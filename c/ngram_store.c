/* ngram_store.c - NGRAM-FORGE N1 read-only sparse-row store with
 *
 * Modes:
 *   NSR_RAM    synthetic in-RAM table (pattern rows), pure gather ceiling
 *   NSR_MMAP   file-backed mmap; consume copies rows (page-fault driven)
 *   NSR_PREAD  synchronous pread per row
 *   NSR_ASYNC  bounded worker pool, batch prefetch handles, intra-batch
 *              dedupe, optional direct-mapped hot-row cache in all modes
 *
 * Row content (file modes) is written by ngram_make_table: first 8 bytes
 * little-endian row id, rest 0x3c.
 */
#include "ngram_forge.h"

#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#define BATCH_MAX_ROWS 32

typedef struct {
    uint8_t *buf;      /* row_bytes payload, valid when state==2 */
    uint64_t row;
    int state;         /* 0 free, 1 queued, 2 done, 3 failed       */
} nsr_slot;

typedef struct {
    int used;
    int n;
    int slot[BATCH_MAX_ROWS];  /* per requested row: slot index (dedup map) */
    int pending;
    uint64_t t_issue;
    pthread_mutex_t mtx;
    pthread_cond_t  cv;
} nsr_batch;

struct nsr_store {
    nsr_cfg cfg;
    int fd;
    uint8_t *map;
    size_t map_size;
    nsr_stats st;
    /* direct-mapped row cache */
    uint64_t *cache_tag;
    uint8_t  *cache_mem;
    uint32_t cache_assoc, cache_seen_pad;
    uint64_t cache_sets, cache_clock;
    uint32_t *cache_lru;
    uint8_t *cache_seen;
    /* async machinery */
    pthread_t threads[64];
    pthread_mutex_t q_mtx;
    pthread_cond_t  q_cv;
    int q_head, q_tail, q_count, q_stop;
    struct { int batch; int slot; } q[4096];
    nsr_slot *slots;
    int n_slots;
    nsr_batch *batches;
    int n_batches, rr_batch, async_started;
};

static uint64_t now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static void *nsr_worker_run(nsr_store *s);

static void atomic_add(uint64_t *x, uint64_t v)
{
    __atomic_fetch_add(x, v, __ATOMIC_RELAXED);
}

/* one row from backing store into dst; returns 0 ok, -1 read error */
static int row_read(nsr_store *s, uint64_t row, uint8_t *dst)
{
    if (s->cfg.mode == NSR_RAM || s->fd < 0) {
        memset(dst, 0x3c, s->cfg.row_bytes);
        size_t idb = sizeof(row) < s->cfg.row_bytes ? sizeof(row) : s->cfg.row_bytes;
        memcpy(dst, &row, idb);
        atomic_add(&s->st.ram_hits, 1);
        return 0;
    }
    if (s->cfg.mode == NSR_MMAP) {
        size_t off = (size_t)row * s->cfg.row_bytes;
        if (off + s->cfg.row_bytes > s->map_size) return -1;
        memcpy(dst, s->map + off, s->cfg.row_bytes);
        atomic_add(&s->st.issued_reads, 1);
        atomic_add(&s->st.bytes_read, s->cfg.row_bytes);
        return 0;
    }
    ssize_t r = pread(s->fd, dst, s->cfg.row_bytes, (off_t)row * s->cfg.row_bytes);
    if (r != (ssize_t)s->cfg.row_bytes) return -1;
    atomic_add(&s->st.issued_reads, 1);
    atomic_add(&s->st.bytes_read, s->cfg.row_bytes);
    return 0;
}

static int cache_lookup(nsr_store *s, uint64_t row, uint8_t *dst)
{
    if (!s->cache_tag) return 0;
    uint32_t assoc = s->cache_assoc, set = (uint32_t)(row % s->cache_sets);
    uint64_t *tags = s->cache_tag + (size_t)set * assoc;
    for (uint32_t w = 0; w < assoc; w++) {
        if (tags[w] == row + 1) { /* tag 0 reserved miss */
            uint64_t entry = (uint64_t)set * assoc + w;
            memcpy(dst, s->cache_mem + entry * s->cfg.row_bytes,
                   s->cfg.row_bytes);
            s->cache_lru[entry] = ++s->cache_clock;
            atomic_add(&s->st.cache_hits, 1);
            return 1;
        }
    }
    return 0;
}

/* peek for issue-time serve (async prefetch), no counter side effects */
static const uint8_t *cache_peek(nsr_store *s, uint64_t row)
{
    if (!s->cache_tag) return NULL;
    uint32_t assoc = s->cache_assoc, set = (uint32_t)(row % s->cache_sets);
    uint64_t *tags = s->cache_tag + (size_t)set * assoc;
    for (uint32_t w = 0; w < assoc; w++)
        if (tags[w] == row + 1)
            return s->cache_mem +
                   ((uint64_t)set * assoc + w) * s->cfg.row_bytes;
    return NULL;
}

static void cache_fill(nsr_store *s, uint64_t row, const uint8_t *src)
{
    if (!s->cache_tag) return;
    uint32_t assoc = s->cache_assoc, set = (uint32_t)(row % s->cache_sets);
    uint64_t *tags = s->cache_tag + (size_t)set * assoc;
    /* second-touch admission: first sighting only marks the sketch */
    if (s->cfg.cache_policy == NSR_CACHE_ADMIT2) {
        if (tags[0] != row + 1 && s->cache_seen[set] < 1) {
            s->cache_seen[set] = 1;
            return;
        }
    }
    /* pick way: empty else LRU */
    uint32_t way = 0;
    uint32_t oldest = UINT32_MAX;
    for (uint32_t w = 0; w < assoc; w++) {
        if (tags[w] == row + 1) { way = w; goto do_fill; }
    }
    for (uint32_t w = 0; w < assoc; w++) {
        if (tags[w] == 0) { way = w; oldest = 0; break; }
        if (s->cache_lru[(uint64_t)set * assoc + w] < oldest) {
            oldest = s->cache_lru[(uint64_t)set * assoc + w];
            way = w;
        }
    }
do_fill:;
    uint64_t entry = (uint64_t)set * assoc + way;
    memcpy(s->cache_mem + entry * s->cfg.row_bytes, src, s->cfg.row_bytes);
    tags[way] = row + 1;
    s->cache_lru[entry] = ++s->cache_clock;
    if (s->cfg.cache_policy == NSR_CACHE_ADMIT2) s->cache_seen[set] = 2;
}

static void cache_inval(nsr_store *s)
{
    if (s->cache_tag)
        memset(s->cache_tag, 0,
               (size_t)s->cache_sets * s->cache_assoc * sizeof(uint64_t));
    if (s->cache_seen)
        memset(s->cache_seen, 0, s->cache_sets);
}

nsr_store *nsr_open(const nsr_cfg *cfg, char *err, size_t errlen)
{
    nsr_store *s = calloc(1, sizeof(*s));
    if (!s) { snprintf(err, errlen, "oom"); return NULL; }
    s->cfg = *cfg;
    if (s->cfg.row_bytes == 0) s->cfg.row_bytes = NGRAM_ROW_BYTES;
    s->fd = -1;

    if (cfg->mode != NSR_RAM && cfg->file) {
        s->fd = open(cfg->file, O_RDONLY);
        if (s->fd < 0) {
            snprintf(err, errlen, "open %s: %s", cfg->file, strerror(errno));
            free(s); return NULL;
        }
        struct stat sb;
        if (fstat(s->fd, &sb) != 0 ||
            (uint64_t)sb.st_size < cfg->rows * cfg->row_bytes) {
            snprintf(err, errlen, "file too small for %llu rows",
                     (unsigned long long)cfg->rows);
            close(s->fd); free(s); return NULL;
        }
        if (cfg->fadv_random >= 0)
            posix_fadvise(s->fd, 0, 0, cfg->fadv_random
                                         ? POSIX_FADV_RANDOM : POSIX_FADV_NORMAL);
    }

    if (cfg->mode == NSR_MMAP) {
        s->map_size = (size_t)cfg->rows * cfg->row_bytes;
        s->map = mmap(NULL, s->map_size, PROT_READ, MAP_PRIVATE, s->fd, 0);
        if (s->map == MAP_FAILED) {
            snprintf(err, errlen, "mmap: %s", strerror(errno));
            close(s->fd); free(s); return NULL;
        }
    }

    if (cfg->cache_rows) {
        s->cache_assoc = cfg->cache_assoc ? cfg->cache_assoc : 1;
        s->cache_sets = cfg->cache_rows / s->cache_assoc;
        if (!s->cache_sets) s->cache_sets = 1;
        uint64_t entries = s->cache_sets * s->cache_assoc;
        s->cache_tag = calloc(entries, sizeof(uint64_t));
        s->cache_mem = malloc((size_t)entries * cfg->row_bytes);
        s->cache_lru = calloc(entries, sizeof(uint32_t));
        s->cache_seen = calloc(s->cache_sets, 1);
        if (!s->cache_tag || !s->cache_mem || !s->cache_lru || !s->cache_seen) {
            snprintf(err, errlen, "oom cache");
            nsr_close(s); return NULL;
        }
    }

    if (cfg->mode == NSR_ASYNC) {
        int w = cfg->workers > 0 ? cfg->workers : 4;
        if (w > 64) w = 64;
        s->cfg.workers = w;
        int depth = cfg->queue_depth > 0 ? cfg->queue_depth : 8;
        if (depth > 4096 / BATCH_MAX_ROWS) depth = 4096 / BATCH_MAX_ROWS;
        s->n_batches = depth;
        s->n_slots = depth * BATCH_MAX_ROWS;
        s->slots = calloc((size_t)s->n_slots, sizeof(nsr_slot));
        s->batches = calloc((size_t)s->n_batches, sizeof(nsr_batch));
        if (!s->slots || !s->batches) {
            snprintf(err, errlen, "oom async"); nsr_close(s); return NULL;
        }
        for (int i = 0; i < s->n_slots; i++)
            s->slots[i].buf = malloc(cfg->row_bytes);
        for (int i = 0; i < s->n_batches; i++) {
            pthread_mutex_init(&s->batches[i].mtx, NULL);
            pthread_cond_init(&s->batches[i].cv, NULL);
        }
        pthread_mutex_init(&s->q_mtx, NULL);
        pthread_cond_init(&s->q_cv, NULL);
        s->async_started = 1;
        for (int i = 0; i < w; i++)
            pthread_create(&s->threads[i], NULL,
                           (void *(*)(void *))nsr_worker_run, s);
    }
    return s;
}

static void *nsr_worker_run(nsr_store *s)
{
    for (;;) {
        pthread_mutex_lock(&s->q_mtx);
        while (s->q_count == 0 && !s->q_stop)
            pthread_cond_wait(&s->q_cv, &s->q_mtx);
        if (s->q_count == 0 && s->q_stop) {
            pthread_mutex_unlock(&s->q_mtx);
            return NULL;
        }
        int b = s->q[s->q_head].batch;
        int sl = s->q[s->q_head].slot;
        s->q_head = (s->q_head + 1) % 4096;
        s->q_count--;
        pthread_mutex_unlock(&s->q_mtx);

        nsr_slot *slot = &s->slots[sl];
        int ok = row_read(s, slot->row, slot->buf) == 0;

        nsr_batch *B = &s->batches[b];
        pthread_mutex_lock(&B->mtx);
        slot->state = ok ? 2 : 3;
        if (--B->pending == 0) {
            uint64_t d = now_ns() - B->t_issue;
            unsigned bin = d / NSR_READY_BIN_NS;
            if (bin >= NSR_READY_BINS) bin = NSR_READY_BINS - 1;
            s->st.ready_hist[bin]++;
            pthread_cond_broadcast(&B->cv);
        }
        pthread_mutex_unlock(&B->mtx);
    }
}

void nsr_close(nsr_store *s)
{
    if (!s) return;
    if (s->async_started) {
        pthread_mutex_lock(&s->q_mtx);
        s->q_stop = 1;
        pthread_cond_broadcast(&s->q_cv);
        pthread_mutex_unlock(&s->q_mtx);
        for (int i = 0; i < s->cfg.workers; i++)
            pthread_join(s->threads[i], NULL);
    }
    if (s->map) munmap(s->map, s->map_size);
    if (s->fd >= 0) close(s->fd);
    if (s->slots) {
        for (int i = 0; i < s->n_slots; i++) free(s->slots[i].buf);
        free(s->slots);
    }
    free(s->batches);
    free(s->cache_tag);
    free(s->cache_mem);
    free(s->cache_lru);
    free(s->cache_seen);
    free(s);
}

int nsr_prefetch(nsr_store *s, const uint64_t *rows, int n_rows)
{
    if (s->cfg.mode != NSR_ASYNC) return -1;
    if (n_rows > BATCH_MAX_ROWS) return -1;

    /* find free batch slot */
    int b = -1;
    for (int i = 0; i < s->n_batches; i++) {
        int cand = (s->rr_batch + i) % s->n_batches;
        pthread_mutex_lock(&s->batches[cand].mtx);
        if (!s->batches[cand].used) b = cand;
        pthread_mutex_unlock(&s->batches[cand].mtx);
        if (b >= 0) { s->rr_batch = (cand + 1) % s->n_batches; break; }
    }
    if (b < 0) return -1; /* bounded queue refuses: caller falls back sync */
    atomic_add(&s->st.batches, 1);

    nsr_batch *B = &s->batches[b];
    pthread_mutex_lock(&B->mtx);
    B->used = 1;
    B->n = n_rows;
    B->pending = 0;
    B->t_issue = now_ns();
    int first_slot = b * BATCH_MAX_ROWS;
    for (int i = 0; i < n_rows; i++) {
        int reuse = -1;
        for (int j = 0; j < i; j++)
            if (rows[j] == rows[i]) { reuse = B->slot[j]; break; }
        if (reuse >= 0) {
            B->slot[i] = reuse;
            atomic_add(&s->st.dedup_hits, 1);
            continue;
        }
        nsr_slot *slot = &s->slots[first_slot + i];
        slot->row = rows[i];
        const uint8_t *cp = cache_peek(s, rows[i]);
        if (cp) {
            /* hot row: serve from cache at issue, skip physical read */
            memcpy(slot->buf, cp, s->cfg.row_bytes);
            slot->state = 2;
            B->slot[i] = i;
            atomic_add(&s->st.cache_hits, 1);
            continue;
        }
        slot->state = 1;
        B->slot[i] = i;
        B->pending++;
        pthread_mutex_lock(&s->q_mtx);
        if (s->q_count < 4096) {
            s->q[s->q_tail % 4096].batch = b;
            s->q[s->q_tail % 4096].slot = first_slot + i;
            s->q_tail++;
            s->q_count++;
            pthread_cond_signal(&s->q_cv);
        } else {
            slot->state = 3; /* refused -> consumer fallback */
            B->pending--;
        }
        pthread_mutex_unlock(&s->q_mtx);
    }
    if (B->pending == 0) pthread_cond_broadcast(&B->cv);
    pthread_mutex_unlock(&B->mtx);
    return b;
}

int nsr_consume_batch(nsr_store *s, int batch, void **out_block,
                      uint64_t *wait_ns, uint64_t *lookup_ns, int *hit)
{
    nsr_batch *B = &s->batches[batch];
    uint64_t t0 = now_ns();
    pthread_mutex_lock(&s->q_mtx);
    unsigned outstanding = (unsigned)s->q_count;
    pthread_mutex_unlock(&s->q_mtx);
    s->st.out_sum += outstanding;
    if (outstanding > s->st.out_max) s->st.out_max = outstanding;
    pthread_mutex_lock(&B->mtx);
    int all_ready = 1;
    for (int i = 0; i < B->n; i++) {
        nsr_slot *slot = &s->slots[batch * BATCH_MAX_ROWS + B->slot[i]];
        if (slot->state == 1) { all_ready = 0; break; }
    }
    while (B->pending > 0)
        pthread_cond_wait(&B->cv, &B->mtx);
    pthread_mutex_unlock(&B->mtx);
    uint64_t t1 = now_ns();
    if (hit) *hit = all_ready;
    if (all_ready) atomic_add(&s->st.batch_hits, 1);

    uint8_t *block = malloc((size_t)B->n * s->cfg.row_bytes);
    if (!block) return -1;
    for (int i = 0; i < B->n; i++) {
        nsr_slot *slot = &s->slots[batch * BATCH_MAX_ROWS + B->slot[i]];
        uint8_t *dst = block + (size_t)i * s->cfg.row_bytes;
        atomic_add(&s->st.lookups, 1);
        if (slot->state == 2) {
            memcpy(dst, slot->buf, s->cfg.row_bytes);
            cache_fill(s, slot->row, slot->buf);
        } else if (cache_lookup(s, slot->row, dst)) {
            /* served from cache */
        } else if (row_read(s, slot->row, dst) != 0) {
            free(block);
            pthread_mutex_lock(&B->mtx); B->used = 0; pthread_mutex_unlock(&B->mtx);
            return -1;
        } else {
            cache_fill(s, slot->row, dst);
        }
    }
    uint64_t t2 = now_ns();
    pthread_mutex_lock(&B->mtx);
    B->used = 0;
    pthread_mutex_unlock(&B->mtx);

    atomic_add(&s->st.wait_ns_total, t1 - t0);
    atomic_add(&s->st.consume_ns_total, t2 - t0);
    atomic_add(&s->st.bytes_consumed, (uint64_t)B->n * s->cfg.row_bytes);
    if (wait_ns) *wait_ns = t1 - t0;
    if (lookup_ns) *lookup_ns = t2 - t0;
    *out_block = block;
    return 0;
}

int nsr_consume(nsr_store *s, const uint64_t *rows, int n_rows,
                void **out_block, uint64_t *wait_ns)
{
    uint64_t t0 = now_ns();
    uint8_t *block = malloc((size_t)n_rows * s->cfg.row_bytes);
    if (!block) return -1;
    for (int i = 0; i < n_rows; i++) {
        atomic_add(&s->st.lookups, 1);
        uint8_t *dst = block + (size_t)i * s->cfg.row_bytes;
        if (cache_lookup(s, rows[i], dst)) continue;
        if (row_read(s, rows[i], dst) != 0) { free(block); return -1; }
        cache_fill(s, rows[i], dst);
    }
    uint64_t t1 = now_ns();
    atomic_add(&s->st.consume_ns_total, t1 - t0);
    atomic_add(&s->st.wait_ns_total, t1 - t0);
    atomic_add(&s->st.bytes_consumed, (uint64_t)n_rows * s->cfg.row_bytes);
    if (wait_ns) *wait_ns = t1 - t0;
    *out_block = block;
    return 0;
}

void nsr_stats_get(nsr_store *s, nsr_stats *st) { *st = s->st; }
void nsr_stats_reset(nsr_store *s) { memset(&s->st, 0, sizeof(s->st)); }

/* discard a prefetched batch that will never be consumed (rejected MTP
 * draft). In-flight worker reads still complete into slots (and can be
 * absorbed by the issue-time cache), but the batch returns to the free
 * pool and the row is counted as wasted. */
void nsr_batch_release(nsr_store *s, int batch)
{
    if (batch < 0 || batch >= s->n_batches) return;
    nsr_batch *B = &s->batches[batch];
    pthread_mutex_lock(&B->mtx);
    if (B->used) {
        B->used = 0;
        atomic_add(&s->st.wasted_rows, (uint64_t)B->n);
    }
    pthread_mutex_unlock(&B->mtx);
}

void nsr_evict(nsr_store *s)
{
    if (s->fd >= 0) {
        fsync(s->fd);
        posix_fadvise(s->fd, 0, 0, POSIX_FADV_DONTNEED);
    }
    if (s->map) madvise(s->map, s->map_size, MADV_DONTNEED);
    cache_inval(s);
}

int nsr_cached_fraction(nsr_store *s, const uint64_t *rows, int n_rows)
{
    if (s->cfg.mode == NSR_RAM) return 100;
    if (s->fd < 0) return -1;
    unsigned char vec[4];
    long ps = sysconf(_SC_PAGESIZE);
    int sample = n_rows > 4096 ? 4096 : n_rows;
    int resident = 0, counted = 0;
    for (int i = 0; i < sample; i++) {
        size_t off = (size_t)rows[i] * s->cfg.row_bytes;
        size_t base = off / (size_t)ps * (size_t)ps;
        size_t sz = (size_t)ps * 2;
        void *m = mmap(NULL, sz, PROT_READ, MAP_PRIVATE, s->fd, (off_t)base);
        if (m == MAP_FAILED) continue;
        memset(vec, 0, sizeof(vec));
        int rc = mincore(m, (size_t)ps, vec);
        if (rc == 0) resident += vec[0] & 1;
        counted++;
        munmap(m, sz);
    }
    return counted ? (int)(100.0 * resident / counted + 0.5) : -1;
}
