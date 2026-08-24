/*
 * Shared pread/LRU expert backend — implementation (Forge F1 Phase 3).
 *
 * Extracted from c/qwen36.c's out-of-core machinery (LCache/LRU/victim scan,
 * in-flight reservation registry, duplicate coalescing identity, fused
 * [qs][weights] container read, format-aware byte accounting) and
 * generalized per docs/experiments/forge_f1_census.md section 4.
 * See expert_backend_pread.h for the seam description.
 */

#include "expert_backend_pread.h"

#include <dirent.h>
#include <errno.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "st.h"
#include "tensor.h"

/* ---- slot states ------------------------------------------------------- */
#define PBS_FREE 0     /* buffer owned, no identity (aborted or never used) */
#define PBS_RESERVED 1 /* an active reservation owns the buffers */
#define PBS_RESIDENT 2 /* published bytes answer this identity */

#define PBS_CLASS_DEMAND 0
#define PBS_CLASS_PREFETCH 1

typedef struct {
    int state;
    int pinned_by_key; /* mirrors the key pin map at publish time */
    int last_class;
    int lease_count;
    int io_counted; /* physical-load accounting already charged (prefetch) */
    int layer, index; /* valid when state != PBS_FREE */
    uint64_t used;    /* LRU clock stamp */
    uint8_t *wbuf;
    uint8_t *sbuf;
    size_t wcap, scap;     /* allocated capacities (grow-only) */
    size_t wbytes, sbytes; /* live byte counts when resident */
} PreadSlot;

typedef struct {
    PreadSlot *slots;
    int n;      /* high-water mark of constructed slots (<= cap) */
    int cap;
    int16_t *reserving; /* [n_experts]: slot idx of the ONE active admission, -1 none */
} PreadLayer;

typedef struct {
    ColiExpertStore store; /* public face; ops points at the static table */
    pthread_mutex_t mx;    /* the single mutation lock (g_pilot_mx pattern) */
    PreadLayer *layers;
    int n_layers, n_experts;
    int slots_per_layer;
    uint64_t clock;
    shards S;
    int owns_shards;
    const int *layer_map; /* borrowed; NULL = identity */
    char wtmpl[160];
    char stmpl[160];
    int has_scales;
    int drop_pagecache;
    int fused_read;
    int64_t *emeta_w;  /* [n_layers*n_experts] weights bytes */
    int64_t *emeta_s;  /* scales bytes (0 = none) */
    int8_t *emeta_fmt; /* classified format id per key */
    uint8_t *key_pinned; /* [n_layers*n_experts] pin intent map */
    ColiExpertStoreStats stats;
} PreadBackend;

static const ColiExpertStoreOps pbs_ops;

#define PBS_LOCK(bk) pthread_mutex_lock(&(bk)->mx)
#define PBS_UNLOCK(bk) pthread_mutex_unlock(&(bk)->mx)

static double pbs_now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static int pbs_container_layer(const PreadBackend *bk, int layer) {
    return bk->layer_map ? bk->layer_map[layer] : layer;
}

static void pbs_names(const PreadBackend *bk, int layer, int index,
                      char *wnm, size_t wnm_sz, char *snm, size_t snm_sz) {
    snprintf(wnm, wnm_sz, bk->wtmpl, pbs_container_layer(bk, layer), index);
    if (snm && snm_sz) {
        if (bk->has_scales)
            snprintf(snm, snm_sz, bk->stmpl, pbs_container_layer(bk, layer), index);
        else
            snm[0] = '\0';
    }
}

/* Format classification from the container itself (generic form of the
 * proven expert_fmt size logic): merged-weight numel N vs nbytes B gives
 * bytes-per-element ratio. 1 -> INT8, 1/2 -> INT4, 24/64 -> INT3-g64. */
static int pbs_classify(int64_t nbytes, int64_t numel) {
    if (numel <= 0 || nbytes <= 0) return 0;
    if (nbytes == numel) return 8;           /* INT8 */
    if (nbytes * 2 == numel) return 4;       /* INT4 packed nibbles */
    if (nbytes * 64 == numel * 24) return 3; /* INT3-g64 (24B per 64 weights) */
    return 0;                                /* unknown -> reject */
}

static ColiTensorFormat pbs_view_format(int fmt) {
    switch (fmt) {
    case 3: return COLI_TENSOR_INT3_BLOCK;
    case 4: return COLI_TENSOR_INT4_BLOCK;
    default: return COLI_TENSOR_INT8_BLOCK;
    }
}

static int64_t pbs_ei(const PreadBackend *bk, int layer, int index) {
    return (int64_t)layer * bk->n_experts + index;
}

/* grow-only segment buffer sizing (the slot_ensure_format reuse discipline,
 * without format-specific pointer aliasing: one blob per segment) */
static int pbs_seg_reserve(uint8_t **buf, size_t *cap, size_t need) {
    if (*buf && *cap >= need) return 0;
    uint8_t *nb = (uint8_t *)realloc(*buf, need);
    if (!nb) return -1;
    *buf = nb;
    *cap = need;
    return 0;
}

/* Find the resident slot holding (layer,index), or NULL. Caller holds mx. */
static PreadSlot *pbs_find_resident(PreadBackend *bk, int layer, int index) {
    PreadLayer *pl = &bk->layers[layer];
    for (int i = 0; i < pl->n; i++) {
        PreadSlot *s = &pl->slots[i];
        if (s->state == PBS_RESIDENT && s->layer == layer && s->index == index)
            return s;
    }
    return NULL;
}

/*
 * Victim selection — three stages preserved from qwen36 expert_acquire /
 * pilot_realload:
 *   1. oldest RESIDENT slot neither pinned nor reserved (reserved == the
 *      eid==-1 sentinel of the original);
 *   2. otherwise oldest RESIDENT slot (may be pinned, never reserved);
 *   3. otherwise spin (every candidate buffer is owned by an unlocked pread
 *      that will publish into it) until stage 2 finds a victim. Stealing a
 *      reserved slot corrupts silently — wait instead; reservations drain.
 * Caller holds mx; returns the slot cleared to FREE.
 */
static PreadSlot *pbs_pick_victim(PreadBackend *bk, PreadLayer *pl) {
    int lru = -1;
    for (int i = 0; i < pl->n; i++) {
        PreadSlot *s = &pl->slots[i];
        if (s->state != PBS_RESIDENT || s->pinned_by_key) continue;
        if (lru < 0 || s->used < pl->slots[lru].used) lru = i;
    }
    if (lru < 0) {
        for (int i = 0; i < pl->n; i++) {
            PreadSlot *s = &pl->slots[i];
            if (s->state != PBS_RESIDENT) continue;
            if (lru < 0 || s->used < pl->slots[lru].used) lru = i;
        }
    }
    while (lru < 0) {
        PBS_UNLOCK(bk);
        usleep(1000);
        PBS_LOCK(bk);
        for (int i = 0; i < pl->n; i++) {
            PreadSlot *s = &pl->slots[i];
            if (s->state != PBS_RESIDENT) continue;
            if (lru < 0 || s->used < pl->slots[lru].used) lru = i;
        }
    }
    PreadSlot *s = &pl->slots[lru];
    bk->stats.resident_bytes -= (uint64_t)(s->wbytes + s->sbytes);
    s->state = PBS_FREE;
    s->lease_count = 0;
    s->io_counted = 0;
    s->layer = -1;
    s->index = -1;
    return s;
}

/* Claim a FREE/reusable/victim slot for (layer,index) and mark the in-flight
 * identity. Caller holds mx. busy_kind: 1 = reservation exists, 2 = resident. */
static PreadSlot *pbs_claim(PreadBackend *bk, int layer, int index,
                            int *busy_kind) {
    PreadLayer *pl = &bk->layers[layer];
    if (pl->reserving[index] >= 0) { *busy_kind = 1; return NULL; }
    if (pbs_find_resident(bk, layer, index)) { *busy_kind = 2; return NULL; }

    PreadSlot *s = NULL;
    for (int i = 0; i < pl->n; i++) { /* reuse freed buffers first */
        if (pl->slots[i].state == PBS_FREE) { s = &pl->slots[i]; break; }
    }
    if (!s && pl->n < pl->cap) {
        s = &pl->slots[pl->n++];
        pl->slots[pl->n - 1].state = PBS_FREE;
    }
    if (!s) s = pbs_pick_victim(bk, pl);

    s->state = PBS_RESERVED;
    s->layer = layer;
    s->index = index;
    s->used = ++bk->clock;
    s->last_class = PBS_CLASS_DEMAND;
    s->io_counted = 0;
    pl->reserving[index] = (int16_t)(s - pl->slots);
    bk->stats.reservations_active++;
    return s;
}

/* Release a claim without publishing (abort / failed load). Caller holds mx. */
static void pbs_unclaim(PreadBackend *bk, PreadSlot *s, int charge_abort) {
    PreadLayer *pl = &bk->layers[s->layer];
    if (pl->reserving[s->index] == (int16_t)(s - pl->slots))
        pl->reserving[s->index] = -1;
    if (s->state == PBS_RESERVED) {
        s->state = PBS_FREE;
        s->layer = -1;
        s->index = -1;
        bk->stats.reservations_active--;
        if (charge_abort) bk->stats.aborts++;
    }
}

/* Fill a view from a resident slot. Caller holds mx. */
static void pbs_fill_view(const PreadBackend *bk, const PreadSlot *s,
                          ColiExpertView *view) {
    memset(view, 0, sizeof(*view));
    view->key.layer = s->layer;
    view->key.expert = s->index;
    view->gate.format = pbs_view_format(
        bk->emeta_fmt[pbs_ei(bk, s->layer, s->index)]);
    view->gate.data = s->wbuf;
    view->gate.data_bytes = s->wbytes;
    if (bk->has_scales) {
        view->gate.scale_format = COLI_SCALE_F32;
        view->gate.scales = s->sbuf;
        view->gate.scale_bytes = s->sbytes;
    }
    view->lease = (void *)s;
}

/* Charge publish-side accounting. Caller holds mx. */
static void pbs_charge_publish(PreadBackend *bk, PreadSlot *s) {
    bk->stats.publishes++;
    if (!s->io_counted) bk->stats.physical_loads++;
    s->io_counted = 0;
    bk->stats.admitted_bytes += (uint64_t)(s->wbytes + s->sbytes);
    bk->stats.admitted_weight_bytes += (uint64_t)s->wbytes;
    bk->stats.admitted_scale_bytes += (uint64_t)s->sbytes;
    bk->stats.resident_bytes += (uint64_t)(s->wbytes + s->sbytes);
}

/* Shared I/O core: pread the container bytes for slot s (identity already
 * claimed) into its segment buffers. Unlocked; returns 0 and sets
 * *bytes_read, or -1 on any failure (nothing published either way). */
static int pbs_io_fill(PreadBackend *bk, PreadSlot *s, int64_t *bytes_read) {
    char wnm[192], snm[192];
    pbs_names(bk, s->layer, s->index, wnm, sizeof(wnm), snm, sizeof(snm));
    int64_t ei = pbs_ei(bk, s->layer, s->index);
    st_tensor *tw = st_find(&bk->S, wnm);
    st_tensor *ts = bk->has_scales ? st_find(&bk->S, snm) : NULL;
    if (!tw || tw->nbytes != bk->emeta_w[ei] ||
        (bk->has_scales && (!ts || ts->nbytes != bk->emeta_s[ei])))
        return -1;

    double t0 = pbs_now_ms();
    int64_t total = 0;
    int fused = 0;
    if (bk->fused_read && ts)
        fused = (ts->fd == tw->fd) && ts->off + ts->nbytes == tw->off &&
                ts->nbytes > 0;
    if (fused) {
        /* Container interleaves [qs][merged_weight] with zero gap (verified
         * above per-tensor): one pread spanning both replaces two device
         * commands; identical bytes, split after transfer. */
        int64_t tot = ts->nbytes + tw->nbytes;
        uint8_t *stg = (uint8_t *)malloc((size_t)tot);
        if (!stg) return -1;
        st_pread_full(tw->fd, stg, tot, ts->off, "pbs fused");
        if (bk->drop_pagecache)
            posix_fadvise(tw->fd, ts->off, tot, POSIX_FADV_DONTNEED);
        memcpy(s->sbuf, stg, (size_t)ts->nbytes);
        memcpy(s->wbuf, stg + ts->nbytes, (size_t)tw->nbytes);
        free(stg);
        total = tot;
    } else {
        st_pread_full(tw->fd, s->wbuf, tw->nbytes, tw->off, "pbs weights");
        if (bk->drop_pagecache)
            posix_fadvise(tw->fd, tw->off, tw->nbytes, POSIX_FADV_DONTNEED);
        total = tw->nbytes;
        if (ts) {
            st_pread_full(ts->fd, s->sbuf, ts->nbytes, ts->off, "pbs scales");
            if (bk->drop_pagecache)
                posix_fadvise(ts->fd, ts->off, ts->nbytes, POSIX_FADV_DONTNEED);
            total += ts->nbytes;
        }
    }
    double t1 = pbs_now_ms();

    PBS_LOCK(bk);
    bk->stats.bytes_read += (uint64_t)total;
    bk->stats.admission_ms_total += (t1 - t0);
    PBS_UNLOCK(bk);
    if (bytes_read) *bytes_read = total;
    return 0;
}

/* Internal synchronous admission used by prefetch(): claim -> pread into the
 * reserved buffers (unlocked I/O) -> publish. Returns 0 on success. */
static int pbs_admit_sync(PreadBackend *bk, int layer, int index, int as_prefetch) {
    PBS_LOCK(bk);
    int busy = 0;
    PreadSlot *s = pbs_claim(bk, layer, index, &busy);
    if (!s) { PBS_UNLOCK(bk); return busy == 2 ? 0 : -1; }
    if (as_prefetch) s->last_class = PBS_CLASS_PREFETCH;

    /* Size the destinations from emeta so the pread runs unlocked. */
    int64_t ei = pbs_ei(bk, layer, index);
    int64_t want_w = bk->emeta_w[ei], want_s = bk->has_scales ? bk->emeta_s[ei] : 0;
    int grow = pbs_seg_reserve(&s->wbuf, &s->wcap, (size_t)want_w) == 0 &&
               (!bk->has_scales ||
                pbs_seg_reserve(&s->sbuf, &s->scap, (size_t)want_s) == 0);
    PBS_UNLOCK(bk);
    if (!grow || pbs_io_fill(bk, s, NULL) != 0) {
        PBS_LOCK(bk);
        pbs_unclaim(bk, s, 1);
        PBS_UNLOCK(bk);
        return -1;
    }

    PBS_LOCK(bk);
    s->wbytes = (size_t)want_w;
    s->sbytes = bk->has_scales ? (size_t)want_s : 0;
    bk->stats.physical_loads++;
    s->io_counted = 1;
    if (s->last_class == PBS_CLASS_PREFETCH)
        bk->stats.prefetch_requests++;
    else
        bk->stats.demand_requests++;
    s->state = PBS_RESIDENT;
    s->pinned_by_key = bk->key_pinned[ei];
    s->used = ++bk->clock;
    bk->layers[layer].reserving[index] = -1;
    bk->stats.reservations_active--;
    pbs_charge_publish(bk, s);
    PBS_UNLOCK(bk);
    return 0;
}

/* ---- ops: lease surface ------------------------------------------------ */

static int pbs_lookup(ColiExpertStore *store, ColiExpertKey key,
                      ColiExpertView *view) {
    PreadBackend *bk = (PreadBackend *)store;
    if (!view) return COLI_EXPERT_ERR_INVALID;
    memset(view, 0, sizeof(*view));
    if (!bk || key.layer < 0 || key.layer >= bk->n_layers ||
        key.expert < 0 || key.expert >= bk->n_experts)
        return -1;

    PBS_LOCK(bk);
    PreadSlot *s = pbs_find_resident(bk, key.layer, key.expert);
    bk->stats.requests++;
    bk->stats.logical_requests++;
    bk->stats.demand_requests++;
    if (!s) {
        bk->stats.misses++;
        bk->stats.demand_misses++;
        PBS_UNLOCK(bk);
        return -1;
    }
    bk->stats.hits++;
    bk->stats.demand_hits++;
    if (s->last_class == PBS_CLASS_PREFETCH) bk->stats.prefetch_hits++;
    s->used = ++bk->clock;
    s->lease_count++;
    pbs_fill_view(bk, s, view);
    PBS_UNLOCK(bk);
    return 0;
}

static void pbs_release(ColiExpertStore *store, ColiExpertView *view) {
    PreadBackend *bk = (PreadBackend *)store;
    if (!view) return;
    if (view->lease && bk) {
        PBS_LOCK(bk);
        PreadSlot *s = (PreadSlot *)view->lease;
        if (s->lease_count > 0) s->lease_count--;
        PBS_UNLOCK(bk);
    }
    memset(view, 0, sizeof(*view));
}

static int pbs_prefetch(ColiExpertStore *store, const ColiExpertKey *keys,
                        size_t count) {
    PreadBackend *bk = (PreadBackend *)store;
    if (!keys && count) return COLI_EXPERT_ERR_INVALID;
    int admitted = 0;
    for (size_t i = 0; i < count; i++) {
        int layer = keys[i].layer, index = keys[i].expert;
        if (layer < 0 || layer >= bk->n_layers ||
            index < 0 || index >= bk->n_experts)
            continue;
        if (pbs_admit_sync(bk, layer, index, 1) == 0) admitted++;
    }
    return admitted;
}

/* ---- ops: reservation surface ------------------------------------------ */

static int pbs_reserve(ColiExpertStore *store, const ColiExpertCoreKey *key,
                       ColiExpertReservation *out) {
    PreadBackend *bk = (PreadBackend *)store;
    if (!key || !out) return COLI_EXPERT_ERR_INVALID;
    memset(out, 0, sizeof(*out));
    if (!bk || key->role != COLI_EXPERT_ROLE_EXPERT ||
        key->layer < 0 || key->layer >= bk->n_layers ||
        key->index < 0 || key->index >= bk->n_experts)
        return COLI_EXPERT_ERR_INVALID;

    PBS_LOCK(bk);
    int busy = 0;
    PreadSlot *s = pbs_claim(bk, key->layer, key->index, &busy);
    if (!s) {
        if (busy == 1) bk->stats.coalesced_requests++; /* saw in-flight loader */
        PBS_UNLOCK(bk);
        return COLI_EXPERT_ERR_BUSY;
    }
    int64_t ei = pbs_ei(bk, key->layer, key->index);
    if (pbs_seg_reserve(&s->wbuf, &s->wcap, (size_t)bk->emeta_w[ei]) != 0 ||
        (bk->has_scales &&
         pbs_seg_reserve(&s->sbuf, &s->scap, (size_t)bk->emeta_s[ei]) != 0)) {
        pbs_unclaim(bk, s, 1);
        PBS_UNLOCK(bk);
        return COLI_EXPERT_ERR_NOCAP;
    }

    /* Segment table: backend-owned, handed to the loader. The destination
     * pointers are stable for the reservation's lifetime because the buffers
     * are grown NOW and eviction cannot touch a RESERVED slot. */
    ColiExpertSegment *tbl =
        (ColiExpertSegment *)calloc(2, sizeof(ColiExpertSegment));
    if (!tbl) {
        pbs_unclaim(bk, s, 1);
        PBS_UNLOCK(bk);
        return COLI_EXPERT_ERR_NOCAP;
    }
    tbl[0].data = s->wbuf;
    tbl[0].bytes = (size_t)bk->emeta_w[ei];
    tbl[1].data = bk->has_scales ? s->sbuf : NULL;
    tbl[1].bytes = bk->has_scales ? (size_t)bk->emeta_s[ei] : 0;

    out->key = *key;
    out->segment_count = bk->has_scales ? 2 : 1;
    out->segments = tbl;
    out->impl = s;
    PBS_UNLOCK(bk);
    return COLI_EXPERT_OK;
}

static int pbs_publish(ColiExpertStore *store, ColiExpertReservation *res,
                       ColiExpertView *out_view) {
    PreadBackend *bk = (PreadBackend *)store;
    if (!res || !res->segments || !res->impl) return COLI_EXPERT_ERR_INVALID;
    PreadSlot *s = (PreadSlot *)res->impl;
    int layer = s->layer, index = s->index;

    PBS_LOCK(bk);
    PreadLayer *pl = &bk->layers[layer];
    if (pl->reserving[index] != (int16_t)(s - pl->slots) ||
        s->state != PBS_RESERVED) {
        PBS_UNLOCK(bk);
        return COLI_EXPERT_ERR_INVALID;
    }
    int64_t ei = pbs_ei(bk, layer, index);
    /* Sizes are authoritative from emeta: loaders write exactly these. */
    s->wbytes = (size_t)bk->emeta_w[ei];
    s->sbytes = bk->has_scales ? (size_t)bk->emeta_s[ei] : 0;
    s->state = PBS_RESIDENT;
    s->pinned_by_key = bk->key_pinned[ei];
    s->used = ++bk->clock;
    pl->reserving[index] = -1;
    bk->stats.reservations_active--;
    pbs_charge_publish(bk, s);
    if (out_view) {
        pbs_fill_view(bk, s, out_view);
        s->lease_count++;
    }
    PBS_UNLOCK(bk);

    free(res->segments);
    res->segments = NULL;
    res->segment_count = 0;
    res->impl = NULL;
    return COLI_EXPERT_OK;
}

static void pbs_abort(ColiExpertStore *store, ColiExpertReservation *res) {
    PreadBackend *bk = (PreadBackend *)store;
    if (!res) return;
    if (res->impl && bk) {
        PBS_LOCK(bk);
        /* an explicit abort charges the abort counter; failed internal
         * loads go through pbs_unclaim(..., 1) at their own sites */
        pbs_unclaim(bk, (PreadSlot *)res->impl, 1);
        PBS_UNLOCK(bk);
    }
    free(res->segments);
    memset(res, 0, sizeof(*res));
}

/* ---- ops: pins, batch, stats, destroy ---------------------------------- */

static int pbs_pin(ColiExpertStore *store, const ColiExpertCoreKey *key) {
    PreadBackend *bk = (PreadBackend *)store;
    if (!key || key->role != COLI_EXPERT_ROLE_EXPERT ||
        key->layer < 0 || key->layer >= bk->n_layers ||
        key->index < 0 || key->index >= bk->n_experts)
        return COLI_EXPERT_ERR_INVALID;
    PBS_LOCK(bk);
    int64_t ei = pbs_ei(bk, key->layer, key->index);
    if (!bk->key_pinned[ei]) {
        bk->key_pinned[ei] = 1;
        bk->stats.pinned_count++;
    }
    PreadSlot *s = pbs_find_resident(bk, key->layer, key->index);
    if (s) s->pinned_by_key = 1; /* pin_hot_experts discipline: immediate */
    PBS_UNLOCK(bk);
    return COLI_EXPERT_OK;
}

static void pbs_unpin(ColiExpertStore *store, const ColiExpertCoreKey *key) {
    PreadBackend *bk = (PreadBackend *)store;
    if (!key || key->layer < 0 || key->layer >= bk->n_layers ||
        key->index < 0 || key->index >= bk->n_experts)
        return;
    PBS_LOCK(bk);
    int64_t ei = pbs_ei(bk, key->layer, key->index);
    if (bk->key_pinned[ei]) {
        bk->key_pinned[ei] = 0;
        if (bk->stats.pinned_count) bk->stats.pinned_count--;
    }
    PreadSlot *s = pbs_find_resident(bk, key->layer, key->index);
    if (s) s->pinned_by_key = 0;
    PBS_UNLOCK(bk);
}

static int pbs_lookup_batch(ColiExpertStore *store, const ColiExpertKey *keys,
                            size_t count, ColiExpertView *views) {
    if (count && (!keys || !views)) return 0;
    int ok = 0;
    for (size_t i = 0; i < count; i++) {
        /* dedupe within the batch: a later duplicate cannot take a second
         * lease on a slot this batch already leased */
        int dup = 0;
        for (size_t j = 0; j < i; j++) {
            if (keys[j].layer == keys[i].layer &&
                keys[j].expert == keys[i].expert) {
                dup = 1;
                break;
            }
        }
        if (dup) {
            memset(&views[i], 0, sizeof(views[i]));
            continue;
        }
        if (coli_expert_lookup(store, keys[i], &views[i]) == 0) ok++;
    }
    return ok;
}

static void pbs_stats(const ColiExpertStore *store, ColiExpertStoreStats *stats) {
    PreadBackend *bk = (PreadBackend *)store;
    if (!stats) return;
    PBS_LOCK(bk);
    *stats = bk->stats;
    PBS_UNLOCK(bk);
}

static void pbs_destroy(ColiExpertStore *store) {
    PreadBackend *bk = (PreadBackend *)store;
    if (!bk) return;
    PBS_LOCK(bk);
#ifndef NDEBUG
    if (bk->stats.reservations_active != 0)
        fprintf(stderr, "pbs_destroy: %llu active reservations\n",
                (unsigned long long)bk->stats.reservations_active);
#endif
    PBS_UNLOCK(bk);

    for (int l = 0; l < bk->n_layers; l++) {
        PreadLayer *pl = &bk->layers[l];
        for (int i = 0; i < pl->n; i++) {
#ifndef NDEBUG
            if (pl->slots[i].lease_count)
                fprintf(stderr, "pbs_destroy: layer %d slot %d holds %d leases\n",
                        l, i, pl->slots[i].lease_count);
#endif
            free(pl->slots[i].wbuf);
            free(pl->slots[i].sbuf);
        }
        free(pl->slots);
        free(pl->reserving);
    }
    free(bk->layers);
    free(bk->emeta_w);
    free(bk->emeta_s);
    free(bk->emeta_fmt);
    free(bk->key_pinned);
    if (bk->owns_shards) {
        for (int i = 0; i < bk->S.nfd; i++) {
            if (bk->S.fds[i] >= 0) close(bk->S.fds[i]);
#if !defined(_WIN32)
            /* O_DIRECT twins are opened lazily; -2 marks untried */
#endif
        }
        /* strdup'd name/path tables are intentionally leaked here, mirroring
         * st_init_multi's own one-shot ownership convention */
    }
    pthread_mutex_destroy(&bk->mx);
    free(bk);
}

static const ColiExpertStoreOps pbs_ops = {
    pbs_lookup,   pbs_release, pbs_prefetch, pbs_stats, pbs_destroy,
    pbs_reserve,  pbs_publish, pbs_abort,    pbs_pin,   pbs_unpin,
    pbs_lookup_batch,
};

/* ---- open ---------------------------------------------------------------- */

static int pbs_template_ok(const char *t) {
    if (!t || !*t) return 0;
    const char *first = strstr(t, "%d");
    if (!first) return 0;
    return strstr(first + 2, "%d") != NULL; /* layer + index placeholders */
}

int coli_expert_backend_pread_open(const ColiExpertStoreDescriptor *desc,
                                   ColiExpertStore **output,
                                   char *error, size_t error_size) {
    if (!output || !error) return -1;
    *output = NULL;
    error[0] = '\0';
    if (!desc) {
        snprintf(error, error_size, "pread backend: descriptor required");
        return -1;
    }
    if (desc->shard_table) {
        snprintf(error, error_size,
                 "pread backend: borrowed shard_table not supported yet "
                 "(hash-map pointers do not survive copies); pass storage_path");
        return -1;
    }
    if (desc->n_layers <= 0 || desc->n_experts <= 0 ||
        !pbs_template_ok(desc->weights_name_template)) {
        snprintf(error, error_size,
                 "pread backend: need positive n_layers/n_experts and a "
                 "weights_name_template with two %%d placeholders");
        return -1;
    }
    int has_scales = desc->scales_name_template && *desc->scales_name_template;
    if (has_scales && !pbs_template_ok(desc->scales_name_template)) {
        snprintf(error, error_size,
                 "pread backend: scales_name_template must contain two %%d");
        return -1;
    }
    if (!desc->storage_path || !*desc->storage_path) {
        snprintf(error, error_size, "pread backend: storage_path required");
        return -1;
    }

    PreadBackend *bk = (PreadBackend *)calloc(1, sizeof(*bk));
    if (!bk) return -1;
    pthread_mutex_init(&bk->mx, NULL);
    bk->n_layers = desc->n_layers;
    bk->n_experts = desc->n_experts;
    bk->layer_map = desc->layer_map;
    bk->has_scales = has_scales;
    bk->drop_pagecache = desc->drop_pagecache ? 1 : 0;
    bk->fused_read = desc->fused_read ? 1 : 0;
    snprintf(bk->wtmpl, sizeof(bk->wtmpl), "%s", desc->weights_name_template);
    if (has_scales)
        snprintf(bk->stmpl, sizeof(bk->stmpl), "%s", desc->scales_name_template);

    memset(&bk->S, 0, sizeof(bk->S));
    /* st_init_multi exits on an unreadable dir (engine convention); validate
     * first so a bad descriptor returns a clean error instead. */
    {
        DIR *d = opendir(desc->storage_path);
        if (!d) {
            snprintf(error, error_size,
                     "pread backend: cannot read storage_path %s: %s",
                     desc->storage_path, strerror(errno));
            pthread_mutex_destroy(&bk->mx);
            free(bk);
            return -1;
        }
        closedir(d);
    }
    st_init_multi(&bk->S, desc->storage_path, NULL);
    bk->owns_shards = 1;
    if (bk->S.nfd <= 0) {
        snprintf(error, error_size,
                 "pread backend: no safetensors shards found under %s",
                 desc->storage_path);
        pthread_mutex_destroy(&bk->mx);
        free(bk);
        return -1;
    }

    int rc = -1;
    const int64_t nk = (int64_t)bk->n_layers * bk->n_experts;
    bk->emeta_w = (int64_t *)calloc((size_t)nk, sizeof(int64_t));
    bk->emeta_s = (int64_t *)calloc((size_t)nk, sizeof(int64_t));
    bk->emeta_fmt = (int8_t *)calloc((size_t)nk, sizeof(int8_t));
    bk->key_pinned = (uint8_t *)calloc((size_t)nk, 1);
    bk->layers = (PreadLayer *)calloc((size_t)bk->n_layers, sizeof(PreadLayer));
    if (!bk->emeta_w || !bk->emeta_s || !bk->emeta_fmt || !bk->key_pinned ||
        !bk->layers) {
        snprintf(error, error_size, "pread backend: OOM");
        goto fail;
    }
    for (int l = 0; l < bk->n_layers; l++) {
        bk->layers[l].reserving =
            (int16_t *)malloc((size_t)bk->n_experts * sizeof(int16_t));
        if (!bk->layers[l].reserving) {
            snprintf(error, error_size, "pread backend: OOM");
            goto fail;
        }
        memset(bk->layers[l].reserving, 0xFF,
               (size_t)bk->n_experts * sizeof(int16_t)); /* all -1 */
    }

    /* Probe per-key metadata up front (fail-closed: every expert present and
     * classifiable, so admission never discovers a hole mid-decode). */
    int64_t max_slot = 0;
    char wnm[192], snm[192];
    for (int l = 0; l < bk->n_layers; l++) {
        for (int e = 0; e < bk->n_experts; e++) {
            pbs_names(bk, l, e, wnm, sizeof(wnm), snm, sizeof(snm));
            st_tensor *tw = st_find(&bk->S, wnm);
            if (!tw) {
                snprintf(error, error_size, "pread backend: missing tensor %s",
                         wnm);
                goto fail;
            }
            int fmt = pbs_classify(tw->nbytes, tw->numel);
            if (!fmt) {
                snprintf(error, error_size,
                         "pread backend: %s (%lld B / %lld elems) is not "
                         "INT8/INT4/INT3-g64 classifiable",
                         wnm, (long long)tw->nbytes, (long long)tw->numel);
                goto fail;
            }
            int64_t sb = 0;
            if (has_scales) {
                st_tensor *ts = st_find(&bk->S, snm);
                if (!ts) {
                    snprintf(error, error_size,
                             "pread backend: missing tensor %s", snm);
                    goto fail;
                }
                sb = ts->nbytes;
            }
            int64_t ei = pbs_ei(bk, l, e);
            bk->emeta_w[ei] = tw->nbytes;
            bk->emeta_s[ei] = sb;
            bk->emeta_fmt[ei] = (int8_t)fmt;
            if (tw->nbytes + sb > max_slot) max_slot = tw->nbytes + sb;
        }
    }

    /* Capacity: explicit budget derives uniform per-layer slots; without one,
     * a documented minimal floor applies (callers should always size). */
    int spl = 2;
    if (desc->capacity_bytes > 0 && max_slot > 0) {
        spl = (int)(desc->capacity_bytes /
                    ((uint64_t)max_slot * (uint64_t)bk->n_layers));
        if (spl < 1) spl = 1;
        if (spl > bk->n_experts) spl = bk->n_experts;
    }
    bk->slots_per_layer = spl;
    bk->stats.capacity_bytes =
        (uint64_t)spl * (uint64_t)max_slot * (uint64_t)bk->n_layers;

    for (int l = 0; l < bk->n_layers; l++) {
        bk->layers[l].cap = spl;
        bk->layers[l].slots = (PreadSlot *)calloc((size_t)spl, sizeof(PreadSlot));
        if (!bk->layers[l].slots) {
            snprintf(error, error_size, "pread backend: OOM");
            goto fail;
        }
    }

    bk->store.ops = &pbs_ops;
    bk->store.state = NULL;
    bk->store.gpu = NULL;
    *output = &bk->store;
    return 0;

fail:
    for (int l = 0; l < bk->n_layers; l++) {
        free(bk->layers[l].slots);
        free(bk->layers[l].reserving);
    }
    for (int l = 0; l < bk->n_layers; l++)
        for (int i = 0; i < bk->layers[l].n; i++) {
            free(bk->layers[l].slots[i].wbuf);
            free(bk->layers[l].slots[i].sbuf);
        }
    free(bk->layers);
    free(bk->emeta_w);
    free(bk->emeta_s);
    free(bk->emeta_fmt);
    free(bk->key_pinned);
    for (int i = 0; i < bk->S.nfd; i++)
        if (bk->S.fds[i] >= 0) close(bk->S.fds[i]);
    pthread_mutex_destroy(&bk->mx);
    free(bk);
    return rc;
}

int coli_expert_backend_pread_register(const char *name) {
    return coli_expert_store_backend_register_v2(
        name, coli_expert_backend_pread_open);
}

/* ColiAdmissionLoadFn adapter: fill an active reservation's segments with
 * the expert's container bytes. userdata = the ColiExpertStore*. */
int coli_expert_backend_pread_load(void *userdata,
                                   const ColiExpertCoreKey *key,
                                   ColiExpertReservation *res) {
    PreadBackend *bk = (PreadBackend *)userdata;
    if (!bk || !res || !res->impl || !key ||
        key->role != COLI_EXPERT_ROLE_EXPERT)
        return -1;
    PreadSlot *s = (PreadSlot *)res->impl;
    if (s->state != PBS_RESERVED || s->layer != key->layer ||
        s->index != key->index)
        return -1;
    int64_t ei = pbs_ei(bk, s->layer, s->index);
    /* buffers were grown at reserve(); be defensive about capacity anyway */
    if (pbs_seg_reserve(&s->wbuf, &s->wcap, (size_t)bk->emeta_w[ei]) != 0)
        return -1;
    if (bk->has_scales &&
        pbs_seg_reserve(&s->sbuf, &s->scap, (size_t)bk->emeta_s[ei]) != 0)
        return -1;
    if (pbs_io_fill(bk, s, NULL) != 0) return -1;
    s->wbytes = (size_t)bk->emeta_w[ei];
    s->sbytes = bk->has_scales ? (size_t)bk->emeta_s[ei] : 0;

    PBS_LOCK(bk);
    bk->stats.physical_loads++;
    s->io_counted = 1; /* publish will not double-charge */
    PBS_UNLOCK(bk);
    return 0;
}
