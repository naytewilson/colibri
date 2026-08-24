/*
 * Admission scheduler — implementation (Forge F1 Phase 4).
 * See admission.h for the policy contract. The store is the only
 * authoritative residency/in-flight table; this file adds orchestration.
 */

#include "admission.h"

#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    ColiExpertStore *store;
    ColiAdmissionLoadFn load_fn;
    void *load_userdata;
    ColiAdmissionConfig cfg;
    ColiAdmissionStats stats;
} Admission;

ColiAdmission *coli_admission_new(ColiExpertStore *store,
                                  ColiAdmissionLoadFn load_fn,
                                  void *load_userdata,
                                  const ColiAdmissionConfig *cfg) {
    if (!store || !store->ops || !load_fn) return NULL;
    Admission *a = (Admission *)calloc(1, sizeof(*a));
    if (!a) return NULL;
    a->store = store;
    a->load_fn = load_fn;
    a->load_userdata = load_userdata;
    if (cfg) a->cfg = *cfg; /* zeroed fields keep default-OFF */
    return (ColiAdmission *)a;
}

void coli_admission_free(ColiAdmission *adm) { free(adm); }

const ColiAdmissionStats *coli_admission_stats(const ColiAdmission *adm) {
    return adm ? &((const Admission *)adm)->stats : NULL;
}

const ColiAdmissionConfig *coli_admission_config(const ColiAdmission *adm) {
    return adm ? &((const Admission *)adm)->cfg : NULL;
}

typedef struct {
    void (*job)(void *arg, int idx);
    void *arg;
    int from, to;
} RunRange;

static void *run_worker(void *p) {
    RunRange *r = (RunRange *)p;
    for (int i = r->from; i < r->to; i++) r->job(r->arg, i);
    return NULL;
}

void coli_admission_run_parallel(ColiAdmission *adm,
                                 void (*job)(void *arg, int idx),
                                 void *arg, int count, int max_workers) {
    if (!job || count <= 0) return;
    int w = max_workers > 0 ? max_workers : 1;
    if (w > count) w = count;
    if (w <= 1) {
        for (int i = 0; i < count; i++) job(arg, i);
        return;
    }
    pthread_t *tid = (pthread_t *)malloc((size_t)w * sizeof(pthread_t));
    RunRange *ranges = (RunRange *)malloc((size_t)w * sizeof(RunRange));
    if (!tid || !ranges) {
        free(tid);
        free(ranges);
        for (int i = 0; i < count; i++) job(arg, i);
        return;
    }
    int launched = 0;
    int base = count / w, extra = count % w, at = 0;
    for (int i = 0; i < w; i++) {
        ranges[i].job = job;
        ranges[i].arg = arg;
        ranges[i].from = at;
        at += base + (i < extra ? 1 : 0);
        ranges[i].to = at;
        if (i == w - 1) continue; /* main thread takes the last range */
        if (pthread_create(&tid[launched++], NULL, run_worker,
                           &ranges[i]) != 0) {
            launched--;
            run_worker(&ranges[i]);
        }
    }
    run_worker(&ranges[w - 1]);
    for (int i = 0; i < launched; i++) pthread_join(tid[i], NULL);
    free(tid);
    free(ranges);
    (void)adm;
}

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

/* Single-threaded core of one acquisition: hit -> lease; miss -> reserve,
 * load outside locks, publish with lease. BUSY propagates to the caller
 * (which may coalesce). Returns 0 / COLI_EXPERT_ERR_BUSY / store error. */
static int acquire_once(Admission *a, const ColiExpertKey *key,
                        ColiExpertView *out) {
    if (coli_expert_lookup(a->store, *key, out) == 0) {
        a->stats.hits++;
        return 0;
    }
    ColiExpertCoreKey core = coli_expert_core_key(*key);
    ColiExpertReservation res;
    int rc = coli_expert_reserve(a->store, &core, &res);
    if (rc == COLI_EXPERT_ERR_BUSY) return rc;
    if (rc != COLI_EXPERT_OK) return rc;

    if (a->load_fn(a->load_userdata, &core, &res) != 0) {
        coli_expert_abort(a->store, &res);
        a->stats.load_failures++;
        return -5;
    }
    rc = coli_expert_publish(a->store, &res, out);
    if (rc != COLI_EXPERT_OK) return rc;
    a->stats.loads++;
    return 0;
}

int coli_admission_acquire(ColiAdmission *adm, const ColiExpertKey *key,
                           ColiExpertView *out) {
    Admission *a = (Admission *)adm;
    if (!a || !key || !out) return COLI_EXPERT_ERR_INVALID;
    memset(out, 0, sizeof(*out));
    a->stats.acquires++;

    int rc = acquire_once(a, key, out);
    if (rc != COLI_EXPERT_ERR_BUSY || a->cfg.wait_busy_ms <= 0) return rc;

    /* Coalescing window: someone else holds the reservation identity —
     * poll for their publish instead of duplicating the load. */
    double deadline =
        a->cfg.wait_busy_max_ms > 0 ? now_ms() + a->cfg.wait_busy_max_ms : 0.0;
    for (;;) {
        struct timespec ts = {0, (long)a->cfg.wait_busy_ms * 1000000L};
        nanosleep(&ts, NULL);
        a->stats.coalesced_waits++;
        rc = acquire_once(a, key, out);
        if (rc != COLI_EXPERT_ERR_BUSY) return rc;
        if (deadline > 0.0 && now_ms() >= deadline) {
            a->stats.busy_timeouts++;
            return COLI_EXPERT_ERR_BUSY;
        }
        if (a->cfg.wait_busy_max_ms <= 0 && a->stats.coalesced_waits > 100000)
            return COLI_EXPERT_ERR_BUSY; /* safety valve vs infinite loop */
    }
}

/* ---- batch orchestration ------------------------------------------------ */

typedef struct {
    Admission *a;
    ColiExpertKey key;
    ColiExpertView view; /* handed to caller on success */
    int done;            /* 0 pending, 1 leased, -1 failed */
    int serial;          /* run inline in the joining thread */
} BatchJob;

static void batch_run_one(BatchJob *j) {
    int rc = acquire_once(j->a, &j->key, &j->view);
    j->done = (rc == 0) ? 1 : -1;
}

typedef struct {
    BatchJob *jobs;
    int from, to;
} BatchRange;

static void *batch_worker(void *arg) {
    BatchRange *r = (BatchRange *)arg;
    for (int i = r->from; i < r->to; i++)
        if (r->jobs[i].done == 0) batch_run_one(&r->jobs[i]);
    return NULL;
}

/* Contiguous round-robin split of jobs over `w` worker ranges. */
static void split_ranges(int njobs, int w, BatchRange *ranges) {
    int base = njobs / w, extra = njobs % w, at = 0;
    for (int i = 0; i < w; i++) {
        ranges[i].from = at;
        at += base + (i < extra ? 1 : 0);
        ranges[i].to = at;
        ranges[i].jobs = NULL; /* filled by caller */
    }
}

int coli_admission_acquire_batch(ColiAdmission *adm,
                                 const ColiExpertKey *keys, size_t count,
                                 ColiExpertView *views) {
    Admission *a = (Admission *)adm;
    if (!a || count == 0) return 0;
    if (!keys || !views) return 0;
    memset(views, 0, count * sizeof(*views));
    a->stats.batches++;
    a->stats.batch_keys_in += count;

    /* dedupe map: uniq_key[j] = key index of unique job j (first-touch
     * order); entry_job[i] = job index serving caller entry i */
    int *entry_job = (int *)malloc(count * sizeof(int));
    int *uniq_key = (int *)malloc(count * sizeof(int));
    if (!entry_job || !uniq_key) {
        free(entry_job);
        free(uniq_key);
        return 0;
    }
    int nuniq = 0;
    for (size_t i = 0; i < count; i++) {
        int found = -1;
        for (int u = 0; u < nuniq; u++) {
            if (keys[uniq_key[u]].layer == keys[i].layer &&
                keys[uniq_key[u]].expert == keys[i].expert) {
                found = u;
                break;
            }
        }
        if (found >= 0) {
            entry_job[i] = found;
            a->stats.batch_dedupe_removed++;
            continue;
        }
        uniq_key[nuniq] = (int)i;
        entry_job[i] = nuniq;
        nuniq++;
    }

    BatchJob *jobs = (BatchJob *)calloc((size_t)nuniq, sizeof(BatchJob));
    if (!jobs) {
        free(entry_job);
        free(uniq_key);
        return 0;
    }
    for (int u = 0; u < nuniq; u++) {
        jobs[u].a = a;
        jobs[u].key = keys[uniq_key[u]];
        jobs[u].done = 0;
    }

    int w = a->cfg.max_parallel > 0 ? a->cfg.max_parallel : 1;
    if (w > nuniq) w = nuniq;
    if (w <= 1) {
        for (int u = 0; u < nuniq; u++) batch_run_one(&jobs[u]);
    } else {
        pthread_t *tid = (pthread_t *)malloc((size_t)w * sizeof(pthread_t));
        BatchRange *ranges = (BatchRange *)malloc((size_t)w * sizeof(BatchRange));
        if (tid && ranges) {
            int launched = 0;
            split_ranges(nuniq, w, ranges);
            for (int i = 0; i < w; i++) {
                ranges[i].jobs = jobs;
                /* main thread takes the last range */
                if (i == w - 1) continue;
                if (pthread_create(&tid[launched++], NULL, batch_worker,
                                   &ranges[i]) != 0) {
                    launched--; /* run it inline below via its range */
                    batch_worker(&ranges[i]);
                }
            }
            batch_worker(&ranges[w - 1]);
            for (int i = 0; i < launched; i++) pthread_join(tid[i], NULL);
        } else {
            for (int u = 0; u < nuniq; u++) batch_run_one(&jobs[u]);
        }
        free(tid);
        free(ranges);
    }

    /* leasing phase: every caller entry gets its own lease in order */
    int ok = 0;
    for (size_t i = 0; i < count; i++) {
        int u = entry_job[i];
        if (jobs[u].done == 1) {
            if (coli_expert_lookup(a->store, keys[i], &views[i]) == 0) ok++;
        }
    }

    free(jobs);
    free(entry_job);
    free(uniq_key);
    return ok;
}

size_t coli_admission_order_prefetch(ColiExpertKey *keys, size_t count) {
    if (!keys || count == 0) return 0;
    /* insertion sort: candidate lists are small (scheduler-capped), and
     * stability preserves equal-key insertion order deterministically */
    size_t kept = 0;
    for (size_t i = 0; i < count; i++) {
        ColiExpertKey k = keys[i];
        int dup = 0;
        for (size_t j = 0; j < kept; j++) {
            if (keys[j].layer == k.layer && keys[j].expert == k.expert) {
                dup = 1;
                break;
            }
        }
        if (dup) continue;
        size_t pos = kept;
        while (pos > 0 && (keys[pos - 1].layer > k.layer ||
                           (keys[pos - 1].layer == k.layer &&
                            keys[pos - 1].expert > k.expert))) {
            keys[pos] = keys[pos - 1];
            pos--;
        }
        keys[pos] = k;
        kept++;
    }
    return kept;
}

int coli_admission_prefetch(ColiAdmission *adm, const ColiExpertKey *keys,
                            size_t count) {
    Admission *a = (Admission *)adm;
    if (!a || count == 0) return 0;
    int rc = a->store->ops->prefetch
                 ? a->store->ops->prefetch(a->store, keys, count)
                 : 0;
    a->stats.prefetch_enqueued += (uint64_t)(rc > 0 ? rc : 0);
    return rc;
}
