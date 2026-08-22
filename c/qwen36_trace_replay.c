/* qwen36_trace_replay — offline replay gate for the MoE admission trace.
 *
 * Replays the CURRENT cache policy from a qwen36_moe_trace v3 TSV by mirroring
 * the runtime LCache slot arrays exactly (per-layer, cap slots, LRU clock,
 * in-flight protocol). Every mutation row carries its exact slot index, so
 * replay is deterministic — including the pilot-publish duplicate-residency
 * race the current policy permits (two slots may briefly hold one eid; the
 * replay models that instead of hiding it).
 *
 * Gate (all must pass for TRACE_REPLAY_SELF_CONSISTENT):
 *   - HIT   row:  slot held exactly that eid at touch time
 *   - EVICT row:  slot held exactly that victim eid at eviction time
 *   - INSERT row: slot was free/in-flight (-1) at publish time
 *   - occupancy never exceeds cap on any layer
 *   - optional --expect-* counters match recorded runtime control exactly
 *
 * No policy evaluation happens here: this tool exists solely to prove the
 * simulator input can reproduce the runtime it claims to model.
 *
 * Exit 0 = self-consistent, 1 = not self-consistent, 2 = usage/IO error.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define MAX_LAYERS 64

typedef struct {
    int    cap;
    int    nslots;
    int    slot_eid[512];      /* eid resident in slot i, -1 free/in-flight */
    int    fmt[512];
    long   used[512];
} LCacheSim;

static LCacheSim g_lcs[MAX_LAYERS];
static long g_clock = 0;

static long demand_hits, demand_misses, win_demand_hits, win_demand_misses;
static long evict_demand, evict_pilot, insert_demand, insert_pilot;
static long dup_resident_events = 0;
static unsigned long long bytes_demand, bytes_pilot;
static long violations = 0;
static int n_layers_seen = 0;

static void violation(const char *kind, long seq, const char *cls, const char *ev,
                      int layer, int eid, int slot) {
    if (violations < 20)
        fprintf(stderr, "VIOLATION[%s] seq=%ld %s %s layer=%d eid=%d slot=%d\n",
                kind, seq, cls, ev, layer, eid, slot);
    violations++;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <trace.tsv> [--cap N] [--expect-hits N] "
                "[--expect-misses N] [--expect-bytes B] "
                "[--expect-evictions-demand N] [--expect-evictions-pilot N]\n", argv[0]);
        return 2;
    }
    int cap = 128;
    long expect_hits = -1, expect_misses = -1, expect_ed = -1, expect_ep = -1;
    unsigned long long expect_bytes = 0; int have_expect_bytes = 0;
    const char *path = NULL;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--cap") && i + 1 < argc) cap = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--expect-hits") && i + 1 < argc) expect_hits = atol(argv[++i]);
        else if (!strcmp(argv[i], "--expect-misses") && i + 1 < argc) expect_misses = atol(argv[++i]);
        else if (!strcmp(argv[i], "--expect-bytes") && i + 1 < argc) { expect_bytes = strtoull(argv[++i], NULL, 10); have_expect_bytes = 1; }
        else if (!strcmp(argv[i], "--expect-evictions-demand") && i + 1 < argc) expect_ed = atol(argv[++i]);
        else if (!strcmp(argv[i], "--expect-evictions-pilot") && i + 1 < argc) expect_ep = atol(argv[++i]);
        else if (!path) path = argv[i];
        else { fprintf(stderr, "unexpected arg %s\n", argv[i]); return 2; }
    }

    FILE *f = fopen(path, "r");
    if (!f) { perror(path); return 2; }

    char line[4096];
    while (fgets(line, sizeof line, f)) {
        if (line[0] == '#') continue;
        unsigned long long seq;
        char cls[16], event[16];
        long long tok, victim, bytes;
        int layer, eid, fmt2, slot; double adm;
        if (sscanf(line, "%llu\t%15s\t%15s\t%lld\t%d\t%d\t%d\t%lld\t%lf\t%lld\t%d",
                   &seq, cls, event, &tok, &layer, &eid, &fmt2, &bytes, &adm, &victim, &slot) != 11) {
            fprintf(stderr, "PARSE FAIL: %s", line);
            violations++; break;
        }
        if (layer < 0 || layer >= MAX_LAYERS || slot < 0 || slot >= 512) {
            violation("range", seq, cls, event, layer, eid, slot); continue;
        }
        LCacheSim *c = &g_lcs[layer];
        if (c->cap == 0) {
            c->cap = cap; c->nslots = 0; n_layers_seen++;
            for (int i2 = 0; i2 < 512; i2++) c->slot_eid[i2] = -1;  /* static arrays start zeroed */
        }
        int is_pilot = !strcmp(cls, "PILOT");
        int is_hit = !strcmp(event, "HIT");
        int is_evict = !strcmp(event, "EVICT");
        int is_insert = !strcmp(event, "INSERT");

        if (is_hit) {
            if (slot >= c->cap || c->slot_eid[slot] != eid) { violation("hit_slot_mismatch", seq, cls, event, layer, eid, slot); continue; }
            c->used[slot] = ++g_clock;
            if (!is_pilot) {
                demand_hits++;
                if (tok >= 0) win_demand_hits++;
            }
        } else if (is_evict) {
            if (slot >= c->cap || c->slot_eid[slot] != eid) { violation("evict_slot_mismatch", seq, cls, event, layer, eid, slot); continue; }
            c->slot_eid[slot] = -1; c->fmt[slot] = 0;
            if (is_pilot) evict_pilot++; else evict_demand++;
        } else if (is_insert) {
            if (slot >= c->cap) { violation("insert_slot_range", seq, cls, event, layer, eid, slot); continue; }
            if (c->slot_eid[slot] != -1) { violation("insert_slot_busy", seq, cls, event, layer, eid, slot); continue; }
            /* informational: does this eid already live in another slot?
             * the current policy permits transient duplicates via the
             * pilot publish race — modeled, not treated as a violation */
            for (int i2 = 0; i2 < c->cap; i2++)
                if (i2 != slot && c->slot_eid[i2] == eid) { dup_resident_events++; break; }
            c->slot_eid[slot] = eid; c->fmt[slot] = fmt2; c->used[slot] = ++g_clock;
            if (is_pilot) { insert_pilot++; bytes_pilot += (unsigned long long)bytes; }
            else {
                insert_demand++; bytes_demand += (unsigned long long)bytes;
                demand_misses++;
                if (tok >= 0) win_demand_misses++;
            }
        } else {
            violation("unknown_event", seq, cls, event, layer, eid, slot);
        }
    }
    fclose(f);

    long occ_total = 0; int occ_max = 0;
    for (int l = 0; l < MAX_LAYERS; l++) {
        if (!g_lcs[l].cap) continue;
        int occ = 0;
        for (int i = 0; i < g_lcs[l].cap; i++) if (g_lcs[l].slot_eid[i] >= 0) occ++;
        occ_total += occ;
        if (occ > occ_max) occ_max = occ;
    }

    printf("replay rows processed: hits=%ld (window tok>=0: %ld) misses=%ld (window: %ld)\n",
           demand_hits, win_demand_hits, demand_misses, win_demand_misses);
    printf("demand hit rate: %.2f%% (cumulative) | %.2f%% (decode window)\n",
           (demand_hits + demand_misses) ? 100.0 * demand_hits / (demand_hits + demand_misses) : 0.0,
           (win_demand_hits + win_demand_misses) ? 100.0 * win_demand_hits / (win_demand_hits + win_demand_misses) : 0.0);
    printf("evictions: demand=%ld pilot=%ld | inserts: demand=%ld pilot=%ld | duplicate-resident inserts=%ld\n",
           evict_demand, evict_pilot, insert_demand, insert_pilot, dup_resident_events);
    printf("admitted bytes: demand=%llu (%.2f MB) pilot=%llu (%.2f MB)\n",
           bytes_demand, bytes_demand / 1048576.0, bytes_pilot, bytes_pilot / 1048576.0);
    printf("final occupancy: total=%ld layers=%d max/layer=%d cap=%d\n",
           occ_total, n_layers_seen, occ_max, cap);

    int fail = violations > 0;
    if (expect_hits >= 0 && demand_hits != expect_hits) {
        fprintf(stderr, "CONTROL MISMATCH: demand hits replay=%ld expected=%ld\n", demand_hits, expect_hits); fail = 1;
    }
    if (expect_misses >= 0 && demand_misses != expect_misses) {
        fprintf(stderr, "CONTROL MISMATCH: demand misses replay=%ld expected=%ld\n", demand_misses, expect_misses); fail = 1;
    }
    if (have_expect_bytes && bytes_demand != expect_bytes) {
        fprintf(stderr, "CONTROL MISMATCH: demand bytes replay=%llu expected=%llu\n", bytes_demand, expect_bytes); fail = 1;
    }
    if (expect_ed >= 0 && evict_demand != expect_ed) {
        fprintf(stderr, "CONTROL MISMATCH: demand evictions replay=%ld expected=%ld\n", evict_demand, expect_ed); fail = 1;
    }
    if (expect_ep >= 0 && evict_pilot != expect_ep) {
        fprintf(stderr, "CONTROL MISMATCH: pilot evictions replay=%ld expected=%ld\n", evict_pilot, expect_ep); fail = 1;
    }

    printf("VERDICT: %s\n", fail ? "TRACE_REPLAY_NOT_SELF_CONSISTENT" : "TRACE_REPLAY_SELF_CONSISTENT");
    return fail ? 1 : 0;
}
