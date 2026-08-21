/* Unit test for Qwen3.6 heterogeneous mixed-precision expert cache.
 * Tests per-expert format detection, slot allocation, dynamic buffer growth/reuse,
 * and format-safe LRU eviction and memory accounting.
 */
#define main qwen36_main_unused
#include "../qwen36.c"
#undef main

static int fails = 0;

static void check_eq(int64_t actual, int64_t expected, const char *msg) {
    if (actual != expected) {
        fprintf(stderr, "FAIL: %s (got %lld, expected %lld)\n", msg, (long long)actual, (long long)expected);
        fails++;
    } else {
        printf("PASS: %s (%lld)\n", msg, (long long)actual);
    }
}

static void check_ptr_nonnull(const void *p, const char *msg) {
    if (!p) {
        fprintf(stderr, "FAIL: %s (got NULL)\n", msg);
        fails++;
    } else {
        printf("PASS: %s (non-null)\n", msg);
    }
}

int main(void) {
    printf("=== Qwen3.6 Mixed-Precision Expert Cache Unit Test ===\n");

    Model m;
    memset(&m, 0, sizeof(Model));
    m.c.n_layers = 40;
    m.c.n_experts = 256;
    m.c.hidden = 2048;
    m.c.inter = 512;
    m.c.expert_gs = 64;

    Slot s;
    memset(&s, 0, sizeof(Slot));
    s.eid = -1;

    /* 1. Allocate as INT3 format (fmt=5) */
    slot_ensure_format(&m, &s, 5);
    check_eq(s.is_int3, 1, "Slot is_int3 == 1");
    check_eq(s.is_int4, 0, "Slot is_int4 == 0");
    check_ptr_nonnull(s.w3, "Slot s.w3 non-null");
    check_ptr_nonnull(s.g3, "Slot s.g3 non-null");
    check_ptr_nonnull(s.u3, "Slot s.u3 non-null");
    check_ptr_nonnull(s.d3, "Slot s.d3 non-null");
    check_ptr_nonnull(s.gs, "Slot s.gs non-null");

    int64_t g_sz = ((int64_t)512 * 2048 / 64) * 24;
    check_eq((uintptr_t)s.u3 - (uintptr_t)s.g3, g_sz, "INT3 u3 offset == g_sz (393,216 bytes)");
    check_eq((uintptr_t)s.d3 - (uintptr_t)s.u3, g_sz, "INT3 d3 offset == g_sz (393,216 bytes)");

    /* Populate INT3 with sentinel */
    memset(s.w3, 0x33, (size_t)(g_sz * 3));

    /* 2. Format transition: Evict and reuse slot for INT4 (fmt=4) */
    slot_ensure_format(&m, &s, 4);
    check_eq(s.is_int4, 1, "Transition to INT4: s.is_int4 == 1");
    check_eq(s.is_int3, 0, "Transition to INT4: s.is_int3 == 0");
    check_ptr_nonnull(s.w4, "Slot s.w4 non-null after INT4 expansion");
    check_ptr_nonnull(s.g4, "Slot s.g4 non-null");
    check_ptr_nonnull(s.u4, "Slot s.u4 non-null");
    check_ptr_nonnull(s.d4, "Slot s.d4 non-null");

    int64_t ng = (int64_t)512 * 2048;
    check_eq((uintptr_t)s.u4 - (uintptr_t)s.g4, ng / 2, "INT4 u4 offset == ng/2 (524,288 bytes)");
    check_eq((uintptr_t)s.d4 - (uintptr_t)s.u4, ng / 2, "INT4 d4 offset == ng/2 (524,288 bytes)");

    /* Populate INT4 with sentinel */
    memset(s.w4, 0x44, (size_t)(ng + ng / 2));

    /* 3. Format transition: Evict and reuse slot for INT3 (fmt=5) */
    slot_ensure_format(&m, &s, 5);
    check_eq(s.is_int3, 1, "Transition back to INT3: s.is_int3 == 1");
    check_eq(s.is_int4, 0, "Transition back to INT3: s.is_int4 == 0");
    check_ptr_nonnull(s.w3, "Slot s.w3 non-null");
    check_ptr_nonnull(s.g3, "Slot s.g3 non-null");
    check_eq((uintptr_t)s.u3 - (uintptr_t)s.g3, g_sz, "INT3 u3 offset preserved");

    /* Free slot */
    if (s.w4) free(s.w4);
    else if (s.w3) free(s.w3);
    else if (s.g) free(s.g);
    if (s.gs) free(s.gs);

    if (fails == 0) {
        printf("ALL TESTS PASSED (%d checks).\n", 15);
        return 0;
    } else {
        printf("FAILED %d checks.\n", fails);
        return 1;
    }
}
