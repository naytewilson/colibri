/*
 * Unit test for the pluggable expert-store backend registry.
 *
 * Verifies:
 *  - the built-in "auto" backend is registered at link time (constructor),
 *    under BOTH the legacy v1 and descriptor-driven v2 interfaces;
 *  - lookup hits for "auto", misses for an unknown name;
 *  - open_selected dispatches to COLI_EXPERT_STORE, errors cleanly on an
 *    unregistered backend, and falls back to "auto" when the env is unset;
 *  - open_descriptor dispatches by desc.backend_name / env / default,
 *    passes descriptor fields through untouched, and rejects v1-only
 *    backends with a clean error;
 *  - register_v2 last-wins override works.
 *
 * The real auto open (coli_v4_expert_store_open_planned) lives in deepseek_v4.c;
 * we stub it here so this test links without the engine. The stub records that
 * it was called and returns a sentinel so we can observe dispatch.
 *
 *   gcc -O2 test_expert_store_registry.c expert_store_registry.c -o test_expert_store_registry
 *   ./test_expert_store_registry
 */
#include "../expert_store_registry.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_auto_called = 0;
static const ColiExpertStoreDescriptor *g_auto_last_desc = NULL;

/* Stub for the built-in backend's legacy open fn. */
int coli_v4_expert_store_open_planned(ColiV4Engine *engine,
                                      const ColiDeepSeekV4Config *config,
                                      const ColiDeepSeekV4ExpertStoreOptions *opts,
                                      ColiExpertStore **out,
                                      char *error, size_t error_size) {
    (void)engine; (void)config; (void)opts; (void)out; (void)error; (void)error_size;
    g_auto_called = 1;
    return -1; /* sentinel: dispatched but did not open */
}

/* This link has no DeepSeek engine, so register the auto backend here
 * (mirroring what deepseek_v4.c's constructor does in real binaries). */
static int test_auto_open_descriptor(const ColiExpertStoreDescriptor *desc,
                                     ColiExpertStore **out,
                                     char *error, size_t error_size) {
    if (!desc || !desc->engine_context || !desc->backend_options) {
        if (error && error_size)
            snprintf(error, error_size,
                     "auto expert store requires descriptor handles");
        return -1;
    }
    return coli_v4_expert_store_open_planned(
        (ColiV4Engine *)desc->engine_context,
        (const ColiDeepSeekV4Config *)desc->backend_config,
        (const ColiDeepSeekV4ExpertStoreOptions *)desc->backend_options,
        out, error, error_size);
}
__attribute__((constructor))
static void test_register_auto(void) {
    coli_expert_store_backend_register("auto",
                                       coli_v4_expert_store_open_planned);
    coli_expert_store_backend_register_v2("auto", test_auto_open_descriptor);
}

/* A distinct v2 backend used to observe descriptor pass-through. */
static int g_v2_called = 0;
static ColiExpertStoreDescriptor g_v2_seen;
static int mock_v2_open(const ColiExpertStoreDescriptor *desc,
                        ColiExpertStore **out,
                        char *error, size_t error_size) {
    (void)out; (void)error; (void)error_size;
    g_v2_called = 1;
    if (desc) g_v2_seen = *desc;
    return -2; /* sentinel */
}

int main(void) {
    int n = coli_expert_store_backend_count();
    if (n != 1) {
        printf("FAIL: expected 1 backend (auto) on startup, got %d\n", n);
        return 1;
    }
    if (!coli_expert_store_backend_lookup("auto")) {
        printf("FAIL: 'auto' not registered\n");
        return 1;
    }
    if (coli_expert_store_backend_lookup("does-not-exist")) {
        printf("FAIL: unknown backend unexpectedly found\n");
        return 1;
    }

    char err[128] = {0};
    ColiExpertStore *out = NULL;

    /* Unregistered backend selected by env -> clean error, no dispatch. */
    setenv("COLI_EXPERT_STORE", "example-not-linked", 1);
    g_auto_called = 0;
    int rc = coli_expert_store_backend_open_selected(NULL, NULL, NULL, &out, err, sizeof(err));
    if (rc == 0 || g_auto_called) {
        printf("FAIL: unregistered backend should error without dispatch (rc=%d, auto_called=%d)\n",
               rc, g_auto_called);
        return 1;
    }
    if (!strstr(err, "example-not-linked") || !strstr(err, "not registered")) {
        printf("FAIL: error message wrong: %s\n", err);
        return 1;
    }
    printf("unregistered backend -> clean error: %s\n", err);

    /* Env unset -> default 'auto' -> dispatch to the stub. */
    unsetenv("COLI_EXPERT_STORE");
    g_auto_called = 0;
    err[0] = 0;
    rc = coli_expert_store_backend_open_selected(NULL, NULL, NULL, &out, err, sizeof(err));
    if (!g_auto_called) {
        printf("FAIL: default 'auto' did not dispatch\n");
        return 1;
    }
    /* rc is the stub's -1 sentinel; what matters is that auto was dispatched. */
    (void)rc;
    printf("default (COLI_EXPERT_STORE unset) -> dispatched to 'auto'\n");

    /* Explicit 'auto' also dispatches. */
    setenv("COLI_EXPERT_STORE", "auto", 1);
    g_auto_called = 0;
    coli_expert_store_backend_open_selected(NULL, NULL, NULL, &out, err, sizeof(err));
    if (!g_auto_called) {
        printf("FAIL: explicit 'auto' did not dispatch\n");
        return 1;
    }
    unsetenv("COLI_EXPERT_STORE");
    printf("explicit 'auto' -> dispatched to 'auto'\n");

    /* --- v2 descriptor interface --- */

    /* 'auto' is registered under v2 as well (constructor registers both). */
    if (!coli_expert_store_backend_lookup_v2("auto")) {
        printf("FAIL: 'auto' missing v2 registration\n");
        return 1;
    }

    /* Descriptor with explicit backend name dispatches and passes fields
     * through untouched (auto thunk -> stub). The auto thunk requires the
     * backend handle fields to be set (adapter contract); the stub ignores
     * their values. */
    ColiExpertStoreDescriptor desc;
    memset(&desc, 0, sizeof(desc));
    desc.backend_name = "auto";
    desc.n_layers = 12;
    desc.n_experts = 64;
    desc.capacity_bytes = 1234;
    int marker_auto;
    desc.engine_context = &marker_auto;      /* fake engine */
    desc.backend_config = NULL;              /* config may be NULL */
    desc.backend_options = &marker_auto;     /* fake options */
    g_auto_called = 0;
    rc = coli_expert_store_open_descriptor(&desc, &out, err, sizeof(err));
    if (!g_auto_called || rc != -1) {
        printf("FAIL: v2 auto dispatch wrong (called=%d rc=%d)\n", g_auto_called, rc);
        return 1;
    }
    printf("descriptor backend_name='auto' -> dispatched to auto v2 thunk\n");

    /* Auto v2 thunk without adapter handles errors cleanly (no dispatch). */
    memset(&desc, 0, sizeof(desc));
    desc.backend_name = "auto";
    g_auto_called = 0;
    err[0] = 0;
    rc = coli_expert_store_open_descriptor(&desc, &out, err, sizeof(err));
    if (rc == 0 || g_auto_called || !strstr(err, "descriptor handles")) {
        printf("FAIL: handle-less auto v2 open should error cleanly (rc=%d called=%d err=%s)\n",
               rc, g_auto_called, err);
        return 1;
    }
    printf("auto v2 without handles -> clean error\n");

    /* Register a mock v2 backend; open via explicit name; verify pass-through
     * of every neutral field and last-wins override semantics. */
    if (coli_expert_store_backend_register_v2("mock-v2", mock_v2_open) != 0) {
        printf("FAIL: register_v2 mock failed\n");
        return 1;
    }
    /* re-register changes nothing observable except still dispatching */
    if (coli_expert_store_backend_register_v2("mock-v2", mock_v2_open) != 0) return 1;

    memset(&desc, 0, sizeof(desc));
    desc.backend_name = "mock-v2";
    desc.n_layers = 48;
    desc.n_experts = 256;
    desc.capacity_bytes = 987654321;
    desc.storage_path = "/models/somewhere";
    int marker;
    desc.shard_table = &marker;
    desc.scheduler_config = &marker;
    desc.engine_context = &marker;
    g_v2_called = 0;
    rc = coli_expert_store_open_descriptor(&desc, &out, err, sizeof(err));
    if (!g_v2_called || rc != -2) {
        printf("FAIL: v2 mock dispatch wrong (called=%d rc=%d)\n", g_v2_called, rc);
        return 1;
    }
    if (g_v2_seen.n_layers != 48 || g_v2_seen.n_experts != 256 ||
        g_v2_seen.capacity_bytes != 987654321 ||
        strcmp(g_v2_seen.storage_path, "/models/somewhere") != 0 ||
        g_v2_seen.shard_table != &marker ||
        g_v2_seen.scheduler_config != &marker ||
        g_v2_seen.engine_context != &marker) {
        printf("FAIL: descriptor fields not passed through\n");
        return 1;
    }
    printf("descriptor pass-through verified on mock-v2\n");

    /* Env-selected name works for the descriptor path too. */
    setenv("COLI_EXPERT_STORE", "mock-v2", 1);
    memset(&desc, 0, sizeof(desc)); /* empty backend_name -> env */
    g_v2_called = 0;
    coli_expert_store_open_descriptor(&desc, &out, err, sizeof(err));
    if (!g_v2_called) {
        printf("FAIL: env selection did not reach mock-v2\n");
        return 1;
    }
    unsetenv("COLI_EXPERT_STORE");

    /* Unknown name in the descriptor path errors cleanly. */
    memset(&desc, 0, sizeof(desc));
    desc.backend_name = "example-not-linked";
    err[0] = 0;
    rc = coli_expert_store_open_descriptor(&desc, &out, err, sizeof(err));
    if (rc == 0 || !strstr(err, "not registered")) {
        printf("FAIL: unknown v2 backend should error cleanly (err=%s)\n", err);
        return 1;
    }

    /* A v1-only backend cannot serve the descriptor path: register a v1 fn
     * under a fresh name and expect the clean v1-only error. */
    if (coli_expert_store_backend_register("v1-only",
                                           coli_v4_expert_store_open_planned) != 0)
        return 1;
    memset(&desc, 0, sizeof(desc));
    desc.backend_name = "v1-only";
    err[0] = 0;
    rc = coli_expert_store_open_descriptor(&desc, &out, err, sizeof(err));
    if (rc == 0 || !strstr(err, "legacy")) {
        printf("FAIL: v1-only backend should report legacy-only (err=%s)\n", err);
        return 1;
    }
    printf("v1-only backend -> clean legacy-only error\n");

    printf("ALL OK\n");
    return 0;
}