/*
 * Pluggable expert-store backend registry — implementation.
 *
 * See expert_store_registry.h for the design. Registration happens from C
 * constructors at static-link time (before main), so the table is read-only
 * once the engine runs and no locking is needed.
 *
 * The table carries both interface generations per name: a backend may
 * register v1 (DeepSeek-coupled), v2 (descriptor-driven) or both.
 */

#include "expert_store_registry.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Keep the public registry symbols external even under -flto: backends live
 * in separately-compiled object files (e.g. a custom backend linked in later)
 * and call register() from their own constructors, so these symbols must
 * survive LTO inlining. Without this, gcc drops them when the only caller in
 * the current link is inside the same LTO set. */
#if defined(__GNUC__)
#define COLI_ESR_EXPORT __attribute__((externally_visible, used))
#else
#define COLI_ESR_EXPORT
#endif

/* The built-in on-disk/mmap backend lives in the COLI_V4_UNIT_EXPERT_STORE_AUTO
 * amalgamation unit of deepseek_v4.c and REGISTERS ITSELF from a constructor
 * there (both interfaces). This module stays engine-free: engines that do not
 * link DeepSeek (qwen36, olmoe) simply have no "auto" backend and select
 * whichever backends their binary carries. */

#define COLI_EXPERT_STORE_MAX_BACKENDS 8

typedef struct {
    const char *name;
    ColiExpertStoreBackendOpenFn open_fn;       /* v1 (may be NULL) */
    ColiExpertStoreBackendOpenV2Fn open_v2_fn;  /* v2 (may be NULL) */
} BackendEntry;

static BackendEntry g_backends[COLI_EXPERT_STORE_MAX_BACKENDS];
static int g_backend_count;

static int backend_slot(const char *name) {
    for (int i = 0; i < g_backend_count; i++)
        if (strcmp(g_backends[i].name, name) == 0) return i;
    return -1;
}

static int backend_put(const char *name, ColiExpertStoreBackendOpenFn v1,
                       ColiExpertStoreBackendOpenV2Fn v2) {
    if (!name || (!v1 && !v2)) return -1;
    int slot = backend_slot(name);
    if (slot >= 0) {
        /* last-wins override, per-interface */
        if (v1) g_backends[slot].open_fn = v1;
        if (v2) g_backends[slot].open_v2_fn = v2;
        return 0;
    }
    if (g_backend_count >= COLI_EXPERT_STORE_MAX_BACKENDS) return -1;
    g_backends[g_backend_count].name = name;
    g_backends[g_backend_count].open_fn = v1;
    g_backends[g_backend_count].open_v2_fn = v2;
    g_backend_count++;
    return 0;
}

COLI_ESR_EXPORT
int coli_expert_store_backend_register(const char *name,
                                       ColiExpertStoreBackendOpenFn open_fn) {
    return backend_put(name, open_fn, NULL);
}

COLI_ESR_EXPORT
int coli_expert_store_backend_register_v2(const char *name,
                                          ColiExpertStoreBackendOpenV2Fn open_fn) {
    return backend_put(name, NULL, open_fn);
}

COLI_ESR_EXPORT
int coli_expert_store_backend_count(void) {
    return g_backend_count;
}

COLI_ESR_EXPORT
ColiExpertStoreBackendOpenFn
coli_expert_store_backend_lookup(const char *name) {
    if (!name) return NULL;
    int slot = backend_slot(name);
    return slot >= 0 ? g_backends[slot].open_fn : NULL;
}

COLI_ESR_EXPORT
ColiExpertStoreBackendOpenV2Fn
coli_expert_store_backend_lookup_v2(const char *name) {
    if (!name) return NULL;
    int slot = backend_slot(name);
    return slot >= 0 ? g_backends[slot].open_v2_fn : NULL;
}

static const char *select_backend_name(char *buf, size_t bufsz) {
    const char *name = getenv("COLI_EXPERT_STORE");
    if (!name || !*name) name = "auto";
    snprintf(buf, bufsz, "%s", name);
    return buf;
}

COLI_ESR_EXPORT
int coli_expert_store_backend_open_selected(
    ColiV4Engine *engine,
    const ColiDeepSeekV4Config *config,
    const ColiDeepSeekV4ExpertStoreOptions *options,
    ColiExpertStore **output,
    char *error, size_t error_size) {
    char name[64];
    select_backend_name(name, sizeof(name));
    ColiExpertStoreBackendOpenFn fn = coli_expert_store_backend_lookup(name);
    if (!fn) {
        if (error && error_size)
            snprintf(error, error_size,
                     "expert store backend '%s' is not registered "
                     "(set COLI_EXPERT_STORE to a linked backend; "
                     "default is 'auto')",
                     name);
        return -1;
    }
    return fn(engine, config, options, output, error, error_size);
}

COLI_ESR_EXPORT
int coli_expert_store_open_descriptor(
    const ColiExpertStoreDescriptor *desc,
    ColiExpertStore **output,
    char *error, size_t error_size) {
    char fallback[64];
    const char *name = (desc && desc->backend_name && *desc->backend_name)
                           ? desc->backend_name
                           : select_backend_name(fallback, sizeof(fallback));
    ColiExpertStoreBackendOpenV2Fn fn = coli_expert_store_backend_lookup_v2(name);
    if (!fn) {
        if (error && error_size) {
            if (coli_expert_store_backend_lookup(name))
                snprintf(error, error_size,
                         "expert store backend '%s' only supports the legacy "
                         "engine-coupled open (no descriptor interface)",
                         name);
            else
                snprintf(error, error_size,
                         "expert store backend '%s' is not registered "
                         "(set COLI_EXPERT_STORE or desc.backend_name; "
                         "default is 'auto')",
                         name);
        }
        return -1;
    }
    return fn(desc, output, error, error_size);
}

/* The built-in "auto" backend registers itself from the DeepSeek link (see
 * deepseek_v4.c, COLI_V4_UNIT_EXPERT_STORE_AUTO): both interfaces under one
 * name. Engines without DeepSeek carry whichever backends they register. */
