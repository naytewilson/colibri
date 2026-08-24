#ifndef COLIBRI_EXPERT_STORE_REGISTRY_H
#define COLIBRI_EXPERT_STORE_REGISTRY_H

/*
 * Pluggable expert-store backend registry (model-neutral since Forge F1).
 *
 * A backend registers a name plus an open function and is selected either
 * explicitly (descriptor field / COLI_EXPERT_STORE environment variable) or
 * by the default name "auto". Backends ship as separate object files and
 * register themselves at link time via a C constructor, so an engine source
 * carries no backend-specific branching — adding a backend is a link-time
 * concern, not an engine edit.
 *
 * Two interface generations coexist:
 *  - v1 ColiExpertStoreBackendOpenFn: the original DeepSeek-V4-coupled open
 *    (engine + config + options pointers). Retained untouched for zero drift;
 *    the built-in "auto" backend still speaks it.
 *  - v2 ColiExpertStoreBackendOpenV2Fn: opens from a model-neutral
 *    ColiExpertStoreDescriptor. New backends implement this; engines build a
 *    descriptor and call coli_expert_store_open_descriptor().
 *
 * The built-in "auto" backend answers BOTH interfaces: its v2 entry adapts
 * the descriptor to the v1 planned-open through the descriptor's backend
 * handle fields (engine_context/backend_config/backend_options), which the
 * DeepSeek adapter (coli_dsv4_build_descriptor, deepseek_v4.c) fills.
 */

#include <stddef.h>
#include "expert_store.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Legacy v1 pointer types. Forward-declared here (opaque). ColiDeepSeekV4Config
 * must be skipped when its concrete header is present: deepseek_v4.h typedefs
 * an ANONYMOUS struct as ColiDeepSeekV4Config, so a blind forward typedef
 * would collide. Engine/Options are tagged structs; repeating their forward
 * typedefs is legal C11 and keeps standalone consumers compiling.
 * Concrete fields live in deepseek_v4.h/deepseek_v4_internal.h, which
 * backends that need them include themselves. */
typedef struct ColiV4Engine ColiV4Engine;
typedef struct ColiDeepSeekV4ExpertStoreOptions ColiDeepSeekV4ExpertStoreOptions;
#ifndef COLIBRI_DEEPSEEK_V4_H
typedef struct ColiDeepSeekV4Config ColiDeepSeekV4Config;
#endif

/* Open a store for the given DeepSeek-V4 engine + config + options (v1).
 * Same contract as documented in deepseek_v4.h: zero on success with
 * *output written, non-zero on failure with a message. Requires a non-NULL
 * engine for the built-in "auto" backend. */
typedef int (*ColiExpertStoreBackendOpenFn)(
    ColiV4Engine *engine,
    const ColiDeepSeekV4Config *config,
    const ColiDeepSeekV4ExpertStoreOptions *options,
    ColiExpertStore **output,
    char *error, size_t error_size);

/*
 * Model-neutral expert-store descriptor (v2 open contract input).
 * Geometry + storage + budget describe WHAT to cache; scheduler_config
 * carries policy knobs owned by the admission layer (Phase 4). Backend-
 * private handles carry whatever a concrete backend additionally needs —
 * they are opaque to the registry and to model-neutral callers.
 */
typedef struct {
    /* Backend selection. Empty string -> COLI_EXPERT_STORE env -> "auto". */
    const char *backend_name;

    /* Geometry. */
    int n_layers;              /* MoE layers addressed by the store */
    int n_experts;             /* experts per layer */
    uint64_t capacity_bytes;   /* target resident-expert budget */

    /* Storage. Either a path to the container directory or an already-opened
     * shard table handle; at most one is meaningful per backend. */
    const char *storage_path;
    void *shard_table;         /* opaque (st.h shards*) when pre-opened */

    /* Policy knobs consumed by the admission-scheduler layer; NULL keeps
     * every knob at its default-OFF value. */
    void *scheduler_config;

    /* Backend-private handles (opaque). For the built-in "auto" backend:
     * engine_context = ColiV4Engine*, backend_config = ColiDeepSeekV4Config*,
     * backend_options = options. Other backends define their own contracts. */
    void *engine_context;
    void *backend_config;
    void *backend_options;
} ColiExpertStoreDescriptor;

/* Open a store from a model-neutral descriptor (v2). Zero on success with
 * *output written; non-zero on failure with a message. */
typedef int (*ColiExpertStoreBackendOpenV2Fn)(
    const ColiExpertStoreDescriptor *desc,
    ColiExpertStore **output,
    char *error, size_t error_size);

/* --- registration ------------------------------------------------------- */

/* Register a legacy (v1) backend under `name`. Intended to be called from a
 * constructor in the backend's object file. Names match case-sensitively; a
 * later registration with the same name replaces the earlier (last-wins).
 * Returns 0 on success, -1 if name/open_fn is NULL or the registry is full. */
int coli_expert_store_backend_register(const char *name,
                                       ColiExpertStoreBackendOpenFn open_fn);

/* Register a descriptor-driven (v2) backend. Same rules as above. */
int coli_expert_store_backend_register_v2(const char *name,
                                          ColiExpertStoreBackendOpenV2Fn open_fn);

/* Number of backends currently registered (tests/diagnostics). */
int coli_expert_store_backend_count(void);

/* Look up a backend's v1 open function by name, or NULL if not registered. */
ColiExpertStoreBackendOpenFn
coli_expert_store_backend_lookup(const char *name);

/* Look up a backend's v2 open function by name, or NULL if not registered. */
ColiExpertStoreBackendOpenV2Fn
coli_expert_store_backend_lookup_v2(const char *name);

/* --- opening ------------------------------------------------------------ */

/* v1 entry point (legacy): open via the backend selected by the
 * COLI_EXPERT_STORE env var (default "auto"). Unchanged behavior. */
int coli_expert_store_backend_open_selected(
    ColiV4Engine *engine,
    const ColiDeepSeekV4Config *config,
    const ColiDeepSeekV4ExpertStoreOptions *options,
    ColiExpertStore **output,
    char *error, size_t error_size);

/* v2 entry point: open via desc->backend_name (empty -> COLI_EXPERT_STORE
 * env -> "auto"). Fails cleanly when the selected backend exists only as a
 * v1 registration (a v1 backend cannot consume a generic descriptor). */
int coli_expert_store_open_descriptor(
    const ColiExpertStoreDescriptor *desc,
    ColiExpertStore **output,
    char *error, size_t error_size);

#ifdef __cplusplus
}
#endif
#endif /* COLIBRI_EXPERT_STORE_REGISTRY_H */
