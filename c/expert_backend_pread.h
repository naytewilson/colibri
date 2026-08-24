#ifndef COLIBRI_EXPERT_BACKEND_PREAD_H
#define COLIBRI_EXPERT_BACKEND_PREAD_H

/*
 * Shared pread/LRU expert backend (Forge F1 Phase 3).
 *
 * One implementation of the full ExpertStore seam — lease lookup, advisory
 * prefetch, atomic residency-coupled reservations with direct-write segment
 * buffers, pin/unpin eviction protection, batch lookup and widened stats —
 * extracted from the qwen36 out-of-core machinery (with olmoe.c cross-check)
 * and generalized:
 *   - keyed internally on (layer, role, index); role=EXPERT first-class;
 *   - tensor names come from caller-supplied printf templates in the
 *     descriptor, never hardcoded model specifics;
 *   - format classes are classified from the container itself
 *     (bytes/numel ratio: 1 -> INT8, 1/2 -> INT4, 24/64 -> INT3-g64),
 *     mirroring the proven expert_fmt size logic without Qwen geometry;
 *   - one global mutation lock (the qwen36 g_pilot_mx pattern); pread I/O
 *     runs unlocked so device transfers never serialize compute threads;
 *   - LRU victim scan skips pinned and reserved slots; the all-reserved
 *     spin-wait fallback is preserved verbatim in semantics (never steal a
 *     buffer an unlocked pread owns);
 *   - fused single-pread [scales][weights] admission when configured and
 *     the pair is verified contiguous zero-gap per tensor.
 */

#include <stddef.h>
#include <stdint.h>

#include "expert_store.h"
#include "expert_store_registry.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Open a descriptor-driven pread/LRU store. The descriptor must carry
 * n_layers/n_experts, either shard_table (borrowed, already indexed) or
 * storage_path (indexed at open), weights_name_template, and optionally
 * scales_name_template / capacity_bytes. See expert_store_registry.h for
 * the field contracts. Zero on success; non-zero with a message on failure.
 * The returned store implements every ColiExpertStoreOps v1+v1.1 entry. */
int coli_expert_backend_pread_open(const ColiExpertStoreDescriptor *desc,
                                   ColiExpertStore **output,
                                   char *error, size_t error_size);

/* ColiAdmissionLoadFn-shaped loader for this backend: preads the container
 * bytes into an active reservation's segments (fused path honored when
 * configured). Pass the ColiExpertStore* as userdata. Zero on success. */
int coli_expert_backend_pread_load(void *userdata,
                                   const ColiExpertCoreKey *key,
                                   ColiExpertReservation *res);

/* Register this backend under `name` (constructor-style helper for links
 * that want it selectable through COLI_EXPERT_STORE). */
int coli_expert_backend_pread_register(const char *name);

#ifdef __cplusplus
}
#endif
#endif /* COLIBRI_EXPERT_BACKEND_PREAD_H */
