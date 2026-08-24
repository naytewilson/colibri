/* l3_tracecmp — native comparator for two ling3.c L3_TRACE f32 dumps.
 *
 * The trace stream is raw little-endian f32, one record per (token, stage):
 * hidden state after every layer's MLP residual plus a final post-norm
 * record. Two runs over the SAME frozen prompt with the same L3_LAYERS /
 * teacher-forcing configuration produce equal-length streams, so kernel A/B
 * adjudication (e.g. scalar vs AVX2 expert matvec, or L3_NO_AVX2 fallback)
 * is an element-wise comparison:
 *
 *   ./l3_tracecmp <trace_a> <trace_b> [T tokens] [H dim] [n_layers]
 *
 * With T/H/L given, diffs are grouped per stage boundary (H floats per
 * record, T records per stage) so a divergence can be pinned to ONE layer;
 * without them it degrades to a whole-stream verdict. Exit 0 = compared,
 * exit 1 = usage/size mismatch. The numeric verdict is printed, not
 * thresholded: the caller classifies (drift ~1e-5 == fp32 reduction-order
 * noise; structural disagreement == real defect — logit-level argmax parity
 * stays the job of ling3_check against a real fixture).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

static float *slurp(const char *path, int64_t *n_out) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz % 4) { fprintf(stderr, "%s: size %ld not a multiple of 4 (f32)\n", path, sz); exit(1); }
    float *buf = malloc((size_t)sz);
    if (!buf) { fprintf(stderr, "OOM %ld\n", sz); exit(1); }
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) { fprintf(stderr, "%s: short read\n", path); exit(1); }
    fclose(f);
    *n_out = sz / 4;
    return buf;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <trace_a> <trace_b> [T tokens] [H dim] [n_stages]\n", argv[0]);
        return 1;
    }
    int64_t na, nb;
    float *a = slurp(argv[1], &na);
    float *b = slurp(argv[2], &nb);
    if (na != nb) {
        fprintf(stderr, "SIZE MISMATCH: %lld vs %lld floats — runs are not comparable\n",
                (long long)na, (long long)nb);
        return 1;
    }
    int64_t T = argc > 3 ? atoll(argv[3]) : 0;
    int64_t H = argc > 4 ? atoll(argv[4]) : 0;
    int64_t stages = argc > 5 ? atoll(argv[5]) : 1;

    double gmax = 0; int64_t garg = -1;
    if (T > 0 && H > 0 && na == T * H * stages) {
        /* per-stage report: pin any divergence to its layer */
        for (int64_t s = 0; s < stages; s++) {
            double smax = 0; int64_t sarg = -1;
            for (int64_t r = 0; r < T; r++)
                for (int64_t d = 0; d < H; d++) {
                    int64_t i = (s * T + r) * H + d;
                    double diff = fabs((double)a[i] - (double)b[i]);
                    if (diff > smax) { smax = diff; sarg = i; }
                }
            printf("stage %lld/%lld: max|diff| = %.6g @float %lld\n",
                   (long long)s + 1, (long long)stages, smax, (long long)sarg);
            if (smax > gmax) { gmax = smax; garg = sarg; }
        }
    } else {
        if (T > 0 && H > 0)
            fprintf(stderr, "note: %lld floats != T*H*stages (%lld*%lld*%lld) — whole-stream mode\n",
                    (long long)na, (long long)T, (long long)H, (long long)stages);
        for (int64_t i = 0; i < na; i++) {
            double diff = fabs((double)a[i] - (double)b[i]);
            if (diff > gmax) { gmax = diff; garg = i; }
        }
    }
    printf("GLOBAL max|diff| = %.6g @float %lld (of %lld)\n",
           gmax, (long long)garg, (long long)na);
    return 0;
}
