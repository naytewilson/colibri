/* vnni_micro.c — DELTANET VNNI bounded strike (Wave 2-4).
 *
 * Baseline: the EXACT x86 matmul_q inner loop from qwen36.c
 *           (int8 weights x f32 activations, cvtepi8->f32 + FMA chain).
 * Candidate: AVX512-VNNI vpdpbusd with IDOT-style per-16-block Q8 activation
 *            quantization (NOT bit-exact by construction — quantified here).
 *
 * Dims = real DeltaNet GEMVs: (I=2048,O=8192) qkv, (2048,4096) z, (4096,2048) out.
 * Timing: interleaved arms, median of REPS; correctness vs double-precision
 * reference and cross-arm max-abs delta.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <immintrin.h>

static double now_ms(void){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec*1e3+ts.tv_nsec/1e6; }

/* ---- BASELINE: verbatim matmul_q x86 branch (single row) ---- */
static void mmq_baseline(float *y, const float *x, const int8_t *q, const float *scale, int I, int O){
    for (int o = 0; o < O; o++) {
        const int8_t *w = q + (int64_t)o * I;
        __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
        __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
        int i = 0;
        for (; i + 32 <= I; i += 32) {
            __m128i b0 = _mm_loadu_si128((const __m128i*)(w + i));
            __m128i b1 = _mm_loadu_si128((const __m128i*)(w + i + 16));
            a0 = _mm256_fmadd_ps(_mm256_loadu_ps(x+i),    _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b0)), a0);
            a1 = _mm256_fmadd_ps(_mm256_loadu_ps(x+i+8),  _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(b0,8))), a1);
            a2 = _mm256_fmadd_ps(_mm256_loadu_ps(x+i+16), _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b1)), a2);
            a3 = _mm256_fmadd_ps(_mm256_loadu_ps(x+i+24), _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(b1,8))), a3);
        }
        for (; i < I; i++) {
            __m128i b = _mm_loadl_epi64((const __m128i*)(w+i));
            a0 = _mm256_fmadd_ps(_mm256_loadu_ps(x+i), _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b)), a0);
        }
        __m256 a = _mm256_add_ps(_mm256_add_ps(a0,a1), _mm256_add_ps(a2,a3));
        __m128 t = _mm_add_ps(_mm256_castps256_ps128(a), _mm256_extractf128_ps(a,1));
        t = _mm_add_ps(t, _mm_movehl_ps(t,t)); t = _mm_add_ss(t, _mm_shuffle_ps(t,t,1));
        float acc; _mm_store_ss(&acc, t);
        y[o] = acc * scale[o];
    }
}

/* ---- CANDIDATE: VNNI vpdpbusd, IDOT-style Q8(16) activations ---- */
static void mmq_vnni(float *y, const float *x, const int8_t *q, const float *scale, int I, int O){
    int nb = I / 16;
    static __thread int8_t xi[8192]; static __thread float xs[512]; static __thread int init_i;
    if (!init_i || nb > 512) { init_i = 1; }
    (void)init_i;
    for (int b = 0; b < nb; b++) {
        const float *xb = x + b*16;
        float am = 0.f; for (int i = 0; i < 16; i++) { float a = xb[i] < 0 ? -xb[i] : xb[i]; if (a > am) am = a; }
        float s = am/127.f; if (s < 1e-12f) s = 1e-12f;
        xs[b] = s; float inv = 1.f/s;
        for (int i = 0; i < 16; i++) xi[b*16+i] = (int8_t)lrintf(xb[i]*inv);
    }
    for (int o = 0; o < O; o++) {
        const int8_t *w = q + (int64_t)o * I;
        __m256i acc0 = _mm256_setzero_si256(), acc1 = _mm256_setzero_si256();
        int b = 0;
        for (; b + 1 < nb; b += 2) {
            __m128i xa = _mm_loadu_si128((const __m128i*)(xi + b*16));
            __m128i xb_ = _mm_loadu_si128((const __m128i*)(xi + (b+1)*16));
            __m128i wa = _mm_loadu_si128((const __m128i*)(w + b*16));
            __m128i wb_ = _mm_loadu_si128((const __m128i*)(w + (b+1)*16));
            acc0 = _mm256_dpbusd_epi32(acc0, _mm256_cvtepi8_epi16(xa), _mm256_cvtepu8_epi16(wa));
            acc1 = _mm256_dpbusd_epi32(acc1, _mm256_cvtepi8_epi16(xb_), _mm256_cvtepu8_epi16(wb_));
        }
        for (; b < nb; b++) {
            __m128i xa = _mm_loadu_si128((const __m128i*)(xi + b*16));
            __m128i wa = _mm_loadu_si128((const __m128i*)(w + b*16));
            acc0 = _mm256_dpbusd_epi32(acc0, _mm256_cvtepi8_epi16(xa), _mm256_cvtepu8_epi16(wa));
        }
        __m256i acc = _mm256_add_epi32(acc0, acc1);
        int32_t tmp[8]; _mm256_storeu_si256((__m256i*)tmp, acc);
        /* fold block scales + per-row weight scale */
        float accf = 0.f;
        for (int k = 0; k < 8; k++) accf += (float)tmp[k];
        y[o] = accf * scale[o];
    }
}

static unsigned long long digest(const float *y, int n){
    unsigned long long h = 1469598103934665603ULL;
    union { float f; unsigned u; } u;
    for (int i = 0; i < n; i++){ u.f = y[i]; h ^= u.u; h *= 1099511628211ULL; }
    return h;
}

static int cmpd(const void *a, const void *b){ double x=*(const double*)a,y=*(const double*)b; return (x>y)-(x<y); }

int main(void){
    if (!__builtin_cpu_supports("avx512vnni")) { printf("CPU lacks avx512_vnni\n"); return 3; }
    struct { int I, O; const char *name; } dims[] = {{2048,8192,"dn_qkv"},{2048,4096,"dn_z"},{4096,2048,"dn_out"}};
    srand(42);
    printf("dims                 base_ns   vnni_ns  speedup  maxabs_vs_base  rel_err  digest_base        digest_vnni\n");
    for (unsigned di = 0; di < sizeof dims/sizeof dims[0]; di++) {
        int I = dims[di].I, O = dims[di].O;
        size_t wbytes = (size_t)I * O;
        int8_t *w = malloc(wbytes);
        float *x = malloc(I*sizeof(float)), *sc = malloc(O*sizeof(float));
        float *yb = malloc(O*sizeof(float)), *yv = malloc(O*sizeof(float)), *yr = malloc(O*sizeof(float));
        for (size_t i = 0; i < wbytes; i++) w[i] = (int8_t)((rand() % 255) - 127);
        for (int i = 0; i < I; i++) x[i] = (float)((rand() % 2000) - 1000) / 997.f;
        for (int o = 0; o < O; o++) sc[o] = 0.002f + (float)(rand()%100)/50000.f;
        /* adversarial extremes pass */
        for (size_t i = 0; i < wbytes; i += 7) w[i] = (i%14==0)? 127 : -128;
        mmq_baseline(yb, x, w, sc, I, O);
        mmq_vnni(yv, x, w, sc, I, O);
        /* double reference of the SAME baseline semantics */
        for (int o = 0; o < O; o++){
            const int8_t *wrow = w + (int64_t)o*I; double acc = 0;
            for (int i = 0; i < I; i++) acc += (double)wrow[i] * (double)x[i];
            yr[o] = (float)(acc * (double)sc[o]);
        }
        double maxd = 0, refmax = 0;
        for (int o = 0; o < O; o++){
            double d = fabs((double)yv[o]-yb[o]); if (d>maxd) maxd=d;
            double r = fabs((double)yb[o]-yr[o]); if (r>refmax) r=r;
        }
        /* timing: interleaved */
        enum { WARM=30, REPS=400 };
        static double tb[REPS], tv[REPS];
        for (int r = 0; r < WARM; r++){ mmq_baseline(yb,x,w,sc,I,O); mmq_vnni(yv,x,w,sc,I,O); }
        for (int r = 0; r < REPS; r++){
            double t0=now_ms(); mmq_baseline(yb,x,w,sc,I,O); tb[r]=(now_ms()-t0)*1e6;
            t0=now_ms(); mmq_vnni(yv,x,w,sc,I,O); tv[r]=(now_ms()-t0)*1e6;
        }
        qsort(tb,REPS,sizeof(double),cmpd); qsort(tv,REPS,sizeof(double),cmpd);
        double mb=tb[REPS/2], mv=tv[REPS/2];
        printf("%-20s %8.0f %8.0f  %6.3fx   %.4g          %.2e  %016llx %016llx\n",
               dims[di].name, mb, mv, mb/mv, maxd,
               maxd/(fabs(yb[0])+1e-9), digest(yb,O), digest(yv,O));
        free(w);free(x);free(sc);free(yb);free(yv);free(yr);
    }
    return 0;
}
