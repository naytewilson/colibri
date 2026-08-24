/* Ling 3.0 Tiny inference engine in pure C — sibling of kimi_k3.c / olmoe.c /
 * inkling.c, sharing st.h / json.h / tok.h / quant.h.
 *
 * Architecture (inclusionAI/Ling-3.0-tiny, bailing_hybrid /
 * BailingMoeV3ForCausalLM, pinned BF16 revision b61f4338de3e68ffc9c0bc1ed5e902981a4a929e,
 * INT4 revision 65a6d1d71e01f73ba01e572992bbd69ea92c865f):
 * 7.9B total / 1.3B active params.
 *   - 24 layers: 18 KDA + 6 gated MLA ((i+1)%layer_group_size==0 -> MLA),
 *     hidden 1536, RMSNorm eps 1e-6, NO AttnRes (plain residual stream).
 *   - KDA (fla fused_recurrent/chunk_kda semantics, head_dim 128, 16 heads):
 *       q,k,v = SiLU(CausalConv4(W{q,k,v} x));  q,k L2-normalized (eps 1e-6
 *       inside sqrt), q *= 128^-0.5
 *       z = W_f x + dt_bias;  gk = lb * sigmoid(exp(A_log[h]) * z), lb=-5
 *       alpha = exp(gk);  beta = sigmoid(W_b x)
 *       S[k][:] *= alpha[k]; vt = beta*(v - sum_k S'[k][:] k[k]);
 *         S[k][:] += k[k]*vt[:]; o[v] = sum_k q[k] S_new[k][v]
 *       out = W_o [ sigmoid(W_g x) * RMSNorm_head(o) ]        (full-rank gate)
 *     A_log[16], dt_bias[2048] are F32 in the checkpoint; f_proj/g_proj are
 *     DIRECT projections (no_kda_lora=true).
 *   - Gated MLA (partial RoPE, rope_interleave=true, rotary_dim 64 of qk 192,
 *     scale 192^-0.5, rope_theta 6e6):
 *       qa = W_qb(RMSNorm(W_qa x));  q split [128 nope | 64 rope]
 *       ckv = W_kv_a x;  L = RMSNorm(ckv[:512]) cached; k_rot = ckv[512:]
 *       k,v per head = split(W_kv_b L); k_rot shared across heads (MQA part)
 *       interleaved-pair RoPE applied to q_rot(pos) and cached k_rot(pos);
 *       ctx[h] *= sigmoid(W_g x)[h]      (gate is HEAD-WISE: [H], NOT [H*vhd])
 *       out = W_dense(ctx)
 *   - MoE (layers >= first_k_dense_replace=1; BailingMoeV3 MLPs, plain SiLU):
 *       logits = x_f32 @ gate.W_f32.T; scores = sigmoid(logits)
 *       routing = scores + expert_bias
 *       grouped top-k (noaux_tc): per group of n_experts/n_group take top-2
 *         routing values, pick top-4 groups, top-8 experts among survivors
 *       weights = ORIGINAL scores of selected, /(sum+1e-20), * routed_scaling
 *       y = sum_k w_k Expert_k(x) + Shared(x); Expert = down(silu(gate)*up)
 *   - Layer 0 dense MLP (intermediate 4608), same SiLU form. Final RMSNorm,
 *     untied lm_head, embedding model.word_embeddings.weight.
 *
 * Routed experts are consumed NATIVELY from the official INT4 checkpoint
 * (compressed-tensors pack-quantized int4-g32 symmetric):
 *     weight_packed I32 [O, I/8]: dense little-endian 4-bit bitstream, element
 *       i of each row at bit 4*i (word i/8, nibble i%8), stored unsigned
 *       (q + 8), q in [-8,7];
 *     weight_scale BF16 [O, I/32] -> converted to f32 once at load;
 *     weight_shape I64 [2].
 * Layout verified against the official BF16 checkpoint (cosine 0.9953,
 * relerr 10.2% == expected g32 noise). Packed words are kept AS STORED;
 * decode never re-expands tensors. Everything else loads from the BF16
 * snapshot and is either kept f32 or quantized at load time (L3_BITS).
 *
 * FULL RESIDENCY: every expert matrix of every layer is loaded once at startup
 * into resident buffers behind a direct [layer][expert][matrix] pointer table.
 * Steady-state decode performs zero model-weight file reads, zero tensor-name
 * lookups, zero locking. NOT yet zero-allocation: the Wave-1 AVX2 exp_matvec
 * allocates/frees its x-even/x-odd scratch on every call and moe_forward /
 * attention keep small per-call temporaries. A persistent scratch/workspace
 * pool is the next exact optimization wave; do not claim a zero-alloc hot
 * path until that lands.
 *
 * ENV:
 *   L3_BITS=4|8|32        load-time quant of attention/dense/shared/embed (def 32)
 *   L3_HEAD_BITS=4|8|32   lm_head (default follows L3_BITS)
 *   L3_LAYERS=N           truncate to first N layers (validation)
 *   L3_TRACE=path         dump f32 hidden state after every layer
 *   L3_LOGITS=path        dump f32 logits per PREFILL position
 *   L3_ROUTE=path         dump router decisions (layer,t,idx[topk],w[topk])
 *   L3_X0=path            inject f32 hidden states as layer inputs (validation)
 *   L3_CHUNK=N            prefill chunk size (default 32)
 *   L3_THREADS=N          generic thread count (prefill AND decode unless the
 *                         specific var below overrides)
 *   L3_THREADS_PREF=N     prefill threads only (default: omp_get_max_threads())
 *   L3_THREADS_DEC=N      decode threads only  (default: min(8, max_threads);
 *                         8 = measured Dell i7-11700 knee, NOT a universal
 *                         constant). Invalid values (<=0) are rejected.
 *   L3_PHASES=0|1         phase decomposition timers to stderr (default 0)
 *   L3_MAXT=N             KV/context capacity (default prompt+ngen+8)
 *   COLI_TEMP=F           0 = greedy (default)
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#if defined(__APPLE__) || defined(__linux__) || defined(__FreeBSD__)
#include <sys/resource.h>
#include <unistd.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif
#include "st.h"
#include "tok.h"
#include "quant.h"

/* ---------- config ---------- */
typedef struct {
    int hidden, n_layers, vocab, first_dense, dense_inter;
    /* MLA */
    int n_heads, q_lora, kv_lora, qk_nope, qk_rope, qk_head, v_head;
    float attn_scale, rope_theta;
    int rotary_dim;
    /* KDA */
    int kda_heads, kda_hd, kda_proj, conv_k;
    float gate_lb;
    /* MoE */
    int n_experts, topk, moe_inter, sh_inter;
    int n_group, topk_group;
    float routed_scale;
    float eps;
    int8_t is_mla[128];
    int bos, eos[8], n_eos;
} Cfg;

/* ---------- RAM-resident weight, quantized at load ---------- */
typedef struct { int fmt; float *f; int8_t *q8; uint8_t *q4; float *s; int O, I, gs; } W;

/* native compressed-tensors int4-g32 matrix, kept in the container layout.
 * dense != NULL selects the EXACT track (bf16->f32 experts, no quantization):
 * used by the correctness ladder so quantization drift never masquerades as
 * a math bug; the serving artifact uses the native packed path. */
typedef struct {
    uint32_t *packed;                       /* [O, I/8] words as stored on disk */
    float *scl;                             /* [O, I/32] f32 (converted from bf16) */
    float *dense;                           /* [O, I] f32 exact-track alternative */
    int O, I, nw, ng;
} Exp;

typedef struct {                          /* KDA layer */
    W q, k, v, o, g, f;
    float *conv_q, *conv_k, *conv_v;      /* [proj*4] depthwise taps */
    float *bp;                            /* beta proj f32 [heads,hidden] */
    float *dt, *A, *onw;                  /* dt_bias[proj], exp(A_log)[heads], o_norm[hd] */
} Kda;

typedef struct {                          /* gated partial-RoPE MLA layer */
    W qa, qb, kva, kvb, o, g;
    float *qa_ln, *kva_ln;
} Mla;

typedef struct {                          /* BailingMoeV3 MoE */
    float *router, *rbias;                /* [E,hidden] f32, [E] f32 */
    W sh_gate, sh_up, sh_down;
    Exp *exps;                            /* [E*3] gate/up/down per expert */
} Moe;

typedef struct {
    int mla, sparse;
    Kda a; Mla m; Moe moe;
    W d_gate, d_up, d_down;               /* dense layer only */
    float *in_ln, *post_ln;
} Layer;

typedef struct {
    Cfg c;
    shards S;
    Layer *L;
    float *final_norm;
    W embed, lm_head;
    float **kstate;                       /* [layer] -> [heads*hd*hd], S[k][v] */
    float **cwq, **cwk, **cwv;            /* conv windows [proj*conv_k], oldest slot first */
    float **Lc, **Rc; int max_t;          /* MLA cache: latent rows + ROTATED k_rot rows */
    float *cos_t, *sin_t;                 /* [max_t][rotary_dim/2] */
    int64_t wbytes;                       /* resident model bytes */
    double t_attn, t_router, t_expert, t_shared, t_norm, t_head, t_embed;
    /* STRIKE 1: attention sub-walls (KDA/MLA and inner stages) */
    double t_kda, t_mla;
    double t_kda_proj, t_kda_conv, t_kda_rec, t_kda_out;
    double t_mla_proj, t_mla_abs, t_mla_ctx, t_mla_val, t_mla_out;
    FILE *trace, *routef;
    /* STRIKE 3 persistent decode/prefill workspace (sized once at kv_alloc) */
    struct {
        int cmax;
        float *hidden,*nrm,*att,*mlp;                       /* [cmax][D] */
        float *kq,*kk_,*kv,*kgp,*kon,*kgraw,*kbraw;         /* KDA chunk temps */
        float *qa,*qv,*ckv,*gv,*ctx;                        /* MLA chunk temps */
        float *moe_U; int *moe_idx; float *moe_w;           /* MoE route arrays */
        float *gate,*up,*hz;                                /* expert temps */
        float *xev,*xod;                                    /* AVX2 split (max I/2) */
        float *logits;                                      /* [V] */
        float *uid_f,*wlist,*poslist_f;                     /* MoE union scratch (int-packed) */
        int *uid,*pcnt,*pfirst,*poslist,*cur;
        float *sg,*su,*sd;                                  /* shared expert */
        float *dg,*du;                                      /* dense layer */
    } ws;
} Model;

static double now_s(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static double rss_gb(void){ struct rusage r; getrusage(RUSAGE_SELF,&r);
#if defined(__APPLE__)
    return r.ru_maxrss/(1024.0*1024.0*1024.0);
#else
    return r.ru_maxrss/(1024.0*1024.0);
#endif
}
static double peak_rss_gb(void){ return rss_gb(); }
/* read one L3_THREADS* env var: unset/empty -> -1 (caller applies its default);
 * set-but-invalid (<=0 or non-numeric) -> reject explicitly, never clamp. */
static int l3_thread_count(const char *name){
    const char *v=getenv(name);
    if(!v||!*v) return -1;
    char *end=NULL; long n=strtol(v,&end,10);
    if(n<=0||!end||*end){
        fprintf(stderr,"[L3] %s=%s rejected: must be a positive integer thread count\n",name,v); exit(1); }
    return (int)n;
}
static long g_allocs=0, g_frees=0;
static int g_heap_phase=0;   /* 0=INIT 1=PREFILL/DECODE: counters only accrue when >0 */
static float *falloc(int64_t n){ float *p=malloc((size_t)n*sizeof(float)); if(!p){fprintf(stderr,"OOM %lld floats\n",(long long)n);exit(1);} if(g_heap_phase)g_allocs++; return p; }
static float *fcalloc(int64_t n){ float *p=calloc(1,(size_t)n*sizeof(float)); if(!p){fprintf(stderr,"OOM %lld floats\n",(long long)n);exit(1);} if(g_heap_phase)g_allocs++; return p; }
static inline float sigmoidf_(float x){ return 1.f/(1.f+expf(-x)); }
static inline float siluf_(float x){ return x/(1.f+expf(-x)); }
static void rmsnorm_(float *out, const float *x, const float *w, int D, float eps){
    double ms=0; for(int i=0;i<D;i++) ms+=(double)x[i]*x[i];
    float r=1.f/sqrtf((float)(ms/D)+eps);
    for(int i=0;i<D;i++) out[i]=x[i]*r*w[i];
}

/* ---------- W: load-time quantization + matvec (kimi_k3.c machinery) ----- */
static void w_matmul(float *y, const float *x, const W *w, int S){
    if(w->fmt==0)      matmul(y,x,w->f,S,w->I,w->O);
    else if(w->fmt==1) matmul_q(y,x,w->q8,w->s,S,w->I,w->O);
    else if(w->fmt==4) matmul_i4_grouped(y,x,w->q4,w->s,S,w->I,w->O,w->gs);
    else { fprintf(stderr,"w_matmul: bad fmt %d\n",w->fmt); exit(1); }
}
/* acc[0..I) += coef * row r (MLA absorb builds q_abs from kv_b rows) */
static void w_addrow(const W *w, int r, float coef, float *acc){
    int I=w->I;
    if(w->fmt==0){ const float *p=w->f+(int64_t)r*I; for(int i=0;i<I;i++) acc[i]+=coef*p[i]; }
    else if(w->fmt==1){ const int8_t *p=w->q8+(int64_t)r*I; float s=w->s[r]*coef;
        for(int i=0;i<I;i++) acc[i]+=s*p[i]; }
    else { int rb=(I+1)/2, ng=(I+w->gs-1)/w->gs; const uint8_t *p=w->q4+(int64_t)r*rb;
        const float *scl=w->s+(int64_t)r*ng;
        for(int g=0;g*w->gs<I;g++){ float s=scl[g]*coef; int e=(g+1)*w->gs; if(e>I)e=I;
            for(int i=g*w->gs;i<e;i+=2){ uint8_t b=p[i>>1];
                acc[i]+=s*(float)((int)(b&0xF)-8);
                if(i+1<e) acc[i+1]+=s*(float)((int)(b>>4)-8); } } }
}
static float w_rowdot(const W *w, int r, const float *x){
    int I=w->I; float a=0;
    if(w->fmt==0){ const float *p=w->f+(int64_t)r*I; for(int i=0;i<I;i++) a+=x[i]*p[i]; return a; }
    if(w->fmt==1){ const int8_t *p=w->q8+(int64_t)r*I; for(int i=0;i<I;i++) a+=x[i]*p[i]; return a*w->s[r]; }
    { int rb=(I+1)/2, ng=(I+w->gs-1)/w->gs; const uint8_t *p=w->q4+(int64_t)r*rb;
      const float *scl=w->s+(int64_t)r*ng;
      for(int g=0;g*w->gs<I;g++){ float ga=0; int e=(g+1)*w->gs; if(e>I)e=I;
          for(int i=g*w->gs;i<e;i+=2){ uint8_t b=p[i>>1];
              ga+=x[i]*(float)((int)(b&0xF)-8);
              if(i+1<e) ga+=x[i+1]*(float)((int)(b>>4)-8); }
          a+=ga*scl[g]; } }
    return a;
}

#define QCHUNK 1024
static int g_bytes_matrix(int O,int I,int bits){
    if(bits>=32) return (int)((int64_t)O*I*4);
    if(bits>=8)  return (int)((int64_t)O*I+(int64_t)O*4);
    return (int)((int64_t)O*(I/2)+(int64_t)O*(I/64)*4);
}
static void w_load(Model *m, W *w, const char *name, int O, int I, int bits){
    st_tensor *t=st_find(&m->S,name);
    if(!t) st_die_missing(&m->S,name);
    memset(w,0,sizeof(*w)); w->O=O; w->I=I;
    if(t->numel!=(int64_t)O*I){ fprintf(stderr,"%s: numel %lld != %dx%d\n",name,(long long)t->numel,O,I); exit(1); }
    m->wbytes += g_bytes_matrix(O,I,bits);
    if(bits>=32){ w->fmt=0; w->f=falloc((int64_t)O*I); st_read_f32(&m->S,name,w->f,0); return; }
    int gs=64;
    if(bits<=4 && I%gs){ bits=8; }
    float *scr=falloc((int64_t)QCHUNK*I);
    if(bits>4){ w->fmt=1; w->q8=malloc((int64_t)O*I); w->s=falloc(O);
        if(!w->q8){fprintf(stderr,"OOM int8 %s\n",name);exit(1);}
        for(int r0=0;r0<O;r0+=QCHUNK){ int n=O-r0<QCHUNK?O-r0:QCHUNK;
            st_read_slice_f32(&m->S,name,(int64_t)r0*I,(int64_t)n*I,scr,0);
            for(int r=0;r<n;r++){ const float *src=scr+(int64_t)r*I;
                float am=0; for(int i=0;i<I;i++){ float a=fabsf(src[i]); if(a>am)am=a; }
                float s=am/127.f; if(s<1e-20f)s=1e-20f; w->s[r0+r]=s; float inv=1.f/s;
                int8_t *dst=w->q8+(int64_t)(r0+r)*I;
                for(int i=0;i<I;i++){ int v=(int)lrintf(src[i]*inv); if(v>127)v=127; if(v<-127)v=-127; dst[i]=(int8_t)v; } } }
    } else { w->fmt=4; w->gs=gs; int rb=I/2, ng=I/gs;
        w->q4=malloc((int64_t)O*rb); w->s=falloc((int64_t)O*ng);
        if(!w->q4){fprintf(stderr,"OOM int4 %s\n",name);exit(1);}
        for(int r0=0;r0<O;r0+=QCHUNK){ int n=O-r0<QCHUNK?O-r0:QCHUNK;
            st_read_slice_f32(&m->S,name,(int64_t)r0*I,(int64_t)n*I,scr,0);
            for(int r=0;r<n;r++){ const float *src=scr+(int64_t)r*I;
                uint8_t *dst=w->q4+(int64_t)(r0+r)*rb; float *scl=w->s+(int64_t)(r0+r)*ng;
                for(int g=0;g<ng;g++){ const float *gp=src+g*gs;
                    float am=0; for(int i=0;i<gs;i++){ float a=fabsf(gp[i]); if(a>am)am=a; }
                    float s=am/7.f; if(s<1e-20f)s=1e-20f; scl[g]=s; float inv=1.f/s;
                    for(int i=0;i<gs;i+=2){
                        int v0=(int)lrintf(gp[i]*inv);   if(v0>7)v0=7; if(v0<-8)v0=-8;
                        int v1=(int)lrintf(gp[i+1]*inv); if(v1>7)v1=7; if(v1<-8)v1=-8;
                        dst[(g*gs+i)>>1]=(uint8_t)((v0+8)|((v1+8)<<4)); } } } }
    }
    free(scr);
}
static float *f32_load(Model *m, const char *name, int64_t want){
    st_tensor *t=st_find(&m->S,name);
    if(!t) st_die_missing(&m->S,name);
    if(want>0 && t->numel!=want){ fprintf(stderr,"%s: numel %lld != %lld\n",name,(long long)t->numel,(long long)want); exit(1); }
    float *p=falloc(t->numel); st_read_f32(&m->S,name,p,0);
    m->wbytes += t->numel*4;
    return p;
}

/* ---------- native compressed-tensors int4-g32 residency ---------- */
static void exp_load(Model *m, Exp *e, const char *base, int O, int I){
    char nm[600];
    memset(e,0,sizeof(*e));
    e->O=O; e->I=I; e->nw=I/8; e->ng=I/32;
    snprintf(nm,sizeof(nm),"%s.weight_packed",base);
    st_tensor *t=st_find(&m->S,nm);
    if(!t) st_die_missing(&m->S,nm);
    int64_t want=(int64_t)O*e->nw*4;
    /* FAIL-CLOSED on the exact declared dtype: ST_I32 is required, not "any raw
     * tensor with the right byte span" — the old dtype==3 check could not tell
     * an I32 [O,I/8] payload from an I64/I16 tensor of the same size. */
    if(t->dtype!=ST_I32 || t->nbytes!=want){
        fprintf(stderr,"%s: dtype %d nbytes %lld != expected ST_I32 %lld (I32 [%d,%d])\n",
                nm,t->dtype,(long long)t->nbytes,(long long)want,O,e->nw); exit(1); }
    e->packed=malloc((size_t)want);
    if(!e->packed){fprintf(stderr,"OOM expert packed %s\n",nm);exit(1);}
    st_read_raw(&m->S,nm,e->packed,0);
    snprintf(nm,sizeof(nm),"%s.weight_scale",base);
    st_tensor *ts=st_find(&m->S,nm);
    if(!ts) st_die_missing(&m->S,nm);
    if(ts->numel!=(int64_t)O*e->ng){
        fprintf(stderr,"%s: numel %lld != %dx%d\n",nm,(long long)ts->numel,O,e->ng); exit(1); }
    /* official compressed-tensors scales are BF16 [O, I/32]: require that exact
     * float dtype instead of trusting st_read_f32's generic conversion. */
    if(ts->dtype!=ST_BF16){
        fprintf(stderr,"%s: scale dtype %d != ST_BF16 (official weight_scale is BF16 [O,I/32])\n",
                nm,ts->dtype); exit(1); }
    e->scl=falloc((int64_t)O*e->ng);
    st_read_f32(&m->S,nm,e->scl,0);            /* bf16 -> f32 once, at load */
    m->wbytes += want + (int64_t)O*e->ng*4;
}

static float *g_ws_xev=NULL,*g_ws_xod=NULL;   /* persistent prepared-input split */

/* y[r] = sum_c x[c]*(nibble(r,c)-8)*scale(r,c/32), straight from packed words.
 *
 * AVX2 path: each 16-byte block holds one 32-group as sequential element
 * PAIRS (byte j = elements 2j,2j+1). Since a dot product is permutation
 * invariant, x is split ONCE into even/odd streams so both sides stay
 * contiguous: nibble-lo pairs against x-even, nibble-hi against x-odd.
 * Group scales applied per 32-block. Scalar reference kept for non-x86. */
static void exp_matvec(float *y, const float *x, const Exp *e){
    int O=e->O, ng=e->ng;
    if(e->dense){ matmul(y,x,e->dense,1,e->I,O); return; }
#if defined(__x86_64__) && defined(__AVX2__) && !defined(L3_NO_AVX2)
    int I=e->I;
    /* STRIKE 4: prepared-input split lives in the persistent workspace */
    float *xev=g_ws_xev, *xod=g_ws_xod;
    (void)I;
    for(int c=0;c<I;c+=2){ xev[c>>1]=x[c]; xod[c>>1]=x[c+1]; }
    const __m256i m4=_mm256_set1_epi8(0x0F);
    const __m128i m4s=_mm_set1_epi8(0x0F);
    const __m128i e8=_mm_set1_epi8(8);
    #pragma omp parallel for schedule(static)
    for(int r=0;r<O;r++){
        const uint8_t *pb=(const uint8_t*)(e->packed+(int64_t)r*e->nw);
        const float *sc=e->scl+(int64_t)r*ng;
        __m256 accE=_mm256_setzero_ps(), accO=_mm256_setzero_ps();
        int g=0;
        for(;g<ng;g++){
            const uint8_t *blk=pb+(int64_t)g*16;
            __m128i b=_mm_loadu_si128((const __m128i*)blk);
            __m128i lo=_mm_sub_epi8(_mm_and_si128(b,m4s),e8);
            __m128i hi=_mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(b,4),m4s),e8);
            __m256i l0=_mm256_cvtepi8_epi32(lo);
            __m256i l1=_mm256_cvtepi8_epi32(_mm_bsrli_si128(lo,8));
            __m256i h0=_mm256_cvtepi8_epi32(hi);
            __m256i h1=_mm256_cvtepi8_epi32(_mm_bsrli_si128(hi,8));
            const float *xe=xev+g*16, *xo=xod+g*16;
            __m256 pe=_mm256_add_ps(
                _mm256_mul_ps(_mm256_cvtepi32_ps(l0),_mm256_loadu_ps(xe)),
                _mm256_mul_ps(_mm256_cvtepi32_ps(l1),_mm256_loadu_ps(xe+8)));
            __m256 po=_mm256_add_ps(
                _mm256_mul_ps(_mm256_cvtepi32_ps(h0),_mm256_loadu_ps(xo)),
                _mm256_mul_ps(_mm256_cvtepi32_ps(h1),_mm256_loadu_ps(xo+8)));
            __m256 sf=_mm256_set1_ps(sc[g]);
            accE=_mm256_fmadd_ps(pe,sf,accE);
            accO=_mm256_fmadd_ps(po,sf,accO);
        }
        __m128 s=_mm_add_ps(_mm256_castps256_ps128(accE),_mm256_extractf128_ps(accE,1));
        s=_mm_add_ps(s,_mm_add_ps(_mm256_castps256_ps128(accO),_mm256_extractf128_ps(accO,1)));
        s=_mm_add_ps(s,_mm_movehl_ps(s,s));
        s=_mm_add_ss(s,_mm_shuffle_ps(s,s,1));
        y[r]=_mm_cvtss_f32(s);
    }
#else
    #pragma omp parallel for schedule(static)
    for(int r=0;r<O;r++){
        const uint32_t *pw=e->packed+(int64_t)r*e->nw;
        const float *sc=e->scl+(int64_t)r*ng;
        float acc=0;
        for(int g=0;g<ng;g++){
            const uint32_t *wp=pw+(int64_t)g*4;
            const float *xp=x+g*32;
            float ga=0;
            for(int w4=0;w4<4;w4++){
                uint32_t word=wp[w4];
                for(int nb=0;nb<8;nb++)
                    ga+=xp[w4*8+nb]*(float)((int)((word>>(4*nb))&0xF)-8);
            }
            acc+=ga*sc[g];
        }
        y[r]=acc;
    }
#endif
}

static double g_rec_work=0,g_abs_work=0,g_val_work=0,g_ctx_work=0;
static int g_kda_scalar=0;                             /* L3_KDA_SCALAR=1: Wave-1 reference recurrence */
static int g_phases=0;                                 /* L3_PHASES: sub-wall telemetry */

/* ---------- config ---------- */
static double req_num(jval *r, const char *k){
    jval *v=json_get(r,k);
    if(!v||v->t!=J_NUM){ fprintf(stderr,"config.json: missing or non-numeric \"%s\"\n",k); exit(1); }
    return v->num;
}
static void load_cfg(Cfg *c, const char *snap){
    char path[2048]; snprintf(path,sizeof(path),"%s/config.json",snap);
    long n; char *buf;
    { FILE *f=fopen(path,"rb"); if(!f){perror(path);exit(1);}
      fseek(f,0,SEEK_END); n=ftell(f); fseek(f,0,SEEK_SET);
      if(n<0||n>(64L<<20)){ fprintf(stderr,"%s: bad size\n",path); exit(1); }
      buf=malloc((size_t)n+1); if(!buf){fprintf(stderr,"OOM cfg\n");exit(1);}
      if(fread(buf,1,(size_t)n,f)!=(size_t)n){ fprintf(stderr,"%s: short read\n",path); exit(1); }
      buf[n]=0; fclose(f); }
    char *arena=NULL; jval *root=json_parse(buf,&arena);
    jval *tc=json_get(root,"text_config"); if(!tc||tc->t!=J_OBJ) tc=root;
    memset(c,0,sizeof(*c));
    c->hidden      =(int)req_num(tc,"hidden_size");
    c->n_layers    =(int)req_num(tc,"num_hidden_layers");
    c->vocab       =(int)req_num(tc,"vocab_size");
    c->first_dense =(int)req_num(tc,"first_k_dense_replace");
    c->dense_inter =(int)req_num(tc,"intermediate_size");
    c->n_heads     =(int)req_num(tc,"num_attention_heads");
    c->q_lora      =(int)req_num(tc,"q_lora_rank");
    c->kv_lora     =(int)req_num(tc,"kv_lora_rank");
    c->qk_nope     =(int)req_num(tc,"qk_nope_head_dim");
    c->qk_rope     =(int)req_num(tc,"qk_rope_head_dim");
    c->v_head      =(int)req_num(tc,"v_head_dim");
    c->n_experts   =(int)req_num(tc,"num_experts");
    c->topk        =(int)req_num(tc,"num_experts_per_tok");
    c->moe_inter   =(int)req_num(tc,"moe_intermediate_size");
    c->sh_inter    =(int)req_num(tc,"moe_shared_expert_intermediate_size")*(int)req_num(tc,"num_shared_experts");
    c->n_group     =(int)req_num(tc,"n_group");
    c->topk_group  =(int)req_num(tc,"topk_group");
    c->routed_scale=(float)req_num(tc,"routed_scaling_factor");
    c->rope_theta  =(float)req_num(tc,"rope_theta");
    /* modeling_bailing_moe_v3.RotaryEmbedding.__init__ force-sets
     * head_dim=qk_rope_head_dim and partial_rotary_factor=1.0: RoPE covers
     * the FULL 64-dim rope slice (config rotary_dim == qk_rope_head_dim). */
    c->rotary_dim = c->qk_rope;
    jval *ep=json_get(tc,"rms_norm_eps"); c->eps=ep?(float)ep->num:1e-6f;
    c->qk_head=c->qk_nope+c->qk_rope;
    c->attn_scale=1.f/sqrtf((float)c->qk_head);
    c->kda_heads=c->n_heads;
    { jval *hds=json_get(tc,"head_dim"); c->kda_hd=hds?(int)hds->num:128; }
    if(c->kda_hd<=0) c->kda_hd=128;
    { jval *ck=json_get(tc,"short_conv_kernel_size"); c->conv_k=ck?(int)ck->num:4; }
    { jval *lb=json_get(tc,"kda_lower_bound"); c->gate_lb=lb?(float)lb->num:-5.f; }
    c->kda_proj=c->kda_heads*c->kda_hd;
    if(c->hidden<1||c->hidden>65536||c->n_layers<1||c->n_layers>128||
       c->n_experts<1||c->n_experts>4096||c->topk<1||c->topk>64||c->topk>c->n_experts||
       c->vocab<1||c->vocab>(1<<22)||c->kda_proj<1||c->kda_proj>(1<<20)||
       c->conv_k<1||c->conv_k>8||c->moe_inter%32||c->kv_lora>4096||
       c->n_group<1||(c->n_experts%c->n_group)||c->topk_group<1||c->topk_group>c->n_group){
        fprintf(stderr,"config.json: dimension out of range\n"); exit(1); }
    { int g=4; jval *lg=json_get(tc,"layer_group_size"); if(lg) g=(int)lg->num;
      if(g<1) g=4;
      for(int i=0;i<c->n_layers;i++)
          c->is_mla[i]=(((i+1)%g)==0)||(i>=(c->n_layers/g*g)); }
    jval *b=json_get(root,"bos_token_id"); if(!b) b=json_get(tc,"bos_token_id");
    c->bos = b&&b->t==J_NUM ? (int)b->num : -1;
    jval *e=json_get(root,"eos_token_id"); if(!e) e=json_get(tc,"eos_token_id");
    if(e&&e->t==J_NUM) c->eos[c->n_eos++]=(int)e->num;
    else if(e&&e->t==J_ARR) for(int i=0;i<e->len&&c->n_eos<8;i++) c->eos[c->n_eos++]=(int)e->kids[i]->num;
    free(buf); (void)arena;
}

/* ---------- init ---------- */
static void model_init(Model *m, const char *snap, int n_layers_env){
    memset(m,0,sizeof(*m));
    load_cfg(&m->c,snap);
    Cfg *c=&m->c;
    if(n_layers_env>0&&n_layers_env<c->n_layers) c->n_layers=n_layers_env;
    st_init_multi(&m->S,snap,NULL);
    int is_int4 = st_has(&m->S,"model.layers.1.mlp.experts.0.gate_proj.weight_packed");
    if(!is_int4) fprintf(stderr,"[L3] note: no packed experts in snapshot — %s\n",
        getenv("L3_EXACT_EXPERTS")?"EXACT track (f32 experts)":"bf16-emulated-int4 experts");
    int bits      = getenv("L3_BITS")?atoi(getenv("L3_BITS")):32;
    int hbits     = getenv("L3_HEAD_BITS")?atoi(getenv("L3_HEAD_BITS")):bits;
    double t0=now_s();
    m->L=calloc(c->n_layers,sizeof(Layer));
    m->kstate=calloc(c->n_layers,sizeof(float*));
    m->cwq=calloc(c->n_layers,sizeof(float*));
    m->cwk=calloc(c->n_layers,sizeof(float*));
    m->cwv=calloc(c->n_layers,sizeof(float*));
    char nm[600];
    #define NM(...) (snprintf(nm,sizeof(nm),__VA_ARGS__),nm)
    for(int i=0;i<c->n_layers;i++){
        Layer *l=&m->L[i];
        l->mla=c->is_mla[i];
        l->sparse=(i>=c->first_dense);
        l->in_ln  =f32_load(m,NM("model.layers.%d.input_layernorm.weight",i),c->hidden);
        l->post_ln=f32_load(m,NM("model.layers.%d.post_attention_layernorm.weight",i),c->hidden);
        if(!l->mla){                                  /* KDA */
            Kda *a=&l->a; int P=c->kda_proj;
            w_load(m,&a->q,NM("model.layers.%d.attention.q_proj.weight",i),P,c->hidden,bits);
            w_load(m,&a->k,NM("model.layers.%d.attention.k_proj.weight",i),P,c->hidden,bits);
            w_load(m,&a->v,NM("model.layers.%d.attention.v_proj.weight",i),P,c->hidden,bits);
            w_load(m,&a->o,NM("model.layers.%d.attention.o_proj.weight",i),c->hidden,P,bits);
            w_load(m,&a->g,NM("model.layers.%d.attention.g_proj.weight",i),P,c->hidden,bits);
            w_load(m,&a->f,NM("model.layers.%d.attention.f_proj.weight",i),P,c->hidden,bits);
            a->conv_q=f32_load(m,NM("model.layers.%d.attention.q_conv1d.weight",i),(int64_t)P*c->conv_k);
            a->conv_k=f32_load(m,NM("model.layers.%d.attention.k_conv1d.weight",i),(int64_t)P*c->conv_k);
            a->conv_v=f32_load(m,NM("model.layers.%d.attention.v_conv1d.weight",i),(int64_t)P*c->conv_k);
            a->bp=f32_load(m,NM("model.layers.%d.attention.b_proj.weight",i),(int64_t)c->kda_heads*c->hidden);
            a->dt=f32_load(m,NM("model.layers.%d.attention.dt_bias",i),P);
            a->onw=f32_load(m,NM("model.layers.%d.attention.o_norm.weight",i),c->kda_hd);
            a->A=falloc(c->kda_heads);
            { float *al=f32_load(m,NM("model.layers.%d.attention.A_log",i),c->kda_heads);
              for(int h=0;h<c->kda_heads;h++) a->A[h]=expf(al[h]); free(al); }
            m->kstate[i]=fcalloc((int64_t)c->kda_heads*c->kda_hd*c->kda_hd);
            m->cwq[i]=fcalloc((int64_t)P*c->conv_k);
            m->cwk[i]=fcalloc((int64_t)P*c->conv_k);
            m->cwv[i]=fcalloc((int64_t)P*c->conv_k);
        } else {                                      /* gated partial-RoPE MLA */
            Mla *a=&l->m;
            w_load(m,&a->qa, NM("model.layers.%d.attention.q_a_proj.weight",i),c->q_lora,c->hidden,bits);
            w_load(m,&a->qb, NM("model.layers.%d.attention.q_b_proj.weight",i),c->n_heads*c->qk_head,c->q_lora,bits);
            w_load(m,&a->kva,NM("model.layers.%d.attention.kv_a_proj_with_mqa.weight",i),c->kv_lora+c->qk_rope,c->hidden,bits);
            w_load(m,&a->kvb,NM("model.layers.%d.attention.kv_b_proj.weight",i),c->n_heads*(c->qk_nope+c->v_head),c->kv_lora,bits);
            w_load(m,&a->o,  NM("model.layers.%d.attention.dense.weight",i),c->hidden,c->n_heads*c->v_head,bits);
            w_load(m,&a->g,  NM("model.layers.%d.attention.g_proj.weight",i),c->n_heads,c->hidden,bits);
            a->qa_ln =f32_load(m,NM("model.layers.%d.attention.q_a_layernorm.weight",i),c->q_lora);
            a->kva_ln=f32_load(m,NM("model.layers.%d.attention.kv_a_layernorm.weight",i),c->kv_lora);
        }
        if(l->sparse){
            Moe *o=&l->moe;
            o->router =f32_load(m,NM("model.layers.%d.mlp.gate.weight",i),(int64_t)c->n_experts*c->hidden);
            o->rbias  =f32_load(m,NM("model.layers.%d.mlp.gate.expert_bias",i),c->n_experts);
            w_load(m,&o->sh_gate,NM("model.layers.%d.mlp.shared_experts.gate_proj.weight",i),c->sh_inter,c->hidden,bits);
            w_load(m,&o->sh_up,  NM("model.layers.%d.mlp.shared_experts.up_proj.weight",i),c->sh_inter,c->hidden,bits);
            w_load(m,&o->sh_down,NM("model.layers.%d.mlp.shared_experts.down_proj.weight",i),c->hidden,c->sh_inter,bits);
            o->exps=calloc((size_t)c->n_experts*3,sizeof(Exp));
            if(is_int4){
                for(int e2=0;e2<c->n_experts;e2++){
                    exp_load(m,&o->exps[e2*3+0],NM("model.layers.%d.mlp.experts.%d.gate_proj",i,e2),c->moe_inter,c->hidden);
                    exp_load(m,&o->exps[e2*3+1],NM("model.layers.%d.mlp.experts.%d.up_proj",i,e2),c->moe_inter,c->hidden);
                    exp_load(m,&o->exps[e2*3+2],NM("model.layers.%d.mlp.experts.%d.down_proj",i,e2),c->hidden,c->moe_inter);
                }
            } else if(getenv("L3_EXACT_EXPERTS")){
                /* EXACT TRACK: keep routed experts at bf16->f32 precision so
                 * parity adjudication measures MATH, not quantization. */
                for(int e2=0;e2<c->n_experts;e2++){
                    static const char *mt[3]={"gate_proj","up_proj","down_proj"};
                    int OO[3]={c->moe_inter,c->moe_inter,c->hidden};
                    int II[3]={c->hidden,c->hidden,c->moe_inter};
                    for(int mi=0;mi<3;mi++){
                        Exp *ex=&o->exps[e2*3+mi];
                        ex->O=OO[mi]; ex->I=II[mi];
                        ex->dense=falloc((int64_t)OO[mi]*II[mi]);
                        char full[700];
                        snprintf(full,sizeof(full),"model.layers.%d.mlp.experts.%d.%s.weight",i,e2,mt[mi]);
                        st_read_f32(&m->S,full,ex->dense,0);
                        m->wbytes += (int64_t)OO[mi]*II[mi]*4;
                    }
                }
            } else {
                /* BF16 snapshot without exact-expert flag: emulate the native
                 * int4-g32 layout by load-time quantization (shared kernel). */
                for(int e2=0;e2<c->n_experts;e2++){
                    static const char *mt[3]={"gate_proj","up_proj","down_proj"};
                    int OO[3]={c->moe_inter,c->moe_inter,c->hidden};
                    int II[3]={c->hidden,c->hidden,c->moe_inter};
                    for(int mi=0;mi<3;mi++){
                        Exp *ex=&o->exps[e2*3+mi];
                        ex->O=OO[mi]; ex->I=II[mi]; ex->nw=II[mi]/8; ex->ng=II[mi]/32;
                        ex->packed=malloc((size_t)OO[mi]*ex->nw*4);
                        ex->scl=falloc((int64_t)OO[mi]*ex->ng);
                        char full[700];
                        snprintf(full,sizeof(full),"model.layers.%d.mlp.experts.%d.%s.weight",i,e2,mt[mi]);
                        float *src=falloc((int64_t)OO[mi]*II[mi]);
                        st_read_f32(&m->S,full,src,0);
                        for(int r=0;r<OO[mi];r++){
                            uint32_t *dst=ex->packed+(int64_t)r*ex->nw;
                            float *sc2=ex->scl+(int64_t)r*ex->ng;
                            const float *rw=src+(int64_t)r*II[mi];
                            for(int g2=0;g2<ex->ng;g2++){
                                const float *gp=rw+g2*32;
                                float am=0; for(int z=0;z<32;z++){ float a=fabsf(gp[z]); if(a>am)am=a; }
                                float s=am/7.f; if(s<1e-20f)s=1e-20f; sc2[g2]=s; float inv=1.f/s;
                                uint32_t word[4]={0,0,0,0};
                                for(int z=0;z<32;z++){
                                    int v=(int)lrintf(gp[z]*inv); if(v>7)v=7; if(v<-8)v=-8;
                                    word[z>>3] |= ((uint32_t)(v+8))<<(4*(z&7));
                                }
                                memcpy(dst+g2*4,word,16);
                            }
                        }
                        free(src);
                        m->wbytes += (int64_t)OO[mi]*ex->nw*4 + (int64_t)OO[mi]*ex->ng*4;
                    }
                }
            }
        } else {
            w_load(m,&l->d_gate,NM("model.layers.%d.mlp.gate_proj.weight",i),c->dense_inter,c->hidden,bits);
            w_load(m,&l->d_up,  NM("model.layers.%d.mlp.up_proj.weight",i),c->dense_inter,c->hidden,bits);
            w_load(m,&l->d_down,NM("model.layers.%d.mlp.down_proj.weight",i),c->hidden,c->dense_inter,bits);
        }
        if(i%8==0) fprintf(stderr,"[L3] loaded layer %d/%d (%.1fs, RSS %.1f GB)\n",i+1,c->n_layers,now_s()-t0,rss_gb());
    }
    m->final_norm=f32_load(m,"model.norm.weight",c->hidden);
    w_load(m,&m->embed,"model.word_embeddings.weight",c->vocab,c->hidden,bits);
    w_load(m,&m->lm_head,"lm_head.weight",c->vocab,c->hidden,hbits);
    int nmla=0; for(int i=0;i<c->n_layers;i++) if(c->is_mla[i]) nmla++;
    fprintf(stderr,"[L3] init done in %.1fs | %d layers (%d MLA/%d KDA) | %s | resident weights %.2f GB | RSS %.1f GB\n",
            now_s()-t0,c->n_layers,nmla,c->n_layers-nmla,
            is_int4?"native-int4-resident":"bf16-emulated-int4",
            m->wbytes/1e9,rss_gb());
    #undef NM
}

static void kv_alloc(Model *m, int max_t){
    Cfg *c=&m->c; m->max_t=max_t;
    { /* STRIKE 3: one-time workspace sizing */
        int cm=getenv("L3_CHUNK")?atoi(getenv("L3_CHUNK")):32;
        if(cm<1)cm=1; if(cm>512)cm=512;
        m->ws.cmax=cm;
        long D=c->hidden, P=c->kda_proj;
        #define WSA(f,n) m->ws.f=falloc(n)
        WSA(hidden,cm*D); WSA(nrm,cm*D); WSA(att,cm*D); WSA(mlp,cm*D);
        WSA(kq,cm*P); WSA(kk_,cm*P); WSA(kv,cm*P); WSA(kgp,cm*P);
        WSA(kon,cm*P); WSA(kgraw,cm*P); WSA(kbraw,(long)cm*c->kda_heads);
        WSA(qa,(long)cm*c->q_lora); WSA(qv,(long)cm*c->n_heads*c->qk_head);
        WSA(ckv,(long)cm*(c->kv_lora+c->qk_rope)); WSA(gv,(long)cm*c->n_heads);
        WSA(ctx,(long)cm*c->n_heads*c->v_head);
        WSA(moe_U,(long)cm*D); m->ws.moe_idx=(int*)falloc(cm*(long)c->topk); g_allocs--;
        m->ws.moe_w=falloc((long)cm*c->topk);
        WSA(gate,c->moe_inter); WSA(up,c->moe_inter); WSA(hz,D);
        { int Imax=c->hidden>c->moe_inter?c->hidden:c->moe_inter;
          WSA(xev,Imax/2+16); WSA(xod,Imax/2+16); }
        WSA(logits,c->vocab);
        { long cap=(long)cm*c->topk;
          WSA(uid_f,cap); WSA(wlist,cap); WSA(poslist_f,cap);
          m->ws.uid=(int*)falloc(cap); m->ws.pcnt=(int*)falloc(cap);
          m->ws.pfirst=(int*)falloc(cap); m->ws.poslist=(int*)falloc(cap);
          m->ws.cur=(int*)falloc(cap); }
        WSA(sg,(long)cm*c->sh_inter); WSA(su,(long)cm*c->sh_inter);
        WSA(sd,(long)cm*D);
        WSA(dg,(long)cm*c->dense_inter); WSA(du,(long)cm*c->dense_inter);
        #undef WSA
        g_ws_xev=m->ws.xev; g_ws_xod=m->ws.xod;
    }
    m->Lc=calloc(c->n_layers,sizeof(float*));
    m->Rc=calloc(c->n_layers,sizeof(float*));
    for(int i=0;i<c->n_layers;i++) if(m->L[i].mla){
        m->Lc[i]=falloc((int64_t)max_t*c->kv_lora);
        m->Rc[i]=falloc((int64_t)max_t*c->qk_rope);
    }
    m->cos_t=falloc((int64_t)max_t*(c->rotary_dim/2));
    m->sin_t=falloc((int64_t)max_t*(c->rotary_dim/2));
    for(int p=0;p<max_t;p++)
        for(int j=0;j<c->rotary_dim/2;j++){
            float th=p*powf(c->rope_theta,-2.f*j/c->rotary_dim);
            m->cos_t[(int64_t)p*(c->rotary_dim/2)+j]=cosf(th);
            m->sin_t[(int64_t)p*(c->rotary_dim/2)+j]=sinf(th);
        }
}

static void model_state_reset(Model *m){
    Cfg *c=&m->c;
    for(int i=0;i<c->n_layers;i++){
        if(m->L[i].mla) continue;
        memset(m->kstate[i],0,(size_t)c->kda_heads*c->kda_hd*c->kda_hd*sizeof(float));
        memset(m->cwq[i],0,(size_t)c->kda_proj*c->conv_k*sizeof(float));
        memset(m->cwk[i],0,(size_t)c->kda_proj*c->conv_k*sizeof(float));
        memset(m->cwv[i],0,(size_t)c->kda_proj*c->conv_k*sizeof(float));
    }
    for(int i=0;i<c->n_layers;i++) if(m->L[i].mla){
        memset(m->Lc[i],0,(size_t)m->max_t*c->kv_lora*sizeof(float));
        memset(m->Rc[i],0,(size_t)m->max_t*c->qk_rope*sizeof(float));
    }
}

static int g_debug=-1;
/* ---------- KDA layer (chunk of C tokens) ---------- */
static void kda_forward(Model *m, Layer *l, int li, const float *x, int C, float *out){
    Cfg *c=&m->c; Kda *a=&l->a;
    int P=c->kda_proj, H=c->kda_heads, hd=c->kda_hd, K=c->conv_k;
    float *q=m->ws.kq, *k=m->ws.kk_, *v=m->ws.kv;
    float *gp=m->ws.kgp, *on=m->ws.kon;
    float *graw=m->ws.kgraw, *braw=m->ws.kbraw;
    w_matmul(q,x,&a->q,C); w_matmul(k,x,&a->k,C); w_matmul(v,x,&a->v,C);
    w_matmul(gp,x,&a->g,C); w_matmul(graw,x,&a->f,C);
    matmul(braw,x,a->bp,C,c->hidden,H);
    if(g_debug<0) g_debug=getenv("L3_DEBUG")?atoi(getenv("L3_DEBUG")):0;
    if(g_debug&&li==0){
        fprintf(stderr,"DBG xn[0:4] %.6f %.6f %.6f %.6f\n",x[0],x[1],x[2],x[3]);
        fprintf(stderr,"DBG q[0:4] %.6f %.6f %.6f %.6f\n",q[0],q[1],q[2],q[3]);
        fprintf(stderr,"DBG graw[0:4] %.6f %.6f %.6f %.6f\n",graw[0],graw[1],graw[2],graw[3]);
        fprintf(stderr,"DBG braw[0:2] %.6f %.6f\n",braw[0],braw[1]);
    }
    float qscale=1.f/sqrtf((float)hd);
    for(int t=0;t<C;t++){
        float *qt=q+(int64_t)t*P, *kt=k+(int64_t)t*P, *tv=v+(int64_t)t*P;
        float *gpt=gp+(int64_t)t*P, *ont=on+(int64_t)t*P;
        const float *rgt=graw+(int64_t)t*P, *bt=braw+(int64_t)t*H;
        float *wins[3]={m->cwq[li],m->cwk[li],m->cwv[li]};
        float *vecs[3]={qt,kt,tv}; float *taps[3]={a->conv_q,a->conv_k,a->conv_v};
        for(int w2=0;w2<3;w2++){
            float *win=wins[w2], *vec=vecs[w2]; const float *cw=taps[w2];
            #pragma omp parallel for schedule(static)
            for(int d=0;d<P;d++){
                float *wd=win+(int64_t)d*K;
                for(int j=0;j<K-1;j++) wd[j]=wd[j+1];
                wd[K-1]=vec[d];
                float acc=0; const float *cd=cw+(int64_t)d*K;
                for(int j=0;j<K;j++) acc+=cd[j]*wd[j];
                vec[d]=siluf_(acc);
            }
        }
        #pragma omp parallel for schedule(static) reduction(+:g_rec_work)
        for(int h=0;h<H;h++){
            const float *qh=qt+(int64_t)h*hd, *kh=kt+(int64_t)h*hd, *vh=tv+(int64_t)h*hd;
            float qn[512], kn[512], alpha[512], vt[512], oh[512];
            float sq=0,sk=0;
            for(int i=0;i<hd;i++){ sq+=qh[i]*qh[i]; sk+=kh[i]*kh[i]; }
            sq=1.f/sqrtf(sq+1e-6f); sk=1.f/sqrtf(sk+1e-6f);
            for(int i=0;i<hd;i++){ qn[i]=qh[i]*sq*qscale; kn[i]=kh[i]*sk; }
            const float *z=rgt+(int64_t)h*hd;
            for(int i=0;i<hd;i++)
                alpha[i]=expf(c->gate_lb*sigmoidf_(a->A[h]*(z[i]+a->dt[(int64_t)h*hd+i])));
            float beta=sigmoidf_(bt[h]);
            float *S=m->kstate[li]+(int64_t)h*hd*hd;
            double tr0=now_s();
            if(g_debug&&li==0&&t==0&&h==0){
                float s0=0; for(int z=0;z<hd*hd;z++) s0+=fabsf(S[z]);
                fprintf(stderr,"DBG S_abssum_before %.6f\n",s0);
            }
            if(g_kda_scalar){
                /* scalar reference path (4 sweeps, Wave-1 semantics) */
                for(int kk=0;kk<hd;kk++){
                    float *row=S+(int64_t)kk*hd; float al=alpha[kk];
                    for(int vv=0;vv<hd;vv++) row[vv]*=al;
                }
                for(int vv=0;vv<hd;vv++){
                    float s2=0; for(int kk=0;kk<hd;kk++) s2+=S[(int64_t)kk*hd+vv]*kn[kk];
                    vt[vv]=(vh[vv]-s2)*beta;
                }
                for(int kk=0;kk<hd;kk++){
                    float *row=S+(int64_t)kk*hd; float kv=kn[kk];
                    for(int vv=0;vv<hd;vv++) row[vv]+=kv*vt[vv];
                }
                memset(oh,0,sizeof(oh));
                for(int kk=0;kk<hd;kk++){
                    const float *row=S+(int64_t)kk*hd; float qq=qn[kk];
                    for(int vv=0;vv<hd;vv++) oh[vv]+=qq*row[vv];
                }
            } else {
                /* STRIKE 2 two-pass row-major recurrence. Per-element math and
                 * kk accumulation order are IDENTICAL to the reference; only
                 * the S^T k strided traversal is fused into pass 1. SIMD lanes
                 * map to independent vv, so vectorization stays bit-faithful
                 * to this loop nest. */
                /* STRIKE 2: two row-major sweeps replace four full traversals.
                 * Bit-faithful to the Wave-1 reference: every product is
                 * rounded exactly where the scalar loop rounds it (explicit
                 * mul then add — no fused contracts), and each vv lane keeps
                 * the original kk accumulation order. */
                float sk[512] __attribute__((aligned(32)));
                for(int vv=0;vv<hd;vv++) sk[vv]=0.f;
                for(int kk=0;kk<hd;kk++){
                    float *row=S+(int64_t)kk*hd;
                    const float al=alpha[kk], kv=kn[kk];
#if defined(__x86_64__) && defined(__AVX2__) && !defined(L3_NO_AVX2)
                    const __m256 va=_mm256_set1_ps(al), vk=_mm256_set1_ps(kv);
                    /* lanes span distinct vv: each chunk accumulates its OWN
                     * sk slice (a hoisted register mixes chunks). */
                    for(int vv=0;vv<hd;vv+=8){
                        __m256 r=_mm256_mul_ps(_mm256_loadu_ps(row+vv),va);
                        _mm256_storeu_ps(row+vv,r);
                        _mm256_store_ps(sk+vv,_mm256_add_ps(_mm256_load_ps(sk+vv),
                                                            _mm256_mul_ps(r,vk)));
                    }
#else
                    for(int vv=0;vv<hd;vv++){ row[vv]*=al; sk[vv]+=row[vv]*kv; }
#endif
                }
                for(int vv=0;vv<hd;vv++) vt[vv]=(vh[vv]-sk[vv])*beta;
                memset(oh,0,sizeof(oh));
                for(int kk=0;kk<hd;kk++){
                    float *row=S+(int64_t)kk*hd;
                    const float kv=kn[kk], qq=qn[kk];
#if defined(__x86_64__) && defined(__AVX2__) && !defined(L3_NO_AVX2)
                    const __m256 vk=_mm256_set1_ps(kv), vq=_mm256_set1_ps(qq);
                    for(int vv=0;vv<hd;vv+=8){
                        __m256 vt8=_mm256_loadu_ps(vt+vv);
                        __m256 pr=_mm256_mul_ps(vk,vt8);
                        __m256 r=_mm256_add_ps(_mm256_loadu_ps(row+vv),pr);
                        _mm256_storeu_ps(row+vv,r);
                        _mm256_storeu_ps(oh+vv,_mm256_add_ps(_mm256_loadu_ps(oh+vv),
                                                             _mm256_mul_ps(vq,r)));
                    }
#else
                    for(int vv=0;vv<hd;vv++){ row[vv]+=kv*vt[vv]; oh[vv]+=qq*row[vv]; }
#endif
                }
            }
            double ms=0; for(int vv=0;vv<hd;vv++) ms+=(double)oh[vv]*oh[vv];
            if(g_debug&&li==0&&t==0&&h==0){
                fprintf(stderr,"DBG alpha[0:2] %.6f %.6f\n",alpha[0],alpha[1]);
                fprintf(stderr,"DBG qn[0:2] %.6f %.6f kn[0:2] %.6f %.6f beta %.6f\n",
                        qn[0],qn[1],kn[0],kn[1],beta);
                fprintf(stderr,"DBG vh[0:2] %.6f %.6f\n",vh[0],vh[1]);
                fprintf(stderr,"DBG oh[0:4] %.6f %.6f %.6f %.6f ms %.6e\n",oh[0],oh[1],oh[2],oh[3],ms);
                fprintf(stderr,"DBG vt[0:4] %.6f %.6f %.6f %.6f\n",vt[0],vt[1],vt[2],vt[3]);
                fprintf(stderr,"DBG S_after[0:4] %.6f %.6f %.6f %.6f\n",S[0],S[1],S[2],S[3]);
            }
            g_rec_work+=now_s()-tr0;
            float r=1.f/sqrtf((float)(ms/hd)+c->eps);
            float *dst=ont+(int64_t)h*hd;
            for(int vv=0;vv<hd;vv++) dst[vv]=oh[vv]*r*a->onw[vv]*sigmoidf_(gpt[(int64_t)h*hd+vv]);
        }
    }
    w_matmul(out,on,&a->o,C);
}

/* ---------- gated MLA with partial interleaved RoPE ---------- */
static void mla_forward(Model *m, Layer *l, int li, const float *x, int pos0, int C, float *out){
    Cfg *c=&m->c; Mla *a=&l->m;
    int H=c->n_heads, vh=c->v_head, kvl=c->kv_lora, qr=c->qk_rope, rd=c->rotary_dim, half=rd/2;
    float *qa=m->ws.qa, *qv=m->ws.qv, *ckv=m->ws.ckv, *gv=m->ws.gv, *ctx=m->ws.ctx;
    double mp0=now_s();
    w_matmul(qa,x,&a->qa,C);
    for(int t=0;t<C;t++)
        rmsnorm_(qa+(int64_t)t*c->q_lora,qa+(int64_t)t*c->q_lora,a->qa_ln,c->q_lora,c->eps);
    w_matmul(qv,qa,&a->qb,C);
    w_matmul(ckv,x,&a->kva,C);
    w_matmul(gv,x,&a->g,C);
    if(g_phases) m->t_mla_proj+=now_s()-mp0;
    /* cache chunk: latent rows + rotated k_rot; rotate this chunk's q_rot */
    for(int t=0;t<C;t++){
        int pos=pos0+t;
        float *Lrow=m->Lc[li]+(int64_t)pos*kvl, *Rrow=m->Rc[li]+(int64_t)pos*qr;
        const float *cv=ckv+(int64_t)t*(kvl+qr);
        rmsnorm_(Lrow,cv,a->kva_ln,kvl,c->eps);
        memcpy(Rrow,cv+kvl,qr*sizeof(float));
        const float *cs=m->cos_t+(int64_t)pos*half, *sn=m->sin_t+(int64_t)pos*half;
        for(int j=0;j<half;j++){                       /* interleaved pairs */
            float x0=Rrow[2*j], x1=Rrow[2*j+1], co=cs[j], si=sn[j];
            Rrow[2*j]  =x0*co-x1*si;
            Rrow[2*j+1]=x0*si+x1*co;
        }
        for(int h=0;h<H;h++){
            float *qrot=qv+(int64_t)t*H*c->qk_head+(int64_t)h*c->qk_head+c->qk_nope;
            for(int j=0;j<half;j++){
                float x0=qrot[2*j], x1=qrot[2*j+1], co=cs[j], si=sn[j];
                qrot[2*j]  =x0*co-x1*si;
                qrot[2*j+1]=x0*si+x1*co;
            }
        }
    }
    for(int tt=0;tt<C;tt++){
        int nt=pos0+tt+1;
        const float *qvt=qv+(int64_t)tt*H*c->qk_head, *gvt=gv+(int64_t)tt*H;
        float *ctxt=ctx+(int64_t)tt*H*vh;
        #pragma omp parallel for schedule(static) reduction(+:g_abs_work,g_val_work,g_ctx_work)
        for(int h=0;h<H;h++){
            const float *qp=qvt+(int64_t)h*c->qk_head, *qrp=qp+c->qk_nope;
            int rbase=h*(c->qk_nope+vh);
            float qabs[4096] __attribute__((aligned(32)));
            memset(qabs,0,kvl*sizeof(float));
            double wa0=now_s();
            for(int d=0;d<c->qk_nope;d++) w_addrow(&a->kvb,rbase+d,qp[d],qabs);
            g_abs_work+=now_s()-wa0;
            float sc[16384]; float mx=-1e30f;
            for(int t=0;t<nt;t++){
                const float *Lt=m->Lc[li]+(int64_t)t*kvl, *Rt=m->Rc[li]+(int64_t)t*qr;
                float s2=0; for(int i=0;i<kvl;i++) s2+=qabs[i]*Lt[i];
                for(int i=0;i<qr;i++) s2+=qrp[i]*Rt[i];
                sc[t]=s2*c->attn_scale;
                if(sc[t]>mx) mx=sc[t];
            }
            float sm=0; for(int t=0;t<nt;t++){ sc[t]=expf(sc[t]-mx); sm+=sc[t]; }
            for(int t=0;t<nt;t++) sc[t]/=sm;
            float clat[4096] __attribute__((aligned(32)));
            memset(clat,0,kvl*sizeof(float));
            double wv0=now_s(), wc0=0;
            for(int t=0;t<nt;t++){
                const float *Lt=m->Lc[li]+(int64_t)t*kvl; float s2=sc[t];
                wc0-=now_s();
                for(int i=0;i<kvl;i++) clat[i]+=s2*Lt[i];
                wc0+=now_s();
            }
            g_val_work+=now_s()-wv0;
            g_ctx_work+=wc0;
            float *cx=ctxt+(int64_t)h*vh;
            float gate=sigmoidf_(gvt[h]);               /* head-wise gate */
            if(getenv("L3_MLA_DEBUG")&&li==3&&tt==0&&h==0){
                fprintf(stderr,"MLADBG Lc0[0:3] %.6f %.6f %.6f\n",
                        m->Lc[li][0],m->Lc[li][1],m->Lc[li][2]);
                fprintf(stderr,"MLADBG qabs[0:3] %.6f %.6f %.6f\n",qabs[0],qabs[1],qabs[2]);
                fprintf(stderr,"MLADBG clat[0:3] %.6f %.6f %.6f\n",clat[0],clat[1],clat[2]);
                fprintf(stderr,"MLADBG gate %.6f sc0 %.6f\n",gate,sc[0]);
            }
            for(int d=0;d<vh;d++)
                cx[d]=w_rowdot(&a->kvb,rbase+c->qk_nope+d,clat)*gate;
        }
    }
    if(getenv("L3_CTX_DUMP")&&li==3){
        char cp2[600]; snprintf(cp2,sizeof(cp2),"%s/ctx_L%d.f32",getenv("L3_CTX_DUMP"),li);
        FILE *cf=fopen(cp2,"wb"); if(cf){ fwrite(ctx,sizeof(float),(size_t)C*H*vh,cf); fclose(cf);} else perror(cp2);
    }
    w_matmul(out,ctx,&a->o,C);
}

/* ---------- grouped sigmoid top-k router (noaux_tc, exact) ---------- */
static void route_select(const float *x, const Moe *o, const Cfg *c,
                         int li, int t, int *idx, float *wsel){
    float sco[4096], scores[4096], rt[4096], taken[4096];
    matmul(sco,x,o->router,1,c->hidden,c->n_experts);
    for(int e=0;e<c->n_experts;e++){
        scores[e]=sigmoidf_(sco[e]);
        rt[e]=scores[e]+o->rbias[e];
        taken[e]=0;
    }
    int Eg=c->n_experts/c->n_group;
    int gsel[128];
    float gsv[128];
    for(int g=0;g<c->n_group;g++){
        float b1=-1e30f,b2=-1e30f;
        for(int j=0;j<Eg;j++){
            float s=rt[g*Eg+j];
            if(s>b1){ b2=b1; b1=s; } else if(s>b2){ b2=s; }
        }
        gsv[g]=b1+b2; gsel[g]=g;
    }
    /* full selection of the top-topk_group groups: a partial pass leaves
     * slot topk_group-1 as an arbitrary leftover (measured at L3/t20: a
     * group with gsum 0.194 lost its slot to a 0.108 leftover). */
    for(int a=0;a<c->topk_group;a++)
        for(int b=a+1;b<c->n_group;b++)
            if(gsv[gsel[b]]>gsv[gsel[a]]){ int t=gsel[a];gsel[a]=gsel[b];gsel[b]=t; }
    unsigned char mask[4096]; memset(mask,0,(size_t)c->n_experts);
    for(int a=0;a<c->topk_group;a++)
        for(int j=0;j<Eg;j++) mask[gsel[a]*Eg+j]=1;
    for(int kk=0;kk<c->topk;kk++){
        int best=-1; float bv=-1e30f;
        for(int e=0;e<c->n_experts;e++)
            if(mask[e]&&!taken[e]&&rt[e]>bv){ bv=rt[e]; best=e; }
        idx[kk]=best; taken[best]=1;
    }
    if(getenv("L3_DEBUG_ROUTE")){
        const char *dp=getenv("L3_DEBUG_ROUTE");
        int dl=atoi(dp),dt=atoi(strchr(dp,',')+1);
        if(li==dl&&t==dt){
            int ord[4096]; for(int e=0;e<c->n_experts;e++) ord[e]=e;
            for(int a2=0;a2<c->n_experts-1;a2++)
                for(int b2=a2+1;b2<c->n_experts;b2++)
                    if(rt[ord[b2]]>rt[ord[a2]]){int tmp=ord[a2];ord[a2]=ord[b2];ord[b2]=tmp;}
            fprintf(stderr,"RTDBG L%d t%d selected(score):",li,t);
            for(int kk=0;kk<c->topk;kk++) fprintf(stderr," %d(%.5f)",idx[kk],scores[idx[kk]]);
            fprintf(stderr,"\nRTDBG top12 rt:");
            for(int j=0;j<12;j++) fprintf(stderr," %d(%.5f%s)",ord[j],rt[ord[j]],mask[ord[j]]?"":" MASKED");
            fprintf(stderr,"\nRTDBG gsums:");
            for(int g2=0;g2<c->n_group;g2++){
                float b1=-1e30f,b2=-1e30f;
                for(int j=0;j<Eg;j++){ float s2=rt[g2*Eg+j]; if(s2>b1){b2=b1;b1=s2;} else if(s2>b2)b2=s2; }
                fprintf(stderr," %.5f",b1+b2);
            }
            fprintf(stderr,"\n");
        }
    }
    float sm=0;
    for(int kk=0;kk<c->topk;kk++){ wsel[kk]=scores[idx[kk]]; sm+=wsel[kk]; }
    for(int kk=0;kk<c->topk;kk++) wsel[kk]=wsel[kk]/(sm+1e-20f)*c->routed_scale;
}

/* ---------- MoE forward (resident direct-indexed) ---------- */
static void moe_forward(Model *m, Layer *l, int li, const float *x, int C, float *out){
    Cfg *c=&m->c; Moe *o=&l->moe;
    int LT=c->hidden, MI=c->moe_inter;
    memset(out,0,(size_t)C*LT*sizeof(float));
    float *U=m->ws.moe_U; memset(U,0,(size_t)C*LT*sizeof(float));
    int *idxs=m->ws.moe_idx; float *wsels=m->ws.moe_w;
    double tr0=now_s();
    for(int t=0;t<C;t++)
        route_select(x+(int64_t)t*LT,o,c,li,t,idxs+(int64_t)t*c->topk,wsels+(int64_t)t*c->topk);
    m->t_router+=now_s()-tr0;
    if(m->routef) for(int t=0;t<C;t++){
        fwrite(&li,sizeof(int),1,m->routef);
        fwrite(&t,sizeof(int),1,m->routef);
        fwrite(idxs+(int64_t)t*c->topk,sizeof(int),c->topk,m->routef);
        fwrite(wsels+(int64_t)t*c->topk,sizeof(float),c->topk,m->routef);
    }
    double te0=now_s();
    {
        int map[4096]; for(int e=0;e<c->n_experts;e++) map[e]=-1;
        int nu=0;
        int *uid=(int*)m->ws.uid, *pcnt=(int*)m->ws.pcnt, *pfirst=(int*)m->ws.pfirst;
        int *poslist=(int*)m->ws.poslist, *cur=(int*)m->ws.cur;
        float *wlist=m->ws.wlist;
        memset(pcnt,0,(size_t)nu*sizeof(int));
        for(int t=0;t<C;t++) for(int kk=0;kk<c->topk;kk++){
            int e=idxs[(int64_t)t*c->topk+kk];
            if(map[e]<0){ map[e]=nu; uid[nu]=e; pcnt[nu]=0; nu++; }
            pcnt[map[e]]++;
        }
        int acc=0;
        for(int j=0;j<nu;j++){ pfirst[j]=acc; cur[j]=acc; acc+=pcnt[j]; }
        for(int t=0;t<C;t++) for(int kk=0;kk<c->topk;kk++){
            int j=map[idxs[(int64_t)t*c->topk+kk]];
            poslist[cur[j]]=t; wlist[cur[j]]=wsels[(int64_t)t*c->topk+kk]; cur[j]++;
        }
        float *gate=m->ws.gate, *up=m->ws.up, *hz=m->ws.hz;
        for(int j=0;j<nu;j++){
            Exp *eg=&o->exps[uid[j]*3+0], *eu=&o->exps[uid[j]*3+1], *ed=&o->exps[uid[j]*3+2];
            for(int p2=0;p2<pcnt[j];p2++){
                int t=poslist[pfirst[j]+p2];
                const float *xt=x+(int64_t)t*LT;
                exp_matvec(gate,xt,eg);
                exp_matvec(up,xt,eu);
                for(int i2=0;i2<MI;i2++) gate[i2]=siluf_(gate[i2])*up[i2];
                exp_matvec(hz,gate,ed);
                float wk=wlist[pfirst[j]+p2];
                for(int i2=0;i2<LT;i2++) U[(int64_t)t*LT+i2]+=wk*hz[i2];
            }
        }
        (void)0;
    }
    m->t_expert+=now_s()-te0;
    for(int64_t i2=0;i2<(int64_t)C*LT;i2++) out[i2]+=U[i2];
    (void)0;
    double ts0=now_s();
    {
        int SI=c->sh_inter;
        float *sg=m->ws.sg, *su=m->ws.su, *sd=m->ws.sd;
        w_matmul(sg,x,&o->sh_gate,C); w_matmul(su,x,&o->sh_up,C);
        for(int64_t i2=0;i2<(int64_t)C*SI;i2++) sg[i2]=siluf_(sg[i2])*su[i2];
        w_matmul(sd,sg,&o->sh_down,C);
        for(int64_t d=0;d<(int64_t)C*c->hidden;d++) out[d]+=sd[d];
    }
    m->t_shared+=now_s()-ts0;
}

static void dense_forward(Model *m, Layer *l, const float *x, int C, float *out){
    Cfg *c=&m->c; int DI=c->dense_inter;
    float *g=m->ws.dg, *u=m->ws.du;
    w_matmul(g,x,&l->d_gate,C); w_matmul(u,x,&l->d_up,C);
    for(int64_t i=0;i<(int64_t)C*DI;i++) g[i]=siluf_(g[i])*u[i];
    w_matmul(out,g,&l->d_down,C);
}

/* ---------- a CHUNK of C tokens through the stack ---------- */
static float *g_x0=NULL; static int g_x0_n=0;
static float *g_xl[128]={NULL}; static int g_xl_n=0;   /* per-layer injected inputs */
static float *g_mi[128]={NULL};                        /* per-layer injected MoE inputs */
static FILE *g_lfp=NULL;
static int g_th_dec=0;                                 /* decode-width thread policy */
static int g_th_wide=0;              /* L3_KDA_SCALAR=1 forces Wave-1 reference recurrence */                                /* prefill width */
static float *step_chunk(Model *m, const int *ids, int pos0, int C){
    Cfg *c=&m->c; int D=c->hidden;
    float *hidden=m->ws.hidden, *nrm=m->ws.nrm, *att=m->ws.att, *mlp=m->ws.mlp;
    double te0=now_s();
    if(g_x0){
        for(int t=0;t<C;t++){
            if(pos0+t>=g_x0_n){ fprintf(stderr,"L3_X0: pos %d beyond %d injected rows\n",pos0+t,g_x0_n); exit(1); }
            memcpy(hidden+(int64_t)t*D,g_x0+(int64_t)(pos0+t)*D,D*sizeof(float));
        }
    } else {
        const W *E=&m->embed;
        if(getenv("L3_GUARD")&&(!E->f||((uintptr_t)E->f<(uintptr_t)1<<40)))
            fprintf(stderr,"GUARD: suspicious E->f=%p\n",(void*)E->f);
        for(int t=0;t<C;t++){
            if(getenv("L3_GUARD")&&(ids[t]<0||ids[t]>=c->vocab)){
                fprintf(stderr,"GUARD: BAD id[%d]=%d (E->f=%p)\n",t,ids[t],(void*)E->f); exit(9); }
            if(E->fmt==0) memcpy(hidden+(int64_t)t*D,E->f+(int64_t)ids[t]*D,D*sizeof(float));
            else if(E->fmt==1){ const int8_t *p=E->q8+(int64_t)ids[t]*D; float s=E->s[ids[t]];
                for(int d=0;d<D;d++) hidden[(int64_t)t*D+d]=(float)p[d]*s; }
            else { int gs=E->gs,ng=D/gs; const uint8_t *p=E->q4+(int64_t)ids[t]*(D/2);
                const float *sl=E->s+(int64_t)ids[t]*ng;
                for(int g=0;g<ng;g++){ float s=sl[g];
                    for(int i=g*gs;i<(g+1)*gs;i+=2){ uint8_t b=p[i>>1];
                        hidden[(int64_t)t*D+i]=s*(float)((int)(b&0xF)-8);
                        hidden[(int64_t)t*D+i+1]=s*(float)((int)(b>>4)-8); } } }
        }
    }
    m->t_embed+=now_s()-te0;
    for(int i=0;i<c->n_layers;i++){
        Layer *l=&m->L[i];
        if(g_xl[i]){
            /* per-layer teacher forcing: restart the residual stream from the
             * reference implementation's state so each layer is validated
             * against identical inputs (defeats MoE routing chaos) */
            if(pos0!=0||C>g_xl_n){ fprintf(stderr,"L3_XLDIR requires single-chunk prefill\n"); exit(1); }
            memcpy(hidden,g_xl[i],(size_t)C*D*sizeof(float));
        }
        double tn0=now_s();
        for(int t=0;t<C;t++) rmsnorm_(nrm+(int64_t)t*D,hidden+(int64_t)t*D,l->in_ln,D,c->eps);
        m->t_norm+=now_s()-tn0;
        double ta0=now_s();
#ifdef _OPENMP
        if(g_th_wide>0){
            if(C==1){ if(g_th_dec>0) omp_set_num_threads(g_th_dec); }
            else omp_set_num_threads(g_th_wide);
        }
#endif
        if(l->mla){ double t0=now_s(); mla_forward(m,l,i,nrm,pos0,C,att); double dt=now_s()-t0; m->t_mla+=dt; m->t_attn+=dt; }
        else      { double t0=now_s(); kda_forward(m,l,i,nrm,C,att);  double dt=now_s()-t0; m->t_kda+=dt; m->t_attn+=dt; }
        if(getenv("L3_ATT_DUMP")){
            char ap[600]; snprintf(ap,sizeof(ap),"%s/att_L%d.f32",getenv("L3_ATT_DUMP"),i);
            FILE *af=fopen(ap,"wb"); if(af){ fwrite(att,sizeof(float),(size_t)C*D,af); fclose(af);} else perror(ap);
        }
        for(int t=0;t<C;t++)
            for(int d=0;d<D;d++) hidden[(int64_t)t*D+d]+=att[(int64_t)t*D+d];
        tn0=now_s();
        for(int t=0;t<C;t++) rmsnorm_(nrm+(int64_t)t*D,hidden+(int64_t)t*D,l->post_ln,D,c->eps);
        m->t_norm+=now_s()-tn0;
        if(g_mi[i]) memcpy(nrm,g_mi[i],(size_t)C*D*sizeof(float));
        if(l->sparse) moe_forward(m,l,i,nrm,C,mlp);
        else          dense_forward(m,l,nrm,C,mlp);
        if(getenv("L3_MLP_DUMP")){
            char ap2[600]; snprintf(ap2,sizeof(ap2),"%s/mlp_L%d.f32",getenv("L3_MLP_DUMP"),i);
            FILE *mf=fopen(ap2,"wb"); if(mf){ fwrite(mlp,sizeof(float),(size_t)C*D,mf); fclose(mf);} else perror(ap2);
        }
        for(int t=0;t<C;t++){
            for(int d=0;d<D;d++) hidden[(int64_t)t*D+d]+=mlp[(int64_t)t*D+d];
            if(m->trace) fwrite(hidden+(int64_t)t*D,sizeof(float),D,m->trace);
        }
    }
    float *logits=NULL;
    {
        double th0=now_s();
        for(int t=0;t<C;t++){
            if(!g_lfp && t<C-1) continue;
            float mix[16384];
            rmsnorm_(mix,hidden+(int64_t)t*D,m->final_norm,D,c->eps);
            if(m->trace) fwrite(mix,sizeof(float),(size_t)D,m->trace);
            float *lo=m->ws.logits;
            w_matmul(lo,mix,&m->lm_head,1);
            if(g_lfp) fwrite(lo,sizeof(float),(size_t)c->vocab,g_lfp);
            if(t==C-1) logits=lo;
        }
        m->t_head+=now_s()-th0;
    }
    /* workspace is persistent: no frees here (STRIKE 3) */
    return logits;
}

static int sample_greedy(const float *lo, int V){
    int b=0; for(int i=1;i<V;i++) if(lo[i]>lo[b]) b=i; return b;
}

int main(int argc, char **argv){
    const char *snap=NULL, *prompt=NULL, *ids_file=NULL;
    int ngen=16, maxt_env=0;
    for(int i=1;i<argc;i++){
        if(!strcmp(argv[i],"--model")&&i+1<argc) snap=argv[++i];
        else if(!strcmp(argv[i],"--prompt")&&i+1<argc) prompt=argv[++i];
        else if(!strcmp(argv[i],"--ids-file")&&i+1<argc) ids_file=argv[++i];
        else if(!strcmp(argv[i],"--ngen")&&i+1<argc) ngen=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--max-t")&&i+1<argc) maxt_env=atoi(argv[++i]);
        else { fprintf(stderr,"unknown arg %s\n",argv[i]); return 1; }
    }
    if(!snap){ fprintf(stderr,"usage: ling3 --model DIR [--prompt TEXT|--ids-file F|--ids env] [--ngen N]\n"); return 1; }
    Model *m=calloc(1,sizeof(Model));
    model_init(m,snap,getenv("L3_LAYERS")?atoi(getenv("L3_LAYERS")):0);
    if(getenv("L3_TRACE")) m->trace=fopen(getenv("L3_TRACE"),"wb");
    if(getenv("L3_LOGITS")) g_lfp=fopen(getenv("L3_LOGITS"),"wb");
    if(getenv("L3_ROUTE")) m->routef=fopen(getenv("L3_ROUTE"),"wb");
    if(getenv("L3_X0")){
        FILE *f=fopen(getenv("L3_X0"),"rb"); if(!f){perror("L3_X0");return 1;}
        fseek(f,0,SEEK_END); long nb=ftell(f); fseek(f,0,SEEK_SET);
        g_x0_n=(int)(nb/(4*(long)m->c.hidden));
        g_x0=falloc((int64_t)g_x0_n*m->c.hidden);
        if(fread(g_x0,4,(size_t)g_x0_n*m->c.hidden,f)!=(size_t)g_x0_n*m->c.hidden){fprintf(stderr,"L3_X0 short read\n");return 1;}
        fclose(f);
    }
    if(getenv("L3_XLDIR")){
        char p2[600];
        for(int i=0;i<m->c.n_layers;i++){
            snprintf(p2,sizeof(p2),"%s/xin_L%d.f32",getenv("L3_XLDIR"),i);
            FILE *f=fopen(p2,"rb");
            if(!f) break;
            fseek(f,0,SEEK_END); long nb=ftell(f); fseek(f,0,SEEK_SET);
            int rows=(int)(nb/(4*(long)m->c.hidden));
            if(g_xl_n&&rows!=g_xl_n){ fprintf(stderr,"L3_XLDIR row mismatch L%d\n",i); exit(1); }
            g_xl_n=rows;
            g_xl[i]=falloc((int64_t)rows*m->c.hidden);
            if(fread(g_xl[i],4,(size_t)rows*m->c.hidden,f)!=(size_t)rows*m->c.hidden){fprintf(stderr,"L3_XLDIR short %s\n",p2);exit(1);}
            fclose(f);
        }
        int nxl=0; for(int i=0;i<m->c.n_layers;i++) if(g_xl[i]) nxl++;
        fprintf(stderr,"[L3] per-layer teacher forcing active (%d layers, %d rows)\n",nxl,g_xl_n);
    }
    if(getenv("L3_MOE_IN_DIR")){
        char p3[700];
        for(int i=0;i<m->c.n_layers;i++){
            snprintf(p3,sizeof(p3),"%s/x_mlp_L%d.f32",getenv("L3_MOE_IN_DIR"),i);
            FILE *f=fopen(p3,"rb");
            if(!f) continue;
            fseek(f,0,SEEK_END); long nb3=ftell(f); fseek(f,0,SEEK_SET);
            int rows=(int)(nb3/(4*(long)m->c.hidden));
            g_mi[i]=falloc((int64_t)rows*m->c.hidden);
            if(fread(g_mi[i],4,(size_t)rows*m->c.hidden,f)!=(size_t)rows*m->c.hidden){fprintf(stderr,"MOE_IN short %s\n",p3);exit(1);}
            fclose(f);
        }
        int nmi=0; for(int i=0;i<m->c.n_layers;i++) if(g_mi[i]) nmi++;
        fprintf(stderr,"[L3] MoE-input teacher forcing active on %d layers\n",nmi);
    }
#ifdef _OPENMP
    /* measured knee (bench_v1 sweep): prefill scales to all cores, decode
     * regresses past 8 (OMP barrier cost on small per-layer regions).
     * Hybrid policy: wide during multi-token chunks, narrow for C==1.
     *
     * Thread contract (Wave-1 closeout):
     *   th_pref = L3_THREADS_PREF > L3_THREADS > omp_get_max_threads()
     *   th_dec  = L3_THREADS_DEC  > L3_THREADS > min(8, max_threads)
     * min(8,...) keeps hosts with fewer useful workers from being
     * oversubscribed by a hardcoded 8; the 8 itself is the MEASURED DELL
     * DEFAULT KNEE (i7-11700), not a universal architecture constant.
     * Set-but-invalid values (<=0) are rejected explicitly instead of being
     * silently clamped or ignored. */
    int omp_maxt=omp_get_max_threads();
    int th_pref=l3_thread_count("L3_THREADS_PREF");
    if(th_pref<=0) th_pref=l3_thread_count("L3_THREADS");
    if(th_pref<=0) th_pref=omp_maxt;
    int th_dec =l3_thread_count("L3_THREADS_DEC");
    if(th_dec<=0)  th_dec=l3_thread_count("L3_THREADS");
    if(th_dec<=0)  th_dec=(8<omp_maxt?8:omp_maxt);
    omp_set_dynamic(0);
    omp_set_num_threads(th_pref);
    g_th_dec=th_dec;
    g_th_wide=th_pref;
    fprintf(stderr,"[L3] threads: prefill=%d decode=%d (max=%d)\n",th_pref,th_dec,omp_maxt);
#endif
    int cap=65536, *ids=malloc((size_t)cap*sizeof(int)), np=0;
    if(m->c.bos>=0) ids[np++]=m->c.bos;
    if(ids_file){
        FILE *f=fopen(ids_file,"r"); if(!f){perror("ids-file");return 1;}
        int v; while(np<cap&&fscanf(f,"%d,",&v)==1) ids[np++]=v;
        fclose(f);
    } else if(prompt){
        Tok T; char tp[2048]; snprintf(tp,sizeof(tp),"%s/tokenizer.json",snap);
        tok_load(&T,tp);
        int en=tok_encode(&T,prompt,(int)strlen(prompt),ids+np,cap-np);
        if(en<0){fprintf(stderr,"tokenizer overflow\n");return 1;}
        np+=en;
    } else if(getenv("L3_IDS")){
        char *s=getenv("L3_IDS"),*p=s;
        while(*p&&np<cap){ int v=(int)strtol(p,&p,10); ids[np++]=v; while(*p==','||*p==' ')p++; }
    }
    if(g_x0) np=g_x0_n;
    if(g_xl[0]) np=g_xl_n;
    if(np<1){ fprintf(stderr,"empty prompt\n"); return 1; }
    int maxt=maxt_env?maxt_env:np+ngen+8;
    kv_alloc(m,maxt);
    model_state_reset(m);
    int phases=getenv("L3_PHASES")?atoi(getenv("L3_PHASES")):0;
    g_phases=phases;
    g_kda_scalar=getenv("L3_KDA_SCALAR")?atoi(getenv("L3_KDA_SCALAR")):0;
    g_heap_phase=1;
    double t_wall0=now_s();
    int chunk=getenv("L3_CHUNK")?atoi(getenv("L3_CHUNK")):32;
    if(chunk<1)chunk=1; if(chunk>512)chunk=512;
    double ttft=0;
    float *lo=NULL;
    for(int i=0;i<np;i+=chunk){
        int nc=np-i<chunk?np-i:chunk;
        lo=step_chunk(m,ids,i,nc);
        if(i==0) ttft=now_s()-t_wall0;
    }
    double t_prefill=now_s()-t_wall0;
    double t_dec0=now_s(); int produced=0;
    for(int g=0;g<ngen;g++){
        int next=sample_greedy(lo,m->c.vocab);
        printf("%d ",next); fflush(stdout);
        int stop=0; for(int e2=0;e2<m->c.n_eos;e2++) if(next==m->c.eos[e2]) stop=1;
        produced++;
        if(stop) break;
        lo=step_chunk(m,&next,np+g,1);
    }
    printf("\n");
    double t_dec=now_s()-t_dec0;
    fprintf(stderr,"[L3] prompt=%d prefill=%.3fs (%.1f tok/s) ttft~%.3fs decode=%d in %.3fs (%.1f tok/s) RSS=%.2fGB peakRSS=%.2fGB\n",
            np,t_prefill,t_prefill>0?np/t_prefill:0,ttft,produced,t_dec,t_dec>0?produced/t_dec:0,
            rss_gb(),peak_rss_gb());
    if(phases)
        fprintf(stderr,"[L3-PHASE] embed %.3fs attn %.3fs [kda %.3fs mla %.3fs | WORK kda_rec %.3fs mla_abs %.3fs mla_lat %.3fs mla_val %.3fs] norms %.3fs router %.3fs experts %.3fs shared %.3fs head %.3fs total_accounted %.3fs\n",
                m->t_embed,m->t_attn,m->t_kda,m->t_mla,
                g_rec_work,g_abs_work,g_ctx_work,g_val_work,
                m->t_norm,m->t_router,m->t_expert,m->t_shared,m->t_head,
                m->t_embed+m->t_attn+m->t_norm+m->t_router+m->t_expert+m->t_shared+m->t_head);
    if(m->trace) fclose(m->trace);
    if(g_lfp) fclose(g_lfp);
    if(m->routef) fclose(m->routef);
    if(getenv("L3_ALLOC_COUNT"))
        fprintf(stderr,"[L3-ALLOC] total heap allocs=%ld frees=%ld\n",g_allocs,g_frees);
    return 0;
}
