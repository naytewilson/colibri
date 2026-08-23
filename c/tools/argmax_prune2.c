/* argmax_prune2.c — skip-fraction discriminator on REAL lm_head F16 rows.
 * Exact-law check: Cauchy-Schwarz pruning preserves argmax; measures how many
 * vocab rows are skippable on genuine weights with synthetic activations.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <immintrin.h>

static double now_ms(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e3+t.tv_nsec*1e-6; }
static float f16u(unsigned u){ unsigned s=u>>15,e=(u>>10)&31,m=u&1023; double v;
    if(!e) v=ldexp(m,-10)* (1); else { e-=15; v=ldexp(1024+m,-10)*ldexp(1,e);} return (float)(s?-v:v); }

int main(int argc,char**argv){
    /* args: f16file offset bytes len D */
    const char*fn=argv[1]; long long off=atoll(argv[2]), nb=atoll(argv[3]); int D=atoi(argv[4]);
    long long O=nb/(D*2);
    FILE*f=fopen(fn,"rb"); if(!f)return 1;
    unsigned short*h=malloc(nb);
    fseek(f,off,SEEK_SET); fread(h,1,nb,f); fclose(f);
    float*W=malloc((size_t)O*D*sizeof(float));
    for(long long i=0;i<O*D;i++) W[i]=f16u(h[i]);
    free(h);
    fprintf(stderr,"loaded O=%lld D=%d\n",O,D);
    double*rj=malloc(O*sizeof(double)); int*ord=malloc(O*sizeof(int));
    for(long long o=0;o<O;o++){
        const float*w=W+o*D; __m256 acc=_mm256_setzero_ps();
        for(int i=0;i<D;i+=32){
            __m256 v0=_mm256_loadu_ps(w+i),v1=_mm256_loadu_ps(w+i+8),v2=_mm256_loadu_ps(w+i+16),v3=_mm256_loadu_ps(w+i+24);
            acc=_mm256_add_ps(acc,_mm256_add_ps(_mm256_mul_ps(v0,v0),_mm256_mul_ps(v1,v1)));
            acc=_mm256_add_ps(acc,_mm256_add_ps(_mm256_mul_ps(v2,v2),_mm256_mul_ps(v3,v3)));
        }
        float s[8];_mm256_storeu_ps(s,acc);
        rj[o]=sqrt(s[0]+s[1]+s[2]+s[3]+s[4]+s[5]+s[6]+s[7]); ord[o]=(int)o;
    }
    for(long long i=1;i<O;i++){int k=ord[i];double v=rj[k];long long j=i-1;while(j>=0&&rj[ord[j]]<v){ord[j+1]=ord[j];j--;}ord[j+1]=k;}
    srand(11);
    static float x[4096],yb[300000];
    long long sk=0,tot=0; int mm=0;
    double tf=0,tp=0,t0=0;
    enum{TOKS=12};
    for(int tk=0;tk<TOKS;tk++){
        for(int i=0;i<D;i++) x[i]=(float)((rand()%2001)-1000)/997.f;
        double xn=0;for(int i=0;i<D;i++)xn+=(double)x[i]*x[i]; xn=sqrt(xn);
        t0=now_ms();
        for(long long o=0;o<O;o++){
            const float*w=W+o*D; __m256 a0=_mm256_setzero_ps(),a1=_mm256_setzero_ps(),a2=_mm256_setzero_ps(),a3=_mm256_setzero_ps();
            for(int i=0;i+32<=D;i+=32){
                a0=_mm256_fmadd_ps(_mm256_loadu_ps(x+i),_mm256_loadu_ps(w+i),a0);
                a1=_mm256_fmadd_ps(_mm256_loadu_ps(x+i+8),_mm256_loadu_ps(w+i+8),a1);
                a2=_mm256_fmadd_ps(_mm256_loadu_ps(x+i+16),_mm256_loadu_ps(w+i+16),a2);
                a3=_mm256_fmadd_ps(_mm256_loadu_ps(x+i+24),_mm256_loadu_ps(w+i+24),a3);
            }
            __m256 a=_mm256_add_ps(_mm256_add_ps(a0,a1),_mm256_add_ps(a2,a3));
            __m128 t=_mm_add_ps(_mm256_castps256_ps128(a),_mm256_extractf128_ps(a,1));
            t=_mm_add_ps(t,_mm_movehl_ps(t,t)); t=_mm_add_ss(t,_mm_shuffle_ps(t,t,1));
            _mm_store_ss(&yb[o],t);
        }
        tf+=now_ms()-t0;
        t0=now_ms();
        float M=-1e30f; long long best=-1,sk2=0;
        for(long long ii=0;ii<O;ii++){
            int o=ord[ii];
            if(rj[o]*xn<=(double)M){sk2++;continue;}
            const float*w=W+(long long)o*D; __m256 a0=_mm256_setzero_ps(),a1=_mm256_setzero_ps(),a2=_mm256_setzero_ps(),a3=_mm256_setzero_ps();
            for(int i=0;i+32<=D;i+=32){
                a0=_mm256_fmadd_ps(_mm256_loadu_ps(x+i),_mm256_loadu_ps(w+i),a0);
                a1=_mm256_fmadd_ps(_mm256_loadu_ps(x+i+8),_mm256_loadu_ps(w+i+8),a1);
                a2=_mm256_fmadd_ps(_mm256_loadu_ps(x+i+16),_mm256_loadu_ps(w+i+16),a2);
                a3=_mm256_fmadd_ps(_mm256_loadu_ps(x+i+24),_mm256_loadu_ps(w+i+24),a3);
            }
            __m256 a=_mm256_add_ps(_mm256_add_ps(a0,a1),_mm256_add_ps(a2,a3));
            __m128 t=_mm_add_ps(_mm256_castps256_ps128(a),_mm256_extractf128_ps(a,1));
            t=_mm_add_ps(t,_mm_movehl_ps(t,t)); t=_mm_add_ss(t,_mm_shuffle_ps(t,t,1));
            float ac;_mm_store_ss(&ac,t);
            if(best<0||ac>M){M=ac;best=o;}
        }
        tp+=now_ms()-t0;
        sk+=sk2; tot+=O;
        float bm=-1e30f; long long bb=-1;
        for(long long o=0;o<O;o++) if(yb[o]>bm){bm=yb[o];bb=o;}
        if(bb!=best)mm++;
    }
    printf("skip%%=%.1f argmax_mismatch=%d full=%.1fms prune=%.1fms speedup=%.2fx\n",
           100.0*sk/tot,mm,tf/TOKS,tp/TOKS,tf/tp);
    return 0;
}
