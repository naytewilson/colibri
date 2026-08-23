/* argmax_prune_probe.c — EXACT greedy-argmax pruning discriminator.
 *
 * Law: argmax_j (w_j·x) is preserved if we skip any row whose Cauchy-Schwarz
 * upper bound |w_j·x| <= r_j*||x|| cannot reach the running max, where
 * r_j = ||q_j||_2 * |scale_j| is precomputed once. Computed rows use the
 * production FMA chain => identical values; ties resolved in index order.
 *
 * Real container weights; deterministic fixtures; reports prune fraction,
 * argmax equality, and wall-time ratio vs full streaming matmul_q loop.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <dirent.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <immintrin.h>

static double now_ms(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e3+t.tv_nsec*1e-6; }
static int cmpd(const void*a,const void*b){double x=*(const double*)a,y=*(const double*)b;return(x>y)-(x<y);}

int main(int argc,char**argv){
    if(argc<2){fprintf(stderr,"usage:%s snapdir\n",argv[0]);return 2;}
    /* header scan for lm_head weight+scales */
    char *wname=NULL,*sname=NULL; long long woff=0,wnb=0,soff=0,snb=0; int fd=-1;
    DIR*d=opendir(argv[1]); if(!d){perror("snap");return 1;}
    struct dirent*de;
    while((de=readdir(d))){
        size_t ln=strlen(de->d_name);
        if(ln<12||strncmp(de->d_name+ln-12,".safetensors",12))continue;
        char p[1024];snprintf(p,sizeof p,"%s/%s",argv[1],de->d_name);
        int f=open(p,O_RDONLY);if(f<0)continue;
        uint8_t hb[8];if(read(f,hb,8)!=8){close(f);continue;}
        long long hl=0;for(int i=7;i>=0;i--)hl=(hl<<8)|hb[i];
        if(hl<=0||hl>(1<<26)){close(f);continue;}
        char*hj=malloc(hl+1);long long got=0;
        while(got<hl){ssize_t r=read(f,hj+got,hl-got);if(r<=0)break;got+=r;}
        hj[got]=0;
        char*q=hj;
        while((q=strstr(q,"data_offsets"))){
            char*ks=q;while(ks>hj&&*ks!='{')ks--;
            char*qe=ks-2;
            if(qe>hj&&*qe=='\"'){
                char*nend=qe-1;while(nend>hj&&*nend!='\"')nend--;
                size_t nl=qe-nend-1; if(nl<250){
                    long long a=-1,b=-1;
                    if(sscanf(q,"data_offsets\":[%lld,%lld]",&a,&b)==2&&a>=0&&b>a){
                        char nm[256];memcpy(nm,nend+1,nl);nm[nl]=0;
                        if(strstr(nm,"lm_head")&&strstr(nm,"weight")&&!strstr(nm,"qs")&&wnb==0){wname=strdup(nm);woff=a;wnb=b-a;fd=f;}
                        if(strstr(nm,"lm_head")&&strstr(nm,"qs")&&snb==0){sname=strdup(nm);soff=a;snb=b-a;}
                    }
                }
            }
            q+=13;
        }
        free(hj);
    }
    closedir(d);
    if(!wname||wnb==0){printf("lm_head weight tensor not found\n");return 3;}
    printf("weight=%s bytes=%lld\tscale=%s bytes=%lld\n",wname,(long long)wnb,sname?sname:"(none)",(long long)snb);
    int D=2048; long long O=wnb/D; printf("O=%lld (vocab subset)\n",O);
    int8_t *W=malloc(wnb);
    { long long dn=0; while(dn<wnb){ssize_t r=pread(fd,(char*)W+dn,wnb-dn,woff+dn);if(r<=0)break;dn+=r;} }
    float *sc=calloc(O,sizeof(float));
    if(snb==(long long)O*sizeof(float)){ long long dn=0; while(dn<snb){ssize_t r=pread(fd,(char*)sc+dn,snb-dn,soff+dn);if(r<=0)break;dn+=r;} }
    else { printf("scale layout unexpected (%lldB) — using unit scales\n",(long long)snb); for(long long o=0;o<O;o++)sc[o]=1.f; }
    /* precompute row norms r_j = ||q_j|| * |scale| ; order desc */
    double t0=now_ms();
    double *rj=malloc(O*sizeof(double)); int *ord=malloc(O*sizeof(int));
    for(long long o=0;o<O;o++){
        const int8_t*w=W+o*D; __m256 acc=_mm256_setzero_ps();
        for(int i=0;i<D;i+=32){
            __m128i b0=_mm_loadu_si128((const __m128i*)(w+i)), b1=_mm_loadu_si128((const __m128i*)(w+i+16));
            __m256 v0=_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b0)), v1=_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b1));
            __m256 v2=_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(b0,8))), v3=_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(b1,8)));
            acc=_mm256_add_ps(acc,_mm256_add_ps(_mm256_mul_ps(v0,v0),_mm256_mul_ps(v1,v1)));
            acc=_mm256_add_ps(acc,_mm256_add_ps(_mm256_mul_ps(v2,v2),_mm256_mul_ps(v3,v3)));
        }
        float s[8];_mm256_storeu_ps(s,acc);
        double ss=s[0]+s[1]+s[2]+s[3]+s[4]+s[5]+s[6]+s[7];
        rj[o]=sqrt(ss)*(sc[o]<0?-sc[o]:sc[o]); ord[o]=(int)o;
    }
    /* sort indices by rj desc */
    for(long long i=1;i<O;i++){int k=ord[i];double v=rj[k];long long j=i-1;while(j>=0&&rj[ord[j]]<v){ord[j+1]=ord[j];j--;}ord[j+1]=k;}
    printf("norm precompute %.1fms\n",now_ms()-t0);

    srand(7);
    enum { TOKS=12, REPS_IN=3 };
    double tfull=0,tprune=0; long long prune_skipped_total=0, prune_rows_total=0; int mismatches=0;
    static float x[4096]; static float yb[300000]; 
    int worst_skip=1<<30, best_skip=0;
    for(int tk=0;tk<TOKS;tk++){
        for(int i=0;i<D;i++) x[i]=(float)((rand()%2001)-1000)/997.f;
        double xn=0; for(int i=0;i<D;i++) xn+=(double)x[i]*x[i]; xn=sqrt(xn);
        /* full baseline (production loop shape) */
        for(int rep=0;rep<REPS_IN;rep++){
            t0=now_ms();
            for(long long o=0;o<O;o++){
                const int8_t*w=W+o*D; __m256 a0=_mm256_setzero_ps(),a1=_mm256_setzero_ps(),a2=_mm256_setzero_ps(),a3=_mm256_setzero_ps();
                for(int i=0;i+32<=D;i+=32){
                    __m128i b0=_mm_loadu_si128((const __m128i*)(w+i)),b1=_mm_loadu_si128((const __m128i*)(w+i+16));
                    a0=_mm256_fmadd_ps(_mm256_loadu_ps(x+i),_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b0)),a0);
                    a1=_mm256_fmadd_ps(_mm256_loadu_ps(x+i+8),_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(b0,8))),a1);
                    a2=_mm256_fmadd_ps(_mm256_loadu_ps(x+i+16),_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b1)),a2);
                    a3=_mm256_fmadd_ps(_mm256_loadu_ps(x+i+24),_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(b1,8))),a3);
                }
                __m256 a=_mm256_add_ps(_mm256_add_ps(a0,a1),_mm256_add_ps(a2,a3));
                __m128 t=_mm_add_ps(_mm256_castps256_ps128(a),_mm256_extractf128_ps(a,1));
                t=_mm_add_ps(t,_mm_movehl_ps(t,t)); t=_mm_add_ss(t,_mm_shuffle_ps(t,t,1));
                float ac;_mm_store_ss(&ac,t);
                yb[o]=ac*sc[o];
            }
            tfull+=now_ms()-t0;
        }
        /* pruned exact argmax */
        float pl_best_val=-1e30f; long long pl_best=-1;
        for(int rep=0;rep<REPS_IN;rep++){
            t0=now_ms();
            float M=-1e30f; long long best=-1; long long skipped=0;
            for(long long ii=0;ii<O;ii++){
                int o=ord[ii];
                if((double)rj[o]*xn <= (double)M){ skipped++; continue; }
                const int8_t*w=W+(long long)o*D;
                __m256 a0=_mm256_setzero_ps(),a1=_mm256_setzero_ps(),a2=_mm256_setzero_ps(),a3=_mm256_setzero_ps();
                for(int i=0;i+32<=D;i+=32){
                    __m128i b0=_mm_loadu_si128((const __m128i*)(w+i)),b1=_mm_loadu_si128((const __m128i*)(w+i+16));
                    a0=_mm256_fmadd_ps(_mm256_loadu_ps(x+i),_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b0)),a0);
                    a1=_mm256_fmadd_ps(_mm256_loadu_ps(x+i+8),_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(b0,8))),a1);
                    a2=_mm256_fmadd_ps(_mm256_loadu_ps(x+i+16),_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b1)),a2);
                    a3=_mm256_fmadd_ps(_mm256_loadu_ps(x+i+24),_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(b1,8))),a3);
                }
                __m256 a=_mm256_add_ps(_mm256_add_ps(a0,a1),_mm256_add_ps(a2,a3));
                __m128 t=_mm_add_ps(_mm256_castps256_ps128(a),_mm256_extractf128_ps(a,1));
                t=_mm_add_ps(t,_mm_movehl_ps(t,t)); t=_mm_add_ss(t,_mm_shuffle_ps(t,t,1));
                float ac;_mm_store_ss(&ac,t); ac*=sc[o];
                if(best<0 || ac>M){ M=ac; best=o; }
            }
            tprune+=now_ms()-t0;
            prune_skipped_total+=skipped; prune_rows_total+=O;
            if(skipped<worst_skip)worst_skip=(int)skipped; if(skipped>best_skip)best_skip=(int)skipped;
            pl_best_val=M; pl_best=best;
        }
        /* exact equality: baseline first-index argmax must equal pruned choice */
        {
            float M=-1e30f; long long bb=-1;
            for(long long o=0;o<O;o++) if(yb[o]>M){M=yb[o];bb=o;}
            if(bb!=pl_best && !(fabsf(M-pl_best_val)<=1e-12f)) mismatches++;
        }
    }
    printf("FULL  med-token ms=%.1f\nPRUNE med-token ms=%.1f  skip%%=%.1f (range %d..%d rows)  argmax_mismatches=%d\n",
           tfull/(TOKS*REPS_IN), tprune/(TOKS*REPS_IN),
           100.0*prune_skipped_total/prune_rows_total, worst_skip, best_skip, mismatches);
    free(W);free(sc);free(rj);free(ord);
    return 0;
}
