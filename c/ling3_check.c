/* ling3_check — NATIVE parity adjudicator for the Ling3 Colibri port.
 * Replaces the Python parity script: the Dell validation path stays native.
 *
 * Compares engine diagnostic dumps (raw f32) against reference fixture dumps
 * (raw f32 companions emitted by the fixture generator):
 *
 *   hidden:  ref [n_stages][T][D], got [<= n_stages][T][D] (trailing
 *            post-final-norm rows ignored)
 *   logits:  ref [T][Vr], got [T][Vg] (V = min(Vr,Vg) columns compared)
 *   router:  fixture JSON [{layer,idx[],w[]}...] vs engine binary records
 *            {int layer,int t,int idx[topk],float w[topk]}
 *
 * usage:
 *   ling3_check --ref-hidden F --got-hidden F --stages N --T N --D N
 *               [--hidden-atol A] [--hidden-rtol R]
 *   ling3_check --ref-logits F --got-logits F --T N --V N
 *   ling3_check --ref-router fixture.json --got-router route.bin [--topk K]
 * Exit code 0 only when every supplied check passes.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "json.h"

static void *slurp(const char *path, long *n){
    FILE *f=fopen(path,"rb"); if(!f){perror(path);exit(2);}
    fseek(f,0,SEEK_END); *n=ftell(f); fseek(f,0,SEEK_SET);
    if(*n<0||(*n&3)){ fprintf(stderr,"%s: bad f32 size %ld\n",path,*n); exit(2); }
    void *b=malloc((size_t)*n? (size_t)*n:1);
    if(!b){fprintf(stderr,"OOM\n");exit(2);}
    if(fread(b,1,(size_t)*n,f)!=(size_t)*n){fprintf(stderr,"%s: short read\n",path);exit(2);}
    fclose(f); return b;
}
static int fails=0, checks=0;
static void verdict(const char *name,double maxdiff,double atol,double rtol,long agree,long total){
    checks++;
    if(agree>=0){
        int ok=(agree==total);
        if(!ok) fails++;
        printf("CHECK %s: %ld/%ld -> %s\n",name,agree,total,ok?"PASS":"FAIL");
    } else {
        int ok=(maxdiff<=atol+rtol);
        if(!ok) fails++;
        printf("CHECK %s: max_abs_diff=%g atol=%g rtol=%g -> %s\n",
               name,maxdiff,atol,rtol,ok?"PASS":"FAIL");
    }
}
static long g_ref_off=0,g_got_off=0;
static double cmp_hidden(const char *ref,const char *got,long stages,long T,long D,
                         double atol,double rtol){
    long nr,ng;
    float *a=slurp(ref,&nr),*b=slurp(got,&ng);
    nr/=4; ng/=4;
    if(nr%(T*D)){ long keep=nr-(nr%(T*D)); fprintf(stderr,"NOTE ref: %ld trailing floats ignored\n",nr-keep); nr=keep; }
    if(ng%(T*D)){ long keep=ng-(ng%(T*D)); fprintf(stderr,"NOTE got: %ld trailing floats ignored\n",ng-keep); ng=keep; }
    long sr=nr/(T*D), sg=ng/(T*D);
    if(sr<stages||sg<stages){fprintf(stderr,"stages %ld/%ld < %ld\n",sr,sg,stages);exit(2);}
    if(sr<stages+g_ref_off||sg<stages+g_got_off){fprintf(stderr,"stage offset exceeds dumps (%ld/%ld)\n",sr,sg);exit(2);}
    if(g_ref_off||g_got_off) printf("NOTE mapping: ref[%ld..%ld] <-> got[%ld..%ld]\n",
        g_ref_off,g_ref_off+stages-1,g_got_off,g_got_off+stages-1);
    double worst=0;
    for(long s=0;s<stages;s++)
        for(long i=0;i<T*D;i++){
            double d=fabs((double)a[(s+g_ref_off)*T*D+i]-b[(s+g_got_off)*T*D+i]);
            if(d>worst) worst=d;
        }
    free(a);free(b);
    return worst;
}
static double cmp_logits_argmax(const char *ref,const char *got,long T,long V,
                                double atol,double rtol,long *agree){
    long nr,ng;
    float *a=slurp(ref,&nr),*b=slurp(got,&ng);
    nr/=4; ng/=4;
    if(nr<T*V||ng<T*V){fprintf(stderr,"logits too small (%ld/%ld vs T*V=%ld)\n",nr,ng,T*V);exit(2);}
    double worst=0; long ag=0;
    for(long t=0;t<T;t++){
        const float *ra=a+t*V,*gb=b+t*V;
        int ma=0,mb=0;
        for(long v=1;v<V;v++){
            double d=fabs((double)ra[v]-gb[v]);
            if(d>worst) worst=d;
            if(ra[v]>ra[ma]) ma=(int)v;
            if(gb[v]>gb[mb]) mb=(int)v;
        }
        if(ma==mb) ag++;
    }
    free(a);free(b);
    *agree=ag;
    return worst;
}
/* returns max |w| diff over matched records; sets exact-match count */
static double cmp_router(const char *refjson,const char *gotbin,int topk,long *exact,long *total){
    FILE *f=fopen(refjson,"rb"); if(!f){perror(refjson);exit(2);}
    fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
    char *buf=malloc((size_t)n+1);
    if(fread(buf,1,(size_t)n,f)!=(size_t)n){fprintf(stderr,"short read\n");exit(2);}
    buf[n]=0; fclose(f);
    char *arena=NULL;
    jval *root=json_parse(buf,&arena);
    if(!root||root->t!=J_ARR){fprintf(stderr,"%s: not a JSON array\n",refjson);exit(2);}
    /* index fixture records by (layer,pos-in-layer); fixture stores
     * idx/w as [T][topk] nested arrays per layer record */
    struct { int layer,pos; jval *idx,*w; } tab[4096]; int nt=0;
    for(int i=0;i<root->len&&nt<4096;i++){
        jval *r=root->kids[i];
        jval *lay=json_get(r,"layer"),*idx=json_get(r,"idx"),*w=json_get(r,"w");
        if(!lay||!idx||!w) continue;
        int npos=idx->t==J_ARR&&idx->len>0&&idx->kids[0]->t==J_ARR?idx->len:1;
        for(int p2=0;p2<npos&&nt<4096;p2++){
            tab[nt].layer=(int)lay->num; tab[nt].pos=p2;
            tab[nt].idx=npos>1?idx->kids[p2]:idx;
            tab[nt].w  =npos>1?w->kids[p2]:w;
            nt++;
        }
    }
    long nb; unsigned char *bin=slurp(gotbin,&nb);
    if(getenv("CHK_DEBUG"))
        fprintf(stderr,"DBG nt=%d rec0.layer=%d npos_idx=%d nb=%ld\n",nt,
                nt?tab[0].layer:-1,
                (nt&&tab[0].idx->t==J_ARR)?tab[0].idx->len:-1,nb);
    double worst=0; long ex=0,tot=0,off=0;
    while(off+(long)(8+topk*8)<=(long)nb && tot<nt){
        int li,t; memcpy(&li,bin+off,4); memcpy(&t,bin+off+4,4); off+=8;
        const int *gidx=(const int*)(bin+off); off+=4*topk;
        const float *gw=(const float*)(bin+off); off+=4*topk;
        /* find matching fixture record (same layer, same position order) */
        for(int i=0;i<nt;i++){
            if(tab[i].layer!=li||tab[i].pos!=t) continue;
            tot++;
            int same=1;
            for(int kk=0;kk<topk;kk++){
                jval *iv=tab[i].idx->kids[kk];
                if((int)iv->num!=gidx[kk]) same=0;
                double d=fabs((double)tab[i].w->kids[kk]->num-gw[kk]);
                if(d>worst) worst=d;
            }
            if(same) ex++;
            break;
        }
    }
    free(bin);
    *exact=ex;
    *total=tot;
    return worst;
}
int main(int argc,char **argv){
    const char *rh=0,*gh=0,*rl=0,*gl=0,*rr=0,*gr=0;
    long stages=0,T=0,D=0,V=0,topk=8;
    double h_atol=0.02,h_rtol=0.02,latol=0.15,lrtol=0.02;
    for(int i=1;i<argc;i++){
        #define ARG(s,v) if(!strcmp(argv[i],s)&&i+1<argc){v=argv[++i];continue;}
        ARG("--ref-hidden",rh) ARG("--got-hidden",gh)
        ARG("--ref-logits",rl) ARG("--got-logits",gl)
        ARG("--ref-router",rr) ARG("--got-router",gr)
        #undef ARG
        if(!strcmp(argv[i],"--stages")&&i+1<argc){stages=atol(argv[++i]);continue;}
        if(!strcmp(argv[i],"--T")&&i+1<argc){T=atol(argv[++i]);continue;}
        if(!strcmp(argv[i],"--D")&&i+1<argc){D=atol(argv[++i]);continue;}
        if(!strcmp(argv[i],"--V")&&i+1<argc){V=atol(argv[++i]);continue;}
        if(!strcmp(argv[i],"--topk")&&i+1<argc){topk=atoi(argv[++i]);continue;}
        if(!strcmp(argv[i],"--ref-offset")&&i+1<argc){g_ref_off=atol(argv[++i]);continue;}
        if(!strcmp(argv[i],"--got-offset")&&i+1<argc){g_got_off=atol(argv[++i]);continue;}
        if(!strcmp(argv[i],"--hidden-atol")&&i+1<argc){h_atol=atof(argv[++i]);continue;}
        if(!strcmp(argv[i],"--hidden-rtol")&&i+1<argc){h_rtol=atof(argv[++i]);continue;}
        if(!strcmp(argv[i],"--logits-atol")&&i+1<argc){latol=atof(argv[++i]);continue;}
        if(!strcmp(argv[i],"--logits-rtol")&&i+1<argc){lrtol=atof(argv[++i]);continue;}
        fprintf(stderr,"unknown arg %s\n",argv[i]); return 2;
    }
    if(rh&&gh){
        if(stages<1||T<1||D<1){fprintf(stderr,"need --stages --T --D\n");return 2;}
        double md=cmp_hidden(rh,gh,stages,T,D,h_atol,h_rtol);
        verdict("hidden_states",md,h_atol,h_rtol,-1,0);
    }
    if(rl&&gl){
        if(T<1||V<1){fprintf(stderr,"need --T --V\n");return 2;}
        long ag=0;
        double md=cmp_logits_argmax(rl,gl,T,V,latol,lrtol,&ag);
        verdict("logits_values",md,latol,lrtol,-1,0);
        verdict("logits_argmax",0,0,0,ag,T);
    }
    if(rr&&gr){
        long ex=0,tot=0;
        double wd=cmp_router(rr,gr,(int)topk,&ex,&tot);
        char nm[64]; snprintf(nm,sizeof(nm),"router_exact (max w-diff %.3g)",wd);
        verdict(nm,0,0,0,ex,tot?tot:-1);
    }
    if(!checks){fprintf(stderr,"nothing to check\n");return 2;}
    printf("PARITY_VERDICT: %s (%d checks)\n",fails?"FAIL":"PASS",checks);
    return fails?1:0;
}
