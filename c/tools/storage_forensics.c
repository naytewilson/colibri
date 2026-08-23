/* storage_forensics.c — Wave D/E/F offline tools for qwen36 containers.
 *
 * scan <snapdir>            : safetensors header census — expert tensor sizes,
 *                             offsets, alignment, inter-expert gaps.
 * read <snapdir> <L> <eid>  : cold-read microbench of one routed expert
 *                             (fadvise DONTNEED between reps), current bytes
 *                             vs a gs128-equivalent byte budget (-7.14%).
 * Both paths are read-only; container is never modified.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#ifndef POSIX_FADV_DONTNEED
#define POSIX_FADV_DONTNEED 4
#endif
#if defined(__APPLE__)
static int posix_fadvise(int fd, off_t o, off_t n, int a){ (void)fd;(void)o;(void)n;(void)a; return 0; }
#endif

typedef struct { char name[256]; long long off, nbytes; int fd; } TEN;

static double now_ms(void){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec*1e3+ts.tv_nsec/1e6; }

static int cmp_ten(const void *a, const void *b){ const TEN *x=a,*y=b; return (x->off>y->off)-(x->off<y->off); }

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr,"usage: %s scan <dir> | %s read <dir> <L> <eid>\n",argv[0],argv[0]); return 2; }
    int doread = strcmp(argv[1],"read")==0;
    int wantL=-1,wantE=-1;
    if (doread){ wantL=atoi(argv[3]); wantE=atoi(argv[4]); }
    DIR *d = opendir(argv[2]); if(!d){perror(argv[2]);return 1;}
    struct dirent *de;
    static TEN tens[200000]; int nt=0;
    while ((de=readdir(d))) {
        size_t ln=strlen(de->d_name);
        if (ln<5 || strcmp(de->d_name+ln-5,".json")||0) ; /* fallthrough below */
        if (ln<12 || strncmp(de->d_name+ln-12,".safetensors",12)!=0) continue;
        char path[1024]; snprintf(path,sizeof path,"%s/%s",argv[2],de->d_name);
        int fd=open(path,O_RDONLY); if(fd<0)continue;
        unsigned char hb[8]; if(read(fd,hb,8)!=8){close(fd);continue;}
        long long hlen=0; for(int i=7;i>=0;i--) hlen=(hlen<<8)|hb[i];
        if(hlen<=0||hlen>(1<<26)){close(fd);continue;}
        char *hj=malloc(hlen+1); 
        long long got=0; while(got<hlen){ ssize_t r=read(fd,hj+got,hlen-got); if(r<=0)break; got+=r; }
        hj[got]=0;
        /* crude but robust scan: "name":{"dtype":..,"shape":[..],"data_offsets":[a,b]} */
        char *q=hj;
        while ((q=strstr(q,"data_offsets"))) {
            char *nb=q;
            /* walk back to the key start: find preceding '"name":{' */
            char *ks=q; int depth=0;
            while (ks>hj) { if(*ks=='{')break; ks--; }
            char *ke=ks-2; /* should point at ':' after name quote end */
            /* find name start: last '"' before ks that closes a key string */
            char *qe=ks-2; /* char before '{' is '"'? pattern: "name":{"dtype" */
            if (qe>hj && *qe=='"') {
                char *qs=qe-1; while(qs>hj && *(qs)!='"') qs--; qs++;
                size_t nlen=qe-qs; if(nlen<sizeof(((TEN*)0)->name)){
                    long long a=-1,b=-1;
                    if (sscanf(q,"data_offsets\":[%lld,%lld]",&a,&b)==2 && a>=0&&b>a){
                        TEN *t=&tens[nt];
                        memcpy(t->name,qs,nlen); t->name[nlen]=0; t->off=a; t->nbytes=b-a; t->fd=fd; nt++;
                    }
                }
            }
            (void)nb;(void)ke;(void)depth;
            q+=13;
        }
        free(hj);
        /* keep fd open until end */
        if (nt>=200000) break;
        continue;
    }
    closedir(d);
    fprintf(stderr,"tensors=%d\n",nt);

    if (!doread) {
        long long ecount=0, emin=1LL<<60, emax=0, esum=0;
        int misaligned=0; long long gapmin=1LL<<60,gapmax=-1,gapsum=0; int ngaps=0;
        long long maxgap_off_file=-1; char where[256]="";
        for (int i=0;i<nt;i++){
            TEN*t=&tens[i];
            if (!strstr(t->name,"mlp.experts.")||!strstr(t->name,"merged_weight")) continue;
            ecount++; esum+=t->nbytes; if(t->nbytes<emin)emin=t->nbytes; if(t->nbytes>emax)emax=t->nbytes;
            if (t->off & 4095) misaligned++;
        }
        /* gaps between consecutive expert tensors within same file */
        for (int i=1;i<nt;i++){
            TEN*a=&tens[i-1],*b=&tens[i];
            if (a->fd!=b->fd) continue;
            if (!strstr(a->name,"mlp.experts.")||!strstr(a->name,"merged_weight")) continue;
            if (!strstr(b->name,"mlp.experts.")||!strstr(b->name,"merged_weight")) continue;
            long long g=b->off-(a->off+a->nbytes);
            if(g>0){ngaps++;gapsum+=g;if(g<gapmin)gapmin=g;if(g>gapmax){gapmax=g;} if(g==gapmax&&g>4096)snprintf(where,sizeof where,"~%lldB before %s",g,b->name);}
        }
        printf("EXPERTS=%d min_bytes=%lld max_bytes=%lld avg_bytes=%.0f misaligned_4k=%d\n",
               ecount,emin==1LL<<60?-1:emin,emax,(double)esum/(ecount?ecount:1),misaligned);
        printf("INTER_EXPERT_GAPS n=%d min=%lld max=%lld sum=%lld (%s)\n",ngaps,
               gapmin==1LL<<60?-1:gapmin,gapmax<0?-1:gapmax,gapsum,where);
        /* gs128 model-wide bound: INT3 experts save scales: nbytes_w3 = packed(24B/64w)+4B*scale per group
           groups = floor(W/64); gs128 saves 4*(groups - floor(W/128)) bytes ≈ groups*4/2 */
        double w3_total_g64=0, w3_total_g128=0;
        for (int i=0;i<nt;i++){
            TEN*t=&tens[i];
            if (!strstr(t->name,"merged_weight")) continue;
            if (t->nbytes!=1376256 && t->nbytes!=1835008) continue; /* int3-g64 / int4-ish */
            if (t->nbytes==1376256){
                long long W=3145728; /* 2*inter*hidden + hidden*inter at 512x2048 */
                long long g64=W/64, g128=W/128;
                long long b64=t->nbytes;                 /* includes 4*g64 scales */
                long long b128=b64 - 4*(g64-g128);
                w3_total_g64+=b64; w3_total_g128+=b128;
            }
        }
        if (w3_total_g64>0)
            printf("MODEL_GS128_BOUND g64_bytes=%.3fGB g128_bytes=%.3fGB saving=%.2f%%\n",
                   w3_total_g64/1e9,w3_total_g128/1e9,100.0*(1-w3_total_g128/w3_total_g64));
        return 0;
    }

    /* READ micro: locate target expert weight tensor */
    char want[256]; snprintf(want,sizeof want,"model.layers.%d.mlp.experts.%d.merged_weight",wantL,wantE);
    TEN *tw=NULL;
    for (int i=0;i<nt;i++) if (strcmp(tens[i].name,want)==0){ tw=&tens[i]; break; }
    if(!tw){ printf("MISSING %s\n",want); return 3; }
    long long W=3145728, g64=W/64, g128=W/128;
    long long cur=tw->nbytes;
    long long alt = (cur==1376256) ? cur - 4*(g64-g128) : (cur*15)/16; /* gs128 for int3; ~-6.25% generic */
    printf("TARGET %s bytes_cur=%lld bytes_gs128=%lld reduction=%.2f%%\n",want,cur,alt,100.0*(1-(double)alt/cur));
    char *buf=malloc(cur);
    double tcur[32],talt[32]; int nr=24;
    /* warm page out first pass */
    for (int r=-2;r<nr;r++){
        posix_fadvise(tw->fd,tw->off,cur,POSIX_FADV_DONTNEED);
        double t0=now_ms();
        long long done=0; while(done<cur){ ssize_t rr=pread(tw->fd,buf+done,cur-done,tw->off+done); if(rr<=0)break; done+=rr; }
        double dt=now_ms()-t0;
        volatile long long sink=0; for(long long z=0;z<cur;z+=4096) sink+=buf[z]; (void)sink;
        if(r>=0)tcur[r]=dt;
    }
    for (int r=-2;r<nr;r++){
        posix_fadvise(tw->fd,tw->off,cur,POSIX_FADV_DONTNEED);
        double t0=now_ms();
        long long done=0; while(done<alt){ ssize_t rr=pread(tw->fd,buf+done,alt-done,tw->off+done); if(rr<=0)break; done+=rr; }
        double dt=now_ms()-t0;
        volatile long long sink=0; for(long long z=0;z<alt;z+=4096) sink+=buf[z]; (void)sink;
        if(r>=0)talt[r]=dt;
    }
    for(int i=1;i<nr;i++){double k=tcur[i];int j=i-1;while(j>=0&&tcur[j]>k){tcur[j+1]=tcur[j];j--;}tcur[j+1]=k;}
    for(int i=1;i<nr;i++){double k=talt[i];int j=i-1;while(j>=0&&talt[j]>k){talt[j+1]=talt[j];j--;}talt[j+1]=k;}
    printf("COLD_READ median_cur=%.3fms median_gs128size=%.3fms delta=%.3fms ratio=%.3f (reps=%d)\n",
           tcur[nr/2],talt[nr/2],tcur[nr/2]-talt[nr/2],talt[nr/2]/tcur[nr/2],nr);
    free(buf);
    return 0;
}
