/* krs_probe.c — Kernel-Readahead Speculation cheap discriminators.
 * P1: does fadvise(WILLNEED)+delay turn a cold 1.x MB expert read into a fast
 *     page-cache-served pread ON THIS DEVICE? With what delay? Under one
 *     intervening demand-sized read (collision)?
 * P2: is the qs scale tensor physically adjacent to its merged_weight tensor?
 * Read-only against the container; no model serving.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <stdint.h>

#ifndef POSIX_FADV_DONTNEED
#define POSIX_FADV_DONTNEED 4
#endif
#ifndef POSIX_FADV_WILLNEED
#define POSIX_FADV_WILLNEED 3
#endif

typedef struct { char name[256]; long long off, nbytes; int fd; } TEN;
static TEN tens[200000]; static int nt;
static double now_ms(void){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec*1e3+ts.tv_nsec/1e6; }
static int cmpd(const void*a,const void*b){double x=*(const double*)a,y=*(const double*)b;return(x>y)-(x<y);}
static double med(double*v,int n){qsort(v,n,sizeof(double),cmpd);return v[n/2];}
static void rd(int fd,long long off,long long n,uint8_t*buf){long long d=0;while(d<n){ssize_t r=pread(fd,buf+d,n-d,off+d);if(r<=0)return;d+=r;}}

int main(int argc,char**argv){
    if(argc<2){fprintf(stderr,"usage: %s snapdir\n",argv[0]);return 2;}
    DIR*d=opendir(argv[1]); if(!d){perror(argv[1]);return 1;}
    struct dirent*de;
    while((de=readdir(d))){
        size_t ln=strlen(de->d_name);
        if(ln<12||strncmp(de->d_name+ln-12,".safetensors",12))continue;
        char p[1024];snprintf(p,sizeof p,"%s/%s",argv[1],de->d_name);
        int fd=open(p,O_RDONLY);if(fd<0)continue;
        uint8_t hb[8];if(read(fd,hb,8)!=8){close(fd);continue;}
        long long hl=0;for(int i=7;i>=0;i--)hl=(hl<<8)|hb[i];
        if(hl<=0||hl>(1<<26)){close(fd);continue;}
        char*hj=malloc(hl+1);long long got=0;
        while(got<hl){ssize_t r=read(fd,hj+got,hl-got);if(r<=0)break;got+=r;}
        hj[got]=0;
        char*q=hj;
        while((q=strstr(q,"data_offsets"))){
            char*ks=q;while(ks>hj&&*ks!='{')ks--;
            char*qe=ks-2;
            if(qe>hj&&*qe=='\"'){
                char*qs=qe-1;while(qs>hj&&*qs!='\"')qs--;qs++;
                size_t nlen=qe-qs;
                long long a=-1,b=-1;
                if(sscanf(q,"data_offsets\":[%lld,%lld]",&a,&b)==2&&a>=0&&b>a&&nlen<sizeof(((TEN*)0)->name)){
                    TEN*t=&tens[nt];memcpy(t->name,qs,nlen);t->name[nlen]=0;t->off=a;t->nbytes=b-a;t->fd=fd;nt++;
                }
            }
            q+=13;
        }
        free(hj);
        if(nt>=200000)break;
    }
    closedir(d);
    fprintf(stderr,"tensors=%d\n",nt);

    /* P2: adjacency of qs to its merged_weight */
    printf("== P2 WEIGHT/SCALE ADJACENCY ==\n");
    int checked=0,gapzero=0;
    for(int i=0;i<nt&&checked<16;i++){
        TEN*w=&tens[i];
        char*s=strstr(w->name,"mlp.experts.");if(!s)continue;
        if(!strstr(w->name,"merged_weight")||w->nbytes!=1179648)continue;
        char qn[300];snprintf(qn,sizeof qn,"%s",w->name);
        char*e=strstr(qn,"merged_weight");strcpy(e,"qs");
        for(int j=0;j<nt;j++){
            if(strcmp(tens[j].name,qn)||tens[j].fd!=w->fd)continue;
            long long gap=tens[j].off-(w->off+w->nbytes);
            printf("adj %s weight_end=%lld qs_off=%lld gap=%lldB\n",qn,w->off+w->nbytes,tens[j].off,gap);
            checked++;if(gap==0)gapzero++;
            break;
        }
    }

    /* timing target: an INT3 expert */
    TEN*tw=NULL;
    for(int i=0;i<nt;i++){
        if(strstr(tens[i].name,"mlp.experts.")&&strstr(tens[i].name,"merged_weight")&&tens[i].nbytes==1179648){tw=&tens[i];break;}
    }
    if(!tw){printf("no INT3 expert found\n");return 3;}
    printf("== P1 READ TRANSFORM target=%s bytes=%lld ==\n",tw->name,tw->nbytes);
    enum{N=24};
    uint8_t*buf=malloc(tw->nbytes);
    static double tA[N],tB[N],tC[N],tD[N];
    for(int r=0;r<N;r++){ /* A cold */
        posix_fadvise(tw->fd,tw->off,tw->nbytes,POSIX_FADV_DONTNEED);
        double t0=now_ms();rd(tw->fd,tw->off,tw->nbytes,buf);tA[r]=now_ms()-t0;
    }
    for(int r=0;r<N;r++){ /* B willneed + 2ms */
        posix_fadvise(tw->fd,tw->off,tw->nbytes,POSIX_FADV_DONTNEED);
        posix_fadvise(tw->fd,tw->off,tw->nbytes,POSIX_FADV_WILLNEED);
        usleep(2000);
        double t0=now_ms();rd(tw->fd,tw->off,tw->nbytes,buf);tB[r]=now_ms()-t0;
    }
    for(int r=0;r<N;r++){ /* C willneed immediate */
        posix_fadvise(tw->fd,tw->off,tw->nbytes,POSIX_FADV_DONTNEED);
        posix_fadvise(tw->fd,tw->off,tw->nbytes,POSIX_FADV_WILLNEED);
        double t0=now_ms();rd(tw->fd,tw->off,tw->nbytes,buf);tC[r]=now_ms()-t0;
    }
    /* neighbor for collision arm */
    TEN*nb=NULL;
    for(int i=0;i<nt;i++){
        if(&tens[i]!=tw&&tens[i].fd==tw->fd&&strstr(tens[i].name,"mlp.experts.")&&strstr(tens[i].name,"merged_weight")&&tens[i].nbytes==1179648){nb=&tens[i];break;}
    }
    for(int r=0;r<N;r++){ /* D willneed then one intervening demand read */
        posix_fadvise(tw->fd,tw->off,tw->nbytes,POSIX_FADV_DONTNEED);
        if(nb)posix_fadvise(nb->fd,nb->off,nb->nbytes,POSIX_FADV_DONTNEED);
        posix_fadvise(tw->fd,tw->off,tw->nbytes,POSIX_FADV_WILLNEED);
        if(nb)rd(nb->fd,nb->off,nb->nbytes,buf); /* demand-class read */
        double t0=now_ms();rd(tw->fd,tw->off,tw->nbytes,buf);tD[r]=now_ms()-t0;
    }
    printf("cold_pread      med=%.3fms\n",med(tA,N));
    printf("willneed+2ms    med=%.3fms\n",med(tB,N));
    printf("willneed+0ms    med=%.3fms\n",med(tC,N));
    printf("willneed+collide med=%.3fms\n",med(tD,N));
    free(buf);
    return 0;
}
