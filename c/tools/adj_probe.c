/* adj_probe.c — do cold-miss candidates cluster by expert-id adjacency?
 * Input: lines "tok layer eid" from the v4 demand stream.
 * Rare experts (low total demand count) proxy misses.
 * For consecutive tokens within a layer, measure min|deid| from previous
 * token's demanded set; histogram + uniform-random baseline comparison.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAXTOK 4096
#define MAXL 64
#define MAXE 256

static int cnt[MAXL][MAXE];              /* demand count per (layer,eid) */
static int cur[MAXL][MAXE], prv[MAXL][MAXE]; /* presence bitmaps per token pair */
static int tok_seen[MAXTOK];
static long long hist_all[10], hist_rare[10];
static long long pairs_all, pairs_rare;

static int bucket(int d){ int b=0; while(d>0){ d>>=1; b++; } return b<9?b:9; }

int main(int argc,char**argv){
    if(argc<2){fprintf(stderr,"usage:%s dem.txt\n",argv[0]);return 2;}
    FILE*f=fopen(argv[1],"r"); if(!f)return 1;
    long long t,l,e,mx=0;
    while(fscanf(f,"%lld %lld %lld",&t,&l,&e)==3){
        if(t>=MAXTOK||l>=MAXL||e>=MAXE)continue;
        cnt[l][e]++; if(t>mx)mx=t;
    }
    fclose(f);
    /* pass2: rebuild stream for consecutive-token comparisons */
    f=fopen(argv[1],"r");
    long long pt=-1,pl=-1;
    memset(prv,0,sizeof prv);
    long long curtok=-1;
    int have_prv=0;
    while(fscanf(f,"%lld %lld %lld",&t,&l,&e)==3){
        if(t>=MAXTOK||l>=MAXL||e>=MAXE)continue;
        if(t!=curtok){ memcpy(prv,cur,sizeof prv); memset(cur,0,sizeof cur); curtok=t; have_prv=(pt==t-1); pt=t; }
        cur[l][e]=1;
        /* flush per-layer comparison lazily: compare when layer seen and prev token had that layer */
        if(have_prv){
            for(int ee=0;ee<MAXE;ee++){
                if(!cur[l][ee])continue;
                int best=999;
                for(int d=0;d<MAXE;d++){ if(prv[l][d]){ int ad=d-ee; if(ad<0)ad=-ad; if(ad<best)best=ad; } }
                if(best==999)continue;
                int rare = cnt[l][ee]<=40; /* bottom tail proxies misses */
                hist_all[bucket(best)]++; pairs_all++;
                if(rare){ hist_rare[bucket(best)]++; pairs_rare++; }
            }
        }
    }
    fclose(f);
    printf("pairs all=%lld rare=%lld\n",pairs_all,pairs_rare);
    printf("bucket(maxdist) all%% rare%%  uniform_baseline%%\n");
    long long ca=0,cr=0;
    for(int b=0;b<10;b++)ca+=hist_all[b],cr+=hist_rare[b];
    double ub=0;
    for(int b=0;b<10;b++){
        int maxd=(1<<b)-1; if(b==9)maxd=MAXE;
        double p=1.0;
        for(int k=0;k<8;k++){ double single=(2*maxd+1.0)<256?(2*maxd+1.0)/256.0:1.0; p*= (1-single); }
        double base=100.0*(1.0-p);
        printf("d<=%3d  %6.2f  %6.2f  %6.2f\n",maxd,
               ca?100.0*hist_all[b]/ca:0, cr?100.0*hist_rare[b]/cr:0, base);
    }
    return 0;
}
