/* gomp_probe.c — test-only interposer counting GOMP_parallel team launches,
 * split by engine phase (same seam as l3_allocprobe). Reports at exit. */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stddef.h>
#include <unistd.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

static void (*r_gomp)(void (*)(void *), void *, void (*)(void *, void *), size_t, unsigned, unsigned);
static long counts[3];
static int tried;

static int phase(void){
    static int *pp;
    if(!tried){ tried=1; pp=(int *)dlsym(RTLD_DEFAULT,"g_l3_alloc_phase"); }
    int v=pp?*pp:0;
    return (v<0||v>2)?0:v;
}
__attribute__((destructor)) static void report(void){
    char line[160];
    int fd=STDOUT_FILENO;
    const char *p=getenv("GOMPPROBE_OUT");
    if(p){ int f=open(p,O_WRONLY|O_CREAT|O_TRUNC,0644); if(f>=0) fd=f; }
    int n=snprintf(line,sizeof line,"[L3-GOMPPROBE] team_launches INIT=%ld PREFILL=%ld DECODE=%ld\n",
                   counts[0],counts[1],counts[2]);
    if(n>0) (void)!write(fd,line,(size_t)n);
    if(fd!=STDOUT_FILENO) close(fd);
}
void GOMP_parallel(void (*fn)(void *), void *data, void (*cpy)(void*,void*), size_t sz, unsigned num, unsigned flags){
    if(!r_gomp){
        r_gomp=(void (*)(void (*)(void *), void *, void (*)(void *, void *), size_t, unsigned, unsigned))
            dlsym(RTLD_NEXT,"GOMP_parallel");
        if(!r_gomp){ const char *m="gompprobe: GOMP_parallel not found\n"; (void)!write(2,m,strlen(m)); abort(); }
    }
    counts[phase()]++;
    r_gomp(fn,data,cpy,sz,num,flags);
}
