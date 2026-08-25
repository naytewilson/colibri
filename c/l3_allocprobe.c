/* l3_allocprobe.c — test-only process-global allocator interposer for the
 * Ling Wave-2 closeout allocation proof.
 *
 * Build:   make -C c l3_allocprobe.so ling3_probe
 * Run:     LD_PRELOAD=c/l3_allocprobe.so [L3_ALLOCPROBE_OUT=path] \
 *          c/ling3_probe --model ... (exact production invocation otherwise)
 *
 * Interposes malloc/calloc/realloc/free/aligned_alloc/posix_memalign via
 * RTLD_NEXT and counts CALLS (+bytes for allocations) split by the engine
 * phase published in the exported int g_l3_alloc_phase of the -DL3_ALLOC_PROBE
 * binary (0=INIT, 1=PREFILL, 2=DECODE). This measures process-global truth,
 * INCLUDING libc/libgomp/tokenizer allocations the engine does not issue
 * directly — the complement of the in-engine falloc-family counters.
 *
 * Semantics recorded here so the receipt cannot drift:
 *   - free(NULL) is NOT counted (no deallocation happens).
 *   - realloc is counted as one realloc call (glibc realloc(p,0) frees, but
 *     the call site chose the realloc primitive).
 *   - dlsym bootstrap allocations land in a static arena and are uncounted.
 *   - counters are atomic; report uses write(2)/snprintf only (no stdio
 *     streams, no further heap traffic) from a destructor at exit.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <stdlib.h>
#include <stdio.h>

enum { EV_MALLOC, EV_CALLOC, EV_REALLOC, EV_FREE, EV_ALIGNED, EV_PMEM, EV_N };
static const char *ev_name[EV_N]={"malloc","calloc","realloc","free","aligned","pmem"};

static void *(*r_malloc)(size_t);
static void (*r_free)(void *);
static void *(*r_calloc)(size_t,size_t);
static void *(*r_realloc)(void *,size_t);
static void *(*r_aligned)(size_t,size_t);
static int (*r_pmem)(void **,size_t,size_t);

static char boot[1<<16];
static size_t boot_off;
static volatile int busy;              /* >0 inside bootstrap */

/* counts[phase][event]: calls; bytes_[phase][event]: allocation bytes */
static long counts[3][EV_N];
static unsigned long long bytes_[3][EV_N];

static int phase(void){
    static int *pp; static int tried;
    if(!tried){
        /* Mark tried FIRST and bail out while bootstrapping: dlsym itself may
         * allocate, and re-entering this path would recurse forever. */
        tried=1;
        if(busy) return 0;
        pp=(int *)dlsym(RTLD_DEFAULT,"g_l3_alloc_phase");
    }
    int v=pp?*pp:0;
    return (v<0||v>2)?0:v;
}
static void bump(int ev,long dc,unsigned long long db){
    int ph=phase();
    __atomic_fetch_add(&counts[ph][ev],dc,__ATOMIC_RELAXED);
    __atomic_fetch_add(&bytes_[ph][ev],db,__ATOMIC_RELAXED);
}
static void *boot_alloc(size_t n){
    n=(n+15UL)&~15UL;
    if(boot_off+n>sizeof boot) return NULL;
    void *p=boot+boot_off; boot_off+=n; return p;
}
static void ensure(void){
    if(r_malloc) return;
    __atomic_fetch_add(&busy,1,__ATOMIC_RELAXED);
    r_calloc =(void *(*)(size_t,size_t))dlsym(RTLD_NEXT,"calloc");
    r_malloc =(void *(*)(size_t))dlsym(RTLD_NEXT,"malloc");
    r_free   =(void (*)(void *))dlsym(RTLD_NEXT,"free");
    r_realloc=(void *(*)(void *,size_t))dlsym(RTLD_NEXT,"realloc");
    r_aligned=(void *(*)(size_t,size_t))dlsym(RTLD_NEXT,"aligned_alloc");
    r_pmem   =(int (*)(void **,size_t,size_t))dlsym(RTLD_NEXT,"posix_memalign");
    __atomic_fetch_sub(&busy,1,__ATOMIC_RELAXED);
}

void *malloc(size_t n){
    ensure(); void *p=r_malloc(n); if(!busy) bump(EV_MALLOC,1,(unsigned long long)n); return p;
}
void *calloc(size_t n,size_t s){
    size_t tot=n*s;
    if(!r_calloc){
        ensure();
        if(!r_calloc){ void *p=boot_alloc(tot); if(p&&tot) memset(p,0,tot); return p; }
    }
    void *p=r_calloc(n,s); if(!busy) bump(EV_CALLOC,1,(unsigned long long)tot); return p;
}
void *realloc(void *q,size_t n){
    ensure(); void *p=r_realloc(q,n); if(!busy) bump(EV_REALLOC,1,(unsigned long long)n); return p;
}
void free(void *q){
    if(!q) return;                            /* no-op by C standard, uncounted */
    ensure(); r_free(q); if(!busy) bump(EV_FREE,1,0);
}
void *aligned_alloc(size_t a,size_t n){
    ensure(); void *p=r_aligned(a,n); if(!busy) bump(EV_ALIGNED,1,(unsigned long long)n); return p;
}
int posix_memalign(void **out,size_t a,size_t n){
    ensure(); int rc=r_pmem(out,a,n); if(!busy) bump(EV_PMEM,1,rc?0:(unsigned long long)n); return rc;
}

__attribute__((destructor)) static void l3ap_report(void){
    int fd=STDOUT_FILENO;
    const char *path=getenv("L3_ALLOCPROBE_OUT");
    if(path){ int f=open(path,O_WRONLY|O_CREAT|O_TRUNC,0644); if(f>=0) fd=f; }
    static const char *ph_name[3]={"INIT   ","PREFILL","DECODE "};
    char line[192];
    for(int ph=0;ph<3;ph++){
        int off=snprintf(line,sizeof line,"[L3-ALLOCPROBE] %s",ph_name[ph]);
        for(int ev=0;ev<EV_N && off>0 && off<(int)sizeof line-1;ev++)
            off+=snprintf(line+off,sizeof line-off," %s=%ld/%lluMB",
                          ev_name[ev],counts[ph][ev],bytes_[ph][ev]>>20);
        if(off>0&&off<(int)sizeof line){ line[off++]='\n'; (void)!write(fd,line,(size_t)off); }
    }
    if(fd!=STDOUT_FILENO) close(fd);
}
