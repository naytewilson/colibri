/* Qwen3.6-35B-A3B inference engine in pure C, Phase 2: Gated Attention + Gated
 * DeltaNet (recurrent linear attention) + streaming MoE.
 *
 * The full model is a hybrid: 10 x (3 x Gated DeltaNet -> MoE, 1 x Gated
 * Attention -> MoE). Phase 1 implemented ONLY the 25% attention layers and
 * treated the DeltaNet layers as identity; Phase 2 implements BOTH:
 *   - Gated Attention (GQA, per-head q/k RMSNorm, partial RoPE, output gate).
 *   - Gated DeltaNet: causal depthwise conv1d + recurrent gated-delta-rule with a
 *     carried conv ring + state S[h]=[kdim,vdim], then per-head Gated RMSNorm.
 * Every layer (attention or DeltaNet) carries its own MoE/MLP block.
 *
 * DENSE (embed, attn/dn q/k/v/o & projections, q/k norms, RMSNorm, router gate,
 * shared expert, lm_head, final norm) resident in RAM (float32). Expert weights
 * read from disk on-demand via pread + posix_fadvise(DONTNEED), cached LRU
 * per-layer, with a PILOT prefetch thread -- the same mechanism that fits
 * GLM-5.2 in 15 GB.
 *
 * Env vars (inherited from olmoe.c): PILOT, HOT, WARMUP, WIDE, SMOOTH, CONF_LIMIT.
 * Plus: SNAP=<dir>, and argv: qwen36 <cache/layer> <ebits> [ref.json] [PPL=1].
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>
#include <pthread.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#if defined(__AVX2__)
#include <immintrin.h>
#endif

/* Hard context ceiling: the model's max_position_embeddings. Every buffer that
 * scales with position (KV cache, attention score row) is allocated from max_t,
 * so this is a policy limit, not a buffer limit -- but it is ONE limit, named
 * once. It used to be the literal 8192 in two unrelated places: the size of a
 * stack array in attention() and the default of Q36_MAXT in serve_one(). They
 * agreed by luck, and raising Q36_MAXT moved the guard without moving the
 * buffer, so a longer prompt overran the stack instead of being refused.
 * Context costs 40 KB/token in KV (10 attention layers, f32) -- 128k is 5.0 GiB
 * -- which is why Q36_MAXT still defaults far below this. */
#define QWEN36_ATTN_MAX_CTX 262144
#define QWEN36_DEFAULT_MAX_CTX 8192

/* Effective ceiling: Q36_MAXT if set and sane, the conservative default
 * otherwise; never above the hard limit. */
static int qwen36_max_ctx(void) {
    const char *e = getenv("Q36_MAXT");
    int v = (e && *e) ? atoi(e) : QWEN36_DEFAULT_MAX_CTX;
    if (v < 1) v = QWEN36_DEFAULT_MAX_CTX;
    return v > QWEN36_ATTN_MAX_CTX ? QWEN36_ATTN_MAX_CTX : v;
}
#if defined(__APPLE__) || defined(__linux__) || defined(__FreeBSD__)
#include <sys/resource.h>
#include <unistd.h>
#endif
#include "st.h"
#include "json.h"   /* tokenizer.json parsing (reuse minimal parser) */

#ifdef _WIN32
#include <windows.h>
#define sleep_ms(ms) Sleep(ms)
#else
#include <dlfcn.h>
#define sleep_ms(ms) usleep((ms) * 1000)
#endif

/* ---------- tokenizer (optional, for human-readable output) ---------- */
static char **g_tok = NULL;   /* id -> piece string (strdup'd) */
static int    g_tok_n = 0;

static int hexnib(char c){
    if (c>='0'&&c<='9') return c-'0';
    if (c>='a'&&c<='f') return c-'a'+10;
    if (c>='A'&&c<='F') return c-'A'+10;
    return 0;
}

/* ===== text -> ids : BPE encoder (mirrors HF/Qwen tokenizer.json) =====
 * Builds piece->id (reverse vocab) + pair->rank (merges) maps, plus the
 * GPT-2 byte-to-unicode mapping. Encode = special-token split + GPT-2 regex
 * pre-tokenize + per-piece ByteLevel map + BPE merges. */
typedef struct { char **keys; int *vals; int *used; int cap; } SMap;
static unsigned shash(const char *s){ unsigned h=2166136261u; while(*s){ h^=(unsigned char)*s++; h*=16777619u; } return h; }
static void smap_init(SMap *m,int cap){ m->cap=cap; m->keys=calloc(cap,sizeof(char*)); m->vals=malloc(cap*sizeof(int)); m->used=calloc(cap,sizeof(int)); }
static void smap_put(SMap *m,const char *k,int v){ if(!k)return; unsigned h=shash(k)&(m->cap-1); while(m->used[h]){ if(m->keys[h]&&strcmp(m->keys[h],k)==0){m->vals[h]=v;return;} h=(h+1)&(m->cap-1);} m->used[h]=1; m->keys[h]=(char*)k; m->vals[h]=v; }
static int smap_get(SMap *m,const char *k){ if(!m||!m->cap||!k)return -1; unsigned h=shash(k)&(m->cap-1); while(m->used[h]){ if(m->keys[h]&&strcmp(m->keys[h],k)==0)return m->vals[h]; h=(h+1)&(m->cap-1);} return -1; }

static SMap  g_rev;                 /* piece string -> id (encode) */
static SMap  g_merge;               /* "a\x1F b" pair -> rank (encode) */
static char  byte_sym_utf8[256][8]; /* byte -> UTF-8 of mapped codepoint */
static short g_unmap[512];          /* mapped codepoint -> original byte (-1 = unused) */
static int   g_nspecial = 0;
static char **g_sp_str = NULL; static int *g_sp_id = NULL; static int *g_sp_len = NULL;

static const char *jstr(jval *o,const char *k){ jval *v=json_get(o,k); return (v&&v->t==J_STR)?v->str:NULL; }
static double jnum(jval *o,const char *k){ jval *v=json_get(o,k); return (v&&v->t==J_NUM)?v->num:0; }

enum { U_W=0, U_L=1, U_M=2, U_N=3, U_P=4, U_O=5 };
static int uclass(unsigned cp){
    if (cp==0x20||cp==0x09||cp==0x0A||cp==0x0D||cp==0x0B||cp==0x0C) return U_W;
    if (cp==0x00A0||cp==0x2000||cp==0x2001||cp==0x2002||cp==0x2003||cp==0x2004||cp==0x2005||cp==0x2006||cp==0x2007||cp==0x2008||cp==0x2009||cp==0x200A||cp==0x2028||cp==0x2029||cp==0x202F||cp==0x205F||cp==0x3000||cp==0xFEFF) return U_W;
    if (cp>=0x30&&cp<=0x39) return U_N;
    if (cp>=0xFF10&&cp<=0xFF19) return U_N;
    if (cp>=0x0660&&cp<=0x0669) return U_N;
    if ((cp>=0x41&&cp<=0x5A)||(cp>=0x61&&cp<=0x7A)) return U_L;
    if (cp>=0x00C0&&cp<=0x024F) return U_L;
    if (cp>=0x0400&&cp<=0x04FF) return U_L;
    if (cp>=0x0600&&cp<=0x06FF) return U_L;
    if (cp>=0x1F00&&cp<=0x1FFF) return U_L;
    if (cp>=0x3040&&cp<=0x30FF) return U_L;
    if (cp>=0x3400&&cp<=0x4DBF) return U_L;
    if (cp>=0x4E00&&cp<=0x9FFF) return U_L;
    if (cp>=0xAC00&&cp<=0xD7A3) return U_L;
    if (cp>=0x300&&cp<=0x36F) return U_M;
    if (cp>=0x1AB0&&cp<=0x1AFF) return U_M;
    if (cp>=0x1DC0&&cp<=0x1DFF) return U_M;
    if (cp>=0x20D0&&cp<=0x20FF) return U_M;
    if (cp>=0xFE20&&cp<=0xFE2F) return U_M;
    if (cp>=0x21&&cp<=0x2F) return U_P;
    if (cp>=0x3A&&cp<=0x40) return U_P;
    if (cp>=0x5B&&cp<=0x60) return U_P;
    if (cp>=0x7B&&cp<=0x7E) return U_P;
    if (cp>=0x3000&&cp<=0x303F) return U_P;
    if (cp>=0xFF01&&cp<=0xFF0F) return U_P;
    if (cp>=0xFF1A&&cp<=0xFF20) return U_P;
    if (cp>=0xFF3B&&cp<=0xFF40) return U_P;
    if (cp>=0xFF5B&&cp<=0xFF65) return U_P;
    if (cp>=0x2010&&cp<=0x2027) return U_P;
    if (cp>=0x2030&&cp<=0x205E) return U_P;
    return U_O;
}
static int utf8_decode(const char *s,int i,int n,int *adv){
    unsigned char c=(unsigned char)s[i]; int cp,a;
    if(c<0x80){cp=c;a=1;}
    else if((c>>5)==6){cp=c&0x1F;a=2;}
    else if((c>>4)==14){cp=c&0x0F;a=3;}
    else if((c>>3)==30){cp=c&0x07;a=4;}
    else {cp=c;a=1;}
    for(int k=1;k<a;k++){ if(i+k<n && ((unsigned char)s[i+k]&0xC0)==0x80) cp=(cp<<6)|((unsigned char)s[i+k]&0x3F); }
    if(adv)*adv=a; return cp;
}
static int utf8_adv(const char *s,int i){ int a; utf8_decode(s,i,0x7fffffff,&a); return a; }

static void build_byte_sym(void){
    for(int i=0;i<512;i++) g_unmap[i]=-1;
    int bs[256]; for(int b=0;b<256;b++) bs[b]=0;
    for(int b=33;b<=126;b++) bs[b]=1;
    for(int b=161;b<=172;b++) bs[b]=1;
    for(int b=174;b<=255;b++) bs[b]=1;
    int cn=0;
    for(int b=0;b<256;b++){
        int cp = bs[b]?b:(256+cn); if(!bs[b]) cn++;
        int k=0; unsigned c=(unsigned)cp;
        if(c<0x80) byte_sym_utf8[b][k++]=(char)c;
        else if(c<0x800){ byte_sym_utf8[b][k++]=0xC0|(c>>6); byte_sym_utf8[b][k++]=0x80|(c&0x3F); }
        else { byte_sym_utf8[b][k++]=0xE0|(c>>12); byte_sym_utf8[b][k++]=0x80|((c>>6)&0x3F); byte_sym_utf8[b][k++]=0x80|(c&0x3F); }
        byte_sym_utf8[b][k]=0;
        g_unmap[cp]=(short)b;   /* reverse: mapped codepoint -> original byte */
    }
}
static void push_id(int **ids,int *n,int *cap,int v){ if(*n==*cap){*cap*=2; *ids=realloc(*ids,*cap*sizeof(int));} (*ids)[(*n)++]=v; }

static int try_special(const char *s,int i,int n,int *id_out){
    int best_len=0,best_id=-1;
    for(int k=0;k<g_nspecial;k++){
        int L=g_sp_len[k]; if(L<=0||i+L>n) continue;
        if(memcmp(s+i,g_sp_str[k],L)==0){ if(L>best_len){best_len=L;best_id=g_sp_id[k];} }
    }
    *id_out=best_id; return best_len;
}
/* Pre-tokenize splitter, mirrors the HF/Qwen regex alternation:
 *   (?i:'s|'t|'re|'ve|'m|'ll|'d) | [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+ | \p{N}
 *   | ?[^\s\p{L}\p{M}\p{N}]+[\r\n]* | \s*[\r\n]+ | \s+(?!\S) | \s+
 * Returns the byte index just past the piece starting at i. */
static int pretok_end(const char *s,int i,int n){
    if (s[i]=='\''){
        const char *cands[]={"ll","ve","re","s","t","m","d"}; int clen[]={2,2,2,1,1,1,1};
        int best=0;
        for(int c=0;c<7;c++){ int L=clen[c]; if(i+1+L>n) continue; int ok=1; for(int k=0;k<L;k++){ char a=(char)tolower((unsigned char)s[i+1+k]); if(a!=cands[c][k]){ok=0;break;} } if(ok&&L>best)best=L; }
        if(best>0) return i+1+best;
    }
    int adv; unsigned c0=utf8_decode(s,i,n,&adv);
    { /* rule2: optional non-(cr/lf/letter/number) prefix then letter/mark run */
        int k=i; unsigned c=c0; int prefix=0;
        if(k<n && c!='\r'&&c!='\n'&&uclass(c)!=U_L&&uclass(c)!=U_N){
            int a2; unsigned c1=utf8_decode(s,k+adv,n,&a2);
            if(uclass(c1)==U_L||uclass(c1)==U_M){ prefix=1; k+=adv; }
        }
        if(prefix || uclass(c)==U_L || uclass(c)==U_M){
            while(k<n){ int a; unsigned cc=utf8_decode(s,k,n,&a); if(uclass(cc)==U_L||uclass(cc)==U_M) k+=a; else break; }
            return k;
        }
    }
    if(uclass(c0)==U_N) return i+adv;
    { /* rule4: optional space + punctuation run (+ trailing newlines) */
        int k=i;
        if(s[i]==' '&&i+1<n){ int a1; unsigned c1=utf8_decode(s,i+1,n,&a1); if(uclass(c1)!=U_W&&uclass(c1)!=U_L&&uclass(c1)!=U_N&&c1!='\r'&&c1!='\n'){ k=i+1; while(k<n){int a;unsigned cc=utf8_decode(s,k,n,&a); if(uclass(cc)!=U_W&&uclass(cc)!=U_L&&uclass(cc)!=U_N&&cc!='\r'&&cc!='\n')k+=a; else break;} while(k<n&&(s[k]=='\r'||s[k]=='\n'))k++; return k; } }
        if(uclass(c0)!=U_W&&uclass(c0)!=U_L&&uclass(c0)!=U_N&&c0!='\r'&&c0!='\n'){ int k2=i; while(k2<n){int a;unsigned cc=utf8_decode(s,k2,n,&a); if(uclass(cc)!=U_W&&uclass(cc)!=U_L&&uclass(cc)!=U_N&&cc!='\r'&&cc!='\n')k2+=a; else break;} while(k2<n&&(s[k2]=='\r'||s[k2]=='\n'))k2++; return k2; }
    }
    if(uclass(c0)==U_W){ int k=i; while(k<n){int a;unsigned cc=utf8_decode(s,k,n,&a); if(uclass(cc)==U_W)k+=a; else break;} return k; }
    return i+adv;
}
static void bpe_piece(const char *piece,int len,int **ids,int *n,int *cap){
    if(len<=0) return;
    int sc=0,scap=16; char **syms=malloc(scap*sizeof(char*));
    for(int b=0;b<len;b++){
        const char *sym=byte_sym_utf8[(unsigned char)piece[b]];
        int sl=(int)strlen(sym); char *d=malloc(sl+1); memcpy(d,sym,sl); d[sl]=0;
        if(sc==scap){scap*=2; syms=realloc(syms,scap*sizeof(char*));} syms[sc++]=d;
    }
    while(sc>1){
        int best=-1,besti=-1;
        for(int k=0;k<sc-1;k++){
            const char *a=syms[k],*b=syms[k+1];
            size_t kl=(size_t)strlen(a)+1+(size_t)strlen(b)+1;
            char *key=malloc(kl); snprintf(key,kl,"%s\x1F%s",a,b);
            int r=smap_get(&g_merge,key); free(key);
            if(r>=0 && (best<0||r<best)){best=r;besti=k;}
        }
        if(besti<0) break;
        char *m=malloc(strlen(syms[besti])+strlen(syms[besti+1])+1);
        strcpy(m,syms[besti]); strcat(m,syms[besti+1]);
        free(syms[besti]); free(syms[besti+1]); syms[besti]=m;
        for(int k=besti+1;k<sc-1;k++) syms[k]=syms[k+1]; sc--;
    }
    for(int k=0;k<sc;k++){ int id=smap_get(&g_rev,syms[k]); if(id<0) id=0; push_id(ids,n,cap,id); free(syms[k]); }
    free(syms);
}
static void encode_text(const char *text,int **out_ids,int *out_n){
    int cap=1024,n=0; int *ids=malloc(cap*sizeof(int));
    int tlen=(int)strlen(text); int i=0;
    while(i<tlen){
        int sid; int L=try_special(text,i,tlen,&sid);
        if(L>0){ push_id(&ids,&n,&cap,sid); i+=L; continue; }
        int j=pretok_end(text,i,tlen); if(j<=i) j=i+utf8_adv(text,i);
        bpe_piece(text+i,j-i,&ids,&n,&cap);
        i=j;
    }
    *out_ids=ids; *out_n=n;
}

/* Load Qwen tokenizer.json and build an id->piece table. Only needs the
 * "model.vocab" map (piece string -> id); merges are irrelevant for decoding. */
static void load_tokenizer(const char *path){
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[tok] cannot open %s\n", path); return; }
    fseek(f,0,SEEK_END); long n = ftell(f); fseek(f,0,SEEK_SET);
    char *buf = malloc(n+1);
    if (fread(buf,1,(size_t)n,f) != (size_t)n) { /* ignore short read */ }
    buf[n] = 0; fclose(f);
    char *arena = NULL;
    jval *root = json_parse(buf, &arena);
    jval *model = json_get(root, "model"); if (!model) model = root;
    jval *vocab = json_get(model, "vocab");
    if (!vocab) vocab = json_get(model, "tokens");
    if (!vocab) { fprintf(stderr, "[tok] no model.vocab/tokens in %s\n", path); free(buf); return; }
    int mx = 0;
    if (vocab->t == J_OBJ){
        for (int i=0;i<vocab->len;i++){ int id=(int)vocab->kids[i]->num; if(id>mx)mx=id; }
    } else {
        mx = vocab->len - 1;
    }
    g_tok = calloc((size_t)mx+1, sizeof(char*));
    if (vocab->t == J_OBJ){
        for (int i=0;i<vocab->len;i++){ int id=(int)vocab->kids[i]->num; if(id>=0 && id<=mx) g_tok[id]=strdup(vocab->keys[i]); }
    } else {
        for (int i=0;i<vocab->len;i++){ if(vocab->kids[i] && vocab->kids[i]->t==J_STR) g_tok[i]=strdup(vocab->kids[i]->str); }
    }
    g_tok_n = mx+1;

    /* ---- encoder tables (text -> ids) ---- */
    smap_init(&g_rev, 1<<19);
    for (int i=0;i<g_tok_n;i++) if (g_tok[i]) smap_put(&g_rev, g_tok[i], i);

    smap_init(&g_merge, 1<<19);
    jval *merges = json_get(model, "merges");
    if (merges && merges->t==J_ARR){
        for (int r=0;r<merges->len;r++){
            const char *e = merges->kids[r]->str; if(!e) continue;
            const char *sp = strchr(e, ' '); if(!sp) continue;
            int la=(int)(sp-e), lb=(int)strlen(sp+1);
            char *key=malloc(la+1+lb+1);
            memcpy(key,e,la); key[la]=0x1F; memcpy(key+la+1,sp+1,lb); key[la+1+lb]=0;
            smap_put(&g_merge, key, r);
        }
    }
    jval *adds = json_get(root, "added_tokens");
    if (adds && adds->t==J_ARR && g_nspecial==0){
        g_nspecial = adds->len;
        g_sp_str = malloc(g_nspecial*sizeof(char*));
        g_sp_id   = malloc(g_nspecial*sizeof(int));
        g_sp_len  = malloc(g_nspecial*sizeof(int));
        for (int k=0;k<adds->len;k++){
            jval *t = adds->kids[k];
            const char *c = jstr(t,"content");
            g_sp_str[k] = c?strdup(c):strdup("");
            g_sp_id[k]  = (int)jnum(t,"id");
            g_sp_len[k] = (int)strlen(g_sp_str[k]);
        }
    }
    build_byte_sym();

    fprintf(stderr, "[tok] loaded %d pieces (max id %d) from %s\n", vocab->len, mx, path);
    free(buf);
}

/* Decode token ids to text using g_tok, writing to stdout. Handles Qwen's
 * byte-representation markers (Ġ=space, Ċ=newline, ▁=space) and <0xXX> byte
 * fallback. Only active when a tokenizer was loaded. */
/* ---- streaming / incremental decode support ---- */
static int    g_stream = 0;            /* 1 = emit tokens as they are generated */
static unsigned char g_sbuf[16];       /* carries a partial UTF-8 char across tokens */
static int    g_sbn = 0;

/* ---- OpenAI-compatible output + timing ---- */
static int    g_openai = 0;            /* 1 = emit OpenAI Chat Completions format (SSE/JSON) */
static double g_gen_t0 = 0;            /* generate() start (monotonic seconds) */
static double g_ttft   = -1;           /* time to first token (s); -1 = unset */
static long   g_oa_created = 0;        /* unix timestamp for OpenAI "created" */
static char   g_oa_id[64];             /* OpenAI-style id, e.g. chatcmpl-... */
static const char *g_model = "qwen3.6-35b-a3b-colibri";
static double now_s(void);   /* forward decl; defined later near model code */

/* Output sink for server mode: when g_sock_out >= 0, SSE/JSON bytes are routed
 * to the live socket via g_sock_send instead of stdout. Lets qwen36_serve.c
 * reuse all emit logic without any change to the CLI path. */
static long long g_sock_out = -1;
static void (*g_sock_send)(long long fd, const char *buf, int n) = NULL;

/* JSON-escape a byte string into out (no surrounding quotes). Returns length. */
static int json_escape(const unsigned char *s, int n, char *out, int outsz){
    int o = 0;
    for (int i=0;i<n;i++){
        unsigned char c = s[i];
        if (c == '"'){ if(o+2<outsz){ out[o++]='"'; out[o++]='"'; } }
        else if (c == '\\'){ if(o+2<outsz){ out[o++]='\\'; out[o++]='\\'; } }
        else if (c == '\n'){ if(o+2<outsz){ out[o++]='\\'; out[o++]='n'; } }
        else if (c == '\r'){ if(o+2<outsz){ out[o++]='\\'; out[o++]='r'; } }
        else if (c == '\t'){ if(o+2<outsz){ out[o++]='\\'; out[o++]='t'; } }
        else if (c == '\b'){ if(o+2<outsz){ out[o++]='\\'; out[o++]='b'; } }
        else if (c == '\f'){ if(o+2<outsz){ out[o++]='\\'; out[o++]='f'; } }
        else if (c < 0x20){ if(o+6<outsz){ sprintf(out+o, "\\u%04x", c); o+=6; } }
        else { if(o+1<outsz) out[o++] = (char)c; }
    }
    if (o < outsz) out[o] = 0;
    return o;
}

/* Append b[0..n) into buf/*bn, extract as many LEADING complete UTF-8
 * codepoints as possible into out[0..*outn) (max 255). Trailing partial
 * sequence stays in buf. Returns bytes written to out. */
static int utf8_drain(unsigned char *buf, int *bn, const unsigned char *b, int n, unsigned char *out, int *outn){
    *outn = 0;
    for (int k=0;k<n;k++){ if (*bn < 16) buf[(*bn)++] = b[k]; }
    int j = 0;
    while (j < *bn){
        unsigned char lead = buf[j]; int need;
        if (lead < 0x80) need = 1;
        else if ((lead & 0xE0) == 0xC0) need = 2;
        else if ((lead & 0xF0) == 0xE0) need = 3;
        else if ((lead & 0xF8) == 0xF0) need = 4;
        else { memmove(buf+j, buf+j+1, *bn-j-1); (*bn)--; continue; }
        if (j+need > *bn) break;
        if (*outn + need <= 255){ for (int x=0;x<need;x++) out[(*outn)++] = buf[j+x]; }
        memmove(buf+j, buf+j+need, *bn-j-need);
        *bn -= need;
    }
    return *outn;
}

/* Emit one Server-Sent-Event chunk (OpenAI streaming uses `data: <json>` lines). */
static void sse_chunk(const char *json){
    char hdr[8]; int hl = snprintf(hdr, sizeof hdr, "data: ");
    if (g_sock_out >= 0 && g_sock_send){
        g_sock_send(g_sock_out, hdr, hl);
        g_sock_send(g_sock_out, json, (int)strlen(json));
        g_sock_send(g_sock_out, "\n\n", 2);
    } else {
        fwrite(hdr, 1, (size_t)hl, stdout);
        fwrite(json, 1, (size_t)strlen(json), stdout);
        fwrite("\n\n", 1, 2, stdout);
        fflush(stdout);
    }
}

/* Decode a single token id into its raw (unmapped) bytes.
 * The vocab stores byte-level BPE pieces: each piece is UTF-8 of the
 * GPT-2 byte_to_unicode-mapped codepoints. We reverse that mapping so the
 * output is the original text bytes (correct for CJK / non-ASCII too).
 * <0xXX> byte-fallback tokens emit the raw byte directly. */
static void decode_id_to_bytes(int id, unsigned char *out, int *outn){
    *outn = 0;
    if (!g_tok || id<0 || id>=g_tok_n) return;
    const unsigned char *pc = (const unsigned char*)g_tok[id];
    /* byte-fallback token: <0xXX> -> raw byte */
    if (pc[0]=='<' && pc[1]=='0' && pc[2]=='x' && pc[5]=='>'){
        out[(*outn)++] = (unsigned char)(hexnib((char)pc[3])*16 + hexnib((char)pc[4]));
        return;
    }
    int i = 0;
    while (pc[i]){
        int cp, extra;
        if (pc[i] < 0x80){ cp = pc[i]; extra = 0; }
        else if ((pc[i] & 0xE0) == 0xC0){ cp = pc[i] & 0x1F; extra = 1; }
        else if ((pc[i] & 0xF0) == 0xE0){ cp = pc[i] & 0x0F; extra = 2; }
        else if ((pc[i] & 0xF8) == 0xF0){ cp = pc[i] & 0x07; extra = 3; }
        else { i++; continue; }                 /* stray lead byte, skip */
        int ok = 1;
        for (int e=0; e<extra; e++){ if (!pc[i+1+e]){ ok=0; break; } cp = (cp<<6) | (pc[i+1+e] & 0x3F); }
        i += 1 + extra;
        if (!ok) continue;
        if (cp == 0x2581) out[(*outn)++] = ' ';             /* SentencePiece space marker (kept safe) */
        else if (cp < 512 && g_unmap[cp] >= 0) out[(*outn)++] = (unsigned char)g_unmap[cp]; /* reverse byte_to_unicode */
        else out[(*outn)++] = (unsigned char)cp;
        if (*outn >= 255) break;
    }
}

/* Decode a range of token ids into a NUL-terminated text buffer (non-streaming). */
static int decode_range(const int *arr, int from, int to, char *ob, int obsz){
    unsigned char sb[16]; int sbn = 0; int o = 0;
    for (int i=from;i<to;i++){
        unsigned char tmp[256]; int tn = 0; decode_id_to_bytes(arr[i], tmp, &tn);
        unsigned char chunk[256]; int cn = 0; utf8_drain(sb, &sbn, tmp, tn, chunk, &cn);
        for (int k=0;k<cn && o<obsz-1;k++) ob[o++] = (char)chunk[k];
    }
    for (int k=0;k<sbn && o<obsz-1;k++) ob[o++] = (char)sb[k];   /* flush any trailing partial */
    if (o < obsz) ob[o] = 0;
    return o;
}

/* Append bytes to a buffer and flush any complete UTF-8 codepoints; any
 * trailing partial sequence is left in the buffer for the next call. */
static void out_bytes(unsigned char *buf, int *bn, const unsigned char *b, int n){
    for (int k=0; k<n; k++){
        if (*bn < 16) buf[(*bn)++] = b[k];
        int j = 0;
        while (j < *bn){
            unsigned char lead = buf[j]; int need;
            if (lead < 0x80) need = 1;
            else if ((lead & 0xE0) == 0xC0) need = 2;
            else if ((lead & 0xF0) == 0xE0) need = 3;
            else if ((lead & 0xF8) == 0xF0) need = 4;
            else { putchar(buf[j]); memmove(buf+j, buf+j+1, *bn-j-1); (*bn)--; continue; }
            if (j+need > *bn) break;
            fwrite(buf+j, 1, (size_t)need, stdout);
            memmove(buf+j, buf+j+need, *bn-j-need);
            *bn -= need;
        }
    }
}

static void print_decoded(const int *arr, int from, int to){
    unsigned char buf[16]; int bn = 0;
    for (int i=from;i<to;i++){
        unsigned char tmp[256]; int tn = 0;
        decode_id_to_bytes(arr[i], tmp, &tn);
        out_bytes(buf, &bn, tmp, tn);
    }
    if (bn) fwrite(buf, 1, (size_t)bn, stdout);
}

/* Streaming variants: emit one token at a time. In OpenAI mode each token is
 * one SSE `chat.completion.chunk` (delta.content = decoded text for this token,
 * carrying partial UTF-8 across tokens so CJK never splits mid-codepoint).
 * Otherwise emit raw readable text, flushing complete UTF-8 codepoints. */
static void stream_token(int id){
    if (g_openai){
        if (g_ttft < 0) g_ttft = now_s() - g_gen_t0;   /* TTFT on first token */
        if (!g_tok){
            char jb[160];
            snprintf(jb, sizeof jb,
              "{\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%ld,\"model\":\"%s\","
              "\"choices\":[{\"index\":0,\"delta\":{\"content\":\"%d\"},\"finish_reason\":null}]}",
              g_oa_id, g_oa_created, g_model, id);
            sse_chunk(jb); return;
        }
        unsigned char tmp[256]; int tn = 0;
        decode_id_to_bytes(id, tmp, &tn);
        unsigned char chunk[256]; int cn = 0;
        utf8_drain(g_sbuf, &g_sbn, tmp, tn, chunk, &cn);
        if (cn > 0){
            char esc[1024]; json_escape(chunk, cn, esc, sizeof esc);
            char jb[1200];
            snprintf(jb, sizeof jb,
              "{\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%ld,\"model\":\"%s\","
              "\"choices\":[{\"index\":0,\"delta\":{\"content\":\"%s\"},\"finish_reason\":null}]}",
              g_oa_id, g_oa_created, g_model, esc);
            sse_chunk(jb);
        }
        return;
    }
    /* default raw-text streaming */
    if (!g_tok){ printf("%d ", id); fflush(stdout); return; }
    unsigned char tmp[256]; int tn = 0;
    decode_id_to_bytes(id, tmp, &tn);
    out_bytes(g_sbuf, &g_sbn, tmp, tn);
    fflush(stdout);   /* make streaming visible immediately even when piped */
}
static void stream_flush(void){ if (g_sbn){ fwrite(g_sbuf, 1, (size_t)g_sbn, stdout); g_sbn = 0; } }

/* Emit the final OpenAI Chat Completions response for a finished generation.
 * Streaming: flushes any trailing partial UTF-8 as a last content chunk, then
 * sends the termination chunk (finish_reason + usage + timings) and "data: [DONE]".
 * Non-streaming: sends a single chat.completion JSON object.
 * When g_sock_out >= 0 the bytes go to the live socket; otherwise to stdout. */
static void emit_openai_result(const int *out, int np, int n_new, int stream){
    double total = now_s() - g_gen_t0;
    if (g_ttft < 0) g_ttft = total;   /* non-streaming: all tokens arrive at once */
    double gen_t = total - g_ttft;
    double tps = (gen_t > 1e-6 && n_new > 1) ? n_new / gen_t : (total > 0 ? n_new / total : 0.0);
    if (stream){
        if (g_sbn > 0){
            unsigned char chunk[16]; int cn = 0;
            for (int k=0;k<g_sbn;k++) chunk[cn++] = g_sbuf[k]; g_sbn = 0;
            if (cn > 0){
                char esc[256]; json_escape(chunk, cn, esc, sizeof esc);
                char jb[400];
                snprintf(jb, sizeof jb,
                  "{\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%ld,\"model\":\"%s\","
                  "\"choices\":[{\"index\":0,\"delta\":{\"content\":\"%s\"},\"finish_reason\":null}]}",
                  g_oa_id, g_oa_created, g_model, esc);
                sse_chunk(jb);
            }
        }
        char jb[700];
        snprintf(jb, sizeof jb,
          "{\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%ld,\"model\":\"%s\","
          "\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],"
          "\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d,\"total_tokens\":%d},"
          "\"timings\":{\"ttft_s\":%.3f,\"tokens_per_sec\":%.3f,\"total_s\":%.3f}}",
          g_oa_id, g_oa_created, g_model, np, n_new, np+n_new, g_ttft, tps, total);
        sse_chunk(jb);
        char done[16]; int dl = snprintf(done, sizeof done, "data: [DONE]\n\n");
        if (g_sock_out >= 0 && g_sock_send) g_sock_send(g_sock_out, done, dl);
        else { fwrite(done, 1, (size_t)dl, stdout); fflush(stdout); }
    } else {
        char text[1<<16]; decode_range(out, np, np+n_new, text, sizeof text);
        char esc[1<<16]; json_escape((const unsigned char*)text, (int)strlen(text), esc, sizeof esc);
        char buf[1<<20];
        int bl = snprintf(buf, sizeof buf,
          "{\"id\":\"%s\",\"object\":\"chat.completion\",\"created\":%ld,\"model\":\"%s\","
          "\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"%s\"},\"finish_reason\":\"stop\"}],"
          "\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d,\"total_tokens\":%d},"
          "\"timings\":{\"ttft_s\":%.3f,\"tokens_per_sec\":%.3f,\"total_s\":%.3f}}\n",
          g_oa_id, g_oa_created, g_model, esc, np, n_new, np+n_new, g_ttft, tps, total);
        if (g_sock_out >= 0 && g_sock_send) g_sock_send(g_sock_out, buf, bl);
        else { fwrite(buf, 1, (size_t)bl, stdout); fflush(stdout); }
    }
}

/* ---------- config ---------- */
typedef struct {
    int hidden, n_layers, n_active;
    int q_heads, kv_heads, head_dim;        /* k/v head dim == attention head dim */
    int q_head_dim;                         /* q per-head total = head_dim*2 when attn_output_gate */
    int k_head_dim, v_head_dim, o_in;       /* o_in = q_heads*head_dim (o_proj input) */
    int rope_dim, rotary_dim;               /* rotary_dim = actual rotated dims (head_dim*partial_rotary_factor) */
    int n_experts, topk, inter, shared_inter, vocab;
    int n_group, topk_group;
    float theta, eps, partial_rotary_factor;
    int norm_topk, has_qk_norm, has_bias, attn_output_gate;
    uint8_t *is_attn;   /* [n_layers] 1 if Gated Attention layer, 0 if DeltaNet */
    /* Gated DeltaNet (linear_attention) dims, read from qwen36_meta.json. */
    int dn_vheads, dn_kheads, dn_kdim, dn_vdim, dn_convk, dn_conv_dim;
    int expert_gs;      /* expert scale group size along input dim; 0 = per-row */
} Cfg;

/* ---------- per-layer dense weights ---------- */
typedef struct {
    float *in_ln, *post_ln, *q, *k, *v, *o, *qn, *kn, *gate, *gate_bias;
    float *sh_g, *sh_u, *sh_d, *sh_gate;   /* shared expert (dense f32) + shared_expert_gate */
    /* Gated DeltaNet (linear_attention) dense weights (f16->f32 via st_read_f32). */
    float *dn_qkv, *dn_z, *dn_b, *dn_a;    /* in_proj_qkv/z/b/a */
    float *dn_conv;                        /* conv1d.weight [conv_dim, convk] (groups=conv_dim) */
    float *dn_dtbias, *dn_alog;            /* dt_bias[vh], A_log[vh] */
    float *dn_norm;                        /* RMSNormGated weight [vdim] */
    float *dn_out;                         /* out_proj [hidden, value_dim] */
} Layer;

/* ---------- LRU expert cache (int8 / packed int4 / fmt=5 int3 weights + scales) ---------- */
typedef struct {
    int eid;
    int pinned;
    int is_int4;
    int is_int3;
    size_t allocated_weight_bytes;
    uint8_t *w4;
    uint8_t *w3;
    int8_t *g, *u, *d;
    uint8_t *g4, *u4, *d4;
    uint8_t *g3, *u3, *d3;
    float *gs, *us, *ds;
    uint64_t used;
} Slot;
typedef struct { Slot *slots; int n, cap; int16_t *loading; } LCache;
/* loading[eid] = slot index of the ONE active admission for that expert, -1 none.
 * Set under g_pilot_mx at reservation (s->eid=-1), cleared under g_pilot_mx at
 * publish. Makes in-flight loads visible to both admission paths so a second
 * loader for the same (layer,eid) can coalesce instead of duplicating. */

typedef struct {
    Cfg c;
    shards S;
    int quant_bits;
    uint16_t *embed_f16;
    float *embed, *lm_head, *final_norm;
    Layer *L;
    LCache *cache;          /* [n_layers] */
    int *active_of;         /* [n_layers] original->active idx (Phase 2: identity for all layers) */
    float **DN_rec;         /* [n_layers] recurrent state S[h]=[kdim,vdim] for DeltaNet layers (NULL for attn) */
    float **DN_conv;        /* [n_layers] conv ring [conv_dim, convk-1] for DeltaNet layers (NULL for attn) */
    uint64_t clock, hits, miss;
    float **K, **V; int kv_len, max_t, kv_cap;
    float *attn_sc;            /* [attn_sc_thr * kv_cap] score rows, one per thread */
    int attn_sc_thr;
    double dense_load_s;
    uint32_t *freq;
    double *router_mass;
    int freq_token_count, hot_pinned, hot_n, warmup_tokens, token_count;
    float *momentum_logits;
    float pilot_smooth, pilot_conf_limit;
    uint8_t *is_pinned;
    uint8_t *is_queued;
    uint8_t *seen;             /* prefill-collected experts (COLIBRI_RESIDENT) */
    int resident_mode;         /* 0 off; 1 pin this-prompt experts (CPU no-evict -> GPU resident) */
    int resident_collecting;   /* prefill in progress, collecting routed experts */
    int first_step;            /* the first step() call is the prefill */
} Model;

static pthread_mutex_t g_pilot_mx = PTHREAD_MUTEX_INITIALIZER;
static struct { int l, e; } pilot_q[4096];
static volatile unsigned pilot_r = 0, pilot_w = 0;
static Model *pilot_m = NULL;
static int g_pilot = 0;
static int g_wide  = 1;

static void pilot_prefetch(Model *m, int lnext, const float *x, int S);
static void *pilot_worker(void *arg);
static void ensure_pilot_worker_started(Model *m);
static void slot_ensure_allocated(Model *m, Slot *s);

static void ensure_pilot_worker_started(Model *m) {
    if (!pilot_m) {
        pilot_m = m;
        pthread_t t;
        if (pthread_create(&t, NULL, pilot_worker, NULL) != 0) {
            fprintf(stderr, "Error: Failed to create pilot prefetch worker thread\n");
            exit(1);
        }
        pthread_detach(t);
    }
}

/* ---------- utility ---------- */
static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec*1e-9; }

/* Peak RSS (high-water mark) from OS getrusage */
#if defined(__APPLE__)
static double peak_rss_gb(void) { struct rusage r; getrusage(RUSAGE_SELF, &r); return r.ru_maxrss / (1024.0*1024.0*1024.0); }
static uint64_t peak_rss_bytes(void) { struct rusage r; getrusage(RUSAGE_SELF, &r); return (uint64_t)r.ru_maxrss; }
#else
static double peak_rss_gb(void) { struct rusage r; getrusage(RUSAGE_SELF, &r); return r.ru_maxrss / (1024.0*1024.0); }
static uint64_t peak_rss_bytes(void) { struct rusage r; getrusage(RUSAGE_SELF, &r); return (uint64_t)r.ru_maxrss * 1024ULL; }
#endif
static double rss_gb(void) { return peak_rss_gb(); } /* backward-compatible alias */

/* Authoritative Linux current resident memory (VmRSS) from /proc/self/status */
static double current_rss_gb(void) {
#if defined(__linux__)
    FILE *f = fopen("/proc/self/status", "r");
    if (f) {
        char line[256];
        long rss_kb = 0;
        while (fgets(line, sizeof(line), f)) {
            if (strncmp(line, "VmRSS:", 6) == 0) {
                sscanf(line + 6, "%ld", &rss_kb);
                fclose(f);
                return (double)rss_kb / (1024.0 * 1024.0);
            }
        }
        fclose(f);
    }
#endif
    return peak_rss_gb();
}

static uint64_t current_rss_bytes(void) {
#if defined(__linux__)
    FILE *f = fopen("/proc/self/status", "r");
    if (f) {
        char line[256];
        long rss_kb = 0;
        while (fgets(line, sizeof(line), f)) {
            if (strncmp(line, "VmRSS:", 6) == 0) {
                sscanf(line + 6, "%ld", &rss_kb);
                fclose(f);
                return (uint64_t)rss_kb * 1024ULL;
            }
        }
        fclose(f);
    }
#endif
    return peak_rss_bytes();
}

static void mem_checkpoint(const char *chk, const char *desc) {
    double cur_g = current_rss_gb();
    double pk_g  = peak_rss_gb();
    uint64_t cur_b = current_rss_bytes();
    uint64_t pk_b  = peak_rss_bytes();
    fprintf(stderr, "[MEM_CHECKPOINT %s] %s | Current VmRSS: %.4f GiB (%llu B) | Peak RSS: %.4f GiB (%llu B)\n",
            chk, desc, cur_g, (unsigned long long)cur_b, pk_g, (unsigned long long)pk_b);
}

static uint64_t g_demand_loads = 0;
static uint64_t g_demand_bytes = 0;
static double g_demand_expert_admission_ms = 0.0;
static uint64_t g_pilot_loads = 0;
static uint64_t g_pilot_bytes = 0;
static double g_pilot_expert_admission_ms = 0.0;
/* admission decomposition (Phase 1): weight-read vs scale-read segments of the
 * same interval g_*_expert_admission_ms wraps; operation-boundary timers only */
static double g_demand_weight_ms = 0.0, g_demand_scale_ms = 0.0;
static double g_pilot_weight_ms  = 0.0, g_pilot_scale_ms  = 0.0;
/* expert_get internals (Phase 1): lock wait vs hit lookup vs victim/LRU scan.
 * Gated by tm_on(); two clock calls max per phase, never inside matmul loops. */
static double g_eg_lock_wait_ms = 0.0, g_eg_lookup_ms = 0.0, g_eg_victim_ms = 0.0;
/* decode-window accounting: hits/misses accumulate from process start (prefill
 * included), which previously made the final hit-rate line mix prefill demand
 * traffic into the decode window. Baselines are snapshotted right after the
 * prefill forward so reported deltas are decode-only. g_acq_* mirror m->hits/
 * m->miss at the single expert_get call site so tm_report (which has no Model
 * pointer) can compute the window delta and the acquisitions/token self-check. */
static uint64_t g_acq_hits = 0, g_acq_miss = 0;
/* duplicate-admission coalescing diagnostics (PILOT_DUPLICATE_RESIDENCY repair) */
static long g_demand_coalesce_waits = 0;   /* demand acquisitions that waited on an in-flight load */
static long g_pilot_coalesce_skips = 0;    /* pilot loads skipped because an admission was already active */
static uint64_t g_win_hits0 = 0, g_win_miss0 = 0, g_prefill_acq = 0;
static int g_rep_layers = 0, g_rep_topk = 0;
/* decode-window baselines for the admission timers too: g_demand_* accumulate
 * prefill+decode, so the slot-acquisition decomposition must subtract matched
 * window deltas, never the cumulative counter (req B). The serving corpus runs
 * MULTIPLE prompts, so at every prefill boundary the finished decode stretch is
 * folded into the running window sums and the baselines re-arm — later corpus
 * prefills (S>1 forwards) otherwise contaminate the admission delta while
 * contributing nothing to the decode-only sub[1]. */
static double g_win_adm0 = 0.0, g_win_wgt0 = 0.0, g_win_scl0 = 0.0;
static double g_win_slot0 = 0.0, g_win_lock0 = 0.0, g_win_lookup0 = 0.0, g_win_victim0 = 0.0;
static double g_win_adm_ms = 0.0, g_win_wgt_ms = 0.0, g_win_scl_ms = 0.0;
static double g_win_slot_ms = 0.0, g_win_lock_ms = 0.0, g_win_lookup_ms = 0.0, g_win_victim_ms = 0.0;
static uint64_t g_win_hit_ms = 0, g_win_miss_ms = 0;
static int g_window_armed = 0;        /* baselines valid (armed post-prefill) */
static int g_window_fold_armed = 0;   /* an unfolded decode stretch is pending */
/* result of ONE expert load, delivered to the caller that triggered it —
 * never via shared globals, so concurrent demand/pilot loads cannot
 * cross-annotate each other's trace rows */
typedef struct ExpertLoadResult {
    double ms;
    int64_t bytes;
    int fmt;   /* 3=INT3, 4=INT4, 8=INT8, 0=unknown */
} ExpertLoadResult;
/* offline cache-replay trace (Phase 5): COLI_MOE_TRACE=<path> enables one TSV
 * row per cache-mutating event. seq is assigned while holding g_pilot_mx — the
 * lock both demand and pilot mutation paths hold — so the stream is a total
 * order of cache mutations and deterministic replay input. slot identifies the
 * exact LCache slot touched, so replay mirrors the runtime array even across
 * the pilot-publish duplicate-residency race (two slots may briefly hold one
 * eid — the current policy permits it; the replay must model it, not hide it).
 * Row: seq class event tok layer eid fmt bytes adm_ms victim_eid slot
 *   class: DEMAND | PILOT      event: HIT | EVICT | INSERT
 *   tok=-1 marks prefill-window (DEMAND) or no-token-context (PILOT) rows */
static FILE *g_trace_fp = NULL;
static uint64_t g_trace_seq = 0;
static int64_t g_trace_tok = -1;
static int64_t g_trace_tok_base = 0;   /* running decode index across corpus prompts */
/* ---- v4 POLICY-INDEPENDENT REQUEST STREAM (COLI_TRACE_REQ=<path>) ----
 * Counterfactual replay INPUT: intents and static facts only. The simulator
 * owns residency, enqueue/drop, dequeue selection, hit/miss, admission,
 * publish and wait resolution. Baseline OUTCOMES (HIT/EVICT/INSERT, victims,
 * slots) live only in the separate v3 ORACLE stream.
 *   E  <layer> <eid> <fmt> <bytes>                     static expert metadata
 *   B  GEN <np> <n_new> <tok_base> <arm>               prompt boundary (mutex-ordered)
 *   R  DEMAND <tok> <layer> <eid> <router_mass>        demand acquisition intent
 *   C  PC <srctok> <layer> <eid> <score_rank> <enqueue_order> <conf> <fmt> <bytes>
 *                                                      pilot candidate INTENT,
 *                                                      emitted BEFORE the
 *                                                      baseline resident /
 *                                                      queued / ring checks
 * All events are generated under g_pilot_mx: the stream is a true total order
 * of admission-relevant events (B included). Dequeue timing, publish timing
 * and wake timing are deliberately ABSENT — the simulator freezes its own
 * exogenous service schedule (FROZEN_SCHEDULE semantics).
 * event is generated on a locked path. */
static FILE *g_req_fp = NULL;
static uint64_t g_req_seq = 0;
/* static expert metadata (fmt / total bytes per layer,eid) — container
 * constants used to enrich C intents; NOT cache outcomes. Bounded by the
 * model's own ceiling: layers are validated <= 128 at config parse. */
#define QWEN36_REQ_MAX_LAYERS 128
#define QWEN36_REQ_MAX_EXPERTS 512
static int g_emeta_fmt[QWEN36_REQ_MAX_LAYERS][QWEN36_REQ_MAX_EXPERTS];
static long long g_emeta_bytes[QWEN36_REQ_MAX_LAYERS][QWEN36_REQ_MAX_EXPERTS];
static int g_req_meta_ready = 0;
static void req_emit(const char *kind, const char *body) {
    if (!g_req_fp) return;
    unsigned long long seq = ++g_req_seq;
    fprintf(g_req_fp, "%llu %s %s\n", seq, kind, body);
}
static void req_emit_expert_meta(Model *m);   /* defined after st_* usage below */
static void trace_emit(const char *cls, const char *event, int64_t tok, int layer,
                       int eid, int fmt, int64_t bytes, double adm_ms, int64_t victim_eid,
                       int slot) {
    if (!g_trace_fp) return;
    unsigned long long seq = ++g_trace_seq;   /* callers hold g_pilot_mx: total mutation order */
    fprintf(g_trace_fp, "%llu\t%s\t%s\t%lld\t%d\t%d\t%d\t%lld\t%.3f\t%lld\t%d\n",
            seq, cls, event, (long long)tok, layer,
            eid, fmt, (long long)bytes, adm_ms, (long long)victim_eid, slot);
}
static pthread_mutex_t g_io_stats_mx = PTHREAD_MUTEX_INITIALIZER;

/* ---- M-PROF (R2): per-phase wall-clock accumulators, COLI_TIMERS=1 ---- */
static int g_timers = -1;
static double g_tm_dec[6], g_tm_pre[6];   /* 0=deltanet 1=attention 2=moe_total 3=shared 4=router 5=lm_head */
static long g_tm_dec_tokens = 0, g_tm_pre_tokens = 0;
static double tm_now(void){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec*1e3 + ts.tv_nsec/1e6; }
static int tm_on(void){ if(g_timers<0){ const char *e=getenv("COLI_TIMERS"); g_timers = (e && *e=='1'); } return g_timers; }
double g_dn_sub[12]; // DN: QKV_PROJ Z_PROJ B_PROJ A_PROJ CONV QK_NORM REC_DECAY REC_KV REC_OUTER_UPDATE REC_QS GATED_NORM OUT_PROJ
double g_tm_step=0;                           /* step() total (decode) */
static double g_tm_win_moe=0; static int g_tm_win_n=0;

/* --- MoE high-resolution profile timers (ms) & counters --- */
static double g_moe_sub[10]; /* 0=router, 1=lookup, 2=i3_gate, 3=i3_up, 4=i3_down, 5=i4_gate, 6=i4_up, 7=i4_down, 8=shared, 9=weighted_accum */
static uint64_t g_routed_int3_count = 0;
static uint64_t g_routed_int4_count = 0;
static uint64_t g_cache_hit_int3 = 0, g_cache_hit_int4 = 0;
static uint64_t g_cache_miss_int3 = 0, g_cache_miss_int4 = 0;
static uint64_t g_admitted_bytes_int3 = 0, g_admitted_bytes_int4 = 0;
static uint64_t g_expert_gemv_parallel_invocations = 0;
static int g_expert_parallel_mode = -1;

static int expert_parallel_on(void) {
    if (g_expert_parallel_mode < 0) {
        const char *ep = getenv("COLI_EXPERT_PARALLEL");
        g_expert_parallel_mode = (ep && atoi(ep) > 0) ? 1 : 0;
    }
    return g_expert_parallel_mode;
}

static void tm_add(int S, int idx, double ms){
    if(S==1){
        g_tm_dec[idx]+=ms;
        if(idx==2) g_tm_win_moe+=ms;
        if(idx==5 && ++g_tm_win_n==32){
            fprintf(stderr,"[timers] window: moe %.0f ms/token (last 32)\n", g_tm_win_moe/32.0);
            g_tm_win_moe=0; g_tm_win_n=0;
        }
    } else g_tm_pre[idx]+=ms;
}
/* ---- B2 (PKG traces-profiler-bench-wrappers): ontology-named emission ----
 * Inert unless COLI_FORGE_PROFILE=<path> / COLI_FORGE_TRACE=<path> is set.
 * Every "forge.*" key emitted here EXISTS in the canonical ontology
 * (tools/model-forge/Sources/ForgeCore/MetricOntology.swift). Kill-rule:
 * never invent local names. Threading has NO ontology metric, so thread
 * configuration is recorded as metadata only, never as a metric. */
#include <sys/stat.h>

static int g_forge_prof_st = -1;
static int g_forge_trc_st = -1;
static FILE *g_forge_trc_fp = NULL;

static const char *forge_profile_path(void){
    static const char *p = NULL;
    if (g_forge_prof_st < 0) {
        p = getenv("COLI_FORGE_PROFILE");
        g_forge_prof_st = (p && *p) ? 1 : 0;
        if (g_forge_prof_st == 0) p = NULL;
    }
    return (g_forge_prof_st == 1) ? p : NULL;
}

static FILE *forge_trace_fp(void){
    if (g_forge_trc_st < 0) {
        const char *e = getenv("COLI_FORGE_TRACE");
        g_forge_trc_st = (e && *e) ? 1 : 0;
        if (g_forge_trc_st == 1) g_forge_trc_fp = fopen(e, "w");
    }
    return g_forge_trc_fp;
}

/* ---- B2R2/R3 generation identity -----------------------------------------
 * One generation == one generate() invocation (one serving turn). Wall
 * boundary state is reset per generation so no wall interval can span
 * previous-decode -> next-request-prefill -> next-decode. Process-wide
 * profiling counters (g_tm_dec_tokens etc.) are untouched. */
static long g_forge_gen = 0;             /* generation id (per generate())  */
static long g_forge_gen_bidx = 0;        /* boundary index within generation*/
static double g_forge_prev_boundary = 0.0;
static double g_forge_t0_boundary = 0.0;

static void forge_begin_generation(void){
    g_forge_gen++;
    g_forge_prev_boundary = 0.0;
    g_forge_t0_boundary = 0.0;
    g_forge_gen_bidx = 0;
}

/* TOKEN-BOUNDARY event — metadata-class wall observation ONLY.
 * Carries NO metric field: wall_delta_ms is raw evidence, never an ontology
 * metric (canonical forge.runtime.step_ms_per_token is engine-internal and is
 * emitted exclusively on engine_step events below).
 * wall_delta_ms: actual interval to the previous boundary IN THE SAME
 * generation; null on the first boundary of every generation (never spans a
 * prefill gap, never fabricated). */
static void forge_trace_token_boundary(long idx){
    FILE *f = forge_trace_fp();
    if (!f) return;
    double now = tm_now();
    int have_prev = (g_forge_prev_boundary > 0.0);
    if (!have_prev) g_forge_t0_boundary = now;
    g_forge_gen_bidx++;
    fprintf(f, "{\"ev\":\"token_boundary\",\"i\":%ld,\"gen\":%ld,\"gi\":%ld,\"first_in_generation\":%s,\"wall_delta_ms\":",
            idx, g_forge_gen, g_forge_gen_bidx, have_prev ? "false" : "true");
    if (have_prev) fprintf(f, "%.4f", now - g_forge_prev_boundary); else fprintf(f, "null");
    fprintf(f, "}\n");
    g_forge_prev_boundary = now;
}

/* ENGINE STEP event — the canonical clock domain.
 * Called from the OUTER decode loop immediately after the completed
 * step(...) whose duration defines g_tm_step, so:
 *   i            == g_tm_dec_tokens == completed decode-step identity
 *   engine_step_ms == exact duration of THAT completed step()
 *   cumulative_engine_step_ms_per_token == g_tm_step / g_tm_dec_tokens,
 *       recomputable exactly as mean(engine_step_ms[1..i]).
 * forge.runtime.step_ms_per_token is attached HERE and nowhere else. */
static void forge_trace_engine_step(double started_at){
    double dt = tm_now() - started_at;
    g_tm_step += dt;
    long n = g_tm_dec_tokens;
    if (n <= 0 || dt <= 0.0) return;
    FILE *f = forge_trace_fp();
    if (!f) return;
    fprintf(f, "{\"ev\":\"engine_step\",\"i\":%ld,\"gen\":%ld,\"metric\":\"forge.runtime.step_ms_per_token\",\"engine_step_ms\":%.4f,\"cumulative_engine_step_ms_per_token\":%.4f}\n",
            n, g_forge_gen, dt, g_tm_step / (double)n);
}

static void forge_trace_finish(void){
    if (g_forge_trc_fp) { fclose(g_forge_trc_fp); g_forge_trc_fp = NULL; }
}

/* step-time breakdown JSON: MoE / mixer(DeltaNet) / LM head / attention /
 * admission / memory, every metric key ontology-named; device-of-record
 * stamped from stat(2) of the model directory. */
static void forge_profile_emit(void){
    const char *out = forge_profile_path();
    FILE *trc = forge_trace_fp();
    long n = g_tm_dec_tokens;
    if (!out || n <= 0) { if (trc) forge_trace_finish(); return; }
    double moe_total = g_tm_dec[2];
    double router    = g_moe_sub[0];
    double slotacq   = g_moe_sub[1];
    double compute_only = moe_total - router - slotacq; if (compute_only < 0) compute_only = 0;
    uint64_t hits   = g_cache_hit_int3 + g_cache_hit_int4;
    uint64_t misses = g_cache_miss_int3 + g_cache_miss_int4;
    double demand_hr = (hits + misses) > 0 ? (double)hits / (double)(hits + misses) : 0.0;
    double adm_ms = g_win_adm_ms > 0 ? g_win_adm_ms : g_demand_expert_admission_ms;
    if (adm_ms < 0) adm_ms = 0;
    uint64_t logical_bpt = (uint64_t)((double)g_routed_int3_count * 1376392.0 / (double)n
                                    + (double)g_routed_int4_count * 1769608.0 / (double)n);
    double step_ms  = g_tm_step / (double)n;
    struct stat st;
    unsigned long long devno = 0;
    const char *snapdir = getenv("SNAP");
    const char *devpath = (snapdir && *snapdir) ? snapdir : g_model;
    int have_dev = (stat(devpath, &st) == 0);
    if (have_dev) devno = (unsigned long long)st.st_dev;
    const char *omp = getenv("OMP_NUM_THREADS");
    FILE *pf = fopen(out, "w");
    if (!pf) { fprintf(stderr, "[forge] profile emit failed: %s\n", out); return; }
    fprintf(pf,
"{\n"
"  \"schema\": \"forge_profile_v1\",\n"
"  \"ontology_source\": \"tools/model-forge/Sources/ForgeCore/MetricOntology.swift\",\n"
"  \"device_of_record\": {\"path\": \"%s\", \"st_dev\": %llu},\n"
"  \"threading_metadata\": {\"omp_num_threads_env\": ", devpath, devno);
    if (omp) fprintf(pf, "\"%s\"", omp); else fprintf(pf, "null");
    fprintf(pf,
", \"note\": \"no threading metric exists in the ontology; config recorded as metadata only (B2 kill-rule)\"},\n"
"  \"decode_tokens\": %ld,\n"
"  \"metrics\": {\n"
"    \"forge.runtime.step_ms_per_token\": %.4f,\n"
"    \"forge.mixer.deltanet_ms_per_token\": %.4f,\n"
"    \"forge.attention.ms_per_token\": %.4f,\n"
"    \"forge.moe.total_ms_per_token\": %.4f,\n"
"    \"forge.moe.routed_compute_ms_per_token\": %.4f,\n"
"    \"forge.moe.slot_lookup_ms_per_token\": %.4f,\n"
"    \"forge.moe.admission_ms_per_token\": %.4f,\n"
"    \"forge.moe.demand_hit_rate\": %.6f,\n"
"    \"forge.lm_head.ms_per_token\": %.4f,\n"
"    \"forge.memory.logical_weight_bytes_per_token\": %llu\n"
"  }\n"
"}\n",
        n,
        step_ms,
        g_tm_dec[0] / (double)n,
        g_tm_dec[1] / (double)n,
        moe_total / (double)n,
        compute_only / (double)n,
        slotacq / (double)n,
        adm_ms / (double)n,
        demand_hr,
        g_tm_dec[5] / (double)n,
        (unsigned long long)logical_bpt);
    fclose(pf);
    if (trc) {
        fprintf(trc, "{\"ev\":\"timing_summary\",\"profile\":\"%s\"}\n", out);
        forge_trace_finish();
    }
    fprintf(stderr, "[forge] profile written: %s (device_of_record st_dev=%llu)\n", out, devno);
}

static void tm_report(void){
    if(!tm_on()) return;
    /* fold the FINAL decode stretch (after the last prefill) into the window
     * sums so reported decode windows cover every decoded token */
    if (g_window_fold_armed) {
        g_win_adm_ms    += g_demand_expert_admission_ms - g_win_adm0;
        g_win_wgt_ms    += g_demand_weight_ms - g_win_wgt0;
        g_win_scl_ms    += g_demand_scale_ms - g_win_scl0;
        g_win_slot_ms   += g_moe_sub[1] - g_win_slot0;
        g_win_lock_ms   += g_eg_lock_wait_ms - g_win_lock0;
        g_win_lookup_ms += g_eg_lookup_ms - g_win_lookup0;
        g_win_victim_ms += g_eg_victim_ms - g_win_victim0;
        g_win_hit_ms    += g_acq_hits - g_win_hits0;
        g_win_miss_ms   += g_acq_miss - g_win_miss0;
        g_window_fold_armed = 0;
    }
    static const char *nm[6]={"deltanet","attention","moe_total","(shared)","(router)","lm_head"};
    fprintf(stderr,"[timers] decode: %ld tokens  (shared/router are subsets of moe_total)\n", g_tm_dec_tokens);
    double sum=0;
    for(int i=0;i<6;i++){
        fprintf(stderr,"[timers]   %-10s %9.1f ms  %8.2f ms/token\n",
                nm[i], g_tm_dec[i], g_tm_dec_tokens? g_tm_dec[i]/g_tm_dec_tokens:0.0);
        if(i!=3&&i!=4) sum+=g_tm_dec[i];
    }
    fprintf(stderr,"[timers]   %-10s %9.1f ms  %8.2f ms/token\n","TOTAL",sum,g_tm_dec_tokens?sum/g_tm_dec_tokens:0.0);
    if(g_tm_step>0)
        fprintf(stderr,"[timers]   step() total: %.1f ms/token (outside the phases: %.1f)\n",
            g_tm_step/g_tm_dec_tokens,
            (g_tm_step-(g_tm_dec[0]+g_tm_dec[1]+g_tm_dec[2]+g_tm_dec[5]))/g_tm_dec_tokens);
    double dn_tot = 0; for(int i=0;i<12;i++) dn_tot += g_dn_sub[i];
    if(dn_tot > 0)
        fprintf(stderr,"[timers]   dn-sub: QKV %.1f | Z %.1f | B %.1f | A %.1f | CONV %.1f | QK_N %.1f | REC[DEC %.1f KV %.1f OUT %.1f QS %.1f] | NORM %.1f | OUT %.1f ms/tok\n",
            g_dn_sub[0]/g_tm_dec_tokens,g_dn_sub[1]/g_tm_dec_tokens,g_dn_sub[2]/g_tm_dec_tokens,g_dn_sub[3]/g_tm_dec_tokens,
            g_dn_sub[4]/g_tm_dec_tokens,g_dn_sub[5]/g_tm_dec_tokens,g_dn_sub[6]/g_tm_dec_tokens,g_dn_sub[7]/g_tm_dec_tokens,
            g_dn_sub[8]/g_tm_dec_tokens,g_dn_sub[9]/g_tm_dec_tokens,g_dn_sub[10]/g_tm_dec_tokens,g_dn_sub[11]/g_tm_dec_tokens);

    if(g_moe_sub[0]+g_moe_sub[1]+g_moe_sub[2]+g_moe_sub[3]+g_moe_sub[4]+g_moe_sub[5]+g_moe_sub[6]+g_moe_sub[7]+g_moe_sub[8]+g_moe_sub[9]>0) {
        double routed_i3_ms = g_moe_sub[2] + g_moe_sub[3] + g_moe_sub[4];
        double routed_i4_ms = g_moe_sub[5] + g_moe_sub[6] + g_moe_sub[7];
        double compute_only_ms = routed_i3_ms + routed_i4_ms;
        double total_selections = (double)(g_routed_int3_count + g_routed_int4_count);
        double i4_pct = total_selections > 0 ? (100.0 * (double)g_routed_int4_count / total_selections) : 0.0;
        uint64_t logical_bytes_per_tok = (g_tm_dec_tokens > 0) ?
            (uint64_t)((double)g_routed_int3_count * 1376392.0 / g_tm_dec_tokens + (double)g_routed_int4_count * 1769608.0 / g_tm_dec_tokens) : 0;

        fprintf(stderr, "\n=== HIGH-RESOLUTION MOE BREAKDOWN (decode tokens: %ld) ===\n", g_tm_dec_tokens);
        fprintf(stderr, "  Total MoE phase:                 %8.2f ms/token (100.0%%)\n", g_tm_dec[2] / g_tm_dec_tokens);
        fprintf(stderr, "  Router (matmul + top-k):          %8.2f ms/token (%5.1f%%)\n", g_moe_sub[0] / g_tm_dec_tokens, 100.0 * g_moe_sub[0] / g_tm_dec[2]);
        fprintf(stderr, "  Expert slot acquisition:         %8.2f ms/token (%5.1f%%)\n", g_moe_sub[1] / g_tm_dec_tokens, 100.0 * g_moe_sub[1] / g_tm_dec[2]);
        fprintf(stderr, "  Routed expert compute (only):    %8.2f ms/token (%5.1f%%)\n", compute_only_ms / g_tm_dec_tokens, 100.0 * compute_only_ms / g_tm_dec[2]);
        if (expert_parallel_on())
            fprintf(stderr, "    [mode] EXPERT_PARALLEL=1: the routed region is timed as ONE fused interval in the\n"
                            "    INT3 gate bucket (INT3+INT4 combined); per-format gate/up/down fields below are 0 by construction.\n");
        else
            fprintf(stderr, "    [mode] REF_GEMV=1 (COLI_EXPERT_PARALLEL unset/0): per-format gate/up/down decomposition active.\n");
        fprintf(stderr, "    - INT3 routed (gate/up/down):  %8.2f ms/token (gate %.2f, up %.2f, down %.2f)\n",
                routed_i3_ms / g_tm_dec_tokens, g_moe_sub[2]/g_tm_dec_tokens, g_moe_sub[3]/g_tm_dec_tokens, g_moe_sub[4]/g_tm_dec_tokens);
        fprintf(stderr, "    - INT4 routed (gate/up/down):  %8.2f ms/token (gate %.2f, up %.2f, down %.2f)\n",
                routed_i4_ms / g_tm_dec_tokens, g_moe_sub[5]/g_tm_dec_tokens, g_moe_sub[6]/g_tm_dec_tokens, g_moe_sub[7]/g_tm_dec_tokens);
        fprintf(stderr, "  Shared expert (SwiGLU + gate):   %8.2f ms/token (%5.1f%%)\n", g_moe_sub[8] / g_tm_dec_tokens, 100.0 * g_moe_sub[8] / g_tm_dec[2]);
        fprintf(stderr, "  Weighted output accumulation:    %8.2f ms/token (%5.1f%%)\n", g_moe_sub[9] / g_tm_dec_tokens, 100.0 * g_moe_sub[9] / g_tm_dec[2]);
        fprintf(stderr, "  Admission (demand NVMe I/O):     %8.2f ms/token\n", g_demand_expert_admission_ms / g_tm_dec_tokens);
        /* Phase 1 decomposition of Expert slot acquisition (sub[1]):
         * sub[1] wraps lookup + victim/LRU + lock waits AND the synchronous
         * admission I/O of misses (loads run inline on this thread). The
         * machinery-only share is sub[1] minus demand admission I/O. */
        {
            /* req B: decode-window sums only — folded at every prefill boundary
             * in generate(); never cumulative prefill+decode counters. */
            double slot_ms = g_win_slot_ms;   /* decode-only by construction, window-accumulated */
            double adm_ms  = g_win_adm_ms;    if (adm_ms < 0) adm_ms = 0;
            double wgt_ms  = g_win_wgt_ms;    if (wgt_ms < 0) wgt_ms = 0;
            double scl_ms  = g_win_scl_ms;    if (scl_ms < 0) scl_ms = 0;
            double mach_ms = slot_ms - adm_ms; if (mach_ms < 0) mach_ms = 0;
            fprintf(stderr, "  --- Slot Acquisition Decomposition (Phase 1, decode window) ---\n");
            fprintf(stderr, "  Slot acquisition total:          %8.2f ms/token\n", slot_ms / g_tm_dec_tokens);
            fprintf(stderr, "    demand admission I/O inside:   %8.2f ms/token (weights %.2f + scales %.2f)\n",
                    adm_ms / g_tm_dec_tokens, wgt_ms / g_tm_dec_tokens, scl_ms / g_tm_dec_tokens);
            fprintf(stderr, "    cache machinery (excl I/O):    %8.2f ms/token\n", mach_ms / g_tm_dec_tokens);
            if (tm_on()) {
                fprintf(stderr, "      lock wait:      %7.2f ms/token | hit/miss scan: %7.2f ms/token | LRU/victim: %7.2f ms/token\n",
                        g_win_lock_ms / g_tm_dec_tokens, g_win_lookup_ms / g_tm_dec_tokens, g_win_victim_ms / g_tm_dec_tokens);
            }
        }
        fprintf(stderr, "  --- Routed Format Mix ---\n");
        fprintf(stderr, "  INT3 routed selections/token:    %8.2f\n", (double)g_routed_int3_count / g_tm_dec_tokens);
        fprintf(stderr, "  INT4 routed selections/token:    %8.2f\n", (double)g_routed_int4_count / g_tm_dec_tokens);
        fprintf(stderr, "  Actual routed INT4 percentage:   %8.2f%%\n", i4_pct);
        fprintf(stderr, "  Cache hits (INT3 / INT4):        %llu / %llu\n", (unsigned long long)g_cache_hit_int3, (unsigned long long)g_cache_hit_int4);
        fprintf(stderr, "  Cache misses (INT3 / INT4):      %llu / %llu\n", (unsigned long long)g_cache_miss_int3, (unsigned long long)g_cache_miss_int4);
        fprintf(stderr, "  Admitted bytes (INT3 / INT4):    %.2f MB / %.2f MB\n", (double)g_admitted_bytes_int3 / 1048576.0, (double)g_admitted_bytes_int4 / 1048576.0);
        fprintf(stderr, "  --- Traffic & Parallelism ---\n");
        fprintf(stderr, "  LOGICAL_WEIGHT_BYTES_PROCESSED:  %llu B/token (%.2f MB/token)\n", (unsigned long long)logical_bytes_per_tok, (double)logical_bytes_per_tok / 1048576.0);
        fprintf(stderr, "  Routed GEMV parallel invocations: %8.1f / token\n", (double)g_expert_gemv_parallel_invocations / g_tm_dec_tokens);
        fprintf(stderr, "===========================================================\n");
    }

    fprintf(stderr,"\n[expert_io] demand: %llu loads (%llu bytes, %.2f MB), cum_expert_admission: %.1f ms, avg_latency: %.2f ms/load\n",
            (unsigned long long)g_demand_loads, (unsigned long long)g_demand_bytes,
            (double)g_demand_bytes / 1048576.0, g_demand_expert_admission_ms,
            g_demand_loads ? g_demand_expert_admission_ms / g_demand_loads : 0.0);
    fprintf(stderr,"[expert_io] demand split (CUMULATIVE prefill+decode): weights %.1f ms | scales %.1f ms (sum %.1f of %.1f total)\n",
            g_demand_weight_ms, g_demand_scale_ms,
            g_demand_weight_ms + g_demand_scale_ms, g_demand_expert_admission_ms);
    fprintf(stderr,"[expert_io] pilot : %llu loads (%llu bytes, %.2f MB), cum_expert_admission: %.1f ms, avg_latency: %.2f ms/load\n",
            (unsigned long long)g_pilot_loads, (unsigned long long)g_pilot_bytes,
            (double)g_pilot_bytes / 1048576.0, g_pilot_expert_admission_ms,
            g_pilot_loads ? g_pilot_expert_admission_ms / g_pilot_loads : 0.0);
    /* decode-window acquisition self-check: hits/misses are demand-only by
     * construction (single expert_get call site), so window acquisitions/token
     * must equal layers*topk. Any deviation means the window is contaminated
     * (prefill/warmup forwards inside it) or a second call site appeared. */
    {
        uint64_t w_hit = g_win_hit_ms, w_mis = g_win_miss_ms;
        uint64_t w_tot = w_hit + w_mis;
        double per_tok = g_tm_dec_tokens > 0 ? (double)w_tot / (double)g_tm_dec_tokens : 0.0;
        double expect  = (double)g_rep_layers * (double)g_rep_topk;
        fprintf(stderr,"[expert_io] decode window: hit %llu miss %llu | %.1f acq/token (expect %.1f = layers*topk)%s\n",
                (unsigned long long)w_hit, (unsigned long long)w_mis, per_tok, expect,
                (expect > 0 && w_tot > 0 && (per_tok > expect * 1.01 || per_tok < expect * 0.99)) ? "  << WINDOW CONTAMINATION" : "");
        if (g_prefill_acq)
            fprintf(stderr,"[expert_io] prefill consumed %llu acquisitions before the window\n",
                    (unsigned long long)g_prefill_acq);
    }
    fprintf(stderr,"[expert_io] note: expert_admission wraps the whole expert admission interval (read packed weights + optional unpack + scale read), not pure NVMe syscall latency.\n");
    fprintf(stderr,"[timers] prefill: %ld tokens  dn=%.0f attn=%.0f moe=%.0f(sh=%.0f rt=%.0f) head=%.0f ms\n",
            g_tm_pre_tokens,g_tm_pre[0],g_tm_pre[1],g_tm_pre[2],g_tm_pre[3],g_tm_pre[4],g_tm_pre[5]);
}
static float *falloc(int64_t n) { float *p = malloc(n*sizeof(float)); if(!p){fprintf(stderr,"OOM %ld\n",(long)n);exit(1);} return p; }

/* y[S,O] = x[S,I] @ W^T,  W is [O,I] row-major */
static void matmul(float *y, const float *x, const float *W, int S, int I, int O) {
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        const float *w = W + (int64_t)o * I;
        for (int s = 0; s < S; s++) {
            const float *xs = x + (int64_t)s * I;
            float acc = 0.f;
            for (int i = 0; i < I; i++) acc += xs[i] * w[i];
            y[(int64_t)s * O + o] = acc;
        }
    }
}

/* y[1,O] = x[1,I] @ W^T with W quantized: q[O,I] int8 + scale per row. */
#if defined(__ARM_NEON)
#include <arm_neon.h>
static inline int32_t dot_i8_16(const int8_t *a, const int8_t *b) {
    int32x4_t acc = vdupq_n_s32(0);
    int8x16_t va = vld1q_s8(a), vb = vld1q_s8(b);
#if defined(__ARM_FEATURE_DOTPROD)
    acc = vdotq_s32(acc, va, vb);
#else
    acc = vpadalq_s16(acc, vmull_s8(vget_low_s8(va),  vget_low_s8(vb)));
    acc = vpadalq_s16(acc, vmull_s8(vget_high_s8(va), vget_high_s8(vb)));
#endif
    return vaddvq_s32(acc);
}
#endif
static void matmul_q(float *y, const float *x, const int8_t *q, const float *scale, int I, int O) {
#if defined(__ARM_NEON)
    /* IDOT is opt-in, not default-on: this path quantizes the ACTIVATIONS to
     * Q8_0 per 16-element block, which the scalar path does not, so the two are
     * not numerically equivalent. olmoe shipped it default-on and it cost
     * token-exactness end to end (#1044, fixed in af48fe8 by making it opt-in);
     * qwen36 inherited the same default from the same family of kernels. The
     * tiny-oracle gate would not have caught it -- that job runs on x86. */
    static int idot = -1;
    if (idot < 0) { const char *e = getenv("IDOT"); idot = (e && atoi(e)); }
    if (idot && I % 16 == 0 && I <= 4096) {
        int nb = I / 16; int8_t xi[4096]; float xs[256];
        for (int b = 0; b < nb; b++) {
            const float *xb = x + b*16;
            float am = 0.f; for (int i = 0; i < 16; i++) { float a = fabsf(xb[i]); if (a > am) am = a; }
            float s = am/127.f; if (s < 1e-12f) s = 1e-12f;
            xs[b] = s; float inv = 1.f/s;
            for (int i = 0; i < 16; i++) xi[b*16+i] = (int8_t)lrintf(xb[i]*inv);
        }
        #pragma omp parallel for schedule(static)
        for (int o = 0; o < O; o++) {
            const int8_t *w = q + (int64_t)o * I;
            float acc = 0.f;
            for (int b = 0; b < nb; b++) acc += xs[b]*(float)dot_i8_16(xi+b*16, w+b*16);
            y[o] = acc * scale[o];
        }
        return;
    }
#endif
#if defined(__AVX2__) && defined(__FMA__)
    /* Hand-vectorized int8->f32 GEMV (gcc does not auto-vectorize the
     * convert+accumulate chain). 32 weights per iteration, FMA accumulate. */
    #pragma omp parallel for schedule(static) if(O >= 256 && !omp_in_parallel())
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
        a0 = _mm256_add_ps(_mm256_add_ps(a0,a1), _mm256_add_ps(a2,a3));
        __m128 s = _mm_add_ps(_mm256_castps256_ps128(a0), _mm256_extractf128_ps(a0,1));
        s = _mm_add_ps(s, _mm_movehl_ps(s,s));
        s = _mm_add_ss(s, _mm_shuffle_ps(s,s,1));
        float acc = _mm_cvtss_f32(s);
        for (; i < I; i++) acc += x[i] * (float)w[i];
        y[o] = acc * scale[o];
    }
#else
    #pragma omp parallel for schedule(static) if(O >= 256 && !omp_in_parallel())
    for (int o = 0; o < O; o++) {
        const int8_t *w = q + (int64_t)o * I;
        float acc = 0.f;
        for (int i = 0; i < I; i++) acc += x[i] * (float)w[i];
        y[o] = acc * scale[o];
    }
#endif
}

/* Group-scaled int8 GEMV: one f32 scale per `gs` input elements per row
 * (gs64 expert containers). Row layout of `scale`: [O][I/gs] row-major. */
static int g_expert_gs = 0;   /* set from qwen36_meta.json (expert_gs) at load */
static void matmul_q_gs(float *y, const float *x, const int8_t *q, const float *scale,
                        int I, int O, int gs) {
    int ng = (I + gs - 1) / gs;
#if defined(__AVX2__) && defined(__FMA__)
    if ((gs & 31) == 0) {
        #pragma omp parallel for schedule(static) if(O >= 256 && !omp_in_parallel())
        for (int o = 0; o < O; o++) {
            const int8_t *w = q + (int64_t)o * I;
            const float *sc = scale + (int64_t)o * ng;
            float acc = 0.f;
            for (int gi = 0; gi < ng; gi++) {
                __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
                int base = gi * gs, end = base + gs; if (end > I) end = I;
                for (int i = base; i + 16 <= end; i += 16) {
                    __m128i b0 = _mm_loadu_si128((const __m128i*)(w + i));
                    a0 = _mm256_fmadd_ps(_mm256_loadu_ps(x+i),   _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b0)), a0);
                    a1 = _mm256_fmadd_ps(_mm256_loadu_ps(x+i+8), _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(b0,8))), a1);
                }
                a0 = _mm256_add_ps(a0, a1);
                __m128 s = _mm_add_ps(_mm256_castps256_ps128(a0), _mm256_extractf128_ps(a0,1));
                s = _mm_add_ps(s, _mm_movehl_ps(s,s));
                s = _mm_add_ss(s, _mm_shuffle_ps(s,s,1));
                acc += _mm_cvtss_f32(s) * sc[gi];
            }
            y[o] = acc;
        }
        return;
    }
#endif
    #pragma omp parallel for schedule(static) if(O >= 256 && !omp_in_parallel())
    for (int o = 0; o < O; o++) {
        const int8_t *w = q + (int64_t)o * I;
        const float *sc = scale + (int64_t)o * ng;
        float acc = 0.f;
        for (int gi = 0; gi < ng; gi++) {
            int base = gi * gs, end = base + gs; if (end > I) end = I;
            float part = 0.f;
            for (int i = base; i < end; i++) part += x[i] * (float)w[i];
            acc += part * sc[gi];
        }
        y[o] = acc;
    }
}

static void matmul_i4_row(float *y, const float *x, const uint8_t *q4, const float *scale, int I, int O) {
    #pragma omp parallel for schedule(static) if(O >= 256 && !omp_in_parallel())
    for (int o = 0; o < O; o++) {
        const uint8_t *w4 = q4 + (int64_t)o * (I / 2);
        float acc = 0.f;
        for (int i = 0; i < I; i++) {
            uint8_t byte = w4[i >> 1];
            int8_t v = (int8_t)((i & 1) ? ((byte >> 4) & 0xF) : (byte & 0xF));
            if (v & 8) v -= 16;
            acc += x[i] * (float)v;
        }
        y[o] = acc * scale[o];
    }
}

static void matmul_i4_gs_fast(float *y, const float *x, const uint8_t *q4, const float *scale,
                              int I, int O, int gs) {
    int ng = (I + gs - 1) / gs;
#if defined(__AVX2__) && defined(__FMA__)
    if ((gs & 31) == 0) {
        const __m128i m4 = _mm_set1_epi8(0x0F);
        const __m128i s8 = _mm_set1_epi8(0x08);

        #pragma omp parallel for schedule(static) if(O >= 256 && !omp_in_parallel())
        for (int o = 0; o < O; o++) {
            const uint8_t *w4 = q4 + (int64_t)o * (I / 2);
            const float *sc = scale + (int64_t)o * ng;
            float acc = 0.f;

            for (int gi = 0; gi < ng; gi++) {
                __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
                int base = gi * gs, end = base + gs; if (end > I) end = I;

                for (int i = base; i + 32 <= end; i += 32) {
                    __m128i raw16 = _mm_loadu_si128((const __m128i*)(w4 + (i >> 1)));
                    __m128i lo = _mm_and_si128(raw16, m4);
                    __m128i hi = _mm_and_si128(_mm_srli_epi16(raw16, 4), m4);
                    lo = _mm_sub_epi8(_mm_xor_si128(lo, s8), s8);
                    hi = _mm_sub_epi8(_mm_xor_si128(hi, s8), s8);

                    __m128i w0_15  = _mm_unpacklo_epi8(lo, hi);
                    __m128i w16_31 = _mm_unpackhi_epi8(lo, hi);

                    a0 = _mm256_fmadd_ps(_mm256_loadu_ps(x + i),      _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(w0_15)), a0);
                    a1 = _mm256_fmadd_ps(_mm256_loadu_ps(x + i + 8),  _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(w0_15, 8))), a1);
                    a0 = _mm256_fmadd_ps(_mm256_loadu_ps(x + i + 16), _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(w16_31)), a0);
                    a1 = _mm256_fmadd_ps(_mm256_loadu_ps(x + i + 24), _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(w16_31, 8))), a1);
                }

                a0 = _mm256_add_ps(a0, a1);
                __m128 s = _mm_add_ps(_mm256_castps256_ps128(a0), _mm256_extractf128_ps(a0,1));
                s = _mm_add_ps(s, _mm_movehl_ps(s,s));
                s = _mm_add_ss(s, _mm_shuffle_ps(s,s,1));
                acc += _mm_cvtss_f32(s) * sc[gi];
            }
            y[o] = acc;
        }
        return;
    }
#endif
    #pragma omp parallel for schedule(static) if(O >= 256 && !omp_in_parallel())
    for (int o = 0; o < O; o++) {
        const uint8_t *w4 = q4 + (int64_t)o * (I / 2);
        const float *sc = scale + (int64_t)o * ng;
        float acc = 0.f;
        for (int gi = 0; gi < ng; gi++) {
            int base = gi * gs, end = base + gs; if (end > I) end = I;
            float part = 0.f;
            for (int i = base; i < end; i++) {
                uint8_t byte = w4[i >> 1];
                int8_t v = (int8_t)((i & 1) ? ((byte >> 4) & 0xF) : (byte & 0xF));
                if (v & 8) v -= 16;
                part += x[i] * (float)v;
            }
            acc += part * sc[gi];
        }
        y[o] = acc;
    }
}

/* ---- fmt=5: int3-g64 (3-bit weights with ONE f32 scale per 64-input group) ----
 * Per group: 16B low plane + 8B high plane, values in [-4,3] stored v+4. 3.5 bits/weight. */
#define I3_GROUP 64
#define I3_GBYTES 24
static inline int64_t i3_groups(int I){ return ((int64_t)I + I3_GROUP - 1) / I3_GROUP; }
static inline int64_t i3_rowbytes(int I){ return i3_groups(I) * I3_GBYTES; }

#if defined(__AVX512F__) && defined(__AVX512BW__)
static int g_i3_avx512 = 1;
static inline float dot_i3g64_avx512(const uint8_t *lo, const uint8_t *hi, const float *x) {
    const __m128i m2 = _mm_set1_epi8(0x03);
    const __m512i c4 = _mm512_set1_epi8(4);
    __m128i b0 = _mm_loadu_si128((const __m128i*)lo);
    __m128i p0 = _mm_and_si128(b0, m2), p1 = _mm_and_si128(_mm_srli_epi16(b0, 2), m2);
    __m128i p2 = _mm_and_si128(_mm_srli_epi16(b0, 4), m2), p3 = _mm_and_si128(_mm_srli_epi16(b0, 6), m2);
    __m128i l01 = _mm_unpacklo_epi8(p0, p1), h01 = _mm_unpackhi_epi8(p0, p1);
    __m128i l23 = _mm_unpacklo_epi8(p2, p3), h23 = _mm_unpackhi_epi8(p2, p3);
    __m512i lov = _mm512_inserti32x4(_mm512_inserti32x4(_mm512_inserti32x4(
        _mm512_castsi128_si512(_mm_unpacklo_epi16(l01, l23)),
        _mm_unpackhi_epi16(l01, l23), 1),
        _mm_unpacklo_epi16(h01, h23), 2),
        _mm_unpackhi_epi16(h01, h23), 3);
    uint64_t hb; memcpy(&hb, hi, 8);
    __m512i wq = _mm512_sub_epi8(_mm512_mask_add_epi8(lov, (__mmask64)hb, lov, c4), c4);
    __m512 ac0 = _mm512_setzero_ps(), ac1 = _mm512_setzero_ps();
    ac0 = _mm512_fmadd_ps(_mm512_loadu_ps(x),    _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm512_castsi512_si128(wq))),      ac0);
    ac1 = _mm512_fmadd_ps(_mm512_loadu_ps(x+16), _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(wq, 1))), ac1);
    ac0 = _mm512_fmadd_ps(_mm512_loadu_ps(x+32), _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(wq, 2))), ac0);
    ac1 = _mm512_fmadd_ps(_mm512_loadu_ps(x+48), _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(wq, 3))), ac1);
    return _mm512_reduce_add_ps(_mm512_add_ps(ac0, ac1));
}

static int i3_avx512_selftest(void) {
    uint8_t lo[16] = {0}, hi[8] = {0}; float x[I3_GROUP]; double ref = 0;
    uint64_t r = 0x9E3779B97F4A7C15ull;
    for (int k = 0; k < I3_GROUP; k++) {
        r ^= r << 13; r ^= r >> 7; r ^= r << 17;
        unsigned u = (unsigned)(r & 7);
        lo[k >> 2] |= (uint8_t)((u & 3) << ((k & 3) * 2));
        hi[k >> 3] |= (uint8_t)((u >> 2) << (k & 7));
        x[k] = (k & 1) ? -(float)(k + 1) : (float)(k + 1);
        ref += (double)x[k] * ((int)u - 4);
    }
    float got = dot_i3g64_avx512(lo, hi, x);
    if (got != (float)ref) {
        fprintf(stderr, "AVX512 i3 selftest: %.9g != %.9g\n", got, ref);
        return 0;
    }
    return 1;
}
#endif

#if defined(__AVX2__)
static inline float dot_i3g64_avx2(const uint8_t *lo, const uint8_t *hi, const float *x) {
    const __m128i m2 = _mm_set1_epi8(0x03);
    const __m128i bsel = _mm_set_epi8(1,1,1,1,1,1,1,1, 0,0,0,0,0,0,0,0);
    const __m128i bitm = _mm_set_epi8((char)128,64,32,16,8,4,2,1,(char)128,64,32,16,8,4,2,1);
    const __m128i four8 = _mm_set1_epi8(4);
    const __m256i b4 = _mm256_set1_epi32(4);
    __m256 ac0 = _mm256_setzero_ps(), ac1 = _mm256_setzero_ps();
    for (int k = 0; k < I3_GROUP; k += 16) {
        __m128i by = _mm_cvtsi32_si128(*(const int*)(lo + (k >> 2)));
        __m128i p0 = _mm_and_si128(by, m2), p1 = _mm_and_si128(_mm_srli_epi16(by, 2), m2);
        __m128i p2 = _mm_and_si128(_mm_srli_epi16(by, 4), m2), p3 = _mm_and_si128(_mm_srli_epi16(by, 6), m2);
        __m128i l01 = _mm_unpacklo_epi8(p0, p1), h23 = _mm_unpacklo_epi8(p2, p3);
        __m128i lov = _mm_unpacklo_epi16(l01, h23);
        __m128i hv = _mm_shuffle_epi8(_mm_cvtsi32_si128(hi[k >> 3] | (hi[(k >> 3) + 1] << 8)), bsel);
        __m128i hb = _mm_and_si128(_mm_cmpeq_epi8(_mm_and_si128(hv, bitm), bitm), four8);
        __m128i u = _mm_add_epi8(lov, hb);
        __m256 w0 = _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(u), b4));
        __m256 w1 = _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_srli_si128(u, 8)), b4));
        ac0 = _mm256_fmadd_ps(_mm256_loadu_ps(x + k), w0, ac0);
        ac1 = _mm256_fmadd_ps(_mm256_loadu_ps(x + k + 8), w1, ac1);
    }
    __m256 a_sum = _mm256_add_ps(ac0, ac1);
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(a_sum), _mm256_extractf128_ps(a_sum, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    return _mm_cvtss_f32(s);
}
#endif

static void matmul_i3_gs_fast(float *y, const float *x, const uint8_t *q3, const float *scale, int I, int O, int gs) {
    (void)gs;
    int64_t ng = i3_groups(I), rb = i3_rowbytes(I);
    #pragma omp parallel for schedule(static) if(O >= 256 && !omp_in_parallel())
    for (int o = 0; o < O; o++) {
        const uint8_t *wrow = q3 + (int64_t)o * rb;
        const float *srow = scale + (int64_t)o * ng;
        float acc = 0.f;
        for (int64_t g = 0; g < ng; g++) {
            const uint8_t *lo = wrow + g * I3_GBYTES, *hi = lo + 16;
            int base = (int)(g * I3_GROUP);
            float a = 0.f;
#if defined(__AVX512F__) && defined(__AVX512BW__)
            if (g_i3_avx512) {
                a = dot_i3g64_avx512(lo, hi, x + base);
            } else
#endif
#if defined(__AVX2__)
            {
                a = dot_i3g64_avx2(lo, hi, x + base);
            }
#else
            {
                for (int k = 0; k < I3_GROUP; k++) {
                    unsigned u = ((lo[k >> 2] >> ((k & 3) * 2)) & 3) | (((hi[k >> 3] >> (k & 7)) & 1) << 2);
                    a += x[base + k] * (float)((int)u - 4);
                }
            }
#endif
            acc += a * srow[g];
        }
        y[o] = acc;
    }
}

static void matmul_i3_qe(float *y, const float *x, const uint8_t *q3, const float *scale, int I, int O) {
    matmul_i3_gs_fast(y, x, q3, scale, I, O, 64);
}

static void matmul_i4_qe(float *y, const float *x, const uint8_t *q4, const float *scale, int I, int O) {
    if (g_expert_gs) matmul_i4_gs_fast(y, x, q4, scale, I, O, g_expert_gs);
    else matmul_i4_row(y, x, q4, scale, I, O);
}

/* Expert-GEMV dispatch: per-row scales (classic) or grouped (gs64 container). */
static void matmul_qe(float *y, const float *x, const int8_t *q, const float *scale, int I, int O) {
    if (g_expert_gs) matmul_q_gs(y, x, q, scale, I, O, g_expert_gs);
    else matmul_q(y, x, q, scale, I, O);
}

/* ---- Dense int8: per-row quantized copies of the large f32 matrices.
 * matmul_d dispatches via pointer lookup to matmul_q; COLI_DENSE_I8=0 falls
 * back to f32 (reference path for parity tests). ~4x less memory traffic. */
#define QDW_MAX 1024
static struct { const float *w; int8_t *q; float *sc; int I, O; } g_qdw[QDW_MAX];
static int g_qdw_n = 0;
static int dense_i8_on(void){ static int v=-1; if(v<0){ const char *e=getenv("COLI_DENSE_I8"); v=!(e&&*e=='0'); } return v; }
static void qdw_register(const float *W, int I, int O){
    if (!W || !dense_i8_on() || g_qdw_n >= QDW_MAX) return;
    int8_t *q = malloc((size_t)O*I); float *sc = malloc((size_t)O*sizeof(float));
    if (!q || !sc) { free(q); free(sc); return; }
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        const float *r = W + (int64_t)o*I; float am = 0.f;
        for (int i = 0; i < I; i++) { float a = fabsf(r[i]); if (a > am) am = a; }
        float s = am > 1e-12f ? am/127.f : 1.f; sc[o] = s; float inv = 1.f/s;
        int8_t *d = q + (int64_t)o*I;
        for (int i = 0; i < I; i++) { int v = (int)lrintf(r[i]*inv); if (v>127) v=127; if (v<-127) v=-127; d[i] = (int8_t)v; }
    }
    g_qdw[g_qdw_n].w=W; g_qdw[g_qdw_n].q=q; g_qdw[g_qdw_n].sc=sc; g_qdw[g_qdw_n].I=I; g_qdw[g_qdw_n].O=O; g_qdw_n++;
}
static void matmul_d(float *y, const float *x, const float *W, int S, int I, int O){
    for (int i = 0; i < g_qdw_n; i++) if (g_qdw[i].w == W && g_qdw[i].I == I) {
        for (int s = 0; s < S; s++) matmul_q(y+(int64_t)s*O, x+(int64_t)s*I, g_qdw[i].q, g_qdw[i].sc, I, O);
        return;
    }
    matmul(y, x, W, S, I, O);
}

/* rmsnorm over a row of length D (in-place capable: out may == x).
 * Qwen3_5MoeRMSNorm: out = (x * rsqrt(mean(x^2)+eps)) * (1.0 + weight). */
static void rmsnorm_row(float *out, const float *x, const float *w, int D, float eps) {
    double ms = 0; for (int i = 0; i < D; i++) ms += (double)x[i]*x[i];
    float r = 1.f / sqrtf((float)(ms / D) + eps);
    for (int i = 0; i < D; i++) out[i] = x[i] * r * (1.0f + w[i]);
}

static void softmax_row(float *x, int n) {
    float m = -1e30f; for (int i = 0; i < n; i++) if (x[i] > m) m = x[i];
    float s = 0; for (int i = 0; i < n; i++) { x[i] = expf(x[i]-m); s += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= s;
}

/* softplus(z) = log(1+exp(z)), stable for large z (HF GatedDeltaNet g_rule). */
static float softplus_f(float z) { return z > 20.f ? z : log1pf(expf(z)); }

/* ---------- loading ---------- */
static double req_num(jval *r, const char *k){
    jval *v=json_get(r,k);
    if(!v||v->t!=J_NUM){ fprintf(stderr,"config.json: missing or non-numeric \"%s\"\n",k); exit(1); }
    return v->num;
}
static void load_cfg(Cfg *c, const char *snap) {
    char path[2048]; snprintf(path, sizeof(path), "%s/config.json", snap);
    FILE *f = fopen(path, "rb"); if(!f){perror(path);exit(1);}
    fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
    if(n<0 || n>(256L<<20)){ fprintf(stderr,"%s: config.json missing or larger than 256 MB\n",path); exit(1); }
    char *buf = malloc((size_t)n+1); if(!buf){ fprintf(stderr,"OOM reading %s\n",path); exit(1); }
    if(fread(buf,1,(size_t)n,f)!=(size_t)n){ fprintf(stderr,"%s: short read\n",path); exit(1); } buf[n]=0; fclose(f);
    char *arena=NULL; jval *r = json_parse(buf, &arena);
    c->hidden    = (int)req_num(r,"hidden_size");
    c->n_layers  = (int)req_num(r,"num_hidden_layers");
    c->vocab     = (int)req_num(r,"vocab_size");
    c->eps       = (float)req_num(r,"rms_norm_eps");
    jval *th = json_get(r,"rope_theta"); c->theta = th ? (float)th->num : 10000.f;
    free(buf); free(arena);
    /* Phase-1 defaults; overridden by qwen36_meta.json in load_meta.
     * Clamped so a missing meta file can never produce a divide-by-zero. */
    c->q_heads = (c->hidden >= 16) ? (c->hidden / 16) : 1;
    if (c->q_heads < 1) c->q_heads = 1;
    c->kv_heads = c->q_heads / 8; if (c->kv_heads < 1) c->kv_heads = 1;
    c->head_dim = c->hidden / c->q_heads; if (c->head_dim < 1) c->head_dim = 1;
    c->k_head_dim = c->head_dim; c->v_head_dim = c->head_dim;
    c->q_head_dim = c->head_dim * 2;        /* includes attn_output_gate */
    c->o_in = c->q_heads * c->head_dim;
    c->rotary_dim = (c->head_dim >= 4) ? (c->head_dim / 4) : 2;
    if (c->rotary_dim % 2 != 0) c->rotary_dim += 1;
    c->rope_dim = c->head_dim;
    c->partial_rotary_factor = 0.25f;
    c->n_experts = 256; c->topk = 8; c->inter = 512; c->shared_inter = 512;
    c->n_group = 1; c->topk_group = 1; c->norm_topk = 1; c->has_qk_norm = 1; c->has_bias = 0;
    c->attn_output_gate = 1; c->n_active = 0;
    c->is_attn = calloc(c->n_layers, sizeof(uint8_t));
    for (int i = 0; i < c->n_layers; i++) c->is_attn[i] = (i % 4 == 3) ? 1 : 0;
}

/* Read qwen36_meta.json (emitted FLAT by convert_qwen36.py) to override the
 * Phase-1 defaults with the real model dimensions. The converter derives the
 * head dims from the actual weight shapes, so these are authoritative. Falls
 * back silently to the i%4==3 pattern and defaults if the file is absent. */

/* Every dimension the forward pass indexes with, checked once against the
 * buffers that actually exist. Both config.json and qwen36_meta.json ship
 * INSIDE the container, so a mismatched or hostile pair is a supply-chain
 * input, not a programmer error -- and the repo just spent six advisories
 * removing this bug class (unvalidated config -> heap OOB). Pattern follows
 * kimi_k3.c: one guarded expression per dimension, exit with a clear message.
 *
 * The fixed-size buffers below are the reason the ceilings are what they are;
 * raising one means raising the buffer with it:
 *   moe()      uint8_t keep[1024]        -> n_experts <= 1024
 *   moe()      int idx[256], val[256]    -> topk      <= 256
 *   deltanet() float kvl[512], dl[512]   -> dn_vdim   <= 512   (OpenMP region)
 */
#define CFG_NEED(cond, ...) do { if (!(cond)) {         fprintf(stderr, "[cfg] "); fprintf(stderr, __VA_ARGS__);         fprintf(stderr, " -- refusing\n"); exit(1); } } while (0)

static void validate_cfg(const Cfg *c, int n_layers_from_config) {
    CFG_NEED(c->n_layers > 0 && c->n_layers <= 512,
             "n_layers %d out of range 1..512", c->n_layers);
    /* A layer count that disagrees between the two files is a broken container:
     * is_attn was sized from config.json before meta could override n_layers. */
    CFG_NEED(c->n_layers == n_layers_from_config,
             "config.json says %d layers, qwen36_meta.json says %d",
             n_layers_from_config, c->n_layers);
    CFG_NEED(c->hidden > 0 && c->hidden <= 65536, "hidden %d out of range", c->hidden);
    CFG_NEED(c->vocab > 0, "vocab %d must be positive", c->vocab);
    CFG_NEED(c->n_experts > 0 && c->n_experts <= 1024,
             "num_experts %d out of range 1..1024 (keep[] in moe())", c->n_experts);
    CFG_NEED(c->topk > 0 && c->topk <= 256,
             "topk %d out of range 1..256 (idx[]/val[] in moe())", c->topk);
    CFG_NEED(c->topk <= c->n_experts, "topk %d exceeds num_experts %d",
             c->topk, c->n_experts);
    CFG_NEED(c->inter > 0 && c->shared_inter > 0,
             "moe_inter %d / shared_inter %d must be positive", c->inter, c->shared_inter);
    CFG_NEED(c->q_heads > 0 && c->kv_heads > 0 && c->head_dim > 0,
             "attention dims q_heads=%d kv_heads=%d head_dim=%d must be positive",
             c->q_heads, c->kv_heads, c->head_dim);
    CFG_NEED(c->q_heads % c->kv_heads == 0,
             "q_heads %d is not a multiple of kv_heads %d (GQA grouping)",
             c->q_heads, c->kv_heads);
    CFG_NEED(c->k_head_dim > 0 && c->v_head_dim > 0 && c->q_head_dim > 0,
             "per-head dims must be positive");
    /* DeltaNet: every one of these indexes a buffer or divides. */
    if (c->n_active < c->n_layers) {          /* at least one DeltaNet layer */
        CFG_NEED(c->dn_vheads > 0 && c->dn_kheads > 0,
                 "dn_vheads %d / dn_kheads %d must be positive (rep = vh / vk)",
                 c->dn_vheads, c->dn_kheads);
        CFG_NEED(c->dn_vheads % c->dn_kheads == 0,
                 "dn_vheads %d is not a multiple of dn_kheads %d",
                 c->dn_vheads, c->dn_kheads);
        CFG_NEED(c->dn_kdim > 0, "dn_kdim %d must be positive", c->dn_kdim);
        CFG_NEED(c->dn_vdim > 0 && c->dn_vdim <= 512,
                 "dn_vdim %d out of range 1..512 (kvl[]/dl[] in deltanet())",
                 c->dn_vdim);
        CFG_NEED(c->dn_convk >= 2, "dn_convk %d must be >= 2 (conv ring is convk-1)",
                 c->dn_convk);
        CFG_NEED(c->dn_conv_dim ==
                 2 * c->dn_kheads * c->dn_kdim + c->dn_vheads * c->dn_vdim,
                 "dn_conv_dim %d != 2*kheads*kdim + vheads*vdim (%d)",
                 c->dn_conv_dim,
                 2 * c->dn_kheads * c->dn_kdim + c->dn_vheads * c->dn_vdim);
    }
}

static void load_meta(Cfg *c, const char *snap) {
    char path[2048]; snprintf(path, sizeof(path), "%s/qwen36_meta.json", snap);
    FILE *f = fopen(path, "rb"); if (!f) { printf("[meta] %s not found; using i%%4==3 + defaults\n", path); return; }
    fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
    char *buf = malloc((size_t)n+1); if(!buf){fclose(f);return;}
    if(fread(buf,1,(size_t)n,f)!=(size_t)n){ free(buf); fclose(f); return; } buf[n]=0; fclose(f);
    char *arena=NULL; jval *r = json_parse(buf, &arena);
    if (r && r->t == J_OBJ) {
        jval *v;
        #define G(name,field) if((v=json_get(r,name))&&v->t==J_NUM) c->field=(int)v->num
        G("hidden", hidden); G("n_layers", n_layers); G("n_active", n_active);
        G("q_heads", q_heads); G("kv_heads", kv_heads); G("head_dim", head_dim);
        G("q_head_dim", q_head_dim); G("k_head_dim", k_head_dim); G("v_head_dim", v_head_dim);
        G("o_in", o_in); G("rope_dim", rope_dim); G("qk_rope_head_dim", rope_dim);
        G("expert_gs", expert_gs);
        G("num_experts", n_experts); G("topk", topk);
        G("moe_inter", inter); G("shared_inter", shared_inter);
        G("n_group", n_group); G("topk_group", topk_group);
        G("dn_vheads", dn_vheads); G("dn_kheads", dn_kheads); G("dn_kdim", dn_kdim);
        G("dn_vdim", dn_vdim); G("dn_convk", dn_convk); G("dn_conv_dim", dn_conv_dim);
        #undef G
        if((v=json_get(r,"partial_rotary_factor"))&&v->t==J_NUM) c->partial_rotary_factor=(float)v->num;
        if((v=json_get(r,"rope_theta"))&&v->t==J_NUM) c->theta=(float)v->num;
        if((v=json_get(r,"rms_eps"))&&v->t==J_NUM) c->eps=(float)v->num;
        if((v=json_get(r,"attn_output_gate"))&&v->t==J_BOOL) c->attn_output_gate=v->boolean;
        if((v=json_get(r,"norm_topk_prob"))&&v->t==J_BOOL) c->norm_topk=v->boolean;
        if((v=json_get(r,"has_bias"))&&v->t==J_BOOL) c->has_bias=v->boolean;
        if((v=json_get(r,"has_qk_norm"))&&v->t==J_BOOL) c->has_qk_norm=v->boolean;
        /* derive rotary_dim from head_dim * partial_rotary_factor (HF formula) */
        if (c->partial_rotary_factor > 0.f)
            c->rotary_dim = (int)(c->head_dim * c->partial_rotary_factor + 0.5f);
        else
            c->rotary_dim = c->head_dim;
        if (c->rotary_dim < 2) c->rotary_dim = 2;
        if (c->rotary_dim % 2 != 0) c->rotary_dim += 1;
        if (c->rotary_dim > c->head_dim) c->rotary_dim = c->head_dim;
        /* rebuild is_attn from explicit layer_types if present */
        jval *lt = json_get(r,"layer_types");
        if (lt && lt->t==J_ARR) {
            for (int i=0;i<c->n_layers;i++) c->is_attn[i]=0;
            c->n_active=0;
            for (int i=0;i<lt->len && i<c->n_layers;i++){
                const char *s = (lt->kids[i]->t==J_STR)? lt->kids[i]->str : "";
                if (s && strcmp(s,"full_attention")==0) { c->is_attn[i]=1; c->n_active++; }
            }
        }
    }
    free(buf); free(arena);
    fprintf(stderr, "[meta] loaded: q_heads=%d kv_heads=%d head_dim=%d q_head_dim=%d k_head_dim=%d v_head_dim=%d "
           "o_in=%d rotary_dim=%d n_experts=%d topk=%d inter=%d shared_inter=%d n_group=%d topk_group=%d "
           "attn_output_gate=%d n_active=%d\n",
           c->q_heads, c->kv_heads, c->head_dim, c->q_head_dim, c->k_head_dim, c->v_head_dim,
           c->o_in, c->rotary_dim, c->n_experts, c->topk, c->inter, c->shared_inter,
           c->n_group, c->topk_group, c->attn_output_gate, c->n_active);
    if (c->dn_vheads > 0)
        fprintf(stderr, "[meta] DeltaNet: vheads=%d kheads=%d kdim=%d vdim=%d convk=%d conv_dim=%d\n",
               c->dn_vheads, c->dn_kheads, c->dn_kdim, c->dn_vdim, c->dn_convk, c->dn_conv_dim);
}

/* `want` is the element count the forward pass will index with. The container
 * is a file, not an invariant: this used to allocate whatever st_numel reported
 * while every read afterwards used CONFIG dims, so a short tensor was a plain
 * heap OOB read (embed is indexed as m->embed + ids[s]*D). The expert path
 * already refuses a wrong size; this is the same discipline for the dense set. */
static float *load_t_n(Model *m, const char *name, int64_t want) {
    int64_t n = st_numel(&m->S, name);
    if (n < 0) { fprintf(stderr, "missing %s\n", name); exit(1); }
    if (want > 0 && n != want) {
        fprintf(stderr, "%s: %lld elements, config implies %lld -- refusing\n",
                name, (long long)n, (long long)want); exit(1);
    }
    float *p = falloc(n);
    st_read_f32(&m->S, name, p, 0);
    return p;
}

static void model_init(Model *m, const char *snap, int cap, int bits) {
    memset(m, 0, sizeof(*m));
    m->quant_bits = bits;
    load_cfg(&m->c, snap);
    int n_layers_from_config = m->c.n_layers;
    load_meta(&m->c, snap);
    validate_cfg(&m->c, n_layers_from_config);
    if (m->c.rotary_dim > m->c.head_dim || m->c.rotary_dim % 2 != 0) {
        fprintf(stderr, "rotary_dim %d invalid for head_dim %d\n", m->c.rotary_dim, m->c.head_dim); exit(1);
    }
    st_init(&m->S, snap);
    Cfg *c = &m->c;
    double t0 = now_s();
    int embed_f32_fallback = getenv("COLI_EMBED_F32") && atoi(getenv("COLI_EMBED_F32")) == 1;
    int64_t embed_numel = (int64_t)c->vocab * c->hidden;
    st_tensor *et = st_find(&m->S, "model.embed_tokens.weight");
    if (et && et->dtype == 1 && !embed_f32_fallback) {
        /* Native FP16 resident embedding: 1.017 GB instead of 2.034 GB */
        m->embed_f16 = malloc((size_t)embed_numel * sizeof(uint16_t));
        if (!m->embed_f16) { fprintf(stderr, "OOM allocating FP16 embedding\n"); exit(1); }
        st_read_raw(&m->S, "model.embed_tokens.weight", m->embed_f16, 0);
        m->embed = NULL;
        fprintf(stderr, "[embed] native FP16 embedding resident (%lld elements, %.2f MiB / %.2f GiB)\n",
                (long long)embed_numel, (double)embed_numel*2.0/1048576.0, (double)embed_numel*2.0/1073741824.0);
    } else {
        m->embed = load_t_n(m, "model.embed_tokens.weight", embed_numel);
        m->embed_f16 = NULL;
        fprintf(stderr, "[embed] FP32 embedding resident (%lld elements, %.2f MiB / %.2f GiB)\n",
                (long long)embed_numel, (double)embed_numel*4.0/1048576.0, (double)embed_numel*4.0/1073741824.0);
    }
    m->lm_head    = load_t_n(m, "lm_head.weight", (int64_t)c->vocab * c->hidden);
    m->final_norm = load_t_n(m, "model.norm.weight", c->hidden);
    m->L = calloc(c->n_layers, sizeof(Layer));
    /* Phase 2: the converter stores EVERY layer (Gated-Attention + Gated DeltaNet)
     * under its OWN original index model.layers.{i}. So active_of is the identity
     * map; experts and dense weights are read from model.layers.{i} for all i. */
    m->active_of = malloc((size_t)c->n_layers * sizeof(int));
    for (int i = 0; i < c->n_layers; i++) m->active_of[i] = i;
    char nm[256];
    for (int i = 0; i < c->n_layers; i++) {
        int ai = m->active_of[i];        /* == i for Phase 2 */
        Layer *l = &m->L[i];
        /* input/post layernorms + MoE exist for every layer */
        #define LD(field, suffix, want) snprintf(nm,sizeof(nm),"model.layers.%d." suffix,ai); l->field = load_t_n(m,nm,(want))
        LD(in_ln,  "input_layernorm.weight", c->hidden);
        LD(post_ln,"post_attention_layernorm.weight", c->hidden);
        LD(gate, "mlp.gate.weight", (int64_t)c->n_experts * c->hidden);
        #undef LD
        /* q/k norms are per-head [head_dim]; only on attention layers, load if present */
        if (c->has_qk_norm) {
            snprintf(nm,sizeof(nm),"model.layers.%d.self_attn.q_norm.weight", ai);
            l->qn = st_has(&m->S, nm) ? load_t_n(m, nm, c->head_dim) : NULL;
            snprintf(nm,sizeof(nm),"model.layers.%d.self_attn.k_norm.weight", ai);
            l->kn = st_has(&m->S, nm) ? load_t_n(m, nm, c->head_dim) : NULL;
        } else { l->qn = NULL; l->kn = NULL; }
        /* router correction bias (optional) */
        snprintf(nm,sizeof(nm),"model.layers.%d.mlp.gate.e_score_correction_bias", ai);
        if (st_has(&m->S, nm)) { l->gate_bias = falloc(c->n_experts); st_read_f32(&m->S, nm, l->gate_bias, 0); }
        else l->gate_bias = NULL;
        /* shared expert (dense f32) */
        #define LD2(field, suffix, want) snprintf(nm,sizeof(nm),"model.layers.%d.mlp.shared_expert." suffix,ai); l->field = load_t_n(m,nm,(want))
        LD2(sh_g, "gate_proj.weight", (int64_t)c->shared_inter * c->hidden);
        LD2(sh_u, "up_proj.weight",   (int64_t)c->shared_inter * c->hidden);
        LD2(sh_d, "down_proj.weight", (int64_t)c->hidden * c->shared_inter);
        #undef LD2
        /* shared_expert_gate: Linear(hidden -> 1), sigmoid-gated shared expert */
        snprintf(nm,sizeof(nm),"model.layers.%d.mlp.shared_expert_gate.weight", ai);
        l->sh_gate = st_has(&m->S, nm) ? load_t_n(m, nm, c->hidden) : NULL;
        if (c->is_attn[i]) {
            /* Gated Attention (full_attention) layer */
            #define LD3(field, suffix, want) snprintf(nm,sizeof(nm),"model.layers.%d.self_attn." suffix,ai); l->field = load_t_n(m,nm,(want))
            LD3(q, "q_proj.weight", (int64_t)c->q_heads * c->q_head_dim * c->hidden);
            LD3(k, "k_proj.weight", (int64_t)c->kv_heads * c->k_head_dim * c->hidden);
            LD3(v, "v_proj.weight", (int64_t)c->kv_heads * c->v_head_dim * c->hidden);
            LD3(o, "o_proj.weight", (int64_t)c->hidden * c->o_in);
            #undef LD3
            l->dn_qkv=l->dn_z=l->dn_b=l->dn_a=l->dn_conv=NULL;
            l->dn_dtbias=l->dn_alog=l->dn_norm=l->dn_out=NULL;
        } else {
            /* Gated DeltaNet (linear_attention) layer */
            l->q=l->k=l->v=l->o=NULL;
            #define LD4(field, suffix, want) snprintf(nm,sizeof(nm),"model.layers.%d.linear_attn." suffix,ai); l->field = load_t_n(m,nm,(want))
            int64_t vdim_tot = (int64_t)c->dn_vheads * c->dn_vdim;
            LD4(dn_qkv, "in_proj_qkv.weight", (int64_t)c->dn_conv_dim * c->hidden);
            LD4(dn_z,   "in_proj_z.weight",   vdim_tot * c->hidden);
            LD4(dn_b,   "in_proj_b.weight",   (int64_t)c->dn_vheads * c->hidden);
            LD4(dn_a,   "in_proj_a.weight",   (int64_t)c->dn_vheads * c->hidden);
            LD4(dn_conv,"conv1d.weight",      (int64_t)c->dn_conv_dim * c->dn_convk);
            LD4(dn_dtbias, "dt_bias",         c->dn_vheads);
            LD4(dn_alog,"A_log",              c->dn_vheads);
            LD4(dn_norm, "norm.weight",       c->dn_vdim);
            LD4(dn_out, "out_proj.weight",    (int64_t)c->hidden * vdim_tot);
            #undef LD4
        }
    }
    m->cache = calloc(c->n_layers, sizeof(LCache));
    for (int i = 0; i < c->n_layers; i++) {
        m->cache[i].cap = cap;
        m->cache[i].slots = calloc(cap, sizeof(Slot));
        m->cache[i].loading = malloc((size_t)c->n_experts * sizeof(int16_t));
        memset(m->cache[i].loading, 0xFF, (size_t)c->n_experts * sizeof(int16_t));   /* all -1 */
    }
    /* per-layer DeltaNet recurrent + conv state (only for linear_attention layers) */
    m->DN_rec = calloc(c->n_layers, sizeof(float*));
    m->DN_conv = calloc(c->n_layers, sizeof(float*));
    for (int i = 0; i < c->n_layers; i++) {
        if (c->is_attn[i]) { m->DN_rec[i] = NULL; m->DN_conv[i] = NULL; continue; }
        if (c->dn_vheads <= 0) { fprintf(stderr, "layer %d is DeltaNet but dn dims missing from meta\n", i); exit(1); }
        m->DN_rec[i]  = calloc((size_t)c->dn_vheads * c->dn_kdim * c->dn_vdim, sizeof(float));
        m->DN_conv[i] = calloc((size_t)c->dn_conv_dim * (c->dn_convk - 1), sizeof(float));
    }
    m->freq = calloc((size_t)c->n_layers * c->n_experts, sizeof(uint32_t));
    m->router_mass = calloc((size_t)c->n_layers * c->n_experts, sizeof(double));
    m->hot_pinned = 0; m->freq_token_count = 0;
    m->hot_n         = getenv("HOT")    ? atoi(getenv("HOT"))    : 0;
    m->warmup_tokens = getenv("WARMUP") ? atoi(getenv("WARMUP")) : 5;
    m->token_count = 0;
    m->momentum_logits = calloc((size_t)c->n_layers * c->n_experts, sizeof(float));
    float sv = getenv("SMOOTH") ? (float)atof(getenv("SMOOTH")) : 0.3f;
    if (sv < 0.f) sv = 0.f; if (sv > 0.95f) sv = 0.95f;
    m->pilot_smooth = sv;
    m->is_pinned = calloc((size_t)c->n_layers * c->n_experts, sizeof(uint8_t));
    m->seen = calloc((size_t)c->n_layers * c->n_experts, 1);
    m->resident_mode = getenv("COLIBRI_RESIDENT") ? atoi(getenv("COLIBRI_RESIDENT")) : 0;
    m->resident_collecting = 0;
    m->first_step = 1;
    m->is_queued = calloc((size_t)c->n_layers * c->n_experts, sizeof(uint8_t));
    float cl = getenv("CONF_LIMIT") ? (float)atof(getenv("CONF_LIMIT")) : 0.92f;
    if (cl < 0.1f) cl = 0.1f; if (cl > 1.0f) cl = 1.0f;
    m->pilot_conf_limit = cl;
    m->dense_load_s = now_s() - t0;
}

/* scale counts per expert matrix: per-row (gs=0) or grouped along input dim */
static int64_t scale_count_gu(const Cfg *c){ return c->expert_gs ? (int64_t)c->inter * ((c->hidden + c->expert_gs - 1) / c->expert_gs) : c->inter; }
static int64_t scale_count_d (const Cfg *c){ return c->expert_gs ? (int64_t)c->hidden * ((c->inter  + c->expert_gs - 1) / c->expert_gs) : c->hidden; }

static int expert_fmt(Model *m, int layer, int eid) {
    char nm[256];
    int la = m->active_of[layer];
    snprintf(nm, sizeof(nm), "model.layers.%d.mlp.experts.%d.merged_weight", la, eid);
    st_tensor *tw = st_find(&m->S, nm);
    if (!tw) return 1;
    Cfg *cc = &m->c;
    int64_t ng = (int64_t)cc->inter * cc->hidden, nd = (int64_t)cc->hidden * cc->inter;
    int64_t want_w = ng + ng + nd;
    int64_t want_w3 = (want_w / 64) * 24;
    if (tw->nbytes == want_w3) return 5; /* fmt=5 INT3-g64 (1,179,648 B) */
    if (tw->nbytes == want_w / 2) return 4; /* fmt=4 INT4-g64 (1,572,864 B) */
    if (tw->nbytes == want_w) return 1;     /* fmt=1 INT8 (3,145,728 B) */
    return 1;
}

static int container_fmt(Model *m) {
    int f0 = expert_fmt(m, 0, 0);
    for (int l = 0; l < m->c.n_layers; l++) {
        for (int e = 0; e < m->c.n_experts; e++) {
            int fe = expert_fmt(m, l, e);
            if (fe > 0 && fe != f0) return 0; /* fmt=0 indicates heterogeneous/mixed container */
        }
    }
    return f0;
}

static int container_is_int3(Model *m) { return container_fmt(m) == 5; }
static int container_is_int4(Model *m) { return container_fmt(m) == 4; }

static int unpack_int8_mode(void) {
    static int v = -1;
    if (v < 0) { const char *e = getenv("COLI_UNPACK_INT8"); v = (e && *e == '1'); }
    return v;
}

static void slot_ensure_format(Model *m, Slot *s, int fmt) {
    Cfg *c = &m->c;
    int64_t ng = (int64_t)c->inter * c->hidden;
    int64_t nd = (int64_t)c->hidden * c->inter;
    int64_t want_w = ng + ng + nd;
    int64_t want_w3 = (want_w / 64) * 24;
    int64_t want_w4 = want_w / 2;

    if (!s->gs) {
        float *s_block = falloc(2 * scale_count_gu(c) + scale_count_d(c));
        s->gs = s_block;
        s->us = s_block + scale_count_gu(c);
        s->ds = s_block + 2 * scale_count_gu(c);
        s->pinned = 0;
    }

    if (fmt == 5) {
        /* INT3-g64: 24 bytes per 64 weights */
        int64_t g_sz = (ng / 64) * 24;
        if (!s->w3) {
            if (s->w4) {
                s->w3 = s->w4;
            } else if (s->g) {
                s->w3 = (uint8_t*)s->g;
            } else {
                s->w3 = malloc((size_t)want_w3); s->allocated_weight_bytes = (size_t)want_w3;
                if (!s->w3) { fprintf(stderr, "Error: OOM allocating INT3 slot weights\n"); exit(1); }
            }
        }
        s->g3 = s->w3;
        s->u3 = s->w3 + g_sz;
        s->d3 = s->w3 + g_sz + g_sz;
        s->is_int3 = 1;
        s->is_int4 = 0;
    } else if (fmt == 4 && !unpack_int8_mode()) {
        /* Packed INT4 */
        if (!s->w4) {
            if (s->g) {
                s->w4 = (uint8_t*)s->g;
            } else if (s->w3) {
                s->w4 = realloc(s->w3, (size_t)want_w4); s->allocated_weight_bytes = (size_t)want_w4;
                if (!s->w4) { fprintf(stderr, "Error: OOM expanding slot weights to INT4\n"); exit(1); }
                s->w3 = NULL;
            } else {
                s->w4 = malloc((size_t)want_w4); s->allocated_weight_bytes = (size_t)want_w4;
                if (!s->w4) { fprintf(stderr, "Error: OOM allocating packed slot weights\n"); exit(1); }
            }
        }
        s->g4 = s->w4;
        s->u4 = s->w4 + ng / 2;
        s->d4 = s->w4 + (ng + ng) / 2;
        s->is_int4 = 1;
        s->is_int3 = 0;
    } else {
        /* Unpacked INT8 */
        if (!s->g) {
            if (s->w4) {
                s->g = (int8_t*)realloc(s->w4, (size_t)want_w);
                s->w4 = NULL;
            } else if (s->w3) {
                s->g = (int8_t*)realloc(s->w3, (size_t)want_w);
                s->w3 = NULL;
            } else {
                s->g = malloc((size_t)want_w);
            }
            if (!s->g) { fprintf(stderr, "Error: OOM allocating slot weights\n"); exit(1); }
        }
        s->u = s->g + ng;
        s->d = s->g + ng + ng;
        s->is_int4 = 0;
        s->is_int3 = 0;
    }
}

static void slot_ensure_allocated(Model *m, Slot *s) {
    int fmt = container_fmt(m);
    if (fmt == 0) fmt = 5; /* default to INT3 slot size for initial pre-alloc */
    slot_ensure_format(m, s, fmt);
}

static void load_expert_merged(Model *m, int layer, int eid, Slot *s, int is_pilot, ExpertLoadResult *out) {
    char nm[256], qsnm[256];
    int la = m->active_of[layer];   /* container stores experts under active index */
    snprintf(nm, sizeof(nm), "model.layers.%d.mlp.experts.%d.merged_weight", la, eid);
    snprintf(qsnm, sizeof(qsnm), "model.layers.%d.mlp.experts.%d.qs", la, eid);
    Cfg *cc = &m->c;
    int64_t ng = (int64_t)cc->inter * cc->hidden, nd = (int64_t)cc->hidden * cc->inter;
    int64_t want_w = ng + ng + nd;
    int64_t want_w3 = (want_w / 64) * 24;
    int64_t want_s = 2 * scale_count_gu(cc) + scale_count_d(cc);
    st_tensor *tw = st_find(&m->S, nm), *ts = st_find(&m->S, qsnm);
    if (!tw || (tw->nbytes != want_w && tw->nbytes != want_w / 2 && tw->nbytes != want_w3)) {
        fprintf(stderr, "%s: expert weight is %lld bytes — expected %lld (int8), %lld (int4) or %lld (int3)\n",
                nm, (long long)(tw ? tw->nbytes : -1), (long long)want_w, (long long)(want_w / 2), (long long)want_w3); exit(1); }
    if (!ts || ts->numel != want_s) {
        fprintf(stderr, "%s: scale array is %lld elems — expected %lld (refusing)\n",
                qsnm, (long long)(ts ? ts->numel : -1), (long long)want_s); exit(1); }
    double _t_io0 = tm_now();
    if (tw->nbytes == want_w3) {
        static int noted_3 = 0;
        if (!noted_3) { fprintf(stderr, "[qwen36] packed INT3-g64 expert CPU residency active (1.38 MB/slot)\n"); noted_3 = 1; }
        slot_ensure_format(m, s, 5);
        st_read_raw(&m->S, nm, s->w3, 1);
        s->is_int3 = 1; s->is_int4 = 0;
    } else if (tw->nbytes == want_w / 2) {
        if (!unpack_int8_mode()) {
            static int noted_p = 0;
            if (!noted_p) { fprintf(stderr, "[qwen36] packed INT4 expert CPU residency active (1.77 MB/slot)\n"); noted_p = 1; }
            slot_ensure_format(m, s, 4);
            st_read_raw(&m->S, nm, s->w4, 1);
            s->is_int4 = 1; s->is_int3 = 0;
        } else {
            static int noted_u = 0;
            if (!noted_u) { fprintf(stderr, "[qwen36] int4 packed weights detected — unpacking to int8 in slot\n"); noted_u = 1; }
            slot_ensure_format(m, s, 1);
            uint8_t *raw = (uint8_t *)malloc((size_t)(want_w / 2));
            if (!raw) { fprintf(stderr, "OOM reading int4 expert %s\n", nm); exit(1); }
            st_read_raw(&m->S, nm, raw, 1);
            for (int64_t i = 0; i < want_w; i++) {
                uint8_t byte = raw[i >> 1];
                int8_t v = (int8_t)((i & 1) ? ((byte >> 4) & 0xF) : (byte & 0xF));
                if (v & 8) v -= 16;                 /* sign-extend signed 4-bit */
                s->g[i] = v;
            }
            s->is_int4 = 0; s->is_int3 = 0;
            free(raw);
        }
    } else {
        slot_ensure_format(m, s, 1);
        s->is_int3 = 0;
        s->is_int4 = 0;
        st_read_raw(&m->S, nm, s->g, 1);
    }
    double _t_io_w = tm_now();                       /* weight read (+ optional unpack) done */
    st_read_f32(&m->S, qsnm, s->gs, 0);
    double _t_io1 = tm_now();
    double _io_dt = _t_io1 - _t_io0;
    double _io_w  = _t_io_w - _t_io0;                /* weights segment */
    double _io_s  = _t_io1 - _t_io_w;                /* scales segment */
    int64_t total_loaded_bytes = tw->nbytes + ts->nbytes;
    pthread_mutex_lock(&g_io_stats_mx);
    if (out) {
        out->ms = _io_dt;
        out->bytes = total_loaded_bytes;
        out->fmt = s->is_int3 ? 3 : (s->is_int4 ? 4 : 8);
    }
    if (is_pilot) {
        g_pilot_loads++;
        g_pilot_bytes += total_loaded_bytes;
        g_pilot_expert_admission_ms += _io_dt;
        g_pilot_weight_ms += _io_w;
        g_pilot_scale_ms  += _io_s;
    } else {
        g_demand_loads++;
        g_demand_bytes += total_loaded_bytes;
        g_demand_expert_admission_ms += _io_dt;
        g_demand_weight_ms += _io_w;
        g_demand_scale_ms  += _io_s;
        if (s->is_int3) g_admitted_bytes_int3 += total_loaded_bytes;
        else if (s->is_int4) g_admitted_bytes_int4 += total_loaded_bytes;
    }
    pthread_mutex_unlock(&g_io_stats_mx);
}

/* Static per-(layer,eid) expert metadata for the v4 request stream: format and
 * byte size are container constants, NOT cache outcomes — any policy needs
 * them for byte-budget decisions. Mirrors load_expert_merged sizing exactly. */
static void req_emit_expert_meta(Model *m) {
    Cfg *cc = &m->c;
    int64_t ng = (int64_t)cc->inter * cc->hidden, nd = (int64_t)cc->hidden * cc->inter;
    int64_t want_w = ng + ng + nd;
    int64_t want_w3 = (want_w / 64) * 24;
    int64_t want_s = 2 * scale_count_gu(cc) + scale_count_d(cc);
    for (int layer = 0; layer < cc->n_layers && layer < QWEN36_REQ_MAX_LAYERS; layer++) {
        int la = m->active_of[layer];
        for (int eid = 0; eid < cc->n_experts && eid < QWEN36_REQ_MAX_EXPERTS; eid++) {
            char nm[256], qsnm[256];
            snprintf(nm, sizeof nm, "model.layers.%d.mlp.experts.%d.merged_weight", la, eid);
            snprintf(qsnm, sizeof qsnm, "model.layers.%d.mlp.experts.%d.qs", la, eid);
            st_tensor *tw = st_find(&m->S, nm), *ts = st_find(&m->S, qsnm);
            int fmt = -1; int64_t bytes = -1;
            if (tw && ts && ts->numel == want_s) {
                if (tw->nbytes == want_w3) fmt = 3;
                else if (tw->nbytes == want_w / 2) fmt = 4;
                else if (tw->nbytes == want_w) fmt = 8;
                bytes = tw->nbytes + ts->nbytes;
                if (layer < QWEN36_REQ_MAX_LAYERS && eid < QWEN36_REQ_MAX_EXPERTS) {
                    g_emeta_fmt[layer][eid] = fmt; g_emeta_bytes[layer][eid] = bytes;
                }
            }
            if (g_req_fp) { unsigned long long s_ = ++g_req_seq;
                fprintf(g_req_fp, "%llu E %d %d %d %lld\n", s_, layer, eid, fmt, (long long)bytes); }
        }
    }
    g_req_meta_ready = 1;
}

static void expert_get(Model *m, int layer, int eid, Slot **out, float router_mass) {
    LCache *lc = &m->cache[layer];
    int _tp = tm_on();
    double _tl0 = _tp ? tm_now() : 0;
    pthread_mutex_lock(&g_pilot_mx);
    /* v4 request event: emitted BEFORE any cache decision — carries intent
     * (who wants which expert, when, with what router mass), never outcomes */
    if (g_req_fp) { unsigned long long s_ = ++g_req_seq;
        fprintf(g_req_fp, "%llu R DEMAND %lld %d %d %.6f\n",
                s_, (long long)g_trace_tok, layer, eid, (double)router_mass); }
    double _tl1 = _tp ? tm_now() : 0;
    double _ts0 = _tl1;
    for (int i = 0; i < lc->n; i++) if (lc->slots[i].eid == eid) {
        m->hits++; g_acq_hits++; lc->slots[i].used = ++m->clock; *out = &lc->slots[i];
        if ((*out)->is_int3) g_cache_hit_int3++;
        else if ((*out)->is_int4) g_cache_hit_int4++;
        if (_tp) { double _tn = tm_now(); g_eg_lock_wait_ms += _tl1 - _tl0; g_eg_lookup_ms += _tn - _tl1; }
        trace_emit("DEMAND", "HIT", g_trace_tok, layer, eid,
                   (*out)->is_int3 ? 3 : ((*out)->is_int4 ? 4 : 8), 0, 0.0, -1,
                   (int)(*out - lc->slots));
        pthread_mutex_unlock(&g_pilot_mx); return;
    }
    /* NOTE: miss accounting happens only when we actually proceed to load —
     * a coalesced acquisition (waited for an in-flight load, then hit the
     * published copy) must count exactly ONCE, as a hit. */
    Cfg *c = &m->c; Slot *s;
    if (_tp) { double _tn = tm_now(); g_eg_lookup_ms += _tn - _ts0; }   /* failed hit scan */
    /* COALESCE: another admission (pilot worker) may already be loading this
     * expert. Wait for its publish instead of launching a duplicate load.
     * Lock is RELEASED while waiting so the publisher can take it. */
    if (lc->loading[eid] >= 0) {
        g_demand_coalesce_waits++;
        while (lc->loading[eid] >= 0) {
            pthread_mutex_unlock(&g_pilot_mx);
            sleep_ms(1);
            pthread_mutex_lock(&g_pilot_mx);
            for (int i = 0; i < lc->n; i++) if (lc->slots[i].eid == eid) {
                /* the coalesced load published: serve as a resident hit */
                m->hits++; g_acq_hits++; lc->slots[i].used = ++m->clock; *out = &lc->slots[i];
                if ((*out)->is_int3) g_cache_hit_int3++;
                else if ((*out)->is_int4) g_cache_hit_int4++;
                trace_emit("DEMAND", "HIT", g_trace_tok, layer, eid,
                           (*out)->is_int3 ? 3 : ((*out)->is_int4 ? 4 : 8), 0, 0.0, -1, i);
                pthread_mutex_unlock(&g_pilot_mx); return;
            }
            if (lc->loading[eid] < 0 && lc->n < lc->cap) break;   /* loader vanished without publish (cannot happen today) — recover */
        }
        /* re-scan once more after the registry cleared; fall through to a
         * normal miss only if still absent */
        for (int i = 0; i < lc->n; i++) if (lc->slots[i].eid == eid) {
            m->hits++; g_acq_hits++; lc->slots[i].used = ++m->clock; *out = &lc->slots[i];
            if ((*out)->is_int3) g_cache_hit_int3++;
            else if ((*out)->is_int4) g_cache_hit_int4++;
            trace_emit("DEMAND", "HIT", g_trace_tok, layer, eid,
                       (*out)->is_int3 ? 3 : ((*out)->is_int4 ? 4 : 8), 0, 0.0, -1, i);
            pthread_mutex_unlock(&g_pilot_mx); return;
        }
    }
    /* proceeding to a real load: count the miss here (post-coalesce decision) */
    m->miss++;
    g_acq_miss++;
    int64_t _victim_eid = -1;   /* -1 = free capacity (no eviction) */
    if (lc->n < lc->cap) { s = &lc->slots[lc->n++]; slot_ensure_allocated(m, s); }
    else {
        /* LRU eviction — skip pinned and in-flight (eid==-1) slots */
        double _tv0 = _tp ? tm_now() : 0;
        int lru = -1;
        for (int i = 0; i < lc->n; i++) {
            if (lc->slots[i].pinned || lc->slots[i].eid < 0) continue;
            if (lru < 0 || lc->slots[i].used < lc->slots[lru].used) lru = i;
        }
        if (lru < 0) {
            /* All slots are pinned or in-flight; find the oldest non-in-flight
             * slot (may be pinned, but never one currently being loaded). */
            for (int i = 0; i < lc->n; i++) { if (lc->slots[i].eid < 0) continue; if (lru < 0 || lc->slots[i].used < lc->slots[lru].used) lru = i; }
        }
        while (lru < 0) {
            /* EVERY slot is in flight: each buffer is owned by an unlocked pread
             * in the pilot worker (or a demand load) that will publish into it.
             * The old last resort (lru=0) stole such a slot mid-load — two writers
             * racing the same slab, then whichever published last decided the
             * expert id the resident bytes answered to. Wait for a publish instead
             * and rescan; in-flight always drains because a load either finishes
             * or the process is already dead in the water.
             *
             * Taken verbatim from olmoe.c, which this cache derives from and
             * where this exact fallback was deleted for exactly this reason.
             * Reachable whenever cap is smaller than the number of candidates a
             * layer has in flight — PILOT queues up to 128 per layer — i.e. on
             * any small-RAM box, and it corrupts silently rather than crashing. */
            pthread_mutex_unlock(&g_pilot_mx);
            sleep_ms(1);
            pthread_mutex_lock(&g_pilot_mx);
            for (int i = 0; i < lc->n; i++) {
                if (lc->slots[i].eid < 0) continue;
                if (lru < 0 || lc->slots[i].used < lc->slots[lru].used) lru = i;
            }
        }
        s = &lc->slots[lru]; s->pinned = 0;
        if (_tp) g_eg_victim_ms += tm_now() - _tv0;   /* includes in-flight wait spins */
        _victim_eid = s->eid;   /* evicted expert id (trace) */
        trace_emit("DEMAND", "EVICT", g_trace_tok, layer, _victim_eid,
                   s->is_int3 ? 3 : (s->is_int4 ? 4 : 8), 0, 0.0, -1, lru);
    }
    s->eid = -1; s->used = ++m->clock;
    lc->loading[eid] = (int16_t)(s - lc->slots);   /* reserve identity: one loader per (layer,eid) */
    pthread_mutex_unlock(&g_pilot_mx);
    ExpertLoadResult res;
    load_expert_merged(m, layer, eid, s, 0, &res);
    double _tl2 = _tp ? tm_now() : 0;
    pthread_mutex_lock(&g_pilot_mx);
    s->eid = eid; s->pinned = m->is_pinned[layer * c->n_experts + eid]; s->used = ++m->clock;
    lc->loading[eid] = -1;   /* publish: exactly one resident slot now represents (layer,eid) */
    if (s->is_int3) g_cache_miss_int3++;
    else if (s->is_int4) g_cache_miss_int4++;
    if (_tp) g_eg_lock_wait_ms += tm_now() - _tl2;
    trace_emit("DEMAND", "INSERT", g_trace_tok, layer, eid,
               res.fmt, res.bytes, res.ms, _victim_eid, (int)(s - lc->slots));
    *out = s; pthread_mutex_unlock(&g_pilot_mx);
}

static void pin_hot_experts(Model *m) {
    Cfg *c = &m->c;
    if (m->hot_n <= 0 || m->hot_pinned) return;
    m->hot_pinned = 1;
    int is_dynamic = (m->hot_n >= 100);
    double thresh = is_dynamic ? (double)m->hot_n / 1000.0 : 0.0;
    int pinned_total = 0;
    for (int l = 0; l < c->n_layers; l++) {
        uint32_t *freq_l = m->freq + (int64_t)l * c->n_experts;
        uint64_t layer_total = 0;
        for (int e = 0; e < c->n_experts; e++) layer_total += freq_l[e];
        if (layer_total == 0) continue;
        int max_pin = m->cache[l].cap - 8; if (max_pin < 4) max_pin = 4;
        int hn = is_dynamic ? max_pin : (m->hot_n < c->n_experts ? m->hot_n : c->n_experts);
        if (hn > 256) hn = 256;
        int hot_eids[256], actual_hn = 0;
        for (int k = 0; k < hn; k++) {
            int best = -1; uint32_t bv = 0;
            for (int e = 0; e < c->n_experts; e++) {
                int already = 0;
                for (int j = 0; j < k; j++) if (hot_eids[j] == e) { already = 1; break; }
                if (!already && freq_l[e] > bv) { bv = freq_l[e]; best = e; }
            }
            if (best < 0 || bv == 0) break;
            if (is_dynamic && bv < thresh * layer_total) break;
            hot_eids[k] = best; actual_hn++;
        }
        for (int k = 0; k < actual_hn; k++) {
            int eid = hot_eids[k];
            m->is_pinned[l * c->n_experts + eid] = 1;
            LCache *lc = &m->cache[l];
            int found = 0;
            pthread_mutex_lock(&g_pilot_mx);
            for (int i = 0; i < lc->n; i++) if (lc->slots[i].eid == eid) { lc->slots[i].pinned = 1; found = 1; break; }
            pthread_mutex_unlock(&g_pilot_mx);
            if (!found && g_pilot > 0) {
                ensure_pilot_worker_started(m);
                unsigned w = __atomic_load_n(&pilot_w, __ATOMIC_RELAXED);
                unsigned r = __atomic_load_n(&pilot_r, __ATOMIC_ACQUIRE);
                int gidx = l * c->n_experts + eid;
                pthread_mutex_lock(&g_pilot_mx);
                int already = m->is_queued[gidx];
                if (!already && w - r < 4096) {
                    pilot_q[w & 4095].l = l; pilot_q[w & 4095].e = eid; m->is_queued[gidx] = 1;
                    __atomic_store_n(&pilot_w, w + 1, __ATOMIC_RELEASE);
                }
                pthread_mutex_unlock(&g_pilot_mx);
            }
            pinned_total++;
        }
    }
    fprintf(stderr, "[HOT] Pinned %d experts (top-%d/layer) after %d warmup tokens\n", pinned_total, m->hot_n, m->freq_token_count);
}

/* COLIBRI_RESIDENT: after prefill (mode 1) and continuously through decode (mode 2),
 * pin every expert this prompt routed to, so the CPU LRU never evicts their RAM
 * slots. `quiet` suppresses the log line when nothing new was pinned
 * (used for the per-token mid-decode calls). A per-layer pin budget = cap prevents
 * pinning more experts than fit in the cache (which would deadlock the LRU). */
static int apply_resident(Model *m, int quiet) {
    Cfg *c = &m->c;
    int newly = 0, over = 0;
    for (int l = 0; l < c->n_layers; l++) {
        uint8_t *row = m->seen + (int64_t)l * c->n_experts;
        int cap = m->cache[l].cap;
        int already = 0;
        for (int e = 0; e < c->n_experts; e++) if (m->is_pinned[l * c->n_experts + e]) already++;
        int budget = cap - already;                 /* free pin slots in this layer */
        int seen = 0;
        for (int e = 0; e < c->n_experts; e++) {
            if (!row[e]) continue;
            seen++;
            if (m->is_pinned[l * c->n_experts + e]) continue;   /* already pinned */
            if (budget <= 0) { over++; continue; }             /* layer full, skip */
            m->is_pinned[l * c->n_experts + e] = 1;
            newly++; budget--;
        }
        if (seen > cap) over += seen - cap;
        LCache *lc = &m->cache[l];
        for (int i = 0; i < lc->n; i++)
            if (lc->slots[i].eid >= 0 && row[lc->slots[i].eid])
                lc->slots[i].pinned = 1;
    }
    if (!quiet || newly > 0)
        fprintf(stderr, "[RESIDENT] Pinned %d new experts (CPU no-evict -> GPU resident)%s\n",
                newly, over > 0 ? " | WARN: exceed per-layer cap, raise cap for full coverage" : "");
    return newly;
}

/* ---------- RoPE: applied to the FIRST rope_dim dims of each head (Qwen3 partial rope) ---------- */
static void rope_head_partial(float *x, int pos, int rope_dim, int head_dim, float theta) {
    int h = rope_dim / 2;
    for (int j = 0; j < h; j++) {
        float inv = powf(theta, -2.0f * j / rope_dim);
        float ang = pos * inv, cs = cosf(ang), sn = sinf(ang);
        float a = x[j], b = x[j+h];
        x[j]   = a*cs - b*sn;
        x[j+h] = b*cs + a*sn;
    }
}

/* Gated Attention (GQA) matching HF Qwen3_5MoeAttention:
 *  - q_proj outputs query(head_dim) ++ attn_output_gate(head_dim); k/v are head_dim.
 *  - per-head q/k RMSNorm (weight [head_dim], 1.0+weight).
 *  - partial RoPE on the first rotary_dim dims of each head (text: mRoPE == standard).
 *  - scale = head_dim^-0.5; GQA repeat_kv.
 *  - attn_out = attn_out * sigmoid(gate), then o_proj (input dim = q_heads*head_dim). */
static void attention(Model *m, Layer *l, int layer, float *x, int S, int pos_base, float *out) {
    Cfg *c = &m->c;
    int H = c->q_heads, KV = c->kv_heads, hd = c->head_dim, D = c->hidden;
    int kvd = c->k_head_dim;
    int qdim = c->q_head_dim;                  /* per-head q total (query+gate) */
    int q_out = H * qdim;                      /* q_proj output dim */
    int kv_out = KV * kvd;                     /* k/v_proj output dim */
    int q_per_kv = H / KV;
    int rotary = c->rotary_dim;
    /* HF always chunks q_proj output into query(head_dim) ++ gate(head_dim),
     * regardless of the attn_output_gate config flag -- so split whenever the
     * q per-head dim exceeds the (k/v) head dim. */
    int gate_dim = (qdim > hd) ? (qdim - hd) : 0;
    float *q = falloc((int64_t)S*q_out);
    float *k = falloc((int64_t)S*kv_out);
    float *vv= falloc((int64_t)S*kv_out);
    matmul_d(q, x, l->q, S, D, q_out);
    matmul_d(k, x, l->k, S, D, kv_out);
    matmul_d(vv, x, l->v, S, D, kv_out);
    /* split q into query (first hd) and gate (next gate_dim), both per head */
    float *query = falloc((int64_t)S*H*hd);
    float *gate  = falloc((int64_t)S*H*gate_dim);
    for (int s = 0; s < S; s++) {
        for (int hh = 0; hh < H; hh++) {
            const float *qs = q + (int64_t)s*q_out + hh*qdim;
            memcpy(query + ((int64_t)s*H + hh)*hd, qs, hd*sizeof(float));
            if (gate_dim) memcpy(gate + ((int64_t)s*H + hh)*gate_dim, qs + hd, gate_dim*sizeof(float));
        }
    }
    for (int s = 0; s < S; s++) {
        for (int hh = 0; hh < H; hh++) {
            float *qh = query + ((int64_t)s*H + hh)*hd;
            if (l->qn) rmsnorm_row(qh, qh, l->qn, hd, c->eps);
            rope_head_partial(qh, pos_base + s, rotary, hd, c->theta);
        }
        for (int kvh = 0; kvh < KV; kvh++) {
            float *kh = k + (int64_t)s*KV*kvd + kvh*kvd;
            if (l->kn) rmsnorm_row(kh, kh, l->kn, kvd, c->eps);
            rope_head_partial(kh, pos_base + s, rotary, kvd, c->theta);
        }
    }
    for (int s = 0; s < S; s++) for (int kvh = 0; kvh < KV; kvh++) {
        int t = pos_base + s;
        memcpy(m->K[layer] + ((int64_t)kvh*m->max_t + t)*kvd, k + (int64_t)s*KV*kvd + kvh*kvd, kvd*sizeof(float));
        memcpy(m->V[layer] + ((int64_t)kvh*m->max_t + t)*kvd, vv + (int64_t)s*KV*kvd + kvh*kvd, kvd*sizeof(float));
    }
    float scale = 1.f / sqrtf((float)hd);
    float *ctx = falloc((int64_t)S*H*hd);
    #pragma omp parallel for collapse(2) schedule(static)
    for (int hh = 0; hh < H; hh++) {
        for (int s = 0; s < S; s++) {
            int kvh = hh / q_per_kv;
            int qpos = pos_base + s;
            const float *qv = query + ((int64_t)s*H + hh)*hd;
            int tid = 0;
#ifdef _OPENMP
            tid = omp_get_thread_num();
#endif
            float *sc = m->attn_sc + (int64_t)tid * m->kv_cap;
            for (int t = 0; t <= qpos; t++) {
                const float *kv = m->K[layer] + ((int64_t)kvh*m->max_t + t)*kvd;
                float acc = 0; for (int dd = 0; dd < kvd; dd++) acc += qv[dd]*kv[dd];
                sc[t] = acc * scale;
            }
            softmax_row(sc, qpos+1);
            float *cx = ctx + ((int64_t)s*H + hh)*hd;
            for (int dd = 0; dd < kvd; dd++) cx[dd] = 0;
            for (int t = 0; t <= qpos; t++) {
                const float *vrow = m->V[layer] + ((int64_t)kvh*m->max_t + t)*kvd;
                float a = sc[t]; for (int dd = 0; dd < kvd; dd++) cx[dd] += a * vrow[dd];
            }
        }
    }
    /* apply attn_output_gate: attn_out *= sigmoid(gate) */
    float *ag = falloc((int64_t)S*H*hd);
    for (int s = 0; s < S; s++) for (int hh = 0; hh < H; hh++) for (int dd = 0; dd < hd; dd++) {
        int o = ((int64_t)s*H + hh)*hd + dd;
        float g = gate_dim ? gate[o] : 0.f;
        ag[o] = ctx[o] * (1.f / (1.f + expf(-g)));
    }
    matmul_d(out, ag, l->o, S, H*hd, D);
    free(q); free(k); free(vv); free(query); free(gate); free(ctx); free(ag);
}

/* MoE: grouped top-k routing (+ optional router bias) + shared expert.
 * Mirrors HF Qwen3 MoE: softmax(gate), optional group-limited top-k, normalized
 * weights, sum routed experts, then add the un-gated shared expert. */
static void moe(Model *m, Layer *l, int layer, float *x, int S, float *out) {
    Cfg *c = &m->c; int D = c->hidden, E = c->n_experts, K = c->topk, I = c->inter;
    float *logits = falloc((int64_t)S*E);
    double _tr0 = tm_on() ? tm_now() : 0.0;
    matmul_d(logits, x, l->gate, S, D, E);
    if (c->has_bias && l->gate_bias) {
        for (int s = 0; s < S; s++) { float *pr = logits + (int64_t)s*E; for (int e = 0; e < E; e++) pr[e] += l->gate_bias[e]; }
    }
    memset(out, 0, (int64_t)S*D*sizeof(float));
    float *g = falloc(I), *u = falloc(I), *hh = falloc(D);
    float *sh = falloc(I), *shu = falloc(I), *shd = falloc(D);  /* shared expert scratch */

    for (int s = 0; s < S; s++) {
        float *pr = logits + (int64_t)s*E;
        if (m->momentum_logits && m->pilot_smooth > 0.f) {
            float *ema = m->momentum_logits + (int64_t)layer * E;
            int is_zero = 1; for (int e = 0; e < E; e++) if (ema[e] != 0.f) { is_zero = 0; break; }
            if (is_zero) { for (int e = 0; e < E; e++) ema[e] = pr[e]; }
            else { for (int e = 0; e < E; e++) ema[e] = (1.f - m->pilot_smooth)*pr[e] + m->pilot_smooth*ema[e]; }
        }
        softmax_row(pr, E);
        /* group-limited top-k selection */
        uint8_t keep[1024]; int Ec = E < 1024 ? E : 1024;
        if (c->n_group > 1 && c->n_group <= Ec) {
            int per = E / c->n_group;
            float gs[1024];
            for (int gi = 0; gi < c->n_group; gi++) {
                float b1 = -1e30f, b2 = -1e30f;
                for (int e = gi*per; e < gi*per+per; e++) { float v = pr[e]; if (v > b1) { b2=b1; b1=v; } else if (v > b2) b2=v; }
                gs[gi] = b1 + b2;
            }
            uint8_t gkeep[1024] = {0};
            for (int kk = 0; kk < c->topk_group; kk++) {
                int bg = -1; float bv = -1e30f;
                for (int gi = 0; gi < c->n_group; gi++) { if (!gkeep[gi] && gs[gi] > bv) { bv = gs[gi]; bg = gi; } }
                if (bg < 0) break; gkeep[bg] = 1;
            }
            for (int e = 0; e < Ec; e++) keep[e] = 0;
            for (int gi = 0; gi < c->n_group; gi++) if (gkeep[gi]) for (int e = gi*per; e < gi*per+per; e++) keep[e] = 1;
        } else {
            for (int e = 0; e < Ec; e++) keep[e] = 1;
        }
        int idx[256]; float val[256];
        for (int kk = 0; kk < K; kk++) {
            int best = -1; float bv = -1e30f;
            for (int e = 0; e < E; e++) {
                if (!keep[e]) continue;
                int taken = 0; for (int j = 0; j < kk; j++) if (idx[j]==e){taken=1;break;}
                if (!taken && pr[e] > bv) { bv = pr[e]; best = e; }
            }
            idx[kk] = best; val[kk] = bv;
        }
        if (m->resident_collecting) {
            for (int kk = 0; kk < K; kk++) if (idx[kk] >= 0) m->seen[(int64_t)layer * E + idx[kk]] = 1;
        }
        /* HF renormalizes the top-k router weights unconditionally */
        { float sm=0; for (int kk=0;kk<K;kk++) sm+=val[kk]; if (sm>0) for (int kk=0;kk<K;kk++) val[kk]/=sm; }

        if (tm_on()) {
            double dt_r = tm_now() - _tr0;
            tm_add(S, 4, dt_r);
            if (S == 1) g_moe_sub[0] += dt_r;
        }

        const float *xs = x + (int64_t)s*D;
        if (m->freq) {
            uint32_t *freq_l = m->freq + (int64_t)layer * E;
            for (int kk = 0; kk < K; kk++) if (idx[kk] >= 0) freq_l[idx[kk]]++;
        }
        if (m->router_mass) {
            double *mass_l = m->router_mass + (int64_t)layer * E;
            for (int kk = 0; kk < K; kk++) if (idx[kk] >= 0) mass_l[idx[kk]] += val[kk];
        }

        /* Expert slot acquisition */
        double _t_lk = tm_on() ? tm_now() : 0.0;
        Slot *e_slots[256];
        for (int kk = 0; kk < K; kk++) {
            expert_get(m, layer, idx[kk], &e_slots[kk], val[kk]);
            if (S == 1) {
                if (e_slots[kk]->is_int3) g_routed_int3_count++;
                else if (e_slots[kk]->is_int4) g_routed_int4_count++;
            }
        }
        if (tm_on() && S == 1) g_moe_sub[1] += tm_now() - _t_lk;

        if (expert_parallel_on() && K <= 8) {
            /* TOP-K EXPERT-PARALLEL TOPOLOGY (Gate 4) */
            double _t_ep = tm_on() ? tm_now() : 0.0;
            float ep_g[8][512], ep_u[8][512], ep_hh[8][2048];
            #pragma omp parallel for schedule(static) num_threads(K)
            for (int kk = 0; kk < K; kk++) {
                Slot *e = e_slots[kk];
                float *gk = ep_g[kk], *uk = ep_u[kk], *hk = ep_hh[kk];
                if (e->is_int3 && e->w3) {
                    matmul_i3_qe(gk, xs, e->g3, e->gs, D, I);
                    matmul_i3_qe(uk, xs, e->u3, e->us, D, I);
                    for (int i = 0; i < I; i++) { float gv = gk[i]; gk[i] = (gv / (1.f + expf(-gv))) * uk[i]; }
                    matmul_i3_qe(hk, gk, e->d3, e->ds, I, D);
                } else if (e->is_int4 && e->w4) {
                    matmul_i4_qe(gk, xs, e->g4, e->gs, D, I);
                    matmul_i4_qe(uk, xs, e->u4, e->us, D, I);
                    for (int i = 0; i < I; i++) { float gv = gk[i]; gk[i] = (gv / (1.f + expf(-gv))) * uk[i]; }
                    matmul_i4_qe(hk, gk, e->d4, e->ds, I, D);
                } else {
                    matmul_qe(gk, xs, e->g, e->gs, D, I);
                    matmul_qe(uk, xs, e->u, e->us, D, I);
                    for (int i = 0; i < I; i++) { float gv = gk[i]; gk[i] = (gv / (1.f + expf(-gv))) * uk[i]; }
                    matmul_qe(hk, gk, e->d, e->ds, I, D);
                }
            }
            if (tm_on() && S == 1) {
                double dt_ep = tm_now() - _t_ep;
                g_moe_sub[2] += dt_ep; /* record into routed expert compute */
                g_expert_gemv_parallel_invocations += 1; /* 1 parallel region per layer */
            }

            /* Deterministic reduction in exact k order */
            double _t_acc = tm_on() ? tm_now() : 0.0;
            float *os = out + (int64_t)s * D;
            for (int kk = 0; kk < K; kk++) {
                float w = val[kk];
                float *hk = ep_hh[kk];
                for (int d = 0; d < D; d++) os[d] += w * hk[d];
            }
            if (tm_on() && S == 1) g_moe_sub[9] += tm_now() - _t_acc;
        } else {
            /* REFERENCE TOPOLOGY (Expert-serial, GEMV-parallel) */
            for (int kk = 0; kk < K; kk++) {
                Slot *e = e_slots[kk];
                double _t_g = 0, _t_u = 0, _t_d = 0;
                if (e->is_int3 && e->w3) {
                    _t_g = tm_on() ? tm_now() : 0;
                    matmul_i3_qe(g, xs, e->g3, e->gs, D, I);
                    if (tm_on() && S == 1) g_moe_sub[2] += tm_now() - _t_g;

                    _t_u = tm_on() ? tm_now() : 0;
                    matmul_i3_qe(u, xs, e->u3, e->us, D, I);
                    if (tm_on() && S == 1) g_moe_sub[3] += tm_now() - _t_u;

                    for (int i = 0; i < I; i++) { float gv = g[i]; g[i] = (gv / (1.f + expf(-gv))) * u[i]; }

                    _t_d = tm_on() ? tm_now() : 0;
                    matmul_i3_qe(hh, g, e->d3, e->ds, I, D);
                    if (tm_on() && S == 1) g_moe_sub[4] += tm_now() - _t_d;
                } else if (e->is_int4 && e->w4) {
                    _t_g = tm_on() ? tm_now() : 0;
                    matmul_i4_qe(g, xs, e->g4, e->gs, D, I);
                    if (tm_on() && S == 1) g_moe_sub[5] += tm_now() - _t_g;

                    _t_u = tm_on() ? tm_now() : 0;
                    matmul_i4_qe(u, xs, e->u4, e->us, D, I);
                    if (tm_on() && S == 1) g_moe_sub[6] += tm_now() - _t_u;

                    for (int i = 0; i < I; i++) { float gv = g[i]; g[i] = (gv / (1.f + expf(-gv))) * u[i]; }

                    _t_d = tm_on() ? tm_now() : 0;
                    matmul_i4_qe(hh, g, e->d4, e->ds, I, D);
                    if (tm_on() && S == 1) g_moe_sub[7] += tm_now() - _t_d;
                } else {
                    matmul_qe(g, xs, e->g, e->gs, D, I);
                    matmul_qe(u, xs, e->u, e->us, D, I);
                    for (int i = 0; i < I; i++) { float gv = g[i]; g[i] = (gv / (1.f + expf(-gv))) * u[i]; }
                    matmul_qe(hh, g, e->d, e->ds, I, D);
                }
                if (tm_on() && S == 1) g_expert_gemv_parallel_invocations += 3; /* 3 GEMV parallel regions */

                double _t_acc = tm_on() ? tm_now() : 0;
                float w = val[kk];
                float *os = out + (int64_t)s*D;
                for (int d = 0; d < D; d++) os[d] += w * hh[d];
                if (tm_on() && S == 1) g_moe_sub[9] += tm_now() - _t_acc;
            }
        }

        /* shared expert (SwiGLU), sigmoid-gated by shared_expert_gate */
        double _ts = tm_on() ? tm_now() : 0.0;
        int Ish = c->shared_inter;
        matmul_d(sh, xs, l->sh_g, 1, D, Ish);
        matmul_d(shu, xs, l->sh_u, 1, D, Ish);
        for (int i = 0; i < Ish; i++) { float sv = sh[i]; sh[i] = (sv / (1.f + expf(-sv))) * shu[i]; }
        matmul_d(shd, sh, l->sh_d, 1, Ish, D);
        float sgate = 1.f;
        if (l->sh_gate) {
            float sg = 0.f; const float *wg = l->sh_gate;
            for (int i = 0; i < D; i++) sg += xs[i] * wg[i];
            sgate = 1.f / (1.f + expf(-sg));
        }
        float *os = out + (int64_t)s*D;
        for (int d = 0; d < D; d++) os[d] += sgate * shd[d];
        if (tm_on()) {
            double dt_sh = tm_now() - _ts;
            tm_add(S, 3, dt_sh);
            if (S == 1) g_moe_sub[8] += dt_sh;
        }
    }
    free(logits); free(g); free(u); free(hh); free(sh); free(shu); free(shd);
}

static void deltanet(Model *m, Layer *l, int layer, float *x, int S, int pos_base, float *out) {
    (void)pos_base;
    Cfg *c = &m->c;
    int vh = c->dn_vheads, vk = c->dn_kheads, kdim = c->dn_kdim, vdim = c->dn_vdim;
    int convk = c->dn_convk, conv_dim = c->dn_conv_dim;
    int rep = vh / vk;
    int key_dim_tot = vk * kdim;
    int value_dim = vh * vdim;
    float scale = 1.f / sqrtf((float)kdim);
    int H = c->hidden;

    float *qkv = falloc(conv_dim);
    float *z   = falloc(value_dim);
    float *b   = falloc(vh);
    float *a   = falloc(vh);
    float *beta= falloc(vh);
    float *gg  = falloc(vh);
    float *conv_out = falloc(conv_dim);
    float *q = falloc(vh * kdim);
    float *k = falloc(vh * kdim);
    float *outv = falloc(value_dim);
    float *outr = falloc(value_dim);
    float *kv = falloc(vdim);
    float *delta = falloc(vdim);

    float *rec = m->DN_rec[layer];      /* [vh*kdim*vdim] */
    float *ring = m->DN_conv[layer];    /* [conv_dim*(convk-1)] */

    for (int s = 0; s < S; s++) {
        const float *xs = x + (int64_t)s * H;
        double _t0;
        /* [0] DN_QKV_PROJ */
        _t0 = tm_on() ? tm_now() : 0;
        matmul_d(qkv, xs, l->dn_qkv, 1, H, conv_dim);
        if (tm_on() && S==1) g_dn_sub[0] += tm_now() - _t0;

        /* [1] DN_Z_PROJ */
        _t0 = tm_on() ? tm_now() : 0;
        matmul_d(z, xs, l->dn_z, 1, H, value_dim);
        if (tm_on() && S==1) g_dn_sub[1] += tm_now() - _t0;

        /* [2] DN_B_PROJ */
        _t0 = tm_on() ? tm_now() : 0;
        matmul(b, xs, l->dn_b, 1, H, vh);
        if (tm_on() && S==1) g_dn_sub[2] += tm_now() - _t0;

        /* [3] DN_A_PROJ */
        _t0 = tm_on() ? tm_now() : 0;
        matmul(a, xs, l->dn_a, 1, H, vh);
        if (tm_on() && S==1) g_dn_sub[3] += tm_now() - _t0;

        /* [4] DN_CONV */
        _t0 = tm_on() ? tm_now() : 0;
        for (int h = 0; h < vh; h++) {
            beta[h] = 1.f / (1.f + expf(-b[h]));
            gg[h] = -expf(l->dn_alog[h]) * softplus_f(a[h] + l->dn_dtbias[h]);
        }
        for (int cc = 0; cc < conv_dim; cc++) {
            const float *w = l->dn_conv + (int64_t)cc * convk;
            const float *rg = ring + (int64_t)cc * (convk - 1);
            float acc = 0.f;
            for (int kk = 0; kk < convk - 1; kk++) acc += w[kk] * rg[kk];
            acc += w[convk - 1] * qkv[cc];
            conv_out[cc] = acc / (1.f + expf(-acc));   /* silu */
        }
        for (int cc = 0; cc < conv_dim; cc++) {
            float *rg = ring + (int64_t)cc * (convk - 1);
            for (int kk = 0; kk < convk - 2; kk++) rg[kk] = rg[kk + 1];
            rg[convk - 2] = qkv[cc];
        }
        if (tm_on() && S==1) g_dn_sub[4] += tm_now() - _t0;

        /* [5] DN_QK_NORM */
        _t0 = tm_on() ? tm_now() : 0;
        const float *q_in = conv_out;
        const float *k_in = conv_out + key_dim_tot;
        const float *v_in = conv_out + 2 * key_dim_tot;
        for (int h = 0; h < vh; h++) {
            int vk_idx = h / rep;
            memcpy(q + (int64_t)h * kdim, q_in + (int64_t)vk_idx * kdim, kdim * sizeof(float));
            memcpy(k + (int64_t)h * kdim, k_in + (int64_t)vk_idx * kdim, kdim * sizeof(float));
        }
        for (int oh = 0; oh < vh; oh++) {
            float *qh = q + (int64_t)oh * kdim;
            double sq = 1e-6; for (int d = 0; d < kdim; d++) sq += (double)qh[d] * qh[d];
            double nq = sqrt(sq);
            for (int d = 0; d < kdim; d++) qh[d] = (float)((double)qh[d] / nq * scale);
            float *kh = k + (int64_t)oh * kdim;
            double sk = 1e-6; for (int d = 0; d < kdim; d++) sk += (double)kh[d] * kh[d];
            double nk = sqrt(sk);
            for (int d = 0; d < kdim; d++) kh[d] = (float)((double)kh[d] / nk);
        }
        if (tm_on() && S==1) g_dn_sub[5] += tm_now() - _t0;

        /* [6..9] Recurrence */
        _t0 = tm_on() ? tm_now() : 0;
        #pragma omp parallel for schedule(static)
        for (int h = 0; h < vh; h++) {
            float kvl[512], dl[512];   /* vdim <= 512 */
            float *Sh = rec + (int64_t)h * kdim * vdim;
            float egh = expf(gg[h]);
            for (int t = 0; t < kdim * vdim; t++) Sh[t] *= egh;
            const float *kd = k + (int64_t)h * kdim;
            const float *vd = v_in + (int64_t)h * vdim;
            for (int vv = 0; vv < vdim; vv++) kvl[vv] = 0.f;
            for (int kk = 0; kk < kdim; kk++) {
                float kkd = kd[kk]; const float *Sr = Sh + (int64_t)kk * vdim;
                for (int vv = 0; vv < vdim; vv++) kvl[vv] += kkd * Sr[vv];
            }
            for (int vv = 0; vv < vdim; vv++) dl[vv] = (vd[vv] - kvl[vv]) * beta[h];
            for (int kk = 0; kk < kdim; kk++) {
                float kkd = kd[kk]; float *Sr = Sh + (int64_t)kk * vdim;
                for (int vv = 0; vv < vdim; vv++) Sr[vv] += kkd * dl[vv];
            }
            const float *qd = q + (int64_t)h * kdim;
            float *ov = outv + (int64_t)h * vdim;
            for (int vv = 0; vv < vdim; vv++) ov[vv] = 0.f;
            for (int kk = 0; kk < kdim; kk++) {
                float qkd = qd[kk]; const float *Sr = Sh + (int64_t)kk * vdim;
                for (int vv = 0; vv < vdim; vv++) ov[vv] += qkd * Sr[vv];
            }
        }
        if (tm_on() && S==1) {
            double dt_rec = tm_now() - _t0;
            g_dn_sub[6] += dt_rec * 0.10;
            g_dn_sub[7] += dt_rec * 0.40;
            g_dn_sub[8] += dt_rec * 0.25;
            g_dn_sub[9] += dt_rec * 0.25;
        }

        /* [10] DN_GATED_NORM */
        _t0 = tm_on() ? tm_now() : 0;
        #pragma omp parallel for schedule(static)
        for (int h = 0; h < vh; h++) {
            const float *o = outv + (int64_t)h * vdim;
            const float *zr = z + (int64_t)h * vdim;
            const float *w = l->dn_norm;
            double ms = 0; for (int d = 0; d < vdim; d++) ms += (double)o[d] * o[d];
            float r = 1.f / sqrtf((float)(ms / vdim) + c->eps);
            for (int d = 0; d < vdim; d++) {
                float val = o[d] * r * w[d];
                outr[(int64_t)h * vdim + d] = val * zr[d] / (1.f + expf(-zr[d]));
            }
        }
        if (tm_on() && S==1) g_dn_sub[10] += tm_now() - _t0;

        /* [11] DN_OUT_PROJ */
        _t0 = tm_on() ? tm_now() : 0;
        matmul_d(out + (int64_t)s * H, outr, l->dn_out, 1, value_dim, H);
        if (tm_on() && S==1) g_dn_sub[11] += tm_now() - _t0;

        if (layer == 0 && s == 0 && getenv("DN_DBG")) {
            FILE *dbg = fopen(getenv("DN_DBG"), "wb");
            if (dbg) {
                fwrite(conv_out, sizeof(float), conv_dim, dbg);
                fwrite(q, sizeof(float), (int64_t)vh * kdim, dbg);
                fwrite(outv, sizeof(float), value_dim, dbg);
                fwrite(z, sizeof(float), value_dim, dbg);
                fwrite(outr, sizeof(float), value_dim, dbg);
                fwrite(out + (int64_t)s * H, sizeof(float), H, dbg);
                fwrite(b, sizeof(float), vh, dbg);
                fwrite(a, sizeof(float), vh, dbg);
                fwrite(beta, sizeof(float), vh, dbg);
                fwrite(gg, sizeof(float), vh, dbg);
                fclose(dbg);
            }
        }
    }
    free(qkv); free(z); free(b); free(a); free(beta); free(gg);
    free(conv_out); free(q); free(k); free(outv); free(outr); free(kv); free(delta);
}

static float *step(Model *m, const int *ids, int S, int pos_base) {
    Cfg *c = &m->c; int D = c->hidden;
    if (m->resident_mode && m->first_step) m->resident_collecting = 1;
    /* Per-layer residual dump (last token) for torch-free cosine debugging.
     * Set DUMP_LAYERS=<path> to write n_layers * D raw float32 rows. */
    FILE *lf = NULL; const char *lfn = getenv("DUMP_LAYERS");
    if (lfn) { lf = fopen(lfn, "wb"); if (!lf) fprintf(stderr, "DUMP_LAYERS: cannot open %s\n", lfn); }
    if (g_pilot && m->token_count > 0) {
        pthread_mutex_lock(&g_pilot_mx);
        memset(m->is_queued, 0, (size_t)c->n_layers * c->n_experts);
        pthread_mutex_unlock(&g_pilot_mx);
    }
    float *x = falloc((int64_t)S*D);
    for (int s = 0; s < S; s++) {
        /* The gather indexes embed by token id, so an id outside the vocabulary
         * reads off the end. Ids reach here from the tokenizer, from a serve
         * request and from the engine's own sampler -- three sources, one of
         * which is remote, and none of them checked until now. */
        if (ids[s] < 0 || ids[s] >= c->vocab) {
            fprintf(stderr, "token id %d out of range 0..%d -- refusing\n",
                    ids[s], c->vocab - 1);
            exit(1);
        }
        if (m->embed_f16) {
            const uint16_t *row_f16 = m->embed_f16 + (int64_t)ids[s] * D;
            float *row_f32 = x + (int64_t)s * D;
            for (int d = 0; d < D; d++) row_f32[d] = f16_to_f32(row_f16[d]);
        } else {
            memcpy(x + (int64_t)s*D, m->embed + (int64_t)ids[s]*D, D*sizeof(float));
        }
    }
    float *nrm = falloc((int64_t)S*D), *tmp = falloc((int64_t)S*D);
    for (int i = 0; i < c->n_layers; i++) {
        Layer *l = &m->L[i];
        for (int s = 0; s < S; s++) rmsnorm_row(nrm + (int64_t)s*D, x + (int64_t)s*D, l->in_ln, D, c->eps);
        double _t0 = tm_on() ? tm_now() : 0.0;
        if (c->is_attn[i]) {
            attention(m, l, i, nrm, S, pos_base, tmp);
            if (tm_on()) tm_add(S, 1, tm_now()-_t0);
        } else {
            deltanet(m, l, i, nrm, S, pos_base, tmp);
            if (tm_on()) tm_add(S, 0, tm_now()-_t0);
        }
        if (lf) fwrite(tmp + (int64_t)(S-1)*D, sizeof(float), D, lf);   /* sublayer output */
        for (int64_t j = 0; j < (int64_t)S*D; j++) x[j] += tmp[j];
        if (lf) fwrite(x + (int64_t)(S-1)*D, sizeof(float), D, lf);   /* post-deltanet residual */
        if (g_pilot >= 1 && S <= 8 && i + 1 < c->n_layers)
            pilot_prefetch(m, i + 1, x, S);
        for (int s = 0; s < S; s++) rmsnorm_row(nrm + (int64_t)s*D, x + (int64_t)s*D, l->post_ln, D, c->eps);
        _t0 = tm_on() ? tm_now() : 0.0;
        moe(m, l, i, nrm, S, tmp);
        if (tm_on()) tm_add(S, 2, tm_now()-_t0);
        for (int64_t j = 0; j < (int64_t)S*D; j++) x[j] += tmp[j];
        if (lf) fwrite(x + (int64_t)(S-1)*D, sizeof(float), D, lf);
        if (g_pilot >= 2 && S <= 8 && i + 2 < c->n_layers)
            pilot_prefetch(m, i + 2, x, S);
        if (g_pilot >= 3 && S <= 8 && i + 3 < c->n_layers)
            pilot_prefetch(m, i + 3, x, S);
    }
    m->token_count += S; m->freq_token_count += S;
    if (!m->hot_pinned && m->hot_n > 0 && m->freq_token_count >= m->warmup_tokens) pin_hot_experts(m);
    m->kv_len = pos_base + S;
    float *last = falloc(D);
    rmsnorm_row(last, x + (int64_t)(S-1)*D, m->final_norm, D, c->eps);
    float *logit = falloc(c->vocab);
    double _th = tm_on() ? tm_now() : 0.0;
    matmul_d(logit, last, m->lm_head, 1, D, c->vocab);
    if (tm_on()) { tm_add(S, 5, tm_now()-_th);
        if (S==1) { g_tm_dec_tokens++; forge_trace_token_boundary(g_tm_dec_tokens); }
        else g_tm_pre_tokens += S; }
    free(x); free(nrm); free(tmp); free(last);
    if (lf) fclose(lf);
    if (m->resident_collecting) {
        int prefill_end = m->first_step;
        apply_resident(m, prefill_end ? 0 : 1);   /* always report after prefill; quiet mid-decode */
        if (prefill_end) {
            m->first_step = 0;
            if (m->resident_mode < 2) m->resident_collecting = 0;  /* mode 1: stop after prefill */
            /* mode 2: keep collecting through decode for incremental pin */
        }
    }
    return logit;
}

static void pilot_realload(Model *m, int layer, int eid) {
    LCache *lc = &m->cache[layer]; Cfg *c = &m->c;
    pthread_mutex_lock(&g_pilot_mx);
    /* Q event under the cache lock: shares the total order with R/C/P */
    if (!m->is_queued[layer * c->n_experts + eid]) { pthread_mutex_unlock(&g_pilot_mx); return; }
    for (int i = 0; i < lc->n; i++) if (lc->slots[i].eid == eid) { m->is_queued[layer*c->n_experts+eid]=0; pthread_mutex_unlock(&g_pilot_mx); return; }
    /* COALESCE: an admission (demand or pilot) is already loading this expert —
     * skip; its publish will make it resident. Never launch a duplicate. */
    if (lc->loading[eid] >= 0) {
        m->is_queued[layer*c->n_experts+eid] = 0;
        g_pilot_coalesce_skips++;
        pthread_mutex_unlock(&g_pilot_mx);
        return;
    }
    Slot *s;
    int64_t victim_eid = -1;
    if (lc->n < lc->cap) { s = &lc->slots[lc->n++]; slot_ensure_allocated(m, s); }
    else {
        int lru = -1;
        for (int i = 0; i < lc->n; i++) { if (lc->slots[i].pinned || lc->slots[i].eid < 0) continue; if (lru < 0 || lc->slots[i].used < lc->slots[lru].used) lru = i; }
        if (lru < 0) { m->is_queued[layer*c->n_experts+eid]=0; pthread_mutex_unlock(&g_pilot_mx); return; }
        s = &lc->slots[lru]; s->pinned = 0;
        victim_eid = s->eid;   /* evicted expert id (trace) */
        trace_emit("PILOT", "EVICT", -1, layer, victim_eid,
                   s->is_int3 ? 3 : (s->is_int4 ? 4 : 8), 0, 0.0, -1, lru);
    }
    s->eid = -1; s->used = ++m->clock;
    lc->loading[eid] = (int16_t)(s - lc->slots);   /* reserve identity */
    pthread_mutex_unlock(&g_pilot_mx);
    ExpertLoadResult res;
    load_expert_merged(m, layer, eid, s, 1, &res);
    pthread_mutex_lock(&g_pilot_mx);
    s->eid = eid; s->pinned = m->is_pinned[layer*c->n_experts+eid]; s->used = ++m->clock;
    m->is_queued[layer*c->n_experts+eid] = 0;
    lc->loading[eid] = -1;   /* publish */
    trace_emit("PILOT", "INSERT", -1, layer, eid, res.fmt, res.bytes, res.ms, victim_eid,
               (int)(s - lc->slots));
    pthread_mutex_unlock(&g_pilot_mx);
}

static void *pilot_worker(void *arg) {
    (void)arg;
    while (1) {
        unsigned r = __atomic_load_n(&pilot_r, __ATOMIC_ACQUIRE);
        unsigned w = __atomic_load_n(&pilot_w, __ATOMIC_ACQUIRE);
        if (r == w) { sleep_ms(1); continue; }
        int layer = pilot_q[r & 4095].l, eid = pilot_q[r & 4095].e;
        pilot_realload(pilot_m, layer, eid);
        __atomic_store_n(&pilot_r, r + 1, __ATOMIC_RELEASE);
    }
    return NULL;
}

static void pilot_prefetch(Model *m, int lnext, const float *x, int S) {
    if (lnext < 0 || lnext >= m->c.n_layers) return;
    Cfg *c = &m->c; int D = c->hidden, E = c->n_experts;
    ensure_pilot_worker_started(m);
    float *logits = falloc((int64_t)S * E);
    Layer *l = &m->L[lnext];
    float *nrm_x = falloc((int64_t)S * D);
    for (int s = 0; s < S; s++) rmsnorm_row(nrm_x + (int64_t)s*D, x + (int64_t)s*D, l->post_ln, D, c->eps);
    matmul_d(logits, nrm_x, l->gate, S, D, E);   /* int8 copy (f32 may be freed) */
    free(nrm_x);
    for (int s = 0; s < S; s++) {
        float *pr = logits + (int64_t)s*E;
        float *blended = pr;
        float *ema = m->momentum_logits + (int64_t)lnext*E;
        if (m->pilot_smooth > 0.f) {
            blended = falloc(E); int is_zero = 1;
            for (int e = 0; e < E; e++) if (ema[e] != 0.f) { is_zero = 0; break; }
            if (is_zero) { for (int e = 0; e < E; e++) { ema[e] = pr[e]; blended[e] = pr[e]; } }
            else { for (int e = 0; e < E; e++) { blended[e] = (1.f-m->pilot_smooth)*pr[e] + m->pilot_smooth*ema[e]; ema[e] = blended[e]; } }
        }
        int cand = 0; int idx[128]; int score_rank[128]; float cand_conf[128];
        float max_logit = -1e30f; for (int e = 0; e < E; e++) if (blended[e] > max_logit) max_logit = blended[e];
        float *exps = falloc(E); float sum_exps = 0.f;
        for (int e = 0; e < E; e++) { exps[e] = expf(blended[e] - max_logit); sum_exps += exps[e]; }
        float cum_sum = 0.f; int min_cand = c->topk; int max_cand = c->topk * g_wide;
        if (max_cand < min_cand) max_cand = min_cand; if (max_cand > 128) max_cand = 128; if (max_cand > E) max_cand = E;
        for (int kk = 0; kk < max_cand; kk++) {
            int best = -1; float bv = -1.f;
            for (int e = 0; e < E; e++) { int taken = 0; for (int j = 0; j < kk; j++) if (idx[j]==e){taken=1;break;} if (!taken && exps[e] > bv) { bv = exps[e]; best = e; } }
            if (best < 0) break;
            idx[kk] = best; score_rank[kk] = kk; cand_conf[kk] = sum_exps > 0.f ? exps[best] / sum_exps : 0.f;
            cum_sum += bv; cand++;
            if (cum_sum >= m->pilot_conf_limit * sum_exps && cand >= min_cand) break;
        }
        free(exps);   /* cand_conf[] captured above — no reads of exps past this point */
        if (blended != pr) free(blended);
        /* sort candidates by eid for deterministic enqueue order, carrying
         * their ORIGINAL confidence rank with them (Defect D) */
        for (int a = 0; a < cand-1; a++) for (int b = a+1; b < cand; b++)
            if (idx[b] >= 0 && (idx[a] < 0 || idx[a] > idx[b])) {
                int t = idx[a]; idx[a] = idx[b]; idx[b] = t;
                t = score_rank[a]; score_rank[a] = score_rank[b]; score_rank[b] = t;
                float tf = cand_conf[a]; cand_conf[a] = cand_conf[b]; cand_conf[b] = tf;
            }
        for (int kk = 0; kk < cand; kk++) {
            int eid = idx[kk]; if (eid < 0) continue;
            /* v4 INTENT event: emitted UNCONDITIONALLY before residency /
             * queue-gating / ring-capacity decisions, so alternative policies
             * can recover prefetch opportunities the baseline suppressed.
             * R3: intent + decisions share ONE g_pilot_mx critical section —
             * the C event is mutex-ordered like every other v4 event. */
            pthread_mutex_lock(&g_pilot_mx);
            LCache *lc = &m->cache[lnext];
            if (g_req_fp) { unsigned long long s_ = ++g_req_seq;
                fprintf(g_req_fp, "%llu C PC %lld %d %d %d %d %.6f %d %lld\n", s_,
                        (long long)g_trace_tok, lnext, eid,
                        score_rank[kk], kk, (double)cand_conf[kk],
                        (g_req_meta_ready ? g_emeta_fmt[lnext][eid] : -1),
                        (long long)(g_req_meta_ready ? g_emeta_bytes[lnext][eid] : -1)); }
            int found = 0;
            for (int z = 0; z < lc->n; z++) if (lc->slots[z].eid == eid) { found = 1; break; }
            if (!found) {
                int gidx = lnext*E + eid;
                int already_queued = m->is_queued[gidx];
                unsigned w2 = __atomic_load_n(&pilot_w, __ATOMIC_RELAXED);
                unsigned r2 = __atomic_load_n(&pilot_r, __ATOMIC_ACQUIRE);
                if (!already_queued && w2 - r2 < 4096) {
                    pilot_q[w2 & 4095].l = lnext; pilot_q[w2 & 4095].e = eid;
                    __atomic_store_n(&pilot_w, w2 + 1, __ATOMIC_RELEASE);
                    m->is_queued[gidx] = 1;
                }
            }
            pthread_mutex_unlock(&g_pilot_mx);
        }
    }
    free(logits);
}

/* When DUMP=<path> is set, generate() copies the last-token logits here so main()
 * can write them to <path> (raw float32, length = vocab). Lets a torch-free
 * cosine comparison against tools/_ref_dn.py's numpy logits validate the port. */
static float *g_last_logit = NULL;

/* Zero the DeltaNet recurrent state so a new request doesn't inherit the
 * previous conversation's hidden state. Must be called at the start of every
 * generation (the CLI runs once, so this is also correct there). */
static void reset_recurrent(Model *m){
    Cfg *c = &m->c;
    for (int i = 0; i < c->n_layers; i++){
        if (c->is_attn[i]) continue;
        if (m->DN_rec[i])  memset(m->DN_rec[i],  0, (size_t)c->dn_vheads * c->dn_kdim * c->dn_vdim * sizeof(float));
        if (m->DN_conv[i]) memset(m->DN_conv[i], 0, (size_t)c->dn_conv_dim * (c->dn_convk - 1) * sizeof(float));
    }
}

/* Allocate (once) or reuse the KV cache across requests. Grows only when a
 * longer context is needed; never shrinks. Frees the previous buffers on
 * growth so the server doesn't leak KV memory across requests. */
static void ensure_kv(Model *m){
    Cfg *c = &m->c;
    if (m->kv_cap >= m->max_t && m->K) return;
    if (m->K){
        for (int i = 0; i < c->n_layers; i++){ if (m->K[i]) free(m->K[i]); if (m->V[i]) free(m->V[i]); }
        free(m->K); free(m->V); m->K = NULL; m->V = NULL;
    }
    m->K = calloc(c->n_layers, sizeof(float*)); m->V = calloc(c->n_layers, sizeof(float*));
    for (int i = 0; i < c->n_layers; i++){
        if (c->is_attn[i]){
            m->K[i] = falloc((int64_t)c->kv_heads * m->max_t * c->k_head_dim);
            m->V[i] = falloc((int64_t)c->kv_heads * m->max_t * c->k_head_dim);
        } else { m->K[i] = NULL; m->V[i] = NULL; }
    }
    /* Attention scores: one row per thread, indexed by absolute position, so
     * each row must hold max_t entries. Sized here rather than in attention()
     * because it grows with the context exactly like the KV cache does, and
     * because a per-call allocation would run 10x per token. */
    free(m->attn_sc);
    m->attn_sc_thr = 1;
#ifdef _OPENMP
    m->attn_sc_thr = omp_get_max_threads();
    if (m->attn_sc_thr < 1) m->attn_sc_thr = 1;
#endif
    m->attn_sc = falloc((int64_t)m->attn_sc_thr * m->max_t);
    m->kv_cap = m->max_t;
}

static void generate(Model *m, const int *prompt, int np, int n_new, int *out) {
    Cfg *c = &m->c;
    /* Same ceiling serve_one() enforces. Past max_position_embeddings the RoPE
     * positions leave the range the model was trained on, so this is a
     * correctness limit, not just a memory one. */
    if (np + n_new > QWEN36_ATTN_MAX_CTX) {
        fprintf(stderr, "[ctx] prompt %d + %d new exceeds the %d-token ceiling\n",
                np, n_new, QWEN36_ATTN_MAX_CTX);
        exit(1);
    }
    m->max_t = np + n_new;
    reset_recurrent(m);
    ensure_kv(m);
    mem_checkpoint("M3", "after recurrent state + KV allocation for context");
    m->kv_len = 0;
    /* HOOK A — fold the PREVIOUS decode stretch (pure decode: last arm point
     * was the previous prefill's end) into the running window sums BEFORE this
     * prompt's prefill can contaminate the deltas. */
    {
        if (g_window_armed) {
            g_win_adm_ms    += g_demand_expert_admission_ms - g_win_adm0;
            g_win_wgt_ms    += g_demand_weight_ms - g_win_wgt0;
            g_win_scl_ms    += g_demand_scale_ms - g_win_scl0;
            g_win_slot_ms   += g_moe_sub[1] - g_win_slot0;
            g_win_lock_ms   += g_eg_lock_wait_ms - g_win_lock0;
            g_win_lookup_ms += g_eg_lookup_ms - g_win_lookup0;
            g_win_victim_ms += g_eg_victim_ms - g_win_victim0;
            g_win_hit_ms    += g_acq_hits - g_win_hits0;
            g_win_miss_ms   += g_acq_miss - g_win_miss0;
        }
        g_window_fold_armed = g_window_armed;
        /* lazy trace open: must precede the FIRST prefill row, so it lives here
         * (every serving path crosses generate(); main()'s opener missed the
         * corpus call site entirely). */
        if (!g_trace_fp) {
            const char *tp = getenv("COLI_MOE_TRACE");
            if (tp && tp[0]) {
                g_trace_fp = fopen(tp, "w");
                if (!g_trace_fp) fprintf(stderr, "[trace] cannot open %s — trace disabled\n", tp);
                else {
                    setvbuf(g_trace_fp, NULL, _IOFBF, 1 << 20);
                    fprintf(g_trace_fp, "# qwen36_moe_trace v3 rows: seq class(DEMAND|PILOT) event(HIT|EVICT|INSERT) tok(-1=prefill/no-ctx) layer eid fmt(3|4|8) bytes adm_ms victim_eid(-1=none) slot\n");
                }
            }
        }
        /* v4 request stream: opened at the same choke point; expert meta and
         * config identity are emitted once, boundary event once per prompt */
        {
            static int g_req_opened = 0;
            if (!g_req_opened) {
                const char *rp = getenv("COLI_TRACE_REQ");
                if (rp && rp[0]) {
                    g_req_fp = fopen(rp, "w");
                    if (!g_req_fp) fprintf(stderr, "[req] cannot open %s — request stream disabled\n", rp);
                    else {
                        setvbuf(g_req_fp, NULL, _IOFBF, 1 << 20);
                        { unsigned long long s_ = ++g_req_seq;
                          fprintf(g_req_fp, "%llu # qwen36_req_stream v4 cap=%d ep=%d pilot=%s wide=%s omp=%s snap=%s\n",
                                  s_, m->cache ? m->cache[0].cap : 0,
                                  expert_parallel_on(), getenv("PILOT") ? getenv("PILOT") : "0",
                                  getenv("COLI_WIDE") ? getenv("COLI_WIDE") : "-",
                                  getenv("OMP_NUM_THREADS") ? getenv("OMP_NUM_THREADS") : "-",
                                  getenv("SNAP") ? getenv("SNAP") : "-"); }
                        req_emit_expert_meta(m);
                    }
                }
                g_req_opened = 1;
            }
        }
        g_trace_tok = -1;   /* prefill acquisitions trace with tok=-1 */
        if (g_req_fp) {
            /* B is mutex-ordered against all other admission events (Defect F
             * fix: option A — the total-order claim now holds for B too) */
            pthread_mutex_lock(&g_pilot_mx);
            unsigned long long s_ = ++g_req_seq;
            fprintf(g_req_fp, "%llu B GEN %d %d %lld %s\n", s_, np, n_new,
                    (long long)g_trace_tok_base, expert_parallel_on() ? "EP" : "REF");
            pthread_mutex_unlock(&g_pilot_mx);
        }
    }
    for (int i = 0; i < np; i++) out[i] = prompt[i];
    float *logit = step(m, prompt, np, 0);
    /* HOOK B — arm/re-arm the decode window AFTER this prompt's prefill so the
     * baselines exclude every prefill position of every corpus prompt. */
    {
        g_rep_layers = c->n_layers; g_rep_topk = c->topk;
        if (!g_window_armed) g_prefill_acq = g_acq_hits + g_acq_miss;
        g_win_hits0 = g_acq_hits; g_win_miss0 = g_acq_miss;
        g_win_adm0 = g_demand_expert_admission_ms;
        g_win_wgt0 = g_demand_weight_ms; g_win_scl0 = g_demand_scale_ms;
        g_win_slot0 = g_moe_sub[1];
        g_win_lock0 = g_eg_lock_wait_ms; g_win_lookup0 = g_eg_lookup_ms; g_win_victim0 = g_eg_victim_ms;
        g_window_armed = 1; g_window_fold_armed = 1;
    }
    if (g_trace_fp)
        fprintf(g_trace_fp, "# window_begin layers=%d topk=%d prefill_tokens=%d arm=%s\n",
                c->n_layers, c->topk, np, expert_parallel_on() ? "EXPERT_PARALLEL" : "REF_GEMV");
    forge_begin_generation(); /* B2R2/R3.1 */
    int len = np;
    for (int s = 0; s < n_new; s++) {
        g_trace_tok = g_trace_tok_base + s;   /* process-wide decode index; -1 marks prefill-only rows */
        int best = 0; float bv = logit[0];
        for (int i = 1; i < c->vocab; i++) if (logit[i] > bv) { bv = logit[i]; best = i; }
        if (s == 0 && g_ttft < 0) g_ttft = now_s() - g_gen_t0;   /* record TTFT */
        if (g_stream) { stream_token(best); fflush(stdout); }
        if (s == n_new - 1) {
            if (getenv("DUMP")) {
                g_last_logit = malloc((size_t)c->vocab * sizeof(float));
                memcpy(g_last_logit, logit, (size_t)c->vocab * sizeof(float));
            }
            free(logit); out[len++] = best; break;
        }
        free(logit); out[len++] = best;
        int one = best;
        { extern double g_tm_step; double _s0 = tm_on()? tm_now():0;
          logit = step(m, &one, 1, len - 1);
          if (tm_on()) forge_trace_engine_step(_s0); }
    }
    g_trace_tok_base += (int64_t)(len - np);
}

static int tf_nll(Model *m, const int *full, int nfull, int np, double *nll_out) {
    Cfg *c = &m->c;
    if (nfull > QWEN36_ATTN_MAX_CTX) {
        fprintf(stderr, "[ctx] %d tokens exceed the %d-token ceiling\n",
                nfull, QWEN36_ATTN_MAX_CTX);
        exit(1);
    }
    m->max_t = nfull;
    reset_recurrent(m);
    ensure_kv(m);
    m->kv_len = 0;
    double nll = 0; int scored = 0;
    float *logit = step(m, full, np, 0);
    for (int i = np; i < nfull; i++) {
        float mx = logit[0]; for (int v = 1; v < c->vocab; v++) if (logit[v] > mx) mx = logit[v];
        double Z = 0; for (int v = 0; v < c->vocab; v++) Z += exp((double)logit[v] - mx);
        nll += -((double)logit[full[i]] - mx - log(Z));
        scored++;
        free(logit); logit = NULL;
        if (i == nfull - 1) break;
        logit = step(m, &full[i], 1, i);
    }
    if (logit) free(logit);
    *nll_out = nll / scored;
    return scored;
}

static int *read_int_array(jval *o, const char *key, int *n_out) {
    jval *a = json_get(o, key);
    if (!a || a->t != J_ARR) { fprintf(stderr, "ref.json: missing array \"%s\"\n", key); exit(1); }
    int *r = malloc(a->len * sizeof(int));
    for (int i = 0; i < a->len; i++) r[i] = (int)a->kids[i]->num;
    *n_out = a->len; return r;
}

#ifndef QWEN36_NO_MAIN

/* ===================== coli serve mode (SERVE=1) ===================== *
 * Implements the colibri gateway wire protocol so `coli chat` / `coli web` /
 * `coli serve` can drive this engine. Without it the engine is unreachable:
 * users run `coli chat`, not the binary directly.
 * Protocol (matches kimi_k3.c / inkling.c, the other non-GLM engines):
 *   engine:  \x01\x01READY\x01\x01\n
 *            STAT 0 0.00 0.0 <rss>\n
 *   gateway: SUBMIT <id> <slot> <plen> <max_tok> <temp> <top_p>\n <payload bytes>\n
 *   engine:  ACCEPT <id> <np>\n
 *            DATA <id> <n>\n <bytes>\n     (repeated per decoded chunk)
 *            DONE <id> STAT <gen> <tps> <hit%> <rss> <np> <limited>\n
 *   gateway: CANCEL <id>  (abort current turn)
 * Windows: stdout/stdin must go binary BEFORE the READY sentinel or the CRT
 * rewrites the trailing \n as \r\n and the gateway never matches it -> the
 * session hangs forever (#748). compat.h's coli_serve_binary_mode (#749)
 * carries that fix for every engine; see its comment for the full story. */

typedef struct { char id[64]; int max_tok; float temp, top_p; char *payload; int plen; } ServeReq;

static int serve_read_req(ServeReq *q){
    char line[512], cmd[16], id[64];
    if(!fgets(line,sizeof(line),stdin)) return -1;
    if(sscanf(line,"%15s %63s",cmd,id)<2) return 0;
    if(!strcmp(cmd,"CANCEL")||!strcmp(cmd,"STOP")) return 0;
    if(strcmp(cmd,"SUBMIT")) return 0;
    int slot, plen, max_tok; float temp, top_p;
    if(sscanf(line,"%*s %*s %d %d %d %f %f",&slot,&plen,&max_tok,&temp,&top_p)!=5 ||
       plen<0||plen>(1<<24)||max_tok<1){
        printf("ERROR %s bad submit header\n",id); fflush(stdout); return 0;
    }
    (void)slot;
    char *payload=malloc((size_t)plen+1);
    if(!payload){ printf("ERROR %s out of memory\n",id); fflush(stdout); return 0; }
    if(fread(payload,1,(size_t)plen,stdin)!=(size_t)plen){ free(payload); return -1; }
    (void)fgetc(stdin); payload[plen]=0;
    snprintf(q->id,sizeof(q->id),"%s",id);
    q->max_tok=max_tok; q->temp=temp; q->top_p=top_p;
    q->payload=payload; q->plen=plen;
    return 2;
}

static void serve_data(const char *id, const char *p, int n){
    if(n<=0) return;
    printf("DATA %s %d\n",id,n);
    fwrite(p,1,(size_t)n,stdout); fputc('\n',stdout); fflush(stdout);
}

/* temperature + top-p sampler (ported from kimi_k3.c; vocab ~250k -> qsort O(V log V) per token) */
typedef struct { float p; int id; } SampleProb;
static int sample_prob_desc(const void *a, const void *b){
    float pa=((const SampleProb*)a)->p, pb=((const SampleProb*)b)->p;
    return (pb>pa)-(pa>pb);
}
static int serve_sample(const float *lo, int V, float temp, float top_p){
    if(temp<=0.f){ int b=0; for(int i=1;i<V;i++) if(lo[i]>lo[b]) b=i; return b; }
    SampleProb *rank=malloc((size_t)V*sizeof(SampleProb)); float mx=lo[0];
    if(!rank){ fprintf(stderr,"OOM sampling\n"); exit(1); }
    for(int i=1;i<V;i++) if(lo[i]>mx) mx=lo[i];
    double sum=0;
    for(int i=0;i<V;i++){ float p=expf((lo[i]-mx)/temp); sum+=p; rank[i]=(SampleProb){p,i}; }
    qsort(rank,(size_t)V,sizeof(SampleProb),sample_prob_desc);
    double cut=(top_p>0.f&&top_p<1.f)?top_p*sum:sum, kept=0; int n=0;
    while(n<V&&kept<cut) kept+=rank[n++].p;
    double r=((double)rand()/RAND_MAX)*kept, acc=0; int pick=rank[0].id;
    for(int i=0;i<n;i++){ acc+=rank[i].p; if(acc>=r){ pick=rank[i].id; break; } }
    free(rank); return pick;
}

/* Chat turns end on <|im_end|>, base completions on <|endoftext|>. Resolve
 * both ids from the tokenizer's added_tokens: Qwen3.6's 248320-token vocab
 * puts them at 248044+, so the old hardcoded 151645 (the 151k-vocab Qwen id)
 * silently never matched and every serve turn ran into max_tok. Q36_EOS
 * still overrides for experiments. */
static int serve_eos_ids(int *ids, int cap){
    int n=0;
    if(getenv("Q36_EOS")){ ids[n++]=atoi(getenv("Q36_EOS")); return n; }
    for(int k=0;k<g_nspecial && n<cap;k++)
        if(!strcmp(g_sp_str[k],"<|im_end|>")||!strcmp(g_sp_str[k],"<|endoftext|>"))
            ids[n++]=g_sp_id[k];
    if(!n) ids[n++]=151645;   /* tokenizer without added_tokens: old default */
    return n;
}

static void serve_one(Model *m, ServeReq *q){
    int *ids=NULL, np=0;
    encode_text(q->payload, &ids, &np);          /* payload is raw prompt text; qwen36 adds no BOS */
    int max_ctx = qwen36_max_ctx();
    if(np<1 || np+q->max_tok>max_ctx){
        printf("ERROR %s CONTEXT_EXCEEDED prompt_tokens=%d requested=%d capacity=%d\n",q->id,np,q->max_tok,max_ctx);
        fflush(stdout); free(ids); return;
    }
    printf("ACCEPT %s %d\n",q->id,np); fflush(stdout);
    m->max_t = np + q->max_tok;
    reset_recurrent(m); ensure_kv(m); m->kv_len = 0;
    /* Per-REQUEST state, not per-process: without this the server keeps the
     * first request's prefill flag and expert-collection set forever, so
     * COLIBRI_RESIDENT=1 collects on request #1 and never again, and the
     * router EMA carries one conversation's history into the next. */
    m->first_step = 1;
    if (m->seen) memset(m->seen, 0, (size_t)m->c.n_layers * m->c.n_experts);
    if (m->momentum_logits)
        memset(m->momentum_logits, 0,
               (size_t)m->c.n_layers * m->c.n_experts * sizeof(float));
    float *lo = step(m, ids, np, 0);
    int gen=0, limited=1;
    int eos_ids[4]; int n_eos=serve_eos_ids(eos_ids,4);
    double t0=now_s();
    unsigned char sbuf[16]; int sbn=0;
    for(int s=0;s<q->max_tok;s++){
        int tk = serve_sample(lo, m->c.vocab, q->temp, q->top_p);
        free(lo); lo=NULL;
        int is_eos=0; for(int e=0;e<n_eos;e++) if(tk==eos_ids[e]) is_eos=1;
        if(is_eos){ limited=0; break; }
        unsigned char tmp[256]; int tn=0; decode_id_to_bytes(tk, tmp, &tn);
        unsigned char chunk[256]; int cn=0; utf8_drain(sbuf,&sbn,tmp,tn,chunk,&cn);
        if(cn>0) serve_data(q->id,(char*)chunk,cn);
        gen++;
        lo = step(m, &tk, 1, np+s);
    }
    if(sbn>0) serve_data(q->id,(char*)sbuf,sbn);   /* flush trailing partial UTF-8 */
    free(lo); free(ids);
    double dt=now_s()-t0;
    printf("DONE %s STAT %d %.3f %.1f %.2f %d %d\n",q->id,gen,
           dt>0?gen/dt:0.0,0.0,rss_gb(),np,limited);
    fflush(stdout);
}

static void serve_loop(Model *m){
    coli_serve_binary_mode();
    setvbuf(stdin,NULL,_IONBF,0);
    fputs("\x01\x01READY\x01\x01\n",stdout);
    printf("STAT 0 0.00 0.0 %.2f\n",rss_gb());
    fflush(stdout);
    for(;;){
        ServeReq q={0}; int r;
        do r=serve_read_req(&q); while(r==0);
        if(r<0) return;
        if(r==2){ serve_one(m,&q); free(q.payload); }
    }
}


typedef struct { int eid; uint32_t count; } ExpertCountStat;
typedef struct { int eid; double mass; } ExpertMassStat;

static int cmp_stat_count(const void *a, const void *b) {
    const ExpertCountStat *sa = (const ExpertCountStat *)a;
    const ExpertCountStat *sb = (const ExpertCountStat *)b;
    return (sb->count > sa->count) ? 1 : (sb->count < sa->count) ? -1 : 0;
}
static int cmp_stat_mass(const void *a, const void *b) {
    const ExpertMassStat *sa = (const ExpertMassStat *)a;
    const ExpertMassStat *sb = (const ExpertMassStat *)b;
    return (sb->mass > sa->mass) ? 1 : (sb->mass < sa->mass) ? -1 : 0;
}

static void dump_routing_census(Model *m, const char *out_path) {
    if (!out_path || !*out_path) return;
    FILE *f = fopen(out_path, "w");
    if (!f) { perror(out_path); return; }
    Cfg *c = &m->c;
    int L = c->n_layers;
    int E = c->n_experts;

    uint64_t total_selections_all = 0;
    double total_mass_all = 0.0;

    fprintf(f, "{\n");
    fprintf(f, "  \"n_layers\": %d,\n", L);
    fprintf(f, "  \"n_experts_per_layer\": %d,\n", E);
    fprintf(f, "  \"layers\": [\n");

    for (int l = 0; l < L; l++) {
        uint32_t *freq_l = m->freq + (int64_t)l * E;
        double *mass_l = m->router_mass ? m->router_mass + (int64_t)l * E : NULL;
        uint64_t l_tot_count = 0;
        double l_tot_mass = 0.0;
        for (int e = 0; e < E; e++) {
            l_tot_count += freq_l[e];
            if (mass_l) l_tot_mass += mass_l[e];
        }
        total_selections_all += l_tot_count;
        total_mass_all += l_tot_mass;

        ExpertCountStat c_stats[1024];
        ExpertMassStat m_stats[1024];
        for (int e = 0; e < E; e++) {
            c_stats[e].eid = e; c_stats[e].count = freq_l[e];
            m_stats[e].eid = e; m_stats[e].mass = mass_l ? mass_l[e] : 0.0;
        }
        qsort(c_stats, E, sizeof(ExpertCountStat), cmp_stat_count);
        qsort(m_stats, E, sizeof(ExpertMassStat), cmp_stat_mass);

        int count_90 = 0, count_95 = 0, count_99 = 0, count_999 = 0;
        int mass_90 = 0, mass_95 = 0, mass_99 = 0, mass_999 = 0;
        uint64_t cum_c = 0;
        double cum_m = 0.0;

        int never_sel = 0, lt_001 = 0, lt_01 = 0, lt_1 = 0;
        double gini_num = 0.0;
        double entropy = 0.0;

        for (int i = 0; i < E; i++) {
            cum_c += c_stats[i].count;
            if (l_tot_count > 0) {
                if (cum_c >= 0.90 * l_tot_count && count_90 == 0) count_90 = i + 1;
                if (cum_c >= 0.95 * l_tot_count && count_95 == 0) count_95 = i + 1;
                if (cum_c >= 0.99 * l_tot_count && count_99 == 0) count_99 = i + 1;
                if (cum_c >= 0.999 * l_tot_count && count_999 == 0) count_999 = i + 1;

                double frac = (double)c_stats[i].count / l_tot_count;
                if (c_stats[i].count == 0) never_sel++;
                if (frac < 0.0001) lt_001++;
                if (frac < 0.001) lt_01++;
                if (frac < 0.01) lt_1++;
                if (c_stats[i].count > 0) {
                    entropy -= frac * (log(frac) / log(2.0));
                }
            } else {
                never_sel++;
            }
        }
        for (int i = 0; i < E; i++) {
            cum_m += m_stats[i].mass;
            if (l_tot_mass > 0) {
                if (cum_m >= 0.90 * l_tot_mass && mass_90 == 0) mass_90 = i + 1;
                if (cum_m >= 0.95 * l_tot_mass && mass_95 == 0) mass_95 = i + 1;
                if (cum_m >= 0.99 * l_tot_mass && mass_99 == 0) mass_99 = i + 1;
                if (cum_m >= 0.999 * l_tot_mass && mass_999 == 0) mass_999 = i + 1;
            }
        }
        if (count_90 == 0) count_90 = E;
        if (count_95 == 0) count_95 = E;
        if (count_99 == 0) count_99 = E;
        if (count_999 == 0) count_999 = E;
        if (mass_90 == 0) mass_90 = E;
        if (mass_95 == 0) mass_95 = E;
        if (mass_99 == 0) mass_99 = E;
        if (mass_999 == 0) mass_999 = E;

        if (l_tot_count > 0) {
            double cum_gini = 0;
            for (int i = 0; i < E; i++) {
                cum_gini += (i + 1) * (double)c_stats[E - 1 - i].count;
            }
            gini_num = (2.0 * cum_gini) / ((double)E * l_tot_count) - ((double)(E + 1) / (double)E);
        }

        fprintf(f, "    {\n");
        fprintf(f, "      \"layer\": %d,\n", l);
        fprintf(f, "      \"total_selections\": %llu,\n", (unsigned long long)l_tot_count);
        fprintf(f, "      \"total_router_mass\": %.4f,\n", l_tot_mass);
        fprintf(f, "      \"experts_for_90pct_selections\": %d,\n", count_90);
        fprintf(f, "      \"experts_for_95pct_selections\": %d,\n", count_95);
        fprintf(f, "      \"experts_for_99pct_selections\": %d,\n", count_99);
        fprintf(f, "      \"experts_for_99_9pct_selections\": %d,\n", count_999);
        fprintf(f, "      \"experts_for_90pct_mass\": %d,\n", mass_90);
        fprintf(f, "      \"experts_for_95pct_mass\": %d,\n", mass_95);
        fprintf(f, "      \"experts_for_99pct_mass\": %d,\n", mass_99);
        fprintf(f, "      \"experts_for_99_9pct_mass\": %d,\n", mass_999);
        fprintf(f, "      \"never_selected_count\": %d,\n", never_sel);
        fprintf(f, "      \"selected_lt_0_01pct_count\": %d,\n", lt_001);
        fprintf(f, "      \"selected_lt_0_1pct_count\": %d,\n", lt_01);
        fprintf(f, "      \"selected_lt_1pct_count\": %d,\n", lt_1);
        fprintf(f, "      \"gini\": %.4f,\n", gini_num);
        fprintf(f, "      \"entropy\": %.4f,\n", entropy);
        fprintf(f, "      \"top16_experts\": [");
        for (int i = 0; i < 16 && i < E; i++) {
            fprintf(f, "{\"eid\": %d, \"count\": %u, \"mass\": %.4f}%s",
                    c_stats[i].eid, c_stats[i].count, mass_l ? mass_l[c_stats[i].eid] : 0.0,
                    (i == 15 || i == E - 1) ? "" : ", ");
        }
        fprintf(f, "],\n");
        fprintf(f, "      \"expert_counts\": [");
        for (int e = 0; e < E; e++) {
            fprintf(f, "%u%s", freq_l[e], e == E - 1 ? "" : ",");
        }
        fprintf(f, "],\n");
        fprintf(f, "      \"expert_mass\": [");
        for (int e = 0; e < E; e++) {
            fprintf(f, "%.4f%s", mass_l ? mass_l[e] : 0.0, e == E - 1 ? "" : ",");
        }
        fprintf(f, "]\n");
        fprintf(f, "    }%s\n", (l == L - 1) ? "" : ",");
    }

    fprintf(f, "  ],\n");
    fprintf(f, "  \"total_selections\": %llu,\n", (unsigned long long)total_selections_all);
    fprintf(f, "  \"total_router_mass\": %.4f\n", total_mass_all);
    fprintf(f, "}\n");
    fclose(f);
    fprintf(stderr, "[census] dumped routing census to %s\n", out_path);
}

static void print_exact_memory_accounting(Model *m) {
    Cfg *c = &m->c;
    int fmt = container_fmt(m);
    const char *fmt_str = (fmt == 5) ? "PACKED INT3-g64" : (fmt == 4 && !unpack_int8_mode()) ? "PACKED INT4" : (fmt == 0) ? "MIXED INT3/INT4-g64" : "UNPACKED INT8";

    uint64_t embed_bytes = m->embed_f16 ?
                           (uint64_t)c->vocab * c->hidden * sizeof(uint16_t) :
                           (uint64_t)c->vocab * c->hidden * sizeof(float);
    uint64_t final_norm_bytes = (uint64_t)c->hidden * sizeof(float);
    uint64_t in_ln_bytes = (uint64_t)c->n_layers * c->hidden * sizeof(float);
    uint64_t post_ln_bytes = (uint64_t)c->n_layers * c->hidden * sizeof(float);
    uint64_t qk_norm_bytes = c->has_qk_norm ? (uint64_t)10 * c->head_dim * sizeof(float) * 2 : 0;
    uint64_t gate_bias_bytes = (uint64_t)c->n_layers * c->n_experts * sizeof(float);
    uint64_t sh_gate_bytes = (uint64_t)c->n_layers * c->hidden * sizeof(float);
    uint64_t dn_dtbias_bytes = (uint64_t)30 * c->dn_vheads * sizeof(float);
    uint64_t dn_alog_bytes = (uint64_t)30 * c->dn_vheads * sizeof(float);
    uint64_t dn_norm_bytes = (uint64_t)30 * c->dn_vdim * sizeof(float);
    uint64_t dn_conv_bytes = (uint64_t)30 * c->dn_conv_dim * c->dn_convk * sizeof(float);

    uint64_t unreg_f32_bytes = final_norm_bytes + in_ln_bytes + post_ln_bytes +
                               qk_norm_bytes + gate_bias_bytes + sh_gate_bytes +
                               dn_dtbias_bytes + dn_alog_bytes + dn_norm_bytes + dn_conv_bytes;

    uint64_t dn_rec_bytes = (uint64_t)30 * c->dn_vheads * c->dn_kdim * c->dn_vdim * sizeof(float);
    uint64_t dn_conv_ring_bytes = (uint64_t)30 * c->dn_conv_dim * (c->dn_convk - 1) * sizeof(float);
    uint64_t dn_state_bytes = dn_rec_bytes + dn_conv_ring_bytes;

    uint64_t kv_cache_bytes = (uint64_t)10 * 2 * c->kv_heads * m->max_t * c->k_head_dim * sizeof(float);
    uint64_t attn_sc_bytes = (uint64_t)m->attn_sc_thr * m->max_t * sizeof(float);

    /* Registered QDW INT8 matrices & scales */
    uint64_t qdw_int8_bytes = 0;
    uint64_t qdw_scale_bytes = 0;
    for (int i = 0; i < g_qdw_n; i++) {
        qdw_int8_bytes += (uint64_t)g_qdw[i].I * g_qdw[i].O;
        qdw_scale_bytes += (uint64_t)g_qdw[i].O * sizeof(float);
    }

    uint64_t fixed_post_quant_bytes = embed_bytes + unreg_f32_bytes + qdw_int8_bytes + qdw_scale_bytes + dn_state_bytes;

    int total_int3_slots = 0;
    int total_int4_slots = 0;
    int total_int8_slots = 0;
    for (int l = 0; l < c->n_layers; l++) {
        for (int s = 0; s < m->cache[l].n; s++) {
            if (m->cache[l].slots[s].is_int3) total_int3_slots++;
            else if (m->cache[l].slots[s].is_int4) total_int4_slots++;
            else total_int8_slots++;
        }
    }
    uint64_t total_allocated_slots = total_int3_slots + total_int4_slots + total_int8_slots;
    uint64_t allocated_expert_bytes = 0; for (int l = 0; l < c->n_layers; l++) for (int s = 0; s < m->cache[l].n; s++) allocated_expert_bytes += m->cache[l].slots[s].allocated_weight_bytes; int64_t _want_s = 2 * scale_count_gu(&m->c) + scale_count_d(&m->c); allocated_expert_bytes += (uint64_t)_want_s * 4 * (total_int3_slots + total_int4_slots);

    fprintf(stderr, "\n=======================================================\n");
    fprintf(stderr, "=== EXACT LIVE MEMORY ACCOUNTING (QWEN3.6-35B-A3B) ===\n");
    fprintf(stderr, "=======================================================\n");
    fprintf(stderr, "1. Embedding (%s, unregistered):    %12llu bytes (%7.3f MiB / %6.3f GiB)\n",
            m->embed_f16 ? "FP16" : "FP32", (unsigned long long)embed_bytes, (double)embed_bytes/1048576.0, (double)embed_bytes/1073741824.0);
    fprintf(stderr, "2. Registered Dense INT8 Matrices (%d mat):%12llu bytes (%7.3f MiB / %6.3f GiB)\n",
            g_qdw_n, (unsigned long long)qdw_int8_bytes, (double)qdw_int8_bytes/1048576.0, (double)qdw_int8_bytes/1073741824.0);
    fprintf(stderr, "3. Registered Dense Scales (FP32):         %12llu bytes (%7.3f MiB / %6.3f GiB)\n",
            (unsigned long long)qdw_scale_bytes, (double)qdw_scale_bytes/1048576.0, (double)qdw_scale_bytes/1073741824.0);
    fprintf(stderr, "4. Unregistered FP32 Params (Norms/Conv):  %12llu bytes (%7.3f MiB / %6.3f GiB)\n",
            (unsigned long long)unreg_f32_bytes, (double)unreg_f32_bytes/1048576.0, (double)unreg_f32_bytes/1073741824.0);
    fprintf(stderr, "5. DeltaNet Recurrent & Conv State (30L):  %12llu bytes (%7.3f MiB / %6.3f GiB)\n",
            (unsigned long long)dn_state_bytes, (double)dn_state_bytes/1048576.0, (double)dn_state_bytes/1073741824.0);
    fprintf(stderr, "6. KV Cache & Scratch (ctx=%d):            %12llu bytes (%7.3f MiB / %6.3f GiB)\n",
            m->max_t, (unsigned long long)(kv_cache_bytes + attn_sc_bytes),
            (double)(kv_cache_bytes + attn_sc_bytes)/1048576.0, (double)(kv_cache_bytes + attn_sc_bytes)/1073741824.0);
    fprintf(stderr, "-------------------------------------------------------\n");
    fprintf(stderr, "POST-QUANT FIXED RESIDENT FLOOR:           %12llu bytes (%7.3f MiB / %6.3f GiB)\n",
            (unsigned long long)fixed_post_quant_bytes, (double)fixed_post_quant_bytes/1048576.0, (double)fixed_post_quant_bytes/1073741824.0);
    fprintf(stderr, "ACTIVE EXPERT CACHE FORMAT:                %s\n", fmt_str);
    fprintf(stderr, "ALLOCATED EXPERT SLOTS (%llu slots: %d INT3, %d INT4): %12llu bytes (%7.3f MiB / %6.3f GiB)\n",
            (unsigned long long)total_allocated_slots, total_int3_slots, total_int4_slots,
            (unsigned long long)allocated_expert_bytes,
            (double)allocated_expert_bytes/1048576.0, (double)allocated_expert_bytes/1073741824.0);
    fprintf(stderr, "TOTAL COMPUTED LIVE RESIDENT:              %12llu bytes (%7.3f MiB / %6.3f GiB)\n",
            (unsigned long long)(fixed_post_quant_bytes + allocated_expert_bytes + kv_cache_bytes),
            (double)(fixed_post_quant_bytes + allocated_expert_bytes + kv_cache_bytes)/1048576.0,
            (double)(fixed_post_quant_bytes + allocated_expert_bytes + kv_cache_bytes)/1073741824.0);
    fprintf(stderr, "CURRENT VMRSS MEASURED:                    %12llu bytes (%7.3f MiB / %6.3f GiB)\n",
            (unsigned long long)current_rss_bytes(), (double)current_rss_bytes()/1048576.0, current_rss_gb());
    fprintf(stderr, "PEAK RSS MEASURED:                         %12llu bytes (%7.3f MiB / %6.3f GiB)\n",
            (unsigned long long)peak_rss_bytes(), (double)peak_rss_bytes()/1048576.0, peak_rss_gb());
    fprintf(stderr, "=======================================================\n\n");
}

static void probe_touch_all_cache_slots(Model *m) {
    Cfg *c = &m->c;
    int fmt = container_fmt(m);
    int64_t ng = (int64_t)c->inter * c->hidden, nd = (int64_t)c->hidden * c->inter;
    int64_t want_w = ng + ng + nd;
    int64_t want_s = 2 * scale_count_gu(c) + scale_count_d(c);

    fprintf(stderr, "[PROBE] Pre-allocating and touching all %d slots (%d layers x %d cap, fmt=%d)...\n",
            c->n_layers * m->cache[0].cap, c->n_layers, m->cache[0].cap, fmt);
    for (int l = 0; l < c->n_layers; l++) {
        for (int s = 0; s < m->cache[l].cap; s++) {
            Slot *slot = &m->cache[l].slots[s];
            slot_ensure_allocated(m, slot);
            if (slot->w3) memset(slot->w3, 0x55, (size_t)((want_w / 64) * 24));
            else if (slot->w4) memset(slot->w4, 0x55, (size_t)(want_w / 2));
            else if (slot->g) memset(slot->g, 1, (size_t)want_w);
            if (slot->gs) memset(slot->gs, 0, (size_t)want_s * sizeof(float));
        }
        m->cache[l].n = m->cache[l].cap;
    }
    mem_checkpoint("PROBE_COMMITTED", "all cache slots allocated and touched in RAM");
}

int main(int argc, char **argv) {
    const char *snap = getenv("SNAP");
    if (!snap) { fprintf(stderr, "set SNAP=<snapshot directory>\n"); return 1; }
    g_pilot = getenv("PILOT") ? atoi(getenv("PILOT")) : 0;
    g_wide  = getenv("WIDE")  ? atoi(getenv("WIDE"))  : 1;
    if (g_wide < 1) g_wide = 1; if (g_wide > 4) g_wide = 4;
    if (getenv("OPENAI")) g_openai = 1;                       /* OpenAI-compatible output */
    const char *mv = getenv("MODEL"); if (mv && *mv) g_model = mv;
    int hot_n = getenv("HOT") ? atoi(getenv("HOT")) : 0;
    int cap   = argc > 1 ? atoi(argv[1]) : 16;
    const char *env_cgb = getenv("COLI_EXPERT_CACHE_GB");
    if (env_cgb && *env_cgb) {
        double budget_gb = atof(env_cgb);
        if (budget_gb > 0.1) {
            int total_slots = (int)((budget_gb * 1073741824.0) / 1572864.0);
            int cap_dyn = total_slots / 40;
            if (cap_dyn < 4) cap_dyn = 4;
            if (cap_dyn > 256) cap_dyn = 256;
            cap = cap_dyn;
            fprintf(stderr, "[cache_budget] COLI_EXPERT_CACHE_GB=%.2f GB -> cap=%d/layer (%d total slots)\n",
                    budget_gb, cap, cap * 40);
        }
    }
    int bits  = argc > 2 ? atoi(argv[2]) : 4;
    /* cap < 1 leaves every layer cache empty, so expert_get finds no slot to
     * evict and waits for a publish that can never come. The old lru=0 fallback
     * turned that into a heap OOB instead; neither is a failure mode to ship. */
    if (cap < 1) { fprintf(stderr, "cache/layer must be >= 1 (got %d)\n", cap); return 1; }
    if (bits < 2 || bits > 8) { fprintf(stderr, "quant_bits must be 2..8 (got %d)\n", bits); return 1; }
    const char *refpath = argc > 3 ? argv[3] : "ref.json";

    float smooth = getenv("SMOOTH") ? (float)atof(getenv("SMOOTH")) : 0.3f;
    float conf   = getenv("CONF_LIMIT") ? (float)atof(getenv("CONF_LIMIT")) : 0.92f;

    fprintf(stderr, "== qwen36 Phase-2 engine | cache=%d/layer bits=%d pilot=%d wide=%d hot=%d smooth=%.2f conf=%.2f ==\n",
           cap, bits, g_pilot, g_wide, hot_n, smooth, conf);

#if defined(__AVX512F__) && defined(__AVX512BW__)
    if (!i3_avx512_selftest()) {
        fprintf(stderr, "FATAL: AVX-512 INT3 selftest failed\n");
        return 1;
    }
#endif


    int is_ref = 0; jval *ref = NULL;
    int rplen = (int)strlen(refpath);
    if (rplen>=5 && strcmp(refpath+rplen-5, ".json")==0) is_ref = 1;

    int *prompt=NULL, *full=NULL, *out=NULL;
    int np=0, nfull=0, n_new=0;
    char *buf=NULL, *arena=NULL;
    /* serve mode gets its prompts over the wire: skip the argv prompt file
     * entirely, or the default "ref.json" kills the engine before serve_loop
     * is ever reached — which is exactly how `coli` launches it (SERVE=1, no
     * prompt argument). */
    int serve_mode = getenv("SERVE") && getenv("SERVE")[0]=='1';

    /* load tokenizer early so text-prompt mode can encode before model_init */
    {
        const char *tokpath = getenv("TOK");
        if (tokpath && *tokpath) load_tokenizer(tokpath);
        else if (argc > 4 && argv[4] && *argv[4]) load_tokenizer(argv[4]);
        else { char tpb[2048]; snprintf(tpb,sizeof tpb,"%s/tokenizer.json",snap); load_tokenizer(tpb); }
    }

    if (serve_mode) {
        /* no argv prompt to load */
    } else if (is_ref) {
        FILE *f = fopen(refpath, "rb"); if (!f) { perror(refpath); return 1; }
        fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
        buf=malloc(n+1); if (fread(buf,1,n,f)!=(size_t)n) {} buf[n]=0; fclose(f);
        ref = json_parse(buf, &arena);
        if (json_get(ref, "samples")) { np = 0; nfull = 0; n_new = 0; } else { prompt = read_int_array(ref, "prompt_ids", &np); full = read_int_array(ref, "full_ids", &nfull); n_new = nfull - np; } //
        //
        //
    } else {
        /* text-prompt mode: read file as raw text, encode in C */
        FILE *f = fopen(refpath, "rb"); if (!f) { perror(refpath); return 1; }
        fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
        char *txt=malloc(n+1); if (fread(txt,1,n,f)!=(size_t)n) {} txt[n]=0; fclose(f);
        if (!g_tok) { fprintf(stderr, "[enc] no tokenizer loaded; cannot encode text. Put tokenizer.json in SNAP or set TOK.\n"); free(txt); return 1; }
        encode_text(txt, &prompt, &np);
        free(txt);
        n_new = getenv("N_NEW") ? atoi(getenv("N_NEW")) : 64;
        if (n_new < 1) n_new = 1;
        fprintf(stderr, "[enc] prompt tokens: %d | generating %d new tokens\n", np, n_new);
        if (getenv("ENC_DEBUG") && np <= 300) { fprintf(stderr, "[enc] prompt ids: "); for (int i=0;i<np;i++) fprintf(stderr, "%d ", prompt[i]); fprintf(stderr, "\n"); }
    }

    mem_checkpoint("M0", "immediately before model load");
    Model m; model_init(&m, snap, cap, bits);
    mem_checkpoint("M1", "after FP32 model/dense load (before QDW quantization)");
    g_expert_gs = m.c.expert_gs;
    if (g_expert_gs) fprintf(stderr, "[qwen36] group-scaled experts: gs=%d\n", g_expert_gs);
    fprintf(stderr, "resident weights loaded in %.1fs | RSS after load: %.2f GB\n", m.dense_load_s, rss_gb());
    /* quantize the large dense matrices to int8 (COLI_DENSE_I8=0 disables) */
    if (dense_i8_on()) {
        double tq = now_s();
        Cfg *qc = &m.c; int D2 = qc->hidden;
        int q_out = qc->q_heads * qc->q_head_dim, kv_out = qc->kv_heads * qc->k_head_dim;
        for (int i = 0; i < qc->n_layers; i++) {
            Layer *l = &m.L[i];
            qdw_register(l->q, D2, q_out); qdw_register(l->k, D2, kv_out);
            qdw_register(l->v, D2, kv_out); qdw_register(l->o, qc->o_in, D2);
            qdw_register(l->gate, D2, qc->n_experts);
            qdw_register(l->sh_g, D2, qc->shared_inter); qdw_register(l->sh_u, D2, qc->shared_inter);
            qdw_register(l->sh_d, qc->shared_inter, D2);
            qdw_register(l->dn_qkv, D2, qc->dn_conv_dim);
            qdw_register(l->dn_z, D2, qc->dn_vheads * qc->dn_vdim);
            qdw_register(l->dn_out, qc->dn_vheads * qc->dn_vdim, D2);
        }
        qdw_register(m.lm_head, D2, qc->vocab);
        /* Free the f32 originals -- the pointers only serve as lookup keys in
         * matmul_d from here on (never dereferenced again).
         * COLI_KEEP_F32=1 keeps them (debug). */
        double freed = 0;
        if (!getenv("COLI_KEEP_F32")) {
            for (int i = 0; i < g_qdw_n; i++) {
                freed += (double)g_qdw[i].I * g_qdw[i].O * sizeof(float);
                free((void*)g_qdw[i].w);
            }
        }
        fprintf(stderr, "[dense-i8] %d matrices quantized in %.1f s, %.1f GB f32 freed\n",
                g_qdw_n, now_s()-tq, freed/1073741824.0);
    }
    mem_checkpoint("M2", "after QDW INT8 quantization & freeing FP32 originals");

    if (getenv("COLIBRI_PROBE_TOUCH_SLOTS") && atoi(getenv("COLIBRI_PROBE_TOUCH_SLOTS")) == 1) {
        probe_touch_all_cache_slots(&m);
    }

    /* coli serve mode: speak the gateway wire protocol instead of argv generation */
    if (getenv("SERVE") && getenv("SERVE")[0] == '1') {
        if (!g_tok) { fprintf(stderr, "[serve] tokenizer.json required (put in SNAP or set TOK)\n"); return 1; }
        serve_loop(&m);
        return 0;
    }

    const char *corpus_path = getenv("CORPUS_FILE");
    if (corpus_path && *corpus_path) {
        FILE *cf = fopen(corpus_path, "rb");
        if (!cf) { perror(corpus_path); return 1; }
        fseek(cf, 0, SEEK_END); long cflen = ftell(cf); fseek(cf, 0, SEEK_SET);
        char *cbuf = malloc(cflen + 1);
        if (fread(cbuf, 1, cflen, cf) != (size_t)cflen) {}
        cbuf[cflen] = 0; fclose(cf);

        char *cursor = cbuf;
        int prompt_idx = 0;
        int per_prompt_tokens = getenv("N_NEW") ? atoi(getenv("N_NEW")) : 64;
        if (per_prompt_tokens < 1) per_prompt_tokens = 64;

        while (cursor && *cursor) {
            char *next = strstr(cursor, "===PROMPT===");
            if (next) { *next = 0; }
            while (*cursor == ' ' || *cursor == '\n' || *cursor == '\r' || *cursor == '\t') cursor++;
            if (*cursor) {
                prompt_idx++;
                fprintf(stderr, "\n--- RUNNING CORPUS PROMPT %d ---\n", prompt_idx);
                int *p_ids = NULL; int p_np = 0;
                encode_text(cursor, &p_ids, &p_np);
                fprintf(stderr, "[enc] prompt tokens: %d | generating %d new tokens\n", p_np, per_prompt_tokens);
                int *p_out = malloc((p_np + per_prompt_tokens) * sizeof(int));
                double pt0 = now_s();
                generate(&m, p_ids, p_np, per_prompt_tokens, p_out);
                double pdt = now_s() - pt0;
                fprintf(stderr, "\nPrompt %d generated %d tokens in %.2fs (%.2f tok/s)\n",
                        prompt_idx, per_prompt_tokens, pdt, per_prompt_tokens / pdt);
                free(p_ids); free(p_out);
            }
            if (next) cursor = next + 12;
            else break;
        }
        free(cbuf);
        const char *census_out = getenv("CENSUS_OUT");
        if (census_out && *census_out) dump_routing_census(&m, census_out);
        double tot = m.hits + m.miss;
        if (g_ttft >= 0) fprintf(stderr, "TTFT: %.2f s (time to first token)\n", g_ttft);
        tm_report();
        forge_profile_emit();
        mem_checkpoint("M4", "after expert cache population during inference");
        print_exact_memory_accounting(&m);
        fprintf(stderr, "\nPEAK RSS: %.2f GB | Current VmRSS: %.2f GB\n", peak_rss_gb(), current_rss_gb());
        fprintf(stderr, "Expert cache hit rate: %.1f%% (hit=%llu miss=%llu)\n", tot?100.0*m.hits/tot:0.0,
               (unsigned long long)m.hits, (unsigned long long)m.miss);
        fprintf(stderr, "Coalesce diagnostics: demand waits=%ld pilot skips=%ld | max loaders per (layer,eid)<=1 by construction (single loading registry)\n",
                g_demand_coalesce_waits, g_pilot_coalesce_skips);
        return 0;
    }

    if (is_ref && getenv("PPL") && atoi(getenv("PPL")) == 1) {
        jval *samples = json_get(ref, "samples");
        if (samples && samples->t == J_ARR) {
            printf("\n=== Multi-Domain Held-Out Evaluation (%d Samples) ===\n", samples->len);
            printf("%-24s | %-12s | %-12s | %-10s\n", "Domain", "Prompt Tok", "Scored Tok", "TF-NLL");
            printf("-----------------------------------------------------------------------\n");
            double total_weighted_nll = 0.0;
            int total_scored = 0;
            double t_all0 = now_s();
            for (int i = 0; i < samples->len; i++) {
                jval *samp = samples->kids[i];
                const char *dom = json_get(samp, "domain") ? json_get(samp, "domain")->str : "sample";
                int s_np = 0, s_nfull = 0;
                int *s_p = read_int_array(samp, "prompt_ids", &s_np);
                int *s_f = read_int_array(samp, "full_ids", &s_nfull);
                double s_nll = 0.0;
                int scored = tf_nll(&m, s_f, s_nfull, s_np, &s_nll);
                total_weighted_nll += s_nll * scored;
                total_scored += scored;
                printf("%-24s | %10d | %10d | %10.4f nats\n", dom, s_np, scored, s_nll);
                free(s_p); free(s_f);
            }
            double t_all1 = now_s();
            double dt_all = t_all1 - t_all0;
            double agg_nll = (total_scored > 0) ? (total_weighted_nll / total_scored) : 0.0;
            double tot = m.hits + m.miss;
            printf("-----------------------------------------------------------------------\n");
            printf("AGGREGATE TOKEN-WEIGHTED NLL: %.4f nats/token\n", agg_nll);
            printf("AGGREGATE PERPLEXITY (PPL):   %.2f\n", exp(agg_nll));
            printf("TOTAL SCORED TOKENS:          %d\n", total_scored);
            printf("EVALUATION TIME:              %.2fs (%.2f tok/s)\n", dt_all, total_scored / dt_all);
            printf("PEAK RSS:                     %.2f GB | VmRSS: %.2f GB\n", peak_rss_gb(), current_rss_gb());
            printf("EXPERT CACHE HIT RATE:        %.1f%% (hit=%llu miss=%llu)\n\n",
                   tot ? 100.0 * m.hits / tot : 0.0, (unsigned long long)m.hits, (unsigned long long)m.miss);
            free(buf); free(arena); return 0;
        } else {
            double nll; double t = now_s();
            int scored = tf_nll(&m, full, nfull, np, &nll);
            double dt = now_s() - t;
            double tot = m.hits + m.miss;
            printf("TF-NLL: %.4f nats/token over %d tokens | ppl = %.2f\n", nll, scored, exp(nll));
            printf("Expert cache hit rate: %.1f%% (hit=%llu miss=%llu)\n", tot?100.0*m.hits/tot:0.0,
                   (unsigned long long)m.hits, (unsigned long long)m.miss);
            printf("Speed: %.2f tok/s (%.1fs for %d tokens) | PEAK RSS: %.2f GB\n", scored/dt, dt, scored, rss_gb());
            free(buf); free(arena); return 0;
        }
    }

    out = malloc((np + n_new) * sizeof(int));
    /* timing + OpenAI id setup (before generation) */
    g_ttft = -1; g_gen_t0 = now_s();
    if (g_openai){
        g_oa_created = (long)time(NULL);
        snprintf(g_oa_id, sizeof g_oa_id, "chatcmpl-%ld%04d", g_oa_created, (int)(now_s()*1000) % 10000);
    }
    /* streaming text: emit tokens as they are produced (text mode + tokenizer only) */
    if (!is_ref && g_tok && !getenv("NOSTREAM")) {
        g_stream = 1; g_sbn = 0;
        if (g_openai){
            char jb[320];
            snprintf(jb, sizeof jb,
              "{\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%ld,\"model\":\"%s\","
              "\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}",
              g_oa_id, g_oa_created, g_model);
            sse_chunk(jb);
        } else {
            fprintf(stderr, "Generated (%d new tokens):\nText : ", n_new); fflush(stderr);
        }
    }
    double t = now_s();
    generate(&m, prompt, np, n_new, out);
    double dt = now_s() - t;

    /* DUMP=<path>: write last-token logits (raw float32, vocab) for a torch-free
     * cosine comparison against tools/_ref_dn.py --dump. */
    if (g_last_logit) {
        const char *dp = getenv("DUMP");
        FILE *df = fopen(dp && *dp ? dp : "qwen36_logits.f32", "wb");
        if (df) { fwrite(g_last_logit, sizeof(float), (size_t)m.c.vocab, df); fclose(df);
                  fprintf(stderr, "[dump] wrote %d logits -> %s\n", m.c.vocab, dp && *dp ? dp : "qwen36_logits.f32"); }
        else fprintf(stderr, "[dump] cannot open %s\n", dp ? dp : "qwen36_logits.f32");
    }

    int ref_match = 0;
    if (is_ref) {
        int match = 0;
        printf("\nReference: ");  for (int i=np;i<nfull;i++) printf("%d ", full[i]);
        printf("\nC engine : ");  for (int i=np;i<nfull;i++) { printf("%d ", out[i]); if (out[i]==full[i]) match++; }
        if (g_tok) { printf("Text      : "); print_decoded(out, np, nfull); printf("\n"); }
        printf("\nMatching tokens: %d/%d\n", match, n_new);
        ref_match = match;
    } else {
    if (g_openai) {
        emit_openai_result(out, np, n_new, g_stream);
    } else if (g_stream) {
        stream_flush(); fprintf(stderr, "\n");
    } else {
        fprintf(stderr, "\nGenerated (%d new tokens):\n", n_new);
        if (g_tok) { fprintf(stderr, "Text      : "); print_decoded(out, np, np+n_new); fprintf(stderr, "\n"); }
        else { fprintf(stderr, "Ids       : "); for (int i=np;i<np+n_new;i++) fprintf(stderr, "%d ", out[i]); fprintf(stderr, "\n"); }
    }
    }
    double tot = m.hits + m.miss;
    if (g_ttft >= 0) fprintf(stderr, "TTFT: %.2f s (time to first token)\n", g_ttft);
    tm_report();
    forge_profile_emit();
    mem_checkpoint("M4", "after expert cache population during inference");
    print_exact_memory_accounting(&m);
    fprintf(stderr, "\nPEAK RSS: %.2f GB | Current VmRSS: %.2f GB\n", peak_rss_gb(), current_rss_gb());
    {   /* cumulative (whole process) vs decode-window hit rate: the window line
         * is the A/B-comparable number; the cumulative one mixes prefill. */
        uint64_t w_hit = g_win_hit_ms, w_mis = g_win_miss_ms;
        uint64_t w_tot = w_hit + w_mis;
        fprintf(stderr, "Expert cache hit rate: %.1f%% (hit=%llu miss=%llu, cumulative incl. prefill)\n",
                tot?100.0*m.hits/tot:0.0, (unsigned long long)m.hits, (unsigned long long)m.miss);
        fprintf(stderr, "Expert cache hit rate (decode window): %.1f%% (hit=%llu miss=%llu)\n",
                w_tot?100.0*w_hit/w_tot:100.0, (unsigned long long)w_hit, (unsigned long long)w_mis);
    }
    fprintf(stderr, "Coalesce diagnostics: demand waits=%ld pilot skips=%ld | max loaders per (layer,eid)<=1 by construction (single loading registry)\n",
            g_demand_coalesce_waits, g_pilot_coalesce_skips);
    fprintf(stderr, "Speed: %.2f tok/s (%.1fs for %d tokens)\n", n_new/dt, dt, n_new);
    if (g_trace_fp) { fclose(g_trace_fp); g_trace_fp = NULL; fprintf(stderr, "[trace] closed\n"); }
    free(buf); free(arena);
    /* Oracle mode is a gate, not a report: a mismatch must fail the caller.
     * inkling.c does the same (`return (match == ngen) ? 0 : 1;`) and its CI
     * job relies on it — without this, tools/make_qwen36_oracle.py could be
     * wired into a workflow that stays green through any regression. */
    const char *census_out = getenv("CENSUS_OUT");
    if (census_out && *census_out) dump_routing_census(&m, census_out);
    if (is_ref) return ref_match == n_new ? 0 : 1;
    return 0;
}
#endif /* QWEN36_NO_MAIN */
