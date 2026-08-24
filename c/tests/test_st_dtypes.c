/* st.h raw-dtype identity contract (Wave-1 closeout, MISSION B).
 *
 * Before the split, U8/I8/I16/I32/I64 all collapsed to generic code 3: ling3's
 * packed-expert loader "proved" I32 [O,I/8] from nbytes alone, and a float
 * reader's else-branch would have silently reinterpreted any raw tensor
 * through its F16 path. These tests pin the new fail-closed contract:
 *   - every declared safetensors dtype gets its own ST_* code;
 *   - float readers accept ONLY BF16/F16/F32;
 *   - st_read_raw accepts ONLY raw integer dtypes;
 *   - U8 quantized-model behavior is byte-exact unchanged;
 *   - BF16/F16/F32 conversions are unchanged. */
#define _GNU_SOURCE
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifndef _WIN32
#include <sys/wait.h>
#include <unistd.h>
#endif

#include "../st.h"

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #condition); \
        return 1; \
    } \
} while (0)

/* one shard holding one tensor of each dtype we must distinguish */
static void write_snap(const char *dir) {
    char path[512];
    snprintf(path, sizeof(path), "%s/model.safetensors", dir);
    /* payloads laid out back to back; offsets computed below */
    static const float f32v[4] = { 1.0f, -2.5f, 65504.0f, -0.0f };
    static const uint16_t bf16v[3] = { 0x3f80, 0xc020, 0x0000 };   /* 1.0, -2.5, 0 */
    static const uint16_t f16v[2]  = { 0x3c00, 0xc100 };           /* 1.0, -2.5 */
    static const unsigned char u8v[6] = { 1, 2, 3, 250, 254, 255 };
    static const int8_t i8v[4] = { -8, -1, 0, 127 };
    static const int16_t i16v[2] = { -32768, 32767 };
    static const int32_t i32v[3] = { -2147483647 - 1, -1, 123456789 };
    static const int64_t i64v[2] = { 1536, 128 };
    struct { const char *name, *dtype; const void *data; int64_t nbytes, numel; } ts[] = {
        { "t_f32", "F32", f32v, sizeof(f32v), 4 },
        { "t_bf16", "BF16", bf16v, sizeof(bf16v), 3 },
        { "t_f16", "F16", f16v, sizeof(f16v), 2 },
        { "t_u8", "U8", u8v, sizeof(u8v), 6 },
        { "t_i8", "I8", i8v, sizeof(i8v), 4 },
        { "t_i16", "I16", i16v, sizeof(i16v), 2 },
        { "t_i32", "I32", i32v, sizeof(i32v), 3 },
        { "t_i64", "I64", i64v, sizeof(i64v), 2 },
    };
    char hdr[2048]; int64_t off = 0; size_t hlen = 0;
    hlen += (size_t)snprintf(hdr + hlen, sizeof(hdr) - hlen, "{");
    for (int i = 0; i < 8; i++) {
        hlen += (size_t)snprintf(hdr + hlen, sizeof(hdr) - hlen,
            "%s\"%s\":{\"dtype\":\"%s\",\"shape\":[%lld],\"data_offsets\":[%lld,%lld]}",
            i ? "," : "", ts[i].name, ts[i].dtype,
            (long long)ts[i].numel, (long long)off, (long long)(off + ts[i].nbytes));
        off += ts[i].nbytes;
    }
    hlen += (size_t)snprintf(hdr + hlen, sizeof(hdr) - hlen, "}");
    FILE *f = fopen(path, "wb");
    if (!f) { perror("fopen snap"); exit(1); }
    uint64_t hl = hlen;
    fwrite(&hl, 8, 1, f);
    fwrite(hdr, 1, hlen, f);
    for (int i = 0; i < 8; i++) fwrite(ts[i].data, 1, (size_t)ts[i].nbytes, f);
    fclose(f);
}

/* run `fn` in a child with stderr captured; require exit(1) and a message tag */
#ifndef _WIN32
typedef void (*raw_fn)(shards *, const char *, void *, int);

static int expect_exit1(shards *S, const char *tensor, raw_fn fn, const char *tag) {
    int pipefd[2]; CHECK(pipe(pipefd) == 0);
    pid_t pid = fork(); CHECK(pid >= 0);
    if (pid == 0) {
        dup2(pipefd[1], 2); close(pipefd[0]); close(pipefd[1]);
        unsigned char buf[64];
        fn(S, tensor, buf, 0);          /* must exit(1) inside */
        _exit(42);                       /* reaching here = bug */
    }
    close(pipefd[1]);
    char err[512] = {0};
    ssize_t n = read(pipefd[0], err, sizeof(err) - 1); (void)n;
    close(pipefd[0]);
    int status = 0; waitpid(pid, &status, 0);
    if (!(WIFEXITED(status) && WEXITSTATUS(status) == 1)) return 1;
    return strstr(err, tag) == NULL;     /* 0 = message present = pass */
}

static void call_raw(shards *S, const char *n, void *b, int d){ st_read_raw(S, n, b, d); }
static void call_f32(shards *S, const char *n, void *b, int d){ st_read_f32(S, n, (float *)b, d); }
#endif

int main(void) {
    /* 1) code identity + helpers */
    CHECK(ST_BF16 == 0 && ST_F16 == 1 && ST_F32 == 2 && ST_U8 == 3);   /* historical codes frozen */
    CHECK(st_dtype_esz(ST_BF16) == 2 && st_dtype_esz(ST_F16) == 2 && st_dtype_esz(ST_F32) == 4);
    CHECK(st_dtype_esz(ST_U8) == 1 && st_dtype_esz(ST_I8) == 1 && st_dtype_esz(ST_I16) == 2 &&
          st_dtype_esz(ST_I32) == 4 && st_dtype_esz(ST_I64) == 8);
    CHECK(st_is_float_dtype(ST_BF16) && st_is_float_dtype(ST_F16) && st_is_float_dtype(ST_F32));
    CHECK(!st_is_float_dtype(ST_U8) && !st_is_float_dtype(ST_I8) && !st_is_float_dtype(ST_I16) &&
          !st_is_float_dtype(ST_I32) && !st_is_float_dtype(ST_I64));

    /* 2) indexing keeps the declared identity per tensor */
    char dir[] = "test_st_dtypes_XXXXXX";
    if (!mkdtemp(dir)) { perror("mkdtemp"); return 1; }
    write_snap(dir);
    shards S; st_init(&S, dir);
    CHECK(st_find(&S, "t_bf16")->dtype == ST_BF16);
    CHECK(st_find(&S, "t_f16")->dtype  == ST_F16);
    CHECK(st_find(&S, "t_f32")->dtype  == ST_F32);
    CHECK(st_find(&S, "t_u8")->dtype   == ST_U8);
    CHECK(st_find(&S, "t_i8")->dtype   == ST_I8);
    CHECK(st_find(&S, "t_i16")->dtype  == ST_I16);
    CHECK(st_find(&S, "t_i32")->dtype  == ST_I32);
    CHECK(st_find(&S, "t_i64")->dtype  == ST_I64);

    /* 3) float readers unchanged on real float dtypes */
    float fv[4];
    CHECK(st_read_f32(&S, "t_f32", fv, 0) == 4);
    CHECK(fv[0] == 1.0f && fv[1] == -2.5f && fv[2] == 65504.0f && fv[3] == -0.0f);
    CHECK(st_read_f32(&S, "t_bf16", fv, 0) == 3);
    CHECK(fv[0] == 1.0f && fv[1] == -2.5f && fv[2] == 0.0f);
    CHECK(st_read_f32(&S, "t_f16", fv, 0) == 2);
    CHECK(fv[0] == 1.0f && fv[1] == -2.5f);

    /* 4) raw readers byte-exact on ALL raw dtypes (U8 legacy + new identities) */
    static const unsigned char u8_expect[6] = { 1, 2, 3, 250, 254, 255 };
    static const int8_t     i8_expect[4]    = { -8, -1, 0, 127 };
    static const int16_t    i16_expect[2]   = { -32768, 32767 };
    static const int32_t    i32_expect[3]   = { -2147483647 - 1, -1, 123456789 };
    static const int64_t    i64_expect[2]   = { 1536, 128 };
    unsigned char rb[32];
    st_read_raw(&S, "t_u8", rb, 0);  CHECK(memcmp(rb, u8_expect, 6) == 0);
    st_read_raw(&S, "t_i8", rb, 0);  CHECK(memcmp(rb, i8_expect, 4) == 0);
    st_read_raw(&S, "t_i16", rb, 0); CHECK(memcmp(rb, i16_expect, 4) == 0);
    st_read_raw(&S, "t_i32", rb, 0); CHECK(memcmp(rb, i32_expect, 12) == 0);
    st_read_raw(&S, "t_i64", rb, 0); CHECK(memcmp(rb, i64_expect, 16) == 0);

#ifndef _WIN32
    /* 5) FAIL-CLOSED: float readers refuse EVERY raw dtype (an I16 tensor can no
     * longer satisfy the old 2-byte width relation and masquerade as F16) */
    CHECK(expect_exit1(&S, "t_u8",  call_f32, "only BF16/F16/F32") == 0);
    CHECK(expect_exit1(&S, "t_i8",  call_f32, "only BF16/F16/F32") == 0);
    CHECK(expect_exit1(&S, "t_i16", call_f32, "only BF16/F16/F32") == 0);
    CHECK(expect_exit1(&S, "t_i32", call_f32, "only BF16/F16/F32") == 0);
    CHECK(expect_exit1(&S, "t_i64", call_f32, "only BF16/F16/F32") == 0);
    /* ...and st_read_raw refuses float tensors (wrong-API bug, caught at read) */
    CHECK(expect_exit1(&S, "t_f32",  call_raw, "only raw integer") == 0);
    CHECK(expect_exit1(&S, "t_bf16", call_raw, "only raw integer") == 0);
    CHECK(expect_exit1(&S, "t_f16",  call_raw, "only raw integer") == 0);
#else
    printf("test_st_dtypes: fork subtests skipped on Windows\n");
#endif

    char cmd[600];
#ifdef _WIN32
    snprintf(cmd, sizeof(cmd), "rmdir /s /q %s", dir);
#else
    snprintf(cmd, sizeof(cmd), "rm -rf %s", dir);
#endif
    if (system(cmd)) {}
    printf("test_st_dtypes: raw-dtype identity + fail-closed readers: ok\n");
    return 0;
}
