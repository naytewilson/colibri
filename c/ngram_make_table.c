/* ngram_make_table.c - materialize a sparse-row table file for N1
 * tomography. Row format: 8-byte LE row id, then 0x3c fill. Every block is
 * really allocated (no holes): hole semantics would falsify cold-read
 * measurement. Verify with stat st_blocks.
 */
#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define ROW_BYTES 320

int main(int argc, char **argv)
{
    const char *out = NULL;
    uint64_t rows = 320001536ull;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--out") && i + 1 < argc) out = argv[++i];
        else if (!strcmp(argv[i], "--rows") && i + 1 < argc) rows = strtoull(argv[++i], NULL, 10);
        else { fprintf(stderr, "usage: ngram_make_table --out FILE [--rows N]\n"); return 2; }
    }
    if (!out) { fprintf(stderr, "missing --out\n"); return 2; }

    int fd = open(out, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) { perror("open"); return 1; }
    size_t chunk_rows = 1u << 20; /* 1 Mi rows = 320 MiB chunk */
    uint8_t *buf = malloc(chunk_rows * ROW_BYTES);
    if (!buf) { fprintf(stderr, "oom\n"); return 1; }

    size_t off = 0;
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (uint64_t base = 0; base < rows; base += chunk_rows) {
        uint64_t n = rows - base < chunk_rows ? rows - base : chunk_rows;
        for (uint64_t r = 0; r < n; r++) {
            uint8_t *p = buf + (size_t)r * ROW_BYTES;
            uint64_t id = base + r;
            memcpy(p, &id, 8);
            memset(p + 8, 0x3c, ROW_BYTES - 8);
        }
        size_t bytes = (size_t)n * ROW_BYTES;
        size_t done = 0;
        while (done < bytes) {
            ssize_t w = write(fd, buf + done, bytes - done);
            if (w <= 0) {
                if (w < 0 && (errno == EINTR)) continue;
                perror("write"); return 1;
            }
            done += (size_t)w;
        }
        off += bytes;
        if (base % (16ull << 20) == 0) {
            fprintf(stderr, "\r%zu MiB / %llu MiB", off >> 20,
                    (unsigned long long)(rows * ROW_BYTES) >> 20);
        }
    }
    fprintf(stderr, "\n");
    if (fsync(fd) != 0) perror("fsync");
    close(fd);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double secs = (double)(t1.tv_sec - t0.tv_sec) + 1e-9 * (double)(t1.tv_nsec - t0.tv_nsec);
    struct stat sb;
    stat(out, &sb);
    printf("rows=%llu apparent_bytes=%zu real_blocks_bytes=%lld wall_s=%.1f mib_s=%.1f\n",
           (unsigned long long)rows, off, (long long)sb.st_blocks * 512,
           secs, (double)off / 1048576.0 / (secs > 0 ? secs : 1));
    return 0;
}
