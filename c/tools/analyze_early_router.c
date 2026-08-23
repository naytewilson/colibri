/* analyze_early_router.c — Wave A/B/K offline analyzer.
 *
 * Input: COLI_LEADFILE TSV from the instrumented runtime:
 *   T <tok> <t0>
 *   K <tok> <layer> <t>                    post-mixer anchor
 *   G <tok> <layer> <t>                    post-moe anchor
 *   R <tok> <layer> <t>                    authoritative router done (T_NEED)
 *   A <tok> <layer> e0..e7                 actual native top-8
 *   P <tok> <src> <anchor 0|1> <tgt> id:score x<=16
 *   M <tok> <layer> <eid> <io_ms> <t0> <t1>
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAXTOK 4096
#define MAXL   64
#define MAXM   262144

typedef struct { double t; int seen; } TMARK;
typedef struct { int ids[8]; int seen; } ACT;
typedef struct { int ids[16]; double sc[16]; int n; } PRED;
typedef struct { int src; PRED p; } PSRC;
typedef struct { long long tok; int layer, eid; double io, t0, t1; } MISS;

static TMARK r_need[MAXTOK][MAXL], k_anch[2][MAXTOK][MAXL];
static ACT   act[MAXTOK][MAXL];
static PSRC  psrc[MAXTOK][MAXL][2][8];
static int   npsrc[MAXTOK][MAXL][2];

static MISS  miss[MAXM]; static int nmiss = 0;
static int   miss_head[MAXTOK][MAXL], miss_next[MAXM];
static long long maxtok = -1;

static int in_list(const PRED *p, int id, int budget) {
    for (int j = 0; j < p->n && j < budget; j++) if (p->ids[j] == id) return j;
    return -1;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s leadfile.tsv\n", argv[0]); return 2; }
    for (int t = 0; t < MAXTOK; t++) for (int l = 0; l < MAXL; l++) miss_head[t][l] = -1;
    FILE *f = fopen(argv[1], "r");
    if (!f) { perror(argv[1]); return 1; }
    char line[4096];
    while (fgets(line, sizeof line, f)) {
        if (line[0] == '#' || line[0] < 'A' || line[0] > 'Z') continue;
        char op = line[0]; char *p = line + 1;
        long long t; int l, a, g, c;
        double v, v2, v3;
        switch (op) {
        case 'T': if (sscanf(p, "%lld %lf", &t, &v) == 2 && t < MAXTOK && t > maxtok) maxtok = t; break;
        case 'R': if (sscanf(p, "%lld %d %lf", &t, &l, &v) == 3 && t < MAXTOK && l < MAXL) { r_need[t][l].t = v; r_need[t][l].seen = 1; } break;
        case 'K': case 'G':
            if (sscanf(p, "%lld %d %lf", &t, &l, &v) == 3 && t < MAXTOK && l < MAXL) {
                a = (op == 'G'); k_anch[a][t][l].t = v; k_anch[a][t][l].seen = 1;
            } break;
        case 'A': {
            int cons = 0;
            if (sscanf(p, "%lld %d%n", &t, &l, &cons) == 2 && t < MAXTOK && l < MAXL) {
                char *q = p + cons; act[t][l].seen = 1;
                for (int i = 0; i < 8; i++) { while (*q == ' ') q++; act[t][l].ids[i] = (int)strtol(q, &q, 10); }
            } break; }
        case 'P': {
            int cons = 0;
            if (sscanf(p, "%lld %d %d %d%n", &t, &l, &a, &g, &cons) == 4 && t < MAXTOK && g < MAXL && a < 2 && l < MAXL) {
                PSRC *sp = NULL;
                for (int i = 0; i < npsrc[t][g][a]; i++) if (psrc[t][g][a][i].src == l) { sp = &psrc[t][g][a][i]; break; }
                if (!sp && npsrc[t][g][a] < 8) sp = &psrc[t][g][a][npsrc[t][g][a]++];
                if (!sp) break;
                sp->src = l; sp->p.n = 0;
                char *q = p + cons;
                while (*q && sp->p.n < 16) {
                    while (*q == ' ') q++;
                    if (sscanf(q, "%d:%lf%n", &c, &v, &cons) != 2) break;
                    sp->p.ids[sp->p.n] = c; sp->p.sc[sp->p.n] = v; sp->p.n++;
                    q += cons;
                }
            } break; }
        case 'M':
            if (nmiss < MAXM && sscanf(p, "%lld %d %d %lf %lf %lf", &t, &l, &c, &v, &v2, &v3) == 6 && t < MAXTOK && l < MAXL) {
                miss[nmiss].tok = t; miss[nmiss].layer = l; miss[nmiss].eid = c;
                miss[nmiss].io = v; miss[nmiss].t0 = v2; miss[nmiss].t1 = v3;
                miss_next[nmiss] = miss_head[t][l]; miss_head[t][l] = nmiss; nmiss++;
            } break;
        }
    }
    fclose(f);
    fprintf(stderr, "loaded: maxtok=%lld misses=%d\n", maxtok, nmiss);

    printf("== LEAD-TIME WINDOWS (ms) ==\n");
    for (int d = 1; d <= 4; d++)
        for (int a = 0; a < 2; a++) {
            static double w[200000]; int cw = 0;
            for (long long t = 0; t <= maxtok && t < MAXTOK; t++)
                for (int l = 0; l < MAXL; l++) {
                    int sl = l - d; if (sl < 0 || sl >= MAXL) continue;
                    if (r_need[t][l].seen && k_anch[a][t][sl].seen && cw < 200000) w[cw++] = r_need[t][l].t - k_anch[a][t][sl].t;
                }
            if (!cw) continue;
            double sum = 0, mn = 1e30, mx = -1e30;
            for (int i = 0; i < cw; i++) { sum += w[i]; if (w[i] < mn) mn = w[i]; if (w[i] > mx) mx = w[i]; }
            double *cp = malloc(sizeof(double) * cw); memcpy(cp, w, sizeof(double) * cw);
            for (int i = 1; i < cw; i++) { double key = cp[i]; int j = i - 1; while (j >= 0 && cp[j] > key) { cp[j+1] = cp[j]; j--; } cp[j+1] = key; }
            printf("[LEAD] anchor=%s d=%d n=%d min=%.3f p10=%.3f med=%.3f mean=%.3f max=%.3f\n",
                   a ? "G(postmoe)" : "K(postmix)", d, cw, mn, cp[cw/10], cp[cw/2], sum/cw, mx);
            free(cp);
        }

    printf("== PERFECT-PREDICTION ORACLE (d<=4, both anchor types) ==\n");
    double base_io = 0, exp_best = 0, exp_G = 0;
    int nmv = 0, nh_best = 0, nh_G = 0;
    for (int i = 0; i < nmiss; i++) {
        MISS *mp = &miss[i]; if (mp->tok >= MAXTOK) continue;
        base_io += mp->io; nmv++;
        double bw = -1, bg = -1;
        for (int sl = mp->layer - 4; sl < mp->layer; sl++) {
            if (sl < 0 || sl >= MAXL) continue;
            if (k_anch[0][mp->tok][sl].seen) { double w = r_need[mp->tok][mp->layer].t - k_anch[0][mp->tok][sl].t; if (w > bw) bw = w; }
            if (k_anch[1][mp->tok][sl].seen) { double w = r_need[mp->tok][mp->layer].t - k_anch[1][mp->tok][sl].t; if (w > bg) bg = w; }
        }
        if (bw < 0) bw = 0; if (bg < 0) bg = 0;
        double e = mp->io - bw; if (e <= 0) nh_best++; else exp_best += e;
        e = mp->io - bg; if (e <= 0) nh_G++; else exp_G += e;
    }
    printf("[ORACLE] misses=%d base_io=%.1fms | K-any(d<=4): fully_hidden=%d exposed=%.1fms | G-any(d<=4): fully_hidden=%d exposed=%.1fms\n",
           nmv, base_io, nh_best, exp_best, nh_G, exp_G);
    long long ntok = maxtok + 1;
    printf("[ORACLE] ms/token: base=%.2f K_exposed=%.2f G_exposed=%.2f (dumped tokens=%lld)\n",
           ntok ? base_io / ntok : 0, ntok ? exp_best / ntok : 0, ntok ? exp_G / ntok : 0, ntok);

    printf("== PREDICTOR RECALL (vs authoritative top-8) ==\n");
    for (int a = 0; a < 2; a++)
        for (int d = 1; d <= 4; d++) {
            long long hit[3] = {0,0,0}, totA = 0;
            long long cm = 0, m8 = 0, m12 = 0, u8 = 0, u12 = 0, u16 = 0;
            double wbytes = 0; long long nfp = 0;
            long long h1[3] = {0,0,0}, t1 = 0, h2[3] = {0,0,0}, t2 = 0;
            for (long long t = 0; t <= maxtok && t < MAXTOK; t++)
                for (int l = 0; l < MAXL; l++) {
                    if (!r_need[t][l].seen || !act[t][l].seen) continue;
                    int src = l - d; if (src < 0) continue;
                    PSRC *sp = NULL;
                    for (int i = 0; i < npsrc[t][l][a]; i++) if (psrc[t][l][a][i].src == src) { sp = &psrc[t][l][a][i]; break; }
                    if (!sp) continue;
                    int wk = k_anch[a][t][src].seen;
                    double win = wk ? r_need[t][l].t - k_anch[a][t][src].t : -1;
                    int ch[3] = {0,0,0};
                    for (int kk = 0; kk < 8; kk++) {
                        int j = in_list(&sp->p, act[t][l].ids[kk], 16);
                        if (j < 0) continue;
                        if (j < 8) { ch[0]++; ch[1]++; ch[2]++; }
                        else if (j < 12) { ch[1]++; ch[2]++; }
                        else ch[2]++;
                    }
                    hit[0] += ch[0]; hit[1] += ch[1]; hit[2] += ch[2]; totA++;
                    int half2 = (t > maxtok / 2);
                    for (int mi = miss_head[t][l]; mi >= 0; mi = miss_next[mi]) {
                        cm++;
                        int j8 = in_list(&sp->p, miss[mi].eid, 8);
                        int j12 = in_list(&sp->p, miss[mi].eid, 12);
                        int j16 = in_list(&sp->p, miss[mi].eid, 16);
                        if (j8 >= 0) m8++;
                        if (j12 >= 0) m12++;
                        if (wk && win >= miss[mi].io) {
                            if (j8 >= 0) u8++;
                            if (j12 >= 0) u12++;
                            if (j16 >= 0) u16++;
                        }
                        int any = (j8 >= 0 || j12 >= 0 || j16 >= 0);
                        if (half2) { t2++; if (j8 >= 0) h2[0]++; if (j12 >= 0) h2[1]++; if (any) h2[2]++; }
                        else { t1++; if (j8 >= 0) h1[0]++; if (j12 >= 0) h1[1]++; if (any) h1[2]++; }
                    }
                    for (int j = 0; j < 8 && j < sp->p.n; j++) {
                        int isact = 0;
                        for (int kk = 0; kk < 8; kk++) if (act[t][l].ids[kk] == sp->p.ids[j]) { isact = 1; break; }
                        if (!isact) { nfp++; wbytes += 1.376256; }
                    }
                }
            if (!totA) continue;
            printf("[RECALL] anchor=%s d=%d acts=%lld top8=%.3f top12=%.3f top16=%.3f | misses=%lld missrec8=%.3f missrec12=%.3f useful8=%.3f useful12=%.3f useful16=%.3f | fp8=%.2f/tok waste~%.0fMB/tok | half1 mr8=%.3f(n=%lld) half2 mr8=%.3f(n=%lld)\n",
                   a ? "G" : "K", d, totA,
                   (double)hit[0]/(totA*8), (double)hit[1]/(totA*8), (double)hit[2]/(totA*8),
                   cm, cm ? (double)m8/cm : 0, cm ? (double)m12/cm : 0,
                   cm ? (double)u8/cm : 0, cm ? (double)u12/cm : 0, cm ? (double)u16/cm : 0,
                   (double)nfp/totA, wbytes/totA,
                   t1 ? (double)h1[0]/t1 : 0, t1, t2 ? (double)h2[0]/t2 : 0, t2);
        }
    return 0;
}
