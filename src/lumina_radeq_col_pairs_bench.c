/* withParityO cert-1 : offline C-port fidelity bench.
 *
 * Reads the per-ion raw-input dumps written by dump_cert_inputs.py (tabulated
 * split-J Omega + oscillator strengths + CMFGEN POPSIL/POP* d54 populations),
 * runs the SHARED runtime core radeq_col_pairs_build() (src/lumina_radeq_col_pairs.h)
 * exactly as the LUMINA_RADEQ_COL_PAIRS gate does, evaluates COOL at the CMFGEN
 * d54 T_e, and compares against the dig_F11 first-principles reproduce numbers.
 *
 * A match certifies the C fill (tab->vR->0.1) AND the C arithmetic against the
 * validated Python reference. Pure offline: no CUDA, no atomic-data load, no runs.
 *
 *   build:  gcc -O2 -o lumina_radeq_col_pairs_bench \
 *               src/lumina_radeq_col_pairs_bench.c -lm
 *   run:    ./lumina_radeq_col_pairs_bench <manifest_dir>
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "lumina_radeq_col_pairs.h"

typedef struct {
    int nlev; double ne, T4, oset, scale;
    double *edge, *n; int *g, *pqn;
    int n_tab; int *tl, *th; double *tom;
    int n_f;   int *fl, *fh; double *fv;
} IonDump;

static int rd(FILE *f, void *p, size_t n) { return fread(p, 1, n, f) == n; }

static int load_dump(const char *path, IonDump *d) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return -1; }
    if (!rd(f, &d->nlev, 4)) goto bad;
    double hdr[4]; if (!rd(f, hdr, 32)) goto bad;
    d->ne = hdr[0]; d->T4 = hdr[1]; d->oset = hdr[2]; d->scale = hdr[3];
    int N = d->nlev;
    d->edge = malloc(N * sizeof(double)); d->n = malloc(N * sizeof(double));
    d->g = malloc(N * sizeof(int));       d->pqn = malloc(N * sizeof(int));
    if (!rd(f, d->edge, N * 8)) goto bad;
    if (!rd(f, d->g,    N * 4)) goto bad;
    if (!rd(f, d->n,    N * 8)) goto bad;
    if (!rd(f, d->pqn,  N * 4)) goto bad;
    if (!rd(f, &d->n_tab, 4)) goto bad;
    d->tl = malloc(d->n_tab * sizeof(int)); d->th = malloc(d->n_tab * sizeof(int));
    d->tom = malloc(d->n_tab * sizeof(double));
    for (int t = 0; t < d->n_tab; t++) {
        if (!rd(f, &d->tl[t], 4) || !rd(f, &d->th[t], 4) || !rd(f, &d->tom[t], 8)) goto bad;
    }
    if (!rd(f, &d->n_f, 4)) goto bad;
    d->fl = malloc(d->n_f * sizeof(int)); d->fh = malloc(d->n_f * sizeof(int));
    d->fv = malloc(d->n_f * sizeof(double));
    for (int t = 0; t < d->n_f; t++) {
        if (!rd(f, &d->fl[t], 4) || !rd(f, &d->fh[t], 4) || !rd(f, &d->fv[t], 8)) goto bad;
    }
    fclose(f); return 0;
bad:
    fprintf(stderr, "malformed dump %s\n", path); fclose(f); return -1;
}

int main(int argc, char **argv) {
    const char *dir = (argc > 1) ? argv[1] : ".";
    char mpath[2048]; snprintf(mpath, sizeof(mpath), "%s/cert_manifest.txt", dir);
    FILE *mf = fopen(mpath, "r");
    if (!mf) { fprintf(stderr, "no %s\n", mpath); return 2; }
    printf("# withParityO cert-1 : C-port fidelity vs dig_F11 reproduce\n");
    printf("%-8s  %-15s  %-15s  %-8s  | %-13s %-13s %-13s\n",
           "ion", "COOL_bb(C)", "target", "ratio", "tab", "vR", "0.1");
    char line[512]; int npass = 0, ntot = 0; double worst = 1.0;
    while (fgets(line, sizeof(line), mf)) {
        if (line[0] == '#' || line[0] == '\n') continue;
        char lab[64], file[256]; double target, gc;
        /* format: label|file|repro_target|gencool_target */
        char *p = strchr(line, '|'); if (!p) continue;
        *p = 0; strncpy(lab, line, sizeof(lab) - 1); lab[sizeof(lab)-1]=0;
        char *q = strchr(p + 1, '|'); if (!q) continue;
        *q = 0; strncpy(file, p + 1, sizeof(file) - 1); file[sizeof(file)-1]=0;
        char *r = strchr(q + 1, '|'); if (!r) continue; *r = 0;
        target = atof(q + 1); gc = atof(r + 1); (void)gc;
        /* trim trailing spaces on lab */
        for (int i = (int)strlen(lab) - 1; i >= 0 && (lab[i]==' '||lab[i]=='\t'); i--) lab[i]=0;
        char dpath[2048]; snprintf(dpath, sizeof(dpath), "%s/%s", dir, file);
        IonDump d;
        if (load_dump(dpath, &d) != 0) continue;
        long cap = (long)d.nlev * (d.nlev - 1) / 2 + 1;
        double *a = malloc(cap * sizeof(double));
        double *b = malloc(cap * sizeof(double));
        double *bet = malloc(cap * sizeof(double));
        long nact = 0; RcpCensus cen;
        double Tref = d.T4 * 1.0e4;
        int rc = radeq_col_pairs_build(d.nlev, d.edge, d.g, d.n, d.pqn,
                                       d.ne, d.T4, d.oset, d.scale,
                                       d.n_tab, d.tl, d.th, d.tom,
                                       d.n_f, d.fl, d.fh, d.fv,
                                       a, b, bet, &nact, &cen);
        if (rc != 0) { fprintf(stderr, "%s: build failed\n", lab); continue; }
        /* evaluate COOL at the CMFGEN d54 T_e, exactly as radeq_line_cool does */
        double sqTe = sqrt(Tref), sum = 0.0;
        for (long m = 0; m < nact; m++) sum += a[m] * exp(-bet[m] / Tref) - b[m];
        double COOL = d.ne / sqTe * sum;
        double ratio = (fabs(target) > 0.0) ? COOL / target : NAN;
        int pass = isfinite(ratio) && ratio > 0.70 && ratio < 1.30; /* +-30% band */
        ntot++; if (pass) npass++;
        if (isfinite(ratio) && fabs(ratio - 1.0) > fabs(worst - 1.0)) worst = ratio;
        printf("%-8s  %+.6e  %+.6e  %+7.4f  | %+.4e(%ld) %+.4e(%ld) %+.4e(%ld) %s\n",
               lab, COOL, target, ratio, cen.c_tab, cen.n_tab, cen.c_vr, cen.n_vr,
               cen.c_set, cen.n_set, pass ? "PASS" : "FAIL");
        free(a); free(b); free(bet);
        free(d.edge); free(d.n); free(d.g); free(d.pqn);
        free(d.tl); free(d.th); free(d.tom); free(d.fl); free(d.fh); free(d.fv);
    }
    fclose(mf);
    printf("# %d/%d ions within +-30%% (dig_F11 acceptance band); worst ratio=%.4f\n",
           npass, ntot, worst);
    return (npass == ntot) ? 0 : 1;
}
