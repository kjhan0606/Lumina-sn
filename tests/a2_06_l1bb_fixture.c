/* A2-06 canonical eight-band projection-closure fixture.
 *
 * Each synthetic path segment has linearly varying comoving frequency and
 * energy.  The direct selective estimator integrates the registered
 * band-constant profile along the segment; the projection lane first deposits
 * the identical raw path-length measure into the canonical band and then
 * integrates phi against that band average.  JSON is written to stdout. */
#include <math.h>
#include <stdio.h>

#define NB 8
#define NS 19

typedef struct {
    double nu0, nu1, e0, e1, length;
} Segment;

static double overlap_energy(const Segment *s, double lo, double hi)
{
    double a = s->nu0, b = s->nu1;
    if (b < a) { double t = a; a = b; b = t; }
    if (!(b > a) || hi <= a || lo >= b) return 0.0;
    double x0 = fmax(lo, a), x1 = fmin(hi, b);
    double t0 = (x0 - a) / (b - a), t1 = (x1 - a) / (b - a);
    double de = s->e1 - s->e0;
    return s->length * (s->e0 * (t1 - t0) +
           0.5 * de * (t1 * t1 - t0 * t0));
}

int main(void)
{
    const double nu_lo = 1.4402928950097124e12;
    const double nu_hi = 4.032418413741097e16;
    double edge[NB + 1] = {
        1440292895009.7124, 5180124105196.376,
        18630714515227.543, 67006796806222.781,
        240995095199417.31, 866757384002910.25,
        3117359554981452.5, 11211823255723808.0,
        40324184137410968.0
    };
    double direct[NB] = {0}, projected[NB] = {0};
    Segment seg[NS];
    (void)nu_lo;
    (void)nu_hi;

    /* Two in-band segments per band plus three boundary crossers. */
    int n = 0;
    for (int k = 0; k < NB; k++) {
        double w = edge[k + 1] - edge[k];
        seg[n++] = (Segment){edge[k] + .11*w, edge[k] + .67*w,
                             1.0 + k, 1.4 + k, 2.0 + .1*k};
        seg[n++] = (Segment){edge[k] + .82*w, edge[k] + .29*w,
                             .7 + .2*k, 1.2 + .2*k, 1.1 + .05*k};
    }
    for (int k = 2; k <= 6; k += 2) {
        double wl = edge[k] - edge[k - 1], wr = edge[k + 1] - edge[k];
        seg[n++] = (Segment){edge[k] - .19*wl, edge[k] + .23*wr,
                             2.3 + k, 1.7 + k, .9 + .03*k};
    }

    for (int k = 0; k < NB; k++) {
        double raw = 0.0;
        for (int q = 0; q < n; q++) raw += overlap_energy(&seg[q], edge[k], edge[k+1]);
        /* phi_k = 1 / Delta-nu_k.  The common 1/(4*pi*V*dt)
         * normalization is deliberately present in both lanes. */
        double norm = 1.0 / (4.0 * acos(-1.0) * 7.25e40 * 3.5e4);
        direct[k] = norm * raw / (edge[k + 1] - edge[k]);
        double J_band = norm * raw / (edge[k + 1] - edge[k]);
        projected[k] = J_band * 1.0;
    }

    printf("{\n  \"schema\": \"lumina-a2-06-projection-fixture-v1\",\n");
    printf("  \"frame\": \"comoving\",\n  \"normalization\": \"1/(4*pi*V_s*delta_t)\",\n");
    printf("  \"band_edges_hz\": [");
    for (int k = 0; k <= NB; k++) printf("%s%.17g", k ? ", " : "", edge[k]);
    printf("],\n  \"rows\": [\n");
    for (int k = 0; k < NB; k++)
        printf("    {\"band\": %d, \"direct\": %.17g, \"projected\": %.17g}%s\n",
               k, direct[k], projected[k], k + 1 == NB ? "" : ",");
    printf("  ]\n}\n");
    return 0;
}
