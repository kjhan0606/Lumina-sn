/*
 * Stream-select a Gate-B raw-Jbar capture without loading the ledger.
 *
 * The production dump is tens of GB.  This helper makes one sequential pass,
 * writes only the requested consumer iteration and oracle shells, and uses the
 * final production shell as a completeness sentinel.  Per-ion line-id
 * signatures must be identical for every selected shell and the sentinel.
 */
#define _GNU_SOURCE
#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N_IONS 8
#define N_SHELLS 4

static const int ion_z[N_IONS] = {14, 14, 16, 16, 26, 26, 26, 27};
static const int ion_stage[N_IONS] = {1, 2, 1, 2, 1, 2, 3, 2};
static const int shells[N_SHELLS] = {0, 8, 43, 49};

typedef struct {
    uint64_t count;
    uint64_t hash;
    uint64_t sum;
    int first;
    int last;
    int strictly_increasing;
} Signature;

static int field_long(char **p, long *value)
{
    char *end = NULL;
    errno = 0;
    long v = strtol(*p, &end, 10);
    if (errno || end == *p || (*end != ',' && *end != '\n' && *end != '\0'))
        return -1;
    *value = v;
    *p = (*end == ',') ? end + 1 : end;
    return 0;
}

static int ion_slot(long z, long stage)
{
    for (int i = 0; i < N_IONS; ++i)
        if (ion_z[i] == z && ion_stage[i] == stage)
            return i;
    return -1;
}

static int shell_slot(long shell)
{
    for (int i = 0; i < N_SHELLS; ++i)
        if (shells[i] == shell)
            return i;
    return -1;
}

static void add_signature(Signature *sig, int line_id)
{
    if (sig->count == 0) {
        sig->first = line_id;
        sig->strictly_increasing = 1;
    } else if (line_id <= sig->last) {
        sig->strictly_increasing = 0;
    }
    sig->last = line_id;
    sig->count++;
    sig->sum += (uint64_t)(unsigned int)line_id;
    uint64_t x = (uint64_t)(unsigned int)line_id;
    for (int b = 0; b < 8; ++b) {
        sig->hash ^= (x >> (8 * b)) & 0xffu;
        sig->hash *= UINT64_C(1099511628211);
    }
}

static int same_signature(const Signature *a, const Signature *b)
{
    if (a->count == 0 && b->count == 0)
        return 1;
    return a->count == b->count && a->hash == b->hash &&
           a->sum == b->sum && a->first == b->first &&
           a->last == b->last && a->strictly_increasing &&
           b->strictly_increasing;
}

int main(int argc, char **argv)
{
    if (argc != 5) {
        fprintf(stderr,
                "usage: %s INPUT.csv CONSUMER_ITER OUTPUT.csv MANIFEST.csv\n",
                argv[0]);
        return 2;
    }
    char *iter_end = NULL;
    long wanted_iter = strtol(argv[2], &iter_end, 10);
    if (!iter_end || *iter_end || wanted_iter < 0) {
        fprintf(stderr, "invalid consumer iteration: %s\n", argv[2]);
        return 2;
    }

    FILE *in = fopen(argv[1], "r");
    FILE *out = fopen(argv[3], "w");
    if (!in || !out) {
        fprintf(stderr, "cannot open input/output: %s\n", strerror(errno));
        if (in) fclose(in);
        if (out) fclose(out);
        return 1;
    }

    char *line = NULL;
    size_t cap = 0;
    ssize_t len = getline(&line, &cap, in);
    const char *header =
        "iter,shell,Z,ion,line_idx,lambda_A,jbar_line,jbar_count,beta,mode,B_planck_Te\n";
    if (len <= 0 || strcmp(line, header) != 0) {
        fprintf(stderr, "unexpected raw-Jbar header\n");
        free(line);
        fclose(in);
        fclose(out);
        return 1;
    }
    fputs(header, out);

    Signature sig[N_SHELLS][N_IONS];
    memset(sig, 0, sizeof(sig));
    for (int s = 0; s < N_SHELLS; ++s)
        for (int q = 0; q < N_IONS; ++q)
            sig[s][q].hash = UINT64_C(1469598103934665603);

    uint64_t input_rows = 0, selected_rows = 0, malformed_rows = 0;
    while ((len = getline(&line, &cap, in)) >= 0) {
        input_rows++;
        char *p = line;
        long iter, shell, z, stage, line_id;
        if (field_long(&p, &iter) || field_long(&p, &shell) ||
            field_long(&p, &z) || field_long(&p, &stage) ||
            field_long(&p, &line_id)) {
            malformed_rows++;
            continue;
        }
        if (iter != wanted_iter)
            continue;
        int ss = shell_slot(shell);
        int qi = ion_slot(z, stage);
        if (ss < 0 || qi < 0)
            continue;
        if (line_id < 0 || line_id > 100000000L) {
            malformed_rows++;
            continue;
        }
        add_signature(&sig[ss][qi], (int)line_id);
        if (ss < N_SHELLS - 1) {
            if (fwrite(line, 1, (size_t)len, out) != (size_t)len) {
                fprintf(stderr, "write failed: %s\n", strerror(errno));
                free(line);
                fclose(in);
                fclose(out);
                return 1;
            }
            selected_rows++;
        }
    }
    int input_error = ferror(in);
    free(line);
    if (fclose(in) || fclose(out) || input_error) {
        fprintf(stderr, "stream read/write close failed\n");
        return 1;
    }

    FILE *mf = fopen(argv[4], "w");
    if (!mf) {
        fprintf(stderr, "cannot open manifest: %s\n", strerror(errno));
        return 1;
    }
    fprintf(mf, "consumer_iter,shell,Z,stage,count,first_line_id,last_line_id,"
                "line_id_sum,fnv1a64,strictly_increasing,matches_s49\n");
    int complete = malformed_rows == 0;
    for (int ss = 0; ss < N_SHELLS; ++ss) {
        for (int qi = 0; qi < N_IONS; ++qi) {
            int match = same_signature(&sig[ss][qi],
                                       &sig[N_SHELLS - 1][qi]);
            if (!match)
                complete = 0;
            fprintf(mf, "%ld,%d,%d,%d,%" PRIu64 ",%d,%d,%" PRIu64
                        ",%016" PRIx64 ",%d,%d\n",
                    wanted_iter, shells[ss], ion_z[qi], ion_stage[qi],
                    sig[ss][qi].count, sig[ss][qi].first, sig[ss][qi].last,
                    sig[ss][qi].sum, sig[ss][qi].hash,
                    sig[ss][qi].strictly_increasing, match);
        }
    }
    fprintf(mf, "# input_rows=%" PRIu64 ",selected_rows=%" PRIu64
                ",malformed_rows=%" PRIu64 ",complete=%d\n",
            input_rows, selected_rows, malformed_rows, complete);
    fclose(mf);

    fprintf(stderr, "[GATEB-SELECT] input=%" PRIu64 " selected=%" PRIu64
                    " malformed=%" PRIu64 " complete=%d\n",
            input_rows, selected_rows, malformed_rows, complete);
    return complete ? 0 : 1;
}
