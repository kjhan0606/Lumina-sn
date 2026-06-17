/* lumina_atomic.c — Phase 2 - Step 7: Load TARDIS reference data from CSV/NPY files.
 * Reads the exact converged plasma state exported by export_tardis_reference.py.
 * This ensures bit-for-bit matching with TARDIS ground truth. */

#include "lumina.h" /* Phase 2 - Step 7 */

#ifdef __cplusplus   /* Phase 6 - Step 9: extern C guard for NVCC */
extern "C" {         /* Phase 6 - Step 9 */
#endif               /* Phase 6 - Step 9 */

/* ============================================================ */
/* Phase 2 - Step 8: NPY file reader (NumPy .npy format)       */
/* ============================================================ */

/* Phase 2 - Step 8: Read NPY header, return data pointer */
static double *read_npy_f64(const char *path, int *out_rows, int *out_cols) {
    FILE *fp = fopen(path, "rb"); /* Phase 2 - Step 8 */
    if (!fp) { /* Phase 2 - Step 8 */
        fprintf(stderr, "ERROR: Cannot open %s\n", path); /* Phase 2 - Step 8 */
        return NULL; /* Phase 2 - Step 8 */
    }

    /* A7: short-read aware macro — abort cleanly instead of trusting garbage. */
    #define NPY_FREAD(buf, elt, cnt) \
        do { \
            size_t _need = (size_t)(cnt); \
            size_t _got  = fread((buf), (elt), _need, fp); \
            if (_got != _need) { \
                fprintf(stderr, "ERROR: %s short read (%zu/%zu) at line %d\n", \
                        path, _got, _need, __LINE__); \
                fclose(fp); \
                return NULL; \
            } \
        } while (0)

    /* Phase 2 - Step 8: Read magic number */
    unsigned char magic[6]; /* Phase 2 - Step 8 */
    NPY_FREAD(magic, 1, 6); /* A7 */
    if (magic[0] != 0x93 || magic[1] != 'N' || magic[2] != 'U' || /* Phase 2 - Step 8 */
        magic[3] != 'M' || magic[4] != 'P' || magic[5] != 'Y') { /* Phase 2 - Step 8 */
        fprintf(stderr, "ERROR: %s is not a valid .npy file\n", path); /* Phase 2 - Step 8 */
        fclose(fp); /* Phase 2 - Step 8 */
        return NULL; /* Phase 2 - Step 8 */
    }

    /* Phase 2 - Step 8: Read version */
    unsigned char version[2]; /* Phase 2 - Step 8 */
    NPY_FREAD(version, 1, 2); /* A7 */

    /* Phase 2 - Step 8: Read header length */
    uint16_t header_len; /* Phase 2 - Step 8 */
    if (version[0] == 1) { /* Phase 2 - Step 8 */
        NPY_FREAD(&header_len, 2, 1); /* A7 */
    } else { /* Phase 2 - Step 8 */
        uint32_t hl32; /* Phase 2 - Step 8 */
        NPY_FREAD(&hl32, 4, 1); /* A7 */
        header_len = (uint16_t)hl32; /* Phase 2 - Step 8 */
    }

    /* Phase 2 - Step 8: Read header string */
    char *header = (char *)malloc(header_len + 1); /* Phase 2 - Step 8 */
    {
        size_t _got = fread(header, 1, header_len, fp);
        if (_got != (size_t)header_len) {
            fprintf(stderr, "ERROR: %s short read of header (%zu/%u)\n",
                    path, _got, (unsigned)header_len);
            free(header); fclose(fp); return NULL;
        }
    }
    header[header_len] = '\0'; /* Phase 2 - Step 8 */

    /* Phase 2 - Step 8: Parse shape from header */
    int rows = 0, cols = 0; /* Phase 2 - Step 8 */
    char *shape_start = strstr(header, "'shape': ("); /* Phase 2 - Step 8 */
    if (!shape_start) shape_start = strstr(header, "\"shape\": ("); /* Phase 2 - Step 8 */
    if (shape_start) { /* Phase 2 - Step 8 */
        shape_start = strchr(shape_start, '(') + 1; /* Phase 2 - Step 8 */
        char *shape_end = strchr(shape_start, ')'); /* Phase 2 - Step 8 */
        char shape_str[256]; /* Phase 2 - Step 8 */
        int len = (int)(shape_end - shape_start); /* Phase 2 - Step 8 */
        strncpy(shape_str, shape_start, len); /* Phase 2 - Step 8 */
        shape_str[len] = '\0'; /* Phase 2 - Step 8 */

        /* Phase 2 - Step 8: Count commas for dimensionality */
        rows = atoi(shape_str); /* Phase 2 - Step 8 */
        char *comma = strchr(shape_str, ','); /* Phase 2 - Step 8 */
        if (comma) { /* Phase 2 - Step 8 */
            /* Phase 2 - Step 8: Skip whitespace after comma */
            char *after = comma + 1; /* Phase 2 - Step 8 */
            while (*after == ' ') after++; /* Phase 2 - Step 8 */
            if (*after != '\0' && *after != ')') { /* Phase 2 - Step 8: 2D */
                cols = atoi(after); /* Phase 2 - Step 8 */
                if (cols <= 0) cols = 1; /* Phase 2 - Step 8 */
            } else { /* Phase 2 - Step 8: 1D with trailing comma */
                cols = 1; /* Phase 2 - Step 8 */
            }
        } else { /* Phase 2 - Step 8: 1D no comma */
            cols = 1; /* Phase 2 - Step 8 */
        }
    }

    /* Phase 2 - Step 8: Check dtype */
    bool is_int = false; /* Phase 2 - Step 8 */
    if (strstr(header, "'<i8'") || strstr(header, "\"<i8\"") || /* Phase 2 - Step 8 */
        strstr(header, "'<i4'") || strstr(header, "\"<i4\"")) { /* Phase 2 - Step 8 */
        is_int = true; /* Phase 2 - Step 8 */
    }

    /* Phase 2 - Step 8: Check Fortran order */
    bool fortran_order = false; /* Phase 2 - Step 8 */
    if (strstr(header, "'fortran_order': True") || /* Phase 2 - Step 8 */
        strstr(header, "\"fortran_order\": true")) { /* Phase 2 - Step 8 */
        fortran_order = true; /* Phase 2 - Step 8 */
    }

    free(header); /* Phase 2 - Step 8 */

    int total = rows * cols; /* Phase 2 - Step 8 */
    double *data = (double *)malloc(total * sizeof(double)); /* Phase 2 - Step 8 */

    if (is_int) { /* Phase 2 - Step 8: Read as int64 and convert */
        int64_t *idata = (int64_t *)malloc(total * sizeof(int64_t)); /* Phase 2 - Step 8 */
        {
            size_t _got = fread(idata, sizeof(int64_t), (size_t)total, fp);
            if (_got != (size_t)total) {
                fprintf(stderr, "ERROR: %s short read of int64 body (%zu/%d)\n",
                        path, _got, total);
                free(idata); free(data); fclose(fp); return NULL;
            }
        }
        for (int i = 0; i < total; i++) { /* Phase 2 - Step 8 */
            data[i] = (double)idata[i]; /* Phase 2 - Step 8 */
        }
        free(idata); /* Phase 2 - Step 8 */
    } else { /* Phase 2 - Step 8: Read as float64 directly */
        size_t _got = fread(data, sizeof(double), (size_t)total, fp); /* A7 */
        if (_got != (size_t)total) {
            fprintf(stderr, "ERROR: %s short read of float64 body (%zu/%d)\n",
                    path, _got, total);
            free(data); fclose(fp); return NULL;
        }
    }
    #undef NPY_FREAD

    /* Phase 2 - Step 8: If Fortran order, transpose to C order */
    if (fortran_order && cols > 1) { /* Phase 2 - Step 8 */
        double *transposed = (double *)malloc(total * sizeof(double)); /* Phase 2 - Step 8 */
        for (int i = 0; i < rows; i++) { /* Phase 2 - Step 8 */
            for (int j = 0; j < cols; j++) { /* Phase 2 - Step 8 */
                transposed[i * cols + j] = data[j * rows + i]; /* Phase 2 - Step 8 */
            }
        }
        free(data); /* Phase 2 - Step 8 */
        data = transposed; /* Phase 2 - Step 8 */
    }

    fclose(fp); /* Phase 2 - Step 8 */
    *out_rows = rows; /* Phase 2 - Step 8 */
    *out_cols = cols; /* Phase 2 - Step 8 */
    return data; /* Phase 2 - Step 8 */
}

/* Phase 2 - Step 8b: Read NPY as int array */
static int *read_npy_int(const char *path, int *out_n) {
    int rows, cols; /* Phase 2 - Step 8b */
    double *ddata = read_npy_f64(path, &rows, &cols); /* Phase 2 - Step 8b */
    if (!ddata) return NULL; /* Phase 2 - Step 8b */
    int total = rows * cols; /* Phase 2 - Step 8b */
    int *idata = (int *)malloc(total * sizeof(int)); /* Phase 2 - Step 8b */
    for (int i = 0; i < total; i++) { /* Phase 2 - Step 8b */
        idata[i] = (int)ddata[i]; /* Phase 2 - Step 8b */
    }
    free(ddata); /* Phase 2 - Step 8b */
    *out_n = total; /* Phase 2 - Step 8b */
    return idata; /* Phase 2 - Step 8b */
}

/* ============================================================ */
/* Phase 2 - Step 9: CSV readers                                */
/* ============================================================ */

/* Phase 2 - Step 9: Read a CSV column by name, return array */
static double *read_csv_column(const char *path, const char *col_name,
                                int *out_n) {
    FILE *fp = fopen(path, "r"); /* Phase 2 - Step 9 */
    if (!fp) { /* Phase 2 - Step 9 */
        fprintf(stderr, "ERROR: Cannot open %s\n", path); /* Phase 2 - Step 9 */
        return NULL; /* Phase 2 - Step 9 */
    }

    /* Phase 2 - Step 9: Read header line */
    char line[4096]; /* Phase 2 - Step 9 */
    if (!fgets(line, sizeof(line), fp)) { /* Phase 2 - Step 9 */
        fclose(fp); return NULL; /* Phase 2 - Step 9 */
    }

    /* Phase 2 - Step 9: Find column index */
    /* Phase 2 - Step 9: Manual CSV parse — strtok skips empty leading fields! */
    int col_idx = -1, idx = 0; /* Phase 2 - Step 9 */
    char *p_hdr = line; /* Phase 2 - Step 9 */
    while (*p_hdr && *p_hdr != '\n' && *p_hdr != '\r') { /* Phase 2 - Step 9 */
        /* Phase 2 - Step 9: Find end of current field */
        char *field_start = p_hdr; /* Phase 2 - Step 9 */
        while (*p_hdr && *p_hdr != ',' && *p_hdr != '\n' && *p_hdr != '\r') { /* Phase 2 - Step 9 */
            p_hdr++; /* Phase 2 - Step 9 */
        }
        /* Phase 2 - Step 9: Null-terminate this field temporarily */
        char saved = *p_hdr; /* Phase 2 - Step 9 */
        *p_hdr = '\0'; /* Phase 2 - Step 9 */
        /* Phase 2 - Step 9: Strip leading whitespace from field */
        while (*field_start == ' ') field_start++; /* Phase 2 - Step 9 */
        if (strcmp(field_start, col_name) == 0) { /* Phase 2 - Step 9 */
            col_idx = idx; /* Phase 2 - Step 9 */
            *p_hdr = saved; /* Phase 2 - Step 9 */
            break; /* Phase 2 - Step 9 */
        }
        *p_hdr = saved; /* Phase 2 - Step 9 */
        if (*p_hdr == ',') p_hdr++; /* Phase 2 - Step 9: skip comma */
        idx++; /* Phase 2 - Step 9 */
    }

    if (col_idx < 0) { /* Phase 2 - Step 9 */
        fprintf(stderr, "ERROR: Column '%s' not found in %s\n", col_name, path); /* Phase 2 - Step 9 */
        fclose(fp); return NULL; /* Phase 2 - Step 9 */
    }

    /* Phase 2 - Step 9: Count rows */
    int capacity = 1024; /* Phase 2 - Step 9 */
    double *data = (double *)malloc(capacity * sizeof(double)); /* Phase 2 - Step 9 */
    int n = 0; /* Phase 2 - Step 9 */

    while (fgets(line, sizeof(line), fp)) { /* Phase 2 - Step 9 */
        if (line[0] == '\n' || line[0] == '\r') continue; /* Phase 2 - Step 9 */
        /* Phase 2 - Step 9: Walk to correct column */
        char *p = line; /* Phase 2 - Step 9 */
        for (int i = 0; i < col_idx; i++) { /* Phase 2 - Step 9 */
            p = strchr(p, ','); /* Phase 2 - Step 9 */
            if (!p) break; /* Phase 2 - Step 9 */
            p++; /* Phase 2 - Step 9 */
        }
        if (!p) continue; /* Phase 2 - Step 9 */

        if (n >= capacity) { /* Phase 2 - Step 9 */
            capacity *= 2; /* Phase 2 - Step 9 */
            data = (double *)realloc(data, capacity * sizeof(double)); /* Phase 2 - Step 9 */
        }
        data[n++] = atof(p); /* Phase 2 - Step 9 */
    }

    fclose(fp); /* Phase 2 - Step 9 */
    *out_n = n; /* Phase 2 - Step 9 */
    return data; /* Phase 2 - Step 9 */
}

/* Phase 2 - Step 9b: Read CSV column as int */
static int *read_csv_column_int(const char *path, const char *col_name,
                                 int *out_n) {
    int n; /* Phase 2 - Step 9b */
    double *ddata = read_csv_column(path, col_name, &n); /* Phase 2 - Step 9b */
    if (!ddata) return NULL; /* Phase 2 - Step 9b */
    int *idata = (int *)malloc(n * sizeof(int)); /* Phase 2 - Step 9b */
    for (int i = 0; i < n; i++) { /* Phase 2 - Step 9b */
        idata[i] = (int)ddata[i]; /* Phase 2 - Step 9b */
    }
    free(ddata); /* Phase 2 - Step 9b */
    *out_n = n; /* Phase 2 - Step 9b */
    return idata; /* Phase 2 - Step 9b */
}

/* ============================================================ */
/* Phase 2 - Step 10: Main data loader                          */
/* ============================================================ */

int load_tardis_reference_data(const char *ref_dir, Geometry *geo,
                                OpacityState *opacity, PlasmaState *plasma,
                                MCConfig *config) {
    char path[512]; /* Phase 2 - Step 10 */
    int n; /* Phase 2 - Step 10 */

    printf("Loading TARDIS reference data from %s...\n", ref_dir); /* Phase 2 - Step 10 */

    /* Phase 2 - Step 10a: Load geometry */
    snprintf(path, sizeof(path), "%s/geometry.csv", ref_dir); /* Phase 2 - Step 10a */
    geo->r_inner = read_csv_column(path, "r_inner", &n); /* Phase 2 - Step 10a */
    geo->n_shells = n; /* Phase 2 - Step 10a */
    geo->r_outer = read_csv_column(path, "r_outer", &n); /* Phase 2 - Step 10a */
    geo->v_inner = read_csv_column(path, "v_inner", &n); /* Phase 2 - Step 10a */
    geo->v_outer = read_csv_column(path, "v_outer", &n); /* Phase 2 - Step 10a */
    printf("  Geometry: %d shells\n", geo->n_shells); /* Phase 2 - Step 10a */
    printf("    r_inner[0] = %.6e cm, r_outer[%d] = %.6e cm\n", /* Phase 2 - Step 10a */
           geo->r_inner[0], n - 1, geo->r_outer[n - 1]); /* Phase 2 - Step 10a */
    printf("    v_inner[0] = %.6e cm/s, v_outer[%d] = %.6e cm/s\n", /* Phase 2 - Step 10a */
           geo->v_inner[0], n - 1, geo->v_outer[n - 1]); /* Phase 2 - Step 10a */

    /* Phase 2 - Step 10b: Load config */
    snprintf(path, sizeof(path), "%s/config.json", ref_dir); /* Phase 2 - Step 10b */
    FILE *fp = fopen(path, "r"); /* Phase 2 - Step 10b */
    if (!fp) {
        fprintf(stderr, "ERROR: cannot open %s\n", path);
        return -1;
    }
    {
        char buf[4096]; /* Phase 2 - Step 10b */
        size_t nr = fread(buf, 1, sizeof(buf) - 1, fp); /* Phase 2 - Step 10b */
        buf[nr] = '\0'; /* Phase 2 - Step 10b */
        fclose(fp); /* Phase 2 - Step 10b */

        /* A5: track which required keys are present.  Previously missing keys
         * silently left the struct zeroed (T_inner=0, n_packets=0 etc.) and the
         * run proceeded with nonsense — would show up as "T_inner: 0.00 K" in
         * the banner.  Now: require all 6, abort on first missing. */
        char *p; /* Phase 2 - Step 10b */
        int missing = 0;
        #define PARSE_REQ(key, sink, conv)                                  \
            do {                                                             \
                p = strstr(buf, "\"" key "\"");                              \
                if (!p || !(p = strchr(p, ':'))) {                           \
                    fprintf(stderr,                                          \
                        "ERROR: %s missing required key \"%s\"\n",           \
                        path, key);                                          \
                    missing = 1;                                             \
                } else {                                                     \
                    sink = conv(p + 1);                                      \
                }                                                            \
            } while (0)
        PARSE_REQ("time_explosion_s",       geo->time_explosion,         atof);
        PARSE_REQ("T_inner_K",              config->T_inner,             atof);
        PARSE_REQ("luminosity_inner_erg_s", config->luminosity_requested,atof);
        PARSE_REQ("n_packets",              config->n_packets,           atoi);
        PARSE_REQ("n_iterations",           config->n_iterations,        atoi);
        PARSE_REQ("seed",                   config->seed,                (uint64_t)atol);
        #undef PARSE_REQ
        if (missing) return -1;

        /* Optional: T_e/T_rad ratio (default 0.9 if absent) */
        plasma->T_e_T_rad_ratio = 0.9;
        p = strstr(buf, "\"T_e_T_rad_ratio\"");
        if (p) { p = strchr(p, ':'); plasma->T_e_T_rad_ratio = atof(p + 1); }
        /* Env override (perturbation test: does the 0.9 seed/fallback anchor the
         * converged T_e?). LUMINA_TE_TRAD_RATIO=0.7 etc. overrides config. */
        { const char *e = getenv("LUMINA_TE_TRAD_RATIO");
          if (e && atof(e) > 0.0) plasma->T_e_T_rad_ratio = atof(e); }

        printf("  Config: t_exp=%.6e s, T_inner=%.2f K, L=%.3e erg/s\n", /* Phase 2 - Step 10b */
               geo->time_explosion, config->T_inner, config->luminosity_requested); /* Phase 2 - Step 10b */
        printf("    n_packets=%d, n_iter=%d, seed=%lu, T_e/T_rad=%.3f\n",
               config->n_packets, config->n_iterations, config->seed,
               plasma->T_e_T_rad_ratio);
    }

    /* Phase 2 - Step 10c: Load electron densities */
    snprintf(path, sizeof(path), "%s/electron_densities.csv", ref_dir); /* Phase 2 - Step 10c */
    opacity->electron_density = read_csv_column(path, "n_e", &n); /* Phase 2 - Step 10c */
    printf("  Electron densities: n_e[0]=%.6e, n_e[%d]=%.6e cm^-3\n", /* Phase 2 - Step 10c */
           opacity->electron_density[0], n - 1, opacity->electron_density[n - 1]); /* Phase 2 - Step 10c */

    /* Phase 2 - Step 10d: Load plasma state (W, T_rad) */
    snprintf(path, sizeof(path), "%s/plasma_state.csv", ref_dir); /* Phase 2 - Step 10d */
    plasma->n_shells = geo->n_shells; /* Phase 2 - Step 10d */
    plasma->W = read_csv_column(path, "W", &n); /* Phase 2 - Step 10d */
    plasma->T_rad = read_csv_column(path, "T_rad", &n); /* Phase 2 - Step 10d */
    printf("  Plasma: W[0]=%.6f, T_rad[0]=%.2f K\n", /* Phase 2 - Step 10d */
           plasma->W[0], plasma->T_rad[0]); /* Phase 2 - Step 10d */

    /* Phase 2 - Step 10d2: Load density */
    snprintf(path, sizeof(path), "%s/density.csv", ref_dir); /* Phase 2 - Step 10d2 */
    plasma->rho = read_csv_column(path, "rho", &n); /* Phase 2 - Step 10d2 */

    /* Phase 2 - Step 10d3: T_electrons = T_rad for now (TARDIS uses T_e ≈ 0.9 * T_rad) */
    opacity->t_electrons = (double *)malloc(geo->n_shells * sizeof(double)); /* Phase 2 - Step 10d3 */
    for (int i = 0; i < geo->n_shells; i++) { /* Phase 2 - Step 10d3 */
        opacity->t_electrons[i] = plasma->T_rad[i]; /* Phase 2 - Step 10d3 */
    }

    /* Phase 2 - Step 10e: Load line list (nu, sorted descending) */
    snprintf(path, sizeof(path), "%s/line_list.csv", ref_dir); /* Phase 2 - Step 10e */
    opacity->line_list_nu = read_csv_column(path, "nu", &n); /* Phase 2 - Step 10e */
    opacity->n_lines = n; /* Phase 2 - Step 10e */
    opacity->n_shells = geo->n_shells; /* Phase 2 - Step 10e */
    printf("  Lines: %d total, nu[0]=%.6e Hz (%.1f A), nu[%d]=%.6e Hz (%.1f A)\n", /* Phase 2 - Step 10e */
           n, opacity->line_list_nu[0], /* Phase 2 - Step 10e */
           C_SPEED_OF_LIGHT / opacity->line_list_nu[0] * 1e8, /* Phase 2 - Step 10e */
           n - 1, opacity->line_list_nu[n - 1], /* Phase 2 - Step 10e */
           C_SPEED_OF_LIGHT / opacity->line_list_nu[n - 1] * 1e8); /* Phase 2 - Step 10e */

    /* Task #072: Store line_list.csv path for later atomic data loading */
    /* (line_atomic_number etc. loaded in load_atomic_data) */

    /* A9: Verify strictly non-ascending order. Sobolev binary search assumes
     * descending nu; a single out-of-order pair silently mis-routes packets
     * onto the wrong line and warps every per-band ratio. Abort hard. */
    int desc_violations = 0;
    int first_bad_i = -1;
    for (int i = 1; i < n; i++) {
        if (opacity->line_list_nu[i] > opacity->line_list_nu[i - 1]) {
            if (first_bad_i < 0) first_bad_i = i;
            desc_violations++;
        }
    }
    if (desc_violations > 0) {
        fprintf(stderr,
                "FATAL: line_list_nu not descending — %d violation(s), first at i=%d "
                "(nu[%d]=%.6e > nu[%d]=%.6e). Sobolev binary search REQUIRES descending.\n"
                "  Regenerate reference with sort_values('nu', ascending=False, kind='stable').\n",
                desc_violations, first_bad_i,
                first_bad_i, opacity->line_list_nu[first_bad_i],
                first_bad_i - 1, opacity->line_list_nu[first_bad_i - 1]);
        return -1;
    }
    printf("  Line order: DESCENDING (correct, %d pairs)\n", n - 1);

    /* Phase 2 - Step 10f: Load tau_sobolev [n_lines, n_shells] */
    snprintf(path, sizeof(path), "%s/tau_sobolev.npy", ref_dir); /* Phase 2 - Step 10f */
    int tr, tc; /* Phase 2 - Step 10f */
    opacity->tau_sobolev = read_npy_f64(path, &tr, &tc); /* Phase 2 - Step 10f */
    printf("  tau_sobolev: [%d x %d] (expect [%d x %d])\n", /* Phase 2 - Step 10f */
           tr, tc, opacity->n_lines, opacity->n_shells); /* Phase 2 - Step 10f */
    if (tr != opacity->n_lines || tc != opacity->n_shells) {
        fprintf(stderr, "WARNING: tau_sobolev [%d x %d] != expected [%d x %d], reinitializing\n",
                tr, tc, opacity->n_lines, opacity->n_shells);
        free(opacity->tau_sobolev);
        opacity->tau_sobolev = (double *)calloc((size_t)opacity->n_lines * opacity->n_shells, sizeof(double));
    }

    /* CMF: per-line NLTE source function, populated during plasma/NLTE update.
     * 0 (calloc default) signals "use fallback" in the CMF solver. */
    opacity->line_source_S = (double *)calloc((size_t)opacity->n_lines * opacity->n_shells, sizeof(double));

    /* Phase 2 - Step 10g: Load transition probabilities [n_trans, n_shells] */
    snprintf(path, sizeof(path), "%s/transition_probabilities.npy", ref_dir); /* Phase 2 - Step 10g */
    opacity->transition_probabilities = read_npy_f64(path, &tr, &tc); /* Phase 2 - Step 10g */
    opacity->n_macro_transitions = tr; /* Phase 2 - Step 10g */
    printf("  transition_probabilities: [%d x %d]\n", tr, tc); /* Phase 2 - Step 10g */
    if (tc != opacity->n_shells) {
        fprintf(stderr, "WARNING: transition_probabilities cols %d != n_shells %d, reinitializing\n",
                tc, opacity->n_shells);
        free(opacity->transition_probabilities);
        opacity->transition_probabilities = (double *)calloc((size_t)tr * opacity->n_shells, sizeof(double));
        /* Initialize with equal branching per block */
    }

    /* Phase 2 - Step 10h: Load macro-atom references */
    snprintf(path, sizeof(path), "%s/macro_atom_references.csv", ref_dir); /* Phase 2 - Step 10h */
    int *block_refs = read_csv_column_int(path, "block_references", &n); /* Phase 2 - Step 10h */
    opacity->n_macro_levels = n; /* Phase 2 - Step 10h */
    /* Phase 2 - Step 10h: Build block_references array [n_levels + 1] */
    opacity->macro_block_references = (int *)malloc((n + 1) * sizeof(int)); /* Phase 2 - Step 10h */
    for (int i = 0; i < n; i++) { /* Phase 2 - Step 10h */
        opacity->macro_block_references[i] = block_refs[i]; /* Phase 2 - Step 10h */
    }
    opacity->macro_block_references[n] = opacity->n_macro_transitions; /* Phase 2 - Step 10h */
    free(block_refs); /* Phase 2 - Step 10h */
    printf("  Macro-atom: %d levels, %d transitions\n", /* Phase 2 - Step 10h */
           opacity->n_macro_levels, opacity->n_macro_transitions); /* Phase 2 - Step 10h */

    /* Phase 2 - Step 10i: Load macro-atom transition data */
    snprintf(path, sizeof(path), "%s/macro_atom_data.csv", ref_dir); /* Phase 2 - Step 10i */
    opacity->transition_type = read_csv_column_int(path, "transition_type", &n); /* Phase 2 - Step 10i */
    opacity->destination_level_id = read_csv_column_int(path, "destination_level_idx", &n); /* Phase 2 - Step 10i */
    opacity->transition_line_id = read_csv_column_int(path, "lines_idx", &n); /* Phase 2 - Step 10i */
    printf("  Macro transitions loaded: %d entries\n", n); /* Phase 2 - Step 10i */

    /* Phase 2 - Step 10j: Load line2macro_level_upper */
    snprintf(path, sizeof(path), "%s/line2macro_level_upper.npy", ref_dir); /* Phase 2 - Step 10j */
    opacity->line2macro_level_upper = read_npy_int(path, &n); /* Phase 2 - Step 10j */
    printf("  line2macro_level_upper: %d entries\n", n); /* Phase 2 - Step 10j */

    /* k-packet tables: lazily built by compute_transition_probabilities when
     * LUMINA_KPACKET is enabled; NULL until then. */
    opacity->p_kpacket = NULL;
    opacity->kpacket_cdf = NULL;

    printf("Data loading complete.\n"); /* Phase 2 - Step 10 */
    return 0; /* Phase 2 - Step 10 */
}

/* ============================================================ */
/* Phase 2 - Step 11: Memory management                         */
/* ============================================================ */

void free_geometry(Geometry *geo) { /* Phase 2 - Step 11 */
    free(geo->r_inner); /* Phase 2 - Step 11 */
    free(geo->r_outer); /* Phase 2 - Step 11 */
    free(geo->v_inner); /* Phase 2 - Step 11 */
    free(geo->v_outer); /* Phase 2 - Step 11 */
}

void free_opacity_state(OpacityState *op) { /* Phase 2 - Step 11 */
    free(op->line_list_nu); /* Phase 2 - Step 11 */
    free(op->tau_sobolev); /* Phase 2 - Step 11 */
    free(op->line_source_S);
    free(op->electron_density); /* Phase 2 - Step 11 */
    free(op->t_electrons); /* Phase 2 - Step 11 */
    free(op->macro_block_references); /* Phase 2 - Step 11 */
    free(op->transition_type); /* Phase 2 - Step 11 */
    free(op->destination_level_id); /* Phase 2 - Step 11 */
    free(op->transition_line_id); /* Phase 2 - Step 11 */
    free(op->transition_probabilities); /* Phase 2 - Step 11 */
    free(op->line2macro_level_upper); /* Phase 2 - Step 11 */
}

void free_plasma_state(PlasmaState *ps) { /* Phase 2 - Step 11 */
    free(ps->W); /* Phase 2 - Step 11 */
    free(ps->T_rad); /* Phase 2 - Step 11 */
    free(ps->rho); /* Phase 2 - Step 11 */
    free(ps->n_electron); /* Task #072 */
    free(ps->T_e); /* P6: per-shell electron temperature */
}

Estimators *create_estimators(int n_shells, int n_lines) { /* Phase 2 - Step 11 */
    Estimators *est = (Estimators *)calloc(1, sizeof(Estimators)); /* Phase 2 - Step 11 */
    est->n_shells = n_shells; /* Phase 2 - Step 11 */
    est->n_lines = n_lines; /* Phase 2 - Step 11 */
    est->j_estimator = (double *)calloc(n_shells, sizeof(double)); /* Phase 2 - Step 11 */
    est->nu_bar_estimator = (double *)calloc(n_shells, sizeof(double)); /* Phase 2 - Step 11 */
    est->j_blue_estimator = (double *)calloc((size_t)n_lines * n_shells, sizeof(double)); /* Phase 2 - Step 11 */
    est->Edotlu_estimator = (double *)calloc((size_t)n_lines * n_shells, sizeof(double)); /* Phase 2 - Step 11 */
    return est; /* Phase 2 - Step 11 */
}

void reset_estimators(Estimators *est) { /* Phase 2 - Step 11 */
    memset(est->j_estimator, 0, est->n_shells * sizeof(double)); /* Phase 2 - Step 11 */
    memset(est->nu_bar_estimator, 0, est->n_shells * sizeof(double)); /* Phase 2 - Step 11 */
    memset(est->j_blue_estimator, 0, (size_t)est->n_lines * est->n_shells * sizeof(double)); /* Phase 2 - Step 11 */
    memset(est->Edotlu_estimator, 0, (size_t)est->n_lines * est->n_shells * sizeof(double)); /* Phase 2 - Step 11 */
}

void free_estimators(Estimators *est) { /* Phase 2 - Step 11 */
    free(est->j_estimator); /* Phase 2 - Step 11 */
    free(est->nu_bar_estimator); /* Phase 2 - Step 11 */
    free(est->j_blue_estimator); /* Phase 2 - Step 11 */
    free(est->Edotlu_estimator); /* Phase 2 - Step 11 */
    free(est); /* Phase 2 - Step 11 */
}

Spectrum *create_spectrum(double lambda_min, double lambda_max, int n_bins) { /* Phase 2 - Step 11 */
    Spectrum *spec = (Spectrum *)calloc(1, sizeof(Spectrum)); /* Phase 2 - Step 11 */
    spec->n_bins = n_bins; /* Phase 2 - Step 11 */
    spec->lambda_min = lambda_min; /* Phase 2 - Step 11 */
    spec->lambda_max = lambda_max; /* Phase 2 - Step 11 */
    spec->flux = (double *)calloc(n_bins, sizeof(double)); /* Phase 2 - Step 11 */
    spec->wavelength = (double *)malloc(n_bins * sizeof(double)); /* Phase 2 - Step 11 */
    double dlambda = (lambda_max - lambda_min) / n_bins; /* Phase 2 - Step 11 */
    for (int i = 0; i < n_bins; i++) { /* Phase 2 - Step 11 */
        spec->wavelength[i] = lambda_min + (i + 0.5) * dlambda; /* Phase 2 - Step 11 */
    }
    return spec; /* Phase 2 - Step 11 */
}

void reset_spectrum(Spectrum *spec) { /* Phase 2 - Step 11 */
    memset(spec->flux, 0, spec->n_bins * sizeof(double)); /* Phase 2 - Step 11 */
}

void free_spectrum(Spectrum *spec) { /* Phase 2 - Step 11 */
    free(spec->flux); /* Phase 2 - Step 11 */
    free(spec->wavelength); /* Phase 2 - Step 11 */
    free(spec); /* Phase 2 - Step 11 */
}

/* ============================================================ */
/* Task #072: Load atomic data for plasma solver                 */
/* ============================================================ */

int load_atomic_data(AtomicData *atom, const char *ref_dir, int n_shells) {
    char path[512];
    int n;

    memset(atom, 0, sizeof(AtomicData));
    printf("\nLoading atomic data for plasma solver...\n");

    /* --- Line columns from line_list.csv --- */
    snprintf(path, sizeof(path), "%s/line_list.csv", ref_dir);
    atom->line_atomic_number = read_csv_column_int(path, "atomic_number", &n);
    atom->line_ion_number    = read_csv_column_int(path, "ion_number", &n);
    atom->line_level_lower   = read_csv_column_int(path, "level_number_lower", &n);
    atom->line_level_upper   = read_csv_column_int(path, "level_number_upper", &n);
    atom->line_f_lu          = read_csv_column(path, "f_lu", &n);
    atom->line_wavelength_cm = read_csv_column(path, "wavelength_cm", &n);

    /* NLTE: Einstein coefficients and line frequencies */
    atom->line_A_ul = read_csv_column(path, "A_ul", &n);
    atom->line_B_lu = read_csv_column(path, "B_lu", &n);
    atom->line_B_ul = read_csv_column(path, "B_ul", &n);
    atom->line_nu   = read_csv_column(path, "nu", &n);
    atom->n_lines   = n;
    printf("  Line columns: %d lines loaded (including A_ul, B_lu, B_ul, nu)\n", n);

    /* LUMINA_AUL_SCALE[N]_*: per-(Z,ion,λ-band) A_ul/f_lu/B_lu/B_ul scale.
     * Pass N="" (primary) and N="2","3" (stacked) for independent per-species rules.
     *   LUMINA_AUL_SCALE[N]_FACTOR    multiplier (default 1.0 = off)
     *   LUMINA_AUL_SCALE[N]_ZMASK     comma-list of Z (default: all)
     *   LUMINA_AUL_SCALE[N]_IONMASK   comma-list of ion stages (default: II)
     *   LUMINA_AUL_SCALE[N]_LAMBDA_MIN  Å, scale lines with λ >= this (default: 0)
     *   LUMINA_AUL_SCALE[N]_LAMBDA_MAX  Å, scale lines with λ < this  (default: 4000)
     */
    {
        const char *suffixes[] = {"", "2", "3", "4", "5", "6", "7", "8", "9"};
        for (int s = 0; s < 9; s++) {
            char name[64];
            snprintf(name, sizeof(name), "LUMINA_AUL_SCALE%s_FACTOR", suffixes[s]);
            const char *e_fac = getenv(name);
            if (!e_fac || !(atof(e_fac) > 0.0) || atof(e_fac) == 1.0) continue;
            double fac = atof(e_fac);
            unsigned int zmask = 0;
            unsigned int imask = (1u << 1);
            double lam_min_A = 0.0;
            double lam_max_A = 4000.0;
            snprintf(name, sizeof(name), "LUMINA_AUL_SCALE%s_ZMASK", suffixes[s]);
            const char *e_z = getenv(name);
            snprintf(name, sizeof(name), "LUMINA_AUL_SCALE%s_IONMASK", suffixes[s]);
            const char *e_ion = getenv(name);
            snprintf(name, sizeof(name), "LUMINA_AUL_SCALE%s_LAMBDA_MIN", suffixes[s]);
            const char *e_lmn = getenv(name);
            snprintf(name, sizeof(name), "LUMINA_AUL_SCALE%s_LAMBDA_MAX", suffixes[s]);
            const char *e_lm = getenv(name);
            if (e_z) {
                char buf[256]; strncpy(buf, e_z, 255); buf[255]=0;
                for (char *p = strtok(buf, ", "); p; p = strtok(NULL, ", ")) {
                    int z = atoi(p);
                    if (z > 0 && z < 32) zmask |= (1u << z);
                }
            } else {
                for (int z = 1; z < 32; z++) zmask |= (1u << z);
            }
            if (e_ion) {
                char buf[64]; strncpy(buf, e_ion, 63); buf[63]=0;
                imask = 0;
                for (char *p = strtok(buf, ", "); p; p = strtok(NULL, ", ")) {
                    int ii = atoi(p);
                    if (ii >= 0 && ii < 8) imask |= (1u << ii);
                }
            }
            if (e_lmn) lam_min_A = atof(e_lmn);
            if (e_lm) lam_max_A = atof(e_lm);
            int nhit = 0;
            for (int i = 0; i < atom->n_lines; i++) {
                int Z = atom->line_atomic_number[i];
                int ion = atom->line_ion_number[i];
                double lam_A = atom->line_wavelength_cm[i] * 1e8;
                if (lam_A >= lam_min_A && lam_A < lam_max_A &&
                    Z >= 0 && Z < 32 && ion >= 0 && ion < 8 &&
                    (zmask & (1u << Z)) && (imask & (1u << ion))) {
                    atom->line_A_ul[i] *= fac;
                    atom->line_B_lu[i] *= fac;
                    atom->line_B_ul[i] *= fac;
                    atom->line_f_lu[i] *= fac;
                    nhit++;
                }
            }
            printf("  [AUL_SCALE%s] fac=%.3f λ∈[%.0f,%.0f)Å zmask=0x%x imask=0x%x → %d lines scaled\n",
                   suffixes[s], fac, lam_min_A, lam_max_A, zmask, imask, nhit);
        }
    }

    /* --- Level data from levels.csv --- */
    snprintf(path, sizeof(path), "%s/levels.csv", ref_dir);
    atom->level_Z          = read_csv_column_int(path, "atomic_number", &n);
    atom->level_ion        = read_csv_column_int(path, "ion_number", &n);
    atom->level_num        = read_csv_column_int(path, "level_number", &n);
    atom->level_energy_eV  = read_csv_column(path, "energy_eV", &n);
    atom->level_g          = read_csv_column_int(path, "g", &n);
    atom->level_metastable = read_csv_column_int(path, "metastable", &n);
    atom->n_levels = n;
    /* Super-level index (CMFGEN f_to_s). Optional column: older references
     * lack it -> default to identity (each full level is its own super level)
     * so the NLTE solve reproduces the level-truncation behaviour. */
    atom->level_super = read_csv_column_int(path, "super_level", &n);
    if (atom->level_super == NULL) {
        atom->level_super = (int *)malloc((size_t)atom->n_levels * sizeof(int));
        for (int l = 0; l < atom->n_levels; l++)
            atom->level_super[l] = atom->level_num[l];
        printf("  Levels: %d loaded (no super_level column -> identity)\n",
               atom->n_levels);
    } else {
        int super_active = 0;
        for (int l = 0; l < atom->n_levels; l++) {
            if (atom->level_super[l] != atom->level_num[l]) { super_active = 1; break; }
        }
        printf("  Levels: %d loaded (super_level column present%s)\n",
               atom->n_levels,
               super_active ? ", super-levels active" : ", all identity");
    }

    /* --- Ionization energies --- */
    snprintf(path, sizeof(path), "%s/ionization_energies.csv", ref_dir);
    atom->ioniz_Z         = read_csv_column_int(path, "atomic_number", &n);
    atom->ioniz_ion       = read_csv_column_int(path, "ion_number", &n);
    atom->ioniz_energy_eV = read_csv_column(path, "ionization_energy_eV", &n);
    atom->n_ionization = n;
    printf("  Ionization: %d entries\n", n);

    /* --- Zeta data --- */
    snprintf(path, sizeof(path), "%s/zeta_ions.csv", ref_dir);
    atom->zeta_Z   = read_csv_column_int(path, "atomic_number", &n);
    atom->zeta_ion = read_csv_column_int(path, "ion_number", &n);
    atom->n_zeta_ions = n;

    snprintf(path, sizeof(path), "%s/zeta_temps.csv", ref_dir);
    atom->zeta_temps = read_csv_column(path, "temperature", &n);
    atom->n_zeta_temps = n;

    snprintf(path, sizeof(path), "%s/zeta_data.npy", ref_dir);
    int zr, zc;
    atom->zeta_data = read_npy_f64(path, &zr, &zc);
    printf("  Zeta: %d ions x %d temps, data [%d x %d]\n",
           atom->n_zeta_ions, atom->n_zeta_temps, zr, zc);

    /* --- Atom masses --- */
    snprintf(path, sizeof(path), "%s/atom_masses.csv", ref_dir);
    atom->element_Z        = read_csv_column_int(path, "atomic_number", &n);
    atom->element_mass_amu = read_csv_column(path, "mass_amu", &n);
    atom->n_elements = n;
    printf("  Elements: %d (", n);
    for (int i = 0; i < n; i++) printf("%s%d", i ? "," : "", atom->element_Z[i]);
    printf(")\n");

    /* --- Abundances --- */
    snprintf(path, sizeof(path), "%s/abundances.csv", ref_dir);
    atom->abundances = (double *)calloc((size_t)atom->n_elements * n_shells, sizeof(double));
    FILE *fp = fopen(path, "r");
    if (fp) {
        char line[8192];
        fgets(line, sizeof(line), fp); /* skip header */
        int elem_idx = 0;
        while (fgets(line, sizeof(line), fp) && elem_idx < atom->n_elements) {
            /* format: atomic_number,shell0,shell1,...,shell29 */
            char *p = line;
            int z_csv = (int)strtol(p, &p, 10);
            /* Find matching element index */
            int eidx = -1;
            for (int i = 0; i < atom->n_elements; i++) {
                if (atom->element_Z[i] == z_csv) { eidx = i; break; }
            }
            if (eidx < 0) continue;
            for (int s = 0; s < n_shells; s++) {
                if (*p == ',') p++;
                atom->abundances[eidx * n_shells + s] = strtod(p, &p);
            }
            elem_idx++;
        }
        fclose(fp);
    }

    /* --- Build ion population table --- */
    /* For each element, ion stages go from 0 to n_ionization_entries_for_element */
    /* Count total ion populations */
    atom->elem_ion_offset = (int *)calloc(atom->n_elements + 1, sizeof(int));
    int total_ion_pops = 0;
    for (int e = 0; e < atom->n_elements; e++) {
        int z = atom->element_Z[e];
        int n_ioniz = 0;
        for (int i = 0; i < atom->n_ionization; i++) {
            if (atom->ioniz_Z[i] == z) n_ioniz++;
        }
        atom->elem_ion_offset[e] = total_ion_pops;
        total_ion_pops += n_ioniz + 1; /* n_ioniz energies -> n_ioniz+1 populations */
    }
    atom->elem_ion_offset[atom->n_elements] = total_ion_pops;
    atom->n_ion_pops = total_ion_pops;

    atom->ion_pop_Z     = (int *)calloc(total_ion_pops, sizeof(int));
    atom->ion_pop_stage = (int *)calloc(total_ion_pops, sizeof(int));
    for (int e = 0; e < atom->n_elements; e++) {
        int z = atom->element_Z[e];
        int n_pops = atom->elem_ion_offset[e + 1] - atom->elem_ion_offset[e];
        /* The ion ladder starts at the LOWEST stage that has atomic data, not
           necessarily neutral. CMFGEN omits neutral Ti I / Mn I, so their
           ionization-energy entries start at stage 1; labelling pops relatively
           (k) dumped all mass into a phantom level-less neutral slot. Anchor the
           ladder at base_stage = min ionization-energy stage for the element. */
        int base_stage = 0;
        int found_base = 0;
        for (int i = 0; i < atom->n_ionization; i++) {
            if (atom->ioniz_Z[i] == z &&
                (!found_base || atom->ioniz_ion[i] < base_stage)) {
                base_stage = atom->ioniz_ion[i];
                found_base = 1;
            }
        }
        /* found_base==0 (no ioniz data): leave base_stage=0 (assume neutral) */
        for (int k = 0; k < n_pops; k++) {
            int idx = atom->elem_ion_offset[e] + k;
            atom->ion_pop_Z[idx] = z;
            atom->ion_pop_stage[idx] = base_stage + k;
        }
    }
    printf("  Ion populations: %d total\n", total_ion_pops);

    /* --- Build level lookup: level_offset[ion_pop_idx] --- */
    /* Levels are sorted by (Z, ion, level_num) in levels.csv */
    atom->level_offset = (int *)calloc(total_ion_pops + 1, sizeof(int));
    for (int ip = 0; ip < total_ion_pops; ip++) {
        int z = atom->ion_pop_Z[ip];
        int ion_stage = atom->ion_pop_stage[ip];
        int count = 0;
        for (int l = 0; l < atom->n_levels; l++) {
            if (atom->level_Z[l] == z && atom->level_ion[l] == ion_stage) count++;
        }
        atom->level_offset[ip + 1] = atom->level_offset[ip] + count;
    }
    printf("  Level offsets built: %d total levels mapped\n",
           atom->level_offset[total_ion_pops]);

    /* Verify level_offset total matches n_levels */
    if (atom->level_offset[total_ion_pops] != atom->n_levels) {
        fprintf(stderr, "WARNING: level_offset total %d != n_levels %d\n",
                atom->level_offset[total_ion_pops], atom->n_levels);
    }

    /* --- Allocate per-shell computed arrays --- */
    atom->ion_number_density  = (double *)calloc((size_t)total_ion_pops * n_shells, sizeof(double));
    atom->partition_functions = (double *)calloc((size_t)total_ion_pops * n_shells, sizeof(double));

    printf("Atomic data loading complete.\n");
    return 0;
}

/* Load pre-baked CMFGEN sigma_bf grid (cmfgen_sigma_bf.bin).
 * Binary layout (little-endian, written by scripts/expand_atomic_data_cmfgen.py):
 *   uint32 magic   = 0x434D4644 ('CMFD')
 *   uint32 version = 1
 *   int32  n_levels, n_freq_bins
 *   double nu_min_Hz, nu_max_Hz
 *   int8   has_cmfgen[n_levels]   (padded to 8-byte alignment)
 *   double sigma_cm2[n_levels * n_freq_bins]
 *
 * Returns 0 on success (grid loaded into atom->cmfgen_*), -1 on missing file
 * or schema mismatch (atom->cmfgen_loaded stays 0 → Kramers fallback). */
int load_cmfgen_sigma_bf(AtomicData *atom, const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        printf("  CMFGEN sigma_bf: %s not found, using Kramers fallback\n", path);
        atom->cmfgen_loaded = 0;
        return -1;
    }
    uint32_t magic = 0, version = 0;
    int32_t n_lev = 0, n_freq = 0;
    double nu_min = 0.0, nu_max = 0.0;
    if (fread(&magic, 4, 1, fp) != 1 || fread(&version, 4, 1, fp) != 1 ||
        fread(&n_lev, 4, 1, fp) != 1 || fread(&n_freq, 4, 1, fp) != 1 ||
        fread(&nu_min, 8, 1, fp) != 1 || fread(&nu_max, 8, 1, fp) != 1) {
        fprintf(stderr, "ERROR: %s header read failed\n", path);
        fclose(fp);
        return -1;
    }
    if (magic != 0x434D4644u || version != 1u) {
        fprintf(stderr, "ERROR: %s bad magic/version (0x%08x v%u)\n",
                path, magic, version);
        fclose(fp);
        return -1;
    }
    if (n_lev != atom->n_levels) {
        fprintf(stderr, "ERROR: %s n_levels=%d != atom->n_levels=%d\n",
                path, n_lev, atom->n_levels);
        fclose(fp);
        return -1;
    }
    if (n_freq != NLTE_N_FREQ_BINS) {
        fprintf(stderr, "ERROR: %s n_freq_bins=%d != NLTE_N_FREQ_BINS=%d\n",
                path, n_freq, NLTE_N_FREQ_BINS);
        fclose(fp);
        return -1;
    }

    /* Read has_cmfgen[n_levels] as int8 then promote to int */
    int8_t *flag8 = (int8_t *)malloc((size_t)n_lev);
    if (fread(flag8, 1, (size_t)n_lev, fp) != (size_t)n_lev) {
        fprintf(stderr, "ERROR: %s has_cmfgen read failed\n", path);
        free(flag8);
        fclose(fp);
        return -1;
    }
    /* Skip 8-byte alignment pad */
    int pad = (8 - (n_lev % 8)) % 8;
    if (pad > 0) fseek(fp, pad, SEEK_CUR);

    atom->cmfgen_has_sigma = (int *)malloc((size_t)n_lev * sizeof(int));
    int n_with = 0;
    for (int i = 0; i < n_lev; i++) {
        atom->cmfgen_has_sigma[i] = flag8[i];
        if (flag8[i]) n_with++;
    }
    free(flag8);

    size_t grid_n = (size_t)n_lev * (size_t)n_freq;
    atom->cmfgen_sigma_bf = (double *)malloc(grid_n * sizeof(double));
    if (fread(atom->cmfgen_sigma_bf, sizeof(double), grid_n, fp) != grid_n) {
        fprintf(stderr, "ERROR: %s sigma grid read failed\n", path);
        free(atom->cmfgen_has_sigma);
        free(atom->cmfgen_sigma_bf);
        atom->cmfgen_has_sigma = NULL;
        atom->cmfgen_sigma_bf = NULL;
        fclose(fp);
        return -1;
    }
    fclose(fp);

    atom->cmfgen_n_freq_bins = n_freq;
    atom->cmfgen_nu_min = nu_min;
    atom->cmfgen_nu_max = nu_max;
    atom->cmfgen_loaded = 1;
    printf("  CMFGEN sigma_bf: %d/%d levels (%.1f%%) on %d-bin grid\n",
           n_with, n_lev, 100.0 * n_with / (double)n_lev, n_freq);
    return 0;
}

/* TOP-STAGE CONTINUUM ANCHOR (LUMINA_TOPSTAGE_ANCHOR=1): inject synthetic
 * ground-only IV ion stages so the top NLTE stage III gets a bound-free
 * continuum partner. Without it, top-stage III EXCITED levels have zero
 * continuum coupling and float super-thermal (no recombination drain) ->
 * featureless emergent spectrum. The (III,IV) recombination is the missing
 * thermalizing anchor (validated: thermal levels -> MC features; FORCE_LTE 166340).
 *
 * APPEND at the end of the level arrays: global level indices are UNCHANGED so
 * existing line/macro-transition references stay intact (vs an insert, which
 * would shift them). The synthetic levels are NLTE-scan orphans -- nlte_init
 * finds them by (Z,ion) scan, independent of atom->level_offset (left stale; the
 * bf recomb reads g_ion from level_g directly, so the unused IV partition fn is
 * harmless). Kramers sigma_bf (cmfgen_has_sigma=0). KPACKET must be off (synthetic
 * levels exceed n_macro_levels-sized kpacket arrays; empty macro block otherwise).
 *
 * Increment 1: O IV only (Z=8, ground 2s2 2p 2P, g=6). O is the last NLTE element
 * (slots 28-30 = O I/II/III) so O IV at slot 31 is adjacent (hi=lo+1) -- no
 * explicit-hi pair plumbing needed. S IV/C IV (mid-table III) come next with
 * explicit pair-hi. Gated default-off. */
void inject_topstage_continuum_levels(AtomicData *atom, OpacityState *opacity) {
    const char *e = getenv("LUMINA_TOPSTAGE_ANCHOR");
    if (!(e && atoi(e) != 0)) return;
    /* {Z, ion(0-based, 3=IV), ground g} */
    int syn_Z[]  = {8};
    int syn_ion[]= {3};
    int syn_g[]  = {6};       /* O IV 2s2 2p 2P deg = 6 */
    int n_syn = (int)(sizeof(syn_Z)/sizeof(syn_Z[0]));
    int old_n = atom->n_levels, new_n = old_n + n_syn;

    atom->level_Z          = (int *)   realloc(atom->level_Z,          (size_t)new_n*sizeof(int));
    atom->level_ion        = (int *)   realloc(atom->level_ion,        (size_t)new_n*sizeof(int));
    atom->level_num        = (int *)   realloc(atom->level_num,        (size_t)new_n*sizeof(int));
    atom->level_energy_eV  = (double *)realloc(atom->level_energy_eV,  (size_t)new_n*sizeof(double));
    atom->level_g          = (int *)   realloc(atom->level_g,          (size_t)new_n*sizeof(int));
    atom->level_metastable = (int *)   realloc(atom->level_metastable, (size_t)new_n*sizeof(int));
    atom->level_super      = (int *)   realloc(atom->level_super,      (size_t)new_n*sizeof(int));
    atom->cmfgen_has_sigma = (int *)   realloc(atom->cmfgen_has_sigma, (size_t)new_n*sizeof(int));
    for (int s = 0; s < n_syn; s++) {
        int l = old_n + s;
        atom->level_Z[l] = syn_Z[s];   atom->level_ion[l] = syn_ion[s];
        atom->level_num[l] = 0;        atom->level_energy_eV[l] = 0.0;
        atom->level_g[l] = syn_g[s];   atom->level_metastable[l] = 1;
        atom->level_super[l] = 0;      atom->cmfgen_has_sigma[l] = 0;
    }
    if (atom->cmfgen_sigma_bf && atom->cmfgen_n_freq_bins > 0) {
        size_t nf = (size_t)atom->cmfgen_n_freq_bins;
        atom->cmfgen_sigma_bf = (double *)realloc(atom->cmfgen_sigma_bf,
                                                  (size_t)new_n*nf*sizeof(double));
        memset(&atom->cmfgen_sigma_bf[(size_t)old_n*nf], 0, (size_t)n_syn*nf*sizeof(double));
    }
    atom->n_levels = new_n;

    if (opacity && opacity->macro_block_references) {
        int sentinel = opacity->macro_block_references[opacity->n_macro_levels];
        opacity->macro_block_references = (int *)realloc(opacity->macro_block_references,
                                                         (size_t)(new_n + 1)*sizeof(int));
        for (int s = 0; s <= n_syn; s++)
            opacity->macro_block_references[old_n + s] = sentinel;  /* empty blocks */
        opacity->n_macro_levels = new_n;
    }
    printf("  [TOPSTAGE-ANCHOR] injected %d synthetic IV ground level(s) (O IV g=6); "
           "n_levels %d->%d\n", n_syn, old_n, new_n);
}

void free_atomic_data(AtomicData *atom) {
    free(atom->line_atomic_number);
    free(atom->line_ion_number);
    free(atom->line_level_lower);
    free(atom->line_level_upper);
    free(atom->line_f_lu);
    free(atom->line_wavelength_cm);
    free(atom->line_A_ul);
    free(atom->line_B_lu);
    free(atom->line_B_ul);
    free(atom->line_nu);
    free(atom->level_Z);
    free(atom->level_ion);
    free(atom->level_num);
    free(atom->level_energy_eV);
    free(atom->level_g);
    free(atom->level_metastable);
    free(atom->level_super);
    free(atom->ioniz_Z);
    free(atom->ioniz_ion);
    free(atom->ioniz_energy_eV);
    free(atom->zeta_Z);
    free(atom->zeta_ion);
    free(atom->zeta_data);
    free(atom->zeta_temps);
    free(atom->element_Z);
    free(atom->element_mass_amu);
    free(atom->abundances);
    free(atom->elem_ion_offset);
    free(atom->ion_pop_Z);
    free(atom->ion_pop_stage);
    free(atom->level_offset);
    free(atom->ion_number_density);
    free(atom->partition_functions);
    free(atom->cmfgen_has_sigma);
    free(atom->cmfgen_sigma_bf);
}

/* ============================================================ */
/* Phase 2 - Step 12: RNG implementation (xoshiro256**)         */
/* ============================================================ */

/* Phase 2 - Step 12: SplitMix64 for seeding */
static uint64_t splitmix64(uint64_t *state) { /* Phase 2 - Step 12 */
    uint64_t z = (*state += 0x9e3779b97f4a7c15ULL); /* Phase 2 - Step 12 */
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL; /* Phase 2 - Step 12 */
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL; /* Phase 2 - Step 12 */
    return z ^ (z >> 31); /* Phase 2 - Step 12 */
}

void rng_init(RNG *rng, uint64_t seed) { /* Phase 2 - Step 12 */
    uint64_t s = seed; /* Phase 2 - Step 12 */
    rng->s[0] = splitmix64(&s); /* Phase 2 - Step 12 */
    rng->s[1] = splitmix64(&s); /* Phase 2 - Step 12 */
    rng->s[2] = splitmix64(&s); /* Phase 2 - Step 12 */
    rng->s[3] = splitmix64(&s); /* Phase 2 - Step 12 */
}

/* Phase 2 - Step 12: Rotate left helper */
static inline uint64_t rotl(const uint64_t x, int k) { /* Phase 2 - Step 12 */
    return (x << k) | (x >> (64 - k)); /* Phase 2 - Step 12 */
}

double rng_uniform(RNG *rng) { /* Phase 2 - Step 12 */
    const uint64_t result = rotl(rng->s[1] * 5, 7) * 9; /* Phase 2 - Step 12 */
    const uint64_t t = rng->s[1] << 17; /* Phase 2 - Step 12 */
    rng->s[2] ^= rng->s[0]; /* Phase 2 - Step 12 */
    rng->s[3] ^= rng->s[1]; /* Phase 2 - Step 12 */
    rng->s[1] ^= rng->s[2]; /* Phase 2 - Step 12 */
    rng->s[0] ^= rng->s[3]; /* Phase 2 - Step 12 */
    rng->s[2] ^= t; /* Phase 2 - Step 12 */
    rng->s[3] = rotl(rng->s[3], 45); /* Phase 2 - Step 12 */
    return (result >> 11) * 0x1.0p-53; /* Phase 2 - Step 12: [0, 1) */
}

double rng_mu(RNG *rng) { /* Phase 2 - Step 12 */
    return 2.0 * rng_uniform(rng) - 1.0; /* Phase 2 - Step 12: [-1, 1) */
}

#ifdef __cplusplus   /* Phase 6 - Step 9: close extern C guard */
}                    /* Phase 6 - Step 9 */
#endif               /* Phase 6 - Step 9 */
