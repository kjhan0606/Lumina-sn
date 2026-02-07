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

    /* Phase 2 - Step 8: Read magic number */
    unsigned char magic[6]; /* Phase 2 - Step 8 */
    fread(magic, 1, 6, fp); /* Phase 2 - Step 8 */
    if (magic[0] != 0x93 || magic[1] != 'N' || magic[2] != 'U' || /* Phase 2 - Step 8 */
        magic[3] != 'M' || magic[4] != 'P' || magic[5] != 'Y') { /* Phase 2 - Step 8 */
        fprintf(stderr, "ERROR: %s is not a valid .npy file\n", path); /* Phase 2 - Step 8 */
        fclose(fp); /* Phase 2 - Step 8 */
        return NULL; /* Phase 2 - Step 8 */
    }

    /* Phase 2 - Step 8: Read version */
    unsigned char version[2]; /* Phase 2 - Step 8 */
    fread(version, 1, 2, fp); /* Phase 2 - Step 8 */

    /* Phase 2 - Step 8: Read header length */
    uint16_t header_len; /* Phase 2 - Step 8 */
    if (version[0] == 1) { /* Phase 2 - Step 8 */
        fread(&header_len, 2, 1, fp); /* Phase 2 - Step 8 */
    } else { /* Phase 2 - Step 8 */
        uint32_t hl32; /* Phase 2 - Step 8 */
        fread(&hl32, 4, 1, fp); /* Phase 2 - Step 8 */
        header_len = (uint16_t)hl32; /* Phase 2 - Step 8 */
    }

    /* Phase 2 - Step 8: Read header string */
    char *header = (char *)malloc(header_len + 1); /* Phase 2 - Step 8 */
    fread(header, 1, header_len, fp); /* Phase 2 - Step 8 */
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
        fread(idata, sizeof(int64_t), total, fp); /* Phase 2 - Step 8 */
        for (int i = 0; i < total; i++) { /* Phase 2 - Step 8 */
            data[i] = (double)idata[i]; /* Phase 2 - Step 8 */
        }
        free(idata); /* Phase 2 - Step 8 */
    } else { /* Phase 2 - Step 8: Read as float64 directly */
        fread(data, sizeof(double), total, fp); /* Phase 2 - Step 8 */
    }

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
    if (fp) { /* Phase 2 - Step 10b */
        char buf[4096]; /* Phase 2 - Step 10b */
        size_t nr = fread(buf, 1, sizeof(buf) - 1, fp); /* Phase 2 - Step 10b */
        buf[nr] = '\0'; /* Phase 2 - Step 10b */
        fclose(fp); /* Phase 2 - Step 10b */

        /* Phase 2 - Step 10b: Parse JSON manually (simple flat struct) */
        char *p; /* Phase 2 - Step 10b */
        p = strstr(buf, "\"time_explosion_s\""); /* Phase 2 - Step 10b */
        if (p) { p = strchr(p, ':'); geo->time_explosion = atof(p + 1); } /* Phase 2 - Step 10b */
        p = strstr(buf, "\"T_inner_K\""); /* Phase 2 - Step 10b */
        if (p) { p = strchr(p, ':'); config->T_inner = atof(p + 1); } /* Phase 2 - Step 10b */
        p = strstr(buf, "\"luminosity_inner_erg_s\""); /* Phase 2 - Step 10b */
        if (p) { p = strchr(p, ':'); config->luminosity_requested = atof(p + 1); } /* Phase 2 - Step 10b */
        p = strstr(buf, "\"n_packets\""); /* Phase 2 - Step 10b */
        if (p) { p = strchr(p, ':'); config->n_packets = atoi(p + 1); } /* Phase 2 - Step 10b */
        p = strstr(buf, "\"n_iterations\""); /* Phase 2 - Step 10b */
        if (p) { p = strchr(p, ':'); config->n_iterations = atoi(p + 1); } /* Phase 2 - Step 10b */
        p = strstr(buf, "\"seed\""); /* Phase 2 - Step 10b */
        if (p) { p = strchr(p, ':'); config->seed = (uint64_t)atol(p + 1); } /* Phase 2 - Step 10b */

        printf("  Config: t_exp=%.6e s, T_inner=%.2f K, L=%.3e erg/s\n", /* Phase 2 - Step 10b */
               geo->time_explosion, config->T_inner, config->luminosity_requested); /* Phase 2 - Step 10b */
        printf("    n_packets=%d, n_iter=%d, seed=%lu\n", /* Phase 2 - Step 10b */
               config->n_packets, config->n_iterations, config->seed); /* Phase 2 - Step 10b */
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

    /* Phase 2 - Step 10e2: Verify descending order */
    int desc_ok = 1; /* Phase 2 - Step 10e2 */
    for (int i = 1; i < n; i++) { /* Phase 2 - Step 10e2 */
        if (opacity->line_list_nu[i] > opacity->line_list_nu[i - 1]) { /* Phase 2 - Step 10e2 */
            desc_ok = 0; /* Phase 2 - Step 10e2 */
            fprintf(stderr, "WARNING: line_list_nu not descending at i=%d\n", i); /* Phase 2 - Step 10e2 */
            break; /* Phase 2 - Step 10e2 */
        }
    }
    printf("  Line order: %s\n", desc_ok ? "DESCENDING (correct)" : "NOT DESCENDING"); /* Phase 2 - Step 10e2 */

    /* Phase 2 - Step 10f: Load tau_sobolev [n_lines, n_shells] */
    snprintf(path, sizeof(path), "%s/tau_sobolev.npy", ref_dir); /* Phase 2 - Step 10f */
    int tr, tc; /* Phase 2 - Step 10f */
    opacity->tau_sobolev = read_npy_f64(path, &tr, &tc); /* Phase 2 - Step 10f */
    printf("  tau_sobolev: [%d x %d] (expect [%d x %d])\n", /* Phase 2 - Step 10f */
           tr, tc, opacity->n_lines, opacity->n_shells); /* Phase 2 - Step 10f */
    if (tr != opacity->n_lines || tc != opacity->n_shells) { /* Phase 2 - Step 10f */
        fprintf(stderr, "ERROR: tau_sobolev shape mismatch!\n"); /* Phase 2 - Step 10f */
        return -1; /* Phase 2 - Step 10f */
    }

    /* Phase 2 - Step 10g: Load transition probabilities [n_trans, n_shells] */
    snprintf(path, sizeof(path), "%s/transition_probabilities.npy", ref_dir); /* Phase 2 - Step 10g */
    opacity->transition_probabilities = read_npy_f64(path, &tr, &tc); /* Phase 2 - Step 10g */
    opacity->n_macro_transitions = tr; /* Phase 2 - Step 10g */
    printf("  transition_probabilities: [%d x %d]\n", tr, tc); /* Phase 2 - Step 10g */

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
