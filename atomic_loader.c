/**
 * LUMINA-SN Atomic Data HDF5 Loader
 * atomic_loader.c - Load TARDIS atomic database from HDF5 format
 *
 * Reads pandas DataFrames stored in HDF5 format using the "table" layout.
 * Handles multi-index structures and converts to flat C arrays.
 *
 * Compile with: -I$HOME/local/include -L$HOME/local/lib -lhdf5 -lhdf5_hl
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <hdf5.h>
#include <hdf5_hl.h>
#include "atomic_data.h"

/* ============================================================================
 * CONSTANTS
 * ============================================================================ */

/* Zeta temperature grid [K] */
const double ZETA_TEMPERATURES[ZETA_N_TEMPERATURES] = {
    2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000,
    22000, 24000, 26000, 28000, 30000, 32000, 34000, 36000, 38000, 40000
};

/* Known reference values for sanity checks */
#define REF_H_I_IONIZATION_EV   13.598434599702
#define REF_HE_II_IONIZATION_EV 54.4177650

/* ============================================================================
 * HELPER FUNCTIONS
 * ============================================================================ */

static void print_error(const char *msg) {
    fprintf(stderr, "[ATOMIC_LOADER ERROR] %s\n", msg);
}

static void print_info(const char *msg) {
    printf("[ATOMIC_LOADER] %s\n", msg);
}

/* Count number of rows in an HDF5 table dataset */
static hsize_t get_table_nrows(hid_t file_id, const char *table_path) {
    hid_t dataset_id = H5Dopen2(file_id, table_path, H5P_DEFAULT);
    if (dataset_id < 0) {
        return 0;
    }

    hid_t space_id = H5Dget_space(dataset_id);
    hsize_t dims[2];
    H5Sget_simple_extent_dims(space_id, dims, NULL);

    H5Sclose(space_id);
    H5Dclose(dataset_id);

    return dims[0];
}

/* Read a 1D double array from an HDF5 dataset */
static int read_double_array(hid_t file_id, const char *path,
                              double *buffer, hsize_t n_elements) {
    hid_t dataset_id = H5Dopen2(file_id, path, H5P_DEFAULT);
    if (dataset_id < 0) {
        print_error("Cannot open dataset");
        return -1;
    }

    herr_t status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
                            H5P_DEFAULT, buffer);
    H5Dclose(dataset_id);

    return (status < 0) ? -1 : 0;
}

/* Read a 1D int64 array from an HDF5 dataset */
static int read_int64_array(hid_t file_id, const char *path,
                             int64_t *buffer, hsize_t n_elements) {
    hid_t dataset_id = H5Dopen2(file_id, path, H5P_DEFAULT);
    if (dataset_id < 0) {
        return -1;
    }

    herr_t status = H5Dread(dataset_id, H5T_NATIVE_INT64, H5S_ALL, H5S_ALL,
                            H5P_DEFAULT, buffer);
    H5Dclose(dataset_id);

    return (status < 0) ? -1 : 0;
}

/* Read a 1D int8 array from an HDF5 dataset */
static int read_int8_array(hid_t file_id, const char *path,
                            int8_t *buffer, hsize_t n_elements) {
    hid_t dataset_id = H5Dopen2(file_id, path, H5P_DEFAULT);
    if (dataset_id < 0) {
        return -1;
    }

    herr_t status = H5Dread(dataset_id, H5T_NATIVE_INT8, H5S_ALL, H5S_ALL,
                            H5P_DEFAULT, buffer);
    H5Dclose(dataset_id);

    return (status < 0) ? -1 : 0;
}

/* Read a 1D int16 array */
static int read_int16_array(hid_t file_id, const char *path,
                             int16_t *buffer, hsize_t n_elements) {
    hid_t dataset_id = H5Dopen2(file_id, path, H5P_DEFAULT);
    if (dataset_id < 0) {
        return -1;
    }

    herr_t status = H5Dread(dataset_id, H5T_NATIVE_INT16, H5S_ALL, H5S_ALL,
                            H5P_DEFAULT, buffer);
    H5Dclose(dataset_id);

    return (status < 0) ? -1 : 0;
}

/* Read a 1D int32 array */
static int read_int32_array(hid_t file_id, const char *path,
                             int32_t *buffer, hsize_t n_elements) {
    hid_t dataset_id = H5Dopen2(file_id, path, H5P_DEFAULT);
    if (dataset_id < 0) {
        return -1;
    }

    herr_t status = H5Dread(dataset_id, H5T_NATIVE_INT32, H5S_ALL, H5S_ALL,
                            H5P_DEFAULT, buffer);
    H5Dclose(dataset_id);

    return (status < 0) ? -1 : 0;
}

/* Read a 1D bool array (stored as int8) */
static int read_bool_array(hid_t file_id, const char *path,
                            bool *buffer, hsize_t n_elements) {
    int8_t *temp = (int8_t *)malloc(n_elements * sizeof(int8_t));
    if (!temp) return -1;

    int status = read_int8_array(file_id, path, temp, n_elements);
    if (status == 0) {
        for (hsize_t i = 0; i < n_elements; i++) {
            buffer[i] = (temp[i] != 0);
        }
    }

    free(temp);
    return status;
}

/* ============================================================================
 * ELEMENT DATA LOADER
 * ============================================================================ */

static int load_atom_data(hid_t file_id, AtomicData *data) {
    print_info("Loading atom_data...");

    /* Atom data is stored as a pandas Series/DataFrame with string columns */
    /* The structure is: /atom_data/table with columns */

    /* For TARDIS format, atom_data has: atomic_number (index), symbol, name, mass */
    /* We'll read from the block format */

    hid_t group_id = H5Gopen2(file_id, "/atom_data", H5P_DEFAULT);
    if (group_id < 0) {
        print_error("Cannot open /atom_data group");
        return -1;
    }

    /* Read the values arrays from block0_values, block1_values, etc. */
    /* For simplicity, we'll initialize with hardcoded element data */
    /* A full implementation would parse the pandas table format */

    data->n_elements = MAX_ATOMIC_NUMBER;
    data->elements = (Element *)calloc(MAX_ATOMIC_NUMBER, sizeof(Element));
    if (!data->elements) {
        H5Gclose(group_id);
        return -1;
    }

    /* Initialize with standard element data */
    /* This is a simplified approach - full implementation would read from HDF5 */
    static const char *symbols[] = {
        "", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
        "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
        "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn"
    };

    static const char *names[] = {
        "", "Hydrogen", "Helium", "Lithium", "Beryllium", "Boron",
        "Carbon", "Nitrogen", "Oxygen", "Fluorine", "Neon",
        "Sodium", "Magnesium", "Aluminum", "Silicon", "Phosphorus",
        "Sulfur", "Chlorine", "Argon", "Potassium", "Calcium",
        "Scandium", "Titanium", "Vanadium", "Chromium", "Manganese",
        "Iron", "Cobalt", "Nickel", "Copper", "Zinc"
    };

    static const double masses[] = {
        0.0, 1.008, 4.003, 6.94, 9.012, 10.81, 12.011, 14.007, 15.999,
        18.998, 20.180, 22.990, 24.305, 26.982, 28.086, 30.974, 32.06,
        35.45, 39.948, 39.098, 40.078, 44.956, 47.867, 50.942, 51.996,
        54.938, 55.845, 58.933, 58.693, 63.546, 65.38
    };

    for (int z = 1; z <= MAX_ATOMIC_NUMBER; z++) {
        data->elements[z-1].atomic_number = z;
        strncpy(data->elements[z-1].symbol, symbols[z], 3);
        strncpy(data->elements[z-1].name, names[z], 15);
        data->elements[z-1].mass = masses[z];
        data->elements[z-1].mass_cgs = masses[z] * 1.66054e-24;  /* amu to grams */
    }

    H5Gclose(group_id);
    printf("  Loaded %d elements\n", data->n_elements);
    return 0;
}

/* ============================================================================
 * IONIZATION DATA LOADER
 * ============================================================================
 * Structure in HDF5 (pandas Series format):
 *   /ionization_data/index_label0  -> atomic_number (int8)
 *   /ionization_data/index_label1  -> ion_number (int8)
 *   /ionization_data/values        -> ionization_energy (float64)
 */

static int load_ionization_data(hid_t file_id, AtomicData *data) {
    print_info("Loading ionization_data...");

    hid_t group_id = H5Gopen2(file_id, "/ionization_data", H5P_DEFAULT);
    if (group_id < 0) {
        print_error("Cannot open /ionization_data group");
        return -1;
    }

    /* Get number of ions from index_label0 */
    hid_t label0_id = H5Dopen2(group_id, "index_label0", H5P_DEFAULT);
    if (label0_id < 0) {
        H5Gclose(group_id);
        print_error("Cannot open index_label0");
        return -1;
    }

    hid_t space_id = H5Dget_space(label0_id);
    hsize_t dims[1];
    H5Sget_simple_extent_dims(space_id, dims, NULL);
    H5Sclose(space_id);

    int32_t n_ions = (int32_t)dims[0];
    data->n_ions = n_ions;

    /* Allocate arrays */
    data->ions = (Ion *)calloc(n_ions, sizeof(Ion));
    int8_t *atomic_nums = (int8_t *)malloc(n_ions * sizeof(int8_t));
    int8_t *ion_nums = (int8_t *)malloc(n_ions * sizeof(int8_t));
    double *ion_energies = (double *)malloc(n_ions * sizeof(double));

    if (!data->ions || !atomic_nums || !ion_nums || !ion_energies) {
        H5Dclose(label0_id);
        H5Gclose(group_id);
        free(atomic_nums);
        free(ion_nums);
        free(ion_energies);
        return -1;
    }

    /* Read atomic numbers */
    H5Dread(label0_id, H5T_NATIVE_INT8, H5S_ALL, H5S_ALL, H5P_DEFAULT, atomic_nums);
    H5Dclose(label0_id);

    /* Read ion numbers */
    hid_t label1_id = H5Dopen2(group_id, "index_label1", H5P_DEFAULT);
    if (label1_id >= 0) {
        H5Dread(label1_id, H5T_NATIVE_INT8, H5S_ALL, H5S_ALL, H5P_DEFAULT, ion_nums);
        H5Dclose(label1_id);
    }

    /* Read ionization energies */
    hid_t values_id = H5Dopen2(group_id, "values", H5P_DEFAULT);
    if (values_id >= 0) {
        H5Dread(values_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, ion_energies);
        H5Dclose(values_id);
    }

    /* Populate ion structures and build index lookup */
    memset(data->ion_index, -1, sizeof(data->ion_index));

    for (int32_t i = 0; i < n_ions; i++) {
        /* HDF5 uses 0-indexed atomic numbers (H=0, He=1, ...)
           Convert to 1-indexed (H=1, He=2, ...) for standard convention */
        int z = atomic_nums[i] + 1;  /* Convert to 1-indexed */
        int ion = ion_nums[i];

        data->ions[i].atomic_number = z;
        data->ions[i].ion_number = ion;
        /* HDF5 stores ionization energy in eV; convert to erg for CGS convention */
        data->ions[i].ionization_energy = ion_energies[i] * CONST_EV_TO_ERG;
        data->ions[i].n_levels = 0;
        data->ions[i].level_start_idx = -1;
        data->ions[i].n_lines = 0;
        data->ions[i].line_start_idx = -1;

        /* Build lookup table */
        if (z > 0 && z <= MAX_ATOMIC_NUMBER && ion >= 0 && ion <= z) {
            data->ion_index[z][ion] = i;
        }
    }

    free(atomic_nums);
    free(ion_nums);
    free(ion_energies);
    H5Gclose(group_id);

    printf("  Loaded %d ions\n", n_ions);
    return 0;
}

/* ============================================================================
 * LEVEL DATA LOADER
 * ============================================================================
 * Structure in HDF5 (pandas DataFrame format):
 *   /levels_data/axis1_label0  -> atomic_number (int8)
 *   /levels_data/axis1_label1  -> ion_number (int8)
 *   /levels_data/axis1_label2  -> level_number (int16)
 *   /levels_data/block0_values -> energy (float64, shape [n, 1])
 *   /levels_data/block1_values -> g (int64, shape [n, 1])
 *   /levels_data/block2_values -> metastable (uint8, shape [n, 1])
 */

static int load_levels_data(hid_t file_id, AtomicData *data) {
    print_info("Loading levels_data...");

    hid_t group_id = H5Gopen2(file_id, "/levels_data", H5P_DEFAULT);
    if (group_id < 0) {
        print_error("Cannot open /levels_data group");
        return -1;
    }

    /* Get number of levels from axis1_label0 */
    hid_t label0_id = H5Dopen2(group_id, "axis1_label0", H5P_DEFAULT);
    if (label0_id < 0) {
        H5Gclose(group_id);
        return -1;
    }

    hid_t space_id = H5Dget_space(label0_id);
    hsize_t dims[1];
    H5Sget_simple_extent_dims(space_id, dims, NULL);
    H5Sclose(space_id);

    int32_t n_levels = (int32_t)dims[0];
    data->n_levels = n_levels;

    /* Allocate arrays */
    data->levels = (Level *)calloc(n_levels, sizeof(Level));
    int8_t *atomic_nums = (int8_t *)malloc(n_levels * sizeof(int8_t));
    int8_t *ion_nums = (int8_t *)malloc(n_levels * sizeof(int8_t));
    int16_t *level_nums = (int16_t *)malloc(n_levels * sizeof(int16_t));
    double *energies = (double *)malloc(n_levels * sizeof(double));
    int64_t *g_values = (int64_t *)malloc(n_levels * sizeof(int64_t));
    uint8_t *metastable = (uint8_t *)malloc(n_levels * sizeof(uint8_t));

    if (!data->levels || !atomic_nums || !ion_nums || !level_nums ||
        !energies || !g_values || !metastable) {
        goto cleanup_levels;
    }

    /* Read index labels */
    H5Dread(label0_id, H5T_NATIVE_INT8, H5S_ALL, H5S_ALL, H5P_DEFAULT, atomic_nums);
    H5Dclose(label0_id);

    hid_t label1_id = H5Dopen2(group_id, "axis1_label1", H5P_DEFAULT);
    if (label1_id >= 0) {
        H5Dread(label1_id, H5T_NATIVE_INT8, H5S_ALL, H5S_ALL, H5P_DEFAULT, ion_nums);
        H5Dclose(label1_id);
    }

    hid_t label2_id = H5Dopen2(group_id, "axis1_label2", H5P_DEFAULT);
    if (label2_id >= 0) {
        H5Dread(label2_id, H5T_NATIVE_INT16, H5S_ALL, H5S_ALL, H5P_DEFAULT, level_nums);
        H5Dclose(label2_id);
    }

    /* Read block0_values: energy [n_levels, 1] */
    hid_t block0_id = H5Dopen2(group_id, "block0_values", H5P_DEFAULT);
    if (block0_id >= 0) {
        H5Dread(block0_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, energies);
        H5Dclose(block0_id);
    }

    /* Read block1_values: g [n_levels, 1] */
    hid_t block1_id = H5Dopen2(group_id, "block1_values", H5P_DEFAULT);
    if (block1_id >= 0) {
        H5Dread(block1_id, H5T_NATIVE_INT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, g_values);
        H5Dclose(block1_id);
    }

    /* Read block2_values: metastable [n_levels, 1] */
    hid_t block2_id = H5Dopen2(group_id, "block2_values", H5P_DEFAULT);
    if (block2_id >= 0) {
        H5Dread(block2_id, H5T_NATIVE_UINT8, H5S_ALL, H5S_ALL, H5P_DEFAULT, metastable);
        H5Dclose(block2_id);
    }

    /* Populate level structures */
    int current_ion_idx = -1;
    int current_z = -1, current_ion = -1;

    for (int32_t i = 0; i < n_levels; i++) {
        /* HDF5 uses 0-indexed atomic numbers; convert to 1-indexed */
        int z = atomic_nums[i] + 1;
        int ion = ion_nums[i];

        data->levels[i].atomic_number = z;
        data->levels[i].ion_number = ion;
        data->levels[i].level_number = level_nums[i];
        /* HDF5 stores level energy in eV; convert to erg for CGS convention */
        data->levels[i].energy = energies[i] * CONST_EV_TO_ERG;
        data->levels[i].g = (int32_t)g_values[i];
        data->levels[i].metastable = (metastable[i] != 0);

        if (z != current_z || ion != current_ion) {
            current_z = z;
            current_ion = ion;

            if (z > 0 && z <= MAX_ATOMIC_NUMBER && ion >= 0) {
                current_ion_idx = data->ion_index[z][ion];
                if (current_ion_idx >= 0) {
                    data->ions[current_ion_idx].level_start_idx = i;
                    data->ions[current_ion_idx].n_levels = 0;
                }
            }
        }

        if (current_ion_idx >= 0) {
            data->ions[current_ion_idx].n_levels++;
        }
    }

cleanup_levels:
    free(atomic_nums);
    free(ion_nums);
    free(level_nums);
    free(energies);
    free(g_values);
    free(metastable);
    H5Gclose(group_id);

    printf("  Loaded %d levels\n", n_levels);
    return 0;
}

/* ============================================================================
 * LINES DATA LOADER
 * ============================================================================
 * Structure in HDF5 (pandas DataFrame format):
 *   /lines_data/axis1_label0  -> atomic_number (int8)
 *   /lines_data/axis1_label1  -> ion_number (int8)
 *   /lines_data/axis1_label2  -> level_number_lower (int16)
 *   /lines_data/axis1_label3  -> level_number_upper (int16)
 *   /lines_data/block0_values -> line_id (int64, shape [n, 1])
 *   /lines_data/block1_values -> wavelength,nu,f_ul,f_lu,B_ul,B_lu,A_ul (float64, [n, 7])
 */

static int load_lines_data(hid_t file_id, AtomicData *data) {
    print_info("Loading lines_data...");

    hid_t group_id = H5Gopen2(file_id, "/lines_data", H5P_DEFAULT);
    if (group_id < 0) {
        print_error("Cannot open /lines_data group");
        return -1;
    }

    /* Get number of lines from axis1_label0 */
    hid_t label0_id = H5Dopen2(group_id, "axis1_label0", H5P_DEFAULT);
    if (label0_id < 0) {
        H5Gclose(group_id);
        return -1;
    }

    hid_t space_id = H5Dget_space(label0_id);
    hsize_t dims[1];
    H5Sget_simple_extent_dims(space_id, dims, NULL);
    H5Sclose(space_id);

    int64_t n_lines = (int64_t)dims[0];
    data->n_lines = n_lines;

    /* Allocate arrays */
    data->lines = (Line *)calloc(n_lines, sizeof(Line));
    data->sorted_line_indices = (int64_t *)malloc(n_lines * sizeof(int64_t));
    data->sorted_line_nu = (double *)malloc(n_lines * sizeof(double));

    int8_t *atomic_nums = (int8_t *)malloc(n_lines * sizeof(int8_t));
    int8_t *ion_nums = (int8_t *)malloc(n_lines * sizeof(int8_t));
    int16_t *level_lower = (int16_t *)malloc(n_lines * sizeof(int16_t));
    int16_t *level_upper = (int16_t *)malloc(n_lines * sizeof(int16_t));
    int64_t *line_ids = (int64_t *)malloc(n_lines * sizeof(int64_t));

    if (!data->lines || !data->sorted_line_indices || !data->sorted_line_nu ||
        !atomic_nums || !ion_nums || !level_lower || !level_upper || !line_ids) {
        goto cleanup_lines;
    }

    /* Read index labels */
    H5Dread(label0_id, H5T_NATIVE_INT8, H5S_ALL, H5S_ALL, H5P_DEFAULT, atomic_nums);
    H5Dclose(label0_id);

    hid_t label1_id = H5Dopen2(group_id, "axis1_label1", H5P_DEFAULT);
    if (label1_id >= 0) {
        H5Dread(label1_id, H5T_NATIVE_INT8, H5S_ALL, H5S_ALL, H5P_DEFAULT, ion_nums);
        H5Dclose(label1_id);
    }

    hid_t label2_id = H5Dopen2(group_id, "axis1_label2", H5P_DEFAULT);
    if (label2_id >= 0) {
        H5Dread(label2_id, H5T_NATIVE_INT16, H5S_ALL, H5S_ALL, H5P_DEFAULT, level_lower);
        H5Dclose(label2_id);
    }

    hid_t label3_id = H5Dopen2(group_id, "axis1_label3", H5P_DEFAULT);
    if (label3_id >= 0) {
        H5Dread(label3_id, H5T_NATIVE_INT16, H5S_ALL, H5S_ALL, H5P_DEFAULT, level_upper);
        H5Dclose(label3_id);
    }

    /* Read block0_values: line_id [n_lines, 1] */
    hid_t block0_id = H5Dopen2(group_id, "block0_values", H5P_DEFAULT);
    if (block0_id >= 0) {
        H5Dread(block0_id, H5T_NATIVE_INT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, line_ids);
        H5Dclose(block0_id);
    }

    /* Read column names from block1_items for dynamic mapping */
    int col_wavelength = -1, col_nu = -1, col_f_ul = -1, col_f_lu = -1;
    int col_A_ul = -1, col_B_ul = -1, col_B_lu = -1;

    hid_t items_id = H5Dopen2(group_id, "block1_items", H5P_DEFAULT);
    if (items_id >= 0) {
        hid_t items_space = H5Dget_space(items_id);
        hsize_t items_dims[1];
        H5Sget_simple_extent_dims(items_space, items_dims, NULL);
        H5Sclose(items_space);

        int n_items = (int)items_dims[0];

        /* Read column names (fixed-length strings) */
        hid_t str_type = H5Dget_type(items_id);
        size_t str_size = H5Tget_size(str_type);
        char *col_names = (char *)malloc(n_items * str_size);

        if (col_names) {
            H5Dread(items_id, str_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, col_names);

            /* Map column names to indices */
            for (int c = 0; c < n_items; c++) {
                char *name = &col_names[c * str_size];
                if (strncmp(name, "wavelength", str_size) == 0) col_wavelength = c;
                else if (strncmp(name, "nu", str_size) == 0) col_nu = c;
                else if (strncmp(name, "f_ul", str_size) == 0) col_f_ul = c;
                else if (strncmp(name, "f_lu", str_size) == 0) col_f_lu = c;
                else if (strncmp(name, "A_ul", str_size) == 0) col_A_ul = c;
                else if (strncmp(name, "B_ul", str_size) == 0) col_B_ul = c;
                else if (strncmp(name, "B_lu", str_size) == 0) col_B_lu = c;
            }
            free(col_names);
        }
        H5Tclose(str_type);
        H5Dclose(items_id);
    }

    /* Fallback to default column order if items not found */
    if (col_wavelength < 0) col_wavelength = 0;
    if (col_f_ul < 0) col_f_ul = 1;
    if (col_f_lu < 0) col_f_lu = 2;
    if (col_nu < 0) col_nu = 3;
    if (col_B_lu < 0) col_B_lu = 4;
    if (col_B_ul < 0) col_B_ul = 5;
    if (col_A_ul < 0) col_A_ul = 6;

    /* Read block1_values: spectroscopic data [n_lines, 7] */
    hid_t block1_id = H5Dopen2(group_id, "block1_values", H5P_DEFAULT);
    if (block1_id >= 0) {
        hid_t block_space = H5Dget_space(block1_id);
        hsize_t block_dims[2];
        H5Sget_simple_extent_dims(block_space, block_dims, NULL);
        H5Sclose(block_space);

        int n_cols = (int)block_dims[1];

        double *block_data = (double *)malloc(n_lines * n_cols * sizeof(double));
        if (block_data) {
            H5Dread(block1_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, block_data);

            /* Use dynamically mapped column indices */
            /* HDF5 stores wavelength in Angstroms; convert to cm for CGS */
            for (int64_t i = 0; i < n_lines; i++) {
                double *row = &block_data[i * n_cols];
                data->lines[i].wavelength = row[col_wavelength] * CONST_ANGSTROM;  /* Å → cm */
                data->lines[i].nu = (col_nu < n_cols) ? row[col_nu] : 0.0;          /* Hz */
                data->lines[i].f_ul = (col_f_ul < n_cols) ? row[col_f_ul] : 0.0;
                data->lines[i].f_lu = (col_f_lu < n_cols) ? row[col_f_lu] : 0.0;
                data->lines[i].A_ul = (col_A_ul < n_cols) ? row[col_A_ul] : 0.0;
                data->lines[i].B_ul = (col_B_ul < n_cols) ? row[col_B_ul] : 0.0;
                data->lines[i].B_lu = (col_B_lu < n_cols) ? row[col_B_lu] : 0.0;
            }
            free(block_data);
        }
        H5Dclose(block1_id);
    }

    /* Populate Line structures */
    for (int64_t i = 0; i < n_lines; i++) {
        data->lines[i].line_id = line_ids[i];
        /* HDF5 uses 0-indexed atomic numbers; convert to 1-indexed */
        data->lines[i].atomic_number = atomic_nums[i] + 1;
        data->lines[i].ion_number = ion_nums[i];
        data->lines[i].level_number_lower = level_lower[i];
        data->lines[i].level_number_upper = level_upper[i];

        /* Build sorted frequency index */
        data->sorted_line_indices[i] = i;
        data->sorted_line_nu[i] = data->lines[i].nu;
    }

cleanup_lines:
    free(atomic_nums);
    free(ion_nums);
    free(level_lower);
    free(level_upper);
    free(line_ids);
    H5Gclose(group_id);

    printf("  Loaded %ld lines\n", (long)n_lines);
    return 0;
}

/* ============================================================================
 * SI II 6347/6371 LINE INJECTION
 * The Si II 6355 doublet is the most important diagnostic for Type Ia SNe.
 * It's missing from the kurucz_cd23_chianti database, so we inject it here.
 * ============================================================================ */

#define SI_II_INJECTED_LINES 2
static const double SI_II_LAMBDA[SI_II_INJECTED_LINES] = {6347.10, 6371.37};  /* Å */
static const double SI_II_F_LU[SI_II_INJECTED_LINES] = {0.708, 0.419};
static const double SI_II_A_UL[SI_II_INJECTED_LINES] = {6.10e7, 7.21e7};
static const int SI_II_G_LOWER[SI_II_INJECTED_LINES] = {2, 4};
static const int SI_II_G_UPPER[SI_II_INJECTED_LINES] = {4, 4};
static const int SI_II_LEVEL_LOWER[SI_II_INJECTED_LINES] = {0, 1};
static const int SI_II_LEVEL_UPPER[SI_II_INJECTED_LINES] = {5, 5};

static int inject_si_ii_6355_lines(AtomicData *data)
{
    printf("[ATOMIC_LOADER] Injecting Si II 6347/6371 Å doublet...\n");

    /* Reallocate lines array with space for new lines */
    int64_t old_n = data->n_lines;
    int64_t new_n = old_n + SI_II_INJECTED_LINES;

    Line *new_lines = (Line *)realloc(data->lines, new_n * sizeof(Line));
    int64_t *new_indices = (int64_t *)realloc(data->sorted_line_indices,
                                               new_n * sizeof(int64_t));
    double *new_nu = (double *)realloc(data->sorted_line_nu, new_n * sizeof(double));

    if (!new_lines || !new_indices || !new_nu) {
        print_error("Failed to allocate memory for Si II lines");
        return -1;
    }

    data->lines = new_lines;
    data->sorted_line_indices = new_indices;
    data->sorted_line_nu = new_nu;

    /* Physical constants */
    const double c_cgs = 2.99792458e10;  /* cm/s */
    const double h_cgs = 6.62607015e-27; /* erg·s */
    const double m_e = 9.1093837e-28;    /* g */
    const double e_cgs = 4.80320451e-10; /* esu */
    const double B_lu_factor = M_PI * e_cgs * e_cgs / (m_e * c_cgs);  /* B_lu = factor * f_lu / nu */

    /* Add Si II lines */
    for (int i = 0; i < SI_II_INJECTED_LINES; i++) {
        int64_t idx = old_n + i;
        Line *line = &data->lines[idx];

        double lambda_cm = SI_II_LAMBDA[i] * 1e-8;  /* Convert Å to cm */
        double nu = c_cgs / lambda_cm;

        line->line_id = 1400000 + i;  /* Synthetic line ID */
        line->atomic_number = 14;     /* Si */
        line->ion_number = 1;         /* Si II */
        line->level_number_lower = SI_II_LEVEL_LOWER[i];
        line->level_number_upper = SI_II_LEVEL_UPPER[i];
        line->wavelength = lambda_cm; /* Wavelength in cm */
        line->nu = nu;
        line->f_lu = SI_II_F_LU[i];
        line->A_ul = SI_II_A_UL[i];
        line->f_ul = SI_II_F_LU[i] * SI_II_G_LOWER[i] / (double)SI_II_G_UPPER[i];
        line->B_lu = B_lu_factor * SI_II_F_LU[i] / nu;
        line->B_ul = line->B_lu * SI_II_G_LOWER[i] / (double)SI_II_G_UPPER[i];

        /* Update sorted arrays (will be re-sorted later) */
        data->sorted_line_indices[idx] = idx;
        data->sorted_line_nu[idx] = nu;

        printf("  Added Si II %.2f Å: f_lu=%.3f, nu=%.4e Hz\n",
               SI_II_LAMBDA[i], SI_II_F_LU[i], nu);
    }

    data->n_lines = new_n;

    /* Re-sort the frequency index */
    /* Simple bubble sort for just 2 new elements */
    for (int64_t i = old_n; i < new_n; i++) {
        int64_t j = i;
        while (j > 0 && data->sorted_line_nu[j] < data->sorted_line_nu[j-1]) {
            /* Swap */
            double tmp_nu = data->sorted_line_nu[j];
            int64_t tmp_idx = data->sorted_line_indices[j];
            data->sorted_line_nu[j] = data->sorted_line_nu[j-1];
            data->sorted_line_indices[j] = data->sorted_line_indices[j-1];
            data->sorted_line_nu[j-1] = tmp_nu;
            data->sorted_line_indices[j-1] = tmp_idx;
            j--;
        }
    }

    printf("  Total lines after injection: %ld\n", (long)new_n);
    return 0;
}

/* ============================================================================
 * MAIN LOADER FUNCTION
 * ============================================================================ */

int atomic_data_load_hdf5(const char *filename, AtomicData *data) {
    printf("\n");
    print_info("Opening HDF5 file...");
    printf("  File: %s\n", filename);

    /* Initialize structure */
    memset(data, 0, sizeof(AtomicData));
    strncpy(data->source_file, filename, 255);
    strcpy(data->format_version, "2.0");
    data->owns_memory = true;

    /* Initialize ion_index to -1 (invalid) */
    memset(data->ion_index, -1, sizeof(data->ion_index));

    /* Open HDF5 file */
    hid_t file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id < 0) {
        print_error("Cannot open HDF5 file");
        return -1;
    }

    int status = 0;

    /* Load each dataset */
    if ((status = load_atom_data(file_id, data)) != 0) goto cleanup;
    if ((status = load_ionization_data(file_id, data)) != 0) goto cleanup;
    if ((status = load_levels_data(file_id, data)) != 0) goto cleanup;
    if ((status = load_lines_data(file_id, data)) != 0) goto cleanup;

    /* Inject missing Si II 6347/6371 Å lines (critical for SN Ia) */
    if ((status = inject_si_ii_6355_lines(data)) != 0) goto cleanup;

    /* Optional datasets (macro-atom, collisions, zeta) */
    /* These are loaded on-demand for non-LTE calculations */

    print_info("Atomic data loaded successfully!");

cleanup:
    H5Fclose(file_id);
    return status;
}

/* ============================================================================
 * MEMORY CLEANUP
 * ============================================================================ */

void atomic_data_free(AtomicData *data) {
    if (!data->owns_memory) return;

    free(data->elements);
    free(data->ions);
    free(data->levels);
    free(data->lines);
    free(data->sorted_line_indices);
    free(data->sorted_line_nu);
    free(data->macro_atom_transitions);
    free(data->macro_atom_references);
    free(data->collisions);
    free(data->zeta_data);

    /* Free downbranch table */
    atomic_free_downbranch_table(data);

    memset(data, 0, sizeof(AtomicData));
}

/* ============================================================================
 * SUMMARY AND VALIDATION
 * ============================================================================ */

void atomic_data_print_summary(const AtomicData *data) {
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║              LUMINA-SN ATOMIC DATA SUMMARY                    ║\n");
    printf("╠═══════════════════════════════════════════════════════════════╣\n");
    printf("║  Source: %-52s ║\n", data->source_file);
    printf("║  Format: %-52s ║\n", data->format_version);
    printf("╠═══════════════════════════════════════════════════════════════╣\n");
    printf("║  Elements:  %10d                                        ║\n", data->n_elements);
    printf("║  Ions:      %10d                                        ║\n", data->n_ions);
    printf("║  Levels:    %10d                                        ║\n", data->n_levels);
    printf("║  Lines:     %10ld                                        ║\n", (long)data->n_lines);
    printf("╚═══════════════════════════════════════════════════════════════╝\n");
    printf("\n");
}

int atomic_data_sanity_check(const AtomicData *data) {
    printf("\n");
    print_info("Running sanity checks...\n");

    int failures = 0;

    /* Check H I ionization energy */
    double h1_ion = atomic_get_ionization_energy(data, 1, 0);
    double h1_ion_ev = erg_to_ev(h1_ion);
    double h1_error = fabs(h1_ion_ev - REF_H_I_IONIZATION_EV) / REF_H_I_IONIZATION_EV;

    printf("  H I ionization energy:\n");
    printf("    Loaded:   %.6f eV\n", h1_ion_ev);
    printf("    Expected: %.6f eV\n", REF_H_I_IONIZATION_EV);
    printf("    Error:    %.2e\n", h1_error);

    if (h1_error > 1e-4) {
        printf("    Status:   FAIL\n");
        failures++;
    } else {
        printf("    Status:   PASS\n");
    }

    /* Check He II ionization energy */
    double he2_ion = atomic_get_ionization_energy(data, 2, 1);
    double he2_ion_ev = erg_to_ev(he2_ion);
    double he2_error = fabs(he2_ion_ev - REF_HE_II_IONIZATION_EV) / REF_HE_II_IONIZATION_EV;

    printf("\n  He II ionization energy:\n");
    printf("    Loaded:   %.6f eV\n", he2_ion_ev);
    printf("    Expected: %.6f eV\n", REF_HE_II_IONIZATION_EV);
    printf("    Error:    %.2e\n", he2_error);

    if (he2_error > 1e-4) {
        printf("    Status:   FAIL\n");
        failures++;
    } else {
        printf("    Status:   PASS\n");
    }

    /* Check Hα line (H I, 6564.6 Å = 4.5668×10¹⁴ Hz) */
    /* Reference: NIST Hα wavelength = 6564.614 Å (vacuum), A_ul ≈ 4.41×10⁷ s⁻¹ */
    #define REF_HALPHA_WAVELENGTH_A  6564.6
    #define REF_HALPHA_NU            4.5668e14
    #define REF_HALPHA_A_UL          4.41e7  /* NIST approx */

    printf("\n  Hα line (H I, 6564.6 Å):\n");

    /* Search for H I lines near Hα */
    double halpha_nu_min = wavelength_angstrom_to_nu(6570.0);
    double halpha_nu_max = wavelength_angstrom_to_nu(6560.0);

    int halpha_found = 0;
    double best_halpha_wl = 0.0, best_halpha_nu = 0.0, best_halpha_A = 0.0;
    double best_halpha_dist = 1e10;

    for (int64_t i = 0; i < data->n_lines; i++) {
        Line *line = &data->lines[i];
        /* H I = Z=1, ion=0 */
        if (line->atomic_number == 1 && line->ion_number == 0) {
            if (line->nu >= halpha_nu_min && line->nu <= halpha_nu_max) {
                double wl_A = line->wavelength / CONST_ANGSTROM;
                double dist = fabs(wl_A - REF_HALPHA_WAVELENGTH_A);
                if (dist < best_halpha_dist) {
                    best_halpha_dist = dist;
                    best_halpha_wl = wl_A;
                    best_halpha_nu = line->nu;
                    best_halpha_A = line->A_ul;
                    halpha_found = 1;
                }
            }
        }
    }

    if (halpha_found) {
        printf("    Found H I line at λ = %.4f Å\n", best_halpha_wl);
        printf("    Frequency: %.4e Hz (expected %.4e Hz)\n", best_halpha_nu, REF_HALPHA_NU);
        printf("    A_ul: %.4e s⁻¹ (expected ~%.2e s⁻¹)\n", best_halpha_A, REF_HALPHA_A_UL);

        /* Verify wavelength matches frequency: λ = c/ν */
        double computed_wl = CONST_C / best_halpha_nu / CONST_ANGSTROM;
        double wl_error = fabs(best_halpha_wl - computed_wl) / best_halpha_wl;
        printf("    λ-ν consistency: λ=%.4f Å, c/ν=%.4f Å (error %.2e)\n",
               best_halpha_wl, computed_wl, wl_error);

        if (wl_error < 1e-3) {
            printf("    Status:   PASS\n");
        } else {
            printf("    Status:   FAIL (λ-ν mismatch)\n");
            failures++;
        }
    } else {
        printf("    WARNING: Hα line not found in database\n");
        printf("    Status:   SKIP (line not present)\n");
    }

    /* Check total line count */
    printf("\n  Total lines: %ld\n", (long)data->n_lines);

    if (data->n_lines < 100000) {
        printf("    WARNING: Fewer lines than expected (>100k for full dataset)\n");
    }

    printf("\n");
    if (failures == 0) {
        print_info("All sanity checks PASSED");
    } else {
        printf("[ATOMIC_LOADER] %d sanity check(s) FAILED\n", failures);
    }

    return failures;
}

/* ============================================================================
 * LOOKUP FUNCTIONS
 * ============================================================================ */

const Element *atomic_get_element(const AtomicData *data, int atomic_number) {
    if (atomic_number < 1 || atomic_number > data->n_elements) {
        return NULL;
    }
    return &data->elements[atomic_number - 1];
}

const Ion *atomic_get_ion(const AtomicData *data, int atomic_number, int ion_number) {
    if (atomic_number < 1 || atomic_number > MAX_ATOMIC_NUMBER) return NULL;
    if (ion_number < 0 || ion_number > atomic_number) return NULL;

    int idx = data->ion_index[atomic_number][ion_number];
    if (idx < 0) return NULL;

    return &data->ions[idx];
}

const Level *atomic_get_level(const AtomicData *data, int atomic_number,
                               int ion_number, int level_number) {
    const Ion *ion = atomic_get_ion(data, atomic_number, ion_number);
    if (!ion || level_number < 0 || level_number >= ion->n_levels) {
        return NULL;
    }
    return &data->levels[ion->level_start_idx + level_number];
}

double atomic_get_ionization_energy(const AtomicData *data,
                                     int atomic_number, int ion_number) {
    const Ion *ion = atomic_get_ion(data, atomic_number, ion_number);
    if (!ion) return 0.0;
    return ion->ionization_energy;
}

int atomic_get_g(const AtomicData *data, int atomic_number,
                 int ion_number, int level_number) {
    const Level *level = atomic_get_level(data, atomic_number, ion_number, level_number);
    if (!level) return 0;
    return level->g;
}

int64_t atomic_find_lines_in_range(const AtomicData *data,
                                    double nu_min, double nu_max,
                                    int64_t *indices, int64_t max_lines) {
    /*
     * Binary search implementation for finding lines in frequency range.
     *
     * The sorted_line_nu array contains frequencies in ascending order,
     * and sorted_line_indices maps back to the original line indices.
     *
     * Algorithm:
     * 1. Binary search for first line with nu >= nu_min
     * 2. Binary search for first line with nu > nu_max
     * 3. Copy indices in the range [start, end)
     *
     * Complexity: O(log n + k) where k is the number of lines in range
     * vs O(n) for linear search
     */

    if (!data->sorted_line_nu || !data->sorted_line_indices) {
        /* Fall back to linear search if sorted arrays not available */
        int64_t count = 0;
        for (int64_t i = 0; i < data->n_lines && count < max_lines; i++) {
            double nu = data->lines[i].nu;
            if (nu >= nu_min && nu <= nu_max) {
                indices[count++] = i;
            }
        }
        return count;
    }

    int64_t n = data->n_lines;

    /* Binary search for first line with nu >= nu_min */
    int64_t left = 0, right = n;
    while (left < right) {
        int64_t mid = left + (right - left) / 2;
        if (data->sorted_line_nu[mid] < nu_min) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    int64_t start = left;

    /* Binary search for first line with nu > nu_max */
    left = start;  /* Start from where we left off */
    right = n;
    while (left < right) {
        int64_t mid = left + (right - left) / 2;
        if (data->sorted_line_nu[mid] <= nu_max) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    int64_t end = left;

    /* Copy indices in range */
    int64_t count = 0;
    for (int64_t i = start; i < end && count < max_lines; i++) {
        indices[count++] = data->sorted_line_indices[i];
    }

    return count;
}

/* ============================================================================
 * DOWNBRANCH TABLE IMPLEMENTATION (Line Fluorescence)
 * ============================================================================
 * Pre-compute branching ratios for line fluorescence cascade.
 *
 * Physics:
 * --------
 * When a line absorbs a photon, the atom is excited to the upper level.
 * From there, it can de-excite through any transition from that level.
 * The branching probability to each emission channel is:
 *
 *   p_k = A_ul(k) / Σ_j A_ul(j)
 *
 * where the sum is over all lines with the same upper level.
 *
 * This enables UV → optical fluorescence: a UV photon absorbed by a
 * high-excitation line can cascade down through intermediate levels,
 * producing optical photons.
 */

int atomic_build_downbranch_table(AtomicData *data)
{
    printf("[ATOMIC_LOADER] Building downbranch table for %ld lines...\n",
           (long)data->n_lines);

    if (data->n_lines == 0 || data->lines == NULL) {
        return -1;
    }

    /* Initialize downbranch structure */
    data->downbranch.initialized = false;

    /* Allocate per-line start/count arrays */
    data->downbranch.emission_line_start = (int64_t *)calloc(data->n_lines, sizeof(int64_t));
    data->downbranch.emission_line_count = (int64_t *)calloc(data->n_lines, sizeof(int64_t));

    if (!data->downbranch.emission_line_start || !data->downbranch.emission_line_count) {
        atomic_free_downbranch_table(data);
        return -1;
    }

    /*
     * First pass: For each line, count emission candidates from its upper level.
     *
     * An emission candidate is any line with:
     *   - Same atomic number and ion number
     *   - Same upper level (level_number_upper)
     *   - Non-zero A_ul
     */
    int64_t total_entries = 0;

    for (int64_t i = 0; i < data->n_lines; i++) {
        const Line *absorbing = &data->lines[i];
        int8_t Z = absorbing->atomic_number;
        int8_t ion = absorbing->ion_number;
        int16_t upper = absorbing->level_number_upper;

        int64_t count = 0;

        /* Find all emission lines from this upper level */
        for (int64_t j = 0; j < data->n_lines; j++) {
            const Line *emission = &data->lines[j];
            if (emission->atomic_number == Z &&
                emission->ion_number == ion &&
                emission->level_number_upper == upper &&
                emission->A_ul > 0.0) {
                count++;
            }
        }

        data->downbranch.emission_line_start[i] = total_entries;
        data->downbranch.emission_line_count[i] = count;
        total_entries += count;
    }

    data->downbranch.total_emission_entries = total_entries;

    if (total_entries == 0) {
        printf("[ATOMIC_LOADER] Warning: No downbranch entries found\n");
        data->downbranch.initialized = true;
        return 0;
    }

    /* Allocate emission line ID and probability arrays */
    data->downbranch.emission_lines = (int64_t *)malloc(total_entries * sizeof(int64_t));
    data->downbranch.branching_probs = (double *)malloc(total_entries * sizeof(double));

    if (!data->downbranch.emission_lines || !data->downbranch.branching_probs) {
        atomic_free_downbranch_table(data);
        return -1;
    }

    /*
     * Second pass: Populate emission line IDs and compute cumulative probabilities.
     *
     * For each absorbing line:
     *   1. Find all emission candidates
     *   2. Sum their A_ul values
     *   3. Compute cumulative branching probabilities: p_k = Σ_{j≤k} A_ul(j) / Σ_all A_ul
     */
    for (int64_t i = 0; i < data->n_lines; i++) {
        const Line *absorbing = &data->lines[i];
        int8_t Z = absorbing->atomic_number;
        int8_t ion = absorbing->ion_number;
        int16_t upper = absorbing->level_number_upper;

        int64_t start = data->downbranch.emission_line_start[i];
        int64_t count = data->downbranch.emission_line_count[i];

        if (count == 0) continue;

        /* First: collect emission lines and sum A_ul */
        double total_A = 0.0;
        int64_t entry = 0;

        for (int64_t j = 0; j < data->n_lines && entry < count; j++) {
            const Line *emission = &data->lines[j];
            if (emission->atomic_number == Z &&
                emission->ion_number == ion &&
                emission->level_number_upper == upper &&
                emission->A_ul > 0.0) {
                data->downbranch.emission_lines[start + entry] = j;
                total_A += emission->A_ul;
                entry++;
            }
        }

        /* Second: compute cumulative probabilities */
        double cumulative = 0.0;
        for (int64_t k = 0; k < count; k++) {
            int64_t line_idx = data->downbranch.emission_lines[start + k];
            cumulative += data->lines[line_idx].A_ul / total_A;
            data->downbranch.branching_probs[start + k] = cumulative;
        }

        /* Ensure last probability is exactly 1.0 */
        if (count > 0) {
            data->downbranch.branching_probs[start + count - 1] = 1.0;
        }
    }

    data->downbranch.initialized = true;

    /* Statistics */
    int64_t lines_with_branches = 0;
    for (int64_t i = 0; i < data->n_lines; i++) {
        if (data->downbranch.emission_line_count[i] > 1) {
            lines_with_branches++;
        }
    }

    printf("[ATOMIC_LOADER] Downbranch table built:\n");
    printf("  Total emission entries: %ld\n", (long)total_entries);
    printf("  Lines with multiple branches: %ld (%.1f%%)\n",
           (long)lines_with_branches, 100.0 * lines_with_branches / data->n_lines);

    return 0;
}

void atomic_free_downbranch_table(AtomicData *data)
{
    free(data->downbranch.emission_line_start);
    free(data->downbranch.emission_line_count);
    free(data->downbranch.emission_lines);
    free(data->downbranch.branching_probs);

    data->downbranch.emission_line_start = NULL;
    data->downbranch.emission_line_count = NULL;
    data->downbranch.emission_lines = NULL;
    data->downbranch.branching_probs = NULL;
    data->downbranch.total_emission_entries = 0;
    data->downbranch.initialized = false;
}

int64_t atomic_sample_downbranch(const AtomicData *data, int64_t line_id, double xi)
{
    /*
     * Sample an emission line using pre-computed branching probabilities.
     *
     * @param data    AtomicData with initialized downbranch table
     * @param line_id Index of the absorbing line
     * @param xi      Random number in [0, 1)
     * @return Index of emission line, or line_id if resonant scatter
     */

    if (!data->downbranch.initialized || line_id < 0 || line_id >= data->n_lines) {
        return line_id;  /* Resonant scatter */
    }

    int64_t start = data->downbranch.emission_line_start[line_id];
    int64_t count = data->downbranch.emission_line_count[line_id];

    if (count <= 1) {
        /* Only one emission channel - resonant scatter */
        return line_id;
    }

    /* Binary search for the emission line */
    int64_t left = 0, right = count;
    while (left < right) {
        int64_t mid = left + (right - left) / 2;
        if (data->downbranch.branching_probs[start + mid] < xi) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }

    /* Return the selected emission line */
    if (left < count) {
        return data->downbranch.emission_lines[start + left];
    }

    /* Fallback to last entry */
    return data->downbranch.emission_lines[start + count - 1];
}
