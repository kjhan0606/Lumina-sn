/* 제거된 분기 원문 — 2026-08-07 스크랩 */

/* ---- LUMINA_SPEC_RANGE ---- */
    if (getenv("LUMINA_SPEC_RANGE")) {
        /* A4: previously unchecked sscanf — malformed string left arbitrary defaults. */
        int nf = sscanf(getenv("LUMINA_SPEC_RANGE"), "%lf,%lf,%d",
                        &spec_min, &spec_max, &spec_bins);
        if (nf != 3 || spec_min <= 0.0 || spec_max <= spec_min || spec_bins <= 0) {
            fprintf(stderr,
                "ERROR: LUMINA_SPEC_RANGE='%s' must be 'min,max,bins' with "
                "min>0, max>min, bins>0 (parsed %d fields → %.3f,%.3f,%d)\n",
                getenv("LUMINA_SPEC_RANGE"), nf, spec_min, spec_max, spec_bins);
            return 1;
        }
        printf("  Spectrum range: %.0f-%.0f A, %d bins\n", spec_min, spec_max, spec_bins);
    }

/* ---- LUMINA_T_INNER_FIX / LUMINA_DIFFUSION_INNER_BC ---- */
            const char *t_pin_env = getenv("LUMINA_T_INNER_FIX");
            double t_pin = t_pin_env ? atof(t_pin_env) : 0.0;
            const char *diff_bc_env = getenv("LUMINA_DIFFUSION_INNER_BC");
            int diff_bc = diff_bc_env ? atoi(diff_bc_env) : 0;
            if (diff_bc) {
                /* A1 (path-A, 2-agent verified): fixed-L diffusion inner BC.
                 * CMFGEN fixes the base luminosity and lets T_inner follow the
                 * diffusion relation; no feedback controller chasing emergent
                 * L_em (which ping-pongs when ionization shifts). HD2012 3.2.2. */
                double R_in = geo.r_inner[0];
                config.T_inner = pow(config.luminosity_requested /
                                     (4.0 * M_PI_VAL * R_in * R_in * SIGMA_SB),
                                     0.25);
                printf("  T_inner: %.2f K (fixed-L diffusion BC, L_req=%.3e, "
                       "L_em=%.3e)\n",
                       config.T_inner, config.luminosity_requested, L_emitted);
            } else if (t_pin > 0.0) {
                config.T_inner = t_pin;
                printf("  T_inner: %.2f K (pinned LUMINA_T_INNER_FIX, L_em=%.3e, L_req=%.3e)\n",
                       config.T_inner, L_emitted, config.luminosity_requested);

/* ---- LUMINA_TRANSPORT=cmf + CMF_NZ/NIMPACT/VTURB_KMS ---- */
    /* CMF formal solver (paper-method line transfer), gated by LUMINA_TRANSPORT=cmf */
    {
        const char *_transport = getenv("LUMINA_TRANSPORT");
        if (_transport && strcmp(_transport, "cmf") == 0) {
            const char *_nz   = getenv("LUMINA_CMF_NZ");
            const char *_nimp = getenv("LUMINA_CMF_NIMPACT");
            const char *_vt   = getenv("LUMINA_CMF_VTURB_KMS");
            int cmf_nz   = _nz   ? atoi(_nz)   : 2000;
            int cmf_nimp = _nimp ? atoi(_nimp) : 50;
            /* Blondin+2013 microturbulence not specified in-repo; default below
             * is a documented placeholder — tune via LUMINA_CMF_VTURB_KMS. */
            double v_turb_cms = (_vt ? atof(_vt) : 0.0) * 1.0e5;
            if (cmf_nz < 1) cmf_nz = 2000;
            if (cmf_nimp < 1) cmf_nimp = 50;

            Spectrum *spec_cmf = create_spectrum(spec_min, spec_max, spec_bins);
            compute_cmf_formal_spectrum(
                &geo, &plasma, &opacity, &atom_data,
                nlte.enabled ? &nlte : NULL,
                bf_opacity_enabled ? &bf : NULL,
                config.T_inner, spec_cmf, cmf_nimp, cmf_nz, v_turb_cms);
            FILE *cf = fopen("lumina_spectrum_cmf.csv", "w");
            if (cf) {
                fprintf(cf, "wavelength_angstrom,flux\n");
                for (int i = 0; i < spec_cmf->n_bins; i++)
                    fprintf(cf, "%.6f,%.6e\n", spec_cmf->wavelength[i], spec_cmf->flux[i]);
                fclose(cf);
                printf("CMF formal spectrum written to lumina_spectrum_cmf.csv\n");
            }
            free_spectrum(spec_cmf);
        }
    }
