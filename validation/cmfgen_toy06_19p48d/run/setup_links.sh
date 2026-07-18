#!/bin/bash
# Create all symlinks CMFGEN needs (atomic data, generic data, txt files, decay data)
cmfdist=/gpfs/kjhan/cmfgen_src/cur_cmf
ATOMIC=/gpfs/kjhan/cmfgen_21jun23/atomic
cd /gpfs/kjhan/cmfgen_runs/toy06_19.48d

# --- generic data ---
ln -sf $ATOMIC/misc/two_phot_data.dat        TWO_PHOT_DATA
ln -sf $ATOMIC/misc/xray_phot_fits.dat       XRAY_PHOT_FITS
ln -sf $ATOMIC/misc/rs_xray_fluxes_sol.dat   RS_XRAY_FLUXES
ln -sf $ATOMIC/HYD/I/5dec96/hyd_l_data.dat   HYD_L_DATA
ln -sf $ATOMIC/HYD/I/5dec96/gbf_n_data.dat   GBF_N_DATA

# --- txt option/data files ---
ln -sf $cmfdist/txt_files/maingen_opt_desc.txt   MAINGEN_OPT_DESC
ln -sf $cmfdist/txt_files/maingen_options.txt    MAINGEN_OPTIONS
ln -sf $cmfdist/txt_files/plt_spec_options.txt   PLT_SPEC_OPTIONS
ln -sf $cmfdist/txt_files/plt_spec_opt_desc.txt  PLT_SPEC_OPT_DESC
ln -sf $cmfdist/txt_files/plt_jh_opt_desc.txt    PLT_JH_OPT_DESC
ln -sf $cmfdist/txt_files/plt_jh_options.txt     PLT_JH_OPTIONS
ln -sf $cmfdist/txt_files/wr_f_opt_desc.txt      WR_F_OPT_DESC
ln -sf $cmfdist/txt_files/wr_f_options.txt       WR_F_OPTIONS
ln -sf $cmfdist/txt_files/sol_abund.dat          SOL_ABUND
ln -sf $cmfdist/txt_files/hi_is_line_list.dat    HI_IS_LINE_LIST 2>/dev/null
ln -sf $cmfdist/txt_files/h2_is_line_list.dat    H2_IS_LINE_LIST 2>/dev/null

# --- radioactive decay data (Ni56 -> Co56 -> Fe56) ---
ln -sf $ATOMIC/rad_decay_data/NUC_DECAY_DATA_ni56_co56_fe56  NUC_DECAY_DATA

# --- atomic data for the 21 model ions ---
ln -sf /gpfs/kjhan/cmfgen_21jun23/atomic/SIL/II/19apr23/osc_data    Sk2_F_OSCDAT
ln -sf /gpfs/kjhan/cmfgen_21jun23/atomic/SIL/II/19apr23/f_to_s_79   Sk2_F_TO_S
ln -sf /gpfs/kjhan/cmfgen_21jun23/atomic/SIL/II/19apr23/phot_data_A PHOTSk2_A
ln -sf /gpfs/kjhan/cmfgen_21jun23/atomic/SIL/II/19apr23/col_data    Sk2_COL_DATA
ln -sf /gpfs/kjhan/cmfgen_21jun23/atomic/SIL/III/19apr23/osc_data   SkIII_F_OSCDAT
ln -sf /gpfs/kjhan/cmfgen_21jun23/atomic/SIL/III/19apr23/f_to_s_ls  SkIII_F_TO_S
ln -sf /gpfs/kjhan/cmfgen_21jun23/atomic/SIL/III/19apr23/phot_data_A PHOTSkIII_A
ln -sf /gpfs/kjhan/cmfgen_21jun23/atomic/SIL/III/19apr23/col_data   SkIII_COL_DATA
ln -sf /gpfs/kjhan/cmfgen_21jun23/atomic/SIL/IV/5dec96/osc_op_split.dat SkIV_F_OSCDAT
ln -sf /gpfs/kjhan/cmfgen_21jun23/atomic/SIL/IV/5dec96/f_to_s_split.dat SkIV_F_TO_S
ln -sf /gpfs/kjhan/cmfgen_21jun23/atomic/SIL/IV/5dec96/phot_op.dat  PHOTSkIV_A
ln -sf /gpfs/kjhan/cmfgen_21jun23/atomic/SIL/IV/5dec96/col_guess.dat SkIV_COL_DATA
ln -sf /gpfs/kjhan/cmfgen_21jun23/atomic/SIL/V/19apr23/osc_data     SkV_F_OSCDAT
ln -sf /gpfs/kjhan/cmfgen_21jun23/atomic/SIL/V/19apr23/f_to_s_52    SkV_F_TO_S
ln -sf /gpfs/kjhan/cmfgen_21jun23/atomic/SIL/V/19apr23/phot_data_A_sm_3000 PHOTSkV_A
ln -sf /gpfs/kjhan/cmfgen_21jun23/atomic/SIL/V/19apr23/col_data     SkV_COL_DATA
ln -sf /gpfs/kjhan/cmfgen_21jun23/atomic/SUL/II/19apr23/osc_data    S2_F_OSCDAT
ln -sf /gpfs/kjhan/cmfgen_21jun23/atomic/SUL/II/19apr23/f_to_s_56   S2_F_TO_S
ln -sf /gpfs/kjhan/cmfgen_21jun23/atomic/SUL/II/19apr23/phot_data_A PHOTS2_A
ln -sf /gpfs/kjhan/cmfgen_21jun23/atomic/SUL/II/19apr23/col_data    S2_COL_DATA
ln -sf /gpfs/kjhan/cmfgen_21jun23/atomic/SUL/III/3oct00/siiiosc_fin.dat SIII_F_OSCDAT
ln -sf /gpfs/kjhan/cmfgen_21jun23/atomic/SUL/III/3oct00/f_to_s_127.dat SIII_F_TO_S
ln -sf /gpfs/kjhan/cmfgen_21jun23/atomic/SUL/III/3oct00/phot_sm_3000.dat PHOTSIII_A
ln -sf /gpfs/kjhan/cmfgen_21jun23/atomic/SUL/III/3oct00/col_siii.dat SIII_COL_DATA
ln -sf /gpfs/kjhan/cmfgen_21jun23/atomic/SUL/IV/3oct00/sivosc_fin.dat SIV_F_OSCDAT
ln -sf /gpfs/kjhan/cmfgen_21jun23/atomic/SUL/IV/3oct00/f_to_s_69.dat SIV_F_TO_S
ln -sf /gpfs/kjhan/cmfgen_21jun23/atomic/SUL/IV/3oct00/phot_sm_3000.dat PHOTSIV_A
ln -sf /gpfs/kjhan/cmfgen_21jun23/atomic/SUL/IV/3oct00/col_siv.dat  SIV_COL_DATA
ln -sf /gpfs/kjhan/cmfgen_21jun23/atomic/SUL/V/3oct00/svosc_fin.dat SV_F_OSCDAT
ln -sf /gpfs/kjhan/cmfgen_21jun23/atomic/SUL/V/3oct00/f_to_s_50.dat SV_F_TO_S
ln -sf /gpfs/kjhan/cmfgen_21jun23/atomic/SUL/V/3oct00/phot_sm_3000.dat PHOTSV_A
ln -sf /gpfs/kjhan/cmfgen_21jun23/atomic/SUL/V/3oct00/col_sv.dat    SV_COL_DATA
ln -sf /gpfs/kjhan/cmfgen_21jun23/atomic/CA/II/19apr23/osc_data     Ca2_F_OSCDAT
ln -sf /gpfs/kjhan/cmfgen_21jun23/atomic/CA/II/19apr23/f_to_s_43    Ca2_F_TO_S
ln -sf /gpfs/kjhan/cmfgen_21jun23/atomic/CA/II/19apr23/phot_data_A  PHOTCa2_A
ln -sf /gpfs/kjhan/cmfgen_21jun23/atomic/CA/II/19apr23/col_data     Ca2_COL_DATA
ln -sf /gpfs/kjhan/cmfgen_21jun23/atomic/CA/III/10apr99/osc_op_sp.dat CaIII_F_OSCDAT
ln -sf /gpfs/kjhan/cmfgen_21jun23/atomic/CA/III/10apr99/f_to_s.dat  CaIII_F_TO_S
ln -sf /gpfs/kjhan/cmfgen_21jun23/atomic/CA/III/10apr99/phot_smooth.dat PHOTCaIII_A
ln -sf /gpfs/kjhan/cmfgen_21jun23/atomic/CA/III/10apr99/col_guess.dat CaIII_COL_DATA
ln -sf /gpfs/kjhan/cmfgen_21jun23/atomic/CA/IV/10apr99/osc_op_sp.dat CaIV_F_OSCDAT
ln -sf /gpfs/kjhan/cmfgen_21jun23/atomic/CA/IV/10apr99/f_to_s.dat   CaIV_F_TO_S
ln -sf /gpfs/kjhan/cmfgen_21jun23/atomic/CA/IV/10apr99/phot_smooth.dat PHOTCaIV_A
ln -sf /gpfs/kjhan/cmfgen_21jun23/atomic/CA/IV/10apr99/col_guess.dat CaIV_COL_DATA
ln -sf /gpfs/kjhan/cmfgen_21jun23/atomic/CA/V/10apr99/osc_op_sp.dat CaV_F_OSCDAT
ln -sf /gpfs/kjhan/cmfgen_21jun23/atomic/CA/V/10apr99/f_to_s.dat    CaV_F_TO_S
ln -sf /gpfs/kjhan/cmfgen_21jun23/atomic/CA/V/10apr99/phot_smooth.dat PHOTCaV_A
ln -sf /gpfs/kjhan/cmfgen_21jun23/atomic/CA/V/10apr99/col_guess.dat CaV_COL_DATA
ln -sf /gpfs/kjhan/cmfgen_21jun23/atomic/FE/II/19apr23/osc_data     Fe2_F_OSCDAT
ln -sf /gpfs/kjhan/cmfgen_21jun23/atomic/FE/II/19apr23/f_to_s_135   Fe2_F_TO_S
ln -sf /gpfs/kjhan/cmfgen_21jun23/atomic/FE/II/19apr23/phot_data_A  PHOTFe2_A
ln -sf /gpfs/kjhan/cmfgen_21jun23/atomic/FE/II/19apr23/col_data     Fe2_COL_DATA
ln -sf /gpfs/kjhan/cmfgen_21jun23/atomic/FE/III/19apr23/osc_data    FeIII_F_OSCDAT
ln -sf /gpfs/kjhan/cmfgen_21jun23/atomic/FE/III/19apr23/f_to_s_105  FeIII_F_TO_S
ln -sf /gpfs/kjhan/cmfgen_21jun23/atomic/FE/III/19apr23/phot_data_A PHOTFeIII_A
ln -sf /gpfs/kjhan/cmfgen_21jun23/atomic/FE/III/19apr23/col_data    FeIII_COL_DATA
ln -sf /gpfs/kjhan/cmfgen_21jun23/atomic/FE/IV/18oct00/feiv_osc.dat FeIV_F_OSCDAT
ln -sf /gpfs/kjhan/cmfgen_21jun23/atomic/FE/IV/18oct00/f_to_s_63.dat FeIV_F_TO_S
ln -sf /gpfs/kjhan/cmfgen_21jun23/atomic/FE/IV/18oct00/phot_sm_3000.dat PHOTFeIV_A
ln -sf /gpfs/kjhan/cmfgen_21jun23/atomic/FE/IV/18oct00/col_data.dat FeIV_COL_DATA
ln -sf /gpfs/kjhan/cmfgen_21jun23/atomic/FE/V/18oct00/fev_osc.dat   FeV_F_OSCDAT
ln -sf /gpfs/kjhan/cmfgen_21jun23/atomic/FE/V/18oct00/f_to_s_45.dat FeV_F_TO_S
ln -sf /gpfs/kjhan/cmfgen_21jun23/atomic/FE/V/18oct00/phot_sm_3000.dat PHOTFeV_A
ln -sf /gpfs/kjhan/cmfgen_21jun23/atomic/FE/V/18oct00/col_guess.dat FeV_COL_DATA
ln -sf /gpfs/kjhan/cmfgen_21jun23/atomic/FE/VI/18oct00/fevi_osc.dat FeSIX_F_OSCDAT
ln -sf /gpfs/kjhan/cmfgen_21jun23/atomic/FE/VI/18oct00/f_to_s_67.dat FeSIX_F_TO_S
ln -sf /gpfs/kjhan/cmfgen_21jun23/atomic/FE/VI/18oct00/phot_sm_3000.dat PHOTFeSIX_A
ln -sf /gpfs/kjhan/cmfgen_21jun23/atomic/FE/VI/18oct00/col_data.dat FeSIX_COL_DATA
ln -sf /gpfs/kjhan/cmfgen_21jun23/atomic/COB/II/18oct00/coii_osc.dat Co2_F_OSCDAT
ln -sf /gpfs/kjhan/cmfgen_21jun23/atomic/COB/II/18oct00/f_to_s_55.dat Co2_F_TO_S
ln -sf /gpfs/kjhan/cmfgen_21jun23/atomic/COB/II/18oct00/phot_data.dat PHOTCo2_A
ln -sf /gpfs/kjhan/cmfgen_21jun23/atomic/COB/II/18oct00/col_guess.dat Co2_COL_DATA
ln -sf /gpfs/kjhan/cmfgen_21jun23/atomic/COB/III/18oct00/coiii_osc.dat CoIII_F_OSCDAT
ln -sf /gpfs/kjhan/cmfgen_21jun23/atomic/COB/III/18oct00/f_to_s_52.dat CoIII_F_TO_S
ln -sf /gpfs/kjhan/cmfgen_21jun23/atomic/COB/III/18oct00/phot_data.dat PHOTCoIII_A
ln -sf /gpfs/kjhan/cmfgen_21jun23/atomic/COB/III/18oct00/col_data.dat CoIII_COL_DATA
ln -sf /gpfs/kjhan/cmfgen_21jun23/atomic/COB/IV/18oct00/coiv_osc.dat CoIV_F_OSCDAT
ln -sf /gpfs/kjhan/cmfgen_21jun23/atomic/COB/IV/18oct00/f_to_s_56.dat CoIV_F_TO_S
ln -sf /gpfs/kjhan/cmfgen_21jun23/atomic/COB/IV/18oct00/phot_data.dat PHOTCoIV_A
ln -sf /gpfs/kjhan/cmfgen_21jun23/atomic/COB/IV/18oct00/col_guess.dat CoIV_COL_DATA
ln -sf /gpfs/kjhan/cmfgen_21jun23/atomic/COB/V/18oct00/cov_osc.dat  CoV_F_OSCDAT
ln -sf /gpfs/kjhan/cmfgen_21jun23/atomic/COB/V/18oct00/f_to_s_43.dat CoV_F_TO_S
ln -sf /gpfs/kjhan/cmfgen_21jun23/atomic/COB/V/18oct00/phot_data.dat PHOTCoV_A
ln -sf /gpfs/kjhan/cmfgen_21jun23/atomic/COB/V/18oct00/col_guess.dat CoV_COL_DATA
ln -sf /gpfs/kjhan/cmfgen_21jun23/atomic/COB/VI/18oct00/covi_osc.dat CoSIX_F_OSCDAT
ln -sf /gpfs/kjhan/cmfgen_21jun23/atomic/COB/VI/18oct00/f_to_s_41.dat CoSIX_F_TO_S
ln -sf /gpfs/kjhan/cmfgen_21jun23/atomic/COB/VI/18oct00/phot_data.dat PHOTCoSIX_A
ln -sf /gpfs/kjhan/cmfgen_21jun23/atomic/COB/VI/18oct00/col_guess.dat CoSIX_COL_DATA
ln -sf /gpfs/kjhan/cmfgen_21jun23/atomic/NICK/II/18oct00/nkii_osc.dat Nk2_F_OSCDAT
ln -sf /gpfs/kjhan/cmfgen_21jun23/atomic/NICK/II/18oct00/f_to_s_59.dat Nk2_F_TO_S
ln -sf /gpfs/kjhan/cmfgen_21jun23/atomic/NICK/II/18oct00/phot_data.dat PHOTNk2_A
ln -sf /gpfs/kjhan/cmfgen_21jun23/atomic/NICK/II/18oct00/col_guess.dat Nk2_COL_DATA
ln -sf /gpfs/kjhan/cmfgen_21jun23/atomic/NICK/III/18oct00/nkiii_osc.dat NkIII_F_OSCDAT
ln -sf /gpfs/kjhan/cmfgen_21jun23/atomic/NICK/III/18oct00/f_to_s_47.dat NkIII_F_TO_S
ln -sf /gpfs/kjhan/cmfgen_21jun23/atomic/NICK/III/18oct00/phot_data.dat PHOTNkIII_A
ln -sf /gpfs/kjhan/cmfgen_21jun23/atomic/NICK/III/18oct00/col_guess.dat NkIII_COL_DATA
ln -sf /gpfs/kjhan/cmfgen_21jun23/atomic/NICK/IV/18oct00/nkiv_osc.dat NkIV_F_OSCDAT
ln -sf /gpfs/kjhan/cmfgen_21jun23/atomic/NICK/IV/18oct00/f_to_s_54.dat NkIV_F_TO_S
ln -sf /gpfs/kjhan/cmfgen_21jun23/atomic/NICK/IV/18oct00/phot_data.dat PHOTNkIV_A
ln -sf /gpfs/kjhan/cmfgen_21jun23/atomic/NICK/IV/18oct00/col_guess.dat NkIV_COL_DATA
ln -sf /gpfs/kjhan/cmfgen_21jun23/atomic/NICK/V/18oct00/nkv_osc.dat NkV_F_OSCDAT
ln -sf /gpfs/kjhan/cmfgen_21jun23/atomic/NICK/V/18oct00/f_to_s_54.dat NkV_F_TO_S
ln -sf /gpfs/kjhan/cmfgen_21jun23/atomic/NICK/V/18oct00/phot_data.dat PHOTNkV_A
ln -sf /gpfs/kjhan/cmfgen_21jun23/atomic/NICK/V/18oct00/col_guess.dat NkV_COL_DATA
ln -sf /gpfs/kjhan/cmfgen_21jun23/atomic/NICK/VI/18oct00/nkvi_osc.dat NkSIX_F_OSCDAT
ln -sf /gpfs/kjhan/cmfgen_21jun23/atomic/NICK/VI/18oct00/f_to_s_62.dat NkSIX_F_TO_S
ln -sf /gpfs/kjhan/cmfgen_21jun23/atomic/NICK/VI/18oct00/phot_data.dat PHOTNkSIX_A
ln -sf /gpfs/kjhan/cmfgen_21jun23/atomic/NICK/VI/18oct00/col_guess.dat NkSIX_COL_DATA
# Si II has 2 photoionization routes (_A + _B)
ln -sf /gpfs/kjhan/cmfgen_21jun23/atomic/SIL/II/19apr23/phot_data_B PHOTSk2_B
