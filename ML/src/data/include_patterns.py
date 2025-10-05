# Column include patterns for exoplanet datasets
KEPLER_INCLUDE_PATTERNS = [
    "koi_disposition","koi_pdisposition","koi_score","koi_fpflag_nt","koi_fpflag_ss","koi_fpflag_co","koi_fpflag_ec","koi_period","koi_time0bk","koi_duration","koi_impact","koi_eccen","koi_model_snr","koi_max_mult_ev","koi_max_sngle_ev","koi_num_transits","koi_steff","koi_slogg","koi_smet","koi_srad","koi_smass","koi_kepmag","koi_gmag","koi_rmag","koi_imag","koi_zmag","koi_jmag","koi_hmag","koi_kmag","koi_fwm_sra","koi_fwm_sdec","koi_fwm_srao","koi_fwm_sdeco","koi_fwm_prao","koi_fwm_pdeco","koi_dicco_mra","koi_dicco_mdec","koi_dicco_msky"
]

K2_INCLUDE_PATTERNS = [
    'disp_refname', 'disc_refname', 'default_flag', 'rv_flag', 'pl_nnotes',
    'pl_trandep', 'st_refname', 'st_masserr2', 'pl_letter', 'disc_pubdate',
    'pl_ratrorerr2', 'pl_refname', 'pl_radjerr2', 'st_masserr1', 'st_tefferr2',
    'pl_radeerr2', 'sy_disterr2', 'sy_plx', 'st_tefferr1', 'sy_dist',
    'pl_tranmid', 'disc_year', 'pl_tsystemref', 'sy_gaiamagerr1', 'pl_rade',
    'rowupdate', 'sy_disterr1', 'pl_radj', 'pl_pubdate', 'sy_gaiamagerr2',
    'sy_pmraerr2'
]

TESS_INCLUDE_PATTERNS = [
    'pl_pnum', 'rastr', 'ra', 'decstr', 'dec', 'st_pmra', 'st_pmraerr1', 'st_pmraerr2',
    'pl_tranmid', 'pl_tranmiderr1', 'pl_tranmiderr2', 'pl_orbper', 'pl_orbpererr1', 'pl_orbpererr2',
    'pl_trandurh', 'pl_trandurherr1', 'pl_trandurherr2', 'pl_trandep', 'pl_trandeperr1', 'pl_trandeperr2',
    'pl_rade', 'pl_radeerr1', 'pl_radeerr2', 'pl_insol', 'pl_eqt', 'st_tmag', 'st_tmagerr1', 'st_tmagerr2',
    'st_dist', 'st_disterr1', 'st_disterr2', 'st_teff', 'st_tefferr1', 'st_tefferr2', 'st_logg', 'st_loggerr1',
    'st_loggerr2', 'st_rad', 'st_raderr1', 'st_raderr2', 'toi_created', 'rowupdate'
]
