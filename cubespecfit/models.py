import numpy as np

# ------------------------------------------------------------------------------------------------------------

def ha_nii_model(x, z, I_Ha, sig, NIIHa, c, R=2000):
    # emission line wavelengths
    Ha, NIIb, NIIr = np.array([0.656461, 0.654985, 0.658528]) * (1 + z)
    I_NII = I_Ha * NIIHa                                              # get NII flux from Ha flux and [NII]/Ha ratio

    sig_inst = Ha / R / 2.35                            # calculate instrumental sigma from R
    sig_obs = (sig ** 2 + sig_inst ** 2) ** 0.5         # convert to observed sigma

    # combine the three Gaussians plus constant continuum
    f1 = I_Ha / ((2 * np.pi) ** 0.5 * sig_obs) * np.exp(-0.5 * (x - Ha) ** 2 / sig_obs ** 2)
    f2 = I_NII / ((2 * np.pi) ** 0.5 * sig_obs) * np.exp(-0.5 * (x - NIIr) ** 2 / sig_obs ** 2)
    f3 = I_NII / 2.8 / ((2 * np.pi) ** 0.5 * sig_obs) * np.exp(-0.5 * (x - NIIb) ** 2 / sig_obs ** 2)
    model = c + f1 + f2 + f3

    return model

# ------------------------------------------------------------------------------------------------------------

def ha_nii_sii_model(x, z, I_Ha, sig, NIIHa, I_SIIr, SII_ratio, c, R=2000):
    # emission line wavelengths
    Ha, NIIb, NIIr, SIIb, SIIr = np.array([0.656461, 0.654985, 0.658528, 0.671829, 0.673267]) * (1 + z)
    I_NII = I_Ha * NIIHa                # get NII flux from Ha flux and [NII]/Ha ratio
    I_SIIb = I_SIIr * SII_ratio         # get blue [SII] flux from red [SII] flux and [SII] ratio

    sig_inst = Ha / R / 2.35                            # calculate instrumental sigma from R
    sig_obs = (sig ** 2 + sig_inst ** 2) ** 0.5         # convert to observed sigma

    # combine the five Gaussians plus constant continuum
    f1 = I_Ha / ((2 * np.pi) ** 0.5 * sig_obs) * np.exp(-0.5 * (x - Ha) ** 2 / sig_obs ** 2)
    f2 = I_NII / ((2 * np.pi) ** 0.5 * sig_obs) * np.exp(-0.5 * (x - NIIr) ** 2 / sig_obs ** 2)
    f3 = I_NII / 2.8 / ((2 * np.pi) ** 0.5 * sig_obs) * np.exp(-0.5 * (x - NIIb) ** 2 / sig_obs ** 2)
    f4 = I_SIIb / ((2 * np.pi) ** 0.5 * sig_obs) * np.exp(-0.5 * (x - SIIb) ** 2 / sig_obs ** 2)
    f5 = I_SIIr / ((2 * np.pi) ** 0.5 * sig_obs) * np.exp(-0.5 * (x - SIIr) ** 2 / sig_obs ** 2)
    model = c + f1 + f2 + f3 + f4 + f5

    return model

# ------------------------------------------------------------------------------------------------------------

def oiii_model(x, z, I_OIIIr, sig, c, R=2000):
    # emission line wavelengths
    OIIIb, OIIIr = np.array([0.4958911, 0.5006843]) * (1 + z)
    I_OIIIb = I_OIIIr / 2.98            # get OIII4959 flux from OIII5007 flux and line ratio

    sig_inst = OIIIr / R / 2.35                         # calculate instrumental sigma from R
    sig_obs = (sig ** 2 + sig_inst ** 2) ** 0.5         # convert to observed sigma

    # combine the two Gaussians plus constant continuum
    f1 = I_OIIIb / ((2 * np.pi) ** 0.5 * sig_obs) * np.exp(-0.5 * (x - OIIIb) ** 2 / sig_obs ** 2)
    f2 = I_OIIIr / ((2 * np.pi) ** 0.5 * sig_obs) * np.exp(-0.5 * (x - OIIIr) ** 2 / sig_obs ** 2)
    model = c + f1 + f2

    return model

# ------------------------------------------------------------------------------------------------------------