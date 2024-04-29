import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit

# ------------------------------------------------------------------------------------------------------------

def fitspec(wave, flux, err, model, p0, bounds):
    """
    fit 1-D spectrum with a given model using scipy.curve_fit
    :param wave: wavelength array of input spectrum
    :param flux: flux array of input spectrum
    :param err: error array of input spectrum
    :param model: model to fit to the data
    :param p0: initial guess for model parameters
    :param bounds: bounds for model parameters
    :return:
    """
    params, cov = curve_fit(model, wave, flux, p0=p0, sigma=err, bounds=bounds)     # do the fitting
    errors = np.array([cov[n][n] ** 0.5 for n in range(len(params))])               # estimate uncertainties from
                                                                                    #    covariance matrix
    bestfit = model(wave, *params)                                                  # get best fit array
    chi2 = np.sum((flux-bestfit)**2/err**2)                                         # get chi-squared

    return params, errors, bestfit, chi2

# ------------------------------------------------------------------------------------------------------------

def fitcube(cube, err_cube, wl, z_sys, model, p0, bounds, l0, no_line, dl=0.05, snr_cut=5,
           bin_lower=0, bin_upper=0, snr_max=50, save_grid=False, grid_file="fit_grid.pdf"):
    """
    derive maps of velocity/velocity dispersion/line flux/continuum flux by fitting a given emission line
    model on a pixel-by-pixel basis, using adaptive binning
    :param cube: data cube to use in the fitting
    :param err_cube: error cube to use in the fitting
    :param wl: wavelength axis of the corresponding data cube
    :param z_sys: systemic redshift of the source in the cube
    :param model: model to fit to the data
    :param p0: initial guess for model parameters
    :param bounds: bounds for model parameters
    :param l0: central wavelength of emission line(s)
    :param no_line: indices of the wavelength array to use for continuum estimation
    :param dl: wavelength range to cut around the line, in um
    :param snr_cut: signal-to-noise threshold for the adaptive binning
    :param bin_lower: minimum bin size to use in adaptive binning
    :param bin_upper: maximum bin size to use in adaptive binning
    :param snr_max: approx. highest expected S/N, will set color scale for grid spectra plots
    :param save_grid: whether or not to save a PDF of the resultant grid spectra
    :param grid_file: filename to save grid spectra
    :return: best fit parameter and error cubes
    """

    color_norm_sn = matplotlib.colors.Normalize(vmin=snr_cut, vmax=snr_max)
    cmap = matplotlib.cm.plasma             # generate color map for plotting

    l = l0*(1+z_sys)                        # observed wavelength of the main emission line
    gd = (wl >= l - dl) & (wl <= l + dl)    # extract a small region around the wavelength of the line...
    cube = cube[gd, :, :]                   #   ...from the cube...
    err_cube = err_cube[gd, :, :]           #   ...the error cube...
    wave = wl[gd]                           #   ...the wavelength array...
    no_line = no_line[gd]                   #   ...and the no_line array

    sz = cube.shape

    if save_grid:                                       # if the user wishes to save a plot of the results
        plt.figure(figsize=(sz[2],sz[1]))               # then initialize the figure and gridspec
        gs = gridspec.GridSpec(sz[1],sz[2])

    param_cube = np.zeros((len(p0), sz[1], sz[2]))      # zero arrays to store results...
    param_err_cube = np.zeros((len(p0), sz[1], sz[2]))  #   ...uncertainties...
    snr_map = np.zeros([sz[1], sz[2]])                  #   ...the S/N map...
    bin_map = np.zeros([sz[1], sz[2]])                  #   ... the binsize map
    norm_map = np.zeros([sz[1], sz[2]])                  #   ... and the flux normalization map

    # now begin the fitting
    # loop over pixels and fit 1d spectra
    for p in range(0, sz[2]):
        for q in range(0, sz[1]):
            for binsize in range(bin_lower, bin_upper + 1):
                xlo, xhi = p + np.array([-binsize, binsize+1])      # get spatial indices of pixels to use in the
                ylo, yhi = q + np.array([-binsize, binsize+1])      # fitting, binning if desired

                xlo = 0 if xlo <= 0 else xlo
                xhi = sz[2] if xhi > sz[2] else xhi     # set xlo/xhi to 0/Nx if pixel near boundaries
                ylo = 0 if ylo <= 0 else ylo
                yhi = sz[1] if yhi >= sz[1] else yhi    # do the same for ylo/yhi

                flux = np.nanmean(cube[:, ylo:yhi, xlo:xhi], axis=(1,2))        # extract flux
                err = np.nanmean(err_cube[:, ylo:yhi, xlo:xhi], axis=(1, 2))    # extract errors

                if np.isnan(flux).all():
                    continue

                # mask NaNs in all arrays
                wave_m = wave.copy()
                flux_m = flux.copy()
                err_m = err.copy()
                no_line_m = no_line.copy()
                wave_m = wave_m[np.isnan(flux) == False]
                flux_m = flux_m[np.isnan(flux) == False]
                err_m = err_m[np.isnan(flux) == False]
                no_line_m = no_line_m[np.isnan(flux) == False]

                norm = np.nanmax(flux_m)      # get max value of flux array
                flux_m/=norm                  # normalise flux and error to avoid very small values
                err_m/=norm
                norm_map[q, p] = norm

                # first, calculate chi2 without including emission  (*_c = "continuum")
                med_c, std_c = np.nanmedian(flux_m[no_line_m]), np.nanstd(flux_m[no_line_m])
                chi2_SLF = np.sum((flux_m-med_c)**2/err_m**2)

                # now try to fit the emission lines
                try:
                    params, errors, bestfit, chi2_GF = fitspec(wave_m, flux_m, err_m, model, p0=p0,
                                                               bounds=bounds)
                    SNR = (chi2_SLF - chi2_GF) ** 0.5

                    if SNR >= snr_cut:  # check if pixel meets S/N threshold
                        if save_grid:
                            ax = plt.subplot(gs[sz[1] - 1 - q, p])
                            ax.set_xlim([l - dl, l + dl])
                            ax.set_ylim([-0.2, 1.5])
                            X = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 1000)
                            ax.set_xticks([])
                            ax.set_yticks([])

                            ax.step(wave_m, flux_m, lw=1, where="mid", c="w")
                            ax.plot(X, model(X, *params), c="k", lw=1, alpha=0.8)

                            ax.fill_between((ax.get_xlim()[0], ax.get_xlim()[1]),
                                            ax.get_ylim()[0], ax.get_ylim()[1],
                                            color=cmap(color_norm_sn(SNR)), zorder=1)
                            ax.fill_between(wave, ax.get_ylim()[0], ax.get_ylim()[1], where=no_line==False,
                                            color="w", alpha=0.2)
                            ax.axvline(l0 * (1 + z_sys), c="k", ls="--", lw=1, alpha=0.75, zorder=2)

                        # store the results
                        param_cube[0:len(p0), q, p] = params[0:len(p0)]
                        param_err_cube[0:len(p0), q, p] = errors[0:len(p0)]
                        snr_map[q, p] = SNR
                        bin_map[q, p] = binsize

                        break

                except ValueError as e:
                    if str(e) == "Residuals are not finite in the initial point.":
                        continue
                except RuntimeError as e:
                    if str(e) == "Optimal parameters not found: The maximum number of function evaluations is exceeded.":
                        continue
                except:
                    raise


    if save_grid:
        gs.update(hspace=0, wspace=0)
        plt.tight_layout()
        plt.savefig(grid_file)
        plt.close()

    return param_cube, param_err_cube, snr_map, bin_map, norm_map

# ------------------------------------------------------------------------------------------------------------