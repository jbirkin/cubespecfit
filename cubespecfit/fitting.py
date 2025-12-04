import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from lmfit import Model, Parameters
import tqdm
import logging
import warnings

from cubespecfit.wcs_helpers import _save_results

# Suppress specific numpy warnings globally
warnings.filterwarnings('ignore', message='Mean of empty slice')
warnings.filterwarnings('ignore', message='Degrees of freedom <= 0 for slice')

# ------------------------------------------------------------------------------------------------------------

logging.basicConfig(
    level=logging.DEBUG,
    format="%(levelname)s: %(message)s",
    filename="fitcube.log",  # Save to file
    filemode="w"  # Overwrite each run (use 'a' to append)
)


def fitspec_lmfit(wave, flux, err, model, params_dict, constraints=None):
    """
    Fit 1-D spectrum with a given model using lmfit

    Parameters
    ----------
    wave : array
        Wavelength array of input spectrum
    flux : array
        Flux array of input spectrum
    err : array
        Error array of input spectrum
    model : function
        Model function to fit. Should take (x, param1, param2, ...) as arguments
    params_dict : dict
        Dictionary of parameter settings. Each key is a parameter name, and value is a dict with:
        - 'value': initial guess (required)
        - 'min': lower bound (optional)
        - 'max': upper bound (optional)
        - 'vary': whether to vary this parameter (optional, default True)
        - 'expr': expression to constrain parameter (optional)
    constraints : dict, optional
        Dictionary of constraint callbacks to apply AFTER fitting.
        Format: {'param_name': callback_function}
        Each callback receives params object and modifies constrained parameters

    Returns
    -------
    params : array
        Best-fit parameter values (only free/constrained parameters)
    errors : array
        1-sigma uncertainties on parameters
    bestfit : array
        Best-fit model evaluated at input wavelengths
    chi2 : float
        Chi-squared of the fit
    result : lmfit.ModelResult
        Full lmfit result object (contains much more info)
    """
    # Create lmfit Model
    lm_model = Model(model)

    # Set up Parameters object
    params = Parameters()

    # Add parameters from params_dict
    for param_name, param_info in params_dict.items():
        params.add(param_name, **param_info)

    # Standard unconstrained fit
    result = lm_model.fit(flux, params, x=wave, weights=1.0 / err)

    # Extract results
    # Filter for only varying (non-fixed) parameters
    varying_params = [p for p in params_dict.keys() if not result.params[p].vary == False]

    param_values = np.array([result.params[p].value for p in varying_params])
    param_errors = np.array([result.params[p].stderr if result.params[p].stderr is not None
                             else np.nan for p in varying_params])

    bestfit = result.best_fit
    chi2 = result.chisqr
    result_obj = result

    return param_values, param_errors, bestfit, chi2, result_obj


def bic(flux, err, model, params):
    """
    Compute Bayesian Information Criterion.
    """
    n = len(flux)
    k = len(params)
    residual = flux - model(params)
    chi2 = np.sum((residual / err) ** 2)
    return k * np.log(n) + chi2


# ------------------------------------------------------------------------------------------------------------

def fitcube(cube, err_cube, wl, z_sys, model, params_dict, constraints, l0, no_line, dl=0.05, snr_cut=5,
            bin_lower=0, bin_upper=0, snr_max=50, save_grid=False, grid_file="fit_grid.pdf"):
    """
    derive maps of velocity/velocity dispersion/line flux/continuum flux by fitting a given emission line
    model on a pixel-by-pixel basis, using adaptive binning
    :param cube: data cube to use in the fitting
    :param err_cube: error cube to use in the fitting
    :param wl: wavelength axis of the corresponding data cube
    :param z_sys: systemic redshift of the source in the cube
    :param model: model to fit to the data
    :param params_dict: dictionary of parameter settings for lmfit
    :param constraints: dictionary of parameter constraints/expressions
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
    cmap = matplotlib.cm.plasma  # generate color map for plotting

    l = l0 * (1 + z_sys)  # observed wavelength of the main emission line
    gd = (wl >= l - dl) & (wl <= l + dl)  # extract a small region around the wavelength of the line...
    cube = cube[gd, :, :]  # ...from the cube...
    err_cube = err_cube[gd, :, :]  # ...the error cube...
    wave = wl[gd]  # ...the wavelength array...
    no_line = no_line[gd]  # ...and the no_line array

    sz = cube.shape

    if save_grid:  # if the user wishes to save a plot of the results
        plt.figure(figsize=(sz[2] * 2, sz[1] * 2))  # then initialize the figure and gridspec
        gs = gridspec.GridSpec(sz[1], sz[2])

    n_params = len(params_dict)
    param_cube = np.zeros((n_params, sz[1], sz[2]))  # zero arrays to store results...
    param_err_cube = np.zeros((n_params, sz[1], sz[2]))  # ...uncertainties...
    snr_map = np.zeros([sz[1], sz[2]])  # ...the S/N map...
    bin_map = np.zeros([sz[1], sz[2]])  # ... the binsize map
    norm_map = np.zeros([sz[1], sz[2]])  # ... and the flux normalization map

    # now begin the fitting
    # loop over pixels and fit 1d spectra
    print("Fitting...")
    for p in tqdm.tqdm(range(0, sz[2])):
        for q in range(0, sz[1]):
            for binsize in range(bin_lower, bin_upper + 1):
                xlo, xhi = p + np.array([-binsize, binsize + 1])  # get spatial indices of pixels to use in the
                ylo, yhi = q + np.array([-binsize, binsize + 1])  # fitting, binning if desired

                xlo = 0 if xlo <= 0 else xlo
                xhi = sz[2] if xhi > sz[2] else xhi  # set xlo/xhi to 0/Nx if pixel near boundaries
                ylo = 0 if ylo <= 0 else ylo
                yhi = sz[1] if yhi >= sz[1] else yhi  # do the same for ylo/yhi

                # Extract flux and error with proper NaN handling
                with np.errstate(invalid='ignore'):  # Suppress warnings for all-NaN slices
                    flux = np.nanmean(cube[:, ylo:yhi, xlo:xhi], axis=(1, 2))
                    err = np.nanmean(err_cube[:, ylo:yhi, xlo:xhi], axis=(1, 2))

                # Skip if flux is all NaN or invalid
                if np.isnan(flux).all() or not np.isfinite(flux).any():
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

                norm = np.nanmax(flux_m)  # get max value of flux array
                flux_m /= norm  # normalise flux and error to avoid very small values
                err_m /= norm
                norm_map[q, p] = norm

                # first, calculate chi2 without including emission  (*_c = "continuum")
                med_c, std_c = np.nanmedian(flux_m[no_line_m]), np.nanstd(flux_m[no_line_m])
                chi2_SLF = np.sum((flux_m - med_c) ** 2 / err_m ** 2)

                # now try to fit the emission lines
                try:
                    params, errors, bestfit, chi2_GF, result = fitspec_lmfit(wave_m, flux_m, err_m, model,
                                                                             params_dict, constraints)
                    SNR = (chi2_SLF - chi2_GF) ** 0.5

                    if SNR >= snr_cut:  # check if pixel meets S/N threshold
                        if save_grid:
                            ax = plt.subplot(gs[sz[1] - 1 - q, p])
                            ax.set_xlim([l - dl / 2, l + dl / 2])
                            ax.set_ylim([-0.2, 1.5])
                            X = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 1000)
                            ax.set_xticks([])
                            ax.set_yticks([])

                            ax.step(wave_m, flux_m, lw=2.5, where="mid", c="w", zorder=3)
                            ax.plot(X, model(X, *params), c="k", lw=2, alpha=1, zorder=3.5)

                            ax.fill_between((ax.get_xlim()[0], ax.get_xlim()[1]),
                                            ax.get_ylim()[0], ax.get_ylim()[1],
                                            color=cmap(color_norm_sn(SNR)), zorder=1)
                            ax.fill_between(wave, ax.get_ylim()[0], ax.get_ylim()[1], where=no_line == False,
                                            color="w", alpha=0.2)
                            ax.axvline(l0 * (1 + z_sys), c="k", ls="--", lw=1, alpha=0.75, zorder=2)

                        # store the results
                        param_cube[0:n_params, q, p] = params[0:n_params]
                        param_err_cube[0:n_params, q, p] = errors[0:n_params]
                        snr_map[q, p] = SNR
                        bin_map[q, p] = binsize + 1

                        break

                except ValueError as e:
                    if str(e) == "Residuals are not finite in the initial point.":
                        continue
                except RuntimeError as e:
                    if str(e) == "Optimal parameters not found: The maximum number of function evaluations is exceeded.":
                        continue
                except:
                    raise

    print("Done.")

    bin_map[bin_map == 0] = np.nan
    snr_map[snr_map == 0] = np.nan

    if save_grid:
        gs.update(hspace=0, wspace=0)
        plt.tight_layout()
        plt.savefig(grid_file)
        plt.close()

    return param_cube, param_err_cube, snr_map, bin_map, norm_map


# ------------------------------------------------------------------------------------------------------------

def fitcube_two_model(
        cube, err_cube, wl, z_sys, model_narrow, model_broad,
        params_dict_n, params_dict_b, constraints_n, constraints_b, l0, no_line,
        dl=0.05, snr_cut=5, bin_lower=0, bin_upper=0, snr_max=50,
        save_grid=False, grid_file="fit_grid_two_models.pdf",
        save_to=None, ref_header=None, detailed_plot=True, bic_threshold=10,
        use_rms=False
):
    color_norm_sn = matplotlib.colors.Normalize(vmin=snr_cut, vmax=snr_max)
    cmap = matplotlib.cm.plasma

    l = l0 * (1 + z_sys)
    gd = (wl >= l - dl) & (wl <= l + dl)
    cube = cube[gd, :, :]
    err_cube = err_cube[gd, :, :]
    wave = wl[gd]
    no_line = no_line[gd]

    sz = cube.shape
    if save_grid:
        plt.figure(figsize=(sz[2] * 2, sz[1] * 2))
        gs = gridspec.GridSpec(sz[1], sz[2])
        plt.title(grid_file.split("/")[-1], color="w")

    binned_cube = np.zeros((sz[0], sz[1], sz[2]))
    binned_err_cube = np.zeros((sz[0], sz[1], sz[2]))
    n_params = len(params_dict_b)  # Use broad model params as it has more parameters
    param_cube = np.zeros((n_params, sz[1], sz[2]))
    param_err_cube = np.zeros((n_params, sz[1], sz[2]))
    snr_map = np.zeros([sz[1], sz[2]])
    bin_map = np.zeros([sz[1], sz[2]])
    norm_map = np.zeros([sz[1], sz[2]])
    broad_map = np.zeros([sz[1], sz[2]])
    bic_map = np.zeros([sz[1], sz[2]])

    logging.info("Fitting started...")
    for p in tqdm.tqdm(range(0, sz[2])):
        for q in range(0, sz[1]):
            for binsize in range(bin_lower, bin_upper + 1):
                xlo, xhi = p + np.array([-binsize, binsize + 1])
                ylo, yhi = q + np.array([-binsize, binsize + 1])
                xlo = 0 if xlo <= 0 else xlo
                xhi = sz[2] if xhi > sz[2] else xhi
                ylo = 0 if ylo <= 0 else ylo
                yhi = sz[1] if yhi >= sz[1] else yhi

                # Extract flux and error with proper NaN handling
                with np.errstate(invalid='ignore'):  # Suppress warnings for all-NaN slices
                    flux = np.nanmean(cube[:, ylo:yhi, xlo:xhi], axis=(1, 2))
                    err = np.nanmean(err_cube[:, ylo:yhi, xlo:xhi], axis=(1, 2))

                # Skip if flux is all NaN or invalid
                if np.isnan(flux).all() or not np.isfinite(flux).any():
                    continue

                if use_rms:
                    with np.errstate(invalid='ignore'):  # Suppress warnings for nanstd
                        rms = np.nanstd(flux[no_line])
                        if np.isfinite(rms) and rms > 0:
                            err = rms * np.ones(len(flux))
                        else:
                            continue  # Skip if RMS calculation fails

                mask = ~np.isnan(flux)
                wave_m = wave[mask]
                flux_m = flux[mask]
                err_m = err[mask]
                no_line_m = no_line[mask]

                norm = np.nanmax(flux_m)
                flux_m /= norm
                err_m /= norm
                norm_map[q, p] = norm

                med_c, std_c = np.nanmedian(flux_m[no_line_m]), np.nanstd(flux_m[no_line_m])
                chi2_SLF = np.sum((flux_m - med_c) ** 2 / err_m ** 2)

                try:
                    params_n, errors_n, bestfit_n, chi2_n, result_n = fitspec_lmfit(
                        wave_m, flux_m, err_m, model_narrow, params_dict_n, constraints_n
                    )
                    bic_n = bic(flux_m, err_m, lambda p: model_narrow(wave_m, *p), params_n)

                    params_b, errors_b, bestfit_b, chi2_b, result_b = fitspec_lmfit(
                        wave_m, flux_m, err_m, model_broad, params_dict_b, constraints_b
                    )
                    bic_b = bic(flux_m, err_m, lambda p: model_broad(wave_m, *p), params_b)

                    delta_bic = bic_n - bic_b
                    logging.info(f"Pixel ({q},{p}) BIC_n={bic_n:.2f}, BIC_b={bic_b:.2f}, ΔBIC={delta_bic:.2f}")

                    if delta_bic > bic_threshold:
                        params, errors, bestfit, is_broad, chi2_GF = params_b, errors_b, bestfit_b, True, chi2_b
                    else:
                        params, errors, bestfit, is_broad, chi2_GF = params_n, errors_n, bestfit_n, False, chi2_n

                    SNR = (chi2_SLF - chi2_GF) ** 0.5

                    if SNR >= snr_cut:
                        if save_grid:
                            ax = plt.subplot(gs[sz[1] - 1 - q, p])
                            ax.set_xlim([l - dl / 2, l + dl / 2])
                            ax.set_ylim([-0.2, 1.5])
                            X = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 1000)
                            ax.set_xticks([])
                            ax.set_yticks([])

                            # Plot the data and error
                            ax.step(wave_m, flux_m, lw=1.5, where="mid", c="w", zorder=3)
                            ax.plot(X, model_broad(X, *params) if is_broad else model_narrow(X, *params),
                                    c="k" if is_broad else "c", lw=0.9, alpha=1, zorder=3.5)

                            # draw the individual broad components (if applicable) ---
                            if is_broad:
                                try:
                                    y_narrow_only = model_broad(X, *params[0:5], I_Ha_broad=0.0, sig_broad=1)
                                    y_broad_only = model_broad(X, *params) - y_narrow_only + params[4]
                                    ax.plot(X, y_narrow_only, c="k", ls="--", lw=1, alpha=0.9, zorder=3.5)
                                    ax.plot(X, y_broad_only, c="k", ls="--", lw=1, alpha=0.9, zorder=3.5)
                                except TypeError:
                                    pass
                                except Exception as e:
                                    logging.debug(f"Component plotting skipped at ({q},{p}): {e}")

                            ax.fill_between((ax.get_xlim()[0], ax.get_xlim()[1]),
                                            ax.get_ylim()[0], ax.get_ylim()[1],
                                            color=cmap(color_norm_sn(SNR)), zorder=1)
                            ax.fill_between(wave, ax.get_ylim()[0], ax.get_ylim()[1], where=no_line == True,
                                            color="w", alpha=0.2)
                            ax.axvline(l0 * (1 + z_sys), c="k", ls="--", lw=1, alpha=0.75, zorder=2)
                            if detailed_plot:
                                ax.text(0.02, 0.98, f'({q}, {p})', va="top", ha="left", transform=ax.transAxes, c="w")
                                ax.text(0.98, 0.98, fr'BIC$_{{\rm n}}$ = {bic_n:.1f}', va="top", ha="right",
                                        transform=ax.transAxes, c="w", fontsize=8)
                                ax.text(0.98, 0.9, fr'BIC$_{{\rm b}}$ = {bic_b:.1f}', va="top", ha="right",
                                        transform=ax.transAxes, c="w", fontsize=8)
                                ax.text(0.98, 0.82, fr'$\Delta$BIC = {delta_bic:.1f}', va="top", ha="right",
                                        transform=ax.transAxes, c="w", fontsize=8)

                        # Store results
                        if delta_bic > bic_threshold:
                            param_cube[0:len(params_b), q, p] = params_b[0:len(params_b)]
                            param_err_cube[0:len(params_b), q, p] = errors_b[0:len(params_b)]
                        else:
                            n_params_n = len(params_n)
                            param_cube[0:n_params_n, q, p] = params_n[0:n_params_n]
                            param_err_cube[0:n_params_n, q, p] = errors_n[0:n_params_n]
                            # Set remaining params to NaN if narrow model has fewer params
                            if n_params_n < n_params:
                                param_cube[n_params_n:, q, p] = np.nan
                                param_err_cube[n_params_n:, q, p] = np.nan

                        binned_cube[:, q, p] = flux
                        binned_err_cube[:, q, p] = err
                        snr_map[q, p] = SNR
                        bin_map[q, p] = binsize + 1
                        broad_map[q, p] = int(is_broad) + 1
                        bic_map[q, p] = delta_bic
                        break

                except ValueError as e:
                    if str(e) == "Residuals are not finite in the initial point.":
                        logging.warning(f"Pixel ({q},{p}) ValueError: {e}")
                        continue
                except RuntimeError as e:
                    if str(e) == "Optimal parameters not found: The maximum number of function evaluations is exceeded.":
                        logging.warning(f"Pixel ({q},{p}) RuntimeError: {e}")
                        continue
                except Exception as e:
                    logging.error(f"Unexpected error at pixel ({q},{p}) binsize={binsize}: {e}")
                    continue

    logging.info("Fitting complete.")

    binned_cube[binned_cube == 0] = np.nan
    binned_err_cube[binned_err_cube == 0] = np.nan
    param_cube[param_cube == 0] = np.nan
    param_err_cube[param_err_cube == 0] = np.nan
    bin_map[bin_map == 0] = np.nan
    snr_map[snr_map == 0] = np.nan
    broad_map[broad_map == 0] = np.nan
    bic_map[bic_map == 0] = np.nan

    if save_grid:
        gs.update(hspace=0, wspace=0)
        plt.tight_layout()
        plt.savefig(grid_file)
        plt.close()

    # persist outputs if requested
    if save_to is not None:
        meta = {
            "z_sys": z_sys,
            "l0": l0,
            "dl": dl,
            "snr_cut": snr_cut,
            "bin_lower": bin_lower,
            "bin_upper": bin_upper,
            "snr_max": snr_max,
            "grid_file": grid_file,
        }
        _save_results(save_to, meta=meta, param_cube=param_cube,
                      param_err_cube=param_err_cube, snr_map=snr_map,
                      bin_map=bin_map, norm_map=norm_map, broad_map=broad_map,
                      bic_map=bic_map, wave=wave, binned_cube=binned_cube,
                      binned_err_cube=binned_err_cube, ref_header=ref_header)

    # return param_cube, param_err_cube, snr_map, bin_map, norm_map, broad_map, bic_map