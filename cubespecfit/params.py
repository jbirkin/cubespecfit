"""Build parameter dictionaries in physical units.

`fitspec_lmfit` and `fitcube` want a `params_dict` whose `sig` and `z` entries are in
wavelength units. Converting velocities to microns by hand is where miscalibrated bounds
come from: the example notebook did it six times with expressions like

    sig_guess = 300 / 2.35 / 3e05 * ((1 + z_sys) * 0.65646)

which nobody can check by eye. A bound that is wrong in that form is invisible — it does
not raise, it just quietly pins a parameter and the map looks plausible.

The helpers here take km/s and return the dictionary the fitting code expects, so bounds
can be read and argued about in the units the physics is actually in.

Example
-------
    from cubespecfit.params import build_params

    params = build_params(z_sys=3.760, l0=0.65646,
                          dv=500.,                 # +/- 500 km/s of z_sys
                          fwhm=(0., 1500.),        # intrinsic FWHM range, km/s
                          fwhm_guess=300.,
                          I_Ha={'value': 0.005, 'min': 5e-4, 'max': 0.1},
                          NIIHa={'value': 0.2, 'min': 0.0, 'max': 3.0},
                          c={'value': 0.5, 'min': 0.1, 'max': 1.0})
"""
from __future__ import annotations

__all__ = ['C_KMS', 'FWHM_PER_SIGMA', 'velocity_to_sigma', 'sigma_to_velocity',
           'redshift_bounds', 'build_params']

C_KMS = 299792.458          # speed of light, km/s
FWHM_PER_SIGMA = 2.3548200  # 2*sqrt(2*ln2)


def velocity_to_sigma(fwhm_kms: float, l_obs: float) -> float:
    """Convert a velocity FWHM (km/s) to a Gaussian sigma in the units of `l_obs`."""
    return fwhm_kms / FWHM_PER_SIGMA / C_KMS * l_obs


def sigma_to_velocity(sigma: float, l_obs: float) -> float:
    """Inverse of `velocity_to_sigma` — sigma in wavelength units to FWHM in km/s."""
    return sigma * FWHM_PER_SIGMA * C_KMS / l_obs


def redshift_bounds(z_sys: float, dv_kms: float) -> tuple[float, float]:
    """Redshift bounds spanning +/- `dv_kms` about `z_sys`."""
    dz = dv_kms / C_KMS * (1 + z_sys)
    return z_sys - dz, z_sys + dz


def build_params(z_sys, l0, model=None, dv=500.0, fwhm=(0.0, 1500.0),
                 fwhm_guess=300.0, **extra):
    """Assemble a `params_dict` with `z` and `sig` specified in km/s.

    Parameters
    ----------
    z_sys : float
        Systemic redshift; the initial guess for `z`.
    l0 : float
        Rest wavelength of the primary line, in the units of the data (microns here).
        Used to convert velocities to wavelength units at the observed wavelength.
    model : callable, optional
        One of the functions in `cubespecfit.models`. If given, the returned dictionary
        is ordered to match the model's signature, so the planes of `param_cube` line up
        with the model arguments. Strongly recommended — the plane order is what the
        plotting code indexes.
    dv : float
        Half-width of the redshift search, km/s. 500 km/s is generous for a systemic
        redshift that is already approximately known.
    fwhm : (float, float)
        (min, max) intrinsic velocity FWHM, km/s. **The minimum should normally be 0**:
        the models add the instrumental width in quadrature, so the spectral resolution
        already sets the floor. A non-zero minimum imposes a second, redundant floor and
        will pin narrow lines against it.
    fwhm_guess : float
        Initial intrinsic FWHM, km/s.
    **extra
        Further parameters as lmfit-style dicts, e.g.
        ``I_Ha={'value': 0.005, 'min': 5e-4, 'max': 0.1}``.

    Returns
    -------
    dict
        A `params_dict` suitable for `fitcube` / `fitspec_lmfit`.
    """
    l_obs = l0 * (1 + z_sys)
    z_min, z_max = redshift_bounds(z_sys, dv)
    fwhm_min, fwhm_max = fwhm

    if fwhm_min < 0 or fwhm_max <= fwhm_min:
        raise ValueError(f"fwhm must be (min, max) with max > min >= 0, got {fwhm}")
    if not (fwhm_min <= fwhm_guess <= fwhm_max):
        raise ValueError(
            f"fwhm_guess {fwhm_guess} km/s lies outside the fwhm bounds {fwhm}"
        )

    params = {
        'z':   {'value': z_sys, 'min': z_min, 'max': z_max},
        'sig': {'value': velocity_to_sigma(fwhm_guess, l_obs),
                'min':   velocity_to_sigma(fwhm_min, l_obs),
                'max':   velocity_to_sigma(fwhm_max, l_obs)},
    }
    params.update(extra)

    if model is None:
        return params

    # Order to match the model signature so param_cube planes are predictable.
    import inspect
    names = [n for n in inspect.signature(model).parameters if n != 'x']
    missing = [n for n in params if n not in names]
    if missing:
        raise ValueError(
            f"parameters {missing} are not arguments of {model.__name__}; "
            f"it accepts {names}"
        )
    return {n: params[n] for n in names if n in params}
