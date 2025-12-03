import os
import numpy as np
from astropy.io import fits

_WCS_KEYS_2D = [
    # core celestial WCS
    "WCSAXES", "RADESYS", "EQUINOX", "LONPOLE", "LATPOLE",
    "CTYPE1", "CTYPE2", "CUNIT1", "CUNIT2",
    "CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2",
    "CD1_1", "CD1_2", "CD2_1", "CD2_2",
    "PC1_1", "PC1_2", "PC2_1", "PC2_2", "CDELT1", "CDELT2",
    "CROTA2",
    "A_ORDER", "B_ORDER", "AP_ORDER", "BP_ORDER",
]

def _propagate_celestial_wcs(dst_hdr, ref_hdr):
    """
    Copy 2D celestial WCS cards from ref_hdr into dst_hdr.
    """
    if ref_hdr is None:
        return

    # Remove conflicting representations if needed
    has_cd = any(k in ref_hdr for k in ("CD1_1", "CD1_2", "CD2_1", "CD2_2"))
    if has_cd:
        for k in ("PC1_1","PC1_2","PC2_1","PC2_2","CDELT1","CDELT2"):
            if k in dst_hdr:
                del dst_hdr[k]
    else:
        for k in ("CD1_1","CD1_2","CD2_1","CD2_2"):
            if k in dst_hdr:
                del dst_hdr[k]

    # Copy known cards if present in reference header
    for k in _WCS_KEYS_2D:
        if k in ref_hdr:
            dst_hdr[k] = ref_hdr[k]

    # Ensure NAXIS=2 in the new HDU header
    dst_hdr["NAXIS"] = 2


def _save_results(save_to, meta, param_cube, param_err_cube, snr_map, bin_map, norm_map,
                  broad_map, bic_map, wave, binned_cube, binned_err_cube, ref_header=None):
    """
    Save results to .npz or .fits
    If ref_header is provided, its 2D celestial WCS is copied into all 2D image maps.
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_to)), exist_ok=True)

    if str(save_to).lower().endswith(".fits"):
        phdr = fits.Header()
        for k, v in meta.items():
            key = (k[:8].upper() if isinstance(k, str) else "META")
            try:
                phdr[key] = v
            except Exception:
                phdr[key] = str(v)
        hdus = [fits.PrimaryHDU(header=phdr)]

        def _img(name, data, ref_hdr=None, bunit=None):
            h = fits.Header()
            h["EXTNAME"] = name.upper()
            if bunit is not None:
                h["BUNIT"] = bunit
            # If this is a 2D image, stamp its WCS
            if isinstance(data, np.ndarray) and data.ndim == 2:
                _propagate_celestial_wcs(h, ref_hdr)
            return fits.ImageHDU(data=data, header=h)

        # 3D cubes (params/errors) are (nparam, y, x)
        hdus += [
            _img("BINNED_CUBE", binned_cube, ref_hdr=ref_header),
            _img("BINNED_ERR_CUBE", binned_err_cube, ref_hdr=ref_header),
            _img("PARAM_CUBE", param_cube, ref_hdr=ref_header),
            _img("PARAM_ERR", param_err_cube, ref_hdr=ref_header),
            _img("SNR_MAP",   snr_map,   ref_hdr=ref_header),
            _img("BIN_MAP",   bin_map,   ref_hdr=ref_header),
            _img("NORM_MAP",  norm_map,  ref_hdr=ref_header),
            _img("BROAD_MAP", broad_map, ref_hdr=ref_header),
            _img("BIC_MAP",   bic_map,   ref_hdr=ref_header),
            _img("WAVE",      wave,      ref_hdr=None, bunit="um"),
            _img("MASK",      (~np.isnan(param_cube)).any(axis=0).astype(np.uint8), ref_hdr=ref_header),
        ]
        fits.HDUList(hdus).writeto(save_to, overwrite=True)
    else:
        np.savez_compressed(
            save_to,
            param_cube=param_cube,
            param_err_cube=param_err_cube,
            snr_map=snr_map,
            bin_map=bin_map,
            norm_map=norm_map,
            broad_map=broad_map,
            bic_map=bic_map,
            wave=wave,
            **{f"meta_{k}": v for k, v in meta.items()},
        )