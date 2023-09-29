Spectral line fitting with cubespecfit 
=======================================

This package is designed to enable adaptive binning and spaxel-by-spaxel fitting of emission lines in a 3-D datacube.
I have implemented this code primarily with observations of the H-alpha and [NII] emission lines 
(see [this paper using VLT/KMOS data](https://ui.adsabs.harvard.edu/abs/2023arXiv230105720B/abstract)
and [this one using JWST NIRSpec data](https://ui.adsabs.harvard.edu/abs/2023arXiv230710412B/abstract)). This allows
the user to derive the spatial variation of galaxy properties such as star-formation rate, metallicity etc. and
also to produce maps such as the one shown below.

The code loops over every spaxel in a given cube and attempts to fit a model provided by the user (some models
are available to be imported from cubespecfit.models). If the signal-to-noise ratio (S/N) of the fit does not achieve a 
user-defined threshold, the code bins that pixel with neighbouring spaxels up to a maximum which is also defined by
the user.

Questions about this code can be directed to Jack Birkin: [jbirkin@tamu.edu](mailto:jbirkin@tamu.edu)

![Example grid spectra](example_map.png)