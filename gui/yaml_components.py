base_yaml = """
cosmology: &cosmo
  name: cosmo
  kind: FlatLambdaCDM

src: &src
  name: source
  kind: {sourcename}
{sourceparams}
{sourcekwargs}

{lenses}

lens: &lens
  name: lens
  kind: SinglePlane
  init_kwargs:
    cosmology: *cosmo
    lenses:
{lenslist}
  params:
    z_l: 1.0

lnslt: &lnslt
  name: lenslight
  kind: Sersic
  params:
    x0: 0.0
    y0: 0.0
    q: 0.8
    phi: 0.0
    n: 1.0
    Re: 1.0
    Ie: 0.0001

simulator:
  name: sim
  kind: LensSource
  init_kwargs:
    lens: *lens
    source: *src
    pixelscale: {pixelscale}
    pixels_x: {numpixels}
    lens_light: *lnslt
  params:
    z_s: 2.0
"""

lens_yaml = """
lens: &{lensname}
  name: {lensname}
  kind: {lenskind}
  params:
    z_l: 1.0
{lensparams}
  init_kwargs:
    cosmology: *cosmo
"""
