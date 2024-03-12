import os
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.cm import inferno
import torch
import numpy as np
import caustics
from app_configs import (
    lens_slider_configs,
    source_slider_configs,
    name_map,
    default_params,
)

st.set_page_config(layout="wide")
css = """
<style>
    section.main > div {max-width:75rem}
</style>
"""
st.markdown(css, unsafe_allow_html=True)
logo_url = (
    "https://github.com/Ciela-Institute/caustics/raw/main/media/caustics_logo_white.png?raw=true"
)
st.sidebar.image(logo_url)
docs_url = "https://caustics.readthedocs.io/"
st.sidebar.write("Check out the [documentation](%s)!" % docs_url)

lens_menu = st.sidebar.multiselect(
    "Select your Lens(es)", lens_slider_configs.keys(), default=["EPL", "Shear"]
)
source_menu = st.sidebar.radio("Select your Source (more to come)", source_slider_configs.keys())
st.sidebar.write("Get it for yourself: pip install caustics")

st.title("Caustics Gravitational Lensing Simulator")
st.header(f"{'+'.join(lens_menu)} and {source_menu} Source")
simulation_size = st.number_input(
    "Simulation resolution", min_value=32, value=500 if source_menu == "Pixelated" else 64
)
fov = 6.5
deltam = fov / simulation_size
# Create a two-column layout
col1, col2, col3 = st.columns([4, 4, 5])

# Sliders for lens parameters in the first column
with col1:
    st.header(r"$\textsf{\tiny Lens Parameters}$", divider="blue")
    # z_lens = st.slider("Lens redshift", min_value=0.0, max_value=10.0, step=0.01)
    x_lens = []
    for lens in lens_menu:
        for param, label, bounds in lens_slider_configs[lens]:
            x_lens.append(
                st.slider(label, min_value=bounds[0], max_value=bounds[1], value=bounds[2])
            )

    x_lens = torch.tensor(x_lens)

with col2:
    st.header(r"$\textsf{\tiny Source Parameters}$", divider="blue")
    # z_source = st.slider("Source redshift", min_value=z_lens, max_value=10.0, step=0.01)
    if source_menu == "Pixelated":
        source_file = st.file_uploader(
            "Upload a source image", type=["png", "jpg"], accept_multiple_files=False
        )
        if source_file is None:
            selfloc = os.path.dirname(os.path.abspath(__file__))
            source_file = os.path.join(selfloc, "logo.png")
        img = plt.imread(source_file)
        source_shape = img.shape[:-1][::-1]
        source_img = torch.tensor(img).permute(2, 0, 1).float()
        if torch.any(source_img > 1).item():
            source_img /= 255.0
        x_source = torch.tensor([])
        src_pixelscale = fov / (max(source_shape))
        fov_scale = st.slider("FOV scale", min_value=0.1, max_value=2.0, value=1.0)
    else:
        x_source = []
        for param, label, bounds in source_slider_configs[source_menu]:
            x_source.append(
                st.slider(label, min_value=bounds[0], max_value=bounds[1], value=bounds[2])
            )
        x_source = torch.tensor(x_source)
x_all = torch.cat((x_lens, x_source))
z_lens = 1.0
z_source = 2.0
cosmology = caustics.FlatLambdaCDM(name="cosmo")
lenses = []
for lens in lens_menu:
    lenses.append(name_map[lens](cosmology, **default_params[lens], z_l=z_lens))
lens = caustics.SinglePlane(lenses=lenses, cosmology=cosmology, z_l=z_lens)
if source_menu == "Pixelated":
    src = list(
        name_map[source_menu](
            name="src",
            image=img,
            pixelscale=src_pixelscale * fov_scale,
            **default_params[source_menu],
        )
        for img in source_img
    )
    minisim = list(
        caustics.Lens_Source(
            lens=lens,
            source=subsrc,
            pixelscale=deltam,
            pixels_x=simulation_size,
            z_s=z_source,
        )
        for subsrc in src
    )
else:
    src = name_map[source_menu](name="src", **default_params[source_menu])
    minisim = caustics.Lens_Source(
        lens=lens, source=src, pixelscale=deltam, pixels_x=simulation_size, z_s=z_source
    )


# Plot the caustic trace and lensed image in the second column
with col3:
    st.header(r"$\textsf{\tiny Visualization}$", divider="blue")

    # Plot the unlensed image
    if source_menu == "Pixelated":
        st.image(
            np.stack(
                list(subsim(x_all, lens_source=False).detach().numpy() for subsim in minisim),
                axis=2,
            ),
            caption="Unlensed image",
            use_column_width="always",
            clamp=True,
        )
    else:
        res = minisim(x_all, lens_source=False).numpy()
        res = (res - np.min(res)) / (np.max(res) - np.min(res))
        st.image(
            inferno(res),
            caption="Unlensed image",
            use_column_width="always",
            clamp=True,
        )

    if source_menu == "Pixelated":
        st.image(
            np.stack(
                list(subsim(x_all, lens_source=True).detach().numpy() for subsim in minisim),
                axis=2,
            ),
            caption="Lensed image",
            use_column_width="always",
            clamp=True,
        )
    else:
        res = minisim(x_all, lens_source=True).numpy()
        res = (res - np.min(res)) / (np.max(res) - np.min(res))
        st.image(
            inferno(res),
            caption="Lensed image",
            use_column_width="always",
            clamp=True,
        )
