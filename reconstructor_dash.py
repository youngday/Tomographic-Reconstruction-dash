import base64
import io
import os
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple

import dash
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from dash import Input, Output, State, callback_context, dcc, html
from PIL import Image
from scipy.interpolate import RectBivariateSpline
from skimage.transform import iradon, radon, rotate

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Set plotly template
pio.templates.default = "plotly_white"

# Performance settings
IMAGE_TARGET_SIZE = 256
ANGLE_STEP = 2
MAX_WORKERS = min(4, os.cpu_count() or 1)


class CTReconstructor:
    """CT Reconstruction class with methods from reconstructor.py"""

    def __init__(self):
        self.image = None
        self.thetas = None
        self.rs = None
        self.p = None
        self.fBP = None
        self.f = None
        self.reconstruction_img = None
        self.sinogram = None

    def load_image_from_upload(self, contents):
        """Load image from uploaded content"""
        try:
            # Decode the base64 string
            content_type, content_string = contents.split(",")
            decoded = base64.b64decode(content_string)

            # Open image and convert to numpy array
            with Image.open(io.BytesIO(decoded)) as img:
                img = img.convert("L")
                # Resize to target size for better performance
                if max(img.size) > IMAGE_TARGET_SIZE:
                    img.thumbnail(
                        (IMAGE_TARGET_SIZE, IMAGE_TARGET_SIZE), Image.Resampling.LANCZOS
                    )
                image = np.array(img, dtype=np.float32)

            # Normalize image
            image = image / (np.max(image) + 1e-8)

            # Optimized padding calculation
            padded_size = int(np.sqrt(2) * max(image.shape)) + 10
            padded_image = np.zeros((padded_size, padded_size), dtype=np.float32)

            start_row = (padded_size - image.shape[0]) // 2
            start_col = (padded_size - image.shape[1]) // 2
            padded_image[
                start_row : start_row + image.shape[0],
                start_col : start_col + image.shape[1],
            ] = image

            self.image = padded_image
            return True, "Image loaded successfully!"

        except Exception as e:
            return False, f"Error loading image: {str(e)}"

    def perform_radon_transform(self):
        """Perform Radon transform"""
        if self.image is None:
            return False, "No image loaded"

        self.thetas = np.arange(0, 180, ANGLE_STEP) * np.pi / 180
        self.rs = np.linspace(-1, 1, self.image.shape[0])

        # Vectorized rotation computation
        angles_deg = self.thetas * 180 / np.pi
        rotations = np.array([rotate(self.image, angle) for angle in angles_deg])

        dr = np.diff(self.rs)[0]

        # Vectorized projection computation
        self.p = np.sum(rotations, axis=1) * dr
        self.p = self.p.T

        return True, "Radon transform completed"

    def filtered_back_projection(self):
        """Perform filtered back projection"""
        if self.p is None or self.rs is None or self.thetas is None:
            return False, "Radon transform not performed"

        dtheta = np.diff(self.thetas)[0]

        # Use linear interpolation for better performance
        p_interp = RectBivariateSpline(self.rs, self.thetas, self.p, kx=1, ky=1)

        # Vectorized computation using meshgrid
        n_points = len(self.rs)
        X, Y = np.meshgrid(self.rs, self.rs, indexing="ij")

        # Precompute all r values for vectorized interpolation
        r_vals = X[:, :, np.newaxis] * np.cos(self.thetas) + Y[
            :, :, np.newaxis
        ] * np.sin(self.thetas)

        # Evaluate interpolation for all points at once
        fBP = p_interp.ev(r_vals.reshape(-1), np.tile(self.thetas, n_points * n_points))
        fBP = fBP.reshape(n_points, n_points, len(self.thetas))
        self.fBP = np.sum(fBP, axis=2) * dtheta

        return True, "Filtered back projection completed"

    def fourier_reconstruction(self):
        """Perform Fourier reconstruction"""
        if self.p is None or self.rs is None or self.thetas is None:
            return False, "Radon transform not performed"

        dtheta = np.diff(self.thetas)[0]
        dr = np.diff(self.rs)[0]

        # Use numpy FFT
        P = np.fft.fft(self.p, axis=0)
        nu = np.fft.fftfreq(P.shape[0], d=dr)

        # Vectorized frequency domain filtering
        integrand = P.T * np.abs(nu)
        integrand = integrand.T

        # Use numpy's IFFT for consistency
        p_p = np.real(np.fft.ifft(integrand, axis=0))

        # Use linear interpolation for reconstruction
        p_p_interp = RectBivariateSpline(self.rs, self.thetas, p_p, kx=1, ky=1)

        # Vectorized reconstruction
        n_points = len(self.rs)
        X, Y = np.meshgrid(self.rs, self.rs, indexing="ij")
        r_vals = X[:, :, np.newaxis] * np.cos(self.thetas) + Y[
            :, :, np.newaxis
        ] * np.sin(self.thetas)

        f = p_p_interp.ev(r_vals.reshape(-1), np.tile(self.thetas, n_points * n_points))
        f = f.reshape(n_points, n_points, len(self.thetas))
        self.f = np.sum(f, axis=2) * dtheta

        return True, "Fourier reconstruction completed"

    def scikit_reconstruction(self):
        """Perform scikit-image reconstruction"""
        if self.image is None:
            return False, "No image loaded"

        theta = np.arange(0.0, 180.0, 5)
        self.sinogram = radon(self.image, theta=theta)
        self.reconstruction_img = iradon(self.sinogram, theta=theta, filter_name="ramp")

        return True, "Scikit-image reconstruction completed"


# Initialize Dash app
app = dash.Dash(__name__, title="CT Reconstruction Dashboard")
app.title = "CT Reconstruction Dashboard"

# Initialize reconstructor
reconstructor = CTReconstructor()

# App layout
app.layout = html.Div(
    [
        html.Div(
            [
                html.H1(
                    "CT Reconstruction Dashboard",
                    style={
                        "textAlign": "center",
                        "color": "#2c3e50",
                        "marginBottom": "30px",
                    },
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                dcc.Upload(
                                    id="upload-image",
                                    children=html.Div(
                                        ["Drag and Drop or ", html.A("Select an Image")]
                                    ),
                                    style={
                                        "width": "100%",
                                        "height": "60px",
                                        "lineHeight": "60px",
                                        "borderWidth": "1px",
                                        "borderStyle": "dashed",
                                        "borderRadius": "5px",
                                        "textAlign": "center",
                                        "margin": "10px",
                                    },
                                    multiple=False,
                                ),
                                html.Div(id="upload-status", style={"margin": "10px"}),
                            ],
                            className="six columns",
                        ),
                        html.Div(
                            [
                                html.Button(
                                    "Load Sample Image",
                                    id="load-sample",
                                    n_clicks=0,
                                    style={
                                        "width": "100%",
                                        "height": "40px",
                                        "margin": "10px",
                                    },
                                ),
                                html.Button(
                                    "Run Reconstruction",
                                    id="run-reconstruction",
                                    n_clicks=0,
                                    style={
                                        "width": "100%",
                                        "height": "40px",
                                        "margin": "10px",
                                    },
                                ),
                                html.Div(
                                    id="reconstruction-status", style={"margin": "10px"}
                                ),
                            ],
                            className="six columns",
                        ),
                    ],
                    className="row",
                ),
                html.Hr(),
                # Original Image
                html.Div(
                    [
                        html.H3("Original Image", style={"textAlign": "center"}),
                        dcc.Graph(id="original-image"),
                    ],
                    style={"marginBottom": "30px"},
                ),
                # Reconstruction Results
                html.Div(
                    [
                        html.H3(
                            "Reconstruction Results", style={"textAlign": "center"}
                        ),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.H4("Filtered Back Projection"),
                                        dcc.Graph(id="fBP-reconstruction"),
                                    ],
                                    className="six columns",
                                ),
                                html.Div(
                                    [
                                        html.H4("Fourier Reconstruction"),
                                        dcc.Graph(id="fourier-reconstruction"),
                                    ],
                                    className="six columns",
                                ),
                            ],
                            className="row",
                        ),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.H4("Scikit-image Reconstruction"),
                                        dcc.Graph(id="scikit-reconstruction"),
                                    ],
                                    className="six columns",
                                ),
                                html.Div(
                                    [
                                        html.H4("Sinogram"),
                                        dcc.Graph(id="sinogram"),
                                    ],
                                    className="six columns",
                                ),
                            ],
                            className="row",
                        ),
                    ]
                ),
                # Slice Analysis
                html.Div(
                    [
                        html.H3("Slice Analysis", style={"textAlign": "center"}),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.H4("Fourier Reconstruction Slice"),
                                        dcc.Graph(id="fourier-slice"),
                                    ],
                                    className="six columns",
                                ),
                                html.Div(
                                    [
                                        html.H4("Scikit-image Reconstruction Slice"),
                                        dcc.Graph(id="scikit-slice"),
                                    ],
                                    className="six columns",
                                ),
                            ],
                            className="row",
                        ),
                        html.Div(
                            [
                                html.Label("Slice Row:"),
                                dcc.Slider(
                                    id="slice-slider",
                                    min=0,
                                    max=255,
                                    value=110,
                                    marks={0: "0", 127: "127", 255: "255"},
                                    step=1,
                                ),
                            ],
                            style={"margin": "20px"},
                        ),
                    ]
                ),
            ],
            style={"padding": "20px"},
        )
    ]
)


# Callbacks
@app.callback(
    [Output("upload-status", "children"), Output("original-image", "figure")],
    [Input("upload-image", "contents")],
)
def update_uploaded_image(contents):
    if contents is not None:
        success, message = reconstructor.load_image_from_upload(contents)
        if success:
            # Create figure for original image
            fig = go.Figure()
            fig.add_trace(
                go.Heatmap(z=reconstructor.image, colorscale="gray", showscale=True)
            )
            fig.update_layout(
                width=400,
                height=400,
                title_text="Uploaded Image",
                margin=dict(l=20, r=20, t=40, b=20),
            )
            return html.Div(message, style={"color": "green"}), fig
        else:
            return html.Div(message, style={"color": "red"}), go.Figure()

    return html.Div("No image uploaded"), go.Figure()


@app.callback(
    Output("reconstruction-status", "children"),
    [Input("run-reconstruction", "n_clicks"), Input("load-sample", "n_clicks")],
    prevent_initial_call=True,
)
def run_reconstruction(run_clicks, sample_clicks):
    ctx = callback_context
    if not ctx.triggered:
        return ""

    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if button_id == "load-sample":
        # Load sample image (you can replace this with your sample image path)
        sample_image_path = "data/ct_scan_1.jpg"  # Adjust path as needed
        if os.path.exists(sample_image_path):
            with Image.open(sample_image_path) as img:
                img = img.convert("L")
                if max(img.size) > IMAGE_TARGET_SIZE:
                    img.thumbnail(
                        (IMAGE_TARGET_SIZE, IMAGE_TARGET_SIZE), Image.Resampling.LANCZOS
                    )
                image = np.array(img, dtype=np.float32)

            image = image / (np.max(image) + 1e-8)
            padded_size = int(np.sqrt(2) * max(image.shape)) + 10
            padded_image = np.zeros((padded_size, padded_size), dtype=np.float32)
            start_row = (padded_size - image.shape[0]) // 2
            start_col = (padded_size - image.shape[1]) // 2
            padded_image[
                start_row : start_row + image.shape[0],
                start_col : start_col + image.shape[1],
            ] = image
            reconstructor.image = padded_image

            return html.Div(
                "Sample image loaded successfully!", style={"color": "green"}
            )
        else:
            return html.Div(
                "Sample image not found. Please upload an image.",
                style={"color": "red"},
            )

    elif button_id == "run-reconstruction":
        start_time = time.time()

        # Perform all reconstruction steps
        steps = [
            ("Radon Transform", reconstructor.perform_radon_transform),
            ("Filtered Back Projection", reconstructor.filtered_back_projection),
            ("Fourier Reconstruction", reconstructor.fourier_reconstruction),
            ("Scikit-image Reconstruction", reconstructor.scikit_reconstruction),
        ]

        results = []
        for step_name, step_func in steps:
            success, message = step_func()
            results.append(f"{step_name}: {'✅' if success else '❌'} {message}")

        total_time = time.time() - start_time
        results.append(f"Total time: {total_time:.2f} seconds")

        return html.Div(
            [
                html.H4("Reconstruction Results:"),
                html.Ul([html.Li(result) for result in results]),
            ],
            style={"color": "green"},
        )


@app.callback(
    [
        Output("fBP-reconstruction", "figure"),
        Output("fourier-reconstruction", "figure"),
        Output("scikit-reconstruction", "figure"),
        Output("sinogram", "figure"),
    ],
    [Input("run-reconstruction", "n_clicks")],
)
def update_reconstruction_figures(n_clicks):
    if n_clicks is None or n_clicks == 0:
        return go.Figure(), go.Figure(), go.Figure(), go.Figure()

    # Create figures for each reconstruction method
    fig_fBP = go.Figure()
    if reconstructor.fBP is not None:
        fig_fBP.add_trace(
            go.Heatmap(z=reconstructor.fBP, colorscale="gray", showscale=True)
        )
    fig_fBP.update_layout(
        width=400,
        height=400,
        title_text="Filtered Back Projection",
        margin=dict(l=20, r=20, t=40, b=20),
    )

    fig_fourier = go.Figure()
    if reconstructor.f is not None:
        fig_fourier.add_trace(
            go.Heatmap(z=reconstructor.f, colorscale="gray", showscale=True)
        )
    fig_fourier.update_layout(
        width=400,
        height=400,
        title_text="Fourier Reconstruction",
        margin=dict(l=20, r=20, t=40, b=20),
    )

    fig_scikit = go.Figure()
    if reconstructor.reconstruction_img is not None:
        fig_scikit.add_trace(
            go.Heatmap(
                z=reconstructor.reconstruction_img, colorscale="gray", showscale=True
            )
        )
    fig_scikit.update_layout(
        width=400,
        height=400,
        title_text="Scikit-image Reconstruction",
        margin=dict(l=20, r=20, t=40, b=20),
    )

    fig_sinogram = go.Figure()
    if reconstructor.sinogram is not None:
        fig_sinogram.add_trace(
            go.Heatmap(z=reconstructor.sinogram, colorscale="gray", showscale=True)
        )
    fig_sinogram.update_layout(
        width=400,
        height=400,
        title_text="Sinogram",
        margin=dict(l=20, r=20, t=40, b=20),
    )

    return fig_fBP, fig_fourier, fig_scikit, fig_sinogram


@app.callback(
    [Output("fourier-slice", "figure"), Output("scikit-slice", "figure")],
    [Input("slice-slider", "value"), Input("run-reconstruction", "n_clicks")],
)
def update_slice_figures(slice_row, n_clicks):
    if n_clicks is None or n_clicks == 0:
        return go.Figure(), go.Figure()

    # Fourier reconstruction slice
    fig_fourier_slice = go.Figure()
    if reconstructor.f is not None and slice_row < reconstructor.f.shape[0]:
        fig_fourier_slice.add_trace(
            go.Scatter(
                x=list(range(len(reconstructor.f[slice_row]))),
                y=reconstructor.f[slice_row],
                mode="lines",
                line=dict(width=2, color="blue"),
            )
        )
    fig_fourier_slice.update_layout(
        width=400,
        height=400,
        title_text=f"Slice at Row {slice_row} - Fourier",
        xaxis_title="Column",
        yaxis_title="Intensity",
        margin=dict(l=20, r=20, t=40, b=20),
    )

    # Scikit reconstruction slice
    fig_scikit_slice = go.Figure()
    if (
        reconstructor.reconstruction_img is not None
        and slice_row < reconstructor.reconstruction_img.shape[0]
    ):
        fig_scikit_slice.add_trace(
            go.Scatter(
                x=list(range(len(reconstructor.reconstruction_img[slice_row]))),
                y=reconstructor.reconstruction_img[slice_row],
                mode="lines",
                line=dict(width=2, color="red"),
            )
        )
    fig_scikit_slice.update_layout(
        width=400,
        height=400,
        title_text=f"Slice at Row {slice_row} - Scikit",
        xaxis_title="Column",
        yaxis_title="Intensity",
        margin=dict(l=20, r=20, t=40, b=20),
    )

    return fig_fourier_slice, fig_scikit_slice


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
