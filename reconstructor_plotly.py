"""
Simple Async-optimized CT Reconstruction with parallel Plotly image saving
This version focuses on async image saving without modifying the core algorithm.
"""

import asyncio
import os
import shutil
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from loguru import logger
from PIL import Image
from plotly.subplots import make_subplots
from scipy.interpolate import RectBivariateSpline
from skimage.transform import iradon, radon, rotate

# Configure loguru - will be set up after output_dir is created

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Create output directory if it doesn't exist
output_dir = "outputs"
# Remove existing directory and recreate to ensure clean state

if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir, exist_ok=True)
logger.info(f"Output directory created: {output_dir}")

# Configure loguru now that output_dir exists
logger.remove()  # Remove default handler

# Console handler with colors
logger.add(
    sink=sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> |<level>{level: <8}</level>|:<cyan>{function}</cyan>:<cyan>{line}</cyan>|<level>{message}</level>",
    # level="INFO",
    level="DEBUG",
    colorize=True,
)

# File handler with full path
log_file_path = os.path.join(output_dir, "ct_reconstruction_async.log")
logger.add(
    sink=log_file_path,
    format="{time:YYYY-MM-DD HH:mm:ss.SSS}|{level: <8}|{function}:{line}|{message}",
    level="DEBUG",
    rotation="10 MB",  # Rotate when file reaches 10MB
    retention="7 days",  # Keep logs for 7 days
    compression="zip",  # Compress old logs
)

logger.info(f"Loguru logging configured - logs saved to {log_file_path}")

# Set plotly template
pio.templates.default = "plotly_white"

# Performance settings
IMAGE_TARGET_SIZE = 256  # Reduced from 256 for better performance
ANGLE_STEP = 2  # Increased from 2° for better performance
MAX_WORKERS = min(4, os.cpu_count() or 1)  # Limit concurrent workers


# Try to use Kaleido engine if available
KALEIDO_AVAILABLE = False
try:
    import importlib.util

    if importlib.util.find_spec("kaleido") is not None:
        KALEIDO_AVAILABLE = True
        logger.info("✅ Kaleido engine available for fast plotting")
    else:
        logger.warning(
            "⚠️ Kaleido not installed. Install with: pip install kaleido for better performance"
        )
except ImportError:
    logger.warning(
        "Kaleido not installed. Install with: pip install kaleido for better performance"
    )


async def save_image_async(fig, filename: str):
    """Asynchronously save a Plotly figure to file."""
    try:
        # Use thread pool executor to avoid blocking
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            await loop.run_in_executor(
                executor,
                lambda: fig.write_image(
                    filename,
                    engine="kaleido" if KALEIDO_AVAILABLE else None,
                    width=800,
                    height=600,
                    scale=1,
                    format="png",
                ),
            )
        logger.debug(f"✅ Async image saved: {filename}")
    except Exception as e:
        logger.error(f"❌ Failed to save image {filename}: {e}")


def load_image_fast(image_path: str | None = None) -> np.ndarray:
    """Fast image loading with memory-efficient preprocessing."""
    if image_path is None:
        image_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), ".", "data", "ct_scan_1.jpg"
        )
    logger.debug(f"Loading image from: {image_path}")
    logger.debug(f"Target image size: {IMAGE_TARGET_SIZE}, Angle step: {ANGLE_STEP}°")

    with Image.open(image_path) as img:
        img = img.convert("L")
        # Resize to target size for better performance
        if max(img.size) > IMAGE_TARGET_SIZE:
            img.thumbnail(
                (IMAGE_TARGET_SIZE, IMAGE_TARGET_SIZE), Image.Resampling.LANCZOS
            )
        image = np.array(img, dtype=np.float32)

    # Normalize image
    image = image / (np.max(image) + 1e-8)
    logger.debug(
        f"Image loaded and normalized. Shape: {image.shape}, Range: [{image.min():.3f}, {image.max():.3f}]"
    )

    # Optimized padding calculation
    padded_size = int(np.sqrt(2) * max(image.shape)) + 10
    padded_image = np.zeros((padded_size, padded_size), dtype=np.float32)

    start_row = (padded_size - image.shape[0]) // 2
    start_col = (padded_size - image.shape[1]) // 2
    padded_image[
        start_row : start_row + image.shape[0], start_col : start_col + image.shape[1]
    ] = image

    logger.debug(f"Image padded to shape: {padded_image.shape}")
    return padded_image


def perform_radon_transform_fast(
    image: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Fast Radon transform using Numba-optimized operations."""
    logger.info(f"Starting Radon transform for image shape: {image.shape}")
    thetas = np.arange(0, 180, ANGLE_STEP) * np.pi / 180
    rs = np.linspace(-1, 1, image.shape[0])
    logger.debug(f"Number of angles: {len(thetas)}, Number of r values: {len(rs)}")

    # Vectorized rotation computation
    angles_deg = thetas * 180 / np.pi
    rotations = np.array([rotate(image, angle) for angle in angles_deg])

    dtheta = np.diff(thetas)[0]
    dr = np.diff(rs)[0]

    # Vectorized projection computation
    p = np.sum(rotations, axis=1) * dr
    p = p.T  # Transpose to match expected shape

    return thetas, rs, p, dtheta


def filtered_back_projection_fast(
    p: np.ndarray, rs: np.ndarray, thetas: np.ndarray, dtheta: float
) -> np.ndarray:
    """Fast filtered back projection with vectorized interpolation."""
    # Use linear interpolation for better performance
    p_interp = RectBivariateSpline(rs, thetas, p, kx=1, ky=1)

    # Vectorized computation using meshgrid
    n_points = len(rs)
    X, Y = np.meshgrid(rs, rs, indexing="ij")

    # Precompute all r values for vectorized interpolation
    r_vals = X[:, :, np.newaxis] * np.cos(thetas) + Y[:, :, np.newaxis] * np.sin(thetas)

    # Evaluate interpolation for all points at once
    fBP = p_interp.ev(r_vals.reshape(-1), np.tile(thetas, n_points * n_points))
    fBP = fBP.reshape(n_points, n_points, len(thetas))
    fBP = np.sum(fBP, axis=2) * dtheta

    return fBP.astype(np.float32)


def fourier_transform_fast(
    p: np.ndarray, rs: np.ndarray, thetas: np.ndarray, dtheta: float
) -> np.ndarray:
    """Fast Fourier transform reconstruction."""
    # Use numpy FFT to avoid Dispatchable tuple issues
    P = np.fft.fft(p, axis=0)
    nu = np.fft.fftfreq(P.shape[0], d=np.diff(rs)[0])

    # Vectorized frequency domain filtering
    integrand = P.T * np.abs(nu)
    integrand = integrand.T

    # Use numpy's IFFT for consistency
    p_p = np.real(np.fft.ifft(integrand, axis=0))

    # Use linear interpolation for reconstruction
    p_p_interp = RectBivariateSpline(rs, thetas, p_p, kx=1, ky=1)

    # Vectorized reconstruction
    n_points = len(rs)
    X, Y = np.meshgrid(rs, rs, indexing="ij")
    r_vals = X[:, :, np.newaxis] * np.cos(thetas) + Y[:, :, np.newaxis] * np.sin(thetas)

    f = p_p_interp.ev(r_vals.reshape(-1), np.tile(thetas, n_points * n_points))
    f = f.reshape(n_points, n_points, len(thetas))
    f = np.sum(f, axis=2) * dtheta

    return f.astype(np.float32)


async def plot_all_images_async(
    image: np.ndarray,
    image_rot: np.ndarray,
    thetas: np.ndarray,
    rs: np.ndarray,
    p: np.ndarray,
    fBP: np.ndarray,
    f: np.ndarray,
    colormap: str = "gray",
):
    """Async plotting - create all figures first, then save them in parallel."""
    logger.info("Creating all figures for async saving...")

    # Create all figures first
    figures = []

    # 1. Original and rotated combined
    fig_combined = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Original Image", "Rotated Image (45°)"],
        horizontal_spacing=0.1,
    )
    fig_combined.add_trace(
        go.Heatmap(z=image, colorscale=colormap, showscale=True), row=1, col=1
    )
    fig_combined.add_trace(
        go.Heatmap(z=image_rot, colorscale=colormap, showscale=True), row=1, col=2
    )
    fig_combined.update_layout(
        width=1200, height=600, title_text="Original and Rotated Images"
    )
    figures.append((fig_combined, "original_and_rotated.png"))

    # 2. Single original
    fig_single = go.Figure()
    fig_single.add_trace(go.Heatmap(z=image, colorscale=colormap, showscale=True))
    fig_single.update_layout(width=600, height=600, title_text="Original Image")
    figures.append((fig_single, "original.png"))

    # 3. Projection vs r
    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(x=rs, y=p[:, 0], mode="lines", line=dict(width=2)))
    fig_line.update_layout(width=800, height=600, title_text="Projection vs r")
    figures.append((fig_line, "p_vs_r.png"))

    # 4. Sinogram
    fig_sinogram = go.Figure()
    fig_sinogram.add_trace(
        go.Heatmap(z=p, x=thetas, y=rs, colorscale=colormap, showscale=True)
    )
    fig_sinogram.update_layout(width=800, height=600, title_text="Sinogram")
    figures.append((fig_sinogram, "sinogram.png"))

    # 5. Reconstruction combined
    fig_recon = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Filtered Back Projection", "Fourier Reconstruction"],
        horizontal_spacing=0.05,
    )
    fig_recon.add_trace(
        go.Heatmap(z=fBP, colorscale=colormap, showscale=False), row=1, col=1
    )
    fig_recon.add_trace(
        go.Heatmap(z=f, colorscale=colormap, showscale=False), row=1, col=2
    )
    fig_recon.update_layout(
        width=1000,
        height=500,
        title_text="Reconstruction Results",
        showlegend=False,
        margin=dict(l=20, r=20, t=60, b=20),
        font=dict(size=12),
    )
    figures.append((fig_recon, "reconstruction_combined.png"))

    # 6. Slice
    fig_slice = go.Figure()
    fig_slice.add_trace(
        go.Scatter(
            x=list(range(len(f[110]))),
            y=f[110],
            mode="lines",
            line=dict(width=2),
        )
    )
    fig_slice.update_layout(
        width=800, height=600, title_text="Slice at Row 110 - Reconstructed Image"
    )
    figures.append((fig_slice, "slice_of_reconstructed_image.png"))

    # 7. Scikit-image reconstruction
    theta = np.arange(0.0, 180.0, 5)
    sinogram = radon(image, theta=theta)
    reconstruction_img = iradon(sinogram, theta=theta, filter_name="ramp")

    fig_sinogram_radon = go.Figure()
    fig_sinogram_radon.add_trace(
        go.Heatmap(z=sinogram, colorscale=colormap, showscale=True)
    )
    fig_sinogram_radon.update_layout(
        width=800, height=600, title_text="Sinogram (scikit-image radon)"
    )
    figures.append((fig_sinogram_radon, "sinogram_radon.png"))

    fig_iradon = go.Figure()
    fig_iradon.add_trace(
        go.Heatmap(z=reconstruction_img, colorscale=colormap, showscale=True)
    )
    fig_iradon.update_layout(
        width=600, height=600, title_text="Reconstruction (scikit-image iradon)"
    )
    figures.append((fig_iradon, "iradon_reconstruction.png"))

    fig_slice_iradon = go.Figure()
    fig_slice_iradon.add_trace(
        go.Scatter(
            x=list(range(len(reconstruction_img[110]))),
            y=reconstruction_img[110],
            mode="lines",
            line=dict(width=2),
        )
    )
    fig_slice_iradon.update_layout(
        width=800, height=600, title_text="Slice at Row 110 - iRadon Reconstruction"
    )
    figures.append((fig_slice_iradon, "slice_of_iradon_reconstruction.png"))

    # Save all images asynchronously with timing
    logger.info(f"Saving {len(figures)} images asynchronously...")
    save_start_time = time.time()

    async def timed_save(fig, filename):
        start_time = time.time()
        await save_image_async(fig, filename)
        save_time = time.time() - start_time
        return filename, save_time

    tasks = []
    for fig, filename in figures:
        filepath = f"{output_dir}/{filename}"
        tasks.append(timed_save(fig, filepath))

    results = await asyncio.gather(*tasks)

    # Print timing information
    logger.info("=== Image Saving Timing Results ===")
    for filename, save_time in results:
        logger.info(f"⏱️  {filename}: {save_time:.3f} seconds")

    total_save_time = time.time() - save_start_time
    logger.info(f"All images saved asynchronously in {total_save_time:.2f}s!")


async def main_async(colormap: str = "gray"):
    """Main function with async image saving."""
    start_time = time.time()
    logger.info("🚀 Starting ASYNC CT reconstruction pipeline")
    logger.info(
        f"Performance settings - Image size: {IMAGE_TARGET_SIZE}, Angle step: {ANGLE_STEP}°"
    )

    logger.info("Loading and preprocessing image...")
    image = load_image_fast()
    image_rot = rotate(image, 45)

    logger.info("Performing Radon transform...")
    thetas, rs, p, dtheta = perform_radon_transform_fast(image)

    logger.info("Performing filtered back projection...")
    fBP = filtered_back_projection_fast(p, rs, thetas, dtheta)

    logger.info("Performing Fourier reconstruction...")
    f = fourier_transform_fast(p, rs, thetas, dtheta)

    logger.info("Creating and saving all images asynchronously...")
    await plot_all_images_async(image, image_rot, thetas, rs, p, fBP, f, colormap)

    end_time = time.time()
    total_time = end_time - start_time
    logger.success(
        f"🚀 ASYNC CT reconstruction completed! Total time: {total_time:.2f} seconds"
    )
    print(f"🎉 ASYNC reconstruction complete! Total time: {total_time:.2f} seconds")
    print(f"📁 Outputs saved to: {output_dir}")


def main():
    """Entry point for async execution."""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
