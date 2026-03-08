#!/usr/bin/env python3
"""
Convert an image to a stroke-based SVG via Canny edge detection.
The output SVG uses fill:none + stroke paths, making it compatible with Vivus animation.
"""

import cv2
import numpy as np
import sys
from pathlib import Path
from scipy.ndimage import gaussian_filter1d


def smooth_contour_to_bezier(pts: np.ndarray, smooth_sigma: float = 2.0) -> str:
    """
    Smooth a contour's x/y coordinates with a Gaussian filter, then fit cubic
    Bézier curves through the points using auto-computed control points derived
    from the Catmull-Rom tangent formula.  Returns an SVG path `d` string.
    """
    if len(pts) < 4:
        # Fall back to a simple polyline for very short contours
        d = f"M {pts[0][0]:.1f},{pts[0][1]:.1f}"
        for x, y in pts[1:]:
            d += f" L {x:.1f},{y:.1f}"
        return d

    # Smooth coordinates independently
    xs = gaussian_filter1d(pts[:, 0].astype(float), sigma=smooth_sigma)
    ys = gaussian_filter1d(pts[:, 1].astype(float), sigma=smooth_sigma)

    # Build cubic Bézier path using Catmull-Rom control points:
    #   cp1_i = p[i]   + (p[i+1] - p[i-1]) / 6
    #   cp2_i = p[i+1] - (p[i+2] - p[i])   / 6
    d = f"M {xs[0]:.1f},{ys[0]:.1f}"
    n = len(xs)
    for i in range(n - 1):
        # Clamp neighbour indices to valid range
        im1 = max(i - 1, 0)
        ip1 = min(i + 1, n - 1)
        ip2 = min(i + 2, n - 1)

        cp1x = xs[i]   + (xs[ip1] - xs[im1]) / 6
        cp1y = ys[i]   + (ys[ip1] - ys[im1]) / 6
        cp2x = xs[ip1] - (xs[ip2] - xs[i])   / 6
        cp2y = ys[ip1] - (ys[ip2] - ys[i])   / 6

        d += f" C {cp1x:.1f},{cp1y:.1f} {cp2x:.1f},{cp2y:.1f} {xs[ip1]:.1f},{ys[ip1]:.1f}"

    return d


def image_to_svg(
    input_path: str,
    output_path: str,
    blur_radius: int = 5,        # Gaussian blur kernel size (odd number); higher = smoother
    canny_low: int = 50,         # Canny lower threshold; lower = more edges
    canny_high: int = 100,       # Canny upper threshold; higher = fewer edges
    min_contour_len: int = 50,   # Discard contours shorter than this (noise reduction)
    smooth_sigma: float = 1.0,   # Gaussian sigma for coordinate smoothing; higher = smoother curves
    stroke_color: str = "#1a1a1a",
    stroke_width: float = 1,
):
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: could not read {input_path}")
        sys.exit(1)

    h, w = img.shape[:2]

    # Convert to grayscale and blur to reduce noise
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (blur_radius | 1, blur_radius | 1), 0)

    # Canny edge detection
    edges = cv2.Canny(blurred, canny_low, canny_high)

    # Find contours on edge map
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_KCOS)

    # Build SVG path data from contours
    paths = []
    for contour in contours:
        if cv2.arcLength(contour, False) < min_contour_len:
            continue

        pts = contour.reshape(-1, 2)
        if len(pts) < 2:
            continue

        paths.append(smooth_contour_to_bezier(pts, smooth_sigma=smooth_sigma))

    print(f"Image size: {w}x{h}")
    print(f"Total contours found: {len(contours)}")
    print(f"Contours kept (after min-length filter): {len(paths)}")

    # Write SVG
    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">',
        f'  <style>path {{ fill: none; stroke: {stroke_color}; stroke-width: {stroke_width}; stroke-linecap: round; stroke-linejoin: round; }}</style>',
    ]
    for d in paths:
        svg_lines.append(f'  <path d="{d}"/>')
    svg_lines.append("</svg>")

    Path(output_path).write_text("\n".join(svg_lines))
    print(f"SVG written to: {output_path}  ({len(paths)} paths)")


if __name__ == "__main__":
    input_path = sys.argv[1] if len(sys.argv) > 1 else "tea-shop-inspo.png"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "tea-shop-edges.svg"
    image_to_svg(input_path, output_path)
