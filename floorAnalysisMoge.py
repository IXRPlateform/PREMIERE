#!/usr/bin/env python
"""
This script processes a video using the MoGe model to:
  - Infer a depth map, intrinsics, and a 3D point cloud from an input video frame.
  - Estimate the floor angle and fit a plane to the floor points.
  - Visualize the 3D point cloud (with and without the fitted plane) using Plotly.
  - Record various metadata (depth range, field of view, camera intrinsics, etc.) into a pickle file.

Usage:
    python video_analysis_moge.py <input_video> <output_pkl> <fov>
       - <input_video>: Path to the input video.
       - <output_pkl>: Path for the output pickle file.
       - <fov>: A non-zero horizontal FOV (in degrees) to force a fixed FOV; if 0, the MoGe estimation is used.
"""

import os
import sys
import cv2
import numpy as np
import torch
import pickle
from tqdm import tqdm
import plotly.graph_objects as go

# Import functions from external modules
from premiere.functionsMoge import initMoGeModel, computeFloorAngle, computeFov, colorizeDepthImage
from premiere.functionsDepth import packDepthImage, computeMinMaxDepth
from sklearn.linear_model import RANSACRegressor

from colorama import init, Fore, Style
init(autoreset=True)

# Get models path from environment variable
models_path = os.environ["MODELS_PATH"]


def visualize_point_cloud_3d(point_cloud, max_display_points=5000):
    """
    Visualizes a 3D point cloud using Plotly.
    
    Args:
        point_cloud (np.ndarray): Point cloud array of shape (N, 3).
        max_display_points (int): Maximum number of points to display.
    """
    # Subsample points if needed
    if len(point_cloud) > max_display_points:
        step = len(point_cloud) // max_display_points
        display_points = point_cloud[::step]
    else:
        display_points = point_cloud

    # Print point cloud statistics
    print("Point Cloud Statistics:")
    print(f"  Total points: {len(point_cloud)}")
    print(f"  Displayed points: {len(display_points)}")
    print(f"  X range: [{point_cloud[:, 0].min():.2f}, {point_cloud[:, 0].max():.2f}]")
    print(f"  Y range: [{point_cloud[:, 1].min():.2f}, {point_cloud[:, 1].max():.2f}]")
    print(f"  Z range: [{point_cloud[:, 2].min():.2f}, {point_cloud[:, 2].max():.2f}]")

    # Normalize the colors based on the Z coordinate
    z_values = display_points[:, 2]
    normalized_z = (z_values - z_values.min()) / (z_values.max() - z_values.min())

    # Create Plotly figure with scatter3d
    fig = go.Figure(data=[
        go.Scatter3d(
            x=display_points[:, 0],
            y=display_points[:, 1],
            z=display_points[:, 2],
            mode='markers',
            marker=dict(
                size=2,
                color=normalized_z,
                colorscale='Viridis',
                opacity=0.7,
                colorbar=dict(title="Z Height")
            ),
            hovertemplate='<b>X</b>: %{x:.2f}<br><b>Y</b>: %{y:.2f}<br><b>Z</b>: %{z:.2f}'
        )
    ])

    # Layout and camera settings
    fig.update_layout(
        title=f'3D Point Cloud ({len(display_points)} points displayed)',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        width=900,
        height=800,
        margin=dict(r=20, l=10, b=10, t=40),
        scene_camera=dict(eye=dict(x=1.5, y=1.5, z=0.8))
    )
    fig.show()


def select_points_near_plane(points, la, lb, delta=0.05):
    """
    Selects points that are within a specified delta from the floor plane.
    
    The plane is defined as: Z = la * Y + lb
    
    Args:
        points (np.ndarray): 3D point cloud of shape (N, 3).
        la (float): Slope parameter of the floor line.
        lb (float): Intercept parameter of the floor line.
        delta (float): Maximum distance from the plane to include points.
    
    Returns:
        np.ndarray: Subset of points within delta distance of the floor plane.
    """
    distances = np.abs(points[:, 2] - (la * points[:, 1] + lb))
    mask = distances <= delta
    selected_points = points[mask]
    print(f"Selected {len(selected_points)} points out of {len(points)} within {delta} m of the floor plane")
    return selected_points


def fit_plane_to_points(points, max_trials=1000, residual_threshold=0.02):
    """
    Fits a plane to 3D points using RANSAC for robustness to outliers.
    
    For this implementation, the plane is modeled as: z = a*x + b*y + c.
    It returns the coefficients in the form: ax + by + cz + d = 0.
    
    Args:
        points (np.ndarray): 3D points of shape (N, 3).
        max_trials (int): Maximum number of RANSAC iterations.
        residual_threshold (float): Threshold for considering a point as an inlier.
    
    Returns:
        tuple: (a, b, c, d) representing the plane coefficients.
    """
    # Prepare input for RANSAC
    X = points[:, :2]  # Use x and y coordinates
    y = points[:, 2]   # z coordinate

    # Fit with RANSAC
    ransac = RANSACRegressor(
        max_trials=max_trials,
        residual_threshold=residual_threshold,
        random_state=0
    )
    ransac.fit(X, y)
    a, b = ransac.estimator_.coef_
    c = ransac.estimator_.intercept_
    
    # Convert from z = a*x + b*y + c to plane equation ax + by + (-1)z + c = 0
    return a, b, -1, c


def visualize_point_cloud_3d_with_plane(point_cloud, plane_coeffs, max_display_points=5000, plane_color='rgba(255, 0, 0, 0.5)'):
    """
    Visualizes a 3D point cloud along with a fitted plane using Plotly.
    
    Args:
        point_cloud (np.ndarray): 3D point cloud of shape (N, 3).
        plane_coeffs (tuple): Coefficients (a, b, c, d) of the plane (ax + by + cz + d = 0).
        max_display_points (int): Maximum number of points to display.
        plane_color (str): RGBA color string for the plane.
    """
    # Subsample points for display if necessary
    if len(point_cloud) > max_display_points:
        step = len(point_cloud) // max_display_points
        display_points = point_cloud[::step]
    else:
        display_points = point_cloud

    # Print point cloud statistics
    print("Point Cloud Statistics:")
    print(f"  Total points: {len(point_cloud)}")
    print(f"  Displayed points: {len(display_points)}")
    print(f"  X range: [{point_cloud[:, 0].min():.2f}, {point_cloud[:, 0].max():.2f}]")
    print(f"  Y range: [{point_cloud[:, 1].min():.2f}, {point_cloud[:, 1].max():.2f}]")
    print(f"  Z range: [{point_cloud[:, 2].min():.2f}, {point_cloud[:, 2].max():.2f}]")

    # Normalize colors based on Z coordinate
    z_values = display_points[:, 2]
    normalized_z = (z_values - z_values.min()) / (z_values.max() - z_values.min())

    fig = go.Figure()

    # Add point cloud trace
    fig.add_trace(
        go.Scatter3d(
            x=display_points[:, 0],
            y=display_points[:, 1],
            z=display_points[:, 2],
            mode='markers',
            marker=dict(
                size=2,
                color=normalized_z,
                colorscale='Viridis',
                opacity=0.7,
                colorbar=dict(title="Z Height")
            ),
            hovertemplate='<b>X</b>: %{x:.2f}<br><b>Y</b>: %{y:.2f}<br><b>Z</b>: %{z:.2f}',
            name='Points'
        )
    )

    # Create a grid for the fitted plane
    a, b, c, d = plane_coeffs
    x_min, x_max = point_cloud[:, 0].min(), point_cloud[:, 0].max()
    y_min, y_max = point_cloud[:, 1].min(), point_cloud[:, 1].max()
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 20),
        np.linspace(y_min, y_max, 20)
    )
    if c != 0:
        zz = (-d - a * xx - b * yy) / c
    else:
        zz = np.zeros_like(xx)  # Vertical plane (special case)

    # Add the plane as a surface trace
    fig.add_trace(
        go.Surface(
            x=xx,
            y=yy,
            z=zz,
            colorscale=[[0, plane_color], [1, plane_color]],
            showscale=False,
            opacity=0.5,
            name='Fitted Plane'
        )
    )

    # Layout and camera settings
    fig.update_layout(
        title=f'3D Point Cloud with Fitted Plane ({len(display_points)} points displayed)',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        width=900,
        height=800,
        margin=dict(r=20, l=10, b=10, t=40),
        scene_camera=dict(eye=dict(x=1.5, y=1.5, z=0.8))
    )
    fig.show()


def plane_coeffs_to_angles(plane_coeffs):
    """
    Converts plane coefficients (ax + by + cz + d = 0) to tilt angles.
    
    Args:
        plane_coeffs (tuple): (a, b, c, d) of the plane.
    
    Returns:
        tuple: (angle_x, angle_y) in degrees.
          - angle_x: tilt around the Y-axis (inclination along X).
          - angle_y: tilt around the X-axis (inclination along Y).
    """
    a, b, c, d = plane_coeffs
    norm = np.sqrt(a*a + b*b + c*c)
    a, b, c = a / norm, b / norm, c / norm

    # Calculate angles using arctan (angle between normal vector and Z-axis)
    angle_x = np.degrees(np.arctan(a / abs(c)))
    angle_y = np.degrees(np.arctan(b / abs(c)))

    # Adjust sign if necessary based on c
    if c < 0:
        angle_x = 180 - angle_x if angle_x > 0 else -180 - angle_x
        angle_y = 180 - angle_y if angle_y > 0 else -180 - angle_y

    return angle_x, angle_y


def is_floor_angle_valid(la, lb, angle, floor_points, plane_coeffs, angle_y, total_points, 
                         max_angle_diff=10.0, min_floor_ratio=5.0, max_floor_angle=45.0,
                         min_inlier_ratio=0.6, max_residual=0.05):
    """
    Comprehensively validates the floor estimation using multiple criteria.

    Args:
        la, lb (float): Parameters of the floor line from computeFloorAngle.
        angle (float): Floor angle computed by computeFloorAngle (in degrees).
        floor_points (np.ndarray): Points selected as belonging to the floor.
        plane_coeffs (tuple): Coefficients of the fitted plane (ax + by + cz + d = 0).
        angle_y (float): Tilt angle (Y) computed from the plane coefficients.
        total_points (int): Total number of points in the rotated point cloud.
        max_angle_diff (float): Maximum allowed difference between angle estimation methods.
        min_floor_ratio (float): Minimum percentage of points that should be floor.
        max_floor_angle (float): Maximum plausible floor angle (degrees).
        min_inlier_ratio (float): Minimum ratio of points within the residual threshold.
        max_residual (float): Maximum average residual for a good plane fit.

    Returns:
        tuple: (is_valid, reasons, metrics) where:
            - is_valid (bool): Indicates if the floor estimation is reliable.
            - reasons (list): A list of issues if the estimation is not reliable.
            - metrics (dict): Quantitative metrics about the floor estimation quality.
    """
    reasons = []
    metrics = {}
    
    # 1. Check angle consistency between methods
    # Compare signed angles to catch direction inconsistencies
    angle_diff = abs(angle_y - angle)
    # Handle angle wrapping (e.g., -175° vs 175°)
    if angle_diff > 180:
        angle_diff = 360 - angle_diff
    metrics['angle_diff'] = angle_diff
    
    if angle_diff > max_angle_diff:
        reasons.append(f"Inconsistency between angle estimation methods: {angle_diff:.2f}°")

    # 2. Check if enough points have been selected as floor
    floor_ratio = len(floor_points) / total_points * 100
    metrics['floor_ratio'] = floor_ratio
    print(f"Floor points ratio: {floor_ratio:.1f}%")
    
    if floor_ratio < min_floor_ratio:
        reasons.append(f"Too few points identified as floor: {floor_ratio:.1f}%")

    # 3. Check if the estimated floor angle is within plausible range
    abs_angle = abs(angle)
    metrics['abs_angle'] = abs_angle
    
    if abs_angle > max_floor_angle:
        reasons.append(f"Floor angle too steep: {abs_angle:.1f}°")

    # 4. Check the quality of the plane fit using point-to-plane distances
    a, b, c, d = plane_coeffs
    norm = np.sqrt(a*a + b*b + c*c)
    
    # Calculate distances from floor points to the fitted plane
    distances = np.abs(a*floor_points[:,0] + b*floor_points[:,1] + 
                       c*floor_points[:,2] + d) / norm
    
    avg_residual = np.mean(distances)
    inlier_ratio = np.sum(distances < max_residual) / len(distances)
    
    metrics['avg_residual'] = avg_residual
    metrics['inlier_ratio'] = inlier_ratio
    
    if avg_residual > max_residual:
        reasons.append(f"Poor plane fit: average residual {avg_residual:.3f}m")
        
    if inlier_ratio < min_inlier_ratio:
        reasons.append(f"Low inlier ratio: {inlier_ratio:.1%}")
    
    # 5. Check the distribution of points (should be well distributed across the floor)
    xy_points = floor_points[:, :2]
    x_range = np.ptp(xy_points[:, 0])
    y_range = np.ptp(xy_points[:, 1])
    metrics['floor_coverage'] = min(x_range, y_range)
    
    if min(x_range, y_range) < 0.5:  # At least 0.5m coverage in both directions
        reasons.append(f"Limited floor coverage: {min(x_range, y_range):.2f}m")
    
    return (len(reasons) == 0), reasons, metrics


def detect_floor_presence(pointCloud3D, la, lb, angle, floor_points, plane_coeffs, metrics=None):
    """
    Detects if the image contains a usable floor for analysis.
    Works with relative scale point clouds (not assuming metric units).
    
    Args:
        pointCloud3D (np.ndarray): Complete 3D point cloud (after rotation)
        la, lb (float): Parameters of the floor line from computeFloorAngle
        angle (float): Floor angle calculated by computeFloorAngle (in degrees)
        floor_points (np.ndarray): Points selected as belonging to the floor
        plane_coeffs (tuple): Coefficients of the fitted plane (ax + by + cz + d = 0)
        metrics (dict, optional): Pre-calculated metrics, if available
        
    Returns:
        tuple: (floor_present, reasons, metrics)
    """
    reasons = []
    if metrics is None:
        metrics = {}
    
    # 1. Check the absolute number of points identified as floor
    if len(floor_points) < 100:
        reasons.append(f"Too few floor points detected: {len(floor_points)} points (minimum 100)")
    
    # 2. Check the proportion of points identified as floor
    floor_ratio = len(floor_points) / len(pointCloud3D) * 100
    metrics['floor_ratio'] = floor_ratio
    if floor_ratio < 3.0:
        reasons.append(f"Floor points proportion too low: {floor_ratio:.1f}%")
    
    # 3. Check the geometric consistency of the detected floor
    a, b, c, d = plane_coeffs
    norm = np.sqrt(a*a + b*b + c*c)
    
    # 3.1 Check if the plane is almost vertical (not a floor)
    if abs(c/norm) < 0.7:
        reasons.append(f"Detected plane is too vertical to be a floor")
    
    # 3.2 Calculate the dispersion of points relative to the plane (residuals)
    if len(floor_points) > 0:
        distances = np.abs(a*floor_points[:,0] + b*floor_points[:,1] + 
                        c*floor_points[:,2] + d) / norm
        
        # Calculate characteristic scale of the scene to use for threshold scaling
        scene_scale = np.median(np.linalg.norm(pointCloud3D, axis=1))
        metrics['scene_scale'] = scene_scale
        
        avg_residual = np.mean(distances)
        metrics['avg_residual'] = avg_residual
        
        # Use relative residual (as % of scene scale) instead of absolute
        relative_residual = avg_residual / scene_scale if scene_scale > 0 else 0
        metrics['relative_residual'] = relative_residual
        
        # Check if points fit poorly to the plane (using relative threshold)
        if relative_residual > 0.02:  # 2% of scene scale
            reasons.append(f"Floor points too dispersed around the plane: {relative_residual:.3f} of scene scale")
            
        # Calculate standard deviation of distances to the plane
        std_residual = np.std(distances)
        metrics['std_residual'] = std_residual
        relative_std = std_residual / scene_scale if scene_scale > 0 else 0
        metrics['relative_std'] = relative_std
        
        if relative_std > 0.06:  # 6% of scene scale
            reasons.append(f"High variation in distances to the plane: {relative_std:.3f} of scene scale")
    
    # 4. Check spatial coverage of floor points
    if len(floor_points) > 0:
        xy_points = floor_points[:, :2]  
        x_range = np.ptp(xy_points[:, 0])
        y_range = np.ptp(xy_points[:, 1])
        
        # Scale coverage relative to overall point cloud dimensions
        pc_x_range = np.ptp(pointCloud3D[:, 0])
        pc_y_range = np.ptp(pointCloud3D[:, 1])
        
        rel_x_coverage = x_range / pc_x_range if pc_x_range > 0 else 0
        rel_y_coverage = y_range / pc_y_range if pc_y_range > 0 else 0
        metrics['rel_x_coverage'] = rel_x_coverage
        metrics['rel_y_coverage'] = rel_y_coverage
        
        # Check relative spatial distribution of floor points
        if min(rel_x_coverage, rel_y_coverage) < 0.3:  # At least 30% coverage in both directions
            reasons.append(f"Insufficient floor spatial coverage: {min(rel_x_coverage, rel_y_coverage):.2f} of scene dimensions")
        
        # Calculate floor area and point density relative to total area
        area_coverage = x_range * y_range
        total_area = pc_x_range * pc_y_range
        relative_area = area_coverage / total_area if total_area > 0 else 0
        metrics['relative_area'] = relative_area
        
        # Check point density as points per relative area unit
        floor_density = len(floor_points) / area_coverage if area_coverage > 0 else 0
        metrics['floor_density'] = floor_density
        
        # Compare density to the average point cloud density
        avg_density = len(pointCloud3D) / total_area if total_area > 0 else 0
        relative_density = floor_density / avg_density if avg_density > 0 else 0
        metrics['relative_density'] = relative_density
        
        if relative_density < 0.25:  # Floor density should be at least 25% of average
            reasons.append(f"Floor point density too low: {relative_density:.1f} of average density")
    
    # 5. Check floor angle (should be close to horizontal) - this is scale-invariant
    abs_angle = abs(angle)
    metrics['abs_angle'] = abs_angle
    
    if abs_angle > 30:  # Floor too tilted
        reasons.append(f"Floor angle too steep: {abs_angle:.1f}°")
    
    # 6. Check height distribution of floor points
    if len(floor_points) > 10:
        z_values = floor_points[:, 2]
        z_range = np.ptp(z_values)
        
        # Scale height variation relative to XY dimensions
        xy_extent = max(x_range, y_range)
        relative_height_range = z_range / xy_extent if xy_extent > 0 else 0
        metrics['relative_height_range'] = relative_height_range
        
        # If the relative height variation is too large, it's probably not a flat floor
        if relative_height_range > 0.3:  # Height variation > 30% of horizontal extent
            reasons.append(f"Floor height variation too large: {relative_height_range:.2f} of horizontal extent")
    
    # Determine if the floor is present and usable
    floor_present = len(reasons) == 0
    
    return floor_present, reasons, metrics


def main():
    # Validate command-line arguments
    if len(sys.argv) != 4:
        print("Usage: python video_analysis_moge.py <input_video> <output_pkl> <fov>")
        sys.exit(1)

    displayCharts = False
    video_in_name = sys.argv[1]
    output_pkl_name = sys.argv[2]
    fovh_degrees = float(sys.argv[3])
    use_fixed_fov = (fovh_degrees != 0)

    # Initialize the MoGe model on CUDA
    device_name = 'cuda'
    device, model = initMoGeModel(device_name)

    # Open the input video and retrieve basic metadata
    video = cv2.VideoCapture(video_in_name)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    frames_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Read the first frame for processing
    ret, frame = video.read()
    if not ret:
        print("[!] Error reading the video frame.")
        sys.exit(1)

    # Convert the frame to a normalized torch tensor in CHW order
    image_tensor = torch.tensor(frame / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1)

    # Run inference with the MoGe model using fixed FOV if provided
    if use_fixed_fov:
        output = model.infer(image_tensor, fov_x=fovh_degrees)
    else:
        output = model.infer(image_tensor)

    # Extract outputs from the model
    depth = output['depth'].cpu().numpy()           # Depth map
    intrinsics = output['intrinsics'].cpu().numpy()   # Camera intrinsics
    pointCloud3D = output['points'].cpu().numpy()     # 3D point cloud
    mask = output['mask'].cpu().numpy()               # Validity mask
    max_points = 40000

    # Compute floor angle using a subset of points
    la, lb, angle_radians, angle = computeFloorAngle(
        pointCloud3D, mask, max_points=max_points, quantile=0.025, displayChart=displayCharts
    )

    # Compute the field of view from the intrinsics
    fovh, fovv = computeFov(intrinsics)
    fovh_degrees = np.degrees(fovh)
    fovv_degrees = np.degrees(fovv)

    # Compute minimum and maximum depth values within the mask
    min_depth, max_depth = computeMinMaxDepth(depth, mask)

    print(f"FOV: {fovh_degrees:.2f} x {fovv_degrees:.2f} degrees")
    print(f"Min depth: {min_depth:.2f}, Max depth: {max_depth:.2f}")
    print(f"Rotation angle: {angle:.2f} degrees")

    # Prepare the point cloud:
    #   - Reshape to (N, 3) and filter out points based on the mask.
    pointCloud3D = pointCloud3D.reshape(-1, 3)
    mask1D = mask.reshape(-1)
    pointCloud3D = pointCloud3D[mask1D]

    # Limit the number of points if needed
    if len(pointCloud3D) > max_points:
        step = len(pointCloud3D) // max_points
        pointCloud3D = pointCloud3D[::step][:max_points]

    # Apply a 90° rotation around the X-axis (to match computeFloorAngle)
    R_90x = np.array([
        [1, 0, 0],
        [0, np.cos(np.radians(90)), -np.sin(np.radians(90))],
        [0, np.sin(np.radians(90)), np.cos(np.radians(90))]
    ])
    pointCloud3D_rotated = np.dot(pointCloud3D, R_90x)

    if displayCharts:
        # Visualize the rotated point cloud
        visualize_point_cloud_3d(pointCloud3D_rotated, max_display_points=max_points)

    # Select floor points based on the computed line (Z = la * Y + lb) with a tolerance delta
    floor_points = select_points_near_plane(pointCloud3D_rotated, la, lb, delta=0.1)

    # Fit a robust plane to the floor points using RANSAC
    plane_coeffs = fit_plane_to_points(floor_points, residual_threshold=0.02)
    print(f"Fitted plane coefficients (ax + by + cz + d = 0): {plane_coeffs}")

    # Convert plane coefficients to tilt angles
    angle_x, angle_y = plane_coeffs_to_angles(plane_coeffs)
    # Adjust angles to match the image rotation convention
    angle_x = (angle_x % 360)
    angle_y = (angle_y % 360)
    angle_x = 180 - angle_x
    angle_x_radians = np.radians(angle_x)
    angle_y = 180 - angle_y
    angle_y_radians = np.radians(angle_y)
    print(f"Plane tilt angles: X={angle_x:.2f}°, Y={angle_y:.2f}°")

    if displayCharts:
        # Visualize the floor points together with the fitted plane
        visualize_point_cloud_3d_with_plane(floor_points, plane_coeffs, max_display_points=max_points)

    # (Optional) Validate the floor angle estimation.
    is_valid, reasons, metrics = is_floor_angle_valid(la, lb, angle, floor_points, plane_coeffs, angle_y, len(pointCloud3D_rotated))
    if not is_valid:
        print("WARNING: Floor estimation may be unreliable:")
        for reason in reasons:
            print(f"  - {reason}")
    else:
        print("Floor estimation appears reliable.")

    # Detect floor presence
    floor_present, floor_issues, floor_metrics = detect_floor_presence(
        pointCloud3D_rotated, la, lb, angle, floor_points, plane_coeffs
    )

    if not floor_present:
        print(Fore.RED +"Warning: No usable floor detected in the image:")
        for issue in floor_issues:
            print(f"  - {issue}")
        angle_y_radians = 0
        angle_x_radians = 0
        angle_y = 0
        angle_x = 0
        print (Style.RESET_ALL)
    else:
        print("Floor correctly detected and usable.")
 

    # Record metadata for the frame
    mogeData = {
        'min_depth': float(min_depth),
        'max_depth': float(max_depth),
        'fov_x': float(fovh),
        'fov_y': float(fovv),
        'fov_x_degrees': float(fovh_degrees),
        'fov_y_degrees': float(fovv_degrees),
        'intrinsics': intrinsics.tolist(),
        'rotation_x_degrees': float(angle_y),
        'rotation_x_radians': float(angle_y_radians),
        'rotation_camera_degrees': [angle_y, angle_x, 0],
        'rotation_camera_radians': [angle_y_radians, angle_x_radians, 0],
        'la': float(la),
        'lb': float(lb),
        'pointCloud3D': pointCloud3D,  # Unrotated, filtered point cloud
        'floor_detected': floor_present,
        'floor_metrics': floor_metrics,
        'floor_issues': floor_issues,
    }

    # Save metadata to a pickle file
    print("Saving data to pickle file...")
    with open(output_pkl_name, 'wb') as file:
        pickle.dump(mogeData, file, protocol=pickle.HIGHEST_PROTOCOL)

    video.release()


if __name__ == "__main__":
    main()
