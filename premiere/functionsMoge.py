"""
Utilities for MoGe model initialization, depth colorization, and geometry calculations.

This module includes:
- `initMoGeModel`: Loads a MoGe model from a pretrained file and moves it to a specified device.
- `colorizeDepthImage`: Converts a raw depth map to a colorized BGR image.
- `findBelowLineWithQuantile`: Fits a line to points using quantile regression (lower quantile).
- `computeFloorAngle`: Computes the approximate floor angle based on a 3D point cloud.
- `intrinsicsToFovNumpy`: Calculates the FOV in x and y directions from an intrinsic matrix.
- `computeFov`: Wrapper around `intrinsicsToFovNumpy` to return (fovx, fovy).
"""

import os
import cv2
import torch
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

from moge.model import MoGeModel
from moge.utils.geometry_numpy import intrinsics_to_fov_numpy, fov_to_focal_numpy
from moge.utils.vis import colorize_depth

models_path = os.environ["MODELS_PATH"]

def computeInitialRotation(rotationCameraDegrees):
    """
    Calculates the initial rotation matrix R_np from the angles in the MoGe dictionary.
    """
    rad_x = np.radians(-rotationCameraDegrees[0])
    rad_y = np.radians(-rotationCameraDegrees[1])
    rad_z = np.radians(-rotationCameraDegrees[2])
    
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(rad_x), -np.sin(rad_x)],
        [0, np.sin(rad_x), np.cos(rad_x)]
    ])
    
    R_y = np.array([
        [np.cos(rad_y), 0, np.sin(rad_y)],
        [0, 1, 0],
        [-np.sin(rad_y), 0, np.cos(rad_y)]
    ])
    
    R_z = np.array([
        [np.cos(rad_z), -np.sin(rad_z), 0],
        [np.sin(rad_z), np.cos(rad_z), 0],
        [0, 0, 1]
    ])
    
    R_np = R_z @ R_x @ R_y
    return R_np


def initMoGeModel(device_name='cuda'):
    """
    Initialize a MoGe model on the specified device.

    :param device_name: Name of the device to use ('cpu' or 'cuda'), defaults to 'cuda'.
    :type device_name: str, optional
    :return: A tuple containing the device and the loaded MoGe model in evaluation mode.
    :rtype: (torch.device, MoGeModel)
    """
    device = torch.device(device_name)
    dtype = torch.bfloat16
    model_name = os.path.join(models_path, 'moge', 'moge-large.pt')

    print("Loading model:", model_name)
    model = MoGeModel.from_pretrained(model_name).to(device).eval()
    return device, model


def colorizeDepthImage(depth):
    """
    Convert a raw depth map to a colorized BGR image.

    :param depth: A 2D numpy array representing depth values.
    :type depth: np.ndarray
    :return: A colorized depth image in BGR format.
    :rtype: np.ndarray
    """
    # The colorize_depth function returns an RGB image, so we convert to BGR.
    colorized_depth_rgb = colorize_depth(depth)
    colorized_depth_bgr = cv2.cvtColor(colorized_depth_rgb, cv2.COLOR_RGB2BGR)
    return colorized_depth_bgr


def findBelowLineWithQuantile(pointCloud2D, displayChart=False, quantile=0.01):
    """
    Fit a line to 2D points using quantile regression at a specified lower quantile.

    This method finds a line that approximates the lower boundary of data points 
    using StatsModels' QuantReg.

    :param pointCloud2D: A 2D numpy array of shape (N, 2) representing points in XY space.
    :type pointCloud2D: np.ndarray
    :param displayChart: If True, display a chart of the points and the fitted line, defaults to False.
    :type displayChart: bool, optional
    :param quantile: The quantile for the regression, defaults to 0.01 (1%).
    :type quantile: float, optional
    :return: A tuple (a, b, angle_radians, angle_degrees), where:
             - a is the slope
             - b is the y-intercept
             - angle_radians is the line angle in radians
             - angle_degrees is the line angle in degrees
    :rtype: (float, float, float, float)
    """
    # Separate x and y coordinates
    x = pointCloud2D[:, 0]
    y = pointCloud2D[:, 1]
    
    # Add a column of ones for the intercept in the regression model
    X = sm.add_constant(x)
    
    # Fit a quantile regression model
    model = sm.QuantReg(y, X)
    res = model.fit(q=quantile)  # Lower quantile regression
    
    # Retrieve slope (a) and intercept (b)
    a = res.params[1]
    b = res.params[0]
    
    # Compute angle in radians and degrees
    angle_radians = np.arctan(a)
    angle_degrees = np.degrees(angle_radians)
    
    if displayChart:
        def line(x_data):
            return a * x_data + b

        # Plot the points and the fitted line
        plt.scatter(x, y, color='blue', label='Points')
        plt.plot(x, line(x), color='red', label=f'y = {a:.2f}x + {b:.2f} (quantile {quantile})')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.show()

    return a, b, angle_radians, angle_degrees


def computeFloorAngle(pointCloud3D, mask, max_points=20000, quantile=0.01, displayChart=False):
    """
    Compute the approximate floor angle based on a 3D point cloud.

    The function extracts valid points from a 3D point cloud using a mask, 
    rotates them by 90 degrees around the x-axis, projects them onto a 2D plane, 
    and then fits a line to the lower quantile of these points to estimate the floor angle.

    :param pointCloud3D: A 3D numpy array of shape (H, W, 3) or (N, 3) representing 3D points.
    :type pointCloud3D: np.ndarray
    :param mask: A binary mask of shape (H, W) or (N,) indicating valid points.
    :type mask: np.ndarray
    :param max_points: Maximum number of points to sample for computation, defaults to 20000.
    :type max_points: int, optional
    :param quantile: Lower quantile used for line fitting, defaults to 0.01.
    :type quantile: float, optional
    :param displayChart: If True, display a chart of the fitted line, defaults to False.
    :type displayChart: bool, optional
    :return: A tuple (la, lb, angle_radians, angle_degrees), where:
             - la is the slope of the fitted line
             - lb is the y-intercept of the fitted line
             - angle_radians is the floor angle in radians
             - angle_degrees is the floor angle in degrees
    :rtype: (float, float, float, float)
    """
    # Reshape inputs
    pointCloud3D = pointCloud3D.reshape(-1, 3)
    mask1D = mask.reshape(-1)

    # Filter points based on the mask
    pointCloud3D = pointCloud3D[mask1D]

    # Limit the number of points
    if len(pointCloud3D) > max_points:
        step = len(pointCloud3D) // max_points
        pointCloud3D = pointCloud3D[::step][:max_points]

    # Rotate points by 90 degrees around the x-axis
    R = np.array([
        [1, 0, 0],
        [0, np.cos(np.radians(90)), -np.sin(np.radians(90))],
        [0, np.sin(np.radians(90)),  np.cos(np.radians(90))]
    ])
    pointCloud3D = np.dot(pointCloud3D, R)

    # Project 3D points onto 2D plane (take y and z)
    pointCloud2D = pointCloud3D[:, [1, 2]]

    # Compute the floor angle via quantile regression line fitting
    la, lb, angle_radians, angle_degrees = findBelowLineWithQuantile(pointCloud2D, displayChart, quantile)
    
    return la, lb, angle_radians, angle_degrees


def intrinsicsToFovNumpy(intrinsics):
    """
    Calculate the field of view (FOV) in both x and y directions from the intrinsic matrix.

    :param intrinsics: The 3x3 intrinsic camera matrix.
    :type intrinsics: np.ndarray
    :return: A tuple (fov_x, fov_y) in radians.
    :rtype: (float, float)
    """
    fov_x = 2 * np.arctan2(intrinsics[0, 2], intrinsics[0, 0])
    fov_y = 2 * np.arctan2(intrinsics[1, 2], intrinsics[1, 1])
    return fov_x, fov_y


def computeFov(intrinsics):
    """
    Wrapper function to compute the FOV in radians along the x and y axes.

    :param intrinsics: The 3x3 intrinsic camera matrix.
    :type intrinsics: np.ndarray
    :return: A tuple (fovx, fovy) in radians.
    :rtype: (float, float)
    """
    fovx, fovy = intrinsicsToFovNumpy(intrinsics)
    return fovx, fovy
