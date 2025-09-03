"""
Depth Image Packing/Unpacking Utilities.

This module provides a set of functions to pack/unpack a 16-bit depth image
into a 3-channel 8-bit image, as well as compute minimum/maximum depth.

Functions:
    deinterlaceImage(image): Split a 16-bit image into two 8-bit images.
    reinterlaceImage(high_bits, low_bits): Recombine two 8-bit images into a single 16-bit image.
    packDepthImage(depth, mask, min_depth, max_depth): Normalize and encode a depth map + mask into a 3-channel image.
    packDepthImageSimple(depth): Encode a 16-bit depth image into 2 channels of an 8-bit image.
    unpackDepthImage(colorImageDepth, min_depth, max_depth, brg2rgb=False): Decode a 3-channel image back into a floating depth map + mask.
    computeMinMaxDepth(depth, mask): Compute the min and max depth values within the mask.

Usage Example:
    packed_img = packDepthImage(depth_map, mask, min_depth, max_depth)
    depth_map_decoded, mask_decoded = unpackDepthImage(packed_img, min_depth, max_depth)

Note:
    - Channels in the packed 8-bit image are organized so that one 8-bit channel stores
      the high bits, another channel stores the low bits, and the optional third channel
      can store the mask (or remain unused).
    - The `brg2rgb` parameter in `unpackDepthImage` allows for channel reordering if
      the image is loaded in a different color format (e.g., BGR vs. RGB).
"""

import numpy as np

def deinterlaceImage(image):
    """
    De-interlace a 16-bit image into two 8-bit images (high and low bits).

    This function takes a 16-bit image (np.uint16) and splits it into two
    separate np.uint8 images:
      - high_bits: The upper 8 bits of each pixel.
      - low_bits: The lower 8 bits of each pixel.

    :param image: A 16-bit input image (H x W), type np.uint16.
    :type image: np.ndarray
    :return: A tuple of two 8-bit images (high_bits, low_bits).
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    # Ensure the input is 16-bit
    image = image.astype(np.uint16)
    # Extract high and low 8 bits
    high_bits = (image >> 8).astype(np.uint8)
    low_bits = (image & 0xFF).astype(np.uint8)
    return high_bits, low_bits


def reinterlaceImage(high_bits, low_bits):
    """
    Re-interlace two 8-bit images into a single 16-bit image.

    This function combines two np.uint8 images (high_bits, low_bits) 
    back into a single np.uint16 image.

    :param high_bits: An 8-bit image storing the high 8 bits.
    :type high_bits: np.ndarray
    :param low_bits: An 8-bit image storing the low 8 bits.
    :type low_bits: np.ndarray
    :return: A single 16-bit image reconstructing the original data.
    :rtype: np.ndarray
    """
    high_bits = high_bits.astype(np.uint16)
    low_bits = low_bits.astype(np.uint16)
    interleaved = (high_bits << 8) | low_bits
    return interleaved


def packDepthImage(depth, mask, min_depth, max_depth):
    """
    Normalize a depth map and encode it with a mask into a 3-channel 8-bit image.

    - Step 1: Depth values are first normalized to the [0, 1] range based on `min_depth` and `max_depth`.
    - Step 2: The normalized depth is scaled to 16 bits (0..65535).
    - Step 3: The 16 bits are split into high_bits (red channel) and low_bits (green channel).
    - Step 4: The `mask` is placed in the blue channel (could be used as a validity channel).

    :param depth: A 2D floating-point array of shape (H, W), representing the depth map.
    :type depth: np.ndarray
    :param mask: A boolean array of shape (H, W), representing valid regions in the depth map.
    :type mask: np.ndarray
    :param min_depth: The minimum depth value used for normalization.
    :type min_depth: float
    :param max_depth: The maximum depth value used for normalization.
    :type max_depth: float
    :return: A 3D np.uint8 image of shape (H, W, 3). 
             - channel 0 (R) = high bits of depth
             - channel 1 (G) = low bits of depth
             - channel 2 (B) = mask
    :rtype: np.ndarray
    """
    # Normalize depth values to [0, 1]
    depth = (depth - min_depth) / (max_depth - min_depth)
    
    if mask is not None:
        # Apply mask (set invalid depths to 0)
        depth[~mask] = 0

    # Scale depth to 16-bit
    depth = (depth * 65535).astype(np.uint16)
    
    # De-interlace into high and low bits
    odd_image, even_image = deinterlaceImage(depth)
    
    height, width = depth.shape
    colorImage = np.zeros((height, width, 3), dtype=np.uint8)

    # Assign high bits to the red channel (channel 0)
    colorImage[:, :, 0] = odd_image  
    # Assign low bits to the green channel (channel 1)
    colorImage[:, :, 1] = even_image
    if mask is None:
        # Put the mask in the blue channel (channel 2)
        colorImage[:, :, 2] = 1

    return colorImage


def packDepthImageSimple(depth):
    """
    Encode an already 16-bit depth map into two channels of an 8-bit image (no mask channel).

    This function is a simpler version of `packDepthImage` that only takes the 16-bit depth
    image and splits it into two 8-bit channels (red and green). The blue channel is left at zero.

    :param depth: A 16-bit depth map of shape (H, W).
    :type depth: np.ndarray
    :return: A 3D np.uint8 image of shape (H, W, 3) where:
             - channel 0 (R) = high bits
             - channel 1 (G) = low bits
             - channel 2 (B) = 0 (unused)
    :rtype: np.ndarray
    """
    # De-interlace into high and low bits
    odd_image, even_image = deinterlaceImage(depth)
    
    height, width = depth.shape
    colorImage = np.zeros((height, width, 3), dtype=np.uint8)

    # Assign high bits to the red channel
    colorImage[:, :, 0] = odd_image
    # Assign low bits to the green channel
    colorImage[:, :, 1] = even_image

    return colorImage


def unpackDepthImage(colorImageDepth, min_depth, max_depth, brg2rgb=False):
    """
    Decode a 3-channel 8-bit depth image back into a floating-point depth map and mask.

    This function reverses the operation done by `packDepthImage`. It takes a 3-channel image
    and:
    1. Extracts the 16-bit depth from the first two channels (R/G by default).
    2. Retrieves the mask from the third channel (B by default).
    3. Re-scales the depth back to [min_depth, max_depth].

    :param colorImageDepth: A 3-channel 8-bit image (H, W, 3) containing the packed depth.
    :type colorImageDepth: np.ndarray
    :param min_depth: The minimum depth value that was used for normalization.
    :type min_depth: float
    :param max_depth: The maximum depth value that was used for normalization.
    :type max_depth: float
    :param brg2rgb: If True, swap the first and the third channels. 
                    (Useful if the image is in BGR format instead of RGB.)
    :type brg2rgb: bool, optional
    :return: A tuple (depth_scaled, mask) where:
             - depth_scaled is a floating-point array (H, W) with depth values in [min_depth, max_depth]
             - mask is a boolean array (H, W) of valid pixels
    :rtype: (np.ndarray, np.ndarray)
    """
    # By default, assume c0=0 (R), c1=1 (G), c2=2 (B)
    c0, c1, c2 = 0, 1, 2
    if brg2rgb:
        # If the image is in BGR format, swap
        c0, c2 = 2, 0
        
    # Extract channels from the packed image
    odd_image = colorImageDepth[:, :, c0].astype(np.uint8)
    even_image = colorImageDepth[:, :, c1].astype(np.uint8)
    mask = colorImageDepth[:, :, c2].astype(np.uint8)
    mask = mask > 0  # Convert to boolean

    # Recombine into a 16-bit depth image
    depth_16bit = reinterlaceImage(odd_image, even_image)

    # Convert to floating [0..1]
    depth_normalized = depth_16bit.astype(np.float32) / 65535.0

    # Scale back to [min_depth..max_depth]
    depth_scaled = min_depth + (depth_normalized * (max_depth - min_depth))

    return depth_scaled, mask


def computeMinMaxDepth(depth, mask):
    """
    Compute the minimum and maximum depth values in a depth map, optionally masked.

    :param depth: A 2D depth array of shape (H, W).
    :type depth: np.ndarray
    :param mask: A boolean mask of the same shape (H, W). If None, the entire depth array is used.
    :type mask: np.ndarray or None
    :return: A tuple (min_depth, max_depth).
    :rtype: (float, float)
    """
    if mask is None:
        min_depth = np.min(depth)
        max_depth = np.max(depth)
    else:
        min_depth = np.min(depth[mask])
        max_depth = np.max(depth[mask])
    return min_depth, max_depth
