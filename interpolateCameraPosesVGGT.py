import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt

from premiere.functionsRBF import buildInterpolator
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import RBFInterpolator  # Add this import

def extractFromArray(array, componenent):
    data = []
    for i in range(len(array)):
        data.append(array[i][componenent])
    data = np.array(data)
    return data

if len(sys.argv) != 6:
    print("Usage: python interpolateCameraPoses.py <vggt_pkl> <output_pkl> <kernel> <smooth> <epsilon>")
    sys.exit(1)
    
display = False

vggtPklName = sys.argv[1]
outputPklName = sys.argv[2]
kernel = sys.argv[3]
smooth = float(sys.argv[4])
epsilon = float(sys.argv[5])

print("Reading PKL: ", vggtPklName)
with open(vggtPklName, 'rb') as file:
    vggtPKL = pickle.load(file)

frameNumbers = vggtPKL['frames']
print("Frame numbers: ", len(frameNumbers))
# print(frameNumbers)
totalFrames = vggtPKL['total_frames']
print("Total frames: ", totalFrames)

relativePositions = np.array(vggtPKL['relative_positions'])
# print("Relative positions: ", relativePositions)
relativeQuaternions = np.array(vggtPKL['relative_quaternions'])
fov_x = vggtPKL['fov_x']
fov_x_degrees = vggtPKL['fov_x_degrees']

# Check if fov_x data exists and has correct length
if fov_x is None or len(fov_x) == 0:
    print("WARNING: No FOV data present, setting to default value of 60 degrees")
    fov_x = np.full(len(frameNumbers), 60.0)  # Default 60 degrees as fallback
elif len(fov_x) != len(frameNumbers):
    print(f"WARNING: FOV data length ({len(fov_x)}) doesn't match frame count ({len(frameNumbers)})")
    print("Using available FOV data and padding with last value if needed")
    # Extend fov_x with its last value if it's shorter than frameNumbers
    if len(fov_x) < len(frameNumbers):
        last_value = fov_x[-1]
        fov_x = np.append(fov_x, [last_value] * (len(frameNumbers) - len(fov_x)))
    # Truncate fov_x if it's longer than frameNumbers
    else:
        fov_x = fov_x[:len(frameNumbers)]

interpolatedRelativePositions = np.zeros((totalFrames, 3))
interpolatedRelativeQuaternions = np.zeros((totalFrames, 4))

# Create a range of all frame numbers
frames = np.arange(0, totalFrames, 1)

# Extract just the frame indices from frameNumbers for interpolation
frame_indices = np.array(frameNumbers)

# Interpolate each position component
for c in range(3):
    # print(f"Interpolating position component {c}...")
    # Create the interpolator using just the frame indices
    interpolator = buildInterpolator(frame_indices, relativePositions[:, c], 
                                    rbfkernel=kernel, rbfsmooth=smooth, rbfepsilon=epsilon)
    
    # Apply the interpolator to each frame - convert each frame to a 2D array with shape (1,2)
    for i, frame in enumerate(frames):
        result = interpolator(np.array([[frame, 0.0]]))
        interpolatedRelativePositions[i, c] = result[0]  # Extract the first element explicitly

# Replace component-wise quaternion interpolation with whole-quaternion interpolation
print("Interpolating quaternions...")
# Process quaternions to ensure continuity by checking signs
quats = np.array(relativeQuaternions)
for i in range(1, len(quats)):
    if np.dot(quats[i-1], quats[i]) < 0:
        quats[i] = -quats[i]  # Flip sign if dot product is negative

# Create input points for RBFInterpolator (2D points for compatibility)
X = np.column_stack((frame_indices, np.zeros_like(frame_indices)))
# Build interpolator for all 4 quaternion components at once
rbfi = RBFInterpolator(X, quats, kernel=kernel, epsilon=epsilon, smoothing=smooth)

# Apply interpolator to all frames
for i, frame in enumerate(frames):
    x_eval = np.array([[frame, 0.0]])
    interpolatedRelativeQuaternions[i] = rbfi(x_eval)[0]
    # Normalize the quaternion
    norm = np.linalg.norm(interpolatedRelativeQuaternions[i])
    if norm > 0:
        interpolatedRelativeQuaternions[i] /= norm

# Convert interpolated quaternions back to Euler angles
interpolatedRelativeRotations = np.zeros((totalFrames, 3))

print("Converting quaternions back to Euler angles...")
for i in range(totalFrames):
    # Get quaternion for this frame
    quat = interpolatedRelativeQuaternions[i]
    
    # Create rotation object from quaternion (format: x,y,z,w)
    rot = R.from_quat(quat)
    
    # Convert to Euler angles using the same convention as before ('xyz')
    euler = rot.as_euler('xyz')
    
    # Store in result array
    interpolatedRelativeRotations[i] = euler

# Interpolate FOV
print(f"Interpolating FOV data...")

# Check again if fov_x length matches frame_indices length before interpolation
if len(fov_x) != len(frame_indices):
    print(f"WARNING: FOV length mismatch before interpolation: fov_x({len(fov_x)}) vs frame_indices({len(frame_indices)})")
    # Fix the mismatch by either extending with the last value or truncating
    if len(fov_x) < len(frame_indices):
        last_value = fov_x[-1]
        fov_x = np.append(fov_x, [last_value] * (len(frame_indices) - len(fov_x)))
    else:
        fov_x = fov_x[:len(frame_indices)]
    print(f"Adjusted FOV length to {len(fov_x)}")

interpolator = buildInterpolator(frame_indices, fov_x, 
                                rbfkernel=kernel, rbfsmooth=smooth, rbfepsilon=epsilon)

print (fov_x_degrees)

interpolatedFov_x = np.zeros(totalFrames)
for i, frame in enumerate(frames):
    interpolatedFov_x[i] = interpolator(np.array([[frame, 0.0]]))[0]  # Extract scalar value

vggtPKL['interpolated_relative_positions'] = interpolatedRelativePositions
vggtPKL['interpolated_relative_rotations'] = interpolatedRelativeRotations
vggtPKL['interpolated_fov_x'] = interpolatedFov_x

print("Saving PKL: ", outputPklName)
with open(outputPklName, 'wb') as file:
    pickle.dump(vggtPKL, file, protocol=pickle.HIGHEST_PROTOCOL)