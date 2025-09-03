"""
SMPLX Human Update Function.

This module provides a function to update a human's 3D joints and body parameters
using an SMPLX model. The resulting 3D joints are transformed by applying rotation,
scaling, and translation, then stored back into the `human` dictionary.

Functions:
    updateHumanFromSMPLX(human, modelSMPLX, useExpression, scaleX=1.0, scaleY=1.0, scaleZ=1.0, offsetZ=0.0):
        Recomputes SMPLX joints (and optionally expressions) from the provided parameters
        and updates the `human` dictionary with the new 3D keypoints.
"""

import torch
import roma
import math
import numpy as np


def updateHumanFromSMPLX(human, modelSMPLX, useExpression, scaleX=1.0, scaleY=1.0, scaleZ=1.0, offsetZ=0.0):
    """
    Update the 3D keypoints of a human dictionary using an SMPLX model.

    This function:
      1. Loads shape (betas) and pose (rotvec) from `human`.
      2. Configures the SMPLX model input parameters (e.g., body pose, hand pose, optional expressions).
      3. Runs SMPLX forward pass to obtain 3D joints.
      4. Applies rotation using the root orientation and aligns the pelvis to the origin.
      5. Scales and offsets the translation, then applies it to all joints.
      6. Stores the resulting joint positions in the `human` dictionary.

    :param human: A dictionary containing human parameters and placeholders to store outputs. Keys must include:
                  - 'shape' (np.ndarray): SMPLX body shape parameters, typically size (10,).
                  - 'rotvec' (np.ndarray): SMPLX rotational vectors, typically size (N, 3).
                  - 'transl' (np.ndarray): Translation vector of size (3,).
                  - 'expression' (np.ndarray, optional): Facial expression parameters if `useExpression` is True.
    :type human: dict
    :param modelSMPLX: A loaded SMPLX model (e.g., via `smplx.create(...)`).
    :type modelSMPLX: SMPLX
    :param useExpression: Whether to use expressions (e.g., for facial keypoints). 
                          If True, the function reads 'expression' from `human` and applies face/jaw parameters.
    :type useExpression: bool
    :param scaleX: Scaling factor along the X axis, defaults to 1.0.
    :type scaleX: float, optional
    :param scaleY: Scaling factor along the Y axis, defaults to 1.0.
    :type scaleY: float, optional
    :param scaleZ: Scaling factor along the Z axis, defaults to 1.0.
    :type scaleZ: float, optional
    :param offsetZ: Additional offset to apply after scaling in the Z dimension, defaults to 0.0.
    :type offsetZ: float, optional
    :return: Updated `human` dictionary with new key 'j3d_smplx' (3D keypoints) and 'trans_pelvis' (pelvis position).
    :rtype: dict
    """
    # Load shape (betas) and pose from CPU memory to GPU
    betas = torch.from_numpy(np.expand_dims(human['shape'], axis=0)).cuda()
    pose = torch.from_numpy(np.expand_dims(human['rotvec'], axis=0)).cuda()
    bs = pose.shape[0]  # batch size, typically 1

    # Build dictionary of SMPLX forward-pass arguments
    kwargs_pose = {
        'betas': betas,
        'return_verts': False,
        'pose2rot': True
    }
    
    # Set global orientation to zero
    t = torch.zeros(1, 3).cuda()
    kwargs_pose['global_orient'] = t.repeat(bs, 1)
    
    # Body pose (21 joints, ignoring the global orientation)
    kwargs_pose['body_pose'] = pose[:, 1:22].flatten(1)

    # Depending on useExpression, set different slices for hand poses
    # If we have expressions, set the appropriate SMPLX parameters
    if useExpression:
        kwargs_pose['left_hand_pose'] = pose[:, 22:37].flatten(1)
        kwargs_pose['right_hand_pose'] = pose[:, 37:52].flatten(1)
        expression = torch.from_numpy(np.expand_dims(human['expression'], axis=0)).cuda()
        kwargs_pose['leye_pose'] = t.repeat(bs, 1)
        kwargs_pose['reye_pose'] = t.repeat(bs, 1)
        # Jaw and expression (if applicable)
        kwargs_pose['jaw_pose'] = pose[:, 52:53].flatten(1)
        kwargs_pose['expression'] = expression.flatten(1)
    else:
        kwargs_pose['jaw_pose'] = pose[:,22:23].flatten(1)
        kwargs_pose['leye_pose'] = pose[:,23:24].flatten(1)
        kwargs_pose['reye_pose'] = pose[:,24:25].flatten(1)
        kwargs_pose['left_hand_pose'] = pose[:, 25:40].flatten(1)
        kwargs_pose['right_hand_pose'] = pose[:, 40:55].flatten(1)

    # Forward pass through the SMPLX model
    output = modelSMPLX(**kwargs_pose)

    # Extract 3D joints from the model output (size ~45)
    j3d = output.joints

    # Convert root orientation (pose[:, 0]) from rotvec to rotmat
    R = roma.rotvec_to_rotmat(pose[:, 0])

    # Pelvis is the first joint (index 0 in SMPLX)
    pelvis = j3d[:, [0]]
    # Subtract pelvis from all joints to center them, then rotate
    j3d = (R.unsqueeze(1) @ (j3d - pelvis).unsqueeze(-1)).squeeze(-1)

    # Identify the person's center (joint index 15 in SMPLX is the head)
    person_center = j3d[:, [15]]

    # Adjust the translation from `human` with scaling and offset
    transCPU = human['transl']
    transCPU[0] = transCPU[0] * scaleX
    transCPU[1] = transCPU[1] * scaleY
    transCPU[2] = (transCPU[2] * scaleZ) + offsetZ
    trans = torch.from_numpy(transCPU).cuda()

    # Subtract person center, then add to all joints
    trans = trans - person_center
    j3d = j3d + trans

    # Move final joints back to CPU
    keypoints = j3d.detach().cpu().numpy().squeeze()

    # Store new SMPLX keypoints in the human dictionary
    human['j3d_smplx'] = keypoints
    # Store the pelvis position (first joint) for reference
    human['trans_pelvis'] = keypoints[0]

    return human

def updateHumanFromSMPLXForHMR2 (human, modelSMPLX, pose_rotvecs, shape_betas, trans, scaleX=1.0, scaleY=1.0, scaleZ=1.0, offsetZ=0.0):
#    betas = torch.from_numpy(np.expand_dims(shape_betas,axis=0)).cuda()
    expression = torch.zeros(1,10).cuda()
    pose = pose_rotvecs
#    pose = torch.from_numpy(np.expand_dims(pose_rotvecs,axis=0)).cuda()
    bs = 1
    kwargs_pose = {
        'return_verts': False,
    }
    t = torch.zeros(1, 3).cuda()
    kwargs_pose['global_orient'] = t.repeat(bs,1)
    kwargs_pose['body_pose'] = pose[:,1*3:22*3]

    # Face and hand pose    
    kwargs_pose['jaw_pose'] = pose[:,22*3:23*3]
    kwargs_pose['leye_pose'] = pose[:,23*3:24*3]
    kwargs_pose['reye_pose'] = pose[:,24*3:25*3]
    kwargs_pose['left_hand_pose'] = pose[:,25*3:40*3]
    kwargs_pose['right_hand_pose'] = pose[:,40*3:55*3]
    kwargs_pose['expression'] = expression
    kwargs_pose['betas'] = shape_betas

    # default - to be generalized
    kwargs_pose['leye_pose'] = t.repeat(bs,1)
    kwargs_pose['reye_pose'] = t.repeat(bs,1)      
            
    output = modelSMPLX(**kwargs_pose)

    j3d = output.joints # 45 joints
    R = roma.rotvec_to_rotmat(pose[:,0:3])

    pelvis = j3d[:,[15]]
    j3d = (R.unsqueeze(1) @ (j3d - pelvis).unsqueeze(-1)).squeeze(-1)

    person_center = pelvis
    # person_center = j3d[:, [0]]

    transCPU = trans.cpu().numpy().squeeze()
    transCPU[0] = transCPU[0] * scaleX
    transCPU[1] = transCPU[1] * scaleY
    transCPU[2] = (transCPU[2] * scaleZ) + offsetZ
    trans = torch.from_numpy( transCPU ).cuda()

    # trans = trans - person_center
    j3d = j3d + trans
    keypoints = j3d.detach().cpu().numpy().squeeze()    
    human['j3d_smplx'] = keypoints[0:127]
    human['transl_pelvis'] = [keypoints[0]]
    human['expression'] = np.zeros(10, dtype=np.float32)
    human['shape'] = shape_betas.detach().cpu().numpy().squeeze()  
    human['rotvec'] = pose_rotvecs.detach().cpu().numpy().reshape(55, 3)
    return human

def updateHumanFromSMPLXForAIST (human, modelSMPLX, pose_rotvecs, shape_betas, trans, scaleX=1.0, scaleY=1.0, scaleZ=1.0, offsetZ=0.0):
#    betas = torch.from_numpy(np.expand_dims(shape_betas,axis=0)).cuda()
    expression = torch.zeros(1,10).cuda()
    pose = pose_rotvecs
#    pose = torch.from_numpy(np.expand_dims(pose_rotvecs,axis=0)).cuda()
    bs = 1
    kwargs_pose = {
        'return_verts': False,
    }
    t = torch.zeros(1, 3).cuda()
    kwargs_pose['global_orient'] = t.repeat(bs,1)
    kwargs_pose['body_pose'] = pose[:,1*3:22*3]

    # Face and hand pose    
    kwargs_pose['jaw_pose'] = pose[:,22*3:23*3]
    kwargs_pose['leye_pose'] = pose[:,23*3:24*3]
    kwargs_pose['reye_pose'] = pose[:,24*3:25*3]
    kwargs_pose['left_hand_pose'] = pose[:,25*3:40*3]
    kwargs_pose['right_hand_pose'] = pose[:,40*3:55*3]
    kwargs_pose['expression'] = expression
    kwargs_pose['betas'] = shape_betas

    # default - to be generalized
    kwargs_pose['leye_pose'] = t.repeat(bs,1)
    kwargs_pose['reye_pose'] = t.repeat(bs,1)      
            
    output = modelSMPLX(**kwargs_pose)

    j3d = output.joints # 45 joints
    global_orient = pose[:,0:3]
    R = roma.rotvec_to_rotmat(global_orient)

    pelvis = j3d[:,[0]]
    
    j3d = (R.unsqueeze(1) @ (j3d - pelvis).unsqueeze(-1)).squeeze(-1)

    person_center = pelvis

    transCPU = trans.cpu().numpy().squeeze()
    transCPU[0] = transCPU[0] * scaleX
    transCPU[1] = transCPU[1] * scaleY
    transCPU[2] = (transCPU[2] * scaleZ) + offsetZ
    trans = torch.from_numpy( transCPU ).cuda()

    trans = trans - person_center
    j3d = j3d + trans
    keypoints = j3d.detach().cpu().numpy().squeeze()
    keypoints = -keypoints
    print (keypoints[0:127])
    human['j3d_smplx'] = keypoints[0:127]
    human['transl_pelvis'] = [keypoints[0]]
    human['transl'] = [keypoints[0]]
    human['expression'] = np.zeros(10, dtype=np.float32)
    human['shape'] = shape_betas.detach().cpu().numpy().squeeze()  
    human['rotvec'] = pose_rotvecs.detach().cpu().numpy().reshape(55, 3)
    return human

def updateHumanFromSMPLXForAISTold (human, modelSMPLX, pose_rotvecs, shape_betas, trans, scaleX=1.0, scaleY=1.0, scaleZ=1.0, offsetZ=0.0):
#    betas = torch.from_numpy(np.expand_dims(shape_betas,axis=0)).cuda()
    expression = torch.zeros(1,10).cuda()
    pose = pose_rotvecs
#    pose = torch.from_numpy(np.expand_dims(pose_rotvecs,axis=0)).cuda()
    bs = 1
    kwargs_pose = {
        'return_verts': False,
    }
    t = torch.zeros(1, 3).cuda()
    kwargs_pose['global_orient'] = t.repeat(bs,1)
    kwargs_pose['body_pose'] = pose[:,1*3:22*3]

    # Face and hand pose    
    kwargs_pose['jaw_pose'] = pose[:,22*3:23*3]
    kwargs_pose['leye_pose'] = pose[:,23*3:24*3]
    kwargs_pose['reye_pose'] = pose[:,24*3:25*3]
    kwargs_pose['left_hand_pose'] = pose[:,25*3:40*3]
    kwargs_pose['right_hand_pose'] = pose[:,40*3:55*3]
    kwargs_pose['expression'] = expression
    kwargs_pose['betas'] = shape_betas

    # default - to be generalized
    kwargs_pose['leye_pose'] = t.repeat(bs,1)
    kwargs_pose['reye_pose'] = t.repeat(bs,1)      
            
    output = modelSMPLX(**kwargs_pose)

    j3d = output.joints # 45 joints
    global_orient = pose[:,0:3]
    R = roma.rotvec_to_rotmat(global_orient)

    pelvis = j3d[:,[0]]
    head = j3d[:,[15]]
    
    j3d = (R.unsqueeze(1) @ (j3d - pelvis).unsqueeze(-1)).squeeze(-1)
    # j3d = (R.unsqueeze(1) @ (j3d - head).unsqueeze(-1)).squeeze(-1)

    person_center = pelvis
    # person_center = j3d[:,[0]]

    transCPU = trans.cpu().numpy().squeeze()
    transCPU[0] = transCPU[0] * scaleX
    transCPU[1] = transCPU[1] * scaleY
    transCPU[2] = (transCPU[2] * scaleZ) + offsetZ
    trans = torch.from_numpy( transCPU ).cuda()

    trans = trans - person_center
    j3d = j3d + trans
    keypoints = j3d.detach().cpu().numpy().squeeze()    
    human['j3d_smplx'] = keypoints[0:127]
    human['transl_pelvis'] = [keypoints[0]]
    human['transl'] = [keypoints[0]]
    human['expression'] = np.zeros(10, dtype=np.float32)
    human['shape'] = shape_betas.detach().cpu().numpy().squeeze()  
    human['rotvec'] = pose_rotvecs.detach().cpu().numpy().reshape(55, 3)
    return human
