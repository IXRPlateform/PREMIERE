import numpy as np
import torch
import roma
import os
import math
import trimesh
from scipy.spatial.transform import Rotation as R

def compute_vertex_normals(vertices, faces):
    """
    Compute per-vertex normals given vertex positions and triangle faces.
    vertices: (N, 3) array
    faces:    (F, 3) array of integer indices (0-based)
    Returns (N, 3) array of vertex normals.
    """
    normals = np.zeros_like(vertices)
    for f in faces:
        v0, v1, v2 = vertices[f[0]], vertices[f[1]], vertices[f[2]]
        # Face normal (unnormalized)
        fn = np.cross(v1 - v0, v2 - v0)
        normals[f[0]] += fn
        normals[f[1]] += fn
        normals[f[2]] += fn
    # Normalize
    lens = np.linalg.norm(normals, axis=1)
    lens[lens == 0] = 1e-8
    normals /= lens[:, None]
    return normals


def save_obj_with_mtl(obj_filename, mtl_filename, all_verts, all_faces, group_materials, material_colors):
    """
    Save an .obj file + .mtl file that references multiple groups (humans).
    
    all_verts: list of arrays, each shape = (num_vertices_i, 3)
    all_faces: list of arrays, each shape = (num_faces_i, 3) (0-based)
    group_materials: list of strings, each is the material name for that group
    material_colors: dict of mat_name -> (r,g,b), each in [0,1]
    """
    # 1) Write the MTL file
    with open(mtl_filename, 'w') as f_mtl:
        for mat_name, rgb in material_colors.items():
            r, g, b = rgb
            f_mtl.write(f"newmtl {mat_name}\n")
            f_mtl.write(f"Kd {r:.3f} {g:.3f} {b:.3f}\n")  # Diffuse color
            f_mtl.write(f"Ka {r:.3f} {g:.3f} {b:.3f}\n")  # Ambient color
            f_mtl.write("Ks 0.0 0.0 0.0\n")
            f_mtl.write("d 1.0\n")   # opacity
            f_mtl.write("Ns 1.0\n\n")

    # 2) Write the OBJ file
    with open(obj_filename, 'w') as f_obj:
        # Reference the .mtl
        mtl_basename = os.path.basename(mtl_filename)
        f_obj.write(f"mtllib {mtl_basename}\n")

        vertex_offset = 0
        normal_offset = 0

        for i, (verts_i, faces_i) in enumerate(zip(all_verts, all_faces)):
            mat_name = group_materials[i]

            # Compute normals for this set
            normals_i = compute_vertex_normals(verts_i, faces_i)

            # Write group and material
            f_obj.write(f"o Human_{i}\n")
            f_obj.write(f"usemtl {mat_name}\n")

            # Write out the vertices and the normals
            for v in verts_i:
                f_obj.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            for n in normals_i:
                f_obj.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")

            # Write faces referencing the new v and vn. OBJ indexing is 1-based.
            for f in faces_i:
                f_obj.write(
                    "f {}//{} {}//{} {}//{}\n".format(
                        f[0] + 1 + vertex_offset, f[0] + 1 + normal_offset,
                        f[1] + 1 + vertex_offset, f[1] + 1 + normal_offset,
                        f[2] + 1 + vertex_offset, f[2] + 1 + normal_offset
                    )
                )
            vertex_offset += len(verts_i)
            normal_offset += len(normals_i)

def _prepare_model_input(human_data, angle, useExpression, device='cpu', estimateVertices=True):
    device_t = torch.device(device)
    t = torch.zeros(1, 3, device=device_t)
    betas = torch.from_numpy(human_data['shape'][None, :]).to(device_t)
    pose = torch.from_numpy(human_data['rotvec'][None, :]).to(device_t)

    kwargs_pose = {
        'betas': betas,
        'return_verts': estimateVertices,
        'pose2rot': True
    }
    bs = pose.shape[0]
    kwargs_pose['global_orient'] = t.repeat(bs,1)
    kwargs_pose['body_pose'] = pose[:,1:22].flatten(1)

    if useExpression:
        expression = torch.from_numpy(human_data['expression'][None, :]).to(device_t)
        kwargs_pose['expression'] = expression.flatten(1)
        kwargs_pose['left_hand_pose'] =  pose[:,22:37].flatten(1)
        kwargs_pose['right_hand_pose'] = pose[:,37:52].flatten(1)
        kwargs_pose['jaw_pose'] = pose[:,52:53].flatten(1)
        kwargs_pose['leye_pose'] = t.repeat(bs,1)
        kwargs_pose['reye_pose'] = t.repeat(bs,1)
    else:
        kwargs_pose['jaw_pose'] = pose[:,22:23].flatten(1)
        kwargs_pose['leye_pose'] = pose[:,23:24].flatten(1)
        kwargs_pose['reye_pose'] = pose[:,24:25].flatten(1)
        kwargs_pose['left_hand_pose'] =  pose[:,25:40].flatten(1)
        kwargs_pose['right_hand_pose'] = pose[:,40:55].flatten(1)

    Rmat = roma.rotvec_to_rotmat(pose[:,0])
    trans = torch.from_numpy(human_data['transl']).to(device_t)

    theta = math.pi - angle
    cos_ = math.cos(theta)
    sin_ = math.sin(theta)
    R_x = torch.tensor([
        [1,    0,    0],
        [0, cos_, -sin_],
        [0, sin_,  cos_]
    ], dtype=torch.float32, device=device_t)

    return device_t, kwargs_pose, Rmat, trans, R_x

def _compute_verts_for_person(human_data, model, angle=0, Zoffset=0, useExpression=False, device='cpu', estimateVertices=True):
    device_t, kwargs_pose, Rmat, trans, R_x = _prepare_model_input(human_data, angle, useExpression, device, estimateVertices)
    output = model(**kwargs_pose)
    verts = output.vertices
    j3d  = output.joints
    pelvis = j3d[:, [0]]
    verts = (Rmat.unsqueeze(1) @ (verts - pelvis).unsqueeze(-1)).squeeze(-1)
    person_center = j3d[:, [15]]
    trans = trans - person_center
    verts = verts + trans
    verts = (R_x.unsqueeze(0) @ verts.unsqueeze(-1)).squeeze(-1)
    verts[:, :, 1] += Zoffset
    return verts.detach().cpu().numpy().squeeze(0)

def estimateFromSMPLXJ3DWithFloor(human_data, model, angle=0, Zoffset=0, useExpression=False, device='cpu', estimateVertices=False):
    device_t, kwargs_pose, Rmat, trans, R_x = _prepare_model_input(human_data, angle, useExpression, device, estimateVertices)
    output = model(**kwargs_pose)
    j3d = output.joints
    pelvis = j3d[:, [0]]
    j3dRot = (Rmat.unsqueeze(1) @ (j3d - pelvis).unsqueeze(-1)).squeeze(-1)
    person_center = j3d[:, [15]]
    trans = trans - person_center
    j3dRot = j3dRot + trans
    j3dRot = (R_x.unsqueeze(0) @ j3dRot.unsqueeze(-1)).squeeze(-1)
    j3dRot[:, :, 1] += Zoffset
    return j3dRot.detach().cpu().numpy().squeeze(0)


def estimateJ3DWithFloor(human_data, angle=0, Zoffset=0):
    """
    Numpy version mirroring the PyTorch logic in estimateFromSMPLXJ3DWithFloor.
    """
    # 1) Global rotation from the first 3 elements of 'rotvec'
    global_rotvec = human_data['rotvec'][:3]  # shape (3,)
    Rmat = R.from_rotvec(global_rotvec).as_matrix()  # (3, 3)

    # 2) Get the SMPL-X joints
    j3d = human_data['j3d_smplx']  # shape (N, 3), e.g. (55, 3)
    
    # 3) Subtract pelvis (joint 0), then rotate
    pelvis = j3d[0]  # (3,)
    j3d_centered = j3d - pelvis
    # np.matmul(Rmat, j3d_centered.T) => (3, N).T => (N, 3)
    # or equivalently j3d_centered @ Rmat.T => (N, 3)
    j3dRot = (Rmat @ j3d_centered.T).T

    # 4) Subtract person_center (joint 15) from trans
    person_center = j3d[15]  # (3,)
    trans = human_data['transl'] - person_center  # shape (3,)

    # 5) Add the translation
    j3dRot = j3dRot + trans  # (N, 3)

    # 6) Build rotation R_x by (π - angle) around the X-axis
    theta = math.pi - angle
    cos_ = math.cos(theta)
    sin_ = math.sin(theta)
    R_x = np.array([
        [1.0,   0.0,    0.0],
        [0.0,  cos_,  -sin_],
        [0.0,  sin_,   cos_]
    ], dtype=np.float32)

    # 7) Rotate around the X-axis
    j3dRot = (R_x @ j3dRot.T).T  # (N, 3)

    # 8) Add Zoffset to the Y dimension (same as PyTorch code)
    j3dRot[:, 1] += Zoffset



    return j3dRot
"""
def estimateJ3DWithFloor(human_data, angle=0, Zoffset=0):
    pose = human_data['rotvec'][None, :]
    Rmat = R.from_rotvec(pose[:,0]).as_matrix()
    trans = human_data['transl']
    j3d = human_data['j3d_smplx']

    theta = math.pi - angle
    cos_ = math.cos(theta)
    sin_ = math.sin(theta)
    R_x = np.array([
        [1,    0,    0],
        [0, cos_, -sin_],
        [0, sin_,  cos_]
    ], dtype=np.float32)

    pelvis = j3d[:, [0]]
    j3dRot = np.matmul(Rmat[:, np.newaxis], (j3d - pelvis)[..., np.newaxis]).squeeze(-1)
    person_center = j3d[:, [15]]
    trans = trans - person_center
    j3dRot = j3dRot + trans
    j3dRot = np.matmul(R_x[np.newaxis], j3dRot[..., np.newaxis]).squeeze(-1)
    j3dRot[:, :, 1] += Zoffset
    return j3dRot.squeeze(0)
"""
def save_obj(humans_in_frame, filename, model, angle=0, Zoffset=0, useExpression=False, device='cpu'):
    """
    Save all humans in the frame as OBJ + MTL.
    """
    mtl_filename = filename + ".mtl"
    
    all_verts_list = []
    all_faces_list = []
    group_materials = []
    
    # Simple palette for up to 10 people
    palette = [
        (0.8, 0.0, 0.0),  # Red
        (0.0, 0.8, 0.0),  # Green
        (0.0, 0.0, 0.8),  # Blue
        (0.8, 0.8, 0.0),  # Yellow
        (0.8, 0.0, 0.8),  # Magenta
        (0.0, 0.8, 0.8),  # Cyan
        (0.8, 0.5, 0.0),  # Orange
        (0.5, 0.0, 0.8),  # Purple
        (0.5, 0.5, 0.5),  # Gray
        (0.0, 0.5, 0.5),  # Teal
    ]
    material_colors = {}

    # Faces are the same for all people from this SMPL-X model
    faces_np = (
        model.faces.detach().cpu().numpy() 
        if torch.is_tensor(model.faces) 
        else model.faces
    ).astype(np.int32)

    # Compute each person's vertices
    for idx, human_data in enumerate(humans_in_frame):
        verts_np = _compute_verts_for_person(
            human_data, model, angle=angle, Zoffset=Zoffset, useExpression=useExpression, device=device
        )

        all_verts_list.append(verts_np)
        all_faces_list.append(faces_np)

        mat_name = f"color_{idx % len(palette)}"
        group_materials.append(mat_name)
        material_colors[mat_name] = palette[idx % len(palette)]

    # Write OBJ + MTL
    save_obj_with_mtl(
        obj_filename=filename,
        mtl_filename=mtl_filename,
        all_verts=all_verts_list,
        all_faces=all_faces_list,
        group_materials=group_materials,
        material_colors=material_colors
    )

    return f"OBJ saved as {filename}, MTL as {mtl_filename}."


def save_glb(humans_in_frame, filename, model, angle=0, Zoffset=0, useExpression=False, device='cpu'):
    """
    Save all humans in the frame as a GLB using trimesh.
    """
    scene = trimesh.Scene()

    # Simple palette for up to 10 people
    palette = [
        (0.8, 0.0, 0.0),  # Red
        (0.0, 0.8, 0.0),  # Green
        (0.0, 0.0, 0.8),  # Blue
        (0.8, 0.8, 0.0),  # Yellow
        (0.8, 0.0, 0.8),  # Magenta
        (0.0, 0.8, 0.8),  # Cyan
        (0.8, 0.5, 0.0),  # Orange
        (0.5, 0.0, 0.8),  # Purple
        (0.5, 0.5, 0.5),  # Gray
        (0.0, 0.5, 0.5),  # Teal
    ]

    # Faces are the same for all people from this SMPL-X model
    faces_np = (
        model.faces.detach().cpu().numpy() 
        if torch.is_tensor(model.faces) 
        else model.faces
    ).astype(np.int32)

    # Compute each person's vertices, build a mesh
    for idx, human_data in enumerate(humans_in_frame):
        verts_np = _compute_verts_for_person(
            human_data, model, angle=angle, Zoffset=Zoffset, useExpression=useExpression, device=device
        )

        mesh_color = np.array(palette[idx % len(palette)])
        # Expand to Nx4 for RGBA with alpha=1.0
        color_rgba = np.hstack([mesh_color, [1.0]])
        colors_per_vertex = np.tile(color_rgba, (verts_np.shape[0], 1))

        tri_mesh = trimesh.Trimesh(vertices=verts_np,
                                   faces=faces_np,
                                   vertex_colors=colors_per_vertex)

        scene.add_geometry(tri_mesh, node_name=f"Human_{idx}")

    # Export the scene as GLB
    glb_bytes = scene.export(file_type='glb')
    with open(filename, 'wb') as f:
        f.write(glb_bytes)

    return f"GLB saved as {filename}."

