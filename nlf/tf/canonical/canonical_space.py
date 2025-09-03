import argparse

import natinterp3d
import numpy as np
import scipy.optimize
import scipy.stats.qmc
import simplepyutils as spu
import trimesh
from smplfitter.np import SMPLBodyModel as SMPL_np, SMPLFitter as SMPLFitter_np
from smplfitter.pt.converter import load_vertex_converter_csr

from nlf.paths import DATA_ROOT, PROJDIR

def save(path, obj):
    np.save(path, obj)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-canonical-verts', action='store_true')
    parser.add_argument('--save-face-probs', action='store_true')
    parser.add_argument('--save-internal-points', action='store_true')
    parser.add_argument('--save-canonical-joints', action='store_true')
    parser.add_argument('--save-internal-regressor', action='store_true')
    parser.add_argument('--body-model', type=str, default='smpl')
    parser.add_argument('--gender', type=str, default='neutral')
    args = parser.parse_args()

    if args.save_canonical_verts:
        save_canonical_verts()

    if args.save_internal_points:
        save_internal_points()

    if args.save_face_probs:
        save_face_probs()

    if args.save_canonical_joints or args.save_internal_regressor:
        maybe_x = 'x' if args.body_model.startswith('smplx') else ''
        canonical_verts = np.load(f'{PROJDIR}/canonical_vertices_smpl{maybe_x}.npy')
        g0 = args.gender[0]
        body_model = SMPL_np(args.body_model, args.gender)

        if args.save_canonical_joints:
            canonical_joints, canonical_joints_144 = get_canonical_joints(
                body_model, canonical_verts)
            save(f'{PROJDIR}/canonical_joints_{args.body_model}_{g0}.npy', canonical_joints)

            if canonical_joints_144 is not None:
                save(f'{PROJDIR}/canonical_joints_{args.body_model}_{g0}_144.npy',
                     canonical_joints_144)

        if args.save_internal_regressor:
            canonical_internal_points = np.load(f'{PROJDIR}/canonical_internal_points_smpl.npy')
            canonical_joints = np.load(f'{PROJDIR}/canonical_joints_{args.body_model}_{g0}.npy')
            internal_regressor = get_internal_regressor(
                body_model, canonical_verts, canonical_joints, canonical_internal_points)
            save_csr(f'{PROJDIR}/internal_regressor_{args.body_model}_{g0}.csr',
                     internal_regressor)


def save_canonical_verts():
    body_model = SMPL_np('smpl', 'neutral')
    stretched = body_model.single(pose_rotvecs=get_canonical_pose()[np.newaxis])
    verts = stretched['vertices']
    canonical_vertices_smpl = symmetrize_verts(verts)
    save(f'{PROJDIR}/canonical_vertices_smpl.npy', canonical_vertices_smpl)

    smpl2smplx_csr = load_vertex_converter_csr(
        f'{DATA_ROOT}/body_models/smpl2smplx_deftrafo_setup.pkl')
    canonical_vertices_smplx = smpl2smplx_csr @ canonical_vertices_smpl
    save(f'{PROJDIR}/canonical_vertices_smplx.npy', canonical_vertices_smplx)


def save_face_probs():
    canonical_vertices_smpl = np.load(f'{PROJDIR}/canonical_vertices_smpl.npy')
    faces_smpl = SMPL_np('smpl', 'neutral').faces
    np.save(f'{PROJDIR}/smpl_faces.npy', faces_smpl)

    face_probs_smpl = get_face_probs(canonical_vertices_smpl, faces_smpl)
    save(f'{PROJDIR}/smpl_face_probs.npy', face_probs_smpl)

    canonical_vertices_smplx = np.load(f'{PROJDIR}/canonical_vertices_smplx.npy')
    faces_smplx = SMPL_np('smplx', 'neutral').faces
    np.save(f'{PROJDIR}/smplx_faces.npy', faces_smplx)

    face_probs_smplx = get_face_probs(canonical_vertices_smplx, faces_smplx)
    save(f'{PROJDIR}/smplx_face_probs.npy', face_probs_smplx)


def get_face_probs(verts, faces):
    face_areas = trimesh.Trimesh(verts, faces).area_faces
    return face_areas / np.sum(face_areas)


def get_canonical_joints(body_model, canonical_verts):
    # since we changed the vertices when symmetrizing, we need to check
    # where the joints would be for such a mesh. We could probably directly use the joint regressor
    # here, but the cleaner way is to fit actual parameters to this changed mesh
    # and see where the joints are for this fit (the regressor should not be used on a posed mesh.)
    fitter = SMPLFitter_np(body_model, num_betas=300)
    res = fitter.fit(
        canonical_verts[np.newaxis], n_iter=10, beta_regularizer=0,
        requested_keys=['joints', 'orientations'])

    canonical_joints = res['joints'][0]

    if body_model.num_vertices == 10475:
        smplx_data = np.load(f'{DATA_ROOT}/body_models/smplx/SMPLX_NEUTRAL.npz')
        landmarks = get_landmarks_smplx(
            res['vertices'], res['orientations'], smplx_data['f'],
            smplx_data['lmk_faces_idx'], smplx_data['lmk_bary_coords'],
            smplx_data['dynamic_lmk_faces_idx'], smplx_data['dynamic_lmk_bary_coords'])[0]
        canonical_joints_144 = np.concatenate([canonical_joints, landmarks], axis=0)
    else:
        canonical_joints_144 = None

    return canonical_joints, canonical_joints_144


def get_internal_regressor(
        body_model, canonical_verts, canonical_joints, canonical_internal_points):
    # Let's now introduce some points along each bone.
    # Explanation: remember that the overall goal is to approximate how SMPL deforms
    # within the mesh. From SMPL, we only know how the surface vertices and the joints
    # transform (this is the forward function of SMPL). For arbitrary internal points we will
    # need to interpolate. The idea is to interpolate not just using the mesh vertices but also
    # the joints and also points along the bones. We can reasonably assume that the bones are rigid
    # so we know their deformation. By using these as given, we can ensure that our interpolated
    # approximation
    # coincides with SMPL's joint and bone transformations and only interpolates the rest.
    # We create 100 points along each bone.
    # The two ends are the joints themselves, so we use 98 points in between.
    alphas = np.linspace(0, 1, 100)[1:-1]
    points_per_bone = len(alphas)
    n_joints = body_model.num_joints
    weights_bonepoints = np.zeros(((n_joints - 1) * points_per_bone, n_joints), np.float32)
    for i_joint, i_parent in enumerate(body_model.kintree_parents[1:], start=1):
        i_bone = i_joint - 1
        start_index = i_bone * points_per_bone
        end_index = start_index + points_per_bone
        weights_bonepoints[start_index:end_index, i_joint] = alphas
        weights_bonepoints[start_index:end_index, i_parent] = 1 - alphas

    # We now have weights that go from joints to bonepoints
    # We can use these to compute the locations of the bonepoints in canonical pose.
    bonepoints = weights_bonepoints @ canonical_joints

    # The natural neighbor interpolations's reference points will be
    # the vertices, the joints and the bonepoints.
    reference_points = np.concatenate([canonical_verts, canonical_joints, bonepoints], axis=0)

    # We now compute the weights (Sibson coordinates) for the internal points that we sampled
    # using the Sobol sequence.
    weights = natinterp3d.get_weights(canonical_internal_points, reference_points)
    n_bonepoints = bonepoints.shape[0]
    weights_new = weights[:, :-n_bonepoints]

    # So now we got weights in terms of verts, joints and bonepoints
    # But of course the bonepoints are by definition linearly interpolated
    # between the joints. So we can convert this to just weights in terms of
    # vertices and joints. We had to do this circumroute because if we just
    # used vertices and joints during natural neighbor interpolation, the joints
    # may be far away to become natural neighbors, so by adding bonepoints, we encouraged the
    # interpolation
    # to express points actually in reference to the bones, i.e. ultimately to joints
    # even though joints may have been further away.
    # So we now change the weights placed on the joints, to also account for the bonepoints
    weights_new[:, -n_joints:] += weights[:, -n_bonepoints:] @ weights_bonepoints
    # save_csr(f'{PROJDIR}/internal_regressor_n.csr', weights_new)
    return weights_new


def save_csr(path, csr):
    save(path + '.indices.npy', csr.indices)
    save(path + '.indptr.npy', csr.indptr)
    save(path + '.data.npy', csr.data)
    save(path + '.shape.npy', csr.shape)


def symmetrize_verts(verts):
    verts_mirror = verts.copy()
    verts_mirror[:, 0] *= -1
    dist = np.linalg.norm(verts_mirror[np.newaxis] - verts[:, np.newaxis], axis=-1)
    vert_indices, mirror_indices = scipy.optimize.linear_sum_assignment(dist)
    return (verts + verts_mirror[mirror_indices]) / 2


def get_canonical_pose():
    return np.concatenate([np.array(
        [0, 0, 0,  # pelvis
         np.deg2rad(0), 0, np.deg2rad(40),  # lhip
         -np.deg2rad(0), 0, -np.deg2rad(40),  # rhip
         *((10 * 3) * [0]),
         0, 0, np.deg2rad(15),  # lcla
         0, 0, -np.deg2rad(15),  # rcla
         0, 0, 0,
         0, 0, np.deg2rad(15),  # lsho
         0, 0, -np.deg2rad(15),  # rsho
         ], np.float32), np.zeros([(24 - 18) * 3])])


def parallel_map_with_progbar(fn, items):
    with spu.ThrottledPool() as pool:
        result = [None] * len(items)
        for i, x in enumerate(spu.progressbar(items)):
            pool.apply_async(fn, (x,), callback=spu.itemsetter(result, i))
    return result


def contains(mesh_and_point):
    mesh, point = mesh_and_point
    return mesh.contains(point)


def get_sobol_points_in_mesh_parallel(mesh, count_two_power, batch_size=8192):
    # Sample points inside the mesh, using a Sobol sequence and checking if they are inside the mesh
    dimensions = 3
    sobol_sampler = scipy.stats.qmc.Sobol(d=dimensions, scramble=True, seed=42)
    samples = sobol_sampler.random_base2(m=count_two_power)
    samples = samples * (mesh.bounds[1] - mesh.bounds[0]) + mesh.bounds[0]
    n_samples = samples.shape[0]
    c = parallel_map_with_progbar(
        contains, [(mesh, samples[i:i + batch_size]) for i in range(0, n_samples, batch_size)])
    return samples[np.concatenate(c, axis=0)]


def save_internal_points():
    canonical_verts = np.load(f'{PROJDIR}/canonical_vertices_smpl.npy')
    faces = np.load(f'{PROJDIR}/smpl_faces.npy')
    canonical_mesh = trimesh.Trimesh(canonical_verts, faces)

    internal_points = get_sobol_points_in_mesh_parallel(
        canonical_mesh, 24, batch_size=4096)
    save(f'{PROJDIR}/canonical_internal_points_smpl.npy', internal_points)


def get_landmarks_smplx(
        vertices, orientations, faces,
        lmk_faces_idx, lmk_bary_coords,
        dynamic_lmk_faces_idx, dynamic_lmk_bary_coords):
    batch_size = vertices.shape[0]
    neck_rot = orientations[:, 12]

    neck_rot_y = np.arctan2(neck_rot[:, 2, 0], np.linalg.norm(neck_rot[:, :2, 0], axis=1))
    neck_rot_idx = np.round(np.clip(neck_rot_y * 180.0 / np.pi, -39, 39)).astype(int)  # [-39..39]
    neck_rot_idx = np.where(neck_rot_idx < 0, 39 - neck_rot_idx, neck_rot_idx)  # [0..78]

    dyn_lmk_faces_idx = dynamic_lmk_faces_idx[neck_rot_idx]
    dyn_lmk_bary_coords = dynamic_lmk_bary_coords[neck_rot_idx]

    lmk_faces_idx = np.concatenate([
        np.repeat(lmk_faces_idx[np.newaxis], batch_size, axis=0),
        dyn_lmk_faces_idx], 1)
    lmk_bary_coords = np.concatenate([
        np.repeat(lmk_bary_coords[np.newaxis], batch_size, axis=0),
        dyn_lmk_bary_coords], 1)

    lmk_faces = faces[lmk_faces_idx]  # b, l, 3
    batch_index = np.arange(batch_size)[:, np.newaxis, np.newaxis]
    lmk_vertices = vertices[batch_index, lmk_faces]  # b, l, 3, 3
    return np.einsum('blfi,blf->bli', lmk_vertices, lmk_bary_coords)

if __name__ == '__main__':
    main()