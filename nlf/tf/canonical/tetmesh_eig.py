import numpy as np
import tetgen
from scipy.sparse.linalg import LinearOperator, eigsh
from nlf.paths import PROJDIR
import sksparse.cholmod
import scipy.sparse as sparse
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-tetmesh', action='store_true')
    parser.add_argument('--save-laplacian-eig', action='store_true')
    args = parser.parse_args()

    if args.save_tetmesh:
        print('Saving tetrahedral mesh...')
        save_tetmesh()

    if args.save_laplacian_eig:
        save_laplacian_eig()


def save_tetmesh():
    verts = np.load(f'{PROJDIR}/canonical_vertices_smpl.npy')
    faces = np.load(f'{PROJDIR}/smpl_faces.npy')

    # Tetrahedralize the canonical mesh
    tgen = tetgen.TetGen(verts, faces)
    tgen.make_manifold(verbose=True)
    nodes, elem = tgen.tetrahedralize(switches='VDqpa3e-8')
    np.save(f'{PROJDIR}/canonical_nodes3.npy', nodes)
    np.save(f'{PROJDIR}/canonical_elems3.npy', elem)

def save_laplacian_eig():
    nodes = np.load(f'{PROJDIR}/canonical_nodes3.npy')
    elem = np.load(f'{PROJDIR}/canonical_elems3.npy')
    # Compute the stiffness and mass matrices of the tetrahedral mesh
    print("Computing stiffness and mass matrices...")
    stiffness, mass = fem_tetra(nodes, elem)
    # Compute the eigenvalues and eigenvectors of the Laplacian
    tet_eigva, tet_eigve = eigs(stiffness, mass, 1025)
    np.save(f'{PROJDIR}/canonical_eigval3.npy', tet_eigva)
    np.save(f'{PROJDIR}/canonical_eigvec3.npy', tet_eigve)

def eigs(stiffness, mass, k: int = 10):
    sigma = -0.01
    print("Computing Cholesky decomposition using scikit-sparse cholmod...")
    chol = sksparse.cholmod.cholesky(stiffness - sigma * mass, use_long=True)
    op_inv = LinearOperator(matvec=chol, shape=stiffness.shape, dtype=stiffness.dtype)

    print("Computing eigenvalues and eigenvectors...\n"
          "This takes 1 hour 20 minutes on an AMD Ryzen 9 5900X CPU.")
    eigenvalues, eigenvectors = eigsh(stiffness, k, mass, sigma=sigma, OPinv=op_inv, tol=1e-1)
    return eigenvalues, eigenvectors


def fem_tetra(nodes, elems):
    # This code is modified from the Lapy library.
    # MIT License Copyright (c) 2020 Deep Medical Imaging Lab (PI Reuter)
    t1 = elems[:, 0]
    t2 = elems[:, 1]
    t3 = elems[:, 2]
    t4 = elems[:, 3]
    v1 = nodes[t1, :]
    v2 = nodes[t2, :]
    v3 = nodes[t3, :]
    v4 = nodes[t4, :]
    e1 = v2 - v1
    e2 = v3 - v2
    e3 = v1 - v3
    e4 = v4 - v1
    e5 = v4 - v2
    e6 = v4 - v3
    # Compute cross product and 6 * vol for each triangle:
    cr = np.cross(e1, e3)
    vol = np.abs(np.sum(e4 * cr, axis=1))
    # zero vol will cause division by zero below, so set to small value:
    vol_mean = 0.0001 * np.mean(vol)
    vol[vol == 0] = vol_mean
    # compute dot products of edge vectors
    e11 = np.sum(e1 * e1, axis=1)
    e22 = np.sum(e2 * e2, axis=1)
    e33 = np.sum(e3 * e3, axis=1)
    e44 = np.sum(e4 * e4, axis=1)
    e55 = np.sum(e5 * e5, axis=1)
    e66 = np.sum(e6 * e6, axis=1)
    e12 = np.sum(e1 * e2, axis=1)
    e13 = np.sum(e1 * e3, axis=1)
    e14 = np.sum(e1 * e4, axis=1)
    e15 = np.sum(e1 * e5, axis=1)
    e23 = np.sum(e2 * e3, axis=1)
    e25 = np.sum(e2 * e5, axis=1)
    e26 = np.sum(e2 * e6, axis=1)
    e34 = np.sum(e3 * e4, axis=1)
    e36 = np.sum(e3 * e6, axis=1)
    # compute entries for A (negations occur when one edge direction is flipped)
    # these can be computed multiple ways
    # basically for ij, take opposing edge (call it Ek) and two edges from the
    # starting point of Ek to point i (=El) and to point j (=Em), then these are of
    # the scheme:   (El * Ek)  (Em * Ek) - (El * Em) (Ek * Ek)
    # where * is vector dot product
    a12 = (-e36 * e26 + e23 * e66) / vol
    a13 = (-e15 * e25 + e12 * e55) / vol
    a14 = (e23 * e26 - e36 * e22) / vol
    a23 = (-e14 * e34 + e13 * e44) / vol
    a24 = (e13 * e34 - e14 * e33) / vol
    a34 = (-e14 * e13 + e11 * e34) / vol
    # compute diagonals (from row sum = 0)
    a11 = -a12 - a13 - a14
    a22 = -a12 - a23 - a24
    a33 = -a13 - a23 - a34
    a44 = -a14 - a24 - a34
    # stack columns to assemble data
    local_a = np.column_stack(
        (a12, a12, a23, a23, a13, a13, a14, a14, a24, a24, a34, a34, a11, a22, a33, a44)
    ).reshape(-1)
    i = np.column_stack(
        (t1, t2, t2, t3, t3, t1, t1, t4, t2, t4, t3, t4, t1, t2, t3, t4)
    ).reshape(-1)
    j = np.column_stack(
        (t2, t1, t3, t2, t1, t3, t4, t1, t4, t2, t4, t3, t1, t2, t3, t4)
    ).reshape(-1)
    local_a = local_a / 6.0
    # Construct sparse matrix:
    a = sparse.csc_matrix((local_a, (i, j)))
    bii = vol / 60.0
    bij = vol / 120.0
    local_b = np.column_stack(
        (bij, bij, bij, bij, bij, bij, bij, bij, bij, bij, bij, bij, bii, bii, bii, bii)
    ).reshape(-1)
    b = sparse.csc_matrix((local_b, (i, j)))
    stiffness = a
    mass = b
    return stiffness, mass


if __name__ == '__main__':
    main()
