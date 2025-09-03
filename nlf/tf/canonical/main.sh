#!/usr/bin/env bash
set -euo pipefail

# In the following, each of the "save" parameters will save its output, so the computation
# can be resumed by skipping the corresponding those "save" parameters if it's already been done.
# This is to help with the long computation times of some of the steps.

# General initial setup
python -m nlf.tf.canonical.canonical_space --save-canonical-verts --save-internal-points --save-face-probs

# Now for each body model we can do the fitting to the canonical space, ie fit the canonical joints
# and then create a regressor for the internal points from verts and joints using natinterp
python -m nlf.tf.canonical.fit_canonical_space --save-canonical-joints --save-internal-points --body-model=smpl --gender=neutral
python -m nlf.tf.canonical.fit_canonical_space --save-canonical-joints --save-internal-points --body-model=smplx --gender=neutral
python -m nlf.tf.canonical.fit_canonical_space --save-canonical-joints --save-internal-points --body-model=smplh --gender=female
python -m nlf.tf.canonical.fit_canonical_space --save-canonical-joints --save-internal-points --body-model=smplh16 --gender=neutral

# Compute Laplacian eigenbasis. This can take 1.5h
python -m nlf.tf.canonical.tetmesh_eig --save-tetmesh --save-laplacian-eig

# Now distill the Laplacian eigenbasis into a learnable Fourier feature network
# This needs a GPU for speed. On an RTX 3090 it takes about 20-30 minutes.
python -m nlf.tf.canonical.distill

# Create regressor from SMPL vertices to skeletal joints
python -m nlf.tf.canonical.smpl2skel_pseudo

# Create the initial canoncial positions for the skeletal keypoints along with JointInfo objects etc.
python -m nlf.tf.canonical.canonical_skel