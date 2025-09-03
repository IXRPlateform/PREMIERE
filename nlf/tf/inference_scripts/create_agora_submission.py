import argparse
import os.path as osp
import pickle
import zipfile

import cameralib

import simplepyutils as spu
from humcentr_cli.util.serialization import Reader
from simplepyutils import FLAGS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-pred-path', type=str, required=True)
    parser.add_argument('--out-pred-path', type=str, required=True)
    parser.add_argument('--fov', type=float, default=65)
    spu.argparse.initialize(parser)

    camera = cameralib.Camera.from_fov(FLAGS.fov, (2160, 3840))
    with Reader(FLAGS.in_pred_path) as xz_in:
        with zipfile.ZipFile(
                FLAGS.out_pred_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=4) as zipf_out:
            for data in spu.progressbar(xz_in):
                image_name = osp.splitext(osp.basename(data['filename']))[0]
                for i_person, (vertices, joints) in enumerate(
                        zip(data['vertices'], data['joints'])):
                    person_filename = f'predictions/{image_name}_personId_{i_person}.pkl'
                    j2d = camera.camera_to_image(joints)
                    result = dict(
                        allSmplJoints3d=joints / 1000, verts=vertices / 1000, joints=j2d[:24])
                    zipf_out.writestr(person_filename, pickle.dumps(result))


if __name__ == '__main__':
    main()
