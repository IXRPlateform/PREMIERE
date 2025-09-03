import argparse

import cameralib
import poseviz
import simplepyutils as spu
import torch
import torchvision.io


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str)
    parser.add_argument('--video-path', type=str)



    with torch.inference_mode(), torch.device('cuda'):
        with poseviz.PoseViz(joint_names, joint_edges, paused=True) as viz:
            image_filepath = get_image(spu.argparse.FLAGS.image_path)
            image = torchvision.io.read_image(image_filepath).cuda()
            camera = cameralib.Camera.from_fov(fov_degrees=55, imshape=image.shape[1:])

            for num_aug in range(1, 50):
                pred = multiperson_model_pt.detect_poses(
                    image, detector_threshold=0.01, suppress_implausible_poses=False,
                    max_detections=1, intrinsic_matrix=camera.intrinsic_matrix,
                    skeleton=skeleton, num_aug=num_aug)

                viz.update(
                    frame=image.cpu().numpy().transpose(1, 2, 0),
                    boxes=pred['boxes'].cpu().numpy(),
                    poses=pred['poses3d'].cpu().numpy(), camera=camera)


if __name__ == '__main__':
    main()
