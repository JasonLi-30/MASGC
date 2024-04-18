# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
import os.path as osp
import warnings
from argparse import ArgumentParser

import cv2
import mmcv
import numpy as np

from mmpose.apis import (collect_multi_frames, extract_pose_sequence,
                         get_track_id, inference_pose_lifter_model,
                         inference_top_down_pose_model, init_pose_model,
                         process_mmdet_results, vis_3d_pose_result)
from mmpose.core import Smoother
from mmpose.datasets import DatasetInfo
from mmpose.models import PoseLifter, TopDown

try:
    from mmdet.apis import inference_detector, init_detector

    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False


def convert_keypoint_definition(keypoints, pose_det_dataset,
                                pose_lift_dataset):
    """Convert pose det dataset keypoints definition to pose lifter dataset
    keypoints definition.

    Args:
        keypoints (ndarray[K, 2 or 3]): 2D keypoints to be transformed.
        pose_det_dataset, (str): Name of the dataset for 2D pose detector.
        pose_lift_dataset (str): Name of the dataset for pose lifter model.
    """
    coco_style_datasets = [
        'TopDownCocoDataset', 'TopDownPoseTrack18Dataset',
        'TopDownPoseTrack18VideoDataset'
    ]
    if pose_det_dataset == 'TopDownH36MDataset' and \
            pose_lift_dataset == 'Body3DH36MDataset':
        return keypoints
    elif pose_det_dataset in coco_style_datasets and \
            pose_lift_dataset == 'Body3DH36MDataset':
        keypoints_new = np.zeros((17, keypoints.shape[1]))
        # pelvis is in the middle of l_hip and r_hip
        keypoints_new[0] = (keypoints[11] + keypoints[12]) / 2
        # thorax is in the middle of l_shoulder and r_shoulder
        keypoints_new[8] = (keypoints[5] + keypoints[6]) / 2
        # in COCO, head is in the middle of l_eye and r_eye
        # in PoseTrack18, head is in the middle of head_bottom and head_top
        keypoints_new[10] = (keypoints[1] + keypoints[2]) / 2
        # spine is in the middle of thorax and pelvis
        keypoints_new[7] = (keypoints_new[0] + keypoints_new[8]) / 2
        # rearrange other keypoints
        keypoints_new[[1, 2, 3, 4, 5, 6, 9, 11, 12, 13, 14, 15, 16]] = \
            keypoints[[12, 14, 16, 11, 13, 15, 0, 5, 7, 9, 6, 8, 10]]
        return keypoints_new
    else:
        raise NotImplementedError


def main():
    parser = ArgumentParser()
    parser.add_argument('det_config', help='Config file for detection')
    parser.add_argument('det_checkpoint', help='Checkpoint file for detection')
    parser.add_argument(
        'pose_detector_config',
        type=str,
        default=None,
        help='Config file for the 1st stage 2D pose detector')
    parser.add_argument(
        'pose_detector_checkpoint',
        type=str,
        default='/media/star/zjurobot/402/WYH_402/mmpose/checkpoints/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth',
        help='Checkpoint file for the 1st stage 2D pose detector')
    parser.add_argument(
        'pose_lifter_config',
        help='Config file for the 2nd stage pose lifter model')
    parser.add_argument(
        'pose_lifter_checkpoint',
        help='Checkpoint file for the 2nd stage pose lifter model')
    parser.add_argument(
        '--video-path', type=str, default='', help='Video path')
    parser.add_argument(
        '--rebase-keypoint-height',
        action='store_true',
        help='Rebase the predicted 3D pose so its lowest keypoint has a '
        'height of 0 (landing on the ground). This is useful for '
        'visualization when the model do not predict the global position '
        'of the 3D pose.')
    parser.add_argument(
        '--norm-pose-2d',
        action='store_true',
        help='Scale the bbox (along with the 2D pose) to the average bbox '
        'scale of the dataset, and move the bbox (along with the 2D pose) to '
        'the average bbox center of the dataset. This is useful when bbox '
        'is small, especially in multi-person scenarios.')
    parser.add_argument(
        '--num-instances',
        type=int,
        default=-1,
        help='The number of 3D poses to be visualized in every frame. If '
        'less than 0, it will be set to the number of pose results in the '
        'first frame.')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show visualizations.')
    parser.add_argument(
        '--out-video-root',
        type=str,
        default='vis_results',
        help='Root of the output video file. '
        'Default not saving the visualization video.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device for inference')
    parser.add_argument(
        '--det-cat-id',
        type=int,
        default=1,
        help='Category id for bounding box detection model')
    parser.add_argument(
        '--bbox-thr',
        type=float,
        default=0.9,
        help='Bounding box score threshold')
    parser.add_argument('--kpt-thr', type=float, default=0.3)
    parser.add_argument(
        '--use-oks-tracking', action='store_true', help='Using OKS tracking')
    parser.add_argument(
        '--tracking-thr', type=float, default=0.3, help='Tracking threshold')
    parser.add_argument(
        '--radius',
        type=int,
        default=8,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=2,
        help='Link thickness for visualization')
    parser.add_argument(
        '--smooth',
        action='store_true',
        help='Apply a temporal filter to smooth the pose estimation results. '
        'See also --smooth-filter-cfg.')
    parser.add_argument(
        '--smooth-filter-cfg',
        type=str,
        default='configs/_base_/filters/one_euro.py',
        help='Config file of the filter to smooth the pose estimation '
        'results. See also --smooth.')

    parser.add_argument(
        '--use-multi-frames',
        action='store_true',
        default=False,
        help='whether to use multi frames for inference in the 2D pose'
        'detection stage. Default: False.')
    parser.add_argument(
        '--online',
        action='store_true',
        default=False,
        help='inference mode. If set to True, can not use future frame'
        'information when using multi frames for inference in the 2D pose'
        'detection stage. Default: False.')

    assert has_mmdet, 'Please install mmdet to run the demo.'

    args = parser.parse_args()
    assert args.show or (args.out_video_root != '')
    assert args.det_config is not None
    assert args.det_checkpoint is not None

    video = mmcv.VideoReader(args.video_path)
    assert video.opened, f'Failed to load video file {args.video_path}'

    # First stage: 2D pose detection
    print('Stage 1: 2D pose detection.')

    print('Initializing model...')
    person_det_model = init_detector(
        args.det_config, args.det_checkpoint, device=args.device.lower())

    pose_det_model = init_pose_model(
        args.pose_detector_config,
        args.pose_detector_checkpoint,
        device=args.device.lower())

    assert isinstance(pose_det_model, TopDown), 'Only "TopDown"' \
        'model is supported for the 1st stage (2D pose detection)'

    # frame index offsets for inference, used in multi-frame inference setting
    if args.use_multi_frames:
        assert 'frame_indices_test' in pose_det_model.cfg.data.test.data_cfg
        indices = pose_det_model.cfg.data.test.data_cfg['frame_indices_test']

    pose_det_dataset = pose_det_model.cfg.data['test']['type']
    # get datasetinfo
    dataset_info = pose_det_model.cfg.data['test'].get('dataset_info', None)
    if dataset_info is None:
        warnings.warn(
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
    else:
        dataset_info = DatasetInfo(dataset_info)

    pose_det_results_list = []
    next_id = 0
    pose_det_results = []

    # whether to return heatmap, optional
    return_heatmap = False

    # return the output of some desired layers,
    # e.g. use ('backbone', ) to return backbone feature
    output_layer_names = None

    print('Running 2D pose detection inference...')
    for frame_id, cur_frame in enumerate(mmcv.track_iter_progress(video)):
        pose_det_results_last = pose_det_results

        # test a single image, the resulting box is (x1, y1, x2, y2)
        mmdet_results = inference_detector(person_det_model, cur_frame)

        # keep the person class bounding boxes.
        person_det_results = process_mmdet_results(mmdet_results,
                                                   args.det_cat_id)

        if args.use_multi_frames:
            frames = collect_multi_frames(video, frame_id, indices,
                                          args.online)

        # make person results for current image
        pose_det_results, _ = inference_top_down_pose_model(
            pose_det_model,
            frames if args.use_multi_frames else cur_frame,
            person_det_results,
            bbox_thr=args.bbox_thr,
            format='xyxy',
            dataset=pose_det_dataset,
            dataset_info=dataset_info,
            return_heatmap=return_heatmap,
            outputs=output_layer_names)

        # get track id for each person instance
        pose_det_results, next_id = get_track_id(
            pose_det_results,
            pose_det_results_last,
            next_id,
            use_oks=args.use_oks_tracking,
            tracking_thr=args.tracking_thr)

        pose_det_results_list.append(copy.deepcopy(pose_det_results))

    # Second stage: Pose lifting
    print('Stage 2: 2D-to-3D pose lifting.')

    print('Initializing model...')
    pose_lift_model = init_pose_model(
        args.pose_lifter_config,
        args.pose_lifter_checkpoint,
        device=args.device.lower())

    assert isinstance(pose_lift_model, PoseLifter), \
        'Only "PoseLifter" model is supported for the 2nd stage ' \
        '(2D-to-3D lifting)'
    pose_lift_dataset = pose_lift_model.cfg.data['test']['type']

    if args.out_video_root == '':
        save_out_video = False
    else:
        os.makedirs(args.out_video_root, exist_ok=True)
        save_out_video = True

    if save_out_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = video.fps
        writer = None

    # convert keypoint definition
    for pose_det_results in pose_det_results_list:
        for res in pose_det_results:
            keypoints = res['keypoints']
            res['keypoints'] = convert_keypoint_definition(
                keypoints, pose_det_dataset, pose_lift_dataset)

    # load temporal padding config from model.data_cfg
    if hasattr(pose_lift_model.cfg, 'test_data_cfg'):
        data_cfg = pose_lift_model.cfg.test_data_cfg
    else:
        data_cfg = pose_lift_model.cfg.data_cfg

    # build pose smoother for temporal refinement
    if args.smooth:
        smoother = Smoother(filter_cfg=args.smooth_filter_cfg, keypoint_dim=3)
    else:
        smoother = None

    num_instances = args.num_instances
    print('Running 2D-to-3D pose lifting inference...')
    for i, pose_det_results in enumerate(
            mmcv.track_iter_progress(pose_det_results_list)):
        # extract and pad input pose2d sequence
        pose_results_2d = extract_pose_sequence(
            pose_det_results_list,
            frame_idx=i,
            causal=data_cfg.causal,
            seq_len=data_cfg.seq_len,
            step=data_cfg.seq_frame_interval)
        # 2D-to-3D pose lifting
        pose_lift_results = inference_pose_lifter_model(
            pose_lift_model,
            pose_results_2d=pose_results_2d,
            dataset=pose_lift_dataset,
            with_track_id=True,
            image_size=video.resolution,
            norm_pose_2d=args.norm_pose_2d)

        # Pose processing
        pose_lift_results_vis = []
        for idx, res in enumerate(pose_lift_results):
            keypoints_3d = res['keypoints_3d']
            # exchange y,z-axis, and then reverse the direction of x,z-axis
            keypoints_3d = keypoints_3d[..., [0, 2, 1]]
            keypoints_3d[..., 0] = -keypoints_3d[..., 0]
            keypoints_3d[..., 2] = -keypoints_3d[..., 2]
            # rebase height (z-axis)
            if args.rebase_keypoint_height:
                keypoints_3d[..., 2] -= np.min(
                    keypoints_3d[..., 2], axis=-1, keepdims=True)
            res['keypoints_3d'] = keypoints_3d
            # add title
            det_res = pose_det_results[idx]
            instance_id = det_res['track_id']
            res['title'] = f'Prediction ({instance_id})'
            # only visualize the target frame
            res['keypoints'] = det_res['keypoints']
            res['bbox'] = det_res['bbox']
            res['track_id'] = instance_id
            pose_lift_results_vis.append(res)

        # Smoothing
        if smoother:
            pose_lift_results = smoother.smooth(pose_lift_results)

        # Visualization
        if num_instances < 0:
            num_instances = len(pose_lift_results_vis)
        img_vis = vis_3d_pose_result(
            pose_lift_model,
            result=pose_lift_results_vis,
            img=video[i],
            out_file=None,
            radius=args.radius,
            thickness=args.thickness,
            num_instances=num_instances)

        if save_out_video:
            if writer is None:
                writer = cv2.VideoWriter(
                    osp.join(args.out_video_root,
                             f'vis3D_{osp.basename(args.video_path)}'), fourcc,
                    fps, (img_vis.shape[1], img_vis.shape[0]))
            writer.write(img_vis)
    if save_out_video:
        writer.release()


if __name__ == '__main__':
    main()