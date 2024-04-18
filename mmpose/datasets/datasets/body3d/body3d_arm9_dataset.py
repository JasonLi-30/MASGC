# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile
import warnings
from collections import OrderedDict, defaultdict

import mmcv
import numpy as np
from mmcv import Config, deprecated_api_warning

from mmpose.core.evaluation import keypoint_mpjpe
from mmpose.core.evaluation import keypoint_mpjpe_wyh
from mmpose.datasets.datasets.base import Kpt3dSviewKpt2dDataset
from ...builder import DATASETS


@DATASETS.register_module()
class Body3DArm9Dataset(Kpt3dSviewKpt2dDataset):
    """Arm9 dataset for 3D human pose estimation.

    keypoint indexes::

        0: 'root',
        1: 'left_shoulder',
        2: 'left_elbow',
        3: 'left_wrist',
        4: 'right_shoulder',
        5: 'right_elbow',
        6: 'right_wrist',
        7: 'left_hand',
        8: 'right_hand',

    Args:
        ann_file (str): Path to the annotation file.
        img_prefix (str): 保存图像的目录的路径.  [Default: None]
        data_cfg (dict): config
        pipeline (list[dict | callable]): A sequence of data transforms 一系列数据转换.
        dataset_info (DatasetInfo): A class containing all dataset info.
        test_mode (bool): Store True when building test or
            validation dataset. Default: False.

        - num_joints: Number of joints.
        - seq_len: 序列中的帧数. Default: 1.
        - seq_frame_interval: 以一定的间隔从视频中提取帧. Default: 1.
        - causal: If set to True, 最右边的输入帧将是目标帧.
            Otherwise, 中间的输入帧将是目标帧. Default: True. (配置文件一般为False)
        - temporal_padding: Whether to pad the video so that poses will be
            predicted for every frame in the video. Default: False
        - subset: Reduce dataset size by fraction. Default: 1.
        - need_2d_label: Whether need 2D joint labels or not.
            Default: False.
    """

    JOINT_NAMES = [
        'Root', 'LShoulder', 'LElbow', 'LWrist', 'RShoulder', 'RElbow', 'RWrist',
        'LHand', 'RHand'
    ]

    # --2D joint source options--:
    # "gt": from the annotation file
    # "detection": from a detection result file of 2D keypoint
    # "pipeline": will be generate by the pipeline
    SUPPORTED_JOINT_2D_SRC = {'gt', 'detection', 'pipeline'}

    # metric
    ALLOWED_METRICS = {'mpjpe', 'p-mpjpe', 'n-mpjpe'}

    def __init__(self,
                 ann_file,
                 img_prefix,
                 data_cfg,
                 pipeline,
                 dataset_info=None,
                 test_mode=False):

        if dataset_info is None:
            warnings.warn(
                'dataset_info is missing. '
                'Check https://github.com/open-mmlab/mmpose/pull/663 '
                'for details.', DeprecationWarning)
            cfg = Config.fromfile('configs/_base_/datasets/h36m.py')
            dataset_info = cfg._cfg_dict['dataset_info']

        super().__init__(
            ann_file,
            img_prefix,
            data_cfg,
            pipeline,
            dataset_info=dataset_info,
            test_mode=test_mode)

    def load_config(self, data_cfg):
        super().load_config(data_cfg)
        # arm9 特定属性
        self.joint_2d_src = data_cfg.get('joint_2d_src', 'gt')
        if self.joint_2d_src not in self.SUPPORTED_JOINT_2D_SRC:
            raise ValueError(
                f'Unsupported joint_2d_src "{self.joint_2d_src}". '
                f'Supported options are {self.SUPPORTED_JOINT_2D_SRC}')

        self.joint_2d_det_file = data_cfg.get('joint_2d_det_file', None)

        self.need_camera_param = data_cfg.get('need_camera_param', False)
        if self.need_camera_param:
            assert 'camera_param_file' in data_cfg
            self.camera_param = self._load_camera_param(data_cfg['camera_param_file'])

        # arm9 specific annotation info  ----无用---
        ann_info = {}
        ann_info['use_different_joint_weights'] = False

        self.ann_info.update(ann_info)

    def load_annotations(self):
        data_info = super().load_annotations()  # from kpt_3d_sview_kpt_2d_dataset.py

        # get 2D joints
        if self.joint_2d_src == 'gt':
            data_info['joints_2d'] = data_info['joints_2d']
        elif self.joint_2d_src == 'detection':  # wyh 使用detect的结果覆盖gt的2d结果
            data_info['joints_2d'] = self._load_joint_2d_detection(
                self.joint_2d_det_file)
            assert data_info['joints_2d'].shape[0] == data_info[
                'joints_3d'].shape[0]
            assert data_info['joints_2d'].shape[2] == 3
        elif self.joint_2d_src == 'pipeline':
            # joint_2d will be generated in the pipeline
            pass
        else:
            raise NotImplementedError(
                f'Unhandled joint_2d_src option {self.joint_2d_src}')

        return data_info

    @staticmethod
    def _parse_arm9_imgname(imgname):
        """解析 imgname 获取主题、动作和相机信息（重载为获取所属视频信息）
        A typical arm9 image filename is like:  Nov_marker12_00001.jpg
        """
        subj, rest = osp.basename(imgname).rsplit('_', 1)  # Nov_marker12
        frameIndex, rest = rest.split('.', 1)  # 00001

        return subj, frameIndex

    def build_sample_indices(self):
        """将原始视频分割成序列并构建帧索引  （此方法将覆盖基类中的默认方法.  """
        # Group frames into videos. Assume that self.data_info is chronological按时间顺序排列的.
        video_frames = defaultdict(list)
        for idx, imgname in enumerate(self.data_info['imgnames']):  # 遍历所有帧（打包）
            subj, frameIndex = self._parse_arm9_imgname(imgname)  # 分解
            video_frames[subj].append(idx)

        # 构建样本索引
        sample_indices = []
        _len = (self.seq_len - 1) * self.seq_frame_interval + 1
        _step = self.seq_frame_interval
        for _, _indices in sorted(video_frames.items()):
            n_frame = len(_indices)  # wyh 估计有451

            if self.temporal_padding:  # 时序填充
                # 填充序列，这样序列中的每一帧都将被预测
                if self.causal:  # 【因果卷积？】
                    frames_left = self.seq_len - 1
                    frames_right = 0
                else:  # 对称卷积
                    frames_left = (self.seq_len - 1) // 2
                    frames_right = frames_left
                for i in range(n_frame):
                    pad_left = max(0, frames_left - i // _step)
                    pad_right = max(0,
                                    frames_right - (n_frame - 1 - i) // _step)
                    start = max(i % _step, i - frames_left * _step)
                    end = min(n_frame - (n_frame - 1 - i) % _step,
                              i + frames_right * _step + 1)
                    sample_indices.append([_indices[0]] * pad_left +
                                          _indices[start:end:_step] +
                                          [_indices[-1]] * pad_right)
            else:
                seqs_from_video = [
                    _indices[i:(i + _len):_step]
                    for i in range(0, n_frame - _len + 1)
                ]
                sample_indices.extend(seqs_from_video)

        # reduce dataset size if self.subset < 1
        assert 0 < self.subset <= 1
        subset_size = int(len(sample_indices) * self.subset)
        start = np.random.randint(0, len(sample_indices) - subset_size + 1)  # 基本就是0
        end = start + subset_size

        return sample_indices[start:end]

    def _load_joint_2d_detection(self, det_file):
        """"Load 2D joint detection results from file."""
        joints_2d = np.load(det_file).astype(np.float32)

        return joints_2d

    @deprecated_api_warning(name_dict=dict(outputs='results'))
    def evaluate(self, results, res_folder=None, metric='mpjpe', **kwargs):
        metrics = metric if isinstance(metric, list) else [metric]
        for _metric in metrics:
            if _metric not in self.ALLOWED_METRICS:
                raise ValueError(
                    f'Unsupported metric "{_metric}" for arm9 dataset.'
                    f'Supported metrics are {self.ALLOWED_METRICS}')

        if res_folder is not None:
            tmp_folder = None
            res_file = osp.join(res_folder, 'result_keypoints.json')
        else:
            tmp_folder = tempfile.TemporaryDirectory()
            res_file = osp.join(tmp_folder.name, 'result_keypoints.json')

        kpts = []
        for result in results:
            preds = result['preds']  # (batchsize, 9, 3)
            image_paths = result['target_image_paths']
            batch_size = len(image_paths)
            for i in range(batch_size):
                target_id = self.name2id[image_paths[i]]
                kpts.append({
                    'keypoints': preds[i],
                    'target_id': target_id,
                })

        mmcv.dump(kpts, res_file)

        name_value_tuples = []
        for _metric in metrics:
            if _metric == 'mpjpe':
                _nv_tuples = self._report_mpjpe(kpts)
            elif _metric == 'p-mpjpe':
                _nv_tuples = self._report_mpjpe(kpts, mode='p-mpjpe')
            elif _metric == 'n-mpjpe':
                _nv_tuples = self._report_mpjpe(kpts, mode='n-mpjpe')
            else:
                raise NotImplementedError
            name_value_tuples.extend(_nv_tuples)  # 列表尾部追加值（类似append）

        if tmp_folder is not None:
            tmp_folder.cleanup()

        return OrderedDict(name_value_tuples)

    def _report_mpjpe(self, keypoint_results, mode='mpjpe'):
        """Cauculate mean per joint position error (MPJPE) or its variants like P-MPJPE or N-MPJPE.
            计算每个关节位置误差的平均值以及变体
        Args:
            keypoint_results (list): Keypoint predictions. See
                'Body3DH36MDataset.evaluate' for details.
            mode (str): Specify mpjpe variants. Supported options are:

                - 'mpjpe': Standard MPJPE.
                - 'p-mpjpe': MPJPE after aligning prediction to groundtruth via a rigid transformation (scale, rotation
                        and translation).
                - 'n-mpjpe': MPJPE after aligning prediction to groundtruth in scale only.
        """

        preds = []
        gts = []
        masks = []
        # action_category_indices = defaultdict(list)
        subj_category_indices = defaultdict(list)
        for idx, result in enumerate(keypoint_results):
            pred = result['keypoints']
            target_id = result['target_id']
            gt, gt_visible = np.split(  # gt (9,3)    gt_visible (9,1)
                self.data_info['joints_3d'][target_id], [3], axis=-1)
            preds.append(pred)
            gts.append(gt)
            masks.append(gt_visible)

            subj = self._parse_arm9_imgname(self.data_info['imgnames'][target_id])[0]  # 返回类别
            # action_category = subj.split('_')[0]
            # action_category_indices[subj].append(idx)
            subj_category_indices[subj].append(idx)

        preds = np.stack(preds)  # (FrameTotal, 9, 3)
        gts = np.stack(gts)  # (FrameTotal, 9, 3)
        masks = np.stack(masks).squeeze(-1) > 0  # (FrameTotal, 9)  True or False

        err_name = mode.upper()  # 大写转换
        if mode == 'mpjpe':
            alignment = 'none'
        elif mode == 'p-mpjpe':
            alignment = 'procrustes'
        elif mode == 'n-mpjpe':
            alignment = 'scale'
        else:
            raise ValueError(f'Invalid mode: {mode}')

        error, error_all = keypoint_mpjpe_wyh(preds, gts, masks, alignment)  # ---- 计算MPJPE --- eg.0.04
        name_value_tuples = [(err_name, error), (err_name + '_ALL', error_all)]

        # 分类计算
        for subj_category, indices in subj_category_indices.items():  # action_category类别
            _error, _error_all = keypoint_mpjpe_wyh(preds[indices], gts[indices], masks[indices], alignment)
            name_value_tuples.append((f'{err_name}_{subj_category}', _error))
            name_value_tuples.append((f'{err_name}_ALL_{subj_category}', _error_all))

        # divide in 9 keypoints
        if mode == 'mpjpe':
            for k in range(9):
                errorK, error_allK = keypoint_mpjpe_wyh(preds[:, k], gts[:, k], masks[:, k])
                name_value_tuples.append((f'{err_name}_keypoint{k}', errorK))
                name_value_tuples.append((f'{err_name}_ALL_keypoint{k}', error_allK))

        return name_value_tuples

    def _load_camera_param(self, camera_param_file):
        """Load camera parameters from file."""
        return mmcv.load(camera_param_file)

    def get_camera_param(self, imgname):
        """Get camera parameters of a frame by its image name."""
        assert hasattr(self, 'camera_param')
        # subj, _ = self._parse_arm9_imgname(imgname)
        return self.camera_param['colorCamera']   # 直接给彩色相机参数
