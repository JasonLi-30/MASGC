# Copyright (c) OpenMMLab. All rights reserved.
from .fpn import FPN
from .gap_neck import GlobalAveragePooling
from .posewarper_neck import PoseWarperNeck
from .cbams_fpn import CBAMSFPN
from .pcbam_fpn import PCBAM_FPN
from .cbam_ori import CBAM_ori
from .cbam_res import CBAM_res
from .tripcbam import TriPCBAM_FPN
from .coordAttention import CA_FPN
from .pcbam_res import PCBAM_res
from .fpca import FPCA
from .tcformer_mta_neck import MTA

__all__ = ['FPN', 'GlobalAveragePooling',
           'PoseWarperNeck', 'CBAMSFPN', 'PCBAM_FPN', 'CBAM_ori', 'CBAM_res', 'TriPCBAM_FPN', 'CA_FPN',
           'PCBAM_res', 'FPCA', 'MTA']
