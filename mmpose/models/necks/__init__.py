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
from .tripcbamv2 import TriPCBAM_v2
from .tripcbam_res import TriPCBAM_res
from .tripcbam_res_noCoor import TriPCBAM_res_NC
from .tripcbam_res_noCoorAndSpatial import TriPCBAM_res_NCS
from .tripcbamv2_res import TriPCBAMv2_res
from .tripcbamv2_res2 import TriPCBAMv2_res2
from .tripcbamv2_res_noCoor import TriPCBAMv2_res_NC
from .tripcbamv2_res_noCoorAndSpatial import TriPCBAMv2_res_NCS
from .tripcbamv3 import TriPCBAMv3
from .tripcbamv4 import TriPCBAMv4
from .tripcbamv5 import TriPCBAMv5
from .tcformer_mta_neck import MTA

__all__ = ['GlobalAveragePooling', 'PoseWarperNeck', 'FPN', 'CBAMSFPN',
           'PCBAM_FPN', 'CBAM_ori', 'CBAM_res', 'TriPCBAM_FPN', 'CA_FPN', 'PCBAM_res', 'TriPCBAM_v2',
           'TriPCBAM_res', 'TriPCBAM_res_NC', 'TriPCBAM_res_NCS', 'TriPCBAMv2_res', 'TriPCBAMv2_res2']
