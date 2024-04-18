dataset_info = dict(
    dataset_name='Arm9',
    paper_info=dict(
        author='WYH',
        title='ZJUHRI Arm9',
        container='',
        year='2022',
        homepage='http://vision.imar.ro/human3.6m/description.php',
    ),
    keypoint_info={
        0:
        dict(
            name='root',
            id=0,
            color=[51, 153, 255],
            type='upper',
            swap=''),
        1:
        dict(
            name='left_shoulder',
            id=1,
            color=[0, 255, 0],
            type='upper',
            swap='right_shoulder'),
        2:
        dict(
            name='left_elbow',
            id=2,
            color=[0, 255, 0],
            type='upper',
            swap='right_elbow'),
        3:
        dict(
            name='left_wrist',
            id=3,
            color=[0, 255, 0],
            type='upper',
            swap='right_wrist'),
        4:
        dict(
            name='right_shoulder',
            id=4,
            color=[255, 128, 0],
            type='upper',
            swap='left_shoulder'),
        5:
        dict(
            name='right_elbow',
            id=5,
            color=[255, 128, 0],
            type='upper',
            swap='left_elbow'),
        6:
        dict(
            name='right_wrist',
            id=6,
            color=[255, 128, 0],
            type='upper',
            swap='left_wrist'),
        7:
        dict(
            name='left_hand',
            id=7,
            color=[0, 255, 0],
            type='upper',
            swap='right_hand'),
        8:
        dict(
            name='right_hand',
            id=8,
            color=[255, 128, 0],
            type='upper',
            swap='left_hand')
    },
    skeleton_info={
        0:
        dict(link=('root', 'left_shoulder'), id=0, color=[51, 153, 255]),
        1:
        dict(link=('root', 'right_shoulder'), id=1, color=[51, 153, 255]),
        2:
        dict(link=('left_shoulder', 'left_elbow'), id=2, color=[0, 255, 0]),
        3:
        dict(link=('left_elbow', 'left_wrist'), id=3, color=[0, 255, 0]),
        4:
        dict(link=('left_wrist', 'left_hand'), id=4, color=[0, 255, 0]),
        5:
        dict(link=('right_shoulder', 'right_elbow'), id=5, color=[255, 128, 0]),
        6:
        dict(link=('right_elbow', 'right_wrist'), id=6, color=[255, 128, 0]),
        7:
        dict(link=('right_wrist', 'right_hand'), id=7, color=[255, 128, 0])
    },
    joint_weights=[1.] * 9,
    sigmas=[],
    stats_info=dict(bbox_center=(528., 427.), bbox_scale=400.))  # 存疑 wyh
