dataset_info = dict(
    dataset_name='Arm6',
    paper_info=dict(
        author='WYH',
        title='ZJUHRI Arm6',
        container='',
        year='2022',
        homepage='http://vision.imar.ro/human3.6m/description.php',
    ),
    keypoint_info={
        0:
        # dict(name='root', id=0, color=[51, 153, 255], type='lower', swap=''),
        dict(
            name='left_shoulder',
            id=0,
            color=[0, 255, 0],
            type='upper',
            swap='right_shoulder'),
        1:
        dict(
            name='left_elbow',
            id=1,
            color=[0, 255, 0],
            type='upper',
            swap='right_elbow'),
        2:
        dict(
            name='left_wrist',
            id=2,
            color=[0, 255, 0],
            type='upper',
            swap='right_wrist'),
        3:
        dict(
            name='right_shoulder',
            id=3,
            color=[255, 128, 0],
            type='upper',
            swap='left_shoulder'),
        4:
        dict(
            name='right_elbow',
            id=4,
            color=[255, 128, 0],
            type='upper',
            swap='left_elbow'),
        5:
        dict(
            name='right_wrist',
            id=5,
            color=[255, 128, 0],
            type='upper',
            swap='left_wrist')
    },
    skeleton_info={
        0:
        dict(link=('left_shoulder', 'left_elbow'), id=0, color=[0, 255, 0]),
        1:
        dict(link=('left_elbow', 'left_wrist'), id=1, color=[0, 255, 0]),
        2:
        dict(link=('left_shoulder', 'right_shoulder'), id=2, color=[51, 153, 255]),
        3:
        dict(link=('right_shoulder', 'right_elbow'), id=3, color=[255, 128, 0]),
        4:
        dict(link=('right_elbow', 'right_wrist'), id=4, color=[255, 128, 0])
    },
    joint_weights=[1.] * 17,
    sigmas=[],
    stats_info=dict(bbox_center=(528., 427.), bbox_scale=400.))  # 存疑 wyh
