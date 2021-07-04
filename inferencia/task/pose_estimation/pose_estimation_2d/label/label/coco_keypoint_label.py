import numpy as np


class COCOKeypointLabel():
    key_points = {
        'nose': 0,
        'left_eye': 1,
        'right_eye': 2,
        'left_ear': 3,
        'right_ear': 4,
        'left_shoulder': 5,
        'right_shoulder': 6,
        'left_elbow': 7,
        'right_elbow': 8,
        'left_wrist': 9,
        'right_wrist': 10,
        'left_hip': 11,
        'right_hip': 12,
        'left_knee': 13,
        'right_knee': 14,
        'left_ankle': 15,
        'right_ankle': 16
    }

    body_edges = np.array(
        [
            [0, 1],
            [0, 2],
            [1, 3],
            [2, 4],
            [0, 5],
            [0, 6],
            [5, 7],
            [7, 9],
            [6, 8],
            [8, 10],
            [5, 6],
            [5, 11],
            [6, 12],
            [11, 12],
            [11, 13],
            [13, 15],
            [12, 14],
            [14, 16]
        ]
    )
