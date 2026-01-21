import os
import pickle as pkl
from pathlib import Path
import numpy as np
import torch

SMPLH_JOINT_NAMES = [
    'pelvis',
    'left_hip',
    'right_hip',
    'spine1',
    'left_knee',
    'right_knee',
    'spine2',
    'left_ankle',
    'right_ankle',
    'spine3',
    'left_foot',
    'right_foot',
    'neck',
    'left_collar',
    'right_collar',
    'head',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'left_index1',
    'left_index2',
    'left_index3',
    'left_middle1',
    'left_middle2',
    'left_middle3',
    'left_pinky1',
    'left_pinky2',
    'left_pinky3',
    'left_ring1',
    'left_ring2',
    'left_ring3',
    'left_thumb1',
    'left_thumb2',
    'left_thumb3',
    'right_index1',
    'right_index2',
    'right_index3',
    'right_middle1',
    'right_middle2',
    'right_middle3',
    'right_pinky1',
    'right_pinky2',
    'right_pinky3',
    'right_ring1',
    'right_ring2',
    'right_ring3',
    'right_thumb1',
    'right_thumb2',
    'right_thumb3',
    'nose',
    'right_eye',
    'left_eye',
    'right_ear',
    'left_ear',
    'left_big_toe',
    'left_small_toe',
    'left_heel',
    'right_big_toe',
    'right_small_toe',
    'right_heel',
    'left_thumb',
    'left_index',
    'left_middle',
    'left_ring',
    'left_pinky',
    'right_thumb',
    'right_index',
    'right_middle',
    'right_ring',
    'right_pinky',
]

SMPLH_LEFT_LEG = ['left_hip', 'left_knee', 'left_ankle', 'left_foot']
SMPLH_RIGHT_LEG = ['right_hip', 'right_knee', 'right_ankle', 'right_foot']
SMPLH_LEFT_ARM = ['left_collar', 'left_shoulder', 'left_elbow', 'left_wrist']
SMPLH_RIGHT_ARM = ['right_collar', 'right_shoulder', 'right_elbow', 'right_wrist']
SMPLH_HEAD = ['neck', 'head']
SMPLH_SPINE = ['spine1', 'spine2', 'spine3']

# name to 21 index (without pelvis, hand, and extra)
_name_2_idx = {j: i for i, j in enumerate(SMPLH_JOINT_NAMES[1:22])}
SMPLH_PART_IDX = {
    'left_leg': [_name_2_idx[x] for x in SMPLH_LEFT_LEG],
    'right_leg': [_name_2_idx[x] for x in SMPLH_RIGHT_LEG],
    'left_arm': [_name_2_idx[x] for x in SMPLH_LEFT_ARM],
    'right_arm': [_name_2_idx[x] for x in SMPLH_RIGHT_ARM],
    'two_legs': [_name_2_idx[x] for x in SMPLH_LEFT_LEG + SMPLH_RIGHT_LEG],
    'left_arm_and_leg': [_name_2_idx[x] for x in SMPLH_LEFT_ARM + SMPLH_LEFT_LEG],
    'right_arm_and_leg': [_name_2_idx[x] for x in SMPLH_RIGHT_ARM + SMPLH_RIGHT_LEG],
}

# name to full index
_name_2_idx_full = {j: i for i, j in enumerate(SMPLH_JOINT_NAMES)}
SMPLH_PART_IDX_FULL = {
    'lower_body': [_name_2_idx_full[x] for x in ['pelvis'] + SMPLH_LEFT_LEG + SMPLH_RIGHT_LEG]
}

# ===== ⬇️ Fitting optimizer ⬇️ ===== #
SMPL_JOINTS = {'hips': 0, 'leftUpLeg': 1, 'rightUpLeg': 2, 'spine': 3, 'leftLeg': 4, 'rightLeg': 5,
               'spine1': 6, 'leftFoot': 7, 'rightFoot': 8, 'spine2': 9, 'leftToeBase': 10, 'rightToeBase': 11,
               'neck': 12, 'leftShoulder': 13, 'rightShoulder': 14, 'head': 15, 'leftArm': 16, 'rightArm': 17,
               'leftForeArm': 18, 'rightForeArm': 19, 'leftHand': 20, 'rightHand': 21}

# chosen virtual mocap markers that are "keypoints" to work with
KEYPT_VERTS = [4404, 920, 3076, 3169, 823, 4310, 1010, 1085, 4495, 4569, 6615, 3217, 3313, 6713,
               6785, 3383, 6607, 3207, 1241, 1508, 4797, 4122, 1618, 1569, 5135, 5040, 5691, 5636,
               5404, 2230, 2173, 2108, 134, 3645, 6543, 3123, 3024, 4194, 1306, 182, 3694, 4294, 744]


# From https://github.com/vchoutas/smplify-x/blob/master/smplifyx/utils.py
# Please see license for usage restrictions.
def smpl_to_openpose(model_type='smplx', use_hands=True, use_face=True,
                     use_face_contour=False, openpose_format='coco25'):
    ''' Returns the indices of the permutation that maps SMPL to OpenPose

        Parameters
        ----------
        model_type: str, optional
            The type of SMPL-like model that is used. The default mapping
            returned is for the SMPLX model
        use_hands: bool, optional
            Flag for adding to the returned permutation the mapping for the
            hand keypoints. Defaults to True
        use_face: bool, optional
            Flag for adding to the returned permutation the mapping for the
            face keypoints. Defaults to True
        use_face_contour: bool, optional
            Flag for appending the facial contour keypoints. Defaults to False
        openpose_format: bool, optional
            The output format of OpenPose. For now only COCO-25 and COCO-19 is
            supported. Defaults to 'coco25'

    '''
    if openpose_format.lower() == 'coco25':
        if model_type == 'smpl':
            return np.array([24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4,
                             7, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34],
                            dtype=np.int32)
        elif model_type == 'smplh':
            body_mapping = np.array([52, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5,
                                     8, 1, 4, 7, 53, 54, 55, 56, 57, 58, 59,
                                     60, 61, 62], dtype=np.int32)
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array([20, 34, 35, 36, 63, 22, 23, 24, 64,
                                          25, 26, 27, 65, 31, 32, 33, 66, 28,
                                          29, 30, 67], dtype=np.int32)
                rhand_mapping = np.array([21, 49, 50, 51, 68, 37, 38, 39, 69,
                                          40, 41, 42, 70, 46, 47, 48, 71, 43,
                                          44, 45, 72], dtype=np.int32)
                mapping += [lhand_mapping, rhand_mapping]
            return np.concatenate(mapping)
        # SMPLX
        elif model_type == 'smplx':
            body_mapping = np.array([55, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5,
                                     8, 1, 4, 7, 56, 57, 58, 59, 60, 61, 62,
                                     63, 64, 65], dtype=np.int32)
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array([20, 37, 38, 39, 66, 25, 26, 27,
                                          67, 28, 29, 30, 68, 34, 35, 36, 69,
                                          31, 32, 33, 70], dtype=np.int32)
                rhand_mapping = np.array([21, 52, 53, 54, 71, 40, 41, 42, 72,
                                          43, 44, 45, 73, 49, 50, 51, 74, 46,
                                          47, 48, 75], dtype=np.int32)

                mapping += [lhand_mapping, rhand_mapping]
            if use_face:
                #  end_idx = 127 + 17 * use_face_contour
                face_mapping = np.arange(76, 127 + 17 * use_face_contour,
                                         dtype=np.int32)
                mapping += [face_mapping]

            return np.concatenate(mapping)
        else:
            raise ValueError('Unknown model type: {}'.format(model_type))
    elif openpose_format == 'coco19':
        if model_type == 'smpl':
            return np.array([24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8,
                             1, 4, 7, 25, 26, 27, 28],
                            dtype=np.int32)
        elif model_type == 'smplh':
            body_mapping = np.array([52, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5,
                                     8, 1, 4, 7, 53, 54, 55, 56],
                                    dtype=np.int32)
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array([20, 34, 35, 36, 57, 22, 23, 24, 58,
                                          25, 26, 27, 59, 31, 32, 33, 60, 28,
                                          29, 30, 61], dtype=np.int32)
                rhand_mapping = np.array([21, 49, 50, 51, 62, 37, 38, 39, 63,
                                          40, 41, 42, 64, 46, 47, 48, 65, 43,
                                          44, 45, 66], dtype=np.int32)
                mapping += [lhand_mapping, rhand_mapping]
            return np.concatenate(mapping)
        # SMPLX
        elif model_type == 'smplx':
            body_mapping = np.array([55, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5,
                                     8, 1, 4, 7, 56, 57, 58, 59],
                                    dtype=np.int32)
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array([20, 37, 38, 39, 60, 25, 26, 27,
                                          61, 28, 29, 30, 62, 34, 35, 36, 63,
                                          31, 32, 33, 64], dtype=np.int32)
                rhand_mapping = np.array([21, 52, 53, 54, 65, 40, 41, 42, 66,
                                          43, 44, 45, 67, 49, 50, 51, 68, 46,
                                          47, 48, 69], dtype=np.int32)

                mapping += [lhand_mapping, rhand_mapping]
            if use_face:
                face_mapping = np.arange(70, 70 + 51 +
                                         17 * use_face_contour,
                                         dtype=np.int32)
                mapping += [face_mapping]

            return np.concatenate(mapping)
        else:
            raise ValueError('Unknown model type: {}'.format(model_type))
    else:
        raise ValueError('Unknown joint format: {}'.format(openpose_format))

def load_hand_faces_and_colors(smplx_faces, mano_smplx_path="inputs/checkpoints/body_models/smplx/MANO_SMPLX_vertex_ids.pkl"):
    idxs_data = pkl.loads(Path(mano_smplx_path).read_bytes())
    left_hand_idx, right_hand_idx = idxs_data['left_hand'], idxs_data['right_hand']
    hand_idxs = np.concatenate([left_hand_idx, right_hand_idx])
    hand_set = set(hand_idxs.tolist())
    old_to_new = {old: new for new, old in enumerate(hand_idxs)}
    hand_faces = []
    for face in smplx_faces:
        v0, v1, v2 = face[0].item(), face[1].item(), face[2].item()
        if all(v in hand_set for v in [v0, v1, v2]):
            new_face = [old_to_new[v0], old_to_new[v1], old_to_new[v2]]
            hand_faces.append(new_face)
    hand_faces = np.array(hand_faces, dtype=np.int64)[None]  # [1, Y, 3]
    # stitching mesh faces from hands
    faces_new = np.array([[92, 38, 234], [234, 38, 239], [38, 122, 239], [239, 122, 279], [122, 118, 279],
                            [279, 118, 215], [118, 117, 215], [215, 117, 214], [117, 119, 214], [214, 119, 121],
                            [119, 120, 121], [121, 120, 78], [120, 108, 78], [78, 108, 79]]).reshape((1, -1, 3))
    hand_faces = torch.from_numpy(np.concatenate([hand_faces, faces_new], axis=1)).to(dtype=torch.long)


    left_hand_colors = np.tile(np.array([0.5, 1.0, 0.5]).reshape((1, 1, 3)), (1, left_hand_idx.shape[0], 1))
    right_hand_colors = np.tile(np.array([1.0, 0.5, 0.5]).reshape((1, 1, 3)), (1, right_hand_idx.shape[0], 1))
    hand_colors = torch.from_numpy(np.concatenate([left_hand_colors, right_hand_colors], axis=1)).to(dtype=torch.float32)
    return hand_idxs, hand_faces, hand_colors