"""
Code for computing 6DoF pose of objects from line predictions done on mtevents dataset samples.
"""
import numpy as np
from letr_prediction import get_data_unified
from scipy.spatial.transform import Rotation as R, Slerp
from scipy.interpolate import interp1d
import os
import json
class Annotation():
    def __init__(self, dataset_root, matching_name, rgb_file_name):
        sample_name = os.path.basename(rgb_file_name).split('.')[0]
        annotation_path = os.path.join(dataset_root, 'matched_poses', matching_name, sample_name + '.json')
        #self.__init__(annotation_path)
        with open(annotation_path) as f:
            self.annotation = json.load(f)

    #def load_from_path(self, annotation_path):
    #    with open(annotation_path) as f:
    #        self.annotation = json.load(f)

    def tracked_objects_ids(self):
        return [obj['id'] for obj in self.annotation['tracked_objects']]

    def _compute_rotation_mat_from_pose(self, pose):
        eangles = pose['euler_angles_deg']
        roll, pitch, yaw = np.deg2rad(eangles[0]), np.deg2rad(eangles[1]), np.deg2rad(eangles[2])

        cx, cy, cz = np.cos([roll, pitch, yaw])
        sx, sy, sz = np.sin([roll, pitch, yaw])

        R_x = np.array([[1, 0, 0],
                        [0, cx, -sx],
                        [0, sx, cx]])

        R_y = np.array([[cy, 0, sy],
                        [0, 1, 0],
                        [-sy, 0, cy]])

        R_z = np.array([[cz, -sz, 0],
                        [sz, cz, 0],
                        [0, 0, 1]])
        return R_z @ R_y @ R_x

    def _compute_rotation_mat_from_pose1(self, pose):
        return R.from_euler('xyz', pose['euler_angles_deg'], degrees=True).as_matrix()


    def get_transform(self, object_id):
        if str(object_id) not in self.annotation['matched_objects']:
            raise(Exception('This image do not have match for object with id' + int(object_id)))

        poses = self.annotation['matched_objects'][str(object_id)]
        poses = sorted(poses, key=lambda p: p['pose_ts'])

        if len(poses) > 1:
            if self.annotation['rgb_ts'] >= poses[0]['pose_ts'] and self.annotation['rgb_ts'] <= poses[-1]['pose_ts']:
                key_times = [p['pose_ts'] for p in poses]
                key_rots = R.from_matrix([ self._compute_rotation_mat_from_pose1(p) for p in poses ])
                slerp = Slerp(key_times, key_rots)
                r = slerp(self.annotation['rgb_ts']).as_matrix()

                key_positions = np.array([p['center_3d'] for p in poses])
                t = interp1d(key_times, key_positions, axis=0, kind='linear')(self.annotation['rgb_ts'])
            else:
                raise(Exception('Now what?'))
        elif len(poses) == 1:
            r = self._compute_rotation_mat_from_pose1(poses[0])
            t = np.array(poses[0]['center_3d'])
        else:
            raise(Exception('Missing poses for object ' + object_id))

        transform = np.eye(4)
        transform[:3, :3] = r
        transform[:3, 3] = t
        return transform

def get_rotation(pred_lines, centered_points):
    u, s, vh = np.linalg.svd(centered_points)
    # get z direction
    z = vh[2]
    if z[2]>0:
        z = -1*z

    # get x direction
    segment_lengths = []
    for l in pred_lines:
        e1, e2 = l[:3], l[3:]
        segment_lengths.append(np.linalg.norm(e1-e2))

    segment_lengths = np.array(segment_lengths)
    sorti = np.argsort(segment_lengths)

    longest_2 = pred_lines[sorti[:2]]
    d1 = longest_2[0][:3] - longest_2[0][3:]
    d2 = longest_2[1][:3] - longest_2[1][3:]
    if np.dot(d1, -d2) > 0:
        x = np.mean(np.array([d1, -d2]), axis=0)
    else:
        x = np.mean(np.array([d1, d2]), axis=0)

    # orthogonalize x, z
    z /= np.linalg.norm(z)

    x = x - np.dot(z, x)*z
    x /= np.linalg.norm(x)

    # get y direction
    y = np.linalg.cross(z, x)

    # rotation matrix
    R = np.zeros([3, 3])
    R[:, 0] = x
    R[:, 1] = y
    R[:, 2] = z

    return R


def compute_distances(arr):
    dist_matrix = np.linalg.norm(arr[:, np.newaxis] - arr, axis=2)
    np.fill_diagonal(dist_matrix, np.inf)  # Don't consider distance to self
    return dist_matrix


def merge_closest(arr):
    while len(arr) > 4:
        # Compute pairwise distances
        dist_matrix = compute_distances(arr)

        # Find the indices of the closest points
        idx1, idx2 = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)

        # Merge the two closest points by replacing them with their mean
        new_point = np.mean([arr[idx1], arr[idx2]], axis=0)

        # Remove the two original points and add the new merged point
        arr = np.delete(arr, [idx1, idx2], axis=0)
        arr = np.vstack([arr, new_point])

    return arr


def get_translation(segment_points, z_direction, bin_height):
    processed = merge_closest(segment_points)
    segments_center = processed.mean(axis=0)
    T = segments_center - z_direction * (bin_height / 2)
    return T


def get_bin_z_offset(object_id):
    obj_id2offset = {
        2: 110.5,
        3: 110.0,
        4: 110.0,
        9: 161.0030212402343750,
        10: 161.0030212402343750,
        11: 126.7993621826171875,
        12: 128.5,
        13: 161.0970153808593750,
        16: 136.801239014,
    }
    return obj_id2offset[object_id]


def calculate_pose(entry, lines, scores):
    #target_lines = np.array(target['lines'])

    normalized = entry['normalized']
    means = np.array(normalized['means'])
    stds = np.array(normalized['stds'])

    # pick 4 lines with highest confidence
    keep = np.array(np.argsort(scores)[::-1][:4])
    best_4_lines = lines[keep]*1000

    # undo normalization
    #best_4_lines = ((best_4_lines.reshape(-1,3) * stds) + means).reshape(-1, 6)

    segment_points = best_4_lines.reshape(8, 3)
    segments_center = segment_points.mean(axis=0)

    #bin_height = entry['bin_height'] if 'bin_height' in entry else get_bin_z_offset(entry['dir'])*2
    bin_height = get_bin_z_offset(entry['object_id'])*2#*0.001

    R = get_rotation(best_4_lines, segment_points - segments_center)
    T = get_translation(segment_points, R[:, 2], bin_height)

    return R, T


def get_poses(data):
    _, _, entry, lines, scores = get_data_unified(data)

    transform = np.array(entry['transform_4x4'])

    gt_R = transform[:3, :3]
    gt_t = transform[:3, 3]

    R, t = calculate_pose(entry, lines, scores)

    msg = None
    z = gt_R[2, 2]
    if z >= 0:
        msg = 'facing away' + str(z)
    else:
        msg = None

    # to identify sample in eval log
    sample_title = entry['img_path']

    return R, t, gt_R, gt_t*1000, msg, sample_title
