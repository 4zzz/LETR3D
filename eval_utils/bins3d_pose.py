"""
Code for computing 6DoF pose of a bin from predicted line segment.
"""
import numpy as np
import os
from letr_prediction import get_data_unified

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

    longest_2 = pred_lines[sorti[-2:]]
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


def get_bin_z_offset(bin_dir):
    dir2type = {
        'TestBin': 3,
        'TestCarton': 7,
        'TestGold': 3,
        'TestGray': 0,
        'TestSynth': 5,
        'dataset0': 0,
        'dataset1': 1,
        'dataset2': 0,
        'dataset3': 2,
        'dataset4': 2,
        'ElavatedGrayBox': 0,
        'ElevatedGreyBox': 0,
        'ElevatedGreyFullBeer': 0,
        'FirstRealSet': 3,
        'GoldBinAdditional': 3,
        'GrayBoxPad': 0,
        'LargeWoodenBoxDynamic': 2,
        'LargeWoodenBoxStatic': 2,
        'ShallowGreyBox': 0,
        'SmalGreyBasket': 4,
        'SmallGoldenBox': 3,
        'SmallWhiteBasket': 1,
        'synth_dataset5_random_origin': 5,
        'synth_dataset6_random_origin': 6,
        'synth_dataset7_random_origin': 5,
        'synth_dataset8_random_origin': 5,
    }
    type2offset = {
        0: 67.8125,
        1: 95.115,
        2: 160.398,
        3: 75.6255,
        4: 119.425,
        5: 108.75,
        6: 56.25,
        7: 48.0,
    }
    return type2offset[dir2type[bin_dir]]


def calculate_pose(entry, lines, scores):
    #target_lines = np.array(target['lines'])

    normalized = entry['normalized']
    means = np.array(normalized['means'])
    stds = np.array(normalized['stds'])

    # pick 4 lines with highest confidence
    keep = np.array(np.argsort(scores)[::-1][:4])
    best_4_lines = lines[keep]

    # undo normalization
    best_4_lines = ((best_4_lines.reshape(-1,3) * stds) + means).reshape(-1, 6)

    segment_points = best_4_lines.reshape(8, 3)
    segments_center = segment_points.mean(axis=0)

    bin_height = entry['bin_height'] if 'bin_height' in entry else get_bin_z_offset(entry['dir'])*2

    R = get_rotation(best_4_lines, segment_points - segments_center)
    T = get_translation(segment_points, R[:, 2], bin_height)

    return R, T


def get_poses(data):
    _, _, entry, lines, scores = get_data_unified(data)
    if 'proper_transform' in entry:
        # old format
        transform = np.array(entry['proper_transform'])
    elif 'transform' in entry:
        # new format
        transform = np.array(entry['transform'])
    else:
        raise (Exception('Ground truth transform not found in prediction file'))
    gt_R = transform[:3, :3]
    gt_t = transform[:3, 3]

    R, t = calculate_pose(entry, lines, scores)

    # to identify sample in eval log
    sample_title = entry['dir'] + '/' + os.path.basename(entry['exr_positions_path'])

    return R, t, gt_R, gt_t, None, sample_title
