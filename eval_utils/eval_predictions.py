import argparse
import os
import numpy as np

from err_metrics import calculate_eRE, calculate_eTE

from letr_prediction import read_prediction_data
from bins3d_pose import get_poses as bins3d_get_poses
from mtevents_pose import get_poses as mtevents_get_poses


def get_pose(data):
    if 'dataset_name' in data:
        dataset_name = data['dataset_name']
    else:
        dataset_name = 'bins'

    if dataset_name == 'bins':
        return bins3d_get_poses(data)
    elif dataset_name == 'mtevents_rgb':
        return mtevents_get_poses(data)


def main(args):

    if os.path.isfile(args.target):
        pred_files = [args.target]
    elif os.path.isdir(args.target):
        pred_files = [
            os.path.join(args.target, f) for f in os.listdir(args.target) if (
                    f.lower().endswith('.gzjson')
                    or f.lower().endswith('.json')
                )
        ]

    stats = {
        'eTE': [],
        'eRE': [],
    }

    for f in pred_files:
        data = read_prediction_data(f)
        R, t, gt_R1, gt_t, msg, title = get_pose(data)

        gt_R2 = np.matrix.copy(gt_R1)
        gt_R2[:, :2] *= -1

        eRE = min(calculate_eRE(gt_R1, R), calculate_eRE(gt_R2, R))
        eTE = calculate_eTE(gt_t, t)


        stats['eTE'].append(eTE)
        stats['eRE'].append(eRE)

        print(os.path.basename(f), title, ': eTE =', eTE, ', eRE =', eRE, '' if msg == None else msg)

    for k in stats.keys():
        metric = np.array(stats[k])

        vals = {
            'mean': metric.mean(),
            'std': metric.std(),
            'median': np.median(metric)
        }
        line = ' '.join([f'{k} = {vals[k]}' for k in vals.keys()])
        print(k,':', line)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluate estimated poses from line predictions')
    parser.add_argument('target', help='Specify a file to evaluate a single prediction, or a directory to evaluate all files in it (with extensions *.json or *.gzjson).')

    args = parser.parse_args()
    main(args)
