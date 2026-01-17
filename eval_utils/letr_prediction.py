"""
Code for loading prediction data from prediction files. Provides a compatibility layer for the various formats we used.
"""
import numpy as np
import os
import json
import gzip


def read_prediction_data(path):
    ext = path.split('.')[-1].lower()
    if ext == 'json':
        f = open(path)
    elif ext == 'gzjson':
        f = gzip.open(path, "rt", encoding="utf-8")
    data = json.load(f)
    data['_prediction_path'] = path
    return data


def calculate_line_probabilities(class_logits):
    e = np.exp(class_logits - np.atleast_2d(np.max(class_logits, axis=1)).T)
    class_probabilities = e / np.atleast_2d(np.sum(e, axis=1)).T
    return class_probabilities[:, 0]


def get_predicted_lines(data):
    if 'prediction' in data:
        # old storage format
        pred_scores = np.array(data['prediction']['scores'])
        pred_lines = np.array(data['prediction']['lines'])
        return pred_lines, pred_scores
    else:
        # new storage format
        lines = np.array(data['output']['pred_lines'])
        probs = calculate_line_probabilities(np.array(data['output']['pred_logits']))
        return lines, probs


def get_sample(data):
    if 'xyz' in data:
        # old storage format
        return data['xyz']
    else:
        # new storage format
        return data['sample']


def get_data_unified(data):
    sample = get_sample(data)
    lines, probs = get_predicted_lines(data)
    entry = data['entry']
    target = data['targets'] if 'targets' in data else data['target']
    return sample, target, entry, lines, probs


def get_linked_file(link_name, data):
    if 'linked_files' in data and link_name in data['linked_files']:
        parent = os.path.dirname(data['_prediction_path'])
        f = os.path.join(parent, data['linked_files'][link_name])
        if os.path.exists(f):
            return f
    return None
