"""
Saving prediction data for evaluation, debugging and visualization

"""
import numpy as np
import os
import json
import torch.nn.functional as F
import gzip
from pathlib import Path
import shutil

def read_transform_file(file):
    with open(file, 'r') as tfile:
        P = tfile.readline().strip().split(' ')
        R = np.array([[float(P[0]), float(P[4]), float(P[8])],
                      [float(P[1]), float(P[5]), float(P[9])],
                      [float(P[2]), float(P[6]), float(P[10])]])
        t = np.array([float(P[12]), float(P[13]), float(P[14])])
        return R, t


def jsonify_value(v):
    def unwrap(method, v):
        c = getattr(v, method, None)
        if callable(c):
            return c()
        else:
            return v

    if isinstance(v, dict):
        return { k: jsonify_value(v[k]) for k in v.keys() }

    v = unwrap('detach', v)
    v = unwrap('cpu', v)
    v = unwrap('tolist', v)

    return v

import hashlib
file2hash = dict()


def save_prediction_data1(dataset_name, samples, targets, entries, linked_files, model_outputs, losses, out_filename, index=0, copy_linked_files=True, remove_sample=False):
    samples, _ = samples.decompose()

    sample = {} if remove_sample else jsonify_value(samples[index])
    target = jsonify_value(targets[index])
    entry = jsonify_value(entries[index])
    losses = jsonify_value(losses)
    linked_files = linked_files[index]
    output = { k: jsonify_value(model_outputs[k][index]) for k in  model_outputs.keys()}

    if copy_linked_files:
        new_links = dict()
        for link_name in linked_files.keys():
            linked_path = linked_files[link_name]
            if linked_path in file2hash:
                hash = file2hash[linked_path]
            else:
                with open(linked_path, "rb") as f:
                    checksum = hashlib.file_digest(f, "md5")
                file2hash[linked_path] = checksum.hexdigest()
            basename = os.path.basename(linked_path)
            ext = '.' + basename.split('.')[-1]
            if ext == basename:
                ext = ''

            dest_dir = 'linked_files'
            dest_dir_path = os.path.join(os.path.dirname(out_filename), dest_dir)
            copied_name = file2hash[linked_path] + ext

            Path(dest_dir_path).mkdir(parents=False, exist_ok=True)
            new_link_path = os.path.join(dest_dir_path, copied_name)
            if not os.path.isfile(new_link_path):
                shutil.copy(linked_path, new_link_path)

            new_links[link_name] = os.path.join(dest_dir, copied_name)
        linked_files = new_links

    with gzip.open(out_filename, "wt", encoding="utf-8") as f:
        json.dump({
            'dataset_name': dataset_name,
            'sample': sample,
            'target': target,
            'entry': entry,
            'output': output,
            'losses': losses,
            'linked_files': linked_files
        }, f)


def save_prediction_data(samples, outputs, targets,
                         entry, out_filename,
                         criterion=None, index=0):
    out_logits, out_line = outputs['pred_logits'][index].detach().cpu(), outputs['pred_lines'][index].detach().cpu()

    prob = F.softmax(out_logits, -1)
    scores, labels = prob[..., :-1].max(-1)

    lines = out_line
    xyz, _ = samples.decompose()

    transform = np.array(entry[index]['proper_transform'])
    gt_R1 = transform[:3, :3]
    gt_T = transform[:3, 3]
    gt_R2 = np.matrix.copy(gt_R1)
    gt_R2[:, :2] *= -1

    entry[index]['lines'] = entry[index]['lines'].tolist()
    data = {
        'entry': {k:jsonify_value(v) for (k, v) in entry[index].items() if k != 'xyz'},
        'targets': {key: value.tolist() for (key,value) in targets[index].items()},
        'prediction': {
            'scores': scores.tolist(),
            'labels': labels.tolist(),
            'lines': lines.tolist(),
        },
        'xyz': xyz[index].tolist(),
    }

    with open(out_filename, 'w') as f:
        json.dump(data, f)


def view3D(points, pred_lines, target_lines):
    from matplotlib import pyplot as plt
    points = points[np.random.rand(points.shape[0]) < 0.05]

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(points[:,0], points[:,1], points[:,2], marker='o')

    plot_lines_3D(target_lines, ax, 'g', 'annotation')
    plot_lines_3D(pred_lines, ax, 'r', 'prediction')

    plt.legend()
    return plt, ax


def view2D(img, pred_lines, target_lines):
    from matplotlib import pyplot as plt
    from PIL import Image, ImageDraw

    mean = np.array([0.538, 0.494, 0.453])
    std = np.array([0.257, 0.263, 0.273])

    img = np.moveaxis(img, 0, -1)

    img = std * img + mean
    img = np.clip(img, 0, 1)

    pil_img = Image.fromarray(np.uint8(img*255))
    draw = ImageDraw.Draw(pil_img)

    width = img.shape[1]
    height = img.shape[0]

    for line in target_lines:
        x1, y1, x2, y2 = line
        draw.line((x1*width, y1*height, x2*width, y2*height), fill=(0, 255, 0), width=2)

    for line in pred_lines:
        x1, y1, x2, y2 = line
        draw.line((x1*width, y1*height, x2*width, y2*height), fill=(255, 0, 0), width=2)

    plt.imshow(np.asarray(pil_img))
    return plt


def save_prediction_visualization(samples, outputs, filename, entry, index=0):
    from matplotlib import pyplot as plt
    # find lines
    out_logits, out_line = outputs['pred_logits'][index], outputs['pred_lines'][index]
    prob = F.softmax(out_logits, -1)
    scores, labels = prob[..., :-1].max(-1)
    lines = out_line.detach().cpu()
    scores = scores.detach().cpu().numpy()
    keep = np.array(np.argsort(scores)[::-1][:12])
    lines = lines[keep]

    xyz, _ = samples.decompose()
    xyz = xyz[index].detach().cpu().tolist()
    if lines.shape[1] == 4:
        plt = view2D(xyz, lines, [])
    else:
        plt, _ = view3D(np.array(xyz), lines, [], entry[index])
    plt.savefig(filename)
    plt.close()
