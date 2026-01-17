"""
Loader for bin dataset.
"""
import os
import json
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation
import numpy as np


def farthest_point_sampling(points, n_samples):
    """ Farthest point sampling.

    """
    selected_pts = np.zeros((n_samples,), dtype=int)
    dist_mat = pairwise_distance(points, points)
    # start from first point
    pt_idx = 0
    dist_to_set = dist_mat[:, pt_idx]
    for i in range(n_samples):
        selected_pts[i] = pt_idx
        dist_to_set = np.minimum(dist_to_set, dist_mat[:, pt_idx])
        pt_idx = np.argmax(dist_to_set)
    return selected_pts

def pairwise_distance(A, B):
    """ Compute pairwise distance of two point clouds.point

    Args:
        A: n x 3 numpy array
        B: m x 3 numpy array

    Return:
        C: n x m numpy array

    """
    diff = A[:, :, None] - B[:, :, None].T
    C = np.sqrt(np.sum(diff**2, axis=1))

    return C

def random_point(face_vertices):
    """ Sampling point using Barycentric coordiante.

    """
    r1, r2 = np.random.random(2)
    sqrt_r1 = np.sqrt(r1)
    point = (1 - sqrt_r1) * face_vertices[0, :] + \
        sqrt_r1 * (1 - r2) * face_vertices[1, :] + \
        sqrt_r1 * r2 * face_vertices[2, :]

    return point

def uniform_sample(vertices, faces, n_samples, with_normal=False):
    """ Sampling points according to the area of mesh surface.

    """
    sampled_points = np.zeros((n_samples, 3), dtype=float)
    normals = np.zeros((n_samples, 3), dtype=float)
    faces = vertices[faces]
    vec_cross = np.cross(faces[:, 1, :] - faces[:, 0, :],
                         faces[:, 2, :] - faces[:, 0, :])
    face_area = 0.5 * np.linalg.norm(vec_cross, axis=1)
    cum_area = np.cumsum(face_area)
    for i in range(n_samples):
        face_id = np.searchsorted(cum_area, np.random.random() * cum_area[-1])
        sampled_points[i] = random_point(faces[face_id, :, :])
        normals[i] = vec_cross[face_id]
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    if with_normal:
        sampled_points = np.concatenate((sampled_points, normals), axis=1)
    return sampled_points

def sample_points_from_mesh(path, n_pts, fps=False, ratio=2):
    """ Uniformly sampling points from mesh model with caching.

    Args:
        path: path to STL file.
        n_pts: int, number of points being sampled.
        fps: whether to use fps for post-processing, default False.
        ratio: int, if use fps, sample ratio*n_pts first, then use fps to sample final output.

    Returns:
        points: n_pts x 3
        bbox: 2 x 3 (min, max)
    """
    # build cache filename next to the model file (remove .stl extension and encode args)
    base, _ = os.path.splitext(path)
    cache_path = f"{base}_pts{n_pts}_fps{int(fps)}_ratio{ratio}.txt"

    # try load from cache
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r') as f:
                first_line = f.readline().strip()
            bbox_vals = np.fromstring(first_line, sep=' ')
            points = np.loadtxt(cache_path, skiprows=1)
            if points.shape[0] == n_pts and points.shape[1] == 3 and bbox_vals.size == 6:
                bbox = np.array([bbox_vals[:3], bbox_vals[3:6]])
                return points, bbox
            # if cache is inconsistent fall through to recompute
        except Exception:
            # corrupted cache -> recompute and overwrite
            pass

    # compute from mesh
    mesh = trimesh.load(path)
    vertices, faces = mesh.vertices, mesh.faces
    if fps:
        points = uniform_sample(vertices, faces, ratio * n_pts, with_normal=False)
        pts_idx = farthest_point_sampling(points[:, :3], n_pts)
        points = points[pts_idx]
    else:
        points = uniform_sample(vertices, faces, n_pts, with_normal=False)

    bbox = np.array([np.min(vertices, axis=0), np.max(vertices, axis=0)])

    # save cache: first line = bbox flattened (6 floats), following lines = points
    try:
        with open(cache_path, 'w') as f:
            f.write(' '.join(map(str, bbox.flatten())) + '\n')
            np.savetxt(f, points, fmt='%.6f')
    except Exception:
        # ignore caching failures
        pass

    return points, bbox


class Dataset(Dataset):
    def __init__(self, args, path, split,
                 width, height, lines_annotation_dir=None, normalize=True, keep_dim_aspect_ratio=True, preload=True, use_resize_cache=True,
                 cutout_prob=0.0, cutout_inside=True,
                 max_cutout_size=0.8, min_cutout_size=0.2,
                 crop_prob=0.0, crop_max_amount=0.3, crop_min_amount=0.0,
                 noise_sigma=None, t_sigma=0.0, random_rot=False):
        self.dataset_dir = os.path.dirname(path)
        self.lines_annotation_dir = lines_annotation_dir
        self.split = split
        self.width = width
        self.height = height
        self.keep_dim_aspect_ratio = keep_dim_aspect_ratio
        self.use_resize_cache = use_resize_cache
        self.preload = preload
        self.noise_sigma = noise_sigma
        self.t_sigma = t_sigma
        self.random_rot = random_rot
        self.use_bb_mask = args.bins_bb_segment

        self.cutout_prob = cutout_prob
        self.use_cutout = cutout_prob > 0.0
        self.cutout_inside = cutout_inside
        self.max_cutout_size = max_cutout_size
        self.min_cutout_size = min_cutout_size

        self.crop_prob = crop_prob
        self.crop_max_amount = crop_max_amount
        self.crop_min_amount = crop_min_amount
        self.use_crop = crop_prob > 0.0

        self.used_size = None

        if self.split != 'train' and self.cutout_prob > 0.0:
            print("***** Split is not train, but cutout is enabled! *****")
            exit(1)

        self.load_data(path)

        # synthetic, real filter
        if args.bins_pick_samples != 'all':
            picked = []
            print('picking', args.bins_pick_samples, 'samples')
            for i in range(len(self.entries)):
                if self.entries[i]['is_synthetic'] is True and args.bins_pick_samples == 'synthetic':
                    picked.append(self.entries[i])
                elif self.entries[i]['is_synthetic'] is False and args.bins_pick_samples == 'real':
                    picked.append(self.entries[i])
            self.entries = picked
            print('Picked', len(self.entries), 'samples')

        # subsample dataset
        if args.bins_subsample_batch < 1.0:
            reduced = []
            for i in range(len(self.entries)):
                if np.random.rand() < args.bins_subsample_batch:
                    reduced.append(self.entries[i])
            self.entries = reduced

        self.normalize = normalize
        self.means = [-7.317206859588623, -7.509462833404541, 621.6871337890625]
        self.stds = [222.503662109375, 165.90419006347656, 681.2403564453125]
        #self.compute_normalization_constants()

        if os.path.isdir(args.output_dir):
            path = os.path.join(args.output_dir, f'used_entries_{split}.txt')
            with open(path, 'w')  as f:
                f.write(path + '\n')
                for e in self.entries:
                    f.write(e['exr_positions_path'] + '\n')

        print("Split: ", self.split)
        print("Size: ", len(self))
        if self.preload:
            print("Preloading dataset to memory")
            self.preloaded = []
            for entry in self.entries:
                print(entry)
                entry['xyz'], entry['xyz_full'], entry['changes'] = self.load_xyz(entry)

            for i in range(len(self.entries)):
                print('preparing sample', i)
                self.preloaded.append(self.load_item(i))

    def load_data(self, json_path):
        print("Loading dataset from path: ", json_path)
        with open(json_path, 'r') as f:
            self.entries = json.load(f)

        if not self.split in self.entries and isinstance(self.entries, list):
            print('***************** OLD DATASET FORMAT *********************')
            if self.split == 'train':
                self.entries = [entry for i, entry in enumerate(self.entries) if i % 5 != 0]
            elif self.split == 'val':
                self.entries = [entry for i, entry in enumerate(self.entries) if i % 5 == 0]
        else:
            self.entries = self.entries[self.split]

        # convert paths to host format
        for i in range(len(self.entries)):
            for p in ['exr_positions_path']:
                if p in self.entries[i]:
                    self.entries[i][p] = os.path.join(*self.entries[i][p].split('\\'))

        # get paths in new dataset structure
        for i in range(len(self.entries)):
            if 'exr_positions_path' not in self.entries[i]:
                assert('exr_positions_name' in self.entries[i])
                dir = self.entries[i]['dir']
                exr_name = self.entries[i]['exr_positions_name']
                self.entries[i]['exr_positions_path'] = os.path.join(
                    'scans',
                    dir,
                    exr_name
                )

        for i in range(len(self.entries)):
            # load lines
            if self.lines_annotation_dir is not None:
                # new dataset structure
                assert ('bin_model_id' in self.entries[i])
                model_id = self.entries[i]['bin_model_id']
                lines_filename = f'bin_{model_id:03}.txt'
                lines_path = os.path.join(self.lines_annotation_dir, lines_filename)
                if not os.path.isfile(lines_path):
                    raise (Exception('Expected lines annotation at' + lines_path))
                self.entries[i]['lines'] = np.loadtxt(lines_path)
            else:
                # old dataset
                if 'lines' not in self.entries[i]:
                    sample_dir = os.path.dirname(self.entries[i]['exr_positions_path'])
                    lines_file = os.path.join(self.dataset_dir, sample_dir, 'bin_lines.txt')
                    if os.path.exists(lines_file):
                        self.entries[i]['lines'] = np.loadtxt(lines_file)
                else:
                    self.entries[i]['lines'] = np.array(self.entries[i]['lines'])
            if (
                'lines' not in self.entries[i]
                or self.entries[i]['lines'] is None
                or len(self.entries[i]['lines']) == 0
            ):
                raise (Exception(f'No lines for sample {i}'))

            # get bin model path
            model_path = None
            if 'bin_model_id' in self.entries[i]:
                # new dataset structure
                model_id = self.entries[i]['bin_model_id']
                model_filename = f'bin_{model_id:03}.stl'
                model_path = os.path.join(self.dataset_dir, 'bin_models', model_filename)
            else:
                # old dataset
                sample_dir = os.path.dirname(self.entries[i]['exr_positions_path'])
                model_path = os.path.join(self.dataset_dir, sample_dir, 'bin.stl')

            if model_path is None or not os.path.isfile(model_path):
                raise (Exception(f'No model for sample {i} (model path: "{model_path}")'))
            self.entries[i]['model_path'] = model_path

            if 'proper_transform' in self.entries[i] and not 'transform' in self.entries[i]:
                self.entries[i]['transform'] = self.entries[i]['proper_transform']

            # add ids
            self.entries[i]['sample_id'] = i

    def get_nomalization_constants(self):
        return self.means, self.stds

    def compute_normalization_constants(self):
        xyzs = []
        for i in range(len(self.entries)):
            xyz, _ = self.__getitem__(i)
            xyzs.append(xyz.tolist())
        xyzs = torch.tensor(xyzs)
        means = [xyzs[:, 0, :, :].mean().item(), xyzs[:, 1, :, :].mean().item(), xyzs[:, 2, :, :].mean().item()]
        stds = [xyzs[:, 0, :, :].std().item(), xyzs[:, 1, :, :].std().item(), xyzs[:, 2, :, :].std().item()]

        self.means = means
        self.stds = stds

        print('Per channel means:', means)
        print('Per channel stds:', stds)

    def cutout(self, xyz):
        mask_width = np.random.randint(int(self.min_cutout_size * self.width), int(self.max_cutout_size * self.width))
        mask_height = np.random.randint(int(self.min_cutout_size * self.height), int(self.max_cutout_size * self.height))

        mask_width_half = mask_width // 2
        offset_width = 1 if mask_width % 2 == 0 else 0

        mask_height_half = mask_height // 2
        offset_height = 1 if mask_height % 2 == 0 else 0

        xyz = xyz.copy()

        h, w = self.height, self.width

        if self.cutout_inside:
            cxmin, cxmax = mask_width_half, w + offset_width - mask_width_half
            cymin, cymax = mask_height_half, h + offset_height - mask_height_half
        else:
            cxmin, cxmax = 0, w + offset_width
            cymin, cymax = 0, h + offset_height

        cx = np.random.randint(cxmin, cxmax)
        cy = np.random.randint(cymin, cymax)
        xmin = cx - mask_width_half
        ymin = cy - mask_height_half
        xmax = xmin + mask_width
        ymax = ymin + mask_height
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(w, xmax)
        ymax = min(h, ymax)

        xyz[:, ymin:ymax, xmin:xmax] = 0.0

        return xyz

    def crop(self, xyz_full):
        width, height = xyz_full.shape[0], xyz_full.shape[1]
        crop_amount = self.crop_max_amount * np.random.rand() + self.crop_min_amount
        crop_width = int(width * (1.0-crop_amount))
        crop_height = int(height * (1.0-crop_amount))
        crop_x_start = np.random.randint(width - crop_width)
        crop_x_end = crop_x_start + crop_width
        crop_y_start = np.random.randint(height - crop_height)
        crop_y_end = crop_y_start + crop_height
        cropped = xyz_full[crop_x_start:crop_x_end, crop_y_start:crop_y_end, :]
        resized = cv2.resize(cropped, (self.width, self.height), interpolation=cv2.INTER_NEAREST_EXACT)
        return np.transpose(resized, [2, 0, 1])
    
    def get_masked_points(self, xyz, entry):
        bin_dims = self.models[entry['model_path']]['dims']   # (3,)
        transform = entry['transform']          # 4x4
        inv_tf = np.linalg.inv(transform)
        H, W = xyz.shape[1], xyz.shape[2]

        pts = xyz.reshape(3, -1).T
        valid_mask = ~(np.all(pts == 0.0, axis=1))
        pts_valid = pts[valid_mask]

        pts_h = np.hstack([pts_valid, np.ones((pts_valid.shape[0], 1))])
        pts_local = (inv_tf @ pts_h.T).T[:, :3]
        half = (bin_dims*1.1) / 2.0

        inside_mask = np.all((pts_local >= -half) & (pts_local <= half), axis=1)
        full_mask_flat = np.zeros(H * W, dtype=bool)
        full_mask_flat[valid_mask] = inside_mask
        full_mask = full_mask_flat.reshape(H, W)

        masked_points = np.zeros((3, H, W), dtype=xyz.dtype)
        masked_points[:, full_mask] = xyz[:, full_mask]

        return masked_points

    def __len__(self):
        """
        Length of dataset
        :return: number of elements in dataset
        """
        return len(self.entries)

    def is_synthetic(self, entry):
        sample_dir = os.path.dirname(entry['exr_positions_path'])
        flag_file = os.path.join(self.dataset_dir, sample_dir, 'synthetic')
        return os.path.exists(flag_file)
        
    def get_transformed_lines(self, lines, transform):
        #lines = entry['lines']
        
        starts = np.c_[(lines[:, :3], np.ones(lines.shape[0]).T)]
        ends = np.c_[(lines[:, 3:], np.ones(lines.shape[0]).T)]
        
        t_starts = transform @ starts.T
        t_ends = transform @ ends.T
        
        return np.hstack((t_starts.T[:, :3], t_ends.T[:, :3]))

    def get_resized_size(self, orig_width, orig_height):
        width = self.width
        height = self.height
        if self.keep_dim_aspect_ratio:
            if orig_width > orig_height:
                width = self.width
                height = int(orig_height * (self.width / orig_width))
            else:
                width = int(orig_width * (self.height / orig_height))
                height = self.height
        return width, height

    def proper_size(self, entry, xyz, changes):
        width, height = self.get_resized_size(xyz.shape[1], xyz.shape[0])
        if xyz.shape[1] != width or xyz.shape[0] != height:
            xyz = cv2.resize(xyz, (width, height), interpolation=cv2.INTER_NEAREST_EXACT)
            changes['resized'] = {'width': width, 'height': height}

        if width < self.width or height < self.height:
            top = (self.height - height) // 2
            bottom = self.height - height - top
            left = (self.width - width) // 2
            right = self.width - width - left

            # Apply padding (black border here)
            xyz = cv2.copyMakeBorder(xyz, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0.0, 0.0, 0.0))
            changes['padded'] = {'top': top, 'bottom': bottom, 'left': left, 'right': right}
        return xyz, changes

    def load_xyz(self, entry, need_orig=False):
        """
        Loads pointcloud for a given entry
        :param entry: entry from self.entries
        :return: pointcloud wit shape (3, height, width)
        """
        exr_path = None
        xyz = None
        cached = False
        xyz_full = None
        changes = {}
        if self.use_resize_cache:
            exr_path = os.path.join(self.dataset_dir, f'_cache_{self.width}x{self.height}', entry['exr_positions_path'])
            if os.path.exists(exr_path):
                xyz = cv2.imread(exr_path,  cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
                if xyz is None:
                    print(exr_path)
                    raise ValueError("Image at path ", exr_path)
                #print('Loaded resized sample from cache!')
                cached = True

        exr_path = os.path.join(self.dataset_dir, entry['exr_positions_path'])
        if xyz is None:
            #print('reading original')
            xyz = cv2.imread(exr_path,  cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            xyz_full = xyz.copy()
            if xyz is None:
                print(exr_path)
                raise ValueError("Image at path ", exr_path)

        if need_orig and xyz_full is None:
            xyz_full = cv2.imread(exr_path,  cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

        width, height = self.get_resized_size(xyz.shape[1], xyz.shape[0])
        if self.used_size is not None:
            if width != self.used_size[0] or height != self.used_size[1]:
                #raise ValueError("Image at path ", exr_path, "has different aspect ratio")
                pass
        else:
            self.used_size = (width, height)

        xyz, changes = self.proper_size(entry, xyz, changes)

        if self.use_resize_cache and cached is False and len(changes.keys()) > 0:
            exr_path = os.path.join(self.dataset_dir, f'_cache_{self.width}x{self.height}', entry['exr_positions_path'])
            exr_dir = os.path.dirname(exr_path)
            Path(exr_dir).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(exr_path, xyz)
            #print('Saved resized sample to cache')

        xyz = np.transpose(xyz, [2, 0, 1])
        return xyz, xyz_full, changes

    def get_aug_transform(self):
        """
        Generates random transformation using. R is from SO(3) thanks to QR decomposition.
        :return: random transformation matrix
        """
        if self.random_rot:
            R, _ = np.linalg.qr(np.random.randn(3, 3))
        else:
            R = np.eye(3)

        t = self.t_sigma * np.random.randn(3)

        out = np.zeros([4, 4])
        out[:3, :3] = R
        out[:3, 3] = t
        out[3, 3] = 1

        #if np.random.rand() < 0.5:
        #    out[0, 0] = out[0, 0] * -1

        return out

    def aug(self, xyz_gt, transform):
        """
        Applies transformation matrix to pointcloud
        :param xyz_gt: original pointcloud with shape (3, height, width)
        :param transform: (4, 4) transformation matrix
        :return: Transformed pointcloud with shape (3, height, width)
        """
        orig_shape = xyz_gt.shape
        xyz = np.reshape(xyz_gt, [-1, 3])
        xyz = np.concatenate([xyz, np.ones([xyz.shape[0], 1])], axis=-1)

        xyz_t = (transform @ xyz.T).T

        xyz_t = xyz_t[:, :3] / xyz_t[:, 3, np.newaxis]
        xyz_t = np.reshape(xyz_t, orig_shape)
        return xyz_t

    def normalize_lines(self, lines):
        stds = np.array(self.stds + self.stds)
        means = np.array(self.means + self.means)
        return (lines - means) / stds

    def normalize_xyz(self, xyz):
        no_point_mask = np.linalg.norm(xyz, axis=0) == 0
        means = np.array(self.means).reshape((3, 1, 1))
        stds = np.array(self.stds).reshape((3, 1, 1))
        normalized = (xyz - means) / stds
        normalized[:, no_point_mask] = 0
        return normalized

    def load_item(self, index):
        randomly_transformed = False
        entry = self.entries[index]

        gt_transform = np.array(entry['transform'])
        if gt_transform[0, 1] < 0.0:
            gt_transform[:, :2] *= -1

        if self.split == 'train' and (self.random_rot or self.t_sigma != 0.0):
            randomly_transformed = True
            aug_transform = self.get_aug_transform()
            transform = aug_transform @ gt_transform
        else:
            transform = gt_transform

        transform = transform.astype(np.float32)

        rot = Rotation.from_matrix(transform[:3, :3])
        rotvec = rot.as_rotvec()
        t = transform[:3, 3]

        if self.preload:
            xyz = entry['xyz']
        else:
            xyz, xyz_full, _ = self.load_xyz(entry)

        if self.use_bb_mask:
            xyz = self.get_masked_points(xyz, entry)

        if self.use_crop:
            randomly_transformed = True
            if self.preload:
                xyz_full = entry['xyz_full']
            if np.random.rand() < self.crop_prob:
                xyz = self.crop(xyz_full)

        if self.split == 'train' and (self.random_rot or self.t_sigma != 0.0):
            xyz = self.aug(xyz, aug_transform)

        xyz = xyz.astype(np.float32)

        if self.noise_sigma is not None:
            randomly_transformed = True
            xyz += self.noise_sigma * np.random.randn(*xyz.shape)

        if self.use_cutout:
            randomly_transformed = True
            if np.random.rand() < self.cutout_prob:
                xyz = self.cutout(xyz)

        target = {}
        lines = entry['lines']#[[0, 2]]
        target['image_id'] = np.array(entry['sample_id'])
        target['labels'] = np.array([0 for _ in lines], dtype=np.int64)
        target['area'] = np.array([1 for _ in lines])
        target['iscrowd'] = np.array([0 for _ in lines])
        target['lines'] = self.get_transformed_lines(lines, transform).astype(np.float32)

        if self.normalize:
            xyz = self.normalize_xyz(xyz)
            target['lines'] = self.normalize_lines(target['lines']).astype(np.float32)
            entry['normalized'] = {'means': self.means, 'stds': self.stds}
        else:
            entry['normalized'] = {'means': [0.0, 0.0, 0.0], 'stds': [1.0, 1.0, 1.0]}

        links = {
            'bin_model': entry['model_path']
        }

        return randomly_transformed, xyz, target, entry, links, rotvec, t, transform

    def __getitem__(self, index):
        if self.preload:
            item = self.preloaded[index]
            randomly_transformed = item[0]
            if randomly_transformed:
                item = self.load_item(index)
        else:
            item = self.load_item(index)

        _, xyz, target, entry, links, rotvec, t, transform = item

        return torch.tensor(xyz, dtype=torch.float32), {k: torch.from_numpy(v) for k, v in target.items()}, entry, links


def build_bins(image_set, args):
    if image_set == 'train':
        return Dataset(args, args.bins_path, 'train', args.bins_input_width, args.bins_input_height,
                            lines_annotation_dir=args.bins_lines_annotation_dir,
                            cutout_prob=args.bins_cutout_prob, cutout_inside=args.bins_cutout_inside,
                            max_cutout_size=args.bins_cutout_max_size, min_cutout_size=args.bins_cutout_min_size,
                            noise_sigma=args.bins_noise_sigma, t_sigma=args.bins_t_sigma, random_rot=args.bins_random_rot,
                            preload=not args.bins_no_preload)
    elif image_set == 'val':
        return Dataset(args, args.bins_path, 'val', args.bins_input_width, args.bins_input_height, lines_annotation_dir=args.bins_lines_annotation_dir, preload=not args.bins_no_preload)
    elif image_set == 'test':
        return Dataset(args, args.bins_path, 'test', args.bins_input_width, args.bins_input_height, lines_annotation_dir=args.bins_lines_annotation_dir, preload=not args.bins_no_preload)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('json', help='Path to dataset json file.')
    args = parser.parse_args()
    json_path = args.json

    dataset = Dataset(json_path, 'train', 258, 193, preload=False, noise_sigma=0.0, random_rot=True)
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=1)

    for item in data_loader:
        print(item['xyz'].size())
        xyz = item['xyz'][0].cpu().detach().numpy()

        print(np.mean(xyz))

        #fig = plt.figure()
        #ax = fig.add_subplot(projection='3d')
        #ax.scatter(xyz[0].ravel(), xyz[1].ravel(), xyz[2].ravel(), marker='o')

        #plt.show()
