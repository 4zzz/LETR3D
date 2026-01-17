"""
Script to produce inference files using trained model for all dataset samples.

"""
import argparse
import os
from args import get_args_parser
import torch
from torch.utils.data import DataLoader
from models.letr import build
from datasets import build_dataset
import util.misc as utils
from util.save_prediction import save_prediction_data1


def create_letr(weights_path, set_cuda):
    checkpoint = torch.load(weights_path, weights_only=False, map_location='cpu')
    model_args = checkpoint['args']

    if set_cuda is False:
        model_args.device = 'cpu'
    else:
        model_args.device = 'cuda'
    model, criterion, _ = build(model_args)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    criterion.eval()
    return model, criterion, model_args


def main(args):
    model, criterion, model_args = create_letr(args.model, args.set_cuda)
    print(model_args)

    dataset = build_dataset(args.dataset_name, args.split, args)
    data_loader = DataLoader(dataset, args.batch_size, drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    if args.set_cuda:
        device = 'cuda'
    else:
        device = 'cpu'

    model.to(device)
    i = 0
    for samples, targets, entries, linked_files in data_loader:
        samples = samples.to(device)
        targets = [{k: (v.to(device) if k != 'exr_file' else v) for k, v in t.items()} for t in targets]

        if model_args.LETRpost:
            outputs, origin_indices = model(samples)
            loss_dict = criterion(outputs, targets, origin_indices)
        else:
            outputs = model(samples)
            loss_dict = criterion(outputs, targets)

        #print(loss_dict)
        for index in range(len(entries)):
            prediction_path = os.path.join(args.output_directory, f'eval_{i:03}.gzjson')
            save_prediction_data1(args.dataset_name, samples, targets, entries, linked_files, outputs, {}, prediction_path, index=index, remove_sample=args.dont_save_sample)
            print('saving prediction data to', prediction_path)
            i+=1


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Produce inference files using trained model for all dataset samples', parents=[get_args_parser()])
    parser.add_argument('--split', type=str, choices=('test', 'val'), default='test')
    parser.add_argument('--model', type=str)
    parser.add_argument('--save_png_visualization', action='store_true', default=False)
    parser.add_argument('-d', '--dont_save_sample', action='store_true', default=False)
    parser.add_argument('-o', '--output_directory', type=str)
    parser.add_argument('--set_cuda', action='store_true', default=False)

    args = parser.parse_args()

    if not os.path.exists(args.output_directory):
        os.mkdir(args.output_directory)

    main(args)
