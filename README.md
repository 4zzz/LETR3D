## Detecting 3D Line Segments for 6DoF Pose Estimation with Limited Data (VISAPP 2026)

Deep learning–based method for direct 3D line segment detection in RGB-D data, derived from LETR, together with evaluation code using 3D line segments for 6DoF bin pose estimation.

## Installation 

## Dataset

## Experiments

Training models to test the effect of query count:
```bash
sh experiments/bins_3D/query_count_experiment.sh \
    --bins_path /path/to/dataset/train_val.json \
    --bins_lines_annotation_dir experiments/bins_3D/lines_annotation/outer_edge
```

Training models with and without synthetic samples to evaluate their contribution:
```bash
sh experiments/bins_3D/synth_no_synth_experiment.sh \
    --bins_path /path/to/dataset/train_val.json \
    --bins_lines_annotation_dir experiments/bins_3D/lines_annotation/outer_edge
```
Training models to evaluate the effect of cutout augmentation:
```bash
sh experiments/bins_3D/cutout_experiment.sh \
    --bins_path /path/to/dataset/train_val.json \
    --bins_lines_annotation_dir experiments/bins_3D/lines_annotation/outer_edge
```
Training the final model based on the results of the previous ablation studies:
```bash
sh experiments/bins_3D/train_final.sh \
    --bins_path /path/to/dataset/train_val.json \
    --bins_lines_annotation_dir experiments/bins_3D/lines_annotation/outer_edge
```

## Training

## Evaluation

To evaluate trained model run inference script `infer_dataset.py`, which for every sample produce prediction file. :
```bash
python src/infer_dataset.py \
    --dataset_name bins \
    --model /path/to/model.pth \
    --output_directory predictions \
    --split test \
    --bins_path /path/to/dataset/test.json \
    --bins_lines_annotation_dir experiments/bins_3D/lines_annotation/outer_edge \
    --bins_no_preload \
    --bins_input_width 516 \
    --bins_input_height 386
```

## Citation

## Acknowledgements

This code is based on the implementations of [**LETR: Line Segment Detection Using Transformers without Edges **](https://github.com/mlpc-ucsd/LETR). 
