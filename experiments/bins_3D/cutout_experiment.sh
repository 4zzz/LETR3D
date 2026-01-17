#!/bin/sh

DEBUG_ARGS=(
    --save_prediction_probability 0.01
    --save_prediction_data
    --save_prediction_visualization
    --save_test_prediction_probability 0.005
    --save_test_prediction_data
    --save_test_prediction_visualization
)

ADD_ARGS=(
        --dataset_name bins
        --bins_input_width 516
        --bins_input_height 386
        --backbone resnet50
        --layer1_num 3
        --bins_no_preload
)
if [ -n "$DEBUG" -a "$DEBUG" == "1" ] ; then
    ADD_ARGS+=${DEBUG_ARGS[@]}
fi

set -x
q=4
for cutout in {0.3,0.4,0.6,0.8} ; do
    echo "cutout: $cutout"
    experiment_dir="bins_output/cutout/$cutout"
    OPENCV_IO_ENABLE_OPENEXR=1 python src/main.py \
        --resume  https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth \
        --output_dir "$experiment_dir/stage1" \
        --num_queries $q \
        --bins_cutout_max_size $cutout \
        --bins_cutout_prob 0.5 \
        --lr_drop 150 \
        --epochs 300 \
        ${ADD_ARGS[@]} "$@"

    OPENCV_IO_ENABLE_OPENEXR=1 python src/main.py \
        --output_dir "$experiment_dir/stage2" \
        --num_queries $q \
        --bins_cutout_max_size $cutout \
        --bins_cutout_prob 0.5 \
        --LETRpost \
        --no_opt \
        --layer1_frozen \
        --frozen_weights "$experiment_dir/stage1/checkpoints/checkpoint.pth" \
        --lr_drop 30 \
        --epochs 99 \
        ${ADD_ARGS[@]} "$@"

    OPENCV_IO_ENABLE_OPENEXR=1 python src/main.py \
        --output_dir "$experiment_dir/stage3" \
        --num_queries $q \
        --bins_cutout_max_size $cutout \
        --bins_cutout_prob 0.5 \
        --LETRpost \
        --no_opt \
        --layer1_frozen \
        --resume "$experiment_dir/stage2/checkpoints/checkpoint.pth" \
        --label_loss_func focal_loss \
        --label_loss_params '{"gamma":2.0}' \
        --save_freq 1 \
        --lr 1e-5 \
        --epochs 20 \
        ${ADD_ARGS[@]} "$@"
    exit
done
