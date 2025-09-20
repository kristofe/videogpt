#!/bin/bash

# Create VizDoom dataset for VideoGPT
# This script downloads and processes VizDoom data into the required HDF5 format

mkdir -p ${1}
mkdir -p ~/.cache/vizdoom

echo "Setting up VizDoom dataset..."

# Check if VizDoom is installed
if ! python -c "import vizdoom" 2>/dev/null; then
    echo "Installing VizDoom..."
    pip install vizdoom
fi

# Create dataset directory structure
mkdir -p ${1}/train
mkdir -p ${1}/test

# Generate VizDoom videos
python scripts/preprocess/vizdoom/generate_vizdoom_videos.py \
    --output_dir ${1} \
    --num_train_videos 1000 \
    --num_test_videos 200 \
    --sequence_length 16 \
    --resolution 64

# Convert to HDF5 format
python scripts/preprocess/vizdoom/vizdoom_to_hdf5.py \
    --data_dir ${1} \
    --output_file ${1}/vizdoom.h5 \
    --sequence_length 16 \
    --resolution 64

echo "VizDoom dataset created successfully at ${1}/vizdoom.h5"
