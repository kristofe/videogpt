#!/bin/bash

# Create VideoMNIST dataset for VideoGPT training
# Usage: sh scripts/preprocess/videomnist/create_videomnist_dataset.sh datasets/videomnist

set -e

# Check if output directory is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <output_directory>"
    echo "Example: $0 datasets/videomnist"
    exit 1
fi

OUTPUT_DIR=$1
HDF5_FILE="${OUTPUT_DIR}/videomnist.h5"

echo "Creating VideoMNIST dataset in: ${OUTPUT_DIR}"

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Generate VideoMNIST dataset directly as HDF5
echo "Generating Moving MNIST dataset..."
python scripts/preprocess/videomnist/generate_videomnist_numpy.py \
    --output_file ${HDF5_FILE} \
    --sequence_length 16 \
    --videos_per_digit 1000 \
    --motion_types bounce circular linear \
    --image_size 64 \
    --create_samples \
    --sample_formats gif mp4 \
    --num_samples 20

echo "VideoMNIST dataset creation completed!"
echo ""
echo "Dataset structure:"
echo "  HDF5 file: ${HDF5_FILE}"
echo "  Sample videos: ${OUTPUT_DIR}/samples/"
echo ""
echo "Usage with VideoGPT:"
echo "  Training VQ-VAE:"
echo "    python scripts/train_vqvae.py --data_path ${HDF5_FILE} --resolution 64 --sequence_length 16"
echo ""
echo "  Training VideoGPT:"
echo "    python scripts/train_videogpt.py --data_path ${HDF5_FILE} --resolution 64 --sequence_length 16"
echo ""
echo "  Sampling VideoGPT:"
echo "    python scripts/sample_videogpt.py --data_path ${HDF5_FILE} --resolution 64 --sequence_length 16"
