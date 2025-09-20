#!/bin/bash

# Create sample visualizations from existing VideoMNIST dataset
# Usage: sh scripts/preprocess/videomnist/create_samples.sh datasets/videomnist/videomnist.h5 output_samples

set -e

# Check if arguments are provided
if [ $# -lt 2 ]; then
    echo "Usage: $0 <hdf5_file> <output_directory> [options]"
    echo "Example: $0 datasets/videomnist/videomnist.h5 output_samples"
    echo ""
    echo "Options:"
    echo "  --num_samples N     Number of sample videos (default: 20)"
    echo "  --formats FMT       Output formats: gif, mp4 (default: gif mp4)"
    echo "  --fps N             Frames per second for MP4 (default: 10)"
    echo "  --comparison        Create class comparison GIFs"
    echo "  --montage-only      Only create montage image"
    exit 1
fi

HDF5_FILE=$1
OUTPUT_DIR=$2
shift 2

# Default options
NUM_SAMPLES=20
FORMATS="gif mp4"
FPS=10
COMPARISON=false
MONTAGE_ONLY=false

# Parse additional arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --num_samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --formats)
            FORMATS="$2"
            shift 2
            ;;
        --fps)
            FPS="$2"
            shift 2
            ;;
        --comparison)
            COMPARISON=true
            shift
            ;;
        --montage-only)
            MONTAGE_ONLY=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "Creating VideoMNIST sample visualizations..."
echo "  Input: $HDF5_FILE"
echo "  Output: $OUTPUT_DIR"
echo "  Samples: $NUM_SAMPLES"
echo "  Formats: $FORMATS"
echo "  FPS: $FPS"

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Run visualization script
if [ "$MONTAGE_ONLY" = true ]; then
    python scripts/preprocess/videomnist/visualize_videomnist.py \
        --hdf5_path ${HDF5_FILE} \
        --output_dir ${OUTPUT_DIR} \
        --num_samples ${NUM_SAMPLES} \
        --formats gif \
        --create_montage \
        --fps ${FPS}
else
    python scripts/preprocess/videomnist/visualize_videomnist.py \
        --hdf5_path ${HDF5_FILE} \
        --output_dir ${OUTPUT_DIR} \
        --num_samples ${NUM_SAMPLES} \
        --formats ${FORMATS} \
        --create_montage \
        --fps ${FPS}
fi

# Create class comparisons if requested
if [ "$COMPARISON" = true ]; then
    echo "Creating class comparison GIFs..."
    python scripts/preprocess/videomnist/visualize_videomnist.py \
        --hdf5_path ${HDF5_FILE} \
        --output_dir ${OUTPUT_DIR}/class_comparisons \
        --create_comparison \
        --num_per_class 3
fi

echo "Sample visualization completed!"
echo "Files saved to: ${OUTPUT_DIR}"
echo ""
echo "Generated files:"
echo "  - Individual sample videos (${FORMATS})"
echo "  - Montage image (videomnist_montage.png)"
echo "  - Dataset statistics (dataset_stats.txt)"
if [ "$COMPARISON" = true ]; then
    echo "  - Class comparison GIFs (class_comparisons/)"
fi
