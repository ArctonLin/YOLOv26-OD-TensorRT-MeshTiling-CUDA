#!/bin/bash
set -e

# Export model using config.json if engine does not exist
echo "Checking TRT model status..."
if [ ! -f "yolo26n.engine" ]; then
    echo "Exporting model to ONNX using settings from config.json..."
    python3 scripts/export_model.py --config config.json
    
    echo "Building TensorRT engine using system trtexec to match C++ API version perfectly..."
    trtexec --onnx=yolo26n.onnx --saveEngine=yolo26n.engine
else
    echo "TRT engine already exists, skipping export."
fi

# Build C++ project
echo "Building C++ Project..."
mkdir -p build
cd build
cmake ..
make -j$(nproc)
cd ..

# Run Video Processing
INPUT_VIDEO="4k-highway-traffic.mp4"
OUTPUT_VIDEO="4k-highway-traffic-detect.mp4"

if [ ! -f "$INPUT_VIDEO" ]; then
    echo "Error: Input video $INPUT_VIDEO not found!"
    exit 1
fi

echo "Running YOLO Detector on Video..."
./build/yolo_detector config.json "$INPUT_VIDEO" "$OUTPUT_VIDEO"
