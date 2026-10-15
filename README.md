# YOLO-TensorRT-MeshTiling-CUDADecode

High-Resolution (4K/8K) Real-Time YOLO Object Detection on NVIDIA TensorRT with Custom CUDA Mesh Tiling, 2D Grid Parallel NMS, and Multi-Threaded GPU Streams in C++.

---

## Benchmark & Performance

> [!NOTE]
> Performance evaluated on **4K UHD (3840×2160)** video with an overlapping $7 \times 4$ grid (**28 tiles of 640×640** per frame) using YOLOv26n TensorRT FP16 engine.

| Metric / Hardware Parameter | Benchmark Specification / Performance Result |
| :--- | :--- |
| **CPU** | **AMD Ryzen 7 5700X3D** (8 Cores / 16 Threads) |
| **GPU** | **NVIDIA GeForce RTX 4070 Ti SUPER 16GB** |
| **RAM** | **128 GB** (DDR4-3200 CL16 32GB × 4) |
| **Resolution & Grid** | **4K UHD (3840 × 2160)** / $7 \times 4 = 28$ Tiles ($640 \times 640$) |
| **End-to-End Pipeline Throughput** | **23.5 FPS @ 4K** (Real-Time 4K Video Processing) |
| **Tile Inference Rate** | **658 Tiles/sec** ($28\text{ tiles} \times 23.5\text{ FPS}$) |
| **Input Buffer Rate** | **584.76 MB/s** ($3840 \times 2160 \times 3 \times 23.5\text{ B/s} \approx 557.67\text{ MiB/s}$) |
| **TensorRT Buffer Rate** | **808.55 MB/s** ($640 \times 640 \times 3 \times 28 \times 23.5\text{ B/s} \approx 771.09\text{ MiB/s}$)<br>*(FP16 Buffer Rate: **1,617.10 MB/s** / $1,542.19\text{ MiB/s}$)* |

---

## Table of Contents
- [Benchmark & Performance](#benchmark--performance)
- [Demo](#demo)
- [Introduction](#introduction)
- [Key Features](#key-features)
- [Pipeline Architecture](#pipeline-architecture)
- [Setup Compiler Environment](#setup-compiler-environment)
  - [A. Visual C++ Compiler](#a-visual-c-compiler)
  - [B. NVIDIA CUDA / cuDNN / TensorRT](#b-nvidia-cuda--cudnn--tensorrt)
  - [C. OpenCV with CUDA Support (Build from Source)](#c-opencv-with-cuda-support-build-from-source)
- [Model Export & Engine Generation](#model-export--engine-generation)
- [Build Instructions](#build-instructions)
  - [Windows (Visual Studio / MSBuild)](#windows-visual-studio--msbuild)
  - [Linux / CMake](#linux--cmake)
- [Configuration (`config.json`)](#configuration-configjson)
- [Usage](#usage)
- [Architecture & Performance Highlights](#architecture--performance-highlights)

---

## Demo

<div align="center">
  <video src="4k-highway-traffic-detect.mp4" width="100%" controls="controls" autoplay="autoplay" loop="loop" muted="muted"></video>
  <p><em>4K (3840×2160) Real-Time Highway Traffic Detection with Mesh Tiling & TensorRT</em></p>
</div>

---

## Introduction

High-resolution videos (such as 4K 3840×2160 or 8K) frequently feature very small objects (vehicles, pedestrians, aerial targets, traffic infrastructure). When such frames are directly downsampled to standard YOLO input sizes (e.g., 640×640), fine details are irreversibly lost, drastically degrading detection accuracy.

A common remedy is **image tiling** (splitting the high-res frame into overlapping sub-images). However, traditional tiling approaches execute image slicing, resizing, normalization, and Non-Maximum Suppression (NMS) sequentially on the CPU, creating severe CPU bottlenecks and PCIe bandwidth saturation that prevent real-time performance.

**YOLO-TensorRT-MeshTiling-CUDADecode** solves this with an end-to-end, GPU-accelerated pipeline:
- **CUDA Mesh Tiling**: Direct GPU memory slicing and FP16 tensor layout generation from full-resolution frames.
- **Batched TensorRT 10.x FP16 Inference**: Concurrent execution across multiple CUDA streams.
- **Parallel 2D Grid Local NMS**: High-throughput GPU candidate suppression and object compaction.
- **CPU Global Tile-to-Tile NMS**: Overlapping boundary object merging concurrent with GPU execution.
- **Zero-Allocation Worker Pipeline**: Pinned host memory and pre-allocated GPU buffers throughout.

---

## Key Features

- **Mesh Tiling Kernel (`launch_mesh_tiling`)**:
  Slices the full-resolution frame into a configured mesh grid (e.g., $7 \times 4 = 28$ tiles of $640 \times 640$). Resizing, normalization ($[0, 1]$), and FP16 (`half`) channel conversion are performed directly on GPU memory in a single CUDA kernel pass.
- **Batched TensorRT Execution (`EnqueueV3`)**:
  Processes all 28 tiles simultaneously in a single engine inference call (`batch_size = 28`), maximizing modern GPU tensor core saturation.
- **Parallel 2D Grid Local NMS**:
  Eliminates the $O(N^2)$ candidate comparison bottleneck using a 2D thread block grid on the GPU.
- **AtomicAdd Memory Compaction**:
  Compacts surviving valid objects per tile into a contiguous buffer (`batch_size * max_objects_per_tile`), discarding inactive boxes and reducing PCIe transfer payload by >98%.
- **Single Pinned DtoH Transfer (`cudaMallocHost`)**:
  Valid detections and object counts are copied asynchronously without stalling the GPU or CPU pipeline.
- **CPU Tile-to-Tile Global NMS (`tile2tile_nms`)**:
  Deduplicates objects lying on adjacent tile boundaries in parallel on CPU worker threads while the GPU processes subsequent frames.
- **Multi-Worker Threading**:
  Configurable worker pool (`cpu_threads`), where each worker owns an independent TensorRT execution context, dedicated CUDA stream, and pre-allocated memory buffers.
- **In-RAM Pre-Decoding Benchmark Mode**:
  Pre-decodes video frames into RAM (default capped to 1000 frames) before the benchmark timer begins, isolating pure GPU inference and rendering throughput from video decoder limitations.

---

## Pipeline Architecture

```
                      [Video File]
                           │
             (Pre-decode into RAM Buffer)
                           │
                    [Feeder Thread] 
                           │ (Zero-copy std::move)
                           ▼
               [BoundedQueue<FrameTask>]
                           │
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
      [Worker 0]      [Worker 1]     [Worker N-1]
     (Context 0)     (Context 1)    (Context N-1)
           │               │               │
    ┌──────┴───────────────┴───────────────┴──────────┐
    │ 1. Upload to cv::cuda::GpuMat                   │
    │ 2. launch_mesh_tiling (CUDA Kernel, FP16)       │
    │ 3. TensorRT Batched Inference (FP16 EnqueueV3)  │
    │ 4. decode_candidates_kernel (CUDA Kernel)       │
    │ 5. parallel_local_nms_kernel (2D Grid CUDA)     │
    │ 6. compact_valid_objects_kernel (atomicAdd)     │
    │ 7. Async DtoH Copy via Pinned Host Memory       │
    │ 8. CPU Tile-to-Tile Global NMS (stitch borders) │
    │ 9. Draw Bounding Boxes on CPU (Parallel)        │
    └──────────────────────┬──────────────────────────┘
                           │
                           ▼
                [OrderedFrameCollector] (Re-order in sequence)
                           │
                           ▼
                  [VideoWriter Output]
```

---

## Setup Compiler Environment

This project utilizes OpenCV with CUDA support (`cv::cuda::GpuMat`) for asynchronous memory transfers. OpenCV must be compiled with CUDA enabled.

### A. Visual C++ Compiler
1. Install **Visual Studio 2022** with the **Desktop development with C++** workload (MSVC v143, Windows SDK).
2. Microsoft Visual C++ (MSVC) must be installed **before** installing NVIDIA CUDA Toolkit.
   *(If CUDA was installed prior to Visual Studio, copy files from `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\extras\visual_studio_integration\MSBuildExtensions` to `C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Microsoft\VC\v170\BuildCustomizations`).*

### B. NVIDIA CUDA / cuDNN / TensorRT
1. **CUDA Toolkit**: Install **CUDA 12.8** (or 12.x compatible).
2. **cuDNN**: Install **cuDNN 9.10.2** (e.g. `cudnn-windows-x86_64-9.10.2.21_cuda12-archive.zip`) and extract contents into your CUDA installation directory:
   `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8`
3. **TensorRT**: Extract **TensorRT-10.12.0.36** to `D:\Compile` (e.g. `D:\Compile\TensorRT-10.12.0.36.Windows.win10.cuda-12.9\TensorRT-10.12.0.36`).
4. **Environment Variables**: Open Environment Variables (`rundll32 sysdm.cpl,EditEnvironmentVariables`) and append to `PATH`:
   - `C:\Program Files\NVIDIA\CUDNN\v9.10\bin\12.9`
   - `D:\Compile\TensorRT-10.12.0.36.Windows.win10.cuda-12.9\TensorRT-10.12.0.36\bin`
   - `D:\Compile\TensorRT-10.12.0.36.Windows.win10.cuda-12.9\TensorRT-10.12.0.36\lib`

### C. OpenCV with CUDA Support (Build from Source)
1. Install **CMake 3.28+** (or CMake 4.x).
2. Download and extract **OpenCV 4.11.0** source to `D:\Compile\OpenCV-4.11.0\opencv`.
3. Download and extract **opencv_contrib 4.11.0** source to `D:\Compile\OpenCV-4.11.0\opencv_contrib`.
4. Open **CMake-GUI**:
   - Source code directory: `D:/Compile/OpenCV-4.11.0/opencv`
   - Build binaries directory: `D:/Compile/OpenCV-4.11.0/build`
5. Click **Configure**, select generator **Visual Studio 17 2022**, platform **x64**, and finish configuration.
6. Configure the following CMake flags:
   - Check `WITH_CUDA` = **ON**
   - Check `BUILD_opencv_world` = **ON**
   - Set `OPENCV_EXTRA_MODULES_PATH` = `D:/Compile/OpenCV-4.11.0/opencv_contrib/modules` *(use forward slashes `/`)*
   - Set `CUDA_ARCH_BIN` to match your GPU architecture (e.g., `8.9` for RTX 40-series, `8.6` for RTX 30-series).
   - *(Optional)* Check `WITH_CUDNN` = **ON** and specify `CUDNN_LIBRARY` and `CUDNN_INCLUDE_DIR`.
   - *(Optional)* Integrate NVIDIA Video Codec SDK for hardware video decoding (`WITH_NVCUVID`).
7. Click **Generate** and confirm that CMake reports `NVIDIA CUDA: YES`.
8. Open `D:\Compile\OpenCV-4.11.0\build\OpenCV.sln` in Visual Studio 2022.
9. Build the target `CMakeTargets/INSTALL`:
   - Select **Release / x64** $\rightarrow$ Build (generates `opencv_world4110.lib` and `opencv_world4110.dll`).
   - Select **Debug / x64** $\rightarrow$ Build (generates `opencv_world4110_d.lib` and `opencv_world4110_d.dll`).
10. Verify that `D:\Compile\OpenCV-4.11.0\build\install\x64\vc17\lib\opencv_world4110.lib` exists.
11. Add `D:\Compile\OpenCV-4.11.0\build\install\x64\vc17\bin` to your system `PATH`.

---

## Model Export & Engine Generation

The C++ pipeline requires an ONNX model exported with matching batch size and native FP16 precision, compiled into a TensorRT `.engine` file.

### 1. Export PyTorch to ONNX (FP16)
Use `scripts/export_model.py` to read `config.json` parameters (batch size = 28, input tile = 640×640) and enforce strict FP16 input and output tensors:
```bash
python scripts/export_model.py --config config.json
```
Output: `yolo26n.onnx`

### 2. Build TensorRT Engine via `trtexec`
Compile the ONNX graph into a high-performance TensorRT engine:
```cmd
trtexec --onnx=yolo26n.onnx --saveEngine=yolo26n.engine --fp16
```
*(Tip: `run.bat` automatically checks and performs model export and engine compilation if `yolo26n.engine` is not yet present).*

---

## Build Instructions

### Windows (Visual Studio / MSBuild)

You can build the solution using Visual Studio 2022 or directly from the terminal via MSBuild:

```cmd
"C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe" yolo-tensorrt-meshtiling-cudadecode.sln /p:Configuration=Release /p:Platform=x64 /t:Rebuild
```

The compiled binary will be placed at:
`bin\x64\Release\yolo_detector.exe`

### Linux / CMake

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

---

## Configuration (`config.json`)

All runtime detector parameters can be tuned without recompiling:

```json
{
  "pt_model_path": "yolo26n.pt",
  "model_path": "yolo26n.engine",
  "input_resolution": {
    "width": 3840,
    "height": 2160
  },
  "model_size": {
    "width": 640,
    "height": 640
  },
  "mesh_tiling": {
    "cols": 7,
    "rows": 4
  },
  "batch_size": 28,
  "conf_threshold": 0.25,
  "nms_threshold": 0.45,
  "merge_iou_threshold": 0.05,
  "max_objects_per_tile": 100,
  "max_global_objects": 1000,
  "cpu_threads": 8,
  "classes": [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", ...
  ]
}
```

### Parameter Breakdown
| Parameter | Description |
| :--- | :--- |
| `input_resolution` | Resolution of source video frame ($W \times H$). |
| `model_size` | Dimensions of each model input tile ($640 \times 640$). |
| `mesh_tiling` | Tile grid layout ($cols \times rows$). Overlap is computed automatically to cover the full frame. |
| `batch_size` | Must match $cols \times rows$ (e.g. $7 \times 4 = 28$). |
| `conf_threshold` | Minimum confidence score to consider candidate detections. |
| `nms_threshold` | IoU threshold for intra-tile 2D GPU NMS. |
| `merge_iou_threshold` | IoU threshold for inter-tile CPU Global NMS across neighbor tile boundaries. |
| `max_objects_per_tile` | Upper bound of surviving objects per tile stored in the contiguous GPU buffer (e.g. 100). |
| `max_global_objects` | Upper bound of total final detected objects per frame (e.g. 1000). |
| `cpu_threads` | Number of parallel worker execution contexts and CUDA streams. |

---

## Usage

### Command Syntax

```cmd
yolo_detector.exe <config.json> <input_video.mp4> <output_video.mp4> [max_frames]
```

- `config.json`: Path to detection configuration.
- `input_video.mp4`: Source video path (e.g. 4K highway video).
- `output_video.mp4`: Path to save rendered video with bounding box overlays.
- `[max_frames]`: *(Optional)* Number of frames to pre-decode and benchmark. **Defaults to 1000 frames** to prevent out-of-memory errors on uncompressed 4K video.

### Example Run
```cmd
bin\x64\Release\yolo_detector.exe config.json 4k-highway-traffic.mp4 4k-highway-traffic-detect.mp4 1000
```

### Console Output
```text
Running with 8 parallel worker threads (OpenCV internal threads set to 1).
Video Resolution: 3840x2160 @ 30 FPS
Total Frames: 4509
Pre-decoding up to 1000 video frames into RAM (RAM usage est: ~23.17 GB)...
Loaded 1000/1000 frames into RAM...
Pre-decode complete! Total 1000 frames in RAM.
Warming up 8 worker contexts...
Processed Frame 50/1000 (42.5 ms/frame, 23.5 FPS) - Objects: 21
Processed Frame 100/1000 (42.5 ms/frame, 23.5 FPS) - Objects: 20
...
==================================================
All frames processed! Final flushing and finalizing video...
Total Frames: 1000
Total Elapsed Time: 42.55 s
Average Pipeline Speed: 23.50 FPS (42.55 ms/frame)
Output saved to: 4k-highway-traffic-detect.mp4
==================================================
```

---

## Architecture & Performance Highlights

1. **Static Memory Allocation (Zero `cudaMalloc` in Detection Loop)**:
   All device buffers (`d_batch_input_`, `d_output_`, `d_candidates_`, `d_valid_objects_`, `d_valid_count_`) and pinned host memory are allocated once during `detector.init()`. Zero memory allocations occur during steady-state processing.
2. **GPU Candidate Compaction via `atomicAdd`**:
   Rather than transferring huge sparse candidate grids back to CPU, `compact_valid_objects_kernel` packs only active detections into a contiguous memory block (`batch_size * max_objects_per_tile`).
3. **No GPU Synchronization Bubbles**:
   Worker threads use asynchronous CUDA memory transfers with pinned memory (`cudaMallocHost`). Memory copies do not block CPU execution, and workers overlap tiling, inference, and DtoH copy operations across independent streams.
4. **Tile-to-Tile Overlap Boundary Deduplication (`tile2tile_nms`)**:
   Global NMS evaluates IoU only across adjacent, overlapping tiles rather than computing an exhaustive $O(K^2)$ matrix across the entire 4K frame, drastically accelerating post-processing.
5. **Decoupled Decoded RAM Cache**:
   By caching uncompressed frames in RAM for benchmark runs, video decoding overhead is eliminated, revealing true engine inference, CUDA kernel, and rendering performance.

---

## License

This project is licensed under the Apache 2.0 License.
