#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <opencv2/core/cuda.hpp>

// Structure for decoded bounding boxes (internal representation, continuous per tile)
struct DecodeBox {
    float x1;
    float y1;
    float x2;
    float y2;
    float conf;
    int class_id;
};

// Structure for final output bounding boxes
struct FinalBox {
    float x1;
    float y1;
    float x2;
    float y2;
    float conf;
    int class_id;
};

// CUDA Kernel Launchers

// 1. Tiling: extracts overlapping tiles from the high-res image into a batched TRT input buffer
void launch_mesh_tiling(const cv::cuda::GpuMat& src, half* d_batch_input, 
                        int batch_size, int mesh_cols, int mesh_rows,
                        int tile_w, int tile_h, int stride_x, int stride_y, 
                        cudaStream_t stream);

// 2. Decode & Local NMS (Reference repo GPU NMS architecture):
// - Decodes candidates in parallel.
// - Massively parallel 2D grid NMS across candidate pairs (i, j).
// - Uses atomicAdd index per tile to pack only valid surviving objects into continuous buffer (batch_size * max_objects_per_tile).
void launch_decode_and_local_nms(const half* d_model_output,
                                 float* d_candidates, int* d_cand_counts,
                                 DecodeBox* d_decode_out, int* d_tile_counts,
                                 int batch_size, int num_classes, int num_bboxes,
                                 float conf_thresh, float nms_thresh,
                                 int stride_x, int stride_y, int mesh_cols,
                                 int max_candidates_per_tile, int max_objects_per_tile,
                                 cudaStream_t stream);
