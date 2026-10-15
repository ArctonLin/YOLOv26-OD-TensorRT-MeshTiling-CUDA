#include "cuda_kernels.cuh"
#include <device_launch_parameters.h>
#include <math_constants.h>
#include <algorithm>
#include <iostream>

#define THREADS_PER_BLOCK 256
#define MAX_CANDIDATES 512

// --- 1. Mesh Tiling Kernel ---
__global__ void mesh_tiling_kernel(const unsigned char* src_ptr, int src_step_bytes, int src_w, int src_h, int src_channels,
                                   half* d_batch_input, int batch_size, 
                                   int mesh_cols, int mesh_rows,
                                   int tile_w, int tile_h, int stride_x, int stride_y) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = batch_size * tile_w * tile_h;
    if (idx >= total_pixels) return;

    int batch_idx = idx / (tile_w * tile_h);
    int pixel_in_tile = idx % (tile_w * tile_h);
    int ty = pixel_in_tile / tile_w;
    int tx = pixel_in_tile % tile_w;

    int col = batch_idx % mesh_cols;
    int row = batch_idx / mesh_cols;
    int src_x = col * stride_x + tx;
    int src_y = row * stride_y + ty;

    uchar3 pixel = {0, 0, 0};
    if (src_x < src_w && src_y < src_h) {
        if (src_channels == 4) {
            const uchar4 p4 = ((const uchar4*)(src_ptr + src_y * src_step_bytes))[src_x];
            pixel = make_uchar3(p4.x, p4.y, p4.z);
        } else {
            pixel = ((const uchar3*)(src_ptr + src_y * src_step_bytes))[src_x];
        }
    }

    int out_c_stride = tile_w * tile_h;
    int out_batch_stride = 3 * out_c_stride;
    half* batch_base = d_batch_input + batch_idx * out_batch_stride;
    batch_base[0 * out_c_stride + pixel_in_tile] = __float2half(pixel.z / 255.0f);
    batch_base[1 * out_c_stride + pixel_in_tile] = __float2half(pixel.y / 255.0f);
    batch_base[2 * out_c_stride + pixel_in_tile] = __float2half(pixel.x / 255.0f);
}

void launch_mesh_tiling(const cv::cuda::GpuMat& src, half* d_batch_input, 
                        int batch_size, int mesh_cols, int mesh_rows,
                        int tile_w, int tile_h, int stride_x, int stride_y, 
                        cudaStream_t stream) {
    int total_pixels = batch_size * tile_w * tile_h;
    int blocks = (total_pixels + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    mesh_tiling_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
        src.ptr<unsigned char>(), (int)src.step, src.cols, src.rows, src.channels(),
        d_batch_input, batch_size, mesh_cols, mesh_rows,
        tile_w, tile_h, stride_x, stride_y
    );
}

// --- 2. Decode & Local NMS (Reference Repo Architecture) ---
__device__ inline float box_iou(float a_x1, float a_y1, float a_x2, float a_y2,
                                float b_x1, float b_y1, float b_x2, float b_y2) {
    float left = max(a_x1, b_x1);
    float right = min(a_x2, b_x2);
    float top = max(a_y1, b_y1);
    float bottom = min(a_y2, b_y2);

    if (left < right && top < bottom) {
        float intersection = (right - left) * (bottom - top);
        float area_a = (a_x2 - a_x1) * (a_y2 - a_y1);
        float area_b = (b_x2 - b_x1) * (b_y2 - b_y1);
        float union_area = area_a + area_b - intersection;
        return (union_area > 0.0f) ? (intersection / union_area) : 0.0f;
    }
    return 0.0f;
}

// 1. Decode candidates kernel: extracts candidate boxes where conf >= conf_thresh
__global__ void decode_candidates_kernel(
    const half* __restrict__ d_model_output,
    float* __restrict__ d_candidates,
    int* __restrict__ d_cand_counts,
    int batch_size, int num_classes, int num_bboxes,
    float conf_thresh,
    int stride_x, int stride_y, int mesh_cols,
    int max_candidates_per_tile
) {
    int box_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = blockIdx.y * blockDim.y + threadIdx.y;
    if (batch_idx >= batch_size || box_idx >= num_bboxes) return;

    int num_channels = 4 + num_classes;
    const half* batch_out = d_model_output + batch_idx * num_channels * num_bboxes;

    float max_conf = 0.0f;
    int max_class = -1;
    for (int c = 0; c < num_classes; ++c) {
        float conf = __half2float(batch_out[(4 + c) * num_bboxes + box_idx]);
        if (conf > max_conf) {
            max_conf = conf;
            max_class = c;
        }
    }

    if (max_conf >= conf_thresh) {
        int idx = atomicAdd(&d_cand_counts[batch_idx], 1);
        if (idx >= max_candidates_per_tile) return;

        float cx = __half2float(batch_out[0 * num_bboxes + box_idx]);
        float cy = __half2float(batch_out[1 * num_bboxes + box_idx]);
        float w = __half2float(batch_out[2 * num_bboxes + box_idx]);
        float h = __half2float(batch_out[3 * num_bboxes + box_idx]);

        int col = batch_idx % mesh_cols;
        int row = batch_idx / mesh_cols;
        float global_cx = cx + col * stride_x;
        float global_cy = cy + row * stride_y;

        int offset = (batch_idx * max_candidates_per_tile + idx) * 7;
        d_candidates[offset + 0] = global_cx - w / 2.0f; // x1
        d_candidates[offset + 1] = global_cy - h / 2.0f; // y1
        d_candidates[offset + 2] = global_cx + w / 2.0f; // x2
        d_candidates[offset + 3] = global_cy + h / 2.0f; // y2
        d_candidates[offset + 4] = max_conf;
        d_candidates[offset + 5] = (float)max_class;
        d_candidates[offset + 6] = 1.0f;                 // keep = 1.0f
    }
}

// 2. Parallel 2D NMS Kernel (Reference repo architecture):
// Compares all pairs (i, j) with j < i concurrently across 2D/3D GPU thread blocks.
__global__ void parallel_local_nms_kernel(
    float* __restrict__ d_candidates,
    const int* __restrict__ d_cand_counts,
    int batch_size,
    int max_candidates_per_tile,
    float nms_thresh
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int b = blockIdx.z;

    if (b >= batch_size || i >= d_cand_counts[b] || i >= max_candidates_per_tile || j >= i) return;

    int base_offset = b * max_candidates_per_tile * 7;
    int offset_i = base_offset + i * 7;
    int offset_j = base_offset + j * 7;

    // Check same class
    if (__float2int_rn(d_candidates[offset_i + 5]) != __float2int_rn(d_candidates[offset_j + 5])) return;

    float iou = box_iou(
        d_candidates[offset_i + 0], d_candidates[offset_i + 1],
        d_candidates[offset_i + 2], d_candidates[offset_i + 3],
        d_candidates[offset_j + 0], d_candidates[offset_j + 1],
        d_candidates[offset_j + 2], d_candidates[offset_j + 3]
    );

    if (iou >= nms_thresh) {
        if (d_candidates[offset_j + 4] >= d_candidates[offset_i + 4]) {
            d_candidates[offset_i + 6] = 0.0f; // delete i
        } else {
            d_candidates[offset_j + 6] = 0.0f; // delete j
        }
    }
}

// 3. Compact valid objects kernel (Reference repo get_final architecture):
// Packs ONLY surviving (keep == 1.0f) objects into continuous buffer using atomicAdd index per tile.
__global__ void compact_valid_objects_kernel(
    const float* __restrict__ d_candidates,
    const int* __restrict__ d_cand_counts,
    DecodeBox* __restrict__ d_decode_out,
    int* __restrict__ d_tile_counts,
    int batch_size,
    int max_candidates_per_tile,
    int max_objects_per_tile
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y * blockDim.y + threadIdx.y;

    if (b >= batch_size || i >= d_cand_counts[b] || i >= max_candidates_per_tile) return;

    int base_offset = b * max_candidates_per_tile * 7;
    if (d_candidates[base_offset + i * 7 + 6] == 1.0f) { // keep == 1.0f
        int write_idx = atomicAdd(&d_tile_counts[b], 1);
        if (write_idx < max_objects_per_tile) {
            DecodeBox* tile_out = d_decode_out + b * max_objects_per_tile;
            tile_out[write_idx].x1 = d_candidates[base_offset + i * 7 + 0];
            tile_out[write_idx].y1 = d_candidates[base_offset + i * 7 + 1];
            tile_out[write_idx].x2 = d_candidates[base_offset + i * 7 + 2];
            tile_out[write_idx].y2 = d_candidates[base_offset + i * 7 + 3];
            tile_out[write_idx].conf = d_candidates[base_offset + i * 7 + 4];
            tile_out[write_idx].class_id = __float2int_rn(d_candidates[base_offset + i * 7 + 5]);
        }
    }
}

void launch_decode_and_local_nms(const half* d_model_output,
                                 float* d_candidates, int* d_cand_counts,
                                 DecodeBox* d_decode_out, int* d_tile_counts,
                                 int batch_size, int num_classes, int num_bboxes,
                                 float conf_thresh, float nms_thresh,
                                 int stride_x, int stride_y, int mesh_cols,
                                 int max_candidates_per_tile, int max_objects_per_tile,
                                 cudaStream_t stream) {
    cudaMemsetAsync(d_cand_counts, 0, batch_size * sizeof(int), stream);
    cudaMemsetAsync(d_tile_counts, 0, batch_size * sizeof(int), stream);

    // 1. Decode candidates
    dim3 block_dec(32, 8, 1);
    dim3 grid_dec((num_bboxes + block_dec.x - 1) / block_dec.x,
                  (batch_size + block_dec.y - 1) / block_dec.y,
                  1);
    decode_candidates_kernel<<<grid_dec, block_dec, 0, stream>>>(
        d_model_output, d_candidates, d_cand_counts,
        batch_size, num_classes, num_bboxes, conf_thresh,
        stride_x, stride_y, mesh_cols, max_candidates_per_tile
    );

    // 2. Parallel 2D NMS (Reference repo architecture)
    dim3 block_nms(16, 16, 1);
    dim3 grid_nms((max_candidates_per_tile + block_nms.x - 1) / block_nms.x,
                  (max_candidates_per_tile + block_nms.y - 1) / block_nms.y,
                  batch_size);
    parallel_local_nms_kernel<<<grid_nms, block_nms, 0, stream>>>(
        d_candidates, d_cand_counts, batch_size, max_candidates_per_tile, nms_thresh
    );

    // 3. Compact valid objects into continuous tile buffer using atomicAdd
    dim3 block_comp(32, 8, 1);
    dim3 grid_comp((max_candidates_per_tile + block_comp.x - 1) / block_comp.x,
                   (batch_size + block_comp.y - 1) / block_comp.y,
                   1);
    compact_valid_objects_kernel<<<grid_comp, block_comp, 0, stream>>>(
        d_candidates, d_cand_counts, d_decode_out, d_tile_counts,
        batch_size, max_candidates_per_tile, max_objects_per_tile
    );
}
