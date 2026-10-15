#include "yolo_detector.h"
#include "cuda_kernels.cuh"
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <opencv2/core/cuda_stream_accessor.hpp>

using json = nlohmann::json;

void Logger::log(Severity severity, const char* msg) noexcept {
    if (severity <= Severity::kWARNING) {
        std::cout << "[TRT] " << msg << std::endl;
    }
}

YoloDetector::YoloDetector() {}

YoloDetector::~YoloDetector() {
    freeMemory();
    if (engine_) delete engine_;
    if (runtime_) delete runtime_;
}

void YoloDetector::tile2tile_nms(std::vector<Detection>& rs1, std::vector<Detection>& rs2, float iou_thresh) {
    for (size_t i = 0; i < rs1.size(); ++i) {
        if (!rs1[i].keep) continue;
        for (size_t j = 0; j < rs2.size(); ++j) {
            if (!rs2[j].keep || rs1[i].class_id != rs2[j].class_id) continue;

            float left = std::max(rs1[i].x1, rs2[j].x1);
            float right = std::min(rs1[i].x2, rs2[j].x2);
            float top = std::max(rs1[i].y1, rs2[j].y1);
            float bottom = std::min(rs1[i].y2, rs2[j].y2);

            if (left < right && top < bottom) {
                float intersection = (right - left) * (bottom - top);
                float area_a = (rs1[i].x2 - rs1[i].x1) * (rs1[i].y2 - rs1[i].y1);
                float area_b = (rs2[j].x2 - rs2[j].x1) * (rs2[j].y2 - rs2[j].y1);
                float union_area = area_a + area_b - intersection;
                float iou = (union_area > 0.0f) ? (intersection / union_area) : 0.0f;

                if (iou >= iou_thresh) {
                    if (rs1[i].conf >= rs2[j].conf) {
                        rs1[i].x1 = std::min(rs1[i].x1, rs2[j].x1);
                        rs1[i].y1 = std::min(rs1[i].y1, rs2[j].y1);
                        rs1[i].x2 = std::max(rs1[i].x2, rs2[j].x2);
                        rs1[i].y2 = std::max(rs1[i].y2, rs2[j].y2);
                        rs2[j].keep = false;
                    } else {
                        rs2[j].x1 = std::min(rs1[i].x1, rs2[j].x1);
                        rs2[j].y1 = std::min(rs1[i].y1, rs2[j].y1);
                        rs2[j].x2 = std::max(rs1[i].x2, rs2[j].x2);
                        rs2[j].y2 = std::max(rs1[i].y2, rs2[j].y2);
                        rs1[i].keep = false;
                        break;
                    }
                }
            }
        }
    }
}

void YoloDetector::freeMemory() {
    for (auto& worker : workers_) {
        if (worker.d_input) cudaFree(worker.d_input);
        if (worker.d_output) cudaFree(worker.d_output);
        if (worker.d_candidates) cudaFree(worker.d_candidates);
        if (worker.d_cand_counts) cudaFree(worker.d_cand_counts);
        if (worker.d_decode_buffer) cudaFree(worker.d_decode_buffer);
        if (worker.h_decode_buffer) cudaFreeHost(worker.h_decode_buffer);
        if (worker.context) delete worker.context;
        if (worker.stream) cudaStreamDestroy(worker.stream);
    }
    workers_.clear();
}

bool YoloDetector::init(const std::string& json_path) {
    // 1. Parse JSON
    std::ifstream file(json_path);
    if (!file.is_open()) {
        std::cerr << "Failed to open " << json_path << std::endl;
        return false;
    }
    json j;
    file >> j;

    config_.model_path = j["model_path"];
    config_.input_width = j["input_resolution"]["width"];
    config_.input_height = j["input_resolution"]["height"];
    config_.model_width = j["model_size"]["width"];
    config_.model_height = j["model_size"]["height"];
    config_.mesh_cols = j["mesh_tiling"]["cols"];
    config_.mesh_rows = j["mesh_tiling"]["rows"];
    config_.batch_size = j["batch_size"];
    config_.conf_threshold = j["conf_threshold"];
    config_.nms_threshold = j["nms_threshold"];
    config_.merge_iou_threshold = j["merge_iou_threshold"];
    config_.max_objects_per_tile = j.value("max_objects_per_tile", 100);
    config_.max_global_objects = j.value("max_global_objects", 1000);
    config_.cpu_threads = j.value("cpu_threads", 8);
    for (const auto& c : j["classes"]) {
        config_.classes.push_back(c.get<std::string>());
    }

    stride_x_ = (config_.input_width - config_.model_width) / (config_.mesh_cols - 1);
    stride_y_ = (config_.input_height - config_.model_height) / (config_.mesh_rows - 1);

    // 2. Load TensorRT Engine
    std::ifstream engine_file(config_.model_path, std::ios::binary);
    if (!engine_file.good()) {
        std::cerr << "Error loading TRT engine: " << config_.model_path << std::endl;
        return false;
    }
    engine_file.seekg(0, engine_file.end);
    size_t size = engine_file.tellg();
    engine_file.seekg(0, engine_file.beg);
    std::vector<char> engine_data(size);
    engine_file.read(engine_data.data(), size);
    engine_file.close();

    runtime_ = nvinfer1::createInferRuntime(logger_);
    engine_ = runtime_->deserializeCudaEngine(engine_data.data(), size);
    if (!engine_) {
        std::cerr << "Failed to deserialize TensorRT engine." << std::endl;
        return false;
    }

    // TensorRT 10+ API for dimensions
    const char* output_name = engine_->getIOTensorName(1);
    auto out_dims = engine_->getTensorShape(output_name);
    // YOLO outputs: [batch_size, num_classes + 4, num_bboxes] e.g. [28, 84, 8400]
    out_num_classes_ = static_cast<int>(out_dims.d[1] - 4);
    out_num_bboxes_ = static_cast<int>(out_dims.d[2]);

    std::cout << "Engine loaded. Input: " << engine_->getIOTensorName(0) << ", Output: " << output_name << std::endl;
    std::cout << "out_dims: " << out_dims.d[0] << "x" << out_dims.d[1] << "x" << out_dims.d[2] << std::endl;
    std::cout << "out_num_classes: " << out_num_classes_ << ", out_num_bboxes: " << out_num_bboxes_ << std::endl;

    // 3. Allocate Worker Contexts
    num_workers_ = config_.cpu_threads > 0 ? config_.cpu_threads : 1;
    workers_.resize(num_workers_);

    size_t input_size = config_.batch_size * 3 * config_.model_width * config_.model_height * sizeof(half);
    size_t output_size = config_.batch_size * out_dims.d[1] * out_dims.d[2] * sizeof(half);

    int max_candidates_per_tile = 1024;
    size_t candidates_size = config_.batch_size * max_candidates_per_tile * 7 * sizeof(float);
    size_t counts_bytes = ((config_.batch_size * sizeof(int) + 15) / 16) * 16;
    size_t boxes_bytes = config_.batch_size * config_.max_objects_per_tile * sizeof(DecodeBox);
    size_t decode_buffer_size = counts_bytes + boxes_bytes;

    std::cout << "Creating " << num_workers_ << " parallel worker contexts & CUDA streams..." << std::endl;
    std::cout << "Buffer config: max_objects_per_tile=" << config_.max_objects_per_tile 
              << " (" << (boxes_bytes / 1024.0f) << " KB), max_global_objects=" << config_.max_global_objects << std::endl;
    for (int i = 0; i < num_workers_; ++i) {
        workers_[i].context = engine_->createExecutionContext();
        if (!workers_[i].context) {
            std::cerr << "Failed to create execution context for worker " << i << std::endl;
            return false;
        }
        cudaStreamCreate(&workers_[i].stream);

        cudaMalloc(&workers_[i].d_input, input_size);
        cudaMalloc(&workers_[i].d_output, output_size);
        cudaMalloc(&workers_[i].d_candidates, candidates_size);
        cudaMalloc(&workers_[i].d_cand_counts, config_.batch_size * sizeof(int));
        cudaMalloc(&workers_[i].d_decode_buffer, decode_buffer_size);
        cudaMallocHost(&workers_[i].h_decode_buffer, decode_buffer_size);

        workers_[i].d_tile_counts = (int*)workers_[i].d_decode_buffer;
        workers_[i].d_decode_out = (DecodeBox*)((char*)workers_[i].d_decode_buffer + counts_bytes);
    }

    return true;
}

std::vector<yolo_object> YoloDetector::detect(const cv::Mat& img, int thread_id) {
    if (thread_id < 0 || thread_id >= num_workers_) thread_id = 0;
    auto& worker = workers_[thread_id];
    cv::cuda::Stream cv_stream = cv::cuda::StreamAccessor::wrapStream(worker.stream);
    worker.d_frame.upload(img, cv_stream);
    return detect(worker.d_frame, thread_id);
}

std::vector<yolo_object> YoloDetector::detect(const cv::cuda::GpuMat& img, int thread_id) {
    if (thread_id < 0 || thread_id >= num_workers_) thread_id = 0;
    auto& worker = workers_[thread_id];

    // 1. Mesh Tiling (Pre-processing)
    launch_mesh_tiling(img, (half*)worker.d_input, config_.batch_size, config_.mesh_cols, config_.mesh_rows,
                       config_.model_width, config_.model_height, stride_x_, stride_y_, worker.stream);

    // 2. TensorRT Inference (TRT 10+ API)
    const char* input_name = engine_->getIOTensorName(0);
    const char* output_name = engine_->getIOTensorName(1);
    worker.context->setTensorAddress(input_name, worker.d_input);
    worker.context->setTensorAddress(output_name, worker.d_output);

    worker.context->enqueueV3(worker.stream);

    // 3. Decode & Local NMS on GPU (Reference repo parallel 2D NMS architecture)
    int max_candidates_per_tile = 1024;
    launch_decode_and_local_nms((half*)worker.d_output,
                                worker.d_candidates, worker.d_cand_counts,
                                worker.d_decode_out, worker.d_tile_counts,
                                config_.batch_size, out_num_classes_, out_num_bboxes_,
                                config_.conf_threshold, config_.nms_threshold,
                                stride_x_, stride_y_, config_.mesh_cols,
                                max_candidates_per_tile, config_.max_objects_per_tile,
                                worker.stream);

    // 4. Single Transfer: Copy continuous tile buffer (counts + valid objects) to Host
    size_t counts_bytes = ((config_.batch_size * sizeof(int) + 15) / 16) * 16;
    size_t boxes_bytes = config_.batch_size * config_.max_objects_per_tile * sizeof(DecodeBox);
    size_t decode_buffer_size = counts_bytes + boxes_bytes;
    cudaMemcpyAsync(worker.h_decode_buffer, worker.d_decode_buffer, decode_buffer_size, cudaMemcpyDeviceToHost, worker.stream);
    cudaStreamSynchronize(worker.stream);

    // 5. Global NMS on CPU (Reference repo architecture)
    const int* h_tile_counts = (const int*)worker.h_decode_buffer;
    const DecodeBox* h_decode_boxes = (const DecodeBox*)((const char*)worker.h_decode_buffer + counts_bytes);

    std::vector<std::vector<Detection>> tile_results(config_.batch_size);
    for (int b = 0; b < config_.batch_size; ++b) {
        int cnt = std::min(h_tile_counts[b], config_.max_objects_per_tile);
        tile_results[b].reserve(cnt);
        const DecodeBox* b_boxes = h_decode_boxes + b * config_.max_objects_per_tile;
        for (int i = 0; i < cnt; ++i) {
            tile_results[b].push_back({
                b_boxes[i].x1, b_boxes[i].y1, b_boxes[i].x2, b_boxes[i].y2,
                b_boxes[i].conf, b_boxes[i].class_id, true
            });
        }
    }

    // Neighboring tiles Global NMS
    for (int bx1 = 0; bx1 < config_.mesh_cols; ++bx1) {
        int bx2s = bx1 - 1 > 0 ? bx1 - 1 : 0;
        int bx2e = std::min(config_.mesh_cols - 1, bx1 + 1);
        for (int by1 = 0; by1 < config_.mesh_rows; ++by1) {
            int by2s = by1 - 1 > 0 ? by1 - 1 : 0;
            int by2e = std::min(config_.mesh_rows - 1, by1 + 1);
            int tile_idx1 = by1 * config_.mesh_cols + bx1;

            for (int bx2 = bx2s; bx2 <= bx2e; ++bx2) {
                for (int by2 = by2s; by2 <= by2e; ++by2) {
                    int tile_idx2 = by2 * config_.mesh_cols + bx2;
                    if (tile_idx2 < tile_idx1) {
                        tile2tile_nms(tile_results[tile_idx1], tile_results[tile_idx2], config_.merge_iou_threshold);
                    }
                }
            }
        }
    }

    // 6. Collect final results up to max_global_objects
    std::vector<yolo_object> results;
    results.reserve(config_.max_global_objects);
    for (int b = 0; b < config_.batch_size; ++b) {
        for (const auto& box : tile_results[b]) {
            if (box.keep) {
                if ((int)results.size() >= config_.max_global_objects) break;
                yolo_object obj;
                obj.class_id = box.class_id;
                obj.confidence = box.conf;
                obj.bbox = cv::Rect(
                    cv::Point((int)box.x1, (int)box.y1),
                    cv::Point((int)box.x2, (int)box.y2)
                );
                results.push_back(obj);
            }
        }
        if ((int)results.size() >= config_.max_global_objects) break;
    }

    return results;
}
