#pragma once
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <NvInfer.h>

struct yolo_object {
    int class_id;
    float confidence;
    cv::Rect bbox; // Bounding box in the original high-res image coordinate system
};

#include "cuda_kernels.cuh"

struct Config {
    std::string model_path;
    int input_width;
    int input_height;
    int model_width;
    int model_height;
    int mesh_cols;
    int mesh_rows;
    int batch_size;
    float conf_threshold;
    float nms_threshold;
    float merge_iou_threshold;
    int max_objects_per_tile = 100;
    int max_global_objects = 1000;
    int cpu_threads;
    std::vector<std::string> classes;
};

class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override;
};

struct WorkerContext {
    nvinfer1::IExecutionContext* context = nullptr;
    cudaStream_t stream = nullptr;
    void* d_input = nullptr;        // Model input (Batched)
    void* d_output = nullptr;       // Model output (Batched)
    float* d_candidates = nullptr;  // Pre-NMS candidate buffer [batch_size * max_cand * 7]
    int* d_cand_counts = nullptr;   // Candidate counts per tile [batch_size]
    void* d_decode_buffer = nullptr;// Device buffer: tile_counts[batch_size] + DecodeBox array
    void* h_decode_buffer = nullptr;// Pinned Host buffer: tile_counts[batch_size] + DecodeBox array
    int* d_tile_counts = nullptr;   // Pointer inside d_decode_buffer
    DecodeBox* d_decode_out = nullptr; // Pointer inside d_decode_buffer
    cv::cuda::GpuMat d_frame;       // Worker dedicated GPU frame
};

class YoloDetector {
public:
    YoloDetector();
    ~YoloDetector();

    bool init(const std::string& json_path);
    std::vector<yolo_object> detect(const cv::Mat& img, int thread_id = 0);
    std::vector<yolo_object> detect(const cv::cuda::GpuMat& img, int thread_id = 0);
    
    const Config& getConfig() const { return config_; }
    int getNumWorkers() const { return num_workers_; }

private:
    void freeMemory();

    struct Detection {
        float x1, y1, x2, y2, conf;
        int class_id;
        bool keep = true;
    };

    static void tile2tile_nms(std::vector<Detection>& rs1, std::vector<Detection>& rs2, float iou_thresh);

    Config config_;
    Logger logger_;

    nvinfer1::IRuntime* runtime_ = nullptr;
    nvinfer1::ICudaEngine* engine_ = nullptr;

    int num_workers_ = 1;
    std::vector<WorkerContext> workers_;

    // Stride calculations for overlapping tiles
    int stride_x_ = 0;
    int stride_y_ = 0;
    
    // Model output dimensions (e.g. 84 x 8400)
    int out_num_classes_ = 0;
    int out_num_bboxes_ = 0;
};
