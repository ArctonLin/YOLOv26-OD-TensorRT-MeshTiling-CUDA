#include "yolo_detector.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <random>
#include <iomanip>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <unordered_map>
#include <atomic>

template <typename T>
class BoundedQueue {
public:
    explicit BoundedQueue(size_t max_size) : max_size_(max_size), stopped_(false) {}

    void push(T item) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_not_full_.wait(lock, [this]() { return stopped_ || queue_.size() < max_size_; });
        if (stopped_) return;
        queue_.push(std::move(item));
        cv_not_empty_.notify_one();
    }

    bool pop(T& item) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_not_empty_.wait(lock, [this]() { return stopped_ || !queue_.empty(); });
        if (queue_.empty()) return false;
        item = std::move(queue_.front());
        queue_.pop();
        cv_not_full_.notify_one();
        return true;
    }

    void stop() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopped_ = true;
        }
        cv_not_empty_.notify_all();
        cv_not_full_.notify_all();
    }

    bool isStopped() {
        std::lock_guard<std::mutex> lock(mutex_);
        return stopped_;
    }

private:
    size_t max_size_;
    bool stopped_;
    std::queue<T> queue_;
    std::mutex mutex_;
    std::condition_variable cv_not_empty_;
    std::condition_variable cv_not_full_;
};

struct FrameTask {
    int frame_idx;
    cv::Mat frame;
};

struct ProcessedFrame {
    int frame_idx;
    cv::Mat frame;
    size_t num_objects;
    float process_ms;
};

class OrderedFrameCollector {
public:
    explicit OrderedFrameCollector(size_t max_pending) : max_pending_(max_pending), stopped_(false) {}

    void push(ProcessedFrame frame) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_not_full_.wait(lock, [this]() { return stopped_ || pending_.size() < max_pending_; });
        if (stopped_) return;
        pending_[frame.frame_idx] = std::move(frame);
        cv_has_next_.notify_all();
    }

    bool get(int expected_idx, ProcessedFrame& out) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_has_next_.wait(lock, [this, expected_idx]() {
            return stopped_ || pending_.find(expected_idx) != pending_.end();
        });
        auto it = pending_.find(expected_idx);
        if (it != pending_.end()) {
            out = std::move(it->second);
            pending_.erase(it);
            cv_not_full_.notify_one();
            return true;
        }
        return false;
    }

    void stop() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopped_ = true;
        }
        cv_has_next_.notify_all();
        cv_not_full_.notify_all();
    }

private:
    size_t max_pending_;
    bool stopped_;
    std::unordered_map<int, ProcessedFrame> pending_;
    std::mutex mutex_;
    std::condition_variable cv_has_next_;
    std::condition_variable cv_not_full_;
};

#include <csignal>

std::atomic<bool> g_stop{false};
void signal_handler(int signum) {
    std::cout << "\nReceived interrupt signal (" << signum << "), cleanly finishing and flushing video..." << std::endl;
    g_stop.store(true);
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <config.json> <input_video.mp4> <output_video.mp4> [max_frames (default: 1000)]" << std::endl;
        return -1;
    }

    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    std::string config_path = argv[1];
    std::string input_path = argv[2];
    std::string output_path = argv[3];
    int max_frames = (argc >= 5) ? std::stoi(argv[4]) : 1000;
    if (max_frames <= 0) {
        max_frames = 1000;
    }

    YoloDetector detector;
    if (!detector.init(config_path)) {
        std::cerr << "Failed to initialize detector." << std::endl;
        return -1;
    }

    const Config& config = detector.getConfig();
    int num_workers = detector.getNumWorkers();
    cv::setNumThreads(1); // Avoid thread oversubscription since we use application-level worker threads
    std::cout << "Running with " << num_workers << " parallel worker threads (OpenCV internal threads set to 1)." << std::endl;

    // Generate random colors for each class
    std::vector<cv::Scalar> colors;
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, 255);
    for (size_t i = 0; i < config.classes.size(); ++i) {
        colors.push_back(cv::Scalar(dist(rng), dist(rng), dist(rng)));
    }
    if (colors.empty()) {
        colors.push_back(cv::Scalar(0, 255, 0));
    }

    cv::VideoCapture cap(input_path);
    if (!cap.isOpened()) {
        std::cerr << "Failed to open video: " << input_path << std::endl;
        return -1;
    }

    int width = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    double fps = cap.get(cv::CAP_PROP_FPS);
    int total_frames = (int)cap.get(cv::CAP_PROP_FRAME_COUNT);

    std::cout << "Video Resolution: " << width << "x" << height << " @ " << fps << " FPS" << std::endl;
    std::cout << "Total Frames: " << total_frames << std::endl;

    cv::VideoWriter writer(output_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(width, height));
    if (!writer.isOpened()) {
        std::cerr << "Failed to open output video for writing." << std::endl;
        return -1;
    }

    // 1. Pre-decode video frames into RAM to eliminate video decoding bottleneck from FPS calculation
    // Limit to max_frames (default 1000) to prevent RAM exhaustion on large 4K videos
    const int DEFAULT_MAX_PRELOAD = 1000;
    int target_frames = (max_frames > 0) ? max_frames : DEFAULT_MAX_PRELOAD;
    if (total_frames > 0 && total_frames < target_frames) {
        target_frames = total_frames;
    }

    double estimated_gb = (target_frames * (double)width * height * 3.0) / (1024.0 * 1024.0 * 1024.0);
    std::cout << "Pre-decoding up to " << target_frames << " video frames into RAM (RAM usage est: ~" 
              << std::fixed << std::setprecision(2) << estimated_gb << " GB)..." << std::endl;
    std::vector<cv::Mat> preloaded_frames;
    preloaded_frames.reserve(target_frames);

    int frame_count = 0;
    while (!g_stop.load()) {
        if (target_frames > 0 && frame_count >= target_frames) break;
        cv::Mat frame;
        if (!cap.read(frame) || frame.empty()) break;
        preloaded_frames.push_back(std::move(frame));
        frame_count++;
        if (frame_count == 1 || frame_count % 50 == 0 || frame_count == target_frames) {
            std::cout << "\rLoaded " << frame_count << "/" << target_frames << " frames into RAM..." << std::flush;
        }
    }
    std::cout << "\nPre-decode complete! Total " << preloaded_frames.size() << " frames in RAM." << std::endl;
    cap.release();

    if (preloaded_frames.empty()) {
        std::cerr << "No frames loaded from video." << std::endl;
        return -1;
    }

    int total_loaded = (int)preloaded_frames.size();

    // Warmup all workers using the first preloaded frame
    std::cout << "Warming up " << num_workers << " worker contexts..." << std::endl;
    for (int w = 0; w < num_workers; ++w) {
        detector.detect(preloaded_frames[0], w);
    }

    BoundedQueue<FrameTask> input_queue(num_workers * 2);
    OrderedFrameCollector output_collector(num_workers * 4);
    std::atomic<int> total_read{total_loaded};

    auto start_time = std::chrono::high_resolution_clock::now();
    auto last_time = start_time;
    int last_idx = 0;

    // 2. Start Feeder Thread (zero-copy memory handoff from RAM)
    std::thread feeder_thread([&]() {
        for (int idx = 0; idx < total_loaded && !g_stop.load(); ++idx) {
            input_queue.push({idx, std::move(preloaded_frames[idx])});
        }
        input_queue.stop();
    });

    // 3. Start Worker Threads
    std::vector<std::thread> workers;
    for (int w = 0; w < num_workers; ++w) {
        workers.emplace_back([&, w]() {
            FrameTask task;
            while (input_queue.pop(task)) {
                auto f_start = std::chrono::high_resolution_clock::now();

                // Detection on worker's dedicated TRT context & CUDA stream
                std::vector<yolo_object> results = detector.detect(task.frame, w);

                // Draw bounding boxes on CPU in parallel
                for (const auto& obj : results) {
                    cv::Scalar color = colors[obj.class_id % colors.size()];
                    std::string class_name = "Unknown";
                    if (obj.class_id >= 0 && obj.class_id < (int)config.classes.size()) {
                        class_name = config.classes[obj.class_id];
                    }

                    cv::rectangle(task.frame, obj.bbox, color, 2);

                    std::stringstream ss;
                    ss << class_name << " " << std::fixed << std::setprecision(2) << obj.confidence;
                    std::string label = ss.str();

                    int baseline = 0;
                    cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
                    cv::Rect text_bg(obj.bbox.x, std::max(0, obj.bbox.y - text_size.height - 5), text_size.width, text_size.height + 5);
                    cv::rectangle(task.frame, text_bg, color, cv::FILLED);
                    cv::putText(task.frame, label, cv::Point(obj.bbox.x, std::max(text_size.height, obj.bbox.y - 5)),
                                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
                }

                std::string count_text = "Objects Detected: " + std::to_string(results.size());
                cv::putText(task.frame, count_text, cv::Point(30, 50), cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(0, 0, 255), 3);

                auto f_end = std::chrono::high_resolution_clock::now();
                float ms = (float)std::chrono::duration_cast<std::chrono::milliseconds>(f_end - f_start).count();

                output_collector.push({task.frame_idx, std::move(task.frame), results.size(), ms});
            }
        });
    }

    // 4. Writer loop (in main thread) writes frames in strict sequential order
    int write_idx = 0;
    ProcessedFrame pf;
    while (true) {
        int read_count = total_read.load();
        if (input_queue.isStopped() && read_count >= 0 && write_idx >= read_count) {
            break;
        }

        if (output_collector.get(write_idx, pf)) {
            writer.write(pf.frame);
            write_idx++;

            if (write_idx == 1 || write_idx % 20 == 0 || write_idx == total_loaded) {
                auto now = std::chrono::high_resolution_clock::now();
                float total_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count() / 1000.0f;
                float step_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_time).count() / 1000.0f;
                float rolling_fps = (step_elapsed > 0.0f && write_idx > last_idx) ? (write_idx - last_idx) / step_elapsed : 0.0f;
                float avg_fps = total_elapsed > 0.0f ? write_idx / total_elapsed : 0.0f;

                std::cout << "Processed Frame " << write_idx << "/" << total_loaded
                          << " (Latency: " << pf.process_ms << " ms/frame, "
                          << "Current: " << std::fixed << std::setprecision(1) << rolling_fps << " FPS, "
                          << "Avg: " << avg_fps << " FPS) - Objects: " << pf.num_objects << std::endl;
                last_time = now;
                last_idx = write_idx;
            }
        } else {
            break;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    float total_seconds = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() / 1000.0f;

    feeder_thread.join();
    for (auto& t : workers) {
        t.join();
    }
    output_collector.stop();

    std::cout << "\nProcessing Complete!" << std::endl;
    std::cout << "Total Time: " << total_seconds << " s" << std::endl;
    std::cout << "Average FPS: " << (total_seconds > 0.0f ? write_idx / total_seconds : 0.0f) << std::endl;
    std::cout << "Output saved to: " << output_path << std::endl;

    writer.release();

    return 0;
}
