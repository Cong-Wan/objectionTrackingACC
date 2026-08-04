/*
Author: wilbur
Version: 1.0
Date: 2026-08-04
Description: trackCrop 应用入口：NVDEC 解码 → TensorRT 检测（fullBody/head/upperBody）→
    head 送 ByteTrack 追踪 → IoA 归属匹配（先 cls0 后 cls2）→ 匹配到的 body 框裁剪图按
    output/<视频名stem>/<trackId>/<%06d帧号>.jpg 保存。每 N 帧处理一帧。
*/

#include "byteTracker.h"
#include "frameTypes.h"
#include "logger.h"
#include "nv12ToBgr.h"
#include "nvdecDecoder.h"
#include "trackerConfig.h"
#include "trtInfer.h"

#include <opencv2/imgcodecs.hpp>

#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct AppArgs {
    std::string videoPath;
    std::string enginePath;
    int n = 1;
    std::string outputDir = "./output";
    float conf = 0.01f;
    float ioa = 0.5f;
    std::string trackerConfigPath = "config/tracker.yaml";
};

void printUsage(const char* prog) {
    printf("Usage: %s --video <path> --engine <path> [--N int] [--output dir] [--conf float]\n"
           "          [--ioa float] [--tracker-config path] [--help]\n"
           "  --video           本地视频路径（必填）\n"
           "  --engine          TensorRT engine 路径（必填）\n"
           "  --N               每 N 帧处理一帧，N >= 1（默认 1）\n"
           "  --output          输出根目录（默认 ./output）\n"
           "  --conf            预过滤阈值，必须 <= tracker track_low_thresh（默认 0.01）\n"
           "  --ioa             IoA 命中阈值，(0,1]（默认 0.5）\n"
           "  --tracker-config  追踪参数 yaml（默认 config/tracker.yaml）\n"
           "  --help            打印本用法\n",
           prog);
}

bool parseInt(const std::string& s, int& out) {
    try {
        size_t pos = 0;
        int v = std::stoi(s, &pos);
        if (pos != s.size()) {
            return false;
        }
        out = v;
        return true;
    } catch (...) {
        return false;
    }
}

bool parseFloat(const std::string& s, float& out) {
    try {
        size_t pos = 0;
        float v = std::stof(s, &pos);
        if (pos != s.size()) {
            return false;
        }
        out = v;
        return true;
    } catch (...) {
        return false;
    }
}

// 解析成功返回 true；help 或非法返回 false（调用方据 helpRequested 决定退出码）
bool parseArgs(int argc, char** argv, AppArgs& args, bool& helpRequested) {
    helpRequested = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char* name, std::string& out) -> bool {
            if (i + 1 >= argc) {
                logError(std::string("missing value for ") + name);
                return false;
            }
            out = argv[++i];
            return true;
        };
        if (a == "--help") {
            helpRequested = true;
            return false;
        } else if (a == "--video") {
            if (!next("--video", args.videoPath)) return false;
        } else if (a == "--engine") {
            if (!next("--engine", args.enginePath)) return false;
        } else if (a == "--output") {
            if (!next("--output", args.outputDir)) return false;
        } else if (a == "--tracker-config") {
            if (!next("--tracker-config", args.trackerConfigPath)) return false;
        } else if (a == "--N") {
            std::string v;
            if (!next("--N", v)) return false;
            if (!parseInt(v, args.n) || args.n < 1) {
                logError("--N must be an integer >= 1, got: " + v);
                return false;
            }
        } else if (a == "--conf") {
            std::string v;
            if (!next("--conf", v)) return false;
            if (!parseFloat(v, args.conf)) {
                logError("--conf must be a float, got: " + v);
                return false;
            }
        } else if (a == "--ioa") {
            std::string v;
            if (!next("--ioa", v)) return false;
            if (!parseFloat(v, args.ioa) || args.ioa <= 0.0f || args.ioa > 1.0f) {
                logError("--ioa must be in (0,1], got: " + v);
                return false;
            }
        } else {
            logError("unknown argument: " + a);
            return false;
        }
    }
    if (args.videoPath.empty() || args.enginePath.empty()) {
        logError("--video and --engine are required");
        return false;
    }
    return true;
}

// float 框 → int cv::Rect，clamp 到 [0,w]x[0,h]；宽或高 <= 0 视为无效。
// x2/y2 语义统一按 cv::Rect 闭区间处理，不纠结 1px 差异。
cv::Rect clampToFrame(float x1, float y1, float x2, float y2, int w, int h) {
    float fw = static_cast<float>(w);
    float fh = static_cast<float>(h);
    int left = static_cast<int>(std::floor(std::min(std::max(x1, 0.0f), fw)));
    int top = static_cast<int>(std::floor(std::min(std::max(y1, 0.0f), fh)));
    int right = static_cast<int>(std::ceil(std::min(std::max(x2, 0.0f), fw)));
    int bottom = static_cast<int>(std::ceil(std::min(std::max(y2, 0.0f), fh)));
    return cv::Rect(left, top, right - left, bottom - top);
}

// 返回候选中 IoA >= thresh 且最大者的 body；IoA 并列（±1e-6）取 body 面积最小者；无命中返回 nullptr
const Detection* matchBest(const cv::Rect& head, const std::vector<Detection>& bodies, float thresh,
                           int frameW, int frameH) {
    const Detection* best = nullptr;
    float bestIoa = 0.0f;
    float bestArea = FLT_MAX;
    for (auto& b : bodies) {
        cv::Rect body = clampToFrame(b.x1, b.y1, b.x2, b.y2, frameW, frameH);
        if (body.area() <= 0) continue;
        float ioa = static_cast<float>((head & body).area()) / static_cast<float>(head.area());
        if (ioa < thresh) continue;                                     // >= thresh 才算命中
        if (!best || ioa > bestIoa + 1e-6f ||
            (std::fabs(ioa - bestIoa) <= 1e-6f && static_cast<float>(body.area()) < bestArea)) {
            best = &b;
            bestIoa = ioa;
            bestArea = static_cast<float>(body.area());
        }
    }
    return best;
}

} // namespace

int main(int argc, char** argv) {
    initLogger(false);

    AppArgs args;
    bool helpRequested = false;
    if (!parseArgs(argc, argv, args, helpRequested)) {
        printUsage(argv[0]);
        return helpRequested ? 0 : 1;
    }

    try {
        // 2. 加载追踪配置
        trackerCpp::TrackerConfig cfg = trackerCpp::loadTrackerConfig(args.trackerConfigPath);

        // 3. conf 必须 <= track_low_thresh，保证 ByteTrack 低分两段匹配不被破坏
        if (args.conf > static_cast<float>(cfg.trackLowThresh)) {
            logError("--conf " + std::to_string(args.conf) +
                     " > tracker track_low_thresh " + std::to_string(cfg.trackLowThresh));
            return 1;
        }

        // 4. 打开解码器并校验基本流信息
        NvdecDecoder decoder(args.videoPath);
        decoder.open();
        if (decoder.width() <= 0 || decoder.height() <= 0 || decoder.fps() <= 0.0) {
            logError("invalid video stream info: width=" + std::to_string(decoder.width()) +
                     " height=" + std::to_string(decoder.height()) +
                     " fps=" + std::to_string(decoder.fps()));
            return 1;
        }

        // 5. 加载推理引擎
        TrtInfer infer(args.enginePath, args.conf);
        infer.load();

        // 6. 按 fps/N 设定 tracker 帧率（时间语义与抽帧对齐）
        cfg.frameRate = std::max(1, static_cast<int>(std::lround(decoder.fps() / args.n)));
        trackerCpp::BYTETracker::resetId();
        trackerCpp::BYTETracker tracker(cfg, cfg.frameRate);

        // 7. 输出目录 output/<视频名stem>
        std::filesystem::path outRoot =
            std::filesystem::path(args.outputDir) / std::filesystem::path(args.videoPath).stem();
        std::filesystem::create_directories(outRoot);

        const int frameW = decoder.width();
        const int frameH = decoder.height();
        const int64_t N = args.n;
        int64_t processed = 0;
        auto startTime = std::chrono::steady_clock::now();

        while (true) {
            std::unique_ptr<Frame> frame;
            if (!decoder.readFrame(frame)) {
                break;  // EOF 已排空
            }
            if (frame->frameId % N != 0) {
                continue;  // 跳过帧：不参与推理与追踪
            }
            ++processed;

            std::vector<Detection> dets = infer.infer(*frame);  // 已 clamp 到显示宽高

            // 1) 分流
            std::vector<Detection> heads, bodies0, bodies2;
            for (auto& d : dets) {
                if (d.classId == 1) {
                    heads.push_back(d);
                } else if (d.classId == 0) {
                    bodies0.push_back(d);
                } else if (d.classId == 2) {
                    bodies2.push_back(d);
                }
            }

            // 2) heads → ByteTrack（xyxy 转 center-xywh）；空检测帧也必须 update，保证帧计数推进
            trackerCpp::DetectionResults input;
            input.xywh.resize(static_cast<Eigen::Index>(heads.size()), 4);
            input.conf.resize(static_cast<Eigen::Index>(heads.size()));
            input.cls.resize(static_cast<Eigen::Index>(heads.size()));
            for (size_t i = 0; i < heads.size(); ++i) {
                const Detection& d = heads[i];
                double cx = (d.x1 + d.x2) / 2.0;
                double cy = (d.y1 + d.y2) / 2.0;
                double w = d.x2 - d.x1;
                double h = d.y2 - d.y1;
                input.xywh.row(static_cast<Eigen::Index>(i)) << cx, cy, w, h;
                input.conf(static_cast<Eigen::Index>(i)) = d.confidence;
                input.cls(static_cast<Eigen::Index>(i)) = 1.0;
            }
            Eigen::MatrixXf tracks = tracker.update(input);  // 每行 [x1,y1,x2,y2,trackId,score,cls,idx]

            if (tracks.rows() == 0) {
                continue;
            }

            // 3) 本帧有 track：NV12→BGR 只下载一次（所有命中共用）
            GpuFrame* gpuFrame = static_cast<GpuFrame*>(frame.get());  // 本链路必为 GpuFrame
            cv::Mat bgr;
            bool downloaded = false;

            // 4) 逐 track 匹配与保存
            for (int r = 0; r < tracks.rows(); ++r) {
                cv::Rect head = clampToFrame(tracks(r, 0), tracks(r, 1), tracks(r, 2), tracks(r, 3),
                                             frameW, frameH);
                if (head.area() <= 0) {
                    continue;
                }

                const Detection* hit = matchBest(head, bodies0, args.ioa, frameW, frameH);
                if (!hit) {
                    hit = matchBest(head, bodies2, args.ioa, frameW, frameH);  // 回退 upperBody
                }
                if (!hit) {
                    continue;  // 本帧该 track 不保存（正常分支）
                }

                if (!downloaded) {
                    bgr = nv12ToBgr(*gpuFrame);
                    downloaded = true;
                }

                cv::Rect crop = clampToFrame(hit->x1, hit->y1, hit->x2, hit->y2, frameW, frameH);
                if (crop.area() <= 0) {
                    continue;
                }

                int64_t trackId = static_cast<int64_t>(tracks(r, 4));  // float 存储，转回整型
                std::filesystem::path dir = outRoot / std::to_string(trackId);
                std::error_code ec;
                std::filesystem::create_directories(dir, ec);
                if (ec) {
                    logWarn("create_directories failed: " + dir.string() + " : " + ec.message());
                    continue;
                }
                char name[64];
                snprintf(name, sizeof(name), "%06d.jpg", static_cast<int>(frame->frameId));
                std::filesystem::path outPath = dir / name;
                if (!cv::imwrite(outPath.string(), bgr(crop))) {
                    logWarn("imwrite failed: " + outPath.string());
                }
            }
        }

        auto elapsed = std::chrono::steady_clock::now() - startTime;
        double seconds = std::chrono::duration<double>(elapsed).count();
        double avgFps = seconds > 0.0 ? processed / seconds : 0.0;
        logInfo("done: processed=" + std::to_string(processed) + " frames, elapsed=" +
                std::to_string(seconds) + " s, avg_fps=" + std::to_string(avgFps));
        return 0;
    } catch (const std::exception& e) {
        logError(std::string("fatal: ") + e.what());
        return 1;
    }
}
