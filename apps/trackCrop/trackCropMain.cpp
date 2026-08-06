/*
Author: wilbur
Version: 1.8
Date: 2026-08-06
Description: trackCrop 应用入口：NVDEC 解码 -> TensorRT 检测（fullBody/head/upperBody）->
    head 送 ByteTrack 追踪 -> IoA 归属匹配（先 cls0 后 cls2）-> 匹配到的 body 框裁剪图按
    output/<视频名stem>/<trackId>/<%06d_cls类别>.png 无损保存。每 N 帧处理一帧。
    1.1: 新增 --debug（逐帧各环节耗时+检测匹配明细）；输出改 PNG 无损；trackId 目录拼视频名前缀、
    图片名由帧号+类别构成。
    1.2: --debug 增强：逐帧打印各环节耗时（decode/infer/track/save）、全部 YOLO 检测明细（cls/conf/xyxy）、
    追踪输出行明细，并接入 BYTETracker 的 debugCallback 输出三轮匹配代价与匈牙利耗时。
    1.3: 新增 --min-size（默认 70）：裁剪框宽或高任一小于该值不保存，0 关闭过滤。
    1.4: trackId 子目录不再拼接视频名前缀，改为纯 trackId：output/<视频名stem>/<trackId>/xxx.png。
    1.5: --debug 大幅增强：（1）IoA 匹配过程逐候选明细（cls0/cls2 各候选 origIdx/conf/box/ioa、
    未命中原因行、双侧最高 ioa）；（2）debug 模式额外输出 NVENC 硬编码可视化视频
    output/<stem>/<stem>_debug.mp4，画面含各类检测框/det#N、track 框/id、归属连线/ioa、
    NO-MATCH/SMALL 标注与左上角信息面板。绘制在 bgr.clone() 副本上，裁剪 PNG 不受污染。
    1.6: 保存前增加 body/head 高度比过滤：cls2(半身) bodyH>1.5*headH，cls0(全身) bodyH>3*headH，不满足不保存。
    1.7: 构造 BYTETracker 传入跳帧 N；update 前注入 vf；debug 增加 health 行、lifecycle CSV、tracker stats。
    1.8: 修复 debug 视频左上角面板：行高误用 getFontScaleFromHeight 导致文字上半截被裁；
    改为 getTextSize 计算行高/baseline，长 match 行自动折行。
*/

#include "byteTracker.h"
#include "frameTypes.h"
#include "logger.h"
#include "nv12ToBgr.h"
#include "nvdecDecoder.h"
#include "nvencEncoder.h"
#include "trackerConfig.h"
#include "trtInfer.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
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
    int minSize = 70;
    bool debug = false;
};

void printUsage(const char* prog) {
    printf("Usage: %s --video <path> --engine <path> [--N int] [--output dir] [--conf float]\n"
           "          [--ioa float] [--tracker-config path] [--min-size int] [--debug] [--help]\n"
           "  --video           本地视频路径（必填）\n"
           "  --engine          TensorRT engine 路径（必填）\n"
           "  --N               每 N 帧处理一帧，N >= 1（默认 1）\n"
           "  --output          输出根目录（默认 ./output）\n"
           "  --conf            预过滤阈值，必须 <= tracker track_low_thresh（默认 0.01）\n"
           "  --ioa             IoA 命中阈值，(0,1]（默认 0.5）\n"
           "  --tracker-config  追踪参数 yaml（默认 config/tracker.yaml）\n"
           "  --min-size        裁剪框宽/高任一小于该值不保存，0 关闭过滤（默认 70）\n"
           "  --debug           打印逐帧各环节耗时与检测/匹配明细，便于排查追踪检测问题\n"
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
        } else if (a == "--min-size") {
            std::string v;
            if (!next("--min-size", v)) return false;
            if (!parseInt(v, args.minSize) || args.minSize < 0) {
                logError("--min-size must be an integer >= 0, got: " + v);
                return false;
            }
        } else if (a == "--debug") {
            args.debug = true;
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

// float 框 -> int cv::Rect，clamp 到 [0,w]x[0,h]；宽或高 <= 0 视为无效。
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

// 带原始下标的检测项（origIdx 对应当帧全量 dets 列表序号，与日志 det#N 对齐）
struct IdxDet {
    size_t origIdx;
    Detection det;
};

// 单个候选的 IoA 计算明细，供 debug 打印
struct MatchCand {
    size_t origIdx;     // 全量 dets 列表下标
    int    candIdx;     // cls0/cls2 子列表下标
    float  conf;
    float  x1, y1, x2, y2;
    float  ioa;         // clamp 后零面积的候选记 -1
    bool   hit;         // ioa >= thresh（零面积候选 false）
    bool   chosen;      // 最终选中
    bool   invalid;     // clamp 后零面积（不参与计算）
};

// 返回候选中 IoA >= thresh 且最大者的 body；IoA 并列（±1e-6）取 body 面积最小者；无命中返回 nullptr。
// outIoa 非空时回写命中 IoA 值；outCands 非空时回写全部候选明细（含未命中/无效），供 debug 打印。
const Detection* matchBest(const cv::Rect& head, const std::vector<IdxDet>& bodies, float thresh,
                           int frameW, int frameH, float* outIoa = nullptr,
                           std::vector<MatchCand>* outCands = nullptr) {
    const Detection* best = nullptr;
    float bestIoa = 0.0f;
    float bestArea = FLT_MAX;
    for (size_t i = 0; i < bodies.size(); ++i) {
        const Detection& b = bodies[i].det;
        cv::Rect body = clampToFrame(b.x1, b.y1, b.x2, b.y2, frameW, frameH);
        MatchCand c{};
        c.origIdx = bodies[i].origIdx;
        c.candIdx = static_cast<int>(i);
        c.conf = b.confidence;
        c.x1 = b.x1; c.y1 = b.y1; c.x2 = b.x2; c.y2 = b.y2;
        if (body.area() <= 0) {
            c.ioa = -1.0f;
            c.hit = false;
            c.invalid = true;
            if (outCands) outCands->push_back(c);
            continue;
        }
        float ioa = static_cast<float>((head & body).area()) / static_cast<float>(head.area());
        c.ioa = ioa;
        c.hit = (ioa >= thresh);
        c.invalid = false;
        if (ioa < thresh) {
            if (outCands) outCands->push_back(c);
            continue;                                     // >= thresh 才算命中
        }
        if (!best || ioa > bestIoa + 1e-6f ||
            (std::fabs(ioa - bestIoa) <= 1e-6f && static_cast<float>(body.area()) < bestArea)) {
            best = &b;
            bestIoa = ioa;
            bestArea = static_cast<float>(body.area());
        }
        if (outCands) outCands->push_back(c);
    }
    // 标记选中项
    if (best && outCands) {
        for (auto& c : *outCands) {
            if (c.origIdx == static_cast<size_t>(-1)) continue;
            if (&bodies[c.candIdx].det == best) {
                c.chosen = true;
                break;
            }
        }
    }
    if (outIoa) {
        *outIoa = bestIoa;
    }
    return best;
}

// 打印某类候选的逐条 IoA 明细（debug 用）
void logMatchCandidates(int64_t frameId, int64_t trackId, const char* clsTag,
                        const std::vector<MatchCand>& cands, float thresh) {
    logDebug("frame " + std::to_string(frameId) + " track " + std::to_string(trackId) +
             "  " + std::string(clsTag) + " 候选=" + std::to_string(cands.size()) +
             " (thresh=" + std::to_string(thresh) + "):");
    float topIoa = -1.0f;
    int topCand = -1;
    size_t topOrig = 0;
    for (auto& c : cands) {
        char buf[256];
        if (c.invalid) {
            snprintf(buf, sizeof(buf), "    %s[%d] det#%zu conf=%.4f box=(%.1f,%.1f,%.1f,%.1f) skipped(invalid box)",
                     clsTag, c.candIdx, c.origIdx, c.conf, c.x1, c.y1, c.x2, c.y2);
        } else {
            snprintf(buf, sizeof(buf), "    %s[%d] det#%zu conf=%.4f box=(%.1f,%.1f,%.1f,%.1f) ioa=%.4f %s%s",
                     clsTag, c.candIdx, c.origIdx, c.conf, c.x1, c.y1, c.x2, c.y2, c.ioa,
                     c.hit ? "HIT" : "miss", c.chosen ? " <= 选中" : "");
            if (c.ioa > topIoa) {
                topIoa = c.ioa;
                topCand = c.candIdx;
                topOrig = c.origIdx;
            }
        }
        logDebug("frame " + std::to_string(frameId) + buf);
    }
    // 原因行：未命中时打印最高 ioa
    bool anyHit = false;
    for (auto& c : cands) {
        if (c.hit) { anyHit = true; break; }
    }
    if (!anyHit && topCand >= 0) {
        char buf[128];
        snprintf(buf, sizeof(buf), "  %s 无命中（最高 ioa=%.4f @ %s[%d] det#%zu，< thresh %.2f）",
                 clsTag, topIoa, clsTag, topCand, topOrig, thresh);
        logDebug("frame " + std::to_string(frameId) + " track " + std::to_string(trackId) + buf);
    }
}

// ===== debug 视频绘制 =====
// BGR 顏色（OpenCV 默认 BGR）
const cv::Scalar kColorCls0(0, 255, 0);      // fullBody 绿
const cv::Scalar kColorCls1(0, 255, 255);   // head 黄
const cv::Scalar kColorCls2(255, 255, 0);   // upperBody 青
const cv::Scalar kColorTrack(255, 0, 255);   // track 品红
const cv::Scalar kColorNoMatch(0, 0, 255);   // 未命中 红
const cv::Scalar kColorSmall(0, 165, 255);   // min-size 过滤 橙
const cv::Scalar kColorPanel(0, 0, 0);      // 面板黑底
const cv::Scalar kColorText(255, 255, 255);  // 面板白字

void drawText(cv::Mat& img, const std::string& text, cv::Point org, double scale = 0.5,
              int thickness = 1) {
    // org.y 为文字 baseline；调用方需保证 baseline 下方有足够空间，避免上半截被裁出画面
    cv::putText(img, text, org, cv::FONT_HERSHEY_SIMPLEX, scale, kColorText, thickness, cv::LINE_AA);
}

// 将超长行按像素宽度粗略折行（按空格优先，否则硬切）
std::vector<std::string> wrapPanelLine(const std::string& line, double scale, int thickness,
                                       int maxTextW) {
    std::vector<std::string> out;
    if (line.empty()) {
        out.push_back("");
        return out;
    }
    int baseline = 0;
    cv::Size full = cv::getTextSize(line, cv::FONT_HERSHEY_SIMPLEX, scale, thickness, &baseline);
    if (full.width <= maxTextW) {
        out.push_back(line);
        return out;
    }
    std::string cur;
    size_t i = 0;
    while (i < line.size()) {
        size_t sp = line.find(' ', i);
        std::string token = (sp == std::string::npos) ? line.substr(i) : line.substr(i, sp - i + 1);
        if (token.empty()) {
            i = (sp == std::string::npos) ? line.size() : sp + 1;
            continue;
        }
        std::string trial = cur + token;
        int b = 0;
        cv::Size sz = cv::getTextSize(trial, cv::FONT_HERSHEY_SIMPLEX, scale, thickness, &b);
        if (!cur.empty() && sz.width > maxTextW) {
            out.push_back(cur);
            cur = token;
            // 单 token 仍超宽则硬切
            while (true) {
                int bb = 0;
                cv::Size tsz = cv::getTextSize(cur, cv::FONT_HERSHEY_SIMPLEX, scale, thickness, &bb);
                if (tsz.width <= maxTextW || cur.size() <= 1) break;
                // 二分缩到宽度内
                size_t lo = 1, hi = cur.size();
                while (lo + 1 < hi) {
                    size_t mid = (lo + hi) / 2;
                    int b2 = 0;
                    cv::Size msz = cv::getTextSize(cur.substr(0, mid), cv::FONT_HERSHEY_SIMPLEX,
                                                   scale, thickness, &b2);
                    if (msz.width <= maxTextW) lo = mid;
                    else hi = mid;
                }
                out.push_back(cur.substr(0, lo));
                cur = cur.substr(lo);
            }
        } else {
            cur = trial;
        }
        i = (sp == std::string::npos) ? line.size() : sp + 1;
    }
    if (!cur.empty()) out.push_back(cur);
    if (out.empty()) out.push_back(line);
    return out;
}

void drawPanel(cv::Mat& img, const std::vector<std::string>& lines) {
    // 修复：旧实现把 getFontScaleFromHeight 的返回值当像素行高（实际是 scale），
    // 导致 lineH≈0、面板过矮，putText baseline 把字形上半截裁到面板外。
    const int pad = 8;
    const double scale = 0.55;
    const int thickness = 1;
    const int maxPanelW = std::max(320, img.cols - 20);  // 留边，避免贴死右边界
    const int maxTextW = std::max(80, maxPanelW - pad * 2);

    std::vector<std::string> wrapped;
    wrapped.reserve(lines.size() * 2);
    for (const auto& l : lines) {
        auto parts = wrapPanelLine(l, scale, thickness, maxTextW);
        wrapped.insert(wrapped.end(), parts.begin(), parts.end());
    }
    if (wrapped.empty()) return;

    int maxW = 0;
    int textH = 0;
    int baseline = 0;
    for (const auto& l : wrapped) {
        int b = 0;
        cv::Size sz = cv::getTextSize(l.empty() ? "A" : l, cv::FONT_HERSHEY_SIMPLEX, scale,
                                      thickness, &b);
        if (sz.width > maxW) maxW = sz.width;
        if (sz.height > textH) textH = sz.height;
        if (b > baseline) baseline = b;
    }
    // 行距 = 字形高度 + baseline 下降 + 行间距
    const int lineGap = 4;
    const int lineH = textH + baseline + lineGap;
    int panelW = std::min(maxPanelW, maxW + pad * 2);
    int panelH = lineH * static_cast<int>(wrapped.size()) + pad * 2 - lineGap;
    panelW = std::min(panelW, img.cols);
    panelH = std::min(panelH, img.rows);

    cv::rectangle(img, cv::Rect(0, 0, panelW, panelH), kColorPanel, cv::FILLED);
    // 半透明感：再画一层深灰边框
    cv::rectangle(img, cv::Rect(0, 0, panelW, panelH), cv::Scalar(80, 80, 80), 1);

    for (size_t i = 0; i < wrapped.size(); ++i) {
        // putText 的 y 是 baseline：第一行 baseline = pad + textH
        const int y = pad + textH + static_cast<int>(i) * lineH;
        if (y >= panelH) break;
        drawText(img, wrapped[i], cv::Point(pad, y), scale, thickness);
    }
}

// 每个 track 的匹配结果，供绘制与汇总日志复用
struct TrackMatchInfo {
    int64_t trackId;
    float score;
    cv::Rect headRect;
    const Detection* hit;       // 命中的 body 检测（无则 nullptr）
    float ioa;
    enum class Status { kMatched, kNoMatch, kSmall } status;
    cv::Rect cropRect;
};

// 绘制一帧 debug 画面：检测框 + track 框 + 归属连线 + 面板
void drawDebugFrame(cv::Mat& img, int64_t frameId,
                    const std::vector<IdxDet>& heads, const std::vector<IdxDet>& bodies0,
                    const std::vector<IdxDet>& bodies2,
                    const std::vector<TrackMatchInfo>& trackInfos,
                    int savedCount, double decodeMs, double inferMs, double trackMs,
                    double saveMs, double encMs) {
    auto labelDet = [&](const std::vector<IdxDet>& dets, const cv::Scalar& color,
                        const char* tag) {
        for (auto& d : dets) {
            cv::Rect r = clampToFrame(d.det.x1, d.det.y1, d.det.x2, d.det.y2, img.cols, img.rows);
            if (r.area() <= 0) continue;
            cv::rectangle(img, r, color, 1);
            char buf[48];
            snprintf(buf, sizeof(buf), "%s#%zu %s %.2f", tag, d.origIdx,
                     d.det.classId == 0 ? "c0" : (d.det.classId == 1 ? "c1" : "c2"), d.det.confidence);
            drawText(img, buf, cv::Point(r.x, std::max(r.y - 3, 0)));
        }
    };
    labelDet(bodies0, kColorCls0, "det");
    labelDet(heads, kColorCls1, "det");
    labelDet(bodies2, kColorCls2, "det");

    for (auto& ti : trackInfos) {
        cv::Rect hr = ti.headRect;
        const cv::Scalar* color = &kColorTrack;
        const char* tag = nullptr;
        if (ti.status == TrackMatchInfo::Status::kNoMatch) {
            color = &kColorNoMatch;
            tag = "NO-MATCH";
        } else if (ti.status == TrackMatchInfo::Status::kSmall) {
            color = &kColorSmall;
            char buf[48];
            snprintf(buf, sizeof(buf), "SMALL %dx%d", ti.cropRect.width, ti.cropRect.height);
            // 用临时字符串
            cv::rectangle(img, hr, *color, 2);
            cv::rectangle(img, ti.cropRect, *color, 1);
            drawText(img, std::string("id=") + std::to_string(ti.trackId) + " " + buf,
                     cv::Point(hr.x, std::max(hr.y - 3, 0)));
            continue;
        }
        cv::rectangle(img, hr, *color, 2);
        char buf[48];
        snprintf(buf, sizeof(buf), "id=%lld %.2f", static_cast<long long>(ti.trackId), ti.score);
        drawText(img, buf, cv::Point(hr.x, std::max(hr.y - 3, 0)));
        if (ti.status == TrackMatchInfo::Status::kMatched && ti.hit) {
            cv::Rect br = clampToFrame(ti.hit->x1, ti.hit->y1, ti.hit->x2, ti.hit->y2,
                                      img.cols, img.rows);
            cv::rectangle(img, br, *color, 2);
            char buf2[48];
            snprintf(buf2, sizeof(buf2), "id=%lld<-%s", static_cast<long long>(ti.trackId),
                     ti.hit->classId == 0 ? "c0" : "c2");
            drawText(img, buf2, cv::Point(br.x, std::min(br.y + br.height + 12, img.rows - 2)));
            cv::Point hc(hr.x + hr.width / 2, hr.y + hr.height / 2);
            cv::Point bc(br.x + br.width / 2, br.y + br.height / 2);
            cv::line(img, hc, bc, *color, 1);
            char buf3[32];
            snprintf(buf3, sizeof(buf3), "ioa=%.2f", ti.ioa);
            drawText(img, buf3, cv::Point((hc.x + bc.x) / 2 - 15, (hc.y + bc.y) / 2));
        }
        if (ti.status == TrackMatchInfo::Status::kNoMatch) {
            char buf[48];
            snprintf(buf, sizeof(buf), "id=%lld %s", static_cast<long long>(ti.trackId), tag);
            drawText(img, buf, cv::Point(hr.x, std::max(hr.y - 3, 0)));
        }
    }

    // 信息面板
    std::string matchLine = "match:";
    for (auto& ti : trackInfos) {
        if (ti.status == TrackMatchInfo::Status::kMatched) {
            char buf[32];
            snprintf(buf, sizeof(buf), " id=%lld<-%s(%.2f)", static_cast<long long>(ti.trackId),
                     ti.hit->classId == 0 ? "c0" : "c2", ti.ioa);
            matchLine += buf;
        } else if (ti.status == TrackMatchInfo::Status::kNoMatch) {
            matchLine += " id=" + std::to_string(ti.trackId) + ":X";
        } else {
            matchLine += " id=" + std::to_string(ti.trackId) + ":S";
        }
    }
    char hdr[128];
    snprintf(hdr, sizeof(hdr), "frame %lld  dets=%zu heads=%zu cls0=%zu cls2=%zu  tracks=%zu",
             static_cast<long long>(frameId), heads.size() + bodies0.size() + bodies2.size(),
             heads.size(), bodies0.size(), bodies2.size(), trackInfos.size());
    char tm[128];
    snprintf(tm, sizeof(tm), "saved=%d  decode=%.1f infer=%.1f track=%.1f save=%.1f enc=%.1f ms",
             savedCount, decodeMs, inferMs, trackMs, saveMs, encMs);
    drawPanel(img, {std::string(hdr), matchLine, std::string(tm)});
}

} // namespace

int main(int argc, char** argv) {
    AppArgs args;
    bool helpRequested = false;
    if (!parseArgs(argc, argv, args, helpRequested)) {
        printUsage(argv[0]);
        return helpRequested ? 0 : 1;
    }
    initLogger(args.debug);

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

        // 6. 按 fps/N 设定 tracker 有效帧率（日志/秒数换算）；maxTimeLost=max(1,track_buffer)
        cfg.frameRate = std::max(1, static_cast<int>(std::lround(decoder.fps() / args.n)));
        trackerCpp::BYTETracker::resetId();
        trackerCpp::BYTETracker tracker(cfg, cfg.frameRate, args.n);
        if (args.debug) {
            tracker.setDebugCallback([](const std::string& m) { logDebug(m); });
            logDebug("app tracker: N=" + std::to_string(args.n) +
                     " video_fps=" + std::to_string(decoder.fps()) +
                     " tracker_frameRate=" + std::to_string(cfg.frameRate) +
                     " track_buffer=" + std::to_string(cfg.trackBuffer) +
                     " maxTimeLost=" + std::to_string(tracker.maxTimeLostValue()) +
                     " (video_frames=" + std::to_string(tracker.maxTimeLostValue() * args.n) +
                     ", seconds=" + std::to_string(
                         static_cast<double>(tracker.maxTimeLostValue()) /
                         static_cast<double>(std::max(1, cfg.frameRate))) +
                     ")");
        }

        // 7. 输出目录 output/<视频名stem>，trackId 子目录为纯 trackId
        std::string videoStem = std::filesystem::path(args.videoPath).stem().string();
        std::filesystem::path outRoot = std::filesystem::path(args.outputDir) / videoStem;
        std::filesystem::create_directories(outRoot);
        // debug 生命周期 CSV 写到 debug/<stem>/（与 debug 视频同视频名）
        std::filesystem::path debugRoot = std::filesystem::path("./debug") / videoStem;
        if (args.debug) {
            std::error_code ecDbg;
            std::filesystem::create_directories(debugRoot, ecDbg);
        }

        // 8. debug 模式创建 NVENC 硬编码器，输出 output/<stem>/<stem>_debug.mp4
        //    只编码处理帧（frameId % N == 0），fps = decoder.fps()/N，与日志帧号对齐
        std::unique_ptr<NvencEncoder> encoder;
        if (args.debug) {
            encoder = std::make_unique<NvencEncoder>();
            std::filesystem::path debugVideoPath = outRoot / (videoStem + "_debug.mp4");
            double vFps = decoder.fps() / static_cast<double>(args.n);
            if (!encoder->open(decoder.width(), decoder.height(), vFps, debugVideoPath.string())) {
                logWarn("debug video encoder open failed, debug 可视化视频不生成（不影响日志与裁剪图）");
                encoder.reset();
            }
        }

        const int frameW = decoder.width();
        const int frameH = decoder.height();
        const int64_t N = args.n;
        int64_t processed = 0;       // 实际参与推理+追踪的帧数
        int64_t decoded = 0;         // 解码总帧数（含跳过帧）
        // 各环节累计耗时（毫秒），用于结束汇总
        double decodeMsTotal = 0.0;  // 含跳过帧
        double inferMsTotal = 0.0;
        double trackMsTotal = 0.0;
        double saveMsTotal = 0.0;    // 匹配+D2H+裁剪+存图
        double encMsTotal = 0.0;     // debug 绘制+硬编码
        auto startTime = std::chrono::steady_clock::now();

        while (true) {
            auto tDecode0 = std::chrono::steady_clock::now();
            std::unique_ptr<Frame> frame;
            if (!decoder.readFrame(frame)) {
                break;  // EOF 已排空
            }
            ++decoded;
            double decodeMs = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - tDecode0).count();
            decodeMsTotal += decodeMs;
            if (frame->frameId % N != 0) {
                if (args.debug) {
                    logDebug("frame " + std::to_string(frame->frameId) + " skipped (decode=" +
                             std::to_string(decodeMs) + " ms)");
                }
                continue;  // 跳过帧：不参与推理与追踪
            }
            ++processed;

            auto tInfer0 = std::chrono::steady_clock::now();
            std::vector<Detection> dets = infer.infer(*frame);  // 已 clamp 到显示宽高
            double inferMs = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - tInfer0).count();
            inferMsTotal += inferMs;

            // 1) 分流（携带全量下标 origIdx，供日志 det#N 与绘制标签对齐）
            std::vector<IdxDet> heads, bodies0, bodies2;
            for (size_t i = 0; i < dets.size(); ++i) {
                Detection d = dets[i];
                if (d.classId == 1) {
                    heads.push_back({i, d});
                } else if (d.classId == 0) {
                    bodies0.push_back({i, d});
                } else if (d.classId == 2) {
                    bodies2.push_back({i, d});
                }
            }
            if (args.debug) {
                logDebug("frame " + std::to_string(frame->frameId) + " dets=" +
                         std::to_string(dets.size()) + " heads=" + std::to_string(heads.size()) +
                         " cls0=" + std::to_string(bodies0.size()) + " cls2=" +
                         std::to_string(bodies2.size()));
                for (size_t i = 0; i < dets.size(); ++i) {
                    const Detection& d = dets[i];
                    char detBuf[192];
                    snprintf(detBuf, sizeof(detBuf),
                             "  det[%zu]: cls=%d conf=%.4f xyxy=(%.1f,%.1f,%.1f,%.1f)",
                             i, d.classId, d.confidence, d.x1, d.y1, d.x2, d.y2);
                    logDebug("frame " + std::to_string(frame->frameId) + detBuf);
                }
            }

            // 2) heads -> ByteTrack（xyxy 转 center-xywh）；空检测帧也必须 update，保证帧计数推进
            trackerCpp::DetectionResults input;
            input.xywh.resize(static_cast<Eigen::Index>(heads.size()), 4);
            input.conf.resize(static_cast<Eigen::Index>(heads.size()));
            input.cls.resize(static_cast<Eigen::Index>(heads.size()));
            for (size_t i = 0; i < heads.size(); ++i) {
                const Detection& d = heads[i].det;
                double cx = (d.x1 + d.x2) / 2.0;
                double cy = (d.y1 + d.y2) / 2.0;
                double w = d.x2 - d.x1;
                double h = d.y2 - d.y1;
                input.xywh.row(static_cast<Eigen::Index>(i)) << cx, cy, w, h;
                input.conf(static_cast<Eigen::Index>(i)) = d.confidence;
                input.cls(static_cast<Eigen::Index>(i)) = 1.0;
            }
            auto tTrack0 = std::chrono::steady_clock::now();
            tracker.setCurrentVideoFrameId(frame->frameId);
            Eigen::MatrixXf tracks = tracker.update(input);  // 每行 [x1,y1,x2,y2,trackId,score,cls,idx]
            double trackMs = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - tTrack0).count();
            trackMsTotal += trackMs;
            if (args.debug) {
                // P2 health：orphan heads / orphan tracks（几何粗判：中心落在 track 内计关联）
                int headHigh = 0, headLow = 0;
                for (const auto& hd : heads) {
                    if (hd.det.confidence >= static_cast<float>(cfg.trackHighThresh)) ++headHigh;
                    else if (hd.det.confidence > static_cast<float>(cfg.trackLowThresh)) ++headLow;
                }
                int orphanHeads = 0;
                for (const auto& hd : heads) {
                    const float hcx = 0.5f * (hd.det.x1 + hd.det.x2);
                    const float hcy = 0.5f * (hd.det.y1 + hd.det.y2);
                    bool covered = false;
                    for (int r = 0; r < tracks.rows(); ++r) {
                        if (hcx >= tracks(r, 0) && hcx <= tracks(r, 2) &&
                            hcy >= tracks(r, 1) && hcy <= tracks(r, 3)) {
                            covered = true;
                            break;
                        }
                    }
                    if (!covered) ++orphanHeads;
                }
                int orphanTracks = 0;
                for (int r = 0; r < tracks.rows(); ++r) {
                    const float tcx = 0.5f * (tracks(r, 0) + tracks(r, 2));
                    const float tcy = 0.5f * (tracks(r, 1) + tracks(r, 3));
                    bool hitHead = false;
                    for (const auto& hd : heads) {
                        if (tcx >= hd.det.x1 && tcx <= hd.det.x2 &&
                            tcy >= hd.det.y1 && tcy <= hd.det.y2) {
                            hitHead = true;
                            break;
                        }
                    }
                    if (!hitHead) ++orphanTracks;
                }
                logDebug("frame " + std::to_string(frame->frameId) +
                         " health: heads=" + std::to_string(heads.size()) +
                         " head_high=" + std::to_string(headHigh) +
                         " head_low=" + std::to_string(headLow) +
                         " tracks=" + std::to_string(tracks.rows()) +
                         " orphan_heads=" + std::to_string(orphanHeads) +
                         " orphan_tracks=" + std::to_string(orphanTracks));

                logDebug("frame " + std::to_string(frame->frameId) + " tracks=" +
                         std::to_string(tracks.rows()));
                for (int r = 0; r < tracks.rows(); ++r) {
                    char trkBuf[192];
                    snprintf(trkBuf, sizeof(trkBuf),
                             "  trackOut: id=%d score=%.4f box=(%.1f,%.1f,%.1f,%.1f)",
                             static_cast<int>(tracks(r, 4)), tracks(r, 5),
                             tracks(r, 0), tracks(r, 1), tracks(r, 2), tracks(r, 3));
                    logDebug("frame " + std::to_string(frame->frameId) + trkBuf);
                }
            }

            // 3) NV12->BGR 下载策略：debug 模式下每帧都需绘制画面，提前下载；
            //    非 debug 仅在有命中时才下载。下载后本帧共用。
            GpuFrame* gpuFrame = static_cast<GpuFrame*>(frame.get());  // 本链路必为 GpuFrame
            cv::Mat bgr;
            if (args.debug) {
                bgr = nv12ToBgr(*gpuFrame);
            }

            double saveMs = 0.0;
            if (tracks.rows() == 0) {
                if (args.debug) {
                    // 空 track 帧也要绘制+编码，保证视频帧与日志帧号对齐
                    auto tEnc0 = std::chrono::steady_clock::now();
                    cv::Mat draw = bgr.clone();
                    drawDebugFrame(draw, frame->frameId, heads, bodies0, bodies2, {}, 0,
                                   decodeMs, inferMs, trackMs, 0.0, 0.0);
                    if (encoder) encoder->writeFrame(draw);
                    double encMs = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - tEnc0).count();
                    logDebug("frame " + std::to_string(frame->frameId) +
                             " timing: decode=" + std::to_string(decodeMs) +
                             " infer=" + std::to_string(inferMs) +
                             " track=" + std::to_string(trackMs) +
                             " save=0 enc=" + std::to_string(encMs) + " ms");
                }
                continue;
            }

            // 非 debug 才延迟下载；debug 已提前下载
            bool downloaded = args.debug;

            auto tSave0 = std::chrono::steady_clock::now();
            int savedThisFrame = 0;
            std::vector<TrackMatchInfo> trackInfos;  // 供绘制与汇总日志复用

            // 4) 逐 track 匹配与保存
            for (int r = 0; r < tracks.rows(); ++r) {
                cv::Rect head = clampToFrame(tracks(r, 0), tracks(r, 1), tracks(r, 2), tracks(r, 3),
                                             frameW, frameH);
                int64_t trackId = static_cast<int64_t>(tracks(r, 4));
                float trackScore = tracks(r, 5);
                TrackMatchInfo info{};
                info.trackId = trackId;
                info.score = trackScore;
                info.headRect = head;
                info.hit = nullptr;
                info.ioa = 0.0f;
                info.status = TrackMatchInfo::Status::kNoMatch;
                if (head.area() <= 0) {
                    trackInfos.push_back(info);
                    continue;
                }

                if (args.debug) {
                    char headBuf[160];
                    snprintf(headBuf, sizeof(headBuf),
                             " track %lld head=(%.1f,%.1f,%.1f,%.1f) score=%.4f",
                             static_cast<long long>(trackId),
                             tracks(r, 0), tracks(r, 1), tracks(r, 2), tracks(r, 3), trackScore);
                    logDebug("frame " + std::to_string(frame->frameId) + headBuf);
                }

                float ioa = 0.0f;
                std::vector<MatchCand> cands0, cands2;
                const Detection* hit = matchBest(head, bodies0, args.ioa, frameW, frameH, &ioa,
                                                  args.debug ? &cands0 : nullptr);
                int hitClass = 0;
                if (!hit) {
                    if (args.debug) {
                        logMatchCandidates(frame->frameId, trackId, "cls0", cands0, args.ioa);
                        logDebug("frame " + std::to_string(frame->frameId) + " track " +
                                 std::to_string(trackId) + "  cls0 无命中 -> 回退 cls2");
                    }
                    hit = matchBest(head, bodies2, args.ioa, frameW, frameH, &ioa,
                                   args.debug ? &cands2 : nullptr);
                    hitClass = 2;
                    if (!hit && args.debug) {
                        logMatchCandidates(frame->frameId, trackId, "cls2", cands2, args.ioa);
                        logDebug("frame " + std::to_string(frame->frameId) + " track " +
                                 std::to_string(trackId) + " no body match, skipped");
                    }
                } else if (args.debug) {
                    logMatchCandidates(frame->frameId, trackId, "cls0", cands0, args.ioa);
                }
                if (hit && args.debug) {
                    if (hitClass == 2) logMatchCandidates(frame->frameId, trackId, "cls2", cands2, args.ioa);
                    // 找原始下标与候选下标
                    const std::vector<IdxDet>& src = (hitClass == 0) ? bodies0 : bodies2;
                    size_t origIdx = 0;
                    int candIdx = -1;
                    for (size_t ci = 0; ci < src.size(); ++ci) {
                        if (&src[ci].det == hit) {
                            origIdx = src[ci].origIdx;
                            candIdx = static_cast<int>(ci);
                            break;
                        }
                    }
                    char buf[128];
                    snprintf(buf, sizeof(buf), "  => track %lld 命中 cls%d[%d] det#%zu ioa=%.4f",
                             static_cast<long long>(trackId), hitClass, candIdx, origIdx, ioa);
                    logDebug("frame " + std::to_string(frame->frameId) + buf);
                }
                if (!hit) {
                    trackInfos.push_back(info);
                    continue;  // 本帧该 track 不保存（正常分支）
                }

                if (!downloaded) {
                    bgr = nv12ToBgr(*gpuFrame);
                    downloaded = true;
                }

                cv::Rect crop = clampToFrame(hit->x1, hit->y1, hit->x2, hit->y2, frameW, frameH);
                if (crop.area() <= 0) {
                    info.status = TrackMatchInfo::Status::kNoMatch;
                    trackInfos.push_back(info);
                    continue;
                }
                if (crop.width < args.minSize || crop.height < args.minSize) {
                    if (args.debug) {
                        logDebug("frame " + std::to_string(frame->frameId) + " track " +
                                 std::to_string(trackId) +
                                 " crop " + std::to_string(crop.width) + "x" +
                                 std::to_string(crop.height) + " < min-size " +
                                 std::to_string(args.minSize) + ", skipped");
                    }
                    info.status = TrackMatchInfo::Status::kSmall;
                    info.cropRect = crop;
                    info.hit = hit;
                    info.ioa = ioa;
                    trackInfos.push_back(info);
                    continue;  // 任一边小于 min-size 不保存
                }

                // body 高度相对 head 高度过滤：半身(cls2)>1.5x，全身(cls0)>3x
                const float headH = static_cast<float>(head.height);
                const float bodyH = static_cast<float>(crop.height);
                const float minRatio = (hit->classId == 0) ? 3.0f : 1.5f;
                if (headH <= 0.0f || bodyH <= minRatio * headH) {
                    if (args.debug) {
                        char buf[160];
                        snprintf(buf, sizeof(buf),
                                 " track %lld cls%d bodyH=%.1f headH=%.1f ratio=%.3f <= %.1f, skipped",
                                 static_cast<long long>(trackId), hit->classId, bodyH, headH,
                                 headH > 0.0f ? bodyH / headH : 0.0f, minRatio);
                        logDebug("frame " + std::to_string(frame->frameId) + buf);
                    }
                    info.status = TrackMatchInfo::Status::kSmall;
                    info.cropRect = crop;
                    info.hit = hit;
                    info.ioa = ioa;
                    trackInfos.push_back(info);
                    continue;
                }

                info.hit = hit;
                info.ioa = ioa;
                info.status = TrackMatchInfo::Status::kMatched;
                info.cropRect = crop;
                trackInfos.push_back(info);

                std::filesystem::path dir = outRoot / std::to_string(trackId);
                std::error_code ec;
                std::filesystem::create_directories(dir, ec);
                if (ec) {
                    logWarn("create_directories failed: " + dir.string() + " : " + ec.message());
                    continue;
                }
                char name[64];
                snprintf(name, sizeof(name), "%06d_cls%d.png",
                         static_cast<int>(frame->frameId), hit->classId);
                std::filesystem::path outPath = dir / name;
                // PNG 无损（默认压缩级别仅影响文件大小，不损画质）；bgr 未被绘制污染
                if (!cv::imwrite(outPath.string(), bgr(crop))) {
                    logWarn("imwrite failed: " + outPath.string());
                } else {
                    ++savedThisFrame;
                    if (args.debug) {
                        logDebug("frame " + std::to_string(frame->frameId) + " track " +
                                 std::to_string(trackId) + " <- cls" + std::to_string(hit->classId) +
                                 " ioa=" + std::to_string(ioa) + " -> " + outPath.filename().string());
                    }
                }
            }
            saveMs = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - tSave0).count();
            saveMsTotal += saveMs;

            // 5) debug 模式：在 bgr.clone() 副本上绘制并硬编码送入视频（不污染裁剪 PNG）
            double encMs = 0.0;
            if (args.debug) {
                auto tEnc0 = std::chrono::steady_clock::now();
                cv::Mat draw = bgr.clone();
                drawDebugFrame(draw, frame->frameId, heads, bodies0, bodies2, trackInfos,
                               savedThisFrame, decodeMs, inferMs, trackMs, saveMs, 0.0);
                if (encoder) encoder->writeFrame(draw);
                encMs = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - tEnc0).count();
                encMsTotal += encMs;
            }
            if (args.debug) {
                logDebug("frame " + std::to_string(frame->frameId) + " saved " +
                         std::to_string(savedThisFrame) + " crops");
                logDebug("frame " + std::to_string(frame->frameId) +
                         " timing: decode=" + std::to_string(decodeMs) +
                         " infer=" + std::to_string(inferMs) +
                         " track=" + std::to_string(trackMs) +
                         " save=" + std::to_string(saveMs) +
                         " enc=" + std::to_string(encMs) + " ms");
            }
        }

        // flush 编码器并写 trailer（RAII 析构也会做，但显式调用便于错误捕获）
        if (encoder) {
            encoder->close();
            encoder.reset();
        }

        if (args.debug) {
            tracker.logFinalStats();
            std::filesystem::path csvPath = debugRoot / "tracks_lifecycle.csv";
            if (tracker.dumpLifecycleCsv(csvPath.string())) {
                logDebug("lifecycle csv: " + csvPath.string());
            } else {
                logWarn("lifecycle csv write failed: " + csvPath.string());
            }
        }

        auto elapsed = std::chrono::steady_clock::now() - startTime;
        double seconds = std::chrono::duration<double>(elapsed).count();
        double avgFps = seconds > 0.0 ? processed / seconds : 0.0;
        double avgDecodeMs = decoded > 0 ? decodeMsTotal / decoded : 0.0;
        double avgInferMs = processed > 0 ? inferMsTotal / processed : 0.0;
        double avgTrackMs = processed > 0 ? trackMsTotal / processed : 0.0;
        double avgSaveMs = processed > 0 ? saveMsTotal / processed : 0.0;
        double avgEncMs = processed > 0 ? encMsTotal / processed : 0.0;
        logInfo("done: processed=" + std::to_string(processed) + " frames (decoded=" +
                std::to_string(decoded) + "), elapsed=" + std::to_string(seconds) + " s, avg_fps=" +
                std::to_string(avgFps) +
                ", avg_ms: decode=" + std::to_string(avgDecodeMs) +
                " infer=" + std::to_string(avgInferMs) +
                " track=" + std::to_string(avgTrackMs) +
                " save=" + std::to_string(avgSaveMs) +
                " enc=" + std::to_string(avgEncMs));
        return 0;
    } catch (const std::exception& e) {
        logError(std::string("fatal: ") + e.what());
        return 1;
    }
}
