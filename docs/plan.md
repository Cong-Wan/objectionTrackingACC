# objectionTrackingACC 实施文档（执行版）

> **本文档读者**：负责实现的工程师。请**完整读完**后再动手，严格按 Task 顺序执行，每个 Task 通过验收后再进入下一个。
> 任何与本文档冲突的"顺手优化"都不允许。遇到文档未覆盖的情况，**停下来提问**，不要自行发挥。
> README.md 面向使用；本文档面向实现，是唯一执行依据。

---

## 0. 业务规则（已确认，不许变更）

| 项 | 规则 |
|---|---|
| 追踪对象 | 仅类别 1（head）送 ByteTrack |
| 归属匹配 | `IoA = area(head ∩ body) / area(head)`，先对 cls0（fullBody）取最大值，≥ 阈值（默认 0.5）命中；未命中再对 cls2（upperBody）同样匹配；仍未命中本帧该 track 不保存 |
| 并列处理 | 多个 body 并列最大 IoA 时，取 **body 面积最小者** |
| 多对一 | 允许共享：同一 body 框可保存到多个 trackId 目录，不做独占指派 |
| 保存内容 | 仅匹配到的 body 框（cls0 或 cls2）裁剪图；head 框不保存 |
| 处理频率 | `--N`：每 N 帧处理一帧（frameId % N == 0）；被处理帧上凡有 track 命中且匹配成功即保存 |
| 预过滤 | `--conf` 默认 0.01，必须 ≤ tracker.yaml 的 `track_low_thresh`，**启动时校验，超限报错退出** |
| 追踪参数 | 复用 trackers_cpp 原值；`frame_rate = max(1, round(视频fps / N))` |
| 输出 | `output/<视频名stem>/<trackId>/<%06d帧号>.jpg`；重复运行直接覆盖同名文件 |
| tracker 输出语义（已核实） | `update()` 只输出**本帧匹配/新激活**的 track（两轮未匹配轨迹 `markLost` 不进输出集），输出框为本帧真实检测框 `detXyxy()`，不存在预测框污染，应用层无需额外过滤 |

模型类别（batch1.int8.engine，输入 [1,3,640,640]，输出 [1,7,8400]，box 格式 cxcywh）：
`0 = fullBody`，`1 = head`，`2 = upperBody`。

---

## 1. 编码约定（强制）

1. **文件头**：每个新建代码文件（.h/.cpp/.cu）开头必须有：

```cpp
/*
Author: wilbur
Version: 1.0
Date: <YYYY-MM-DD>
Description: <一句话说明本文件职责>
*/
```

2. **命名**：文件名小驼峰（`nvdecDecoder.cpp`），类名大驼峰，函数/变量小驼峰，常量全大写下划线以外的场景用 `k` 前缀小驼峰（`kNv12`）。文件名不宜过长但要表意。
3. **显式源文件列表**：CMake 中禁止 `file(GLOB)`。
4. **错误处理**：构造/open 期失败抛 `std::runtime_error`（带上下文信息）；运行期可恢复问题（device 不匹配、imwrite 失败）打日志继续，不崩溃。
5. **不引入任何新第三方依赖**。只许用 §3 列出的依赖 + C++17 标准库（`std::filesystem` 允许）。

---

## 2. 代码来源与搬移清单（含完整源路径）

源项目根（下文用缩写）：

- `SRC_PA = /home/Wilbur/project/performanceAnalysis`（解码/推理来源）
- `SRC_TRK = /home/Wilbur/project/trackers_cpp`（追踪来源）

### 2.1 modules/common

| 目标文件 | 源文件 | 动作 |
|---|---|---|
| modules/common/frameTypes.h | `$SRC_PA/include/frameTypes.h` | **重写**（§4.1），保留 Detection / LetterboxParams 定义 |
| modules/common/frameTypes.cpp | —（新增） | GpuFrame 析构定义（§4.1） |
| modules/common/cudaUtils.h | `$SRC_PA/include/cudaUtils.h` | 原样复制 |
| modules/common/cudaUtils.cpp | `$SRC_PA/src/cudaUtils.cpp` | 原样复制 |
| modules/common/logger.h | `$SRC_PA/include/logger.h` | 原样复制 |
| modules/common/logger.cpp | `$SRC_PA/src/logger.cpp` | 原样复制 |
| modules/common/nv12ToBgr.h / nv12ToBgr.cu | —（新增） | 按 §4.4 实现 |

### 2.2 modules/decoder

| 目标文件 | 源文件 | 动作 |
|---|---|---|
| modules/decoder/decoderBase.h | —（新增） | 按 §4.2 接口实现 |
| modules/decoder/nvdecDecoder.h | `$SRC_PA/include/videoDecoder.h` | 复制+改造（§4.2 改动点逐条） |
| modules/decoder/nvdecDecoder.cpp | `$SRC_PA/src/videoDecoder.cpp` | 复制+改造（§4.2 改动点逐条） |

### 2.3 modules/infer

| 目标文件 | 源文件 | 动作 |
|---|---|---|
| modules/infer/inferBase.h | —（新增） | 按 §4.3 接口实现 |
| modules/infer/gpuPreprocess.h | `$SRC_PA/include/gpuPreprocess.h` | 原样复制 |
| modules/infer/gpuPreprocess.cu | `$SRC_PA/src/gpuPreprocess.cu` | 原样复制 |
| modules/infer/trtInfer.h | `$SRC_PA/include/inferRunner.h` + `$SRC_PA/include/postProcessor.h` | **合并改造**（§4.3 第 1~7 条） |
| modules/infer/trtInfer.cpp | `$SRC_PA/src/inferRunner.cpp` + `$SRC_PA/src/postProcessor.cpp` | **合并改造**（§4.3 第 1~7 条） |

### 2.4 modules/tracker（原样复制，一行不许改）

| 目标文件 | 源文件 |
|---|---|
| modules/tracker/include/byteTracker.h | `$SRC_TRK/include/byteTracker.h` |
| modules/tracker/include/baseTrack.h | `$SRC_TRK/include/baseTrack.h` |
| modules/tracker/include/trackerConfig.h | `$SRC_TRK/include/trackerConfig.h` |
| modules/tracker/include/utils/kalmanFilter.h | `$SRC_TRK/include/utils/kalmanFilter.h` |
| modules/tracker/include/utils/matching.h | `$SRC_TRK/include/utils/matching.h` |
| modules/tracker/src/byteTracker.cpp | `$SRC_TRK/src/byteTracker.cpp` |
| modules/tracker/src/baseTrack.cpp | `$SRC_TRK/src/baseTrack.cpp` |
| modules/tracker/src/trackerConfig.cpp | `$SRC_TRK/src/trackerConfig.cpp` |
| modules/tracker/src/utils/kalmanFilter.cpp | `$SRC_TRK/src/utils/kalmanFilter.cpp` |
| modules/tracker/src/utils/matching.cpp | `$SRC_TRK/src/utils/matching.cpp` |
| config/tracker.yaml | `$SRC_TRK/config.yaml` |

不复制：`$SRC_TRK/testTracker.cpp`、`$SRC_TRK/CMakeLists.txt`（构建由本项目 CMake 接管）。

**明确丢弃**（不搬）：appConfig / engineBuilder / int8Calibrator / jsonUtils / metadata.h 的 manifest 部分 / gpuMonitor / metrics / reportWriter / boundedQueue / perfPipeline / perfTestMain / modelToolMain / findNvml.cmake。

---

## 3. 依赖与 CMake（Task 1 的完整规格）

### 3.1 依赖路径（写为 CMake 变量，允许 -D 覆盖）

| 变量 | 默认值 | 用途 |
|---|---|---|
| `TensorRT_ROOT` | `/home/Disk-6T/tensorrt` | include: `$ROOT/include`，lib: `$ROOT/targets/x86_64-linux-gnu/lib` |
| `OpenCV_DIR` | `/home/Disk-6T/opencv/build` | `find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc)` |
| `CUDA_TOOLKIT_ROOT_DIR` | `/usr/local/cuda-12.6` | nvcc + cudart |
| FFmpeg7 | `/usr/local/ffmpeg7` | pkg-config：`PKG_CONFIG_PATH` 前置 `/usr/local/ffmpeg7/lib/pkgconfig`，`pkg_check_modules` 找 `libavformat libavcodec libavutil` |
| Eigen3 / yaml-cpp | 系统 | `find_package(Eigen3 REQUIRED)` / `find_package(yaml-cpp REQUIRED)` |

### 3.2 CMakeLists.txt 要点（照抄 performanceAnalysis 的对应段落并删减）

- `cmake_minimum_required(VERSION 3.24)`；`project(objectionTrackingACC LANGUAGES CXX CUDA)`
- **必须在 project() 之前**：`set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.6/bin/nvcc)`（带 `if(NOT CMAKE_CUDA_COMPILER)` 保护）
- C++17 / CUDA17，`CMAKE_CUDA_ARCHITECTURES` 默认 `86`（RTX 3090），允许 -D 覆盖
- 默认 `Release`
- `add_compile_definitions(__STDC_CONSTANT_MACROS __STDC_FORMAT_MACROS)`（FFmpeg 头需要）
- rpath：`${CUDA_TOOLKIT_ROOT_DIR}/lib64`、`/usr/local/ffmpeg7/lib`、TensorRT lib 目录，同时设 `CMAKE_BUILD_RPATH` 与 `CMAKE_INSTALL_RPATH`
- 去掉：NVML、nlohmann/json、`libnvonnxparser.so`（运行时反序列化不需要）
- 链接库：`PkgConfig::AVFORMAT PkgConfig::AVCODEC PkgConfig::AVUTIL ${OpenCV_LIBS} CUDA::cudart_static CUDA::cuda_driver ${TENSORRT_LIB_DIR}/libnvinfer.so`

### 3.3 构建目标

```
trackerCpp  （STATIC 库：modules/tracker 5 个 .cpp；PUBLIC include modules/tracker/include 与 include/utils；
             PUBLIC 链接 Eigen3::Eigen yaml-cpp::yaml-cpp）
otaCore     （STATIC 库：modules/common + modules/decoder + modules/infer 全部源文件；
             PUBLIC include modules/ 各子目录；链接 §3.2 链接库）
trackCrop   （可执行：apps/trackCrop/trackCropMain.cpp；链接 otaCore trackerCpp；
             输出到 build/bin/：set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)）
```

---

## 4. 模块详细设计

### 4.1 modules/common/frameTypes.h（重写）

```cpp
struct Frame {
    enum class Device { kCuda, kDrm };          // kDrm 预留给 RK
    enum class PixFmt { kNv12, kNv21, kRgb };
    int64_t frameId = 0;
    int width = 0;                              // 显示宽（非 stride 对齐）
    int height = 0;
    Device device = Device::kCuda;
    PixFmt pixFmt = PixFmt::kNv12;
    virtual ~Frame() = default;
};

struct AVFrame;                                 // 前向声明！本头文件禁止 include 任何 FFmpeg 头

struct GpuFrame : Frame {
    uint8_t* data0 = nullptr;                   // Y plane
    uint8_t* data1 = nullptr;                   // UV plane
    int linesize0 = 0;
    int linesize1 = 0;
    // 注意：shared_ptr 默认 deleter 是 delete，AVFrame 必须 av_frame_free！
    // 构造一律照搬 videoDecoder 原写法：shared_ptr<AVFrame>(p, [](AVFrame* f){ if(f) av_frame_free(&f); })
    std::shared_ptr<AVFrame> frameRef;
    ~GpuFrame() override;                       // 定义在 frameTypes.cpp（AVFrame 完整类型可见处）
};

struct Detection {                              // 原样保留（原图坐标系 xyxy）
    int64_t frameId; int classId; float confidence;
    float x1, y1, x2, y2;
};

struct LetterboxParams { /* 原样保留 */ };
```

- 新建 `frameTypes.cpp`，内含 `GpuFrame::~GpuFrame() = default;`（在 include 了 `<libavutil/frame.h>` 的翻译单元里实例化 shared_ptr 析构）。
- **禁止**在 frameTypes.h 中出现 `#include <libavutil/...>`。

### 4.2 modules/decoder

**decoderBase.h（新增）**：

```cpp
class IVideoDecoder {
public:
    virtual ~IVideoDecoder() = default;
    virtual void open() = 0;
    virtual bool readFrame(std::unique_ptr<Frame>& frame) = 0;  // false = 正常结束（EOF 已排空）
    virtual int width() const = 0;
    virtual int height() const = 0;
    virtual double fps() const = 0;
};
```

**nvdecDecoder.h/.cpp（改造自 videoDecoder）**，改动点逐条：

1. 类名 `VideoDecoder` → `NvdecDecoder`，继承 `IVideoDecoder`。
2. 构造函数精简为 `NvdecDecoder(std::string videoPath, AVBufferRef* sharedHwDeviceCtx = nullptr)`。**删除** `maxFrames_`、`debug_` 成员及全部相关分支。
3. `readFrame` 签名改为返回 `std::unique_ptr<Frame>`：内部逻辑不变，出口处把填充好的 `GpuFrame` 包成 `std::unique_ptr<Frame>` 返回。
4. **保留**：EOF flush 逻辑（`flushing_` + `avcodec_send_packet(nullptr)`）、非视频流跳过、NV12/full-range/BT.709 校验（校验失败抛 `std::runtime_error`——这就是"色彩校验失败退出"的落点）、`primary_ctx=1` 的 CUDA context 创建与共享、fps() 的 avg→r_frame_rate 回退。
5. `GpuFrame.frameId` 用解码器内自增计数器（`nextFrameId_`），**这就是输出图片名中的"原始帧号"**。
6. 损坏 packet 导致 `av_read_frame`/`send_packet` 报错：维持原行为抛异常，由应用层 catch 后报错退出（退出码 1）。

### 4.3 modules/infer

**inferBase.h（新增）**：

```cpp
class IInferEngine {
public:
    virtual ~IInferEngine() = default;
    virtual void load() = 0;
    virtual std::vector<Detection> infer(const Frame& frame) = 0;
};
```

**trtInfer.h/.cpp（inferRunner + postProcessor 合并改造）**：

1. 类 `TrtInfer : public IInferEngine`，构造：`TrtInfer(std::string enginePath, float confThreshold, float nmsIouThreshold = 0.45f, int maxDetections = 300)`。
2. 从 inferRunner 搬：engine 文件读取与反序列化、`TrtLogger`、input/output buffer 分配、`enqueueV3` 推理、输出 D2H 到 pinned buffer。input shape 固定 `[1,3,640,640]`，output 校验 `[1,7,8400]`（校验失败抛异常）。
3. **删除**：warmup / inferDummyOnce / NaN 校验统计 / 各阶段计时上报（`preprocessMs` 等）、batch 相关逻辑（本项目 batch 恒为 1）、`metadata.h` 依赖（模型常量直接写为类内常量：640×640、7、8400、3 类）。
4. 从 postProcessor 搬：output0 解析（8400 anchor × cxcywh + 3 类分数）、conf 过滤、letterbox 逆映射、**clamp 到显示宽高（已有，保留）**、class-aware NMS、按 conf 降序限 maxDetections。
5. `infer(const Frame&)` 流程：

```
if (frame.device != Device::kCuda) { logWarn("TrtInfer: unsupported device, skip"); return {}; }
const GpuFrame& gpu = static_cast<const GpuFrame&>(frame);
computeLetterboxParams(gpu.width, gpu.height, 640, 640);
preprocessNv12ToNchw(gpu.data0, gpu.data1, ..., inputBuffer_, stream);   // 零拷贝直写 TRT 输入
enqueueV3(stream) → cudaStreamSynchronize
后处理（含 clamp、NMS）→ return detections（detection.frameId = frame.frameId）
```

6. 单次推理用自己的 `cudaStream_t`（load 时创建，析构销毁）；每次 infer 末尾 `cudaStreamSynchronize`，**返回时结果必须已可用**。
7. 异常：engine 文件不存在/反序列化失败/shape 不符 → `load()` 抛 `std::runtime_error`。

### 4.4 modules/common/nv12ToBgr（新增）

```cpp
// nv12ToBgr.h
// NV12(full-range BT.709) GPU 帧 → 8bit BGR cv::Mat（含 D2H）
cv::Mat nv12ToBgr(const GpuFrame& frame, cudaStream_t stream = nullptr);
```

- kernel 逐像素：Y = data0[y*linesize0 + x]；U/V = data1[(y/2)*linesize1 + (x/2)*2 + {0,1}]
- **BT.709 full-range 公式（照抄，不许换 BT.601）**：

```
R = Y + 1.5748 * (V - 128)
G = Y - 0.1873 * (U - 128) - 0.4681 * (V - 128)
B = Y + 1.8556 * (U - 128)
```

- clamp 到 [0,255] 写 uchar；输出临时 device buffer 复用（类外函数则每次 cudaMalloc/cudaFree 也可接受，D2H 不是瓶颈——**选简单方案**）。
- `cudaMemcpy` D2H 到 `cv::Mat(frame.height, frame.width, CV_8UC3)`；有 stream 则先 `cudaStreamSynchronize`。
- 边界：宽高为奇数时 UV 行数 `(height+1)/2`，kernel 内对最后一行/列的 UV 采样做边界 min 钳制；奇数宽时 `(x/2)*2+1` 最大可取到 `width`（越出显示宽一列），依赖 NVDEC 输出的 `linesize1 ≥ width+1` 对齐 padding 兜底（decoder 对齐保证，不需额外校验）；索引必须一律用 linesize，禁止用 width 当行距。

### 4.5 modules/tracker（原样搬入）

- 文件：byteTracker / baseTrack / trackerConfig / utils/kalmanFilter / utils/matching，共 5 对 .h/.cpp。
- **一行都不许改**。发现 bug 先报告，不许顺手修。
- `trackerConfig.cpp` 中 `loadTrackerConfig()` 默认相对路径逻辑保留；应用层始终显式传路径调用 `loadTrackerConfig(path)`。
- config/tracker.yaml 内容（当前值，勿改）：

```yaml
track_high_thresh: 0.6
track_low_thresh: 0.04
new_track_thresh: 0.2
track_buffer: 5
match_thresh: 0.9
aspect_ratio_thresh: 1.5
min_box_area: 10
mot20: false
fuse_score: false
frame_rate: 30
```

---

## 5. trackCrop 应用详细设计（apps/trackCrop/trackCropMain.cpp）

### 5.1 命令行参数

手写解析（循环 + `std::string` 比较，不引第三方）：

| 参数 | 必填 | 默认 | 说明 |
|---|---|---|---|
| `--video <path>` | 是 | — | 本地视频路径 |
| `--engine <path>` | 是 | — | TensorRT engine 路径 |
| `--N <int>` | 否 | 1 | 每 N 帧处理一帧，N ≥ 1，否则报错退出 |
| `--output <dir>` | 否 | `./output` | 输出根目录 |
| `--conf <float>` | 否 | 0.01 | 预过滤阈值，必须 ≤ tracker 的 track_low_thresh |
| `--ioa <float>` | 否 | 0.5 | IoA 命中阈值，(0,1] |
| `--tracker-config <path>` | 否 | `config/tracker.yaml` | 追踪参数 |
| `--help` | 否 | — | 打印用法退出 |

参数缺失/非法 → 打印用法，退出码 1。

### 5.2 启动顺序（严格按序，任一步失败抛异常→catch→退出码 1）

```
1. 解析参数
2. cfg = loadTrackerConfig(trackerConfigPath)
3. 校验 conf <= cfg.trackLowThresh，违反则 logError + return 1
4. decoder = NvdecDecoder(video); decoder.open()
   → 校验 width/height/fps > 0，否则报错退出
5. infer = TrtInfer(engine, conf); infer.load()
6. cfg.frameRate = max(1, round(decoder.fps() / N))
   tracker = BYTETracker(cfg, cfg.frameRate); BYTETracker::resetId() 在构造前调用一次
7. outRoot = output/<视频文件名stem>（std::filesystem::path(video).stem()）
   create_directories(outRoot)
```

### 5.3 主循环伪代码（照此实现）

```cpp
while (true) {
    std::unique_ptr<Frame> frame;
    if (!decoder.readFrame(frame)) break;              // EOF 已排空

    if (frame->frameId % N != 0) continue;             // 跳过帧：不参与推理与追踪

    std::vector<Detection> dets = infer.infer(*frame); // 已 clamp 到显示宽高

    // 1) 分流
    std::vector<Detection> heads, bodies0, bodies2;
    for (auto& d : dets) {
        if (d.classId == 1) heads.push_back(d);
        else if (d.classId == 0) bodies0.push_back(d);
        else if (d.classId == 2) bodies2.push_back(d);
    }

    // 2) heads → ByteTrack（Detection 是 xyxy，转 center-xywh）
    trackerCpp::DetectionResults input;
    input.xywh.resize(heads.size(), 4);
    input.conf.resize(heads.size());
    input.cls.resize(heads.size());
    for (i...) {
        cx = (d.x1+d.x2)/2; cy = (d.y1+d.y2)/2; w = d.x2-d.x1; h = d.y2-d.y1;
        input.xywh.row(i) << cx, cy, w, h;
        input.conf(i) = d.confidence;
        input.cls(i)  = 1.0;
    }
    Eigen::MatrixXf tracks = tracker.update(input);    // 每行 [x1,y1,x2,y2,trackId,score,cls,idx]

    if (tracks.rows() == 0) continue;

    // 3) 本帧有 track：先做一次 NV12→BGR（所有命中共用，只下载这一次）
    GpuFrame* gpuFrame = static_cast<GpuFrame*>(frame.get());   // 本链路必为 GpuFrame
    cv::Mat bgr;  bool downloaded = false;

    // 4) 逐 track 匹配与保存
    for (int r = 0; r < tracks.rows(); ++r) {
        // tracks(r, 0..3) = x1,y1,x2,y2；tracks(r, 4) = trackId
        Rect head = clampToFrame(tracks(r,0), tracks(r,1), tracks(r,2), tracks(r,3));
        if (head 面积 <= 0) continue;

        const Detection* hit = matchBest(head, bodies0, ioaThresh);   // §5.5
        if (!hit) hit = matchBest(head, bodies2, ioaThresh);          // 回退
        if (!hit) continue;

        if (!downloaded) { bgr = nv12ToBgr(*gpuFrame); downloaded = true; }

        Rect crop = clampToFrame(hit->x1, hit->y1, hit->x2, hit->y2);
        if (crop 面积 <= 0) continue;

        int64_t trackId = (int64_t)tracks(r, 4);   // float 存储，先转回整型（<2^24 无精度问题，见 §6-20）
        std::string dir = outRoot + "/" + std::to_string(trackId);
        create_directories(dir);
        char name[64]; snprintf(name, sizeof(name), "%06d.jpg", (int)frame->frameId);
        if (!cv::imwrite(dir + "/" + name, bgr(crop))) logWarn("imwrite failed: ...");
    }
}
logInfo("done: processed=X frames, elapsed=Y s, avg_fps=Z");   // T7 验收的计时依据
```

> 注意：`frame` 是 `unique_ptr<Frame>`，decoder 已保证 device==kCuda，static_cast 安全。`tracks(r,4)` 中的 trackId 是 float 存储，转 int64_t 使用。

### 5.4 clampToFrame

```cpp
// float 框 → int cv::Rect，clamp 到 [0,w]×[0,h]：
// left = (int)floor(min(max(x1,0),w)), width = (int)ceil(min(max(x2,0),w)) - left，高同理
// 宽或高 ≤ 0 视为无效。x2/y2 语义统一按 cv::Rect 闭区间处理，不纠结 1px 差异。
```

### 5.5 matchBest（IoA 匹配，照此实现）

```cpp
// 返回候选中 IoA ≥ thresh 且最大者的 body；IoA 并列（±1e-6）取 body 面积最小者；无命中返回 nullptr
const Detection* matchBest(const cv::Rect& head, const std::vector<Detection>& bodies, float thresh) {
    const Detection* best = nullptr;
    float bestIoa = 0.0f;
    float bestArea = FLT_MAX;
    for (auto& b : bodies) {
        cv::Rect body = clampToFrame(b.x1, b.y1, b.x2, b.y2);
        if (body.area() <= 0) continue;
        float ioa = (float)(head & body).area() / (float)head.area();   // cv::Rect 交
        if (ioa < thresh) continue;                                     // ≥ thresh 才算命中
        if (!best || ioa > bestIoa + 1e-6f ||
            (std::fabs(ioa - bestIoa) <= 1e-6f && (float)body.area() < bestArea)) {
            best = &b; bestIoa = ioa; bestArea = (float)body.area();
        }
    }
    return best;
}
```

### 5.6 每帧结束

`frame` 离开作用域即释放（GpuFrame 析构释放 AVFrame 引用）。**禁止**跨帧持有 frame 指针。

---

## 6. 边界情况处理总表（实现时逐条对照）

| # | 场景 | 处理 |
|---|---|---|
| 1 | 视频打不开 / 无视频流 / 编码不支持 | decoder.open() 抛异常 → main catch → 打日志退出码 1 |
| 2 | 色彩空间非 full-range BT.709 | open() 抛异常退出（不许静默回退矩阵） |
| 3 | EOF | decoder 内部 flush 延迟帧，排空后 readFrame 返回 false，正常结束 |
| 4 | 解码中途损坏 packet | decoder 抛异常 → 退出码 1（不做跳过续跑，当前版本从严） |
| 5 | 某帧零检测 | dets 为空 → heads 空 → update 空输入（tracker 内部处理，**仍必须调用 update**？——**不**：跳过的帧不送；但被处理帧即使 0 检测也要 `update` 空矩阵，保证 tracker 内部帧计数与 lost 推进正确） |
| 6 | 被处理帧有检测但无 head | 同上，heads 空，仍需 update 空输入；无 track 输出则不保存 |
| 7 | head 匹配不到任何 body | 跳过该 track 本帧保存（正常分支，不打 warn） |
| 8 | track 输出框越界 | clampToFrame 防御；面积 ≤0 跳过 |
| 9 | body 框越界 | 后处理已 clamp；matchBest 内再 clamp 一次防御 |
| 10 | N > 视频总帧数 | 只处理 frameId=0 一帧，正常跑完 |
| 11 | fps 读取为 0（异常容器） | 启动校验报错退出（§5.2 第 4 步） |
| 12 | VFR 视频 | fps 为平均值，frame_rate 换算为近似，可接受，不特殊处理 |
| 13 | 输出目录已存在/图片重名 | 直接覆盖（imwrite 语义），不去重不报警 |
| 14 | create_directories 失败 / 磁盘满 imwrite 失败 | logWarn 记录路径，继续跑（不崩溃） |
| 15 | trackId 数量多导致目录多 | 不限制，按实际分配 |
| 16 | 长视频内存 | 每帧 unique_ptr 释放；nv12ToBgr 临时 buffer 作用域内释放；**验证项 T8** 跑长视频观察 RSS |
| 17 | --conf 传了 > track_low_thresh | 启动即报错退出（§5.2 第 3 步） |
| 18 | 同帧多 track 命中 | BGR 只下载一次（downloaded 标志），逐个裁剪保存 |
| 19 | 视频含旋转 display matrix（手机视频常见） | **不支持旋转矫正**：open 时检测 side data，存在则 logWarn "rotation metadata ignored"，按存储方向处理（画面可能旋转 90°，检测追踪仍工作但裁剪方向随之） |
| 20 | trackId 存于 float（MatrixXf） | 使用时转 int64_t（§5.3）；ID < 2^24（约 1600 万）无精度丢失，单视频场景不可能达到，仅备注不处理 |
| 21 | 奇数宽/高视频 | 支持。NV12 色度采样与 linesize padding 已覆盖（§4.4），不需特殊分支 |

> **第 5 条是易错点**：被处理帧 = 检测帧 = 追踪帧，三者一致；跳过的帧 tracker 完全无感（frame_rate 已按 fps/N 设定，时间语义对齐）。被处理帧即使没有任何检测，也必须用空输入调 `update`，否则 track_buffer 语义错乱。

---

## 7. 实施任务清单（勾选式，严格按序）

**使用规则**：每完成一项把 `- [ ]` 改成 `- [x]`；一个 Task 内所有项勾完且验收通过，才允许进入下一个 Task；**每新增一个源文件必须同步 CMakeLists 显式源列表**（T1.4 后生效）。

### T1 工程骨架

- [x] T1.1 创建目录：`modules/common` `modules/decoder` `modules/infer` `modules/tracker/include/utils` `modules/tracker/src/utils` `apps/trackCrop` `apps/smoke` `config` `docs`
- [x] T1.2 编写 CMakeLists.txt（§3.1 依赖变量 / §3.2 全部要点；本步只建 trackCrop 空目标）
- [x] T1.3 新建 `apps/trackCrop/trackCropMain.cpp`：带文件头的空 main，`printf("trackCrop scaffold ok\n")`
- [x] T1.4 **验收**：`mkdir build && cd build && cmake .. && make -j` 通过；`./bin/trackCrop` 输出 scaffold 信息

### T2 tracker 模块

- [x] T2.1 按 §2.4 表复制 5 个头文件 → `modules/tracker/include/`
- [x] T2.2 按 §2.4 表复制 5 个 .cpp → `modules/tracker/src/`
- [x] T2.3 复制 `$SRC_TRK/config.yaml` → `config/tracker.yaml`
- [x] T2.4 CMake 新增 `trackerCpp` 静态库目标（§3.3），源列表含 5 个 .cpp
- [x] T2.5 新建 `apps/smoke/smokeTracker.cpp`（CMake 同步加 `smokeTracker` 目标输出到 bin/）：从 `argv[1]` 读 tracker 配置路径（缺省 `config/tracker.yaml`），构造 BYTETracker，连续 2 帧喂同一检测框（xywh=100,100,20,40；conf=0.9；cls=1），打印每次 update 返回的行数与 trackId
- [x] T2.6 **验收**：make 通过；**在项根目录运行** `./build/bin/smokeTracker`（相对路径才能解析到 config/tracker.yaml），第二帧输出 trackId ≥ 1
- [x] T2.7 删除 smokeTracker.cpp 及 CMake 对应目标

### T3 common 模块

- [x] T3.1 重写 `modules/common/frameTypes.h`（§4.1 代码照写，含 AVFrame 前向声明与 deleter 注释）
- [x] T3.2 新建 `modules/common/frameTypes.cpp`（`GpuFrame::~GpuFrame() = default;`，include `<libavutil/frame.h>`）
- [x] T3.3 复制 cudaUtils.h/.cpp、logger.h/.cpp（§2.1 路径）
- [x] T3.4 CMake 新增 `otaCore` 静态库目标，加入上述源文件
- [x] T3.5 **验收**：make 通过

### T4 decoder 模块

- [x] T4.1 新建 `modules/decoder/decoderBase.h`（§4.2 接口照写）
- [x] T4.2 复制 `videoDecoder.h` → `nvdecDecoder.h`，按 §4.2 改动点 1~3 改（类名/继承/构造签名/readFrame 签名）
- [x] T4.3 复制 `videoDecoder.cpp` → `nvdecDecoder.cpp`，按 §4.2 改动点 2~6 改；**保留清单逐项核对**：EOF flush / 跳过非视频流 / NV12+full-range+BT.709 校验 / primary_ctx=1 / fps 双回退
- [x] T4.4 两个源文件加入 otaCore 源列表
- [x] T4.5 新建 `apps/smoke/smokeDecoder.cpp`（CMake 同步加 `smokeDecoder` 目标）：打开 §8 测试视频，循环读 5 帧，打印 frameId/width/height/linesize0/linesize1/pixFmt
- [x] T4.6 **验收**：width/height 与 `/usr/local/ffmpeg7/bin/ffprobe` 显示宽高一致；frameId 从 0 连续递增；正常退出无 crash
- [x] T4.7 删除冒烟代码及 CMake 目标

### T5 infer 模块

- [x] T5.1 复制 `gpuPreprocess.h/.cu`（§2.3 路径）并加入 otaCore 源列表
- [x] T5.2 新建 `modules/infer/inferBase.h`（§4.3 接口照写）
- [x] T5.3 改造 `trtInfer.h/.cpp`：§4.3 第 1~7 条逐条对照完成（构造/load/infer 流程/device 契约/异常路径）
- [x] T5.4 新建 `apps/smoke/smokeInfer.cpp`（CMake 同步加 `smokeInfer` 目标）：decoder 读首帧 → `infer.infer()` → 打印检测总数与置信度 Top5（含 classId、xyxy）
- [x] T5.5 **验收**：检测数 > 0；conf ∈ (0,1)；所有坐标 ∈ [0,w]×[0,h]（验证 clamp 生效）
- [x] T5.6 删除冒烟代码及 CMake 目标

### T6 nv12ToBgr

- [x] T6.1 新建 `modules/common/nv12ToBgr.h/.cu`（§4.4，BT.709 公式照抄，禁止换成 BT.601）
- [x] T6.2 源文件加入 otaCore 源列表
- [x] T6.3 新建 `apps/smoke/smokeBgr.cpp`（CMake 同步加 `smokeBgr` 目标）：首帧 → `nv12ToBgr` → `cv::imwrite("bgr.png")`
- [x] T6.4 生成参照图：`/usr/local/ffmpeg7/bin/ffmpeg -y -i <测试视频> -frames:v 1 ref.png`
- [x] T6.5 **验收**：bgr.png 与 ref.png 目视一致（无偏绿/偏灰/条纹错位）
- [x] T6.6 删除冒烟代码及 CMake 目标

### T7 trackCrop 应用

- [x] T7.1 参数解析（§5.1 全表；--help；缺参/非法 → 打印用法退出码 1）
- [x] T7.2 启动顺序 7 步（§5.2）：含 conf ≤ track_low_thresh 校验、fps > 0 校验、`BYTETracker::resetId()`、输出目录创建
- [x] T7.3 主循环（§5.3 伪代码照写）：抽帧 / 三类分流 / xyxy→center-xywh 转换 / **空检测帧也调 update**
- [x] T7.4 `clampToFrame` + `matchBest`（§5.4/§5.5 代码照写）
- [x] T7.5 保存逻辑：每帧 D2H 至多一次 / `create_directories` / `%06d.jpg` / imwrite 失败 logWarn 继续
- [x] T7.6 结束日志：`processed=X frames, elapsed=Y s, avg_fps=Z`
- [x] T7.7 **验收 A**：§8 参考命令跑通，产出 `output/3810_125502/<trackId>/*.jpg`，目录结构正确
- [x] T7.8 **验收 B**：抽样 10 张裁剪图人工核对归属正确率 ≥ 9/10
- [x] T7.9 **验收 C**：以 `--N 1` 验收：结束日志 avg_fps ≥ 视频 fps（avg_fps 口径 = 处理帧数/总耗时，§5.3；N>1 时该口径会高于视频 fps，不作为本项判据，N>1 的吞吐观察归 T8.2）

### T8 边界与稳定性

- [ ] T8.1 §6 边界表 21 条逐条走查/构造验证，逐条记录结果
- [ ] T8.2 `--N 5` 与 `--N 1` 各完整跑一遍测试视频，均正常结束
- [ ] T8.3 长视频跑完，`top` 观察 RSS 无持续增长。长视频来源：用 ffmpeg 循环拼接测试视频生成 ≥10 分钟版本：`/usr/local/ffmpeg7/bin/ffmpeg -stream_loop 9 -i <测试视频> -c copy long.mp4`
- [ ] T8.4 非法参数三连：缺 `--video` / `--N 0` / `--conf 0.5`，均打印原因并退出码 1

每个 Task 完成即 git 提交一次（如已初始化仓库），或在日报中记录验收结果。

---

## 8. 验证资源

- 测试视频：`/home/Disk-6T/AI_Sentinel_data/old_gold_recycling_data/AI_Sentinel_Video_Standard_Library/customer_service/3810_125502.mp4`
- 引擎：`/home/Wilbur/project/performanceAnalysis/engines/batch1.int8.engine`
- 参考命令：

```bash
./build/bin/trackCrop \
  --video /home/Disk-6T/AI_Sentinel_data/old_gold_recycling_data/AI_Sentinel_Video_Standard_Library/customer_service/3810_125502.mp4 \
  --engine /home/Wilbur/project/performanceAnalysis/engines/batch1.int8.engine \
  --N 2 --conf 0.01 --ioa 0.5
```

---

## 9. 常见坑（提前告知）

1. **nvcc 版本**：系统 PATH 默认可能是 CUDA 12.1 的 nvcc，必须按 §3.2 强制 `/usr/local/cuda-12.6/bin/nvcc`，否则与 TensorRT 10.7 ABI 不匹配。
2. **FFmpeg 版本**：系统 `/usr/bin/ffmpeg` 是 6.x，**编译必须链接 7.x**（`/usr/local/ffmpeg7`），靠 pkg-config 路径前置保证，别用 `find_library` 乱找。
3. **OpenCV**：链接的是 `/home/Disk-6T/opencv/build` 下的 4.14 自编译版，**没有 CUDA 模块**，不要 include 任何 `opencv2/cuda*.hpp`。
4. **CUDA context**：解码器用 `primary_ctx=1`，预处理/推理 kernel 都跑在 device 0 primary context，不要在任何模块里 `cudaSetDevice` 到其它设备或自建 context。
5. **GpuFrame 生命周期**：frameRef 释放前 data0/data1 必须已用完（infer 与 nv12ToBgr 都完成后才能让 unique_ptr 析构）。
6. **tracker 输入格式**：是 **center-xywh** 不是 ltwh，转换错会导致追踪全乱（§5.3 已给出公式）。
7. **Eigen 对齐**：trackerCpp 与应用在同一工程同编译选项下构建，不要单独预编译后混链。
8. **临时代码**：T2/T4/T5/T6 的冒烟 main 放 `apps/smoke/`，验完即删，不许留在仓库里。
9. **AVFrame 释放**：一律 `av_frame_free`（§4.1 注释），shared_ptr 默认 delete 会导致泄漏/崩溃。

---

## 10. 审核修订记录

本文档与 README.md 分别经 `pi -p` 独立窗口审核，已落实修订：

1. **（高）接口 CUDA 耦合** → `Frame` 抽象基类，解码/推理接口只依赖 `Frame`，骨架期定稿。
2. **（高）`--conf 0.25` 破坏 ByteTrack 低分两段匹配** → 默认 0.01 + 启动校验超限退出。
3. **（高）ByteTrack 预测框污染疑虑** → 核对 trackers_cpp 源码后排除：`update()` 只输出本帧匹配/激活的 track，输出框为本帧真实检测框（§0 记录核实结论）。
4. **（中）common 头依赖 FFmpeg** → 前向声明 `AVFrame` + `Frame` 增加 `PixFmt` 枚举。
5. **（中）接口契约缺失** → device 不匹配返回空+告警；decoder 构造注入；色彩校验失败退出。
6. **（中）验收主观** → 可度量（处理 fps ≥ 视频 fps；抽样 10 张 ≥ 9/10）。
7. **（中）多对一/并列未定义** → 允许共享；并列取面积最小。
8. **（低）** `%06d.jpg`、stem 不含扩展名、覆盖策略、VFR 近似、README 图改串行回退、补 engine 来源/输入约束/运行环境。
9. 本版按"初级工程师可执行"标准重写：补编码约定、CMake 完整规格、逐文件改造点、伪代码、边界总表、Task 级验收、常见坑。
10. 二审修订：matchBest 阈值语义统一为 ≥ 且修正并列分支（§5.5）；补 AVFrame deleter 坑（§4.1/§9）；伪代码补 gpuFrame/trackId/head 来源与 snprintf sizeof（§5.3）；T2 验收改冒烟链接；冒烟代码统一 apps/smoke/ 验后即删；边界表补 display matrix 旋转/trackId float 精度/奇数尺寸（§6-19~21）；T7 计时依据落到结束日志；clamp 按 cv 闭区间语义注明（§5.4）。
11. 三审修订（按使用反馈）：§2 补全所有源文件的**绝对路径**（SRC_PA/SRC_TRK 缩写 + 逐文件路径表）；§7 由 Task 表格改为**可勾选原子任务清单**（T1.1~T8.4，完成一项划掉一项）。
12. 四审修订：T2.5/T2.6 相对路径冲突（smoke 改为 argv 传参 + 项目根运行）；冒烟任务补"CMake 加目标"半句；T4.6 指定 ffmpeg7 的 ffprobe；T7.9 明确 avg_fps 口径与 --N 1 判据；T8.3 补长视频生成命令。
