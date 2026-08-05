# trackCrop debug 增强方案与执行计划

- Author: wilbur
- Version: 1.3
- Date: 2026-08-05
- 1.1: 按 pi 审核结果修订：H1 绘制与裁剪图隔离（clone 绘制+PNG hash 回归）、H2 空 track 帧编码路径、
  M1 编码 fps 用 double/av_d2q、M2 偶数宽高校验、M3 析构幂等 close、M4 BT.709 色彩矩阵与元数据、
  M5 编码器验证改日志+ffprobe、M6 nvenc 失败降级策略、L1~L6 细节补全。
- 1.2: 按用户要求改为**不使用 FFmpeg 编码器**，自研基于 NVIDIA Video Codec SDK（NVENC API）的
  硬编码模块：nvEncodeAPI.h（/usr/local/include/ffnvcodec）+ libnvidia-encode.so（驱动自带），
  CUDA 上下文直传 NV12 显存帧编码出 H.264 裸流；mp4 封装仅用 libavformat 做 mux（不涉及编码）。
- 1.3: 按二次 pi 审核结果修订：C1 mux 前逐 packet 做 Annex-B→AVCC 转换（扫 start code 改写
  长度前缀、剥离 AUD/SPS/PPS）；C2 extradata 手工构建 avcC（AVCDecoderConfigurationRecord）而非
  SPS/PPS 裸拼接；H1 补全 NVENC 调用链（NvEncodeAPIGetMaxSupportedVersion→CreateInstance→
  OpenEncodeSessionEx→GetEncodePresetConfig→InitializeEncoder→GetSequenceParams→
  CreateBitstreamBuffer→RegisterResource）；H2 补输出路径（LockBitstream→memcpy→Unlock）与销毁顺序；
  H3 关 B 帧（frameIntervalP=1、enableLookahead=0）保证 pts 单调；H4 IDR 置 AV_PKT_FLAG_KEY；
  M1 回改 av_d2q；M2 enableEncodeAsync=0；M3 pts=dts=帧计数、duration=1；M4 EOS flush 细节；
  M5 codecpar 色彩元数据 BT.709；L1 repeatSPSPPS=1；L2 版本不符走降级；L3 pitch/显存高度细节；
  L4 验证补 NVDEC 闭环解码、N>1 帧号映射、降级路径。
  复审补充：Register 路径下每帧 Map/Unmap 必须配对（nvEncUnmapInputResource），销毁顺序含 unmap。

## 一、需求背景

当前 `--debug` 只打印检测明细、tracker 三轮匹配明细、每个 track 的最终归属结果（`track 15 <- cls2 ioa=1.0`）。
**缺失 IoA 归属匹配的中间过程**：无法回答"track 15 为什么没有匹配到 cls0？cls2 匹配的是哪个框？每个候选框的 IoA 各是多少？"。

需求拆解：

1. **增强 debug 打印**：把每个 track 与每个 body 候选框的 IoA 计算过程完整打印，能定位"没匹配到 / 匹配到哪个 / 为什么"。
2. **debug 模式输出硬编码（NVENC）可视化视频**：把 debug 信息画到帧画面上，用硬件编码（h264_nvenc）保存成 mp4，与日志对照排查。

## 二、总体设计

### 2.1 增强 IoA 匹配 debug 打印

**改动点**：`apps/trackCrop/trackCropMain.cpp` 的 `matchBest()` 与逐 track 匹配循环。

`matchBest` 增加可选输出参数，回写每个候选的明细结构体：

```cpp
struct MatchCand {
    size_t origIdx;   // 该候选在当帧全量 dets 列表中的下标（对应日志 det#N）
    int    candIdx;   // 在 cls0/cls2 子列表中的下标
    float  conf;
    float  x1, y1, x2, y2;
    float  ioa;       // clamp 后零面积的候选记 -1，打印 skipped(invalid box)
    bool   hit;       // ioa >= thresh
    bool   chosen;    // 最终选中
};
```

`det#N` 序号来源：分流（heads/bodies0/bodies2）时用 `pair<size_t, Detection>` 携带全量列表原始下标，保证日志中 `det#N` 与顶部全量 det 打印行对号入座。

每个 track 的打印格式（debug 开启时）：

```
[DEBUG] frame 105 track 15 head=(717.9,69.5,771.6,113.2) score=0.2899
[DEBUG]   cls0 候选=7 (thresh=0.5):
[DEBUG]     cls0[0] det#3  conf=0.7930 box=(1213.9,187.5,1352.3,598.0) ioa=0.0000 miss
[DEBUG]     cls0[1] det#5  conf=0.6931 box=(915.5,190.9,1077.5,703.0)  ioa=0.0000 miss
[DEBUG]     ...（全部候选逐个列出，含 ioa < thresh 的；零面积候选标 skipped(invalid box)）
[DEBUG]   cls0 无命中（最高 ioa=0.1234 @ cls0[2] det#5，< thresh 0.5）-> 回退 cls2
[DEBUG]   cls2 候选=11 (thresh=0.5):
[DEBUG]     cls2[3] det#21 conf=0.0336 box=(695.8,68.8,774.8,189.8)    ioa=1.0000 HIT <= 选中
[DEBUG]   => track 15 命中 cls2[3] det#21 ioa=1.0000
[DEBUG] frame 105 track 15 <- cls2 ioa=1.000000 -> 000105_cls2.png   (原有最终行保留)
```

说明：
- `miss`/`HIT` 标注是否过阈值；选中行追加 `<= 选中`。
- cls0 无命中时打印原因行（最高 IoA 及对应候选、与阈值比较），再进入 cls2 回退。
- 两类都无命中时打印 `no body match: cls0 最高 ioa=xx (@ det#N), cls2 最高 ioa=xx (@ det#M), thresh=0.5`（替代现在的 `no body match, skipped`）。
- 被 `--min-size` 过滤的分支现状已有日志，保持不变。
- 非 debug 路径开销仅为一次 `nullptr` 判断，无额外分配。

### 2.2 debug 可视化视频（NVENC 硬编码）

**新增模块**：`modules/encoder/nvencEncoder.h / nvencEncoder.cpp`，加入 `otaCore` 静态库。

**不使用 FFmpeg 的编码器**。直接调用 NVIDIA Video Codec SDK（NVENC API）：

- 头文件：`/usr/local/include/ffnvcodec/nvEncodeAPI.h`（系统已装 nv-codec-headers）；链接 `libnvidia-encode.so`（驱动自带，项目已链 `CUDA::cuda_driver`）。
- CUDA 上下文：复用进程内 CUDA primary context（`cudaFree(0)` 初始化 + `cuCtxGetCurrent`），NVENC 以 `NV_ENC_DEVICE_TYPE_CUDA` 打开 session（与解码侧 nvdecDecoder 的 primary_ctx=1 用法一致）。

**NVENC 完整调用链（open 时）**：
```
NvEncodeAPIGetMaxSupportedVersion          // 校验驱动版本；低于头文件版本 → 报错走降级（M6）
NvEncodeAPICreateInstance                  // 填充 NV_ENCODE_API_FUNCTION_LIST 函数表（所有 nvEnc* 的前提）
nvEncOpenEncodeSessionEx(NV_ENC_DEVICE_TYPE_CUDA, primaryCtx)
nvEncGetEncodePresetConfig(P4) -> 修改 -> nvEncInitializeEncoder
nvEncGetSequenceParams                     // 取 SPS/PPS，用于构建 avcC extradata
nvEncCreateBitstreamBuffer                 // 输出 bitstream buffer
nvEncRegisterResource                      // 注册预分配的 CUDA NV12 输入 buffer
```

**编码参数**：`NV_ENC_CODEC_H264_GUID`，`NV_ENC_PRESET_P4_GUID`，`NV_ENC_PARAMS_RC_VBR` + `averageBitRate` 按分辨率估算（如 1080p 约 4Mbps）+ `maxBitRate`=1.5×avg，`gopLength = 2*fps`；**关 B 帧与 lookahead**：`frameIntervalP=1`、`rcParams.enableLookahead=0`（保证输出即顺序、pts 按帧计数器单调）；**同步模式**：`enableEncodeAsync=0`（单线程 debug 工具，无需 event）；`repeatSPSPPS=1`（零成本提鲁棒性）；帧率用 `AVRational fr = av_d2q(fps, 1001000)`，`frameRateNum=fr.num / frameRateDen=fr.den`（避免 NTSC 30000/1001 截断漂移），mux 侧 `time_base = av_inv_q(fr)` 共用同一有理数；VUI 写入 BT.709（colour_primaries/transfer/matrix）。

**每帧编码流程（writeFrame）**：
```
BGR（CPU 绘制后）-> 按 BT.709 系数手动转 NV12（CPU）-> cudaMemcpy2D 上传到
预分配的 cudaMallocPitch 显存 buffer（pitch 传入 NV_ENC_REGISTER_RESOURCE；
NV12 buffer 总高 = height*3/2，chroma 平面偏移 pitch*height）->
nvEncMapInputResource -> nvEncEncodePicture -> nvEncUnmapInputResource
（已走 Register 路径，Map/Unmap 每帧必须配对；Encode 返回后即可 Unmap，
与 LockBitstream 顺序无关）->
nvEncLockBitstream -> memcpy(bitstreamSizeInBytes) 取出 Annex-B 裸流 ->
nvEncUnlockBitstream -> annexbToAvcc 转换 -> av_interleaved_write_frame
```

**EOS flush（close 时）**：`picParams.inputBuffer=NULL` + `encodePicFlags=NV_ENC_PIC_FLAG_EOS`，循环取包直至 `bitstreamSizeInBytes==0`；销毁顺序：flush → unmap（若仍持有）→ unlock → `nvEncDestroyBitstreamBuffer` → `nvEncUnregisterResource` → `nvEncDestroyEncoder` → mux write trailer。

**Annex-B → AVCC 转换（mux 前必做，缺它 mp4 必坏）**：NVENC 对 H.264 固定输出 Annex-B（start code 分隔），MP4 要求 AVCC（4 字节大端长度前缀）。实现 `annexbToAvcc()`：扫 start code（0x000001/0x00000001）定位 NAL 边界，改写为长度前缀；packet 内剥离 AUD(nal=9)/SPS(7)/PPS(8)（SPS/PPS 已进 extradata 且 repeatSPSPPS 重复出现，mp4 包内只留 VCL 与 SEI）。

**avcC extradata 构造**：mov muxer 要求 extradata 是 AVCDecoderConfigurationRecord（extradata[0]==1），非 SPS/PPS 裸拼接。用 `nvEncGetSequenceParams` 取得的 SPS/PPS 手工构建：version=1、profile/compat/level 取 SPS 负载[1..3]、lengthSizeMinusOne=3、numSPS|0xE0 + SPS 列表 + numPPS + PPS 列表；`codec_tag=0`。

**packet 属性**：每次 `av_new_packet` 新包；`pts = dts = 帧计数器`（time_base=1/fps）、`duration=1`（已关 B 帧故无重排）；lockBitstream 后按 `sliceType == NV_ENC_PIC_TYPE_IDR` 置 `pkt.flags |= AV_PKT_FLAG_KEY`（mov muxer 建 stss 表所需，缺则无法 seek）；`av_interleaved_write_frame` 接管包所有权。

**mux（容器封装）**：H.264 裸流转 AVCC 后用 **libavformat 仅做 mux** 写 mp4——FFmpeg 在此**只负责容器，不参与编码**。`codecpar` 除 avcC 外同步色彩元数据：`color_primaries/trc/space = BT709`、`color_range` 与 BGR→NV12 转换约定（limited/studio swing）一致，与码流 VUI 配套。
- 接口：`bool open(int width, int height, double fps, const std::string& outPath)` / `bool writeFrame(const cv::Mat& bgr)` / `void close()`（nvEncEncodePicture flush 排空 + EOS + trailer）。
- **fps 用 double 传入**；`fps <= 0` 时 open 报错返回 false。
- **奇数宽高校验**：NV12/h264 要求宽高均为偶数，open 时校验，不满足明确报错（不做裁剪兼容）。
- **RAII**：`~NvencEncoder()` 幂等调用 `close()`（内部 `opened_` 标志），异常路径也能 flush + 写 trailer 保住已编码部分。
- **BGR→NV12 色彩矩阵**：与解码侧 `nv12ToBgr.cu` 一致用 **BT.709 系数**（OpenCV 无现成 709 的 I420 枚举，手动按 709 系数转换并打包 NV12），码流 VUI 元数据与矩阵配套、播放器不二次偏移。

**触发条件与降级**：仅 `--debug` 开启时创建。open/writeFrame 失败 → `logError` + 置空编码器**继续跑**（debug 日志能力不受视频失败影响）。输出路径固定：`output/<视频名stem>/<视频名stem>_debug.mp4`。

**NVDEC/NVENC 共存**：解码（NVDEC，经 FFmpeg hwaccel）与编码（NVENC，直接 SDK）是 GPU 上相互独立的硬件引擎，可并行共存；两者不共享 CUDA 上下文冲突（primary context 复用）。

**编码哪些帧**：只编码参与推理/追踪的帧（`frameId % N == 0`），输出视频 fps = `decoder.fps() / N`。
**视频第 i 帧 ⇔ 原始 frameId = i * N**（帧号换算关系，抽查对照时用）。

**空 track 帧的编码路径（重点）**：debug 模式下 `nv12ToBgr` 下载点**提前**到 infer/track 之后、`tracks.rows() == 0` 的提前 continue **之前**；空 track 帧照常画 det 框+信息面板并送编码器，然后再 continue。保证"视频帧与日志帧号严格对齐"。

**绘制与裁剪图隔离（重点）**：裁剪 PNG 的 `imwrite` 全部使用未绘制的 `bgr`；绘制发生在 `cv::Mat bgrDraw = bgr.clone()` 副本上（在当帧全部 crop 保存完成之后进行），绝不在原始 `bgr` 上画。debug 模式产出的裁剪 PNG 必须与非 debug 模式逐文件 hash 一致。

**画面内容**（在 bgrDraw 上用 OpenCV 绘制后送编码器）：

| 元素 | 颜色/样式 | 标注文字 |
|---|---|---|
| cls0 fullBody 检测框 | 绿色细框 | `det#N c0 0.79` |
| cls1 head 检测框 | 黄色细框 | `det#N c1 0.80` |
| cls2 upperBody 检测框 | 青色细框 | `det#N c2 0.82` |
| track 输出的 head 框 | 品红粗框(2px) | `id=15 0.29` |
| 归属连线 | 品红 | head 中心 → 命中 body 框中心连线，中点标 `ioa=1.00` |
| 命中的 body 框 | 品红粗框描边叠加 | `id=15<-c2` |
| 未命中 track | 红色粗框 | `id=18 NO-MATCH` |
| min-size 过滤的 track | 橙色粗框 | `id=X SMALL 43x61` |
| 左上角信息面板 | 半透明黑底白字 | 见下 |

信息面板逐行内容（与日志行对应）：
```
frame 105  dets=28 heads=10 cls0=7 cls2=11  tracks=5
match: id=3<-c0(1.00) id=14<-c0(1.00) id=15<-c2(1.00) id=18:X id=8<-c2(1.00)
saved=4  decode=1.3 infer=1.0 track=0.1 save=7.1 ms
```

逐 track 的 IoA 全候选明细不画进视频（画面会过于拥挤），该类细节查日志；视频负责"空间关系"（谁和谁连上了、框在哪里），日志负责"数值过程"，两者通过 frame 号 + det#N 序号对齐。

### 2.3 文件变更清单

| 文件 | 变更 |
|---|---|
| `apps/trackCrop/trackCropMain.cpp` | M（v1.4 → v1.5）：分流携带原始下标、matchBest 回写候选明细、debug 打印增强、debug 时提前下载+clone 绘制+送编码、空 track 帧编码路径 |
| `modules/encoder/nvencEncoder.h` | A（v1.0）：编码器接口 |
| `modules/encoder/nvencEncoder.cpp` | A（v1.0）：NVENC SDK 直接调用实现（H.264 裸流编码）+ libavformat 仅 mux mp4（含 RAII、pts、BT.709 转换、VUI 元数据） |
| `CMakeLists.txt` | M（v1.0 → v1.1）：nvencEncoder.cpp 加入 otaCore、`target_include_directories` 增加 modules/encoder 与 /usr/local/include/ffnvcodec；链接 `libnvidia-encode.so`（libavformat 已链接，mux 无需新增） |

## 三、执行计划（TODO）

1. **新增 nvencEncoder 模块（NVENC SDK 直接调用）** → 验证：编译通过；`ldd` 确认链接 libnvidia-encode；debug 跑 N=1 与 N=3 两组用例能产出 mp4；运行日志打印 NVENC API 版本与 preset；`ffprobe -show_streams` 校验 `codec_name=h264`、`nb_frames == processed`、`duration ≈ processed / (fps/N)`、色彩元数据为 bt709；**闭环验证：用项目自己的 NVDEC decoder 解码产出 mp4，帧数==processed 且抽查像素容差**（兜住 Annex-B→AVCC / avcC / pts / KEY 标志全部 mux 类错误）；**降级路径验证**：人为制造 open 失败（如临时错误参数）确认 logError 后流程继续、日志与裁剪图不受影响。
2. **matchBest 增加候选明细回写 + debug 打印增强** → 验证：debug 跑同一视频，frame 105 的日志能看到 track 15 对全部 7 个 cls0 候选与 11 个 cls2 候选的 ioa 逐条记录（含 det#N 与全量 det 行对齐），能回答"为什么没匹配 cls0 / cls2 匹配了哪个 det"；无命中 track 打印双侧最高 ioa 原因行。
3. **debug 视频绘制（框/连线/面板）+ 接入编码器** → 验证：抽查原始 frame 105（须 105%N==0，对应视频第 105/N 帧）：28 个 det 框、5 个 track 框、4 条归属连线、id=18 标 NO-MATCH、面板数值与日志一致；空 track 帧在视频中存在且只有 det 框+面板、无连线；N=3 时验证帧号映射落地（抽视频第 1 帧对照日志 frame 3 的 det 框）。
4. **回归验证** → 验证：① 不带 `--debug` 跑一遍，无视频产出、无新打印、裁剪图与改动前一致；② **带 `--debug` 跑一遍，裁剪 PNG 与非 debug 逐文件 hash 一致**（验证绘制未污染 crop）。
5. **更新文件头版本与 description，提交** → 验证：git 提交按个人 commit 规范。

## 四、已定的关键取舍（如需调整请指出）

- **只编码处理帧**（`frameId % N == 0`），不编码跳过帧 —— 保证视频帧与日志帧号严格对齐（视频第 i 帧 ⇔ 原始 frameId = i*N）。
- **编码不用 FFmpeg**：直接调 NVENC SDK（nvEncodeAPI.h + libnvidia-encode.so），H.264 裸流自产出；FFmpeg 仅以 libavformat 做 mp4 mux（容器封装，非编码）。H.264 而非 HEVC —— 兼容性优先；PRESET_P4 + RC_VBR；关 B 帧/lookahead、同步模式，pts=dts=帧计数器。
- **CUDA 上下文复用 primary context**，与 NVDEC 解码链路共存（独立硬件引擎）。
- **绘制在 CPU BGR 上做**：debug 模式下每个处理帧强制 `nv12ToBgr` 下载一次（含空 track 帧），绘制在 `bgr.clone()` 副本上进行，crop PNG 用未绘制原图。
- **BT.709 色彩矩阵**与解码侧一致，AVFrame 元数据配套。
- **nvenc open/写帧失败降级**：logError 后继续跑，不影响 debug 日志与裁剪图输出。
- **IoA 全候选明细只进日志不进视频**，视频只画最终归属关系。
- debug 视频固定保存在 `output/<stem>/<stem>_debug.mp4`，不加命令行参数。
