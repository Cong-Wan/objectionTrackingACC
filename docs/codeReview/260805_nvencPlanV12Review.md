# 代码审核报告 — debugEnhancePlan.md v1.2（NVENC SDK 直调 + libavformat mux）

- 审核日期：2026-08-05
- 审核对象：docs/debugEnhancePlan.md v1.2
- 环境核实：nvEncodeAPI.h 在 /usr/local/include/ffnvcodec；libnvidia-encode.so 可链接；
  解码侧 nvdecDecoder.cpp 确实使用 av_hwdevice_ctx_create + primary_ctx=1（与计划的 primary context 复用一致）。

## 总览

- 发现问题：🔴 2 / 🟠 4 / 🟡 5 / 🔵 4，共 15 个
- 整体评价：方向正确，但恰好漏掉 mux H.264 裸流进 mp4 的两个致命坑
  （Annex-B→AVCC 转换、avcC extradata），以及 NVENC 入口（函数表初始化）与输出 bitstream 路径。
  按现计划直接实现，产出 mp4 必坏。v1.2 不可直接执行，建议修订为 v1.3。

---

## 🔴 Critical

### C1. 缺失 Annex-B → AVCC 转换（packet 数据层）

NVENC 对 H.264 固定输出 Annex-B（start code 分隔），MP4 要求 AVCC（4 字节大端长度前缀）。
计划只提 extradata，未提 packet payload 转换；直接 mux 会得到 NAL 边界全错的 mp4。
修复：实现 annexbToAvcc（扫 start code → 改写长度前缀），并剥离 AUD(9)/SPS(7)/PPS(8)。
计划文档需补：「mux 前必须逐 packet 转 AVCC 长度前缀格式」。

### C2. extradata 必须是 AVCDecoderConfigurationRecord（avcC），非 SPS/PPS 裸拼接

mov muxer 的 ff_isom_write_avcc 要求 extradata[0]==1 的 avcC 结构；裸 Annex-B 字节会写出
损坏的 avcC box。修复：用 nvEncGetSequenceParams 取得的 SPS/PPS 手工构建 avcC
（version=1, profile/compat/level 取 SPS[1..3], lengthSizeMinusOne=3, numSPS|0xE0 + SPS 列表 + PPS 列表）。

---

## 🟠 High

### H1. 缺失 NvEncodeAPICreateInstance 函数表初始化

nvEnc* 全部是 NV_ENCODE_API_FUNCTION_LIST 成员指针，必须先
NvEncodeAPIGetMaxSupportedVersion（校验/钳制驱动版本）→ NvEncodeAPICreateInstance，
否则一个 API 都调不了。计划需补完整调用链：
CreateInstance → nvEncOpenEncodeSessionEx(CUDA, primaryCtx) → nvEncGetEncodePresetConfig(P4)
→ 修改 → nvEncInitializeEncoder → nvEncGetSequenceParams → nvEncCreateBitstreamBuffer
→ nvEncRegisterResource。

### H2. 输出 bitstream 路径缺失

每帧输出链未提：nvEncCreateBitstreamBuffer → picParams.outputBitstream →
nvEncLockBitstream → memcpy(bitstreamSizeInBytes) → nvEncUnlockBitstream。
销毁顺序：EOS flush → unlock → destroyBitstreamBuffer → unregisterResource → destroyEncoder。

### H3. B 帧乱序未声明

preset config 可能 frameIntervalP>1（B 帧），输出重排后"pts 按帧计数器"会产生非单调 pts。
修复：显式 encodeConfig.frameIntervalP=1、rcParams.enableLookahead=0（debug 用途推荐），
或改用 NV_ENC_LOCK_BITSTREAM::outputTimeStamp + dts 处理。

### H4. 缺 AV_PKT_FLAG_KEY

mov muxer 靠 key 标志建 stss 表，不设则无法 seek/起播黑屏。
修复：lockBitstream 后按 sliceType == NV_ENC_PIC_TYPE_IDR 置 pkt.flags |= AV_PKT_FLAG_KEY。

---

## 🟡 Medium

### M1. fps*1000/1000 是 v1.1 M1（av_d2q）的回退

NTSC 30000/1001 会被截断为 29970/1000，duration 漂移。
修复：AVRational fr = av_d2q(fps, 1001000)；编码器 frameRate 与 mux time_base=av_inv_q(fr) 共用。

### M2. 同步/异步模式未声明

enableEncodeAsync 默认 1（需 event）。单线程 debug 工具显式 enableEncodeAsync=0。

### M3. packet 缺 dts/duration 与所有权说明

pkt.pts = pkt.dts = frameIdx; pkt.duration = 1；av_interleaved_write_frame 接管引用，
每次 av_new_packet 新包。

### M4. EOS flush 细节

picParams.inputBuffer=NULL; encodePicFlags=NV_ENC_PIC_FLAG_EOS；循环至 bitstreamSizeInBytes==0。

### M5. mux 侧色彩元数据

codecpar->color_primaries/trc/space = BT709、color_range 与 BGR→NV12 转换约定
（limited/studio swing）一致，写 colr box 与 VUI 配套。

---

## 🔵 Low

- L1. NV_ENC_CONFIG_H264::repeatSPSPPS=1（鲁棒性，成本为零）。
- L2. 驱动低于头文件 NVENCAPI_VERSION 时明确报错并触发 M6 降级，而非 INVALID_VERSION 一退了之。
- L3. 注册资源 pitch 细节：cudaMallocPitch 的 pitch 传入 NV_ENC_REGISTER_RESOURCE；
  NV12 buffer 总高 = height*3/2（chroma 偏移 pitch*height）。
- L4. 验证补充：① 闭环验证——用项目 NVDEC decoder 解码产出 mp4，帧数==processed 且抽查像素容差
  （可兜住 C1/C2/H3/H4 全部 mux 错误）；② N=3 时验证"视频第 i 帧 ⇔ frameId=i*N"落地
  （抽视频第 1 帧对照日志 frame 3 的 det 框）；③ NVENC open 失败降级路径无验证步骤，建议补。

---

## 优点记录

- primary context 复用与解码侧实现核对一致，NVDEC/NVENC 共存判断正确；
- 降级策略、RAII 幂等 close、绘制/裁剪隔离 + PNG hash 回归、det#N 序号对齐设计成熟；
- nvEncGetSequenceParams 取 SPS/PPS、VUI 写 BT.709 方向正确。

## 修复优先级 Top 3

1. C1+C2（Annex-B→AVCC + avcC）——不修产物必坏；
2. H1+H2（函数表初始化 + bitstream 输出路径）——计划 API 链断头尾，照计划写不出能跑的代码；
3. H3+H4（B 帧 + KEY 标志）——特定条件下暴露，排查成本高。
