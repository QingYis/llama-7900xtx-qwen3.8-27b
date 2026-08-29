<div align="center">

**[English](README.md) | 简体中文**

</div>

# llama.cpp（ROCmFP4 移植版）

<div align="center">

**C/C++ LLM 推理引擎**

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

</div>

## 这个项目是什么

**本项目是官方 [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) 的移植增强版**：以官方 DFlash2 推测解码分支（[PR #27342](https://github.com/ggml-org/llama.cpp/pull/27342)，作者 SubSir）为基础，**新增了对 ROCmFP4（STRIX）GGUF 量化格式的完整支持**，其余代码与上游保持一致（MIT 许可）。

ROCmFP4 是 [ROCmFPX](https://github.com/charlie12345/ROCmFPX) 量化工具链面向 AMD RDNA 显卡的 4-bit GGUF 格式。移植前，官方 llama.cpp 无法加载这类模型；移植后，加载、量化、CPU 推理、HIP 全套 GPU 内核（MMVQ / MMQ / 反量化 / MoE 的 `MUL_MAT_ID`）全部可用。

### 格式要点（也是移植中最容易踩的坑）

- 每块 32 个值、18 字节（`_FAST` 变体 17 字节、单刻度）；
- **拆分 nibble 布局**：16 个字节的低 nibble 依次是第 0..15 个值，高 nibble 依次是第 16..31 个值（`qs[j] = val(j) | val(j+16) << 4`）——与 Q4_0/MXFP4 的"一字节两个连续值"不同，按旧习惯解码会得到"量级正常但数值全错"的结果；
- **半刻度 UE4M3**：scale 按标准无符号 E4M3 映射再除以 2（`2^(exp-9)`），CUDA 侧经 `rocmfp4_ue4m3_to_fp32_half()` 解码，`>0x7E`（inf/NaN）钳到 0。

## 快速开始（RX 7900 XTX / gfx1100 实测）

```bash
cmake -B build-rocmfp4 -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1100
cmake --build build-rocmfp4 -j

bash scripts/serve-dflash2.sh          # 一键拉起完整优化配置
```

模型路径、端口、上下文长度均可用环境变量覆盖：

```bash
MODEL=... DRAFT=... PORT=8080 CTX=131072 bash scripts/serve-dflash2.sh
```

## 推荐配置与优化原理（摘要）

完整逐条解释（含实测扫参数据）见 [docs/rocmfp4.md](docs/rocmfp4.md)。

| 配置 | 原理 |
|---|---|
| ROCmFP4 权重 | 4.5 bit/权重，RDNA3 上解码带宽占用大幅低于 f16 |
| `-ctk q4_0 -ctv q4_0` + `LLAMA_ATTN_ROT_DISABLE=1` | 混合架构只有 16 层全注意力；4-bit KV 让 256K 上下文从 ≈19 GiB 降到 ≈4.7 GiB，24G 卡才装得下；量化 KV 必须禁用 HIP 侧的旋转 KV 路径，否则崩溃 |
| `--spec-type ngram-map-k4v,draft-dflash` | 两个推测器叠加：n-gram 精确命中零成本优先，DFlash2 块扩散草稿兜底 |
| `--spec-draft-n-max 5 --spec-draft-p-min 0.4` | 扫参最优点（n5=76 t/s、p0.4=76 t/s）；低置信度尾部草稿的验证成本高于收益 |
| `--no-kv-unified --parallel 1` | DeltaNet 递归状态无法跨请求共享，统一 KV 会导致每轮全量重算历史（5.3 万 token ≈ 90 秒） |
| `--ctx-checkpoints 2 --cache-ram 4096` | 默认 32 个快照要吃 ≈10 GiB；2 个足够覆盖 n5 的回滚深度 |

### 实测性能（同卡同 prompt）

| 内容类型 | tok/s |
|---|---|
| ROCmFP4 纯基线 | 41 |
| + DFlash2，推理/思维链内容 | 68 |
| + 叠加 ngram-map-k4v，重复性内容 | 98 |

> **注意**：推测解码的接受率由**内容可预测性**主导（两个独立实现的草稿逐字节一致），创作类文本在任何配置下都只有 30-35 t/s——测速必须固定内容类型。

## 测试

- `test-backend-ops`：**12967/12967 全部通过**（含 k=5120/k=96 真实模型形状、J=16..128 的 MMQ 回归用例）。

## 致谢

- 格式与参考编解码：[ROCmFPX](https://github.com/charlie12345/ROCmFPX) / [rocmfp4-llama](https://github.com/charlie12345/rocmfp4-llama)（charlie12345）
- DFlash2 推测解码：上游 [PR #27342](https://github.com/ggml-org/llama.cpp/pull/27342)（SubSir）
- 移植集成、拆分 nibble 的 MMQ tile 布局与 RDNA3 验证：本仓库

## 许可

MIT（与上游 llama.cpp 一致）。上游的完整文档（构建矩阵、全部后端、REST API 等）请阅读 [英文 README](README.md)。
