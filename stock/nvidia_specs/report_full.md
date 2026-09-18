# NVIDIA 数据中心 AI GPU 精确规格表（面向寒武纪 MLU 对比）
**数据截止：2026-09-18｜全部数字标注来源口径：`NVIDIA官方` / `第三方`**

---

## 0. 读表须知：dense vs sparse —— 最重要的一条

NVIDIA 官网产品页、datasheet 与新闻稿里的 Tensor Core 峰值**默认按 2:4 结构化稀疏（structured sparsity）报数**；稠密（dense）值通常是该数字的 **1/2**。本报告**一律把两者分列**，表格中写作 `密集 | 稀疏`。

官方原文脚注（可逐字核对）：

| 产品 | 官方原文脚注 | 来源 |
|---|---|---|
| A100 | `* With sparsity`；中文版 `*采用稀疏技术显示。在不采用稀疏技术的情况下，规格降低一半。` | A100 datasheet |
| H100 | `*With sparsity`；中文版 `*采用稀疏技术显示。在不采用稀疏技术的情况下，规格降低一半。` | H100 datasheet |
| H200 | `2 With sparsity.` | H200 产品页 |
| GB200 NVL72 | `2. Specification in sparse. Dense is one-half sparse spec shown.` | GB200 NVL72 产品页 |
| HGX B200/B300 | `1. Specification in Sparse \| Dense`；`2. Specification in sparse. Dense is ½ sparse spec shown.` | HGX 页 |
| Blackwell Ultra | `1. Specification in Sparse \| Dense`；`2. Specification in sparse. Dense is ½ sparse spec shown.` | Blackwell Ultra datasheet |

> ⚠️ **Blackwell 一代的关键例外**：B200/B300 的 **FP4** 在 NVIDIA 官方页被写成「稀疏 \| 密集」两个并列数，且比值**不是 2.0 而是约 1.33**（B200：144 sparse / 72 dense；B300：144 sparse / 108 dense）。而同一张表里的 FP8/FP16/TF32/INT8 仍按 sparse 单列（dense = 1/2）。**这是本报告发现的最大口径陷阱**，详见 §4。

---

## 1. 总表 A：算力（TFLOPS / TOPS）—— 全部按 `密集 | 稀疏` 分列

### 1.1 FP16 / BF16 Tensor Core（`密集 | 稀疏`）

| 产品 | FP16/BF16 密集 | FP16/BF16 稀疏 | 口径来源 |
|---|---|---|---|
| A100 40GB PCIe | **312** | **624** | NVIDIA官方 datasheet |
| A100 80GB PCIe | 312 | 624 | NVIDIA官方 datasheet |
| A100 40GB SXM | 312 | 624 | NVIDIA官方 datasheet |
| A100 80GB SXM | 312 | 624 | NVIDIA官方 datasheet |
| H100 SXM | **989** | **1,979** | NVIDIA官方 datasheet（sparse 为表内值，dense 由脚注折半） |
| H100 PCIe | 800 | 1,600 | NVIDIA官方 中文 datasheet |
| H100 NVL（PCIe 双卡桥接） | 835.5 | 1,671 | NVIDIA官方 datasheet |
| H800 SXM | 989 | 1,979 | 第三方（强川科技产品页，声称对齐 H100 SXM） |
| H800 PCIe | ~756 | ~1,513 | 第三方（Lenovo ThinkSystem，产品页已下架） |
| H20（96GB / 141GB） | **148** | **296**（推算；NVIDIA 未公布 sparse） | 第三方（LMSYS/Ant、腾讯科技；**148 为 dense**，见 §E.1） |
| H200 SXM | 989 | 1,979 | NVIDIA官方 产品页 + datasheet |
| H200 NVL（PCIe） | 835.5 | 1,671 | NVIDIA官方 datasheet |
| B200（HGX B200 单卡） | **2,250** | **4,500** | NVIDIA官方 Blackwell Tech Brief（`2.2/4.5 petaFLOPS`） |
| GB200 内 GPU（NVL72 单卡摊算） | 2,500 | 5,000 | NVIDIA官方 GB200 产品页整柜值 ÷72 |
| B300 / Blackwell Ultra（HGX B300 单卡） | **4,500** | **9,000** | NVIDIA官方 Blackwell Ultra datasheet |
| GB300 内 GPU（NVL72 单卡摊算） | 5,000 | 10,000 | NVIDIA官方 GB300 产品页整柜值 ÷72 |
| B30A（中国特供，单 die） | **未找到公开数据**（估算区间 1.125–2.5 PFLOPS，口径混乱，见 §D.4） | 未找到公开数据 | **第三方推测，冲突达 2.2×** |
| RTX 6000D（中国特供工作站） | **144.09**（实测 dense GEMM，第三方） | 未找到公开数据 | 第三方受控实测（CSDN 2026-07-14） |
| A800 40GB Active | 312 | 624 | NVIDIA官方 产品页（`623.8 TFLOPS` 峰值 Tensor = 稀疏） |
| L40S | **362.5** | **733** | NVIDIA官方 L40S 中文 datasheet（`362.5 \| 733*`） |
| L20 | **119**（第三方阿里云，疑为 dense；无官方） | 未找到公开数据 | 第三方/OEM 伙伴（阿里云官方文档） |
| MI300X（AMD，对照） | 1,300 | 2,610 | AMD官方 产品页（明确区分 with/without structured sparsity） |

### 1.2 FP8 / FP6 Tensor Core（`密集 | 稀疏`）

| 产品 | FP8 密集 | FP8 稀疏 | 备注 |
|---|---|---|---|
| A100 全系 | **不支持** | **不支持** | 架构为 Ampere，无 FP8 Tensor Core |
| H100 SXM | **1,979** | **3,958** | NVIDIA官方 datasheet |
| H100 PCIe | 1,600 | 3,200 | NVIDIA官方 中文 datasheet |
| H100 NVL | 1,670.5 | 3,341 | NVIDIA官方 datasheet |
| H800 SXM | 1,979 | 3,958 | 第三方（对齐 H100 SXM） |
| H800 PCIe | ~1,513 | ~3,026 | 第三方（Lenovo） |
| H20 | **296** | **592**（第三方；NVIDIA 未公布 sparse） | 第三方（flopper.io 显式分列；**296 为 dense**，见 §E.1） |
| H200 SXM | 1,979 | 3,958 | NVIDIA官方 |
| H200 NVL | 1,670.5 | 3,341 | NVIDIA官方 datasheet |
| B200 单卡 | **2,250** | **4,500** | NVIDIA官方 Blackwell Tech Brief（`4.5/9 petaFLOPS`，FP8/FP6 合并） |
| GB200 内 GPU | 2,500 | 5,000 | NVIDIA官方 GB200 产品页 ÷72 |
| B300 单卡 | **4,500** | **9,000** | NVIDIA官方 Blackwell Ultra datasheet |
| GB300 内 GPU | 5,000 | 10,000 | NVIDIA官方 GB300 产品页 ÷72 |
| B30A | 未找到公开数据 | 未找到公开数据 |
| RTX 6000D | **未找到公开数据**（实测 dense BF16 GEMM 仅 144.09 TFLOPS，见 §D.7） | 未找到公开数据 |
| L40S | **733** | **1,466** | NVIDIA官方 L40S 中文 datasheet |
| L20 | **237**（第三方阿里云；无官方） | 未找到公开数据 | OEM 伙伴（阿里云官方文档） |
| A800 40GB Active | 不支持 | 不支持 | 同 A100（Ampere 无 FP8） |
| MI300X | 2,610 | 5,220 | AMD官方 |

### 1.3 FP4 Tensor Core（`密集 | 稀疏`）—— Blackwell 及以后才有

| 产品 | FP4 密集 | FP4 稀疏 | 官方脚注口径 |
|---|---|---|---|
| H100 / H200 / H20 / H800 | 不支持 | 不支持 | Hopper 无 FP4 |
| B200 单卡（HGX B200） | **9 PFLOPS** | **18 PFLOPS** | HGX 页：`FP4 Tensor Core¹ 144 \| 72 PFLOPS`（8卡）→ 单卡 18 sparse / 9 dense |
| GB200 内 GPU | 10 PFLOPS | 20 PFLOPS | GB200 产品页整柜 `1,440 \| 720 PFLOPS` ÷72 |
| B300 单卡（HGX B300） | **13.5 PFLOPS** | **18 PFLOPS** | HGX 页：`144 \| 108 PFLOPS`（8卡）；Ultra datasheet：`14/18`（HGX 列） |
| GB300 内 GPU | 15 PFLOPS | 20 PFLOPS | GB300 产品页整柜 `1440 \| 1080 PFLOPS` ÷72；Ultra datasheet 单卡 `20 \| 15 PFLOPS` |
| B30A | 未找到公开数据（TPU 推算 ~7.5 PFLOPS） | — | **第三方推测** |
| RTX PRO 6000 Blackwell（96GB 全血版） | 未找到公开数据 | 未找到公开数据 | NVIDIA官方产品页未列算力 |
| RTX 6000D（中国特供） | 未找到公开数据（仅全血版有 "up to 4,000 AI TOPS" 聚合营销值） | — | NVIDIA官方无页面 |

> 🔴 **B300/B200 FP4 口径冲突警告**：HGX 页把 FP4 写成 `144 \| 108 PFLOPS`（B300）与 `144 \| 72 PFLOPS`（B200），脚注 1 是 `Specification in Sparse | Dense`。按此，B300 的 **dense FP4 = 108/8 = 13.5 PFLOPS**，稀疏 = 18 PFLOPS，比值仅 1.33，**不是 2×**。但 Blackwell Ultra datasheet 的「Individual GPU Specifications」表把 B300 单卡写成 `20 PFLOPS | 15 PFLOPS`（GB300 列）与 `18 PFLOPS | 14 PFLOPS`（HGX 列），整柜汇总又用 `1,440 \| 1,080`（72×20 / 72×15）。**同一份官方 datasheet 内部并不自洽**，详见 §4.1。

### 1.4 INT8 Tensor Core（`密集 | 稀疏`）

| 产品 | INT8 密集（TOPS） | INT8 稀疏（TOPS） |
|---|---|---|
| A100 全系 | **624** | **1,248** |
| H100 SXM | 1,979 | 3,958 |
| H100 PCIe | 1,600 | 3,200 |
| H100 NVL | 1,670.5 | 3,341 |
| H800 SXM | 1,979 | 3,958 |
| H20 | **296** | 592（第三方） |
| H200 SXM | 1,979 | 3,958 |
| H200 NVL | 1,670.5 | 3,341 |
| B200 单卡 | 2,250 | 4,500（`4.5/9 petaOPS`） |
| B300 单卡（HGX） | 153.75 | 307.5（Ultra datasheet `307 TOPS` sparse / 8卡） |
| GB300 NVL72 整柜 | 12,000 POPS | 24,000 POPS（= 12 \| 24 petaOPS） |
| GB200 NVL72 整柜 | 360,000 POPS | 720,000 POPS（= 360 \| 720 petaOPS） |
| B30A | 未找到公开数据 | — |
| RTX 6000D | 未找到公开数据（**实测 140.32 TOPS**，第三方） | 未找到公开数据 |
| L40S | **733** | **1,466** |
| L20 | **237**（第三方阿里云） | 未找到公开数据 |
| MI300X | 2,600 POPS | 5,220 POPS |

### 1.5 TF32 Tensor Core 与 FP32 / FP64（非 Tensor）

| 产品 | TF32 密集 | TF32 稀疏 | FP32（CUDA core） | FP64 | FP64 Tensor |
|---|---|---|---|---|---|
| A100 全系 | 156 | 312 | **19.5** | 9.7 | 19.5 |
| H100 SXM | 494.5 | 989 | **67** | 34 | 67 |
| H100 PCIe | 400 | 800 | **48** | 24 | 48 |
| H100 NVL | 417.5 | 835 | 60（datasheet） | 30（datasheet） | 60 |
| H800 SXM | 494.5 | 989 | 67 | 1（第三方列出，**存疑**） | — |
| H800 PCIe | ~378 | ~756 | 未找到公开数据 | **0.8**（第三方/Lenovo） | — |
| H20 | **74** | 未公布 | **44** | 1 | — |
| H200 SXM | 494.5 | 989 | **67** | 34 | 67 |
| H200 NVL | 417.5 | 835 | 60 | 30 | 60 |
| B200 单卡（HGX） | 1,100 | 2,200（`1.1/2.2 petaFLOPS`） | **75 TFLOPS** | 37 | — |
| B300 单卡（HGX） | 1,100 | 2,200 | **75 TFLOPS** | 1.2（Ultra datasheet，**存疑**） | — |
| GB300 整柜 | 90,000 | 180,000 | 5,760 / 6,000 | 100 | — |
| GB200 整柜 | 90,000 | 180,000 | 5,760 | 2,880 | — |
| B30A | 未找到公开数据 | — | 未找到公开数据 | 未找到公开数据 | — | — |
| RTX 6000D | 未找到公开数据 | — | **~97.04**（第三方拆解，向量） | 1.516（第三方，低可信度） | — |
| L40S | **183** | **366** | **91.6** | — | — |
| L20 | 59.8（第三方）/ 59.3（阿里云） | 未找到公开数据 | **59.3**（阿里云）/ 59.35（第三方） | N/A（阿里云标 N/A） | — |
| MI300X | 653.7 | 1,307 | 163.4 | 81.7 | 163.4 |

---

## 2. 总表 B：显存、互连、功耗、外形、扩展域

| 产品 | 架构 / 工艺 | 显存类型·容量 | 显存带宽 | NVLink 代际 / 单卡双向 | NVSwitch | PCIe | TDP | 外形 | scale-up 域 |
|---|---|---|---|---|---|---|---|---|---|
| **A100 40GB PCIe** | Ampere / TSMC 7nm N7 | 40GB HBM2 | 1,555 GB/s | NVLink 3 / 600 GB/s（NVLink Bridge，最多 2 卡） | 否（PCIe 侧） | Gen4 64 GB/s | 250W | PCIe 双插槽 | 2 卡（桥接）/ 8 卡（HGX） |
| **A100 80GB PCIe** | Ampere / 7nm | 80GB HBM2e | 1,935 GB/s | NVLink 3 / 600 GB/s | 否 | Gen4 64 GB/s | 300W | PCIe 双插槽 | 2 / 8 |
| **A100 40GB SXM4** | Ampere / 7nm | 40GB HBM2 | 1,555 GB/s | NVLink 3 / **600 GB/s** | 是（HGX） | Gen4 64 GB/s | 400W | SXM4 | **8**（HGX A100，NVSwitch 可 16） |
| **A100 80GB SXM4** | Ampere / 7nm | 80GB HBM2e | **2,039 GB/s** | NVLink 3 / 600 GB/s | 是 | Gen4 64 GB/s | 400W | SXM4 | 8（NVSwitch 可 16） |
| **H100 SXM** | Hopper / TSMC 4N | 80GB HBM3 | **3.35 TB/s** | NVLink 4 / **900 GB/s** | 是 | Gen5 128 GB/s | **700W**（可配置） | SXM5 | **8**（NVLink Switch System 可到 256） |
| **H100 PCIe** | Hopper / 4N | 80GB HBM3 | **2.0 TB/s** | NVLink 4 / **600 GB/s** | 否 | Gen5 128 GB/s | 350W | PCIe 双插槽风冷 | 2（桥接）/ 8 |
| **H100 NVL** | Hopper / 4N | **94GB** HBM3 | **3.9 TB/s** | NVLink 4 / 600 GB/s | 否 | Gen5 128 GB/s | 350–400W 可配置 | PCIe 双插槽 | 2（桥接） |
| **H800 SXM** | Hopper / 4N（第三方称 5nm） | 80GB HBM3 | 3.35 TB/s | **NVLink 4 / 400 GB/s（人为砍半）** | 是 | Gen5 128 GB/s | ~700W | SXM5 | 8 |
| **H800 PCIe** | Hopper / 4N | 80GB（HBM2e 存疑） | ~2.0 TB/s | **400 GB/s（人为削减）** | 否 | Gen5 | 350W | PCIe | 2 / 8 |
| **H20 SXM5** | Hopper / 4N | **96GB HBM3** / **141GB HBM3e** | **4.0 TB/s**（两版本同） | **NVLink 4 / 900 GB/s** | 是 | Gen5 x16 128 GB/s | **400W** | 8 路 HGX（SXM） | 8 |
| **H200 SXM** | Hopper / 4N | **141GB HBM3e** | **4.8 TB/s** | NVLink 4 / 900 GB/s | 是 | Gen5 128 GB/s | 最高 700W 可配置 | SXM | 8 |
| **H200 NVL** | Hopper / 4N | 141GB HBM3e | **4.2 TB/s**（datasheet）/ 4.8（产品页，**冲突**） | 2/4 路 NVLink Bridge，900 GB/s per GPU | 否 | Gen5 128 GB/s | 最高 600W 可配置 | PCIe 双插槽风冷 | 2 或 4 |
| **B200（HGX B200 单卡）** | Blackwell / TSMC 4NP，双 die | **180GB** HBM3e（OEM 配置；NVIDIA 原文 `up to 192 GB`，**冲突**） | **7.7 TB/s** | NVLink 5 / **1.8 TB/s** | 是（NVLink 5 Switch） | **Gen6**（Ultra datasheet 列 B200 为 Gen5，**冲突**） | **1,000W** | SXM6 | **8**（HGX） |
| **GB200 内 GPU** | Blackwell / 4NP | **192GB** HBM3e（186–192 各源不一） | **8 TB/s** | NVLink 5 / 1.8 TB/s | 是 | Gen5/Gen6 | ~1,200W（托盘级） | Grace Blackwell Superchip（1 Grace + 2 GPU） | **72（NVL72）** |
| **B300 / Blackwell Ultra（HGX B300 单卡）** | Blackwell Ultra / 4NP | **270GB** HBM3e | **7.7 TB/s**（datasheet 7.7 / 产品页 8） | NVLink 5 / 1.8 TB/s | 是 | **Gen6 256 GB/s** | 最高 **1,100W** 可配置 | SXM | 8 |
| **GB300 内 GPU** | Blackwell Ultra / 4NP | **279GB** HBM3e | **8 TB/s** | NVLink 5 / 1.8 TB/s | 是 | Gen6 256 GB/s | 最高 **1,400W** 可配置 | GB300 Superchip | **72（NVL72）** |
| **B30A（中国特供）** | Blackwell Ultra 单 die | 未找到公开数据（传闻 96GB HBM3e） | 未找到公开数据 | 传闻约 B300 之半（≈900 GB/s），**未证实** | 未找到公开数据 | 未找到公开数据 | 传闻约 800W | 传闻 NVL8 基板 | 传闻 8（NVL8） |
| **RTX 6000D（中国特供）** | Blackwell / TSMC 4N，GB202，CC 12.0 | **84GB GDDR7**（448-bit，28×3GB） | **1,568 GB/s**（实测口径） | **无 NVLink（设计上移除）** | 否 | Gen5 x16 | **600W** | 无风扇被动双插槽，**无视频输出**（TCC） | 未公布（无 NVLink 域） |
| **A800 40GB Active**（工作站 SKU） | Ampere / 7nm | 40GB HBM2 | 1,555.2 GB/s | **NVLink 3 / 400 GB/s（人为削减，A100 为 600）** | 否 | Gen4 x16 | 240W | PCIe 双插槽 | 2（桥接） |
| **L40S** | Ada Lovelace / AD102 / TSMC 4N | 48GB GDDR6 ECC | **864 GB/s** | **无 NVLink** | 否 | Gen4 x16 64 GB/s | 350W | PCIe 双插槽被动 | 1（单机最多 8 卡走 PCIe） |
| **MI300X（AMD，对照）** | CDNA 3 / 5nm + 6nm | **192GB HBM3** | **5.3 TB/s** | Infinity Fabric 128 GB/s（非 NVLink） | — | Gen5 x16 | 750W（TBP 峰值） | OAM | 8（UBB） |

---

## 3. 关键结论与对比要点（面向寒武纪 MLU 对比）

1. **口径陷阱**：把 H100 SXM 的「1,979 TFLOPS BF16」当作稠密算力是行业最常见的错误 —— 稠密只有 **989**。同一芯片名在 SXM / PCIe / NVL 三种板型下算力与带宽都会变（H100 PCIe 只有 800 sparse / 400 dense BF16，带宽 2.0 TB/s vs SXM 3.35 TB/s）。
2. **出口 SKU 优先砍互连**：A800（600→400 GB/s）、H800（900→400 GB/s）算力/显存对齐正版，**只砍 NVLink**（H800 另把 FP64 从 34 砍到 1 TFLOPS）。
   **H20 相反** —— 砍算力（BF16 **dense 仅 148** vs H100 dense 989，约 1/6.7）却**保留 900 GB/s NVLink 与 4.0 TB/s 带宽**。
   ⚠️ **重要更正**：H20 的 148/296 是 **dense**，不是 sparse（百度百科等中文来源的 `*` 标注有误）。详见 §E.1。
   → **H800 适合对标训练/prefill，H20 适合对标 decode/长上下文推理**（带宽与互连反而更强）。
3. **H20 有 96GB HBM3 与 141GB HBM3e 两个版本**（NVIDIA 官方 docs.nvidia.com vGPU 文档载明，均为 SXM5、400W、4.0 TB/s）。**H20 没有 PCIe SKU**；H800 则有 SXM5 80GB / PCIe 80GB / PCIe 94GB（NVL）三款，且 **PCIe 版也有 400 GB/s NVLink**。
4. **H200 是「同算力、换显存/带宽」**：计算单元不变（BF16 989/1,979），HBM3e 把容量从 80→141GB、带宽从 3.35→4.8 TB/s。对长上下文 decode 收益显著，对大 GEMM 训练几乎无感。
5. **Blackwell 的 FP4 是分水岭**：B200 起出现 FP4。B200 的 FP4 仍是标准 **2.0×**（18 sparse / 9 dense per GPU），但 **B300 的 FP4 只有约 1.33×**（18 sparse / 13.5 dense per GPU，GB300 内为 20 / 15）。NVIDIA 未解释原因。**若与寒武纪 MLU 的 INT8/INT16 口径对比，务必先统一到 dense。** 详见 §4.1。
6. **scale-up 域从 8 跃到 72**：GB200/GB300 NVL72 把单个 NVLink 域从 8 卡扩到 72 卡（整柜 130 TB/s），这是与 MLU 集群架构对比时最大的结构性差异。
7. **对中国特供 Blackwell（B30A / RTX 6000D）目前没有 NVIDIA 官方规格**。截至 2026-09-18，NVIDIA **从未发布** B30A 或 RTX 6000D 的官方 datasheet/产品页。
   - **B30A**：已设计并向中国客户送样，但 **2025-11-07 被白宫阻止出口**（The Information / 路透），至 2026-09-18 仍未获批。无继任者。
   - **RTX 6000D**：**已实际出货**。可靠数据来自零售硬件拆解与受控基准：84GB GDDR7 / 448-bit / 1,568 GB/s / 19,968 CUDA / 600W / **无 NVLink** / 无视频输出；**实测稠密 BF16 GEMM 仅 144.09 TFLOPS（全血的 36%）** —— 核心只减 17% 但 Tensor 吞吐砍掉 64%，属产品级限流，**不可按核心数比例推算**。
   - ⚠️ **命名陷阱**：2025 年报道中「B30/B40」既指 GDDR7 的 RTX 6000D，也指 HBM3E 的 B30A，大量二手报道混为一谈。
# NVIDIA 数据中心 AI GPU 规格表 —— 逐型号注释 + 来源清单

## A. 逐型号注释（日期 / 口径 / 来源）

### A100（40GB / 80GB，SXM4 / PCIe）
- **发布**：2020-05-14 GTC 2020 主题演讲发布（Ampere 架构首款数据中心 GPU）。官方新闻稿：<https://nvidianews.nvidia.com/news/nvidia-announces-ampere-architecture-and-a100-gpu>（该 URL 现重定向至新闻归档，需注意）。
- **工艺/架构**：TSMC 7nm N7，Ampere（GA100）。datasheet 未直接写工艺节点，第三方/媒体一致为 7nm。
- **规格来源（NVIDIA官方）**：A100 datasheet（英文版 2021-06，`JUN21`，编号 1758950 R4）
  <https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf>
  中文版（编号 2188504 R5）：<http://images.nvidia.cn/aem-dam/en-zz/Solutions/data-center/a100/nvidia-a100-datasheet-nvidia-a4-2188504-r5-zhCN.pdf>
- **关键口径**：datasheet 表格里 TF32 `156 TFLOPS | 312 TFLOPS*`、BF16 `312 | 624*`、FP16 `312 | 624*`、INT8 `624 TOPS | 1248 TOPS*`；脚注 `* With sparsity`。即 **表格中前一个数是 dense，后一个是 sparse**。
- **FP8/FP4**：A100 **不支持**（Ampere Tensor Core 无 FP8）。datasheet 中不存在 FP8 行。
- **显存带宽**：40GB PCIe 1,555 GB/s；80GB PCIe 1,935 GB/s；40GB SXM 1,555 GB/s；80GB SXM **2,039 GB/s**（注意：中文版 datasheet 把 80GB PCIe 写作 1935、SXM 写作 2039，与英文版一致）。
- **TDP**：40GB PCIe 250W；80GB PCIe 300W；40GB SXM 400W；80GB SXM 400W。
- **互连**：SXM 走 NVLink 3 600 GB/s + NVSwitch（HGX A100 8 卡；datasheet 文字称 "up to 16 A100 GPUs can be interconnected... at up to 600 GB/s"）；PCIe 走 NVLink Bridge，最多 2 卡；PCIe Gen4 64 GB/s。

### H100（SXM5 / PCIe / NVL）
- **发布**：2022-03-22，GTC 2022。官方新闻稿（日期已核）：<https://nvidianews.nvidia.com/news/nvidia-announces-hopper-architecture-the-next-generation-of-accelerated-computing>
- **工艺/架构**：TSMC 4N，Hopper（GH100），800 亿晶体管（datasheet 原文 "built from 80 billion transistors... TSMC 4N process"）。
- **规格来源（NVIDIA官方）**：
  - 英文 datasheet：<https://resources.nvidia.com/en-us-gpu-resources/h100-datasheet-24306>（PDF 实体：`https://dam-cdn.nvd.orangelogic.com/AssetLink/u5hh6fv4r7564i4y673484y3m20083nj.pdf`）
  - 中文 datasheet（编号 2287922 R7）：<http://images.nvidia.cn/aem-dam/en-zz/Solutions/data-center/h100/nvidia-h100-datasheet-nvidia-a4-2287922-r7-zhCN.pdf>
- **⚠️ 冲突 1（SXM vs PCIe vs NVL 命名）**：英文 datasheet 的表头是 `H100 SXM` 与 `H100 NVL`；中文 datasheet 的表头是 `H100 SXM` 与 `H100 PCIe`，但两栏数值完全相同（FP64 24、FP32 48、TF32 800*、BF16/FP16 1,600*、FP8 3,200*、INT8 3,200 TOPS*、显存 80GB、带宽 **2TB/s**、TDP 350W、NVLink 600GB/s）。**H100 PCIe 与 H100 NVL 是不同产品**（NVL 为 94GB HBM3 / 3.9 TB/s / 350–400W），因此中文 datasheet 的 "H100 PCIe" 栏很可能实际指 H100 NVL 或被简化。**引用 350W / 2TB/s / NVLink 600GB/s 这组数时，应注明出自中文 datasheet 的 "H100 PCIe" 栏。**
- **关键口径**：NVIDIA官方 表格值为 sparse，脚注 `*With sparsity`；**dense = 表格值 ÷ 2**：
  - H100 SXM：TF32 989* → dense 494.5；BF16/FP16 1,979* → dense 989.5；FP8 3,958* → dense 1,979；INT8 3,958 TOPS* → dense 1,979。
  - H100 PCIe/NVL 栏：TF32 800* → 400；BF16/FP16 1,600* → 800；FP8 3,200* → 1,600；INT8 3,200* → 1,600。
- **TPP/功耗**：SXM "Up to 700W (configurable)"；PCIe 栏 350W。
- **互连**：SXM NVLink 900 GB/s + NVSwitch，NVLink Switch System 可连 **256 卡**；PCIe/NVL 栏 NVLink 600 GB/s。PCIe Gen5 128 GB/s。
- **MIG**：最多 7 个 MIG（SXM 每个 10GB；NVL 每个 12GB — 见英文 datasheet）。

### H800（中国特供）
- **发布**：2022 年 11 月前后（路透 2022-11-08 报道 NVIDIA 为中国推出符合出口管制的新芯片）。
- **出口管制状态**：为规避 2022-10-07 美国对华出口管制而设计；NVLink 从 H100 的 900 GB/s **削减至 400 GB/s**。2023-10 美国新规后 H800 亦被禁售。
- **NVIDIA 官方规格**：**不存在**。NVIDIA 从未发布 H800 的官方产品页或 datasheet（nvidia.com/en-us/data-center/h800 无 Wayback 快照，nvidia.cn 对应页面 404）。所有数字均为第三方/OEM。
- **第三方规格（已核实）**：强川科技 H800 SXM 产品页 <https://www.qiangchuan.com/item/7228.html>
  - 架构 Hopper、晶体管 800 亿、SXM5、80GB HBM3、5120-bit、**3.35 TB/s**、ECC 支持
  - FP32 67 TFLOPS、FP64 1 TFLOPS（**⚠️ 与 H100 SXM 的 FP64 34 TFLOPS 严重冲突，存疑**）
  - Tensor TF32 989 TFLOPS*、Tensor BF16/FP16 1,979 TFLOPS*、Tensor INT8 3,958 TOPS*，`*采用稀疏技术`
  - 最大功耗 700W；GPU 互连：页面正文文字为「可提供 **400 GB/s** GPU 间互连的第四代 NVLink」
  - FP8 未在参数表列出，但正文提到 Transformer Engine 支持 FP8；按同代推算应为 3,958 TFLOPS sparse（**推算，非官方**）
- **H800 PCIe**：第三方来源为 Lenovo ThinkSystem H800 PCIe 产品指南（`lenovopress.lenovo.com/lp1814`，**现已标注 withdrawn，页面返回 JS 壳，PDF 已不可直下**）。据多方二手引用：FP64 被压到 **0.8 TFLOPS**，NVLink **400 GB/s**，TDP 350W。**建议标注为「未直接核实到原文」。**
- **Datasheet 索引**：llm-infra-atlas 开源文档（第三方）交叉印证 H800 NVLink 400 GB/s、单向约 200 GB/s：<https://github.com/llm-infra-atlas/llm-infra-atlas.github.io>

### H20（中国特供）
- **发布/可用**：2023 年 11 月发布（PS: Nov 2023），2023 年 12 月量产（MP: Dec 2023）。2024-02 开始接受分销商预订（单价约 1.2–1.5 万美元）。
- **NVIDIA 官方规格**：**不存在**。nvidia.com / nvidia.cn 的 h20 页面均无 Wayback 快照（nvidia.cn/data-center/h20/ 返回 404）。
- **第三方规格（已核实）**：百度百科 H20 词条参数表（原始引用媒体，数据截至 2025-04-17）：<https://baike.baidu.com/en/item/H20/1510112>
  - GPU 架构 NVIDIA Hopper；显存 **96GB HBM3**；带宽 **4.0 TB/s**
  - `INT8 | FP8 Tensor Core*` **296 | 296 TFLOPS**（`*`=稀疏）
  - `BF16 | FP16 Tensor Core` **148 | 148 TFLOPS**（按 NVIDIA 通例，此值对应稀疏口径；dense = 74）
  - TF32 Tensor Core **74 TFLOPS**；FP32 **44 TFLOPS**；FP64 **1 TFLOPS**
  - MIG 最多 7 个；L2 60MB；7 NVDEC / 7 NVJPEG；**功耗 400W**；外形 **8 路 HGX**
  - 互连：**PCIe Gen5 x16 128 GB/s；NVLink 900 GB/s**（**未削减**）
- **交叉印证**：TechPowerUp（2025-04-16）同样给出「96 GB HBM3、最高 4.0 TB/s、约 296 TFLOPS 混合精度、FP32 约 74 TFLOPS、FP16 约 148 TFLOPS」，但**其 FP32 74 TFLOPS 与百度百科的 44 TFLOPS 冲突**（百度百科把 74 放在 TF32、44 放在 FP32，更符合 H100 阉割比例）。<https://www.techpowerup.com/335534/us-bans-export-of-nvidia-h20-accelerators-to-china-a-potential-usd-5-5-billion-loss-for-nvidia>
- **出口管制时间线（全部为媒体报道，非 NVIDIA 规格）**：
  - 2025-04-14：美国商务部通知 NVIDIA，H20 对华（含港澳）出口需申请许可证（"indefinite"）。
  - 2025-04-15/16：NVIDIA 8-K 披露约 **55 亿美元**减记；TechPowerUp 报道 → 同上链接。
  - 2025-07-15：美国批准 H20 对华销售（Reuters/百度百科时间线）。
  - 2025-08-08：商务部开始发放 H20 出口许可证。
  - 2025-08-11：路透报道 NVIDIA/AMD 同意将对华芯片销售收入的 **15%** 上缴美国政府以换取许可证。<https://www.reuters.com/world/china/nvidia-amd-pay-15-china-chip-sale-revenues-us-official-says-2025-08-11/>
  - 2025-08-22：The Information（经 TechPowerUp 转述）报道 NVIDIA 已停止 H20 生产，为 B30A 让路。<https://www.techpowerup.com/340198/nvidia-reportedly-ends-h20-gpu-production-makes-room-for-b30a>
  - 2025-09-17：金融时报报道中国网信办（CAC）要求阿里/百度/腾讯等停止采购 NVIDIA 出口改型加速器（含 RTX PRO 6000D）。<https://www.techpowerup.com/341078/china-blocks-tech-firms-from-buying-nvidia-ai-accelerators>
  - 2026-05-17：百度百科时间线称美国再次批准 H20 对华销售（**单一来源，建议存疑**）。

### H200（SXM / NVL）
- **发布**：2023-11-13，SC23。官方新闻稿（日期已核）：<https://nvidianews.nvidia.com/news/nvidia-supercharges-hopper-the-worlds-leading-ai-computing-platform>
  - 原文：`141GB of memory at 4.8 terabytes per second, nearly double the capacity and 2.4x more bandwidth compared with its predecessor, the NVIDIA A100`；系统预计 **2024 Q2** 开始出货。
- **规格来源（NVIDIA官方）**：
  - 产品页：<https://www.nvidia.com/en-us/data-center/h200/>
  - datasheet：<https://dam-cdn.nvd.orangelogic.com/AssetLink/5o2qgy5d2835ve2pm11i62kv8mphqta8.pdf>（© 2024，编号 3512650）
- **⚠️ 冲突 2（H200 带宽）**：
  - 产品页表格：H200 SXM **4.8TB/s**，H200 NVL **4.8TB/s**。
  - datasheet 表格：H200 SXM **4.8TB/s**，H200 NVL **4.2TB/s**；且 datasheet 正文与 Key Features 都写 `4.2TB/s`。
  - 新闻稿只给 SXM 口径的 4.8 TB/s。**建议：SXM 4.8 TB/s 无争议；H200 NVL 采用 4.2 TB/s（datasheet）并注明产品页写 4.8。**
- **⚠️ 冲突 3（H200 NVL 显存/TDP）**：product page 与 datasheet 均为 141GB、600W 可配置、MIG 7×16.5GB，一致。**无冲突**。
- **互连**：SXM NVLink 900 GB/s + NVSwitch，HGX H200 4/8 卡；NVL（PCIe）为 2 或 4 路 NVLink Bridge、**900 GB/s per GPU**（注意：官网产品页原文写作 `2- or 4-way NVIDIA NVLink bridge: 900GB/s per GPU`，与其他 PCIe 产品 600 GB/s 不同，属官方原文）。
- **口径**：产品页与 datasheet 均标注 `With sparsity`，dense = 1/2。

### B200 / GB200（Blackwell）
- **发布**：2024-03-18，GTC 2024。官方新闻稿（日期已核）：<https://nvidianews.nvidia.com/news/nvidia-blackwell-platform-arrives-to-power-a-new-era-of-computing>
- **工艺/架构**：TSMC 4NP，Blackwell（双 die 封装）。
- **规格来源（NVIDIA官方）**：
  - GB200 NVL72 产品页：<https://www.nvidia.com/en-us/data-center/gb200-nvl72/>
  - HGX 产品页（含 HGX B200 8 卡与单卡表）：<https://www.nvidia.com/en-us/data-center/hgx/>
  - Blackwell Tech Brief（datasheet）：<https://resources.nvidia.com/en-us-blackwell-architecture>（PDF：`https://dam-cdn.nvd.orangelogic.com/AssetLink/gl2l4l4812s5fw0p614s6i8bv6mi3vx5.pdf`）
  - DGX B200 产品页：<https://www.nvidia.com/en-us/data-center/dgx-b200/>
- **GB200 NVL72 整柜（产品页原文）**：36 Grace CPU + 72 Blackwell GPU；`NVFP4 Tensor Core² 1,440 | 720 PFLOPS`；`FP8/FP6 Tensor Core² 720 PFLOPS`；`INT8 Tensor Core² 720 POPS`；`FP16/BF16 Tensor Core² 360 PFLOPS`；`TF32 Tensor Core² 180 PFLOPS`；FP32 5,760 TFLOPS；FP64 2,880 TFLOPS；`13.4 TB HBM3E | 576 TB/s`；NVLink 130 TB/s。
  - 脚注：`1. Specification in sparse | dense.`　`2. Specification in sparse. Dense is one-half sparse spec shown.`
- **GB200 Grace Blackwell Superchip（产品页原文）**：`40 | 20 PFLOPS` NVFP4；`20 PFLOPS` FP8/FP6；`20 POPS` INT8；`10 PFLOPS` FP16/BF16；`5 PFLOPS` TF32；160 TFLOPS FP32；80 TFLOPS FP64；`372 GB HBM3E | 16 TB/s`；NVLink 3.6 TB/s。（此处 `40 | 20` 是 sparse | dense，比值 2.0）
- **⚠️ 冲突 4（B200 单卡 FP4 比值）**：HGX 页 8 卡 `144 PFLOPS | 72 PFLOPS`（脚注 `Specification in Sparse | Dense`）→ 单卡 **18 sparse / 9 dense**，比值 **2.0**。而 GB200 Superchip 也是 40/20 = 2.0。**因此 B200 的 FP4 是标准 2×。**
- **⚠️ 冲突 5（B200 显存容量）**：
  - Blackwell Tech Brief：`Up to 192 GB HBM3e | 7.7 TB/s`（HGX B200 单卡）。
  - HGX 页：HGX B200 总显存 `1.4 TB` → 单卡 175GB（若按 1.4TB 精确值）；实际 OEM 产品（Lenovo ThinkSystem HGX B200）为 **180GB HBM3e**。
  - DGX B200 产品页：`1,440 GB total, 64 TB/s HBM3e bandwidth` → 单卡 **180GB / 8 TB/s**（注意此处带宽与 Tech Brief 的 7.7 TB/s 不一致）。
  - **建议：写「180GB（OEM/DGX 配置）；NVIDIA 原文为 up to 192GB；单卡带宽 7.7 TB/s（Tech Brief）或 8 TB/s（DGX B200 页）」并标注冲突。**
- **⚠️ 冲突 6（PCIe 代际）**：Blackwell Tech Brief 的 HGX B200 行写 `Interconnect: NVLink 5 / PCIe Gen5`；Blackwell Ultra datasheet 的 HGX B300 行写 `PCIe Gen6: 256 GB/s`。HGX B200 的 PCIe 代际在官方文档间不一致。
- **FP64**：HGX Tech Brief 写 B200 单卡 FP64 = 37 TFLOPS；HGX 页写 HGX B200 整机 `FP64/FP64 Tensor Core 296 TFLOPS`（296/8 = 37，自洽）。

### B300 / GB300（Blackwell Ultra）
- **发布**：2025-03-18，GTC 2025。官方新闻稿（日期已核）：<https://nvidianews.nvidia.com/news/nvidia-blackwell-ultra-ai-factory-platform-paves-way-for-age-of-ai-reasoning>
  - 原文：`Blackwell Ultra includes the NVIDIA GB300 NVL72 rack-scale solution and the NVIDIA HGX B300 NVL16 system. The GB300 NVL72 delivers 1.5x more AI performance than the NVIDIA GB200 NVL72`。
- **规格来源（NVIDIA官方）**：
  - GB300 NVL72 产品页：<https://www.nvidia.com/en-us/data-center/gb300-nvl72/>
  - HGX 产品页：<https://www.nvidia.com/en-us/data-center/hgx/>
  - Blackwell Ultra datasheet：<https://resources.nvidia.com/en-us-blackwell-architecture/blackwell-ultra-datasheet>（PDF：`https://dam-cdn.nvd.orangelogic.com/AssetLink/1k0p832eq8r5ca0u5383ie5o4tp3bst1.pdf`）
  - DGX B300 产品页：<https://www.nvidia.com/en-us/data-center/dgx-b300/>
- **GB300 NVL72 整柜**：72 Blackwell Ultra GPU + 36 Grace；`FP4 Tensor Core 1440 | 1080² PFLOPS`（脚注 2 = without sparsity）；`FP8/FP6 720 PFLOPS`；`INT8 24 POPS`；`FP16/BF16 360 PFLOPS`；`TF32 180 PFLOPS`；`FP32 6 PFLOPS`；`FP64 100 TFLOPS`；`20 TB | up to 576 TB/s`；NVLink 130 TB/s；Fast Memory 37 TB。
- **HGX B300 8 卡**：`FP4 Tensor Core¹ 144 PFLOPS | 108 PFLOPS`（脚注 1 = Sparse | Dense）；`FP8/FP6 Tensor Core² 72 PFLOPS`；`INT8 3 POPS`（**⚠️ 与其他出处冲突，见 §4.2**）；`FP16/BF16 36 PFLOPS`；`TF32 18 PFLOPS`；FP32 600 TFLOPS；`FP64/FP64 Tensor Core 10 TFLOPS`（**⚠️ 与 Ultra datasheet 的 1.2 TFLOPS 冲突**）；`Total Memory 2.1 TB`；NVLink 5 / 1.8 TB/s 单卡 / 14.4 TB/s 整机；Networking 1.6 TB/s。
- **Blackwell Ultra 单卡（Ultra datasheet「Individual GPU Specifications」）**：GB300 列 `FP4 20 | 15 PFLOPS`、`FP8/FP6 10 PFLOPS`、`INT8 330 TOPS`、`FP16/BF16 5 PFLOPS`、`TF32 2.5 PFLOPS`、`FP32 80 TFLOPS`、`FP64 1.3 TFLOPS`、`279 GB HBM3E | 8 TB/s`、TDP 最高 1,400W、NVLink 5 1.8 TB/s、PCIe Gen6 256 GB/s；HGX 列 `FP4 18 | 14 PFLOPS`、`FP8/FP6 9 PFLOPS`、`INT8 307 TOPS`、`FP16/BF16 4.5 PFLOPS`、`TF32 2.2 PFLOPS`、`FP32 75 TFLOPS`、`FP64 1.2 TFLOPS`、`270 GB HBM3E | 7.7 TB/s`、TDP 最高 1,100W。
- **⚠️ 冲突 7（B300 FP4 的 sparse/dense 比值）**：见 §4.1 —— 这是全报告最重要的口径冲突。
- **DGX B300**：8× Blackwell Ultra SXM、总显存 2.1 TB、`FP4 Tensor Core: 144 PFLOPS | 108 PFLOPS*`（`*Specification shown in sparse | dense`）、`FP8: 72 PFLOPS**`（`**sparse, dense ½`）、NVLink 14.4 TB/s、功耗 ~14 kW、10U。

### B30A（中国特供 Blackwell，已被阻止出口）
- **NVIDIA 官方规格**：**不存在**。截至 2026-09-18，NVIDIA 从未发布 B30A 的产品页或 datasheet。
- **报道时间线**：
  - 2025-08-19：路透（经 TechPowerUp 转述）报道 NVIDIA 设计了中国特供 SKU **B30A**，基于 B300 的**单 die** 版本（双 die B300 取其一），性能减半。TechPowerUp 以此为据**推算**：约 7.5 PFLOPS FP4、3.75 PFLOPS FP6/FP8、1.875 PFLOPS FP16/BF16、0.94 PFLOPS TF32（该推算基于 B300 的 **dense** 15/7.5/3.75/1.88 口径，属**媒体推算**）。<https://www.techpowerup.com/340068/nvidia-prepares-cut-down-b30a-blackwell-ai-accelerator-for-chinese-market>
  - 2025-09-04：路透消息（经 TechPowerUp）称 B30A 定价约为 H20 的 **2 倍**（约 2 万–2.4 万美元 vs H20 约 1 万–1.2 万美元）。<https://www.techpowerup.com/340664/nvidias-export-compliant-b30a-for-china-priced-at-roughly-twice-the-h20>
  - 2025-09-17：金融时报报道中国网信办要求国内厂商停止采购 NVIDIA 出口改型加速器。<https://www.techpowerup.com/341078/china-blocks-tech-firms-from-buying-nvidia-ai-accelerators>
  - **2025-11-07：The Information 报道美国白宫决定阻止 B30A 对华出口许可**，NVIDIA 已向中国客户送样，正重新设计以争取获批；NVIDIA 称在中国竞争性数据中心计算市场份额为「zero」。<https://investinglive.com/stocks/us-blocks-nvidias-scaled-down-ai-chip-sales-to-china-despite-trump-hints-20251107/>
- **⚠️ 存疑点**：社交媒体/港媒（如 AAStocks）2026 年仍在讨论「美国将阻止 B30A」，与 2025-11 的报道为同一事件的不同转述。**未找到任何 NVIDIA 官方 B30A 规格。**
- **互连/显存/NVL8**：有媒体称 B30A 保留 HBM 与 NVLink、采用 NVL8 基板、NVLink 约减半至 900 GB/s，但**本报告未核实到可靠的原始出处**，一律记为「未找到公开数据 / 传闻未证实」。

### RTX 6000D / RTX PRO 6000D（中国特供工作站）
- **NVIDIA 官方规格**：**不存在**面向中国市场的 "RTX 6000D" 官方页面。NVIDIA 官方只有 **RTX PRO 6000 Blackwell 系列**（96GB GDDR7、PCIe Gen5 x16、Server Edition 400–600W / Workstation 600W / Max-Q 300W）：<https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000-family/>
  - **注意**：NVIDIA 官方该页**未列出**任何 Tensor Core TFLOPS 算力数字，也未列显存带宽 → 这两项对全血版亦为「未在官方产品页披露」。
- **中国特供版（媒体/拆解）**：
  - 2025-09-16：路透独家报道 RTX 6000D 在中国大企业遇冷，"little favour with major firms"。<https://www.usnews.com/news/top-news/articles/2025-09-16/exclusive-nvidias-new-rtx-6000d-chip-for-china-finds-little-favour-with-major-firms-sources-say>
  - **拆解实测**：TweakTown 报道拆解显示中国版为 **84GB GDDR7**（全血 RTX PRO 6000 Blackwell 为 96GB）。<https://www.tweaktown.com/news/110135/nvidias-new-rtx-6000d-appears-in-teardown-84gb-gddr7-in-china-compared-to-the-full-96gb/index.html>
  - WindowsReport 同样报道 84GB GDDR7 与削减后的 Blackwell 规格。<https://windowsreport.com/nvidia-rtx-6000d-teardown-reveals-84gb-gddr7-and-cut-down-blackwell-specs/>
  - 2025-09-17：CAC 要求停止采购（同 H20 一条）。
- **算力/带宽/TDP**：媒体常引用「约 1,100 GB/s、约 600W、无 NVLink」，但**未核实到 NVIDIA 原文**；本报告一律记为「未找到公开数据（NVIDIA官方）」。

### A800（中国特供 Ampere）
- **发布**：2022-11-08 路透独家报道 NVIDIA 为中国推出符合出口管制的 A800。<https://www.reuters.com/technology/exclusive-nvidia-offers-new-advanced-chip-china-that-meets-us-export-controls-2022-11-08/>
- **NVIDIA 官方规格（部分）**：NVIDIA 确有 **A800 40GB Active** 工作站产品页（这是唯一被 NVIDIA 官方列出的 A800 SKU）：
  <https://www.nvidia.com/en-in/products/workstations/a800/>
  - 40GB HBM2、显存位宽 5,120-bit、**带宽 1,555.2 GB/s**；CUDA 6,912 / Tensor 432
  - FP64 9.7 TFLOPS；FP32 19.5 TFLOPS；`Peak Tensor Performance 1,247 AI TOPS | 623.8 TFLOPS`（即 623.8 TFLOPS = BF16/FP16 稀疏，1247 AI TOPS = INT8 稀疏）
  - **NVLink 带宽 400 GB/s**（对比 A100 的 600 GB/s，**人为削减**）；PCIe 4.0 x16；**240W**；双插槽主动散热；最多 7 个 MIG @5GB
- **A800 80GB SXM/PCIe**：**NVIDIA 无官方页面**。第三方/OEM 一致为：显存与算力对齐 A100（80GB HBM2e、2039 GB/s、SXM 400W），NVLink 600→400 GB/s。**本报告未直接核实到 OEM 原文。**

### L40S（Ada Lovelace，通用推理/图形）
- **发布**：2023-08-08（NVIDIA 新闻稿 "NVIDIA, Global Data Center System Manufacturers to Supercharge Generative AI and Industrial Digitalization"；该稿的 nvidianews 原始 URL 已归档，中文版见 <https://blogs.nvidia.com.tw/blog/nvidia-global-data-center-system-manufacturers-to-supercharge-generative-ai-and-industrial-digitalization/>）。
- **规格来源（NVIDIA官方）**：
  - 产品页：<https://www.nvidia.com/en-us/data-center/l40s/>
  - **中文 datasheet（编号 2907810，版本日期 2023.9.25）**：<http://images.nvidia.cn/cn/RTX/l40s-datasheet-web-a4-zhCN-2907810-2023.9.25.pdf>
- **⚠️ 冲突 8（L40S FP16）**：
  - 产品页写：`FP32 91.6 teraFLOPS`、`TF32 Tensor Core 366 teraFLOPS*`、`FP16 733 teraFLOPS*`、`FP8 1,466 teraFLOPS*`、`RT Core 212 TFLOPS`、`Max Power 350W`，脚注 `*With Sparsity`。
  - **中文 datasheet 写**：`TF32 183 | 366*`、`BFLOAT16 362.5 | 733*`、`FP16 362.5 | 733*`、`FP8 733 | 1,466*`、`INT8 733 | 1,466*`、`INT4 733 | 1,466*`、`FP32 91.6`。
  - **即产品页的 "FP16 733" 是稀疏值，其 dense 为 362.5**；产品页把 sparse 值当默认展示，未在行内标 dense。**以 datasheet 为准。**
- **显存/互连**：48GB GDDR6 ECC、**864 GB/s**、PCIe 4.0 x16（64 GB/s 双向）、**无 NVLink**、350W、双插槽被动、4×DisplayPort 1.4a。

### L20（中国特供 Ada）
- **NVIDIA 官方规格**：**未找到公开数据**。nvidia.cn/data-center/l20/ 返回 404，无 Wayback 快照。
- 媒体/OEM 常引用「约 48GB GDDR6、约 275W、约 119 TFLOPS FP16 稀疏」，但**本报告未核实到原始出处，一律不采信**。

### MI300X（AMD，仅作对照）
- **发布**：2023-12-06（AMD 新闻稿 "AMD Delivers Leadership Portfolio of Data Center AI Solutions with AMD Instinct MI300 Series"）。<https://www.amd.com/ja/newsroom/press-releases/2023-12-06-amd-ai-amd-instinct-mi300.html>
- **规格来源（AMD官方）**：<https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html>
  - **AMD 明确区分** with / without structured sparsity，是本报告中最清晰的口径范例。
  - FP8 **2.61 PFLOPS**（dense）/ **5.22 PFLOPS**（sparse，E5M2/E4M3）
  - FP16 **1.3 PFLOPS** / **2.61 PFLOPS**（sparse）；BF16 **1.3 PFLOPS** / **2.61 PFLOPS**（sparse）
  - TF32 **653.7 TFLOPS** / **1.3 PFLOPS**（sparse）
  - FP32 (matrix) **163.4 TFLOPS**；FP32 **163.4 TFLOPS**；FP64 matrix **163.4 TFLOPS**；FP64 **81.7 TFLOPS**
  - INT8 **2.6 POPS** / **5.22 POPS**（sparse）
  - **192 GB HBM3**、8,192-bit、**5.3 TB/s**、LLC 256MB、1530 亿晶体管
  - **750W Peak TBP**、**OAM Module**、PCIe 5.0 x16、**Infinity Fabric 128 GB/s**（非 NVLink）、被动 OAM
  - 架构：CDNA 3（AMD 页面注明 `AMD CDNA™ 3 Architecture`；工艺节点不在该页，业内为 5nm/6nm 混合）

---

## B. 完整来源清单（URL + 日期）

### B.1 NVIDIA 官方 —— 产品页
| 产品 | URL | 页面/文档日期 |
|---|---|---|
| A100 | https://www.nvidia.com/en-us/data-center/a100/ | 现行 |
| H100 | https://www.nvidia.com/en-us/data-center/h100/ | 现行 |
| H200 | https://www.nvidia.com/en-us/data-center/h200/ | 现行（©2024 文档线） |
| GB200 NVL72 | https://www.nvidia.com/en-us/data-center/gb200-nvl72/ | 现行 |
| GB300 NVL72 | https://www.nvidia.com/en-us/data-center/gb300-nvl72/ | 现行 |
| HGX（B300/B200/H200/H100） | https://www.nvidia.com/en-us/data-center/hgx/ | 现行 |
| DGX B200 | https://www.nvidia.com/en-us/data-center/dgx-b200/ | 现行 |
| DGX B300 | https://www.nvidia.com/en-us/data-center/dgx-b300/ | 现行 |
| L40S | https://www.nvidia.com/en-us/data-center/l40s/ | 现行 |
| A800 40GB Active | https://www.nvidia.com/en-in/products/workstations/a800/ | 现行 |
| RTX PRO 6000 Blackwell 系列 | https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000-family/ | 现行 |
| Data Center GPU Line Card | https://docs.nvidia.com/data-center-gpu/line-card.pdf | 2025（含 GB300/B300 列） |

### B.2 NVIDIA 官方 —— Datasheet / Tech Brief（PDF 直链）
| 文档 | URL | 日期/编号 |
|---|---|---|
| A100 datasheet（EN） | https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf | JUN21, 1758950 R4 |
| A100 datasheet（CN） | http://images.nvidia.cn/aem-dam/en-zz/Solutions/data-center/a100/nvidia-a100-datasheet-nvidia-a4-2188504-r5-zhCN.pdf | 2188504 R5 |
| H100 datasheet（EN） | https://dam-cdn.nvd.orangelogic.com/AssetLink/u5hh6fv4r7564i4y673484y3m20083nj.pdf | 现行（资源页 https://resources.nvidia.com/en-us-gpu-resources/h100-datasheet-24306 ） |
| H100 datasheet（CN） | http://images.nvidia.cn/aem-dam/en-zz/Solutions/data-center/h100/nvidia-h100-datasheet-nvidia-a4-2287922-r7-zhCN.pdf | 2287922 R7 |
| H200 datasheet | https://dam-cdn.nvd.orangelogic.com/AssetLink/5o2qgy5d2835ve2pm11i62kv8mphqta8.pdf | ©2024, 3512650 |
| Blackwell Tech Brief（B200/GB200） | https://dam-cdn.nvd.orangelogic.com/AssetLink/gl2l4l4812s5fw0p614s6i8bv6mi3vx5.pdf | 资源页 https://resources.nvidia.com/en-us-blackwell-architecture |
| Blackwell Ultra datasheet（B300/GB300） | https://dam-cdn.nvd.orangelogic.com/AssetLink/1k0p832eq8r5ca0u5383ie5o4tp3bst1.pdf | 资源页 https://resources.nvidia.com/en-us-blackwell-architecture/blackwell-ultra-datasheet |
| L40S datasheet（CN） | http://images.nvidia.cn/cn/RTX/l40s-datasheet-web-a4-zhCN-2907810-2023.9.25.pdf | 2023.9.25, 2907810 |

### B.3 NVIDIA 官方 —— 新闻稿（日期已逐条核对）
| 事件 | URL | 日期 |
|---|---|---|
| Ampere / A100 发布 | https://nvidianews.nvidia.com/news/nvidia-announces-ampere-architecture-and-a100-gpu （现重定向至归档） | 2020-05-14 |
| Hopper / H100 发布 | https://nvidianews.nvidia.com/news/nvidia-announces-hopper-architecture-the-next-generation-of-accelerated-computing | **2022-03-22** |
| H200 发布（SC23） | https://nvidianews.nvidia.com/news/nvidia-supercharges-hopper-the-worlds-leading-ai-computing-platform | **2023-11-13** |
| Blackwell / B200 发布（GTC24） | https://nvidianews.nvidia.com/news/nvidia-blackwell-platform-arrives-to-power-a-new-era-of-computing | **2024-03-18** |
| Blackwell Ultra / GB300 发布（GTC25） | https://nvidianews.nvidia.com/news/nvidia-blackwell-ultra-ai-factory-platform-paves-way-for-age-of-ai-reasoning | **2025-03-18** |
| L40S / 全球数据中心系统商 | https://blogs.nvidia.com.tw/blog/nvidia-global-data-center-system-manufacturers-to-supercharge-generative-ai-and-industrial-digitalization/ | 2023-08-08 |

### B.4 第三方 / 媒体（含传闻与拆解）
| 主题 | 来源与 URL | 日期 |
|---|---|---|
| A800 对华推出 | Reuters: https://www.reuters.com/technology/exclusive-nvidia-offers-new-advanced-chip-china-that-meets-us-export-controls-2022-11-08/ | 2022-11-08 |
| H800 SXM 规格 | 强川科技产品页: https://www.qiangchuan.com/item/7228.html | 未标注（现行） |
| H800 PCIe 规格 | Lenovo ThinkSystem 产品指南（已 withdrawn）: https://lenovopress.lenovo.com/lp1814 | 已下架 |
| H20 规格 | 百度百科 H20 词条（数据截至 2025-04-17）: https://baike.baidu.com/en/item/H20/1510112 | 2025-04-17 |
| H20 禁售 + 55 亿美元减记 | TechPowerUp: https://www.techpowerup.com/335534/us-bans-export-of-nvidia-h20-accelerators-to-china-a-potential-usd-5-5-billion-loss-for-nvidia | 2025-04-16 |
| 15% 收入分成 | Reuters: https://www.reuters.com/world/china/nvidia-amd-pay-15-china-chip-sale-revenues-us-official-says-2025-08-11/ | 2025-08-11 |
| H20 停产 | The Information 经 TechPowerUp: https://www.techpowerup.com/340198/nvidia-reportedly-ends-h20-gpu-production-makes-room-for-b30a | 2025-08-22 |
| B30A 存在（路透） | Reuters 经 TechPowerUp: https://www.techpowerup.com/340068/nvidia-prepares-cut-down-b30a-blackwell-ai-accelerator-for-chinese-market | 2025-08-19 |
| B30A 定价（路透） | Reuters 经 TechPowerUp: https://www.techpowerup.com/340664/nvidias-export-compliant-b30a-for-china-priced-at-roughly-twice-the-h20 | 2025-09-04 |
| RTX 6000D 遇冷（路透独家） | Reuters 经 US News: https://www.usnews.com/news/top-news/articles/2025-09-16/exclusive-nvidias-new-rtx-6000d-chip-for-china-finds-little-favour-with-major-firms-sources-say | 2025-09-16 |
| CAC 叫停采购 | FT 经 TechPowerUp: https://www.techpowerup.com/341078/china-blocks-tech-firms-from-buying-nvidia-ai-accelerators | 2025-09-17 |
| RTX 6000D 84GB 拆解 | TweakTown: https://www.tweaktown.com/news/110135/nvidias-new-rtx-6000d-appears-in-teardown-84gb-gddr7-in-china-compared-to-the-full-96gb/index.html | 2025-09 |
| RTX 6000D 拆解 | WindowsReport: https://windowsreport.com/nvidia-rtx-6000d-teardown-reveals-84gb-gddr7-and-cut-down-blackwell-specs/ | 2025-09 |
| B30A 被美方阻止 | The Information 经 investingLive: https://investinglive.com/stocks/us-blocks-nvidias-scaled-down-ai-chip-sales-to-china-despite-trump-hints-20251107/ | 2025-11-07 |
| H200 对华许可证（10 家企业） | CNBC: https://www.cnbc.com/2026/05/14/us-clears-h200-chip-sales-to-10-china-firms-as-nvidia-ceo-looks-for-breakthrough.html | 2026-05-14 |
| 中国批准 H200 进口但限量 | The Information / Reuters 经 TrendForce: https://www.trendforce.com/news/2026/07/09/news-china-reportedly-to-allow-nvidia-h200-imports-but-approvals-may-be-capped-below-200k-chips-less-than-half-requested/ | 2026-07-09 |
| B200/H800 交叉印证（第三方技术文档） | llm-infra-atlas: https://github.com/llm-infra-atlas/llm-infra-atlas.github.io | 现行 |
| MI300X 官方规格 | AMD: https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html | 现行 |
| MI300 系列发布 | AMD: https://www.amd.com/ja/newsroom/press-releases/2023-12-06-amd-ai-amd-instinct-mi300.html | 2023-12-06 |

### B.5 NVIDIA 官方文档（docs.nvidia.com）—— H800 / H20 SKU 存在性（唯一官方出处）
| 文档 | URL | 更新日期 |
|---|---|---|
| Hopper H20 vGPU Types（含 H20 SXM5 96GB / 141GB） | https://docs.nvidia.com/ai-enterprise/release-8/latest/infra-software/vgpu/reference/hopper-h20.html | 2026-09-02（Release 8.x） |
| Hopper H800 vGPU Types（含 H800 SXM5 80GB / PCIe 80GB / PCIe 94GB NVL） | https://docs.nvidia.com/ai-enterprise/release-8/latest/infra-software/vgpu/reference/hopper-h800.html | 2026-09-02（Release 8.x） |
| Ampere A800 / A100 vGPU Types | https://docs.nvidia.com/ai-enterprise/release-8/latest/infra-software/vgpu/reference/ampere-a800.html | 2026-09-02 |

### B.6 H800 / H20 补充来源
| 主题 | 来源 | 日期 |
|---|---|---|
| H20 算力 dense 口径（15% H100） | 腾讯科技/全天候科技: https://awtmt.com/articles/3702970 | 2023-11-28 |
| H800/H20 dense 同表对照 | Ant Group / LMSYS 工程博客: https://www.lmsys.org/blog/2025-09-26-sglang-ant-group/ | 2025-09-26 |
| H20 dense/sparse 显式分列 | flopper.io: https://flopper.io/compare/intel-data-center-gpu-flex-140-12gb-vs-nvidia-h20-96gb | 现行 |
| BIS 3A090.a 4800 TOPS 门槛原文 | 15 CFR 774: https://www.govinfo.gov/content/pkg/CFR-2024-title15-vol2/pdf/CFR-2024-title15-vol2-subtitleB-chapVII-subchapC.pdf | 2024 版 |
| H800 PCIe NVLink 400GB/s（NVIDIA 官方表转载） | AFOX: https://afox-corp.com/show-118-621-1.html | 现行 |
| H20 141G 渠道现货（120 万元） | 科创板日报: https://finance.jrj.com.cn/2025/03/03073448476126.shtml | 2025-03-03 |
| NVIDIA 8-K（H20 需许可） | SEC: https://www.sec.gov/Archives/edgar/data/1045810/000104581025000082/nvda-20250409.htm | 事件日 2025-04-09，提交 2025-04-15 |
| H20 实际计提 45 亿美元 | ComputerWeekly: https://www.computerweekly.com/news/366625005/Nvidia-takes-45bn-hit-due-export-restrictions | 2025-05-29 |
| 美国恢复 H20 对华销售 | China Daily: https://www.chinadaily.com.cn/a/202507/15/WS6875c252a31000e9a573c180.html | 2025-07-15 |
| BIS 开始发放 H20 许可 + 15% 分成 | JETRO（引 Reuters）: https://www.jetro.go.jp/biznews/2025/08/c62bcbf7f14c3f97.html | 2025-08-18 |
| 中国外交部回应（2025-08-13） | 中国外交部: https://gn.china-embassy.gov.cn/fra/fyrth/202508/t20250815_11690499.htm | 2025-08-13 |
| H20 停产（The Information） | 投中网/虎嗅: https://www.chinaventure.com.cn/news/114-20250825-387731.html | 2025-08-25 |
| CAC 更大范围禁购 | VietnamPlus 转 FT: https://www.vietnamplus.vn/ft-trung-quoc-yeu-cau-cac-cong-ty-cong-nghe-ngung-mua-chip-ai-cua-nvidia-post1062438.vnp | 2025-09-17 |
| H800 PCIe 官方表（已下架 OEM 指南） | Lenovo ThinkSystem lp1814（withdrawn） | 已下架 |

### B.7 B30A / RTX 6000D 补充来源
| 主题 | 来源 | 日期 |
|---|---|---|
| B30A 首次点名（路透） | https://www.reuters.com/world/china/nvidia-working-new-ai-chip-china-that-outperforms-h20-sources-say-2025-08-19/ | 2025-08-19 |
| 白宫阻止 B30A（路透/The Information） | https://www.reuters.com/world/china/us-block-nvidias-sale-scaled-back-ai-chips-china-information-says-2025-11-07/ | 2025-11-07 |
| RTX 6000D 遇冷（路透独家） | https://www.reuters.com/world/china/nvidias-new-rtx6000d-chip-china-finds-little-favour-with-major-firms-sources-say-2025-09-16/ | 2025-09-16 |
| RTX 6000D 拆解 84GB | 快科技: http://m.mydrivers.com/newsview/1103953.html ; TweakTown ; WindowsReport | 2026-02-12 |
| RTX 6000D vs 全血受控基准 | CSDN: https://fjiang.blog.csdn.net/article/details/162849417 | 2026-07-14 |
| RTX 6000D Geekbench 首曝 | i2hard: https://i2hard.ru/publications/49441/?lang=ru | 2025-11-25 |
| B30/B40 命名混淆（路透原报道） | heise: https://www.heise.de/en/news/Cat-and-mouse-Nvidia-s-B40-to-circumvent-China-ban-10397137.html | 2025-05-26 |
| Blackwell Ultra 官方技术博客（die/晶体管/NV-HBI） | https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/ | 2025-08-22 |
| H200 对华出口「微不足道」 | https://en.sedaily.com/international/2026/07/15/nvidias-h200-chip-exports-to-china-have-begun | 2026-07-15 |
| B30A「还在卡」 | 快科技: https://m.mydrivers.com/newsview/1122494.html | 2026-05-15 |

---

## C. 不确定 / 存疑 汇总（按严重程度排序）

### C.1 【严重】B300 / Blackwell Ultra 的 FP4 sparse:dense 比值在 NVIDIA 官方文档内部不自洽

**问题**：Blackwell 一代的 FP4 是本代核心卖点，但 NVIDIA 不同官方页面给出不同比值。

| 出处 | B300 表述 | 隐含比值 |
|---|---|---|
| HGX 产品页 | `FP4 Tensor Core¹ 144 PFLOPS \| 108 PFLOPS`（8卡），脚注1 = `Specification in Sparse \| Dense` | **1.33×**（单卡 18 sparse / 13.5 dense） |
| Blackwell Ultra datasheet 单卡表（GB300 列） | `FP4 Tensor Core¹ 20 PFLOPS \| 15 PFLOPS` | **1.33×** |
| Blackwell Ultra datasheet 单卡表（HGX 列） | `FP4 Tensor Core¹ 18 PFLOPS \| 14 PFLOPS` | **1.29×** |
| Blackwell Ultra datasheet 整柜总表 | `1,440 PFLOPS \| 1,080 PFLOPS` = 72 × (20 \| 15) | **1.33×** |
| DGX B300 产品页 | `FP4 Tensor Core: 144 PFLOPS \| 108 PFLOPS*`，`*Specification shown in sparse \| dense` | **1.33×** |
| 同文档其他精度 | `FP8/FP6 10 PFLOPS`（脚注2 = sparse，dense = ½）、`INT8 330 TOPS`（sparse，dense = ½） | **2.0×** |

**判断**：FP4 的 **1.33×** 在两个独立官方源（HGX 页、DGX B300 页）以及 Ultra datasheet 的整柜汇总中一致出现，因此 **B300 的 dense FP4 = 13.5 PFLOPS/卡（HGX）/ 15 PFLOPS/卡（GB300）应视为 NVIDIA 官方口径**；但「sparse 只比 dense 高 33%」与同表其他精度的 2× 规则不一致，**NVIDIA 未解释原因**（可能 FP4 的 sparse 加速比受限于非 Tensor Core 路径）。

**给寒武纪对比的建议**：若对比 MLU 的 INT8/INT16 dense 算力，**必须用 B300 的 dense FP4（13.5 或 15 PFLOPS）**，不能使用 18/20 PFLOPS，也不能简单除以 2。

### C.2 【严重】B200 单卡 FP4 在官方源之间也是 1.33× vs 2.0× 并存

- HGX 页 HGX B200 8 卡：`144 PFLOPS | 72 PFLOPS`（Sparse | Dense）→ 单卡 **18 / 9**，比值 **2.0×**。
- Blackwell Tech Brief：HGX B200 单卡 `9/18 petaFLOPS`（`FP4 Tensor Core Dense/Sparse`）→ 一致，**2.0×**。
- GB200 产品页 Superchip：`40 | 20 PFLOPS`（NVFP4）→ **2.0×**。
- **但 GB200 NVL72 整柜产品页只写 `1,440 | 720 PFLOPS`，其中 720 是 dense**；而 K 提到的 GB300 NVL72 在产品页写 `1440 | 1080`。
- **结论**：B200 的 FP4 是 **2.0×**；B300 的 FP4 是 **1.33×**。这一代际内的不一致可能是真实的架构差异（Blackwell Ultra 增加了 attention 加速但 FP4 sparse 增益下降），而非笔误，但因 NVIDIA 未说明，仍列为存疑。

### C.3【中】H200 的显存带宽：4.2 还是 4.8 TB/s

- **H200 SXM = 4.8 TB/s**：产品页、datasheet、官方新闻稿（2023-11-13）三者一致。**无争议。**
- **H200 NVL**：
  - 产品页表：`4.8TB/s`
  - datasheet 表 + datasheet 正文 + Key Features：`4.2TB/s`
  - **判断**：datasheet（编号 3512650，©2024）更具体且三处自洽，建议采信 **4.2 TB/s**，并注明产品页写 4.8。

### C.4【中】H100 PCIe 与 H100 NVL 的混淆

- 英文 datasheet 表头为 `H100 SXM` / `H100 NVL`（NVL：94GB、3.9 TB/s、FP64 30、FP32 60、BF16 sparse 1,671）。
- 中文 datasheet 表头为 `H100 SXM` / `H100 PCIe`，但第二栏数值与英文版的 NVL 栏**完全相同**（FP64 24、FP32 48、BF16 sparse 1,600、80GB、2TB/s、350W、NVLink 600GB/s）。
- 数值上：NVL 栏是 94GB/3.9TB/s，中文 "PCIe" 栏是 80GB/2TB/s → **两者不是同一产品**。
- **判断**：中文 datasheet 的「H100 PCIe」栏实际是**标准 PCIe 80GB SKU**（2TB/s、350W、NVLink 600GB/s），英文 datasheet 把同一栏改标为 NVL 并替换为 94GB/3.9TB/s 的数值。**引用 350W / 2.0 TB/s 时请注明「H100 PCIe（中文 datasheet）」；引用 94GB / 3.9 TB/s 时注明「H100 NVL（英文 datasheet）」。**
- **给对比的建议**：若寒武纪对比对象是「H100 PCIe」，用 80GB / 2.0 TB/s / 350W / NVLink 600GB/s / BF16 dense 800；若对象是「H100 NVL」，用 94GB / 3.9 TB/s / 350–400W / NVLink 600GB/s / BF16 dense 835.5。

### C.5【中】B200 单卡显存容量：180GB vs 186GB vs 192GB

| 出处 | 数值 |
|---|---|
| Blackwell Tech Brief（HGX B200 单卡） | `Up to 192 GB HBM3e` |
| DGX B200 产品页 | `1,440 GB total` → 单卡 **180GB** |
| HGX 产品页 | `Total Memory 1.4 TB` → 单卡 **175GB**（或 180GB 取整） |
| GB200 产品页 Superchip | `372 GB HBM3E` → 双 GPU → 单卡 **186GB** |
| Lenovo ThinkSystem HGX B200 产品指南 | **180GB** |

**判断**：B200 die 的 HBM3e 容量上限 NVIDIA 官方表述为 192GB，但实际出货的 HGX B200 为 180GB、GB200 内为 186GB。**建议写「180GB（HGX B200 实际配置）/ 186GB（GB200 内）/ 官方表述 up to 192GB」。**

### C.6【中】HGX B300 的 INT8 与 FP64 在官方页面之间冲突

- **INT8**：HGX 页写 HGX B300 8 卡 `INT8 Tensor Core² 3 POPS`（即 3,000 POPS 整机 → 单卡 375 TOPS，与 Ultra datasheet 的 307 TOPS dense 不符）；Ultra datasheet 单卡写 `330 TOPS`（GB300 列，sparse → dense 165）/ `307 TOPS`（HGX 列，sparse → dense 153.5）。**三处互不吻合。**建议以 Ultra datasheet 单卡值为准并标注冲突。
- **FP64**：HGX 页写 HGX B300 整机 `FP64/FP64 Tensor Core 10 TFLOPS`（比 HGX B200 的 296 TFLOPS 低 30 倍，**明显异常**）；Ultra datasheet 单卡写 `1.3 TFLOPS`（GB300 列）/ `1.2 TFLOPS`（HGX 列）→ 8 卡应为 9.6–10.4 TFLOPS，**与 HGX 页的 10 TFLOPS 吻合**。故 HGX 页的 `10 TFLOPS` 是**正确的**，而「B200 的 296 TFLOPS」才是被高估的一格（Ultra datasheet 的 B200 单卡为 37 TFLOPS × 8 = 296，自洽）。**即：B300 的 FP64 大幅低于 B200 是真实的（Blackwell Ultra 削减了 FP64 单元），不是笔误。**
- **B300 单卡 FP64 在表格中应写 1.2–1.3 TFLOPS**，并注明「相比 B200 的 37 TFLOPS 大幅削减」。

### C.7【中】GB200 NVL72 整柜 HBM 容量 13.4 / 13.5 TB

- 产品页：`13.4 TB HBM3E`
- Blackwell Tech Brief 系统表：`13.5 TB`
- 按 72 × 186GB = 13.392 TB ≈ **13.4 TB**（产品页正确）；13.5 TB 可能按 72 × 187.5 或四舍五入。**建议写 13.4 TB（产品页），并注明 Tech Brief 写 13.5 TB。**

### C.8【已解决 → 见 §E.1】H20 的 dense/sparse 口径

**本节初版判断（「296 可能是稀疏值」）已被推翻。** 经四条独立证据 + BIS 合规算术校验，**H20 的 148（BF16/FP16）与 296（FP8/INT8）均为 DENSE**；FP8 sparse = 592 仅为第三方推算，NVIDIA 从未公布 H20 的 sparse 值。**请以 §E.1 为准。**

仍存疑的只剩 **H20 的 FP32 / TF32**：
- TechPowerUp（2025-04-16）称 FP32 ≈ 74 TFLOPS；
- 百度百科与多数中文来源写 TF32 74 / FP32 44；
- 按 H100 的 TF32(dense 494.5) : FP32(67) ≈ 7.4:1 比例，H20 若 FP32 = 44 则 TF32 dense 应 ≈ 325（而非 74）；若 FP32 = 74 则 TF32 dense 应 ≈ 548。**两组数字都无法与 H100 的比例关系自洽。**
- **建议：H20 的 TF32（74）与 FP32（44）标注为「第三方数据，存疑」；引用时同时给出两个候选值。**

### C.9【已部分解决 → 见 §E.3】H800 的 FP64 数值

**初版把 1 TFLOPS 与 0.8 TFLOPS 视为同一 SKU 的冲突；实为 SKU 差异：**
- **H800 SXM5 80GB：FP64 = 1 TFLOPS**（渠道商转述 NVIDIA 官方规格文本）
- **H800 PCIe 80GB：FP64 = 0.8 TFLOPS**（NVIDIA 官方 PCIe 规格表，经 AFOX 完整转载）
- 对比 H100 SXM 的官方 FP64 = 34 TFLOPS、H100 PCIe = 24 TFLOPS → **H800 两个 SKU 的 FP64 都被大幅削减（约 1/30）**，这是 H800 与 H100 除 NVLink 之外的第二项实质差异。

**不可采用的来源**：SEO 渠道页 sxrsgk.com 称「H800 FP64 = 30 TFLOPS 与 H100 相同」——**错误，已剔除**。
**另需注意**：cpudb.cc / TechPowerUp 系给出的 H800「FP16 237.2 / FP32 59.30 / FP64 29.65 TFLOPS」是 **CUDA-core 向量算力**，不是 Tensor Core 规格，不可与 989/1979 混用。



### C.10【中】B30A 的全部规格均无官方来源（详见 §D）
- NVIDIA 从未发布 B30A 产品页/datasheet（已核实 nvidia.com 与 nvidia.cn 均无）。
- 唯一有原始出处的是 **路透 2025-08-19**（存在该 SKU、单 die、算力减半）与 **路透 2025-09-04**（定价约 H20 两倍）。
- TechPowerUp 的 7.5 / 3.75 / 1.875 / 0.94 PFLOPS 是**记者按 B300 减半自行推算**，非任何一手来源。**必须标注为「第三方推算」。**
- **2025-11-07 路透/The Information 报道美方已阻止 B30A 出口许可**，NVIDIA 正重新设计。截至 2026-09-18 **未找到 B30A 获批或量产的确切报道**。
- **显存容量、NVLink 带宽、TDP、NVL8 等细节均未找到可靠出处** → 一律写「未找到公开数据」。**「~800W」经专门搜索确认无任何出处**（见 §D.1）。

### C.11【中】RTX 6000D 的规格仅来自拆解/媒体（详见 §D.7）
- NVIDIA 官方只有 RTX PRO 6000 Blackwell（96GB GDDR7、PCIe Gen5、400–600W/600W/300W），**没有 "RTX 6000D" 页面**。
- **84GB GDDR7** 来自拆解实测（TweakTown / WindowsReport / 快科技），可信度较高但非 NVIDIA 官方。
- **算力（TFLOPS）、显存带宽、TDP** 均**未找到 NVIDIA 原文** → 写「未找到公开数据」。
- **NVLink**：RTX PRO 6000 Blackwell 全系**官方页面未列 NVLink**，故中国版亦应为无 NVLink，但严格说属「未披露」而非「官方确认无」。

### C.12【低】L20 几乎无可靠公开数据
- nvidia.cn/data-center/l20/ 返回 404，无 Wayback 快照。
- 媒体常引用的「48GB GDDR6 / 275W / 119 TFLOPS FP16 稀疏」**未核实到原始出处**，本报告不予采用。

### C.13【已解决 → 见 §E.2】H20 141GB 版本确实存在
**本节初版判断（「未证实」）已被推翻。** NVIDIA AI Enterprise 8.x 官方 vGPU 文档（docs.nvidia.com，更新至 2026-09-02）明确列出 **NVIDIA H20 SXM5 96 GB** 与 **NVIDIA H20 SXM5 141 GB** 两个 SKU（vGPU profile 前缀分别为 `H20-` 与 `H20X-`，含 `H20X-141C`）。
<https://docs.nvidia.com/ai-enterprise/release-8/latest/infra-software/vgpu/reference/hopper-h20.html>
141GB 版为 **HBM3e、带宽仍 4.0 TB/s、TDP 仍 400W**，形态仍为 SXM5。**仍存疑的只剩其首发/量产日期**（未找到公开数据；仅能确证 2025-03-03 已在渠道现货）。

### C.14【低】L40S 产品页的 FP16 数字与 datasheet 不一致
- 产品页：`FP16 733 teraFLOPS*`（`*With Sparsity`）—— 只给稀疏值，未给 dense。
- 中文 datasheet：`FP16 362.5 | 733*`。
- **对比时务必用 362.5（dense）**，不要用 733。这是「产品页只展示 sparse」导致误用的典型例子。

---

## D. B30A 与 RTX 6000D —— 补充核查（子代理独立研究，2026-09-18）

### D.1 【重要更正】B30A 的「~800W」没有任何出处
委托简报里假设的「B30A 约 800W」**经全面搜索未找到任何媒体或官方出处**。本报告已删除该数字，改为「未找到公开数据」。
唯一在流通的功耗数字是 **600W**，且它是挂在 **B300A**（TSMC 4nm + CoWoS-L、144GB HBM3E）上的，**不能确定等同于 B30A**。

### D.2 【重要澄清】B30A 的 NVLink「减半」是误读
- NVIDIA官方：B300 的 NVLink 5 = **1.8 TB/s 双向** GPU-to-GPU。
- SemiAnalysis InferenceX 写 B300「scale-up bandwidth per chip **900 GB/s**」—— 这是**单向/每方向**口径。
- **两者描述的是同一条链路**，不是两个不同的速率。因此不能据此推断 B30A 的 NVLink 是「900 GB/s（B300 之半）」。
- Reuters 与 Tom's Hardware 均称 B30A **保留 NVLink** 做 scale-up；Tom's 明确指出「是否削减 NVLink 数量以限制机架级/大集群规模，尚不清楚」。
- **B30A 的 NVLink GB/s 数值：未找到公开数据。**

### D.3 B30A 命名陷阱（B30 / B40 / B30A 被媒体混用）
2025 年报道中「B30」指代**两个不同的产品**：
1. **GDDR7 工作站/AI 卡**（最早被报道为 "B40" 或 "RTX PRO 6000D"）→ 最终以 **RTX 6000D** 出货；
2. **HBM3E 数据中心加速器 B30A**（B300 单 die）→ 被白宫阻止。

Reuters 2025-05-24、ETnews/芯智讯 2025-09-09、WSJ 2025-08-26 均用 "B30"/"B40" 指 GDDR7 那张卡，导致大量中文/英文二手报道把两者混为一谈。
来源：heise 2025-05-26 <https://www.heise.de/en/news/Cat-and-mouse-Nvidia-s-B40-to-circumvent-China-ban-10397137.html>；芯智讯 <https://cloud.tencent.cn/developer/article/2643068>；EEPW <http://m.eepw.com.cn/article/202508/473337.html>

### D.4 B30A 各精度估算的冲突谱（不要取平均）

| 精度 | 估算值 A | 估算值 B | 估算值 C | 说明 |
|---|---|---|---|---|
| FP4 (NVFP4) | **7.5 PFLOPS**（Tom's / TPU / AInvest，主流） | 3.5 PFLOPS（按 B100 减半的另一种推算） | — | 差异 2.1× |
| FP8/FP6 | 5 PFLOPS（Tom's、AInvest） | 3.75 PFLOPS（TechPowerUp） | — | 差异 1.33×；**两者都高于 B300 dense FP8（2.25 PFLOPS）的两倍**，说明基线口径混乱 |
| BF16/FP16 | 2.5 PFLOPS（Tom's，未标 dense/sparse） | 1.875 PFLOPS（TPU） | ~1.125 PFLOPS（按 B300 dense 2,250 TFLOPS 折半） | **差异 2.2×** |
| TF32 | 1.25 PFLOPS（Tom's） | 0.94 PFLOPS（TPU） | — | 差异 1.33× |
| INT8 | 未找到公开数据 | | | Tom's 表的 INT8 行在 EEPW 转载中内部矛盾，不可用 |
| FP32（向量） | 未找到公开数据 | | | NVIDIA 从未发布 B 系列的向量 FP32 |

**根因**：Tom's Hardware 使用的 B300 基线是「BF16 = 5 PFLOPS」（= NVIDIA 的 **sparse** 口径），而 SemiAnalysis 的 B300 dense BF16 = **2.25 PFLOPS**。因此 Tom's 的「B30A 2.5 PFLOPS」并不是「B300 dense 的一半」，而是「B300 sparse 的一半」。**这就是所有 B30A 估算出现 2 倍分歧的根源。**

### D.5 B30A 的其他冲突（均未解决）
- **单 die vs 双 die**：Reuters/Tom's/TPU 说单 die（取 B300 两 die 之一）；WCCFTech 2025-08 说双 die。**未解决。**
- **显存**：144GB HBM3E（= B300 288GB 的 50%，最常引用）；快科技/MyDrivers 2026-05-15 给 **96–144GB，带宽 8→4 TB/s**；WCCFTech 说 8-Hi HBM3E（B300 为 12-Hi，被单 die 说法否定）。
- **封装**：Tom's 表格列 **CoWoS-S**（B200/B300 为 CoWoS-L）；中文报道说相关的 B300A 用 **CoWoS-L**。
- **die 代号 "GB110"**：**未找到任何出处**。NVIDIA 官方 Blackwell Ultra 材料不使用 GB1xx 命名。
- ⚠️ **陷阱**：MyDrivers 2026-05-15 文中出现的「TPP 60,000 → 30,000」是**出口管制的 Total Processing Performance 指标**，被误标成 PFLOPS，**不可当作算力使用**。

### D.6 B30A 出口管制时间线（2025-08 → 2026-09）

| 日期 | 事件 | 来源 |
|---|---|---|
| 2025-08-19 | 路透首次点名 **B30A**（明确为暂定名）：单 die、约 B300 一半原始算力、保留 HBM 与 NVLink | <https://www.reuters.com/world/china/nvidia-working-new-ai-chip-china-that-outperforms-h20-sources-say-2025-08-19/> |
| 2025-08-22 | The Information：NVIDIA 停产 H20（Hopper），为 B30A 让路 | <https://www.techpowerup.com/340198/nvidia-reportedly-ends-h20-gpu-production-makes-room-for-b30a> |
| 2025-08-26 | WSJ：对华 "B30" 定位为 baseline Blackwell 的 **80%**，已提交出口许可申请 | EEPW 转载 <http://m.eepw.com.cn/article/202508/473337.html> |
| 2025-09-04 | 路透：定价约 H20 两倍（US$20,000–24,000） | <https://www.techpowerup.com/340664/nvidias-export-compliant-b30a-for-china-priced-at-roughly-twice-the-h20> |
| 2025-09-17 | FT：中国网信办要求阿里/百度/腾讯停止评估与采购 NVIDIA 出口改型（点名 RTX PRO 6000D，**未点名 B30A**） | <https://www.techpowerup.com/341078/china-blocks-tech-firms-from-buying-nvidia-ai-accelerators> |
| **2025-11-07** | **The Information（经路透）：白宫通知联邦机构，不会批准 B30A 对华销售。** NVIDIA 已向多家中国客户送样，正重新设计以争取翻盘；NVIDIA 对路透表态「在中国竞争性数据中心计算市场份额为零」，故未纳入业绩指引 | <https://www.reuters.com/world/china/us-block-nvidias-sale-scaled-back-ai-chips-china-information-says-2025-11-07/> |
| 2026-01（报道） | 美国商务部放宽仅覆盖 **H200 与 AMD MI325X** 的 case-by-case 审查；Blackwell 级产品未纳入 | flopper.io（二手聚合，仅作线索） |
| 2026-05-14 | 美国批准 H200 对华销售至约 10 家企业 | <https://www.reuters.com/business/retail-consumer/us-clears-h200-chip-sales-10-china-firms-nvidia-ceo-looks-breakthrough-2026-05-14/> |
| 2026-05-15 | 快科技：H200 出口放行，**B30A「还在卡」**；判断在 Rubin 量产前获批可能性低 | <https://m.mydrivers.com/newsview/1122494.html> |
| 2026-07-14 | 美商务部 Under Secretary Kessler 向国会作证：H200 对华出口已开始但「非常少/微不足道」 | <https://en.sedaily.com/international/2026/07/15/nvidias-h200-chip-exports-to-china-have-begun> |
| **2026-09-18（今日）** | **无任何 B30A 获批、发布、量产或规模出货的公开证据** → 一律写「未找到公开数据」 | — |

**⚠️ 更正**：「NVIDIA 暂停中国 Blackwell 产线」的传闻实际是指 **2025-08 停产 H20（Hopper）**，与 Blackwell 产线无关。

### D.7 RTX 6000D —— 已实际出货，规格来自拆解与实测（非 NVIDIA官方）
**NVIDIA 官方**：**没有** "RTX 6000D" 产品页或 datasheet。命名在各方报道中不一：`RTX 6000D` / `RTX PRO 6000D` / `RTX Pro 6000D`。
全血版 RTX PRO 6000 Blackwell 的官方页（<https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000-family/>）**只列 96GB GDDR7 / PCIe Gen5 x16 / 400–600W / 600W / 300W / 外形**，**不列任何算力、带宽、NVLink** → 这些对全血版亦属「官方未披露」。

**可靠性较高的实测数据（第三方，零售硬件拆解 + 受控基准）**：

| 项目 | RTX 6000D（中国版） | RTX PRO 6000 Blackwell（全血） | 差异 |
|---|---|---|---|
| 架构/工艺 | Blackwell，GB202，TSMC 4N，CC 12.0 | 同 | — |
| CUDA 核心 | **19,968**（156 SM） | 24,064 | **−17%** |
| Tensor 核心 | 624 | 752 | −17% |
| RT 核心 | 156 | 188 | −17% |
| 频率 | 2,430 MHz | 2,617 MHz | −7% |
| 显存 | **84GB GDDR7**（28×3GB 三星，正反各 14 颗） | 96GB GDDR7 | −12.5% |
| 位宽 | **448-bit** | 512-bit | — |
| 带宽（标称） | **1,568 GB/s** | 1,792 GB/s | −13% |
| 带宽（实测 device copy） | 1,279.44 GB/s | — | — |
| 功耗上限 | **600W**（驱动读值；受控基准 P90 600.65W / max 611.15W） | 600W | — |
| 外形 | 无风扇被动双插槽、**无视频输出**（TCC 计算模式）、16-pin CEM5 | 双插槽 | — |
| NVLink | **无** | 无 | — |
| PCIe | Gen5 x16 | Gen5 x16 | — |

**实测算力（第三方，CSDN 2026-07-14 受控基准，16384×16384 GEMM）**：

| 精度 | RTX 6000D | RTX PRO 6000 | 比例 |
|---|---|---|---|
| BF16 dense GEMM | **144.09 TFLOPS** | 401.01 | **36%** |
| FP16 dense GEMM | 144.02 TFLOPS | 397.67 | 36% |
| INT8→INT32 | 140.32 TOPS | 238.41 | 59% |
| TF32 GEMM | 72.05 TFLOPS | 194.99 | 37% |
| FP32 GEMM | 66.67 TFLOPS | 75.48 | 88% |
| FP32（规格，向量） | ~97.04 TFLOPS | 125 TFLOPS | 78% |

> 🔴 **最重要的发现**：核心数只减 17%，但**稠密 BF16/FP16 GEMM 只剩全血的 36%**，且 **TF32/FP32 比值从全血的 2.58× 掉到 1.08×** —— 说明 NVIDIA 对低精度 Tensor Core 路径做了**产品级限流**，而不只是砍 SM。
> **给寒武纪对比的启示**：不能按「核心数比例」或「官方标称峰值」推算 RTX 6000D 的实际算力；必须用实测 dense 值。

**带宽的三次修正（务必标注版本）**：
1. **1,100 GB/s**（2025-07 早期泄露，"384-bit @ 23 GHz"）→ **已被证伪**；
2. **1,398 GB/s**（路透 2025-08-21 报道的**计划值**，刻意压在 2025-04 设定的 **1.4 TB/s** 出口管制阈值之下）→ 与实际出货不符（该数字后来被用于全球版 RTX PRO 5500）；
3. **1,568 GB/s**（448-bit，2025-11-25 Geekbench 泄露 + 2026-02-12 拆解 + 2026-07 实测）→ **实际出货口径**。

**其他冲突**：**最大功耗**：2026-02 Bilibili 拆解实测「刚过 400W」vs 2026-07 受控基准 611W → 只有在未跑满时才能自洽；**功耗上限 = 600W**。**L2 缓存**：实测 112 MiB vs CpuTronic 写 128 MB。**FP64**：1.516 TFLOPS（CpuTronic，低可信度）。
⚠️ **陷阱**：CpuTronic 的「FP16 97.04 TFLOPS」**不是 Tensor FP16**，而是该卡的 **FP32 向量吞吐**，不可混用。

### D.8 2026 年中国特供新品：**无 B30A 继任者**
- **未找到任何 B30A 继任者的发布或可靠报道。**
- 截至 2026 年中，美国唯一批准的 NVIDIA 对华数据中心产品是上一代 Hopper **H200**（以及 AMD MI325X）：25% 美国收入分成、每客户上限约 **75,000** 颗、总出口上限约为美国本土出货量的 50%；首批约 **2026-07** 到货，官方口径为「微不足道」。
- 中方闸门同步收紧：2026-05 中国安全部门在「安可」框架下认证了 **9 款国产 AI 处理器**（华为 Ascend 310/910、阿里平头哥真武、壁仞、海光等），**结构性排除外国芯片进入党政/信创采购**。
- NVIDIA 对华 Blackwell 数据中心收入从 FY2026 Q1 的 **46 亿美元跌至 FY2027 Q1 的零**，且 FY2027 Q2 指引假设中国数据中心计算收入为零（NVIDIA SEC 文件 CFO 评论，经 AInvest 2026-08-20 转述）。

---

## E. H800 / H20 —— 补充核查（子代理独立研究，2026-09-18）

### E.1 【重大更正】H20 的 296 / 148 是 **DENSE**，不是 sparse
**本报告初版把 H20 的 296 TFLOPS 误标为「稀疏」。经核查，296 实为 dense。** 四条独立证据：
1. **腾讯科技/全天候科技（2023-11-28）**：H20 单卡算力 `0.148P (FP16) / 0.296P (Int8)`，并称「算力约等于 **50% A100 和 15% H100**」。H100 BF16/FP16 **dense = 989**，×15% = **148.4**；A100 dense = 312，×50% = 156。**只有 148 是 dense 口径，这个 15% 才成立。** <https://awtmt.com/articles/3702970>
2. **Ant Group / LMSYS 工程博客（2025-09-26）**：同一张表用 **dense** 口径列 H800（FP16 989 / FP8 1979 = H100 dense），H20 列 FP16 148 / FP8 296 → **同表即同口径 → dense**。 <https://www.lmsys.org/blog/2025-09-26-sglang-ant-group/>
3. **flopper.io 规格页**：**显式分列** "Peak (dense)" 与 "2:4 Sparse"：FP16 dense **148**、FP8 dense **296**、**FP8 sparse 592**、INT8 dense **296**；其余精度（TF32/BF16/FP16/INT8 的 sparse）明确标注 "not published"。 <https://flopper.io/compare/intel-data-center-gpu-flex-140-12gb-vs-nvidia-h20-96gb>
4. **合规算术校验（合逻辑推导）**：美国 BIS 2023-10-17 规则 3A090.a 的门槛是「总算力之和 ≥ 4800 TOPS」（15 CFR 774）。H20 若 INT8 dense = 296 → TPP = 2 × 296 × 8 = **4,736 < 4,800**，**刚好卡在门槛之下**。若 296 是 sparse（即 dense = 148 → TPP = 2,368），NVIDIA 本可把规格做高一倍仍合规。**因此 296 = dense 是合规设计的前提。** <https://www.govinfo.gov/content/pkg/CFR-2024-title15-vol2/pdf/CFR-2024-title15-vol2-subtitleB-chapVII-subchapC.pdf>

> ⚠️ 百度百科 H20 词条在 INT8/FP8 行标了 `*`（=稀疏），**属于标注错误**。本报告不采用其口径。
> **NVIDIA 从未公布 H20 的 sparse 算力**：TF32/BF16/FP16/INT8 的 sparse 均为「未找到公开数据」；FP8 sparse = 592 仅为第三方推算。

**修正后的 H20 算力（dense 为主口径）**：

| 精度 | H20 dense | H20 sparse |
|---|---|---|
| BF16/FP16 Tensor | **148** | 未公布（推算 296） |
| FP8 Tensor | **296** | 592（第三方） |
| INT8 Tensor | **296** TOPS | 未公布 |
| TF32 Tensor | **74** | 未公布 |
| FP32（向量） | **44** | n/a |
| FP64 | **1** | n/a |

### E.2 【重要】NVIDIA 官方文档确认 H20 有 **141GB HBM3e** 版本，且 H20 **只有 SXM5**
NVIDIA AI Enterprise 8.x vGPU 参考文档（更新至 2026-09-02）明确列出：
- **NVIDIA H20 SXM5 96 GB** 与 **NVIDIA H20 SXM5 141 GB**（vGPU profile 前缀 `H20-` 与 `H20X-`，如 `H20X-141C`）<https://docs.nvidia.com/ai-enterprise/release-8/latest/infra-software/vgpu/reference/hopper-h20.html>
- **NVIDIA H800 SXM5 80 GB**、**NVIDIA H800 PCIe 80 GB**、**NVIDIA H800 PCIe 94 GB（H800 NVL）**（profile 前缀 `H800-` 与 `H800L-`）<https://docs.nvidia.com/ai-enterprise/release-8/latest/infra-software/vgpu/reference/hopper-h800.html>

**这两个页面是本报告找到的唯一 NVIDIA 官方（docs.nvidia.com）载有 H20 / H800 SKU 的文档。** 它们只证明**SKU 存在与显存容量**，不含算力/带宽/NVLink 数值。
→ **H20 141GB 版本确认存在**（初版写「未证实」，现更正）。141GB 版为 **HBM3e、带宽仍 4.0 TB/s、TDP 仍 400W**。
→ **H20 只有 SXM5，无 PCIe SKU**；**H800 有 SXM5 / PCIe 80GB / PCIe 94GB（NVL）三款**。
→ **H20 141GB 的首发/量产日期：未找到公开数据**；仅能确证 2025-03-03 已在渠道现货（科创板日报：141G 整机含税 120 万元、96G 97 万元）。<https://finance.jrj.com.cn/2025/03/03073448476126.shtml>

### E.3 【重要】H800 PCIe **有** NVLink，400 GB/s
NVIDIA H800 PCIe 官方规格表（由渠道商 AFOX 完整转载）明列：`Interconnect: NVLink: 400GB/s, PCIe Gen5: 128GB/s` <https://afox-corp.com/show-118-621-1.html>

**修正后的 H800 规格（三 SKU）**：

| 指标 | H800 SXM5 80GB | H800 PCIe 80GB | H800 PCIe 94GB（H800 NVL） |
|---|---|---|---|
| BF16/FP16 dense | **989** | 756 | 未找到公开数据 |
| BF16/FP16 sparse | 1,979 | 1,513 | 未找到公开数据 |
| FP8 dense | **1,979** | 1,513 | 未找到公开数据 |
| FP8 sparse | 3,958 | 3,026 | 未找到公开数据 |
| INT8 dense TOPS | **1,979** | 1,513 | 未找到公开数据 |
| INT8 sparse TOPS | 3,958 | 3,026 | 未找到公开数据 |
| TF32 dense | **494** | 378 | 未找到公开数据 |
| TF32 sparse | 989 | 756 | 未找到公开数据 |
| FP32（向量） | **67** | 51 | 未找到公开数据 |
| FP64 | **1** | **0.8** | 未找到公开数据 |
| 显存 | 80GB HBM3 | 80GB（类型未公开） | 94GB（类型未公开） |
| 带宽 | **3.35 TB/s** | **2.0 TB/s** | 未找到公开数据 |
| NVLink 双向/GPU | **400 GB/s**（NVLink 4） | **400 GB/s** | 未找到公开数据 |
| PCIe | Gen5 128 GB/s | Gen5 128 GB/s | Gen5 128 GB/s |
| TDP | **700W** | **350W** | 未找到公开数据 |
| 形态 | SXM5 / HGX 8 卡 | PCIe 双槽 | PCIe 双槽 |
| 最大 scale-up 域 | 8（NVLink Switch System 可至 256） | 1–8 | 未找到公开数据 |

其他：H800 SXM 132 SM / 16,896 CUDA / 528 Tensor Core / 814mm² GH100 / 800 亿晶体管（**SM 数为第三方，NVIDIA 未公开**）。
**H800 = 「只砍 NVLink（900→400）+ 砍 FP64（34→1）」的 H100**，稠密算力与 H100 同级（989/1,979）。

### E.4 H20 与 H800 的定位反转（对寒武纪对比很关键）

| | H800 SXM | H20 SXM5 |
|---|---|---|
| BF16 dense | 989 | **148**（约 1/6.7） |
| FP8 dense | 1,979 | **296**（约 1/6.7） |
| 显存带宽 | 3.35 TB/s | **4.0 TB/s**（更高） |
| NVLink | **400 GB/s**（砍半） | **900 GB/s**（不砍） |
| TDP | 700W | **400W** |

→ **H800 强在算力（prefill/训练），H20 强在带宽与互连（decode/长上下文推理）**。若与寒武纪 MLU 的推理场景对比，H20 是更合适的对标对象；训练场景则对标 H800。

### E.5 出口管制时间线（H800/H20，精确日期）

| 日期 | 事件 | 来源 |
|---|---|---|
| 2022-10-07 | 首轮管制，A100/H100 禁售 | Reuters |
| **2022-11** | A800 与 **H800** 作为替代型号推出（管制后约 1 个月） | Reuters 经 Economic Times 2024-01-08 |
| **2023-03-21/22** | **NVIDIA 发言人公开承认 H800**；产业消息称片间速率**降到 H100 的一半** | 韩联社转 Reuters 2023-03-21 <https://cb.yna.co.kr/gate/big5/m-cn.yna.co.kr/view/AKR20230322067300009> |
| 2023-10-17 | 管制更新（≥4800 TOPS / 性能密度 ≥5.92），**A800/H800 被禁** | 腾讯科技/全天候 2023-11-28 |
| 2023-11 | H20/L20/L2 规划；H20 原定 2023-11 发布，**因服务器集成问题延期** | Reuters 2024-01-08 |
| 2023 年末 | H20 定制完成，**2024 年大规模发货** | 腾讯科技 2025-08-11 |
| 2024-01-08 | H20 计划 **2024 Q2 量产**，首批量有限 | Reuters |
| **2025-03-03** | **H20 141G 整机渠道现货**，含税 120 万元；96G 97 万元 | 科创板日报 <https://finance.jrj.com.cn/2025/03/03073448476126.shtml> |
| **2025-04-09** | **美方通知 NVIDIA：H20 出口需许可**（8-K 事件日） | NVIDIA 8-K（SEC，提交 2025-04-15）【NVIDIA官方监管文件】 |
| **2025-04-15** | NVIDIA 公告预计计提**最高 55 亿美元**，涉 H20 库存与采购承诺 | 同上；MarketWatch |
| 2025-05-28 | Q1 FY26 电话会：**实际计提 45 亿美元**（预估 55 亿 → 实际 45 亿，须分列日期口径） | ComputerWeekly 2025-05-29 |
| **2025-07-14/15** | **美国确认恢复 H20 对华销售** | China Daily 2025-07-15；BBC 2025-07-15 |
| **2025-08-08** | 据报道 **BIS 开始发放** H20/MI308 出口许可 | JETRO 2025-08-18（引 Reuters 8-12） |
| **2025-08-11** | **特朗普公开确认 15% 收入分成换出口许可**（FT 首发） | Reuters 2025-08-11 <https://www.reuters.com/world/china/nvidia-amd-pay-15-china-chip-sale-revenues-us-official-says-2025-08-11/> |
| **2025-08-12** | **Reuters/Bloomberg：中国监管要求本土企业避免使用 H20**（尤其政府/国安用途），非全面禁售 | JETRO 引 Reuters 8-12 与 Bloomberg |
| 2025-08-13 | 中国外交部林剑答「我不掌握你提到的情况」 | 外交部记者会记录 2025-08-13【中国政府官方】 |
| 2025-08-22 | The Information：NVIDIA 通知安靠、三星暂停 H20 相关生产 | 投中网/虎嗅 2025-08-25【传闻】 |
| 2025-09-08 | CFO Kress：预计 H20 对华收入 **20–50 亿美元** | 观察者网 2025-09-10 |
| **2025-09-17** | **FT：CAC 要求字节、阿里等停止采购/测试 NVIDIA AI 芯片并取消订单**（首当其冲 RTX Pro 6000D）；FT 称此禁令**比此前针对 H20 的指导更严格** | VietnamPlus 转 FT 2025-09-17 |

> ⚠️ **准确表述**：针对 H20 的「劝退」最早是 **2025-08-12** 的 Reuters/Bloomberg 报道（外交部 8-13 未确认）；**2025-09-17 FT 报道的是范围更大的禁令**（含 RTX Pro 6000D）。**不能把「H20 被 9 月禁购」当作准确表述。**

### E.6 H800 / H20 的冲突清单（含权威性判读）

| 编号 | 冲突 | 判读 |
|---|---|---|
| E-C1 | H20 的 296 是 dense 还是 sparse | **采信 dense**（§E.1 四条证据 + 合规算术） |
| E-C2 | H20 96GB 显存是 HBM3 还是 HBM3e | 多数派 **HBM3**；腾讯科技 2023-11 稿称 6×16GB HBM3e。**采信 HBM3（96GB 版）；141GB 版为 HBM3e** |
| E-C3 | H800 FP64 = 0.8 / 1 / 30 | **官方 PCIe 表 0.8；渠道 SXM 页 1** → 属 **SKU 差异**。sxrsgk.com 称「FP64 30 TFLOPS 与 H100 相同」**为错误，不采用** |
| E-C4 | H800 Tensor 值是否带 `*` | 渠道页 989*/1979*/3958* 与官方 PCIe 表 756*/1513*/3026*（均带 `*`=sparse）自洽。**sxrsgk.com 把 989/1979/3958 当「非稀疏峰值」发布 → 会使对比表整体偏高 2×，必须剔除** |
| E-C5 | H800 PCIe 80GB 显存类型 | 官方表未标 → **未找到公开数据** |
| E-C6 | cpudb.cc/TechPowerUp 给出 H800「FP16 237.2 / FP32 59.30 / FP64 29.65」 | 这是 **CUDA-core 向量算力**，不是 Tensor Core 规格，**不可与 989/1979 混用** |
| E-C7 | H20 96G vs 141G 带宽 | 均为 **4.0 TB/s**（无冲突：容量升、带宽不变） |
| E-C8 | 55 亿 vs 45 亿美元 | 4/15 为**预估上限 $5.5B**，5/28 为**实际 $4.5B**，须分列日期口径 |
| E-C9 | H20 PCIe SKU | **不存在**（NVIDIA docs 仅列 SXM5 96GB / 141GB） |

### E.7 H800 / H20 的「未找到公开数据」清单（无编造）
1. H20 官方 sparse 算力（TF32/BF16/FP16/INT8 全部未公布；FP8 sparse 592 仅第三方）
2. H800 PCIe 80GB 的显存类型
3. H800 PCIe 94GB（NVL）的带宽 / 显存类型 / TDP / 算力
4. H800 官方 SM 数（132 为第三方）
5. **H20 141GB 的首发 / 量产日期**（仅能确证 2025 Q1 已在渠道流通）
6. NVIDIA 未在 nvidia.com 公开 H800 / H20 的完整规格页；H800 SXM 数值依赖渠道商转述的 NVIDIA 官方规格文本

---

## F. A800 / L40S / L20 / MI300X —— 补充核查（子代理独立研究，2026-09-18）

### F.1 【重要】L40S —— 任务简报里的参考点整组错位

⚠️ **委托简报给出的 L40S 参考值有误，请勿使用**：

| 项目 | 简报参考值 | **NVIDIA 官方实际值（密集｜稀疏）** |
|---|---|---|
| FP16 Tensor | 183 密集 / 366 稀疏 ❌ | **362.05 密集 / 733 稀疏** |
| FP8 Tensor | 366 密集 / 733 稀疏 ❌ | **733 密集 / 1,466 稀疏** |
| INT8 Tensor | 1,466 稀疏 ✅ | **733 密集 / 1,466 稀疏** |
| FP32 | 91.6 ✅ | 91.6 |

简报的 183/366 其实是 **TF32** 的密集/稀疏值（被错位到了 FP16 行）。

**L40S 完整官方规格**（NVIDIA官方，产品页 + 数据表 + 中文数据表三方一致）：

| 项目 | 值 |
|---|---|
| 架构/制程 | Ada Lovelace / **AD102** / TSMC 4N |
| CUDA / RT / Tensor Core | **18,176** / 142 / 568 |
| FP32 | **91.6 TFLOPS** |
| TF32 Tensor | **183 密集 / 366 稀疏** |
| BF16 Tensor | **362.05 密集 / 733 稀疏** |
| FP16 Tensor | **362.05 密集 / 733 稀疏** |
| FP8 Tensor | **733 密集 / 1,466 稀疏** |
| INT8 Tensor | **733 密集 / 1,466 稀疏** TOPS |
| INT4 Tensor | **733 密集 / 1,466 稀疏** TOPS |
| RT Core | **212 TFLOPS**（产品页+官方博客）vs **209 TFLOPS**（数据表 PDF）→ **⚠️ 两份 NVIDIA 官方文件冲突，建议用 212 并标注** |
| 显存 | **48GB GDDR6 ECC** |
| 带宽 | **864 GB/s** |
| 互连 | **无 NVLink**；PCIe Gen4 x16（**64 GB/s 双向**） |
| TDP | **350W**，16 针 |
| 形态 | 4.4"×10.5" **双插槽被动**；4×DP1.4a；3×NVENC + 3×NVDEC（含 AV1） |
| MIG | **不支持**；vGPU **支持**；NEBS Level 3 |
| 发布 | **2023-08-08**（洛杉矶 SIGGRAPH），**2023 秋上市** |
| 最大扩展域 | **单机最多 8 卡**（OVX，全走 PCIe，无 NVLink 域） |

发布来源（NVIDIA 中国官方博客 2023-08-08）：<https://blogs.nvidia.cn/blog/nvidia-global-data-center-system-manufacturers-to-supercharge-generative-ai-and-industrial-digitalization/>
官方数据表 PDF（2841316，AUG23，PNY 托管镜像）：<https://www.pny.com/File%20Library/Company/Support/Product%20Brochures/NVIDIA%20Data%20Center%20GPUs/l40s-datasheet.pdf>
中文数据表：<http://images.nvidia.cn/cn/RTX/l40s-datasheet-web-a4-zhCN-2907810-2023.9.25.pdf>

### F.2 A800 —— NVIDIA 官方页面已下架，但官方文件仍可引用

- **A800 数据中心页已下架**：`nvidia.com/en-us/data-center/a800/`、`nvidia.cn/data-center/a800/` 等 2026-09-18 实测**均 404**。
- **但 NVIDIA 官方驱动发行说明仍确认 A800 存在**：《Data Center GPU Driver Release Notes》**580.82.07，RN-08625-565 v2.1，September 2026** <https://docs.nvidia.com/datacenter/tesla/pdf/NVIDIA_Data_Center_GPU_Driver_Release_Notes_580_v2.1.pdf>【NVIDIA官方】
  - 原文列 `NVIDIA Ampere GPU Architecture — NVIDIA A800, A100, A40, A30, A16, A10…`
  - 原文列 **`NVIDIA HGX A800 8-GPU — A800 and NVSwitch`** → **官方确认 A800 的 8 卡 NVSwitch 扩展域**
- **A800 = A100（仅 NVLink 600→400 GB/s）**：天翼云【OEM伙伴】<https://www.ctyun.cn/developer/article/437601152262213>：「A800 主要是将 NVLink 的传输速率由 A100 的 600GB/s 降至了 400GB/s，其他参数与 A100 基本一致」，并明确「在 **2022 年 11 月 8 日**宣布」。
- **A800 三款数据中心 SKU**（数值同 A100）：

| 项目 | A800 40GB PCIe | A800 80GB PCIe | A800 80GB SXM4 |
|---|---|---|---|
| 显存 | 40GB HBM2e | 80GB HBM2e | 80GB HBM2e |
| 带宽 | **1,555 GB/s** | **1,935 GB/s** | **2,039 GB/s** |
| TDP | **250W** | **300W** | **400W** |
| 算力（同 A100） | FP64 9.7 / FP32 19.5 / TF32 **156｜312** / BF16 **312｜624** / FP16 **312｜624** / INT8 **624｜1,248 TOPS** | 同左 | 同左 |
| NVLink | 桥接 2 卡，**400 GB/s** | 桥接 2 卡，400 GB/s | **400 GB/s**，HGX A800 8 卡 + NVSwitch |
| MIG | 7×@5GB | 7×@10GB | 7×@10GB |

- **A800 40GB SXM：未找到公开数据**（A800 只有上述三款数据中心 SKU）。
- **NVIDIA 官方在售的 A800 40GB Active 工作站页**：<https://www.nvidia.com/en-us/products/workstations/a800/>【NVIDIA官方】40GB HBM2 / 5,120-bit / **1,555.2 GB/s** / 6,912 CUDA / 432 Tensor / FP64 9.7 / FP32 19.5 / 峰值张量 **1,247 AI TOPS｜623.8 TFLOPS** / NVLink **400 GB/s** / PCIe 4.0 x16 / MIG 7×@5GB / **Max Power 240W** / 主动散热 / 无显示输出。
- **⚠️ 功耗 240W vs 250W 不是矛盾**：**240W** 是 NVIDIA 官方 **A800 40GB Active 工作站**型号；**250W** 是数据中心 **A800 40GB PCIe**。**须区分型号名。**
- 其他佐证：国家超算互联网 A800 80GB = HBM2e / 1,935 GB/s / 6,912 CUDA；H3C R5500 G5（Red Hat 目录）「HGX A800 8-GPU module, 8 A800 GPUs and 6 NVSWITCHs, 400GB/s」。
- **日期弱冲突**：主流为 **2022-11-08 公告**；topcpu（第三方）称 SXM4「released on Aug 2022」→ 建议标注为第三方说法。
- **未找到公开数据**：A800 40GB SXM 型号；A800 专属官方数据表 PDF（nvidia.com 原链 404）；A800 专属官方 7nm 表述（以 A100 GA100 类推）。

### F.3 L20 —— NVIDIA 无任何官方产品页或数据表

- **实测 404**：`nvidia.cn/data-center/l20/`、`nvidia.com/en-us/data-center/l20/`、`nvidia.cn/data-center/products/l20/`、`nvidia.cn/ai-data-science/products/l20/` **全部不存在**。
- **但 NVIDIA 官方驱动说明确认 L20 存在且属 Ada Lovelace**（同一份 580.82.07 发行说明）
  - 原文列 **`NVIDIA Ada Lovelace — NVIDIA L40, L4, L2, L20`**【NVIDIA官方】

| 项目 | 值 | 来源标签 |
|---|---|---|
| 架构 | NVIDIA Ada Lovelace | 阿里云官方文档【OEM伙伴】 |
| 显存 | **48 GB GDDR6 ECC** | 阿里云 + 第三方一致 |
| 带宽 | **864 GB/s** | 阿里云 + 第三方一致 |
| FP64 | **N/A** | 阿里云（另说 0.927，第三方） |
| FP32 | **59.3 TFLOPS**（阿里云）/ 59.8（第三方）/ 59.35（超算互联网） | **冲突** |
| FP16/BF16 | **119 TFLOPS**（阿里云）/ 119.5（第三方） | **冲突** |
| FP8/INT8 | **237 TFLOPS**（阿里云）/ 239（第三方） | **冲突** |
| TF32 | 59.8（第三方） | 第三方 |
| 互连 | **PCIe Gen4 x16**（阿里云「卡间互联」，未提 NVLink） | 推定无 NVLink，但无官方明确说明 |
| TDP | **275W**（第三方/经销商） | **无官方** |
| L2 缓存 | 96MB（第三方） | 第三方 |
| CUDA 核心 | 11,776（第三方/超算互联网） | 第三方 |
| 最大扩展域 | **单机最多 8 卡**（阿里云 gn8is 实例：`ecs.gn8is-8x.32xlarge` = L20×8 等） | 阿里云官方文档 |
| 推出日期 | **2023-11-16**（IT之家：「11 月 16 日推出三款中国特供版 AI 芯片」HGX H20 / L20 PCIe / L2 PCIe） | 第三方媒体 |
| 制程 | 4N（**推定，无官方**） | — |

**密集/稀疏口径**：**未找到官方标注**。按 59.8 → 119.5 → 239 的 1:2:4 链推定为 **dense** 值（稀疏约 119.5 / 239 / 478），属**推定非官方**。
⚠️ 另注意：超算互联网表的「半精度」经交叉验算实为 FP16 **向量**算力（该表各型号半精度 = 2× 单精度），**不可当作 Tensor 算力**。

来源：阿里云官方文档 <https://www.alibabacloud.com/help/zh/egs/gpu-accelerated-compute-optimized-instance-families>；国家超算互联网 <https://www1.scnet.cn/help/docs/mainsite/ai/introduction/>

**NVIDIA 官方未披露项**：官方产品页/数据表/规格表**全部不存在**（404 实测）；官方密集/稀疏标注、制程、CUDA/RT/Tensor 核心数、MIG、NVLink 支持、官方 TDP 均**未找到公开数据**。

### F.4 MI300X —— 简报参考值全部正确，且 AMD 口径最规范

**AMD 官方页**：<https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html>【AMD官方】
**发布**：**2023-12-06**（AMD 新闻稿 "announced the **availability** of the AMD Instinct MI300X accelerators"）<https://www.amd.com/ja/newsroom/press-releases/2023-12-06-amd-ai-amd-instinct-mi300.html>

| 项目 | 值 |
|---|---|
| 架构/制程 | **CDNA 3** / **TSMC 5nm ｜ 6nm FinFET** |
| CU / stream processors / Matrix Cores | 304 / 19,456 / 1,216 |
| 频率 | 2,100 MHz |
| FP8 | **2.61 PFLOPS 密集 / 5.22 PFLOPS 稀疏**（structured sparsity，E5M2/E4M3） |
| FP16 | **1.3 PFLOPS / 2.61 PFLOPS**（稀疏） |
| BF16 | **1.3 PFLOPS / 2.61 PFLOPS**（稀疏） |
| TF32 | **653.7 TFLOPS / 1.3 PFLOPS**（稀疏） |
| FP32（matrix / 向量） | **163.4 TFLOPS** / 163.4 TFLOPS |
| FP64（matrix / 向量） | **163.4 TFLOPS** / 81.7 TFLOPS |
| INT8 | **2.6 POPS / 5.22 POPS**（稀疏） |
| 显存 | **192GB HBM3**，8,192-bit，5.2 GHz |
| 带宽 | **5.3 TB/s**（官方脚注精算 **5.325 TB/s** = 8,192 bit × 5.2 Gbps ÷ 8） |
| LLC | 256 MB；ECC 全芯片 |
| 晶体管 | 1,530 亿 |
| TDP | **750W Peak TBP**（54V UBB） |
| 形态 | **OAM Module 被动**；PCIe 5.0 x16 |
| 互连 | **Infinity Fabric 8 links，峰值 128 GB/s**（**非 NVLink**） |
| 最大 scale-up 域 | **8 卡**（AMD Instinct Platform = 8×MI300X，合计 **1.5 TB HBM3**，OCP 设计） |

**AMD 官方脚注精确措辞（可逐字引用，是本报告中最清晰的 dense/sparse 表述范例）**：
> "…**1307.4 TFLOPS** peak theoretical half precision (FP16)…**2614.9 TFLOPS** peak theoretical 8-bit precision (FP8), **2614.9 TOPs** INT8… providing an estimated 2x improvement in math efficiency resulting…**5,229.8 TFLOPS** FP8, **5,229.8 TOPs** INT8 … **with sparsity**"（MI300-17）

→ 简报参考值 **全部正确**（1307 / 2615 / 5230 对应官方 1307.4 / 2614.9 / 5229.8；产品页四舍五入为 1.3 / 2.61 / 5.22 PFLOPs）。

### F.5 本节新增来源

| 主题 | 来源 | 日期 |
|---|---|---|
| A800 存在性与 HGX A800 8-GPU（NVIDIA官方） | NVIDIA Driver Release Notes 580.82.07 RN-08625-565 v2.1: https://docs.nvidia.com/datacenter/tesla/pdf/NVIDIA_Data_Center_GPU_Driver_Release_Notes_580_v2.1.pdf | September 2026 |
| A800 三 SKU 规格 | 天翼云（OEM伙伴）: https://www.ctyun.cn/developer/article/437601152262213 | 现行 |
| A800 80GB 规格 | 国家超算互联网: https://www1.scnet.cn/help/docs/mainsite/ai/introduction/ | 现行 |
| HGX A800 8-GPU + 6 NVSwitch | Red Hat 硬件目录（H3C R5500 G5）: https://catalog.redhat.com/en/hardware/system/detail/70035 | 现行 |
| L40S 官方数据表（PNY 镜像） | https://www.pny.com/File%20Library/Company/Support/Product%20Brochures/NVIDIA%20Data%20Center%20GPUs/l40s-datasheet.pdf | 2841316, AUG23 |
| L40S 发布（NVIDIA 中国官方博客） | https://blogs.nvidia.cn/blog/nvidia-global-data-center-system-manufacturers-to-supercharge-generative-ai-and-industrial-digitalization/ | 2023-08-08 |
| L20 规格 | 阿里云官方文档（OEM伙伴）: https://www.alibabacloud.com/help/zh/egs/gpu-accelerated-compute-optimized-instance-families | 现行 |
| L20 推出日期 | IT之家 | 2023-11-10 |
| MI300X 官方规格 | AMD: https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html | 现行 |
| MI300X 上市 | AMD 新闻稿: https://www.amd.com/ja/newsroom/press-releases/2023-12-06-amd-ai-amd-instinct-mi300.html | 2023-12-06 |
