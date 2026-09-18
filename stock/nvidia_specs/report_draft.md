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
| H20 | **148** | **296** | 第三方（百度百科参数表，标注 `*`=稀疏） |
| H200 SXM | 989 | 1,979 | NVIDIA官方 产品页 + datasheet |
| H200 NVL（PCIe） | 835.5 | 1,671 | NVIDIA官方 datasheet |
| B200（HGX B200 单卡） | **2,250** | **4,500** | NVIDIA官方 Blackwell Tech Brief（`2.2/4.5 petaFLOPS`） |
| GB200 内 GPU（NVL72 单卡摊算） | 2,500 | 5,000 | NVIDIA官方 GB200 产品页整柜值 ÷72 |
| B300 / Blackwell Ultra（HGX B300 单卡） | **4,500** | **9,000** | NVIDIA官方 Blackwell Ultra datasheet |
| GB300 内 GPU（NVL72 单卡摊算） | 5,000 | 10,000 | NVIDIA官方 GB300 产品页整柜值 ÷72 |
| B30A（中国特供，单 die） | ~1,875（推算） | ~2,250（推算） | **第三方推测**（TechPowerUp 以 B300 半算力推算） |
| RTX 6000D（中国特供工作站） | 未找到公开数据 | 未找到公开数据 | — |
| A800 40GB Active | 312 | 624 | NVIDIA官方 产品页（`623.8 TFLOPS` 峰值 Tensor = 稀疏） |
| L40S | **362.5** | **733** | NVIDIA官方 L40S 中文 datasheet（`362.5 \| 733*`） |
| L20 | 未找到公开数据 | 未找到公开数据 | — |
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
| H20 | **148** | **296** | 第三方（百度百科，`INT8 \| FP8 Tensor Core* 296 \| 296`） |
| H200 SXM | 1,979 | 3,958 | NVIDIA官方 |
| H200 NVL | 1,670.5 | 3,341 | NVIDIA官方 datasheet |
| B200 单卡 | **2,250** | **4,500** | NVIDIA官方 Blackwell Tech Brief（`4.5/9 petaFLOPS`，FP8/FP6 合并） |
| GB200 内 GPU | 2,500 | 5,000 | NVIDIA官方 GB200 产品页 ÷72 |
| B300 单卡 | **4,500** | **9,000** | NVIDIA官方 Blackwell Ultra datasheet |
| GB300 内 GPU | 5,000 | 10,000 | NVIDIA官方 GB300 产品页 ÷72 |
| B30A | ~1,875（推算） | ~1,875（TPU 原推算值，未区分口径） | **第三方推测，存疑** |
| RTX 6000D | 未找到公开数据 | 未找到公开数据 | — |
| L40S | **733** | **1,466** | NVIDIA官方 L40S 中文 datasheet |
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

> 🔴 **B300/B200 FP4 口径冲突警告**：HGX 页把 FP4 写成 `144 \| 108 PFLOPS`（B300）与 `144 \| 72 PFLOPS`（B200），脚注 1 是 `Specification in Sparse | Dense`。按此，B300 的 **dense FP4 = 108/8 = 13.5 PFLOPS**，稀疏 = 18 PFLOPS，比值仅 1.33，**不是 2×**。但 Blackwell Ultra datasheet 的「Individual GPU Specifications」表把 B300 单卡写成 `20 PFLOPS | 15 PFLOPS`（GB300 列）与 `18 PFLOPS | 14 PFLOPS`（HGX 列），整柜汇总又用 `1,440 \| 1,080`（72×20 / 72×15）。**同一份官方 datasheet 内部并不自洽**，详见 §4.1。

### 1.4 INT8 Tensor Core（`密集 | 稀疏`）

| 产品 | INT8 密集（TOPS） | INT8 稀疏（TOPS） |
|---|---|---|
| A100 全系 | **624** | **1,248** |
| H100 SXM | 1,979 | 3,958 |
| H100 PCIe | 1,600 | 3,200 |
| H100 NVL | 1,670.5 | 3,341 |
| H800 SXM | 1,979 | 3,958 |
| H20 | 148 | 296 |
| H200 SXM | 1,979 | 3,958 |
| H200 NVL | 1,670.5 | 3,341 |
| B200 单卡 | 2,250 | 4,500（`4.5/9 petaOPS`） |
| B300 单卡（HGX） | 153.75 | 307.5（Ultra datasheet `307 TOPS` sparse / 8卡） |
| GB300 NVL72 整柜 | 12,000 POPS | 24,000 POPS（= 12 \| 24 petaOPS） |
| GB200 NVL72 整柜 | 360,000 POPS | 720,000 POPS（= 360 \| 720 petaOPS） |
| B30A | 未找到公开数据 | — |
| L40S | **733** | **1,466** |
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
| H20 | 74 | 未单列 | **44** | 1 | — |
| H200 SXM | 494.5 | 989 | **67** | 34 | 67 |
| H200 NVL | 417.5 | 835 | 60 | 30 | 60 |
| B200 单卡（HGX） | 1,100 | 2,200（`1.1/2.2 petaFLOPS`） | **75 TFLOPS** | 37 | — |
| B300 单卡（HGX） | 1,100 | 2,200 | **75 TFLOPS** | 1.2（Ultra datasheet，**存疑**） | — |
| GB300 整柜 | 90,000 | 180,000 | 5,760 / 6,000 | 100 | — |
| GB200 整柜 | 90,000 | 180,000 | 5,760 | 2,880 | — |
| B30A | 未找到公开数据（TPU 推算 ~940） | — | — | — | — |
| L40S | **183** | **366** | **91.6** | — | — |
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
| **H20** | Hopper / 4N | **96GB HBM3** | **4.0 TB/s** | **NVLink 4 / 900 GB/s** | 是 | Gen5 x16 128 GB/s | **400W** | 8 路 HGX（SXM） | 8 |
| **H200 SXM** | Hopper / 4N | **141GB HBM3e** | **4.8 TB/s** | NVLink 4 / 900 GB/s | 是 | Gen5 128 GB/s | 最高 700W 可配置 | SXM | 8 |
| **H200 NVL** | Hopper / 4N | 141GB HBM3e | **4.2 TB/s**（datasheet）/ 4.8（产品页，**冲突**） | 2/4 路 NVLink Bridge，900 GB/s per GPU | 否 | Gen5 128 GB/s | 最高 600W 可配置 | PCIe 双插槽风冷 | 2 或 4 |
| **B200（HGX B200 单卡）** | Blackwell / TSMC 4NP，双 die | **180GB** HBM3e（OEM 配置；NVIDIA 原文 `up to 192 GB`，**冲突**） | **7.7 TB/s** | NVLink 5 / **1.8 TB/s** | 是（NVLink 5 Switch） | **Gen6**（Ultra datasheet 列 B200 为 Gen5，**冲突**） | **1,000W** | SXM6 | **8**（HGX） |
| **GB200 内 GPU** | Blackwell / 4NP | **192GB** HBM3e（186–192 各源不一） | **8 TB/s** | NVLink 5 / 1.8 TB/s | 是 | Gen5/Gen6 | ~1,200W（托盘级） | Grace Blackwell Superchip（1 Grace + 2 GPU） | **72（NVL72）** |
| **B300 / Blackwell Ultra（HGX B300 单卡）** | Blackwell Ultra / 4NP | **270GB** HBM3e | **7.7 TB/s**（datasheet 7.7 / 产品页 8） | NVLink 5 / 1.8 TB/s | 是 | **Gen6 256 GB/s** | 最高 **1,100W** 可配置 | SXM | 8 |
| **GB300 内 GPU** | Blackwell Ultra / 4NP | **279GB** HBM3e | **8 TB/s** | NVLink 5 / 1.8 TB/s | 是 | Gen6 256 GB/s | 最高 **1,400W** 可配置 | GB300 Superchip | **72（NVL72）** |
| **B30A（中国特供）** | Blackwell Ultra 单 die | 未找到公开数据（传闻 96GB HBM3e） | 未找到公开数据 | 传闻约 B300 之半（≈900 GB/s），**未证实** | 未找到公开数据 | 未找到公开数据 | 传闻约 800W | 传闻 NVL8 基板 | 传闻 8（NVL8） |
| **RTX 6000D（中国特供）** | Blackwell（GB202 衍生） | 传闻 **84GB GDDR7**（拆解实测） | 传闻约 1,100 GB/s | **无 NVLink** | 否 | Gen5 x16 | 传闻 600W | PCIe 双插槽 | 1（无 NVLink） |
| **A800 40GB Active** | Ampere / 7nm | 40GB HBM2 | 1,555.2 GB/s | **NVLink 3 / 400 GB/s（人为削减，A100 为 600）** | 否 | Gen4 x16 | 240W | PCIe 双插槽 | 2（桥接） |
| **L40S** | Ada Lovelace / TSMC 4N | 48GB GDDR6 ECC | **864 GB/s** | **无 NVLink** | 否 | Gen4 x16 64 GB/s | 350W | PCIe 双插槽被动 | 1 |
| **MI300X（AMD，对照）** | CDNA 3 / 5nm + 6nm | **192GB HBM3** | **5.3 TB/s** | Infinity Fabric 128 GB/s（非 NVLink） | — | Gen5 x16 | 750W（TBP 峰值） | OAM | 8（UBB） |

---

## 3. 关键结论与对比要点（面向寒武纪 MLU 对比）

1. **口径陷阱**：把 H100 SXM 的「1,979 TFLOPS BF16」当作稠密算力是行业最常见的错误 —— 稠密只有 **989**。同一芯片名在 SXM / PCIe / NVL 三种板型下算力与带宽都会变（H100 PCIe 只有 800 sparse / 400 dense BF16，带宽 2.0 TB/s vs SXM 3.35 TB/s）。
2. **出口 SKU 优先砍互连**：A800（600→400 GB/s）、H800（900→400 GB/s）算力/显存对齐正版，**只砍 NVLink**。H20 相反 —— 砍算力（BF16 稀疏仅 296 vs H100 的 1,979）却**保留 900 GB/s NVLink 与 4.0 TB/s 带宽**，定位偏向推理与通信密集型负载。B30A 传闻同样是「算力减半 + 互连减半」。
3. **H200 是「同算力、换显存/带宽」**：计算单元不变（BF16 989/1,979），HBM3e 把容量从 80→141GB、带宽从 3.35→4.8 TB/s。对长上下文 decode 收益显著，对大 GEMM 训练几乎无感。
4. **Blackwell 的 FP4 是分水岭**：B200 起出现 FP4，且 sparse/dense 比值不再是 2×（见 §4.1）。若与寒武纪 MLU 的 INT8/INT16 口径对比，务必先统一到 dense。
5. **scale-up 域从 8 跃到 72**：GB200/GB300 NVL72 把单个 NVLink 域从 8 卡扩到 72 卡（整柜 130 TB/s），这是与 MLU 集群架构对比时最大的结构性差异。
6. **对中国特供 Blackwell（B30A / RTX 6000D）目前没有 NVIDIA 官方规格**。截至 2026-09-18，NVIDIA **从未发布** B30A 或 RTX 6000D 的官方 datasheet/产品页；所有数字均来自媒体/泄露，且 B30A 已被美国政府阻止出口（The Information，2025-11-07）。
