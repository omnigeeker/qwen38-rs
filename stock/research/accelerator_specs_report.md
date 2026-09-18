# 数据中心 AI 加速卡规格对比报告（A800 / L40S / L20 / MI300X）

**编制日期：2026-09-18** ｜ 所有数字均附来源 URL、机构名称、日期与信源等级标签。
标签定义：**【NVIDIA官方】/【AMD官方】** = 厂商一手文档；**【OEM伙伴】** = 服务器/云厂商一手技术文档；**【第三方】** = 经销商、GPU 数据库、券商、媒体。

> ⚠️ 本报告更正了任务中若干"已知参考点"的错误，见第 6 节。**最重要的更正：L40S 的 FP16 张量算力不是"366 稀疏 / 183 密集"。**

---

## 1. 速览对比表

| 项目 | NVIDIA A800 40GB PCIe | NVIDIA A800 80GB PCIe | NVIDIA A800 80GB SXM4 | NVIDIA L40S | NVIDIA L20 | AMD MI300X |
|---|---|---|---|---|---|---|
| 架构 | Ampere (GA100) | Ampere (GA100) | Ampere (GA100) | Ada Lovelace (AD102) | Ada Lovelace | CDNA 3 |
| 制程 | TSMC 7nm | TSMC 7nm | TSMC 7nm | TSMC 4N | TSMC 4N（推定，无官方） | TSMC 5nm + 6nm FinFET |
| 显存类型/容量 | 40GB HBM2e | 80GB HBM2e | 80GB HBM2e | 48GB GDDR6 ECC | 48GB GDDR6 ECC | 192GB HBM3 |
| 显存带宽 | 1,555 GB/s | 1,935 GB/s | 2,039 GB/s | 864 GB/s | 864 GB/s | 5.3 TB/s（精算 5.325） |
| FP32 | 19.5 TFLOPS | 19.5 TFLOPS | 19.5 TFLOPS | 91.6 TFLOPS | 59.3–59.8 TFLOPS（冲突） | 163.4 TFLOPS |
| FP16/BF16 张量（密集｜稀疏） | 312｜624 TFLOPS | 312｜624 TFLOPS | 312｜624 TFLOPS | **362.05｜733 TFLOPS** | 119.5｜239 TFLOPS【第三方】 | 1,307.4｜2,614.9 TFLOPS |
| FP8 张量（密集｜稀疏） | 不支持 | 不支持 | 不支持 | **733｜1,466 TFLOPS** | 239｜478 TFLOPS【第三方】 | 2,614.9｜5,229.8 TFLOPS |
| INT8（密集｜稀疏） | 624｜1,248 TOPS | 624｜1,248 TOPS | 624｜1,248 TOPS | **733｜1,466 TOPS** | 239｜478 TOPS【第三方】 | 2,614.9｜5,229.8 TOPs |
| TDP / 最大功耗 | 250W（天翼云）／**240W（NVIDIA 官方 Active 型号）** | 300W | 400W | 350W | 275W【第三方】 | 750W Peak |
| 形态 | PCIe 双插槽风冷／单插槽液冷 | 同左 | SXM4 | PCIe 双插槽被动（4.4"×10.5"） | PCIe 双插槽被动 | OAM Module 被动 |
| 互连 | PCIe 4.0 x16 (64GB/s) + NVLink 桥接 2 卡 400GB/s | 同左 | NVLink 400GB/s + PCIe 4.0 | **无 NVLink**；PCIe Gen4 x16 64GB/s 双向 | 无 NVLink；PCIe Gen4 x16 | 8× Infinity Fabric，峰值 128 GB/s；PCIe 5.0 x16 |
| 最大纵向扩展域 | 2 卡 | 2 卡 | **8 卡（HGX A800 8-GPU + NVSwitch）** | 单机最多 8 卡（NVIDIA OVX，走 PCIe，无 NVLink） | 单机最多 8 卡（阿里云 gn8is，走 PCIe） | 8 卡（AMD Instinct Platform，1.5TB HBM3） |
| MIG | 7×@5GB | 7×@10GB | 7×@10GB | 不支持 | 未见公开数据 | 不适用（SR-IOV 支持） |
| 发布/可用 | 2022-11-08 宣布 | 2022-11-08 宣布 | 2022-11-08 宣布 | 2023-08-08 宣布，2023 秋上市 | 2023-11-16 推出 | 2023-12-06 上市 |

---

## 2. A) NVIDIA A800

### 2.1 官方数据可得性（关键发现）
**NVIDIA 已下架 A800 数据中心产品页**：`nvidia.com/en-us/data-center/a800/`、`nvidia.cn/data-center/a800/`、`nvidia.com/en-in/data-center/a800/`、`nvidia.com/en-eu/data-center/a800/` 均返回 **404**（本次实测，2026-09-18）。
现存唯一 NVIDIA 一手 A800 页面是**工作站型号 "A800 40GB Active"**：<https://www.nvidia.com/en-us/products/workstations/a800/> 【NVIDIA官方，无页面日期，2026-09-18 读取】。

**NVIDIA 官方文件仍确认 A800 的存在、架构与平台形态**：
《NVIDIA Data Center GPU Driver version 580.82.07 (Linux) Release Notes》RN-08625-565 v2.1，**September 2026**（PDF 生成时间 2026-09-17），<https://docs.nvidia.com/datacenter/tesla/pdf/NVIDIA_Data_Center_GPU_Driver_Release_Notes_580_v2.1.pdf> 【NVIDIA官方】原文措辞：
- "NVIDIA **Ampere GPU Architecture** — NVIDIA **A800**, A100, A40, A30, A16, A10, A10G, A2, AX800"
- "NVIDIA **HGX A800 8-GPU** — **A800 and NVSwitch**"（→ **官方确认 A800 SXM 的 8 卡 NVSwitch 纵向扩展域**）
- "NVIDIA **RTX A800 40GB Active** — NVIDIA Ampere architecture"
- "Data Center A-Series Products: NVIDIA **A800, AX800** — NVIDIA Ampere architecture"

### 2.2 A800 = A100（NVLink 由 600GB/s 降至 400GB/s）
中国电信天翼云开发者文章（【OEM伙伴】云厂商技术文档，无页面日期，文中引用 2022-11-08 公告）："A800 主要是将 NVLink 的传输速率由 A100 的 600GB/s 降至了 400GB/s，其他参数与 A100 基本一致" — <https://www.ctyun.cn/developer/article/437601152262213>

**A100 官方基准**（用于印证"A800 其余参数同 A100"）：《NVIDIA A100 Tensor Core GPU Data Sheet》JUN21，<https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf> 【NVIDIA官方】官方规格表（SXM4 与 PCIe）：

| | A100 40GB PCIe | A100 80GB PCIe | A100 40GB SXM | A100 80GB SXM |
|---|---|---|---|---|
| 显存 | 40GB HBM2 | 80GB HBM2e | 40GB HBM2 | 80GB HBM2e |
| 带宽 | 1,555 GB/s | 1,935 GB/s | 1,555 GB/s | 2,039 GB/s |
| TDP | 250W | 300W | 400W | 400W |
| 互连 | NVLink 桥接 2 卡 600GB/s；PCIe Gen4 64GB/s | 同左 | NVLink 600GB/s；PCIe Gen4 64GB/s | 同左 |

同表算力（全形态同值）：FP64 9.7 / FP64 Tensor 19.5 / FP32 19.5 TFLOPS；TF32 156｜312*；BF16 312｜624*；FP16 312｜624*；INT8 624 TOPS｜1,248 TOPS*（*为稀疏）。

### 2.3 A800 三款 SKU 规格（天翼云对比表，【OEM伙伴】）
<https://www.ctyun.cn/developer/article/437601152262213>，表头为：A100 40GB PCIe / 80GB PCIe / 40GB SXM / 80GB SXM / **A800 40GB PCIe / A800 80GB PCIe / A800 80GB SXM**

| 参数 | A800 40GB PCIe | A800 80GB PCIe | A800 80GB SXM |
|---|---|---|---|
| GPU 显存 | 40GB HBM2e | 80GB HBM2e | 80GB HBM2e |
| 显存带宽 | 1,555 GB/s | 1,935 GB/s | **2,039 GB/s** |
| Max TDP | **250W** | 300W | **400W** |
| TF32 Tensor | 156｜312 TFLOPS | 同左 | 同左 |
| BF16 Tensor | 312｜624 TFLOPS | 同左 | 同左 |
| FP16 Tensor | 312｜624 TFLOPS | 同左 | 同左 |
| INT8 Tensor | 624 TOPS | 624 TOPS | 1,248 TOPS |
| MIG | 7×@5GB | 7×@10GB | 7×@10GB |
| 形态 | PCIe 双插槽风冷／单插槽液冷 | 同左 | SXM |
| 互连 | NVLink 桥接（2 卡）400GB/s；PCIe 4.0 64GB/s | 同左 | NVLink 400GB/s；PCIe 4.0 64GB/s |

> **注意**：该表的 INT8 行对 PCIe 给 624、对 SXM 给 1,248，而 NVIDIA A100 官方数据表把 INT8 写作单行 "624 TOPS | 1248 TOPS*"（*稀疏）。据此 A800 全形态应为 **624 密集 / 1,248 稀疏**；天翼云表的 "1,248" 应理解为稀疏值。**此为来源冲突，见第 6 节。**

### 2.4 A800 单SKU佐证
- **A800 40GB PCIe**：hypertek（【第三方】经销商）称 "launched on November 8th, 2022. Built on the 7 nm process, and based on the **GA100** graphics processor"，40GB HBM2e、**1,555 GB/s**、**250W**、双插槽、267mm、8-pin EPS、5,120-bit、7×MIG、NVLink 最高 400 GB/s — <https://www.hypertek.nl/product/nvidia-a800-pcie-40-gb/>
- **A800 80GB SXM4**：cputronic（【第三方】GPU 数据库）列 Ampere、**7nm**、HBM2e、**2,039 GB/s**、**400W**、6,912 CUDA 核心、40MB L2、542 亿晶体管 — <https://cputronic.com/en/gpu/nvidia-a800-sxm4-80-gb>；topcpu 同源数据称 "released on Aug 2022"（与 11-08 公告日冲突，见第 6 节） — <https://topcpu.net/en/cpu/a800-sxm4-80-gb>
- **A800 80GB（带宽）**：国家超算互联网平台表格列 "A800 | 80GB | 半精度 311.84 TFLOPS | 单精度 155.92 | 双精度 9.746 | CUDA 6,912 | HBM2e | **1935GB/s**"（对应 80GB PCIe） — <https://www1.scnet.cn/help/docs/mainsite/ai/introduction/> 【第三方/平台】
- **8 卡扩展域**：H3C UniServer R5500 G5（Red Hat 硬件兼容目录）"supporting **HGX A800 8-GPU module**, with **8 A800 GPUs and 6 NVSWITCHs** to achieve **400GB/s**" — <https://catalog.redhat.com/en/hardware/system/detail/70035> 【OEM伙伴】

### 2.5 A800 40GB Active（NVIDIA 官方工作站型号，独立 SKU）
<https://www.nvidia.com/en-us/products/workstations/a800/> 【NVIDIA官方】官方规格表原文：
- GPU 显存 **40GB HBM2**；显存接口 **5,120-bit**；显存带宽 **1,555.2 GB/s**
- CUDA 核心 **6,912**；Tensor 核心 **432**；双精度 9.7 TFLOPS；单精度 19.5 TFLOPS
- 峰值张量性能 **1,247 AI TOPS | 623.8 TFLOPS**
- **NVIDIA NVLink：Yes；NVLink 带宽：400GB/s**；总线 PCIe 4.0 x16
- MIG：Up to 7 MIG instances @5GB
- **Max Power Consumption：240W**；Thermal：Active；Form Factor：4.4" H × 10.5" L, dual slot；无显示输出

### 2.6 公告/上市日期
- **2022-11-08** 路透社独家："Exclusive: Nvidia offers new advanced chip for China that meets U.S. export controls" — <https://www.reuters.com/technology/exclusive-nvidia-offers-new-advanced-chip-china-that-meets-us-export-controls-2022-11-08/> 【第三方/媒体，URL 日期 2022-11-08】。**注：该页面正文返回 HTTP 401（付费墙/反爬），正文未能读取**；日期由 URL 与天翼云转述佐证（"在 2022 年 11 月 8 日宣布将推出…A800"）。彭博社同日报道：<https://www.bloomberg.com/news/articles/2022-11-08/nvidia-to-sell-new-chip-in-china-it-says-meets-us-export-ban> 【第三方/媒体】
- 超能网 expreview 头条《英伟达推出A800系列计算卡：NVLink带宽限制在400GB/s，专供中国市场》— <https://www.expreview.com/85490.html> 【第三方/媒体】。**注：正文抓取仅返回 39 字节，未能读取正文，仅标题可证。**
- **A800 40GB SXM：未找到公开数据**（天翼云表中 A800 仅有 40GB PCIe / 80GB PCIe / 80GB SXM 三款）。

---

## 3. B) NVIDIA L40S（PCIe，Ada Lovelace）

### 3.1 一手来源
1. **NVIDIA 官方产品页规格表**：<https://www.nvidia.com/en-us/data-center/l40s/> 【NVIDIA官方，无页面日期，2026-09-18 读取】
2. **NVIDIA 官方数据表 PDF**（文档编号 2841316，**AUG23**，PNY 托管）：<https://www.pny.com/File%20Library/Company/Support/Product%20Brochures/NVIDIA%20Data%20Center%20GPUs/l40s-datasheet.pdf> 【NVIDIA官方文档】
3. **NVIDIA 官方中文数据表**（文档编号 2907810，标注"2023 年 8 月"，PDF 生成 2023-09-25）：<http://images.nvidia.cn/cn/RTX/l40s-datasheet-web-a4-zhCN-2907810-2023.9.25.pdf> 【NVIDIA官方】
4. **NVIDIA 中国官方博客发布稿**（**2023 年 8 月 8 日**，洛杉矶 SIGGRAPH）：<https://blogs.nvidia.cn/blog/nvidia-global-data-center-system-manufacturers-to-supercharge-generative-ai-and-industrial-digitalization/> 【NVIDIA官方】
5. **NVIDIA Ada GPU 架构白皮书**（AD102 = 763 亿晶体管、96MB L2、TSMC 4N）：<http://images.nvidia.cn/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf> 【NVIDIA官方】

### 3.2 官方规格（中英数据表与产品页一致）
| 规格标签 | 数值 |
|---|---|
| GPU 架构 | NVIDIA Ada Lovelace 架构（AD102） |
| 制程 | TSMC 4N NVIDIA Custom Process【NVIDIA官方，Ada 白皮书】 |
| GPU 显存 | 48GB GDDR6（支持 ECC） |
| 显存带宽 | **864 GB/s** |
| 连接接口 | PCIe 4.0 Gen4 x16：**64 GB/s 双向** |
| CUDA 核心 | **18,176** |
| 第三代 RT Core | **142** |
| 第四代 Tensor Core | **568** |
| RT Core 性能 | **212 TFLOPS**（产品页/博客）｜**209 TFLOPS**（数据表 PDF）→ 冲突 |
| FP32 | **91.6 TFLOPS** |
| TF32 Tensor Core | **183 ｜ 366*** |
| BFLOAT16 Tensor Core | **362.05 ｜ 733*** |
| FP16 Tensor Core | **362.05 ｜ 733*** |
| FP8 Tensor Core | **733 ｜ 1,466*** |
| INT8 Tensor TOPS（峰值） | **733 ｜ 1,466*** |
| INT4 Tensor TOPS（峰值） | **733 ｜ 1,466*** |
| 外形规格 | 4.4" (H) × 10.5" (L)（中文表：11.2cm × 26.7cm），**双插槽** |
| 最大功耗 | **350W** |
| 电源接口 | 16 针 |
| 散热 | 被动（Passive） |
| 显示端口 | 4× DisplayPort 1.4a |
| NVENC / NVDEC | 3× ｜ 3×（含 AV1 编解码） |
| **NVIDIA NVLink 支持** | **否（No）** |
| MIG 支持 | 否（No） |
| vGPU 软件支持 | 是 |
| NEBS 就绪 | Level 3 |
| 安全启动（信任根） | 是 |

\* 数据表原文脚注：**"\* With sparsity"（采用稀疏技术）**——即左侧为**密集**值、右侧为**稀疏**值。

### 3.3 发布与可用性
- **2023-08-08** NVIDIA 中国官方博客（SIGGRAPH）："NVIDIA 于今日宣布推出搭载全新 NVIDIA® L40S GPU 的 [OVX 系统]"；"NVIDIA OVX 系统的**每台服务器最多支持 8 块 NVIDIA L40S GPU**，每块 GPU 的显存为 48GB"；"可提供**超过 1.45 PFLOP 的张量处理能力**"；"搭载 142 颗第三代 RT Core，可提供 **212 TFLOP 的光线追踪性能**"；"搭载 **18,176 颗 CUDA Core**"；"**NVIDIA L40S 将于今年秋季上市**"。
- 中文数据表页脚："2907810. 2023 年 8 月"；英文数据表页脚："2841316. AUG23"。

---

## 4. C) NVIDIA L20（中国特供，Ada）

### 4.1 结论：NVIDIA 未公开发布 L20 产品页或数据表 → **官方规格 未找到公开数据**
本次实测（2026-09-18）以下 URL 全部返回 **404**：
`nvidia.cn/data-center/l20/`、`nvidia.com/en-us/data-center/l20/`、`nvidia.cn/data-center/products/l20/`、`nvidia.cn/ai-data-science/products/l20/`。

**但 NVIDIA 官方文件确认 L20 存在且属 Ada Lovelace 家族**：《NVIDIA Data Center GPU Driver Release Notes》580 v2.1，RN-08625-565，**September 2026**，原文："NVIDIA **Ada Lovelace** — NVIDIA L40, L4, L2, **L20**" — <https://docs.nvidia.com/datacenter/tesla/pdf/NVIDIA_Data_Center_GPU_Driver_Release_Notes_580_v2.1.pdf> 【NVIDIA官方】

### 4.2 合作方/云厂商官方文档（可引用的最优来源）
**阿里云官方文档**（【OEM伙伴】云厂商一手文档，gn8is / ebmgn8ia / ebmgn8is 实例族，无页面日期，2026-09-18 读取）— <https://www.alibabacloud.com/help/zh/egs/gpu-accelerated-compute-optimized-instance-families>，原文"NVIDIA L20主要参数"：

| 规格标签 | 数值 |
|---|---|
| GPU 架构 | NVIDIA Ada Lovelace |
| 显存容量 | **48 GB** |
| 显存带宽 | **864 GB/s** |
| FP64 | **N/A** |
| FP32 | **59.3 TFLOPS** |
| FP16/BF16 | **119 TFLOPS** |
| FP8/INT8 | **237 TFLOPS** |
| 视频编解码 | 3× Video Encoder（+AV1）、3× Video Decoder、4× JPEG Decoder |
| 卡间互联 | **PCIe 接口：PCIe Gen4 x16**（未提 NVLink） |

同页实例表确认**单机最多 8 卡**：`ecs.gn8is-8x.32xlarge` = L20×8 / 48GB×8；`ecs.ebmgn8is.32xlarge` = L20×8 / 48GB×8；`ecs.ebmgn8ia.64xlarge` = L20×4 / 48GB×4。

### 4.3 其他来源
| 来源 | 类型 | 关键数字 |
|---|---|---|
| 国家超算互联网平台 <https://www1.scnet.cn/help/docs/mainsite/ai/introduction/> | 【第三方/平台】 | L20：**48GB**；半精度 **119.5 TFLOPS**；单精度 **59.35 TFLOPS**；双精度 **0.927 TFLOPS**；**CUDA 核心 11,776**；GDDR6；**864 GB/s** |
| hypertek <https://www.hypertek.nl/product/nvidia-l20-48gb-gpu/> | 【第三方】经销商 | 48GB GDDR6 ECC；**275 W**；PCIe Gen4 x16（64 GB/s）；TF32 Tensor **59.8**；BF16/FP16 Tensor **119.5**；INT8/FP8 Tensor **239** TFLOPS |
| MillionMiner <https://millionminer.com/product/ai-hardware/nvidia-l20-enterprise-48gb> | 【第三方】经销商 | 48GB GDDR6 ECC；**864 GB/s**；FP32 **59.8 TFLOPS**；**275W**；**96MB L2**；3× NVENC（含 AV1）/ 3× NVDEC / 4× NVJPEG；被动双插槽 |
| 物联网世界/爱集微 2024-01-04 <https://www.iotworld.com.cn/html/News/202401/52da19aad1c61b42.shtml> | 【第三方/媒体】 | "L20、L2 均为 **PCIe 4.0 x16 板卡形态**，采用英伟达 **Ada Lovelace 架构**。这两款产品分别搭载 **48GB、24GB GDDR6 显存**" |
| IT之家 2023-11-10 <https://m.ithome.com/html/731545.htm> | 【第三方/媒体】 | "英伟达内部人士确认：**11 月 16 日**推出三款中国特供版 AI 芯片"（HGX H20、L20 PCIe、L2 PCIe） |

### 4.4 L20 密集/稀疏标注
**未找到官方公开的密集/稀疏标注。** 依据 hypertek 的 TF32 59.8 → FP16 119.5 → FP8/INT8 239 呈 1:2:4 的 NVIDIA 典型密集张量链（并对照 L40S 官方 183 : 366 : 733 同为 1:2:4），可推断 **59.8 / 119.5 / 239 为密集张量值，稀疏值约翻倍（119.5 / 239 / 478）**；但此为**推定，非官方数据**。
> 注意另一歧义：国家超算互联网表的"半精度 119.5"经交叉验算（该表各型号"半精度 = 2 × 单精度"，如 4090 的 82.58→165.2、A800 的 155.92→311.84）实为 **FP16 向量算力**，与张量值数值巧合相近。故 L20 的 119.5 存在"FP16 向量"与"FP16 张量密集"两种可能解读。

### 4.5 出口管制背景
2023 年 10 月美国商务部更新高性能芯片出口管制后，NVIDIA 原计划 2023 年推出的 HGX H20 / L20 / L2 延期；L20 与 L2 均为 PCIe 4.0 x16 形态、Ada Lovelace 架构，算力低于 H20 — 【第三方/媒体】物联网世界（爱集微），2024-01-04，链接同上。

---

## 5. D) AMD Instinct MI300X（对照）

### 5.1 AMD 官方产品页
<https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html> 【AMD官方，2026-09-18 读取】

| 规格标签 | 数值（官方原文标签） |
|---|---|
| Family / Series | Instinct / Instinct MI300 Series |
| **Launch Date** | **12/06/2023** |
| GPU Architecture | **CDNA3** |
| Lithography | **TSMC 5nm ｜ 6nm FinFET** |
| Stream Processors | 19,456 |
| Matrix Cores | 1,216 |
| Compute Units | 304 |
| Peak Engine Clock | 2100 MHz |
| Peak Eight-bit Precision (FP8) Performance (E5M2, E4M3) | **2.61 PFLOPs** |
| Peak FP8 Performance **with Structured Sparsity** | **5.22 PFLOPs** |
| Peak Half Precision (FP16) Performance | **1.3 PFLOPs** |
| Peak FP16 Performance **with Structured Sparsity** | **2.61 PFLOPs** |
| Peak bfloat16 / with Structured Sparsity | 1.3 / 2.61 PFLOPs |
| Peak Single Precision (TF32 Matrix) | 653.7 TFLOPs |
| Peak TF32 with Structured Sparsity | 1.3 PFLOPs |
| Peak Single Precision Matrix (FP32) / FP32 | 163.4 / 163.4 TFLOPs |
| Peak Double Precision Matrix (FP64) / FP64 | 163.4 / **81.7** TFLOPs |
| Peak INT8 Performance | **2.6 POPs** |
| Peak INT8 Performance **with Structured Sparsity** | **5.22 POPs** |
| Transistor Count | 153 Billion |
| External Power Connectors | 54V UBB |
| **Typical Board Power (TBP)** | **750W Peak** |
| Last Level Cache (LLC) | 256 MB |
| Dedicated Memory Size / Type | **192 GB** / **HBM3** |
| Memory Interface / Memory Clock | 8192-bit / 5.2 GHz |
| **Peak Memory Bandwidth** | **5.3 TB/s** |
| Memory ECC Support | Yes (Full-Chip) |
| GPU Form Factor | **OAM Module**，Passive OAM |
| Bus Type | **PCIe® 5.0 x16** |
| Infinity Fabric™ Links | **8** |
| Peak Infinity Fabric™ Link Bandwidth | **128 GB/s** |

### 5.2 AMD 官方脚注（精确密集/稀疏措辞，原文摘录）
同页 Footnotes 【AMD官方】：
> **MI300-17**：…resulted in **653.7 TFLOPS** peak theoretical TensorFloat-32 (TF32), **1307.4 TFLOPS** peak theoretical half precision (FP16), **1307.4 TFLOPS** peak theoretical Bfloat16 (BF16), **2614.9 TFLOPS** peak theoretical 8-bit precision (FP8), **2614.9 TOPs** INT8 floating-point performance. The MI300X is expected to be able to take advantage of fine-grained structure sparsity providing an estimated 2x improvement in math efficiency resulting **1,307.4 TFLOPS** TF32, **2,614.9 TFLOPS** FP16, **2,614.9 TFLOPS** BF16, **5,229.8 TFLOPS** FP8, **5,229.8 TOPs** INT8 … **with sparsity**.
> **MI300-05A**：…resulted in 192 GB HBM3 memory capacity and **5.325 TB/s** peak theoretical memory bandwidth… MI300X memory bus interface is 8,192 and memory data rate is 5.2 Gbps for total peak memory bandwidth of **5.325 TB/s** (8,192 bits × 5.2 Gbps / 8).
> **MI300-15**：The AMD Instinct™ MI300X (750W) accelerator has **304 compute units (CUs), 19,456 stream cores, and 1,216 Matrix cores**.

### 5.3 上市与扩展域（AMD 官方新闻稿）
AMD 新闻稿《AMD Delivers Leadership Portfolio of Data Center AI Solutions with AMD Instinct MI300 Series》，**2023 年 12 月 6 日**（日文官方版：<https://www.amd.com/ja/newsroom/press-releases/2023-12-06-amd-ai-amd-instinct-mi300.html>；英文转载带日期戳 "Dec 6, 2023 3:00pm EST"：<https://www.nasdaq.com/press-release/amd-delivers-leadership-portfolio-of-data-center-ai-solutions-with-amd-instinct-mi300>）【AMD官方】：
- "announced the **availability** of the AMD Instinct™ **MI300X** accelerators"
- "features a best-in-class **192 GB of HBM3** memory capacity as well as **5.3 TB/s** peak memory bandwidth"
- "AMD Instinct Platform … **8 基の MI300X** アクセラレータ … **1.5TB** の HBM3 メモリー容量"（搭载 8 颗 MI300X，合计 1.5TB HBM3，OCP 设计）→ **8 卡纵向扩展域**
- 对比："Compared to the Nvidia H100 HGX, the AMD Instinct Platform can offer a throughput increase of up to **1.6x** when running inference on LLMs like BLOOM 176B"

---

## 6. 来源冲突与对本任务"已知参考点"的更正

### 6.1 ⚠️ L40S 参考点整组错误（最重要）
任务给出的 L40S 参考值 **"366 TFLOPS FP16 Tensor 稀疏（183 密集）、733 TFLOPS FP8 稀疏（366 密集）、1466 TOPS INT8 稀疏"** 与 NVIDIA 官方数据表/产品页**不符**：

| 项目 | 任务参考值 | **NVIDIA 官方实际值（密集｜稀疏）** | 判定 |
|---|---|---|---|
| FP16 Tensor | 183 密集 / 366 稀疏 | **362.05 密集 / 733 稀疏** | ❌ 错误 |
| FP8 Tensor | 366 密集 / 733 稀疏 | **733 密集 / 1,466 稀疏** | ❌ 错误 |
| INT8 Tensor | — / 1,466 稀疏 | **733 密集 / 1,466 稀疏** | ✅ 稀疏值正确 |
| FP32 | 91.6 | **91.6** | ✅ 正确 |
| TF32 Tensor | （未给） | **183 密集 / 366 稀疏** | 任务把 TF32 的 183/366 误当作 FP16 |

来源：NVIDIA 官方数据表 AUG23 与中文数据表第 2 页"技术规格"表，以及 <https://www.nvidia.com/en-us/data-center/l40s/> 规格表（三者一致，均为 FP16 362.05｜733）。**建议修正为 362.05｜733（FP16）、733｜1,466（FP8）、733｜1,466（INT8）。**

### 6.2 L40S RT Core 性能：209 vs 212 TFLOPS
- **212 TFLOPS**：NVIDIA 官方产品页规格表；NVIDIA 中国官方博客（2023-08-08）"142 颗第三代 RT Core，可提供 212 TFLOP 的光线追踪性能"。
- **209 TFLOPS**：NVIDIA 官方数据表 PDF（2841316. AUG23）第 2 页"RT Core Performance TFLOPS 209"。
→ **两份 NVIDIA 官方文件互相冲突**，疑为数据表未随最终规格更新。建议以产品页/博客的 **212 TFLOPS** 为准并标注冲突。

### 6.3 A800 40GB 功耗：240W vs 250W
- **240W**：【NVIDIA官方】"A800 **40GB Active**"工作站型号页面（主动散热、无显示输出）——<https://www.nvidia.com/en-us/products/workstations/a800/>
- **250W**：【OEM伙伴】天翼云对比表中 "A800 **40GB PCIe**" 数据中心型号；【第三方】hypertek 同值。
→ **并非直接矛盾**：二者为不同 SKU（Active 工作站版 240W vs 数据中心 PCIe 被动版 250W）。报告中须区分型号名，勿混用。

### 6.4 A800 INT8 数值呈现不一致
- 【NVIDIA官方】A100 数据表：单行 "**624 TOPS | 1248 TOPS***"（*稀疏），即全形态 624 密集 / 1,248 稀疏。
- 【OEM伙伴】天翼云表：PCIe 形态 624、SXM 形态 1,248（未标密集/稀疏）。
→ 建议采用 **624 密集 / 1,248 稀疏**，并注明天翼云表的 SXM "1,248" 为稀疏值。

### 6.5 A800 发布/上市日期
- **2022-11-08 公告**：路透社（URL 日期 2022-11-08，正文 401 未能读取）、彭博社同日、天翼云转述、hypertek"launched on November 8th, 2022"。
- **"Aug 2022 发布"**：topcpu.net（【第三方】GPU 数据库）称 A800 SXM4 "released on Aug 2022"。
→ 存在**弱冲突**；主流与厂商转述均指向 **2022-11-08 公告**，建议以此为准，"2022 年 8 月"标为第三方说法。

### 6.6 L20 算力/FP32 数值三方不一致
| 数值 | 来源 |
|---|---|
| FP32 **59.3** TFLOPS；FP16/BF16 **119**；FP8/INT8 **237** | 阿里云官方文档【OEM伙伴】 |
| FP32 **59.35**；FP16 **119.5**；FP64 0.927 | 国家超算互联网【第三方/平台】 |
| FP32 **59.8**；FP16 **119.5**；INT8/FP8 **239** | hypertek / MillionMiner【第三方】 |
→ 建议以**阿里云官方文档（59.3 / 119 / 237）**为主引，并注明第三方 59.8 / 119.5 / 239。

### 6.7 MI300X 数值精度
任务参考值 "1307 FP16 密集 / 2615 稀疏、2615 FP8 密集 / 5230 稀疏、2615 INT8 密集 / 5230 稀疏" **与 AMD 官方一致**（官方精确值 1307.4 / 2614.9 / 5229.8；产品页四舍五入为 1.3 / 2.61 / 5.22 PFLOPs）。**唯一需注意**：产品页"Peak INT8 Performance **2.6 POPs**"（P=拍，2.6 拍次整数运算），与脚注 2614.9 TOPs 为同一数值的不同量纲表述。

---

## 7. "未找到公开数据"清单

| 项目 | 状态 |
|---|---|
| **NVIDIA L20 官方产品页 / 官方数据表 / 官方规格表** | **未找到公开数据**（nvidia.cn 与 nvidia.com 的 L20 路径全部 404，2026-09-18 实测）。NVIDIA 仅在驱动发行说明中列出 L20 型号与 Ada Lovelace 归属。 |
| L20 官方密集/稀疏标注、制程节点、CUDA/RT/Tensor 核心数、MIG 支持 | 未找到公开数据（CUDA 核心数 11,776 仅见第三方平台） |
| L20 是否支持 NVLink | 无任何来源声明支持；阿里云"卡间互联"仅列 PCIe Gen4 x16 → **推定不支持**，但**未找到官方明确说明** |
| L20 官方 TDP | 未找到官方；275W 见三家第三方（hypertek / MillionMiner / technocity.ru） |
| **NVIDIA A800 40GB SXM 型号** | **未找到公开数据**（天翼云表 A800 仅列 40GB PCIe、80GB PCIe、80GB SXM） |
| A800 官方数据表 PDF（nvidia.com 原链） | 未找到（原页 404；chaoqing-i.com 镜像 20231128 链接现亦 404） |
| A800 官方 7nm 制程表述 | 未找到 A800 专属官方文件；以 GA100 = A100 官方 7nm/A100 架构类推（A100 数据表 + 第三方） |
| A800 官方公告新闻稿（NVIDIA Newsroom） | 未找到；以路透/彭博 + 天翼云转述为据 |
| MI300X 中文官方页面 | 未找到（AMD 中国站未见 mi300x 中文页；以英文官方页 + 日文新闻稿为据） |

---

## 8. 来源清单（含日期与信源等级）

**NVIDIA 官方**
1. NVIDIA L40S 产品页规格表 — <https://www.nvidia.com/en-us/data-center/l40s/> — 无页面日期，读取于 2026-09-18 —【NVIDIA官方】
2. NVIDIA L40S Data Sheet（2841316. **AUG23**，PNY 托管官方文档）— <https://www.pny.com/File%20Library/Company/Support/Product%20Brochures/NVIDIA%20Data%20Center%20GPUs/l40s-datasheet.pdf> —【NVIDIA官方】
3. NVIDIA L40S 中文数据表（2907810，**2023 年 8 月**；PDF 生成 2023-09-25）— <http://images.nvidia.cn/cn/RTX/l40s-datasheet-web-a4-zhCN-2907810-2023.9.25.pdf> —【NVIDIA官方】
4. NVIDIA 中国官方博客《NVIDIA 与全球数据中心系统制造商大力推动 AI 与工业数字化的发展》— **2023-08-08** — <https://blogs.nvidia.cn/blog/nvidia-global-data-center-system-manufacturers-to-supercharge-generative-ai-and-industrial-digitalization/> —【NVIDIA官方】
5. NVIDIA A800 40GB Active 官方产品页 — <https://www.nvidia.com/en-us/products/workstations/a800/> — 无页面日期，读取于 2026-09-18 —【NVIDIA官方】
6. NVIDIA A100 Tensor Core GPU Data Sheet — **JUN21** — <https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf> —【NVIDIA官方】
7. NVIDIA Data Center GPU Driver Release Notes 580.82.07 (Linux)，RN-08625-565 v2.1 — **September 2026** — <https://docs.nvidia.com/datacenter/tesla/pdf/NVIDIA_Data_Center_GPU_Driver_Release_Notes_580_v2.1.pdf> —【NVIDIA官方】
8. NVIDIA Ada GPU Architecture 白皮书 — <http://images.nvidia.cn/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf> —【NVIDIA官方】

**AMD 官方**
9. AMD Instinct MI300X 产品页 — <https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html> — 读取于 2026-09-18 —【AMD官方】
10. AMD 新闻稿（日文官方）《AMD Instinct MI300 シリーズを提供開始》— **2023-12-06** — <https://www.amd.com/ja/newsroom/press-releases/2023-12-06-amd-ai-amd-instinct-mi300.html> —【AMD官方】
11. AMD 新闻稿英文转载（日期戳 Dec 6, 2023 3:00pm EST）— <https://www.nasdaq.com/press-release/amd-delivers-leadership-portfolio-of-data-center-ai-solutions-with-amd-instinct-mi300> —【AMD官方新闻稿，第三方转载】

**OEM / 云厂商伙伴**
12. 天翼云（中国电信）《英伟达受限GPU与常规型号参数对比》— <https://www.ctyun.cn/developer/article/437601152262213> —【OEM伙伴】
13. 阿里云 GPU 计算型实例规格族文档（gn8is / ebmgn8ia / ebmgn8is）— <https://www.alibabacloud.com/help/zh/egs/gpu-accelerated-compute-optimized-instance-families>（繁中版：<https://www.alibabacloud.com/help/tc/egs/gpu-accelerated-compute-optimized-instance-families>）— 读取于 2026-09-18 —【OEM伙伴】
14. H3C UniServer R5500 G5（Red Hat 硬件兼容目录，HGX A800 8-GPU/6 NVSwitch/400GB/s）— <https://catalog.redhat.com/en/hardware/system/detail/70035> —【OEM伙伴】
15. 国家超算互联网平台《国产加速卡介绍》— <https://www1.scnet.cn/help/docs/mainsite/ai/introduction/> —【第三方/平台】

**第三方 / 媒体**
16. 路透社（独家，URL 日期 **2022-11-08**，正文 401 未读取）— <https://www.reuters.com/technology/exclusive-nvidia-offers-new-advanced-chip-china-that-meets-us-export-controls-2022-11-08/> —【第三方/媒体】
17. 彭博社（**2022-11-08**）— <https://www.bloomberg.com/news/articles/2022-11-08/nvidia-to-sell-new-chip-in-china-it-says-meets-us-export-ban> —【第三方/媒体】
18. IT之家《英伟达内部人士确认：11 月 16 日推出三款中国特供版 AI 芯片》— **2023-11-10** — <https://m.ithome.com/html/731545.htm> —【第三方/媒体】
19. 物联网世界（爱集微）《英伟达H20 AI GPU参数曝光》— **2024-01-04** — <https://www.iotworld.com.cn/html/News/202401/52da19aad1c61b42.shtml> —【第三方/媒体】
20. 超能网 expreview《英伟达推出A800系列计算卡：NVLink带宽限制在400GB/s，专供中国市场》— <https://www.expreview.com/85490.html> —【第三方/媒体】（正文抓取失败，仅标题可证）
21. hypertek《NVIDIA A800 PCIe 40 GB》— <https://www.hypertek.nl/product/nvidia-a800-pcie-40-gb/> —【第三方/经销商】
22. hypertek《NVIDIA L20 48GB GPU》— <https://www.hypertek.nl/product/nvidia-l20-48gb-gpu/> —【第三方/经销商】
23. cputronic《NVIDIA A800 SXM4 80 GB》— <https://cputronic.com/en/gpu/nvidia-a800-sxm4-80-gb> —【第三方/GPU数据库】
24. topcpu《NVIDIA A800 SXM4 80 GB》— <https://topcpu.net/en/cpu/a800-sxm4-80-gb> —【第三方/GPU数据库】
25. MillionMiner《NVIDIA L20 48GB Ada Lovelace Data Center GPU》— <https://millionminer.com/product/ai-hardware/nvidia-l20-enterprise-48gb> —【第三方/经销商】

**未能获取（说明）**：TechPowerUp GPU 数据库（机器人校验拦截）、fpsbench.com（403）、alza.cz（403）、华西证券研报 PDF（域名 aigc.idigital.com.cn 连接失败，其搜索摘要含 NVIDIA H20/L20/L2 官方规格表，但**未能读取正文，故本报告未引用其数字**）、联想的 ThinkSystem A800 产品指南 lp1813.pdf（需登录）。
