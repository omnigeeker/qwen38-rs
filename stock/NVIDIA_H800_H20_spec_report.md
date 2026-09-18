# NVIDIA H800 / H20 中国特供数据中心 GPU 规格与出口管制时间线报告

**编制日期：2026-09-18（检索日）｜用途：与寒武纪 MLU 系列做对比表的数据底稿**

**引用体例**：正文中每个数字后附来源代码 `[Nx]`/`[Tx]`，代码对应的完整来源（URL、媒体名、发布日期、官方/第三方标签）见文末 **§8 来源登记表**。所有数字均区分 **dense（稠密）** 与 **sparse（2:4 结构化稀疏，= 2×dense）**。

---

## §0 三条最重要的结论（先看这一段）

1. **H20 的关键纠正：296 不是"稀疏值"，而是 dense 值。**
   广泛流传的 H20 "FP8 296 TFLOPS / INT8 296 TOPS / FP16 148 TFLOPS" 是 **dense 口径**，不是 sparse。三条独立证据：
   - 腾讯科技/全天候科技（2023-11-28）明确写 H20 单卡算力 "0.148P（FP16）/ 0.296P（Int8）"，并称其"**算力水平约等于 50% A100 和 15% H100**" `[T21]`。H100 dense FP16 = 989 TFLOPS，989 × 15% = 148.4 ≈ 148 `[N6]`；A100 dense FP16 = 312 TFLOPS，×50% = 156 ≈ 148 `[T25]`。**只有把 148 当作 dense，这个"15%"才成立**。
   - Ant Group/LMSYS 工程博客（2025-09-26）的 H20 vs H800 对照表用 dense 口径列 H800（FP16 989 / FP8 1979，正是 H100 的 dense 值），同一张表里 H20 写 FP16/BF16 = 148、FP8 = 296 `[T3]` —— 同表同口径，即 dense。
   - flopper.io 的 H20 96GB / 141GB 规格页显式分列 "Peak (dense)" 与 "2:4 Sparse"：FP16/BF16 dense 148、FP8 dense 296、FP8 sparse 592、INT8 dense 296 `[T4][T5]`。
   - 交叉验算（本报告自行推导，仅作逻辑校验）：美国 BIS 2023-10-17 规则对 ECCN 3A090.a 的门槛之一是 "总算力之和 ≥ 4800 TOPS" `[T21]`（规则原文见 15 CFR 774 附录 `[T27]`-A）。H20 dense INT8 = 296 TOPS 时，TPP = 2 × 296 × 8 = **4736 < 4800**，刚好卡在门槛之下。若 296 是 sparse（dense 只有 148，TPP = 2368），NVIDIA 本可把规格做得高一倍。**296 = dense 是 H20 能合规存在的算术前提。**
   → **因此 H20 的官方口径下未见公开的 sparse 数值（未找到公开数据）；按 2:4 稀疏 2× 关系推算，稀疏值应为 FP16/BF16 296、FP8 592、INT8 592。使用时必须注明这是推算值而非 NVIDIA 公布值。**

2. **H800 是"只砍 NVLink（+FP64）"的 H100，算力 dense 口径与 H100 同级。** H800 SXM5：BF16/FP16 **989 dense / 1979 sparse**，FP8 **1979 dense / 3958 sparse**，INT8 **1979 dense / 3958 sparse** `[T1][T3]`；NVLink 从 900 GB/s 砍到 **400 GB/s 双向**（NVIDIA 发言人经路透证实 H800 的片间传输速率"降到 H100 的一半"）`[T11]`。

3. **H800 PCIe 确实有 NVLink，带宽 400 GB/s。** NVIDIA H800 PCIe 规格表（渠道商 AFOX 完整转载 N 官方 H800 规格页文案与表格）明确列 "Interconnect: NVLink: 400GB/s, PCIe Gen5: 128GB/s" `[N5]`。**"H800 PCIe 没有 NVLink"的说法不成立。**

---

## §1 H800 —— 型号谱系（NVIDIA 官方文档确认）

NVIDIA AI Enterprise 官方文档（最后更新 **2026-09-02**）逐条列出 H800 的存在形态：

| SKU（官方命名） | 官方 framebuffer | MIG profile 示例 | 来源 |
|---|---|---|---|
| NVIDIA H800 SXM5 80 GB | 80 GB | MIG 7g.80gb / 4g.40gb / 2g.20gb / 1g.10gb … | `[N1]` |
| NVIDIA H800 PCIe 80 GB | 80 GB | MIG 7g.80gb / 4g.40gb / 2g.20gb / 1g.10gb … | `[N1]` |
| NVIDIA H800 PCIe 94 GB（**H800 NVL**） | 94 GB | MIG 7g.94gb / 4g.47gb / 3g.47gb / 2g.24gb / 1g.12gb … | `[N1]` |

- **性质**：`[N1]` 为 **NVIDIA官方**（docs.nvidia.com）—— 这是"H800 PCIe 94GB / H800 NVL 存在"的最硬证据。
- 官方文档同时给出架构级说明："Hopper … NVLink 4.0 where the board provides it, and high-bandwidth memory (HBM3 or HBM3e depending on product)" `[N1]`。

---

## §2 H800 规格明细（dense / sparse 严格分列）

### 2.1 H800 SXM5 80 GB

| 项目 | 数值 | 口径 | 来源 |
|---|---|---|---|
| 架构 | Hopper（GH100） | — | `[N5][T1][T2]` |
| 制程 | TSMC 4N（渠道页写 5nm / 4nm） | — | `[T1][T2]`（4N 为 NVIDIA 对 Hopper 的官方工艺命名） |
| SM 数 | 132 | — | `[T26]`（第三方数据库；NVIDIA 未公开确认 H800 SM 数） |
| CUDA core | 16,896 | — | `[T1][T2]` |
| Tensor Core | 528（第 4 代） | — | `[T1][T2][T21]` |
| 显存 | 80 GB HBM3，5120-bit | — | `[T1][T2][T3]` |
| 显存带宽 | **3.35 TB/s**（Ant Group 表写 3352 GB/s） | — | `[T1][T2][T3]` |
| FP64（CUDA core） | **1 TFLOPS** | — | `[T1][T2]` |
| FP64 Tensor Core | **1 TFLOPS** | — | `[T1]`（H100 SXM 为 34/67，H800 砍到 1） |
| FP32（非 Tensor） | **67 TFLOPS** | — | `[T1][T2]` |
| TF32 Tensor Core | **494 dense / 989 sparse** | dense/sparse | `[T1]`（989 标 `*`=稀疏）、`[T25]` |
| BF16 Tensor Core | **989 dense / 1979 sparse** | dense/sparse | `[T1][T3]`（989 dense）、`[T1]`（1979 标 `*`） |
| FP16 Tensor Core | **989 dense / 1979 sparse** | dense/sparse | `[T1][T3]`、`[T1]`（1979 `*`） |
| FP8 Tensor Core | **1979 dense / 3958 sparse** | dense/sparse | `[T3]`（1979 dense）、`[T1][T2]`（3958 `*`） |
| INT8 Tensor Core | **1979 dense / 3958 sparse**（TOPS） | dense/sparse | `[T1][T2]`（3958 TOPS `*`） |
| NVLink | **NVLink 4，400 GB/s 双向（每 GPU）** | — | `[N5][T1][T3][T11]` |
| NVSwitch | 支持（HGX 基板 8 卡全互联；NVLink Switch System 可连"最多 256 个 H800"） | — | `[N5]`（渠道页完整转载 NVIDIA 文案） |
| PCIe | **PCIe Gen5 ×16，128 GB/s** | — | `[N5][T1]` |
| TDP | **700 W（可配置）** | — | `[T1][N5-表]` |
| 形态 | SXM5；HGX H800 8-GPU | — | `[T1][N5]` |
| MIG | 最多 7 实例 | — | `[N5][N1]` |
| **最大 scale-up 域** | **8 GPU（单 HGX）**；经 NVLink Switch System 可扩到 **256 GPU** | — | `[N5]` |

> 注：`[T2]`（Cloud Hin，2023-03）页面自注"数据来源：英伟达"，列 FP32 67 TFLOPS / FP64 1 TFLOPS / 80GB HBM3 / 3.35 TB/s / 700W。

### 2.2 H800 PCIe 80 GB

NVIDIA 的 H800 PCIe 规格表（经渠道商 AFOX **逐项完整转载**，含 N 官方文案"With the NVIDIA NVLink Switch System, up to 256 H100 GPUs can be connected"）`[N5]`：

| 项目 | 数值 | 口径 |
|---|---|---|
| FP64 / FP64 Tensor Core | **0.8 TFLOPS / 0.8 TFLOPS** | — |
| FP32（非 Tensor） | **51 TFLOPS** | — |
| TF32 Tensor Core | **378 dense / 756 sparse** | dense/sparse |
| BF16 Tensor Core | **756 dense / 1513 sparse** | dense/sparse |
| FP16 Tensor Core | **756 dense / 1513 sparse** | dense/sparse |
| FP8 Tensor Core | **1513 dense / 3026 sparse** | dense/sparse |
| INT8 Tensor Core | **1513 dense / 3026 sparse**（TOPS） | dense/sparse |
| 显存 | **80 GB** | — |
| 显存带宽 | **2.0 TB/s** | — |
| 显存类型 | **未找到公开数据**（该表未标注；H100 PCIe 80GB 对应型号业界多记为 HBM2e） | — |
| NVLink | **400 GB/s** | — |
| PCIe | PCIe Gen5，128 GB/s | — |
| TDP | **350 W** | — |
| 形态 | PCIe 双槽风冷 | — |
| MIG | 最多 7 × 10 GB | — |
| 服务器形态 | 合作厂商及 NVIDIA-Certified 系统，1–8 GPU | — |

> **回答提问**：H800 PCIe **有 NVLink，400 GB/s**（`[N5]`，非"无 NVLink"）。表中所有 Tensor Core 数字带 `*`，AFOX 转载页未保留脚注，但按 NVIDIA 一贯脚注（"Specification in sparse. Dense is one-half"）以及 H100 同构对照，**带 `*` 者 = sparse，dense 为其一半** `[N5][T25]`。

### 2.3 H800 PCIe 94 GB（H800 NVL）

- **存在性：NVIDIA 官方确认**（`[N1]`，MIG profile 7g.94gb / 4g.47gb / 3g.47gb / 2g.24gb / 1g.12gb）。
- 显存带宽 / 显存类型 / TDP：**未找到权威公开数据**。第三方低可信度渠道页 `[T27]` 声称 94GB HBM3 / 3.9 TB/s / 400W，但该页面同时把 H800 写成"FP64 30 TFLOPS 与 H100 相同"、"80GB HBM2e"，与 NVIDIA 官方 H800 PCIe 表（FP64 0.8 TFLOPS）直接矛盾，**该来源不可采用**（见 §6 冲突）。

---

## §3 H20 规格明细

### 3.1 型号谱系（NVIDIA 官方文档确认）

NVIDIA AI Enterprise 官方文档（最后更新 **2026-09-02**）列出两个 H20 SKU `[N2]`：

| SKU（官方命名） | 官方 framebuffer | MIG profile 示例 |
|---|---|---|
| **NVIDIA H20 SXM5 96 GB** | 96 GB | MIG 7g.96gb / 4g.48gb / 3g.48gb / 2g.24gb / 1g.12gb |
| **NVIDIA H20 SXM5 141 GB** | **141 GB** | MIG 7g.141gb / 4g.71gb / 3g.71gb / 2g.35gb / 1g.18gb |

- **H20 141GB 版本确实存在，且为 NVIDIA 官方文档所载** `[N2]`（**NVIDIA官方**）。
- 形态：官方文档只列 **SXM5**（无 PCIe SKU）`[N2]`。媒体亦确认 H20 为 SXM 板卡形态、兼容 8 路 HGX `[T8][T6][T21]`。

### 3.2 H20 性能 / 存储 / 互联（dense / sparse 严格分列）

| 项目 | H20 96 GB | H20 141 GB | 口径 | 来源 |
|---|---|---|---|---|
| 架构 | Hopper（GH100，814 mm² die） | 同 | — | `[T21][T7]` |
| 制程 | TSMC 4N（4nm） | 同 | — | `[T4][T5]` |
| SM 数 | **78**（GH100 完整为 144；H100 SXM5 启用 132、PCIe 启用 114） | 同 | — | `[T7][T6][T9]`（Geekbench 6 数据，第三方） |
| L2 cache | **60 MB** | 同 | — | `[T7][T6][T9]` |
| 显存 | **96 GB HBM3**（腾讯科技 2023-11 稿称 6×16GB **HBM3e**，存冲突，见 §6） | **141 GB HBM3e** | — | `[T6][T7][T8][T21]`（96GB）；`[N2][T5]`（141GB） |
| 显存带宽 | **4.0 TB/s** | **4.0 TB/s** | — | `[T6][T7][T8][T21][T3][T4]`；`[T5]` |
| **FP64** | 1 TFLOPS | 1 TFLOPS | — | `[T6][T7][T8][T21][T4][T5]` |
| **FP32（非 Tensor）** | **44 TFLOPS** | 44 TFLOPS | — | `[T6][T7][T21][T4][T5]` |
| **TF32 Tensor Core** | **74 dense** | 74 dense | dense | `[T6][T7][T8][T21][T4][T5]` |
| TF32 Tensor Core sparse | **未找到公开数据**（按 2× 推算 148） | 同 | — | `[T4][T5]` 标 "not published" |
| **BF16 Tensor Core** | **148 dense** | 148 dense | dense | `[T3][T6][T7][T8][T21][T4][T5]` |
| BF16 sparse | **未找到公开数据**（按 2× 推算 296） | 同 | — | `[T4][T5]` 标 "not published" |
| **FP16 Tensor Core** | **148 dense** | 148 dense | dense | `[T3][T6][T7][T21][T4][T5]` |
| FP16 sparse | **未找到公开数据**（按 2× 推算 296） | 同 | — | `[T4][T5]` "not published" |
| **FP8 Tensor Core** | **296 dense** | 296 dense | dense | `[T3][T6][T7][T8][T21][T4][T5]` |
| FP8 sparse | **592**（flopper.io 明确列出） | 592 | sparse | `[T4][T5]` |
| **INT8 Tensor Core** | **296 dense（TOPS）** | 296 TOPS | dense | `[T6][T7][T8][T21][T4][T5]` |
| INT8 sparse | **未找到公开数据**（按 2× 推算 592） | 同 | — | `[T4][T5]` "not published" |
| **NVLink** | **NVLink 4，900 GB/s 双向（每 GPU）** | 900 GB/s | — | `[T6][T7][T8][T21][T3][T5]` |
| NVSwitch | 支持（8 路 HGX 基板全互联） | 同 | — | `[T6][T8][T21]` |
| PCIe | PCIe Gen5 ×16（128 GB/s，与 Hopper 平台一致） | 同 | — | `[T21]`（PCIe Gen5 表述）；128 GB/s 为 Hopper SXM 平台通用值 `[N5]` |
| **TDP** | **400 W** | **400 W** | — | `[T6][T7][T8][T21][T4][T5]` |
| 形态 | SXM5（8 路 HGX） | SXM5 | — | `[N2][T8][T6]` |
| MIG | 最多 7 实例 | 最多 7 | — | `[T7][N2]` |
| RDMA NIC（参考） | 4 × 400 Gb/s | — | — | `[T3]`（Ant Group 生产集群实测配置） |
| **最大 scale-up 域** | **8 GPU（单 HGX 8 路）** | 8 GPU | — | `[T8][T6][T21]`（未见 H20 支持 256-GPU NVLink Switch System 的报道） |

### 3.3 H20 141 GB 的时间与价格线索

- **官方存在性**：`[N2]`（NVIDIA官方，文档更新 2026-09-02）。
- **上市/流通时间**：科创板日报（2025-03-03）报道渠道商推广 "3 月中旬到货 H20 141G 整机"，报价 **141G 整机含税 120 万元**、96G 整机含税 97 万元 `[T22]`。→ 可确认 **2025 年 3 月前后 141G 已在渠道现货流通**。
- 腾讯科技（2025-08-11）称 H20 "早期版本搭载 96GB HBM3 …**后期又推出了 141GB 显存的版本**"，8 卡服务器价格一度超 110 万元 `[T16]`。
- TrendForce（2025-07-16）引 Cailian Press：2025-07-15 渠道报价 **H20 141G（不含 IB 卡）125 万元**，与 3 月初基本持平 `[T12]`。
- **141GB 版本的首发日期（官方发布/量产日）：未找到公开数据**（可确证的是"2025 年 Q1 已在渠道流通"，非官方发布时间）。

---

## §4 时间线：发布 / 可用性 / 出口管制（逐条带日期）

| 日期 | 事件 | 来源与性质 |
|---|---|---|
| 2022-10-07 | 美国首轮对华先进芯片出口管制生效，A100/H100 被禁 | `[T10]`（Reuters 回溯叙述） |
| **2022-11（约管制后一个月）** | **NVIDIA 面向中国推出 A800 与 H800 作为替代型号**（"introduced as alternatives for Chinese customers in November 2022 about a month after the U.S. first restricted exports"） | `[T10]`（Reuters，2024-01-08）｜第三方/Reuters |
| **2023-03-21/22** | **NVIDIA 发言人公开承认 H800 存在**，并称其"已被阿里巴巴、百度、腾讯等中国科技企业用于云计算"；NVIDIA 拒答 H800 与 H100 的具体差异，仅称"H800 系列产品完全符合出口管制规定"。一位中国半导体产业消息人士称 H800 的片间数据传输速率**降到 H100 的一半** | `[T11]`（韩联社 2023-03-22 转 Reuters 2023-03-21）｜第三方/Reuters 转述 |
| 2023 年（H800 上市） | 第三方数据库记 H800 SXM5 release date = **2023-03** | `[T26]`（第三方数据库，仅供参考） |
| 2023-10-17 | 美国商务部更新管制标准（总算力之和 ≥4800 TOPS；总算力 ≥1600 且性能密度 ≥5.92 等），**A800/H800 被纳入禁售** | `[T21]`（腾讯科技/全天候科技，2023-11-28）；规则原文 `[T27]`-A |
| 2023-11 | NVIDIA 规划 H20 / L20 / L2 三款新特供型号；H20 原定 2023-11 发布，**因服务器厂商集成问题延期** | `[T10]`（Reuters，2024-01-08） |
| 2023 年末 | NVIDIA 基于 Hopper 为大陆客户定制 H20，**2024 年大规模发货**，接替受管制的 H800 | `[T16]`（腾讯科技，2025-08-11） |
| 2024-01-08 | Reuters：H20 计划 **2024 Q2 量产**，首批量有限、优先满足大客户 | `[T10]` |
| 2024-01-02 | 爱集微曝光 H20 参数（FP8 296 TFLOPS、FP16 148 TFLOPS、96GB HBM3、4.0 TB/s、NVLink 900 GB/s、SXM、8 路 HGX） | `[T8]` |
| 2024-02 | Reuters：H20 定价与华为产品相近 | `[T10]` 同期报道群（Reuters 2024-02-01） |
| 2024-07-10 | Geekbench 6 曝光 H20：78 SM、60MB L2、400W、7 MIG（Wccftech 首发，中文媒体转载） | `[T7][T9][T6]` |
| 2025-03-03 | 渠道现货在售 **H20 141G** 整机，含税 120 万元 | `[T22]`（科创板日报） |
| **2025-04-09** | **美国政府通知 NVIDIA：H20 对华出口需申请许可**（NVIDIA 8-K 披露的事件日） | `[N3]`（NVIDIA 8-K，事件日 2025-04-09，提交日 2025-04-15）｜**NVIDIA官方/监管文件** |
| **2025-04-15** | NVIDIA 提交 8-K / 公告：预计计提**最高 55 亿美元**（$5.5B）费用，涉及 H20 库存与采购承诺；H20 出口"无限期"需许可 | `[N3]`；第三方确认标题 `[T23]`（MarketWatch，2025-04-15/16）、`[T24]`（"Nvidia Faces $5.5B Charge on New H20 Export Curbs to China Placed 'Indefinitely'"） |
| 2025-04-16 | Reuters：NVIDIA 未及时告知部分中国客户新管制（"kept some China customers in the dark"） | Reuters 2025-04-16（检索到标题，正文付费墙） |
| 2025-05-09 | ZOL 引路透：NVIDIA 计划**两个月内**为中国推出**降规版 H20**（内存容量明显减少），并已通知主要云厂商 7 月发布 | `[T28]`（中关村在线，2025-05-09，转路透） |
| 2025-05-28 | NVIDIA Q1 FY2026 财报电话会：Q1 实际计提 **45 亿美元** H20 超额库存与采购义务费用；管制前 H20 当季销售 **46 亿美元**；另有 **25 亿美元** H20 收入无法发货；若无管制本应有约 **80 亿美元** H20 订单 | `[T17]`（ComputerWeekly，2025-05-29，引 NVIDIA 财报电话会）｜第三方引 NVIDIA 官方口径 |
| **2025-07-14/15** | **美国确认将恢复 H20 对华销售**。NVIDIA 公告将恢复 H20 销售；黄仁勋称正在提交许可申请，美方保证将获批；同时发布新的合规 RTX PRO GPU | `[T15]`（China Daily，2025-07-15）、`[T29]`（BBC，2025-07-15）、`[T12]`（TrendForce，2025-07-16，引 Reuters 2025-07-15） |
| 2025-07-16 | 黄仁勋在链博会称"已经有很多订单"，客户在等发货通知；同期 IT之家复述 H20 参数 | `[T6]`（IT之家，2025-07-16） |
| 2025-07-29 | DigiTimes：H20 库存远不足需求（需求约 180 万块），NVIDIA 重新向台积电下单约 **30 万张**（NVIDIA 未正面回应） | `[T16]`（腾讯科技，2025-08-11，转 DigiTimes）｜**传闻** |
| **2025-08-08** | 据报道 **BIS 开始发放** H20 与 AMD MI308 的出口许可 | `[T13]`（JETRO，2025-08-18，引美媒/Reuters 8-12） |
| **2025-08-11** | **特朗普公开确认**：NVIDIA 与 AMD 同意将对华 AI 芯片销售收入的 **15%** 上缴美国政府，以换取出口许可（FT 首发，Reuters 8-12 跟进） | `[T13]`（JETRO，2025-08-18）；`[T16]`（腾讯科技，2025-08-11，引 FT） |
| **2025-08-12** | **Reuters / Bloomberg：中国监管机构要求本土企业避免使用 H20**，尤其政府/国家安全相关项目；非全面禁售 | `[T13]`（JETRO 引 Reuters 8-12 与 Bloomberg）；Reuters 西语版标题《China insta a las empresas locales a no utilizar los chips H20 de Nvidia, según Bloomberg News》（2025-08-12） |
| 2025-08-13 | 中国外交部发言人林剑例行记者会：就被问及"中国劝告企业避免使用 H20"回应"**我不掌握你提到的情况**"；记者提问中提到中国监管机构此前已**约谈 NVIDIA**表达对其芯片安全风险的关切 | `[T20]`（中国驻几内亚使馆发布的外交部记者会记录，2025-08-13）｜**中国政府官方** |
| 2025-08-22 | **The Information**：NVIDIA 已通知安靠（Amkor）、三星等关键供应商**暂停 H20 相关生产** | `[T19]`（投中网/虎嗅，2025-08-25，引 The Information）｜**传闻（The Information）** |
| 2025-09-08 | NVIDIA CFO Colette Kress 在高盛科技大会：已获美方出口许可，并为几家中国主要客户取得许可证，但仍存在问题待协调；预计 H20 对华收入 **20 亿–50 亿美元**；Q2 未计入 H20 收入，仅向其他地区释放 1.8 亿美元 H20 库存 | `[T18]`（观察者网/网易，2025-09-10，引 Investing/Bloomberg 与 NVIDIA CFO） |
| 2025-09-15 | 中国认定 NVIDIA 违反反垄断法（涉 2020 年 70 亿美元收购 Mellanox），宣布进一步调查 | `[T14]`（VietnamPlus，2025-09-17，引 FT）；`[T14]` 相关报道 |
| **2025-09-17** | **FT：中国国家网信办（CAC）本周要求字节跳动、阿里巴巴等头部企业停止采购/测试 NVIDIA AI 芯片，并取消现有订单**（首当其冲是 RTX Pro 6000D）；FT 称该禁令比此前针对 H20 的指导**更严格** | `[T14]`（VietnamPlus，2025-09-17，引 FT，三名知情人士）｜第三方转 FT |

> **关于"中国监管机构是否劝退 H20 采购（2025 年 9 月）"的准确回答**：
> - 针对 **H20** 的监管劝退，最早可追溯至 **2025-08-12** 的 Reuters/Bloomberg 报道（要求本土企业避免在政府/国安相关场景使用 H20）`[T13]`；中国外交部 2025-08-13 **未予确认** `[T20]`。
> - 2025-09-17 FT 报道的是**范围更大的禁令**（含 RTX Pro 6000D，且明确"比此前针对 H20 的指导更严格"）`[T14]`。把 9 月 FT 报道直接等同于"H20 被禁购"并不准确。

---

## §5 制程 / 架构 / 平台归属

| 项目 | H800（全部 SKU） | H20（全部 SKU） | 来源 |
|---|---|---|---|
| 架构世代 | Hopper（第 4 代 Tensor Core、Transformer Engine、FP8、第 2 代 MIG、机密计算） | Hopper（同） | `[N1][N5][N2][T8][T21]` |
| Die | GH100，814 mm²（≈ 全掩模尺寸上限 858 mm²） | GH100，814 mm²（与 H100 同 die） | `[T21]` |
| 制程 | **TSMC 4N**（NVIDIA 对 Hopper 的官方工艺命名；第三方数据库写作 4nm/5nm） | TSMC 4N（4nm） | `[T1][T2][T4][T5]` |
| 晶体管 | 800 亿 | 800 亿（同 die） | `[T1][T21]` |
| 平台 | HGX H800 8-GPU；NVLink Switch System 最多 256 GPU | 8 路 HGX | `[N5][T8][T6]` |

---

## §6 冲突清单与权威性判读

| # | 冲突点 | 各来源说法 | 权威判读 |
|---|---|---|---|
| **C1** | **H20 FP8/INT8 的 296 是 dense 还是 sparse** | (a) 中文媒体通稿式表述"FP8 算力 296 TFLOPS / INT8 296"多未注明口径 `[T8][T6][T7]`；(b) Ant Group/LMSYS 与 flopper.io 明确标为 **dense** `[T3][T4][T5]`；(c) 提问中的参考点称"FP8 ~296 sparse" | **采信 dense**。理由：(i) `[T3]` 同表用 dense 列 H800（989/1979 = H100 dense），H20 数字必为同口径；(ii) `[T21]` 的"15% H100 / 50% A100"换算只在 dense 下成立；(iii) TPP=4736<4800 的合规算术只在 dense 下成立。**NVIDIA 从未公开 H20 的 sparse 数值**，任何 296 稀疏说法属口径误标。 |
| **C2** | **H20 96GB 的显存是 HBM3 还是 HBM3e** | (a) 绝大多数报道：**96GB HBM3** `[T6][T7][T8]`；(b) 腾讯科技 2023-11 稿：**6×16GB HBM3e** = 96GB `[T21]` | 采信 **HBM3**（多数派 + 与"后期才推出 141GB HBM3e"的叙述自洽 `[T16]`）。`[T21]` 的 HBM3e 说法可能是早期爆料误差，**保留为冲突**。 |
| **C3** | **H800 SXM 的 FP64** | (a) NVIDIA H800 PCIe 表：FP64 = **0.8 TFLOPS** `[N5]`；(b) 渠道页：H800 **SXM** FP64 = **1 TFLOPS**（FP32 67）`[T1][T2]`；(c) `[T27]` 低质页：FP64 = **30 TFLOPS"与 H100 相同"** | 采信 (a)/(b) 的分 SKU 差异（SXM 1 / PCIe 0.8）。(c) **不可采信**：与 NVIDIA 官方 H800 PCIe 表直接矛盾。 |
| **C4** | **H800 的 Tensor 算力数值** | (a) `[T1][T2]`：TF32 989\* / BF16 1979\* / INT8 3958\*（**sparse**，标 `*`）；(b) `[N5]` H800 PCIe：TF32 756\* / BF16 1513\* / FP8 3026\*（sparse）；(c) `[T27]` 把 989/1979/3958 **当作非稀疏"峰值算力"**列出 | (a)(b) 一致且自洽（均带稀疏脚注），采信。(c) **错误**：把 sparse 值当 dense 值公布，会导致对比表整体偏高 2×。 |
| **C5** | **H800 显存类型** | (a) `[T1][T2]`：80GB HBM3；(b) `[T27]`：80GB HBM2e（PCIe 版） | SXM 采信 **(a) HBM3**（与 3.35 TB/s 相符）。H800 PCIe 80GB 的显存类型 `[N5]` 未标注 → **未找到官方公开数据**；同带宽（2.0 TB/s）的 H100 PCIe 业界多记为 HBM2e。 |
| **C6** | **H800 的 CUDA core / 制程** | (a) `[T1][T2]`：16,896 CUDA / 528 Tensor，5nm；(b) `[T26]`（cpudb/TechPowerUp 系）：给出 "FP16 237.2 TFLOPS / FP32 59.30 / FP64 29.65 TFLOPS" | (a) 的 CUDA/Tensor 数与 H100 SXM 相同，可信。**(b) 的 237.2/59.30/29.65 是 CUDA-core 向量算力（FP16=2×FP32 的向量吞吐），不是 Tensor Core 规格**，不可与 989/1979 混用。制程以 **TSMC 4N** 为准，5nm/4nm 只是通俗写法。 |
| **C7** | **H800 PCIe 是否有 NVLink** | (a) `[N5]`：**有，400 GB/s**；(b) 坊间说法：无 NVLink | 采信 **(a)**。`[N5]` 完整转载 NVIDIA H800 规格表并含 NVLink Switch System 文案。 |
| **C8** | **H20 96GB vs 141GB 的带宽是否不同** | `[T3][T4]`：两者均 **4.0 TB/s**；`[T5]`：141GB 版 4.0 TB/s | 一致，无冲突。141GB 版**容量提升但带宽不变**（HBM3e 仅换颗粒/密度）。 |
| **C9** | **H20 是否有 PCIe 形态** | NVIDIA 官方文档只列 SXM5 `[N2]`；中文媒体均称 SXM 板卡、8 路 HGX `[T8][T6]` | 采信 **仅 SXM5**。未见任何 HGX H20 之外的 PCIe SKU 证据。 |
| **C10** | **55 亿 vs 45 亿美元的 H20 计提** | (a) NVIDIA 2025-04-15 8-K/公告：预计**最高 55 亿美元** `[N3][T23][T24]`；(b) NVIDIA Q1 FY2026（2025-05-28）实际计提 **45 亿美元** `[T17]`；(c) 腾讯科技 2025-08-11 写"一季度计提了 55 亿美元的损失" `[T16]` | 两者不矛盾：**4/15 是预估上限 $5.5B，5/28 是实际 $4.5B**。引用时须分别标注日期与口径。 |

---

## §7 明确"未找到公开数据"的项目（不可编造）

1. **H20 的官方 sparse（2:4 稀疏）算力值**：TF32 / BF16 / FP16 / INT8 的 sparse 值均未被 NVIDIA 公布（`[T4][T5]` 明确标 "not published"）。FP8 sparse = 592 来自第三方数据库 `[T4][T5]`，非官方。
2. **H800 PCIe 80GB 的显存类型**（HBM2e 或 HBM3）：`[N5]` 规格表未标注。
3. **H800 PCIe 94GB（H800 NVL）的显存带宽、显存类型、TDP、Tensor 算力**：仅有 NVIDIA 官方文档确认容量 94GB 与 MIG profile `[N1]`。
4. **H800 的官方 SM 数**：NVIDIA 未公开；`[T26]` 的 132 SM 为第三方数据库推断。
5. **H20 141GB 版本的首发/量产日期**：只能确证"2025 年 Q1 已在渠道流通" `[T22]`。
6. **H800 SXM / H20 的官方 datasheet PDF**：NVIDIA 未在 nvidia.com 产品页公开 H800/H20 的完整规格页；本报告的 H800 SXM 规格依赖渠道商转述的 NVIDIA 规格文本 `[T1][T2][N5]`，H20 规格依赖媒体与第三方数据库 `[T3][T4][T5][T6][T7][T8]`。
7. **H20 的官方 PCIe 带宽明细（128 GB/s）**：按 Hopper SXM 平台通用值引用 `[N5]`，未见 H20 专属官方声明。

---

## §8 来源登记表（URL / 媒体 / 日期 / 官方·第三方）

### NVIDIA 官方

| 代码 | URL | 媒体/机构 | 日期 | 性质 |
|---|---|---|---|---|
| **N1** | https://docs.nvidia.com/ai-enterprise/release-8/latest/infra-software/vgpu/reference/hopper-h800.html | NVIDIA AI Enterprise 官方文档（"Hopper H800 vGPU Types"） | Last updated **2026-09-02** | **NVIDIA官方** |
| **N2** | https://docs.nvidia.com/ai-enterprise/release-8/latest/infra-software/vgpu/reference/hopper-h20.html | NVIDIA AI Enterprise 官方文档（"Hopper H20 vGPU Types"） | Last updated **2026-09-02** | **NVIDIA官方** |
| **N3** | https://www.sec.gov/Archives/edgar/data/1045810/000104581025000082/nvda-20250409.htm | NVIDIA Form 8-K（SEC EDGAR；事件日 2025-04-09，提交日 2025-04-15） | 2025-04-15 | **NVIDIA官方 / 监管文件**（本次检索被 SEC 反爬拦截，内容经 `[T23][T24][T17]` 转述核对） |
| **N4** | https://www.sec.gov/Archives/edgar/data/1045810/000104581025000115/q1fy26pr.htm | NVIDIA Q1 FY2026 财报新闻稿（SEC EDGAR） | 2025-05-28 | **NVIDIA官方**（同上，被拦截） |
| **N5** | https://afox-corp.com/show-118-621-1.html | AFOX（NVIDIA 渠道商）完整转载 **NVIDIA H800 规格页文案与规格表** | 图片资源目录 2023-11-30 | 第三方渠道商 / **内容为 NVIDIA 官方规格表转载** |
| **N6** | https://www.nvidia.com/en-us/data-center/h100/ | NVIDIA H100 产品页（用于 dense 基线对照） | 检索日 2026-09-18 | **NVIDIA官方** |

### 第三方 / 媒体

| 代码 | URL | 媒体 | 日期 | 性质 |
|---|---|---|---|---|
| **T1** | https://www.qiangchuan.com/item/7228.html | 强川科技（渠道商）H800 SXM 参数表，自注数据来源 NVIDIA | 图片目录 2024-01-29 | 第三方/渠道商 |
| **T2** | http://i.cloudhin.com/xk/showproduct.php?id=293 | Cloud Hin 云轩（渠道商），自注"数据来源：英伟达" | 图片目录 2023-03 | 第三方/渠道商 |
| **T3** | https://www.lmsys.org/blog/2025-09-26-sglang-ant-group/ | LMSYS Org / Ant Group 工程博客（H20-96G vs H800-80G 对照表、生产集群实测） | 2025-09-26 | 第三方/产业工程博客 |
| **T4** | https://flopper.io/compare/intel-data-center-gpu-flex-140-12gb-vs-nvidia-h20-96gb | flopper.io GPU 数据库（H20 96GB） | © 2026 | 第三方/数据库 |
| **T5** | https://flopper.io/gpu/nvidia-h20-141gb/spec-sheet | flopper.io GPU 数据库（H20 141GB HBM3e） | 2026-09-18 | 第三方/数据库 |
| **T6** | https://www.ithome.com/0/868/512.htm | IT之家（转 Wccftech/Geekbench） | 2025-07-16 | 第三方/媒体 |
| **T7** | https://m.mydrivers.com/newsview/990598.html | 快科技 mydrivers（转 Wccftech/Geekbench） | 2024-07-10 | 第三方/媒体 |
| **T8** | https://laoyaoba.com/n/889915 | 爱集微（集微网） | 2024-01-02 | 第三方/媒体 |
| **T9** | https://www.eepw.com.cn/zhuanlan/202407/340272.html | EEPW 电子产品世界（转 WCCFtech/EETOP） | 2024-07-11 | 第三方/媒体 |
| **T10** | https://economictimes.indiatimes.com/tech/technology/nvidia-to-launch-china-focused-ai-chip-in-second-quarter-of-2024/printarticle/106630930.cms | **Reuters**（经 Economic Times 转载） | 2024-01-08 | 第三方/**Reuters** |
| **T11** | https://cb.yna.co.kr/gate/big5/m-cn.yna.co.kr/view/AKR20230322067300009 | 韩联社 Yonhap（转 **Reuters 2023-03-21**） | 2023-03-22 | 第三方/Yonhap 转 Reuters |
| **T12** | https://www.trendforce.com/news/2025/07/16/news-nvidia-h20-sales-restart-sparks-buzz-likely-inventory-sell-off-with-new-blackwell-gpu-for-china-expected/ | TrendForce（引 Cailian Press / Reuters 2025-07-15 / Tom's Hardware） | 2025-07-16 | 第三方/TrendForce |
| **T13** | https://www.jetro.go.jp/biznews/2025/08/c62bcbf7f14c3f97.html | JETRO ビジネス短信（引 Reuters 8-12、Bloomberg） | 2025-08-18 | 第三方/JETRO |
| **T14** | https://www.vietnamplus.vn/ft-trung-quoc-yeu-cau-cac-cong-ty-cong-nghe-ngung-mua-chip-ai-cua-nvidia-post1062438.vnp | VietnamPlus（转 **Financial Times** 2025-09-17） | 2025-09-17 | 第三方/转 FT |
| **T15** | https://www.chinadaily.com.cn/a/202507/15/WS6875c252a31000e9a573c180.html | China Daily | 2025-07-15 | 第三方/媒体 |
| **T16** | https://news.qq.com/rain/a/20250811A02O1300 | 腾讯科技（引 FT、DigiTimes） | 2025-08-11 | 第三方/媒体 |
| **T17** | https://www.computerweekly.com/news/366625005/Nvidia-takes-45bn-hit-due-export-restrictions | ComputerWeekly（引 NVIDIA Q1 FY26 财报电话会 / Seeking Alpha 记录） | 2025-05-29 | 第三方/媒体 |
| **T18** | https://www.163.com/dy/article/K93PPN58051481US.html | 观察者网（经网易转载，引 Investing / Bloomberg + NVIDIA CFO） | 2025-09-10 | 第三方/媒体 |
| **T19** | https://www.chinaventure.com.cn/news/114-20250825-387731.html | 投中网 / 虎嗅（引 **The Information**） | 2025-08-25 | 第三方/媒体 / **传闻（The Information）** |
| **T20** | https://gn.china-embassy.gov.cn/fra/fyrth/202508/t20250815_11690499.htm | 中国外交部发言人林剑例行记者会记录 | 2025-08-13 | **中国政府官方** |
| **T21** | https://awtmt.com/articles/3702970 | 腾讯科技 / 全天候科技（H800→H20 降规分析，含 BIS 门槛与 H20 单卡算力 0.148P/0.296P） | 2023-11-28 | 第三方/媒体分析 |
| **T22** | https://finance.jrj.com.cn/2025/03/03073448476126.shtml | 科创板日报（经金融界转载）：H20 141G 渠道现货与报价 | 2025-03-03 | 第三方/媒体 |
| **T23** | https://www.marketwatch.com/story/nvidias-stock-tumbles-as-it-discloses-5-5-billion-in-charges-due-to-china-licensing-rules-16aa01da | MarketWatch | 2025-04-15/16 | 第三方/媒体 |
| **T24** | https://tgam-tgam-prod.web.arc-cdn.net/investing/markets/stocks/AVGO/pressreleases/31906149/nvidia-faces-55b-charge-on-new-h20-export-curbs-to-china-placed-indefinitely/ | Investing.com / GlobeNewswire（标题含 "Nvidia Faces $5.5B Charge … Placed 'Indefinitely'"） | 2025-04 | 第三方/媒体 |
| **T25** | https://raw.githubusercontent.com/llm-infra-atlas/llm-infra-atlas.github.io/refs/heads/main/docs/hpc/00_gpu_hw_params.md | llm-infra-atlas（GitHub 技术手册，dense/sparse 口径与出口 SKU 分析） | 检索日 2026-09-18 | 第三方/技术手册 |
| **T26** | https://www.cpudb.cc/compare/nvidia-h800-sxm5-vs-nvidia-a800-sxm4-80-gb | cpudb.cc（GPU 数据库） | © 2026 | 第三方/数据库（**含向量算力陷阱，见 C6**） |
| **T27** | https://www.sxrsgk.com/qitaxilie/15753.html | 山西润盛（SEO 渠道页） | 2025-09-15 | 第三方/**低可信度**（**仅作冲突反例，不采用其数据**） |
| **T27-A** | https://www.govinfo.gov/content/pkg/CFR-2024-title15-vol2/pdf/CFR-2024-title15-vol2-subtitleB-chapVII-subchapC.pdf | 15 CFR 第 VII 章 C 分章（ECCN 3A090 / TPP 与 performance density 门槛原文） | 2024 版 | **美国政府官方（法规原文）** |
| **T28** | https://ai.zol.com.cn/981/9812766_all.html | 中关村在线 ZOL（转路透） | 2025-05-09 | 第三方/媒体 |
| **T29** | https://www.bbc.com/news/articles/cy8g22n32d0o | BBC News | 2025-07-15 | 第三方/媒体 |

---

## §9 供对比表使用的"干净取数"清单（供与寒武纪 MLU 对齐）

**强烈建议在对比表中使用以下口径**，并在脚注注明 sparse = 2×dense：

| 指标 | H800 SXM5 80GB | H800 PCIe 80GB | H20 SXM5 96GB | H20 SXM5 141GB |
|---|---|---|---|---|
| 架构 / 制程 | Hopper / TSMC 4N | Hopper / TSMC 4N | Hopper / TSMC 4N | Hopper / TSMC 4N |
| BF16/FP16 dense（TFLOPS） | **989** | 756 | **148** | 148 |
| BF16/FP16 sparse（TFLOPS） | **1979** | 1513 | 官方未公布（推算 296） | 同 |
| FP8 dense（TFLOPS） | **1979** | 1513 | **296** | 296 |
| FP8 sparse（TFLOPS） | **3958** | 3026 | 592（第三方） | 592（第三方） |
| INT8 dense（TOPS） | **1979** | 1513 | **296** | 296 |
| INT8 sparse（TOPS） | **3958** | 3026 | 官方未公布 | 同 |
| TF32 dense（TFLOPS） | **494** | 378 | **74** | 74 |
| TF32 sparse（TFLOPS） | **989** | 756 | 官方未公布 | 同 |
| FP32 非 Tensor（TFLOPS） | **67** | 51 | **44** | 44 |
| FP64（TFLOPS） | **1** | 0.8 | **1** | 1 |
| 显存 / 类型 | 80 GB HBM3 | 80 GB（类型未公开） | 96 GB HBM3 | 141 GB HBM3e |
| 显存带宽 | **3.35 TB/s** | 2.0 TB/s | **4.0 TB/s** | **4.0 TB/s** |
| NVLink 双向/GPU | **400 GB/s**（NVLink 4） | **400 GB/s** | **900 GB/s**（NVLink 4） | 900 GB/s |
| PCIe | Gen5 128 GB/s | Gen5 128 GB/s | Gen5（128 GB/s） | 同 |
| TDP | **700 W** | **350 W** | **400 W** | **400 W** |
| 形态 | SXM5 / HGX 8 卡 | PCIe 双槽；另有 94GB H800 NVL | SXM5 / HGX 8 卡 | SXM5 |
| 最大 scale-up 域 | 8 GPU（NVLink Switch System 可至 256） | 1–8 GPU（合作厂商系统） | **8 GPU** | 8 GPU |
