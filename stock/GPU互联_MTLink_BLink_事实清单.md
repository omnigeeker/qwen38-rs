# 摩尔线程 MTLink/KUAE 与壁仞 BLink/BR 系列 互联技术事实清单

> 可信度标注：【官方】= 厂商官网/官方文档/官方新闻稿；【招股书/交易所公告】= 上交所/港交所披露文件；【Hot Chips 论文】= Hot Chips 34 官方幻灯片；【券商研报】；【媒体/自媒体】。不能确证者标注「未证实」或「未找到公开数据」。

---

## A. 摩尔线程（688795）MTLink / MTT S4000 / S5000 / KUAE

### A1. MTLink 规格（SerDes 速率 / lane 数 / 带宽 / 端口数）

| # | 事实 | 来源 URL | 可信度 |
|---|---|---|---|
| A1-1 | MTT S4000 产品规格书对 MTLink 的原文描述为「**MTLink：x8 Serdes in the gold finger，Speed: up to 56Gbps PAM4**」，即 8 lane SerDes、56Gbps PAM4、位于金手指。 | https://docs.mthreads.com/s4000/s4000-doc-online/product_specifications/ | 【官方】 |
| A1-2 | 招股说明书芯片规格表载明「**片间互连带宽**」：**平湖（S5000）800 GB/s、曲院（S4000）240 GB/s**，春晓/苏堤为 NA。 | https://static.sse.com.cn/stock/disclosure/announcement/c/202509/002098_20250905_YN3R.pdf | 【招股书/交易所公告】 |
| A1-3 | 招股说明书智算集群对比表列示 KUAE2 的「**卡间互连（GB/s）~800**」，同表 NVIDIA H100 集群口径为 900 GB/s（注明「设计性能」）。 | https://static.sse.com.cn/stock/disclosure/announcement/c/202509/002098_20250905_YN3R.pdf | 【招股书/交易所公告】 |
| A1-4 | 招股说明书 KUAE2 规格表另列「**单 GPU 服务器互联 400 Gb/s per GPU**、**单 GPU 互联 800 GB/s**」。 | https://static.sse.com.cn/stock/disclosure/announcement/c/202509/002098_20250905_YN3R.pdf | 【招股书/交易所公告】 |
| A1-5 | 官方尚无公开的 MTLink **端口数（port 数）**、单向带宽拆分口径；MTLink 在 S4000 上通过金手指 + **MTLink Bridge** 桥接相邻卡实现（说明书要求先移除 MTLink Cover，再插入 MTLink Bridge 对准金手指）。 | https://docs.mthreads.com/s4000/s4000-doc-online/product_manual/ | 【官方】 |
| A1-6 | 需核实：x8 lane × 56Gbps PAM4 的物理容量（单向 448Gbps≈56GB/s，双向≈112GB/s）与招股书「240 GB/s」口径不一致，公开资料未给出 240/800 GB/s 的 lane/端口分解 → **端口数与单向/双向拆分口径：未找到公开数据**。 | — | 未证实 |
| A1-7 | 招股说明书称「**MT-Link 3.0 协议接近国外同代系 GPU 节点内传输速率**」，为官方文件中对 MTLink 版本号最明确的一次表述。 | https://static.sse.com.cn/stock/disclosure/announcement/c/202509/002098_20250905_YN3R.pdf | 【招股书/交易所公告】 |
| A1-8 | 2025-12-20 MUSA 开发者大会披露新一代「**MTLink 4.0**」：搭载 MTLink 4.0 + 多种类以太协议，**片间互联速度达 134.5 Gb/s**，支持扩展至 1024 GPUs，支持 SHARP。 | https://www.c114.net.cn/industry/46397.html | 【媒体/自媒体】（转述官方幻灯片） |
| A1-9 | 招股说明书将「MTLINK」列入公司自研完成的 GPU IP 清单（GPUIP、VID、DE、NOC、MTLINK 等）。 | https://static.sse.com.cn/stock/disclosure/announcement/c/202509/002098_20250905_YN3R.pdf | 【招股书/交易所公告】 |

### A2. MTT S4000 互联与 PCIe

| # | 事实 | 来源 URL | 可信度 |
|---|---|---|---|
| A2-1 | MTT S4000 官方规格：PCIe 总线接口 **PCIe 5.0 x16**；MTLink x8 SerDes in the gold finger，up to 56Gbps PAM4；48GB GDDR6、768 GB/s、450W。 | https://docs.mthreads.com/s4000/s4000-doc-online/product_specifications/ | 【官方】 |
| A2-2 | 招股说明书：**曲院（S4000）片间互连带宽 240 GB/s**，PCIe 5.0，最大显存 48GB，显存带宽 768 GB/s。 | https://static.sse.com.cn/stock/disclosure/announcement/c/202509/002098_20250905_YN3R.pdf | 【招股书/交易所公告】 |
| A2-3 | KUAE1 为「支持千卡互联的第一代超大规模智算融合中心产品」，由 MTT S4000（2023 年底推出）承载。 | https://static.sse.com.cn/stock/disclosure/announcement/c/202509/002098_20250905_YN3R.pdf | 【招股书/交易所公告】 |
| A2-4 | 「8 卡全互联、每卡多少 GB/s」在 S4000 上**未找到官方直接披露**；招股书仅给出「片间互连带宽 240 GB/s」这一整卡口径。 | — | 未找到公开数据 |
| A2-5 | MCCX D800 X1 官方/招股书规格：4U 服务器、**8 × MTT S4000，支持 PCIe Gen5**、1TB DDR5、2×200Gb InfiniBand NDR/Ethernet 网卡。 | https://static.sse.com.cn/stock/disclosure/announcement/c/202509/002098_20250905_YN3R.pdf | 【招股书/交易所公告】 |

### A3. MTT S5000 / KUAE 2.0

| # | 事实 | 来源 URL | 可信度 |
|---|---|---|---|
| A3-1 | 官方 S5000 页面：「**8 颗 MTT S5000 OAM 计算模组通过 MTLink 高速互联**」，面向大模型训练、推理与科学计算。 | https://www.mthreads.com/product/S5000 | 【官方】 |
| A3-2 | 招股说明书：**平湖（S5000）片间互连带宽 800 GB/s**，PCIe 5.0，最大显存 80GB；KUAE2 规格表「单 GPU 互联 800 GB/s」。 | https://static.sse.com.cn/stock/disclosure/announcement/c/202509/002098_20250905_YN3R.pdf | 【招股书/交易所公告】 |
| A3-3 | 财报/媒体口径：MTT S5000 单卡 AI 算力最高 **1000 TFLOPS**、80GB 显存、显存带宽 **1.6TB/s**、**卡间互联带宽 784GB/s**，支持 FP8 到 FP64。 | https://finance.cnr.cn/ycbd/20260227/t20260227_527537765.shtml | 【媒体/自媒体】（转述业绩公告与官方参数） |
| A3-4 | 784 GB/s（媒体）与 800 GB/s（招股书）口径略有差异，两者均系公开来源，未获官方统一说明。 | 同上 + 招股书 | 【媒体/自媒体】+【招股书/交易所公告】 |
| A3-5 | 量产/节点规模：**MTT S5000 于 2025 年实现规模化量产**；官方称实现「从单卡到万卡集群的线性性能跃升」，万卡集群扩展线性度达 95%。 | https://finance.cnr.cn/ycbd/20260227/t20260227_527537765.shtml ；https://www.mthreads.com/product/S5000 | 【媒体/自媒体】+【官方】 |
| A3-6 | 是否支持更多卡全互联：**KUAE2「支持万卡互联」**；下一代「华山」芯片配 MTLink 4.0，**单节点最多 1024 卡**，可支撑十万卡以上集群。 | https://static.sse.com.cn/stock/disclosure/announcement/c/202509/002098_20250905_YN3R.pdf ；https://m.21jingji.com/article/20251220/herald/4d2c3d7f86c81484c30d5197f0b1989a_zaker.html | 【招股书/交易所公告】+【媒体/自媒体】 |
| A3-7 | 招股说明书产品代际：S5000 = 第四代「平湖」架构（PH100 芯片），S4000 = 第三代「曲院」架构。 | https://static.sse.com.cn/stock/disclosure/announcement/c/202509/002098_20250905_YN3R.pdf | 【招股书/交易所公告】 |

### A4. KUAE 集群拓扑

| # | 事实 | 来源 URL | 可信度 |
|---|---|---|---|
| A4-1 | 招股说明书 KUAE2 规格表：**GPU 10,240 个**、GPU 显存 800TB、CPU 核心 81,920、**计算网络 IB / RoCE v2（10,240 个端口）**、单 GPU 服务器互联 400 Gb/s per GPU、单 GPU 互联 800 GB/s、训练数据缓存 2TB/s。 | https://static.sse.com.cn/stock/disclosure/announcement/c/202509/002098_20250905_YN3R.pdf | 【招股书/交易所公告】 |
| A4-2 | 招股说明书：KUAE 单集群可部署**超 1,000 个计算节点，每节点集成 8 颗自研 OAM 模组化 GPU，通过 3D 全互联拓扑实现亚微秒级通信延迟**。 | https://static.sse.com.cn/stock/disclosure/announcement/c/202509/002098_20250905_YN3R.pdf | 【招股书/交易所公告】 |
| A4-3 | 机间网络为**商用 InfiniBand/RoCE v2**（非自研私有网络）；官方同时称「创新应用硅光互联技术与自适应路由协议，支持无损 RDMA/IB 传输」。 | https://static.sse.com.cn/stock/disclosure/announcement/c/202509/002098_20250905_YN3R.pdf | 【招股书/交易所公告】 |
| A4-4 | 早期「夸娥」千卡集群节点（MCCX D800）实测配置：每节点 8× MTT S4000、双路 Intel 四代至强、16×64GB 内存、4×3.84TB NVMe、**双路 400Gb IB + 四路 25Gb 以太网**。 | https://m.mydrivers.com/newsview/971986.html | 【媒体/自媒体】 |
| A4-5 | 单集群规模：**KUAE1 支持千卡互联；KUAE2 于 2024 年底推出，支持万卡互联**；夸娥万卡集群浮点算力 10 Exa-FLOPS，Dense 大模型 MFU 60%、MoE MFU 40%、有效训练时间占比 >90%、训练线性扩展效率 95%。 | https://static.sse.com.cn/stock/disclosure/announcement/c/202509/002098_20250905_YN3R.pdf ；https://m.21jingji.com/article/20251220/herald/4d2c3d7f86c81484c30d5197f0b1989a_zaker.html | 【招股书/交易所公告】+【媒体/自媒体】 |
| A4-6 | **OISA 高密超节点参考设计（摩尔线程 + 中国移动研究院 + 之江实验室等，2026-03）**：在主流 32–64 卡互联基础上，实现标准单宽机柜内 **128 卡全互联**，并支持并柜扩展至 **256 卡**。 | https://www.stcn.com/article/detail/3679504.html | 【媒体/自媒体】（转述联合发布的技术规范） |
| A4-7 | 招股说明书：公司研发基于光电混合架构的高速互联技术，旨在实现**单网络支持 10 万卡级 GPU 集群**。 | https://static.sse.com.cn/stock/disclosure/announcement/c/202509/002098_20250905_YN3R.pdf | 【招股书/交易所公告】 |

### A5. 招股书/年报官方原文摘录

| # | 原文摘录（《首次公开发行股票并在科创板上市招股说明书（申报稿）》，2025-09-05 披露） | 来源 URL | 可信度 |
|---|---|---|---|
| A5-1 | 「集群层面，夸娥（KUAE）智算集群，**可扩展至万卡规模**，采用先进网络架构和调度系统……」 | https://static.sse.com.cn/stock/disclosure/announcement/c/202509/002098_20250905_YN3R.pdf | 【招股书/交易所公告】 |
| A5-2 | 「公司代表性产品包括：MTT S4000，系 2023 年底推出的训推一体全功能智算卡；MTT S5000，通过 FP8 精度支持等创新提升性能；**KUAE1，系支持千卡互联的第一代超大规模智算融合中心产品；KUAE2，系 2024 年底推出的第二代超大规模智算融合中心产品，支持万卡互联。**」 | https://static.sse.com.cn/stock/disclosure/announcement/c/202509/002098_20250905_YN3R.pdf | 【招股书/交易所公告】 |
| A5-3 | 「单集群可部署超 1,000 个计算节点，**每节点集成 8 颗自研 OAM 模组化（OCP Accelerator Module）GPU，通过 3D 全互联拓扑实现亚微秒级通信延迟**……**MT-Link 3.0 协议接近国外同代系 GPU 节点内传输速率。**」 | https://static.sse.com.cn/stock/disclosure/announcement/c/202509/002098_20250905_YN3R.pdf | 【招股书/交易所公告】 |
| A5-4 | 芯片规格表：「**片间互连带宽：平湖 800 GB/s；曲院 240 GB/s；春晓 NA；苏堤 NA**」。 | https://static.sse.com.cn/stock/disclosure/announcement/c/202509/002098_20250905_YN3R.pdf | 【招股书/交易所公告】 |
| A5-5 | 「公司积极参与行业标准的制定，如**与中国移动合作主导 OISA（Omni-directional Intelligent Sensing Express Architecture，全向智感互联）GPU 卡间互联协议体系标准**的制定和研发。」 | https://static.sse.com.cn/stock/disclosure/announcement/c/202509/002098_20250905_YN3R.pdf | 【招股书/交易所公告】 |
| A5-6 | 「公司采用 Chiplet 技术，通过先进封装与小芯粒设计，实现了多芯片模块（MCM）架构……结合高速 I/O 接口和优化的片间通信协议，公司进一步提升了高性能数据传输率并降低通信延迟。」 | https://static.sse.com.cn/stock/disclosure/announcement/c/202509/002098_20250905_YN3R.pdf | 【招股书/交易所公告】 |
| A5-7 | 注：本次招股说明书为**申报稿**（2025-09-05）；另有 2025-11-13 招股意向书版本（含 2025 年数据），本次研究未逐页核对意向书文本。 | https://wap.stockstar.com/detail/SN2025111300039492 | 【招股书/交易所公告】 |

### A6. MTLink 是否支持内存语义 / 一致性

| # | 事实 | 来源 URL | 可信度 |
|---|---|---|---|
| A6-1 | **MTLink 本身是否支持内存语义/缓存一致性：招股说明书与 mthreads.com 官方文档均未作出此类表述 → 未找到公开数据。** | https://static.sse.com.cn/stock/disclosure/announcement/c/202509/002098_20250905_YN3R.pdf | 未找到公开数据 |
| A6-2 | 与 MTLink 同属卡间互联体系的 **OISA 协议：OISA 2.0 版本「原生内存语义支持」，实现跨节点无障碍数据访问，卡间带宽推向 TB/s 级别，时延缩短至数百纳秒**（摩尔线程为 OISA 主导厂商之一）。 | https://www.stcn.com/article/detail/3679504.html | 【媒体/自媒体】（转述联合发布的技术规范） |
| A6-3 | 招股说明书仅说明「长江」SoC 基于**统一内存架构（UMA）**，未将其与 MTLink 关联；不能据此推断 MTLink 的内存语义能力。 | https://static.sse.com.cn/stock/disclosure/announcement/c/202509/002098_20250905_YN3R.pdf | 【招股书/交易所公告】 |

---

## B. 壁仞科技（Biren，港股 6082.HK）BR100 / BR104 / 壁砺 106 系列

### B1. BLink 规格

| # | 事实 | 来源 URL | 可信度 |
|---|---|---|---|
| B1-1 | 港交所上市文件原文（英文）：「we developed our proprietary **BLink** system … our BLink technology enables connections between GPU cards, with a **maximum bidirectional data transfer rate up to 64GB/s per link and 4-8 links in total**」。 | https://www1.hkexnews.hk/listedco/listconews/sehk/2026/0102/11974288/sehk25121700480.pdf | 【招股书/交易所公告】 |
| B1-2 | 港交所上市文件中文版原文：「……最大雙向數據傳輸速率**高達每通道 64GB/s，共 4 至 8 條通道**。」 | https://www1.hkexnews.hk/listedco/listconews/sehk/2026/0102/11974286/sehk25121700449_c.pdf | 【招股书/交易所公告】 |
| B1-3 | Hot Chips 34 官方幻灯片 BR100 规格行：「**Interconnections 8 BLink™**」，外部 I/O 带宽 2.3TB/s。 | https://www.hc34.hotchips.org/assets/program/conference/day1/GPU%20HPC/HC2022.BirenTech.MikeHong.LingjieXu.v01.pdf | 【Hot Chips 论文】 |
| B1-4 | 壁仞 CTO 洪洲发布会原话：「我们**有 8 组 BLink 接口，其中 7 个接口可以连接另外 7 个 GPU**，最终将 8 个 GPU 有效连在一起」「通过这项技术，把 8 个 GPU 当成 1 个 GPU 来用」。 | https://www.pingwest.com/w/268809 | 【媒体/自媒体】（发布会现场引述高管） |
| B1-5 | 单卡互联带宽（8 卡节点）：壁仞联合创始人/总裁徐凌杰称「**节点内每张卡能够有 448GB/s 的互连带宽，节点之外还有 64GB 的带宽。8 卡对分带宽 1.8TB/s**」。——448 GB/s = 7 条 × 64 GB/s，与 B1-1/B1-3/B1-4 三项互相自洽。 | https://www.pingwest.com/w/268809 | 【媒体/自媒体】（发布会现场引述高管） |
| B1-6 | BLink 单向带宽、BLink 自身 SerDes 速率（Gbps/lane）与 lane 数：**公开资料未披露**（Hot Chips 仅披露 die-to-die 采用 112G PAM4 SerDes）→ 未找到公开数据。 | — | 未找到公开数据 |
| B1-7 | 日本 EE Times 报道曾给出「每芯片 BLink 8 端口、芯片间带宽 412GB/s」的同代数字（与 448 GB/s 接近，疑为传抄差异），本清单以 B1-1/B1-5 为准。 | https://eetimes.itmedia.co.jp/ee/articles/2209/02/news066_2.html | 【媒体/自媒体】 |

### B2. BR100 单机 8 卡全互联实现 与 Hot Chips 34 数据

| # | 事实 | 来源 URL | 可信度 |
|---|---|---|---|
| B2-1 | Hot Chips 34 结论页原文：「**7nm chiplet design with CoWoS packaging**」「**550W OAM form factor with 8 cards all-to-all interconnection topology**」「Over 300MB on-chip SRAM」。 | https://www.hc34.hotchips.org/assets/program/conference/day1/GPU%20HPC/HC2022.BirenTech.MikeHong.LingjieXu.v01.pdf | 【Hot Chips 论文】 |
| B2-2 | Hot Chips 34 BR100 规格页：面积 **1074mm² @7nm**、**770 亿晶体管**、Host Interface **PCIe Gen 5 x16 w/ CXL**、INT8 2048 TOPS、BF16 1024 TFLOPS、TF32+ 512 TFLOPS、FP32 256 TFLOPS、**64GB HBM2E**、**8 BLink**、外部 I/O 带宽 2.3TB/s。 | https://www.hc34.hotchips.org/assets/program/conference/day1/GPU%20HPC/HC2022.BirenTech.MikeHong.LingjieXu.v01.pdf | 【Hot Chips 论文】 |
| B2-3 | Hot Chips 34「One Tapeout, Multiple Products」页：**896GB/s high speed die-to-die interconnect**；双计算 die + HBM2E 布局；相比单片设计性能 +30%、良率 +20%。 | https://www.hc34.hotchips.org/assets/program/conference/day1/GPU%20HPC/HC2022.BirenTech.MikeHong.LingjieXu.v01.pdf | 【Hot Chips 论文】 |
| B2-4 | **8 卡全互联的具体实现**：把 8 个 OAM 模组放在**通用 UBB 主板**上形成 8 卡点对点全互连拓扑，利用 BR100 上的高速接口与 UBB 互连基础设施实现 8 张卡两两互连；被称为「国内芯片设计厂商中第一个实现在 OAM 系统中单节点 8 卡全互连拓扑」。 | https://www.pingwest.com/w/268809 | 【媒体/自媒体】（发布会现场引述高管） |
| B2-5 | **片内两 die 的互联不是「桥接 die」，而是 CoWoS-S 硅中介层**：BR100 明确采用台积电 2.5D CoWoS-S，两片 die 与周边 HBM2E 放在同一硅中介层上，die-to-die 采用超高速 **112G PAM4 SerDes**，die 间带宽 **896GB/s**。 | https://www.pingwest.com/w/268809 | 【媒体/自媒体】（发布会现场引述高管，与 HC34 幻灯片一致） |
| B2-6 | 8 卡服务器「海玄」（与浪潮合作）：BF16 峰值 8 PFLOPS、512GB HBM2e、支持 PCIe 5.0 与 CXL、**1.8TB/s 对分互连带宽**、最大功耗 7kW。 | https://www.pingwest.com/w/268809 | 【媒体/自媒体】 |

### B3. BR104 与 壁砺 106 / 166 系列的互联规格与差异

| # | 事实 | 来源 URL | 可信度 |
|---|---|---|---|
| B3-1 | BR104 = **单片 die 版本**，算力与 IO 等参数大多为 BR100 的一半，形态为 PCIe 板卡（壁砺 104，300W）。 | https://www.pingwest.com/w/268809 | 【媒体/自媒体】 |
| B3-2 | 壁砺 104（BR104）专为多卡高速互连**设计了高速桥片**，可在 **4 张卡之间形成点对点全互连拓扑，带宽 192GB/s**（= 3 × 64 GB/s）。 | https://www.pingwest.com/w/268809 | 【媒体/自媒体】（发布会现场引述高管） |
| B3-3 | 港交所上市文件的量产产品命名体系为 **BR106 / BR110 / BR166**（未出现 BR100/BR104 字样）：BR106 于 **2023 年 1 月量产**，BR110 于 **2024 年 10 月量产**。 | https://www1.hkexnews.hk/listedco/listconews/sehk/2026/0102/11974288/sehk25121700480.pdf | 【招股书/交易所公告】 |
| B3-4 | BR106 提供四个型号：**BILI 106M（风冷 OAM）、BILI 106L（液冷 OAM）、BILI 106B（2-slot FHFL PCIe 卡）、BILI 106C（PCIe 卡，主打推理）**。 | https://www1.hkexnews.hk/listedco/listconews/sehk/2026/0102/11974288/sehk25121700480.pdf | 【招股书/交易所公告】 |
| B3-5 | 壁砺 106M 官方页面：**风冷 OAM 模组，峰值功耗 400W**，强调「强大的算力和高速互连能力」。 | https://www.birentech.com/product/hardware/106m/ | 【官方】 |
| B3-6 | 壁砺 106B 官方页面：**全高全长、双宽 PCIe 板卡，峰值功耗 300W**。 | https://www.birentech.com/product/hardware/106b/ | 【官方】 |
| B3-7 | 第三方平台文档：壁砺 106M 为 OAM 模组，PCIe 5.0 x8 主机接口，**通过 UBB 主板实现单机 8 卡互联，每张卡支持 4 端口共 256GB/s 双向互联带宽**；32GB HBM2E、819GB/s。 | https://ai.gitee.com/docs/compute/clusters_gpu/biren_gpu | 【媒体/自媒体】（平台厂商文档） |
| B3-8 | BR166（双 BR106 die + 四颗 DRAM 共封装于单一封装）在**峰值算力、内存、视频编解码、互连**等方面性能为 BR106 的**两倍**；两 die 间 **D2D 双向带宽最高 896GB/s**。 | https://www1.hkexnews.hk/listedco/listconews/sehk/2026/0102/11974288/sehk25121700480.pdf | 【招股书/交易所公告】 |
| B3-9 | 壁砺 166 型号：**BILI 166M（风冷 OAM）、166L（液冷 OAM）、166C（2-slot FHFL PCIe）**；**166L/166M 于 2025 年 8 月量产，166C 于 2025 年 12 月量产**。 | https://www1.hkexnews.hk/listedco/listconews/sehk/2026/0102/11974288/sehk25121700480.pdf | 【招股书/交易所公告】 |
| B3-10 | 106M/L 与 106B/C 的互联差异核心在形态：仅 OAM（106M/106L）经 UBB 支持单机 8 卡全互连；PCIe 卡（106B/106C）官网未披露 BLink 端口/带宽。 | https://www.birentech.com/product/hardware/106b/ ；https://ai.gitee.com/docs/compute/clusters_gpu/biren_gpu | 【官方】+【媒体/自媒体】 |
| B3-11 | BR100（2022 发布名）与 BR166（量产名）之间的官方映射关系**未见壁仞或港交所文件明确说明** → 属推断，未证实。 | — | 未证实 |

### B4. 壁仞招股书（港交所上市文件）关于 BLink 的官方原文摘录

| # | 原文摘录 | 来源 URL | 可信度 |
|---|---|---|---|
| B4-1 | 英文版原文：「**Multi-GPU Interconnections: We integrated high-speed Serializer/Deserializer (SerDes)** that enables high-speed communications while minimizing the number of input/output pins and interconnections. We also developed our **proprietary BLink system** that improves the scalability of intelligent computing clusters. Traditional GPU cards can only connect with server hosts, but our **BLink technology enables connections between GPU cards**, with a **maximum bidirectional data transfer rate up to 64GB/s per link and 4-8 links in total**. In addition, we are **the first GPGPU company in China to achieve point-to-point full mesh topology for eight GPU cards in a single server, according to CIC**.」 | https://www1.hkexnews.hk/listedco/listconews/sehk/2026/0102/11974288/sehk25121700480.pdf | 【招股书/交易所公告】 |
| B4-2 | 中文版原文：「**多 GPU 互連**：我們集成了高速串行╱解串器（SerDes），可實現高速通信，同時最大限度地減少輸入╱輸出接腳和互連的數量。我們還開發了專有的 **BLink 系統**，提高了智能計算集群的可擴展性。傳統 GPU 卡只能與服務器主機連接，但我們的 BLink 技術可以實現 GPU 卡之間的連接，**最大雙向數據傳輸速率高達每通道 64GB/s，共 4 至 8 條通道**。此外，根據灼識諮詢的資料，我們是**中國首家在單一服務器中實現八塊 GPU 卡點對點全網狀拓撲的 GPGPU 公司**。」 | https://www1.hkexnews.hk/listedco/listconews/sehk/2026/0102/11974286/sehk25121700449_c.pdf | 【招股书/交易所公告】 |
| B4-3 | 上市文件其他相关原文：「**Multi-GPU Interconnections** … our UBB can connect **up to eight OAM cards with multiple topologies using our in house P2P interface**. Next generation, we will design more flexible and powerful **SerDes connection to scale up our system**.」 | https://www1.hkexnews.hk/listedco/listconews/sehk/2026/0102/11974288/sehk25121700480.pdf | 【招股书/交易所公告】 |
| B4-4 | 上市文件：「communication library … accelerate data transfer across diverse network topologies, including our **BLink intra-node/super-node channels** and large scale GPU clusters.」 | https://www1.hkexnews.hk/listedco/listconews/sehk/2026/0102/11974288/sehk25121700480.pdf | 【招股书/交易所公告】 |
| B4-5 | 上市文件未披露 BLink 的 SerDes 速率、lane 数、端口带宽明细，亦未使用「NVLink」或「内存语义」等表述来描述 BLink 1.0。 | https://www1.hkexnews.hk/listedco/listconews/sehk/2026/0102/11974288/sehk25121700480.pdf | 【招股书/交易所公告】 |

### B5. BLink 是否支持内存语义 / 是否对标 NVLink

| # | 事实 | 来源 URL | 可信度 |
|---|---|---|---|
| B5-1 | **BLink 2.0 官方明确定义为「内存语义互连」**：壁仞官方新闻稿称 BLink™2.0 四大核心能力之一即「**内存语义互连，让最多 1024 张 GPU 共享同一个内存空间，多卡如同一台"超级 GPU"**」。 | https://www.birentech.com/news/odug5ugc29npl8m6slum8d9k/ | 【官方】 |
| B5-2 | BLink 2.0 其余三大能力：**在网计算（通信运算下沉到交换机）、智能拥塞控制、多层链路自愈**；基于 BR2xx + BLink 2.0 构建三级超节点产品矩阵：**16 卡（电互连）/ 128 卡整机柜（电互连）/ 1024 卡分布式解耦（NPO 光互连）**。 | https://www.birentech.com/news/odug5ugc29npl8m6slum8d9k/ | 【官方】 |
| B5-3 | **BLink 1.0（BR100/BR106 世代）是否支持内存语义/缓存一致性：港交所上市文件与壁仞官网均未作出此类表述 → 未找到公开数据。** | https://www1.hkexnews.hk/listedco/listconews/sehk/2026/0102/11974288/sehk25121700480.pdf | 未找到公开数据 |
| B5-4 | **对标 NVLink**：媒体在发布会报道中称 BLink「在壁仞科技的版图中应该也是很重要的——**可类比于英伟达的 NVLink**」；壁仞官方文件未使用「对标 NVLink」措辞。 | https://www.pingwest.com/w/268809 | 【媒体/自媒体】 |
| B5-5 | 壁仞官方称「把 8 个 GPU 当成 1 个 GPU 来用，并且把数据多播、计算核之间同步都通过接口来实现」（CTO 洪洲，发布会）。 | https://www.pingwest.com/w/268809 | 【媒体/自媒体】（引述高管） |

### B6. 量产时间 / 代工厂 / 封装方式

| # | 事实 | 来源 URL | 可信度 |
|---|---|---|---|
| B6-1 | 量产时间表（港交所上市文件）：**BR106 于 2023 年 1 月量产；BR110 于 2024 年 10 月量产；壁砺 166L/166M 于 2025 年 8 月量产，166C 于 2025 年 12 月量产。** | https://www1.hkexnews.hk/listedco/listconews/sehk/2026/0102/11974288/sehk25121700480.pdf | 【招股书/交易所公告】 |
| B6-2 | 工艺与封装：Hot Chips 34 官方幻灯片载「**Area 1074mm² @7nm**」「**7nm chiplet design with CoWoS packaging**」。 | https://www.hc34.hotchips.org/assets/program/conference/day1/GPU%20HPC/HC2022.BirenTech.MikeHong.LingjieXu.v01.pdf | 【Hot Chips 论文】 |
| B6-3 | 港交所上市文件：「According to CIC, we are the **first company in China to package dual AI computing dies using 2.5D chiplet technology**」「we adopt **chiplet technology** in our chip design methodology … involved advanced chip packaging technologies … stacking and connecting multiple dies (such as GPU SoCs, memory, etc.) into a single package」。 | https://www1.hkexnews.hk/listedco/listconews/sehk/2026/0102/11974288/sehk25121700480.pdf | 【招股书/交易所公告】 |
| B6-4 | 代工厂（foundry）名称：**港交所上市文件未披露具体代工厂商**（仅提及「foundries」「IC packaging and testing suppliers」，并有章节讨论出口管制对供应链的影响）→ 代工厂未获官方确认。 | https://www1.hkexnews.hk/listedco/listconews/sehk/2026/0102/11974288/sehk25121700480.pdf | 未找到公开数据 |
| B6-5 | 台积电 2.5D CoWoS-S 封装：媒体在 2022-08-09 发布会报道中称 BR100「明确采用了**台积电的 2.5D CoWoS-S 封装方案**——两片 die 和周边 HBM2e 内存都放在一片硅中介层上」。 | https://www.pingwest.com/w/268809 | 【媒体/自媒体】 |
| B6-6 | 7nm 工艺、770 亿晶体管、die size 约 1000mm²（突破 reticle limit，故将两片 die 封装到一起）由壁仞 CTO 在发布会确认，与 HC34 幻灯片一致。 | https://www.pingwest.com/w/268809 | 【媒体/自媒体】（引述高管） |
| B6-7 | BR100 2022-08 发布时状态为「**正在测试中**」（未量产）；2023 年 1 月量产的是 BR106。 | https://www.expreview.com/84428.html | 【媒体/自媒体】 |
| B6-8 | 南京 1024-GPU 智算集群于 2024 年 9 月交付（与 IT 合作伙伴合作），可作为壁仞大规模互联商用落地的官方佐证。 | https://www1.hkexnews.hk/listedco/listconews/sehk/2026/0102/11974288/sehk25121700480.pdf | 【招股书/交易所公告】 |

---

## 附：明确「未找到公开数据 / 未证实」条目汇总

1. MTLink 的**端口数（port 数）**、**单向/双向带宽拆分口径**：未找到公开数据（官方仅给出整卡「片间互连带宽」240/800 GB/s 与 x8 SerDes 56Gbps PAM4）。
2. MTT S4000 的**「8 卡全互联、每卡 GB/s」**官方口径：未找到公开数据（招股书仅披露曲院片间互连带宽 240 GB/s）。
3. **MTLink 是否支持内存语义/缓存一致性**：官方文件无表述 → 未找到公开数据（同体系的 OISA 2.0 官方联合规范称「原生内存语义支持」）。
4. **BLink 的 SerDes 速率（Gbps/lane）、lane 数、单向带宽**：未找到公开数据（仅披露 64GB/s/链路双向、4–8 链路）。
5. **BLink 1.0 是否支持内存语义/一致性**：未找到公开数据（BLink 2.0 官方明确为内存语义互连）。
6. **壁仞的代工厂**：港交所上市文件未披露；台积电 CoWoS-S 仅见媒体转述 → 官方未证实。
7. **BR100 ↔ BR166 / BR104 ↔ BR106 的官方命名映射**：未证实（属推断）。
8. 壁砺 106B/106C 的 **BLink 端口与带宽**：官方页面未披露。
