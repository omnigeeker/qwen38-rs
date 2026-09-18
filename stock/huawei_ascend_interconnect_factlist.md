# 华为昇腾 910B / 910C / 910D：芯片级与系统级互联技术——结构化事实清单

证据分级：【官方】（huawei.com / hiascend.com / 华为署名论文 / 官方白皮书）＞【第三方机构/官方论文】（NVIDIA 官方页、TechInsights、SemiAnalysis、arXiv 华为署名论文、超节点白皮书）＞【券商研报】＞【媒体】＞【自媒体/未证实】
说明：无法确证的一律标注「未证实」；拿不到的一律写「未找到公开数据」；页面打不开的写「页面不可访问」。
本清单中 910C 单卡互联带宽存在 **392GB/s（单向）** 与 **784GB/s（整封装）** 两种口径，两处均已并列保留，未做合并。

---

## 一、HCCS（Huawei Cache Coherence System）具体规格

### 1.1 910B 单卡 HCCS 带宽与拓扑

- 【官方论文·旁证】华为+硅基流动论文（arXiv 2506.12708，通讯作者为 @huawei.com / @hisilicon.com）明确："All reported network bandwidth values denote unidirectional bandwidth"（全文带宽均为单向口径）；该论文的 Ascend 910 每 die 集成 **7 个高速收发器**接 scale-up 平面。https://ar5iv.labs.arxiv.org/html/2506.12708
- 【媒体·技术分析】910B（8 卡模块）HCCS 总带宽 **392GB/s**，与 A800 NVLink（400GB/s）相当；HCCS 采用**对等（point-to-point）拓扑、无 NVSwitch 类交换芯片**，因此**卡对卡（双向）最大带宽仅 56GB/s**。https://cloud.tencent.cn/developer/article/2473240
- 【第三方社区·鲲鹏昇腾开发者社区】对照表列"H100 NVLink 900GB/s（原文写 900GB）vs 910B/910C HCCS **392GB**"。https://hwcomputing.csdn.net/6a6b1d4f662f9a54cb964ca8.html
- 【官方（页面为 JS 渲染，内容不可读）】华为开发者论坛帖《Ascend 800I A2 2卡间带宽是多少？》——**页面不可访问**（正文由 JS 动态加载，抓取仅得站点框架）。https://developer.huawei.com/home/forum/ascend/thread-02201224295121576526-1-1.html
- 【官方（页面为 JS 渲染，内容不可读）】hiascend.com《Atlas 800T A2 训练服务器》产品页与《Atlas 800T A2 训练服务器 技术白皮书 08》——**页面不可访问**（只返回站点框架，未取到规格表正文）。https://www.hiascend.com/hardware/ai-server?tag=900A2 ｜ https://e.huawei.com/cn/documents/products/computing/8b1341dda54e434284231f905e37c629
- 【官方·间接】华为企业业务文档（Atlas 800T A3 相关彩页摘要）出现描述："超节点架构，节点内全交换无收敛全互联；8 路 NPU 通过总线互联，双向互联带宽达更大；8*400GE RoCE v2 高速接口"——**未给出具体 GB/s 数值，页面正文不可完整读取**。https://e.huawei.com/cn/documents/products/computing/c83a5ba0fd954d238a1e6962d8816620

### 1.2 端口数与 SerDes 速率 / lane 数（关键缺口）

- **910B 的 HCCS 端口数、SerDes 速率（Gbps/lane）、lane 数：未找到公开数据（官方未公布）。** 华为从未发布 910B 的 HCCS 端口/SerDes 规格表；上述 392GB/s、56GB/s 均为第三方/媒体口径。
- 【未证实·第三方拆解帖，仅作线索】**未在权威来源中找到支撑**。"每卡 7 个 HCCS 端口"这一说法与下述官方论文中"**每个 die 集成 7 个高速收发器到 scale-up 平面**"数字吻合，但**官方论文并未把该 7 个收发器称为 HCCS**，也未给出单端口速率。https://ar5iv.labs.arxiv.org/html/2506.12708
- 【官方·下一代可作对照（不是 910B）】昇腾 950 官方白皮书给出代际 SerDes 规格：整芯片集成 **72 Lane HiLink SerDes，划分为 18 个 x4 端口，每端口最高 4×112Gbps，整芯片对外 IO 峰值 2TB/s**；UB 协议版本 Unified Bus 2.0，UB IO 带宽 **2016GB/s（双向）**。这是官方唯一公开的昇腾互联 SerDes/端口级规格，可用于推断华为互联 SerDes 的代际量级，**但不可直接套用到 910B**。https://public-download.obs.cn-east-2.myhuaweicloud.com/ascend/昇腾950%20NPU架构白皮书.pdf

### 1.3 910C 的 HCCS 带宽：是否与 910B 相同

- 【官方·HC2025 高管演讲转述，最硬的"官方数字"】徐直军 2025-09-18 华为全联接大会公布的 910C 规格："910C 算力 800 TFLOPS（FP16），支持 FP32/HF32/FP16/BF16/INT8，**互联带宽 784GB/s**，HBM 容量 128GB、内存带宽 3.2TB/s"。https://www.stcn.com/article/detail/3345926.html
- 【券商研报·中信证券】"910C 在**单芯片互联上达到 392GB/s 的单向带宽**，对比英伟达 NVLink 第四代的双向带宽为 900GB/s"；同文：910C 双 die，每 die 约 376 TFLOPS(BF16/FP16)，整芯片 752 TFLOPS。https://finance.jrj.com.cn/2025/06/25084951290329.shtml
- 【口径核对（本清单换算，非来源原文）】910C 为双 die 封装，**392GB/s（单向/每 die）× 2 die = 784GB/s**，与官方 HC2025 的 784GB/s 口径自洽；两种数字很可能分别指"每 die 单向"与"整封装"。
- 【券商研报】另有"CANN 生态"分析口径写作"910C 单芯片互联 400GB/s、SemiAnalysis 称 392GB/s 单向"。https://www.stcn.com/article/detail/3345926.html
- 【媒体·自媒体，未证实】自媒体报道 910C"HCCS ~400GB/s 双向"（并给出 910B "HCCS ~300GB/s 双向"）；数字与官方/券商口径不一致，**未证实**。https://cloud.tencent.cn/developer/article/2697481
- 【官方·系统级对照】《超节点技术体系白皮书》（上海人工智能实验室发起，社区协作，**非华为官方**）给出 CM384 系统级互联带宽 **269 TB/s（双向）**、单 NPU 互联带宽"392GB/s 级"。https://deeplink-org.github.io/superpod-whitepaper/01-architecture/02-huawei/

### 1.4 HCCS 是否支持内存语义 / 缓存一致性

- 【官方术语表】昇腾 950 白皮书术语表把 **UB Memory** 定义为"Unified Bus 提供的**同步访存语义，支持 Load/Store 操作**"，把 **URMA** 定义为"UB 提供的**异步内存拷贝语义**"；并明确"同步访问即芯片上的 Core 或调度器可以直接向 UB 端口发起 **Load、Store 和 Atomic** 访问"。→ **内存语义（load/store）是 UB（灵衢）的能力，不是 HCCS 的能力**。https://public-download.obs.cn-east-2.myhuaweicloud.com/ascend/昇腾950%20NPU架构白皮书.pdf
- 【官方】昇腾 950 的 UB Memory 支持的具体操作：Write、Read、AtomicStore、AtomicLoad、AtomicSwap、AtomicCompareAndSwap（经 UB Memory Decoder + 对端 UMMU 地址翻译与权限校验后直接访问对端芯片内存）。https://public-download.obs.cn-east-2.myhuaweicloud.com/ascend/昇腾950%20NPU架构白皮书.pdf
- 【官方】UB Memory 同步语义可实现"最高 **128TB** 的 Host-Device 以及 Device-Device 内存共享访问"（950 代际）。https://public-download.obs.cn-east-2.myhuaweicloud.com/ascend/昇腾950%20NPU架构白皮书.pdf
- 【第三方白皮书（非华为官方）】明确指出："**昇腾 AICore 并没有提供类似 GPU 的 Load/Store 等同步的内存语义访问模式**"，而是由 AICore 下发 **MTE（memory transfer engine）指令**经 UB 实现远端 memory 访问，属"异步大块数据搬移的**类 Load/Store 语义**"。https://deeplink-org.github.io/superpod-whitepaper/01-architecture/02-huawei/
- 【官方·openEuler 社区文档】对"缓存一致性"的口径很关键："当考虑共享场景，我们**不对 ScaleUP 互联所能提供的 Cacheable 能力以及 Cache Coherence 能力&范围做任何强假设**（实际上在大规模 ScaleUP 互联系统中也无法做任何强假设）"；"**当 ScaleUP 互联具有 Cache Coherence 能力时，可直接在相应的运行环境中不额外实现 set_ownership 接口**"——即**华为侧文档未宣称 UB/HCCS 在大规模拓扑下提供全局硬件缓存一致性**，一致性由软件（数据所有权 set_ownership / cache flush + 状态维护）承担。https://www.openeuler.openatom.cn/zh/blog/20260206-SIG-Long_02/20260206-SIG-Long_02.html
- 【官方·华为 2026-09-17 全联接大会】灵衢（UB）"**协议归一，内存语义**：将十余种互联协议统一为灵衢协议，互联带宽从百 GB 级提升至 TB 级，RTT 通信时延从 7 微秒压缩至 2 微秒，**支持超节点内全局内存统一编址**"。https://www.huawei.com/cn/news/2026/9/hc-lingqu-agent-ai
- 【官方·昇腾 950 白皮书的"缓存一致性"仅指 die 内】"由**硬件维护 2 个 Die 之间的 L2 Cache 一致性**，软件不感知"——即华为公布的硬件缓存一致性范围是**同封装内 2 个 die 的 L2**，而非跨卡/跨节点全局一致。https://public-download.obs.cn-east-2.myhuaweicloud.com/ascend/昇腾950%20NPU架构白皮书.pdf
- **结论性事实**：HCCS 名称（Huawei Cache Coherence System）本身以"缓存一致性"命名，但**华为公开材料中未找到任何关于 HCCS 在不同卡之间提供硬件缓存一致性/load-store 语义的官方规格说明** → 该问题记为**未找到公开数据**。

### 1.5 HCCS → UB（灵衢）的代际演进（官方口径）

- 【官方】昇腾 950 白皮书："**规模突破：超节点（Super Node）规模从上一代的 384 卡提升至 8K（8192）卡**，整体集群支持规模超过 128K 卡"——官方把 384 卡超节点确认为"上一代"。https://public-download.obs.cn-east-2.myhuaweicloud.com/ascend/昇腾950%20NPU架构白皮书.pdf
- 【第三方白皮书】代际表：Atlas 800（910A）=HCCS v1、8 卡/节点、2020；Atlas 800T A2（910B）=**HCCS v2**、8 卡/节点、2023；CloudMatrix 384（910C）=**UB 灵衢 1.0**、384 NPU、2024 曝光/2025 商用。https://deeplink-org.github.io/superpod-whitepaper/01-architecture/02-huawei/
- 【官方·NVIDIA 对照（HCCS/UB 之外的官方数字）】见第五节。

---

## 二、910C 的封装方式、代工、HBM、基板

### 2.1 是否双 die（2× 910B die）封装

- 【官方·最强证据】华为+硅基流动论文（arXiv 2506.12708）§3.3.1 原文："**The Ascend 910 is a dual-die package: two identical compute dies are co-packaged, sharing eight on-package memory stacks and connected by a high-bandwidth cross-die fabric.**" 每 die 含 24 个 AI Cube（AIC）+ 48 个 AI Vector（AIV）。注：论文用"Ascend 910"表述，语境为 CloudMatrix384（华为对外称 910C）。https://ar5iv.labs.arxiv.org/html/2506.12708
- 【第三方机构·TechInsights（经彭博/Business Times）】"Huawei's current-generation accelerator, the 910C, is made by **packaging two 910B dies together**"，die 由台积电制造；在**两颗不同 910C 样品**中分别发现**三星与 SK 海力士的 HBM2E**。https://www.businesstimes.com.sg/companies-markets/huawei-used-tsmc-samsung-sk-hynix-components-top-ai-chips-techinsights ｜【彭博原文，付费墙，页面不可访问】https://www.bloomberg.com/news/articles/2025-10-03/huawei-used-tsmc-samsung-sk-hynix-components-in-top-ai-chips
- 【媒体·路透】910C 通过"把两颗 910B 处理器封装进单一封装"实现，推理性能约为 H100 的 60%；**路透原文页面不可访问（HTTP 401）**，可访问转述见下。https://www.reuters.com/world/china/huawei-readies-new-ai-chip-mass-shipment-china-seeks-nvidia-alternatives-sources-2025-04-21/ ｜ https://www.sdxcentral.com/news/huawei-unveils-ascend-920-ai-chip-will-start-shipping-910c-to-chinese-customers-from-may-report-sdx/
- 【券商研报·中信证券】"910C 采用双 Die 封装，每个 Die 提供约 376 TFLOPS（BF16/FP16），整个芯片算力高达 752 TFLOPS"；内存为"单芯片集成 8 个内存堆栈（每堆栈 16GB），共 128GB HBM，带宽 3.2TB/s"。
- 【官方·高管】徐直军 HC2025：910C 800 TFLOPS（FP16）——与券商 752 TFLOPS 口径并存，差异未获官方解释。https://www.stcn.com/article/detail/3345926.html

### 2.2 封装技术（CoWoS / 国产 2.5D / 封装厂）

- 【媒体·TrendForce 转分析师】910C 采用"比英伟达更简单的方案：**两个硅中介层（silicon interposer）通过有机基板（organic substrate）互联**"。https://www.trendforce.com/news/2025/03/13/news-huaweis-ascend-910c-takes-on-nvidia-as-chinas-ai-race-heats-up-more-alleged-details/
- **910C 是否采用台积电 CoWoS：未找到公开数据**（华为署名论文、TechInsights、SemiAnalysis 均未明确表述使用 CoWoS）。
- 【媒体·证券时报】盛合晶微（SJ Semiconductor）"也是**华为昇腾系列 AI 芯片的核心代工厂之一**，其 **3 倍光罩尺寸 TSV 硅通孔载板技术**（亚微米级互联精度）和三维多芯片集成封装技术，为昇腾芯片的高密度集成提供了关键支持"；国内首条大规模量产级 TSV 产线，关键工艺良率 99.5% 以上，年产能 50 万片晶圆。https://www.stcn.com/article/detail/2241117.html
- 【招股书（仅经媒体引用可见；**招股书 PDF 直连 HTTP 404，页面不可访问**）】盛合晶微自述"对于业界最主流的**基于硅通孔转接板（TSV Interposer）的 2.5D 集成**，公司是中国大陆量产最早、生产规模最大的企业之一"。https://static.sse.com.cn/stock/disclosure/announcement/c/202601/002104_20260107_5VZI.pdf
- 【媒体·引用招股书】盛合晶微第一大客户收入占比由 2022 年 40.56% 升至 2025H1 **74.40%**；昇腾类 2.5D 封装订单 **2025H1 占收入 56.24%**；2024 年 2.5D 封装市占率 85%。http://news.cnfol.com/chanyejingji/20251107/31775227.shtml
- 【媒体·新快报】"通富微电**深度绑定华为 AI 和手机芯片**，高端算力封装订单持续放量"；长电科技先进封装收入占比超 70%。https://ep.ycwb.com/epaper/xkb/html/2026-06/08/content_1515_754746.htm
- 【官方论文+媒体】"韬定律"/逻辑折叠（何庭波，ChinaXiv，2026-05-25 首版、2026-07-03 V2）：1.5μm 混合键合，晶体管密度 155→238 MTr/mm²；规划 **2030 年昇腾 990 为首款采用逻辑折叠的 AI 加速芯片**——**未用于 910C**。https://ee.ofweek.com/2026-07/ART-12003-2815-30694231.html
- 【媒体】大陆 CoWoS 产能不足 2 万片/月（全球占比 <11%），台积电 2026 年底约 12.7 万片/月；2.5D/3D 高阶技术国产化率不足 10%。https://ee.ofweek.com/2026-07/ART-12003-2815-30694231.html

### 2.3 代工（台积电 N7 vs 中芯 N+2）

- 【第三方机构·SemiAnalysis】"SMIC 虽具备 7nm，但**绝大多数 Ascend 910B 和 910C 都是用台积电 7nm 制造的**。美国政府、TechInsights 等获取的 Ascend 910B 和 910C，**每一颗都使用了台积电 die**"；华为通过第三方公司 Sophgo 采购约 **5 亿美元** 7nm 晶圆规避制裁。https://newsletter.semianalysis.com/p/huawei-ai-cloudmatrix-384-chinas-answer-to-nvidia-gb200-nvl72
- 【第三方机构·SemiAnalysis】台积电累计提供 **290 万颗 die**（2024–2025），足够做 80 万颗 910B + 105 万颗 910C。https://newsletter.semianalysis.com/p/huawei-ai-cloudmatrix-384-chinas-answer-to-nvidia-gb200-nvl72
- 【媒体·路透】台积电可能因美国调查面临 **10 亿美元或更高罚款**；**原文页面不可访问（HTTP 401）**。https://www.reuters.com/technology/tsmc-could-face-1-billion-or-more-fine-us-probe-sources-say-2025-04-08/
- 【官方·台积电声明（经媒体）】台积电回应 TechInsights 调查："该硬件看起来是用本机构 **2024 年 10 月分析过的 die** 制造的，而非最近制造或更先进技术的 die"，且"该芯片的出货与制造自那时起已经停止"；台积电称自 2020 年 9 月中起未再向华为供货。https://www.businesstimes.com.sg/companies-markets/huawei-used-tsmc-samsung-sk-hynix-components-top-ai-chips-techinsights
- 【官方·高管】徐直军 HC2025："由于受美国的制裁，**华为不能到台积电去投片**，单颗芯片的算力相比英伟达存在差距"，因此走"超节点+集群"路线。https://www.stcn.com/article/detail/3345926.html
- 【第三方机构·SemiAnalysis】SMIC 7nm 及以下产能：2025 年底约 **4.5 万片/月**，2026 年 6 万片/月，2027 年 8 万片/月；Ascend 每月最多只需约 **2 万片**。https://newsletter.semianalysis.com/p/huawei-ascend-production-ramp
- **910B 与 910C 的台积电/中芯产能拆分比例：未找到公开数据。**

### 2.4 HBM 供应商与堆栈/容量/带宽

- 【第三方机构·TechInsights】910C 样品中使用的是**上一代 HBM（HBM2E）**，供应商分别为**三星**与 **SK 海力士**（两颗样品不同）。https://www.businesstimes.com.sg/companies-markets/huawei-used-tsmc-samsung-sk-hynix-components-top-ai-chips-techinsights
- 【第三方机构·SemiAnalysis】仅三星就向中国直接供 **1140 万颗 HBM 堆栈**（其中 2024-12 管制生效前一个月窗口内 700 万颗），计入其他渠道合计约 **1300 万颗堆栈 = 可装 160 万颗 910C**（≈**8 堆栈/颗**，与券商口径一致）。https://newsletter.semianalysis.com/p/huawei-ascend-production-ramp
- 【第三方机构·SemiAnalysis】HBM 入境路径：CoAsia Electronics（三星 HBM 大中华区独家分销商）→ Faraday → SPIL "封装"成系统级封装 → 中国解焊回收 HBM（低温弱焊凸点便于拆解）。https://newsletter.semianalysis.com/p/huawei-ai-cloudmatrix-384-chinas-answer-to-nvidia-gb200-nvl72
- 【官方·供应商声明（经媒体）】SK 海力士称"自 2020 年限制措施实施后就已停止与华为的一切交易"；三星称"持续严格遵守"美国出口规定、与清单实体"没有任何业务关系"。https://www.businesstimes.com.sg/companies-markets/huawei-used-tsmc-samsung-sk-hynix-components-top-ai-chips-techinsights
- 【官方·高管】徐直军：2026 年一季度发布的**昇腾 950PR 将采用华为自研 HBM**（**HiBL 1.0**；950DT 升级至 **HiZQ 2.0**）。https://www.stcn.com/article/detail/3345926.html
- 【媒体·引匿名信源，未证实】CXMT（长鑫）已启动 **HBM3 模组量产**，计划 HBM3 月产能 6 万片（约其 2026 年 30 万片/月产能的 20%）；匿名信源称"华为正与长鑫共同开发 HBM，预计即使良率偏低也将进入量产"。**CXMT 是否通过 910C 认证 / 是否已实际用于 910C：未找到公开数据 / 未证实。** https://www.techpowerup.com/346207/cxmt-reportedly-plans-to-dedicate-20-of-mass-production-capacity-to-hbm3-line-in-2026
- 【券商研报·中信证券】910C 单芯片 **8 个 HBM 堆栈 × 16GB = 128GB**，带宽 **3.2TB/s**。
- 【官方·高管】徐直军：910C HBM **128GB、3.2TB/s**。https://www.stcn.com/article/detail/3345926.html
- 【自媒体，未证实】有自媒体称 910C 为"96GB HBM / 1.8TB/s"（低于主流口径），**未证实**。https://cloud.tencent.cn/developer/article/2697481

### 2.5 量产时间与产量

- 【官方·高管】徐直军路线图确认：**910C 为 2025 年 Q1 最新发布**；2025 年 3 月推出基于 910C 的 Atlas 900 超节点（满配 384 卡，最大 300 PFLOPS）。https://www.stcn.com/article/detail/3345926.html ｜ https://paper.cnstock.com/html/2025-09/19/content_2123758.htm
- 【媒体·路透】华为准备**从 2025 年 5 月起向中国客户批量出货 910C**。https://www.sdxcentral.com/news/huawei-unveils-ascend-920-ai-chip-will-start-shipping-910c-to-chinese-customers-from-may-report-sdx/
- 【第三方机构·SemiAnalysis】**2024 年 50.7 万颗**（以 910B 为主）；**2025 年 80.5 万颗，其中 910C 为 65.3 万颗**；台积电"die 银行"约 9 个月内耗尽；2026 年若无外国 HBM，"连 100 万颗昇腾都造不出来"。https://newsletter.semianalysis.com/p/huawei-ascend-production-ramp
- 【媒体·金融时报（经 TrendForce 转述）】华为把最新 AI 芯片**良率翻倍到接近 40%**（一年前约 20%），昇腾产线首次实现盈利。https://www.trendforce.com/news/2025/03/13/news-huaweis-ascend-910c-takes-on-nvidia-as-chinas-ai-race-heats-up-more-alleged-details/
- 【自媒体，未证实】910C 2026 年产量目标约 60 万颗。https://ruibao.news/institute/tech-tracker/2026-04/

### 2.6 基板 / 中介层

- 【媒体·证券时报】盛合晶微提供 **3 倍光罩尺寸 TSV 硅通孔载板**，为昇腾芯片高密度集成的关键支持。https://www.stcn.com/article/detail/2241117.html
- 【公告/互动易·兴森科技（002436）】其 ABF 载板主要应用于 CPU/GPU/FPGA/ASIC；**无公开证据指向其为 910C 供货**。https://basic.10jqka.com.cn/002436/
- **910C 具体 ABF 载板 / 有机基板供应商：未找到公开数据。**

---

## 三、910D 最新进展（2025–2026）

- 【官方·缺失证据】**"昇腾 910D / Ascend 910D" 从未被华为官方发布**：huawei.com 新闻库与 hiascend.com 产品/新闻库中均无任何 910D 页面或新闻稿。https://www.hiascend.com/en/hardware/cluster?tag=950
- 【官方·路线图，最关键】华为全联接大会 2025（2025-09-18，上海）徐直军公布官方路线图：**910C（2025Q1）→ 950PR（2026Q1）→ 950DT（2026Q4）→ 960（2027Q4）→ 970（2028Q4）**；**通篇无 910D**。https://www.huawei.com/cn/news/2025/9/hc-lingqu-ai-superpod ｜ https://www.stcn.com/article/detail/3345926.html
- 【官方】HC2026（2026-09-17）汪涛：**昇腾 910C 超节点已部署超 1000 套，昇腾 950 超节点已开始规模商用**；960 研发超预期，960DT 提前至 2027Q1、960PR 提前至 2027Q3；2028/2029 推 970/980。https://www.huawei.com/cn/news/2026/9/hc-wang-keynote
- 【媒体转述 WSJ，未证实】2025-04-28 WSJ：华为已接触多家中国科技公司测试 910D 技术可行性，**首批样品最早 2025 年 5 月底**；用封装技术集成更多硅晶粒，**比 H100 更强但能效更低**。https://www.silicon.co.uk/cloud/ai/huawei-ai-chip-610953 ｜ https://www.trendforce.com/news/2025/04/28/news-huawei-reportedly-set-to-test-new-ascend-910d-ai-chip-as-early-as-may-aiming-to-challenge-nvidia/
- 【专利为真、用途为推测】华为 2024 年 4 月申请的"一种集成装置、通信芯片和通信设备"专利（国际申请号 **PCT/CN2024/086375**，公开号 **WO2024222427A1**）为**四芯片（quad-chiplet）封装**、硅中介层 + 类 CoWoS-L 桥接；媒体**推测**用于 910D。https://patentimages.storage.googleapis.com/66/fd/f7/a7f894b0022c64/WO2024222427A1.pdf
- 【媒体测算，未证实】910D 为 **4 颗 die（4×910B）**；单颗 910B 约 665mm²，四芯片合计约 2660mm²，16 颗 HBM 堆栈约 1360mm²，**总硅面积约 4020mm²**（约合 5 个 EUV 光罩）。https://www.trendforce.com/news/2025/06/17/news-huaweis-quad-chiplet-910d-reportedly-takes-shape-with-advanced-packaging-to-challenge-tsmc-nvidia/
- 【媒体，未证实且与其它来源矛盾】"四芯片封装可使单卡 FP16 算力提升至 1,400 TFLOPS"；"完全基于中芯国际 14nm 制程与长电科技封装产线"；"四芯片堆叠封装良率不足 65%"。https://wap.seccw.com/document/detail/id/34134.html
- **910D 的 HBM 类型（是否 HBM3/HBM3E）、容量、带宽：未找到公开数据。**
- **910D 的互联方案（HCCS vs 灵衢 UB，是否升级）：未找到公开数据。**
- **910D 的 TDP / 量产时间：未找到公开数据；无任何官方或可靠的流片（tape-out）、良率报道。**
- 【媒体分析，未证实】南风窗（2026-04）认为 910D"**可能不再出现**"，910C 是 2 块 910B 叠加、910D 本应是 4 块 910B 叠加，但因需求转向推理而被跳过，下一代命名直接恢复为 950。https://www.cqcb.com/news/56/2026-04-27/6126680.html
- 【官方·可对照的 950 代际规格（910D 若存在应对标的对象）】950 官方白皮书：950PR 单片 **128GB / 1.6TB/s**；950DT **144GB / 4TB/s**；**单芯片互联带宽 2TB/s**；UB 2.0；超节点最大 **8192 卡**；集群 >128K 卡。https://public-download.obs.cn-east-2.myhuaweicloud.com/ascend/昇腾950%20NPU架构白皮书.pdf

---

## 四、CloudMatrix 384 超节点（Atlas 900 A3 SuperPoD）

### 4.1 拓扑结构（官方论文）

- 【官方】**384 颗昇腾 910 NPU + 192 颗鲲鹏 CPU**，经全对等（peer-to-peer）all-to-all 的 **UB（灵衢 / Unified Bus）** 互联为一个 supernode；三个网络平面：UB（scale-up）、RDMA/RoCE（scale-out）、VPC（数据中心网络）。https://ar5iv.labs.arxiv.org/html/2506.12708
- 【官方】**16 机柜 = 12 个计算柜（共 48 个计算节点 = 384 NPU）+ 4 个通信柜（L2 UB 交换）**；即 **4 节点/计算柜**、**8 NPU/节点**（换算）。https://ar5iv.labs.arxiv.org/html/2506.12708
- 【官方】**每个计算节点 = 8×910 NPU + 4×鲲鹏 CPU + 7 颗板载 L1 UB 交换芯片**；12 个处理器（8 NPU + 4 CPU）接入板载交换机形成节点内单层 UB 平面；4 个鲲鹏 socket 之间为 full-mesh NUMA。https://ar5iv.labs.arxiv.org/html/2506.12708
- 【官方】UB 交换两层：**L1 = 板载 7 颗/节点（合计 336 颗）**；**L2 分 7 个独立子平面，每子平面 16 颗 L2 芯片、每颗 48 端口（合计 112 颗）**；L1 每颗向所属子平面的 16 颗 L2 各拉 1 条链路（16 links）；**L2 层无收敛（non-blocking）**。https://ar5iv.labs.arxiv.org/html/2506.12708
- 【官方】跨节点**带宽衰减 <3%、时延增加 <1 µs**；论文全部带宽数字为**单向**口径。https://ar5iv.labs.arxiv.org/html/2506.12708
- 【官方·高管】徐直军 HC2025：CloudMatrix 384 超节点"累计部署 300+ 套，服务 20+ 客户"；Atlas 950 SuperPoD（8192 卡）2025Q4 上市、Atlas 960 SuperPoD（15488 卡）2027Q4。https://www.stcn.com/article/detail/3345926.html
- 【官方·HC2026】汪涛："截至今天昇腾 910C 超节点已部署**超过 1000 套**，昇腾 950 超节点已开始规模商用。"https://www.huawei.com/cn/news/2026/9/hc-wang-keynote

### 4.2 光互联方案与光模块数量

- 【第三方机构·SemiAnalysis（免费段可读）】一个 CloudMatrix Pod 需 **6,912 个 400G LPO 光模块**，绝大部分用于 scale-up 网络；文章副标题口径为 "**China Abundance of Power, 100% Optics, 0% Copper**"、"**14 Transceivers per Chip**"。https://newsletter.semianalysis.com/p/huawei-ai-cloudmatrix-384-chinas-answer-to-nvidia-gb200-nvl72
- 【媒体·展会实测/华为展台】"**3168 根光纤 + 6912 个 400G 光模块**实现百纳秒级互联，支持 2m 以上长距部署"；384 卡 × 18 = 6912，即**每 NPU 18 个 400G 光模块**。https://cloud.tencent.cn/developer/article/2642597
- 【媒体·科创板日报】"超节点网络交换机采用 **6812 个 400G 光模块**，实现 **2.8Tbps 卡间互联带宽**"（6812 疑为 6912 之笔误）。https://www.chinastarmarket.cn/detail/2003317
- 【券商研报·申万宏源】华为 CM384 光模块需求比 **1:18 = Scale-up 1:14 + Scale-out 1:4**；原文："每台服务器 8 个 NPU 对应 112 个 400G 光模块或 56 个 800G 模块，即 NPU 与 400G 光模块用量比在 1:14，与 800G 光模块用量比在 1:7"；每颗 UB Switch 对外 16×28GBps = 448GBps = 8×400G。https://www.sgpjbg.com/labelsyh/aiguangmokuaixuqiucesuan/1/6545745.html
- 【券商研报·申万宏源】Scale-out 侧：12 个计算机柜为一个 CloudMatrix 节点，按胖树扩容、两层拓扑不收敛，光模块比约 1:4；"超节点内部（Scale-up）光模块用量占主体"。https://www.sgpjbg.com/labelsyh/aiguangmokuaixuqiucesuan/1/6545745.html
- 【媒体·2026 华为中国合作伙伴大会（深圳 3/19–20）】现场展示 **6912 片"星云 400G"光互联模块**，**非标准通用模块、由华为定制**，针对自有芯片与整机优化信号质量。https://www.c114.com.cn/news/126/a1307246.html
- 【官方·HC2026】昇腾 960 超节点（NPO）用 **5500 个自研 Hi-ONE** 替代原本需要的 **48000 颗 800G 光模块，降低超 550 kW 功耗**；Hi-ONE 单引擎 7.2T，"业界首个量产 NPO、唯一内置光源"。https://www.huawei.com/cn/news/2026/9/hc-wang-keynote

### 4.3 每 GPU 聚合互联带宽（含"~2.8Tbps"口径辨析）

- 【官方/第三方白皮书口径】CM384 **系统级 Scale-Up 带宽 269 TB/s（双向）**；单 NPU 互联带宽"392GB/s 级"。https://www.hlzq.com/upload1/yb/20260828/1787882411196.pdf ｜ https://deeplink-org.github.io/superpod-whitepaper/01-architecture/02-huawei/
- 【换算，非来源原文】**269 TB/s ÷ 384 NPU ≈ 700 GB/s/卡（双向）= 5.6 Tbps 双向 ≈ 2.8 Tbps 单向**——中文报道中的 "**2.8Tbps 卡间互联带宽**"（科创板日报）与**单向**口径数值一致。
- 【未证实】"SemiAnalysis 曾报道每 GPU 约 2.8Tbps"这一具体表述**未能在可访问来源中证实**（SemiAnalysis 原文相关段落位于付费墙内，**页面不可访问**）。请勿把 2.8Tbps 直接标注为 SemiAnalysis 数据。
- 【未证实】"2.8Tbps **双向**"的说法**未在任何来源中证实**。
- 【券商研报·中信证券 引 SemiAnalysis】CM384 BF16 总算力为 GB200 NVL72 的 **1.7 倍**、总内存容量 **3.6 倍**、内存带宽 **2.1 倍**；总功耗 **3.9 倍**、单 TFLOP 耗电 **2.3 倍**。
- 【第三方机构·SemiAnalysis（免费段原文）】"it takes **4.1x the power of a GB200 NVL72**, with **2.5x worse power per FLOP**, 1.9x worse power per TB/s memory bandwidth, and 1.2x worse power per TB HBM memory capacity"；CM384 密集 BF16 算力 **300 PFLOPs**，"almost double that of the GB200 NVL72"，"3.6x aggregate memory capacity and 2.1x more memory bandwidth"。https://newsletter.semianalysis.com/p/huawei-ai-cloudmatrix-384-chinas-answer-to-nvidia-gb200-nvl72

### 4.4 功耗、机柜、部署时间、客户

- 【官方】CM384 官方未公布整机功耗与功耗比；华为口径为"单集群 300 PFLOPS BF16 稠密算力，约为 GB200 NVL72 的 1.7 倍"，MFU 从行业平均 30% 提升至 45%+，已用于训练 7180 亿参数的盘古 Ultra MoE 大模型。https://cloud.tencent.cn/developer/article/2642597
- 【第三方白皮书（非华为官方）】系统总功耗 **约 559 kW**（对比 GB200 NVL72 约 145 kW）；16 柜超节点。https://deeplink-org.github.io/superpod-whitepaper/01-architecture/02-huawei/
- 【券商研报·华龙证券，引同一白皮书】对照表：计算芯片 384×910C vs 72×Blackwell；系统算力 ~300 PFLOPs(BF16/FP16) vs 180 PFLOPs；Scale-Up 域 384 NPU vs 72 GPU；系统 HBM **49.2 TB vs 13.8 TB**；系统 HBM 带宽 **1229 TB/s vs 576 TB/s**；Scale-Up 带宽 **269 TB/s(双向) vs 130 TB/s(双向)**；形态 16 柜 vs 单柜；总功耗 **约 559 kW vs 约 145 kW**。https://www.hlzq.com/upload1/yb/20260828/1787882411196.pdf
- 【官方·华为云】华为云在贵州、内蒙古、安徽部署全液冷 AI 数据中心，**实现单机柜 80kW 散热、PUE 低至 1.1**；机柜功率从 10kW 向 70kW/200kW 升级。https://m.thepaper.cn/detail/31634478
- 【官方论文·性能数据】DeepSeek-R1（671B MoE）：prefill **6,688 tokens/s/NPU**（4.45 tokens/s/TFLOPS）、decode **1,943 tokens/s/NPU** @TPOT<50ms（1.29 tokens/s/TFLOPS）、TPOT<15ms 时 538 tokens/s/NPU；支持 **EP320**（每 die 承载 1 个专家）。https://ar5iv.labs.arxiv.org/html/2506.12708
- 【媒体·2025-04-10 华为云生态大会（芜湖）】发布 CloudMatrix 384 超节点，**已在芜湖数据中心规模上线**。https://tidenews.com.cn/news.html?id=3095412
- 【媒体·2025-07-28 WAIC 上海】昇腾 384 超节点真机首展；**已在芜湖、贵安、乌兰察布、和林格尔数据中心全面上线**。https://www.chinastarmarket.cn/detail/2097675
- 【媒体·2025-04-14 科创板日报】硅基流动联合华为云基于 CM384 上线 DeepSeek-R1，单卡 decode 吞吐突破 **1920 tokens/s**。https://www.chinastarmarket.cn/detail/2003317
- 【媒体】CM384 客户案例：**新浪**"智慧小浪"推理交付效率 +50%、**硅基流动**（每天为 600 万用户提供推理）、**中科院 AI4S**、**面壁智能**"小钢炮"、**360** 纳米 AI 搜索、**讯飞**大模型。https://www.chinastarmarket.cn/detail/2097675
- 【媒体·2026-01-22】截至 2025 年底昇腾 AI 云服务客户数突破 **2663 家**（2024 年 321 家）。https://www.qbitai.com/2026/01/371533.html
- 【交易所/集采公告（经媒体）·2026-04-27】中国移动《2026-2027 年人工智能超节点设备集中采购》**6208 张卡（776 套计算节点）、全部采用华为 CANN 生态**，五家报价集中于 20.59–20.70 亿元。https://www.ccidcom.com/company/20260427/wW9f8YsAaxZ2ESojY1clykk6lpsm0.html
- 【媒体·2026-08-26】中国移动呼和浩特智算中心：一期 6208 卡（20.6 亿元）+ 扩容 3840 卡（约 12.9 亿元），**两期累计约 33.5 亿元、部署超 1 万卡，均基于华为昇腾超节点**。https://www.stnn.cc/detail/6a8efba0c3c8142abfe9a7a6.html
- 【未找到公开数据】DeepSeek 本身是否采购 CM384 未见可核实一手证据（论文合作方为硅基流动 SiliconFlow，非 DeepSeek）；蚂蚁、字节、中国电信、中国联通作为 CM384 客户亦未找到可核实一手证据。

---

## 五、HCCS vs NVLink 的量化差距（NVIDIA 官方数字）

- 【官方·NVIDIA】NVLink 规格表（第四代/第五代/第六代）：**NVLink 每 GPU 带宽 900 GB/s / 1,800 GB/s / 3,000 GB/s**；每 GPU 最大链路数 **18 / 18 / 36**；NVLink Switch GPU-to-GPU 带宽 **900 GB/s / 1,800 GB/s / 3,000 GB/s**；**总聚合带宽 7.2 TB/s（Hopper 8-GPU 域）/ 130 TB/s（NVL72）/ 216 TB/s（Vera Rubin NVL72）**；NVLink GPU 域 8 / 8|72 / 8|72。https://www.nvidia.com/en-us/data-center/nvlink/
- 【官方·NVIDIA】NVLink 6 每 GPU **3 TB/s**，"1.7x more bandwidth than the previous generation and 12x the bandwidth of PCIe Gen6"；Vera Rubin NVL72 全互联 72 GPU、聚合 **216 TB/s**。https://www.nvidia.com/en-us/data-center/nvlink/
- 【官方·NVIDIA】NVLink 3（A100）官方 datasheet 原文："**NVLink: 600GB/s**"（每 GPU，SXM4）。https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf
- 【官方·NVIDIA 博客】"第三代 NVLink 将每 GPU 最大带宽翻倍至 **600GB/s**……如今 **18 条第四代 NVLink** 嵌入单颗 H100"；第四代"以高达 **900 GB/s** 的速率连接"；NVLink 能效 1.3 pJ/bit。https://blogs.nvidia.com/blog/what-is-nvidia-nvlink/
- 【官方·NVIDIA GB200 NVL72 产品页】"NVLink Switch System provides **130 terabytes per second (TB/s)** of low-latency GPU communications"，72 GPU NVLink 域。https://www.nvidia.com/en-us/data-center/gb200-nvl72/
- 【官方·NVIDIA 技术博客】Blackwell Ultra 对照表：NVLink 带宽 **H100 900 GB/s → B200 1,800 GB/s → GB300 1,800 GB/s**；HBM 带宽 3.35 / 4.8 / 8 / 8 TB/s；TGP 700W / 1,200W / 1,400W。https://developer.nvidia.com/blog/?p=104887
- 【量化差距汇总（依据上述官方数字，本清单整理）】
  - 910B 单 die HCCS **392 GB/s**（口径分歧：第三方称单向总带宽）vs A100 NVLink 3 **600 GB/s** → 约 **0.65×**
  - 910C 整封装互联 **784 GB/s**（官方 HC2025）vs H100 NVLink 4 **900 GB/s** → 约 **0.87×**
  - 910C 每 die **392 GB/s（单向）** vs H100 NVLink 4 **900 GB/s（双向）** → 约 **0.44×**（中信证券口径："NVLink 碾压 HCCS 2.25 倍"）
  - CM384 系统级 Scale-Up **269 TB/s（双向）** vs GB200 NVL72 **130 TB/s（双向）** → 约 **2.07×**（超节点系统层面华为领先）
  - 910C **784 GB/s** vs B200 NVLink 5 **1,800 GB/s** → 约 **0.44×**
  - 昇腾 950 **2 TB/s**（官方白皮书，单芯片互联）vs B200 NVLink 5 **1,800 GB/s** → 约 **1.11×**
- 【官方·可比性警告】华为 910C 官方口径（784GB/s）与 NVIDIA NVLink 口径（900GB/s 双向）**未明确标注单向/双向，二者不可直接相除**；中信证券明确以"392GB/s 单向 vs 900GB/s 双向"作对比，本清单已原样保留该口径。

---

## 六、关键口径冲突与检索局限（务必随清单引用）

1. **910C 单卡互联带宽存在三个数字**：392GB/s（中信证券，标注"单向"）、784GB/s（徐直军 HC2025 PPT，整封装）、约 400GB/s（自媒体，口径不明）。三者**不要混用**。
2. **910C HBM 容量存在 128GB（官方+券商+TechInsights 口径）与 96GB（自媒体）两种**；以官方 128GB/3.2TB/s 为准。
3. **910C 算力存在 800 TFLOPS（官方 HC2025）与 752 TFLOPS（中信证券，双 die 各 376）两种**。
4. **CloudMatrix 384 整机功耗无官方数据**；559 kW 来自第三方《超节点技术体系白皮书》，SemiAnalysis 仅给出"4.1× GB200 NVL72"的相对值。
5. **"2.8Tbps 每 GPU"并非已证实的 SemiAnalysis 原文数据**；它是 269 TB/s ÷ 384 卡的换算值，且对应**单向**口径。
6. **页面不可访问 / 抓取失败清单**：Reuters（TSMC 罚款 2025-04-08、910C 出货 2025-04-21、字节阿里下单 2026-03-27）HTTP 401；Bloomberg 原文付费墙；SemiAnalysis 付费墙（仅免费段可读）；TechInsights 官方页仅 JS 空壳；盛合晶微招股书 PDF（SSE）HTTP 404；hiascend.com 产品页与开发者论坛帖 JS 渲染、只返回框架；e.huawei.com 产品页/白皮书页 JS 渲染；wccftech HTTP 403；中时新闻网 HTTP 403；联合早报跨域重定向未跟随；百度百科 HTTP 403 安全验证。
7. **华为官方从未公布 910B 的 HCCS 端口数、SerDes 速率、lane 数**；亦从未公布 HCCS 的缓存一致性/内存语义规格。相关数字均属第三方或媒体口径，凡未在权威来源中出现的一律记为**未找到公开数据**。
