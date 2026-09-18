# 芯原股份（VeriSilicon，688521.SH）Chiplet / Die-to-Die IP 与 UCIe 事实清单

核实日期：2026-09-18　|　可信度：【官方】>【招股书/年报/交易所公告】>【标准组织官网】>【券商研报】>【媒体/自媒体】

---

## 1. 产品名称（Die-to-Die / Chiplet IP）

1.1 【官方】芯原 UCIe 产品线共 **3 个具名 IP**，官网"接口IP → UCIe (Chiplet D2D)"栏目下列出：**UCIe-SP（PHY）**、**UCIe Controller-AXI**、**UCIe Controller-CHI**。
https://www.verisilicon.com/cn/IPPortfolio/InterfaceIP

1.2 【官方】UCIe-SP 页面标题即 **"Die-to-Die (D2D) Interface for Chiplet Application"**；原文："The UCIe PHY IP is a market-leading, ultra low-power, and low-latency interface IP for high-bandwidth connections between two dies that are on the same substrate."（Part Number：UCIe-SP）
https://www.verisilicon.com/cn/IPPortfolio/UCIe-SP　https://www.verisilicon.com/en/IPPortfolio/UCIe-SP

1.3 【官方】UCIe Controller-AXI 原文："UCIe Controller (AXI Protocol + Adapter)… The UCIe Controller supports Streaming Protocol."
https://www.verisilicon.com/cn/IPPortfolio/UCIeController-AXI

1.4 【官方】UCIe Controller-CHI 原文："UCIe Controller (CHI Protocol + Adapter)… SoC logics communicate with UCIe Protocol Layer via the CHI Issue G interface."
https://www.verisilicon.com/en/IPPortfolio/UCIeController-CHI

1.5 【交易所公告】**"VeriHealth/VeriHealthi"不是 Die-to-Die/Chiplet 产品线**。年报原文将其列为"芯健康（VeriHealthi）平台"，属健康监测方向："该平台已为全球范围内的多家客户提供了侵入式脑机接口芯片、胶囊内窥镜芯片以及可穿戴低能耗蓝牙芯片等的 ASIC 定制设计服务"。→ **用户线索中的"VeriHealth 是 D2D 产品"不成立。**
https://stock.stockstar.com/notice/SN2026081800000904.shtml

1.6 【未证实】**未找到任何"Glink"属于芯原的证据**。"GLink-3D Die-to-Die Slave PHY"（IGAD2DX03A TSMC CLN7FF）见于 **GUC 创意电子**产品资料，与芯原无关。→ **"Glink 是芯原产品"不成立。**
https://www.guc-asic.com/upload/2025_07_01/4_20250701210546rptoawPgN12.pdf

1.7 【交易所公告】**无 "Chiplet as a Service" 这一官方产品名**。芯原官方提法为商业模式 **SiPaaS（Silicon Platform as a Service）** 与三句技术方针："IP 芯片化（IP as a Chiplet）""芯片平台化（Chiplet as a Platform）""平台生态化（Platform as an Ecosystem）"。
https://stock.stockstar.com/notice/SN2026081800000904.shtml（2026年半年报）

1.8 【官方】一站式 Chiplet 平台的项目级名称：**"AIGC 与智慧出行领域 Chiplet 解决方案平台研发项目"**（再融资募投项目），含"关键功能模块 Chiplet、Die-to-Die 接口、Chiplet 芯片架构、先进封装技术"。
https://www.yicaiglobal.com/star50news/2025_02_136793355338528587776　https://static.cninfo.com.cn/finalpage/2025-04-26/1223311717.PDF（2024年报）

---

## 2. UCIe 版本 / 每 lane 速率 / 封装 / 模块化

2.1 【官方】**三条 UCIe 产品线均声明符合 "UCIe Specification Revision 1.1"**（UCIe-SP、Controller-AXI、Controller-CHI 的 Features 首条/明确列示）。
https://www.verisilicon.com/cn/IPPortfolio/UCIe-SP　https://www.verisilicon.com/cn/IPPortfolio/UCIeController-AXI　https://www.verisilicon.com/en/IPPortfolio/UCIeController-CHI

2.2 【未找到公开数据】**未找到芯原支持 UCIe 2.0 或 UCIe 3.0 的任何官方表述**（官网产品页、2024/2025 年报、2026 半年报、投资者关系活动记录表均无）。→ 官方口径目前仅到 **UCIe 1.1**。

2.3 【官方】UCIe-SP 速率：**"Supports 4, 8, 12, 16, and 24 GT/s data rates"**；概述原文 **"data rate up to 24 Gbps per pin"**（即每 pin 最高 24 Gbps）。
https://www.verisilicon.com/cn/IPPortfolio/UCIe-SP

2.4 【官方】模块化（module）设计：**"Supports muti-module (1, 2, or 4) design"**（原文拼写为 muti-）；**"Flexible configuration: 16 RX and TX pins per module (standard package)"**。→ 单 module 16 路 TX/RX，可 1/2/4 module 扩展。
https://www.verisilicon.com/cn/IPPortfolio/UCIe-SP

2.5 【官方】封装类型：UCIe-SP 明确标注 **"Package type: standard package"**，功耗 **"< 1.2 pJ/bit @ 16 Gbps (standard package)"**。→ **公开规格仅覆盖 standard package（标准封装）**，未见 advanced package（2.5D/3D、CoWoS、INFO）版本 UCIe PHY 的公开规格。
https://www.verisilicon.com/cn/IPPortfolio/UCIe-SP

2.6 【官方】先进封装能力体现在**芯片定制项目**而非 UCIe PHY 规格：2026 半年报原文"已帮助客户设计了基于 Chiplet 架构的 Chromebook 芯片，采用了 SiP（System in Package）先进封装技术，将高性能 SoC 和多颗 IPM 内存合封；已帮助客户的 AIGC 芯片设计了 2.5D CoWos 封装"；"针对新一代面板级封装（Panel level package）技术进行了先行设计开发"。
https://stock.stockstar.com/notice/SN2026081800000904.shtml

2.7 【官方】UCIe-SP 其它规格：sideband channel（初始化与参数交换）、self-calibrating and training、BIST / 内部 loopback / 外部 PHY-to-PHY link test、on-chip termination impedance 校准、最低输出驱动电压 0.5 V、结温 -40°C ~ 125°C。
https://www.verisilicon.com/cn/IPPortfolio/UCIe-SP

2.8 【官方】Controller 规格（AXI/CHI 两版）：FDI/RDI 数据位宽 **32B（CHI）/ 32 或 64B（AXI）**；CFG 位宽 **8/16/32 bit（CHI）/ 32 bit（AXI）**；**256B latency-optimized flit**（支持 Streaming Protocol 可选字节）；APB4 32-bit 地址/数据寄存器接口；CHI 版支持 C2C Packetization Layer、CHI↔UCIe 时钟域穿越；AXI 版支持 AXI4、反压机制、读写超时监控、可配置 ATU region。
https://www.verisilicon.com/en/IPPortfolio/UCIeController-CHI　https://www.verisilicon.com/cn/IPPortfolio/UCIeController-AXI

2.9 【官方】第三方 IP 平台（转载官网规格，可作交叉印证）：Semi IP Hub 列 UCIe-SP 为 PHY / UCIe 1.1 / standard package。
https://semiiphub.com/ip/ucie-sp-ip-24076

---

## 3. 商用 / 客户 / 流片进展 / 授权收入

3.1 【交易所公告】2026 半年报研发项目表原文（逐字）：**"-UCIe PHY 在……完成硅验证，性能达到预期，已经和 UCIe 控制器 IP 一起导入客户的项目。8nm 版本已完成流片。-UCIe 控制器 IP 已经搭配第二版 UCIe 物理层 IP 一起进行回片测试，目前控制器功能正常，整个子系统性能和稳定性压力测试正在进行中。-启动车规级的 UCIe 物理层 IP 研发，计划年度完成流片。-HSS_DSP_SERDES 进行封装和测试 PCB 制板。第二版改进版正在进行中，并计划于年底完成流片。"**
https://stock.stockstar.com/notice/SN2026081800000904.shtml

3.2 【交易所公告】2025 年报研发项目表原文（逐字）：**"-UCIe PHY 在 14nm 和 4nm 已经完成硅验证，性能达到预期 -UCIe 控制器 IP 加上 UCIe 物理层 IP 整个子系统完成硅验证，将于今年完成客户项目流片 -车规级的 UCIe 物理层 IP 第一季度完成流片 -高速 HSS_DSP_SERDES IP 已经完成流片"**
http://static.cninfo.com.cn/finalpage/2026-03-31/1225060349.PDF（2025 年年度报告，第 77 页）

3.3 【交易所公告】2024 年报原文（逐字）：**"已设计研发了针对 Die to Die 连接的 UCIe 物理层接口，相关测试芯片已流片，即将返回进行封装和测试"**（即 2024 年底尚在等待回片）。
https://static.cninfo.com.cn/finalpage/2025-04-26/1223311717.PDF（2024 年年度报告，第 26 页）

3.4 【交易所公告】2026 半年报 Chiplet 成果原文（逐字）：**"公司自主研发的面向 Die-to-Die 连接的 UCIe 物理层接口 IP 已成功通过流片验证并获客户项目导入"**；以及 **"已设计研发了针对 Die to Die 连接的 UCIe 物理层接口，并已顺利完成流片测试，相关技术已被客户采纳用于项目开发"**。
https://stock.stockstar.com/notice/SN2026081800000904.shtml

3.5 【官方】公开点名的 Chiplet 客户案例（采用**处理器 IP**，非 UCIe IP）：南京蓝洋智能采用芯原 **GPGPU IP CC8400、NPU IP VIP9400、VPU IP VC8000D** 部署可扩展 Chiplet 架构 AI 芯片；官方口径"内置芯原 GPGPU IP 的 Chiplet 芯片算力可达 8 TFLOPS，内置芯原 NPU IP 的 Chiplet 芯片算力可达 240 TOPS"；芯原 IP 事业部总经理戴伟进原话："UCIe 和 BOW Die-to-Die 接口的日益成熟，以及 Chiplet 封装成本的逐步降低，将加速 Chiplet 在复杂功能芯片中的应用。"
https://verisilicon.com/cn/PressRelease/BlueOcean（2023-03-30）

3.6 【媒体】2025-02-13 一财全球（引述芯原投资者交流）：芯原 **"has designed and developed UCIe/BoW-compatible physical layer interfaces for die-to-die connections"**（UCIe/BoW 双兼容 PHY），并点名合作方为 Lanyang Intelligence（蓝洋智能）。→ BoW（Bunch of Wires）兼容性仅有此一处媒体表述，**未在年报/官网中得到确认**。
https://www.yicaiglobal.com/star50news/2025_02_136793355338528587776

3.7 【未找到公开数据】**未找到 UCIe IP 单独授权收入金额、UCIe 授权客户名称、或 UCIe 特许权使用费**的公开披露。仅 2026H1 整体口径可参考："报告期内，公司半导体 IP 授权服务业务收入为 3.49 亿元，占营业收入总额比例为 18.72%"（未拆分 UCIe）。
https://stock.stockstar.com/notice/SN2026081800000904.shtml

3.8 【交易所公告】车规 UCIe 路线：2025 年报计划"车规级 UCIe 物理层和控制器 IP 取得功能安全 ASIL-B 证书"；2026H1 表述为"启动车规级的 UCIe 物理层 IP 研发，计划年度完成流片"。→ **截至 2026 半年报，车规级 UCIe IP 尚未取得 ASIL-B 证书（公开口径未确认）。**
http://static.cninfo.com.cn/finalpage/2026-03-31/1225060349.PDF　https://stock.stockstar.com/notice/SN2026081800000904.shtml

3.9 【招股书】H 股招股书（2026-04-01 递交）述及接口 IP 组合"涵盖 UCIe、MIPI、USB、HDMI 和 PCIe 行业标准，并已覆盖到 8 纳米和 4 纳米等先进工艺"；Chiplet 条目标注"自 2020 年起投资"、"加入 UCIe 中国内地较早业之一（2022 年）"。（注：该 PDF 中文字体 ToUnicode 映射损坏，部分中文经字形比对判读，**关键数字建议以年报为准**。）
https://www1.hkexnews.hk/app/sehk/2026/108392/documents/sehk26040104032_c.pdf　https://www1.hkexnews.hk/app/sehk/2026/108392/a131588/sehk26040104050_c.pdf

---

## 4. 年报 / 半年报 / 业绩说明会 / 招股书 官方原文摘录

### 4.1 Chiplet 战略（2024 年报 / 2025 年报 / 2026 半年报 三处措辞一致）
【交易所公告】**"Chiplet 技术及产业化是芯原的发展战略之一，公司已于五年前开始布局 Chiplet 技术的研发。目前，公司正在以"IP 芯片化（IP as a Chiplet）"、"芯片平台化（Chiplet as a Platform）"和"平台生态化（Platform as an Ecosystem）"理念为行动指导方针，从接口 IP、Chiplet 芯片架构、先进封装技术、面向 AIGC 和智慧出行的解决方案等方面入手，持续推进公司 Chiplet 技术、项目的发展和产业化。"**
https://stock.stockstar.com/notice/SN2026081800000904.shtml　http://static.cninfo.com.cn/finalpage/2026-03-31/1225060349.PDF　https://static.cninfo.com.cn/finalpage/2025-04-26/1223311717.PDF

### 4.2 接口 IP 产品组合（2025 年报 / 2026 半年报）
【交易所公告】**"芯原已拥有完善的接口 IP 产品组合，可提供端到端完整解决方案，核心产品涵盖 UCIe 物理层及控制器 IP、USB 物理层及控制器 IP、MIPI D-PHY/C-PHY/A-PHY 物理层及控制器 IP、HDMI 物理层及控制器 IP、DP/eDP 物理层 IP，以及 PCIe 物理层与高速 SerDes IP。目前，绝大部分接口 IP 已成功导入国内外客户多款产品，实现大规模量产落地，助力客户在低能耗智能家居、可穿戴设备、车载高分辨率视频传输、Chiplet 高性能计算等领域实现商业化。"**
https://stock.stockstar.com/notice/SN2026081800000904.shtml　http://static.cninfo.com.cn/finalpage/2026-03-31/1225060349.PDF（第 32 页）

### 4.3 Chiplet 领域"切实成果"清单（2026 半年报，逐字）
【交易所公告】**"公司在 Chiplet 与先进封装领域已取得多项实质性产业化成果。在芯片定制端，公司成功交付基于 Chiplet 架构的 Chromebook 芯片设计，通过 SiP 技术实现高性能 SoC 与多颗 IPM 内存的异构合封；并顺利完成客户 AIGC 芯片的 2.5D CoWoS 封装设计。在核心 IP 与生态构建端，公司自主研发的面向 Die-to-Die 连接的 UCIe 物理层接口 IP 已成功通过流片验证并获客户项目导入；同时，公司向 Chiplet 解决方案的行业头部客户授权 GPGPU、NPU 及 VPU 等多款自有核心处理器 IP，全面赋能其在数据中心、高性能计算及汽车领域的高性能人工智能芯片部署。"**
https://stock.stockstar.com/notice/SN2026081800000904.shtml

### 4.4 2025 年报对口表述（对比：2024 年报尚无"获客户项目导入"）
【交易所公告】2025 年报：**"已设计研发了针对 Die to Die 连接的 UCIe 物理层接口，并已顺利完成流片测试，相关技术已被客户采纳将用于项目开发"**；2024 年报：**"相关测试芯片已流片，即将返回进行封装和测试"**。→ 显示 2024→2025 完成回片测试、2026H1 进入客户项目导入。
http://static.cninfo.com.cn/finalpage/2026-03-31/1225060349.PDF　https://static.cninfo.com.cn/finalpage/2025-04-26/1223311717.PDF

### 4.5 UCIe 联盟身份（2025 年报 / 2026 半年报，逐字）
【交易所公告】**"以 RISC-V International 金牌会员、OpenHW Group 理事会成员、Chips Alliance 理事会成员、UCIe 联盟贡献者成员等多重身份，深入参与前沿技术演进与行业标准制定。"**
https://stock.stockstar.com/notice/SN2026081800000904.shtml　http://static.cninfo.com.cn/finalpage/2026-03-31/1225060349.PDF

### 4.6 2025 年报"未来展望"原文
【交易所公告】**"芯原是中国首批加入 UCIe 产业联盟的企业之一。基于"IP 芯片化、芯片平台化、平台生态化"的路径，公司将持续深化在 AIGC 大数据处理与高端智驾两大赛道的领先布局。依托自主研发的 UCIe 物理层接口 IP、2.5D/CoWoS、3D 堆叠、面板级封装等先进封装项目经验，以及与产业链伙伴的深度合作，目标是在 AIGC 和智慧出行领域率先实现 Chiplet 解决方案的规模化落地。"**
http://static.cninfo.com.cn/finalpage/2026-03-31/1225060349.PDF（第 96 页）

### 4.7 较早的联盟加入公告（2022-04-02）
【官方】**"VeriSilicon… today announced it has officially joined the Universal Chiplet Interconnect Express (UCIe) industry consortium. VeriSilicon is one of the first enterprises in mainland China to join the organization…"**；同文："Under the concepts of 'IP as a Chiplet' and 'Chiplet as a Platform', VeriSilicon has launched a high-end application processor platform based on chiplet architecture. To date, the 12nm SoC version has been taped out and verified, and the upgraded chiplet-based version is in progress."
https://www.verisilicon.com/en/PressRelease/VeriSiliconJoinsUCIe

### 4.8 投资者关系活动记录表（未见 UCIe 专门问答）
【交易所公告】2026-07-16 电话会议记录：问答集中在先进封装布局（"在持续支持 CoWoS 等现有方案的同时，公司前瞻性布局了面板级封装技术（Panel-Level Packaging，PLP）"）、新签订单（年初至 7/16 合计近 150 亿元）；**未涉及 UCIe IP 技术细节或客户**。
http://dataclouds.cninfo.com.cn/shgonggao/investor/2026/20260720/0e01cf58e23f454ae51ece6011384367.pdf

【交易所公告】2026-04-09 公开业绩说明会记录：董事长戴伟民出席；问答涉及港股上市进展、定价与利润率、端侧 AI 与谷歌合作；**未涉及 UCIe**。
https://www.9fzt.com/detail/sh_688521_9_a78c976005a749e6a9208917b803becd.html

【媒体】2026-08-28 投资者关系活动记录表可检索到标题，但正文为 JS 渲染、未能取得全文，**其 UCIe 相关内容未证实**。
https://www.cnfin.com/announ/detail/index.html?id=841505809233&code=688521&dannoun=stibdetail&announ=stib

### 4.9 招股说明书（H 股申请版本）
【招股书】芯原于 2026-04-01 向港交所递交 H 股上市申请并同日刊发申请资料；定义章节将 UCIe 释义为"Chiplet 晶粒间互连"；接口 IP 章节称产品组合涵盖 MIPI、USB、HDMI、PCIe 等标准"并已成功移植至 4 纳米及 8 纳米 FinFET 等先进工艺节点"。
https://www1.hkexnews.hk/app/sehk/2026/108392/documents/sehk26040104032_c.pdf（英文版 sehk26040104033.pdf）

---

## 5. OISA 及其他国产互联标准中的角色

5.1 【未证实】**未找到芯原参与 OISA 的任何公开证据。** OISA 官网"生态伙伴"页面约 60 个成员 logo 中不含芯原（在列的有中国移动、中兴、浪潮信息、联想、海光、昆仑芯、燧原、天数智芯、飞腾、锐捷、烽火、立讯、奇异摩尔、曦智科技、曦望、云脉芯联、云豹、亿铸科技、众星微、太初元芯、汤谷智能等）；站内搜索"芯原"返回 **"No 文章 found!"**。
https://www.oisa.org.cn/　https://www.oisa.org.cn/index.php/about/（相关成员页）　https://www.oisa.org.cn/?s=芯原

5.2 【标准组织官网/媒体】**线索更正：OISA ≠ "Open Interconnect Standard Alliance"。** OISA 全称为 **全向智感互联**（Omni-directional Intelligent Sensing Interconnect），由中国移动研究院主导的开放 GPU 卡间互联技术体系。时间线：2024-10-09 发布 OISA Gen1.1 → 2025-08-23 发布 OISA 2.0 协议 → 2026-05-08 发布商用级 OISA IP Core。官方称"OISA 协同创新平台已汇聚超过 50 家产业链合作伙伴"。
https://www.oisa.org.cn/　https://www.c114.com.cn/news/118/a1309911.html

5.3 【未证实】**未找到芯原参与 UALink、UEC（超以太网联盟）或 CCITA /《芯粒测试规范》等国产小芯片接口总线标准的公开证据。** 芯原官方文件仅在行业趋势段落"提及"这些标准，未声明成员身份。
https://stock.stockstar.com/notice/SN2026081800000904.shtml

5.4 【交易所公告】芯原公开列示的标准组织身份**仅有 4 项**：RISC-V International 金牌会员、OpenHW Group 理事会成员、Chips Alliance 理事会成员、UCIe 联盟贡献者成员；此外牵头成立中国 RISC-V 产业联盟（CRVIC）与上海开放处理器产业创新中心（SOPIC）。
https://stock.stockstar.com/notice/SN2026081800000904.shtml

5.5 【标准组织官网】UCIe 官网仅公开列示 **Promoter Members**（AMD、ASE、Alibaba Cloud、Arm、Google Cloud、Intel、Meta、Microsoft、NVIDIA、Qualcomm、Samsung、TSMC）12 家；Contributor / Adopter 成员页为 Wix 动态渲染，**无静态可抓取名单**，故无法由官网直接复核芯原的 Contributor 身份——该身份以芯原年报自述为准（见 4.5）。
https://www.uciexpress.org/membership　https://www.uciexpress.org/adopter-members

---

## 6. 一站式 Chiplet 方案与 D2D SerDes

6.1 【官方】平台框架：芯原以 **SiPaaS（芯片设计平台即服务）** 为商业模式，Chiplet 侧以"IP 芯片化 / 芯片平台化 / 平台生态化"为方针；官方业务表述为"面向云端 AIGC 计算和智慧驾驶应用提供基于 Chiplet 架构的芯片设计软、硬件解决方案"。
https://www.verisilicon.com/cn/IPPortfolio/UCIe-SP　https://stock.stockstar.com/notice/SN2026081800000904.shtml

6.2 【官方】一站式能力：芯原"拥有从先进 4nm FinFET、22nm FD-SOI 到传统 250nm CMOS 制程的设计能力"，"拥有 14nm/10nm/7nm/6nm/5nm/4nm FinFET 和 28nm/22nm FD-SOI 工艺节点芯片的成功流片经验"。
http://static.cninfo.com.cn/finalpage/2026-03-31/1225060349.PDF

6.3 【官方】先进封装布局：CoWoS（已交付客户 AIGC 芯片 2.5D CoWoS 封装设计）+ 3D 堆叠 + **面板级封装 PLP**（"已针对新一代面板级封装（Panel level package）技术进行了先行设计开发"）+ SiP（Chromebook 芯片 SoC 与多颗 IPM 内存合封）+ HB-PoP（AI/AR 眼镜 SoC 平台，三维封装）。
https://stock.stockstar.com/notice/SN2026081800000904.shtml

6.4 【交易所公告/官方】**112G SerDes 存在性**：2026 半年报"高性能 AIGC 平台"章节原文 **"其配备了关键的高速系统互连技术，包括 PCIe 5.0、56G/112G SerDes 以及 UCIe，以确保在高度复杂的 AIGC 系统内实现高效的数据传输效能与架构可扩展性"**。→ 112G SerDes 作为平台级能力被明确提及。
https://stock.stockstar.com/notice/SN2026081800000904.shtml

6.5 【未找到公开数据】**未找到 112G SerDes 独立 IP 产品页**。官网 SerDes 栏目仅列 **"1.25 Gbps–12.5 Gbps SerDes"** 与 **"1.25 Gbps–16 Gbps SerDes"** 两个具名产品；PCIe PHY 仅到 PCIe 4.0。
https://www.verisilicon.com/cn/IPPortfolio/InterfaceIP

6.6 【交易所公告】自研高速 SerDes 研发项目：**HSS_DSP_SERDES**（2024 年报"架构设计已经完成"→ 2025 年报"已经完成流片"→ 2026 半年报"进行封装和测试 PCB 制板。第二版改进版正在进行中，并计划于年底完成流片"）；应用方向含"Chiplet 应用、数据中心、ADAS 和自动驾驶、AIGC 相关应用"。
https://stock.stockstar.com/notice/SN2026081800000904.shtml　http://static.cninfo.com.cn/finalpage/2026-03-31/1225060349.PDF

6.7 【交易所公告】接口 IP 工艺覆盖：2026 半年报 **"该产品组合涵盖 MIPI、USB、HDMI 及 PCIe 等核心行业标准，并已成功移植至 4nm 及 8nm FinFET 等先进工艺节点"**（与 UCIe PHY "8nm 版本已完成流片"相互印证）。
https://stock.stockstar.com/notice/SN2026081800000904.shtml

6.8 【官方】汽车/智驾侧 Chiplet 平台：**"目前，公司正在积极推进智慧出行领域 Chiplet 解决方案平台研发"**；**"下一代自动驾驶 SoC 平台将基于更前沿的工艺节点，并通过引入 PCIe 与 UCIe 等高速接口技术实现增强的连接能力与数据带宽。"**
https://stock.stockstar.com/notice/SN2026081800000904.shtml

---

## 附：三条"高风险线索"的核实结论

| 线索 | 结论 | 依据 |
|---|---|---|
| "VeriHealth 是芯原 D2D/Chiplet IP 产品" | **不成立** | VeriHealthi（芯健康）为健康监测方案，见 1.5 |
| "Glink 是芯原产品" | **不成立** | Glink-3D 属 GUC 创意电子，见 1.6 |
| "芯原参与 OISA（开放互联标准联盟）" | **未证实** | OISA 全称/成员均不含芯原，见 5.1、5.2 |
| "芯原 UCIe IP 支持 UCIe 2.0" | **未找到公开数据** | 官方口径仅为 UCIe 1.1，见 2.1、2.2 |
| "uCiexpress.org 成员列表可确认芯原" | **不可直接复核** | Contributor 页动态渲染，见 5.5 |
