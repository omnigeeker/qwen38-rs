# 中科曙光（603019）超节点 / 海光 DCU 互联体系 事实核查清单

## 1. 中科曙光 scaleX640 超节点

- 单机柜 **640 卡**，为"全球首个单机柜级 640 卡超节点"；采用"一拖二"高密架构设计，实现单机柜 640 卡超高速总线互连，构建大规模、高带宽、低时延超节点通信域，双 scaleX640 超节点组成 **1280 卡计算单元**，柜间通过高速网络互连。—— 【官方】来源：https://www.sugon.com/x640
- 发布方与时间：中科曙光于 **2025 年 11 月 6 日**（2025 世界互联网大会乌镇峰会期间）正式发布 scaleX640，并在"互联网之光"博览会首次亮相。—— 【媒体】（转述官方发布）来源：http://www.xinhuanet.com/enterprise/20251106/0973cad6baa74c61a421f39165d98e70/c.html
- "一拖二"的具体构成：1 台液体冷凝换热装置 CDM + 2 台计算机柜；CDM 为千卡级计算单元提供高达 **1.72MW** 散热能力。—— 【券商研报】来源：https://pdf.dfcfw.com/pdf/H3_AP202604141821177994_1.pdf
- 卡间互联拓扑（Scale Up 方式）：采用**超高速正交架构**，即计算节点与交换节点垂直交叉部署，以保证集成密度，并"消除背板信号衰减瓶颈"。—— 【券商研报】来源：https://pdf.dfcfw.com/pdf/H3_AP202604141821177994_1.pdf
- 互联协议（Scale Out 方式）：scaleX640 **发布时仍采用 InfiniBand 协议**，后伴随 scaleFabric 发布改用基于 RDMA 架构全栈自研的 400G 无损高速网络。—— 【券商研报】来源：https://pdf.dfcfw.com/pdf/H3_AP202604141821177994_1.pdf
- 单机柜总算力："单机柜总算力超 **600PFLOPs**"，算力密度较业界同类产品最大提升 20 倍。—— 【券商研报】来源：https://4g.stockstar.com/detail/JC2026041400024578
- 更具体的单机柜指标（国海证券 2025-11-10 研报）：**FP16 算力规模达 630PFlops，HBM 总容量 81.9TB，HBM 总带宽 2304TB/s，片间互连总带宽 573TB/s**。—— 【券商研报】来源：http://stock.finance.sina.com.cn/stock/go.php/vReport_Show/kind/search/rptid/816086178548/index.phtml
- 卡间互连总带宽（东吴证券口径）：单机柜"访存总带宽超 **2.3PB/s**、卡间互连总带宽超 **570TB/s**"。—— 【券商研报】来源：https://pdf.dfcfw.com/pdf/H3_AP202604141821177994_1.pdf
- **每卡互联带宽：未找到公开数据（scaleX640 单卡互连带宽）**。所有已查到的一手/研报来源只公布"单机柜 640 卡的卡间互连总带宽"（573TB/s 或"超 570TB/s"），无任何来源给出单卡 GB/s 或 Gbps 数值，故不做换算。—— 综合上述来源
- 效率与可靠性：大模型训练/推理性能提升 **30%-40%**，PUE **＜1.04**，通过 **100+ 项 RAS** 多级架构设计与 **30 天+** 长稳测试，可支撑 **10 万卡级**超大规模 AI 集群扩展。—— 【券商研报】来源：https://pdf.dfcfw.com/pdf/H3_AP202604141821177994_1.pdf
- 开放性：基于 AI 计算开放架构，硬件适配多品牌 AI 加速卡（含海光 DCU），软件兼容主流生态，适配优化 **400+ 主流大模型**。—— 【官方】来源：https://www.sugon.com/x640
- 交付/落地时间线：以 640 卡超节点为基础的 scaleX 万卡超集群，于 **2026 年 2 月**在国家超算互联网核心节点实现"同步建设、同步上线、同步对外提供服务"。—— 【年报/交易所公告】来源：https://stock.stockstar.com/notice/SN2026041400039000.shtml
- 中科曙光在 2025 年年报摘要中明确表述："目前，公司超节点、超集群产品**尚未在市场上实现大规模部署与应用**"，后续将推进商业化落地。—— 【年报/交易所公告】来源：https://stock.stockstar.com/notice/SN2026041400039000.shtml
- 万卡集群总量数据：scaleX 万卡超集群由多个 scaleX640 超节点（单机柜 640 卡）与 scaleFabric 高速网络互连，总计 **10240 块加速卡**，总算力超 **5EFlops**；HBM 总容量超 **650TB**，总带宽超 **18PB/s**；**片间互连总带宽超 4.5PB/s**，柜间互连总带宽超 500TB/s。—— 【媒体】来源：https://m.mydrivers.com/newsview/1094106.html
- 后续规模：scaleX 已完成 **6 万卡级** AI4S 计算集群部署（2026 世界智能产业博览会，天津，2026 年 5 月 28 日报道）。—— 【媒体】来源：https://m.jiemian.com/article/14501095.html

## 2. 中科曙光 scaleFabric 400G 网络

- 发布方与时间：**中科曙光**于 **2026 年 3 月 12 日**（河南郑州）正式发布首款全栈自研 400G 无损高速网络 scaleFabric，官方定位为"国内首款原生 RDMA 架构的高端网络产品"。—— 【官方】来源：https://www.sugon.com/cut?id=2829
- 全栈自研范围："从底层的 **112G SerDes IP**、硬件设备到上层的管理软件实现 **100% 自主研发**"；核心关键 IP、交换芯片、网卡、交换机、驱动与管理软件均自研。—— 【官方】来源：https://www.sugon.com/cut?id=2829
- 物理层 SerDes：基于 **PAM4 架构的自研 112G SerDes IP**，是 400G/800G 端口速率与低时延传输的核心基础。—— 【券商研报】来源：https://pdf.dfcfw.com/pdf/H3_AP202604141821177994_1.pdf
- 交换芯片：自研交换芯片实现 **64Tbps 双向吞吐**，可支持 80 个 400G 端口或 40 个 800G 端口配置。—— 【券商研报】+【官方】（官方称"整机交换容量可达双向 64Tbps，支持 800G×40 或 400G×80"）来源：https://pdf.dfcfw.com/pdf/H3_AP202604141821177994_1.pdf ；https://www.sugon.com/cut?id=2829
- 网卡时延：scaleFabric400 网卡基于 **PCIe 5.0** 接口，端口带宽 **400Gbps**，端到端通信时延低至 **0.9 微秒**（东吴研报口径为"网卡芯片端到端时延低于 1us"）。—— 【官方】来源：https://www.sugon.com/cut?id=2829
- 交换机时延：单端口带宽 **800Gbps**，交换时延约 **260 纳秒**。—— 【官方】来源：https://www.sugon.com/cut?id=2829
- 官方网卡产品页口径："低至 **1us** 的点对点延迟"，基于 Credit 流控机制，并称其为"国内首个采用全国产化芯片的 400G 原生无损 RDMA 高速网卡"。—— 【官方】来源：https://www.sugon.com/product/storage/product_list?cate=288
- 官方交换机产品页口径："国内首款量产 **800G** 原生无损 RDMA 交换机"，"核心技术高速 **112G Serdes IP** 全自研"，支持 40×800G 或 80×400G 端口。—— 【官方】来源：https://www.sugon.com/product/storage/product_list?cate=291
- 协议兼容性：scaleFabric400 交换机与网卡**均支持 InfiniBand 协议**（网卡采用 QSFP_112 连接器）；东吴研报称 2U 风冷交换机为"国内首个采用全国产化芯片的一款 InfiniBand 交换机"。—— 【券商研报】+【官方】来源：https://pdf.dfcfw.com/pdf/H3_AP202604141821177994_1.pdf ；https://www.sugon.com/product/storage/product_list?cate=291
- 稳定性与扩展：采用基于信用的无损流控机制，链路故障恢复时间 **<1 毫秒**；相比英伟达 NDR，交换机端口密度提升 **25%**、网卡最大 QP 数支持提升 **100%**、单子网互连规模是传统 IB 的 **2.33 倍**，**可轻松支持最大 11.4 万卡集群部署**，网络总成本可降低 **30%**。—— 【官方】来源：https://www.sugon.com/cut?id=2829
- 落地验证：已部署于**国家超算互联网郑州核心节点**，支撑三套万卡级 scaleX 智算集群上线运行，总规模达 **3 万卡**，官方称"已支撑近万卡集群持续稳定运行验证超 10 个月"。—— 【官方】来源：https://www.sugon.com/cut?id=2829
- 技术路线表态（中科曙光高级副总裁李斌）：scaleFabric 定位面向超大规模紧耦合算力系统；未来会探索"在原生 RDMA 上面做对 RoCE 的兼容"，网络接口是标准的，可与不同计算芯片互联。—— 【媒体】来源：https://finance.sina.com.cn/stock/relnews/cn/2026-03-17/doc-inhrhicv8232777.shtml
- 6 万卡集群口径（科技日报）：郑州 6 万卡超智融合算力集群于 2026 年 4 月 28 日接入全国一体化算力网，搭载 scaleX 系列超节点 + scaleFabric 高速网络，scaleFabric 实现 **400Gbs 超高带宽和低于 1 微秒的通信延迟**。—— 【媒体】来源：https://www.stdaily.com/web/gdxw/2026-04/29/content_510132.html

## 3. 中科曙光 scaleX40 超节点

- 发布方与时间：**2026 年 3 月 26 日**（2026 中关村论坛年会期间），中科曙光推出"世界首个无线缆箱式超节点 scaleX40"，**同步开启预售**。—— 【媒体】（北京市科委官网转载北京青年报）来源：https://kw.beijing.gov.cn/ztzl/2026zgclt/2026zgcltmtjj/202603/t20260326_4567044.html
- 架构：采用**正交无线缆一级互连架构**，实现计算节点与交换节点直接对插，消除线缆带来的性能损耗与运维风险；采用标准 **19 英寸箱式设计**，实现算力单元与机柜解耦，部署周期从数月级压缩至数小时，系统可靠性达 **99.99%**。—— 【官方】+【券商研报】来源：https://www.sugon.com/product/storage/product_detail?id=569 ；https://pdf.dfcfw.com/pdf/H3_AP202604141821177994_1.pdf
- 算力与显存：单节点集成 **40 张 AI 加速卡**，总算力超过 **28PFLOPS（FP8 精度）**，**HBM 总显存超过 5.62TB**（官方产品页口径）。—— 【官方】来源：https://www.sugon.com/product/storage/product_detail?id=569
- 一级全互连：算力芯片通过一级互联技术实现 Scale-up 全互连，"真正实现内存语义、极低延迟和高扩展性"。—— 【官方】来源：https://www.sugon.com/product/storage/product_detail?id=569
- 关键互联参数（官方产品参数表，scaleX40-3G）：交换节点（4 个）采用**海光 HySW 交换芯片**，支持 **40 张加速卡一级 CLOS 全互联**，**任意两卡间 P2P 互联带宽 448GB/s**。—— 【官方】来源：https://www.sugon.com/product/storage/product_detail?id=569
- 计算节点规格（官方参数表）：计算节点 10 个，每节点 1 个 C86-4G/H CPU + 4 张加速卡；内存最大支持 12 个 DDR5 6400MT/s 插槽。—— 【官方】来源：https://www.sugon.com/product/storage/product_detail?id=569
- 系统规格（官方参数表）：适配 19 英寸机柜，支持单柜单 PoD 或单柜双 PoD；冷板液冷+风冷；单 PoD 系统功耗 <45kW，单 PoD 重量 <450kg。—— 【官方】来源：https://www.sugon.com/product/storage/product_detail?id=569
- **数值冲突（访存总带宽）**：东吴证券研报写"访存总带宽超 **80TB/s**"，北京青年报/北京市科委官网转载稿写"访存总带宽超过 **8TB/s**"。两者相差 10 倍，未能找到第三方权威口径裁决，仅并列记录。—— 【券商研报】来源：https://pdf.dfcfw.com/pdf/H3_AP202604141821177994_1.pdf ；【媒体】来源：https://kw.beijing.gov.cn/ztzl/2026zgclt/2026zgcltmtjj/202603/t20260326_4567044.html
- **数值冲突（HBM 总显存）**：东吴证券研报与北京青年报稿均写"超 5TB"，官方产品页写"HBM 总显存超过 **5.62TB**"，取官方口径为准。—— 【官方】来源：https://www.sugon.com/product/storage/product_detail?id=569
- 扩展能力：具备纵向 Scale Up 构建**百卡级**超节点、横向 Scale Out 搭建**千卡级**集群的双路径扩展能力。—— 【券商研报】来源：https://pdf.dfcfw.com/pdf/H3_AP202604141821177994_1.pdf

## 4. 海光 DCU（深算系列）在曙光超节点中的互联角色 / "HSL+IB"含义

- 分层体系定位：海光 DCU 为 scaleX640、scaleX40 超节点提供自主可控的算力核心选择；其在 **Chiplet 互联、ComboPHY 高速 I/O 接口、HSL** 等**芯片级**互联技术上的积累，与曙光 **scaleFabric 高速网络**形成"从芯片内到系统间的全栈互联协同"。—— 【券商研报】来源：https://pdf.dfcfw.com/pdf/H3_AP202604141821177994_1.pdf
- HSL 定义（海光信息年报原文）："HSL（HygonSystemLink）是海光信息推出的开放、高带宽、低延迟的互联总线协议规范，主要用于连接海光 CPU 与各类 xPU（如 GPU、AI 加速芯片等），实现高效的数据交互和协同计算。"—— 【年报/交易所公告】来源：海光信息 2025 年年度报告（http://dataclouds.cninfo.com.cn/shgonggao/hsomarket/2026/20260407/18000ad88d1a430f8957dd2348548e74.PDF ；工作区文本 /Users/waynewong/dsh/QWen3.8-27B/stock/hygon/hygon_2025_annual.txt 第 234-238 行）
- HSL 发布时间（海光信息年报原文）："2025 年 9 月，公司正式发布 HSL（HygonSystemLink）系统总线互联协议"，目标是实现 xPU、IO、OS、OEM 等厂商与海光 CPU 的"紧耦合"高速互联。—— 【年报/交易所公告】来源：海光信息 2025 年年度报告（同上一文件，约第 674 行）
- HSL 的核心内容（海光信息年报原文）："开放完整的总线协议、提供 IP 参考设计、开放指令集"；伙伴可实现降低访问延迟、自定义简化协议栈、提升链路利用率、**支持缓存一致性和自由的多链路扩展**。—— 【年报/交易所公告】来源：海光信息 2025 年年度报告
- HSL 1.0 规范：2025 年 12 月 18 日（光合组织 2025 人工智能创新大会），海光携手国产 AI 芯片、操作系统、存储与网络模块等伙伴**共同发布 HSL 1.0 规范，并公布未来三年的开放路线图**；HSL 1.0 涵盖完整总线协议栈、IP 参考设计及指令集。—— 【媒体】来源：https://m.ithome.com/html/906914.htm
- HSL 速率与性能（东吴证券口径）："当前起步速率 **112G**，后续将升级至 **240G**，带宽较 32G PCIe 提升 **8 倍**，时延较 PCIe 降低约一半。"—— 【券商研报】来源：https://pdf.dfcfw.com/pdf/H3_AP202604141821177994_1.pdf
- HSL 生态规模：海光已联合 **10 余家厂商**共建生态，全层级互联覆盖芯片、板卡、服务器等任意层级，支持全局地址空间与缓存一致性。—— 【券商研报】来源：https://pdf.dfcfw.com/pdf/H3_AP202604141821177994_1.pdf
- 海光官方口径："公司推出'双芯战略'，通过发布 HSL 总线互联协议、共建 AI 软件栈体系两大举措"；截至报告期末光合组织已聚合 **6,000 余家**合作伙伴。—— 【年报/交易所公告】来源：海光信息 2025 年年度报告
- **"HSL+IB"的准确含义**：东吴研报标题《超节点系列报告二：海光&曙光系超节点，HSL+IB 构建最全互连体系》中，**HSL 指芯片级 Scale Up 互联总线协议**（海光 DCU/CPU 与各 xPU 的紧耦合），**"IB"指网络级 Scale Out 的 InfiniBand（兼容）RDMA 网络**——即 scaleX640 发布时先用 InfiniBand，后改用自研 scaleFabric（scaleFabric 交换机与网卡均支持 InfiniBand 协议）。研报正文并未给出"HSL+IB"的逐字定义句，此为按原文上下文的对应关系整理。—— 【券商研报】来源：https://pdf.dfcfw.com/pdf/H3_AP202604141821177994_1.pdf
- 研报正文原话："海光 DCU 为曙光超节点提供自主可控算力核心，其芯片级互联技术 HSL 与曙光 scaleFabric 高速网络实现全栈互联协同，推动国产智算平台迈向系统领先。"—— 【券商研报】来源：https://4g.stockstar.com/detail/JC2026041400024578
- 海光 DCU 侧与超节点的配合：DCU 支持**内存语义的 ScaleUP 互联网络芯片**；2025 年公司发布自研软件栈升级并全面开放，"为超节点及分布式训练、推理提供软硬件耦合支撑"。—— 【年报/交易所公告】来源：海光信息 2025 年年度报告
- 中科曙光官方表态（描述 scaleX640）："曙光 scaleX640 超节点采用 AI 计算开放架构，支持市场上主流的国产和国际 AI 加速卡。"—— 【媒体】（证券时报，引述中科曙光高管）来源：https://stcn.com/article/detail/3534419.html

## 5. 公开拓扑图描述与每卡带宽数据 / 最大集群规模

- **公开拓扑图情况**：东吴证券研报包含三张架构图——"图 4：scaleX640 架构图以及产品亮点""图 5：scaleFabric 整体解决方案""图 6：scaleX40 架构图以及产品亮点"，均为**图片形式**，PDF 文本层无可提取的拓扑文字描述。文字层面可确认的拓扑描述仅有两处：scaleX640 的"超高速正交架构（计算节点与交换节点垂直交叉部署）"，以及 scaleX40 的"正交无线缆一级互连架构 / 40 卡一级 CLOS 全互联"。—— 【券商研报】来源：https://pdf.dfcfw.com/pdf/H3_AP202604141821177994_1.pdf
- Scale Up 域内带宽（scaleX640，单机柜 640 卡）：卡间互连总带宽超 **570TB/s**（东吴）/ **573TB/s**（国海）；访存总带宽超 **2.3PB/s**。—— 【券商研报】来源：https://pdf.dfcfw.com/pdf/H3_AP202604141821177994_1.pdf ；http://stock.finance.sina.com.cn/stock/go.php/vReport_Show/kind/search/rptid/816086178548/index.phtml
- Scale Up 域内**每卡**带宽（scaleX640）：**未找到公开数据**。仅 scaleX40 公布了单卡口径的 Scale Up 带宽——任意两卡间 P2P 互联带宽 **448GB/s**（官方产品参数表）。—— 【官方】来源：https://www.sugon.com/product/storage/product_detail?id=569
- Scale Out 带宽（每卡对应）：scaleFabric400 标准网卡**单端口 400Gbps**，端到端时延 **~0.9us**；交换机单设备带宽 **400/800Gbps**，端到端交换时延 **~260ns**。—— 【券商研报】+【官方】来源：https://pdf.dfcfw.com/pdf/H3_AP202604141821177994_1.pdf ；https://www.sugon.com/cut?id=2829
- Scale Out 协议切换时间点：scaleX640 发布（2025-11-06）时采用 InfiniBand；scaleFabric 发布（2026-03-12）后适配自研 400G RDMA 网络。—— 【券商研报】来源：https://pdf.dfcfw.com/pdf/H3_AP202604141821177994_1.pdf
- 最大集群规模（官方）：scaleX640 通过 30 天+长稳运行可靠性测试验证，"可保障 **10 万卡级**超大规模集群扩展部署"。—— 【官方】来源：https://www.sugon.com/x640
- 最大集群规模（网络侧）：scaleFabric 单子网互连规模是传统 IB 的 2.33 倍，"可轻松支持最大 **11.4 万卡**集群部署"。—— 【官方】来源：https://www.sugon.com/cut?id=2829
- 集群级冗余/优化技术：万卡超集群采用"超级隧道"三级协同优化（芯片级/系统级/应用级），通过 BurstBuffer、XDS 等技术，大模型训推效率提升 30-40%，GPU 利用率提升最多 55%；集群长期可用性 99.99%，平均每 30 天不可用时间小于 4 分钟。—— 【媒体】来源：https://m.mydrivers.com/newsview/1094106.html
- 研发/量产节奏（券商观点）：华龙证券 2026-04-14 报告称国内超节点"量产+迭代"并进，2026 年有望成为国产超节点放量元年；同期中国移动 2026-2027 年人工智能超节点设备集采（CANN 生态）规模为 6208 张 AI 加速卡、折合 776 套计算节点设备。—— 【券商研报】来源：https://www.hlzq.com/upload1/yb/20260414/1776140111991.pdf

## 6. 中科曙光与海光信息吸收合并/重组进展及影响

- 交易方案：2025 年 5 月 26 日起停牌筹划；2025 年 6 月 6 日董事会通过预案，拟由**海光信息向中科曙光全体 A 股换股股东发行 A 股股票的方式换股吸收合并中科曙光并募集配套资金**，构成关联交易与重大资产重组；2025 年 6 月 10 日复牌。—— 【年报/交易所公告】来源：https://paper.cnstock.com/html/2025-12/10/content_2155669.htm
- 交易对价参数：换股比例 **1∶0.5525**；海光信息换股价格 **143.46 元/股**，中科曙光换股吸收合并定价 **79.26 元/股**；海光信息为本次换股吸收合并发行股份合计 **8.08 亿股**，参与换股的中科曙光总股本 **14.63 亿股**；交易前中科曙光持有海光信息约 6.5 亿股（27.96%），交易完成后该等股份将择机注销。—— 【媒体】（证券时报）来源：https://stcn.com/article/detail/3534419.html
- 终止结果：**2025 年 12 月 9 日**中科曙光第五届董事会第二十六次会议审议通过《关于公司终止重大资产重组的议案》（同意 6 票、反对 0 票、弃权 0 票，关联董事历军回避表决）；**2025 年 12 月 10 日**披露《关于终止重大资产重组的公告》（公告编号：2025-071）。—— 【年报/交易所公告】来源：https://paper.cnstock.com/html/2025-12/10/content_2155669.htm
- 终止原因（公告原文）："由于本次交易规模较大、涉及相关方较多，使得重大资产重组方案论证历时较长，目前市场环境较本次交易筹划之初发生较大变化，本次实施重大资产重组的条件尚不成熟"。—— 【年报/交易所公告】来源：https://paper.cnstock.com/html/2025-12/10/content_2155669.htm
- 终止影响（公告原文）："本次重大资产重组的终止不影响双方后续的持续合作，中科曙光将与海光信息在系统级产品应用上建立更加紧密的合作关系。后续中科曙光仍将继续围绕高端计算机核心业务，在**超节点智算算力**、科学大模型开发平台、超集群系统等前沿技术突破的基础上，持续在智能计算、算力调度、数据中心解决方案等领域开展全栈布局。"—— 【年报/交易所公告】来源：https://paper.cnstock.com/html/2025-12/10/content_2155669.htm
- 承诺事项：公司承诺自终止本次交易事项披露之日起**至少 1 个月内**不再筹划重大资产重组事项。—— 【年报/交易所公告】来源：https://paper.cnstock.com/html/2025-12/10/content_2155669.htm
- 管理层对终止的解释（中科曙光董事、总经理历军）："不合并能让两家企业成为国产算力产业的'双核心'，而非单一龙头"；两家公司保持独立的市场化运作与专业化发展路径，分别聚焦算力基础设施集成和高端芯片设计的核心赛道。—— 【媒体】（证券时报）来源：https://stcn.com/article/detail/3534419.html
- 海光信息侧表态（董事、总经理沙超群）：海光信息将与中科曙光"在系统级产品应用上建立更加紧密的合作关系，充分发挥中科曙光在**超节点算力**、科学大模型开发平台、集群系统等前沿技术的优势"。—— 【媒体】（证券时报）来源：https://stcn.com/article/detail/3534419.html
- 对超节点方案的净影响：**未找到公开数据（重组的终止对曙光/海光系超节点具体技术路线或产品规划的量化影响）**。可确认的是：双方公开表述均为"保持上市公司独立性 + 深化战略协同"，且 scaleX640/scaleX40/scaleFabric 三项产品发布与落地（2025-11 至 2026-05）均发生在重组终止公告（2025-12-10）前后，产品线未出现公开的路线变更。—— 综合上述来源

## 附：本次核查使用的主要来源

- 东吴证券《超节点系列报告二：海光&曙光系超节点，HSL+IB构建最全互连体系》全文 PDF（2026-04-14，11 页，已下载并转文本）：https://pdf.dfcfw.com/pdf/H3_AP202604141821177994_1.pdf
- 证券之星研报摘录页：https://4g.stockstar.com/detail/JC2026041400024578
- 新浪财经研报页：http://vip.stock.finance.sina.com.cn/q/go.php/vReport_Show/kind/lastest/rptid/829497178598/index.phtml
- 水滴研报全文页（含研报原文前 10 页）：http://www.sdyanbao.com/detail/953058
- 中科曙光官网：智算超节点 https://www.sugon.com/x640 ；scaleX40 https://www.sugon.com/product/storage/product_detail?id=569 ；scaleFabric 发布稿 https://www.sugon.com/cut?id=2829 ；液冷交换机 https://www.sugon.com/product/storage/product_list?cate=291 ；网卡 https://www.sugon.com/product/storage/product_list?cate=288
- 海光信息 2025 年年度报告（工作区文本）：/Users/waynewong/dsh/QWen3.8-27B/stock/hygon/hygon_2025_annual.txt
- 中科曙光 2025 年年度报告摘要：https://stock.stockstar.com/notice/SN2026041400039000.shtml
- 中科曙光终止重大资产重组公告（上海证券报）：https://paper.cnstock.com/html/2025-12/10/content_2155669.htm
- 证券时报：海光信息与中科曙光分道扬帆 双双回应终止重组原因：https://stcn.com/article/detail/3534419.html
- 国海证券《中科曙光：SCALEX640全球领先发布》（2025-11-10）：http://stock.finance.sina.com.cn/stock/go.php/vReport_Show/kind/search/rptid/816086178548/index.phtml
- 华龙证券计算机行业周报（2026-04-14）：https://www.hlzq.com/upload1/yb/20260414/1776140111991.pdf
