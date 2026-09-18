# 昆仑芯（Kunlunxin / 百度）P800 与超节点互联技术 —— 事实核实清单

> 核实时间：2026-09-18。可信度分级：【官方】>【招股书/年报/交易所公告】>【券商研报】>【媒体/自媒体】。
> 凡未在公开来源中出现的数字，一律标注「未找到公开数据」或「未证实」，不做推算填充。

## 1. XPU Link（XPU-Link）规格与带宽

- 【官方】昆仑芯自研互联通信协议名为 **XPU Link**（亦写作 XPU-Link），用于超节点内加速卡间 Scale-Up 互联；官方表述为「兼容主流 scale-up 通信标准 OISA」。**公开口径未给出 GB/s、SerDes 速率、lane 数、单向/双向口径**。来源：https://www.kunlunxin.com/news/4541.html
- 【官方】昆仑芯超节点产品页进一步明确：「自研 XPU Link 兼容主流 Scale-Up 通信标准 **OISA**」；CEO 欧阳剑称「在 P800 产品定义阶段就前瞻性采用高性能及生态开放的互联架构」。来源：https://www.kunlunxin.com/news/4717.html
- 【官方】2025 中国算力大会稿称超节点「所使用的自研互联通信协议 XPU Link，**已实现对 OISA 协议的兼容**」——与上一条一致，可交叉确认兼容关系。来源：https://www.kunlunxin.com/news/4712.html
- 【官方】带宽的唯一公开定量表述是**相对值**：超节点「单柜内卡间实现全互联通信，**带宽提升高达 8 倍**」（相对传统单机 8 卡形态），未给出绝对 GB/s。来源：https://www.kunlunxin.com/news/4541.html
- 【官方】同一 8 倍口径在百度智能云渠道复述：「卡间互联带宽提升 8 倍，单整机柜训练性能提升 10 倍，单卡推理性能提升 13 倍」。来源：https://cloud.baidu.com/news/news_cdd5d1d3-4510-47a2-a806-55a795a50167
- ❌ **XPU Link 的绝对带宽 GB/s、SerDes 速率（如 112G/224G）、lane 数、单向/双向口径：未找到公开数据。** 已检索 kunlunxin.com 全部产品页与新闻页、cloud.baidu.com、OISA 2.0 规范正文、券商研报，均无此参数。
- ⚠️【媒体/自媒体，不可引用】有云厂商社区文章称超节点「双向带宽达 1TB/s、延迟 50ns 以内、3D-Torus 拓扑、专用 RDMA 协议」，同文又将 P800 误写为「第二代 AI 芯片、7nm、HBM2e、128 TFLOPS」，与官方口径（第三代、96GB HBM3、345 TFLOPS）直接冲突，判定为 AI 生成内容，**不予采信**。来源（仅供排除）：https://cloud.baidu.com/article/5615257
- ⚠️【未证实】另有「国产算力平台整机柜聚合带宽超 400GB/s」的说法，属 4 台服务器/16 卡集群的 **Scale-Out（RoCEv2）整机柜聚合**实测，**不是** XPU Link 的卡间 Scale-Up 带宽，且原始报告未直接取得。来源：https://www.sohu.com/a/1047763313_468661

### P800 支持几卡互联

- 【官方】超节点内 Scale-Up 域形态为**一层全互联**：单机柜 **32 卡或 64 卡**（「单柜可容纳 32/64 张昆仑芯 AI 加速卡」）。来源：https://www.kunlunxin.com/news/4541.html
- 【官方】32 卡 Scale-Up 域由 **4 台 Switch Tray** 实现全互联，任意两张 XPU 之间通信**仅需 1 跳**。来源：https://cloud.baidu.com/news/news_cdd5d1d3-4510-47a2-a806-55a795a50167
- 【官方】百度智能云天池超节点产品页亦标注「**32 卡一层全互联**」。来源：https://cloud.baidu.com/product/tianchi.html
- 【官方】作为对照，**传统单机形态为 8 卡**（「突破传统单机 8 卡产品形态」「相比传统形态下 8 台 8 卡服务器」）；DeepSeek 一体机亦有「单机 8 卡满血版」配置。来源：https://www.kunlunxin.com/news/4541.html
- 【官方】整柜 64 卡场景机柜配置：**16 个 1U Compute Tray（每 tray 4 卡）+ 8 个 1U Switch Tray + 2 个 2U Power shelf，共 28U**。来源：https://cloud.baidu.com/news/news_cdd5d1d3-4510-47a2-a806-55a795a50167
- 【官方】计算节点为 **1U 4 卡**液冷设计，算力密度较传统 8U 8 卡提升 4 倍。来源：https://cloud.baidu.com/news/news_cdd5d1d3-4510-47a2-a806-55a795a50167

## 2. OISA 与昆仑芯的角色、OISA 与 UALink 的关系

- 【官方】OISA = **Omni-directional Intelligent Sensing Express Architecture，全向智感互联架构**。来源：https://www.oisa.org.cn/index.php/aboutoisa/
- 【官方】OISA 由**中国移动**牵头联合国内互联网、AI 芯片、交换芯片、服务器等领域力量研发；目标是打造「全向、对等、智能」互联新范式，使任意 AI 芯片之间、AI 芯片与 CPU 之间直接高速访问彼此内存。来源：https://www.oisa.org.cn/index.php/aboutoisa/
- 【官方】2025-08-23 中国算力大会主论坛，**中国移动、之江实验室、百度等**数十家单位共同启动「智算开放互联 OISA 生态共建战略合作」并发布 **OISA 2.0 协议**。来源：https://www.stdaily.com/web/gdxw/2025-08/25/content_389911.html
- 【官方】同一启动仪式的昆仑芯官方通稿：**昆仑芯携手中国移动、之江实验室及多家服务器厂商**共同启动 OISA 生态共建并发布 OISA 2.0。来源：https://www.kunlunxin.com/news/4712.html
- 【官方】2024-06-18「多样性算力产业峰会 2024」，北京市科委中关村管委会联合中国移动及**近 50 家产业链上下游单位**启动「北京全向智感 OISA 协同创新平台」，发布国内首个 GPU 卡间互联开放协议 **OISA Gen1** 及 OISA 交换芯片原型。来源：https://kw.beijing.gov.cn/xwdt/bmdt/202406/t20240618_3816335.html
- 【官方/政府】OISA Gen1 规格：支持 **128 张 GPU** 经 **8 个 Switch 芯片**互联，任意卡间互联带宽 **800GB/s**，每 Switch 支持 **128 端口**、交换容量 **51.2 Tb/s**。来源：https://kw.beijing.gov.cn/xwdt/bmdt/202406/t20240618_3816335.html
- 【官方】昆仑芯官方定位为 OISA 体系「**重要合作伙伴**」，自 OISA 1.0 阶段起持续参与演进，**未使用「发起单位」或「理事单位」表述**。来源：https://www.kunlunxin.com/news/4712.html
- ❌ **昆仑芯是否为 OISA 发起单位/理事单位：未找到公开数据。** OISA 官网「生态伙伴」页为动态渲染，抓取仅取到中国移动，成员名单不可得；官网规范文本仅列「编制单位」「主要起草单位」。来源：https://www.oisa.org.cn/index.php/aboutmembers/
- 【官方】OISA 技术规范（交换节点参考设计）主要起草单位署名为：**苏州盛科通信股份有限公司、中国移动研究院、之江实验室等**——**未见昆仑芯/百度署名**。来源：https://www.oisa.org.cn/wp-content/uploads/2026/06/OISA-%E4%BA%A4%E6%8D%A2%E8%8A%82%E7%82%B9%E5%8F%82%E8%80%83%E8%AE%BE%E8%AE%A1%E8%A7%84%E8%8C%83.pdf
- 【官方】中国移动研究院 2026-05-08 发布商用智算卡间互联 **OISA IP Core**，中国移动与**中国电子、中科曙光、之江实验室、百度**等单位共同出席见证。来源：https://www.c114.com.cn/news/118/a1309911.html
- 【官方】OISA 2.0 规范口径：支持 **1024 张 AI 芯片** Scale-up 互联，互联带宽**突破 TB/s**，通信时延缩短至**数百纳秒**。来源：https://www.oisa.org.cn/index.php/pdfdown/
- 【官方】OISA 2.0 物理层：兼容 IEEE 802.3，FEC 提供 RS(544,514) 与 RS(272,257) 轻量级方案；**端口速率支持 50Gb/s 至 1.6Tb/s 聚合速率，通过 25Gb/s、56Gb/s、112Gb/s、224Gb/s 等高速 SerDes 通道绑定实现**。来源：https://www.oisa.org.cn/index.php/seepart/
- 【官方】OISA 2.0 规范给出的示例拓扑：256 个 GPU 节点、每 GPU **8 个通信端口、单端口速率 200 Gb/s**，配 8 个独立交换平面，每交换芯片 **51.2 Tb/s / 256 个 200Gb/s 端口**（属规范举例，非产品强制规格）。来源：https://www.oisa.org.cn/index.php/seepart/
- 【官方】OISA 2.0 规范：直连（无交换）拓扑仅支持少量 GPU（举例 8 张）；支持单级、背脊（跨机柜 Scale-up）、叶脊四类拓扑。来源：https://www.oisa.org.cn/index.php/seepart/
- 【官方】OISA 交换节点参考设计（编号 OISA-2026-CN-002，2026-04-30 发布）：物理层高速信号接口**支持 50Gb/s 至 800Gb/s 速率及 56Gb/s、112Gb/s SerDes**；附录含 **25.6T 交换节点板卡布局**。来源：https://www.oisa.org.cn/wp-content/uploads/2026/06/OISA-%E4%BA%A4%E6%8D%A2%E8%8A%82%E7%82%B9%E5%8F%82%E8%80%83%E8%AE%BE%E8%AE%A1%E8%A7%84%E8%8C%83.pdf
- 【官方】OISA 生态规模：截至 2026-05，OISA 协同创新平台已汇聚**超过 50 家**产业单位。来源：https://www.c114.com.cn/news/118/a1309911.html
- 【官方】OISA IP 实测效果（中国移动口径）：相较传统协议方案**时延降低约 1.7 倍**；DeepSeek-V3 在千卡级集群训练中 **MFU 提升 1.53 倍、训练时间缩短 3.59 天**。来源：https://www.c114.com.cn/news/118/a1309911.html
- 【官方】ODCC 2024-09-03 发布、中国移动李锴牵头《GPU 卡间互联 OISA 系统研究》，为 OISA 体系重要产业成果。来源：https://www.odcc.org.cn/news/p-1890304273705443330.html

### OISA 与 UALink 的关系

- ❌ **OISA 与 UALink 之间的官方合作关系/兼容关系：未找到公开数据。** OISA 官网、OISA 2.0 规范正文、昆仑芯官方稿均未提及 UALink。
- 【媒体】雷峰网 2026-03-31 将全球超节点互联路线并列为并行竞争格局：英伟达 NVLink、华为灵衢、**UALink 联盟（开放标准，被称"反英伟达"联盟）**、ETH-X/SUE 以太网开放协议、以及 **OISA 标准**；并称中国移动 OISA 等开放联盟走「国芯国连、协议共用」路线。来源：https://m.leiphone.com/category/chips/zWXsXt5CiDBI44yI.html
- 【媒体】UALink 背景：2024 年由 **AMD 牵头**，亚马逊、Astera Labs、思科、谷歌、惠普、Intel、Meta、微软等参与成立，开发 CPU/加速器卡间开放互联标准。来源：中兴通讯《中兴通讯技术》2025 年第 2 期 https://www.zte.com.cn/content/dam/zte-site/res-www-zte-com-cn/mediares/magazine/publication/com_cn/pdf/202502.pdf
- 【媒体】结论：二者是**各自独立、相互竞争**的开放 Scale-Up 互联标准，公开资料未见互认或合并。来源：https://m.leiphone.com/category/chips/zWXsXt5CiDBI44yI.html

## 3. 昆仑芯超节点拓扑与百度智能云「天池」超节点

- 【官方】2025-04-25 Create 2025 发布昆仑芯超节点：**单柜可容纳 32/64 张卡**，突破传统单机 8 卡形态；机柜间**支持 IB/RoCE 通信**，实现跨柜高带宽低延迟，**支持万卡以上规模**智算集群构建。来源：https://www.kunlunxin.com/news/4541.html
- 【官方】同一稿：整柜功率可支持到 **120kW**，采用冷板式液冷；MoE 大模型**单节点训练性能提升 5–10 倍、单卡推理效率提升 13 倍**；一个机柜算力最高可达到传统形态下 8 台 8 卡服务器。来源：https://www.kunlunxin.com/news/4541.html
- 【官方】2025-09 云智大会复述：单机柜支持 **32 至 64 张**加速卡灵活部署；跨柜 **IB/RoCE**；整柜功率控制在 **120kW 以内**；DeepSeek V3/R1 PD 分离下**单卡性能提升 95%、单实例推理性能提升高达 8 倍**；水电网「三盲插」。来源：https://www.kunlunxin.com/news/4717.html
- 【官方】2025-12-29 百度智能云技术稿：昆仑芯超节点是 **32/64 卡最小算力交付单元**，可支撑万卡级别集群网络互联；64 卡场景仅需 **28U**（16×1U Compute Tray + 8×1U Switch Tray + 2×2U Power shelf）。来源：https://cloud.baidu.com/news/news_cdd5d1d3-4510-47a2-a806-55a795a50167
- 【官方】Scale-Out 侧：每计算节点预留 **4 张 PCIe 网卡扩展位**，XPU 与 NIC **1:1 绑定**，单节点最高支持 **4 张 400G 网卡**；结合自研 **HPN（High Performance Network）**导轨优化架构，支撑数百卡到上万卡。来源：https://cloud.baidu.com/news/news_cdd5d1d3-4510-47a2-a806-55a795a50167
- 【官方】供电：单 Power shelf 2U 内置 12 个 PSU，10+2 冗余，支持 3300W/5500W，单柜 **33kW~120kW**；散热为液冷+风冷混合，CDU 为自研「天玑 1.0」，可使 XPU 温度下降 20℃ 以上并支持部署于风冷机房。来源：https://cloud.baidu.com/news/news_cdd5d1d3-4510-47a2-a806-55a795a50167
- 【官方】百度智能云天池超节点产品页：主打「**32 卡一层全互联**」「兼容现有机房」「单人可运维」，XPU 温控 **75℃**，标准整柜一体化 0 配置交付，1U 一节点水电网盲插，号称**首个 x86 平台**可平滑迁移。来源：https://cloud.baidu.com/product/tianchi.html
- 【官方】天池超节点性能：相比传统 8 卡服务器，**卡间互联带宽提升 8 倍、单整机柜训练性能提升 10 倍、单卡推理性能提升 13 倍**。来源：https://cloud.baidu.com/news/news_cdd5d1d3-4510-47a2-a806-55a795a50167
- 【官方】2026 WAIC（2026-07-17 开幕）：**天池 256** 从底层 **XPU-Link 互联协议、交换与整机架构，到液冷 CDU 散热均为百度自研**；**Scale-Up 全互联通信域从 32 节点扩展到 1024 节点**。来源：https://news.ifeng.com/c/8ut7vaodSgy
- 【官方/交易所媒体】天池自研四项技术：①自研 XPU-Link 互联协议与可编程交换技术；②自研拥塞控制算法；③报文分类与 QoS；④自研链路高可用算法，**故障切换压缩至 50 微秒以内**；实测**带宽有效率达 77%**；互联成本与单柜网络空间各节省 50%；采用自研 **Cable Tray 铜缆互连**；双 Power Shelf 单柜支持 **66kW**。来源：https://www.cs.com.cn/ssgs/01/2026/07/19/detail_2026071910025376.html
- 【官方】天池 256 较上一代**吞吐性能提升 25%**，**推理效率提升 50%**，已适配文心/DeepSeek/GLM/MiniMax；**天池 512 单个超节点即可支撑万亿参数模型训练**。来源：https://news.ifeng.com/c/8ut7vaodSgy
- 【官方】P800 集群实测：全国产集群完成**文心 5.1** 重要版本训练，**有效训练率 97%**，**万卡规模线性扩展度超过 85%**。来源：https://www.jiemian.com/article/14421027.html
- 【媒体】天池 256 于 2026 年 4 月点亮，**2026 年 6 月正式上市**；网络架构升级至 **HPN5.0**，**端到端时延优化 50%**，支持按需搭建数十万卡乃至百万卡超大集群。来源：https://www.jiemian.com/article/14421027.html
- ⚠️【官方口径不一致，需注意】云智大会称天池超节点**整柜功率 120kW 以内**（2025-04 与 2025-09 口径），WAIC 2026 则称天池 256 **单柜支持到 66kW**（双 Power Shelf）。疑为不同产品世代/配置，非矛盾但不可混用。
- ⚠️【官方口径不一致，需注意】超节点单柜卡数在不同官方稿件中分别出现 **32/64 卡**（昆仑芯 2025-04）、**32~64 卡**（2025-09）、**32/64 卡**（2025-12）与**天池 256 / 天池 512**（2025-11 起）两套命名体系；「256」「512」为超节点整机规模而非单柜卡数，引用时须区分。

## 4. 昆仑芯 3 代及后续产品（P800 / M100 / M300 / X600）

- 【官方】昆仑芯官网产品研发路线图明确：**昆仑芯 1 代（2019）、2 代（2021）、3 代（2024）**，已完成三代 AI 计算芯片研发与商业化落地，均基于 2017 年 Hot Chips 发布的自研 XPU 架构，「目前新一代产品正在研发中」。来源：https://www.kunlunxin.com/%e6%a0%b8%e5%bf%83%e6%8a%80%e6%9c%af
- 【官方】**P800 即昆仑芯 3 代 AI 芯片**（第三代 AI 加速器），2024 年推出，基于新一代自研架构 **XPU-P**，96GB HBM，FP16 算力 345 TFLOPS，支持 FP8/FP16/FP32。来源：https://www.c114.net.cn/industry/101818.html
- 【券商研报】招银国际：百度直接持股昆仑芯 **59.45%**；P800 基于新一代自研架构 XPU-P，**显存规格优于同类主流 GPU 20%–50%**，目前应用的最大集群规模**超 3 万卡**。来源：https://www.cmbi.com.hk/upload/202509/20250919547901.pdf
- 【官方/交易所媒体】2025-11-13 百度世界大会：**M100 面向大规模推理场景优化，2026 年年初上市；M300 面向超大规模多模态大模型训练与推理优化，2027 年年初上市**。来源：https://www.stcn.com/article/detail/3492973.html
- 【官方】2025-11-13 百度世界大会同时披露昆仑芯发展路线图：**2028 年推动千卡级昆仑芯超节点上市；2029 年推出三颗全新 N 系列芯片；2030 年点亮「百度百舸」百万卡级别昆仑芯单集群**。来源：https://news.qq.com/rain/a/20251113A030F500
- 【媒体】2025-11-13 大会另一表述：昆仑芯新一代超节点包括 **256 卡的「天池 256」与 512 卡的「天池 512」，两款产品均于 2026 年上市**；单个天池 512 即可完成万亿参数模型训练。来源：https://news.qq.com/rain/a/20251113A030F500
- 【官方】2026 WAIC（2026-07-20）**M100 实物首次公开展示**，CEO 欧阳剑在央视新闻直播间展示；M100 **基于全国产供应链**，**主要对标 NVIDIA H20**，HBM 容量略低，主打高性价比推理应用。来源：https://www.c114.net.cn/industry/101818.html
- 【招股书/交易所公告】**2026-01-01，昆仑芯已透过其联席保荐人以保密形式向香港联交所提交上市申请表格（A1 表格）**，申请于港交所主板上市及买卖。来源：https://www.cnfin.com/kx/detail/20260102/4359511_1.html
- 【官方】李彦宏在 2026 年二季度财报电话会披露：**昆仑芯已完成三代 AI 芯片的研发及商业化**，产品适配 Kimi K3、GLM 5.2、MiniMax M3、混元 Hy3 等新版本模型；已落地多个万卡规模集群；**M100 已对外公布，M300 将按节奏推进研发上市；昆仑芯独立 IPO 相关工作仍在正常推进**。来源：https://www.trendforce.cn/industry-news/semiconductors/20260819-7330.html
- 【官方】下一代产品规划：百度智能云称**基于昆仑芯 M 系列将推出千卡、四千卡超节点**，拓展国产算力规模化边界。来源：https://news.ifeng.com/c/8ut7vaodSgy
- 【媒体】**昆仑芯正在研发新一代 P 系列高性能 AI 芯片**（面向高端训练，未公布型号与时间表）。来源：https://www.c114.net.cn/industry/101818.html
- ❌ **「昆仑芯 X600」：未找到任何公开来源。** 检索 kunlunxin.com、cloud.baidu.com 及中文媒体均无此型号，**判定为未证实/不存在于公开资料**。
- ❌ **M100 / M300 的互联规格（是否升级 XPU Link、带宽数值、互联规模）：未找到公开数据。** 官方仅披露应用场景定位与上市时间。来源：https://www.stcn.com/article/detail/3492973.html
- 【媒体】IDC 数据：2025 年中国 AI 加速卡总出货约 **400 万张**，国产厂商合计约 **165 万张（占比 41%）**；**昆仑芯与寒武纪各约 11.6 万张并列国产第三**，仅次于华为昇腾 81.2 万张、阿里平头哥 26.5 万张。来源：https://www.c114.net.cn/industry/101818.html
- 【媒体】集微网：昆仑芯 P800 万卡集群已点亮，**计划进一步点亮 3 万卡集群**；训练主流开源模型集群 **MFU 提升至 58%**，**有效训练率达 98%**，**带宽有效性达 90% 以上**。来源：https://m.laoyaoba.com/html/share/news/932199?news_id=932199
- 【官方】商用进展：昆仑芯以**三个标包均第一名**中标中国移动集采项目**十亿级订单**。来源：https://www.kunlunxin.com/news/4712.html
- 【媒体】Reuters 报道：昆仑芯赢得中国移动供应商 **超 1.39 亿美元**订单。来源：https://www.reuters.com/technology/baidu-chip-design-unit-kunlunxin-wins-over-139-million-orders-china-mobile-2025-08-22/

## 5. 补充：P800 集群规模与关键性能节点（供交叉核对）

- 【官方】2025-04 Create 2025 百度 AI 开发者大会：**三万卡昆仑芯集群点亮**。来源：https://www.kunlunxin.com/news/4541.html
- 【媒体】2025 年 4 月基于 P800 的 **3.2 万卡**超级集群完成首次系统级点亮。来源：https://www.c114.net.cn/industry/101818.html
- 【官方/媒体】2025-02 集微网：昆仑芯宣布成功点亮 P800 **万卡集群**（早于 3 万卡节点）。来源：https://m.laoyaoba.com/html/share/news/932199?news_id=932199
- 【官方】OISA 2.0 发布稿称「相较 OISA 1.0，2.0 将支持的 AI 芯片数量提升至 **1024 张**，带宽**突破 TB/s** 级别，互联时延降低至**数百纳秒**」。来源：https://www.kunlunxin.com/news/4712.html

## 6. 明确的「未找到公开数据」清单

1. XPU Link 的绝对带宽（GB/s）、单向/双向口径 —— 未找到公开数据。
2. XPU Link 的 SerDes 速率与 lane 数 —— 未找到公开数据。
3. 昆仑芯在 OISA 中的正式成员身份（发起单位/理事单位/普通成员）—— 未找到公开数据。
4. OISA 与 UALink 之间的官方关系 —— 未找到公开数据（公开资料仅呈现为并行竞争标准）。
5. 昆仑芯「X600」型号 —— 未找到任何公开来源。
6. M100 / M300 的互联规格（协议版本、带宽、互联卡数）—— 未找到公开数据。
