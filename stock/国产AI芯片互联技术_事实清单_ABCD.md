# 国产 AI 芯片互联技术事实清单（沐曦 / 天数智芯 / 燧原 / 昆仑芯）

> 核实日期：2026-09-18。可信度分级：【官方】>【招股书/年报/交易所公告】>【券商研报】>【媒体/自媒体】；无法确证的写「未证实」，拿不到就写「未找到公开数据」。所有数字均来自下列 URL，未做推算填充。

### 题面六处前提更正（先读）

1. 沐曦代码是 **688802**，不是 688798。
2. 沐曦 **C500 板卡官方只支持 2 卡或 4 卡** MetaXLink 互联；8 卡是「8 卡高性能服务器/超节点」形态，不是 C500 板卡形态。
3. 沐曦 **C600 的卡间互连带宽官方公告明确写「略有下降」**（受限于国产工艺制程），不要默认 C600 互联是升级。
4. 天数智芯**没有 iLink / Iluvatar Link 这类硬件互联品牌名**；官方专名只有软件层通信库 **IXCCL**。
5. 燧原互联正式名是 **GCU-LARE**（不是 GCU-Link）；且 **「邃思 2.0 ↔ 云燧 S60」不是同一代**——邃思 2.0 对应云燧 T20/T21，S60 是第三代邃思 320。
6. 燧原 IPO 受理（2026-01-22）与过会（2026-06-16）都在 **2026 年**，不是 2025 年；昆仑芯**「X600」型号未证实**，且 **XPU Link 没有任何公开的绝对 GB/s 数字**。

---

## A. 沐曦 MetaX（曦云 C500 / C600）

### A0. 主体与代码更正

- 【交易所公告】沐曦全称为「沐曦集成电路（上海）股份有限公司」，科创板证券代码为 **688802**（非 688798）；用户题面中的「688798」有误。来源：https://www.sse.com.cn/assortment/stock/list/info/company/index.shtml?COMPANY_CODE=688802
- 【媒体】2025-12-17 于上交所科创板上市，开盘涨 568.83%。来源：https://finance.cnr.cn/ycbd/20251217/t20251217_527462586.shtml

### A1. MetaXLink 规格（带宽 / SerDes / lane / 卡数 / 桥接器）

- 【招股书/交易所公告】「公司单 GPU 芯片创新性集成 **7 个 MetaXLink 接口**，达到了英伟达 4nm 制程工艺下旗舰产品（H200）相当的互连带宽性能，处于国内领先水平。」来源：http://dataclouds.cninfo.com.cn/sjother/documents/2025/20250630/ff35fb52e4824cf8b6624d22905ebc76.pdf
- 【招股书/交易所公告】「在互连拓扑结构方面，MetaXLink 支持 Full-Mesh、Hybrid Cube Mesh 等多种复杂互连拓扑结构，并且通过协议层设计创新，MetaXLink 支持互连拓扑重构……目前公司可实现 **2-64 卡**等多种互连拓扑及超节点架构。」来源同上。
- 【招股书/交易所公告】「通过 MetaXLink 各端口不同连接形式，发行人 GPU 产品支持 **2 卡、4 卡、8 卡全互连拓扑**……通过超节点架构可灵活适配 **16 卡、32 卡、64 卡**等系统规模……突破性实现**单机柜 128 卡**超高密度部署。」来源：http://vip.stock.finance.sina.com.cn/corp/view/vCB_AllBulletinDetail.php?stockid=688802&id=11397324
- 【交易所公告】MetaXLink「根据 GPU 之间数据交互需求优化传输协议并且**支持 MetaXLink 端口之间数据直接转发**」。来源同上。
- 【交易所公告】硬件演进（2024-2025）：「在每个 MetaXLink 上引入了**双控制器**，可以根据系统需求静态配置……通过激活更多的硬件设计能力实现包括**直通互连、交换机互连及直通和交换机混合互连**的方式。」来源同上。
- 【官方】官方开发者文档确认端口数为 7：「默认选择所有正常连接的端口（**1-7**）」，眼图测试 lanes 范围 0-3、phys 范围 0-3。来源：https://developer.metax-tech.com/api/client/document/preview/1335/split_files/metaxlink%E9%AA%8C%E6%94%B6%E6%B5%8B%E8%AF%95%E5%B7%A5%E5%85%B7.html
- 【官方】**SerDes 速率（Gbps/lane）与每端口 lane 数：未找到公开数据**（官网、招股书、问询回复、开发者文档均未披露）。
- 【官方】唯一可查的数值口径来自 mx-diagease 诊断默认配置模板：`"metaxlink": {"bw_uni_p2p": 49000, "bw_bi_p2p": 90000}`（单位 MB/s，即单向 **49 GB/s**、双向 **90 GB/s**）。**注意：这是诊断工具的目标门限/配置模板值，不是官方标称带宽，且未说明是单端口还是整卡汇总。** 来源：https://developer.metax-tech.com/api/client/document/preview/768/split_files/%E8%AF%8A%E6%96%AD%E6%A8%A1%E5%BC%8F.html
- 【未证实】沐曦开发者论坛有用户称「官方给的数据 **384GB/s**」（指 C500 D2D 带宽），官方客服回复仅为「不用加额外参数，正常跑就行」「两者皆可做参考」，**未确认该数字**。来源：https://developer.metax-tech.com/forum/t/4c500-pei-zhi-qiao-jie-qi-hou-ru-he-ce-shi-dai-kuan/466/
- 【官方论坛】桥接器存在：C500 需外接「桥接器/桥接卡」实现卡间互联（论坛实例为 4 张 C500 配桥接卡，官方客服确认 `--devices 0,1,2,3` 可跑 MetaXLink 带宽测试）。来源：https://developer.metax-tech.com/forum/t/4c500-pei-zhi-qiao-jie-qi-hou-ru-he-ce-shi-dai-kuan/466/
- 【官方】**C500 板卡形态不支持 8 卡**：官网 C500 产品页明确写「MetaXLink 互连技术支持 **2 卡或 4 卡**高速卡间互连」。来源：https://www.metax-tech.com/prod.html?cid=107&id=21
- 【交易所公告】8 卡形态需服务器/超节点级方案：「发行人深度参与 OCP 开放标准，将公司自主研发的 MetaXLink 高速互连拓扑转化为可商业化落地的 **8 卡高性能服务器系统**」。来源：http://vip.stock.finance.sina.com.cn/corp/view/vCB_AllBulletinDetail.php?stockid=688802&id=11397324

### A2. C500 的 MetaXLink 官方口径带宽数字

- 【招股书/交易所公告】官方**未给绝对 GB/s**，只有定性对标：「互连带宽达到了与英伟达 4nm 制程工艺下产品（H200）相当的性能」「是国内厂商中率先将与英伟达 NVLink 技术的差距缩小到一代以内的 GPU 供应商」。来源：http://vip.stock.finance.sina.com.cn/corp/view/vCB_AllBulletinDetail.php?stockid=688802&id=11397324
- 【交易所公告】问询回复中同一张对比表设有「卡间互连（GB/s）」列，但**只填了英伟达 H20 = 900 GB/s**，发行人自身一栏未给数字。来源同上（同页 H20 参数表）。
- 【交易所公告】行业区间口径：「国内互连技术……大部分企业单卡互连带宽在 **200-400 GB/s**」，沐曦自述「互连带宽处于国内领先水平」。来源：http://dataclouds.cninfo.com.cn/sjother/documents/2025/20250630/ff35fb52e4824cf8b6624d22905ebc76.pdf
- 【官方】结论：**C500 的 MetaXLink 官方标称带宽数字为「未找到公开数据」**；可引用的官方定量仅为上述「7 个接口」与「H200 相当」。

### A3. C600 升级（2025-2026）/ 互联规格 / 量产时间

- 【官方】C600 产品页：「搭载 MetaXLink 互连技术，支持灵活的多卡互连拓扑……**超低时延 MetaXLink 互连接口，互连灵活可扩展 MetaXLink-E 接口，高达 128 卡规模超节点**；符合 OAM2.0 规范的扣卡式模组，最大功耗 1000W。」来源：https://www.metax-tech.com/prod.html?cid=107&id=68
- 【官方】2025-07-29 WAIC 发布曦云 C600：「支持 **MetaXLink 超节点扩展技术**，并内置 ECC/RAS 多重安全防护模块」「构建从设计、制造到封装测试的**全流程国产供应链闭环**」。来源：https://www.sh.chinanews.com.cn/chanjing/2025-07-29/138561.shtml
- 【交易所公告】研发节点：「曦云 C600 系列已于 **2024 年 10 月交付流片**，并于 **2025 年 7 月回片并成功点亮**。」来源：http://vip.stock.finance.sina.com.cn/corp/view/vCB_AllBulletinDetail.php?stockid=688802&id=11397324
- 【年报/交易所公告·关键反直觉】「然而，**受限于工艺制程，曦云 C600 系列的卡间互连带宽略有下降**，同时功耗有所提升。」发行人靠 SDMA 通信引擎、TP Overlap、MoE Comm Overlap、DualPipeV 等软件优化「缓解理论带宽下降带来的影响」。来源同上。
- 【年报/媒体】量产时间：「2026 年 5 月，曦云 C600 成功实现量产」（沐曦 2026 年半年报，经证券时报报道）。来源：https://www.stcn.com/article/detail/4162800.html
- 【官方】C600 核心芯片 MXC600 于 2026-05-26 通过国家《安全可靠测评》。来源：https://www.metax-tech.com/ndetail/12610.html
- 【媒体】C600 规格：集成 HBM3e、支持 FP8、**144GB 显存容量**。来源：https://www.ithome.com/0/890/942.htm
- 【交易所公告】后续路线：曦云 C700 系列「性能接近英伟达 H100」，「通信能力及能效比等均大幅提升」。来源：http://vip.stock.finance.sina.com.cn/corp/view/vCB_AllBulletinDetail.php?stockid=688802&id=11397324

### A4. 招股书 / 交易所公告中关于 MetaXLink 的官方原文摘录

- 【招股书】「集群性能方面，公司自研的 **MetaXLink 具备国内稀缺的高带宽卡间互连能力**，可实现 2-64 卡多种互连拓扑，并且在智算集群的线性度和稳定性方面具有较强的产品表现。」来源：http://dataclouds.cninfo.com.cn/sjother/documents/2025/20250630/ff35fb52e4824cf8b6624d22905ebc76.pdf
- 【招股书】「公司自主研发的 MetaXLink 高速互连技术突破了传统 PCIe 总线在带宽和延迟方面的限制，为大规模 GPU 集群的高效互连提供了关键支撑，在性能、灵活性和可扩展性方面均处于行业领先地位。」来源同上。
- 【交易所公告·技术演进】「将 MetaXLink 技术与光互连技术相结合，**从单节点内 4 卡、8 卡互连扩展到节点外 16 卡、32 卡、64 卡互连，打通了 PCIe Switch Box 跨节点通信技术，成功实现跨机架级别光互连超节点技术落地**。」来源：http://vip.stock.finance.sina.com.cn/corp/view/vCB_AllBulletinDetail.php?stockid=688802&id=11397324
- 【交易所公告】「报告期内，基于自主创新的 **DragonFly 互连拓扑**，发行人已成功打造 **64 卡光互连超节点产品及 3D-Mesh 64 卡超节点产品**等。」来源同上。
- 【招股书·募投】「研发基于新一代 MetaXLink 光互连的高性价比与鲁棒性的 GPU 集群组网，实现任意到任意、非阻塞高带宽域在多节点、多机架范围的有效覆盖」「研发模组内光电共封（**CPO**）技术」。来源：http://dataclouds.cninfo.com.cn/sjother/documents/2025/20250630/ff35fb52e4824cf8b6624d22905ebc76.pdf
- 上交所正式招股说明书（2025-11-27）原件 URL（40MB，SSE 静态站有反爬挑战）：https://static.sse.com.cn/disclosure/listedinfo/announcement/c/new/2025-11-27/688802_20251127_CZ8L.pdf

### A5. 是否支持内存语义 / 一致性

- 【官方·是（仅限特定超节点产品）】曦云 C550 3D Mesh 超节点产品页：「**单超节点 64 卡内存语义电互连**，通信延迟极低」，3D Mesh 拓扑，最高 8 机 64 卡。来源：https://www.metax-tech.com/prod.html?cid=112&id=55
- 【官方】曦云 C500X 光链超节点：「光电混合互连实现 **DragonFly 互连拓扑**支持最高 8 机 64 卡超节点」「单节点最高支持 64 卡光互连」。来源：https://www.metax-tech.com/prod.html?cid=112&id=54
- 【官方】沐曦 2026-04-01 新闻：「**耀龙 S8000 G2 超节点**业界首创 3D Mesh 互连技术，实现 64 张曦云 C550 通用 GPU 的高速互连」；**Shanghai Cube** 液冷整机柜「以 **128 颗 GPU** 的高密度液冷架构，8 机柜并排即可组成千卡集群」。来源：https://www.metax-tech.com/ndetail/12573.html
- 【官方】沐曦两位研究员深度参与《超节点技术体系白皮书》「**scale up 互连协议**」章节编撰。来源同上。
- 【未找到公开数据】C500 / C600 **单卡或 8 卡域层面**是否提供内存语义（load/store）与缓存一致性：招股书、问询回复全文均无「内存语义」字样，仅在 C550 超节点产品页出现「内存语义电互连」。来源（全文检索）：https://developer.metax-tech.com/api/client/document/preview/768/split_files/%E8%AF%8A%E6%96%AD%E6%A8%A1%E5%BC%8F.html 与上述问询回复 URL。

---

## B. 天数智芯（天垓 100 / 天垓 150）

### B1. 互联方案叫什么 / 带宽多少

- 【官方·名称更正】**不存在 iLink / Iluvatar Link / IxLink 这类硬件互联品牌名**——官网产品页、招股书、媒体稿均无。题面假设的「iLink」**未证实**。来源：https://www.iluvatar.com/productDetails?fullCode=cpjs-yj-xlxl-tg100
- 【官方】**唯一有官方专名的是软件层通信库 IXCCL**（分布式集合通信）：「公司自主研发了 **IXCCL 分布式通信技术**，显著提升多机多卡高速互联性能」。来源：https://cn.wicinternet.org/2026-01/12/content_38531570.htm （英文版：https://www.wicinternet.org/2026-02/26/c_1162495.htm ）
- 【招股书/上市文件】IXCCL 对应招股书附录六第 37 项软件著作权「High-performance Collective Communication Library（適用於天數智芯通用芯片的高性能集合通信庫軟件）」，登记号 2023SR0013102。来源：https://www.hkexnews.hk/listedco/listconews/sehk/2025/1230/11969957/2025123000082_c.pdf
- 【官方】**天垓 100 官方标称带宽 = 64 GB/s，且官方不区分单向/双向**：官网规格原文「接口 PCIe Gen4.0 x 16 lane／**共享 64 GB/s 主控双向带宽**／**共享 64 GB/s 片间互联带宽**」，内存 32GB DRAM HBM2，板级功耗 250W，被动散热，全长全高双槽 PCIe 卡。来源：https://www.iluvatar.com/productDetails?fullCode=cpjs-yj-xlxl-tg100
- 【招股书/上市文件】招股书英文口径一致：「**PCIe-based proprietary peer-to-peer protocol provides 64GB/s bi-directional bandwidth**」（写在 ZK Gen 1 条目下）。来源：https://www.hkexnews.hk/listedco/listconews/sehk/2025/1230/11969957/2025123000082_c.pdf
- 【未找到公开数据】**天垓 150（TG Gen 2）的互联带宽官方未公布**（官网天垓 150 页无「产品规格」表）。来源：https://www.iluvatar.com/productDetails?fullCode=cpjs-yj-xlxl-tg150

### B2. 天垓 100 的多卡互联拓扑（4 卡/8 卡、PCIe 还是专有互联）

- 【官方+招股书·拓扑判定】天垓 100 走 **PCIe**：片间互联带宽数字与 PCIe Gen4.0 x16 主控双向带宽同源（均标「共享 64 GB/s」），**官方未提及任何 NVLink 式桥接器或互联模组**。来源：见 B1 两条 URL。
- 【招股书/上市文件】**4 卡 / 8 卡拓扑官方未披露**；可确证口径仅为「多卡、多机」「多 GPGPU 系统」，以及 peer-to-peer 可把集群扩展到「数千甚至数万张卡」。来源：https://www.hkexnews.hk/listedco/listconews/sehk/2025/1230/11969957/2025123000082_c.pdf
- 【招股书/上市文件】TG Gen 1（=天垓 100）原文：「Its **peer-to-peer communication** enables multiple chips to work together directly for expanded computing power.」，2021 年 3 月发布、**2021 年 9 月量产**，7nm。来源同上。
- 【官方】（佐证走标准 PCIe）官方规格页无任何专有互联接口、桥接卡、互联模组的条目或图片，仅有 PCIe 接口一项。来源：https://www.iluvatar.com/productDetails?fullCode=cpjs-yj-xlxl-tg100

### B3. 天垓 150 规格升级 / 量产时间 / 代工厂

- 【招股书/上市文件】**TG Gen 2（=天垓 150，BI-V150）：2023 年 9 月发布，2023 年 Q4 量产**；升级点为「expanded-capacity high-speed memory」+ **350W TDP** + 增强整型性能。来源：https://www.hkexnews.hk/listedco/listconews/sehk/2025/1230/11969957/2025123000082_c.pdf
- 【官方】官网特点页：「配备了 **64GB 的 HBM** 内存，板级功耗为 **350W**」；第三方技术文档补充：自研 **ivcore11** 架构、**7nm**、**64GB HBM2e**、**PCIe 4.0 x16**、风冷 PCIe 加速卡，官方仅提「支持 PCIe 4.0 x16 标准接口，能够实现多卡、多机的高效算力扩展」，**未提任何专有卡间互联**。来源：https://www.iluvatar.com/productDetails?fullCode=cpjs-yj-xlxl-tg150 、 https://ai.gitee.com/docs/compute/clusters_gpu/iluvatar/iluvatar_BI-V150_gpu
- 【招股书/上市文件】**TG Gen 3（对应天垓 300）：2024 年 Q3 发布**，「significant advancements in large-scale cluster efficiency and connectivity」「implements the latest international standard interfaces including **PCIe Gen5**. The chip significantly improves **peer-to-peer communication bandwidth** and supports a **multi-card architecture**」，**预计 2026 年 Q1 量产**。来源：https://www.hkexnews.hk/listedco/listconews/sehk/2025/1230/11969957/2025123000082_c.pdf
- 【政府公示】军队采购网需求公示（2025-12-10）给出**天垓 150S（BI-V150-64-01）**实测口径：≥56TFLOPS@FP32 / ≥220TFLOPS@FP16 / ≥660TOPS@INT8；≥64GB HBM2e；**PCIe Gen4.0 x16**；板级功耗 ≤450W；**显存带宽 ≥1100GB/s**；支持多机多卡分布式训练与英伟达异构混训。来源：https://www.plap.mil.cn/freecms/site/juncai/ggxx/info/2025/8a1d03a39afc00c5019b06c930335338.html
- 【招股书·代工匿名化】招股书将代工厂**匿名为「Supplier F」**，定义为「1987 年成立、在台交所与纽交所两地上市的全球领先晶圆代工」（特征指向台积电但**未点名**）。来源：https://www.hkexnews.hk/listedco/listconews/sehk/2025/1230/11969957/2025123000082_c.pdf
- 【媒体·2021 年发布稿点名】天垓 100 发布稿明确点名「**台积电 7nm FinFET + 2.5D CoWoS**」。来源（媒体稿）：https://www.iluvatar.com/productDetails?fullCode=cpjs-yj-xlxl-tg100 （补充检索：集微网/半导体行业观察同期报道）

### B4. 官方 / 研报原文摘录 + IPO 前提更正

- 【官方】官网天垓 100 规格原文（可直接引用）：「接口 **PCIe Gen4.0 x 16 lane**」「**共享 64 GB/s 主控双向带宽**」「**共享 64 GB/s 片间互联带宽**」「内存 32 GB DRAM HBM2」「型号 天垓 100 加速卡（BI-V100）」「功耗 板级功耗 250W」。来源：https://www.iluvatar.com/productDetails?fullCode=cpjs-yj-xlxl-tg100
- 【招股书/上市文件】原文：「TG Gen 3 … implements the latest international standard interfaces including PCIe Gen5. The chip significantly improves peer-to-peer communication bandwidth and supports a multi-card architecture.」来源：https://www.hkexnews.hk/listedco/listconews/sehk/2025/1230/11969957/2025123000082_c.pdf
- 【招股书/上市文件】**重要前提更正**：招股书正文以英文为主，**不含「天垓 100/150」中文品名**，产品一律以 **TG Gen 1 / TG Gen 2 / TG Gen 3** 指代；TG↔天垓的映射为按时间线+规格的对应关系。来源同上。
- 【招股书/上市文件·IPO 前提更正】天数智芯 IPO 是**港股**而非科创板：**09903.HK，2026-01-08 上市**，发售价 **HK$144.60**，募资 36.77 亿港元，首日 +8.44%；招股书日期 2025-12-30。**未发现科创板申报文件**。来源：https://www.hkexnews.hk/listedco/listconews/sehk/2025/1230/11969957/2025123000082_c.pdf 、 https://jnzstatic.cs.com.cn/zzb/htmlInfo/112548.html
- 【招股书/上市文件】招股书全文（中文版，7.17MB）：https://www1.hkexnews.hk/listedco/listconews/sehk/2025/1230/2025123000020_c.pdf ；PHIP：https://www1.hkexnews.hk/app/sehk/2025/107974/documents/sehk25121900271.pdf
- 【未找到公开数据】天垓 100 的 4 卡/8 卡具体拓扑与互联聚合带宽：官方与招股书均未披露。

---

## C. 燧原科技（邃思 2.0 / 云燧 S60）

### C1. 互联方案名称与带宽

- 【官方】正式名称是 **GCU-LARE**（**General Compute Unit - Link All Round Engine**），官方中文释义「**燧原科技互联引擎**」；**未见 GCU-Link / Enflame Link / TopsLink 等名称**。来源：https://support.enflame-tech.com/onlinedoc_hw/1-t2x/t20/GCU-LARE_linkcard_product_manual/content/source/GCU-LARE_linkcard_product_manual.html
- 【官方】GCU-LARE **桥接卡**（型号 **ESL-B**，PN EFB-0003002-00）：GCU-LARE 插槽数 4、GCU-LARE 数 3、**桥接卡互联双向带宽 150GB/s**。来源同上。
- 【官方】GCU-LARE **连接线**（型号 **ESL-Q**，PN EFC-2015914000）：**双向传输速率 50GB/s**，QSFP-DD 接口，线长 60cm。来源：https://support.enflame-tech.com/onlinedoc_hw/1-t2x/t20/GCU-LARE_cable_product_manual/content/source/GCU-LARE_cable_product_manual.html
- 【媒体·发布会口径】邃思 2.0 / 云燧 T20 发布稿：GCU-LARE 全域互联技术「**双向 300GB/s**，支持数千张加速卡」。来源：http://m.eepw.com.cn/article/202107/426799.html
- 【未证实】300GB/s 与 150GB/s、50GB/s 之间的换算关系（是否单路 50GB/s × 多路聚合）无公开来源，不作推算。
- 【未找到公开数据】GCU-LARE 的 **SerDes 速率与 lane 数**：官方手册、招股书、问询回复均未披露。

### C2. 邃思 2.0 多卡互联拓扑 / 是否 8 卡

- 【招股书/交易所公告】原文：「自主架构的 **GCU-LARE 片间高速互连技术**，可以实现公司 AI 芯片之间**无需通过 PCIe 接口，高速直接互联**。」来源：http://vip.stock.finance.sina.com.cn/corp/view/vCB_AllBulletinDetail.php?stockid=688801&id=12099322
- 【招股书/交易所公告】互联代际演进原文：第 1 代 GCU-LARE 满足 **8 卡服务器内互联**（卡与主板走 PCIe 4.0x16）；第 2 代升级 GCU-LARE，**支持 4-16 卡间高效协同**；第 3 代主板升级 PCIe 5.0x16 + Chiplet/D2D；第 4 代升级版 GCU-LARE，**支持与交换机直连构建 128 卡高带宽域超节点**；第 5 代（规划）**支持内存语义访问**。来源同上。
- 【官方】桥接形态确认：桥接卡「通过 3 路 GCU-LARE 实现同一个服务器系统内 **4 卡之间的全互联**」；连接线「实现同服务器系统间跨 4 卡之间的互联」，**非 PCIe Switch 方案**。来源：见 C1 两条官方手册 URL。
- 【招股书/交易所公告】官方对标原文：「英伟达国内主流量产的 **Hopper 系列（第四代 NVLink）单卡互连带宽略高于公司产品水平**。」来源：http://vip.stock.finance.sina.com.cn/corp/view/vCB_AllBulletinDetail.php?stockid=688801&id=12099322
- 【招股书/交易所公告】问询回复比较表设有「**互联带宽（GB/s）**」列，但发行人一栏填「**发行人及国内公司产品性能豁免披露**」——**燧原自家互联带宽数字被豁免披露**。来源同上。

### C3. 云燧 S60 互联规格 / 量产时间

- 【官方·名称与代际更正】官方名 **燧原 S60 / Enflame S60**（非「云燧 S60」），基于 **GCU320（邃思 320）**，官方定义为「**第三代**人工智能加速卡」。来源：https://support.enflame-tech.com/onlinedoc_hw/5-s6x/S60/product_manual/content/source/S60_product_manual.html
- 【招股书/交易所公告·重要前提更正】**「邃思 2.0 ↔ 云燧 S60」不是同一代**：官方代际表为第三代 = 邃思 320 → 燧原 S60 推理卡；第四代 = 邃思 400 → 燧原 L600 训推一体模组；智算系统 = 云燧 OGX400、超节点 ESL32/64。**邃思 2.0 对应的是云燧 T2x（T20/T21）训练系列**。来源：http://vip.stock.finance.sina.com.cn/corp/view/vCB_AllBulletinDetail.php?stockid=688801&id=12099322
- 【官方】S60 接口规格仅列 **PCIe Gen5 X16**（FHFL 双槽、300W、48GB GDDR6、672GB/s、SR-IOV 4VF）；**官方参数表未列 GCU-LARE 或卡间互联带宽**。来源：见上条官方手册 URL。
- 【官方】S60 产品手册 V1.0 文档日期 **2024/7/23**，佐证 **2024 年发布/量产**（而非 2023 年）。来源同上。
- 【媒体】燧原 COO 张亚林公开回顾：2020 第一代、2022 第二代、**2024 第三代 = 燧原 S60**，报道称 S60「去年（2024）就已量产」；S60 一体机有 **4/8/16/32 卡**版本。来源：https://www.seccw.com/Document/detail/id/35044.html
- 【未找到公开数据】S60 卡间互联带宽数值。
- 【媒体】后续型号：第四代 **L600**（FP8，144GB / 3.6TB/s 存储带宽 / **800GB/s 互联带宽**）；**云燧 OGX400 单机八卡 OAM 全互联**，单机聚合带宽 2.8TB/s；**云燧 ESL 超节点单节点最高 64 卡全带宽互联**，聚合带宽 51.2TB/s。来源同上。
- 【招股书/交易所公告】「第四代产品 **L600 已回片但尚未大规模量产交付**。」来源：https://stock.stockstar.com/notice/SN2026090900000740.shtml

### C4. 燧原招股说明书中的互联官方原文摘录

- 【招股书·正式稿】「对标英伟达的 Tensor Core 加速计算单元和 NVLink 卡间互联技术，原创自主架构的 **GCU–CARE 加速计算单元和 GCU–LARE 片间高速互连技术**。」来源：https://stock.stockstar.com/notice/SN2026090900000740.shtml
- 【招股书·正式稿】「公司是国内首家自主研发了高密度、高互联带宽的 **OAM 模组方案**，多卡之间直接可通过 **UBB 底座**传输信号。」
- 【招股书·正式稿】超节点原文：「基于**第四代芯片**的超节点产品方案**基于 RoCE 协议**实现跨卡跨节点通信，可实现 **Scale up 规模 64 卡的高效互联**。」
- 【招股书·正式稿】正式招股说明书全文（2026-09）：https://vip.stock.finance.sina.com.cn/corp/view/vCB_AllBulletinDetail.php?CompanyCode=81112134&gather=1&id=12588272
- 【招股书·申报稿】2026-04-16 首轮问询回复全文：http://vip.stock.finance.sina.com.cn/corp/view/vCB_AllBulletinDetail.php?stockid=688801&id=12099322

### C5. 燧原 IPO 时间线（更正题面前提）

- 【交易所公告/媒体】辅导备案 2024-08-26（中金）→ 2025-11-01 重新备案（中信证券）→ **受理 2026-01-22**（上交所 2026 年首单，募资 60 亿）→ 问询函 2026-02-11 → 首轮问询回复 2026-04-16 → **上市委过会 2026-06-16** → 注册获批 → 申购 2026-09-02 → 招股说明书 2026-09-07；科创板代码 **688801**。来源：https://stcn.com/article/detail/3625179.html 、 https://www.cnstock.com/commonDetail/730128
- **题面「燧原 2025 年科创板 IPO 已受理/过会」不成立**：受理与过会均在 2026 年。

---

## D. 昆仑芯（P800 / 昆仑芯 3 代）

### D1. 自研 XPU Link 的规格与带宽

- 【官方】协议名为 **XPU Link / XPU-Link**，用于超节点内 Scale-Up 卡间互联；官方表述为「**已实现对 OISA 协议的兼容**」「自研 XPU Link 兼容主流 Scale-Up 通信标准 OISA」。来源：https://www.kunlunxin.com/news/4712.html 、 https://www.kunlunxin.com/news/4717.html
- 【官方】带宽的**唯一公开定量口径是相对值**：超节点「单柜内卡间实现全互联通信，**带宽提升高达 8 倍**」（相对传统单机 8 卡形态）。来源：https://www.kunlunxin.com/news/4541.html
- 【官方】同一相对口径在百度智能云渠道复述：「卡间互联带宽**提升 8 倍**，单整机柜训练性能提升 10 倍，单卡推理性能提升 13 倍」。来源：https://cloud.baidu.com/news/news_cdd5d1d3-4510-47a2-a806-55a795a50167
- 【未找到公开数据】**XPU Link 的绝对带宽 GB/s、SerDes 速率、lane 数、单向/双向口径**：已检索 kunlunxin.com 产品页与新闻页、cloud.baidu.com、OISA 2.0 规范正文、券商研报，均无此参数。
- 【未证实·应排除】网传「双向带宽达 1TB/s、延迟 50ns 以内、3D-Torus 拓扑」等说法，同文把 P800 误写为「第二代、7nm、HBM2e、128 TFLOPS」，与官方（第三代、96GB HBM3、345 TFLOPS）冲突，判定为 AI 生成内容，**不予采信**。来源（仅供排除）：https://cloud.baidu.com/article/5615257
- 【官方】P800 互联卡数：超节点内一层全互联，**单机柜 32 卡或 64 卡**；32 卡 Scale-Up 域由 **4 台 Switch Tray** 实现，任意两 XPU **仅 1 跳**。来源：https://cloud.baidu.com/news/news_cdd5d1d3-4510-47a2-a806-55a795a50167
- 【官方】传统单机形态为 8 卡（如「昆仑芯 P800 单机 8 卡满血版」）。来源：https://cloud.baidu.com/article/3715556

### D2. OISA 是什么 / 昆仑芯的角色

- 【官方】**OISA = Omni-directional Intelligent Sensing Express Architecture（全向智感互联）**，由**中国移动提出并牵头**的 GPU 卡间互联开放协议。来源：https://www.oisa.org.cn/index.php/aboutoisa/
- 【官方】**OISA Gen1**（2024-06-18 北京发布，国内首个 GPU 卡间互联开放协议）：支持 **128 GPU / 8 Switch**，任意卡间带宽 **800GB/s**，每 Switch **128 端口、51.2 Tb/s**。来源：https://kw.beijing.gov.cn/xwdt/bmdt/202406/t20240618_3816335.html
- 【官方】**OISA 2.0**（2025-08-23 中国算力大会，中国移动 + 之江实验室 + 百度等发布）：**1024 张芯片**、带宽**突破 TB/s**、时延**数百纳秒**。来源：https://www.stdaily.com/web/gdxw/2025-08/25/content_389911.html 、 https://www.oisa.org.cn/index.php/pdfdown/
- 【官方】OISA 2.0 规范实测数字：端口速率 **50Gb/s–1.6Tb/s** 聚合，由 **25/56/112/224 Gb/s** SerDes 通道绑定；示例拓扑 256 GPU × 8 端口、单端口 200 Gb/s、每交换芯片 51.2 Tb/s；直连拓扑仅 8 卡。来源：https://www.oisa.org.cn/index.php/seepart/
- 【官方】交换节点参考设计（OISA-2026-CN-002）：支持 **50–800Gb/s** 与 **56/112 Gb/s SerDes**，含 25.6T 板卡布局；**主要起草单位为盛科通信、中国移动研究院、之江实验室等，未见昆仑芯/百度署名**。来源：https://www.oisa.org.cn/wp-content/uploads/2026/06/OISA-%E4%BA%A4%E6%8D%A2%E8%8A%82%E7%82%B9%E5%8F%82%E8%80%83%E8%AE%BE%E8%AE%A1%E8%A7%84%E8%8C%83.pdf
- 【官方】**昆仑芯角色**：自 OISA 1.0 起即为「**重要合作伙伴**」，参与生态共建启动与 2.0 发布。来源：https://www.kunlunxin.com/news/4712.html 、 https://www.kunlunxin.com/news/4791.html
- 【未找到公开数据】昆仑芯是否为 OISA **发起单位/理事单位**（OISA 官网成员页为动态渲染，抓不到名单）；**OISA 与 UALink 的官方关系**（OISA 官网、规范正文、昆仑芯稿件均不提 UALink）。
- 【媒体】雷峰网将 UALink（AMD 牵头）与 OISA 并列为**相互竞争的开放 Scale-Up 路线**，称中国移动 OISA 走「国芯国连、协议共用」路线。来源：https://m.leiphone.com/category/chips/zWXsXt5CiDBI44yI.html
- 【官方】中国移动研究院 2026-05-08 发布 **OISA IP Core**，中国移动与中国电子、中科曙光、之江实验室、**百度**共同见证；平台已汇聚超 50 家单位；实测时延降低约 1.7 倍，DeepSeek-V3 千卡训练 **MFU 提升 1.53 倍**。来源：https://www.c114.com.cn/news/118/a1309911.html
- 【提醒】OISA 协议侧数字（Gen1 800GB/s@128 卡 → 2.0 TB/s 级@1024 卡）为官方可引用，但**不等于 XPU Link 的卡间带宽**，二者不可混同。

### D3. 昆仑芯超节点拓扑 / 跨柜网络 / 超节点规模

- 【官方】单柜 **32/64 卡**，跨柜**支持 IB/RoCE**，**支持万卡以上规模**；整柜 **120kW**，冷板液冷。来源：https://www.kunlunxin.com/news/4541.html
- 【官方】64 卡场景整柜 **28U**（16 × 1U Compute Tray「**1U 4 卡**」+ 8 × 1U Switch Tray + 2 × 2U Power shelf）；Scale-Out 每节点 4 张 PCIe 网卡位、**XPU:NIC = 1:1**、单节点最高 4×400G，配自研 **HPN** 架构。来源：https://cloud.baidu.com/news/news_cdd5d1d3-4510-47a2-a806-55a795a50167
- 【官方】**百度智能云天池超节点**：32 卡一层全互联、XPU 温控 75℃、标准整柜 0 配置交付、XPU:网卡 1:1。来源：https://cloud.baidu.com/product/tianchi.html
- 【官方】2026 WAIC：**天池 256** 的 XPU-Link 协议、交换与整机架构、液冷 CDU 全自研；**Scale-Up 域从 32 节点扩展到 1024 节点**；**故障切换 <50µs**、**带宽有效率 77%**，互联成本与单柜网络空间各省 50%，自研 Cable Tray 铜互连，**单柜 66kW**；**天池 512 单超节点可支撑万亿参数训练**。来源：https://www.cs.com.cn/ssgs/01/2026/07/19/detail_2026071910025376.html 、 https://news.ifeng.com/c/8ut7vaodSgy
- 【媒体】天池 256 于 2026-04 点亮、**2026-06 上市**，网络升级 **HPN5.0**、端到端时延优化 50%。来源：https://www.jiemian.com/article/14421027.html
- 【口径不一致·需注意】功率 **120kW 以内**（2025 口径）vs **66kW**（2026 天池 256）；单柜卡数 **32/64** vs 产品命名「天池 256 / 512」（后者是整机/超节点命名，非单柜卡数）。

### D4. 昆仑芯 3 代（P800 后续）的进展与互联升级

- 【官方】路线图：**1 代 2019、2 代 2021、3 代 2024**；**P800 即第 3 代**（XPU-P 架构，96GB HBM，FP16 345 TFLOPS）。来源：https://www.kunlunxin.com/%e6%a0%b8%e5%bf%83%e6%8a%80%e6%9c%af
- 【官方/媒体】2025-11-13 百度世界大会：**M100 面向大规模推理，2026 年初上市；M300 面向超大规模多模态训练 + 推理，2027 年初上市**。来源：https://www.stcn.com/article/detail/3492973.html
- 【官方】同一大会路线图：**2028 千卡级超节点、2029 三颗全新 N 系列芯片、2030 百万卡单集群**。来源：https://news.qq.com/rain/a/20251113A030F500
- 【官方】2026-07-20 WAIC **M100 实物首次展出**，**全国产供应链、对标 NVIDIA H20**，HBM 容量略低，主打高性价比推理。来源：https://www.c114.net.cn/industry/101818.html
- 【官方】下一代将**基于 M 系列推出千卡、四千卡超节点**。来源：https://news.ifeng.com/c/8ut7vaodSgy
- 【券商研报】招银国际：百度直接持股昆仑芯 **59.45%**；P800 显存规格**优于同类主流 GPU 20%–50%**，应用最大集群**超 3 万卡**。来源：https://www.cmbi.com.hk/upload/202509/20250919547901.pdf
- 【官方】P800 全国产集群完成**文心 5.1** 训练，**有效训练率 97%**、**万卡线性扩展度 >85%**。来源：https://news.ifeng.com/c/8ut7vaodSgy
- 【交易所公告/公告】2026-01-01 **昆仑芯以保密形式向港交所提交 A1 上市申请表**。来源：https://www.cnfin.com/kx/detail/20260102/4359511_1.html
- 【官方】2026Q2 财报会：李彦宏称**已完成三代 AI 芯片研发与商业化**，M100 已公布、M300 按节奏推进，**独立 IPO 正常推进**。来源：https://www.trendforce.cn/industry-news/semiconductors/20260819-7330.html
- 【未证实】**「昆仑芯 X600」：未找到任何公开来源**，判定为未证实。
- 【未找到公开数据】**M100 / M300 的互联规格升级**（协议版本、带宽、卡数）：未找到公开数据。

---

## 附：四家横向关键数字速查

| 项目 | 互联方案名 | 官方定量带宽 | 单机 8 卡 | 超节点规模 | 数据可信度 |
|---|---|---|---|---|---|
| 沐曦 C500 | MetaXLink（7 接口） | 官方无 GB/s；仅「H200 相当」；diagease 模板 49/90 GB/s | 板卡形态仅 2/4 卡；8 卡需服务器形态 | 16/32/64 卡，单柜 128 卡 | 招股书 + 公告 + 官方文档 |
| 沐曦 C600 | MetaXLink + MetaXLink-E | 官方无 GB/s；**公告明确「卡间互连带宽略有下降」** | 待查 | **128 卡超节点**（OAM2.0，1000W） | 官方产品页 + 交易所公告 |
| 天数智芯 天垓 100 | **无硬件专名**（走 PCIe P2P；软件层叫 IXCCL） | **共享 64 GB/s**（官方，不区分单向） | 官方未披露 | 未找到公开数据 | 官网规格页 + 港股招股书 |
| 天数智芯 天垓 150 | 同上（PCIe 4.0 x16） | **未找到公开数据** | 官方未披露 | 未找到公开数据 | 官网 + 招股书 |
| 燧原 邃思 2.0 / T20 | **GCU-LARE** | 桥接卡 150GB/s（双向）、连接线 50GB/s（双向）；发布会 300GB/s | 第 1 代满足 8 卡服务器内互联 | 第 4 代 128 卡高带宽域 | 官方手册 + 招股书 |
| 燧原 S60 | 官方未列 | **未找到公开数据** | 一体机 4/8/16/32 卡 | — | 官方手册 |
| 昆仑芯 P800 | **XPU Link**（兼容 OISA） | 官方**仅相对值**「提升 8 倍」，无绝对 GB/s | 传统单机 8 卡 | 单柜 32/64 卡；天池 Scale-Up 域 1024 节点 | 官方新闻 + 百度智能云 |
