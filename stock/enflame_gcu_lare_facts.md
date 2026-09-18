# 燧原科技（Enflame）多卡互联技术核实清单

## 一、互联方案正式名称与带宽

1. 【官方】燧原互联方案正式名称为 **GCU-LARE**，全称 **General Compute Unit - Link All Round Engine**，官方中文定义“燧原科技互联引擎”，未见“GCU-Link/Enflame Link/TopsLink”等名称。来源：https://support.enflame-tech.com/onlinedoc_hw/1-t2x/t20/GCU-LARE_linkcard_product_manual/content/source/GCU-LARE_linkcard_product_manual.html （S60/T20 官方词汇表同）

2. 【官方】GCU-LARE 桥接卡（型号 **ESL-B**，PN EFB-0003002-00）：含 3 路 GCU-LARE、4 个 GCU-LARE 插槽，**互联双向带宽 150GB/s**（该手册未区分单向；150GB/s 为“双向”标注值）。来源：同上

3. 【官方】GCU-LARE 连接线（型号 **ESL-Q**，PN EFC-2015914000）：**双向传输速率 50GB/s**，接口类型 QSFP-DD，线长 60cm。来源：https://support.enflame-tech.com/onlinedoc_hw/1-t2x/t20/GCU-LARE_cable_product_manual/content/source/GCU-LARE_cable_product_manual.html

4. 【媒体（发布会新闻稿）】邃思2.0/云燧T20 发布稿称“**GCU-LARE 全域互联技术**……提供**双向 300GB/s 互联带宽**，支持数千张云燧 CloudBlazer 加速卡互联”。来源：http://m.eepw.com.cn/article/202107/426799.html

5. 【未证实】上述 300GB/s 与单件 150GB/s（桥接卡）、50GB/s（连接线）的换算关系（如“300GB/s 为芯片侧多路 GCU-LARE 聚合、50GB/s 为单路”）：**未找到公开数据**可证实，本清单不作推算认定。

6. 【未找到公开数据】**SerDes 速率（Gbps/lane）与 lane 数**：官方产品手册、招股说明书、问询回复、公开新闻均未披露，**未找到公开数据**。

7. 【未找到公开数据】“GCU-Link”“Enflame Link”“TopsLink”作为燧原互联技术名称：**未找到公开数据**（“驭算 TopsRider”为软件栈名称，非互联技术）。

## 二、邃思 2.0 的多卡互联拓扑

8. 【招股书/交易所公告】官方原文：“自主架构的 **GCU-LARE 片间高速互连技术**，可以实现公司 AI 芯片之间**无需通过 PCIe 接口，高速直接互联**。”来源：http://vip.stock.finance.sina.com.cn/corp/view/vCB_AllBulletinDetail.php?stockid=688801&id=12099322

9. 【招股书/交易所公告】互联演进官方原文：“**第 1 代**采用自研专用的 GCU-LARE 技术，能满足 **8 卡服务器内互联**，AI 加速卡与主板通过 PCIe 4.0x16 互联，基于商业网卡可实现千卡规模集群；**第 2 代**升级 GCU-LARE 技术，**支持 4-16 卡间高效协同**，实现 3D 混合并行通信；**第 3 代**AI 加速卡与主板升级为 PCIe 5.0x16 互联，采用 **Chiplet 架构搭配 D2D 互联**，通过网卡 ScaleOut 支撑万卡集群；**第 4 代**采用升级版 GCU-LARE，**支持与交换机直连构建 128 卡高带宽域超节点**；**第 5 代（规划）**支持内存语义访问。”来源：同上

10. 【官方】邃思2.0 训练方案的桥接/互联模组确实存在：**GCU-LARE 桥接卡（板卡）** 用于 T20 系列“同一个服务器系统内 **4 卡之间的全互联**”，另有 **GCU-LARE 连接线**实现“同服务器系统间跨 4 卡之间的互联”——即走专有互联 + 桥接卡/连接线，而非 PCIe Switch。来源：https://support.enflame-tech.com/onlinedoc_hw/1-t2x/t20/GCU-LARE_linkcard_product_manual/content/source/GCU-LARE_linkcard_product_manual.html 与 https://support.enflame-tech.com/onlinedoc_hw/1-t2x/t20/GCU-LARE_cable_product_manual/content/source/GCU-LARE_cable_product_manual.html

11. 【招股书/交易所公告】官方对英伟达的对标表述：“国外的多卡互连技术主导为英伟达的 NVLink……英伟达国内主流量产的 Hopper 系列产品（应用第四代 NVLink 技术）**单卡互连带宽略高于公司产品水平**。”来源：http://vip.stock.finance.sina.com.cn/corp/view/vCB_AllBulletinDetail.php?stockid=688801&id=12099322

12. 【招股书/交易所公告】问询回复中的“发行人 AI 加速卡及模组规格与可比公司比较表”设有“**互联带宽（GB/s）**”列，但发行人一栏为“**发行人及国内公司产品性能豁免披露**”，即**该表中燧原互联带宽数字被豁免披露**。来源：同上

13. 【招股书/交易所公告·正式招股说明书】原文：“公司基于自主指令集，对标英伟达的 Tensor Core 加速计算单元和 NVLink 卡间互联技术，原创自主架构的 **GCU–CARE 加速计算单元和 GCU–LARE 片间高速互连技术**。”来源：https://stock.stockstar.com/notice/SN2026090900000740.shtml

14. 【招股书/交易所公告·正式招股说明书】原文（Scale-Up 定义）：“通过高速互联架构（如英伟达的 NVLink、**公司的 GCU-LARE**）实现 AI 芯片间超低延迟通信以提升单个节点内部的数据流通效率”；并载明“公司是国内首家自主研发了**高密度、高互联带宽的 OAM 模组方案**，多卡之间直接可通过 **UBB 底座**进行信号互相传输”。来源：同上

15. 【招股书/交易所公告·正式招股说明书】超节点官方原文：“基于**第四代芯片**的超节点产品方案**基于 RoCE 协议**实现跨卡和跨节点的卡间通信……可实现 **Scale up 规模为 64 卡的高效互联**。”来源：同上

## 三、云燧 S60 的互联规格、发布/量产时间与后续型号

16. 【官方】型号名称更正：官方产品名为“**燧原 S60 / Enflame S60**”（属“燧原 S60 推理系列”），“云燧”品牌对应 T 系/i 系；S60 基于 **GCU320（邃思320）**，官方定义为“**第三代**人工智能加速卡”。来源：https://support.enflame-tech.com/onlinedoc_hw/5-s6x/S60/product_manual/content/source/S60_product_manual.html

17. 【官方】S60 接口规格只有 **PCIe Gen5 X16**（FHFL 双槽、300W、48GB GDDR6、显存带宽 672GB/s、SR-IOV 4VF、12VHPWR）；**官方参数表中未列出 GCU-LARE 或卡间互联带宽**。来源：同上

18. 【招股书/交易所公告】官方代际对应表：第三代 = 邃思320 芯片 → **燧原 S60 推理卡**；第四代 = 邃思400 芯片 → **燧原 L600 训推一体模组**，智算系统为 **云燧 OGX400、超节点 ESL32/64**。**即“邃思2.0 ↔ 云燧S60”并非同一代**：邃思2.0 对应云燧 T2X（T20/T21）训练系列。来源：http://vip.stock.finance.sina.com.cn/corp/view/vCB_AllBulletinDetail.php?stockid=688801&id=12099322

19. 【官方】S60 产品手册 V1.0 文档日期为 **2024/7/23**（2025/1/16 更新至 V1.2），可佐证 S60 量产/发布时点在 2024 年而非 2023 年。来源：https://support.enflame-tech.com/onlinedoc_hw/5-s6x/S60/product_manual/content/source/S60_product_manual.html

20. 【媒体】燧原 COO 张亚林公开回顾“三代四颗芯片”历程：2020 年第一代、2022 年第二代（训练+推理）、**2024 年第三代——燧原 S60**；同文称 S60 “去年（2024）就已量产”“基于 GCU320”。来源：https://www.seccw.com/Document/detail/id/35044.html

21. 【媒体】S60 多卡形态主要走一体机/系统方案：“燧原 S60 的一体机产品涵盖 **4 卡、8 卡、16 卡、32 卡**等不同扩展版本，8 卡为标准版，16 卡可支持满血版 DeepSeek 671B，32 卡 POD 版用于规模化应用。”来源：https://www.seccw.com/Document/detail/id/35044.html

22. 【媒体】S60 的一体机/多卡方案互联带宽：**未找到公开数据**（该报道未给出 S60 卡间互联带宽数值）。

23. 【媒体】S60 后续升级型号：第四代 **燧原 L600 训推一体模组**（国内首创原生 FP8，**144GB 存储、3.6TB/s 存储带宽、800GB/s 互联带宽**）；基于 L600 的 **云燧 OGX 系列**，其中 **OGX400 单机八卡 OAM 全互联**，单机聚合带宽 2.8TB/s；**云燧 ESL 超节点单节点最高 64 卡全带宽互联**，单节点聚合带宽 51.2TB/s。来源：https://www.seccw.com/Document/detail/id/35044.html

24. 【招股书/交易所公告】正式招股说明书口径：“公司第四代产品 **L600 为训推一体加速模组**，截至招股说明书签署日，该产品**已回片但尚未大规模量产交付**。”来源：https://stock.stockstar.com/notice/SN2026090900000740.shtml

25. 【未证实】“S60 的升级型号为 S70/其他 S 系型号”：**未找到公开数据**；官方披露的下一代为第四代 L600（训推一体）与超节点 ESL32/64。

## 四、科创板 IPO 招股说明书及审核文件中的互联原文

26. 【招股书/交易所公告】时间线更正（用户所述“2025 年已受理/过会”不成立）：辅导备案 2024-08-26（中金）→ 2025-11-01 重新备案（中信证券）→ **受理 2026-01-22**（上交所 2026 年首单）→ 问询函 2026-02-11 → 首轮问询回复 2026-04-16 → **上市委过会 2026-06-16** → 注册获批 → 申购 2026-09-02 → 招股说明书 2026-09-07。来源：https://stcn.com/article/detail/3625179.html 、https://www.cnstock.com/commonDetail/730128 、http://vip.stock.finance.sina.com.cn/corp/view/vCB_AllBulletinDetail.php?stockid=688801&id=12099322

27. 【招股书/交易所公告】招股说明书（申报稿）2026-04-16 全文（新浪公告页，含保荐机构中信证券、审计毕马威）：http://vip.stock.finance.sina.com.cn/corp/view/vCB_AllBulletinDetail.php?stockid=688801&id=12099323

28. 【招股书/交易所公告】正式招股说明书（2026-09）全文（新浪公告页）：https://vip.stock.finance.sina.com.cn/corp/view/vCB_AllBulletinDetail.php?CompanyCode=81112134&gather=1&id=12588272

29. 【招股书/交易所公告】上交所披露的发行人文件 PDF（含“发行人是依法设立且持续经营三年以上……”上市条件核查）：http://static.sse.com.cn/stock/disclosure/announcement/c/202601/002175_20260122_654W.pdf （注：本次核实中该静态 PDF 直连返回 404，仅作为上交所披露路径引用）

30. 【招股书/交易所公告】上交所（big5 镜像）上市公告书 PDF（2026-09-08）：http://big5.sse.com.cn/disclosure/listedinfo/announcement/c/new/2026-09-08/688801_20260908_BIW0.pdf

31. 【媒体】上交所官网披露“燧原科技科创板 IPO 已获受理，中信证券为独家保荐人”，募资 60 亿元，用于第五代、第六代 AI 芯片研发及产业化。来源：https://stcn.com/article/detail/3625179.html

32. 【媒体】过会报道（上海证券报·中国证券网，2026-06-16）：“抢滩万亿国产替代市场，燧原科技 IPO 过会！”来源：https://www.cnstock.com/commonDetail/730128

33. 【官方】燧原官方硬件文档中心（互联相关产品手册索引，含 GCU-LARE 桥接卡/连接线手册）：https://support.enflame-tech.com/onlinedoc_hw/

## 附：可信度分级使用说明
- 【官方】：enflame.com / support.enflame-tech.com 官方产品手册与词汇表。
- 【招股书/交易所公告】：上交所受理文件、首轮审核问询函之回复、正式招股说明书。
- 【媒体】：发布会新闻稿转载、电子工程专辑/证券时报/上海证券报等。
- 【未证实】/【未找到公开数据】：无公开来源支撑，未作数字推定。
