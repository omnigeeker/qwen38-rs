# 华为昇腾 910C：封装 / Die 配置 / 代工 / HBM 供应链——可引用事实清单

- 可信度分级：【官方】>【招股书/年报/交易所公告】>【券商研报】>【媒体/自媒体】；专业机构（TechInsights、SemiAnalysis、Reuters、Bloomberg、TrendForce）在中标注。
- 无法核实一律写「未找到公开数据」/「未证实」。
- 检索时间：本会话（页面日期跨度 2024-10 至 2026-09）。

---

## 1. 是否为「两颗 910B 级 Die 的双 Die 封装」

- 【官方】华为 + 硅基流动论文《Serving Large Language Models on Huawei CloudMatrix384》（arXiv 2506.12708，通讯作者 @huawei.com / @hisilicon.com）§3.3.1 原文：**"The Ascend 910 is a dual-die package: two identical compute dies are co-packaged, sharing eight on-package memory stacks and connected by a high-bandwidth cross-die fabric."** ——两颗相同计算 Die 共封装，共享 8 个片上内存堆栈，Die 间为高带宽 cross-die fabric。URL: https://ar5iv.labs.arxiv.org/html/2506.12708
  - 注意：论文正文用"Ascend 910"表述；论文语境为 CloudMatrix384，华为对外称该超节点搭载 910C。论文未在 §3.3.1 直接写"910C"字样。
- 【官方】同论文 §3.3.1：每颗 Die 含 24 个 AI Cube（AIC）+ 48 个 AI Vector（AIV）核，支持 FP16/BF16/INT8；每颗 Die 集成 7 个 UB 高速收发器（scale-up）+ 独立 RDMA 接口（scale-out）。URL: https://ar5iv.labs.arxiv.org/html/2506.12708
- 【媒体·TechInsights / 彭博】TechInsights 拆解多颗 910C 样品后确认："Huawei's current-generation accelerator, the 910C, is made by packaging two 910B dies together."，且 Die 由台积电制造。URL: https://www.businesstimes.com.sg/companies-markets/huawei-used-tsmc-samsung-sk-hynix-components-top-ai-chips-techinsights （彭博原文：https://www.bloomberg.com/news/articles/2025-10-03/huawei-used-tsmc-samsung-sk-hynix-components-in-top-ai-chips ，付费墙）
- 【媒体·路透】路透 2025-04-21：910C 通过"把两颗 910B 处理器封装进单一封装"实现，推理性能约为英伟达 H100 的 60%。路透原文页面不可访问（HTTP 401）：https://www.reuters.com/world/china/huawei-readies-new-ai-chip-mass-shipment-china-seeks-nvidia-alternatives-sources-2025-04-21/ ；可访问转述：https://www.sdxcentral.com/news/huawei-unveils-ascend-920-ai-chip-will-start-shipping-910c-to-chinese-customers-from-may-report-sdx/
- 【券商研报·中信证券（2025-06-24 电子行业算力专题系列报告5）】910C"采用双 Die 封装，每个 Die 提供约 376 TFLOPS（BF16/FP16）的算力，整个芯片算力高达 752 TFLOPS"。URL: https://finance.sina.com.cn/stock/stockzmt/2025-06-25/doc-infcfxpq8236746.shtml
- 【官方·高管演讲】华为轮值董事长徐直军 2025 全联接大会（2025-09-18）：910C 算力 800 TFLOPS（FP16），支持 FP32/HF32/FP16/BF16/INT8，互联带宽 784GB/s，HBM 容量 128GB、带宽 3.2TB/s；路线图显示 910C 为 2025 年 Q1 最新发布。（与中信证券 752 TFLOPS 口径略有差异，两者并存标注。）URL: https://www.stcn.com/article/detail/3345926.html
- 【官方·高管演讲】徐直军同一演讲：2025 年 3 月推出基于昇腾 910C 的 Atlas 900 超节点，满配 384 卡，"最大算力可达 300 PFLOPS"。URL: https://paper.cnstock.com/html/2025-09/19/content_2123758.htm

---

## 2. 封装技术：CoWoS？国产 2.5D？谁做封装？

### 2.1 910C 的封装形态

- 【券商研报·中信证券】910C 单芯片集成 **8 个内存堆栈（每堆栈 16GB），共 128GB HBM**，内存带宽 3.2TB/s；单芯片互联达 392GB/s 单向带宽（对比英伟达 NVLink 第四代双向 900GB/s）。URL: https://finance.sina.com.cn/stock/stockzmt/2025-06-25/doc-infcfxpq8236746.shtml
- 【媒体·TrendForce 转 Wccftech / 分析师 @ohlennart】910C"采用比英伟达芯片更简单的方案：**两个硅中介层（silicon interposer）通过有机基板（organic substrate）互联**"；规格 800 TFLOP/s FP16、3.2TB/s 内存带宽，约为 H100 的 80%。URL: https://www.trendforce.com/news/2025/03/13/news-huaweis-ascend-910c-takes-on-nvidia-as-chinas-ai-race-heats-up-more-alleged-details/
- 【未证实·自媒体拆解帖】中文社媒 decap 帖称：910C 为 TSMC N7、确为两块 910B 拼贴；封装为"两块 interposer（<2 倍光罩能力）之间透过 substrate 互联"，对比 Blackwell 是 3.3 倍光罩 interposer 且两颗 GPU 直接在中介层互联。属自媒体拆解，无权威机构背书。URL: https://sina.cn/news/detail/5192969583330081.html
- **是否使用台积电 CoWoS：未找到公开数据。** 官方论文、TechInsights、SemiAnalysis 均未明确表述 910C 采用 CoWoS。
- 【媒体·Notebookcheck（转 Bloomberg/SemiAnalysis）】"910C 的封装有些简陋，容易出现散热效率低下的问题"；性能约为 H100 的一半左右。URL: https://www.notebookcheck-cn.com/910C-AI.1131027.0.html

### 2.2 封装/组装厂

- 【媒体·证券时报（2025-06-24）】盛合晶微（SJ Semiconductor）"也是华为昇腾系列 AI 芯片的核心代工厂之一，其 **3 倍光罩尺寸 TSV 硅通孔载板技术（亚微米级互联精度）和三维多芯片集成封装技术**，为昇腾芯片的高密度集成提供了关键支持"；公司建成国内首条大规模量产级 TSV 硅通孔生产线，关键工艺良率 99.5% 以上，实现微凸点间距 <40μm、多层堆叠，年产能达 50 万片晶圆。URL: https://www.stcn.com/article/detail/2241117.html
- 【招股书（仅检索快照可见；PDF 直连 HTTP 404）】盛合晶微招股书自述："在芯粒多芯片集成封装领域……尤其对于业界最主流的**基于硅通孔转接板（TSV Interposer）的 2.5D 集成**，公司是中国大陆量产最早、生产规模最大的企业之一，代表中国大陆在该技术领域的最先进水平"。URL（直接抓取返回 404）: https://static.sse.com.cn/stock/disclosure/announcement/c/202601/002104_20260107_5VZI.pdf
- 【媒体引用招股书】盛合晶微对第一大客户收入占比 2022→2025H1 由 40.56% 飙升至 **74.40%**，前五大客户收入占比突破 90%；核心收入来自 AI 芯片（如华为昇腾）的 **2.5D 封装订单，2025H1 占比 56.24%**；2024 年 2.5D 封装市占率 85%；有息借款突破 80.39 亿元、资产负债率 68.38%；募资 48 亿元投向三维集成封装项目。URL: http://news.cnfol.com/chanyejingji/20251107/31775227.shtml
- 【媒体·新快报（2026-06-08）】"通富微电**深度绑定华为 AI 和手机芯片**，高端算力封装订单持续放量"；长电科技为全球第三、国内第一封测龙头，掌握 XDFOI 高密度堆叠、2.5D/3D 异构集成，先进封装收入占比超 70%。URL: https://ep.ycwb.com/epaper/xkb/html/2026-06/08/content_1515_754746.htm
- 【媒体·OFweek/产联社（2026-07-13，转引 EET China 等）】2026 上半年国内四大封测龙头累计宣布扩产投资 **274.2 亿元**：甬矽电子 124 亿（马来西亚槟城+宁波余姚）、长电科技 78 亿（上海临港，2026 固投预算约 100 亿，2025 先进封装收入 270 亿元、占全年营收 69.5%）、通富微电 44 亿定增、华天科技 30 亿（南京存储封装）。URL: https://ee.ofweek.com/2026-07/ART-12003-2815-30694231.html

### 2.3 华为「韬定律」/ 逻辑折叠（自媒体可信度低，单独标注）

- 【官方论文 + 媒体转述】华为半导体业务负责人何庭波 2026-05-25 在ChinaXiv 首发、2026-07-03 发布 V2《面向多层级电子系统的时间缩微理论》（业内称"韬定律"）：以"时间缩微"替代"几何缩微"，通过**逻辑折叠**把电路拆解并垂直堆叠到多层有源层，以超细间距**混合键合**互连。麒麟 2026 为"保守版"：混合键合间距 1.5μm，晶体管密度 155→238 MTr/mm²（+53.5%），功耗 -41%，面积 -37.5%，主频 3.1GHz。规划 2031 年密度突破 400 MTr/mm²，**2030 年昇腾 990 将成为首款采用逻辑折叠的 AI 加速芯片**。URL: https://ee.ofweek.com/2026-07/ART-12003-2815-30694231.html
- 【券商研报·伯恩斯坦（经 OFweek 转述，2026-06-04）】传统 3D 封装只是"把两颗独立芯片粘在一起"，华为是在设计阶段就把逻辑电路拆分到两层晶圆；预计到 2030 年采用 2.5D/3D 堆叠的晶圆将增长 7 倍至 350 万片/月、渗透率 38%。属第三方转述，未见原始研报。URL: https://ee.ofweek.com/2026-07/ART-12003-2815-30694231.html
- 【媒体·新快报】黄仁勋公开表示"这对华为来说是突破，但对台积电并不是威胁。台积电使用芯片堆叠和 3D 封装技术已经快 10 年"；该说法引发"逻辑折叠 ≠ 3D 封装"的产业争议。何庭波回应称韬定律"不是为了取代摩尔定律"。URL: https://ep.ycwb.com/epaper/xkb/html/2026-06/08/content_1515_754746.htm
- **「韬定律/逻辑折叠」是否已用于 910C：未找到公开数据。** 华为公开材料中首款采用逻辑折叠的 AI 加速芯片为 2030 年昇腾 990。
- 【媒体·OFweek】大陆 CoWoS 产能不足 2 万片/月（全球占比不足 11%），台积电 2026 年底约 12.7 万片/月；2.5D/3D 高阶技术国产化率仍不足 10%，高端渗透率仅 5%–8%。URL: https://ee.ofweek.com/2026-07/ART-12003-2815-30694231.html

---

## 3. 代工：台积电 7nm（N7）vs 中芯 N+2

- 【媒体·SemiAnalysis（2025-04-16）】"SMIC 虽具备 7nm，但**绝大多数 Ascend 910B 和 910C 都是用台积电 7nm 制造的**。事实上，美国政府、TechInsights 等获取的 Ascend 910B 和 910C，每一颗都使用了台积电 Die。"URL: https://newsletter.semianalysis.com/p/huawei-ai-cloudmatrix-384-chinas-answer-to-nvidia-gb200-nvl72
- 【媒体·SemiAnalysis】华为通过第三方公司 Sophgo **采购约 5 亿美元 7nm 晶圆**以规避制裁；并称"台积电因此被罚 10 亿美元，仅为其获利的 2 倍"。URL: https://newsletter.semianalysis.com/p/huawei-ai-cloudmatrix-384-chinas-answer-to-nvidia-gb200-nvl72
- 【媒体·SemiAnalysis】台积电累计提供 **290 万颗 Die**（跨 2024–2025），足够做 80 万颗 910B + 105 万颗 910C；Sept 2025 报告口径为"超过 290 万颗 Ascend Die，可用于 910B 和 910C"。URL: https://newsletter.semianalysis.com/p/huawei-ai-cloudmatrix-384-chinas-answer-to-nvidia-gb200-nvl72 与 https://newsletter.semianalysis.com/p/huawei-ascend-production-ramp
- 【媒体·路透（2025-04-08 独家）】TSMC 可能因美国调查面临 **10 亿美元或更高罚款**。页面不可访问（HTTP 401）：https://www.reuters.com/technology/tsmc-could-face-1-billion-or-more-fine-us-probe-sources-say-2025-04-08/
- 【媒体·路透（经 SDxCentral 转述，2024-10-28）】TSMC 因在华为 Ascend 910B 上发现相似芯片而**暂停向中国公司 Sophgo 出货**（此前已向其出货"数十万颗"芯片）；Sophgo 否认与华为有任何直接或间接业务关系，并称已向台积电提交详细调查报告。URL: https://www.sdxcentral.com/news/china-based-sophgo-implicated-in-tsmc-huawei-chip-row-deny-all-allegations-report/
- 【媒体·TechInsights/彭博】台积电回应：TechInsights 近期调查的 910C 硬件"看起来是用**该机构 2024 年 10 月分析过的 Die 制造的，而不是用最近制造或更先进技术的 Die**"，且"**该芯片的出货与制造自那时起已经停止**"；台积电称自 2020 年 9 月中起未再向华为供货。URL: https://www.businesstimes.com.sg/companies-markets/huawei-used-tsmc-samsung-sk-hynix-components-top-ai-chips-techinsights
- 【官方·高管演讲】徐直军 2025 全联接大会："**由于受美国的制裁，华为不能到台积电去投片**，单颗芯片的算力相比英伟达存在差距"，因此走"超节点+集群"路线。URL: https://www.stcn.com/article/detail/3345926.html
- 【媒体·SemiAnalysis（2025-09-08）】SMIC 7nm 及以下先进节点总产能：2025 年底保守估计约 **4.5 万片/月**，2026 年 6 万片/月，2027 年 8 万片/月；Ascend 每月**最多只需约 2 万片**（wspm）即可产出数百万颗 Die；预计 SMIC 到 2025 年底不再是 Ascend 生产的瓶颈。URL: https://newsletter.semianalysis.com/p/huawei-ascend-production-ramp
- 【媒体·SemiAnalysis】SMIC 7nm 级良率偏低（工艺不成熟 + 出口管制 + Ascend 这类大 Die 本身难做），因此 SMIC 只把相对低的产能比例分配给 Ascend；较小面积的手机处理器商业上更划算。URL: https://newsletter.semianalysis.com/p/huawei-ascend-production-ramp
- 【媒体·SemiAnalysis】华为自建 Fab 网络（含 SiCarrier），报道称其合计产能到 2026 年可能超过 SMIC；2024 年华为 WFE 支出 73 亿美元（同比 +27%），两年内从近乎零升至全球第 4 大设备客户。URL: https://newsletter.semianalysis.com/p/huawei-ascend-production-ramp
- 【券商研报·中信证券】"中芯国际先进制程产能作为国产算力根基，其战略价值进一步凸显。此外先进制程衍生的封测、载板、电源管理等环节也有望同步受益"（未点名具体载板/封测厂商）。URL: https://finance.sina.com.cn/stock/stockzmt/2025-06-25/doc-infcfxpq8236746.shtml
- **910B 与 910C 的台积电/中芯产能拆分比例：未找到公开数据。**

---

## 4. HBM：供应商、堆栈数、容量与带宽

### 4.1 供应商与来源

- 【媒体·TechInsights/彭博（2025-10-03）】TechInsights 在**两颗不同的 910C 样品**中分别发现**三星与 SK 海力士的上一代 HBM，型号指定为 HBM2E**。URL: https://www.businesstimes.com.sg/companies-markets/huawei-used-tsmc-samsung-sk-hynix-components-top-ai-chips-techinsights
- 【媒体·SemiAnalysis（2025-09-08）】三星是向中国供货最多的 HBM 供应商：**仅三星就向中国直接提供 1140 万颗 HBM 堆栈，其中在 BIS 2024-12-02 宣布管制、2024-12-31 全面执行的"1 个月窗口"内高达 700 万颗**；计入其他供应商与渠道，合计约 **1300 万颗堆栈**。URL: https://newsletter.semianalysis.com/p/huawei-ascend-production-ramp
- 【媒体·SemiAnalysis（2025-04-16）】HBM 走私路径：CoAsia Electronics 作为三星 HBM 在大中华区独家分销商，将 HBM2E 供给 ASIC 设计服务公司 Faraday，由 SPIL 与一颗廉价 16nm 逻辑 Die 一起"封装"成系统级封装输入中国，中国厂商再**解焊回收 HBM**（使用低温弱焊凸点便于拆解）。URL: https://newsletter.semianalysis.com/p/huawei-ai-cloudmatrix-384-chinas-answer-to-nvidia-gb200-nvl72
- 【媒体·彭博】SK 海力士声明"自 2020 年限制措施实施后就已停止与华为的一切交易"；三星称"持续严格遵守"美国出口规定，与清单实体"没有任何业务关系"。URL: https://www.businesstimes.com.sg/companies-markets/huawei-used-tsmc-samsung-sk-hynix-components-top-ai-chips-techinsights
- 【官方·高管演讲】徐直军：2026 年一季度发布的**昇腾 950PR 将采用华为自研 HBM**；昇腾 950 系列将在"低精度数据格式、向量算力、互联带宽及自研 HBM 等方面实现根本性提升"。URL: https://www.stcn.com/article/detail/3345926.html
- 【媒体·TechPowerUp 转 MK South Korea / The Korea Times / Wccftech（2026-02-11）】CXMT 已启动 **HBM3 模组量产**；匿名消息源称"华为正与长鑫（CXMT）共同开发 HBM，预计即使良率偏低也将进入量产"；CXMT 计划 HBM3 月产能 **6 万片**，约相当于其 2026 年 30 万片/月总产能的 20%。**属匿名消息源，未证实。** URL: https://www.techpowerup.com/346207/cxmt-reportedly-plans-to-dedicate-20-of-mass-production-capacity-to-hbm3-line-in-2026
- 【未证实·自媒体】锐报研究所：CXMT 2026 年仅能生产约 200 万组 HBM 堆叠，折算**只够装 25–30 万颗 910C**。URL: https://ruibao.news/institute/tech-tracker/2026-04/
- **CXMT 是否已通过 910C 认证 / 是否已实际用于 910C：未找到公开数据 / 未证实。**

### 4.2 堆栈数、容量、带宽

- 【券商研报·中信证券】910C 单芯片集成 **8 个内存堆栈 × 16GB = 128GB HBM**，带宽 **3.2TB/s**。URL: https://finance.sina.com.cn/stock/stockzmt/2025-06-25/doc-infcfxpq8236746.shtml
- 【官方·高管演讲】徐直军：910C HBM 容量 **128GB**、内存带宽 **3.2TB/s**；互联带宽 784GB/s。URL: https://www.stcn.com/article/detail/3345926.html
- 【媒体·SemiAnalysis（推算式）】1300 万颗 HBM 堆栈"足以装配 **160 万颗昇腾 910C 封装**"——按此推算约 **8 个 HBM 堆栈/颗 910C**（与中信证券 8 堆栈口径一致）。URL: https://newsletter.semianalysis.com/p/huawei-ascend-production-ramp
- 【媒体·SemiAnalysis】"1300 万颗 HBM 堆栈……但中国到年底会被 HBM 卡住（bottlenecked by HBM by the end of the year）"；"中国今年从台积电+中芯产能看轻松可做 80.5 万颗以上昇腾，但不会做，因为 HBM 不够"。URL: https://newsletter.semianalysis.com/p/huawei-ascend-production-ramp
- 【媒体·SemiAnalysis】"没有更多外国 HBM，华为明年（2026）连 100 万颗昇腾都造不出来"；NVDA/AMD 在中国将实际上零竞争；替代方案 GDDR/LPDDR 不适合前沿模型。URL: https://newsletter.semianalysis.com/p/huawei-ascend-production-ramp

---

## 5. 量产时点、月度产出、良率、2025–2026 量级

### 5.1 量产时点

- 【媒体·路透（2025-04-21）】华为准备**从 2025 年 5 月起向中国客户批量出货 910C**。路透原文不可访问（401）；转述 URL: https://www.sdxcentral.com/news/huawei-unveils-ascend-920-ai-chip-will-start-shipping-910c-to-chinese-customers-from-may-report-sdx/
- 【官方·高管演讲】徐直军路线图显示 **910C 为 2025 年 Q1 最新发布**；Atlas 900 超节点（384 卡 910C）于 2025 年 3 月推出。URL: https://www.stcn.com/article/detail/3345926.html 与 https://paper.cnstock.com/html/2025-09/19/content_2123758.htm
- 【媒体·TechInsights/彭博】910C "entering mass shipment earlier this year"（2025 年早些时候进入批量出货）。URL: https://www.businesstimes.com.sg/companies-markets/huawei-used-tsmc-samsung-sk-hynix-components-top-ai-chips-techinsights

### 5.2 出货量 / 产量

- 【媒体·SemiAnalysis（2025-09-08）】华为 Ascend 出货：**2024 年 50.7 万颗**（以 910B 为主）；**2025 年 80.5 万颗，其中 910C 为 65.3 万颗**（含台积电与中芯制造的 Die）。URL: https://newsletter.semianalysis.com/p/huawei-ascend-production-ramp
- 【媒体·SemiAnalysis】台积电"Die 银行"将在报告发布后**约 9 个月内耗尽**；2024 与 2025 正是靠这批外来 Die 支撑产量，否则数字会低得多。URL: https://newsletter.semianalysis.com/p/huawei-ascend-production-ramp
- 【媒体·SemiAnalysis】预计 SMIC 当年（2025）生产"100 万颗 910C 和近 50 万颗 910B"——**原文未明确是 Die 还是成品封装，且与 80.5 万颗总出货口径存在张力，引用需谨慎。** URL: https://newsletter.semianalysis.com/p/huawei-ascend-production-ramp
- 【媒体·Notebookcheck（转 Bloomberg/SemiAnalysis，2025-10-03）】按当时生产速度，华为预计产出 **65.3 万颗"使用两颗台积电 Die"的 Ascend 910C**；台积电 Die 库存**足够支撑到"明年夏季之前"的生产**；受 HBM 与 Die 瓶颈，**2026 年只能生产约 100 万颗 910C**。URL: https://www.notebookcheck-cn.com/910C-AI.1131027.0.html
- 【媒体·彭博（2025-09-29）】《Huawei to Double Output of Top AI Chip as Nvidia Wavers in China》。页面付费墙不可访问。URL: https://www.bloomberg.com/news/articles/2025-09-29/huawei-to-double-output-of-top-ai-chip-as-nvidia-wavers-in-china
- 【官方·高管演讲】徐直军：昇腾超节点**自上市以来短短数月已累计部署 300 多套**，服务 20 多个行业客户（华为董事杨超斌在昇腾产业峰会披露）。URL: https://paper.cnstock.com/html/2025-09/19/content_2123758.htm
- 【未证实·自媒体】锐报研究所：华为昇腾 910C 的 **2026 年产量目标约 60 万颗，为上年两倍**；全线 Ascend Die 有望触及 160 万片，由中芯国际增强 7nm 工艺承担。URL: https://ruibao.news/institute/tech-tracker/2026-04/
- 【未证实·媒体转述分析师 @ohlennart】TrendForce 转述：华为计划 2025 年生产 **10 万颗 910C + 30 万颗 910B**（2024 年为 20 万颗 910B、0 颗 910C）。该数字明显低于 SemiAnalysis 的 65.3 万颗，**未证实**。URL: https://www.trendforce.com/news/2025/03/13/news-huaweis-ascend-910c-takes-on-nvidia-as-chinas-ai-race-heats-up-more-alleged-details/
- 【媒体·香港 unwire 转述美智库】美智库估计华为有能力年产 75 万枚 AI 芯片；传中芯目标 2025 年底产 5 万枚 7nm 芯片。抓取正文被截断，内容不可访问。URL: https://unwire.hk/2025/03/11/huawei-can-make-750000-advanced-chip/ai/

### 5.3 良率

- 【媒体·金融时报（经 TrendForce 转述）】华为把最新 AI 芯片的**良率翻倍到接近 40%**（一年前约 20%），使其昇腾芯片产线**首次实现盈利**。URL: https://www.trendforce.com/news/2025/03/13/news-huaweis-ascend-910c-takes-on-nvidia-as-chinas-ai-race-heats-up-more-alleged-details/ （原 FT：https://www.ft.com/content/f46b7f6d-62ed-4b64-8ad7-2417e5ab34f6 ）
- 【媒体·SemiAnalysis】SemiAnalysis 对良率的预测"低于台积电、Intel、三星、ASE、Amkor 等对前道晶圆与封装所做的估计"，即其模型本就采用保守良率假设。URL: https://newsletter.semianalysis.com/p/huawei-ascend-production-ramp
- **910C 具体良率数值：除上述 FT 约 40% 的说法外，未找到公开数据。**

### 5.4 2026 及以后（官方口径）

- 【官方·高管演讲（2025-09-18）】华为昇腾芯片规划到 2028 年：昇腾 950PR 2026 Q1；昇腾 950DT 2026 Q4；昇腾 960 2027 Q4；昇腾 970 2028 Q4。Atlas 950 SuperPoD 支持 8192 张卡、2026 Q4 上市；Atlas 960 SuperPoD 支持 15488 张卡、2027 Q4 上市；Atlas 950 SuperCluster 算力规模超 50 万卡，Atlas 960 SuperCluster 达百万卡。URL: https://paper.cnstock.com/html/2025-09/19/content_2123758.htm
- 【媒体·界面新闻（经证券时报）】昇腾 950PR 算力 1 PFLOPS(FP8)/2 PFLOPS(FP4)，互联带宽 2TB/s，128GB 内存、1.6TB/s；昇腾 950DT 144GB、4TB/s。URL: https://www.stcn.com/article/detail/3345926.html

---

## 6. 基板 / 中介层供应商

- 【媒体·证券时报】盛合晶微以 **3 倍光罩尺寸 TSV 硅通孔载板技术（亚微米级互联精度）** 支持昇腾芯片高密度集成；公司自述为国内基于 TSV Interposer 的 2.5D 集成量产最早、规模最大者之一。URL: https://www.stcn.com/article/detail/2241117.html
- 【公告/互动易·兴森科技（002436）】公司 ABF 载板是芯片封装原材料，主要应用于 CPU、GPU、FPGA、ASIC 等芯片（2026-01-30 互动易答复）；但**无公开证据指向其为 910C 供货**。URL: https://basic.10jqka.com.cn/002436/
- 【券商研报·中信证券】仅笼统指出"先进制程衍生的封测、载板、电源管理等环节也有望同步受益"，未点名 910C 载板/中介层的具体供应商。URL: https://finance.sina.com.cn/stock/stockzmt/2025-06-25/doc-infcfxpq8236746.shtml
- **910C 具体 ABF 载板 / 有机基板供应商：未找到公开数据 / 未证实。**

---

## 7. 页面不可访问 / 抓取失败清单

- 路透《TSMC could face $1 billion or more fine from US probe》（2025-04-08）：HTTP 401（要求 JS 与关闭广告拦截）。https://www.reuters.com/technology/tsmc-could-face-1-billion-or-more-fine-us-probe-sources-say-2025-04-08/
- 路透《Huawei readies new AI chip for mass shipment》（2025-04-21）：HTTP 401。https://www.reuters.com/world/china/huawei-readies-new-ai-chip-mass-shipment-china-seeks-nvidia-alternatives-sources-2025-04-21/
- 彭博 2025-10-03 与 2025-09-29 两篇原文：付费墙（前者有 Business Times 全文转载可读）。
- TechInsights 官方页面：仅返回 JS 空壳（标题 "TechInsights Platform"，无正文）。https://library.techinsights.com/public/hg-content/3d83fe34-6b77-4f8a-bcb9-d2cf5e4ae0ae
- 盛合晶微招股说明书 PDF（SSE）：直连返回 HTTP 404（内容仅通过搜索引擎快照可见）。https://static.sse.com.cn/stock/disclosure/announcement/c/202601/002104_20260107_5VZI.pdf
- marketscreener（TSMC 罚款）、dailymail（路透 Sophgo 转载）、Yahoo Finance 镜像：均 HTTP 403。
- 联合早报（zaobao）：cross-origin 重定向未跟随；中时新闻网（chinatimes）：HTTP 403。
- unwire.hk 两篇（910C 零件、美智库 75 万枚）：正文抓取被截断，**页面内容不可访问**。
- 新浪财经 CloudMatrix 研报页、证券时报、上海证券报、TrendForce、Notebookcheck-cn、Business Times、SemiAnalysis（两篇）、TechPowerUp、OFweek、中金在线、新快报、锐报：均成功抓取。
