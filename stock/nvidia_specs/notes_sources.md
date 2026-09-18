# NVIDIA 数据中心 AI GPU 规格表 —— 逐型号注释 + 来源清单

## A. 逐型号注释（日期 / 口径 / 来源）

### A100（40GB / 80GB，SXM4 / PCIe）
- **发布**：2020-05-14 GTC 2020 主题演讲发布（Ampere 架构首款数据中心 GPU）。官方新闻稿：<https://nvidianews.nvidia.com/news/nvidia-announces-ampere-architecture-and-a100-gpu>（该 URL 现重定向至新闻归档，需注意）。
- **工艺/架构**：TSMC 7nm N7，Ampere（GA100）。datasheet 未直接写工艺节点，第三方/媒体一致为 7nm。
- **规格来源（NVIDIA官方）**：A100 datasheet（英文版 2021-06，`JUN21`，编号 1758950 R4）
  <https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf>
  中文版（编号 2188504 R5）：<http://images.nvidia.cn/aem-dam/en-zz/Solutions/data-center/a100/nvidia-a100-datasheet-nvidia-a4-2188504-r5-zhCN.pdf>
- **关键口径**：datasheet 表格里 TF32 `156 TFLOPS | 312 TFLOPS*`、BF16 `312 | 624*`、FP16 `312 | 624*`、INT8 `624 TOPS | 1248 TOPS*`；脚注 `* With sparsity`。即 **表格中前一个数是 dense，后一个是 sparse**。
- **FP8/FP4**：A100 **不支持**（Ampere Tensor Core 无 FP8）。datasheet 中不存在 FP8 行。
- **显存带宽**：40GB PCIe 1,555 GB/s；80GB PCIe 1,935 GB/s；40GB SXM 1,555 GB/s；80GB SXM **2,039 GB/s**（注意：中文版 datasheet 把 80GB PCIe 写作 1935、SXM 写作 2039，与英文版一致）。
- **TDP**：40GB PCIe 250W；80GB PCIe 300W；40GB SXM 400W；80GB SXM 400W。
- **互连**：SXM 走 NVLink 3 600 GB/s + NVSwitch（HGX A100 8 卡；datasheet 文字称 "up to 16 A100 GPUs can be interconnected... at up to 600 GB/s"）；PCIe 走 NVLink Bridge，最多 2 卡；PCIe Gen4 64 GB/s。

### H100（SXM5 / PCIe / NVL）
- **发布**：2022-03-22，GTC 2022。官方新闻稿（日期已核）：<https://nvidianews.nvidia.com/news/nvidia-announces-hopper-architecture-the-next-generation-of-accelerated-computing>
- **工艺/架构**：TSMC 4N，Hopper（GH100），800 亿晶体管（datasheet 原文 "built from 80 billion transistors... TSMC 4N process"）。
- **规格来源（NVIDIA官方）**：
  - 英文 datasheet：<https://resources.nvidia.com/en-us-gpu-resources/h100-datasheet-24306>（PDF 实体：`https://dam-cdn.nvd.orangelogic.com/AssetLink/u5hh6fv4r7564i4y673484y3m20083nj.pdf`）
  - 中文 datasheet（编号 2287922 R7）：<http://images.nvidia.cn/aem-dam/en-zz/Solutions/data-center/h100/nvidia-h100-datasheet-nvidia-a4-2287922-r7-zhCN.pdf>
- **⚠️ 冲突 1（SXM vs PCIe vs NVL 命名）**：英文 datasheet 的表头是 `H100 SXM` 与 `H100 NVL`；中文 datasheet 的表头是 `H100 SXM` 与 `H100 PCIe`，但两栏数值完全相同（FP64 24、FP32 48、TF32 800*、BF16/FP16 1,600*、FP8 3,200*、INT8 3,200 TOPS*、显存 80GB、带宽 **2TB/s**、TDP 350W、NVLink 600GB/s）。**H100 PCIe 与 H100 NVL 是不同产品**（NVL 为 94GB HBM3 / 3.9 TB/s / 350–400W），因此中文 datasheet 的 "H100 PCIe" 栏很可能实际指 H100 NVL 或被简化。**引用 350W / 2TB/s / NVLink 600GB/s 这组数时，应注明出自中文 datasheet 的 "H100 PCIe" 栏。**
- **关键口径**：NVIDIA官方 表格值为 sparse，脚注 `*With sparsity`；**dense = 表格值 ÷ 2**：
  - H100 SXM：TF32 989* → dense 494.5；BF16/FP16 1,979* → dense 989.5；FP8 3,958* → dense 1,979；INT8 3,958 TOPS* → dense 1,979。
  - H100 PCIe/NVL 栏：TF32 800* → 400；BF16/FP16 1,600* → 800；FP8 3,200* → 1,600；INT8 3,200* → 1,600。
- **TPP/功耗**：SXM "Up to 700W (configurable)"；PCIe 栏 350W。
- **互连**：SXM NVLink 900 GB/s + NVSwitch，NVLink Switch System 可连 **256 卡**；PCIe/NVL 栏 NVLink 600 GB/s。PCIe Gen5 128 GB/s。
- **MIG**：最多 7 个 MIG（SXM 每个 10GB；NVL 每个 12GB — 见英文 datasheet）。

### H800（中国特供）
- **发布**：2022 年 11 月前后（路透 2022-11-08 报道 NVIDIA 为中国推出符合出口管制的新芯片）。
- **出口管制状态**：为规避 2022-10-07 美国对华出口管制而设计；NVLink 从 H100 的 900 GB/s **削减至 400 GB/s**。2023-10 美国新规后 H800 亦被禁售。
- **NVIDIA 官方规格**：**不存在**。NVIDIA 从未发布 H800 的官方产品页或 datasheet（nvidia.com/en-us/data-center/h800 无 Wayback 快照，nvidia.cn 对应页面 404）。所有数字均为第三方/OEM。
- **第三方规格（已核实）**：强川科技 H800 SXM 产品页 <https://www.qiangchuan.com/item/7228.html>
  - 架构 Hopper、晶体管 800 亿、SXM5、80GB HBM3、5120-bit、**3.35 TB/s**、ECC 支持
  - FP32 67 TFLOPS、FP64 1 TFLOPS（**⚠️ 与 H100 SXM 的 FP64 34 TFLOPS 严重冲突，存疑**）
  - Tensor TF32 989 TFLOPS*、Tensor BF16/FP16 1,979 TFLOPS*、Tensor INT8 3,958 TOPS*，`*采用稀疏技术`
  - 最大功耗 700W；GPU 互连：页面正文文字为「可提供 **400 GB/s** GPU 间互连的第四代 NVLink」
  - FP8 未在参数表列出，但正文提到 Transformer Engine 支持 FP8；按同代推算应为 3,958 TFLOPS sparse（**推算，非官方**）
- **H800 PCIe**：第三方来源为 Lenovo ThinkSystem H800 PCIe 产品指南（`lenovopress.lenovo.com/lp1814`，**现已标注 withdrawn，页面返回 JS 壳，PDF 已不可直下**）。据多方二手引用：FP64 被压到 **0.8 TFLOPS**，NVLink **400 GB/s**，TDP 350W。**建议标注为「未直接核实到原文」。**
- **Datasheet 索引**：llm-infra-atlas 开源文档（第三方）交叉印证 H800 NVLink 400 GB/s、单向约 200 GB/s：<https://github.com/llm-infra-atlas/llm-infra-atlas.github.io>

### H20（中国特供）
- **发布/可用**：2023 年 11 月发布（PS: Nov 2023），2023 年 12 月量产（MP: Dec 2023）。2024-02 开始接受分销商预订（单价约 1.2–1.5 万美元）。
- **NVIDIA 官方规格**：**不存在**。nvidia.com / nvidia.cn 的 h20 页面均无 Wayback 快照（nvidia.cn/data-center/h20/ 返回 404）。
- **第三方规格（已核实）**：百度百科 H20 词条参数表（原始引用媒体，数据截至 2025-04-17）：<https://baike.baidu.com/en/item/H20/1510112>
  - GPU 架构 NVIDIA Hopper；显存 **96GB HBM3**；带宽 **4.0 TB/s**
  - `INT8 | FP8 Tensor Core*` **296 | 296 TFLOPS**（`*`=稀疏）
  - `BF16 | FP16 Tensor Core` **148 | 148 TFLOPS**（按 NVIDIA 通例，此值对应稀疏口径；dense = 74）
  - TF32 Tensor Core **74 TFLOPS**；FP32 **44 TFLOPS**；FP64 **1 TFLOPS**
  - MIG 最多 7 个；L2 60MB；7 NVDEC / 7 NVJPEG；**功耗 400W**；外形 **8 路 HGX**
  - 互连：**PCIe Gen5 x16 128 GB/s；NVLink 900 GB/s**（**未削减**）
- **交叉印证**：TechPowerUp（2025-04-16）同样给出「96 GB HBM3、最高 4.0 TB/s、约 296 TFLOPS 混合精度、FP32 约 74 TFLOPS、FP16 约 148 TFLOPS」，但**其 FP32 74 TFLOPS 与百度百科的 44 TFLOPS 冲突**（百度百科把 74 放在 TF32、44 放在 FP32，更符合 H100 阉割比例）。<https://www.techpowerup.com/335534/us-bans-export-of-nvidia-h20-accelerators-to-china-a-potential-usd-5-5-billion-loss-for-nvidia>
- **出口管制时间线（全部为媒体报道，非 NVIDIA 规格）**：
  - 2025-04-14：美国商务部通知 NVIDIA，H20 对华（含港澳）出口需申请许可证（"indefinite"）。
  - 2025-04-15/16：NVIDIA 8-K 披露约 **55 亿美元**减记；TechPowerUp 报道 → 同上链接。
  - 2025-07-15：美国批准 H20 对华销售（Reuters/百度百科时间线）。
  - 2025-08-08：商务部开始发放 H20 出口许可证。
  - 2025-08-11：路透报道 NVIDIA/AMD 同意将对华芯片销售收入的 **15%** 上缴美国政府以换取许可证。<https://www.reuters.com/world/china/nvidia-amd-pay-15-china-chip-sale-revenues-us-official-says-2025-08-11/>
  - 2025-08-22：The Information（经 TechPowerUp 转述）报道 NVIDIA 已停止 H20 生产，为 B30A 让路。<https://www.techpowerup.com/340198/nvidia-reportedly-ends-h20-gpu-production-makes-room-for-b30a>
  - 2025-09-17：金融时报报道中国网信办（CAC）要求阿里/百度/腾讯等停止采购 NVIDIA 出口改型加速器（含 RTX PRO 6000D）。<https://www.techpowerup.com/341078/china-blocks-tech-firms-from-buying-nvidia-ai-accelerators>
  - 2026-05-17：百度百科时间线称美国再次批准 H20 对华销售（**单一来源，建议存疑**）。

### H200（SXM / NVL）
- **发布**：2023-11-13，SC23。官方新闻稿（日期已核）：<https://nvidianews.nvidia.com/news/nvidia-supercharges-hopper-the-worlds-leading-ai-computing-platform>
  - 原文：`141GB of memory at 4.8 terabytes per second, nearly double the capacity and 2.4x more bandwidth compared with its predecessor, the NVIDIA A100`；系统预计 **2024 Q2** 开始出货。
- **规格来源（NVIDIA官方）**：
  - 产品页：<https://www.nvidia.com/en-us/data-center/h200/>
  - datasheet：<https://dam-cdn.nvd.orangelogic.com/AssetLink/5o2qgy5d2835ve2pm11i62kv8mphqta8.pdf>（© 2024，编号 3512650）
- **⚠️ 冲突 2（H200 带宽）**：
  - 产品页表格：H200 SXM **4.8TB/s**，H200 NVL **4.8TB/s**。
  - datasheet 表格：H200 SXM **4.8TB/s**，H200 NVL **4.2TB/s**；且 datasheet 正文与 Key Features 都写 `4.2TB/s`。
  - 新闻稿只给 SXM 口径的 4.8 TB/s。**建议：SXM 4.8 TB/s 无争议；H200 NVL 采用 4.2 TB/s（datasheet）并注明产品页写 4.8。**
- **⚠️ 冲突 3（H200 NVL 显存/TDP）**：product page 与 datasheet 均为 141GB、600W 可配置、MIG 7×16.5GB，一致。**无冲突**。
- **互连**：SXM NVLink 900 GB/s + NVSwitch，HGX H200 4/8 卡；NVL（PCIe）为 2 或 4 路 NVLink Bridge、**900 GB/s per GPU**（注意：官网产品页原文写作 `2- or 4-way NVIDIA NVLink bridge: 900GB/s per GPU`，与其他 PCIe 产品 600 GB/s 不同，属官方原文）。
- **口径**：产品页与 datasheet 均标注 `With sparsity`，dense = 1/2。

### B200 / GB200（Blackwell）
- **发布**：2024-03-18，GTC 2024。官方新闻稿（日期已核）：<https://nvidianews.nvidia.com/news/nvidia-blackwell-platform-arrives-to-power-a-new-era-of-computing>
- **工艺/架构**：TSMC 4NP，Blackwell（双 die 封装）。
- **规格来源（NVIDIA官方）**：
  - GB200 NVL72 产品页：<https://www.nvidia.com/en-us/data-center/gb200-nvl72/>
  - HGX 产品页（含 HGX B200 8 卡与单卡表）：<https://www.nvidia.com/en-us/data-center/hgx/>
  - Blackwell Tech Brief（datasheet）：<https://resources.nvidia.com/en-us-blackwell-architecture>（PDF：`https://dam-cdn.nvd.orangelogic.com/AssetLink/gl2l4l4812s5fw0p614s6i8bv6mi3vx5.pdf`）
  - DGX B200 产品页：<https://www.nvidia.com/en-us/data-center/dgx-b200/>
- **GB200 NVL72 整柜（产品页原文）**：36 Grace CPU + 72 Blackwell GPU；`NVFP4 Tensor Core² 1,440 | 720 PFLOPS`；`FP8/FP6 Tensor Core² 720 PFLOPS`；`INT8 Tensor Core² 720 POPS`；`FP16/BF16 Tensor Core² 360 PFLOPS`；`TF32 Tensor Core² 180 PFLOPS`；FP32 5,760 TFLOPS；FP64 2,880 TFLOPS；`13.4 TB HBM3E | 576 TB/s`；NVLink 130 TB/s。
  - 脚注：`1. Specification in sparse | dense.`　`2. Specification in sparse. Dense is one-half sparse spec shown.`
- **GB200 Grace Blackwell Superchip（产品页原文）**：`40 | 20 PFLOPS` NVFP4；`20 PFLOPS` FP8/FP6；`20 POPS` INT8；`10 PFLOPS` FP16/BF16；`5 PFLOPS` TF32；160 TFLOPS FP32；80 TFLOPS FP64；`372 GB HBM3E | 16 TB/s`；NVLink 3.6 TB/s。（此处 `40 | 20` 是 sparse | dense，比值 2.0）
- **⚠️ 冲突 4（B200 单卡 FP4 比值）**：HGX 页 8 卡 `144 PFLOPS | 72 PFLOPS`（脚注 `Specification in Sparse | Dense`）→ 单卡 **18 sparse / 9 dense**，比值 **2.0**。而 GB200 Superchip 也是 40/20 = 2.0。**因此 B200 的 FP4 是标准 2×。**
- **⚠️ 冲突 5（B200 显存容量）**：
  - Blackwell Tech Brief：`Up to 192 GB HBM3e | 7.7 TB/s`（HGX B200 单卡）。
  - HGX 页：HGX B200 总显存 `1.4 TB` → 单卡 175GB（若按 1.4TB 精确值）；实际 OEM 产品（Lenovo ThinkSystem HGX B200）为 **180GB HBM3e**。
  - DGX B200 产品页：`1,440 GB total, 64 TB/s HBM3e bandwidth` → 单卡 **180GB / 8 TB/s**（注意此处带宽与 Tech Brief 的 7.7 TB/s 不一致）。
  - **建议：写「180GB（OEM/DGX 配置）；NVIDIA 原文为 up to 192GB；单卡带宽 7.7 TB/s（Tech Brief）或 8 TB/s（DGX B200 页）」并标注冲突。**
- **⚠️ 冲突 6（PCIe 代际）**：Blackwell Tech Brief 的 HGX B200 行写 `Interconnect: NVLink 5 / PCIe Gen5`；Blackwell Ultra datasheet 的 HGX B300 行写 `PCIe Gen6: 256 GB/s`。HGX B200 的 PCIe 代际在官方文档间不一致。
- **FP64**：HGX Tech Brief 写 B200 单卡 FP64 = 37 TFLOPS；HGX 页写 HGX B200 整机 `FP64/FP64 Tensor Core 296 TFLOPS`（296/8 = 37，自洽）。

### B300 / GB300（Blackwell Ultra）
- **发布**：2025-03-18，GTC 2025。官方新闻稿（日期已核）：<https://nvidianews.nvidia.com/news/nvidia-blackwell-ultra-ai-factory-platform-paves-way-for-age-of-ai-reasoning>
  - 原文：`Blackwell Ultra includes the NVIDIA GB300 NVL72 rack-scale solution and the NVIDIA HGX B300 NVL16 system. The GB300 NVL72 delivers 1.5x more AI performance than the NVIDIA GB200 NVL72`。
- **规格来源（NVIDIA官方）**：
  - GB300 NVL72 产品页：<https://www.nvidia.com/en-us/data-center/gb300-nvl72/>
  - HGX 产品页：<https://www.nvidia.com/en-us/data-center/hgx/>
  - Blackwell Ultra datasheet：<https://resources.nvidia.com/en-us-blackwell-architecture/blackwell-ultra-datasheet>（PDF：`https://dam-cdn.nvd.orangelogic.com/AssetLink/1k0p832eq8r5ca0u5383ie5o4tp3bst1.pdf`）
  - DGX B300 产品页：<https://www.nvidia.com/en-us/data-center/dgx-b300/>
- **GB300 NVL72 整柜**：72 Blackwell Ultra GPU + 36 Grace；`FP4 Tensor Core 1440 | 1080² PFLOPS`（脚注 2 = without sparsity）；`FP8/FP6 720 PFLOPS`；`INT8 24 POPS`；`FP16/BF16 360 PFLOPS`；`TF32 180 PFLOPS`；`FP32 6 PFLOPS`；`FP64 100 TFLOPS`；`20 TB | up to 576 TB/s`；NVLink 130 TB/s；Fast Memory 37 TB。
- **HGX B300 8 卡**：`FP4 Tensor Core¹ 144 PFLOPS | 108 PFLOPS`（脚注 1 = Sparse | Dense）；`FP8/FP6 Tensor Core² 72 PFLOPS`；`INT8 3 POPS`（**⚠️ 与其他出处冲突，见 §4.2**）；`FP16/BF16 36 PFLOPS`；`TF32 18 PFLOPS`；FP32 600 TFLOPS；`FP64/FP64 Tensor Core 10 TFLOPS`（**⚠️ 与 Ultra datasheet 的 1.2 TFLOPS 冲突**）；`Total Memory 2.1 TB`；NVLink 5 / 1.8 TB/s 单卡 / 14.4 TB/s 整机；Networking 1.6 TB/s。
- **Blackwell Ultra 单卡（Ultra datasheet「Individual GPU Specifications」）**：GB300 列 `FP4 20 | 15 PFLOPS`、`FP8/FP6 10 PFLOPS`、`INT8 330 TOPS`、`FP16/BF16 5 PFLOPS`、`TF32 2.5 PFLOPS`、`FP32 80 TFLOPS`、`FP64 1.3 TFLOPS`、`279 GB HBM3E | 8 TB/s`、TDP 最高 1,400W、NVLink 5 1.8 TB/s、PCIe Gen6 256 GB/s；HGX 列 `FP4 18 | 14 PFLOPS`、`FP8/FP6 9 PFLOPS`、`INT8 307 TOPS`、`FP16/BF16 4.5 PFLOPS`、`TF32 2.2 PFLOPS`、`FP32 75 TFLOPS`、`FP64 1.2 TFLOPS`、`270 GB HBM3E | 7.7 TB/s`、TDP 最高 1,100W。
- **⚠️ 冲突 7（B300 FP4 的 sparse/dense 比值）**：见 §4.1 —— 这是全报告最重要的口径冲突。
- **DGX B300**：8× Blackwell Ultra SXM、总显存 2.1 TB、`FP4 Tensor Core: 144 PFLOPS | 108 PFLOPS*`（`*Specification shown in sparse | dense`）、`FP8: 72 PFLOPS**`（`**sparse, dense ½`）、NVLink 14.4 TB/s、功耗 ~14 kW、10U。

### B30A（中国特供 Blackwell，已被阻止出口）
- **NVIDIA 官方规格**：**不存在**。截至 2026-09-18，NVIDIA 从未发布 B30A 的产品页或 datasheet。
- **报道时间线**：
  - 2025-08-19：路透（经 TechPowerUp 转述）报道 NVIDIA 设计了中国特供 SKU **B30A**，基于 B300 的**单 die** 版本（双 die B300 取其一），性能减半。TechPowerUp 以此为据**推算**：约 7.5 PFLOPS FP4、3.75 PFLOPS FP6/FP8、1.875 PFLOPS FP16/BF16、0.94 PFLOPS TF32（该推算基于 B300 的 **dense** 15/7.5/3.75/1.88 口径，属**媒体推算**）。<https://www.techpowerup.com/340068/nvidia-prepares-cut-down-b30a-blackwell-ai-accelerator-for-chinese-market>
  - 2025-09-04：路透消息（经 TechPowerUp）称 B30A 定价约为 H20 的 **2 倍**（约 2 万–2.4 万美元 vs H20 约 1 万–1.2 万美元）。<https://www.techpowerup.com/340664/nvidias-export-compliant-b30a-for-china-priced-at-roughly-twice-the-h20>
  - 2025-09-17：金融时报报道中国网信办要求国内厂商停止采购 NVIDIA 出口改型加速器。<https://www.techpowerup.com/341078/china-blocks-tech-firms-from-buying-nvidia-ai-accelerators>
  - **2025-11-07：The Information 报道美国白宫决定阻止 B30A 对华出口许可**，NVIDIA 已向中国客户送样，正重新设计以争取获批；NVIDIA 称在中国竞争性数据中心计算市场份额为「zero」。<https://investinglive.com/stocks/us-blocks-nvidias-scaled-down-ai-chip-sales-to-china-despite-trump-hints-20251107/>
- **⚠️ 存疑点**：社交媒体/港媒（如 AAStocks）2026 年仍在讨论「美国将阻止 B30A」，与 2025-11 的报道为同一事件的不同转述。**未找到任何 NVIDIA 官方 B30A 规格。**
- **互连/显存/NVL8**：有媒体称 B30A 保留 HBM 与 NVLink、采用 NVL8 基板、NVLink 约减半至 900 GB/s，但**本报告未核实到可靠的原始出处**，一律记为「未找到公开数据 / 传闻未证实」。

### RTX 6000D / RTX PRO 6000D（中国特供工作站）
- **NVIDIA 官方规格**：**不存在**面向中国市场的 "RTX 6000D" 官方页面。NVIDIA 官方只有 **RTX PRO 6000 Blackwell 系列**（96GB GDDR7、PCIe Gen5 x16、Server Edition 400–600W / Workstation 600W / Max-Q 300W）：<https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000-family/>
  - **注意**：NVIDIA 官方该页**未列出**任何 Tensor Core TFLOPS 算力数字，也未列显存带宽 → 这两项对全血版亦为「未在官方产品页披露」。
- **中国特供版（媒体/拆解）**：
  - 2025-09-16：路透独家报道 RTX 6000D 在中国大企业遇冷，"little favour with major firms"。<https://www.usnews.com/news/top-news/articles/2025-09-16/exclusive-nvidias-new-rtx-6000d-chip-for-china-finds-little-favour-with-major-firms-sources-say>
  - **拆解实测**：TweakTown 报道拆解显示中国版为 **84GB GDDR7**（全血 RTX PRO 6000 Blackwell 为 96GB）。<https://www.tweaktown.com/news/110135/nvidias-new-rtx-6000d-appears-in-teardown-84gb-gddr7-in-china-compared-to-the-full-96gb/index.html>
  - WindowsReport 同样报道 84GB GDDR7 与削减后的 Blackwell 规格。<https://windowsreport.com/nvidia-rtx-6000d-teardown-reveals-84gb-gddr7-and-cut-down-blackwell-specs/>
  - 2025-09-17：CAC 要求停止采购（同 H20 一条）。
- **算力/带宽/TDP**：媒体常引用「约 1,100 GB/s、约 600W、无 NVLink」，但**未核实到 NVIDIA 原文**；本报告一律记为「未找到公开数据（NVIDIA官方）」。

### A800（中国特供 Ampere）
- **发布**：2022-11-08 路透独家报道 NVIDIA 为中国推出符合出口管制的 A800。<https://www.reuters.com/technology/exclusive-nvidia-offers-new-advanced-chip-china-that-meets-us-export-controls-2022-11-08/>
- **NVIDIA 官方规格（部分）**：NVIDIA 确有 **A800 40GB Active** 工作站产品页（这是唯一被 NVIDIA 官方列出的 A800 SKU）：
  <https://www.nvidia.com/en-in/products/workstations/a800/>
  - 40GB HBM2、显存位宽 5,120-bit、**带宽 1,555.2 GB/s**；CUDA 6,912 / Tensor 432
  - FP64 9.7 TFLOPS；FP32 19.5 TFLOPS；`Peak Tensor Performance 1,247 AI TOPS | 623.8 TFLOPS`（即 623.8 TFLOPS = BF16/FP16 稀疏，1247 AI TOPS = INT8 稀疏）
  - **NVLink 带宽 400 GB/s**（对比 A100 的 600 GB/s，**人为削减**）；PCIe 4.0 x16；**240W**；双插槽主动散热；最多 7 个 MIG @5GB
- **A800 80GB SXM/PCIe**：**NVIDIA 无官方页面**。第三方/OEM 一致为：显存与算力对齐 A100（80GB HBM2e、2039 GB/s、SXM 400W），NVLink 600→400 GB/s。**本报告未直接核实到 OEM 原文。**

### L40S（Ada Lovelace，通用推理/图形）
- **发布**：2023-08-08（NVIDIA 新闻稿 "NVIDIA, Global Data Center System Manufacturers to Supercharge Generative AI and Industrial Digitalization"；该稿的 nvidianews 原始 URL 已归档，中文版见 <https://blogs.nvidia.com.tw/blog/nvidia-global-data-center-system-manufacturers-to-supercharge-generative-ai-and-industrial-digitalization/>）。
- **规格来源（NVIDIA官方）**：
  - 产品页：<https://www.nvidia.com/en-us/data-center/l40s/>
  - **中文 datasheet（编号 2907810，版本日期 2023.9.25）**：<http://images.nvidia.cn/cn/RTX/l40s-datasheet-web-a4-zhCN-2907810-2023.9.25.pdf>
- **⚠️ 冲突 8（L40S FP16）**：
  - 产品页写：`FP32 91.6 teraFLOPS`、`TF32 Tensor Core 366 teraFLOPS*`、`FP16 733 teraFLOPS*`、`FP8 1,466 teraFLOPS*`、`RT Core 212 TFLOPS`、`Max Power 350W`，脚注 `*With Sparsity`。
  - **中文 datasheet 写**：`TF32 183 | 366*`、`BFLOAT16 362.5 | 733*`、`FP16 362.5 | 733*`、`FP8 733 | 1,466*`、`INT8 733 | 1,466*`、`INT4 733 | 1,466*`、`FP32 91.6`。
  - **即产品页的 "FP16 733" 是稀疏值，其 dense 为 362.5**；产品页把 sparse 值当默认展示，未在行内标 dense。**以 datasheet 为准。**
- **显存/互连**：48GB GDDR6 ECC、**864 GB/s**、PCIe 4.0 x16（64 GB/s 双向）、**无 NVLink**、350W、双插槽被动、4×DisplayPort 1.4a。

### L20（中国特供 Ada）
- **NVIDIA 官方规格**：**未找到公开数据**。nvidia.cn/data-center/l20/ 返回 404，无 Wayback 快照。
- 媒体/OEM 常引用「约 48GB GDDR6、约 275W、约 119 TFLOPS FP16 稀疏」，但**本报告未核实到原始出处，一律不采信**。

### MI300X（AMD，仅作对照）
- **发布**：2023-12-06（AMD 新闻稿 "AMD Delivers Leadership Portfolio of Data Center AI Solutions with AMD Instinct MI300 Series"）。<https://www.amd.com/ja/newsroom/press-releases/2023-12-06-amd-ai-amd-instinct-mi300.html>
- **规格来源（AMD官方）**：<https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html>
  - **AMD 明确区分** with / without structured sparsity，是本报告中最清晰的口径范例。
  - FP8 **2.61 PFLOPS**（dense）/ **5.22 PFLOPS**（sparse，E5M2/E4M3）
  - FP16 **1.3 PFLOPS** / **2.61 PFLOPS**（sparse）；BF16 **1.3 PFLOPS** / **2.61 PFLOPS**（sparse）
  - TF32 **653.7 TFLOPS** / **1.3 PFLOPS**（sparse）
  - FP32 (matrix) **163.4 TFLOPS**；FP32 **163.4 TFLOPS**；FP64 matrix **163.4 TFLOPS**；FP64 **81.7 TFLOPS**
  - INT8 **2.6 POPS** / **5.22 POPS**（sparse）
  - **192 GB HBM3**、8,192-bit、**5.3 TB/s**、LLC 256MB、1530 亿晶体管
  - **750W Peak TBP**、**OAM Module**、PCIe 5.0 x16、**Infinity Fabric 128 GB/s**（非 NVLink）、被动 OAM
  - 架构：CDNA 3（AMD 页面注明 `AMD CDNA™ 3 Architecture`；工艺节点不在该页，业内为 5nm/6nm 混合）

---

## B. 完整来源清单（URL + 日期）

### B.1 NVIDIA 官方 —— 产品页
| 产品 | URL | 页面/文档日期 |
|---|---|---|
| A100 | https://www.nvidia.com/en-us/data-center/a100/ | 现行 |
| H100 | https://www.nvidia.com/en-us/data-center/h100/ | 现行 |
| H200 | https://www.nvidia.com/en-us/data-center/h200/ | 现行（©2024 文档线） |
| GB200 NVL72 | https://www.nvidia.com/en-us/data-center/gb200-nvl72/ | 现行 |
| GB300 NVL72 | https://www.nvidia.com/en-us/data-center/gb300-nvl72/ | 现行 |
| HGX（B300/B200/H200/H100） | https://www.nvidia.com/en-us/data-center/hgx/ | 现行 |
| DGX B200 | https://www.nvidia.com/en-us/data-center/dgx-b200/ | 现行 |
| DGX B300 | https://www.nvidia.com/en-us/data-center/dgx-b300/ | 现行 |
| L40S | https://www.nvidia.com/en-us/data-center/l40s/ | 现行 |
| A800 40GB Active | https://www.nvidia.com/en-in/products/workstations/a800/ | 现行 |
| RTX PRO 6000 Blackwell 系列 | https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000-family/ | 现行 |
| Data Center GPU Line Card | https://docs.nvidia.com/data-center-gpu/line-card.pdf | 2025（含 GB300/B300 列） |

### B.2 NVIDIA 官方 —— Datasheet / Tech Brief（PDF 直链）
| 文档 | URL | 日期/编号 |
|---|---|---|
| A100 datasheet（EN） | https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf | JUN21, 1758950 R4 |
| A100 datasheet（CN） | http://images.nvidia.cn/aem-dam/en-zz/Solutions/data-center/a100/nvidia-a100-datasheet-nvidia-a4-2188504-r5-zhCN.pdf | 2188504 R5 |
| H100 datasheet（EN） | https://dam-cdn.nvd.orangelogic.com/AssetLink/u5hh6fv4r7564i4y673484y3m20083nj.pdf | 现行（资源页 https://resources.nvidia.com/en-us-gpu-resources/h100-datasheet-24306 ） |
| H100 datasheet（CN） | http://images.nvidia.cn/aem-dam/en-zz/Solutions/data-center/h100/nvidia-h100-datasheet-nvidia-a4-2287922-r7-zhCN.pdf | 2287922 R7 |
| H200 datasheet | https://dam-cdn.nvd.orangelogic.com/AssetLink/5o2qgy5d2835ve2pm11i62kv8mphqta8.pdf | ©2024, 3512650 |
| Blackwell Tech Brief（B200/GB200） | https://dam-cdn.nvd.orangelogic.com/AssetLink/gl2l4l4812s5fw0p614s6i8bv6mi3vx5.pdf | 资源页 https://resources.nvidia.com/en-us-blackwell-architecture |
| Blackwell Ultra datasheet（B300/GB300） | https://dam-cdn.nvd.orangelogic.com/AssetLink/1k0p832eq8r5ca0u5383ie5o4tp3bst1.pdf | 资源页 https://resources.nvidia.com/en-us-blackwell-architecture/blackwell-ultra-datasheet |
| L40S datasheet（CN） | http://images.nvidia.cn/cn/RTX/l40s-datasheet-web-a4-zhCN-2907810-2023.9.25.pdf | 2023.9.25, 2907810 |

### B.3 NVIDIA 官方 —— 新闻稿（日期已逐条核对）
| 事件 | URL | 日期 |
|---|---|---|
| Ampere / A100 发布 | https://nvidianews.nvidia.com/news/nvidia-announces-ampere-architecture-and-a100-gpu （现重定向至归档） | 2020-05-14 |
| Hopper / H100 发布 | https://nvidianews.nvidia.com/news/nvidia-announces-hopper-architecture-the-next-generation-of-accelerated-computing | **2022-03-22** |
| H200 发布（SC23） | https://nvidianews.nvidia.com/news/nvidia-supercharges-hopper-the-worlds-leading-ai-computing-platform | **2023-11-13** |
| Blackwell / B200 发布（GTC24） | https://nvidianews.nvidia.com/news/nvidia-blackwell-platform-arrives-to-power-a-new-era-of-computing | **2024-03-18** |
| Blackwell Ultra / GB300 发布（GTC25） | https://nvidianews.nvidia.com/news/nvidia-blackwell-ultra-ai-factory-platform-paves-way-for-age-of-ai-reasoning | **2025-03-18** |
| L40S / 全球数据中心系统商 | https://blogs.nvidia.com.tw/blog/nvidia-global-data-center-system-manufacturers-to-supercharge-generative-ai-and-industrial-digitalization/ | 2023-08-08 |

### B.4 第三方 / 媒体（含传闻与拆解）
| 主题 | 来源与 URL | 日期 |
|---|---|---|
| A800 对华推出 | Reuters: https://www.reuters.com/technology/exclusive-nvidia-offers-new-advanced-chip-china-that-meets-us-export-controls-2022-11-08/ | 2022-11-08 |
| H800 SXM 规格 | 强川科技产品页: https://www.qiangchuan.com/item/7228.html | 未标注（现行） |
| H800 PCIe 规格 | Lenovo ThinkSystem 产品指南（已 withdrawn）: https://lenovopress.lenovo.com/lp1814 | 已下架 |
| H20 规格 | 百度百科 H20 词条（数据截至 2025-04-17）: https://baike.baidu.com/en/item/H20/1510112 | 2025-04-17 |
| H20 禁售 + 55 亿美元减记 | TechPowerUp: https://www.techpowerup.com/335534/us-bans-export-of-nvidia-h20-accelerators-to-china-a-potential-usd-5-5-billion-loss-for-nvidia | 2025-04-16 |
| 15% 收入分成 | Reuters: https://www.reuters.com/world/china/nvidia-amd-pay-15-china-chip-sale-revenues-us-official-says-2025-08-11/ | 2025-08-11 |
| H20 停产 | The Information 经 TechPowerUp: https://www.techpowerup.com/340198/nvidia-reportedly-ends-h20-gpu-production-makes-room-for-b30a | 2025-08-22 |
| B30A 存在（路透） | Reuters 经 TechPowerUp: https://www.techpowerup.com/340068/nvidia-prepares-cut-down-b30a-blackwell-ai-accelerator-for-chinese-market | 2025-08-19 |
| B30A 定价（路透） | Reuters 经 TechPowerUp: https://www.techpowerup.com/340664/nvidias-export-compliant-b30a-for-china-priced-at-roughly-twice-the-h20 | 2025-09-04 |
| RTX 6000D 遇冷（路透独家） | Reuters 经 US News: https://www.usnews.com/news/top-news/articles/2025-09-16/exclusive-nvidias-new-rtx-6000d-chip-for-china-finds-little-favour-with-major-firms-sources-say | 2025-09-16 |
| CAC 叫停采购 | FT 经 TechPowerUp: https://www.techpowerup.com/341078/china-blocks-tech-firms-from-buying-nvidia-ai-accelerators | 2025-09-17 |
| RTX 6000D 84GB 拆解 | TweakTown: https://www.tweaktown.com/news/110135/nvidias-new-rtx-6000d-appears-in-teardown-84gb-gddr7-in-china-compared-to-the-full-96gb/index.html | 2025-09 |
| RTX 6000D 拆解 | WindowsReport: https://windowsreport.com/nvidia-rtx-6000d-teardown-reveals-84gb-gddr7-and-cut-down-blackwell-specs/ | 2025-09 |
| B30A 被美方阻止 | The Information 经 investingLive: https://investinglive.com/stocks/us-blocks-nvidias-scaled-down-ai-chip-sales-to-china-despite-trump-hints-20251107/ | 2025-11-07 |
| H200 对华许可证（10 家企业） | CNBC: https://www.cnbc.com/2026/05/14/us-clears-h200-chip-sales-to-10-china-firms-as-nvidia-ceo-looks-for-breakthrough.html | 2026-05-14 |
| 中国批准 H200 进口但限量 | The Information / Reuters 经 TrendForce: https://www.trendforce.com/news/2026/07/09/news-china-reportedly-to-allow-nvidia-h200-imports-but-approvals-may-be-capped-below-200k-chips-less-than-half-requested/ | 2026-07-09 |
| B200/H800 交叉印证（第三方技术文档） | llm-infra-atlas: https://github.com/llm-infra-atlas/llm-infra-atlas.github.io | 现行 |
| MI300X 官方规格 | AMD: https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html | 现行 |
| MI300 系列发布 | AMD: https://www.amd.com/ja/newsroom/press-releases/2023-12-06-amd-ai-amd-instinct-mi300.html | 2023-12-06 |

---

## C. 不确定 / 存疑 汇总（按严重程度排序）

### C.1 【严重】B300 / Blackwell Ultra 的 FP4 sparse:dense 比值在 NVIDIA 官方文档内部不自洽

**问题**：Blackwell 一代的 FP4 是本代核心卖点，但 NVIDIA 不同官方页面给出不同比值。

| 出处 | B300 表述 | 隐含比值 |
|---|---|---|
| HGX 产品页 | `FP4 Tensor Core¹ 144 PFLOPS \| 108 PFLOPS`（8卡），脚注1 = `Specification in Sparse \| Dense` | **1.33×**（单卡 18 sparse / 13.5 dense） |
| Blackwell Ultra datasheet 单卡表（GB300 列） | `FP4 Tensor Core¹ 20 PFLOPS \| 15 PFLOPS` | **1.33×** |
| Blackwell Ultra datasheet 单卡表（HGX 列） | `FP4 Tensor Core¹ 18 PFLOPS \| 14 PFLOPS` | **1.29×** |
| Blackwell Ultra datasheet 整柜总表 | `1,440 PFLOPS \| 1,080 PFLOPS` = 72 × (20 \| 15) | **1.33×** |
| DGX B300 产品页 | `FP4 Tensor Core: 144 PFLOPS \| 108 PFLOPS*`，`*Specification shown in sparse \| dense` | **1.33×** |
| 同文档其他精度 | `FP8/FP6 10 PFLOPS`（脚注2 = sparse，dense = ½）、`INT8 330 TOPS`（sparse，dense = ½） | **2.0×** |

**判断**：FP4 的 **1.33×** 在两个独立官方源（HGX 页、DGX B300 页）以及 Ultra datasheet 的整柜汇总中一致出现，因此 **B300 的 dense FP4 = 13.5 PFLOPS/卡（HGX）/ 15 PFLOPS/卡（GB300）应视为 NVIDIA 官方口径**；但「sparse 只比 dense 高 33%」与同表其他精度的 2× 规则不一致，**NVIDIA 未解释原因**（可能 FP4 的 sparse 加速比受限于非 Tensor Core 路径）。

**给寒武纪对比的建议**：若对比 MLU 的 INT8/INT16 dense 算力，**必须用 B300 的 dense FP4（13.5 或 15 PFLOPS）**，不能使用 18/20 PFLOPS，也不能简单除以 2。

### C.2 【严重】B200 单卡 FP4 在官方源之间也是 1.33× vs 2.0× 并存

- HGX 页 HGX B200 8 卡：`144 PFLOPS | 72 PFLOPS`（Sparse | Dense）→ 单卡 **18 / 9**，比值 **2.0×**。
- Blackwell Tech Brief：HGX B200 单卡 `9/18 petaFLOPS`（`FP4 Tensor Core Dense/Sparse`）→ 一致，**2.0×**。
- GB200 产品页 Superchip：`40 | 20 PFLOPS`（NVFP4）→ **2.0×**。
- **但 GB200 NVL72 整柜产品页只写 `1,440 | 720 PFLOPS`，其中 720 是 dense**；而 K 提到的 GB300 NVL72 在产品页写 `1440 | 1080`。
- **结论**：B200 的 FP4 是 **2.0×**；B300 的 FP4 是 **1.33×**。这一代际内的不一致可能是真实的架构差异（Blackwell Ultra 增加了 attention 加速但 FP4 sparse 增益下降），而非笔误，但因 NVIDIA 未说明，仍列为存疑。

### C.3【中】H200 的显存带宽：4.2 还是 4.8 TB/s

- **H200 SXM = 4.8 TB/s**：产品页、datasheet、官方新闻稿（2023-11-13）三者一致。**无争议。**
- **H200 NVL**：
  - 产品页表：`4.8TB/s`
  - datasheet 表 + datasheet 正文 + Key Features：`4.2TB/s`
  - **判断**：datasheet（编号 3512650，©2024）更具体且三处自洽，建议采信 **4.2 TB/s**，并注明产品页写 4.8。

### C.4【中】H100 PCIe 与 H100 NVL 的混淆

- 英文 datasheet 表头为 `H100 SXM` / `H100 NVL`（NVL：94GB、3.9 TB/s、FP64 30、FP32 60、BF16 sparse 1,671）。
- 中文 datasheet 表头为 `H100 SXM` / `H100 PCIe`，但第二栏数值与英文版的 NVL 栏**完全相同**（FP64 24、FP32 48、BF16 sparse 1,600、80GB、2TB/s、350W、NVLink 600GB/s）。
- 数值上：NVL 栏是 94GB/3.9TB/s，中文 "PCIe" 栏是 80GB/2TB/s → **两者不是同一产品**。
- **判断**：中文 datasheet 的「H100 PCIe」栏实际是**标准 PCIe 80GB SKU**（2TB/s、350W、NVLink 600GB/s），英文 datasheet 把同一栏改标为 NVL 并替换为 94GB/3.9TB/s 的数值。**引用 350W / 2.0 TB/s 时请注明「H100 PCIe（中文 datasheet）」；引用 94GB / 3.9 TB/s 时注明「H100 NVL（英文 datasheet）」。**
- **给对比的建议**：若寒武纪对比对象是「H100 PCIe」，用 80GB / 2.0 TB/s / 350W / NVLink 600GB/s / BF16 dense 800；若对象是「H100 NVL」，用 94GB / 3.9 TB/s / 350–400W / NVLink 600GB/s / BF16 dense 835.5。

### C.5【中】B200 单卡显存容量：180GB vs 186GB vs 192GB

| 出处 | 数值 |
|---|---|
| Blackwell Tech Brief（HGX B200 单卡） | `Up to 192 GB HBM3e` |
| DGX B200 产品页 | `1,440 GB total` → 单卡 **180GB** |
| HGX 产品页 | `Total Memory 1.4 TB` → 单卡 **175GB**（或 180GB 取整） |
| GB200 产品页 Superchip | `372 GB HBM3E` → 双 GPU → 单卡 **186GB** |
| Lenovo ThinkSystem HGX B200 产品指南 | **180GB** |

**判断**：B200 die 的 HBM3e 容量上限 NVIDIA 官方表述为 192GB，但实际出货的 HGX B200 为 180GB、GB200 内为 186GB。**建议写「180GB（HGX B200 实际配置）/ 186GB（GB200 内）/ 官方表述 up to 192GB」。**

### C.6【中】HGX B300 的 INT8 与 FP64 在官方页面之间冲突

- **INT8**：HGX 页写 HGX B300 8 卡 `INT8 Tensor Core² 3 POPS`（即 3,000 POPS 整机 → 单卡 375 TOPS，与 Ultra datasheet 的 307 TOPS dense 不符）；Ultra datasheet 单卡写 `330 TOPS`（GB300 列，sparse → dense 165）/ `307 TOPS`（HGX 列，sparse → dense 153.5）。**三处互不吻合。**建议以 Ultra datasheet 单卡值为准并标注冲突。
- **FP64**：HGX 页写 HGX B300 整机 `FP64/FP64 Tensor Core 10 TFLOPS`（比 HGX B200 的 296 TFLOPS 低 30 倍，**明显异常**）；Ultra datasheet 单卡写 `1.3 TFLOPS`（GB300 列）/ `1.2 TFLOPS`（HGX 列）→ 8 卡应为 9.6–10.4 TFLOPS，**与 HGX 页的 10 TFLOPS 吻合**。故 HGX 页的 `10 TFLOPS` 是**正确的**，而「B200 的 296 TFLOPS」才是被高估的一格（Ultra datasheet 的 B200 单卡为 37 TFLOPS × 8 = 296，自洽）。**即：B300 的 FP64 大幅低于 B200 是真实的（Blackwell Ultra 削减了 FP64 单元），不是笔误。**
- **B300 单卡 FP64 在表格中应写 1.2–1.3 TFLOPS**，并注明「相比 B200 的 37 TFLOPS 大幅削减」。

### C.7【中】GB200 NVL72 整柜 HBM 容量 13.4 / 13.5 TB

- 产品页：`13.4 TB HBM3E`
- Blackwell Tech Brief 系统表：`13.5 TB`
- 按 72 × 186GB = 13.392 TB ≈ **13.4 TB**（产品页正确）；13.5 TB 可能按 72 × 187.5 或四舍五入。**建议写 13.4 TB（产品页），并注明 Tech Brief 写 13.5 TB。**

### C.8【中】H20 的 dense/sparse 口径与 FP32 / TF32 数值

- 百度百科参数表写 `INT8 | FP8 Tensor Core* 296 | 296 TFLOPS`（`*`=稀疏）、`BF16 | FP16 148 | 148 TFLOPS`。
- **若 296 是稀疏值，dense 应为 148**；这与「H20 算力约为 H100 的 1/7」的定性描述一致（H100 SXM FP8 sparse 3,958 → H20 296 约为 7.5%）。
- **但百度百科未在 BF16/FP16 行标 `*`**，存在把稀疏值当 dense 展示的可能。
- **FP32/TF32 冲突**：TechPowerUp（2025-04-16）称 FP32 ≈ 74 TFLOPS；百度百科写 TF32 74 / FP32 44。按 H100 的 TF32:FP32 = 989:67 ≈ 14.8:1 的比例，H20 若 FP32 = 44 则 TF32 稀疏 ≈ 652、dense ≈ 326，与 74 不符；反之若 FP32 = 74，则 TF32 dense ≈ 1,096，更不符。**两组数字都无法与 H100 的比例关系自洽，H20 的 FP32/TF32 数值建议标注为「第三方数据，存疑」。**
- **H20 官方规格不存在**：NVIDIA 从未发布 H20 datasheet / 产品页（中英文站点均 404，无 Wayback 快照）。

### C.9【中】H800 的 FP64 数值存在重大冲突

- 强川科技产品页写 **FP64 = 1 TFLOPS**（暗示 FP64 被大幅削减）。
- 但 Lenovo 的 H800 PCIe 产品指南（二手引用）写 **0.8 TFLOPS**。
- 而 H100 SXM 官方为 **FP64 = 34 TFLOPS**、H100 PCIe 官方为 **24 TFLOPS**。
- **判断**：若 H800 SXM 的 FP64 真为 1 TFLOPS，则 H800 不仅砍了 NVLink，还砍了 FP64 —— 这会显著影响 HPC 场景适用性。**目前仅有第三方单一来源支撑，建议标注为「第三方数据，未获 NVIDIA 官方确认，存疑」**；在依赖 FP64 的对比场景中不要采用 H800。

### C.10【中】B30A 的全部规格均无官方来源

- NVIDIA 从未发布 B30A 产品页/datasheet（已核实 nvidia.com 与 nvidia.cn 均无）。
- 唯一有原始出处的是 **路透 2025-08-19**（存在该 SKU、单 die、算力减半）与 **路透 2025-09-04**（定价约 H20 两倍）。
- TechPowerUp 的 7.5 / 3.75 / 1.875 / 0.94 PFLOPS 是**记者按 B300 减半自行推算**，非任何一手来源。**必须标注为「第三方推算」。**
- **2025-11-07 The Information 报道美方已阻止 B30A 出口许可**，NVIDIA 正重新设计。截至 2026-09-18 **未找到 B30A 获批或量产的确切报道**。
- **显存容量、NVLink 带宽、TDP、NVL8 等细节均未找到可靠出处** → 一律写「未找到公开数据」。

### C.11【中】RTX 6000D 的规格仅来自拆解/媒体

- NVIDIA 官方只有 RTX PRO 6000 Blackwell（96GB GDDR7、PCIe Gen5、400–600W/600W/300W），**没有 "RTX 6000D" 页面**。
- **84GB GDDR7** 来自 TweakTown / WindowsReport 的**拆解实测**，可信度较高但非 NVIDIA 官方。
- **算力（TFLOPS）、显存带宽、TDP** 均**未找到 NVIDIA 原文** → 写「未找到公开数据」。
- **NVLink**：RTX PRO 6000 Blackwell 全系**官方页面未列 NVLink**，故中国版亦应为无 NVLink，但严格说属「未披露」而非「官方确认无」。

### C.12【低】L20 几乎无可靠公开数据

- nvidia.cn/data-center/l20/ 返回 404，无 Wayback 快照。
- 媒体常引用的「48GB GDDR6 / 275W / 119 TFLOPS FP16 稀疏」**未核实到原始出处**，本报告不予采用。

### C.13【低】H20 141GB / H20 NVL 版本

- 有传闻称存在 H20 的 141GB HBM3e 版本。**本报告在 NVIDIA 官方与百度百科 H20 词条中均未找到该版本**，搜索亦未找到一手出处。**视为未证实。**

### C.14【低】L40S 产品页的 FP16 数字与 datasheet 不一致

- 产品页：`FP16 733 teraFLOPS*`（`*With Sparsity`）—— 只给稀疏值，未给 dense。
- 中文 datasheet：`FP16 362.5 | 733*`。
- **对比时务必用 362.5（dense）**，不要用 733。这是「产品页只展示 sparse」导致误用的典型例子。
