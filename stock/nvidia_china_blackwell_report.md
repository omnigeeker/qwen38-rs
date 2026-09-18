# NVIDIA China-Specific Blackwell Parts — B30A and RTX 6000D / RTX PRO 6000D

**Research cut-off: 2026-09-18. Analyst note: hardware semiconductor research.**

## 0. Headline reliability statement (read first)

* **Neither part has an NVIDIA-published datasheet.** Zero `NVIDIA官方` performance/memory/power numbers exist for **B30A** (never launched) or **RTX 6000D** (shipped, but NVIDIA never published a spec sheet). Everything below for those two parts is `第三方/媒体推测` unless explicitly marked `NVIDIA官方`.
* The only `NVIDIA官方` numbers in this report are for the **baseline** parts (B300 / Blackwell Ultra and RTX PRO 6000 Blackwell) and are used to judge the leaks.
* **Naming trap — two different parts share the "B30" family name in 2025 reporting:**
  1. a **GDDR7**-based workstation/AI card first reported as **"B40" / "RTX PRO 6000D"** (this is what actually shipped, as **RTX 6000D**); and
  2. an **HBM3E**-based datacenter accelerator **"B30A"** (single-die B300), which was blocked.
  Reuters' 2025-05-24 report (relayed by heise, 2025-05-26) describes part (1) as *"B40 or RTX Pro 6000D"*; ETnews/芯智讯 (2025-09-09) calls the GDDR7 part *"B40"/"B30"*; the WSJ calls the 80 %-of-Blackwell part *"B30"*. Outlets repeatedly conflated the two. ([heise, 2025-05-26](https://www.heise.de/en/news/Cat-and-mouse-Nvidia-s-B40-to-circumvent-China-ban-10397137.html?view=print); [芯智讯 via Tencent Cloud, 2025-09-09/2026-03-20](https://cloud.tencent.cn/developer/article/2643068); [EEPW, 2025-08-26](http://m.eepw.com.cn/article/202508/473337.html); [flopper.io B30A entry](https://flopper.io/gpu/nvidia-b30a))

---

## 1. B30A (datacenter, China-only, never launched)

### 1.1 Announcement / launch date and export-control status

| Date | Event | Outlet / label |
|---|---|---|
| 2025-05-24 | Reuters first reports a cheaper Blackwell part for China (GDDR7, no NVLink) — this lineage became RTX 6000D, **not** B30A | Reuters via [heise, 2025-05-26](https://www.heise.de/en/news/Cat-and-mouse-Nvidia-s-B40-to-circumvent-China-ban-10397137.html?view=print) `第三方/媒体推测` |
| **2025-08-19** | **Reuters names "B30A" (explicitly tentative): single-die design using one of the two dies of the dual-die B300, ~half the raw compute of B300, retains HBM and NVLink; samples to Chinese customers possibly as early as Sept 2025** | [Reuters, 2025-08-19](https://www.reuters.com/world/china/nvidia-working-new-ai-chip-china-that-outperforms-h20-sources-say-2025-08-19/); relayed by [TechPowerUp, 2025-08-19](https://www.techpowerup.com/340068/nvidia-prepares-cut-down-b30a-blackwell-ai-accelerator-for-chinese-market); [Notebookcheck, 2025-08-21](https://www.notebookcheck.net/Nvidia-readies-new-Blackwell-based-accelerators-tailored-for-China-under-US-export-limits.1091993.0.html) `第三方/媒体推测` |
| 2025-08-22 | The Information: NVIDIA halts **H20** (Hopper) production and tells Samsung/Amkor to stop H20 components, "making room for B30A" | [TechPowerUp, 2025-08-22](https://www.techpowerup.com/340198/nvidia-reportedly-ends-h20-gpu-production-makes-room-for-b30a) `第三方/媒体推测` |
| 2025-08-26 | WSJ: China-bound "B30" pitched at **80 % of a baseline Blackwell GPU's performance**; export-licence application submitted to the US government | [EEPW, 2025-08-26](http://m.eepw.com.cn/article/202508/473337.html) `第三方/媒体推测` |
| 2025-09-04 | Reuters: B30A priced at roughly **2× H20** ⇒ **US$20,000–24,000** (~CNY 171,000), H20 at $10,000–12,000 | [TechPowerUp, 2025-09-04](https://www.techpowerup.com/340664/nvidias-export-compliant-b30a-for-china-priced-at-roughly-twice-the-h20) `第三方/媒体推测` |
| 2025-09-17 | FT: China's CAC tells Alibaba/Baidu/Tencent to stop evaluating and procuring NVIDIA China parts — names **RTX PRO 6000D**; **B30A not named** | [TechPowerUp, 2025-09-17](https://www.techpowerup.com/341078/china-blocks-tech-firms-from-buying-nvidia-ai-accelerators) (source: FT) `第三方/媒体推测` |
| **2025-11-07** | **The Information: the White House told other federal agencies it would NOT allow the B30A to be sold to China. NVIDIA had already given samples to several Chinese customers. NVIDIA was re-working the B30A design hoping for reconsideration. NVIDIA's own statement to Reuters: its share of China's competitive datacenter-compute market is zero, so it is not in guidance.** Chinese media framed it as 夭折 ("stillborn") | [Reuters, 2025-11-07](https://www.reuters.com/world/china/us-block-nvidias-sale-scaled-back-ai-chips-china-information-says-2025-11-07/); [HKEJ, 2025-11-07](https://www.hkej.com/landing/articleprint/id/4243408); [PConline/快科技, 2025-11-07](https://news.pconline.com.cn/2013/20133432.html) `第三方/媒体推测` |
| 2025-12 | Trump says H200 sales to China will be allowed with a **25 % US revenue take** | [Seoul Economic Daily, 2026-07-15](https://en.sedaily.com/international/2026/07/15/nvidias-h200-chip-exports-to-china-have-begun) `第三方/媒体推测` |
| 2026-01 (reported) | US Commerce easing moved **only H200 and AMD MI325X** to case-by-case review — **Blackwell-class parts were not included** | [flopper.io B30A entry](https://flopper.io/gpu/nvidia-b30a) `第三方/媒体推测` |
| 2026-02/03 | Commerce licenses H200 to China; NVIDIA restarts production Mar 2026 | [AInvest, 2026-08-20](https://www.ainvest.com/news/4-6-billion-nvidia-door-china-squeeze-means-ai-trade-2608/) (AI-generated aggregation — treat as pointer only) `第三方/媒体推测` |
| 2026-05-14/15 | Reuters: US clears H200 sales to **10 Chinese firms** (~750,000 units). 快科技: **H200 export cleared, B30A "still stuck"**; approval judged unlikely until Rubin ships in volume; both US and Chinese approval plus domestic-chip politics involved | [Reuters, 2026-05-14](https://www.reuters.com/business/retail-consumer/us-clears-h200-chip-sales-10-china-firms-nvidia-ceo-looks-breakthrough-2026-05-14/); [快科技/MyDrivers, 2026-05-15](https://m.mydrivers.com/newsview/1122494.html) `第三方/媒体推测` |
| 2026-05 | Jensen Huang says NVIDIA has **"largely conceded"** China's advanced AI-chip market to Huawei | [AInvest, 2026-08-20](https://www.ainvest.com/news/4-6-billion-nvidia-door-china-squeeze-means-ai-trade-2608/) citing CNBC; [Times of India](https://timesofindia.indiatimes.com/technology/tech-news/after-admitting-that-nvidia-share-in-china-has-fallen-to-zero-ceo-jensen-huang-tells-chinese-company-huawei-we-have-/affcmtoi_articleshow/131240420.cms) `第三方/媒体推测` |
| 2026-07-14 | US Commerce Under Secretary Jeffrey Kessler to Congress: H200 exports to China **have begun but are "very small"/"trivial"**; case-by-case licensing with on-site inspections; 3 more Chinese firms (incl. a ZTE affiliate) approved, on top of a dozen in May | [Seoul Economic Daily, 2026-07-15](https://en.sedaily.com/international/2026/07/15/nvidias-h200-chip-exports-to-china-have-begun); [The Star/Reuters, 2026-07-18](https://www.thestar.com.my/aseanplus/aseanplus-news/2026/07/18/us-says-nvidias-h200-exports-to-china-remain-trivial-despite-approvals) `第三方/媒体推测` |
| 2026-08-20 | B30A reported to **exceed the then-current US export-control performance threshold by >18×**, i.e. no engineering fix was available; the November 2025 White House veto overrode the compliance framework | [AInvest, 2026-08-20](https://www.ainvest.com/news/4-6-billion-nvidia-door-china-squeeze-means-ai-trade-2608/) citing [IFP, "The B30A Decision"](https://ifp.org/the-b30a-decision/) `第三方/媒体推测` |
| **As of 2026-09-18** | **No public evidence that B30A has been approved, launched, sampled at scale, or shipped. No "B30A successor" has been reported by any outlet.** | `未找到公开数据` |

**Correction to a common claim:** the widely repeated "NVIDIA paused China Blackwell production" story is actually about **H20 (Hopper)** production being halted in Aug 2025, not a Blackwell line ([TechPowerUp, 2025-08-22](https://www.techpowerup.com/340198/nvidia-reportedly-ends-h20-gpu-production-makes-room-for-b30a)). H200 production was later restarted around Mar 2026 ([AInvest, 2026-08-20](https://www.ainvest.com/news/4-6-billion-nvidia-door-china-squeeze-means-ai-trade-2608/)).

**NVIDIA's only on-record statements** (never a confirmation of the SKU): *"We evaluate various products to plan our roadmap so we can be prepared to compete to the extent governments allow"* and *"All our products are fully approved by the relevant authorities and designed for beneficial commercial use"* ([EEPW quoting Tom's Hardware, 2025-08-20](https://www.eepw.com.cn/article/202508/473207.htm)); plus the zero-China-share statement to Reuters on 2025-11-07. → **NVIDIA官方 = no B30A confirmation.**

### 1.2 Process node / architecture

* **Baseline (NVIDIA官方, 2025-08-22):** Blackwell Ultra = **TSMC 4NP**, **208 B transistors**, **two reticle-sized dies** joined by **NV-HBI at 10 TB/s**, **160 SMs**, **640 5th-gen Tensor Cores**, 288 GB HBM3E, 8 TB/s, NVLink 5. ([NVIDIA Developer Blog, 2025-08-22](https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/))
* **B30A:** described by Reuters/Tom's Hardware/TechPowerUp as a **single-die** part using one of B300's two dies (≈half compute) — `第三方/媒体推测`. ([Reuters, 2025-08-19](https://www.reuters.com/world/china/nvidia-working-new-ai-chip-china-that-outperforms-h20-sources-say-2025-08-19/); [TechPowerUp, 2025-08-19](https://www.techpowerup.com/340068/nvidia-prepares-cut-down-b30a-blackwell-ai-accelerator-for-chinese-market))
* **Direct conflict:** WCCFTech's August 2025 rumour claimed the China Blackwell part would have a **dual-die design, 8-Hi HBM3E and TSMC N4P**, expected Q4 2025 ([WCCFTech](https://wccftech.com/nvidia-b30a-blackwell-chip-for-china-rumored-to-feature-dual-die-design/)) — `第三方/媒体推测`, contradicted by the Reuters single-die account.
* **Packaging conflict:** Tom's Hardware's comparison table lists the B30A package as **CoWoS-S**, while B200/B300 are CoWoS-L ([EEPW reproduction of Tom's Hardware, 2025-08-20](https://www.eepw.com.cn/article/202508/473207.htm)). A separate Chinese piece says the related **B300A** is TSMC 4 nm + **CoWoS-L**, 144 GB HBM3E, 600 W ([EEPW, 2025-08-26](http://m.eepw.com.cn/article/202508/473337.html); [TechWeb/快科技, 2025-09-05](https://m.techweb.com.cn/marticle/2025-09-05/2965455.shtml)). Unresolved.
* **"GB110" die codename: `未找到公开数据`.** No source found names a GB110 die for B30A. NVIDIA's official Blackwell Ultra materials do not use a GB1xx die name for B300; rumoured Blackwell IDs in circulation are GB112/GB120 ([WCCFTech](https://wccftech.com/five-brand-new-nvidia-blackwell-gpu-pci-ids-spotted-gb112-gb120/)).

### 1.3 Dense FP16/BF16 TFLOPS vs with-sparsity — **sources do NOT distinguish, and they contradict each other**

No outlet publishes a B30A FP16/BF16 figure with an explicit dense/sparse label. All published numbers are arithmetic performed on B300 figures, and several outlets used B300 figures that appear to be **sparse or marketing** numbers rather than dense.

Published B30A estimates:

| Source | B30A BF16/FP16 per package | Stated B300 baseline used | Label |
|---|---|---|---|
| Tom's Hardware (via EEPW, 2025-08-20) | **2.5 PFLOPS BF16** | BF16 5 PFLOPS (i.e. 2× SemiAnalysis' B300 dense 2.25) | `第三方/媒体推测` — no dense/sparse label |
| TechPowerUp (2025-08-19 / 2025-09-04) | **1.875 PFLOPS FP16/BF16** | BF16 3.75 PFLOPS | `第三方/媒体推测` — no dense/sparse label |
| Arithmetic from B300's **dense** BF16 | ≈**1.125 PFLOPS dense** (half of 2,250 TFLOPS) | 2,250 TFLOPS dense (SemiAnalysis InferenceX) | analyst arithmetic, not published by any outlet |

For the real B300 baseline: **BF16 dense = 2,250 TFLOPS, FP8 dense = 4,500 TFLOPS, FP4 dense = 13,500 TFLOPS** ([SemiAnalysis InferenceX, B300 chip page](https://inferencex.semianalysis.com/chips/b300), `第三方/媒体推测` but a benchmark vendor's static hardware table); NVIDIA官方 states **15 PFLOPS dense NVFP4** for the full B300 implementation ([NVIDIA Developer Blog, 2025-08-22](https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/)). → **The dense/sparse conflation is the single biggest source of the factor-of-2 divergence.**

**Verdict: `未找到公开数据` for a dense-vs-sparse-labelled B30A FP16/BF16 figure. Published estimates span ~1.1–2.5 PFLOPS.**

### 1.4 FP8 / FP4

| Precision | Published B30A figure | Source | Label |
|---|---|---|---|
| FP4 (NVFP4) | **7.5 PFLOPS/package** | Tom's Hardware table via [EEPW, 2025-08-20](https://www.eepw.com.cn/article/202508/473207.htm); [TechPowerUp, 2025-08-19](https://www.techpowerup.com/340068/nvidia-prepares-cut-down-b30a-blackwell-ai-accelerator-for-chinese-market) | `第三方/媒体推测` |
| FP4 | **≈7.5 PFLOPS** (one die, four HBM stacks) | [AInvest, 2026-08-20](https://www.ainvest.com/news/4-6-billion-nvidia-door-china-squeeze-means-ai-trade-2608/) | `第三方/媒体推测` |
| FP4 | a competing leak published **3.5 PFLOPS** (by halving the B100 instead of the B300) | noted by [flopper.io](https://flopper.io/gpu/nvidia-b30a) | `第三方/媒体推测` |
| FP8 / FP6 | **5 PFLOPS/package** | Tom's table via EEPW (2025-08-20); AInvest (2026-08-20) | `第三方/媒体推测` |
| FP8 | **3.75 PFLOPS/package** | [TechPowerUp, 2025-08-19](https://www.techpowerup.com/340068/nvidia-prepares-cut-down-b30a-blackwell-ai-accelerator-for-chinese-market) | `第三方/媒体推测` |

`NVIDIA官方`: no B30A figures at all. Note the FP8 estimates conflict (3.75 vs 5 PFLOPS) and both sit **above** a literal half of B300's dense FP8 (2.25 PFLOPS).

### 1.5 INT8 TOPS

`未找到公开数据.` Tom's Hardware's table includes an INT8 row but the version reproduced by EEPW is internally inconsistent (H20 0.296 / H100 2 / B200 4.5 / B300 0.319) and cannot be read as a B30A INT8 figure ([EEPW, 2025-08-20](https://www.eepw.com.cn/article/202508/473207.htm)). No other outlet publishes B30A INT8.

### 1.6 TF32 / FP32

* **TF32:** 1.25 PFLOPS/package (Tom's Hardware table, [EEPW 2025-08-20](https://www.eepw.com.cn/article/202508/473207.htm)) vs 0.94 PFLOPS ([TechPowerUp](https://www.techpowerup.com/340068/nvidia-prepares-cut-down-b30a-blackwell-ai-accelerator-for-chinese-market)). → conflict; `第三方/媒体推测`.
* **FP32:** `未找到公开数据` (NVIDIA does not publish vector FP32 for B-series in the Blackwell Ultra blog; no outlet states it for B30A).

### 1.7 Memory type, capacity, bandwidth

* **144 GB HBM3E** = 50 % of B300's 288 GB — most-cited figure ([Tom's Hardware via EEPW, 2025-08-20](https://www.eepw.com.cn/article/202508/473207.htm); [AInvest, 2026-08-20](https://www.ainvest.com/news/4-6-billion-nvidia-door-china-squeeze-means-ai-trade-2608/)). `第三方/媒体推测`
* **96–144 GB HBM3E, bandwidth cut 8 TB/s → 4 TB/s** — 快科技/MyDrivers, 2026-05-15: *"HBM3e 显存容量从192-288GB降低到96到144GB，带宽从8TB/s降低到4TB/s"* ([MyDrivers, 2026-05-15](https://m.mydrivers.com/newsview/1122494.html)). `第三方/媒体推测` — this is the only source giving a range rather than 144 GB.
* **8-Hi HBM3E** (vs 12-Hi on B300) claimed by the WCCFTech dual-die rumour. `第三方/媒体推测`, contradicted by the single-die account.
* `NVIDIA官方`: none.
* **The "TPP 60,000 → 30,000" figures** quoted by MyDrivers (2026-05-15) are a **cut-and-paste of the export-control Total Processing Performance metric mislabelled as PFLOPS** and should not be used as a throughput spec ([MyDrivers, 2026-05-15](https://m.mydrivers.com/newsview/1122494.html)).

### 1.8 NVLink generation and per-GPU bidirectional GB/s; PCIe gen

* Reuters (2025-08-19) and Tom's Hardware (2025-08-20) both say **NVLink is retained** for scale-up, and Tom's explicitly says **it is unclear whether NVIDIA would cut the NVLink count to limit rack-scale / large-cluster builds** ([Reuters](https://www.reuters.com/world/china/nvidia-working-new-ai-chip-china-that-outperforms-h20-sources-say-2025-08-19/); [EEPW/Tom's Hardware](https://www.eepw.com.cn/article/202508/473207.htm)). `第三方/媒体推测`
* **The premise that B30A NVLink runs at "~half B300 rate" is UNVERIFIED.** No source found states a B30A-specific NVLink bandwidth. → **`未找到公开数据` for B30A NVLink GB/s.**
* The two numbers in circulation for the **baseline B300** are not in conflict once you separate directions:
  * **1.8 TB/s bidirectional GPU-to-GPU** (NVLink 5) — `NVIDIA官方` ([NVIDIA Developer Blog, 2025-08-22](https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/));
  * **900 GB/s per chip** — `第三方/媒体推测`, SemiAnalysis InferenceX's unidirectional/per-direction convention ([InferenceX B300](https://inferencex.semianalysis.com/chips/b300)).
  So **~900 GB/s and ~1.8 TB/s describe the same B300 link, not two different rates.** Use 1.8 TB/s bidirectional when comparing to NVIDIA's datasheet convention.
* **PCIe:** B300 = **PCIe Gen 6 (256 GB/s)** `NVIDIA官方` ([NVIDIA Developer Blog, 2025-08-22](https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/)). **B30A PCIe generation: `未找到公开数据`.**

### 1.9 TDP and form factor

* **~800 W: `未找到公开数据`.** Extensive searching found **no outlet** reporting an 800 W TDP for B30A. Treat the 800 W figure as unsupported.
* The only power figure in circulation is **600 W**, and it is attached to **B300A** (described as TSMC 4 nm, CoWoS-L, 144 GB HBM3E), not necessarily to B30A ([EEPW, 2025-08-26](http://m.eepw.com.cn/article/202508/473337.html); [TechWeb/快科技, 2025-09-05](https://m.techweb.com.cn/marticle/2025-09-05/2965455.shtml)). `第三方/媒体推测`
* **Form factor: `未找到公开数据`.** No source states SXM/OAM/PCIe for B30A. (Packaging is separately disputed as CoWoS-S vs CoWoS-L — see §1.2.)

### 1.10 Max scale-up domain (e.g. NVL8)

`未找到公开数据.` No outlet specifies an NVL8/rack domain for B30A. The Information's claim (relayed by Reuters/HKEJ) is only qualitative — that *"if arranged in large clusters, it could be used to train large language models"* ([HKEJ, 2025-11-07](https://www.hkej.com/landing/articleprint/id/4243408)). Tom's Hardware explicitly flags uncertainty about whether NVIDIA would reduce NVLink to cap cluster size ([EEPW, 2025-08-20](https://www.eepw.com.cn/article/202508/473207.htm)).

---

## 2. RTX 6000D / "RTX PRO 6000D" (Blackwell workstation card for China — shipped)

### 2.1 Announcement / launch date and export-control status

| Date | Event | Outlet / label |
|---|---|---|
| 2025-03 (GTC) | Full **RTX PRO 6000 Blackwell** launched (GB202, 24,064 CUDA cores, 96 GB GDDR7, 512-bit, 1,792 GB/s, 600 W workstation/server or 300 W Max-Q) | [MyDrivers/快科技, 2025-07-16](https://m.mydrivers.com/newsview/1062147.html) `第三方/媒体推测` (specs corroborate NVIDIA官方 product page) |
| 2025-05-24 | Reuters: NVIDIA to launch a cheaper Blackwell AI chip for China after export curbs — **"B40 or RTX Pro 6000D"**, GDDR7 instead of HBM, **no NVLink** (multi-GPU over PCIe), target price **US$6,500–8,000** | Reuters via [heise, 2025-05-26](https://www.heise.de/en/news/Cat-and-mouse-Nvidia-s-B40-to-circumvent-China-ban-10397137.html?view=print) `第三方/媒体推测` |
| 2025-07-14/16 | Jensen Huang in China announces H20 resumption + a new RTX PRO professional card; leaks name it **RTX 6000D**; Q3 2025 shipment start, 1–2 M units and up to **US$10 B** revenue projected in 2025 | [MyDrivers/快科技, 2025-07-16](https://m.mydrivers.com/newsview/1062147.html) `第三方/媒体推测` |
| 2025-08-21 | Reuters-reported plan: memory bandwidth **1,398 GB/s, just below the 1.4 TB/s threshold set in April**; initial shipments to select Chinese clients slated for September | [Notebookcheck, 2025-08-21](https://www.notebookcheck.net/Nvidia-readies-new-Blackwell-based-accelerators-tailored-for-China-under-US-export-limits.1091993.0.html) citing Reuters `第三方/媒体推测` |
| **2025-09-16** | **Reuters exclusive: launches/ships that week at ~CNY 50,000 (~US$7,000); "little favour with major firms"; samples tested slower than the (grey-market, <half-price) RTX 5090; JPMorgan expected 1.5 M units in H2'25, Morgan Stanley 2 M in the pipeline** | [Reuters, 2025-09-16](https://www.reuters.com/world/china/nvidias-new-rtx6000d-chip-china-finds-little-favour-with-major-firms-sources-say-2025-09-16/) / [CGTN mirror of Reuters](https://news.cgtn.com/news/2025-09-16/Nvidia-s-new-RTX6000D-chip-for-China-finds-little-favor-sources-say-1GIE0Yk3CMw/share_amp.html); [3DNews, 2025-09-16](https://3dnews.ru/1129320/sdelanniy-dlya-kitaya-uskoritel-nvidia-rtx-6000d-provalilsya-v-prodage-geforce-rtx-5090-luchshe/print?past-link) `第三方/媒体推测` |
| 2025-09-17 | FT: China's Cyberspace Administration tells Alibaba/Baidu/Tencent to stop evaluating and procuring NVIDIA export-modified accelerators, **explicitly including the RTX PRO 6000D** | [TechPowerUp, 2025-09-17](https://www.techpowerup.com/341078/china-blocks-tech-firms-from-buying-nvidia-ai-accelerators) (source: FT) `第三方/媒体推测` |
| 2025-11-25/26 | First Geekbench 6 OpenCL listing: **390,656** (full RTX PRO 6000 ≈ 450,000–500,000); 156 SMs / 19,968 CUDA cores; 83 GB seen; 2,430 MHz; **448-bit, 1,568 GB/s** | [i2hard, 2025-11-25](https://i2hard.ru/publications/49441/?lang=ru); [集微网/C114, 2025-11-26](https://www.c114.net.cn/industry/39880.html); [Tom's Hardware, 2025-11-26](https://www.tomshardware.com/pc-components/gpus/nvidia-rtx-pro-6000d-squeaks-ahead-of-rtx-5090d-in-geekbench-opencl-china-tailored-ai-card-still-performs-well-despite-regulatory-woes) `第三方/媒体推测` |
| **2026-02-12** | **First public teardown** (Bilibili UP主 "技数犬", relayed by 快科技): 19,968 CUDA (−17 %), 624 Tensor (−17 %), 156 RT (−17 %), 2,430 MHz (−7 %), **448-bit 84 GB, 1.79 → 1.57 TB/s (−13 %)**, FP32 ≈ 97.04 TFLOPS (−23 % total compute), fanless server-style PCB identical to RTX PRO 6000 except 28× 3 GB Samsung GDDR7 (14 per side), **no video output (TCC compute-only mode)**, measured card max power "just over 400 W"; rumoured to have no Chinese customer interest | [MyDrivers/快科技, 2026-02-12](http://m.mydrivers.com/newsview/1103953.html); [TweakTown teardown write-up](https://www.tweaktown.com/news/110135/nvidias-new-rtx-6000d-appears-in-teardown-84gb-gddr7-in-china-compared-to-the-full-96gb/index.html); [WindowsReport](https://windowsreport.com/nvidia-rtx-6000d-teardown-reveals-84gb-gddr7-and-cut-down-blackwell-specs/) `第三方/媒体推测` |
| **2026-07-14** | **First full independent benchmark of a retail RTX 6000D vs RTX PRO 6000 Blackwell Server Edition** (vLLM/Transformers/GEMM/power telemetry) — see §2.3–2.6 | [CSDN blog (SUAT-AIRI benchmark), 2026-07-14](https://fjiang.blog.csdn.net/article/details/162849417) `第三方/媒体推测` |
| 2026-09-15 | NVIDIA launches global **RTX PRO 5500** (GB202, 21,760 CUDA, 84 GB GDDR7, 448-bit, 1,398 GB/s, 600 W) — listed on NVIDIA sites in most regions but **not on the mainland-China site**; RTX 6000D still the China-specific part | [21ic via x-techcon, 2026-09-15](https://www.x-techcon.com/article/187140.html) `第三方/媒体推测` |
| **NVIDIA官方** | **No RTX 6000D product page or datasheet found.** → `未找到公开数据` for official specs. Naming varies by source: "RTX 6000D", "RTX PRO 6000D", "RTX Pro 6000D". | `NVIDIA官方` (absence) |

### 2.2 Process node / architecture

* **TSMC 4N, GB202 die, Blackwell** — `第三方/媒体推测` ([MyDrivers teardown, 2026-02-12](http://m.mydrivers.com/newsview/1103953.html)). Compute Capability **12.0** confirmed by CSDN's driver readout ([CSDN, 2026-07-14](https://fjiang.blog.csdn.net/article/details/162849417)). Same die as RTX PRO 6000 Blackwell (92.2 B transistors, GB202 — [CpuTronic](https://cputronic.com/gpu/nvidia-rtx-6000d)).
* No official NVIDIA die/node statement for the D part.

### 2.3 Dense FP16/BF16 TFLOPS vs with-sparsity

* **`NVIDIA官方`: none.** No NVIDIA figure for RTX 6000D FP16/BF16 tensor throughput at any sparsity.
* **Measured (dense GEMM, no sparsity), CSDN 2026-07-14, 16,384×16,384, BF16/FP16:** RTX 6000D **144.09 / 144.02 TFLOPS** vs RTX PRO 6000 **401.01 / 397.67 TFLOPS** ⇒ the D card retains only **~36 %** of the full card's dense BF16/FP16 GEMM throughput. ([CSDN, 2026-07-14](https://fjiang.blog.csdn.net/article/details/162849417)) `第三方/媒体推测`
* The cut is far larger than the 17 % core-count reduction (188→156 SMs; 2.6→2.43 GHz), and the test authors conclude the limit is a **product-level restriction on low-precision Tensor Core throughput**, not clocks/SM count/thermals. `第三方/媒体推测` — single-source measurement, treat as indicative.
* **With-sparsity figures: `未找到公开数据` for any precision.**
* **Do not** read CpuTronic's "FP16 97.04 TFLOPS" as tensor FP16 — it equals the card's FP32 vector rate ([CpuTronic](https://cputronic.com/gpu/nvidia-rtx-6000d)).

### 2.4 FP8 / FP4

`未找到公开数据` for RTX 6000D specifically. The only aggregate AI figure is for the **full** RTX PRO 6000 Blackwell: **up to 4,000 AI TOPS** (marketing aggregate; no precision or dense/sparse label given in the cited source) ([21ic via x-techcon, 2026-09-15](https://www.x-techcon.com/article/187140.html)) `第三方/媒体推测` transcribing the official spec sheet. No FP8/FP4 TFLOPS are published for the D card by anyone.

### 2.5 INT8 TOPS

* **Measured (INT8→INT32 tensor path), CSDN 2026-07-14:** RTX 6000D **140.32 TOPS** vs RTX PRO 6000 **238.41 TOPS** (1.70× gap) ([CSDN](https://fjiang.blog.csdn.net/article/details/162849417)) `第三方/媒体推测`.
* **NVIDIA官方: none.** `未找到公开数据` for a vendor INT8 rating.

### 2.6 TF32 / FP32

| Metric | RTX 6000D | RTX PRO 6000 Blackwell | Source / label |
|---|---|---|---|
| FP32 (vector, spec) | **~97.04 TFLOPS** (MyDrivers teardown); 95.099 TFLOPS (CpuTronic) | **125 TFLOPS** | [MyDrivers, 2026-02-12](http://m.mydrivers.com/newsview/1103953.html); [CpuTronic](https://cputronic.com/gpu/nvidia-rtx-6000d) `第三方/媒体推测` |
| FP32 (measured GEMM) | **66.67 TFLOPS** | 75.48 TFLOPS (1.13×) | [CSDN, 2026-07-14](https://fjiang.blog.csdn.net/article/details/162849417) `第三方/媒体推测` |
| TF32 (measured GEMM) | **72.05 TFLOPS** | 194.99 TFLOPS (2.71×) | [CSDN, 2026-07-14](https://fjiang.blog.csdn.net/article/details/162849417) `第三方/媒体推测` |
| FP64 | 1.516 TFLOPS (low confidence) | — | [CpuTronic](https://cputronic.com/gpu/nvidia-rtx-6000d) `第三方/媒体推测` |

`NVIDIA官方`: none for RTX 6000D. Note the measured TF32/FP32 ratio on the D card is only **~1.08×**, versus ~2.58× on the full card — again indicating a throttled tensor path in the D part.

### 2.7 Memory type, capacity, bandwidth

| Item | Value | Source / label |
|---|---|---|
| Type | **GDDR7** (28 × 3 GB Samsung modules, 14 front / 14 back); no ECC status published | [MyDrivers teardown, 2026-02-12](http://m.mydrivers.com/newsview/1103953.html) `第三方/媒体推测` |
| Capacity | **84 GB** nominal; `nvidia-smi` reports **85,651 MiB**; OS-visible **83.05 GiB**; Geekbench saw 83 GB | [CSDN, 2026-07-14](https://fjiang.blog.csdn.net/article/details/162849417); [i2hard, 2025-11-25](https://i2hard.ru/publications/49441/?lang=ru) `第三方/媒体推测` |
| Bus width | **448-bit** (14 × 32-bit controllers) | [MyDrivers, 2026-02-12](http://m.mydrivers.com/newsview/1103953.html); [i2hard, 2025-11-25](https://i2hard.ru/publications/49441/?lang=ru) `第三方/媒体推测` |
| Bandwidth (nominal) | **1,568 GB/s (1.57 TB/s)** | [i2hard, 2025-11-25](https://i2hard.ru/publications/49441/?lang=ru); [MyDrivers, 2026-02-12](http://m.mydrivers.com/newsview/1103953.html) `第三方/媒体推测` |
| Bandwidth (measured, device copy, read+write aggregate) | **1,279.44 GB/s** | [CSDN, 2026-07-14](https://fjiang.blog.csdn.net/article/details/162849417) `第三方/媒体推测` |
| Full RTX PRO 6000 Blackwell | 96 GB GDDR7, 512-bit, 1,792 GB/s, 28 Gbps | [MyDrivers, 2025-07-16](https://m.mydrivers.com/newsview/1062147.html) `第三方/媒体推测` |

**Bandwidth conflicts — flag explicitly:**
1. **1,100 GB/s** (guessed 384-bit @ 23 GHz) — early Chinese leak, [MyDrivers, 2025-07-16](https://m.mydrivers.com/newsview/1062147.html). **Superseded / wrong.**
2. **1,398 GB/s**, deliberately just under the April-2025 **1.4 TB/s** export threshold — the Reuters-reported *plan*, [Notebookcheck, 2025-08-21](https://www.notebookcheck.net/Nvidia-readies-new-Blackwell-based-accelerators-tailored-for-China-under-US-export-limits.1091993.0.html). Note this is ~11 % **lower** than what shipped (1,568 GB/s), and 1,398 GB/s is exactly the figure NVIDIA later used for the **global RTX PRO 5500** (448-bit, 2026-09-15, [x-techcon](https://www.x-techcon.com/article/187140.html)).
3. **1,568 GB/s measured on retail hardware** — [i2hard 2025-11-25](https://i2hard.ru/publications/49441/?lang=ru); [MyDrivers 2026-02-12](http://m.mydrivers.com/newsview/1103953.html). A Russian report that the card "fits the 1.4 TB/s limit" (and describes the memory as GDDR6) is **inconsistent** with the 448-bit/1,568 GB/s teardown ([3DNews, 2025-09-16](https://3dnews.ru/1129320/sdelanniy-dlya-kitaya-uskoritel-nvidia-rtx-6000d-provalilsya-v-prodage-geforce-rtx-5090-luchshe/print?past-link)).

### 2.8 NVLink generation and per-GPU bidirectional GB/s; PCIe gen

* **No NVLink.** Reuters' B40/RTX PRO 6000D report states the fast NVLink interconnect would be **completely absent**, with multi-GPU communication over PCIe ([heise, 2025-05-26](https://www.heise.de/en/news/Cat-and-mouse-Nvidia-s-B40-to-circumvent-China-ban-10397137.html?view=print)). No source found indicates NVLink on the full RTX PRO 6000 Blackwell either. → **NVLink generation: none. Per-GPU bidirectional NVLink GB/s: not applicable / `未找到公开数据` (deliberate removal, not a reduced rate).**
* **PCIe: Gen 5 ×16** — [CpuTronic](https://cputronic.com/gpu/nvidia-rtx-6000d); confirmed by CSDN's driver readout ([CSDN, 2026-07-14](https://fjiang.blog.csdn.net/article/details/162849417)). `第三方/媒体推测` (no official listing).
* Practical multi-GPU: teardown notes 4 cards = 336 GB and 8 cards = 672 GB of aggregate memory for quantised 671B-class models, over PCIe ([MyDrivers, 2026-02-12](http://m.mydrivers.com/newsview/1103953.html)).

### 2.9 TDP and form factor

* **Power limit / TDP: 600 W** — [CpuTronic](https://cputronic.com/gpu/nvidia-rtx-6000d); CSDN read a **600 W** power limit directly from the driver and measured **P90 600.65 W / max 611.15 W** ([CSDN, 2026-07-14](https://fjiang.blog.csdn.net/article/details/162849417)); [21ic/x-techcon, 2026-09-15](https://www.x-techcon.com/article/187140.html) also states 600 W. `第三方/媒体推测`
* **Conflict:** the Feb-2026 Bilibili teardown measured "整卡最大功耗400W出头" (just over 400 W) ([MyDrivers, 2026-02-12](http://m.mydrivers.com/newsview/1103953.html)). Reconcilable only if that test never saturated the board; the July-2026 controlled benchmark hit 611 W.
* **Form factor:** fanless (passive) dual-slot server-style card needing chassis airflow — identical PCB layout to the RTX PRO 6000 Server Edition; **16-pin (CEM5)** power connector; 3D-stereo sync (4-pin) and frame-lock/sync headers on the top edge; **four DisplayPort 2.1b connectors physically present but video output disabled — the card runs in compute-only TCC mode under Windows** ([MyDrivers, 2026-02-12](http://m.mydrivers.com/newsview/1103953.html)). `第三方/媒体推测`
* **No NVIDIA官方 form-factor document.**

### 2.10 Max scale-up domain

`未找到公开数据.` With **no NVLink**, the RTX 6000D has no NVLink scale-up domain (contrast: B300 = NVLink 5, world size 8, per [SemiAnalysis InferenceX](https://inferencex.semianalysis.com/chips/b300)). Multi-GPU scaling is PCIe-only. NVIDIA publishes no maximum supported multi-GPU count or domain for this part.

---

## 3. B30A successors / other 2026 China parts (as of 2026-09-18)

* **No successor to B30A has been announced or credibly reported.** `未找到公开数据`. ([flopper.io B30A entry](https://flopper.io/gpu/nvidia-b30a); [MyDrivers, 2026-05-15](https://m.mydrivers.com/newsview/1122494.html); [AInvest, 2026-08-20](https://www.ainvest.com/news/4-6-billion-nvidia-door-china-squeeze-means-ai-trade-2608/))
* **The only US-approved NVIDIA datacenter product for China is last-generation Hopper H200** (plus AMD MI325X), under a 25 % US revenue share, per-buyer caps of roughly 75,000 chips and a total-export ceiling of ~50 % of US-level shipments; first units reached customers around July 2026 and volumes are officially described as "trivial". ([Seoul Economic Daily, 2026-07-15](https://en.sedaily.com/international/2026/07/15/nvidias-h200-chip-exports-to-china-have-begun); [The Star/Reuters, 2026-07-18](https://www.thestar.com.my/aseanplus/aseanplus-news/2026/07/18/us-says-nvidias-h200-exports-to-china-remain-trivial-despite-approvals); [AInvest, 2026-08-20](https://www.ainvest.com/news/4-6-billion-nvidia-door-china-squeeze-means-ai-trade-2608/))
* **China's own gate tightened:** in May 2026 Chinese security authorities certified **nine domestic AI processors** (Huawei Ascend 310/910, Alibaba T-Head Zhenwu, Biren, Hygon and others) under the **Anke** "secure and reliable" framework, structurally excluding foreign silicon from state/Xinchuang procurement ([AInvest, 2026-08-20](https://www.ainvest.com/news/4-6-billion-nvidia-door-china-squeeze-means-ai-trade-2608/) citing Tom's Hardware).
* **NB:** NVIDIA's new **RTX PRO 5500** (2026-09-15, 84 GB GDDR7, 1,398 GB/s, 21,760 CUDA cores, 600 W) is a *global* SKU — it is **not** listed on NVIDIA's mainland-China site and is not a China-specific part, though its bandwidth sits at the old 1.4 TB/s China threshold ([21ic via x-techcon, 2026-09-15](https://www.x-techcon.com/article/187140.html)).
* NVIDIA's Blackwell China revenue collapsed from US$4.6 B in Q1 FY2026 to **zero** in Q1 FY2027, with Q2 FY2027 guidance assuming no China datacenter compute revenue ([AInvest, 2026-08-20](https://www.ainvest.com/news/4-6-billion-nvidia-door-china-squeeze-means-ai-trade-2608/), citing NVIDIA's SEC-filed CFO commentary — treat the aggregation as a pointer, not a primary source).

---

## 4. Consolidated conflict list (do not average these)

| Item | Value A | Value B | Value C | Resolution |
|---|---|---|---|---|
| B30A die | Single die (Reuters/Tom's/TPU) | Dual-die (WCCFTech) | — | Unresolved; Reuters account more widely corroborated |
| B30A BF16/FP16 | 2.5 PFLOPS (Tom's) | 1.875 PFLOPS (TPU) | ~1.125 PFLOPS (half of B300 dense) | All are arithmetic on *different* B300 baselines; dense/sparse never labelled |
| B30A FP8 | 5 PFLOPS (Tom's, AInvest) | 3.75 PFLOPS (TPU) | — | Unresolved |
| B30A FP4 | 7.5 PFLOPS | 3.5 PFLOPS (halving B100) | — | 7.5 is the majority figure |
| B30A HBM | 144 GB | 96–144 GB (MyDrivers) | 8-Hi HBM3E (WCCFTech) | Unresolved |
| B30A packaging | CoWoS-S (Tom's table) | CoWoS-L (B300A Chinese reports) | — | Unresolved |
| B30A TDP | 600 W **for B300A**, not B30A | ~800 W: no source found | — | Treat 800 W as **unsupported** |
| B30A status | Blocked by White House (Nov 2025) | "Revising design for reconsideration" | Not approved as of May 2026 | Blocked/unapproved |
| RTX 6000D bandwidth | 1,100 GB/s (leak) | 1,398 GB/s (Reuters plan) | 1,568 GB/s (measured) | 1,568 GB/s on shipped retail hardware |
| RTX 6000D L2 | 112 MiB (measured) | 128 MB (CpuTronic) | — | Measured value preferred |
| RTX 6000D max power | ~400 W (Feb 2026 teardown) | 611 W (Jul 2026 benchmark) | 600 W limit | 600 W limit; 400 W test was sub-saturated |
| RTX 6000D tensor throughput | 17 % core cut (teardown framing) | ~64 % dense BF16 throughput cut (measured) | — | The tensor path is cut far more than core count implies |
| H200 licensing date | Jan 2026 (flopper) | Feb/Mar 2026 (AInvest) | May 2026 approvals (Reuters) | Licences began early 2026; approvals announced May 2026; exports "trivial" by Jul 2026 |

---

## 5. Official vs unconfirmed — one-line summary per part

* **B30A:** **NVIDIA官方 = 0 specifications; no confirmation the SKU exists.** Every number is `第三方/媒体推测` and the estimates disagree by >2× on most precisions. Status: designed, sampled to Chinese customers, then **blocked by the White House in Nov 2025**, still unapproved as of 2026-09-18.
* **RTX 6000D:** **NVIDIA官方 = no product page, no datasheet.** The reliable figures come from retail-hardware teardowns and a controlled benchmark (84 GB GDDR7 / 448-bit / 1,568 GB/s / 19,968 CUDA / 624 Tensor / 2,430 MHz / 600 W / no NVLink / no video output) plus Reuters' launch-price and demand reporting (≈CNY 50,000, tepid demand, CAC discouraging purchases). Spec-level `第三方/媒体推测` only.
* **Successor:** none. H200 (Hopper) is the only US-approved China datacenter part through mid-2026.
