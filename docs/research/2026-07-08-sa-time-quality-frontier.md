# 2026-07-08 SA 時間-品質前沿研究(deep-research v4 合成)

**問題**:E2 證明時間換分數在 heavy band 成立後——如何更有效率地把 wall-clock/平行度換成 no-runtime 分數,或同分數降時間。
**方法**:deep-research workflow(105 agents;16 claims 3 票 confirmed、6 refuted、3 因 usage limit 未驗證;合成階段因 usage limit 中斷,本文由 scheduler 依 confirmed claims 手動合成)。**本地量測先行**:W1/E2 掃描已在研究完成前實測並促轉(見 [docs/experiments/2026-07-08-w1-e2-scan-promotion.md](../experiments/2026-07-08-w1-e2-scan-promotion.md),促轉後 1.2156),文獻用於解釋機制與指引下一棒。

## 已驗證文獻事實(全 3-0,除註明)

| # | 事實 | 來源 | 對本 repo 的意義 |
|---|---|---|---|
| C1/C4 | PARSAC 在 112 核 = **純獨立 restart、無 worker 間通訊、同 config 只差 seed** | arXiv 2405.05495 + IntelLabs/parsac | W1 的獨立 restart 設計 = SOTA 同款;連 config 多樣性都非必需(我們有,屬加分) |
| C5/C6 | 已知時間預算下,**全預算長鏈勝過多短 restart 取最好**(P-FAL-1,統計顯著) | arXiv 1709.02877 | 加寬 = 加更多全長鏈 ✓(現行設計正確);不要縮短單鏈換 restart 數 |
| C7 | 獨立 multistart **報酬遞減定量:加速 ≈ N/2**(4 鏈→2-3×,8 鏈→4×) | 同上 | 12→32 已收 −0.0064;32→44 邊際更小,**32-40 workers 即甜蜜點**(48 核機) |
| C10/C11 | **Fast-SA 三段式溫度排程:同品質 12-17× 提速**(B*-tree,GSRC n100-300;episode-level,零 per-move 成本) | Chen-Chang ISPD'05 | 換排程 = 零吞吐成本的時間→分數槓桿,恰在我們的 n 區間;風險 =「調校即承重」 |
| C9 | **Slicing 表示 delta-評估定理**:每 move 只需沿受影響 path/fork 重算(O(depth)),原始出處 | Wong-Liu polish(UT Austin 鏡像) | column 局部 move 原則上只需重算受影響欄;FAST_EVAL 快取已是半套實作,`_stack_column` miss 路徑是剩餘礦 |
| C2/C3 | PARSAC 吞吐來源 = **C++ 熱迴圈**(「move loop 永不進直譯器」),非 Python 層技巧;repo 56.7% C++ | 同 C1 | 吞吐的正路是編譯熱路徑,不是再擰 Python |
| C15 | **低改動 AOT 編譯(Nuitka / untyped Cython / mypyc)在 list-heavy 數值碼上無效甚至倒退** | arXiv 2505.02346 | 「懶人編譯」路線證據級否定;要就 **typed rewrite**(C-ext / typed Cython) |
| C13/C14 | PyInstaller frozen + multiprocessing:spawn 需 PyInstaller 版 `freeze_support()`(**所有平台**);**fork 是 frozen 下唯一「可能免改就動」的 start method** | pyinstaller.org | 我們的 pool 已用 fork(column_slicing.py:2520)✓ 設計上最低風險;但見 refuted——**仍必須實測 frozen binary** |
| C16(2-0) | 觀測 solver-trace 特徵的**動態預算政策比固定 cutoff 省 40-65% 時間**(Las Vegas 域,注意域差:backtracking search 非 anytime SA) | Kautz et al. AAAI'02 | **E2-adaptive**(邊際改善驅動的預算延伸)的理論背書;有機會拿 scale3 的 −0.019 而避開 69s worst-case |
| C8 | Luby universal restart schedule:分布無關、距最優僅 log factor | Luby(UT) | 背景理論;我們是 anytime SA 非 Las Vegas,直接適用性低 |
| C12 | 「更快的單候選評估直接轉換成同 wall-clock 更好的最終品質」 | FAST-SP(Illinois) | E2 實測的同一命題之文獻版 |

## 被否決 / 未決(勿引用為依據)

- 「warm-start/reannealing 無益於獨立 restart」**0-3 否決**——注意方向:是該表述不被來源支持,**warm-start 的價值仍未決**,兩個方向都不要引用。
- 「PyInstaller multiprocessing 問題僅限 Windows、POSIX 天然安全」**0-3 否決**——Linux 也不自動安全,frozen binary + fork pool 必須實測。
- 「B*-tree 攤銷線性重評估」「FAST-SP O(n log log n) 可取代全量重推」的具體表述(1-2,弱證據);numba/llvmlite 打包宣稱(0-3)。
- 未驗證(usage-limit 犧牲):Codon/PyPy 在純 Python benchmark 的 17×/7.5×;Luby vs 動態政策的 6× 差;LK-anytime 停時政策效用。

## 合成排序(對 n≥100 band 的 score-per-effort)

| # | 干預 | 期望 | 狀態 / 前置 |
|---|---|---|---|
| 1 | **W1 portfolio 加寬(32 workers/configs)** | −0.0064,**零 runtime 代價** | ✅ 已實測促轉(C1/C4/C6/C7 背書) |
| 2 | **E2 預算 ×2 @ n≥100** | −0.009~−0.013(runtime 代價見 RTF 備忘) | ✅ 已實測促轉(user 核准);提交期重審 |
| 3 | **E2-adaptive 預算延伸**:base 預算後,依「最後窗口 best-of-workers 改善率」決定 +25% 延伸,cap ~2.5×(C16 域轉移假設需實測) | 目標 = scale3 的 −0.019 級增益、worst-case ≪ 69s | 下一棒;mounting = `column_backbone` 預算/`_parallel_solve` 收尾,episode-level |
| 4 | **Fast-SA 三段式排程 probe**(C10/C11) | 文獻 12-17× 同品質提速 → 同預算更深收斂 | dormant flag + paired;「調校即承重」風險 → 先 tail-5 小規模 paired 再全量 |
| 5 | **`_stack_column`/`_finish_layout` typed C-ext / typed Cython**(C2/C3/C15,理論藍圖 C9) | 2×+ tail 吞吐 ≈ 再一份 E2 而不加時 | 3-5 天;**先過 #6 打包 harness**;「懶人編譯」已被 C15 否定 |
| 6 | **PyInstaller 打包驗證(提前至現在)**(C13/C14 + refuted「POSIX 安全」神話) | 風險排除;提交必經;解鎖 #5 | fork pool ✓ 但必須實測 frozen binary + multiprocessing + torch-free import |
| 停 | GPU SA / torch 打包;warm-start 縮時 | — | A100 只在 ML 復活時考慮;warm-start 價值未決 + seed 通道前科 |

## 與 0709 判定的關係

本軸(W1/E2/排程/吞吐)全部作用於 production SA,與 ML 路線正交。scale probe 判死則此軸 + 已收案的理論收束(自洽聯合幾何 = ML 唯一形態)就是剩餘全部提分空間;判活則 best-of-N + cost-select(s4 8.86 → s32 7.33,未平緩)是 ML 兌現端的既測放大器。

## 工件

- 驗證後 claims JSON:session scratchpad `sa_research_claims.json`(16 confirmed / 6 refuted / 3 unverified,來源與逐字 quote 齊)。
- workflow:`wf_7847a250-d8a`(105 agents;合成步因 5h usage limit 中斷,claims 完整)。
