# my_opt_claude / legalizer_claude — 欄位切片 Floorplan 優化器

ICCAD 2026 FloorSet Challenge 的參賽 optimizer。這份文件記錄架構、設計依據、目前成績與後續路線。

**目前成績(2026-07-04)**:完整 100-case validation **Total Score ≈ 1.31–1.32**
(重寫前的 shelf-packing 版本為 4.31),100/100 feasible,平均 runtime ≈ 5.1 s。
目標:**< 1.15**。

```bash
# 完整評分
python3 iccad2026_evaluate.py --evaluate my_opt_claude.py

# 單一 case
python3 iccad2026_evaluate.py --evaluate my_opt_claude.py --test-id 99

# 提交格式檢查
python3 iccad2026_evaluate.py --validate my_opt_claude.py
```

---

## 1. 評分函數分析(所有設計決策的出發點)

```
Cost = (1 + 0.5·(HPWL_gap + Area_gap)) × exp(2·V_rel) × RuntimeFactor
Total = Σ Cost_i · e^{n_i/12} / Σ e^{n_j/12}
```

由 validation set 實測得到的關鍵事實(皆為一般性質、非特定 case):

| 事實 | 設計上的意義 |
|---|---|
| 權重 `e^{n/12}`:n≥100 的 case 佔總分 **83%**、n≥80 佔 97% | 時間預算按 `exp(n/20)` 指數配置,小 case 快速解決 |
| GT utilization 0.95–0.99 | Area 目標:bbox ≈ 總面積 / 0.97 |
| soft block **只驗面積(1% 內),aspect ratio 無限制** | 可以用「精確填滿」的切片幾何,Area_gap 幾乎歸零 |
| MIB 群組成員的 target area **完全相同** | 同欄放置 ⇒ 形狀自動一致 ⇒ MIB 違規恆為 0 |
| 帶 Touch-Top/Right tag 的 preplaced 塊,其邊緣就是 GT frame 邊界 | 直接把 frame 高度 H 釘在該值(一般規則,不是看 case) |
| infeasible = 10 分,遠比任何品質損失貴 | 所有硬限制(overlap、面積、fixed/preplaced)必須由**構造保證**,不靠修補 |

## 2. 架構:欄位切片(column slicing)+ 模擬退火

### 幾何模型
- 選定 frame 高度 `H`(總面積/0.96 與 pins bbox 長寬比推得;若有 Touch-Top 的 preplaced 塊則精確釘住)。
- 把 block 分組成 **unit**(cluster 群、soft-MIB 群,用 union-find 合併)。
- 由左到右排欄。soft slice 的寬 = 欄寬 `w_c`、高 = 面積/`w_c` ⇒ **面積精確、零 overlap、欄高恰為 H**:
  `w_c = soft_area / (H − rigid_h − obstacle_h)`
- **preplaced** 塊固定原位當障礙物;堆疊跳過其 y 區間;障礙物旁的空條以「窄段」回收(否則整條 y 帶 × 欄寬全浪費)。
- **fixed-shape** 塊尺寸精確,欄寬至少撐到其寬度。
- **cluster** unit 內部可攤平成並排 **chunk**(帶狀 2D 排列):多個 Bottom/Top tag 成員各佔一條 chunk 的底/頂;帶 Left/Right tag 的成員放最外側 chunk。chunk 解不可行時逐步合併重解,而不是整組退化成單疊。
- **boundary** 處理全部結構化:L/R-forced unit 進最左/最右欄(且禁止進窄段)、B/T unit 排欄底/欄頂、頂端 unit 上抬貼齊全域頂邊。

以上保證:overlap = 0、面積精確、fixed/preplaced 精確 ⇒ **永遠 feasible**;
MIB 違規 = 0;純 movable cluster 違規 = 0(構造連通)。

### 搜尋(時間預算內)
1. **Probe**:兩種方向(原始/轉置)× 欄數 {C−1, C, C+1} 短跑取優
   (轉置把 L↔B、R↔T 對調;當 L/R unit 明顯多於 B/T 時直接跳過轉置)。
2. **SA**:低溫退火,移動集 = unit 搬欄/交換/欄內重排/unit 內重排,
   35% 移動由連線導向(沿 b2b/p2b edge 把 unit 拉向夥伴所在欄)。
   成本 proxy = 真實評分公式(HPWL 正規化基準週期性再校準)。
3. **Polish**:溢出降頂 → boundary 違規修復 → B/T 分散 → 連線加權欄內排序 → 全 unit 掃描式最佳改善。

### 時間預算
`budget(n) = clamp(0.06·e^{n/20}, 0.8, 24)` 秒 → 100 case 平均 ≈ 5 s,
n=100 約 9 s、n=120 約 24 s(指數成長,符合權重結構)。

## 3. Diffusion checkpoint 的角色與極限

目前用 `checkpoints/diffusion_stable_xywh_order_2day/step_00080000.pt` 做 4 步 DDIM
refine,只拿它的 **相對位置** 當初始欄位分配的種子。

**checkpoint 不是目前的瓶頸**,證據:拿 ground truth 座標當種子(完美 seed)跑同一套管線,
HPWL 仍停在 GT 的 ~1.35 倍 —— 上限卡在「欄位切片的幾何表示 + SA 搜尋品質」,
不在種子好壞。所以短期內重訓模型的邊際效益低;等搜尋端(吞吐、平行化)改善後,
更好的模型才會重新變成有效槓桿(讓 SA 從更接近最優的初始出發)。

## 3.5 重要校準:GT 下限 = 1.108

把 ground truth 原封不動丟進評分器,總分是 **1.1079** —— 因為 **GT 自己就有大量
boundary 違規**(n≥100 平均 vrel 0.053,和我們的 0.04–0.05 同級;例如 case 88 GT
有 7 個 bnd 違規,cost 1.249)。這代表:

- violations 這個槓桿已經打到底(我們已在 GT 水準,個別 case 甚至更好)。
- **目標 1.15 = 只比 GT 差 0.042**,等於 HPWL_gap 要壓到 ~0.08。
- 欄位切片表示的 HPWL 天花板實測 ~1.35–1.44×GT(用 GT 座標當種子也一樣),
  所以 **1.15 在目前架構下達不到**;需要下一代精修層(見路線圖第 0 點)。

## 4. 已知限制與後續路線(按預期收益排序)

0. **突破表示天花板(必要條件)**:欄位模型把 x 量化到欄、y 由堆疊順序決定,
   HPWL 下限 ~1.35×GT。要到 1.15 需要能表達任意 packing 的精修層,例如:
   以目前解(或 diffusion 輸出)建 constraint graph(sequence pair),
   做保拓樸的壓縮 + soft block 連續變形(面積不變、長寬比自由),
   直接最小化 HPWL。欄位解可當它的合法初始解。

1. **Run 變異大**(同 case 的 HPWL_gap 在 0.16–0.45 之間跳;好的 run 已能到單 case 1.16–1.19)
   → **平行多重啟**:多 process 同時跑不同 (方向, 欄數, 種子) 的 SA,取真實成本最佳者。
   牆鐘時間不變、有效搜尋量 ×N。*(評分平台允許平行化/GPU。)*
2. **SA 吞吐**:n=120 目前 ~1000 evals/s(純 Python)→ 向量化/編譯 `_layout` + `_evaluate`
   可再 5–10×,直接改善收斂與變異。
3. 欄內順序用解析式(加權中位數)排序取代隨機搜尋。
4. 個別難解的 tag 組合(同 unit 同時含 L 與 R、B-cluster 底部被 preplaced 擋住)
   目前以違規成本交給搜尋權衡 —— 屬一般結構,不做 case-specific 處理。

## 5. 檔案

| 檔案 | 內容 |
|---|---|
| `legalizer_claude.py` | 欄位切片幾何 + SA/polish 全部邏輯(`legalize_rectangles` 入口) |
| `my_opt_claude.py` | 參賽入口 `MyOptimizer`:diffusion seed → legalizer;時間預算配置 |

歷史注意事項(除錯時曾踩過的坑,已修):
- 欄交換/移動類的操作**必須**保護 L/R-forced unit 不離開邊欄。
- chunk 帶不可行時要漸進合併重解;直接退化成單疊會失去所有 B/T 對齊。
- 欄溢出重試要解 `w₂ = soft/(H − overhead)`(rigid/障礙高度不隨寬度縮小),
  不能整條寬度等比例放大。
- 障礙物遮蔽是「整欄寬」的,寬欄 + 小障礙 = 大浪費;必須回收側邊窄條。
