# 後端方法外部交換報告設計

## 目的

新增 `docs/research/後端報告.md`，作為 `docs/research/前端報告.md` 的配套文件，向外部交換對象說明目前後端求解流程、實測證據、已否定方向與可轉移的工程經驗。文件必須讓未閱讀原始碼的讀者理解後端如何把問題輸入轉成合法版圖，以及品質損失主要發生在哪裡。

## 核心論點

在 FloorSet SoC floorplanning 問題中，我們以 legality-first 的候選組合取代單一路徑求解：統一解析約束後，利用可選 guidance 建立多種 relative-order 候選，經分層修復與品質 refinement，再以硬合法性優先、V10 no-runtime proxy 次之的規則選解；現有證據支持 100/100 可行性，但面積膨脹仍是主要品質瓶頸。

## 讀者與披露邊界

- 讀者是外部技術交換對象，不假設其熟悉本 repository。
- 可以披露穩定的演算法機制、資料流、公式、彙總實驗結果與已封存的負面結果。
- 不披露 checkpoint 檔案、內部調參腳本、未封存策略、尚未由 evaluator 證實的改善宣稱或只存在於 scratchpad 的數字。
- 環境變數與 profile 名稱只在有助於理解機制時出現；不把可選或歷史路徑描述成 production default。

## 資料來源優先序

若來源互相衝突，依下列順序決定正文：

1. `src/architecture_v5_optimizer.py` 與 `src/floorset_arch/` 的現行 production code。
2. 測試所固定的行為契約與 evaluator 驗證輸出。
3. 已封存於 `docs/experiments/`、`docs/design/` 或 `artifacts/` 的實驗證據。
4. `README.md` 的架構說明。
5. `docs/research/前端報告.md` 只提供敘事模板和共用基準數字，不作為後端機制的唯一證據。

無法交叉確認的數字不進正文；必要時改寫為定性描述，並在文件末尾列出證據邊界。

## 術語表

| 正式名稱 | 首次定義 | 文件內規則 |
| --- | --- | --- |
| 後端 | production solver 的 candidate generation、repair、refinement 與 ranking | 不與 diffusion frontend 混稱 |
| `Instance` | evaluator tensors 經解析後的統一問題表示 | 保留程式型別名稱 |
| `Placement` | 候選矩形集合及其輸出轉接物件 | 保留程式型別名稱 |
| anchor guidance | 神經或 deterministic guidance 轉成的錨點、優先度、長寬比與相對軸資訊 | 首次說明後使用英文短名 |
| candidate portfolio | 同一 instance 上的多策略候選集合 | 不簡稱為 ensemble |
| repair | 消除重疊並處理 boundary、grouping、MIB 等約束的修復階段 | 與 quality refinement 分開 |
| hard-legality gate | 在品質分數之前淘汰或降級非法候選的排序規則 | 不寫成軟懲罰 |
| V10 no-runtime proxy | 對官方 V10 成本中非 runtime 部分的本地代理排序 | 不宣稱等同官方總分 |
| official100 | 官方 100 個 validation cases 的整體評估 | 數字必須標示 runtime 是否計入 |

## 章節設計

報告沿用前端報告的外部交換節奏，而非逐檔案 API 文件：

1. **一句話總結**：先交代 legality-first portfolio、100/100 可行性及面積瓶頸。
2. **後端的任務邊界**：說明輸入、輸出、五類約束與官方成本的關係。
3. **統一問題表示**：介紹 parser、`Instance`、`Placement`，解釋為何所有路徑共用同一契約。
4. **Guidance 是偏置而非答案**：說明 shared heads 如何影響 anchor、shape、priority 與 pairwise axis，並保留無 checkpoint 的降級路徑。
5. **Candidate portfolio 與預算**：解釋為何不押單一路徑，以及 instance-aware budget 如何控制候選數與 refinement。
6. **Relative-order 建構式放置**：以資料流描述初始合法結構的生成，不陷入逐函式細節。
7. **Repair 與 quality refinement**：分開說明硬約束修復和可行域內品質改善，指出面積損失風險。
8. **Hard-legality gate 與 V10 proxy**：明確寫出 legality-first 的排序層級及 proxy 邊界。
9. **實測品質**：只列可追溯的 official100 可行性、品質、area/HPWL gap、利用率與 runtime 證據。
10. **被量測否定的方向**：整理已有 paired evaluation 或封存結論支持的負面結果。
11. **給交換對象的三個要點**：提煉可移植的設計決定。
12. **目前位置**：與前端報告使用相容口徑，交代剩餘瓶頸和結論邊界。

## 圖表與表達方式

- 使用一張純文字資料流圖，讓 terminal、GitHub 與一般 Markdown viewer 都可讀；不複製 README 的完整 Mermaid 圖。
- 表格只承載約束分類、排序層級、核心實測與負面結果；避免把內部 profile 清單全部外露。
- 每節先寫結論，再補機制或證據。
- 保留前端報告的直接語氣，但避免「首次」「最佳」「完整解決」等無證據宣稱。

## 證據與驗證

完成報告前必須：

1. 從現行 source code 確認 production wrapper、candidate、repair、quality portfolio 和 ranking 的實際呼叫鏈。
2. 從 tests 與封存實驗文件確認每個量化數字的評估集合與 runtime 口徑。
3. 對報告中的模組名稱、環境開關與 default/opt-in 狀態做 literal search。
4. 掃描所有數字與強結論，逐項建立 claim-evidence 對照。
5. 執行 Markdown 基本檢查、`git diff --check`，並確認只新增預定文件及 graphify 增量更新。
6. 修改完成後執行 `graphify update .`，保持 repository knowledge graph 與文件同步。

## 驗收條件

- 文件可以與 `前端報告.md` 並排閱讀，語氣、深度與篇幅大致一致。
- 未閱讀原始碼的外部讀者能回答：後端做什麼、如何保證合法、如何選候選、瓶頸在哪裡。
- 所有量化結果都有可定位的 repository 證據；缺證據的內容不偽裝成結果。
- production default、opt-in、legacy 與未採用方向清楚分隔。
- 不修改 solver、tests、checkpoint 或 evaluator 行為。
