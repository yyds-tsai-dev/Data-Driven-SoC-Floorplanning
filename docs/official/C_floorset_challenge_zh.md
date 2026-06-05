# The FloorSet Challenge: Data-Driven SoC Floorplanning

### Background

Fixed-outline floorplanning is an NP-complete combinatorial optimization problem that has been extensively studied in the literature. In industrial System-on-Chip (SoC) design, this problem becomes significantly more complex due to hard constraints on block shapes and locations. The objective is to arrange blocks (also called modules) on a 2D canvas such that a multi-objective cost function is minimized while all constraints are satisfied.

固定輪廓佈局規劃（Fixed-outline floorplanning）是一個在文獻中被廣泛研究的 NP 完全組合優化問題。在工業級系統單晶片（SoC）設計中，由於對區塊形狀和位置的硬性限制，這個問題變得更加複雜。其目標是在 2D 畫布上排列區塊（也稱為模組），以最小化多目標成本函數，同時滿足所有限制條件。

SoC floorplanning differs from traditional bin packing in several key ways. Blocks may have flexible shapes but must adhere to predefined area budgets. Certain blocks are subject to specific placement constraints. Blocks are interconnected through nets and often require connectivity to external terminals on the canvas boundary, creating spatial dependencies that directly influence total wirelength—a primary optimization objective. The cost function is inherently multi-objective, aiming to simultaneously minimize chip bounding-box area and total wirelength while satisfying all placement constraints.

SoC 佈局規劃在幾個關鍵方面不同於傳統的裝箱問題（bin packing）。區塊可能具有彈性的形狀，但必須遵守預先定義的面積預算。某些區塊受特定放置限制的約束。區塊透過網線（nets）互連，且通常需要連接到畫布邊界上的外部端點，這產生了直接影響總線長（主要優化目標）的空間依賴性。成本函數本質上是多目標的，旨在同時最小化晶片邊界框面積和總線長，同時滿足所有放置限制條件。

---

### Motivation

Reducing the time-to-convergence for high-complexity designs while navigating multi-dimensional constraint spaces has become a critical bottleneck in the physical design back-end flow. The objective of this competition is to identify highly efficient methodologies capable of addressing these industry-scale challenges.

在處理多維度限制空間的同時，縮短高複雜度設計的收斂時間，已成為實體設計後端流程中的關鍵瓶頸。本次競賽的目標是找出能夠解決這些工業級挑戰的高效方法。

The primary goal is to shift the design paradigm from manual iterations—which currently span several days—to automated cycles completed within minutes. We hypothesize that this transition is best achieved through machine learning (ML)-guided optimization. However, the effectiveness of such models depends critically on the availability of high-fidelity, labeled datasets. To address this gap, we provide a comprehensive suite of benchmark datasets derived from realistic industrial scenarios. By releasing these datasets (e.g., FloorSet), we challenge the community to develop ML-driven solutions that scale to industrial requirements.

主要目標是將設計範式從目前需要數天的手動迭代，轉變為幾分鐘內完成的自動化週期。我們假設這種轉變最能透過機器學習（ML）引導的優化來實現。然而，這類模型的有效性關鍵取決於高保真、帶標籤數據集的可用性。為了解決這個缺口，我們提供了一套源自真實工業場景的綜合基準數據集。透過發布這些數據集（例如 FloorSet），我們向社群發起挑戰，開發能擴展至工業需求的機器學習驅動解決方案。

Our underlying hypothesis is that ML guidance enables rapid exploration of the solution space, serving as a foundational component for agentic AI and conversational floorplanning agents. We envision a long-term design environment where architects interact with floorplanning agents via natural language, achieving design iterations with sub-one-minute latency.

我們的基本假設是，機器學習的引導能夠快速探索解決方案空間，作為代理式 AI 和對話式佈局規劃代理程式的基礎組件。我們預見一個長期的設計環境：架構師透過自然語言與佈局規劃代理互動，實現延遲低於一分鐘的設計迭代。

---

### Related Work

Fixed-outline floorplanning under diverse placement constraints represents a mature yet increasingly complex challenge within the physical design landscape. While the community has historically relied on legacy benchmarks such as MCNC and GSRC to validate a wide array of solvers—ranging from classical meta-heuristics to recent hybrid learning frameworks—these datasets lack the rigor required for modern machine learning validation. Specifically, they fail to provide ground-truth optimal solutions, which are standard in the broader machine learning field, and often treat constraints in isolation rather than integrating them into a holistic design problem.

在多樣化的放置限制下的固定輪廓佈局規劃，代表了實體設計領域中一個成熟但日益複雜的挑戰。雖然社群過去依賴 MCNC 和 GSRC 等傳統基準來驗證各種求解器（從經典的元啟發式演算法到最近的混合學習框架），但這些數據集缺乏現代機器學習驗證所需的嚴謹性。具體來說，它們未能提供在更廣泛的機器學習領域中作為標準的「真實最佳解（ground-truth optimal solutions）」，且通常將限制條件孤立處理，而不是將它們整合成一個整體的設計問題。

In contrast, our proposed dataset FloorSet bridges this gap by providing millions of layout solutions where area and wirelength are demonstrably optimal, and all placement constraints are strictly satisfied. By leveraging a reverse-engineering methodology that ensures near-optimality by construction, these benchmarks establish a definitive ground-truth reference that enables precise quantification of the optimality gap for any proposed solver. Recognizing that the full industrial complexity of rectilinear partitions and soft blocks poses a steep entry barrier for neural architectures, we introduce FloorSet-Lite. This curated subset focuses exclusively on hard blocks, providing a streamlined yet high-fidelity environment for evaluating the next generation of machine learning-driven EDA tools.

相比之下，我們提出的數據集 FloorSet 彌補了這個差距，提供了數百萬個佈局解決方案，其中的面積和線長被證明是最佳的，且嚴格滿足所有放置限制。透過利用確保「建構即近似最佳（near-optimality by construction）」的反向工程方法，這些基準建立了一個明確的真實參考，能夠精確量化任何提出的求解器的最佳化差距。體認到直線分割和軟區塊（soft blocks）的完整工業複雜性對神經網絡架構構成了很高的進入門檻，我們引入了 FloorSet-Lite。這個精選的子集專注於硬區塊（hard blocks），為評估下一代機器學習驅動的 EDA 工具提供了一個精簡但高保真的環境。

---

### Proof of Concept

Our preliminary experiments indicate that traditional heuristic methods, even with massive parallelism such as distributed SA, struggle with the FloorSet-Lite dataset. Even for moderate-sized test cases (e.g., 60 partitions), execution times often exceed 10 minutes without reaching optimality, exhibiting at least a 10% gap in wirelength or area.

我們的初步實驗表明，傳統的啟發式方法，即使具有大規模平行處理能力（例如分散式模擬退火 SA），在處理 FloorSet-Lite 數據集時也會遇到困難。即使對於中等規模的測試案例（例如 60 個分區），執行時間通常超過 10 分鐘且無法達到最佳化，在線長或面積上表現出至少 10% 的差距。

Our internal experiments using a diffusion model serves as a counterpoint, achieving high-fidelity solutions in sub-minute intervals. Although the provided dataset is intended for ML training, the contest objective is performance-oriented rather than methodologically restrictive. We invite the application of any algorithmic paradigm, whether purely stochastic, data-driven, or hybrid, that effectively addresses the trade-off between solution quality and runtime.

我們內部使用擴散模型（diffusion model）的實驗則形成對比，在不到一分鐘的時間內實現了高保真的解決方案。雖然提供的數據集旨在用於機器學習訓練，但競賽的目標是效能導向，而非方法論上的限制。我們邀請應用任何演算法範式，無論是純隨機的、數據驅動的還是混合的，只要能有效解決解決方案品質和運行時間之間的權衡即可。

---

### Problem Statement

#### Inputs

* $B = \{b_1, b_2, . . . , b_k\}$ denotes the set of $k$ blocks, where each block $b_i$ is a soft-block, constrained by its predetermined area target from $A = \{a_i | i = 1, 2, . . . , k\}$.
* $T = \{t_1, t_2, . . . , t_r\}$ represents $r$ terminals, which are predefined fixed points on the 2D plane used for external interfacing. Terminal locations are provided in the input and remain fixed throughout the contest.
* Inter-module connectivity: weighted adjacency matrix $W^{(int)} \in \mathbb{R}^{k \times k}$, where $W^{(int)}_{ij}$ is the weight of the net connecting $b_i$ to $b_j$. A weight of zero indicates no connection.
* External connectivity: weighted adjacency matrix $W^{(ext)} \in \mathbb{R}^{k \times r}$, where $W^{(ext)}_{ij}$ is the weight of the net connecting $b_i$ to terminal $t_j$. A weight of zero indicates no connection.

輸入：

* $B = \{b_1, b_2, \dots, b_k\}$ 表示 $k$ 個區塊的集合，其中每個區塊 $b_i$ 是一個軟區塊，受其預定面積目標限制，該面積來自 $A = \{a_i \mid i = 1, 2, \dots, k\}$。
* $T = \{t_1, t_2, \dots, t_r\}$ 表示 $r$ 個端點，這些是 2D 平面上用於外部介面的預定義固定點。端點位置在輸入中提供，並在整個比賽過程中保持固定。
* 模組間連通性：加權鄰接矩陣 $W^{(int)} \in \mathbb{R}^{k \times k}$，其中 $W^{(int)}_{ij}$ 是連接 $b_i$ 和 $b_j$ 的網線權重。權重為零表示沒有連接。
* 外部連通性：加權鄰接矩陣 $W^{(ext)} \in \mathbb{R}^{k \times r}$，其中 $W^{(ext)}_{ij}$ 是連接 $b_i$ 到端點 $t_j$ 的網線權重。權重為零表示沒有連接。
* Soft constraints (for the purpose of this contest, violations incur a penalty but do not disqualify the solution or make it infeasible):
  – Grouping: $B_{P\_grouping}$ defines $P$ groups of blocks that should be physically abutted (i.e., share a common edge segment of non-zero length, with zero gap). A group is satisfied if all its blocks form a single connected component through shared edges.
  – Multi-Instantiation Blocks (MIB): $B_{Q\_mib}$ defines $Q$ groups where blocks should share identical dimensions (width and height). These represent instances of the same master cell that must have uniform shape.
  – Boundary constraints: $B_{boundary}$ specifies blocks that should be placed such that at least one edge touches the bounding-box boundary (for edge constraints) or such that two edges touch the bounding-box corner (for corner constraints). The specific edge or corner requirement is provided per block in the input.
  – Preplaced blocks: $B_{preplaced}$ specifies blocks with predetermined optimal locations $(x_i, y_i)$ and dimensions $(w_i, h_i)$. For these blocks, the area target $a_i$ is ignored—only the specified width and height matter. Deviating from either the location or dimensions incurs a penalty.
  – Fixed-shape blocks: $B_{fixed}$ specifies blocks with predetermined dimensions $(w_i, h_i)$ only; their locations are flexible. For these blocks, the area target $a_i$ is ignored—only the specified width and height matter. Deviating from the specified dimensions incurs a penalty.
* 軟限制（就本次競賽而言，違規會導致懲罰，但不會使解決方案失去資格或變得不可行）：
  – 分組（Grouping）：$B_{P\_grouping}$ 定義了應物理相鄰（即共享非零長度的公共邊緣段，間隙為零）的 $P$ 個區塊組。如果組內所有區塊透過共享邊緣形成單一連通分量，則該組被滿足。
  – 多重實例化區塊（MIB）：$B_{Q\_mib}$ 定義了 $Q$ 個組，其中的區塊應共享相同的尺寸（寬度和高度）。這些代表了必須具有統一形狀的同一主單元（master cell）的實例。
  – 邊界限制（Boundary constraints）：$B_{boundary}$ 指定應放置的區塊，使得至少一條邊接觸邊界框邊界（針對邊界限制），或者兩條邊接觸邊界框角落（針對角落限制）。每個區塊的具體邊緣或角落要求在輸入中提供。
  – 預先放置區塊（Preplaced blocks）：$B_{preplaced}$ 指定具有預定最佳位置 $(x_i, y_i)$ 和尺寸 $(w_i, h_i)$ 的區塊。對於這些區塊，面積目標 $a_i$ 被忽略——只有指定的寬度和高度重要。偏離位置或尺寸都會產生懲罰。
  – 固定形狀區塊（Fixed-shape blocks）：$B_{fixed}$ 僅指定具有預定尺寸 $(w_i, h_i)$ 的區塊；它們的位置是彈性的。對於這些區塊，面積目標 $a_i$ 被忽略——只有指定的寬度和高度重要。偏離指定尺寸會產生懲罰。
* Hard constraints (violations that make the solution infeasible for that test case):
  – Area Targets and Dimensionality: The dimensions of all blocks must strictly adhere to their specified requirements. For preplaced and fixed-shape blocks, the input dimensions $w_i$ and $h_i$ are immutable. For all other blocks (soft blocks), the realized width $w_i$ and height $h_i$ must satisfy the target area $a_i$ within a 1% relative error threshold:
  $| (w_i h_i - a_i) / a_i | \le 0.01$  (1)
  Any solution that deviates from the fixed dimensions (for preplaced and fixed-shape blocks) or exceeds the 1% area tolerance for soft blocks is classified as infeasible.
  – Overlap-Free Constraint: The solution must be strictly overlap-free. For any two distinct blocks $b_i$ and $b_j$ ($i \neq j$), the area of their intersection must be zero:
  $Area(b_i \cap b_j) = 0$
  Any intersection between block geometries, regardless of magnitude, renders the solution infeasible. Note: Blocks may share an edge (touch) without overlapping.
  Infeasible solutions receive a fixed penalty cost of $M = 10$ for that test case, as defined in the Objective Function (Equation 2). This ensures that any feasible solution, regardless of quality, scores better than an infeasible one.
* 硬限制（違規會使該測試案例的解決方案變為不可行）：
  – 面積目標和尺寸：所有區塊的尺寸必須嚴格遵守其指定要求。對於預先放置和固定形狀的區塊，輸入尺寸 $w_i$ 和 $h_i$ 是不可變的。對於所有其他區塊（軟區塊），實現的寬度 $w_i$ 和高度 $h_i$ 必須滿足目標面積 $a_i$，且相對誤差在 1% 容忍度以內：
  $| (w_i h_i - a_i) / a_i | \le 0.01$  (1)
  任何偏離固定尺寸（對於預置和固定形狀區塊）或超過軟區塊 1% 面積容忍度的解決方案都將歸類為不可行。
  – 無重疊限制：解決方案必須嚴格沒有重疊。對於任何兩個不同的區塊 $b_i$ 和 $b_j$ ($i \neq j$)，它們交集的面積必須為零：
  $Area(b_i \cap b_j) = 0$
  區塊幾何形狀之間的任何交集，無論大小，都會使解決方案不可行。注意：區塊可以共享邊界（接觸）而不重疊。
  不可行的解決方案在該測試案例中會受到固定懲罰成本 $M = 10$，如目標函數（公式 2）中所定義。這確保了任何可行的解決方案，無論品質如何，得分都高於不可行的解決方案。

#### Expected Output

* Overlap-free block locations:
  $L = \{(x_i, y_i) | i = 1, 2, . . . , k\}$
  where $(x_i, y_i)$ are the coordinates of the lower-left corner of block $b_i$. The coordinate system has its origin $(0, 0)$ at the lower-left corner of the canvas, with $x$ increasing to the right and $y$ increasing upward. For preplaced blocks, the coordinates $(x_i, y_i)$ are immutable and must match the input specification exactly.
* Block dimensions:
  $D = \{(w_i, h_i) | i = 1, 2, . . . , k\}$
  where $w_i$ is the width (extent in the $x$ direction) and $h_i$ is the height (extent in the $y$ direction) of block $b_i$. Thus, block $b_i$ occupies the rectangular region $[x_i, x_i + w_i] \times [y_i, y_i + h_i]$.
* Output format: Solutions must be submitted in the format specified in the FloorSet repository. See the repository documentation for file format details and submission instructions.

預期輸出：

* 無重疊區塊位置：
  $L = \{(x_i, y_i) \mid i = 1, 2, \dots, k\}$
  其中 $(x_i, y_i)$ 是區塊 $b_i$ 左下角的座標。座標系統原點 $(0, 0)$ 位於畫布左下角，$x$ 向右增加，$y$ 向上增加。對於預先放置區塊，座標 $(x_i, y_i)$ 不可變且必須與輸入規格完全匹配。
* 區塊尺寸：
  $D = \{(w_i, h_i) \mid i = 1, 2, \dots, k\}$
  其中 $w_i$ 是區塊 $b_i$ 的寬度（$x$ 方向範圍），$h_i$ 是高度（$y$ 方向範圍）。因此，區塊 $b_i$ 佔據矩形區域 $[x_i, x_i + w_i] \times [y_i, y_i + h_i]$。
* 輸出格式：解決方案必須以 FloorSet 儲存庫中指定的格式提交。請參閱儲存庫文件以了解檔案格式詳細資訊和提交說明。

---

### Objective Function

We use the following multi-objective cost function:
Cost =
  $(1 + \alpha \cdot (HPWL_{gap} + Area_{gap\_bbox})) \times e^{\beta \cdot Violations_{relative}} \times \max(0.7, RuntimeFactor^\gamma)$, if feasible
  $M$, if infeasible
(2)
where:

* A solution is infeasible if it violates any hard constraint (block overlap or area tolerance violation). Infeasible solutions receive a fixed penalty cost $M = 10$.
* $\alpha = 0.5$ weights the quality metrics (HPWL and bounding-box area gaps).
* $\beta = 2.0$ controls the exponential violation penalty.
* $\gamma = 0.3$ dampens the runtime factor.
* The $\max(0.7, \cdot)$ caps the runtime benefit at 30%.
* $HPWL_{gap}$ is the relative gap between the achieved wirelength and the baseline (optimal) wirelength.
* $Area_{gap\_bbox}$ is the relative gap between the achieved bounding-box area and the baseline (optimal) area.
* $Violations_{relative} \in$ quantifies soft-constraint violations.
* RuntimeFactor = Your Runtime / Median Runtime of All Submissions

我們使用以下多目標成本函數：
Cost =
  $(1 + \alpha \cdot (HPWL_{gap} + Area_{gap\_bbox})) \times e^{\beta \cdot Violations_{relative}} \times \max(0.7, RuntimeFactor^\gamma)$, 若可行
  $M$, 若不可行
(2)
其中：

* 如果解決方案違反任何硬限制（區塊重疊或面積容忍度違規），則為不可行。不可行的解決方案會受到固定懲罰成本 $M = 10$。
* $\alpha = 0.5$ 衡量品質指標（HPWL 和邊界框面積差距）的權重。
* $\beta = 2.0$ 控制指數型違規懲罰。
* $\gamma = 0.3$ 抑制運行時間因子的影響。
* $\max(0.7, \cdot)$ 將運行時間優勢的上限限制為 30%。
* $HPWL_{gap}$ 是實現的線長與基準（最佳）線長之間的相對差距。
* $Area_{gap\_bbox}$ 是實現的邊界框面積與基準（最佳）面積之間的相對差距。
* $Violations_{relative} \in$ 量化軟限制違規。
* RuntimeFactor = 您的運行時間 / 所有提交的中位數運行時間

Interpretation:

* Feasible solutions have costs typically in the range $[0.7, 7.4]$:
  – Best case: perfect quality, 0 violations, fast $\rightarrow \approx 0.7$
  – Worst feasible case: poor quality, 100% soft violations, slow $\rightarrow \approx 7.4 \times 1.5 \approx 11$
* Infeasible solutions receive $M = 10$, which is:
  – Higher than any reasonable feasible solution
  – But not so extreme that a single failure dominates the entire score
* Example impact on Total Score: With 100 test cases and exponential weighting:
  – A participant who fails 1 large test case (e.g., 120 blocks) will be significantly penalized due to the high weight of larger instances.
  – A participant who fails 1 small test case (e.g., 21 blocks) will be penalized less severely.
* Exponential violation penalty: The $e^{\beta V}$ term creates rapidly increasing penalties...
* Capped runtime factor... (Runtime penalties for slowness are uncapped, while the benefit for speed are subject to a fixed upper limit.)
* Example scores: ... Lower cost is better. Hard constraint violations result in a fixed penalty of $M = 10$, ensuring infeasible solutions are always worse than feasible ones while allowing partial credit across the 100 test cases.

解釋：

* 可行解決方案的成本通常在 $[0.7, 7.4]$ 範圍內：
  – 最佳情況：完美的品質，0 違規，快速 $\rightarrow \approx 0.7$
  – 最差的可行情況：品質差，100% 軟違規，緩慢 $\rightarrow \approx 7.4 \times 1.5 \approx 11$
* 不可行的解決方案得到 $M = 10$，這是：
  – 高於任何合理的可行解決方案
  – 但又不會極端到讓單一失敗主導整個分數
* 對總分的影響範例：在 100 個測試案例和指數加權的情況下：
  – 在 1 個大型測試案例（例如 120 個區塊）失敗的參賽者，由於較大實例的高權重，將受到嚴重懲罰。
  – 在 1 個小型測試案例（例如 21 個區塊）失敗的參賽者，受到的懲罰較輕。
* 指數型違規懲罰：$e^{\beta V}$ 項會產生快速增加的懲罰...
* 設有上限的運行時間因子...（對緩慢的運行時間懲罰沒有上限，但對速度的獎勵設有固定的上限。）
* 範例分數：... 成本越低越好。違反硬限制會導致固定懲罰 $M = 10$，確保不可行解總是比可行解差，同時允許在 100 個測試案例中獲得部分分數。

Half-Perimeter Wirelength (HPWL): HPWL sums the half-perimeters of bounding boxes enclosing connected blocks and terminals, weighted by connection weights.
(Mathematical formulas for $HPWL_{int}$ and $HPWL_{ext}$ omitted for brevity, see original PDF)
Gap-based normalization:
$HPWL_{gap} = (HPWL_{int} + HPWL_{ext} - HPWL_{baseline}) / HPWL_{baseline}$
A value of 0 means the solution matches the baseline wirelength exactly. A value of 0.15 means the solution is 15% worse than baseline.

半周長線長 (HPWL)：HPWL 總和包含連接區塊和端點的邊界框的半周長，並以連線權重進行加權。
（因排版省略 $HPWL_{int}$ 與 $HPWL_{ext}$ 具體數學式，詳見原 PDF）
基於差距的標準化：
$HPWL_{gap} = (HPWL_{int} + HPWL_{ext} - HPWL_{baseline}) / HPWL_{baseline}$
值為 0 表示解決方案與基準線長完全相符。值為 0.15 表示解決方案比基準差 15%。

Bounding-Box Area: For floorplan $M = \{(x_i, y_i, w_i, h_i) | i = 1, . . . , k\}$: Lower-left corner of bounding box: $x_{min} = \min_i x_i, y_{min} = \min_i y_i$. Upper-right corner of bounding box: $x_{max} = \max_i(x_i + w_i), y_{max} = \max_i(y_i + h_i)$. Let $Area_{bbox} = (x_{max} - x_{min}) \times (y_{max} - y_{min})$.
Gap-based normalization:
$Area_{gap\_bbox} = (Area_{bbox} - Area_{baseline\_bbox}) / Area_{baseline\_bbox}$
A value of 0 means the solution matches the baseline area exactly. A value of 0.05 means the solution uses 5% more area than baseline.

邊界框面積：對於佈局 $M = \{(x_i, y_i, w_i, h_i) \mid i = 1, \dots, k\}$：邊界框左下角：$x_{min} = \min_i x_i, y_{min} = \min_i y_i$。邊界框右上角：$x_{max} = \max_i(x_i + w_i), y_{max} = \max_i(y_i + h_i)$。令 $Area_{bbox} = (x_{max} - x_{min}) \times (y_{max} - y_{min})$。
基於差距的標準化：
$Area_{gap\_bbox} = (Area_{bbox} - Area_{baseline\_bbox}) / Area_{baseline\_bbox}$
值為 0 表示解決方案與基準面積完全相符。值為 0.05 表示解決方案比基準多使用 5% 的面積。

Violation Cost: Violations are computed differently depending on the constraint type:

* Per-block constraints (Fixed-shape, Preplaced, Boundary): Each block either satisfies (0) or violates (1) its constraint.
* Per-group constraints (Grouping, MIB): Violations are counted per group based on the degree of fragmentation (connected components − 1) or shape inconsistency (distinct shapes − 1).
  $Violations_{relative} = (V_{fixed} + V_{preplaced} + V_{grouping} + V_{boundary} + V_{mib}) / N_{soft}$  (3)

違規成本：違規的計算方式因限制類型而異：

* 單一區塊限制（固定形狀、預置、邊界）：每個區塊滿足（0）或違反（1）其限制。
* 群組限制（分組、MIB）：違規依據群組計算，基於碎片化程度（連通分量 − 1）或形狀不一致性（不同形狀數 − 1）。
  $Violations_{relative} = (V_{fixed} + V_{preplaced} + V_{grouping} + V_{boundary} + V_{mib}) / N_{soft}$  (3)
  （各項細節公式請參見原文）

---

### Dataset

Machine learning has advanced rapidly with scalable transformers leveraging large pre-trained datasets. However, SoC floorplanning lacks such datasets for supervised learning. We provide the FloorSet-Lite dataset (rectangular blocks with a fixed rectangular outline) containing optimal-by-construction layouts:

* Training: 1M samples with optimal solutions for sizes ranging from 21 to 120 blocks. Participants may use this data for training ML models or analyzing problem structure. Available at Hugging Face. Use get_training_dataloader() from iccad2026_evaluate.py for automatic downloading and data access.
* Validation: 100 samples (one per size from 21 to 120 blocks), accessible to contestants for validating solution generalizability. Available at Hugging Face. Use get_validation_dataloader() from iccad2026_evaluate.py for automatic downloading and data access.
* Test: 100 samples (one per size from 21 to 120 blocks), hidden from candidates. Used for final submission evaluation.
* The repository includes PyTorch DataLoaders, score evaluators, infeasibility checks, and plotting utilities. We strongly recommend using the provided evaluator to verify solutions before submission.
* Baseline values $(HPWL_{baseline}, Area_{baseline\_bbox})$ for each test case are provided in the dataset.

機器學習已透過利用大型預訓練數據集的可擴展 Transformer 取得了快速進展。然而，SoC 佈局規劃缺乏此類用於監督式學習的數據集。我們提供 FloorSet-Lite 數據集（具有固定矩形輪廓的矩形區塊），其中包含建構即最佳（optimal-by-construction）的佈局：

* 訓練集（Training）：100 萬個樣本，包含 21 至 120 個區塊大小的最佳解決方案。參賽者可使用此數據訓練 ML 模型或分析問題結構。可在 Hugging Face 取得。使用 iccad2026_evaluate.py 中的 get_training_dataloader() 進行自動下載與數據存取。
* 驗證集（Validation）：100 個樣本（從 21 到 120 個區塊每種大小各一個），參賽者可用於驗證解決方案的泛化能力。可在 Hugging Face 取得。使用 iccad2026_evaluate.py 中的 get_validation_dataloader() 進行自動下載與數據存取。
* 測試集（Test）：100 個樣本（從 21 到 120 個區塊每種大小各一個），對參賽者隱藏。用於最終提交評估。
* 該儲存庫包含 PyTorch DataLoaders、分數評估器、不可行性檢查和繪圖工具。我們強烈建議在提交前使用提供的評估器驗證解決方案。
* 數據集中提供了每個測試案例的基準值 $(HPWL_{baseline}, Area_{baseline\_bbox})$。

---

### Total Score (Cost)

The total score is the weighted average of benchmark costs over 100 test examples (hidden from the candidates), with exponentially increasing weight for larger instances:
Total Score = $\sum_{i=21}^{120} Cost[i] \cdot \frac{e^{n_i}}{\sum_{j=21}^{120} e^{n_j}}$
where:

* $Cost[i]$ is the cost (computed using Equation 2) for test case $i$.
* $n_i$ is the number of blocks in test case $i$ (i.e., $n_i = i$ for this dataset).
* Equivalently, we are multiplying the cost of each testcase $i$ with a normalized weight $\lambda_i = \frac{e^{n_i}}{Z}$, where $Z = \sum_{j=21}^{120} e^{n_j}$ is the normalization constant and $\sum_{i=21}^{120} \lambda_i = 1$:
  Total Score = $\sum_{i=21}^{120} \lambda_i \cdot Cost[i]$
  This exponential weighting scheme ensures that larger, more challenging instances contribute more heavily to the final score.
* Lower Total Score is better. A perfect solution achieving baseline metrics on all test cases with median runtime would have a Total Score close to 0.1.
  Baseline metrics and a live leaderboard are provided on the leaderboard page.

總分是 100 個測試範例（對參賽者隱藏）基準成本的加權平均值，較大實例的權重呈指數增加：
Total Score = $\sum_{i=21}^{120} Cost[i] \cdot \frac{e^{n_i}}{\sum_{j=21}^{120} e^{n_j}}$
其中：

* $Cost[i]$ 是測試案例 $i$ 的成本（使用公式 2 計算）。
* $n_i$ 是測試案例 $i$ 中的區塊數量（即此數據集中 $n_i = i$）。
* 等效地，我們將每個測試案例 $i$ 的成本乘以標準化權重 $\lambda_i = \frac{e^{n_i}}{Z}$，其中 $Z = \sum_{j=21}^{120} e^{n_j}$ 是標準化常數，且 $\sum_{i=21}^{120} \lambda_i = 1$：
  Total Score = $\sum_{i=21}^{120} \lambda_i \cdot Cost[i]$
  這種指數加權方案確保了更大、更具挑戰性的實例對最終分數有更重的貢獻。
* 總分越低越好。一個完美的解決方案，在所有測試案例上達到基準指標且具備中位數的運行時間，其總分將接近 0.1。
  排行榜頁面上提供了基準指標和即時排行榜。

---

### Incentivizing Machine Learning Solutions

In this contest, we incentivize data-driven ML-guided solutions in the following ways:

* Efficiency-Focused Scoring: Runtime is explicitly integrated into the scoring function, and the exponential weighting by block count penalizes methods that scale poorly with problem size.
* Scalability Barriers: Larger instances present significant challenges for classical methods, which struggle to scale efficiently as runtime penalties increase. ML methods that learn from training data can potentially generalize to larger instances more efficiently.

在本次競賽中，我們透過以下方式激勵數據驅動的機器學習引導解決方案：

* 效能導向的評分：運行時間被明確整合到評分函數中，並且根據區塊數量的指數加權會懲罰那些隨問題規模擴展性較差的方法。
* 可擴展性障礙：較大的實例對傳統方法構成了重大挑戰，隨著運行時間懲罰的增加，這些方法難以有效擴展。從訓練數據中學習的 ML 方法有潛力更有效地泛化到較大的實例中。

沒問題！以下為 `C_20260325.pdf` 後續 References（參考文獻）部分的原文與中文翻譯對照。為了維持學術引用的專業性，作者名稱與期刊名稱皆保留原始英文書寫，主要針對論文與文獻標題進行翻譯整理，您同樣可以直接複製使用：

---

### References

 S.N. Adya and I.L. Markov. “Fixed-outline floorplanning: enabling hierarchical design”. In: IEEE Transactions on Very Large Scale Integration (VLSI) Systems 11.6 (2003), pp. 1120–1135.
 S.N. Adya 與 I.L. Markov。〈固定輪廓佈局規劃：實現階層式設計〉。載於：《IEEE Transactions on Very Large Scale Integration (VLSI) Systems》 11.6 (2003)，頁 1120–1135。

 Mohammad Amini et al. “Generalizable Floorplanner through Corner Block List Representation and Hypergraph Embedding”. In: KDD ’22. Association for Computing Machinery, 2022, pp. 2692–2702. isbn: 9781450393850.
3 Submissions found to be reverse-engineering the dataset generator rather than developing genuine algorithmic solutions will be disqualified.
 Mohammad Amini 等人。〈透過角落區塊列表表示法與超圖嵌入的泛化佈局規劃器〉。載於：KDD ’22。美國電腦協會，2022 年，頁 2692–2702。
(註解3：若發現提交的解決方案是針對數據集生成器進行反向工程，而非開發真正的演算法解決方案，將被取消參賽資格。)

 Suchandra Banerjee, Anand Ratna, and Suchismita Roy. “Satisfiability modulo theory based methodology for floorplanning in VLSI circuits”. In: 2016 Sixth International Symposium on Embedded Computing and System Design (ISED). 2016, pp. 91–95.
 Suchandra Banerjee, Anand Ratna, 與 Suchismita Roy。〈基於可滿足性模理論的 VLSI 電路佈局規劃方法〉。載於：2016 第六屆嵌入式運算與系統設計國際研討會 (ISED)。2016 年，頁 91–95。

 Hayward H. Chan, Saurabh N. Adya, and Igor L. Markov. “Are floorplan representations important in digital design?” In: Proceedings of the 2005 International Symposium on Physical Design. ISPD ’05. San Francisco, California, USA: Association for Computing Machinery, 2005, pp. 129–136. isbn: 1595930213.
 Hayward H. Chan, Saurabh N. Adya, 與 Igor L. Markov。〈佈局規劃表示法在數位設計中重要嗎？〉載於：2005 實體設計國際研討會論文集。ISPD ’05。美國加州舊金山：美國電腦協會，2005 年，頁 129–136。

 Guolong Chen et al. “VLSI floorplanning based on Particle Swarm Optimization”. In: 2008 3rd International Conference on Intelligent System and Knowledge Engineering. Vol. 1. 2008, pp. 1020–1025.
 Guolong Chen 等人。〈基於粒子群最佳化的 VLSI 佈局規劃〉。載於：2008 第三屆智慧系統與知識工程國際會議。第 1 卷。2008 年，頁 1020–1025。

 Tung-Chieh Chen and Yao-Wen Chang. “Modern floorplanning based on B/sup */-tree and fast simulated annealing”. In: IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems 25.4 (2006), pp. 637–650.
 Tung-Chieh Chen 與 Yao-Wen Chang。〈基於 B*-tree 與快速模擬退火的現代佈局規劃〉。載於：《IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems》 25.4 (2006)，頁 637–650。

 Chuan-Wen Chiang. “ANT COLONYOPTIMIZATION FORVLSI FLOORPLANNING WITH CLUSTERING CONSTRAINTS”. In: Journal of the Chinese Institute of Industrial Engineers 26.6 (2009), pp. 440–448.
 Chuan-Wen Chiang。〈具備分群限制之 VLSI 佈局規劃的蟻群最佳化〉。載於：《Journal of the Chinese Institute of Industrial Engineers》 26.6 (2009)，頁 440–448。

 “GSRC”. url: http://vlsicad.eecs.umich.edu/BK/GSRCbench.
 〈GSRC 基準測試〉。網址：http://vlsicad.eecs.umich.edu/BK/GSRCbench。

 Zhuolun He et al. “Learn to Floorplan through Acquisition of Effective Local Search Heuristics”. In: 2020 IEEE 38th International Conference on Computer Design (ICCD). 2020, pp. 324–331.
 Zhuolun He 等人。〈透過獲取有效局部搜尋啟發式方法學習佈局規劃〉。載於：2020 IEEE 第 38 屆電腦設計國際會議 (ICCD)。2020 年，頁 324–331。

 Xianlong Hong et al. “Corner block list: an effective and efficient topological representation of non-slicing floorplan”. In: IEEE/ACM International Conference on Computer Aided Design. ICCAD - 2000. IEEE/ACM Digest of Technical Papers (Cat. No.00CH37140). 2000, pp. 8–12.
 Xianlong Hong 等人。〈角落區塊列表：非切片佈局規劃的有效且高效拓撲表示法〉。載於：IEEE/ACM 電腦輔助設計國際會議。ICCAD - 2000。2000 年，頁 8–12。

 Chyi-Shiang Hoo et al. “Variable-Order Ant System for VLSI multiobjective floorplanning”. In: Applied Soft Computing 13.7 (2013), pp. 3285–3297. issn: 1568-4946.
 Chyi-Shiang Hoo 等人。〈用於 VLSI 多目標佈局規劃的可變階數蟻群系統〉。載於：《Applied Soft Computing》 13.7 (2013)，頁 3285–3297。

 R. Jeyarohini, K. R. Aravind Britto, and M. P. Ramkumar. “Optimization and Representation of Non-Slicing VLSI Floorplanning”. In: 2023 4th International Conference on Smart Electronics and Communication (ICOSEC). 2023, pp. 26– 31.
 R. Jeyarohini, K. R. Aravind Britto, 與 M. P. Ramkumar。〈非切片 VLSI 佈局規劃的最佳化與表示法〉。載於：2023 第四屆智慧電子與通訊國際會議 (ICOSEC)。2023 年，頁 26–31。

 Pengli Ji et al. “A Quasi-Newton-based Floorplanner for fixed-outline floorplanning”. In: Computers & Operations Research 129 (2021), p. 105225.
 Pengli Ji 等人。〈基於擬牛頓法之固定輪廓佈局規劃器〉。載於：《Computers & Operations Research》 129 (2021)，頁 105225。

 Andrew B. Kahng. “Classical floorplanning harmful?” In: ISPD ’00 (2000), pp. 207– 213.
 Andrew B. Kahng。〈傳統佈局規劃有害嗎？〉載於：ISPD ’00 (2000)，頁 207–213。

 Andrew B. Kahng. “Machine Learning for CAD/EDA: The Road Ahead”. In: IEEE Design & Test 40.1 (2023), pp. 8–16.
 Andrew B. Kahng。〈CAD/EDA 的機器學習：未來之路〉。載於：《IEEE Design & Test》 40.1 (2023)，頁 8–16。

 Jae-Gon Kim and Yeong-Dae Kim. “A linear programming-based algorithm for floorplanning in VLSI design”. In: IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems 22.5 (2003), pp. 584–592.
 Jae-Gon Kim 與 Yeong-Dae Kim。〈VLSI 設計中基於線性規劃的佈局規劃演算法〉。載於：《IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems》 22.5 (2003)，頁 584–592。

 Jianbang Lai et al. “Module placement with boundary constraints using the sequence-pair representation”. In: Proceedings of the ASP-DAC 2001. Asia and South Pacific Design Automation Conference 2001 (Cat. No.01EX455). 2001, pp. 515–520.
 Jianbang Lai 等人。〈使用序列對表示法的具邊界限制模組放置〉。載於：2001 亞洲與南太平洋設計自動化會議論文集 (ASP-DAC 2001)。2001 年，頁 515–520。

 Ximeng Li et al. “PeF: Poisson’s Equation-Based Large-Scale Fixed-Outline Floorplanning”. In: IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems 42.6 (2023), pp. 2002–2015.
 Ximeng Li 等人。〈PeF：基於卜瓦松方程式的大規模固定輪廓佈局規劃〉。載於：《IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems》 42.6 (2023)，頁 2002–2015。

 Zhu Lichen et al. “An Efficient Simulated Annealing Based VLSI Floorplanning Algorithm for Slicing Structure”. In: 2012 International Conference on Computer Science and Service System. 2012, pp. 326–330.
 Zhu Lichen 等人。〈針對切片結構的高效模擬退火 VLSI 佈局規劃演算法〉。載於：2012 電腦科學與服務系統國際會議。2012 年，頁 326–330。

 Jai-Ming Lin and Yao-Wen Chang. “TCG-S: orthogonal coupling of P*-admissible representations for general floorplans”. In: Proceedings 2002 Design Automation Conference (IEEE Cat. No.02CH37324). 2002, pp. 842–847.
 Jai-Ming Lin 與 Yao-Wen Chang。〈TCG-S：一般佈局規劃 P*-容許表示法的正交耦合〉。載於：2002 設計自動化會議論文集 (DAC)。2002 年，頁 842–847。

 Jai-Ming Lin and Yao-Wen Chang. “TCG: a transitive closure graph-based representation for non-slicing floorplans”. In: Proceedings of the 38th Design Automation Conference (IEEE Cat. No.01CH37232). 2001, pp. 764–769.
 Jai-Ming Lin 與 Yao-Wen Chang。〈TCG：用於非切片佈局規劃之基於遞移閉包圖表示法〉。載於：第 38 屆設計自動化會議論文集 (DAC)。2001 年，頁 764–769。

 Ke Liu et al. “A Hybrid Reinforcement Learning and Genetic Algorithm for VLSI Floorplanning”. In: Proceedings of the 2023 15th International Conference on Machine Learning and Computing. ICMLC ’23. New York, NY, USA: Association for Computing Machinery, 2023, pp. 412–418. isbn: 9781450398411.
 Ke Liu 等人。〈用於 VLSI 佈局規劃之混合強化學習與基因演算法〉。載於：2023 第 15 屆機器學習與運算國際會議論文集。ICMLC ’23。美國紐約：美國電腦協會，2023 年，頁 412–418。

 Yiting Liu et al. “GraphPlanner: Floorplanning with Graph Neural Network”. In: ACM Trans. Des. Autom. Electron. Syst. 28.2 (Dec. 2022).
 Yiting Liu 等人。〈GraphPlanner：使用圖神經網絡進行佈局規劃〉。載於：《ACM Trans. Des. Autom. Electron. Syst.》 28.2 (2022 年 12 月)。

 Chaomin Luo, Miguel F. Anjos, and Anthony Vannelli. “Large-scale fixed-outline floorplanning design using convex optimization techniques”. In: Proceedings of the 2008 Asia and South Pacific Design Automation Conference. ASP-DAC ’08. Seoul, Korea: IEEE Computer Society Press, 2008, pp. 198–203. isbn: 9781424419227.
 Chaomin Luo, Miguel F. Anjos, 與 Anthony Vannelli。〈使用凸面最佳化技術的大規模固定輪廓佈局規劃設計〉。載於：2008 亞洲與南太平洋設計自動化會議論文集。ASP-DAC ’08。韓國首爾：IEEE，2008 年，頁 198–203。

 Yuchun Ma et al. “VLSI floorplanning with boundary constraints based on corner block list”. In: Proceedings of the ASP-DAC 2001. Asia and South Pacific Design Automation Conference 2001 (Cat. No.01EX455). 2001, pp. 509–514.
 Yuchun Ma 等人。〈基於角落區塊列表之具邊界限制的 VLSI 佈局規劃〉。載於：2001 亞洲與南太平洋設計自動化會議論文集 (ASP-DAC 2001)。2001 年，頁 509–514。

 Uday Mallappa et al. “FloorSet - a VLSI Floorplanning Dataset with Design Constraints of Real-World SOCs.” In: Proceedings of the 43rd IEEE/ACM International Conference on Computer-Aided Design. New York, NY, USA: Association for Computing Machinery, 2025. isbn: 9798400710773.
 Uday Mallappa 等人。〈FloorSet - 具備真實世界 SOC 設計限制的 VLSI 佈局規劃數據集〉。載於：第 43 屆 IEEE/ACM 電腦輔助設計國際會議論文集。美國紐約：美國電腦協會，2025 年。

 “MCNC”. url: http://vlsicad.eecs.umich.edu/BK/MCNCbench.
 〈MCNC 基準測試〉。網址：http://vlsicad.eecs.umich.edu/BK/MCNCbench。

 Hesham Mostafa et al. PARSAC: Fast, Human-quality Floorplanning for Modern SoCs with Complex Design Constraints. 2024. arXiv: 2405.05495 [cs.OH]. url: https://arxiv.org/abs/2405.05495.
 Hesham Mostafa 等人。〈PARSAC：針對具複雜設計限制現代 SoC 的快速、人類品質佈局規劃〉。2024 年。arXiv: 2405.05495 [cs.OH]。

 Hiroshi Murata and Ernest S. Kuh. “Sequence-pair based placement method for hard/soft/pre-placed modules”. In: ISPD ’98. Monterey, California, USA: Association for Computing Machinery, 1998, pp. 167–172. isbn: 158113021X.
 Hiroshi Murata 與 Ernest S. Kuh。〈硬體/軟體/預先放置模組之基於序列對放置方法〉。載於：ISPD ’98。美國加州蒙特雷：美國電腦協會，1998 年，頁 167–172。

 David Z. Pan. “Closing the Virtuous Cycle of AI for IC and IC for AI”. The Council on Electronic Design Automation (CEDA), IEEE. 2021. url: https://ieee-ceda.org/presentation/webinar/closing-virtuous-cycle-ai-ic-and-ic-ai.
 David Z. Pan。〈閉合「AI for IC」與「IC for AI」的良性循環〉。IEEE 電子設計自動化委員會 (CEDA)。2021 年。

 Yingxin Pang, Chung-Kuan Cheng, and Takeshi Yoshimura. “An enhanced perturbing algorithm for floorplan design using the O-tree representation”. In: Proceedings of the 2000 International Symposium on Physical Design. ISPD ’00. San Diego, California, USA: Association for Computing Machinery, 2000, pp. 168–173. isbn: 1581131917.
 Yingxin Pang, Chung-Kuan Cheng, 與 Takeshi Yoshimura。〈使用 O-tree 表示法的佈局設計增強擾動演算法〉。載於：2000 實體設計國際研討會論文集。ISPD ’00。美國加州聖地牙哥：美國電腦協會，2000 年，頁 168–173。

 Martin Rapp et al. “MLCAD: A Survey of Research in Machine Learning for CAD Keynote Paper”. In: IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems 41.10 (2022), pp. 3162–3181.
 Martin Rapp 等人。〈MLCAD：CAD 機器學習研究調查主題演講論文〉。載於：《IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems》 41.10 (2022)，頁 3162–3181。

 T. Singha, H.S. Dutta, and M. De. “Optimization of Floor-Planning using Genetic Algorithm”. In: Procedia Technology 4 (2012). 2nd International Conference on Computer, Communication, Control and Information Technology( C3IT-2012) on February 25 - 26, 2012, pp. 825–829. issn: 2212-0173.
 T. Singha, H.S. Dutta, 與 M. De。〈使用基因演算法進行佈局規劃最佳化〉。載於：《Procedia Technology》 4 (2012)。第二屆電腦、通訊、控制與資訊科技國際會議 (C3IT-2012)，2012 年，頁 825–829。

 Jian Sun et al. Floorplanning of VLSI by Mixed-Variable Optimization. 2024. arXiv: 2401.15317 [cs.NE].
 Jian Sun 等人。〈透過混合變數最佳化進行 VLSI 佈局規劃〉。2024 年。arXiv: 2401.15317 [cs.NE]。

 S. Sutanthavibul, E. Shragowitz, and J.B. Rosen. “An analytical approach to floorplan design and optimization”. In: IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems 10.6 (1991), pp. 761–769.
 S. Sutanthavibul, E. Shragowitz, 與 J.B. Rosen。〈佈局設計與最佳化的分析方法〉。載於：《IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems》 10.6 (1991)，頁 761–769。

 Christine L. Valenzuela and Pearl Y. Wang. “A Genetic Algorithm for VLSI Floorplanning”. In: Parallel Problem Solving from Nature PPSN VI. Ed. by Marc Schoenauer et al. Berlin, Heidelberg: Springer Berlin Heidelberg, 2000, pp. 671–680.
 Christine L. Valenzuela 與 Pearl Y. Wang。〈VLSI 佈局規劃的基因演算法〉。載於：《Parallel Problem Solving from Nature PPSN VI》。德國柏林/海德堡：Springer，2000 年，頁 671–680。

 D.F. Wong and C.L. Liu. “A New Algorithm for Floorplan Design”. In: 23rd ACM/IEEE Design Automation Conference. 1986, pp. 101–107.
 D.F. Wong 與 C.L. Liu。〈佈局設計的新演算法〉。載於：第 23 屆 ACM/IEEE 設計自動化會議 (DAC)。1986 年，頁 101–107。

 Qi Xu, Song Chen, and Bin Li. “Combining the ant system algorithm and simulated annealing for 3D/2D fixed-outline floorplanning”. In: Appl. Soft Comput. 40.C (Mar. 2016), pp. 150–160.
 Qi Xu, Song Chen, 與 Bin Li。〈結合蟻群系統演算法與模擬退火的 3D/2D 固定輪廓佈局規劃〉。載於：《Appl. Soft Comput.》 40.C (2016 年 3 月)，頁 150–160。

 Qi Xu et al. “GoodFloorplan: Graph Convolutional Network and Reinforcement Learning-Based Floorplanning”. In: Trans. Comp.-Aided Des. Integ. Cir. Sys. 41.10 (Oct. 2022), pp. 3492–3502.
 Qi Xu 等人。〈GoodFloorplan：基於圖卷積網絡與強化學習的佈局規劃〉。載於：《Trans. Comp.-Aided Des. Integ. Cir. Sys.》 41.10 (2022 年 10 月)，頁 3492–3502。

 Jackey Z. Yan and Chris Chu. “DeFer: Deferred decision making enabled fixed-outline floorplanner”. In: 2008 45th ACM/IEEE Design Automation Conference. 2008, pp. 161–166.
 Jackey Z. Yan 與 Chris Chu。〈DeFer：啟用延遲決策的固定輪廓佈局規劃器〉。載於：2008 第 45 屆 ACM/IEEE 設計自動化會議 (DAC)。2008 年，頁 161–166。

 E.F.Y. Young, C.C.N. Chu, and M.L. Ho. “Placement constraints in floorplan design”. In: IEEE Transactions on Very Large Scale Integration (VLSI) Systems 12.7 (2004), pp. 735–745.
 E.F.Y. Young, C.C.N. Chu, 與 M.L. Ho。〈佈局設計中的放置限制〉。載於：《IEEE Transactions on Very Large Scale Integration (VLSI) Systems》 12.7 (2004)，頁 735–745。

 F.Y. Young and D.F. Wong. “Slicing floorplans with boundary constraint”. In: Proceedings of the ASP-DAC ’99 Asia and South Pacific Design Automation Conference 1999 (Cat. No.99EX198). 1999, 17–20 vol.1.
 F.Y. Young 與 D.F. Wong。〈具邊界限制的切片佈局規劃〉。載於：1999 亞洲與南太平洋設計自動化會議論文集 (ASP-DAC ’99)。1999 年，頁 17–20 第 1 卷。

 F.Y. Young and D.F. Wong. “Slicing floorplans with pre-placed modules”. In: 1998 IEEE/ACM International Conference on Computer-Aided Design. Digest of Technical Papers (IEEE Cat. No.98CB36287). 1998, pp. 252–258.
 F.Y. Young 與 D.F. Wong。〈具預先放置模組的切片佈局規劃〉。載於：1998 IEEE/ACM 電腦輔助設計國際會議 (ICCAD)。1998 年，頁 252–258。

 Yong Zhan, Yan Feng, and S.S. Sapatnekar. “A fixed-die floorplanning algorithm using an analytical approach”. In: Asia and South Pacific Conference on Design Automation, 2006. 2006, 6 pp.-.
 Yong Zhan, Yan Feng, 與 S.S. Sapatnekar。〈使用分析方法的固定晶片佈局規劃演算法〉。載於：2006 亞洲與南太平洋設計自動化會議 (ASP-DAC)。2006 年。

 Hang Zhao et al. “Online 3D Bin Packing with Constrained Deep Reinforcement Learning”. In: Thirty-Fifth AAAI Conference on Artificial Intelligence, AAAI 2021. AAAI Press, 2021, pp. 741–749.
 Hang Zhao 等人。〈結合受限深度強化學習的線上 3D 裝箱問題〉。載於：第三十五屆 AAAI 人工智慧會議 (AAAI 2021)。AAAI Press，2021 年，頁 741–749。

 Hai Zhou and Jia Wang. “ACG-adjacent constraint graph for general floorplans”. In: IEEE International Conference on Computer Design: VLSI in Computers and Processors, 2004. ICCD 2004. Proceedings. 2004, pp. 572–575.
 Hai Zhou 與 Jia Wang。〈ACG：一般佈局規劃的相鄰限制圖〉。載於：IEEE 電腦設計國際會議：電腦與處理器中的 VLSI (ICCD 2004)。2004 年，頁 572–575。

```

```
