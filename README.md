# Project Work 1: Can We Predict When Greedy Will Fail? (Task T16)

**Master 1 Data Science & Artificial Intelligence (M1 DSAI)**  
**UFAZ — Advanced Data Structures and Algorithms (Fall 2026)**  
**Pair:** Maruf Abdullayev – Muslum Ibadov  

---

## 1. Research Question & Purpose
> **How reliably can a lightweight, fast test ($O(|V| + |E|)$), run before solving, predict whether a greedy algorithm will lose to an exact method on Graph Colouring and on Maximum Weight Independent Set (MWIS) in real-life network topologies?**

Greedy algorithms make myopic, locally optimal choices at each step. While fast, they risk falling into structural traps on certain graph topologies. Exact methods (branch-and-bound, backtracking) guarantee optimal solutions but have exponential worst-case runtime. The goal of this project is to develop and evaluate cheap structural pre-checks that can determine in advance whether an instance requires an exact solver or can be safely delegated to a greedy heuristic.

---

## 2. Real-Life Graph Dataset
* **Source:** *Caenorhabditis elegans* (C. elegans) Neural Network from the Quantware CNRS/FETNADINE Repository ([V. Kandiah & D. L. Shepelyansky, 2014](https://www.quantware.ups-tlse.fr/QWLIB/wormgooglematrix/)).
* **Biological Context:** The complete somatic nervous system of the *C. elegans* hermaphrodite worm.
* **Size:** 
  * Vertices (Neurons): $N = 279$
  * Edges (Synaptic Connections): $|E| = 1{,}961$
  * Average Degree: $\langle k \rangle \approx 14.06$
  * Graph Density: $\approx 0.0506$
* **Workload Extraction (Subgraphs):**
  Because exact solvers for NP-hard problems scale exponentially, evaluation is performed on induced subgraphs ($N \in [15, 25]$) extracted via **Breadth-First Search (BFS) / Snowball crawling**. This preserves the real-world clustering, degree skewness, and local community traps of the biological network while keeping exact verification tractable within strict time bounds.

---

## 3. Formal Hypotheses
* **$H_1$ (Colouring Traps):** Greedy colouring heuristics fail predominantly on subgraphs with high degree variance ($CV_d > 0.6$) and high local clustering ($\bar{C} > 0.25$), where hub-and-clique intersections create color assignment bottlenecks.
* **$H_2$ (MWIS Collateral Damage):** Ratio-based greedy MWIS fails when heavy nodes are surrounded by mutually independent low-degree neighbors whose aggregate value exceeds the hub.
* **$H_3$ (Predictability Without Solving):** A feature-based pre-test running in $O(|V| + |E|)$ can predict greedy failure with an accuracy and ROC-AUC significantly better than random guessing ($\text{ROC-AUC} \ge 0.75$) on unseen test subgraphs.

---

## 4. Problem Formulations & Method Selection

### 4.1. Graph Colouring (Vertex Colouring)
* **Objective:** Assign an integer color to each vertex such that no two adjacent vertices share the same color, minimizing the total number of colors used ($\chi(G)$).
* **Greedy Rule 1 (Largest First / Welsh-Powell):** Sort vertices statically by descending degree; assign the lowest allowable color.
* **Greedy Rule 2 (Saturation Degree / DSATUR):** Dynamically pick the uncolored vertex with the highest number of distinct colors in its neighborhood; break ties using vertex degree.
* **Exact Method (Oracle):** Recursive Backtracking with Forward Pruning and Symmetry Breaking. Uses the maximum clique size as an initial lower bound and greedy coloring as an initial upper bound.

### 4.2. Maximum Weight Independent Set (MWIS)
* **Objective:** Find a subset of vertices $S \subseteq V$ such that no two vertices in $S$ share an edge, maximizing $\sum_{v \in S} w(v)$.
* **Vertex Weights Derivation:** To reflect real structural influence, vertex weights are derived deterministically from the graph topology:
  $$w(v) = \lfloor 10 \cdot C(v) \rfloor + \text{deg}(v) + 1$$
  *(where $C(v)$ is the local clustering coefficient of node $v$)*.
* **Greedy Rule 1 (Weight-First):** Iteratively select the heaviest available vertex and remove its neighbors.
* **Greedy Rule 2 (Weight-to-Degree Ratio):** Iteratively select the vertex maximizing $\frac{w(v)}{\text{deg}_{current}(v) + 1}$ and remove its neighbors.
* **Exact Method (Oracle):** Branch-and-Bound with upper-bound pruning based on the remaining candidate weight sum and vertex domination rules.

---

## 5. Specification of Task Decisions (T16)

1. **Greedy Rules Selection:**
   * *Colouring:* Welsh-Powell (static degree-based) vs. DSATUR (dynamic constraint-based).
   * *MWIS:* Pure Weight-First (points-greedy) vs. Weight-to-Degree Ratio (collateral-damage-aware).
2. **Exact Solvers & Size Limits:**
   * Backtracking (Colouring) and Branch-and-Bound (MWIS).
   * Strict practical timeout: **2.0 seconds per instance**.
   * Operating size threshold: Subgraphs of size $N = 15$ to $25$ finish consistently within this time budget.
3. **Fast Test Architecture & Non-Leakage Tuning:**
   * **Extracted Features ($O(|V| + |E|)$):**
     1. Node count ($n$) and Edge count ($m$)
     2. Density: $2m / (n(n-1))$
     3. Degree skewness / Coefficient of Variation: $\sigma(d) / \mu(d)$
     4. Average Local Clustering: $\bar{C}$
     5. Weight-to-Degree correlation ($r_{w, d}$)
   * **Classification Model:** A lightweight regularized Logistic Regression classifier.
   * **Tuning Protocol:** 200 induced subgraphs are generated. The dataset is partitioned into a **60% Training set** (used to fit feature scalers and model parameters) and a **40% Frozen Test set** (evaluated once without retraining).
4. **Failure Definitions:**
   * *Colouring Failure:* Any suboptimality: $\text{Colors}_{greedy} > \text{Colors}_{exact}$.
   * *MWIS Failure:* Relative degradation beyond threshold $\tau$:
     $$\Delta_{rel} = \frac{W_{exact} - W_{greedy}}{W_{exact}} > 0.05 \quad (\text{tolerance: } 5\%)$$
5. **Vertex Weights:** Computed directly from intrinsic structural metrics ($w(v) = \lfloor 10 \cdot C(v) \rfloor + \text{deg}(v) + 1$) rather than uniform random noise.

---

## 6. Abstract Data Types (ADT) & Complexity Guarantees

In strict accordance with Requirement A3, **all core data structures and search algorithms are custom implementations** without third-party graph solvers.

```python
class GraphADT:
    """Undirected weighted graph implemented via adjacency lists."""
    def add_vertex(self, u: int, weight: float = 1.0) -> None:
        """Time: O(1) amortized"""
        ...

    def add_edge(self, u: int, v: int) -> None:
        """Time: O(1) amortized"""
        ...

    def get_neighbors(self, u: int) -> set[int]:
        """Time: O(1)"""
        ...

    def degree(self, u: int) -> int:
        """Time: O(1)"""
        ...

    def get_weight(self, u: int) -> float:
        """Time: O(1)"""
        ...
```

---

## 7. Project Structure

├── data/
│   ├── ssynlinks.txt              # Primary dataset: chemical synapses (279 nodes, 1961 edges)
│   ├── ssumlinks.txt              # Full network: synapses + gap junctions combined
│   └── nodenames.txt              # Neuron ID to biological name dictionary
├── src/
│   ├── graph.py                   # Custom Graph ADT
│   ├── sampler.py                 # BFS connected subgraph generator
│   ├── greedy_solvers.py          # Welsh-Powell, DSATUR, Weight-First, Ratio Greedy
│   ├── exact_solvers.py           # Exact Backtracking and Branch-and-Bound
│   └── predictor.py               # Feature extractor and fast decision model
├── tests/
│   └── test_oracles.py            # Hand-verified small graph test suite
└── README.md