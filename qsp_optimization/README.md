# QSP — classical floors for the "quantum portfolio" slides

Two self-contained demonstrations that the optimization problems on the QSP
slides are handled by classical methods, with the quantum/annealing routes
showing **no advantage**. Each script runs all solvers on the same fixed-seed
instance and writes a JSON results file.

```
pip install numpy pulp ortools dimod dwave-samplers   # Part A
pip install cvxpy pyscipopt                            # Part B (SCIP = exact MIQP)
python bond_knapsack_3way.py        # Part A
python bond_portfolio_miqp.py       # Part B
```

## Part A — `bond_knapsack_3way.py` (40-bond capital-budget knapsack)

Solved by dynamic programming, MILP/CBC, MILP/CP-SAT, and QUBO + simulated
annealing. All exact methods prove the optimum **$12.241 M/yr** in ~1–10 ms.

Additions in this revision:

- **`[0]` greedy-by-ratio baseline** — the docstring claimed greedy "fails";
  now it actually runs and the shortfall is measured: **$12.207 M/yr, 0.28 %
  below optimal** (fast, but no optimality certificate).
- **`[3b]` analytically-derived QUBO penalty** — the penalty weight is not a
  free knob. A sufficient bound is `lambda > max_i income_i` (≈ 2.485 here; cf.
  Lucas 2014), which makes the true optimum the QUBO ground state. The catch:
  at that provably-correct penalty the landscape is stiff and SA **never
  reaches** the optimum in 2000 reads (95 % feasible, 0 % optimal), while the
  soft penalty that *does* reach it leaks 41 % infeasible reads. The native
  MILP faces neither horn of this feasibility-vs-reachability dilemma — the
  difficulty is manufactured by the QUBO encoding.

## Part B — `bond_portfolio_miqp.py` (cardinality-constrained fixed-income MIQP)

The genuinely NP-hard problem (Bienstock 1996). The original logged a convex
relaxation (provable bound) + relax-and-polish heuristic and reported a
~6.4 % "gap" — but that gap is `(relaxation bound − polished)/bound`, and the
relaxation **drops cardinality**, so it overstates suboptimality and the script
never solved the actual MIQP.

Additions in this revision:

- **`solve_exact_miqp()`** — binary-indicator MIQP (cardinality, min-position,
  big-M linking) solved to **proven optimality** by SCIP branch & bound.
- **Exact-MIQP validation** — decomposes the reported gap at the proven optimum:
  `relaxation looseness = (bound − exact)/|exact|` vs.
  `true heuristic gap = (exact − polished)/|exact|`. Result: under mild
  cardinality binding (γ=8) the relaxation is only ~0.5 % loose and the cheap
  heuristic is within ~1 % of optimal — i.e. the headline 6.4 % is mostly
  relaxation looseness. In the hard high-γ regime the quick heuristic degrades
  (~38 %), but exact branch & bound still proves optimality in ~2 s. **Both the
  bound and the remedy are classical.**
- **Exact-MIQP scaling** — measured B&B time vs N. Exact solves run in seconds
  up to a few hundred names; this open-source MINLP setup does not reach the
  N=1500 production scale (which needs Gurobi/MOSEK), which is exactly why the
  desk ships the heuristic at scale.
- **Robustness fix** — the relax-and-polish support QP could be infeasible at
  small N / tight K (returning a garbage objective); it now drops the cosmetic
  min-position floor and retries. The N=1500 path is unchanged.
- **Adaptive lot-rounding buffer** (`relax_and_polish(..., adaptive_buffer=True)`)
  — the original fixed pre-tightening buffer (`db=0.06` on the duration band,
  `cb=0.004` on caps) is wildly oversized: a 0.06-year buffer on a 1.0-year band
  forces a costly interior portfolio whenever the optimum sits on a band edge
  (the high-γ regime). That, not heuristic weakness, is what produced the ~38%
  validation gap. The adaptive path tries the smallest buffer whose *rounded*
  portfolio is still mandate-feasible and only pays for slack rounding needs:

  | N | K | γ | fixed-buffer gap | adaptive-buffer gap |
  |---|---|---|---|---|
  | 160 | 30 | 8 | 0.68% | 0.68% |
  | 200 | 25 | 8 | 1.15% | 0.45% |
  | 200 | 30 | 25 | 38.29% | 6.71% |

  The hard-regime artifact drops 38% → 7%; the residual is genuine lot-rounding
  cost on a tight band, which exact branch & bound on the K-name support clears
  in ~2s. Default stays `adaptive_buffer=False` so the N=1500 headline/figure
  numbers (6.415% gap, frontier) reproduce exactly.

Results land in `knapsack_results.json` and `portfolio_miqp_results.json`.
All numbers above were reproduced in-repo (NumPy 2.4, cvxpy 1.9 + SCIP 6.2,
OR-Tools, dwave-samplers).
