# AI Usage in dsc26-ai-w02

**Repository:** dsc26-ai-w02  
**Track:** AI & Deep Tech (Week 2 Task)  
**Engineer:** Youssef Mohamed Abdelkarim  

---

## 1. Tools Used and Engineering Workflow
I utilized LLM assistance (Gemini / Claude) strictly as an interactive junior pair-programmer under rigorous human supervision. Rather than asking for open-ended code generation, I drove the design by giving explicit architectural constraints derived from the Rawaj Pay problem framing and the test rubric:

| Tool | Functions / Modules | My Directive & Workflow |
|---|---|---|
| Gemini / Claude | `rawaj/audit.py` | Guided the AI to scaffold column-level checks; personally audited and rewrote missingness masks, timestamp validation, and ROC AUC guardrails. |
| Gemini / Claude | `rawaj/metrics.py` | Directed mathematical formulas for confusion matrix and cost curves; rejected naive suggestions that violated tie-breaking and candidate definitions. |

---

## 2. Where the AI Was Wrong or Incomplete (and How I Caught & Fixed It)

During development, I actively caught and corrected two major flaws where the AI produced misleading or non-compliant logic:

### Flaw A: Naive Tie-Breaking and Candidate Space in `best_threshold`
- **What the AI proposed:** The model suggested replicating the snippet from the lecture slides, using `np.unique(scores)` and NumPy’s `np.argmin(costs)`.
- **Why it was wrong:** I immediately caught that this fails the assignment’s operational specification on two counts:
  1. Ascending `np.unique` paired with `argmin` arbitrarily picks the *lowest* threshold on cost ties, whereas Dina's business rule explicitly mandates picking the *highest* threshold to reduce unnecessary operational alarms for her five analysts.
  2. The AI completely omitted `math.inf` ("flag nothing") as an evaluated candidate.
- **My Detection & Fix:** I caught this by cross-checking the edge-case rubric and writing a targeted test case (`test_best_threshold_and_tie_rule`). I discarded the AI's routine and re-engineered the selection logic myself using a deterministic composite sorting key `(cost, -threshold)`, properly mapping `math.inf` to `-float("inf")` so cost ties always resolve to the highest threshold.

### Flaw B: Type Coercion Trap in `audit()` Leakage Detection
- **What the AI proposed:** For check (a), the AI relied solely on `pd.api.types.is_numeric_dtype(col)`.
- **Why it was wrong:** In real-world data and pandas extracts, columns with numbers or booleans mixed with blank whitespace strings are automatically loaded as `object` dtype. Relying on `is_numeric_dtype` causes leak detection to silently bypass genuine numerical leaks.
- **My Detection & Fix:** I flagged this blind spot during code review and enforced a multi-step type-coercion check (`col_valid.astype(float)` and `pd.to_numeric(col_valid, errors='coerce')`) to ensure numeric/boolean features are reliably inspected regardless of initial pandas object casting.

---

## 3. Functions Written Without AI
I wrote the following components entirely from scratch by hand:

1. **`precision_at_k`**:  
   I wrote this independently to guarantee absolute compliance with stable sorting. I used Python’s native Timsort (`sorted(range(n), key=lambda i: s_arr[i], reverse=True)`) to ensure tied scores strictly preserve their original input array order, avoiding non-deterministic behavior inherent to unstable quicksort implementations. I also hand-crafted the boundary controls for $k \le 0$ and $k > n$.

2. **`_validate_inputs`**:  
   I hand-coded the input validation layer across all metrics functions to strictly enforce array dimensionalities, disallow non-binary labels (strictly validating against anything other than 0 and 1), and safeguard against NaN scores and empty containers before any metric evaluation takes place.
