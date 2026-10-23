# Framing: Rawaj Pay Mule Account Detection

## Decision
Rawaj Pay must decide which incoming and outgoing instant bank transfers to flag for immediate manual investigation by the five-person fraud investigation team, versus which transfers to allow automatically without friction. Flagging a legitimate user creates unnecessary customer friction and support costs (false positive cost), while allowing a money mule account to launder illicit funds exposes Rawaj Pay to severe central bank regulatory fines, financial loss, and reputational risk (false negative cost). The automated threshold routes the highest-risk transactions to analysts while ensuring analyst workload remains sustainable.

## Metric
Because Dina's team consists of exactly five analysts who can inspect a fixed volume of flagged cases per day, Rawaj Pay cannot optimize for unconstrained recall. The primary operational metric is **Precision@k** (where $k$ matches the daily review capacity of the five analysts) alongside **Expected Cost** parameterized by estimated costs of false positives ($cost_{FP}$) versus false negatives ($cost_{FN}$). To guarantee high investigative yield without burying analysts in false alarms, the system sets an operating threshold via cost minimization and precision constraints ($\ge 80\%$).

## Baseline
The initial operational baseline is a deterministic heuristic rule-set derived from known AML patterns: flagging transfers exceeding a high monetary cutoff (e.g. > EGP 50,000) or new accounts receiving multiple consecutive transfers followed by immediate total liquidation within 15 minutes. Any machine learning model or learned risk score must demonstrate statistically significant uplift in Precision@k and cost reduction over this business-rule baseline on historical out-of-time splits.

## Classical ML or LLM
For this task, **Classical Tabular Machine Learning (Gradient Boosted Trees such as XGBoost / LightGBM or CatBoost)** is strongly preferred over Large Language Models (LLMs). Tabular transfer transaction data consists of high-frequency numerical aggregates, timestamps, velocity ratios, and network graph features. Classical GBDTs execute inference in sub-millisecond latency (essential for real-time payment processing), offer calibrated probabilities for thresholding, are cost-efficient on free tiers/commodity hardware, and can be inspected via SHAP feature importance for anti-money laundering (AML) regulatory compliance.
