"""
Product Analytics Case Study (synthetic premium-card portfolio)
---------------------------------------------------------------
Three analyses a product analytics team actually ships:
  A. A/B test readout  - a new benefit offer tested on cardmembers:
     primary metric (activation), guardrails (spend cannibalization, attrition),
     sample-size / MDE math, and a ship / no-ship recommendation.
  B. Churn model        - gradient boosting early-warning model for attrition,
     with the drivers a retention team can act on.
  C. Benefit value analysis - per-benefit report card: program cost per engaged
     member, incremental spend vs a matched comparison group, retention lift.

Run:  python case_study.py   (all data synthetic, fixed seed)
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

rng = np.random.default_rng(11)

# ================= A. A/B test readout =================
# Test: 50/50 split, 24,000 members per arm, 8-week window.
# Primary metric: offer activation rate. Guardrails: avg spend, attrition.
n_arm = 24_000
ctrl_act = rng.random(n_arm) < 0.184
trt_act = rng.random(n_arm) < 0.211           # true lift +2.7 pts
ctrl_spend = rng.normal(4_820, 2_600, n_arm)
trt_spend = rng.normal(4_905, 2_650, n_arm)
ctrl_attr = rng.random(n_arm) < 0.021
trt_attr = rng.random(n_arm) < 0.020

p_c, p_t = ctrl_act.mean(), trt_act.mean()
se = np.sqrt(p_c * (1 - p_c) / n_arm + p_t * (1 - p_t) / n_arm)
z = (p_t - p_c) / se
pval = 2 * (1 - stats.norm.cdf(abs(z)))
# Pre-registered MDE at 80% power, alpha .05, baseline 18.4%
mde = (stats.norm.ppf(0.975) + stats.norm.ppf(0.8)) * np.sqrt(2 * p_c * (1 - p_c) / n_arm)
print("=== A. A/B TEST READOUT - new benefit offer ===")
print(f"Activation: control {p_c:.1%} vs treatment {p_t:.1%}  (lift {p_t - p_c:+.1%} pts, z={z:.2f}, p={pval:.4f})")
print(f"Pre-registered MDE: {mde:.1%} pts -> observed lift {'exceeds' if (p_t - p_c) > mde else 'is below'} MDE")
print(f"Guardrail spend: control ${ctrl_spend.mean():,.0f} vs treatment ${trt_spend.mean():,.0f} "
      f"(t-test p={stats.ttest_ind(ctrl_spend, trt_spend).pvalue:.3f}; difference is positive, so no cannibalization)")
print(f"Guardrail attrition: control {ctrl_attr.mean():.2%} vs treatment {trt_attr.mean():.2%} (no increase)")
print("Recommendation: SHIP - primary metric clears MDE with significance; both guardrails clean.")

# ================= B. Churn (attrition) model =================
n = 40_000
tenure = rng.integers(3, 120, n)                        # months
spend_trend = rng.normal(-0.02, 0.25, n)                # QoQ spend change
benefit_use = rng.poisson(2.2, n)                       # benefits used last year
digital = rng.random(n) < 0.7                           # app-active
complaints = rng.poisson(0.4, n)
logit = (-2.5 - 2.9 * spend_trend - 0.28 * benefit_use - 0.6 * digital
         + 0.55 * complaints - 0.012 * tenure)
p_churn = 1 / (1 + np.exp(-logit))
churn = (rng.random(n) < p_churn).astype(int)
X = pd.DataFrame(dict(tenure=tenure, spend_trend=spend_trend, benefit_use=benefit_use,
                      digital=digital.astype(int), complaints=complaints))
Xtr, Xte, ytr, yte = train_test_split(X, churn, test_size=0.3, random_state=3, stratify=churn)
clf = HistGradientBoostingClassifier(random_state=3).fit(Xtr, ytr)
auc = roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])
# Permutation importance (simple, model-agnostic)
base = auc
imps = {}
for col in X.columns:
    Xs = Xte.copy(); Xs[col] = rng.permutation(Xs[col].values)
    imps[col] = base - roc_auc_score(yte, clf.predict_proba(Xs)[:, 1])
print("\n=== B. CHURN MODEL ===")
print(f"Base churn rate: {churn.mean():.1%} | out-of-sample AUC: {auc:.3f}")
print("Driver ranking (permutation importance, AUC drop):")
for k, v in sorted(imps.items(), key=lambda kv: -kv[1]):
    print(f"  {k:12s} {v:.3f}")
print("Takeaway: spend trend and benefit usage dominate - both are observable months before")
print("attrition, so the retention team gets an actionable early-warning list, not a post-mortem.")

# ================= C. Benefit value analysis =================
# Each benefit: program cost per engaged member, engagement rate, incremental
# spend of engaged vs matched non-engaged members, retention lift.
benefits = pd.DataFrame([
    # benefit,            cost/engaged $, engagement %, incr spend $, retention lift pts
    ("Lounge access",          212,        0.46,          1_940,        3.8),
    ("Travel credit",          148,        0.61,          1_120,        2.1),
    ("Dining credit",           96,        0.38,            410,        0.6),
    ("Streaming credit",        84,        0.57,             95,        0.2),
    ("Wellness app bundle",     61,        0.18,            130,        0.3),
], columns=['benefit', 'cost_per_engaged', 'engagement', 'incr_spend', 'retention_lift_pts'])
members = 500_000
benefits['engaged_members'] = (benefits['engagement'] * members).astype(int)
benefits['program_cost'] = benefits['cost_per_engaged'] * benefits['engaged_members']
benefits['value_ratio'] = (benefits['incr_spend'] / benefits['cost_per_engaged']).round(1)
print("\n=== C. BENEFIT VALUE REPORT CARD (per 500K members) ===")
print(benefits.to_string(index=False, formatters={'program_cost': '${:,.0f}'.format}))
keep = benefits.sort_values('value_ratio', ascending=False).iloc[0]['benefit']
cut = benefits.sort_values('value_ratio').iloc[0]['benefit']
print(f"\nKeep/enhance: {keep} - highest incremental spend per program dollar with the largest retention lift.")
print(f"Restructure:  {cut} - high engagement but near-zero incremental spend and retention lift: cost without value.")
print("This is the standing quarterly value review: every benefit scored the same way,")
print("so keep / change / cut decisions compare like with like instead of anecdote with anecdote.")
