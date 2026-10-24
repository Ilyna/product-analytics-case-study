# Product Analytics Case Study - Premium Card Portfolio

Three analyses a product analytics team actually ships, run end to end on synthetic data: an A/B test readout with a ship decision, a churn early-warning model, and a benefit value analysis of the kind used to run a quarterly value review for a premium card portfolio.

## Problem
Product leadership for a premium (high annual fee) card portfolio needs three recurring answers:
1. Did the new benefit offer work - and is it safe to ship to everyone?
2. Which members are about to attrite, early enough to do something about it?
3. Which benefits in the package earn their cost, and which are cost without value?

## Data
Fully synthetic, generated in-script with fixed seeds. A/B test: 48,000 members in a 50/50 split over an 8-week window. Churn model: 40,000 members with tenure, spend trend, benefit usage, digital engagement, and complaint history. Benefit analysis: portfolio of 500,000 members across five benefits. No real customer data.

## Approach and results

### A. A/B test readout
- Pre-registered primary metric (offer activation) and minimum detectable effect (1.0 pt at 80% power) set *before* looking at results.
- Activation: **18.6% control vs 21.0% treatment (+2.4 pts, p < 0.001)** - the lift clears the MDE, not just the significance bar.
- Guardrails checked before recommending: average spend did not drop (it rose slightly), and attrition did not increase (2.0% vs 1.9%).
- **Recommendation: SHIP.** A significant primary metric with clean guardrails is the whole decision rule - stated in advance, not negotiated after.

### B. Churn early-warning model
- Gradient boosting classifier on behavioral features; base churn rate 3.0%, **out-of-sample AUC 0.734**.
- Permutation importance ranks **spend trend** far ahead of everything else, followed by benefit usage, tenure, and complaints.
- Both top drivers are observable months before attrition - which is what makes the model operationally useful: the retention team gets a ranked save list while intervention can still work.

### C. Benefit value analysis
Every benefit scored the same way - program cost per engaged member, engagement rate, incremental spend of engaged vs matched non-engaged members, retention lift:
- **Lounge access** is the standout: $212 cost per engaged member against $1,940 incremental spend (9.2x value ratio) and +3.8 pts retention lift.
- **Streaming credit** is the problem: 57% engagement but only $95 incremental spend (1.1x) and +0.2 pts retention - members use it and would behave identically without it.
- The deliverable is a standing quarterly report card, so keep / change / cut decisions compare benefits like with like instead of anecdote against anecdote.

## Business takeaway
The pattern across all three: define the decision rule before the analysis (MDE and guardrails, actionable lead time, a single value scorecard), then let the numbers make the call. That is what turns analytics from ad hoc reporting into a product stakeholders can run decisions on.

## Run it
```bash
pip install -r requirements.txt
python case_study.py
```

## Extensions I would add in production
- Sequential monitoring / always-valid p-values for the experimentation platform; CUPED variance reduction.
- Survival analysis for churn timing (not just probability) and uplift modeling to target saves where intervention changes the outcome.
- Matched-cohort construction with propensity diagnostics for the benefit value analysis, refreshed quarterly.
