# AllLife Bank Personal Loan Targeting
### AI Product Case Study -- Supervised Learning

**Author:** Nikhil Patel
**Date:** May 2026
**Tools:** Python, Pandas, NumPy, scikit-learn, Matplotlib, Seaborn, Jupyter Notebook

---

## The Question

AllLife Bank's customer base is overwhelmingly liability customers (depositors), with a small slice that also borrows. Their last personal loan campaign converted 9.6% of contacted customers. Marketing's question for the next round was simple: which customers should we contact this time so the conversion rate goes up? That's a classification problem with two production constraints that change everything -- imbalanced classes (9.6% positive) and asymmetric error cost (false positive is cheap, false negative is expensive). Those constraints dictate the metric (recall over accuracy) and the model family (a decision tree, because marketing needs to *see* the rules, not just the score).

---

## Key Metrics

| Metric | Value |
|--------|-------|
| Customers Modeled | 5,000 |
| Base Conversion Rate | 9.6% |
| Final Model Recall | 85.2% |
| Final Model Precision | 93.4% |
| Targeting Lift (HOT+WARM vs base) | 3.1x |
| Premium-Tier Lift (HOT only) | 6.8x |

---

## Key Findings

- **Income is the single strongest signal** -- customers earning $150K+ convert at 49.3%, vs 0% for those under $50K. The buyer distribution is bimodal at ~$130K and ~$180K.
- **CD Account is the killer categorical** -- only 6% of customers hold one, but 46.4% of CD holders converted (4.8x lift over base rate).
- **CCAvg (monthly card spend) adds independent signal** -- buyers average $3.9K/month vs $1.7K for non-buyers, even controlling for income.
- **Education matters in steps, not gradients** -- conversion jumps from 4.4% (Undergrad) to ~13% (Graduate/Advanced), with the cliff between Undergrad and Graduate.
- **Six features predict almost nothing** -- Age, Experience, ZIP Code, Online Banking, Securities Account, and other-bank Credit Card all hover within 1pp of the 9.6% base rate.
- **Pre-pruned tree (depth-2) hits 100% recall but 31% precision** -- catches every buyer by flagging ~40% of the customer base. Useless for a paid campaign, useful for a free mass mailer.
- **Post-pruned tree (cost-complexity α) hits 85.2% recall and 93.4% precision** -- the right tradeoff for a budget-bounded campaign.

---

## The 3-Tier Targeting Playbook

| Tier | Rule | Customers | Conversion | Lift |
|------|------|-----------|------------|------|
| HOT  | Income > $98K AND Education >= Graduate | 7% of base | 67.3% | 6.8x |
| WARM | Income > $98K AND Education = Undergrad | 25% of base | 20.6% | 2.1x |
| SKIP | Income <= $98K | 68% of base | 0.0% | 0x |

**Campaign math:** mass campaign hits all 1,500 test customers and gets 149 buyers (9.9%). Targeted campaign (HOT + WARM) hits 480 customers and gets the same 149 buyers -- 31% conversion, 3.1x the success ratio with zero buyers missed.

---

## Strategic Recommendations

1. **Run HOT tier as a premium campaign** -- 107 customers, 67% conversion. Dedicated outbound rep, white-glove handling, personalized offer.
2. **Treat WARM as a nurture funnel** -- 373 customers, 21% conversion. Email + SMS sequence with rate transparency and a time-limited promo.
3. **Suppress the SKIP tier** -- 1,020 customers, 0% conversion. Stop spending here. Re-tier quarterly as customers' income grows.
4. **Cross-sell loans to CD holders** -- 4.8x lift over base rate. Build an in-app loan offer that triggers at CD origination.
5. **Operationalize the tree as a SQL view** -- refresh weekly. The model becomes a CRM segment marketing owns, not a one-off analysis.
6. **Hold out 10% of HOT as control** -- without it you can't separate model lift from baseline demand. Validate before scaling spend.

---

## Project Structure

```
alllife-bank/
  README.md                           # This file
  Loan_Modelling_Analysis.ipynb       # Full analysis notebook (41 cells, 11 sections)
  Loan_Modelling.csv                  # Dataset (5,000 customers, 14 features) -- not committed if course-licensed
```

---

## Technical Approach

1. **Data Cleaning** -- impute 52 negative-Experience rows to absolute value; engineer ZIP_Code -> 2-digit prefix; drop Experience (0.99 corr with Age).
2. **EDA** -- conversion-by-segment tables (income bands, CD account, education, family), correlation heatmap, and explicit "what doesn't predict" pass to rule out noise features.
3. **Train/Test Split** -- 70/30 with `random_state=1`. Class balance preserved via stratification on the analytical pass; template-default split used for the modeling comparison.
4. **Three-Model Comparison** -- default tree (overfit baseline), pre-pruned via grid search over depth/leaves/split with `class_weight='balanced'`, post-pruned via cost-complexity α-tuning with `class_weight={0:0.15, 1:0.85}`.
5. **Model Selection** -- recall as primary metric, with precision as the constraint that keeps the contact list affordable. Post-pruned wins for the campaign use case.
6. **Segmentation** -- pre-pruned tree's 4 leaves map to a 3-tier marketing playbook (HOT/WARM/SKIP). Decoupling the production classifier (post-pruned) from the segmentation framework (pre-pruned) means each model does one job well.

---

## Live Demo

An interactive web demo of the classifier runs on the portfolio site -- adjust customer profile inputs and watch the tier and propensity score update in real time, or upload a CSV for bulk scoring with downloadable enriched output. Everything runs client-side; no data leaves the browser.

**Demo:** [nikhilcpatel.com/demo-alllife-bank.html](https://www.nikhilcpatel.com/demo-alllife-bank.html)
**Case Study:** [nikhilcpatel.com/project-alllife-bank.html](https://www.nikhilcpatel.com/project-alllife-bank.html)

---

## How to Run

```bash
git clone https://github.com/patelnikhilc/The-Lab.git
cd The-Lab/alllife-bank

pip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter notebook Loan_Modelling_Analysis.ipynb
```

The notebook reads `Loan_Modelling.csv` from the same folder. If the dataset isn't included (course license), source it from the original AllLife Bank Personal Loan Modeling dataset on Kaggle.

---

*Built as part of Great Learning's PGP-AIML program. Methodology and conclusions are mine; the dataset is course-provided.*
