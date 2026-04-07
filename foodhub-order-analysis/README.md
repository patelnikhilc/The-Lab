# FoodHub Order Analysis
### AI Product Case Study -- Market Analytics

**Author:** Nikhil Patel
**Date:** March 2026
**Tools:** Python, Pandas, NumPy, Matplotlib, Seaborn, Jupyter Notebook

---

## The Question

FoodHub is a food aggregator connecting NYC customers with hundreds of restaurants via a single mobile app. Revenue comes from a tiered commission model (25% on orders above $20, 15% on orders above $5). Leadership needed data-driven answers: Where is demand concentrated? Why are deliveries slower on certain days? Which cuisines drive revenue vs. satisfaction? Where are the untapped growth pockets?

---

## Key Metrics

| Metric | Value |
|--------|-------|
| Total Orders Analyzed | 1,898 |
| Net Commission Revenue | $6,166 |
| Avg Customer Rating | 4.34 |
| Weekend Order Share | 71.2% |
| Unrated Orders | 38.8% |
| High-Value Orders (>$20) | 29.2% (drives 60% of revenue) |

---

## Key Findings

- **American and Japanese cuisines** account for 55.6% of all orders and $3,439 in combined commission
- **Weekend demand is 2.5x weekday** -- 71.2% of orders on Sat-Sun
- **Weekday delivery is 26% slower** than weekends (28.3 min vs 22.5 min) -- counter-intuitive and signals a driver allocation gap
- **38.8% of orders are unrated** -- a product design issue, not a data quality issue
- **10.5% of orders exceed 60 minutes** end-to-end fulfillment
- **Spanish, Thai, and Indian cuisines** score highest ratings (4.5-4.8) despite low volume -- under-promoted, not underperforming

---

## Strategic Recommendations

1. **Scale Weekend Operations** -- increase delivery capacity to match 71.2% demand concentration
2. **Fix Weekday Delivery Speed** -- audit routing, add driver incentives, coordinate pickup windows
3. **Close the Ratings Gap** -- post-delivery push notifications with loyalty incentives
4. **Grow High-Satisfaction Cuisines** -- targeted campaigns for Spanish, Thai, Indian restaurants
5. **Accelerate High-Value Orders** -- promote bundles to push orders above the $20 commission threshold
6. **Activate Loyalty Programs** -- vouchers for top customers, featured placements for high-rated partners

---

## Project Structure

```
foodhub-order-analysis/
  README.md                                # This file
  foodhub_analysis.ipynb                   # Complete EDA notebook (70 cells, code + visualizations)
  foodhub_orders.csv                       # Dataset (1,898 orders, 9 features)
  FoodHub_Executive_Presentation.pptx      # Executive summary presentation
```

---

## Technical Approach

1. **Data Inspection** -- shape, types, missing values, statistical summary
2. **Univariate EDA** -- distributions of cost, prep time, delivery time, ratings, cuisine mix
3. **Multivariate EDA** -- cross-variable relationships (day vs delivery, cuisine vs revenue, rating vs restaurant)
4. **Business Questions** -- 17 specific leadership questions answered with statistical analysis
5. **Revenue Modeling** -- tiered commission structure analysis (25% above $20, 15% above $5)

---

## How to Run

```bash
git clone https://github.com/patelnikhilc/The-Lab.git
cd The-Lab/foodhub-order-analysis

pip install pandas numpy matplotlib seaborn jupyter

jupyter notebook foodhub_analysis.ipynb
```

---

## Part of The Lab

This project is part of [The Lab](https://github.com/patelnikhilc/The-Lab) -- a collection of AI Product Case Studies by Nikhil Patel exploring data science, machine learning, and product strategy.

**Portfolio:** [nikhilcpatel.com](https://www.nikhilcpatel.com) | **GitHub:** [@patelnikhilc](https://github.com/patelnikhilc)
