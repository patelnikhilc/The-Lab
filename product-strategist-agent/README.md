# AI Product Strategist Agent
### AI Product Case Study — Agentic AI / Product Strategy

**Author:** Nikhil Patel
**Date:** April 2026
**Stack:** Python, LangGraph, OpenAI GPT-4o, pandas, matplotlib, Streamlit

---

## The Problem

Product managers spend weeks on market research, competitive analysis, and PRD writing. The data exists, but synthesizing it into actionable strategy takes too long. What if 5 AI agents could do it in 90 seconds?

---

## The Solution

A **5-agent LangGraph pipeline** that takes a company/product name and produces a complete Product Requirements Document with competitive analysis, gap identification, RICE-scored feature backlog, and strategic roadmap.

```
Product → [Market Research] → [Competitor Intel] → [Gap Analysis] → [RICE Prioritization] → [PRD Generator] → PRD
```

| Agent | Role | Output |
|-------|------|--------|
| Market Research Analyst | TAM, trends, customer segments, market forces | Market landscape JSON |
| Competitor Intelligence | Features, pricing, positioning, threat levels | Competitor matrix JSON |
| Product Gap Analyzer | Unmet needs, whitespace, positioning advice | Opportunity map JSON |
| Feature Prioritizer (RICE) | Reach × Impact × Confidence / Effort scoring | Ranked backlog JSON |
| PRD Generator | Synthesizes all agent outputs into a professional PRD | Complete PRD (markdown) |

---

## Key Design Decisions

1. **RICE framework for credibility** — industry-standard prioritization that any PM or hiring manager recognizes
2. **5 specialized agents** — each with domain-specific system prompts optimized for their analysis type
3. **Cumulative state enrichment** — gap analyzer works because it has both market AND competitive context
4. **JSON intermediates** — structured outputs enable dashboards and cross-agent data flow
5. **PRD as final output** — the deliverable every product org uses, not just raw analysis

---

## Project Structure

```
product-strategist-agent/
├── README.md
├── product_strategist.ipynb     # Full notebook with 5-agent pipeline
├── app.py                       # Streamlit web app
├── requirements.txt
├── .streamlit/config.toml
├── sample_outputs/              # Generated PRDs and dashboards
└── .env.example
```

---

## How to Run

```bash
git clone https://github.com/patelnikhilc/The-Lab.git
cd The-Lab/product-strategist-agent

pip install -r requirements.txt

echo "OPENAI_API_KEY=your-key-here" > .env

# Run the notebook
jupyter notebook product_strategist.ipynb

# Or run the web app
streamlit run app.py
```

---

## Part of The Lab

This project is part of [The Lab](https://github.com/patelnikhilc/The-Lab) — a collection of AI Product Case Studies by Nikhil Patel exploring agentic AI, machine learning, and product strategy.

**Portfolio:** [nikhilcpatel.com](https://www.nikhilcpatel.com) | **GitHub:** [@patelnikhilc](https://github.com/patelnikhilc)
