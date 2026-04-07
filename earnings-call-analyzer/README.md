# AI Earnings Call Analyzer
### AI Product Case Study -- Agentic AI / Financial Intelligence

**Author:** Nikhil Patel
**Date:** April 2026
**Stack:** Python, LangGraph, OpenAI GPT-4o, pdfplumber, pandas, matplotlib

---

## The Problem

Earnings calls contain critical investment signals buried in hours of corporate language. Analysts spend 2-4 hours per call manually extracting metrics, reading tone, cross-referencing guidance, and writing briefs. What if a multi-agent AI system could do it in 60 seconds?

---

## The Solution

A **4-agent LangGraph pipeline** that takes a raw earnings call transcript and produces a structured analyst brief with a clear BUY/HOLD/WATCH signal.

```
Transcript --> [Metrics Agent] --> [Sentiment Agent] --> [Guidance Agent] --> [Report Agent] --> Analyst Brief
```

| Agent | Role | Output |
|-------|------|--------|
| Financial Metrics Extractor | Revenue, EPS, margins, guidance, YoY changes | Structured JSON metrics |
| Executive Sentiment Analyzer | Management tone, red/green flags, hedging patterns | Sentiment scorecard |
| Guidance vs Reality Tracker | Prior guidance accuracy, new targets, credibility score | Accountability report |
| Report Generator | Synthesizes all agent outputs into a one-page brief | Investment brief + signal |

---

## Key Design Decisions

1. **LangGraph over simple chains** -- typed state flows between agents, enabling downstream agents to build on upstream analysis
2. **Specialized agents over mega-prompts** -- 4 focused agents produce more accurate results than one prompt trying to do everything
3. **Sentiment as a first-class signal** -- reads between the lines for hedging language, deflections, and competitive anxiety
4. **JSON-structured outputs** -- enables programmatic dashboards and cross-agent data flow
5. **Guidance credibility scoring** -- tracks management's promise-vs-delivery track record over time

---

## Output

The system generates:
- **Analyst Brief** (markdown) -- executive summary, key metrics table, sentiment read, guidance assessment, bull/bear case, watch list
- **Dashboard** (PNG) -- visual summary with signal badge, segment revenue chart, sentiment flags, guidance scorecard
- **Full State** (JSON) -- complete structured data from all agents for further analysis

---

## Project Structure

```
earnings-call-analyzer/
  README.md                           # This file
  earnings_call_analyzer.ipynb        # Full notebook with 4-agent pipeline
  requirements.txt                    # Python dependencies
  sample_transcripts/                 # Sample earnings call transcripts
  output/                             # Generated reports and dashboards
```

---

## How to Run

```bash
# Clone the repo
git clone https://github.com/patelnikhilc/The-Lab.git
cd The-Lab/earnings-call-analyzer

# Install dependencies
pip install -r requirements.txt

# Set your API key
echo "OPENAI_API_KEY=your-key-here" > .env

# A sample Microsoft Q2 FY2026 transcript is included
# Or add your own to sample_transcripts/ (SEC EDGAR, Seeking Alpha, MotleyFool)

# Run the notebook
jupyter notebook earnings_call_analyzer.ipynb
```

---

## Sample Results

*Run the notebook to generate results for any publicly traded company's earnings call.*

---

## Part of The Lab

This project is part of [The Lab](https://github.com/patelnikhilc/The-Lab) -- a collection of AI Product Case Studies by Nikhil Patel exploring agentic AI, machine learning, and product strategy.

**Portfolio:** [nikhilcpatel.com](https://www.nikhilcpatel.com) | **GitHub:** [@patelnikhilc](https://github.com/patelnikhilc)
