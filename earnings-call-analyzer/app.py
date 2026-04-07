"""
AI Earnings Call Analyzer — Streamlit Web App
Multi-Agent Intelligence System for Financial Analysis
Author: Nikhil Patel
"""

import streamlit as st
import os
import json
import re
import time
from datetime import datetime
from typing import TypedDict

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pdfplumber

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END


# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI Earnings Call Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# Custom CSS — dark theme matching portfolio
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=DM+Serif+Display&display=swap');

    .stApp {
        background-color: #0a1128;
        color: #e0e0e0;
    }

    h1, h2, h3 {
        font-family: 'DM Serif Display', serif !important;
        color: #F5C878 !important;
    }

    .main-title {
        font-family: 'DM Serif Display', serif;
        font-size: 2.4rem;
        color: #F5C878;
        margin-bottom: 0;
        line-height: 1.2;
    }

    .subtitle {
        font-family: 'DM Sans', sans-serif;
        color: #a0a0a0;
        font-size: 1rem;
        margin-top: 4px;
        margin-bottom: 24px;
    }

    .signal-badge {
        display: inline-block;
        padding: 12px 32px;
        border-radius: 8px;
        font-size: 2rem;
        font-weight: 700;
        font-family: 'DM Sans', sans-serif;
        letter-spacing: 2px;
        text-align: center;
    }

    .signal-buy { background: rgba(76, 201, 164, 0.15); color: #4CC9A4; border: 2px solid #4CC9A4; }
    .signal-hold { background: rgba(245, 200, 120, 0.15); color: #F5C878; border: 2px solid #F5C878; }
    .signal-watch { background: rgba(231, 76, 60, 0.15); color: #e74c3c; border: 2px solid #e74c3c; }

    .metric-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(245, 200, 120, 0.15);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }

    .metric-value {
        font-family: 'DM Serif Display', serif;
        font-size: 1.8rem;
        color: #F5C878;
    }

    .metric-label {
        font-family: 'DM Sans', sans-serif;
        color: #a0a0a0;
        font-size: 0.85rem;
        margin-top: 4px;
    }

    .agent-status {
        font-family: 'DM Sans', sans-serif;
        padding: 8px 16px;
        border-radius: 8px;
        margin: 4px 0;
        font-size: 0.9rem;
    }

    .agent-running { background: rgba(245, 200, 120, 0.1); border-left: 3px solid #F5C878; }
    .agent-done { background: rgba(76, 201, 164, 0.1); border-left: 3px solid #4CC9A4; }

    .stSidebar {
        background-color: #0d1635;
    }

    .stSidebar .stMarkdown {
        color: #e0e0e0;
    }

    div[data-testid="stFileUploader"] {
        background: rgba(255,255,255,0.03);
        border: 1px dashed rgba(245, 200, 120, 0.3);
        border-radius: 12px;
        padding: 8px;
    }

    .footer-text {
        font-family: 'DM Sans', sans-serif;
        color: #666;
        font-size: 0.8rem;
        text-align: center;
        margin-top: 48px;
        padding-top: 24px;
        border-top: 1px solid rgba(255,255,255,0.05);
    }

    .footer-text a { color: #F5C878; text-decoration: none; }
    .footer-text a:hover { text-decoration: underline; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# State Schema
# ─────────────────────────────────────────────
class AnalystState(TypedDict):
    transcript: dict
    company_name: str
    quarter: str
    metrics: dict
    sentiment: dict
    guidance: dict
    report: str
    signal: str
    processing_log: list


# ─────────────────────────────────────────────
# Transcript Extraction
# ─────────────────────────────────────────────
def extract_transcript(raw_text: str) -> dict:
    """Structure a raw transcript into sections."""
    sections = {
        'full_text': raw_text,
        'prepared_remarks': '',
        'qa_section': '',
        'word_count': len(raw_text.split()),
        'char_count': len(raw_text)
    }

    qa_markers = [
        r'(?i)question.{0,5}and.{0,5}answer',
        r'(?i)q\s*&\s*a\s+session',
        r'(?i)operator.*first question',
        r'(?i)we.{0,10}now.{0,10}open.{0,20}questions',
        r'(?i)let me now turn.{0,30}questions'
    ]

    split_pos = len(raw_text)
    for marker in qa_markers:
        match = re.search(marker, raw_text)
        if match:
            split_pos = min(split_pos, match.start())

    if split_pos < len(raw_text):
        sections['prepared_remarks'] = raw_text[:split_pos].strip()
        sections['qa_section'] = raw_text[split_pos:].strip()
    else:
        sections['prepared_remarks'] = raw_text
        sections['qa_section'] = ''

    return sections


# ─────────────────────────────────────────────
# Agent Definitions
# ─────────────────────────────────────────────
def get_llm(api_key: str):
    return ChatOpenAI(model="gpt-4o", temperature=0, max_tokens=4096, api_key=api_key)


def financial_metrics_agent(state: AnalystState, llm) -> AnalystState:
    text = (state['transcript']['prepared_remarks'] or state['transcript']['full_text'])[:12000]

    prompt = f"""You are a financial analyst extracting metrics from an earnings call transcript.

Analyze this transcript for {state['company_name']} ({state['quarter']}) and extract ALL mentioned financial metrics.

Return ONLY valid JSON in this exact format (use null for any metrics not mentioned):
{{
    "revenue": {{"value": "$XX.XB", "yoy_change": "+X%", "beat_miss": "beat/miss/inline"}},
    "eps": {{"value": "$X.XX", "yoy_change": "+X%", "beat_miss": "beat/miss/inline"}},
    "gross_margin": {{"value": "XX.X%", "yoy_change": "+X.Xpp"}},
    "operating_margin": {{"value": "XX.X%", "yoy_change": "+X.Xpp"}},
    "free_cash_flow": {{"value": "$XX.XB", "yoy_change": "+X%"}},
    "net_income": {{"value": "$XX.XB", "yoy_change": "+X%"}},
    "forward_guidance": {{
        "next_quarter_revenue": "$XX-XXB",
        "full_year_revenue": "$XXB",
        "commentary": "brief summary of forward outlook"
    }},
    "key_segments": [
        {{"name": "segment name", "revenue": "$XB", "growth": "+X%"}}
    ],
    "notable_metrics": [
        "any other significant numbers mentioned"
    ]
}}

TRANSCRIPT:
{text}"""

    response = llm.invoke([
        SystemMessage(content="You are a precise financial data extraction agent. Return ONLY valid JSON, no markdown."),
        HumanMessage(content=prompt)
    ])

    try:
        raw = response.content.strip()
        raw = re.sub(r'^```json\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw)
        metrics = json.loads(raw)
    except json.JSONDecodeError:
        metrics = {"error": "Failed to parse metrics", "raw_response": response.content[:500]}

    state['metrics'] = metrics
    state['processing_log'].append({
        'agent': 'Financial Metrics Extractor',
        'timestamp': datetime.now().isoformat(),
        'status': 'success' if 'error' not in metrics else 'error'
    })
    return state


def sentiment_analysis_agent(state: AnalystState, llm) -> AnalystState:
    text = state['transcript']['full_text'][:15000]

    prompt = f"""You are an expert at reading between the lines in corporate communications.

Analyze the tone and sentiment of this earnings call transcript for {state['company_name']} ({state['quarter']}).

Return ONLY valid JSON:
{{
    "overall_sentiment": "bullish/neutral/cautious/bearish",
    "confidence_score": 0.0-1.0,
    "ceo_tone": {{
        "descriptor": "confident/measured/defensive/evasive/enthusiastic",
        "evidence": "direct quote or paraphrase demonstrating tone"
    }},
    "cfo_tone": {{
        "descriptor": "confident/measured/defensive/evasive/enthusiastic",
        "evidence": "direct quote or paraphrase demonstrating tone"
    }},
    "red_flags": [
        {{"flag": "description of concerning language pattern", "severity": "low/medium/high", "quote": "relevant excerpt"}}
    ],
    "green_flags": [
        {{"flag": "description of positive signal", "evidence": "relevant excerpt"}}
    ],
    "language_patterns": {{
        "hedging_frequency": "low/medium/high",
        "forward_looking_statements": "few/moderate/many",
        "competitive_mentions": "none/few/several",
        "restructuring_language": false
    }},
    "qa_dynamics": {{
        "analyst_pushback": "none/mild/significant",
        "management_deflections": 0,
        "notable_exchanges": "brief description"
    }}
}}

TRANSCRIPT:
{text}"""

    response = llm.invoke([
        SystemMessage(content="You are a corporate communications analyst specializing in detecting subtle sentiment signals. Return ONLY valid JSON."),
        HumanMessage(content=prompt)
    ])

    try:
        raw = response.content.strip()
        raw = re.sub(r'^```json\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw)
        sentiment = json.loads(raw)
    except json.JSONDecodeError:
        sentiment = {"error": "Failed to parse sentiment", "raw_response": response.content[:500]}

    state['sentiment'] = sentiment
    state['processing_log'].append({
        'agent': 'Executive Sentiment Analyzer',
        'timestamp': datetime.now().isoformat(),
        'status': 'success' if 'error' not in sentiment else 'error'
    })
    return state


def guidance_tracker_agent(state: AnalystState, llm) -> AnalystState:
    text = state['transcript']['full_text'][:12000]
    metrics_context = json.dumps(state['metrics'], indent=2, default=str)

    prompt = f"""You are a financial analyst tracking management credibility by comparing guidance vs actual results.

Based on this {state['company_name']} ({state['quarter']}) earnings call and extracted metrics, analyze guidance accuracy.

Previously extracted metrics:
{metrics_context}

Return ONLY valid JSON:
{{
    "guidance_accuracy": {{
        "overall_score": "A/B/C/D/F",
        "description": "How well did management deliver on prior promises?"
    }},
    "beats": [
        {{"metric": "name", "guided": "value", "actual": "value", "delta": "+X%"}}
    ],
    "misses": [
        {{"metric": "name", "guided": "value", "actual": "value", "delta": "-X%"}}
    ],
    "new_guidance": [
        {{"metric": "name", "target": "value", "timeframe": "next quarter/full year", "confidence": "high/medium/low"}}
    ],
    "management_credibility": {{
        "score": 0.0-1.0,
        "pattern": "consistently beats/mixed/consistently misses/sandbagging",
        "commentary": "brief assessment"
    }},
    "risk_factors": ["specific risks mentioned"],
    "catalysts": ["specific positive drivers mentioned"]
}}

TRANSCRIPT:
{text}"""

    response = llm.invoke([
        SystemMessage(content="You are a guidance tracking specialist. Compare promises vs delivery. Return ONLY valid JSON."),
        HumanMessage(content=prompt)
    ])

    try:
        raw = response.content.strip()
        raw = re.sub(r'^```json\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw)
        guidance = json.loads(raw)
    except json.JSONDecodeError:
        guidance = {"error": "Failed to parse guidance", "raw_response": response.content[:500]}

    state['guidance'] = guidance
    state['processing_log'].append({
        'agent': 'Guidance vs Reality Tracker',
        'timestamp': datetime.now().isoformat(),
        'status': 'success' if 'error' not in guidance else 'error'
    })
    return state


def report_generator_agent(state: AnalystState, llm) -> AnalystState:
    metrics_summary = json.dumps(state['metrics'], indent=2, default=str)
    sentiment_summary = json.dumps(state['sentiment'], indent=2, default=str)
    guidance_summary = json.dumps(state['guidance'], indent=2, default=str)

    prompt = f"""You are a senior equity research analyst writing a one-page investment brief.

Using the following analysis from three specialized agents, generate a concise, actionable analyst report for {state['company_name']} ({state['quarter']}).

FINANCIAL METRICS:
{metrics_summary}

EXECUTIVE SENTIMENT:
{sentiment_summary}

GUIDANCE TRACKING:
{guidance_summary}

Write the report in this EXACT markdown format:

# {state['company_name']} — {state['quarter']} Earnings Analysis

**Signal: [BUY / HOLD / WATCH]**
**Confidence: [High / Medium / Low]**
**Date: {datetime.now().strftime('%B %d, %Y')}**

---

## Executive Summary
[3-4 sentences capturing the quarter's story]

## Key Metrics
| Metric | Value | YoY Change |
|--------|-------|------------|
[most important financial metrics]

## Sentiment Read
[2-3 sentences on management tone and flags]

## Guidance Assessment
[2-3 sentences on guidance accuracy and new targets]

## Bull Case
- [3 bullet points]

## Bear Case
- [3 bullet points]

## What to Watch Next Quarter
- [3 specific metrics or events]

---
*Generated by AI Earnings Call Analyzer — Multi-Agent System by Nikhil Patel*

IMPORTANT: The signal (BUY/HOLD/WATCH) must be justified by the data."""

    response = llm.invoke([
        SystemMessage(content="You are a senior equity research analyst. Write clear, data-driven investment briefs. Be decisive on the signal."),
        HumanMessage(content=prompt)
    ])

    report = response.content.strip()
    signal = 'HOLD'
    signal_match = re.search(r'\*\*Signal:\s*(BUY|HOLD|WATCH)\*\*', report)
    if signal_match:
        signal = signal_match.group(1)

    state['report'] = report
    state['signal'] = signal
    state['processing_log'].append({
        'agent': 'Report Generator',
        'timestamp': datetime.now().isoformat(),
        'status': 'success'
    })
    return state


# ─────────────────────────────────────────────
# Dashboard Generator
# ─────────────────────────────────────────────
def create_dashboard(result: dict):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"{result['company_name']} {result['quarter']} — Earnings Intelligence Dashboard",
        fontsize=16, fontweight='bold', y=0.98, color='white'
    )
    fig.patch.set_facecolor('#0a1128')

    for ax in axes.flat:
        ax.set_facecolor('#0d1635')
        ax.tick_params(colors='#a0a0a0')
        ax.spines['bottom'].set_color('#2a2a2a')
        ax.spines['left'].set_color('#2a2a2a')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Signal Badge
    ax1 = axes[0, 0]
    signal = result.get('signal', 'HOLD')
    signal_colors = {'BUY': '#4CC9A4', 'HOLD': '#F5C878', 'WATCH': '#e74c3c'}
    color = signal_colors.get(signal, '#F5C878')
    ax1.text(0.5, 0.6, signal, fontsize=48, fontweight='bold', color=color,
             ha='center', va='center', transform=ax1.transAxes)
    confidence = result.get('sentiment', {}).get('confidence_score', 0)
    ax1.text(0.5, 0.25, f'Confidence: {confidence:.0%}', fontsize=14,
             color='#a0a0a0', ha='center', va='center', transform=ax1.transAxes)
    ax1.set_title('Signal', color='white', fontsize=12, pad=10)
    ax1.set_xticks([])
    ax1.set_yticks([])

    # Segment Revenue
    ax2 = axes[0, 1]
    segments = result.get('metrics', {}).get('key_segments', [])
    if segments:
        names = [s.get('name', '?')[:20] for s in segments[:6]]
        revenues = []
        for s in segments[:6]:
            rev_str = s.get('revenue', '0')
            rev_num = float(re.sub(r'[^\d.]', '', str(rev_str)) or 0)
            revenues.append(rev_num)
        ax2.barh(names, revenues, color='#00d9ff', alpha=0.8)
        ax2.set_xlabel('Revenue ($B)', color='#a0a0a0')
        ax2.set_title('Segment Revenue', color='white', fontsize=12, pad=10)
        ax2.tick_params(axis='y', labelsize=9, colors='#a0a0a0')
    else:
        ax2.text(0.5, 0.5, 'No segment data', color='#a0a0a0',
                 ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Segment Revenue', color='white', fontsize=12, pad=10)

    # Sentiment Flags
    ax3 = axes[1, 0]
    sentiment = result.get('sentiment', {})
    green_count = len(sentiment.get('green_flags', []))
    red_count = len(sentiment.get('red_flags', []))
    ax3.bar(['Green Flags', 'Red Flags'], [green_count, red_count],
            color=['#4CC9A4', '#e74c3c'], width=0.5)
    ax3.set_title('Sentiment Flags', color='white', fontsize=12, pad=10)
    ax3.tick_params(axis='x', colors='#a0a0a0')
    for i, v in enumerate([green_count, red_count]):
        ax3.text(i, v + 0.1, str(v), ha='center', color='white', fontweight='bold')

    # Guidance Scorecard
    ax4 = axes[1, 1]
    guidance = result.get('guidance', {})
    beats = len(guidance.get('beats', []))
    misses = len(guidance.get('misses', []))
    score = guidance.get('guidance_accuracy', {}).get('overall_score', 'N/A')
    ax4.text(0.5, 0.65, score, fontsize=48, fontweight='bold', color='#F5C878',
             ha='center', va='center', transform=ax4.transAxes)
    ax4.text(0.5, 0.30, f'{beats} beats  |  {misses} misses', fontsize=14,
             color='#a0a0a0', ha='center', va='center', transform=ax4.transAxes)
    credibility = guidance.get('management_credibility', {}).get('score', 0)
    ax4.text(0.5, 0.15, f'Credibility: {credibility:.0%}', fontsize=11,
             color='#a0a0a0', ha='center', va='center', transform=ax4.transAxes)
    ax4.set_title('Guidance Accuracy', color='white', fontsize=12, pad=10)
    ax4.set_xticks([])
    ax4.set_yticks([])

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


# ─────────────────────────────────────────────
# Main App
# ─────────────────────────────────────────────
def main():
    # Header
    st.markdown('<p class="main-title">AI Earnings Call Analyzer</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">4 Agents. 60 Seconds. One Analyst Brief. &nbsp;|&nbsp; Built by <a href="https://www.nikhilcpatel.com" target="_blank" style="color:#F5C878;text-decoration:none;">Nikhil Patel</a></p>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("### Configuration")
        api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
        st.markdown("---")
        company_name = st.text_input("Company Name", value="Microsoft Corp.")
        quarter = st.text_input("Quarter", value="Q2 FY2026")
        st.markdown("---")
        st.markdown("### Upload Transcript")
        uploaded_file = st.file_uploader(
            "PDF or TXT file",
            type=["pdf", "txt"],
            help="Upload an earnings call transcript"
        )
        use_sample = st.checkbox("Use sample Microsoft transcript", value=True)
        st.markdown("---")
        st.markdown("""
        <div style="color:#666;font-size:0.8rem;">
        <strong>How it works:</strong><br>
        4 specialized AI agents analyze the transcript in sequence:<br><br>
        1. Financial Metrics Extractor<br>
        2. Executive Sentiment Analyzer<br>
        3. Guidance vs Reality Tracker<br>
        4. Report Generator<br><br>
        Each agent enriches a shared state before passing to the next.
        </div>
        """, unsafe_allow_html=True)

    # Main content area
    if not api_key:
        # Welcome state
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown('<div class="metric-card"><div class="metric-value">4</div><div class="metric-label">AI Agents</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-card"><div class="metric-value">&lt;60s</div><div class="metric-label">Analysis Time</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="metric-card"><div class="metric-value">3</div><div class="metric-label">Signal Types</div></div>', unsafe_allow_html=True)
        with col4:
            st.markdown('<div class="metric-card"><div class="metric-value">17+</div><div class="metric-label">Metrics Extracted</div></div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### How It Works")
        st.markdown("""
        This system takes a quarterly earnings call transcript and runs it through a **4-agent LangGraph pipeline**
        to auto-generate a structured analyst report with a clear **BUY / HOLD / WATCH** signal.
        """)

        st.code("Transcript → [Metrics Agent] → [Sentiment Agent] → [Guidance Agent] → [Report Agent] → Analyst Brief", language=None)

        st.markdown("""
        | Agent | Role | Output |
        |-------|------|--------|
        | **Financial Metrics Extractor** | Revenue, EPS, margins, guidance | Structured JSON |
        | **Executive Sentiment Analyzer** | Management tone, red/green flags | Sentiment scores |
        | **Guidance vs Reality Tracker** | Prior guidance vs actuals | Credibility report |
        | **Report Generator** | Synthesizes all outputs | Investment brief |
        """)

        st.info("Enter your OpenAI API key in the sidebar to get started. Your key is never stored.")
        return

    # Load transcript
    raw_text = None
    if uploaded_file:
        if uploaded_file.name.endswith('.pdf'):
            with pdfplumber.open(uploaded_file) as pdf:
                raw_text = '\n'.join(page.extract_text() or '' for page in pdf.pages)
        else:
            raw_text = uploaded_file.read().decode('utf-8')
    elif use_sample:
        sample_path = os.path.join(os.path.dirname(__file__), 'sample_transcripts', 'microsoft_q2_fy2026.txt')
        if os.path.exists(sample_path):
            with open(sample_path, 'r') as f:
                raw_text = f.read()
        else:
            st.error("Sample transcript not found. Please upload a file.")
            return

    if not raw_text:
        st.warning("Please upload a transcript or use the sample.")
        return

    transcript = extract_transcript(raw_text)

    # Show transcript info
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Words", f"{transcript['word_count']:,}")
    with col2:
        st.metric("Prepared Remarks", f"{len(transcript['prepared_remarks']):,} chars")
    with col3:
        st.metric("Q&A Section", f"{len(transcript['qa_section']):,} chars")

    # Run button
    st.markdown("---")
    if st.button("Analyze Earnings Call", type="primary", use_container_width=True):
        llm = get_llm(api_key)

        state = {
            'transcript': transcript,
            'company_name': company_name,
            'quarter': quarter,
            'metrics': {},
            'sentiment': {},
            'guidance': {},
            'report': '',
            'signal': '',
            'processing_log': []
        }

        start_time = time.time()

        # Agent 1
        with st.status("Agent 1: Financial Metrics Extractor", expanded=True) as status:
            st.write("Scanning transcript for revenue, EPS, margins, guidance...")
            state = financial_metrics_agent(state, llm)
            revenue = state['metrics'].get('revenue', {}).get('value', 'N/A')
            eps = state['metrics'].get('eps', {}).get('value', 'N/A')
            segments = len(state['metrics'].get('key_segments', []))
            st.write(f"Revenue: {revenue} | EPS: {eps} | {segments} segments extracted")
            status.update(label="Agent 1: Financial Metrics Extractor — Done", state="complete")

        # Agent 2
        with st.status("Agent 2: Executive Sentiment Analyzer", expanded=True) as status:
            st.write("Analyzing management tone, red/green flags, hedging patterns...")
            state = sentiment_analysis_agent(state, llm)
            sentiment_val = state['sentiment'].get('overall_sentiment', 'N/A')
            confidence = state['sentiment'].get('confidence_score', 0)
            red_flags = len(state['sentiment'].get('red_flags', []))
            green_flags = len(state['sentiment'].get('green_flags', []))
            st.write(f"Sentiment: {sentiment_val} | Confidence: {confidence:.0%} | {green_flags} green / {red_flags} red flags")
            status.update(label="Agent 2: Executive Sentiment Analyzer — Done", state="complete")

        # Agent 3
        with st.status("Agent 3: Guidance vs Reality Tracker", expanded=True) as status:
            st.write("Comparing prior guidance against actual results...")
            state = guidance_tracker_agent(state, llm)
            score = state['guidance'].get('guidance_accuracy', {}).get('overall_score', 'N/A')
            beats = len(state['guidance'].get('beats', []))
            misses = len(state['guidance'].get('misses', []))
            st.write(f"Grade: {score} | {beats} beats | {misses} misses")
            status.update(label="Agent 3: Guidance vs Reality Tracker — Done", state="complete")

        # Agent 4
        with st.status("Agent 4: Report Generator", expanded=True) as status:
            st.write("Synthesizing all agent outputs into analyst brief...")
            state = report_generator_agent(state, llm)
            st.write(f"Signal: **{state['signal']}** | Report: {len(state['report']):,} chars")
            status.update(label="Agent 4: Report Generator — Done", state="complete")

        elapsed = time.time() - start_time

        # Store results in session state
        st.session_state['result'] = state
        st.session_state['elapsed'] = elapsed

    # Display results
    if 'result' in st.session_state:
        result = st.session_state['result']
        elapsed = st.session_state['elapsed']

        st.markdown("---")

        # Signal Badge
        signal = result['signal']
        signal_class = f"signal-{signal.lower()}"
        st.markdown(f'<div style="text-align:center;margin:24px 0;"><span class="signal-badge {signal_class}">{signal}</span></div>', unsafe_allow_html=True)

        # Quick metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            rev = result['metrics'].get('revenue', {}).get('value', 'N/A')
            rev_chg = result['metrics'].get('revenue', {}).get('yoy_change', '')
            st.markdown(f'<div class="metric-card"><div class="metric-value">{rev}</div><div class="metric-label">Revenue ({rev_chg} YoY)</div></div>', unsafe_allow_html=True)
        with col2:
            eps_val = result['metrics'].get('eps', {}).get('value', 'N/A')
            eps_chg = result['metrics'].get('eps', {}).get('yoy_change', '')
            st.markdown(f'<div class="metric-card"><div class="metric-value">{eps_val}</div><div class="metric-label">EPS ({eps_chg} YoY)</div></div>', unsafe_allow_html=True)
        with col3:
            om = result['metrics'].get('operating_margin', {}).get('value', 'N/A')
            st.markdown(f'<div class="metric-card"><div class="metric-value">{om}</div><div class="metric-label">Operating Margin</div></div>', unsafe_allow_html=True)
        with col4:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{elapsed:.0f}s</div><div class="metric-label">Analysis Time</div></div>', unsafe_allow_html=True)

        st.markdown("---")

        # Tabs for detailed results
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Analyst Report", "Dashboard", "Metrics", "Sentiment", "Guidance"])

        with tab1:
            st.markdown(result['report'])

        with tab2:
            fig = create_dashboard(result)
            st.pyplot(fig)
            plt.close(fig)

        with tab3:
            st.subheader("Extracted Financial Metrics")
            st.json(result['metrics'])

        with tab4:
            st.subheader("Executive Sentiment Analysis")
            sentiment = result['sentiment']

            if 'error' not in sentiment:
                scol1, scol2 = st.columns(2)
                with scol1:
                    st.metric("Overall Sentiment", sentiment.get('overall_sentiment', 'N/A'))
                    st.metric("Confidence Score", f"{sentiment.get('confidence_score', 0):.0%}")
                with scol2:
                    ceo = sentiment.get('ceo_tone', {})
                    cfo = sentiment.get('cfo_tone', {})
                    st.metric("CEO Tone", ceo.get('descriptor', 'N/A'))
                    st.metric("CFO Tone", cfo.get('descriptor', 'N/A'))

                st.markdown("#### Green Flags")
                for flag in sentiment.get('green_flags', []):
                    st.success(f"**{flag.get('flag', '')}**")

                st.markdown("#### Red Flags")
                for flag in sentiment.get('red_flags', []):
                    severity = flag.get('severity', 'low')
                    if severity == 'high':
                        st.error(f"**{flag.get('flag', '')}** (Severity: {severity})")
                    elif severity == 'medium':
                        st.warning(f"**{flag.get('flag', '')}** (Severity: {severity})")
                    else:
                        st.info(f"**{flag.get('flag', '')}** (Severity: {severity})")
            else:
                st.json(sentiment)

        with tab5:
            st.subheader("Guidance vs Reality Tracking")
            guidance = result['guidance']

            if 'error' not in guidance:
                gcol1, gcol2 = st.columns(2)
                with gcol1:
                    acc = guidance.get('guidance_accuracy', {})
                    st.metric("Overall Grade", acc.get('overall_score', 'N/A'))
                    st.write(acc.get('description', ''))
                with gcol2:
                    cred = guidance.get('management_credibility', {})
                    st.metric("Credibility Score", f"{cred.get('score', 0):.0%}")
                    st.write(f"Pattern: {cred.get('pattern', 'N/A')}")

                if guidance.get('beats'):
                    st.markdown("#### Beats")
                    st.dataframe(pd.DataFrame(guidance['beats']), use_container_width=True)

                if guidance.get('misses'):
                    st.markdown("#### Misses")
                    st.dataframe(pd.DataFrame(guidance['misses']), use_container_width=True)

                if guidance.get('risk_factors'):
                    st.markdown("#### Risk Factors")
                    for risk in guidance['risk_factors']:
                        st.warning(risk)

                if guidance.get('catalysts'):
                    st.markdown("#### Catalysts")
                    for cat in guidance['catalysts']:
                        st.success(cat)
            else:
                st.json(guidance)

        # Download options
        st.markdown("---")
        st.markdown("### Download Results")
        dcol1, dcol2, dcol3 = st.columns(3)
        with dcol1:
            st.download_button(
                "Download Report (MD)",
                result['report'],
                file_name=f"{result['company_name'].replace(' ', '_')}_{result['quarter'].replace(' ', '_')}_report.md",
                mime="text/markdown"
            )
        with dcol2:
            state_json = json.dumps({
                'company_name': result['company_name'],
                'quarter': result['quarter'],
                'signal': result['signal'],
                'metrics': result['metrics'],
                'sentiment': result['sentiment'],
                'guidance': result['guidance'],
                'processing_log': result['processing_log']
            }, indent=2, default=str)
            st.download_button(
                "Download Full Data (JSON)",
                state_json,
                file_name=f"{result['company_name'].replace(' ', '_')}_{result['quarter'].replace(' ', '_')}_state.json",
                mime="application/json"
            )

    # Footer
    st.markdown("""
    <div class="footer-text">
        Built by <a href="https://www.nikhilcpatel.com" target="_blank">Nikhil Patel</a> &nbsp;|&nbsp;
        <a href="https://github.com/patelnikhilc/The-Lab" target="_blank">GitHub</a> &nbsp;|&nbsp;
        Part of <a href="https://www.nikhilcpatel.com/#lab" target="_blank">The Lab</a> — AI Product Case Studies
        <br><br>
        <em>Disclaimer: This tool is for educational purposes only. Not financial advice. Always do your own research.</em>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
