"""
AI Product Strategist Agent — Streamlit Web App
Multi-Agent System for Product Analysis & PRD Generation
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

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END


# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI Product Strategist Agent",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=DM+Serif+Display&display=swap');

    .stApp { background-color: #0a1128; color: #e0e0e0; }
    h1, h2, h3 { font-family: 'DM Serif Display', serif !important; color: #F5C878 !important; }

    .main-title {
        font-family: 'DM Serif Display', serif;
        font-size: 2.4rem; color: #F5C878;
        margin-bottom: 0; line-height: 1.2;
    }
    .subtitle {
        font-family: 'DM Sans', sans-serif;
        color: #a0a0a0; font-size: 1rem;
        margin-top: 4px; margin-bottom: 24px;
    }
    .metric-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(245, 200, 120, 0.15);
        border-radius: 12px; padding: 20px; text-align: center;
    }
    .metric-value {
        font-family: 'DM Serif Display', serif;
        font-size: 1.8rem; color: #F5C878;
    }
    .metric-label {
        font-family: 'DM Sans', sans-serif;
        color: #a0a0a0; font-size: 0.85rem; margin-top: 4px;
    }
    .stSidebar { background-color: #0d1635; }
    .footer-text {
        font-family: 'DM Sans', sans-serif; color: #666;
        font-size: 0.8rem; text-align: center;
        margin-top: 48px; padding-top: 24px;
        border-top: 1px solid rgba(255,255,255,0.05);
    }
    .footer-text a { color: #F5C878; text-decoration: none; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# State Schema
# ─────────────────────────────────────────────
class StrategistState(TypedDict):
    product_input: dict
    market_research: dict
    competitors: dict
    gap_analysis: dict
    feature_backlog: dict
    prd: str
    processing_log: list


# ─────────────────────────────────────────────
# Agent Definitions
# ─────────────────────────────────────────────
def get_llm(api_key: str):
    return ChatOpenAI(model="gpt-4o", temperature=0, max_tokens=4096, api_key=api_key)


def market_research_agent(state, llm):
    product = state['product_input']
    prompt = f"""You are a senior market research analyst. Analyze the market landscape for:
Company: {product['company_name']}
Product: {product['product_name']}
Description: {product['description']}
Target Market: {product['target_market']}
Focus: {product.get('analysis_focus', 'General')}

Return ONLY valid JSON:
{{
    "market_overview": {{
        "category": "primary category",
        "subcategories": ["list"],
        "tam": {{"value": "$XXB", "year": "2026", "source": "basis"}},
        "growth_rate": "XX% CAGR",
        "maturity_stage": "emerging/growth/mature/declining"
    }},
    "key_trends": [
        {{"trend": "description", "impact": "high/medium/low", "timeframe": "now/near-term/long-term"}}
    ],
    "customer_segments": [
        {{"segment": "name", "size": "relative", "pain_points": ["list"], "willingness_to_pay": "high/medium/low"}}
    ],
    "market_forces": {{
        "tailwinds": ["forces driving growth"],
        "headwinds": ["forces limiting growth"],
        "disruption_risks": ["emerging threats"]
    }},
    "key_players": [
        {{"name": "company", "positioning": "brief", "estimated_market_share": "XX%"}}
    ]
}}"""

    response = llm.invoke([
        SystemMessage(content="You are a market research expert. Provide specific numbers. Return ONLY valid JSON."),
        HumanMessage(content=prompt)
    ])
    try:
        raw = re.sub(r'^```json\s*', '', response.content.strip())
        raw = re.sub(r'\s*```$', '', raw)
        market = json.loads(raw)
    except json.JSONDecodeError:
        market = {"error": "Failed to parse", "raw": response.content[:500]}

    state['market_research'] = market
    state['processing_log'].append({'agent': 'Market Research Analyst', 'timestamp': datetime.now().isoformat(), 'status': 'success' if 'error' not in market else 'error'})
    return state


def competitor_intel_agent(state, llm):
    product = state['product_input']
    market_context = json.dumps(state['market_research'], indent=2, default=str)

    prompt = f"""You are a competitive intelligence specialist. Analyze competitors for:
Company: {product['company_name']} | Product: {product['product_name']}
Description: {product['description']}
Focus: {product.get('analysis_focus', 'General')}

Market context: {market_context}

Identify top 4-5 competitors. Return ONLY valid JSON:
{{
    "competitors": [
        {{
            "name": "name", "product": "product", "positioning": "one-line",
            "target_audience": "audience",
            "pricing": {{"model": "type", "range": "$X-$X/user/mo"}},
            "key_features": ["top 5"], "ai_capabilities": ["AI features"],
            "strengths": ["top 3"], "weaknesses": ["top 3"],
            "recent_moves": ["notable moves"], "threat_level": "high/medium/low"
        }}
    ],
    "feature_comparison": {{
        "categories": ["categories"],
        "matrix": [{{"feature": "name", "importance": "level", "{product['company_name']}": "rating", "competitor_leader": "who and why"}}]
    }},
    "competitive_dynamics": {{
        "moats": ["defensible advantages"],
        "vulnerabilities": ["areas losing"],
        "consolidation_risk": "level",
        "pricing_pressure": "trend"
    }}
}}"""

    response = llm.invoke([
        SystemMessage(content="Competitive intelligence expert. Be specific. Return ONLY valid JSON."),
        HumanMessage(content=prompt)
    ])
    try:
        raw = re.sub(r'^```json\s*', '', response.content.strip())
        raw = re.sub(r'\s*```$', '', raw)
        competitors = json.loads(raw)
    except json.JSONDecodeError:
        competitors = {"error": "Failed to parse", "raw": response.content[:500]}

    state['competitors'] = competitors
    state['processing_log'].append({'agent': 'Competitor Intelligence', 'timestamp': datetime.now().isoformat(), 'status': 'success' if 'error' not in competitors else 'error'})
    return state


def gap_analysis_agent(state, llm):
    product = state['product_input']
    market = json.dumps(state['market_research'], indent=2, default=str)
    competitors = json.dumps(state['competitors'], indent=2, default=str)

    prompt = f"""You are a product strategist identifying opportunities for:
Company: {product['company_name']} | Product: {product['product_name']}
Focus: {product.get('analysis_focus', 'General')}

MARKET: {market}
COMPETITORS: {competitors}

Return ONLY valid JSON:
{{
    "unmet_needs": [
        {{"need": "description", "segment_affected": "segment", "severity": "level", "current_workaround": "how solved today", "opportunity_size": "large/medium/small"}}
    ],
    "competitive_gaps": [
        {{"gap": "description", "competitors_with_this": ["list"], "impact_on_churn": "level", "difficulty_to_build": "level"}}
    ],
    "whitespace_opportunities": [
        {{"opportunity": "description", "rationale": "why whitespace", "potential_impact": "level", "time_to_market": "months"}}
    ],
    "strategic_recommendations": [
        {{"recommendation": "move", "type": "build/buy/partner", "priority": "timing", "rationale": "why now"}}
    ],
    "positioning_advice": {{
        "current_positioning": "current", "recommended_positioning": "recommended",
        "key_differentiator": "the #1 thing to own", "messaging_pillars": ["3 messages"]
    }}
}}"""

    response = llm.invoke([
        SystemMessage(content="Product strategy expert. Think like a VP of Product. Return ONLY valid JSON."),
        HumanMessage(content=prompt)
    ])
    try:
        raw = re.sub(r'^```json\s*', '', response.content.strip())
        raw = re.sub(r'\s*```$', '', raw)
        gaps = json.loads(raw)
    except json.JSONDecodeError:
        gaps = {"error": "Failed to parse", "raw": response.content[:500]}

    state['gap_analysis'] = gaps
    state['processing_log'].append({'agent': 'Product Gap Analyzer', 'timestamp': datetime.now().isoformat(), 'status': 'success' if 'error' not in gaps else 'error'})
    return state


def feature_prioritizer_agent(state, llm):
    product = state['product_input']
    gaps = json.dumps(state['gap_analysis'], indent=2, default=str)
    market = json.dumps(state['market_research'], indent=2, default=str)

    prompt = f"""You are a senior PM creating a prioritized backlog for:
Product: {product['product_name']} by {product['company_name']}

GAPS: {gaps}
MARKET: {market}

Score using RICE: Reach * Impact * Confidence / Effort. Generate 8-12 features.
Return ONLY valid JSON:
{{
    "features": [
        {{
            "id": "F001", "name": "feature name", "description": "what and why",
            "user_story": "As a..., I want..., so that...",
            "addresses": "which gap", "reach": 10000, "impact": 2, "confidence": 0.8, "effort": 3,
            "rice_score": 5333, "tier": "P0/P1/P2", "category": "AI/UX/Integration/Platform/Growth",
            "dependencies": ["list"], "success_metrics": ["how to measure"]
        }}
    ],
    "roadmap_recommendation": {{
        "now": ["IDs 0-3 months"], "next": ["IDs 3-6 months"], "later": ["IDs 6-12 months"],
        "rationale": "sequencing logic"
    }},
    "resource_estimate": {{
        "total_effort_months": 0, "recommended_team_size": 0, "timeline": "X months for P0"
    }}
}}"""

    response = llm.invoke([
        SystemMessage(content="Senior PM expert in prioritization. RICE scores must be calculated correctly. Return ONLY valid JSON."),
        HumanMessage(content=prompt)
    ])
    try:
        raw = re.sub(r'^```json\s*', '', response.content.strip())
        raw = re.sub(r'\s*```$', '', raw)
        features = json.loads(raw)
    except json.JSONDecodeError:
        features = {"error": "Failed to parse", "raw": response.content[:500]}

    state['feature_backlog'] = features
    state['processing_log'].append({'agent': 'Feature Prioritizer (RICE)', 'timestamp': datetime.now().isoformat(), 'status': 'success' if 'error' not in features else 'error'})
    return state


def prd_generator_agent(state, llm):
    product = state['product_input']
    market = json.dumps(state['market_research'], indent=2, default=str)
    competitors = json.dumps(state['competitors'], indent=2, default=str)
    gaps = json.dumps(state['gap_analysis'], indent=2, default=str)
    features = json.dumps(state['feature_backlog'], indent=2, default=str)

    prompt = f"""You are a VP of Product writing a comprehensive PRD for {product['product_name']} by {product['company_name']}.

MARKET: {market}
COMPETITORS: {competitors}
GAPS: {gaps}
FEATURES: {features}

Write a professional PRD in markdown with these sections:
# Product Requirements Document: {product['product_name']}
### {product['company_name']} — Strategic Product Analysis

**Date:** {datetime.now().strftime('%B %d, %Y')}
**Status:** Draft for Review

## 1. Executive Summary
## 2. Market Context
## 3. Competitive Landscape (include comparison table)
## 4. Problem Statement & Opportunity
## 5. Proposed Solution (5.1 Vision, 5.2 Feature Specs with user stories, 5.3 Prioritized Roadmap table)
## 6. Success Metrics
## 7. Risks & Mitigations
## 8. Resource Requirements
## 9. Appendix (RICE scoring table)

---
*Generated by AI Product Strategist Agent — Multi-Agent System by Nikhil Patel*"""

    response = llm.invoke([
        SystemMessage(content="VP of Product at a top tech company. Write a PRD that would impress a CPO. Be specific and data-driven."),
        HumanMessage(content=prompt)
    ])

    state['prd'] = response.content.strip()
    state['processing_log'].append({'agent': 'PRD Generator', 'timestamp': datetime.now().isoformat(), 'status': 'success'})
    return state


# ─────────────────────────────────────────────
# Dashboard
# ─────────────────────────────────────────────
def create_dashboard(result):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    product = result['product_input']
    fig.suptitle(f"{product['company_name']} — {product['product_name']} Strategy Dashboard",
                 fontsize=16, fontweight='bold', y=0.98, color='white')
    fig.patch.set_facecolor('#0a1128')

    for ax in axes.flat:
        ax.set_facecolor('#0d1635')
        ax.tick_params(colors='#a0a0a0')
        ax.spines['bottom'].set_color('#2a2a2a')
        ax.spines['left'].set_color('#2a2a2a')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # RICE Scores
    ax1 = axes[0, 0]
    features = result.get('feature_backlog', {}).get('features', [])
    if features:
        sf = sorted(features, key=lambda x: x.get('rice_score', 0), reverse=True)[:8]
        names = [f.get('name', '?')[:25] for f in sf]
        scores = [f.get('rice_score', 0) for f in sf]
        tiers = [f.get('tier', 'P2') for f in sf]
        colors = ['#4CC9A4' if t == 'P0' else '#F5C878' if t == 'P1' else '#a0a0a0' for t in tiers]
        ax1.barh(names[::-1], scores[::-1], color=colors[::-1], alpha=0.85)
        ax1.set_xlabel('RICE Score', color='#a0a0a0')
    ax1.set_title('Feature Priority (RICE)', color='white', fontsize=12, pad=10)
    ax1.tick_params(axis='y', labelsize=8, colors='#a0a0a0')

    # Competitor Threats
    ax2 = axes[0, 1]
    comps = result.get('competitors', {}).get('competitors', [])
    if comps:
        cn = [c.get('name', '?')[:15] for c in comps]
        tm = {'high': 3, 'medium': 2, 'low': 1}
        th = [tm.get(c.get('threat_level', 'low'), 1) for c in comps]
        tc = ['#e74c3c' if t == 3 else '#F5C878' if t == 2 else '#4CC9A4' for t in th]
        ax2.barh(cn, th, color=tc, alpha=0.85)
        ax2.set_xticks([1, 2, 3])
        ax2.set_xticklabels(['Low', 'Medium', 'High'], color='#a0a0a0')
    ax2.set_title('Competitor Threat Levels', color='white', fontsize=12, pad=10)
    ax2.tick_params(axis='y', labelsize=9, colors='#a0a0a0')

    # Opportunity Map
    ax3 = axes[1, 0]
    g = result.get('gap_analysis', {})
    cats = ['Unmet Needs', 'Competitive\nGaps', 'Whitespace']
    cts = [len(g.get('unmet_needs', [])), len(g.get('competitive_gaps', [])), len(g.get('whitespace_opportunities', []))]
    ax3.bar(cats, cts, color=['#e74c3c', '#F5C878', '#4CC9A4'], width=0.5)
    for i, v in enumerate(cts):
        ax3.text(i, v + 0.1, str(v), ha='center', color='white', fontweight='bold')
    ax3.set_title('Opportunity Landscape', color='white', fontsize=12, pad=10)
    ax3.tick_params(axis='x', colors='#a0a0a0', labelsize=9)

    # Roadmap
    ax4 = axes[1, 1]
    rm = result.get('feature_backlog', {}).get('roadmap_recommendation', {})
    phases = ['Now\n(0-3 mo)', 'Next\n(3-6 mo)', 'Later\n(6-12 mo)']
    pc = [len(rm.get('now', [])), len(rm.get('next', [])), len(rm.get('later', []))]
    ax4.bar(phases, pc, color=['#4CC9A4', '#F5C878', '#00d9ff'], width=0.5)
    for i, v in enumerate(pc):
        ax4.text(i, v + 0.1, str(v), ha='center', color='white', fontweight='bold')
    ax4.set_title('Roadmap Distribution', color='white', fontsize=12, pad=10)
    ax4.tick_params(axis='x', colors='#a0a0a0')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


# ─────────────────────────────────────────────
# Main App
# ─────────────────────────────────────────────
def main():
    st.markdown('<p class="main-title">AI Product Strategist Agent</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">5 Agents. One PRD. Zero Guesswork. &nbsp;|&nbsp; Built by <a href="https://www.nikhilcpatel.com" target="_blank" style="color:#F5C878;">Nikhil Patel</a></p>', unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### Configuration")
        api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
        st.markdown("---")
        st.markdown("### Product to Analyze")
        company_name = st.text_input("Company Name", value="Notion")
        product_name = st.text_input("Product Name", value="Notion AI")
        product_url = st.text_input("Product URL (optional)", value="https://www.notion.so")
        description = st.text_area("Product Description", value="Notion is an all-in-one workspace for notes, docs, wikis, and project management. Notion AI adds generative AI features for writing, summarization, and data extraction within the workspace.", height=100)
        target_market = st.text_input("Target Market", value="Knowledge workers, product teams, startups, mid-market companies")
        analysis_focus = st.text_input("Analysis Focus", value="AI-powered productivity features and differentiation against Coda, Confluence, ClickUp")
        st.markdown("---")
        st.markdown("""
        <div style="color:#666;font-size:0.8rem;">
        <strong>How it works:</strong><br>
        5 specialized AI agents analyze the product:<br><br>
        1. Market Research Analyst<br>
        2. Competitor Intelligence<br>
        3. Product Gap Analyzer<br>
        4. Feature Prioritizer (RICE)<br>
        5. PRD Generator<br><br>
        Each agent enriches a shared state.
        </div>
        """, unsafe_allow_html=True)

    if not api_key:
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown('<div class="metric-card"><div class="metric-value">5</div><div class="metric-label">AI Agents</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-card"><div class="metric-value">RICE</div><div class="metric-label">Scoring Framework</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="metric-card"><div class="metric-value">12+</div><div class="metric-label">Feature Proposals</div></div>', unsafe_allow_html=True)
        with col4:
            st.markdown('<div class="metric-card"><div class="metric-value">PRD</div><div class="metric-label">Full Document</div></div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### How It Works")
        st.markdown("Enter any company or product, and 5 AI agents will analyze the market, competitors, gaps, and generate a complete PRD with a prioritized feature backlog.")
        st.code("Product → [Market Research] → [Competitor Intel] → [Gap Analysis] → [RICE Prioritization] → [PRD Generator] → PRD", language=None)
        st.info("Enter your OpenAI API key in the sidebar to get started. Your key is never stored.")
        return

    st.markdown("---")
    if st.button("Generate Product Strategy & PRD", type="primary", use_container_width=True):
        llm = get_llm(api_key)

        product_input = {
            'company_name': company_name,
            'product_name': product_name,
            'product_url': product_url,
            'description': description,
            'target_market': target_market,
            'analysis_focus': analysis_focus
        }

        state = {
            'product_input': product_input,
            'market_research': {},
            'competitors': {},
            'gap_analysis': {},
            'feature_backlog': {},
            'prd': '',
            'processing_log': []
        }

        start_time = time.time()

        with st.status("Agent 1: Market Research Analyst", expanded=True) as status:
            st.write("Mapping market landscape, TAM, trends, and key players...")
            state = market_research_agent(state, llm)
            tam = state['market_research'].get('market_overview', {}).get('tam', {}).get('value', 'N/A')
            players = len(state['market_research'].get('key_players', []))
            st.write(f"TAM: {tam} | {players} key players identified")
            status.update(label="Agent 1: Market Research — Done", state="complete")

        with st.status("Agent 2: Competitor Intelligence", expanded=True) as status:
            st.write("Deep-diving competitor features, pricing, and positioning...")
            state = competitor_intel_agent(state, llm)
            cc = len(state['competitors'].get('competitors', []))
            st.write(f"{cc} competitors analyzed with feature comparison matrix")
            status.update(label="Agent 2: Competitor Intelligence — Done", state="complete")

        with st.status("Agent 3: Product Gap Analyzer", expanded=True) as status:
            st.write("Identifying unmet needs, competitive gaps, whitespace...")
            state = gap_analysis_agent(state, llm)
            un = len(state['gap_analysis'].get('unmet_needs', []))
            ws = len(state['gap_analysis'].get('whitespace_opportunities', []))
            st.write(f"{un} unmet needs | {ws} whitespace opportunities found")
            status.update(label="Agent 3: Gap Analysis — Done", state="complete")

        with st.status("Agent 4: Feature Prioritizer (RICE)", expanded=True) as status:
            st.write("Scoring and ranking feature opportunities...")
            state = feature_prioritizer_agent(state, llm)
            fc = len(state['feature_backlog'].get('features', []))
            p0 = len(state['feature_backlog'].get('roadmap_recommendation', {}).get('now', []))
            st.write(f"{fc} features scored | {p0} P0 features for immediate build")
            status.update(label="Agent 4: Feature Prioritization — Done", state="complete")

        with st.status("Agent 5: PRD Generator", expanded=True) as status:
            st.write("Synthesizing all analysis into a professional PRD...")
            state = prd_generator_agent(state, llm)
            st.write(f"PRD generated: {len(state['prd']):,} characters")
            status.update(label="Agent 5: PRD Generator — Done", state="complete")

        elapsed = time.time() - start_time
        st.session_state['result'] = state
        st.session_state['elapsed'] = elapsed

    if 'result' in st.session_state:
        result = st.session_state['result']
        elapsed = st.session_state['elapsed']
        product = result['product_input']

        st.markdown("---")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            tam = result['market_research'].get('market_overview', {}).get('tam', {}).get('value', 'N/A')
            st.markdown(f'<div class="metric-card"><div class="metric-value">{tam}</div><div class="metric-label">TAM</div></div>', unsafe_allow_html=True)
        with col2:
            cc = len(result['competitors'].get('competitors', []))
            st.markdown(f'<div class="metric-card"><div class="metric-value">{cc}</div><div class="metric-label">Competitors Analyzed</div></div>', unsafe_allow_html=True)
        with col3:
            fc = len(result['feature_backlog'].get('features', []))
            st.markdown(f'<div class="metric-card"><div class="metric-value">{fc}</div><div class="metric-label">Features Proposed</div></div>', unsafe_allow_html=True)
        with col4:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{elapsed:.0f}s</div><div class="metric-label">Analysis Time</div></div>', unsafe_allow_html=True)

        st.markdown("---")

        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["PRD", "Dashboard", "Market", "Competitors", "Gaps", "Features"])

        with tab1:
            st.markdown(result['prd'])

        with tab2:
            fig = create_dashboard(result)
            st.pyplot(fig)
            plt.close(fig)

        with tab3:
            st.subheader("Market Research")
            mr = result['market_research']
            if 'error' not in mr:
                mo = mr.get('market_overview', {})
                mcol1, mcol2, mcol3 = st.columns(3)
                with mcol1:
                    st.metric("Category", mo.get('category', 'N/A'))
                with mcol2:
                    st.metric("TAM", mo.get('tam', {}).get('value', 'N/A'))
                with mcol3:
                    st.metric("Growth", mo.get('growth_rate', 'N/A'))

                st.markdown("#### Key Trends")
                for t in mr.get('key_trends', []):
                    st.info(f"**{t.get('trend', '')}** — Impact: {t.get('impact', '')} | Timeframe: {t.get('timeframe', '')}")

                st.markdown("#### Market Forces")
                fcol1, fcol2 = st.columns(2)
                with fcol1:
                    st.markdown("**Tailwinds**")
                    for tw in mr.get('market_forces', {}).get('tailwinds', []):
                        st.success(tw)
                with fcol2:
                    st.markdown("**Headwinds**")
                    for hw in mr.get('market_forces', {}).get('headwinds', []):
                        st.warning(hw)
            else:
                st.json(mr)

        with tab4:
            st.subheader("Competitive Landscape")
            comp = result['competitors']
            if 'error' not in comp:
                for c in comp.get('competitors', []):
                    threat = c.get('threat_level', 'low')
                    icon = "🔴" if threat == 'high' else "🟡" if threat == 'medium' else "🟢"
                    with st.expander(f"{icon} {c.get('name', '?')} — {c.get('positioning', '')}", expanded=False):
                        st.write(f"**Target:** {c.get('target_audience', 'N/A')}")
                        st.write(f"**Pricing:** {c.get('pricing', {}).get('range', 'N/A')} ({c.get('pricing', {}).get('model', '')})")
                        st.write(f"**Key Features:** {', '.join(c.get('key_features', []))}")
                        if c.get('ai_capabilities'):
                            st.write(f"**AI:** {', '.join(c.get('ai_capabilities', []))}")
                        scol1, scol2 = st.columns(2)
                        with scol1:
                            st.markdown("**Strengths**")
                            for s in c.get('strengths', []):
                                st.success(s)
                        with scol2:
                            st.markdown("**Weaknesses**")
                            for w in c.get('weaknesses', []):
                                st.error(w)
            else:
                st.json(comp)

        with tab5:
            st.subheader("Product Gaps & Opportunities")
            ga = result['gap_analysis']
            if 'error' not in ga:
                st.markdown("#### Unmet Needs")
                for n in ga.get('unmet_needs', []):
                    st.warning(f"**{n.get('need', '')}** — Severity: {n.get('severity', '')} | Opportunity: {n.get('opportunity_size', '')}")

                st.markdown("#### Whitespace Opportunities")
                for w in ga.get('whitespace_opportunities', []):
                    st.success(f"**{w.get('opportunity', '')}** — Impact: {w.get('potential_impact', '')} | Time to market: {w.get('time_to_market', '')}")

                pos = ga.get('positioning_advice', {})
                if pos:
                    st.markdown("#### Positioning")
                    st.info(f"**Key Differentiator:** {pos.get('key_differentiator', 'N/A')}")
                    st.write(f"**Current:** {pos.get('current_positioning', '')}")
                    st.write(f"**Recommended:** {pos.get('recommended_positioning', '')}")
            else:
                st.json(ga)

        with tab6:
            st.subheader("Feature Backlog (RICE)")
            fb = result['feature_backlog']
            if 'error' not in fb and fb.get('features'):
                df = pd.DataFrame(fb['features'])
                cols_to_show = ['id', 'name', 'tier', 'rice_score', 'reach', 'impact', 'confidence', 'effort', 'category']
                available_cols = [c for c in cols_to_show if c in df.columns]
                st.dataframe(df[available_cols].sort_values('rice_score', ascending=False), use_container_width=True, hide_index=True)

                rm = fb.get('roadmap_recommendation', {})
                if rm:
                    st.markdown("#### Roadmap")
                    rcol1, rcol2, rcol3 = st.columns(3)
                    with rcol1:
                        st.markdown("**Now (0-3 mo)**")
                        for fid in rm.get('now', []):
                            st.success(fid)
                    with rcol2:
                        st.markdown("**Next (3-6 mo)**")
                        for fid in rm.get('next', []):
                            st.info(fid)
                    with rcol3:
                        st.markdown("**Later (6-12 mo)**")
                        for fid in rm.get('later', []):
                            st.warning(fid)
            else:
                st.json(fb)

        # Downloads
        st.markdown("---")
        st.markdown("### Download Results")
        dcol1, dcol2 = st.columns(2)
        with dcol1:
            st.download_button("Download PRD (MD)", result['prd'],
                file_name=f"{product['company_name'].replace(' ', '_')}_PRD.md", mime="text/markdown")
        with dcol2:
            state_json = json.dumps({
                'product_input': result['product_input'],
                'market_research': result['market_research'],
                'competitors': result['competitors'],
                'gap_analysis': result['gap_analysis'],
                'feature_backlog': result['feature_backlog'],
                'processing_log': result['processing_log']
            }, indent=2, default=str)
            st.download_button("Download Full Analysis (JSON)", state_json,
                file_name=f"{product['company_name'].replace(' ', '_')}_analysis.json", mime="application/json")

    st.markdown("""
    <div class="footer-text">
        Built by <a href="https://www.nikhilcpatel.com" target="_blank">Nikhil Patel</a> &nbsp;|&nbsp;
        <a href="https://github.com/patelnikhilc/The-Lab" target="_blank">GitHub</a> &nbsp;|&nbsp;
        Part of <a href="https://www.nikhilcpatel.com/#lab" target="_blank">The Lab</a> — AI Product Case Studies
        <br><br><em>This tool generates strategic analysis for educational purposes. Always validate with primary research.</em>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
