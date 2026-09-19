"""
FinPilot - Personal Finance Decision Support Agent
Built for Agentic AI Hackathon 2026

Run with: streamlit run app.py

API key setup (no key box in UI - set this before running):
    export ANTHROPIC_API_KEY="your-key-here"      (Mac/Linux)
    setx ANTHROPIC_API_KEY "your-key-here"          (Windows, then reopen terminal)
"""

import os
import streamlit as st
import pandas as pd
import plotly.express as px

from utils.parser import parse_csv, load_sample_data
from utils.categorize import (
    categorize_transactions,
    detect_recurring,
    generate_insight_answer,
    build_data_summary,
)

st.set_page_config(page_title="FinPilot", layout="wide")

# ---------------------------------------------------------------------------
# THEME — dark finance dashboard look
# ---------------------------------------------------------------------------
GOLD = "#D4AF6A"
TEAL = "#4FD1C5"
CORAL = "#E07A5F"
CHART_COLORS = [GOLD, TEAL, "#7C93A8", CORAL, "#8B7FD1", "#5A7A6B"]

CURRENCIES = {
    "USD": "$",
    "EUR": "€",
    "GBP": "£",
    "INR": "₹",
    "KRW": "₩",
    "JPY": "¥",
}

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700&display=swap');

html, body, [class*="css"], p, div, span, label, input, textarea {
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
}

h1, h2, h3, h4, .stMarkdown h3 {
    font-family: 'Montserrat', 'Helvetica Neue', sans-serif !important;
    letter-spacing: -0.01em;
}

/* Metric cards: flat panel + accent top border instead of shadow */
div[data-testid="stMetric"] {
    background-color: #131A22;
    border: 1px solid #232B35;
    border-top: 3px solid #D4AF6A;
    border-radius: 6px;
    padding: 16px 18px;
}

div[data-testid="stMetricLabel"] {
    color: #8593A3;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #0E1319;
    border-right: 1px solid #232B35;
}

.stAlert {
    border-radius: 6px;
}

/* Chat panel container */
.chat-panel-header {
    background: linear-gradient(180deg, #161B22 0%, #131A22 100%);
    border: 1px solid #232B35;
    border-radius: 10px;
    padding: 22px 24px;
    margin-bottom: 4px;
}
.chat-panel-header h4 {
    margin: 0;
    font-family: 'Montserrat', sans-serif;
    color: #E8EDF2;
    font-size: 1.1rem;
}
.chat-panel-header p {
    margin: 4px 0 0 0;
    color: #8593A3;
    font-size: 0.85rem;
}

div[data-testid="stChatInput"] {
    border: 1px solid #232B35;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# LLM CLIENT SETUP
# No API key box in the UI. Set ANTHROPIC_API_KEY as an environment
# variable (or in .streamlit/secrets.toml) before running the app.
# ---------------------------------------------------------------------------
def get_llm_call():
    # Try Gemini first (free, no card needed), then fall back to Anthropic
    gemini_key = ""
    anthropic_key = ""
    try:
        gemini_key = st.secrets.get("GEMINI_API_KEY", "")
        anthropic_key = st.secrets.get("ANTHROPIC_API_KEY", "")
    except Exception:
        pass
    gemini_key = gemini_key or os.environ.get("GEMINI_API_KEY", "")
    anthropic_key = anthropic_key or os.environ.get("ANTHROPIC_API_KEY", "")

    if gemini_key:
        import google.generativeai as genai
        genai.configure(api_key=gemini_key)
        model = genai.GenerativeModel("gemini-3.6-flash")

        def call(prompt: str) -> str:
            response = model.generate_content(prompt)
            return response.text

        return call

    if anthropic_key:
        import anthropic
        client = anthropic.Anthropic(api_key=anthropic_key)

        def call(prompt: str) -> str:
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text

        return call

    return None


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
st.sidebar.title("FinPilot")
st.sidebar.caption("Your personal finance decision-support agent")

currency_code = st.sidebar.selectbox("Currency", list(CURRENCIES.keys()), index=0)
CURRENCY_SYMBOL = CURRENCIES[currency_code]

use_sample = st.sidebar.checkbox("Use sample demo data", value=True)
uploaded_file = None
if not use_sample:
    uploaded_file = st.sidebar.file_uploader("Upload transactions CSV", type=["csv"])

monthly_budget = st.sidebar.number_input(
    f"Monthly budget ({CURRENCY_SYMBOL})", min_value=0, value=25000, step=1000
)


# ---------------------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------------------
if use_sample:
    df = load_sample_data()
elif uploaded_file is not None:
    df = parse_csv(uploaded_file)
else:
    st.info("Upload a CSV or enable sample data from the sidebar to get started.")
    st.stop()

llm_call = get_llm_call()

if llm_call:
    df = categorize_transactions(df, llm_call)
else:
    from utils.categorize import naive_categorize
    df["category"] = df["description"].apply(naive_categorize)

df = detect_recurring(df)


# ---------------------------------------------------------------------------
# HEADER METRICS
# ---------------------------------------------------------------------------
st.markdown("### FinPilot")
st.caption("Your finances, understood.")

total_spend = -df[df["amount"] < 0]["amount"].sum()
total_income = df[df["amount"] > 0]["amount"].sum()
net = total_income - total_spend
budget_used_pct = (total_spend / monthly_budget * 100) if monthly_budget else 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Spend", f"{CURRENCY_SYMBOL}{total_spend:,.0f}")
col2.metric("Total Income", f"{CURRENCY_SYMBOL}{total_income:,.0f}")
col3.metric("Net", f"{CURRENCY_SYMBOL}{net:,.0f}", delta=f"{net:,.0f}")
col4.metric("Budget Used", f"{budget_used_pct:.0f}%",
            delta=f"{CURRENCY_SYMBOL}{monthly_budget - total_spend:,.0f} left",
            delta_color="normal" if budget_used_pct <= 100 else "inverse")

if budget_used_pct > 100:
    st.warning("You've exceeded your monthly budget.")
elif budget_used_pct > 80:
    st.warning("You're close to your monthly budget limit.")


# ---------------------------------------------------------------------------
# CHARTS
# ---------------------------------------------------------------------------
c1, c2 = st.columns(2)

with c1:
    st.subheader("Spend by Category")
    cat_spend = (
        df[df["amount"] < 0]
        .groupby("category")["amount"]
        .sum()
        .abs()
        .sort_values(ascending=False)
        .reset_index()
    )
    fig = px.pie(
        cat_spend, names="category", values="amount", hole=0.55,
        color_discrete_sequence=CHART_COLORS,
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#E8EDF2", font_family="Helvetica Neue, Arial, sans-serif",
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=10, b=10, l=10, r=10),
    )
    st.plotly_chart(fig, use_container_width=True)

with c2:
    st.subheader("Spending Over Time")
    daily = df.groupby(df["date"].dt.date)["amount"].sum().reset_index()
    fig2 = px.line(daily, x="date", y="amount", markers=True,
                    color_discrete_sequence=[GOLD])
    fig2.update_traces(line=dict(width=2), marker=dict(size=6, color=TEAL))
    fig2.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#E8EDF2", font_family="Helvetica Neue, Arial, sans-serif",
        xaxis=dict(gridcolor="#232B35"), yaxis=dict(gridcolor="#232B35"),
        margin=dict(t=10, b=10, l=10, r=10),
    )
    st.plotly_chart(fig2, use_container_width=True)


# ---------------------------------------------------------------------------
# RECURRING / SUBSCRIPTIONS
# ---------------------------------------------------------------------------
st.subheader("Recurring Payments & Subscriptions Detected")
recurring_df = df[df["is_recurring"]][["date", "description", "amount", "category"]]
if not recurring_df.empty:
    st.dataframe(recurring_df, use_container_width=True, hide_index=True)
else:
    st.caption("No recurring transactions detected yet.")


# ---------------------------------------------------------------------------
# TRANSACTION TABLE
# ---------------------------------------------------------------------------
with st.expander("All Transactions"):
    st.dataframe(df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# CHAT / NATURAL LANGUAGE Q&A — scoped strictly to this project's data
# ---------------------------------------------------------------------------
st.markdown("""
<div class="chat-panel-header">
    <h4>Ask FinPilot</h4>
    <p>Answers questions about your transactions, spending, and budget shown above.</p>
</div>
""", unsafe_allow_html=True)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

question = st.chat_input("e.g. Where did I spend the most this month?")

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

OFF_TOPIC_REPLY = "You're on the wrong path. I can only help with questions about your finances in this dashboard."

if question:
    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        if llm_call:
            summary = build_data_summary(df)
            answer = generate_insight_answer(question, summary, llm_call)
        else:
            answer = ("No API key detected, so AI chat is unavailable right now. "
                       "Set GEMINI_API_KEY (or ANTHROPIC_API_KEY) as an environment "
                       "variable, then restart the app. For now, check the charts "
                       "above for insights.")
        st.write(answer)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
