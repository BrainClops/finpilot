"""
categorize.py
Handles AI-powered categorization of transactions, recurring/subscription
detection, and natural-language Q&A over the transaction data.

Uses the Anthropic API (Claude) by default. Swap the client init if you
prefer OpenAI/Gemini - the prompt-based approach stays the same.
"""

import json
import pandas as pd

CATEGORIES = [
    "Food", "Rent", "Subscription", "Transport", "Shopping",
    "Utilities", "Health", "Income", "Entertainment", "Other"
]


def build_categorize_prompt(descriptions: list[str]) -> str:
    return f"""
You are a financial transaction categorizer.

Categorize each of the following transaction descriptions into EXACTLY
one of these categories: {", ".join(CATEGORIES)}.

Return ONLY a JSON array of category strings, in the same order as the
input, with no extra text or explanation.

Transactions:
{json.dumps(descriptions)}
"""


def categorize_transactions(df: pd.DataFrame, llm_call) -> pd.DataFrame:
    """
    llm_call: a function(prompt: str) -> str  that sends the prompt to
    your chosen LLM API and returns the raw text response.
    You wire this up in app.py depending on which API key you have.
    """
    descriptions = df["description"].tolist()
    prompt = build_categorize_prompt(descriptions)
    raw_response = llm_call(prompt)

    try:
        categories = json.loads(raw_response)
    except json.JSONDecodeError:
        # Fallback: naive keyword matching if LLM output isn't clean JSON
        categories = [naive_categorize(d) for d in descriptions]

    df = df.copy()
    df["category"] = categories[: len(df)]
    return df


def naive_categorize(description: str) -> str:
    """Simple keyword fallback so the app never fully breaks without an LLM key."""
    d = description.lower()
    mapping = {
        "Food": ["swiggy", "zomato", "restaurant", "food"],
        "Subscription": ["netflix", "spotify", "prime", "subscription"],
        "Rent": ["rent"],
        "Utilities": ["electricity", "water bill", "utility"],
        "Transport": ["uber", "ola", "fuel", "petrol"],
        "Shopping": ["amazon", "flipkart", "myntra"],
        "Income": ["salary", "credit"],
        "Health": ["gym", "pharmacy", "hospital"],
    }
    for cat, keywords in mapping.items():
        if any(k in d for k in keywords):
            return cat
    return "Other"


def detect_recurring(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flags likely recurring/subscription transactions:
    same description appearing 2+ times with similar amount.
    """
    df = df.copy()
    grouped = df.groupby("description")["amount"].agg(["count", "std"]).reset_index()
    recurring_desc = grouped[
        (grouped["count"] >= 2) & (grouped["std"].fillna(0) < 25)
    ]["description"]

    df["is_recurring"] = df["description"].isin(recurring_desc)
    return df


def generate_insight_answer(question: str, df_summary: str, llm_call) -> str:
    """
    Answers a natural-language question about the user's spending,
    grounded in a text summary of their transaction data.

    Strictly scoped: the assistant only answers questions about the
    user's finances within this app. Anything off-topic gets redirected
    with a fixed English refusal line instead of a real answer.
    """
    prompt = f"""
You are FinPilot, a personal finance assistant embedded inside a finance
dashboard app. You are strictly scoped to this app only.

Rules:
1. Only answer questions about the user's spending, income, budget,
   categories, or recurring payments, using ONLY the data summary below.
2. You may also help the user reason through decisions related to that
   data (e.g. "should I cut this subscription", "how can I stay under
   budget") - always grounded in the summary provided.
3. If the question is unrelated to the user's finances in this app
   (general knowledge, coding help, unrelated topics, anything outside
   this data), do NOT answer it. Instead, reply with EXACTLY this
   sentence and nothing else: "You're on the wrong path. I can only
   help with questions about your finances in this dashboard."
4. Be concise and specific, using actual numbers from the summary.

Data summary:
{df_summary}

User question: {question}
"""
    return llm_call(prompt)


def build_data_summary(df: pd.DataFrame) -> str:
    """Creates a compact text summary of transactions for grounding LLM answers."""
    total_spend = df[df["amount"] < 0]["amount"].sum()
    total_income = df[df["amount"] > 0]["amount"].sum()
    by_category = df.groupby("category")["amount"].sum().sort_values()

    lines = [
        f"Total spend: {total_spend:.2f}",
        f"Total income: {total_income:.2f}",
        "Spend by category:",
    ]
    for cat, amt in by_category.items():
        if amt < 0:
            lines.append(f"  - {cat}: {amt:.2f}")

    return "\n".join(lines)
