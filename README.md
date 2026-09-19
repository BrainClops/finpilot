# FinPilot

Personal finance agent I'm building for the Agentic AI Hackathon 2026.

Basic idea: upload your transactions, it figures out where your money's
going, catches recurring subscriptions you forgot about, checks it
against your budget, and you can just ask it stuff like "where did I
overspend this month" instead of digging through numbers yourself.

## Running it

```
pip install -r requirements.txt
streamlit run app.py
```

Opens up in your browser. Throw in your Anthropic API key in the sidebar
if you want the actual AI categorization + chat working — without it,
it just falls back to basic keyword matching so the app doesn't break.

## What's in here

- `app.py` — the whole app basically, run this one
- `utils/parser.py` — reads the CSV, also has a sample data generator so
  you can demo without a real bank statement
- `utils/categorize.py` — categorizes transactions, finds recurring
  payments, handles the chat answers

## Done so far

- upload + sample data fallback
- categorization (AI + keyword backup)
- recurring/subscription detection
- budget vs actual with warnings
- charts (category split, spending over time)
- chat box for asking questions about your spending

## Still need to add

- Health score (0-100 based on savings rate, budget adherence etc) — this
  is probably the thing that'll make it stand out, should prioritize this
- monthly summary report, AI-written
- basic prediction — "at this rate you'll blow your budget by ₹X"
- PDF support, right now only takes CSV
- make the UI actually look good, add some branding
- deploy it somewhere (streamlit cloud / HF spaces) so there's a live link

## Demo video (3 min, need this for submission)

- 0:00-0:30 — the problem
- 0:30-1:30 — demo: upload, categorization, dashboard, recurring stuff
- 1:30-2:30 — chat + budget alerts + health score
- 2:30-3:00 — why it's an agent and not just a dashboard

## Notes to self

- built this fresh for the problem statement, didn't reuse old project
  code since hackathon rules said no
- sample data function is there so demo doesn't depend on having a real
  statement handy on stage
- LLM client is in app.py under get_llm_call() — swap it if using OpenAI/Gemini instead
