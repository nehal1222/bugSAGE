# bugSAGE — AI Debugging Coach

**Live:** https://bugsage.onrender.com/app &nbsp;|&nbsp; **GitHub:** https://github.com/nehal1222/bugSAGE

> An AI-powered platform that helps developers debug faster, practice skills, review code, and run engineering workflows — all powered by Google Gemini.

---

## Resume Description

**bugSAGE — AI Debugging & Engineering Coach** | [bugsage.onrender.com/app](https://bugsage.onrender.com/app)

Built a full-stack AI debugging platform using **FastAPI**, **Google Gemini 2.0**, and **PostgreSQL (Supabase)**. Features include AI error analysis, practice question generation, GitHub code review, PR risk scoring, and 10 engineering productivity tools (test generation, security scanning, log analysis, incident investigation, and more). Implemented JWT authentication with Google OAuth, a dual-backend database layer (SQLite/PostgreSQL), Gemini rate-limit retry with exponential backoff, and a responsive single-page frontend. Deployed on Render via Docker with automated CI from GitHub.

---

## Features

### Debug
- AI error analysis with language + skill level selection
- Step-by-step explanations powered by Gemini 2.0 Flash
- Rate-limit retry with exponential backoff

### Practice
- Generate coding questions across Python, JavaScript, Go, Rust, Java, and more
- Modes: MCQ, coding challenges, debugging exercises
- Configurable difficulty and question count

### Code Review
- GitHub repo review — paste a URL for AI architecture analysis
- Diff / PR review — paste a raw diff for risk scoring and change summary

### ⚙ Engineering Tools (10 tools)
| Tool | What it does |
|------|-------------|
| Test Generator | Writes unit tests for pasted code |
| Security Scanner | Finds vulnerabilities and CVEs |
| Log Analyzer | Spots errors and anomalies in log files |
| Root Cause Analysis | Diagnoses bugs from symptoms |
| PR Risk Score | Rates the risk level of a pull request |
| Explain Legacy Code | Translates old/unfamiliar code into plain English |
| Regression Detector | Identifies risky code changes |
| Architecture Review | Reviews system design from a description |
| Sprint Assistant | Turns sprint tickets into an actionable plan |
| AI Incident Investigator | Diagnoses production incidents |

### Dashboard
- Session stats: queries, score, accuracy, streak
- Activity history and leaderboard

### Auth
- Username/password register + login (PBKDF2-HMAC-SHA256)
- Google OAuth sign-in
- JWT access tokens (1 hr) + refresh tokens (30 days)
- Draft persistence across page refreshes via localStorage

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI + Uvicorn (Python 3.12) |
| AI | Google Gemini 2.0 Flash |
| Database | PostgreSQL via Supabase (prod) / SQLite (local dev) |
| Auth | PBKDF2 passwords + Google OAuth 2.0 + JWT |
| Frontend | Vanilla JS + CSS (single-page, no framework) |
| Deployment | Render (Docker) + GitHub Actions CI |

---

## Local Development

```bash
git clone https://github.com/nehal1222/bugSAGE.git
cd bugSAGE

pip install -r requirements.txt

# Create .env with your keys (see below)
uvicorn app:app --reload

# Open http://localhost:8000/app
```

### .env

```env
GEMINI_API_KEY=your_gemini_api_key
GOOGLE_CLIENT_ID=your_google_client_id
# DATABASE_URL=postgresql://...   # leave unset to use SQLite locally
```

---

## Project Structure

```
bugSAGE/
├── app.py            # FastAPI app + all API endpoints
├── db.py             # DB abstraction layer (SQLite / PostgreSQL)
├── index.html        # Frontend single-page app
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

---

## Deployment

Hosted on **Render** (Docker, free tier) with **Supabase PostgreSQL**.

> Free tier note: the service sleeps after 15 min of inactivity — first request after sleep takes ~30 s to wake up.
