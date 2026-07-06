from fastapi import FastAPI, HTTPException, Request, Depends, UploadFile, File, Form
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
from pydantic import BaseModel, field_validator
import asyncio
import os
import httpx
import logging
import json
import random
import time
import hashlib
import hmac
import binascii
from datetime import datetime, timedelta
from dotenv import load_dotenv
from pathlib import Path
from typing import Optional, List, Dict, Any
import uuid
from collections import defaultdict
import re
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests
from db import (
    db_connect, db_close, init_db,
    db_execute, db_commit, db_fetchrow, db_fetchall, db_fetchval,
)
# ---------------------------------------------------------------------
# Env + Logging
# ---------------------------------------------------------------------
load_dotenv(dotenv_path=Path(__file__).parent / ".env")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("debug_coach")

ACCESS_TOKEN_EXPIRY_HOURS = 1
REFRESH_TOKEN_EXPIRY_DAYS = 30
DATA_DIR = Path(os.getenv("DATA_DIR", Path(__file__).parent))
DB_PATH  = DATA_DIR / "bugsage.db"

# ---------------------------------------------------------------------
# Shared resources + lifespan
# ---------------------------------------------------------------------
_http_client: Optional[httpx.AsyncClient] = None


class _DBShim:
    """Thin shim so existing _db.execute / _db.commit call sites need no changes."""

    async def execute(self, sql: str, args: tuple = ()) -> None:
        await db_execute(sql, *args)

    async def commit(self) -> None:
        await db_commit()


_db = _DBShim()


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _http_client
    await db_connect(str(DB_PATH))
    _http_client = httpx.AsyncClient(timeout=60.0)
    await init_db()
    logger.info("Startup complete — database and HTTP client ready")
    yield
    await db_close()
    if _http_client:
        await _http_client.aclose()


# ---------------------------------------------------------------------
# FastAPI + CORS
# ---------------------------------------------------------------------
app = FastAPI(title="AI Debugging Coach", version="1.5.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def row_to_dict(row) -> Optional[Dict]:
    if row is None:
        return None
    return {k: row[k] for k in row.keys()}


async def fetchrow(sql: str, *args):
    return await db_fetchrow(sql, *args)


async def fetchall(sql: str, *args):
    return await db_fetchall(sql, *args)


async def fetchval(sql: str, *args):
    return await db_fetchval(sql, *args)

# ---------------------------------------------------------------------
# Password hashing — PBKDF2-HMAC-SHA256 (no extra packages needed)
# ---------------------------------------------------------------------
def hash_password(password: str) -> str:
    salt = os.urandom(32)
    key  = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 100_000)
    return binascii.hexlify(salt + key).decode()


def verify_password(stored: str, provided: str) -> bool:
    try:
        raw  = binascii.unhexlify(stored)
        salt = raw[:32]
        key  = hashlib.pbkdf2_hmac("sha256", provided.encode(), salt, 100_000)
        return hmac.compare_digest(raw[32:], key)
    except Exception:
        return False

# ---------------------------------------------------------------------
# Token auth dependency
# ---------------------------------------------------------------------
security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> str:
    token = credentials.credentials
    row = await fetchrow(
        "SELECT user_id, expires_at FROM tokens WHERE token = ?", token
    )
    if not row:
        raise HTTPException(status_code=401, detail="Invalid or expired token. Please log in again.")
    if datetime.fromisoformat(row["expires_at"]) < datetime.now():
        raise HTTPException(status_code=401, detail="Token expired. Please log in again.")
    return row["user_id"]

# ---------------------------------------------------------------------
# Rate limiting (in-memory, per IP, 10 req/60 s on /debug)
# ---------------------------------------------------------------------
_rate_buckets: Dict[str, List[float]] = defaultdict(list)
RATE_LIMIT  = 10
RATE_WINDOW = 60


def check_rate_limit(req: Request):
    ip  = req.client.host if req.client else "unknown"
    now = time.time()
    _rate_buckets[ip] = [t for t in _rate_buckets[ip] if now - t < RATE_WINDOW]
    if len(_rate_buckets[ip]) >= RATE_LIMIT:
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again in a minute.")
    _rate_buckets[ip].append(now)

# ---------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------
class RegisterRequest(BaseModel):
    user_id:  str
    password: str

    @field_validator("user_id")
    @classmethod
    def validate_uid(cls, v):
        v = v.strip().lower()
        if len(v) < 3 or len(v) > 30:
            raise ValueError("Username must be 3–30 characters")
        if not all(c.isalnum() or c in "-_" for c in v):
            raise ValueError("Username may only contain letters, numbers, hyphens, underscores")
        return v

    @field_validator("password")
    @classmethod
    def validate_pwd(cls, v):
        if len(v) < 6:
            raise ValueError("Password must be at least 6 characters")
        return v


class LoginRequest(BaseModel):
    user_id:  str
    password: str


class DebugRequest(BaseModel):
    language: str
    level:    str
    error:    str
    context:  Optional[str] = None

    @field_validator("language")
    @classmethod
    def validate_language(cls, v):
        if not v or not v.strip():
            raise ValueError("Language is required")
        return v.strip().lower()

    @field_validator("level")
    @classmethod
    def validate_level(cls, v):
        valid = ["beginner", "intermediate", "advanced", "expert"]
        v_lower = v.strip().lower()
        if v_lower not in valid:
            raise ValueError(f"Level must be one of: {', '.join(valid)}")
        return v_lower

    @field_validator("error")
    @classmethod
    def validate_error(cls, v):
        if not v or not v.strip():
            raise ValueError("Error description is required")
        if len(v.strip()) > 10000:
            raise ValueError("Error too long (max 10 000 chars)")
        return v.strip()


class PracticeRequest(BaseModel):
    language:      str
    level:         str
    mode:          str
    num_questions: Optional[int] = 5

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v):
        valid = ["quiz", "debug", "code_review"]
        if v not in valid:
            raise ValueError(f"Mode must be one of: {', '.join(valid)}")
        return v


class AnswerRequest(BaseModel):
    session_id:         str
    question_id:        int
    user_answer:        str
    time_taken_seconds: Optional[int] = 0

# ---------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------
async def ensure_user(user_id: str):
    now = datetime.now().isoformat()
    exists = await fetchval("SELECT 1 FROM users WHERE user_id = ?", user_id)
    if not exists:
        await _db.execute(
            "INSERT INTO users (user_id, session_id, created_at, last_activity) VALUES (?,?,?,?)",
            (user_id, str(uuid.uuid4()), now, now)
        )
        await _db.execute(
            "INSERT INTO leaderboard (user_id, last_activity) VALUES (?,?)",
            (user_id, now)
        )
        await _db.commit()


async def user_exists(user_id: str) -> bool:
    return bool(await fetchval("SELECT 1 FROM users WHERE user_id = ?", user_id))


async def increment_queries(user_id: str, language: str, level: str):
    now = datetime.now().isoformat()
    await _db.execute(
        """UPDATE users
           SET total_queries     = total_queries + 1,
               last_activity     = ?,
               favorite_language = COALESCE(favorite_language, ?),
               skill_level       = COALESCE(skill_level, ?)
           WHERE user_id = ?""",
        (now, language, level, user_id)
    )
    await _db.commit()


async def add_to_history(user_id: str, request_type: str, language: str, level: str,
                         query: str, processing_time: float, score: Optional[int] = None):
    truncated = query[:100] + "..." if len(query) > 100 else query
    now = datetime.now().isoformat()
    await _db.execute(
        """INSERT INTO history
           (user_id, timestamp, request_type, language, level, query, processing_time, score)
           VALUES (?,?,?,?,?,?,?,?)""",
        (user_id, now, request_type, language, level, truncated, processing_time, score)
    )
    # Keep only latest 50 entries per user
    await _db.execute(
        """DELETE FROM history
           WHERE user_id = ?
             AND id NOT IN (
                 SELECT id FROM history
                 WHERE user_id = ?
                 ORDER BY timestamp DESC
                 LIMIT 50
             )""",
        (user_id, user_id)
    )
    await _db.commit()


async def update_leaderboard(user_id: str, points: int):
    now = datetime.now().isoformat()
    await _db.execute(
        """UPDATE leaderboard
           SET total_score                 = total_score + ?,
               practice_sessions_completed = practice_sessions_completed + 1,
               last_activity               = ?
           WHERE user_id = ?""",
        (points, now, user_id)
    )
    await _db.commit()

# ---------------------------------------------------------------------
# Gemini API
# ---------------------------------------------------------------------
GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
]


async def call_gemini_api(prompt: str, temperature: float = 0.7, max_tokens: int = 2000) -> Dict[str, Any]:
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not configured")

    headers = {"Content-Type": "application/json", "x-goog-api-key": GEMINI_API_KEY}
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
            "topP": 0.8,
            "topK": 40,
        },
        "safetySettings": [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH",  "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ],
    }

    for model in GEMINI_MODELS:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
            response = await _http_client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            if "candidates" in data and data["candidates"]:
                return {"model": model, "data": data}
        except httpx.HTTPStatusError as e:
            if e.response.status_code in (401, 403):
                raise HTTPException(status_code=500, detail="Invalid Gemini API key")
            logger.warning(f"Model {model} HTTP {e.response.status_code}: {e}")
        except Exception as e:
            logger.warning(f"Model {model} failed: {e}")

    raise HTTPException(status_code=503, detail="All Gemini models unavailable")


def extract_gemini_text(response: Dict[str, Any]) -> str:
    try:
        parts = response["data"]["candidates"][0]["content"].get("parts", [])
        if parts and "text" in parts[0]:
            return parts[0]["text"]
    except Exception as e:
        logger.error(f"Text extraction failed: {e}")
    return ""

# ---------------------------------------------------------------------
# Question templates
# ---------------------------------------------------------------------
QUESTION_TEMPLATES = {
    "python": {
        "beginner": [
            {
                "question": "What will happen when this code runs?",
                "code": "my_list = [1, 2, 3]\nprint(my_list[3])",
                "options": ["A) Prints 3", "B) IndexError: list index out of range", "C) Prints None", "D) Prints 4"],
                "correct_answer": "B",
                "explanation": "List indices start at 0, so index 3 is out of range for a 3-element list.",
                "points": 10,
            },
            {
                "question": "What error will this code produce?",
                "code": "x = 5\ny = \"hello\"\nz = x + y",
                "options": ["A) No error", "B) SyntaxError", "C) TypeError", "D) NameError"],
                "correct_answer": "C",
                "explanation": "You cannot add an integer and a string in Python.",
                "points": 10,
            },
            {
                "question": "What is wrong with this function call?",
                "code": "def greet(name):\n    return f\"Hello, {name}!\"\n\ngreet()",
                "options": ["A) Missing return", "B) Missing argument", "C) Wrong syntax", "D) No error"],
                "correct_answer": "B",
                "explanation": "The function requires a name argument but none was passed.",
                "points": 10,
            },
            {
                "question": "What does this code print?",
                "code": "x = 10\nif x > 5:\n    print(\"big\")\nelse:\n    print(\"small\")",
                "options": ["A) small", "B) big", "C) Nothing", "D) Error"],
                "correct_answer": "B",
                "explanation": "10 > 5 is True, so the if-branch executes and prints 'big'.",
                "points": 10,
            },
        ],
        "intermediate": [
            {
                "question": "What is the issue with this recursive function?",
                "code": "def factorial(n):\n    if n == 0:\n        return 1\n    return n * factorial(n - 1)",
                "options": ["A) Missing base case", "B) Infinite recursion for negatives", "C) Wrong return", "D) No error"],
                "correct_answer": "B",
                "explanation": "For negative inputs the function never reaches 0, causing infinite recursion.",
                "points": 15,
            },
            {
                "question": "What will happen with this dictionary access?",
                "code": "data = {\"name\": \"John\"}\nprint(data[\"email\"])",
                "options": ["A) Prints None", "B) KeyError", "C) Prints empty string", "D) Creates key"],
                "correct_answer": "B",
                "explanation": "Accessing a non-existent key raises KeyError.",
                "points": 15,
            },
            {
                "question": "What is printed?",
                "code": "def f(x, lst=[]):\n    lst.append(x)\n    return lst\n\nprint(f(1))\nprint(f(2))",
                "options": ["A) [1] then [2]", "B) [1] then [1, 2]", "C) Error", "D) [1, 2] both times"],
                "correct_answer": "B",
                "explanation": "Default mutable arguments are shared across calls — a classic Python gotcha.",
                "points": 15,
            },
        ],
    },
    "javascript": {
        "beginner": [
            {
                "question": "What will this code output?",
                "code": "let arr = [1, 2, 3];\nconsole.log(arr[10]);",
                "options": ["A) 10", "B) undefined", "C) Error", "D) null"],
                "correct_answer": "B",
                "explanation": "Accessing a non-existent index returns undefined in JavaScript.",
                "points": 10,
            },
            {
                "question": "What is the output?",
                "code": "console.log(typeof null);",
                "options": ["A) \"null\"", "B) \"object\"", "C) \"undefined\"", "D) Error"],
                "correct_answer": "B",
                "explanation": "typeof null === 'object' is a long-standing JavaScript quirk.",
                "points": 10,
            },
        ],
        "intermediate": [
            {
                "question": "What does this log?",
                "code": "console.log(1 + \"2\" + 3);",
                "options": ["A) 6", "B) \"123\"", "C) \"15\"", "D) NaN"],
                "correct_answer": "B",
                "explanation": "1 + '2' coerces to '12', then '12' + 3 = '123'.",
                "points": 15,
            },
        ],
    },
    "java": {
        "beginner": [
            {
                "question": "What will this code produce?",
                "code": "String x = null;\nSystem.out.println(x.length());",
                "options": ["A) 0", "B) NullPointerException", "C) Compilation error", "D) null"],
                "correct_answer": "B",
                "explanation": "Calling a method on null throws NullPointerException.",
                "points": 10,
            },
        ],
    },
}


def generate_from_templates(language: str, level: str, mode: str, count: int) -> List[Dict]:
    templates = QUESTION_TEMPLATES.get(language.lower(), {}).get(level.lower(), [])
    if not templates:
        for fallback in ["beginner", "intermediate", "advanced"]:
            templates = QUESTION_TEMPLATES.get(language.lower(), {}).get(fallback, [])
            if templates:
                break
    if not templates:
        templates = [{
            "question": f"What is important in {language} programming?",
            "code": f"// {language} example",
            "options": ["A) Syntax", "B) Logic", "C) Testing", "D) All of the above"],
            "correct_answer": "D",
            "explanation": f"All aspects are important in {language} development.",
            "points": 10,
        }]

    if len(templates) >= count:
        chosen = random.sample(templates, count)
    else:
        shuffled = random.sample(templates, len(templates))
        chosen = []
        while len(chosen) < count:
            chosen.extend(shuffled)
        chosen = chosen[:count]

    return [{**t, "id": i + 1} for i, t in enumerate(chosen)]


async def generate_questions_with_gemini(language: str, level: str, mode: str, count: int) -> Optional[List[Dict]]:
    prompt = f"""Generate {count} programming practice questions for {language} at {level} level.

Return ONLY a valid JSON array:
[
  {{
    "question": "What will happen?",
    "code": "example code",
    "options": ["A) Option 1", "B) Option 2", "C) Option 3", "D) Option 4"],
    "correct_answer": "A",
    "explanation": "Explanation here",
    "points": 10
  }}
]

Requirements:
- Exactly 4 options labeled A, B, C, D
- correct_answer must be A, B, C, or D
- Realistic {language} code snippets
- {level} difficulty, {mode} style"""

    try:
        response = await call_gemini_api(prompt, temperature=0.3)
        text = extract_gemini_text(response)
        if not text:
            return None
        cleaned = text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            start = next((i for i, ln in enumerate(lines) if "[" in ln), 0)
            end   = next((i for i in range(len(lines) - 1, -1, -1) if "]" in lines[i]), len(lines))
            cleaned = "\n".join(lines[start:end + 1])
        cleaned = cleaned.replace("```json", "").replace("```", "").strip()
        questions = json.loads(cleaned)
        if not isinstance(questions, list):
            return None
        validated = []
        for i, q in enumerate(questions[:count], 1):
            if not isinstance(q, dict):
                continue
            vq = {
                "id": i,
                "question": str(q.get("question", "")).strip(),
                "code": str(q.get("code", "")).strip(),
                "options": q.get("options", []),
                "correct_answer": str(q.get("correct_answer", "A")).upper().strip(),
                "explanation": str(q.get("explanation", "")).strip(),
                "points": int(q.get("points", 10)),
            }
            if len(vq["options"]) != 4:
                continue
            if vq["correct_answer"] not in ("A", "B", "C", "D"):
                vq["correct_answer"] = "A"
            validated.append(vq)
        return validated if validated else None
    except Exception as e:
        logger.error(f"Gemini generation failed: {e}")
        return None


async def generate_practice_questions(language: str, level: str, mode: str, count: int) -> List[Dict]:
    gemini_qs = await generate_questions_with_gemini(language, level, mode, count)
    if gemini_qs and len(gemini_qs) >= count:
        return gemini_qs[:count]
    template_qs = generate_from_templates(language, level, mode, count)
    if gemini_qs:
        extra = generate_from_templates(language, level, mode, count - len(gemini_qs))
        for i, q in enumerate(extra):
            q["id"] = len(gemini_qs) + i + 1
        return (gemini_qs + extra)[:count]
    return template_qs[:count]


def create_debug_prompt(language: str, level: str, error: str, context: Optional[str]) -> str:
    instructions = {
        "beginner":     "Use simple terms and explain basic concepts clearly.",
        "intermediate": "Provide detailed explanations with moderate technical depth.",
        "advanced":     "Use technical language and focus on best practices.",
        "expert":       "Provide concise, technical solutions with performance considerations.",
    }
    prompt = f"""You are an expert debugging coach for {language.title()} programming.

PROGRAMMER LEVEL: {level.title()}
INSTRUCTIONS: {instructions.get(level, instructions['intermediate'])}

"""
    if context:
        prompt += f"CONTEXT: {context}\n\n"
    prompt += f"""ERROR/ISSUE:
{error}

Please provide debugging assistance in this format:

## Error Analysis
[Explain what the error means and why it occurs]

## Step-by-Step Debugging
1. [First debugging step]
2. [Second debugging step]
3. [Additional steps as needed]

## Most Likely Solution
[Provide the probable fix with code examples]

## Prevention Tips
[How to avoid this issue in the future]
"""
    return prompt

# ---------------------------------------------------------------------
# Auth routes  (public — no token needed)
# ---------------------------------------------------------------------
async def create_token_pair(user_id: str) -> Dict:
    now             = datetime.now()
    access_tok      = str(uuid.uuid4())
    refresh_tok     = str(uuid.uuid4())
    access_expires  = (now + timedelta(hours=ACCESS_TOKEN_EXPIRY_HOURS)).isoformat()
    refresh_expires = (now + timedelta(days=REFRESH_TOKEN_EXPIRY_DAYS)).isoformat()
    now_str         = now.isoformat()
    await _db.execute("DELETE FROM tokens WHERE user_id = ?", (user_id,))
    await _db.execute(
        "INSERT INTO tokens (token, user_id, token_type, created_at, expires_at) VALUES (?,?,?,?,?)",
        (access_tok, user_id, 'access', now_str, access_expires)
    )
    await _db.execute(
        "INSERT INTO tokens (token, user_id, token_type, created_at, expires_at) VALUES (?,?,?,?,?)",
        (refresh_tok, user_id, 'refresh', now_str, refresh_expires)
    )
    await _db.commit()
    return {"token": access_tok, "refresh_token": refresh_tok,
            "user_id": user_id, "expires_at": access_expires}


@app.post("/auth/register", status_code=201)
async def register(request: RegisterRequest):
    taken = await fetchval("SELECT 1 FROM auth WHERE user_id = ?", request.user_id)
    if taken:
        raise HTTPException(status_code=409, detail="Username already taken")
    now = datetime.now().isoformat()
    await _db.execute(
        "INSERT INTO auth (user_id, password_hash, created_at) VALUES (?,?,?)",
        (request.user_id, hash_password(request.password), now)
    )
    await _db.commit()
    await ensure_user(request.user_id)
    logger.info(f"Registered: {request.user_id}")
    return await create_token_pair(request.user_id)


@app.post("/auth/login")
async def login(request: LoginRequest):
    row = await fetchrow("SELECT password_hash FROM auth WHERE user_id = ?", request.user_id)
    if not row or not verify_password(row["password_hash"], request.password):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    logger.info(f"Login: {request.user_id}")
    return await create_token_pair(request.user_id)


@app.post("/auth/refresh")
async def refresh_token(request: Request):
    body        = await request.json()
    refresh_tok = body.get("refresh_token")
    if not refresh_tok:
        raise HTTPException(status_code=400, detail="Missing refresh token")
    row = await fetchrow(
        "SELECT user_id, expires_at, token_type FROM tokens WHERE token = ?", refresh_tok
    )
    if not row or row["token_type"] != "refresh":
        raise HTTPException(status_code=401, detail="Invalid refresh token")
    if datetime.fromisoformat(row["expires_at"]) < datetime.now():
        raise HTTPException(status_code=401, detail="Session expired. Please log in again.")
    user_id      = row["user_id"]
    access_tok   = str(uuid.uuid4())
    now          = datetime.now()
    access_exp   = (now + timedelta(hours=ACCESS_TOKEN_EXPIRY_HOURS)).isoformat()
    await _db.execute(
        "DELETE FROM tokens WHERE user_id = ? AND token_type = 'access'", (user_id,)
    )
    await _db.execute(
        "INSERT INTO tokens (token, user_id, token_type, created_at, expires_at) VALUES (?,?,?,?,?)",
        (access_tok, user_id, 'access', now.isoformat(), access_exp)
    )
    await _db.commit()
    return {"token": access_tok, "expires_at": access_exp}


@app.post("/auth/logout")
async def logout(
    current_user: str = Depends(get_current_user),
):
    await _db.execute("DELETE FROM tokens WHERE user_id = ?", (current_user,))
    await _db.commit()
    return {"message": "Logged out successfully"}


@app.post("/auth/google")
async def google_login(request: Request):
    body       = await request.json()
    credential = body.get("credential")
    if not credential:
        raise HTTPException(status_code=400, detail="Missing Google credential")
    if not GOOGLE_CLIENT_ID:
        raise HTTPException(status_code=500, detail="GOOGLE_CLIENT_ID not configured")
    try:
        id_info = id_token.verify_oauth2_token(
            credential, google_requests.Request(), GOOGLE_CLIENT_ID
        )
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid Google token")
    email   = id_info.get("email", "")
    raw_id  = email.split("@")[0].lower()
    user_id = re.sub(r'[^a-z0-9_-]', '_', raw_id)[:30]
    exists  = await fetchval("SELECT 1 FROM auth WHERE user_id = ?", user_id)
    if not exists:
        now = datetime.now().isoformat()
        await _db.execute(
            "INSERT INTO auth (user_id, password_hash, created_at) VALUES (?,?,?)",
            (user_id, "google:" + id_info["sub"], now)
        )
        await _db.commit()
        await ensure_user(user_id)
    logger.info(f"Google login: {user_id}")
    return await create_token_pair(user_id)

# ---------------------------------------------------------------------
# Public routes
# ---------------------------------------------------------------------
@app.get("/")
async def root():
    return {"message": "AI Debugging Coach API", "version": "1.5.0", "status": "operational"}


@app.get("/app")
async def serve_frontend():
    return FileResponse(Path(__file__).parent / "index.html")


@app.get("/health")
async def health_check():
    total_users     = await fetchval("SELECT COUNT(*) FROM users")
    active_sessions = await fetchval("SELECT COUNT(*) FROM practice_sessions WHERE completed_at IS NULL")
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.5.0",
        "total_users": total_users or 0,
        "active_sessions": active_sessions or 0,
    }


@app.get("/leaderboard")
async def get_leaderboard(limit: int = 10):
    limit = min(max(1, limit), 100)
    rows  = await fetchall("SELECT * FROM leaderboard ORDER BY total_score DESC LIMIT ?", limit)
    total = await fetchval("SELECT COUNT(*) FROM leaderboard")
    return {
        "leaderboard": [row_to_dict(r) for r in rows],
        "total_users": total or 0,
        "last_updated": datetime.now().isoformat(),
    }

# ---------------------------------------------------------------------
# Protected routes  (require Authorization: Bearer <token>)
# ---------------------------------------------------------------------
@app.post("/debug")
async def debug_endpoint(
    request:      DebugRequest,
    current_user: str = Depends(get_current_user),
    _=Depends(check_rate_limit),
):
    start_time = datetime.now()
    logger.info(f"Debug: user={current_user} lang={request.language} level={request.level}")
    try:
        await increment_queries(current_user, request.language, request.level)
        prompt   = create_debug_prompt(request.language, request.level, request.error, request.context)
        response = await call_gemini_api(prompt)
        answer   = extract_gemini_text(response)
        if not answer or len(answer.strip()) < 50:
            raise HTTPException(status_code=502, detail="AI response too short or empty")
        processing_time = (datetime.now() - start_time).total_seconds()
        await add_to_history(current_user, "debug", request.language, request.level,
                             request.error, processing_time)
        return {
            "response": answer,
            "model_used": response["model"],
            "processing_time_seconds": round(processing_time, 2),
            "timestamp": datetime.now().isoformat(),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Debug error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/practice/generate")
async def generate_practice_endpoint(
    request:      PracticeRequest,
    current_user: str = Depends(get_current_user),
):
    start_time = datetime.now()
    session_id = str(uuid.uuid4())
    try:
        count     = max(1, min(request.num_questions or 5, 10))
        questions = await generate_practice_questions(request.language, request.level, request.mode, count)
        if not questions:
            raise HTTPException(status_code=500, detail="Failed to generate questions")
        total_pts = sum(q.get("points", 10) for q in questions)
        await _db.execute(
            """INSERT INTO practice_sessions
               (session_id, user_id, language, level, mode, questions, started_at, total_possible_points)
               VALUES (?,?,?,?,?,?,?,?)""",
            (session_id, current_user, request.language, request.level,
             request.mode, json.dumps(questions), datetime.now().isoformat(), total_pts)
        )
        await _db.commit()
        processing_time = (datetime.now() - start_time).total_seconds()
        await add_to_history(current_user, "practice_generate", request.language, request.level,
                             f"Generated {len(questions)} questions", processing_time)
        return {
            "session_id": session_id,
            "questions": questions,
            "total_questions": len(questions),
            "total_possible_points": total_pts,
            "processing_time_seconds": round(processing_time, 2),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Practice generation error: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate practice session")


@app.post("/practice/answer")
async def answer_endpoint(
    request:      AnswerRequest,
    current_user: str = Depends(get_current_user),
):
    row = await fetchrow("SELECT * FROM practice_sessions WHERE session_id = ?", request.session_id)
    if not row:
        raise HTTPException(status_code=404, detail="Practice session not found")
    if row["user_id"] != current_user:
        raise HTTPException(status_code=403, detail="Unauthorized")

    questions = json.loads(row["questions"])
    answers   = json.loads(row["answers"])
    question  = next((q for q in questions if q["id"] == request.question_id), None)
    if not question:
        raise HTTPException(status_code=404, detail="Question not found")
    if str(request.question_id) in answers:
        raise HTTPException(status_code=400, detail="Question already answered")

    user_answer    = request.user_answer.upper().strip()
    correct_answer = question.get("correct_answer", "A").upper().strip()
    correct        = user_answer == correct_answer
    points_earned  = question.get("points", 10) if correct else 0

    answers[str(request.question_id)] = {
        "user_answer": user_answer,
        "correct": correct,
        "points_earned": points_earned,
        "time_taken": request.time_taken_seconds,
        "answered_at": datetime.now().isoformat(),
    }
    new_score         = row["total_score"] + points_earned
    session_completed = len(answers) == len(questions)
    completed_at      = datetime.now().isoformat() if session_completed else None

    await _db.execute(
        "UPDATE practice_sessions SET answers=?, total_score=?, completed_at=? WHERE session_id=?",
        (json.dumps(answers), new_score, completed_at, request.session_id)
    )
    await _db.commit()

    if session_completed:
        await update_leaderboard(current_user, new_score)
        await add_to_history(current_user, "practice_complete", row["language"],
                             row["level"], f"Completed: {new_score} pts", 0, new_score)
    return {
        "correct": correct,
        "points_earned": points_earned,
        "correct_answer": correct_answer,
        "explanation": question.get("explanation", "No explanation available"),
        "current_score": new_score,
        "questions_answered": len(answers),
        "total_questions": len(questions),
        "session_completed": session_completed,
    }


@app.get("/user/{user_id}/session")
async def get_user_session(user_id: str, current_user: str = Depends(get_current_user)):
    if current_user != user_id:
        raise HTTPException(status_code=403, detail="Cannot access another user's session")
    if not await user_exists(user_id):
        raise HTTPException(status_code=404, detail="User not found")
    session      = row_to_dict(await fetchrow("SELECT * FROM users WHERE user_id = ?", user_id))
    lb           = await fetchrow("SELECT total_score, practice_sessions_completed FROM leaderboard WHERE user_id = ?", user_id)
    debug_count  = await fetchval("SELECT COUNT(*) FROM history WHERE user_id=? AND request_type='debug'", user_id)
    practice_count = await fetchval("SELECT COUNT(*) FROM history WHERE user_id=? AND request_type='practice_complete'", user_id)
    return {
        "session": session,
        "statistics": {
            "total_queries": session["total_queries"],
            "debug_queries": debug_count or 0,
            "practice_sessions": practice_count or 0,
            "total_practice_score": lb["total_score"] if lb else 0,
            "favorite_language": session.get("favorite_language"),
            "skill_level": session.get("skill_level"),
        },
    }


@app.get("/user/{user_id}/dashboard")
async def get_dashboard(user_id: str, current_user: str = Depends(get_current_user)):
    if current_user != user_id:
        raise HTTPException(status_code=403, detail="Cannot access another user's dashboard")
    user = row_to_dict(await fetchrow("SELECT * FROM users WHERE user_id = ?", user_id))
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    lb          = await fetchrow("SELECT total_score, practice_sessions_completed FROM leaderboard WHERE user_id = ?", user_id)
    debug_count = await fetchval("SELECT COUNT(*) FROM history WHERE user_id=? AND request_type='debug'", user_id)
    sessions    = await fetchall(
        "SELECT * FROM practice_sessions WHERE user_id=? AND completed_at IS NOT NULL ORDER BY completed_at DESC",
        user_id
    )

    total_q = total_correct = total_time = streak = 0
    lang_stats: Dict[str, Dict] = {}
    score_history = []

    for s in sessions:
        sd      = row_to_dict(s)
        answers = json.loads(sd["answers"])
        lang    = sd["language"]
        s_corr  = sum(1 for a in answers.values() if a.get("correct"))
        s_tot   = len(answers)
        s_time  = sum(a.get("time_taken", 0) or 0 for a in answers.values())
        total_q       += s_tot
        total_correct += s_corr
        total_time    += s_time
        lang_stats.setdefault(lang, {"correct": 0, "total": 0})
        lang_stats[lang]["correct"] += s_corr
        lang_stats[lang]["total"]   += s_tot
        if len(score_history) < 10:
            possible = sd["total_possible_points"] or 1
            score_history.append({
                "date": sd["completed_at"][:10],
                "score": sd["total_score"],
                "possible": sd["total_possible_points"],
                "pct": round(sd["total_score"] / possible * 100),
                "language": lang,
                "level": sd["level"],
            })

    # Consecutive-session streak (≥70% accuracy)
    for s in sessions:
        sd      = row_to_dict(s)
        answers = json.loads(sd["answers"])
        s_tot   = len(answers)
        s_corr  = sum(1 for a in answers.values() if a.get("correct"))
        if s_tot and s_corr / s_tot >= 0.7:
            streak += 1
        else:
            break

    accuracy = round(total_correct / total_q * 100) if total_q else 0

    weak_topics = sorted(
        [{"language": lang, "accuracy": round(v["correct"] / v["total"] * 100)}
         for lang, v in lang_stats.items() if v["total"] > 0 and round(v["correct"] / v["total"] * 100) < 65],
        key=lambda x: x["accuracy"]
    )[:3]

    # Difficulty recommendation
    levels      = ["beginner", "intermediate", "advanced", "expert"]
    curr_level  = user.get("skill_level") or "beginner"
    rec_level   = curr_level
    if len(sessions) >= 3:
        avg_pct = sum(
            row_to_dict(s)["total_score"] / max(row_to_dict(s)["total_possible_points"], 1) * 100
            for s in sessions[:3]
        ) / 3
        idx = levels.index(curr_level) if curr_level in levels else 0
        if avg_pct >= 85 and idx < len(levels) - 1:
            rec_level = levels[idx + 1]
        elif avg_pct < 40 and idx > 0:
            rec_level = levels[idx - 1]

    return {
        "user_id": user_id,
        "joined": user["created_at"][:10],
        "stats": {
            "total_questions":    total_q,
            "total_correct":      total_correct,
            "accuracy":           accuracy,
            "total_score":        lb["total_score"] if lb else 0,
            "sessions_completed": lb["practice_sessions_completed"] if lb else 0,
            "debug_queries":      debug_count or 0,
            "streak":             streak,
            "practice_time_mins": round(total_time / 60),
            "favorite_language":  user.get("favorite_language"),
        },
        "weak_topics":       weak_topics,
        "score_history":     score_history,
        "recommended_level": rec_level,
        "recent_sessions":   [row_to_dict(s) for s in sessions[:5]],
    }


@app.get("/user/{user_id}/history")
async def get_user_history(user_id: str, limit: int = 20, current_user: str = Depends(get_current_user)):
    if current_user != user_id:
        raise HTTPException(status_code=403, detail="Cannot access another user's history")
    limit = min(max(1, limit), 100)
    rows  = await fetchall("SELECT * FROM history WHERE user_id=? ORDER BY timestamp DESC LIMIT ?", user_id, limit)
    total = await fetchval("SELECT COUNT(*) FROM history WHERE user_id=?", user_id)
    return {"history": [row_to_dict(r) for r in rows], "total_entries": total or 0}


@app.delete("/user/{user_id}/reset")
async def reset_user_data(user_id: str, current_user: str = Depends(get_current_user)):
    if current_user != user_id:
        raise HTTPException(status_code=403, detail="Cannot reset another user's data")
    for table in ("users", "history", "leaderboard", "practice_sessions", "auth", "tokens"):
        await _db.execute(f"DELETE FROM {table} WHERE user_id = ?", (user_id,))
    await _db.commit()
    return {"message": f"All data for {user_id} has been reset"}

# ---------------------------------------------------------------------
# File Review
# ---------------------------------------------------------------------
_EXT_LANG = {
    ".py": "python", ".js": "javascript", ".ts": "typescript",
    ".jsx": "javascript", ".tsx": "typescript",
    ".java": "java", ".cpp": "c++", ".cc": "c++", ".cxx": "c++",
    ".cs": "csharp", ".go": "go", ".rs": "rust", ".php": "php",
    ".rb": "ruby", ".swift": "swift", ".kt": "kotlin",
    ".html": "html", ".css": "css", ".sql": "sql", ".sh": "bash",
    ".c": "c", ".h": "c",
}

@app.post("/review/file")
async def review_file(
    file: UploadFile = File(...),
    model: str = Form(default="gemini-2.5-flash"),
    current_user: str = Depends(get_current_user),
):
    raw = await file.read()
    if len(raw) > 150_000:
        raise HTTPException(status_code=400, detail="File too large (max 150 KB)")

    filename = file.filename or "unknown"
    ext      = Path(filename).suffix.lower()
    language = _EXT_LANG.get(ext, "code")

    try:
        source = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File must be valid UTF-8 text")

    if not source.strip():
        raise HTTPException(status_code=400, detail="File is empty")

    prompt = f"""You are a senior software engineer performing a thorough code review.
Review the {language} code from file "{filename}" below.

Respond with these exact Markdown sections (use the headings verbatim):

## Summary
One short paragraph: what the code does and overall quality impression.

## Bugs & Errors
Numbered list of concrete bugs or logic errors. Include approximate line numbers.
If none found write: No bugs found.

## Security Issues
Numbered list of security vulnerabilities (injection, secrets, unsafe ops, etc.).
If none write: No security issues found.

## Performance Issues
Numbered list of bottlenecks or inefficiencies.
If none write: No performance issues found.

## Code Quality & Improvements
Numbered list of style, readability, and maintainability suggestions.

## Overall Score
Rate the code X/10 and give one sentence of justification.

---
```{language}
{source}
```"""

    model_id = model if model in GEMINI_MODELS else GEMINI_MODELS[0]
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={GEMINI_API_KEY}"
    t0  = time.time()
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(url, json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.3, "maxOutputTokens": 4096},
            })
        resp.raise_for_status()
        review_text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Gemini error: {str(e)}")

    elapsed = round(time.time() - t0, 2)

    await _db.execute(
        "INSERT INTO history (user_id, request_type, language, level, query, processing_time, score, timestamp) VALUES (?,?,?,?,?,?,?,?)",
        (current_user, "file_review", language, "n/a", filename, elapsed, 0, datetime.now().isoformat()),
    )
    await _db.commit()

    return {
        "filename": filename,
        "language": language,
        "model_used": model_id,
        "processing_time_seconds": elapsed,
        "lines": source.count("\n") + 1,
        "review": review_text,
    }


# ---------------------------------------------------------------------
# GitHub URL Review
# ---------------------------------------------------------------------
class UrlReviewRequest(BaseModel):
    url: str
    model: str = "gemini-2.5-flash"


def _github_to_raw(url: str) -> str:
    """Convert a github.com blob URL to raw.githubusercontent.com."""
    # Already raw
    if "raw.githubusercontent.com" in url:
        return url
    # https://github.com/user/repo/blob/branch/path → raw
    m = re.match(
        r"https?://github\.com/([^/]+)/([^/]+)/blob/(.+)",
        url.strip()
    )
    if m:
        return f"https://raw.githubusercontent.com/{m.group(1)}/{m.group(2)}/{m.group(3)}"
    return ""


@app.post("/review/url")
async def review_url(request: UrlReviewRequest, current_user: str = Depends(get_current_user)):
    raw_url = _github_to_raw(request.url)
    if not raw_url:
        raise HTTPException(status_code=400, detail="Only github.com file URLs are supported (blob or raw links)")

    # Fetch the file
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(raw_url, follow_redirects=True,
                                    headers={"User-Agent": "bugSAGE/1.0"})
        if resp.status_code == 404:
            raise HTTPException(status_code=404, detail="File not found — make sure the URL points to a public file")
        if resp.status_code == 403:
            raise HTTPException(status_code=403, detail="Access denied — private repositories are not supported")
        resp.raise_for_status()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to fetch file: {str(e)}")

    raw_bytes = resp.content
    if len(raw_bytes) > 150_000:
        raise HTTPException(status_code=400, detail="File too large (max 150 KB)")

    try:
        source = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File does not appear to be UTF-8 text")

    if not source.strip():
        raise HTTPException(status_code=400, detail="File is empty")

    filename = raw_url.split("/")[-1]
    ext      = Path(filename).suffix.lower()
    language = _EXT_LANG.get(ext, "code")

    prompt = f"""You are a senior software engineer performing a thorough code review.
Review the {language} code from file "{filename}" (GitHub source).

Respond with these exact Markdown sections (use the headings verbatim):

## Summary
One short paragraph: what the code does and overall quality impression.

## Bugs & Errors
Numbered list of concrete bugs or logic errors. Include approximate line numbers.
If none found write: No bugs found.

## Security Issues
Numbered list of security vulnerabilities (injection, secrets, unsafe ops, etc.).
If none write: No security issues found.

## Performance Issues
Numbered list of bottlenecks or inefficiencies.
If none write: No performance issues found.

## Code Quality & Improvements
Numbered list of style, readability, and maintainability suggestions.

## Overall Score
Rate the code X/10 and give one sentence of justification.

---
```{language}
{source}
```"""

    model_id = request.model if request.model in GEMINI_MODELS else GEMINI_MODELS[0]
    gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={GEMINI_API_KEY}"
    t0 = time.time()
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            gr = await client.post(gemini_url, json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.3, "maxOutputTokens": 4096},
            })
        gr.raise_for_status()
        review_text = gr.json()["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Gemini error: {str(e)}")

    elapsed = round(time.time() - t0, 2)

    await _db.execute(
        "INSERT INTO history (user_id, request_type, language, level, query, processing_time, score, timestamp) VALUES (?,?,?,?,?,?,?,?)",
        (current_user, "url_review", language, "n/a", request.url[:200], elapsed, 0, datetime.now().isoformat()),
    )
    await _db.commit()

    return {
        "filename": filename,
        "language": language,
        "url": request.url,
        "raw_url": raw_url,
        "model_used": model_id,
        "processing_time_seconds": elapsed,
        "lines": source.count("\n") + 1,
        "review": review_text,
    }


# ---------------------------------------------------------------------
# Diff Review
# ---------------------------------------------------------------------
class DiffReviewRequest(BaseModel):
    diff: str
    model: str = "gemini-2.5-flash"


def _parse_diff_meta(diff: str) -> Dict:
    """Extract files changed, lines added/removed from a unified diff."""
    files, added, removed = [], 0, 0
    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            path = line[6:].strip()
            if path not in files:
                files.append(path)
        elif line.startswith("+") and not line.startswith("+++"):
            added += 1
        elif line.startswith("-") and not line.startswith("---"):
            removed += 1
    return {"files": files, "added": added, "removed": removed}


@app.post("/review/diff")
async def review_diff(request: DiffReviewRequest, current_user: str = Depends(get_current_user)):
    diff = request.diff.strip()
    if not diff:
        raise HTTPException(status_code=400, detail="Diff is empty")
    if len(diff) > 200_000:
        raise HTTPException(status_code=400, detail="Diff too large (max 200 KB)")
    if not any(line.startswith("@@") or line.startswith("---") or line.startswith("+++") for line in diff.splitlines()):
        raise HTTPException(status_code=400, detail="Does not look like a valid unified diff — paste the output of `git diff`")

    meta = _parse_diff_meta(diff)
    files_str = ", ".join(meta["files"][:6]) or "unknown files"
    langs = list({_EXT_LANG.get("." + p.split(".")[-1].lower(), "") for p in meta["files"] if "." in p} - {""})
    lang_str = ", ".join(langs) if langs else "code"

    prompt = f"""You are a senior software engineer reviewing a pull request / code change.
Below is a unified diff touching {len(meta['files'])} file(s): {files_str}
+{meta['added']} lines added, -{meta['removed']} lines removed.

Review ONLY the changes shown in the diff. Focus on what was added or modified.

Respond with these exact Markdown sections:

## Summary of Changes
2-3 sentences: what this diff does, its apparent purpose.

## Bugs Introduced
Numbered list of bugs or logic errors in the new/changed code. Be specific — include file name and approximate line. If none write: No bugs found.

## Security Issues
Numbered list of security vulnerabilities introduced by these changes. If none write: No security issues found.

## Code Quality
Numbered list of style, readability, or maintainability issues in the changed lines.

## What's Done Well
1-3 things that are good about this change.

## Verdict
One of: ✅ Approve · ⚠ Approve with minor changes · ❌ Request changes
Follow with one sentence explaining why.

---
{diff}"""

    model_id = request.model if request.model in GEMINI_MODELS else GEMINI_MODELS[0]
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={GEMINI_API_KEY}"
    t0 = time.time()
    try:
        async with httpx.AsyncClient(timeout=90) as client:
            resp = await client.post(url, json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.2, "maxOutputTokens": 4096},
            })
        resp.raise_for_status()
        review_text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Gemini error: {str(e)}")

    elapsed = round(time.time() - t0, 2)

    await _db.execute(
        "INSERT INTO history (user_id, request_type, language, level, query, processing_time, score, timestamp) VALUES (?,?,?,?,?,?,?,?)",
        (current_user, "diff_review", lang_str, "n/a",
         f"{len(meta['files'])} files +{meta['added']} -{meta['removed']}", elapsed, 0,
         datetime.now().isoformat()),
    )
    await _db.commit()

    return {
        "files_changed": meta["files"],
        "lines_added": meta["added"],
        "lines_removed": meta["removed"],
        "language": lang_str,
        "model_used": model_id,
        "processing_time_seconds": elapsed,
        "review": review_text,
    }


# ---------------------------------------------------------------------
# Interview Mode
# ---------------------------------------------------------------------
class InterviewQuestionsRequest(BaseModel):
    language: str
    level: str
    mode: str        # coding | debugging | conceptual
    num_questions: int = 5
    model: str = "gemini-2.5-flash"


class InterviewGradeRequest(BaseModel):
    question: str
    answer: str
    language: str
    level: str
    mode: str
    model: str = "gemini-2.5-flash"


class InterviewFinishRequest(BaseModel):
    language: str
    level: str
    mode: str
    scores: List[int]          # one per question
    total_possible: int


async def _call_gemini(prompt: str, model: str, temperature: float = 0.4,
                       max_tokens: int = 4096, timeout: int = 60) -> str:
    model_id = model if model in GEMINI_MODELS else GEMINI_MODELS[0]
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{model_id}:generateContent?key={GEMINI_API_KEY}")
    for attempt in range(4):
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens},
            })
        if resp.status_code == 429:
            wait = int(resp.headers.get("Retry-After", 2 ** (attempt + 1)))
            logger.warning("Gemini rate limit hit, retrying in %ss (attempt %s)", wait, attempt + 1)
            await asyncio.sleep(wait)
            continue
        resp.raise_for_status()
        return resp.json()["candidates"][0]["content"]["parts"][0]["text"]
    raise HTTPException(status_code=429, detail="Gemini rate limit reached. Wait a moment and try again.")


@app.post("/interview/questions")
async def generate_questions(request: InterviewQuestionsRequest,
                             current_user: str = Depends(get_current_user)):
    count = max(1, min(request.num_questions, 10))
    mode_desc = {
        "coding":     "require writing or completing code in " + request.language,
        "debugging":  "provide a short buggy " + request.language + " snippet and ask the candidate to find and fix the bug",
        "conceptual": "test theoretical knowledge of " + request.language + " concepts, patterns, and best practices",
    }.get(request.mode, "test " + request.language + " knowledge")

    prompt = f"""Generate exactly {count} {request.level}-level technical interview questions for a {request.language} developer.
All questions should {mode_desc}.
Make each question distinct, covering different topics.

Return ONLY a valid JSON array with this exact structure (no markdown, no explanation):
[
  {{
    "question": "Full question text here",
    "hint": "One short hint (10 words max) without giving away the answer",
    "topic": "Short topic tag e.g. recursion, closures, SQL injection"
  }}
]"""

    try:
        raw = await _call_gemini(prompt, request.model, temperature=0.7)
        # Strip markdown code fences if present
        raw = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
        questions = json.loads(raw)
        if not isinstance(questions, list):
            raise ValueError("not a list")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Could not generate questions: {str(e)}")

    return {"questions": questions[:count], "language": request.language,
            "level": request.level, "mode": request.mode}


@app.post("/interview/grade")
async def grade_answer(request: InterviewGradeRequest,
                       current_user: str = Depends(get_current_user)):
    if not request.answer.strip():
        return {"score": 0, "feedback": "No answer provided.",
                "model_answer": "", "strong_points": [], "weak_points": ["No answer given"]}

    prompt = f"""You are a strict but fair technical interviewer grading a {request.level} {request.language} developer interview answer.

Question ({request.mode}):
{request.question}

Candidate's Answer:
{request.answer}

Grade the answer. Respond with these exact Markdown sections:

## Score
X/10  (integer only, no fractions)

## Strong Points
Bullet list of what the candidate did well. If nothing, write: Nothing notable.

## Weak Points / Missing
Bullet list of important things missing or wrong. Be specific.

## Model Answer
A complete, correct answer a strong {request.level} candidate would give.

## One-line Summary
One sentence verdict on this answer."""

    try:
        raw = await _call_gemini(prompt, request.model, temperature=0.2)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Grading failed: {str(e)}")

    # Parse score
    score = 0
    sm = re.search(r"##\s*Score\s*\n\s*(\d{1,2})", raw)
    if sm:
        score = max(0, min(10, int(sm.group(1))))

    def extract(heading: str) -> str:
        m = re.search(rf"##\s*{re.escape(heading)}\s*\n([\s\S]*?)(?=##|$)", raw, re.I)
        return m.group(1).strip() if m else ""

    return {
        "score":        score,
        "strong_points": extract("Strong Points"),
        "weak_points":   extract("Weak Points / Missing"),
        "model_answer":  extract("Model Answer"),
        "summary":       extract("One-line Summary"),
        "full_review":   raw,
    }


@app.post("/interview/finish")
async def finish_interview(request: InterviewFinishRequest,
                           current_user: str = Depends(get_current_user)):
    total = sum(request.scores)
    pct = round(total / max(request.total_possible, 1) * 100)
    now = datetime.now().isoformat()
    session_id = str(uuid.uuid4())
    answers_json = json.dumps({str(i): {"score": s, "correct": s >= 6}
                               for i, s in enumerate(request.scores)})
    await _db.execute(
        """INSERT INTO practice_sessions
           (session_id, user_id, language, level, mode, questions, answers,
            started_at, completed_at, total_score, total_possible_points)
           VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        (session_id, current_user, request.language, request.level,
         "interview_" + request.mode, json.dumps([]), answers_json,
         now, now, total, request.total_possible)
    )
    # Update leaderboard
    existing = await fetchrow("SELECT total_score, practice_sessions_completed FROM leaderboard WHERE user_id=?", current_user)
    if existing:
        await _db.execute(
            "UPDATE leaderboard SET total_score=total_score+?, practice_sessions_completed=practice_sessions_completed+1, last_activity=? WHERE user_id=?",
            (total, now, current_user)
        )
    else:
        await _db.execute(
            "INSERT INTO leaderboard (user_id, total_score, practice_sessions_completed, last_activity) VALUES (?,?,?,?)",
            (current_user, total, 1, now)
        )
    await _db.commit()
    return {"total_score": total, "total_possible": request.total_possible,
            "percentage": pct, "session_id": session_id}


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
