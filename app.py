from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
import os
import httpx
import logging
import json
import random
from datetime import datetime, timedelta
from dotenv import load_dotenv
from pathlib import Path
from typing import Optional, List, Dict, Any
import uuid

# ---------------------------------------------------------------------
# Env + Logging
# ---------------------------------------------------------------------
ENV_PATH = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("debug_coach")

# ---------------------------------------------------------------------
# FastAPI + CORS
# ---------------------------------------------------------------------
app = FastAPI(title="AI Debugging Coach", version="1.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------
# In-Memory State
# ---------------------------------------------------------------------
user_sessions: Dict[str, Dict[str, Any]] = {}
practice_sessions: Dict[str, Dict[str, Any]] = {}
user_history: Dict[str, List[Dict[str, Any]]] = {}
leaderboard_data: Dict[str, Dict[str, Any]] = {}

# ---------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------
# Replace your existing DebugRequest class with this:

class DebugRequest(BaseModel):
    language: str
    level: str
    error: str
    context: Optional[str] = None
    user_id: Optional[str] = None

    @field_validator('language')
    @classmethod
    def validate_language(cls, v):
        if not v or not v.strip():
            raise ValueError('Language is required')
        return v.strip().lower()

    @field_validator('level')
    @classmethod
    def validate_level(cls, v):
        valid_levels = ['beginner', 'intermediate', 'advanced', 'expert']
        v_lower = v.strip().lower()
        if v_lower not in valid_levels:
            raise ValueError(f'Level must be one of: {", ".join(valid_levels)}')
        return v_lower

    @field_validator('error')
    @classmethod
    def validate_error(cls, v):
        if not v or not v.strip():
            raise ValueError('Error description is required')
        if len(v.strip()) > 10000:  # Increased limit
            raise ValueError('Error description too long (max 10000 characters)')
        return v.strip()

    # Alternative: Remove all validators for testing
    # Just comment out all @field_validator decorators and methods

class PracticeRequest(BaseModel):
    language: str
    level: str
    mode: str
    user_id: str
    num_questions: Optional[int] = 10
    suggestion: Optional[str] = None

    @field_validator('mode')
    @classmethod
    def validate_mode(cls, v):
        valid_modes = ['quiz', 'debug', 'code_review']
        if v not in valid_modes:
            raise ValueError(f'Mode must be one of: {", ".join(valid_modes)}')
        return v

class AnswerRequest(BaseModel):
    user_id: str
    session_id: str
    question_id: int
    user_answer: str
    time_taken_seconds: Optional[int] = 0

# ---------------------------------------------------------------------
# Gemini API Configuration
# ---------------------------------------------------------------------
GEMINI_MODELS = [
    "gemini-2.0-flash-exp",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    "gemini-pro"
]

async def call_gemini_api(prompt: str, temperature=0.7, max_tokens=2000) -> Dict[str, Any]:
    """Call Gemini API with fallback models"""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not configured")

    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key
    }

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
            "topP": 0.8,
            "topK": 40
        },
        "safetySettings": [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
        ]
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        for model in GEMINI_MODELS:
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
                
                if "candidates" in data and data["candidates"]:
                    return {"model": model, "data": data}
                    
            except Exception as e:
                logger.warning(f"Model {model} failed: {e}")
                continue

    raise HTTPException(status_code=503, detail="All Gemini models unavailable")

def extract_gemini_text(response: Dict[str, Any]) -> str:
    """Extract text from Gemini response"""
    try:
        candidates = response["data"]["candidates"]
        content = candidates[0]["content"]
        parts = content.get("parts", [])
        if parts and "text" in parts[0]:
            return parts[0]["text"]
    except Exception as e:
        logger.error(f"Text extraction failed: {e}")
    return ""

# ---------------------------------------------------------------------
# Template System
# ---------------------------------------------------------------------
QUESTION_TEMPLATES = {
    'python': {
        'beginner': [
            {
                'question': 'What will happen when this code runs?',
                'code': 'my_list = [1, 2, 3]\nprint(my_list[3])',
                'options': ['A) Prints 3', 'B) IndexError: list index out of range', 'C) Prints None', 'D) Prints 4'],
                'correct_answer': 'B',
                'explanation': 'List indices start at 0, so index 3 is out of range for a list with 3 elements.',
                'points': 10
            },
            {
                'question': 'What error will this code produce?',
                'code': 'x = 5\ny = "hello"\nz = x + y',
                'options': ['A) No error', 'B) SyntaxError', 'C) TypeError', 'D) NameError'],
                'correct_answer': 'C',
                'explanation': 'Cannot add integer and string directly in Python.',
                'points': 10
            },
            {
                'question': 'What is wrong with this function call?',
                'code': 'def greet(name):\n    return f"Hello, {name}!"\n\ngreet()',
                'options': ['A) Missing return', 'B) Missing argument', 'C) Wrong syntax', 'D) No error'],
                'correct_answer': 'B',
                'explanation': 'The function requires a name argument.',
                'points': 10
            }
        ],
        'intermediate': [
            {
                'question': 'What is the issue with this recursive function?',
                'code': 'def factorial(n):\n    if n == 0:\n        return 1\n    return n * factorial(n - 1)',
                'options': ['A) Missing base case', 'B) Infinite recursion for negatives', 'C) Wrong return', 'D) No error'],
                'correct_answer': 'B',
                'explanation': 'For negative numbers, it never reaches 0, causing infinite recursion.',
                'points': 15
            },
            {
                'question': 'What will happen with this dictionary access?',
                'code': 'data = {"name": "John"}\nprint(data["email"])',
                'options': ['A) Prints None', 'B) KeyError', 'C) Prints empty string', 'D) Creates key'],
                'correct_answer': 'B',
                'explanation': 'Accessing non-existent key raises KeyError.',
                'points': 15
            }
        ]
    },
    'javascript': {
        'beginner': [
            {
                'question': 'What will this code output?',
                'code': 'let arr = [1, 2, 3];\nconsole.log(arr[10]);',
                'options': ['A) 10', 'B) undefined', 'C) Error', 'D) null'],
                'correct_answer': 'B',
                'explanation': 'Accessing non-existent index returns undefined in JavaScript.',
                'points': 10
            }
        ]
    },
    'java': {
        'beginner': [
            {
                'question': 'What will this code produce?',
                'code': 'String x = null;\nSystem.out.println(x.length());',
                'options': ['A) 0', 'B) NullPointerException', 'C) Compilation error', 'D) null'],
                'correct_answer': 'B',
                'explanation': 'Calling methods on null reference throws NullPointerException.',
                'points': 10
            }
        ]
    }
}

# ---------------------------------------------------------------------
# Question Generation
# ---------------------------------------------------------------------
async def generate_questions_with_gemini(language: str, level: str, mode: str, count: int) -> Optional[List[Dict]]:
    """Generate questions using Gemini API"""
    prompt = f"""Generate {count} programming practice questions for {language} at {level} level.

Return ONLY valid JSON array with this structure:
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
- Include realistic {language} code
- {level} difficulty level
- Focus on {mode} type questions"""

    try:
        response = await call_gemini_api(prompt, temperature=0.3)
        text = extract_gemini_text(response)
        
        if not text:
            return None
            
        # Clean JSON response
        cleaned = text.strip()
        if cleaned.startswith('```'):
            lines = cleaned.split('\n')
            start_idx = next((i for i, line in enumerate(lines) if '[' in line), 0)
            end_idx = next((i for i in range(len(lines)-1, -1, -1) if ']' in lines[i]), len(lines))
            cleaned = '\n'.join(lines[start_idx:end_idx+1])
        
        cleaned = cleaned.replace('```json', '').replace('```', '').strip()
        
        questions = json.loads(cleaned)
        if not isinstance(questions, list):
            return None
            
        # Validate questions
        validated = []
        for i, q in enumerate(questions[:count], 1):
            if not isinstance(q, dict):
                continue
                
            validated_q = {
                "id": i,
                "question": str(q.get('question', '')).strip(),
                "code": str(q.get('code', '')).strip(),
                "options": q.get('options', []),
                "correct_answer": str(q.get('correct_answer', 'A')).upper().strip(),
                "explanation": str(q.get('explanation', '')).strip(),
                "points": int(q.get('points', 10))
            }
            
            # Validate options
            if len(validated_q['options']) != 4:
                continue
                
            # Validate correct answer
            if validated_q['correct_answer'] not in ['A', 'B', 'C', 'D']:
                validated_q['correct_answer'] = 'A'
                
            validated.append(validated_q)
        
        return validated if validated else None
        
    except Exception as e:
        logger.error(f"Gemini generation failed: {e}")
        return None

def generate_from_templates(language: str, level: str, mode: str, count: int) -> List[Dict]:
    """Generate questions from templates"""
    templates = QUESTION_TEMPLATES.get(language.lower(), {}).get(level.lower(), [])
    
    if not templates:
        # Try fallback levels
        for fallback in ['beginner', 'intermediate', 'advanced']:
            templates = QUESTION_TEMPLATES.get(language.lower(), {}).get(fallback, [])
            if templates:
                break
    
    if not templates:
        # Create generic questions
        templates = [{
            'question': f'What is important in {language} programming?',
            'code': f'// {language} example',
            'options': ['A) Syntax', 'B) Logic', 'C) Testing', 'D) All above'],
            'correct_answer': 'D',
            'explanation': f'All aspects are important in {language} development.',
            'points': 10
        }]
    
    questions = []
    for i in range(count):
        template = random.choice(templates).copy()
        template['id'] = i + 1
        questions.append(template)
    
    return questions

async def generate_practice_questions(language: str, level: str, mode: str, count: int) -> List[Dict]:
    """Main question generation function"""
    # Try Gemini first
    gemini_questions = await generate_questions_with_gemini(language, level, mode, count)
    if gemini_questions and len(gemini_questions) >= count:
        return gemini_questions[:count]
    
    # Fallback to templates
    template_questions = generate_from_templates(language, level, mode, count)
    
    # Combine if we have partial Gemini results
    if gemini_questions:
        remaining = count - len(gemini_questions)
        additional = generate_from_templates(language, level, mode, remaining)
        for i, q in enumerate(additional):
            q['id'] = len(gemini_questions) + i + 1
        return (gemini_questions + additional)[:count]
    
    return template_questions[:count]

# ---------------------------------------------------------------------
# Session Management
# ---------------------------------------------------------------------
def init_user_session(user_id: str):
    """Initialize user session if not exists"""
    if user_id not in user_sessions:
        user_sessions[user_id] = {
            "session_id": str(uuid.uuid4()),
            "created_at": datetime.now().isoformat(),
            "total_queries": 0,
            "favorite_language": None,
            "skill_level": None,
            "last_activity": datetime.now().isoformat()
        }
        user_history[user_id] = []
        leaderboard_data[user_id] = {
            "user_id": user_id,
            "total_score": 0,
            "practice_sessions_completed": 0,
            "last_activity": datetime.now().isoformat()
        }

def add_to_history(user_id: str, request_type: str, language: str, level: str, query: str, processing_time: float, score: Optional[int] = None):
    """Add entry to user history"""
    if user_id not in user_history:
        user_history[user_id] = []

    user_history[user_id].append({
        "timestamp": datetime.now().isoformat(),
        "request_type": request_type,
        "language": language,
        "level": level,
        "query": query[:100] + '...' if len(query) > 100 else query,
        "processing_time": processing_time,
        "score": score
    })
    
    # Keep only last 50 entries
    user_history[user_id] = user_history[user_id][-50:]

# ---------------------------------------------------------------------
# Enhanced Prompting
# ---------------------------------------------------------------------
def create_debug_prompt(language: str, level: str, error: str, context: Optional[str] = None) -> str:
    """Create enhanced debugging prompt"""
    level_instructions = {
        'beginner': 'Use simple terms and explain basic concepts clearly.',
        'intermediate': 'Provide detailed explanations with moderate technical depth.',
        'advanced': 'Use technical language and focus on best practices.',
        'expert': 'Provide concise, technical solutions with performance considerations.'
    }

    prompt = f"""You are an expert debugging coach for {language.title()} programming.

PROGRAMMER LEVEL: {level.title()}
INSTRUCTIONS: {level_instructions.get(level, level_instructions['intermediate'])}

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

## Follow-up Questions (if needed)
[Any clarifying questions if the error is unclear]
"""
    return prompt

# ---------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------
@app.get("/")
async def root():
    return {
        "message": "AI Debugging Coach API",
        "version": "1.2.0",
        "status": "operational",
        "endpoints": {
            "debug": "POST /debug - Get debugging help",
            "practice": "POST /practice/generate - Generate practice questions",
            "answer": "POST /practice/answer - Submit answers",
            "session": "GET /user/{user_id}/session - User session info",
            "history": "GET /user/{user_id}/history - User history",
            "leaderboard": "GET /leaderboard - View leaderboard",
            "health": "GET /health - Health check"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.2.0",
        "active_users": len(user_sessions),
        "active_sessions": len(practice_sessions)
    }

@app.get("/test-api")
async def test_api():
    """Test Gemini API connection"""
    try:
        response = await call_gemini_api("Reply with: API connection successful!", temperature=0.1, max_tokens=50)
        text = extract_gemini_text(response)
        return {
            "success": True,
            "model_used": response["model"],
            "response": text,
            "message": "Gemini API working correctly"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
@app.post("/debug")
async def debug_endpoint(request: DebugRequest):
    """Debug assistance endpoint"""
    start_time = datetime.now()
    
    # Add debug logging
    logger.info(f"Debug request received: {request.model_dump()}")
    
    try:
        # Initialize user session
        if request.user_id:
            init_user_session(request.user_id)
            session = user_sessions[request.user_id]
            session["total_queries"] += 1
            session["last_activity"] = datetime.now().isoformat()
            if not session["favorite_language"]:
                session["favorite_language"] = request.language
                session["skill_level"] = request.level

        # Create prompt and call Gemini
        prompt = create_debug_prompt(request.language, request.level, request.error, request.context)
        response = await call_gemini_api(prompt)
        answer = extract_gemini_text(response)
        
        if not answer or len(answer.strip()) < 50:
            raise HTTPException(status_code=502, detail="AI response too short or empty")

        processing_time = (datetime.now() - start_time).total_seconds()

        # Log to history
        if request.user_id:
            add_to_history(
                request.user_id, "debug", request.language, 
                request.level, request.error, processing_time
            )

        return {
            "response": answer,
            "model_used": response["model"],
            "processing_time_seconds": round(processing_time, 2),
            "timestamp": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Debug endpoint error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Alternative: Add a raw request handler for debugging
from fastapi import Request

@app.post("/debug-raw")
async def debug_raw(request: Request):
    """Raw debug endpoint to see exact request data"""
    try:
        body = await request.body()
        logger.info(f"Raw request body: {body}")
        
        json_data = await request.json()
        logger.info(f"Parsed JSON: {json_data}")
        
        return {"received_data": json_data, "status": "ok"}
    except Exception as e:
        logger.error(f"Raw debug error: {e}")
        return {"error": str(e)}
    
@app.post("/practice/generate")
async def generate_practice_endpoint(request: PracticeRequest):
    """Generate practice questions"""
    start_time = datetime.now()
    session_id = str(uuid.uuid4())
    
    try:
        # Initialize user session
        init_user_session(request.user_id)
        
        # Generate questions
        count = max(1, min(request.num_questions or 10, 20))
        questions = await generate_practice_questions(
            request.language, request.level, request.mode, count
        )
        
        if not questions:
            raise HTTPException(status_code=500, detail="Failed to generate questions")

        # Create practice session
        practice_sessions[session_id] = {
            "session_id": session_id,
            "user_id": request.user_id,
            "language": request.language,
            "level": request.level,
            "mode": request.mode,
            "questions": questions,
            "answers": {},
            "started_at": datetime.now().isoformat(),
            "completed_at": None,
            "total_score": 0,
            "total_possible_points": sum(q.get("points", 10) for q in questions)
        }

        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Log to history
        add_to_history(
            request.user_id, "practice_generate", request.language,
            request.level, f"Generated {len(questions)} questions", processing_time
        )

        return {
            "session_id": session_id,
            "questions": questions,
            "total_questions": len(questions),
            "total_possible_points": practice_sessions[session_id]["total_possible_points"],
            "processing_time_seconds": round(processing_time, 2)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Practice generation error: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate practice session")

@app.post("/practice/answer")
async def answer_endpoint(request: AnswerRequest):
    """Submit practice answer"""
    if request.session_id not in practice_sessions:
        raise HTTPException(status_code=404, detail="Practice session not found")

    session = practice_sessions[request.session_id]
    
    # Verify user
    if session["user_id"] != request.user_id:
        raise HTTPException(status_code=403, detail="Unauthorized")

    # Find question
    question = next((q for q in session["questions"] if q["id"] == request.question_id), None)
    if not question:
        raise HTTPException(status_code=404, detail="Question not found")

    if request.question_id in session["answers"]:
        raise HTTPException(status_code=400, detail="Question already answered")

    # Evaluate answer
    user_answer = request.user_answer.upper().strip()
    correct_answer = question.get("correct_answer", "A").upper().strip()
    correct = user_answer == correct_answer
    points_earned = question.get("points", 10) if correct else 0

    # Store answer
    session["answers"][request.question_id] = {
        "user_answer": user_answer,
        "correct": correct,
        "points_earned": points_earned,
        "time_taken": request.time_taken_seconds,
        "answered_at": datetime.now().isoformat()
    }

    session["total_score"] += points_earned

    # Check if session completed
    session_completed = len(session["answers"]) == len(session["questions"])
    if session_completed:
        session["completed_at"] = datetime.now().isoformat()
        
        # Update leaderboard
        if request.user_id in leaderboard_data:
            leaderboard_data[request.user_id]["total_score"] += session["total_score"]
            leaderboard_data[request.user_id]["practice_sessions_completed"] += 1
            leaderboard_data[request.user_id]["last_activity"] = datetime.now().isoformat()

        # Log completion
        add_to_history(
            request.user_id, "practice_complete", session["language"],
            session["level"], f"Completed session: {session['total_score']} points",
            0, session["total_score"]
        )

    return {
        "correct": correct,
        "points_earned": points_earned,
        "correct_answer": correct_answer,
        "explanation": question.get("explanation", "No explanation available"),
        "current_score": session["total_score"],
        "questions_answered": len(session["answers"]),
        "total_questions": len(session["questions"]),
        "session_completed": session_completed
    }

@app.get("/user/{user_id}/session")
async def get_user_session(user_id: str):
    """Get user session info"""
    init_user_session(user_id)
    
    session = user_sessions[user_id]
    leaderboard = leaderboard_data.get(user_id, {})
    history = user_history.get(user_id, [])
    
    # Calculate statistics
    debug_queries = len([h for h in history if h["request_type"] == "debug"])
    practice_sessions = len([h for h in history if h["request_type"] == "practice_complete"])
    total_score = leaderboard.get("total_score", 0)
    
    stats = {
        "total_queries": session["total_queries"],
        "debug_queries": debug_queries,
        "practice_sessions": practice_sessions,
        "total_practice_score": total_score,
        "favorite_language": session.get("favorite_language"),
        "skill_level": session.get("skill_level")
    }
    
    return {
        "session": session,
        "statistics": stats
    }

@app.get("/user/{user_id}/history")
async def get_user_history(user_id: str, limit: int = 20):
    """Get user history"""
    init_user_session(user_id)
    
    history = user_history.get(user_id, [])
    recent_history = sorted(history, key=lambda x: x["timestamp"], reverse=True)[:limit]
    
    return {
        "history": recent_history,
        "total_entries": len(history)
    }

@app.get("/leaderboard")
async def get_leaderboard(limit: int = 10):
    """Get leaderboard"""
    sorted_users = sorted(
        leaderboard_data.values(),
        key=lambda x: x.get("total_score", 0),
        reverse=True
    )[:limit]
    
    return {
        "leaderboard": sorted_users,
        "total_users": len(leaderboard_data),
        "last_updated": datetime.now().isoformat()
    }

@app.delete("/user/{user_id}/reset")
async def reset_user_data(user_id: str):
    """Reset user data"""
    # Remove user data
    if user_id in user_sessions:
        del user_sessions[user_id]
    if user_id in user_history:
        del user_history[user_id]
    if user_id in leaderboard_data:
        del leaderboard_data[user_id]

    # Remove practice sessions
    sessions_to_remove = [
        sid for sid, session in practice_sessions.items() 
        if session.get("user_id") == user_id
    ]
    for sid in sessions_to_remove:
        del practice_sessions[sid]

    return {"message": f"All data for user {user_id} has been reset"}

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    
    # Check for SSL certificates
    cert_file = Path("cert.pem")
    key_file = Path("key.pem")
    
    if cert_file.exists() and key_file.exists():
        print("Starting with HTTPS at https://127.0.0.1:8000")
        uvicorn.run(
            app, 
            host="127.0.0.1", 
            port=8000,
            ssl_keyfile=str(key_file),
            ssl_certfile=str(cert_file)
        )
    else:
        print("Starting with HTTP at http://127.0.0.1:8000")
        uvicorn.run(app, host="127.0.0.1", port=8000)