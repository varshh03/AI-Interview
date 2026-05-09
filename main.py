"""
AI Interview Platform - Backend (main.py)
FastAPI + Groq + Firebase Auth + SQLite
"""

import os
import json
import uuid
import base64
import tempfile
import logging
from datetime import datetime, timedelta
from typing import Optional, List
from pathlib import Path
from contextlib import asynccontextmanager
from dotenv import load_dotenv
load_dotenv()

import httpx
from fastapi import (
    FastAPI, HTTPException, Depends, UploadFile, File,
    Form, BackgroundTasks, status
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, EmailStr
import sqlite3

# ── Groq ──────────────────────────────────────────────────────────────────────
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    Groq = None

# ── PDF / DOCX parsing ────────────────────────────────────────────────────────
try:
    import pdfplumber
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

try:
    from docx import Document as DocxDocument
    DOCX_SUPPORT = True
except ImportError:
    DOCX_SUPPORT = False

# ── TTS (gTTS fallback when no paid TTS) ─────────────────────────────────────
try:
    from gtts import gTTS
    TTS_SUPPORT = True
except ImportError:
    TTS_SUPPORT = False

# ── Firebase Admin (optional – comment out if not using Firebase) ─────────────
try:
    import firebase_admin
    from firebase_admin import credentials, auth as firebase_auth
    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GROQ_API_KEY        = os.getenv("GROQ_API_KEY", "")
FIREBASE_CREDENTIALS = os.getenv("FIREBASE_CREDENTIALS_JSON", "")  # JSON string
SECRET_KEY          = os.getenv("SECRET_KEY", "changeme-secret-key-32chars-min!!")
DATABASE_PATH       = os.getenv("DATABASE_PATH", "interview_platform.db")
ALLOWED_ORIGINS     = os.getenv("ALLOWED_ORIGINS", "*").split(",")

# ── Groq client ───────────────────────────────────────────────────────────────
groq_client = Groq(api_key=GROQ_API_KEY) if (GROQ_AVAILABLE and GROQ_API_KEY) else None

# ── Firebase init ─────────────────────────────────────────────────────────────
if FIREBASE_AVAILABLE and FIREBASE_CREDENTIALS:
    try:
        cred_dict = json.loads(FIREBASE_CREDENTIALS)
        cred = credentials.Certificate(cred_dict)
        firebase_admin.initialize_app(cred)
        logger.info("Firebase initialised ✓")
    except Exception as e:
        logger.warning(f"Firebase init failed: {e}")
        FIREBASE_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────────────────────────────────────────

def get_db():
    conn = sqlite3.connect(DATABASE_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_db():
    conn = sqlite3.connect(DATABASE_PATH, check_same_thread=False)
    cur = conn.cursor()

    cur.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id          TEXT PRIMARY KEY,
            email       TEXT UNIQUE NOT NULL,
            name        TEXT,
            provider    TEXT DEFAULT 'email',
            password_hash TEXT,
            created_at  TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS sessions (
            id          TEXT PRIMARY KEY,
            user_id     TEXT NOT NULL,
            token       TEXT UNIQUE NOT NULL,
            expires_at  TEXT NOT NULL,
            created_at  TEXT DEFAULT (datetime('now')),
            FOREIGN KEY(user_id) REFERENCES users(id)
        );

        CREATE TABLE IF NOT EXISTS interviews (
            id              TEXT PRIMARY KEY,
            user_id         TEXT NOT NULL,
            domain          TEXT NOT NULL,
            custom_domain   TEXT,
            resume_text     TEXT,
            status          TEXT DEFAULT 'active',
            total_questions INTEGER DEFAULT 0,
            answered        INTEGER DEFAULT 0,
            score           REAL DEFAULT 0,
            feedback        TEXT,
            started_at      TEXT DEFAULT (datetime('now')),
            completed_at    TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id)
        );

        CREATE TABLE IF NOT EXISTS questions (
            id              TEXT PRIMARY KEY,
            interview_id    TEXT NOT NULL,
            question_text   TEXT NOT NULL,
            question_type   TEXT NOT NULL,
            difficulty      TEXT DEFAULT 'medium',
            order_num       INTEGER,
            audio_url       TEXT,
            FOREIGN KEY(interview_id) REFERENCES interviews(id)
        );

        CREATE TABLE IF NOT EXISTS answers (
            id              TEXT PRIMARY KEY,
            interview_id    TEXT NOT NULL,
            question_id     TEXT NOT NULL,
            user_id         TEXT NOT NULL,
            answer_text     TEXT,
            answer_type     TEXT DEFAULT 'text',
            score           REAL,
            max_score       REAL DEFAULT 10,
            feedback        TEXT,
            suggestions     TEXT,
            evaluated_at    TEXT,
            submitted_at    TEXT DEFAULT (datetime('now')),
            FOREIGN KEY(interview_id) REFERENCES interviews(id),
            FOREIGN KEY(question_id) REFERENCES questions(id)
        );
    """)

    conn.commit()
    conn.close()
    logger.info("Database initialised ✓")

# ─────────────────────────────────────────────────────────────────────────────
# AUTH HELPERS  (simple JWT-less token stored in DB)
# ─────────────────────────────────────────────────────────────────────────────
import hashlib
import secrets

security = HTTPBearer(auto_error=False)


def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    hashed = hashlib.sha256(f"{salt}{password}".encode()).hexdigest()
    return f"{salt}:{hashed}"


def verify_password(password: str, stored: str) -> bool:
    try:
        salt, hashed = stored.split(":")
        return hashlib.sha256(f"{salt}{password}".encode()).hexdigest() == hashed
    except Exception:
        return False


def create_session_token(user_id: str, db: sqlite3.Connection) -> str:
    token = secrets.token_urlsafe(48)
    session_id = str(uuid.uuid4())
    expires_at = (datetime.utcnow() + timedelta(days=7)).isoformat()
    db.execute(
        "INSERT INTO sessions (id, user_id, token, expires_at) VALUES (?,?,?,?)",
        (session_id, user_id, token, expires_at)
    )
    db.commit()
    return token


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: sqlite3.Connection = Depends(get_db)
) -> dict:
    if not credentials:
        raise HTTPException(status_code=401, detail="Not authenticated")

    token = credentials.credentials

    # ── Firebase token ────────────────────────────────────────────────────────
    if FIREBASE_AVAILABLE:
        try:
            decoded = firebase_auth.verify_id_token(token)
            uid   = decoded["uid"]
            email = decoded.get("email", "")
            # upsert user
            row = db.execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone()
            if not row:
                db.execute(
                    "INSERT INTO users (id, email, name, provider) VALUES (?,?,?,?)",
                    (uid, email, decoded.get("name", ""), "google")
                )
                db.commit()
            return {"id": uid, "email": email}
        except Exception:
            pass  # fall through to DB session

    # ── DB session token ──────────────────────────────────────────────────────
    row = db.execute(
        """SELECT s.user_id, s.expires_at, u.email, u.name
           FROM sessions s JOIN users u ON u.id=s.user_id
           WHERE s.token=?""",
        (token,)
    ).fetchone()

    if not row:
        raise HTTPException(status_code=401, detail="Invalid token")
    if datetime.fromisoformat(row["expires_at"]) < datetime.utcnow():
        raise HTTPException(status_code=401, detail="Token expired")

    return {"id": row["user_id"], "email": row["email"], "name": row["name"]}

# ─────────────────────────────────────────────────────────────────────────────
# PYDANTIC SCHEMAS
# ─────────────────────────────────────────────────────────────────────────────

class SignUpRequest(BaseModel):
    email: EmailStr
    password: str
    name: Optional[str] = ""

class SignInRequest(BaseModel):
    email: EmailStr
    password: str

class FirebaseTokenRequest(BaseModel):
    firebase_token: str

class StartInterviewRequest(BaseModel):
    domain: str                        # e.g. "Python", "Data Science", "other"
    custom_domain: Optional[str] = ""  # filled when domain=="other"
    num_questions: Optional[int] = 5

class SubmitAnswerRequest(BaseModel):
    interview_id: str
    question_id: str
    answer_text: str
    answer_type: str = "text"          # "text" | "audio_transcript" | "code"

class EvaluateAnswerRequest(BaseModel):
    interview_id: str
    question_id: str
    answer_id: str

# ─────────────────────────────────────────────────────────────────────────────
# GROQ HELPERS
# ─────────────────────────────────────────────────────────────────────────────

DOMAINS = [
    "Python", "JavaScript", "Java", "C++", "Data Science",
    "Machine Learning", "DevOps", "Cloud (AWS/GCP/Azure)",
    "System Design", "SQL & Databases", "Frontend (React/Vue)",
    "Backend (Node/Django/FastAPI)", "Cybersecurity", "Product Management",
    "HR & Behavioural", "other"
]

QUESTION_TYPE_PROMPT = {
    "conceptual":  "theoretical / conceptual",
    "coding":      "coding / problem-solving (write code)",
    "behavioural": "behavioural (STAR method)",
    "system":      "system design",
}


def _groq_chat(messages: list, model="llama-3.3-70b-versatile", temperature=0.7, max_tokens=1024) -> str:
    if not groq_client:
        raise HTTPException(status_code=503, detail="Groq API key not configured")
    try:
        resp = groq_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Groq error: {e}")
        raise HTTPException(status_code=500, detail=f"Groq error: {str(e)}")

def generate_questions_from_domain(domain: str, num: int = 5, resume_text: str = "") -> list[dict]:
    """Ask Groq to produce interview questions, deeply analysing resume if provided."""

    if resume_text and resume_text.strip():
        # ── Resume-based deep analysis mode ──────────────────────────────────
        system = (
            "You are a senior technical interviewer. "
            "You will receive a candidate resume. Carefully read it and extract: "
            "their skills, technologies, projects, job roles, achievements, and experience gaps. "
            "Then generate highly specific interview questions based ONLY on what is in their resume. "
            "Questions must reference actual projects, technologies, or roles mentioned. "
            "Do NOT ask generic questions. Every question must be personalized to this exact resume. "
            "Return ONLY a valid JSON array, no markdown, no extra text. "
            "Each element: {\"question\": \"...\", \"type\": \"conceptual|coding|behavioural|system\", \"difficulty\": \"easy|medium|hard\"}"
        )
        user = (
            f"CANDIDATE RESUME:\n{resume_text[:4000]}\n\n"
            f"TARGET DOMAIN: {domain}\n\n"
            f"Generate exactly {num} interview questions that are SPECIFIC to this resume. "
            "Reference their actual projects, technologies they used, and roles they held. "
            "Mix types: some about their specific projects (behavioural), "
            "some technical deep-dives on technologies they listed (conceptual/coding), "
            "some about challenges in their actual experience (behavioural). "
            f"Return exactly {num} items in the JSON array."
        )
    else:
        # ── Domain-only mode ─────────────────────────────────────────────────
        system = (
            "You are an expert technical interviewer. "
            "Return ONLY a valid JSON array – no markdown, no extra text. "
            "Each element: {\"question\": \"...\", \"type\": \"conceptual|coding|behavioural|system\", \"difficulty\": \"easy|medium|hard\"}"
        )
        user = (
            f"Generate {num} interview questions for the domain: {domain}. "
            "Mix question types: conceptual, coding, behavioural, system design. "
            f"Return exactly {num} items in the JSON array."
        )

    raw = _groq_chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=2048
    )
    raw = raw.replace("```json", "").replace("```", "").strip()
    try:
        questions = json.loads(raw)
        if not isinstance(questions, list):
            questions = questions.get("questions", [])
        return questions[:num]
    except json.JSONDecodeError:
        logger.error(f"Bad JSON from Groq: {raw[:300]}")
        return [{"question": f"Tell me about your experience with {domain}.",
                 "type": "conceptual", "difficulty": "medium"}]


def evaluate_answer(question: str, q_type: str, answer: str, domain: str) -> dict:
    """Score and give suggestions for a candidate answer."""
    system = (
        "You are a senior interviewer evaluating a candidate's answer. "
        "Return ONLY valid JSON: "
        "{\"score\": <0-10 float>, \"verdict\": \"...\", "
        "\"strengths\": [...], \"improvements\": [...], \"model_answer_hint\": \"...\"}"
    )
    user = (
        f"Domain: {domain}\n"
        f"Question type: {QUESTION_TYPE_PROMPT.get(q_type, q_type)}\n"
        f"Question: {question}\n"
        f"Candidate answer: {answer}\n\n"
        "Score the answer from 0-10 and give structured feedback."
    )
    raw = _groq_chat([{"role": "system", "content": system},
                      {"role": "user",   "content": user}],
                     max_tokens=1024)
    raw = raw.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"score": 5.0, "verdict": raw[:200],
                "strengths": [], "improvements": [], "model_answer_hint": ""}


def extract_resume_text(file: UploadFile) -> str:
    """Extract plain text from PDF or DOCX resume."""
    suffix = Path(file.filename).suffix.lower()
    data = file.file.read()

    if suffix == ".pdf" and PDF_SUPPORT:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        text = ""
        with pdfplumber.open(tmp_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
        os.unlink(tmp_path)
        return text

    if suffix in (".docx", ".doc") and DOCX_SUPPORT:
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        doc = DocxDocument(tmp_path)
        text = "\n".join(p.text for p in doc.paragraphs)
        os.unlink(tmp_path)
        return text

    # fallback – raw bytes → utf-8 best-effort
    return data.decode("utf-8", errors="ignore")


def transcribe_audio(audio_bytes: bytes, filename: str = "audio.webm") -> str:
    """Use Groq Whisper to transcribe audio."""
    if not groq_client:
        raise HTTPException(status_code=503, detail="Groq not configured")
    with tempfile.NamedTemporaryFile(suffix=Path(filename).suffix or ".webm", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        with open(tmp_path, "rb") as f:
            transcription = groq_client.audio.transcriptions.create(
                file=(filename, f, "audio/webm"),
                model="whisper-large-v3",
                response_format="text",
            )
        return transcription
    finally:
        os.unlink(tmp_path)


def text_to_speech(text: str) -> bytes:
    """Convert question text to speech audio (MP3 bytes)."""
    if not TTS_SUPPORT:
        return b""
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp_path = tmp.name
    tts = gTTS(text=text, lang="en", slow=False)
    tts.save(tmp_path)
    with open(tmp_path, "rb") as f:
        audio_bytes = f.read()
    os.unlink(tmp_path)
    return audio_bytes

# ─────────────────────────────────────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    logger.info("AI Interview Platform started ✓")
    yield
    logger.info("AI Interview Platform shutting down...")


app = FastAPI(
    title="AI Interview Platform API",
    description="Groq-powered adaptive interview backend",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.options("/{rest_of_path:path}")
async def preflight(rest_of_path: str):
    return JSONResponse(content={}, headers={
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "*",
        "Access-Control-Allow-Headers": "*",
    })



@app.get("/health")
def health():
    return {"status": "ok", "groq": bool(groq_client),
            "firebase": FIREBASE_AVAILABLE, "tts": TTS_SUPPORT}

# ─────────────────────────────────────────────────────────────────────────────
# AUTH ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/auth/signup")
def signup(body: SignUpRequest, db: sqlite3.Connection = Depends(get_db)):
    existing = db.execute("SELECT id FROM users WHERE email=?", (body.email,)).fetchone()
    if existing:
        raise HTTPException(status_code=409, detail="Email already registered")

    user_id = str(uuid.uuid4())
    ph = hash_password(body.password)
    db.execute(
        "INSERT INTO users (id, email, name, provider, password_hash) VALUES (?,?,?,?,?)",
        (user_id, body.email, body.name, "email", ph)
    )
    db.commit()
    token = create_session_token(user_id, db)
    return {"token": token, "user": {"id": user_id, "email": body.email, "name": body.name}}


@app.post("/auth/signin")
def signin(body: SignInRequest, db: sqlite3.Connection = Depends(get_db)):
    row = db.execute("SELECT * FROM users WHERE email=?", (body.email,)).fetchone()
    if not row or not verify_password(body.password, row["password_hash"] or ""):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_session_token(row["id"], db)
    return {"token": token, "user": {"id": row["id"], "email": row["email"], "name": row["name"]}}


@app.post("/auth/firebase")
def firebase_signin(body: FirebaseTokenRequest, db: sqlite3.Connection = Depends(get_db)):
    """Exchange a Firebase ID token for a platform session."""
    if not FIREBASE_AVAILABLE:
        raise HTTPException(status_code=501, detail="Firebase not configured")
    try:
        decoded = firebase_auth.verify_id_token(body.firebase_token)
        uid   = decoded["uid"]
        email = decoded.get("email", "")
        name  = decoded.get("name", "")
        row = db.execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone()
        if not row:
            db.execute(
                "INSERT INTO users (id, email, name, provider) VALUES (?,?,?,?)",
                (uid, email, name, "google")
            )
            db.commit()
        token = create_session_token(uid, db)
        return {"token": token, "user": {"id": uid, "email": email, "name": name}}
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Firebase error: {e}")


@app.post("/auth/signout")
def signout(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: sqlite3.Connection = Depends(get_db)
):
    if creds:
        db.execute("DELETE FROM sessions WHERE token=?", (creds.credentials,))
        db.commit()
    return {"message": "Signed out"}


@app.get("/auth/me")
def me(user: dict = Depends(get_current_user)):
    return user

# ─────────────────────────────────────────────────────────────────────────────
# DOMAINS
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/domains")
def list_domains():
    return {"domains": DOMAINS}


# ─────────────────────────────────────────────────────────────────────────────
# RESUME UPLOAD (standalone — frontend uploads here before starting interview)
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/resume/upload")
async def upload_resume(
    file: UploadFile = File(...),
    user: dict = Depends(get_current_user),
):
    """Parse resume and return extracted text. Frontend stores in state."""
    import io
    allowed = {".pdf", ".docx", ".doc", ".txt"}
    suffix = Path(file.filename or "resume.txt").suffix.lower()
    if suffix not in allowed:
        raise HTTPException(status_code=415,
            detail=f"Unsupported file type '{suffix}'. Allowed: PDF, DOCX, TXT")

    content = await file.read()
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large (max 10 MB)")

    class _FakeUpload:
        def __init__(self, data, filename):
            self.filename = filename
            self.file = io.BytesIO(data)
    fake = _FakeUpload(content, file.filename or "resume.txt")
    resume_text = extract_resume_text(fake)

    if not resume_text.strip():
        raise HTTPException(status_code=422,
            detail="Could not extract text from resume. Try a text-based PDF or DOCX.")

    return {
        "message": "Resume parsed successfully",
        "text": resume_text,
        "resume_text": resume_text,
        "char_count": len(resume_text),
        "preview": resume_text[:300] + ("..." if len(resume_text) > 300 else ""),
    }



@app.post("/interview/start")
def start_interview(
    body: StartInterviewRequest,
    user: dict = Depends(get_current_user),
    db: sqlite3.Connection = Depends(get_db)
):
    try:
        effective_domain = body.custom_domain if body.domain == "other" and body.custom_domain else body.domain
        num = max(1, min(body.num_questions or 5, 15))
        questions_raw = generate_questions_from_domain(effective_domain, num)
        interview_id = str(uuid.uuid4())
        db.execute(
            """INSERT INTO interviews
               (id, user_id, domain, custom_domain, total_questions, status)
               VALUES (?,?,?,?,?,?)""",
            (interview_id, user["id"], body.domain, body.custom_domain or "", num, "active")
        )
        question_rows = []
        for i, q in enumerate(questions_raw):
            qid = str(uuid.uuid4())
            db.execute(
                """INSERT INTO questions
                   (id, interview_id, question_text, question_type, difficulty, order_num)
                   VALUES (?,?,?,?,?,?)""",
                (qid, interview_id, q["question"], q.get("type","conceptual"),
                 q.get("difficulty","medium"), i + 1)
            )
            question_rows.append({
                "id": qid,
                "question": q["question"],
                "type": q.get("type","conceptual"),
                "difficulty": q.get("difficulty","medium"),
                "order": i + 1
            })
        db.commit()
        return {
            "interview_id": interview_id,
            "domain": effective_domain,
            "total_questions": num,
            "questions": question_rows,
        }
    except Exception as e:
        logger.error(f"START INTERVIEW ERROR: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/interview/start-with-resume")
async def start_interview_with_resume(
    domain: str = Form("other"),
    custom_domain: str = Form(""),
    num_questions: int = Form(5),
    resume: UploadFile = File(...),
    user: dict = Depends(get_current_user),
    db: sqlite3.Connection = Depends(get_db)
):
    import io
    # Read file content first (async) then pass to sync extractor
    content_bytes = await resume.read()
    if not content_bytes:
        raise HTTPException(status_code=400, detail="Resume file is empty")

    class _FakeUpload:
        def __init__(self, data, filename):
            self.filename = filename
            self.file = io.BytesIO(data)

    fake = _FakeUpload(content_bytes, resume.filename or "resume.pdf")
    resume_text = extract_resume_text(fake)

    if not resume_text or len(resume_text.strip()) < 50:
        raise HTTPException(status_code=422, detail="Could not extract enough text from resume. Please use a text-based PDF or DOCX.")

    logger.info(f"Resume extracted: {len(resume_text)} chars for user {user['id']}")

    effective_domain = custom_domain if domain == "other" and custom_domain else domain
    num = max(1, min(num_questions, 15))

    # Generate questions deeply analysed from resume
    questions_raw = generate_questions_from_domain(effective_domain, num, resume_text)

    interview_id = str(uuid.uuid4())
    db.execute(
        """INSERT INTO interviews
           (id, user_id, domain, custom_domain, resume_text, total_questions, status)
           VALUES (?,?,?,?,?,?,?)""",
        (interview_id, user["id"], domain, custom_domain, resume_text[:5000], num, "active")
    )

    question_rows = []
    for i, q in enumerate(questions_raw):
        qid = str(uuid.uuid4())
        db.execute(
            """INSERT INTO questions
               (id, interview_id, question_text, question_type, difficulty, order_num)
               VALUES (?,?,?,?,?,?)""",
            (qid, interview_id, q["question"], q.get("type","conceptual"),
             q.get("difficulty","medium"), i + 1)
        )
        question_rows.append({
            "id": qid,
            "question": q["question"],
            "type": q.get("type","conceptual"),
            "difficulty": q.get("difficulty","medium"),
            "order": i + 1
        })
    db.commit()

    return {
        "interview_id": interview_id,
        "domain": effective_domain,
        "total_questions": num,
        "questions": question_rows,
        "resume_parsed": bool(resume_text),
    }


@app.get("/interview/{interview_id}")
def get_interview(
    interview_id: str,
    user: dict = Depends(get_current_user),
    db: sqlite3.Connection = Depends(get_db)
):
    interview = db.execute(
        "SELECT * FROM interviews WHERE id=? AND user_id=?",
        (interview_id, user["id"])
    ).fetchone()
    if not interview:
        raise HTTPException(status_code=404, detail="Interview not found")

    questions = db.execute(
        "SELECT * FROM questions WHERE interview_id=? ORDER BY order_num",
        (interview_id,)
    ).fetchall()

    answers = db.execute(
        "SELECT * FROM answers WHERE interview_id=?",
        (interview_id,)
    ).fetchall()

    return {
        "interview": dict(interview),
        "questions": [dict(q) for q in questions],
        "answers":   [dict(a) for a in answers],
    }


@app.post("/interview/submit-answer")
def submit_answer(
    body: SubmitAnswerRequest,
    user: dict = Depends(get_current_user),
    db: sqlite3.Connection = Depends(get_db)
):
    # validate ownership
    interview = db.execute(
        "SELECT * FROM interviews WHERE id=? AND user_id=?",
        (body.interview_id, user["id"])
    ).fetchone()
    if not interview:
        raise HTTPException(status_code=404, detail="Interview not found")

    question = db.execute(
        "SELECT * FROM questions WHERE id=? AND interview_id=?",
        (body.question_id, body.interview_id)
    ).fetchone()
    if not question:
        raise HTTPException(status_code=404, detail="Question not found")

    # evaluate via Groq
    eval_result = evaluate_answer(
        question=question["question_text"],
        q_type=question["question_type"],
        answer=body.answer_text,
        domain=interview["domain"]
    )

    answer_id = str(uuid.uuid4())
    score       = float(eval_result.get("score", 5.0))
    feedback    = eval_result.get("verdict", "")
    suggestions = json.dumps({
        "strengths":         eval_result.get("strengths", []),
        "improvements":      eval_result.get("improvements", []),
        "model_answer_hint": eval_result.get("model_answer_hint", ""),
    })

    db.execute(
        """INSERT INTO answers
           (id, interview_id, question_id, user_id,
            answer_text, answer_type, score, max_score, feedback, suggestions, evaluated_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        (answer_id, body.interview_id, body.question_id, user["id"],
         body.answer_text, body.answer_type, score, 10.0, feedback, suggestions,
         datetime.utcnow().isoformat())
    )

    # update interview answered count
    db.execute(
        "UPDATE interviews SET answered = answered + 1 WHERE id=?",
        (body.interview_id,)
    )
    db.commit()

    return {
        "answer_id":   answer_id,
        "score":       score,
        "max_score":   10.0,
        "feedback":    feedback,
        "suggestions": json.loads(suggestions),
    }


@app.post("/interview/{interview_id}/complete")
def complete_interview(
    interview_id: str,
    user: dict = Depends(get_current_user),
    db: sqlite3.Connection = Depends(get_db)
):
    interview = db.execute(
        "SELECT * FROM interviews WHERE id=? AND user_id=?",
        (interview_id, user["id"])
    ).fetchone()
    if not interview:
        raise HTTPException(status_code=404, detail="Interview not found")

    answers = db.execute(
        "SELECT score, max_score FROM answers WHERE interview_id=?",
        (interview_id,)
    ).fetchall()

    if not answers:
        raise HTTPException(status_code=400, detail="No answers submitted yet")

    total_score = sum(a["score"] for a in answers)
    max_possible = sum(a["max_score"] for a in answers)
    pct = round((total_score / max_possible) * 100, 1) if max_possible else 0

    # Overall feedback via Groq
    overall_prompt = (
        f"The candidate completed a {interview['domain']} interview. "
        f"Score: {pct}%. "
        "Give a 3-sentence overall performance summary and top 3 action items to improve."
    )
    overall_fb = _groq_chat([{"role": "user", "content": overall_prompt}], max_tokens=300)

    db.execute(
        """UPDATE interviews
           SET status='completed', score=?, feedback=?, completed_at=?
           WHERE id=?""",
        (pct, overall_fb, datetime.utcnow().isoformat(), interview_id)
    )
    db.commit()

    return {
        "interview_id": interview_id,
        "score_percent": pct,
        "total_score": total_score,
        "max_score": max_possible,
        "overall_feedback": overall_fb,
    }

# ─────────────────────────────────────────────────────────────────────────────
# AUDIO ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/audio/transcribe")
async def transcribe(
    audio: UploadFile = File(...),
    user: dict = Depends(get_current_user)
):
    """Transcribe candidate's audio answer via Groq Whisper."""
    data = await audio.read()
    text = transcribe_audio(data, audio.filename or "audio.webm")
    return {"transcript": text}


@app.get("/audio/question/{question_id}")
def question_audio(
    question_id: str,
    user: dict = Depends(get_current_user),
    db: sqlite3.Connection = Depends(get_db)
):
    """Return TTS audio (MP3) for a question so the robot can speak it."""
    if not TTS_SUPPORT:
        raise HTTPException(status_code=501, detail="TTS not available (install gTTS)")

    q = db.execute("SELECT question_text FROM questions WHERE id=?", (question_id,)).fetchone()
    if not q:
        raise HTTPException(status_code=404, detail="Question not found")

    audio_bytes = text_to_speech(q["question_text"])
    return StreamingResponse(
        iter([audio_bytes]),
        media_type="audio/mpeg",
        headers={"Content-Disposition": f"inline; filename=question_{question_id}.mp3"}
    )

# ─────────────────────────────────────────────────────────────────────────────
# HISTORY / ANALYTICS
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/history")
def get_history(
    user: dict = Depends(get_current_user),
    db: sqlite3.Connection = Depends(get_db)
):
    """All completed + active interviews for the current user."""
    rows = db.execute(
        """SELECT id, domain, custom_domain, status, total_questions,
                  answered, score, started_at, completed_at
           FROM interviews WHERE user_id=? ORDER BY started_at DESC""",
        (user["id"],)
    ).fetchall()

    interviews = []
    for row in rows:
        r = dict(row)
        # per-domain score breakdown
        domain_answers = db.execute(
            """SELECT AVG(a.score)*10 as avg_score
               FROM answers a
               JOIN questions q ON q.id=a.question_id
               WHERE a.interview_id=?""",
            (row["id"],)
        ).fetchone()
        r["avg_score_per_q"] = round(domain_answers["avg_score"] or 0, 1)
        interviews.append(r)

    # domain-level aggregation
    domain_stats = db.execute(
        """SELECT domain, COUNT(*) as attempts, AVG(score) as avg_score
           FROM interviews WHERE user_id=? AND status='completed'
           GROUP BY domain ORDER BY attempts DESC""",
        (user["id"],)
    ).fetchall()

    return {
        "interviews":   interviews,
        "domain_stats": [dict(d) for d in domain_stats],
    }


@app.get("/history/{interview_id}/detail")
def interview_detail(
    interview_id: str,
    user: dict = Depends(get_current_user),
    db: sqlite3.Connection = Depends(get_db)
):
    interview = db.execute(
        "SELECT * FROM interviews WHERE id=? AND user_id=?",
        (interview_id, user["id"])
    ).fetchone()
    if not interview:
        raise HTTPException(status_code=404, detail="Not found")

    qa_pairs = db.execute(
        """SELECT q.order_num, q.question_text, q.question_type, q.difficulty,
                  a.answer_text, a.score, a.max_score, a.feedback, a.suggestions
           FROM questions q
           LEFT JOIN answers a ON a.question_id=q.id
           WHERE q.interview_id=?
           ORDER BY q.order_num""",
        (interview_id,)
    ).fetchall()

    result = []
    for row in qa_pairs:
        r = dict(row)
        if r.get("suggestions"):
            try:
                r["suggestions"] = json.loads(r["suggestions"])
            except Exception:
                pass
        result.append(r)

    return {"interview": dict(interview), "qa_pairs": result}

# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

# ── Serve index.html at root (so frontend + backend live on same origin) ──────
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import pathlib

_STATIC_DIR = pathlib.Path(__file__).parent / "static"
if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

@app.get("/", include_in_schema=False)
def serve_index():
    idx = pathlib.Path(__file__).parent / "static" / "index.html"
    if idx.exists():
        return FileResponse(str(idx))
    return {"message": "AI Interview Platform API — see /docs"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
