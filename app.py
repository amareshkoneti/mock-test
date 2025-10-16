import os
import json
import uuid
import time
import logging
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
from flask import Flask, render_template, request, jsonify, redirect, url_for, abort
from dotenv import load_dotenv
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential

# Suppress Google logging warnings
os.environ["GOOGLE_LOGGING_CPP_SUPPRESS_STDERR_LOGS"] = "1"

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Load .env if present
load_dotenv()

# Configuration
APP_ROOT = Path(__file__).parent
QUIZ_DIR = APP_ROOT / "quizzes"
QUIZ_DIR.mkdir(exist_ok=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
SECRET_KEY = os.getenv("SECRET_KEY", "dev_secret_key")

# Flask app
app = Flask(__name__)
app.secret_key = SECRET_KEY
app.config["JSON_SORT_KEYS"] = False

# Custom Jinja2 filter for datetime in IST
def datetime_filter(value):
    try:
        # Convert Unix timestamp to datetime, try IST first
        dt = datetime.fromtimestamp(int(value), tz=ZoneInfo("Asia/Kolkata"))
        return dt.strftime("%Y-%m-%d %I:%M %p")
    except Exception as e:
        logging.error(f"Datetime filter error for IST: {str(e)}")
        try:
            # Fallback to UTC if IST fails
            dt = datetime.fromtimestamp(int(value), tz=timezone.utc)
            return dt.strftime("%Y-%m-%d %I:%M %p UTC")
        except (ValueError, TypeError) as e2:
            logging.error(f"Datetime filter fallback error: {str(e2)}")
            return str(value)  # Final fallback to raw value

# Register the filter with Jinja2
app.jinja_env.filters['datetime'] = datetime_filter

# Helper to create a quiz file
def write_quiz_file(quiz_id: str, data: dict):
    path = QUIZ_DIR / f"quiz_{quiz_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def read_quiz_file(quiz_id: str):
    path = QUIZ_DIR / f"quiz_{quiz_id}.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# Build the prompt for Gemini
def build_gemini_prompt(topic: str):
    prompt = f"""
You are an expert question generator for aptitude, especially following the style of **TCS NQT**. Generate **30 quantitative aptitude multiple-choice questions** on the topic "{topic}".  
Use the style, difficulty, and syllabus coverage similar to TCS NQT’s Quantitative section (foundation and advanced).  
Include:
- Mostly moderate difficulty, with some easy and some challenging questions,
- Topics like percentages, ratio & proportion, time & work, speed & distance, averages, number systems, simplifications, equations, mensuration, profit & loss, mixtures & allegation, etc.
- No negative marking.
- Four options (A, B, C, D), one correct answer.
- A short explanation for each.

Return output purely in JSON — a JSON array of 30 objects, no additional prose. Do NOT add extra text or trailing commas.
Each object:
{{
  "question": "...",
  "options": {{ "A": "...", "B": "...", "C": "...", "D": "..." }},
  "answer": "A" / "B" / "C" / "D",
  "explanation": "..."
}}
"""
    return prompt

# Parse/normalize gemini response (defensive)
import re

def parse_gemini_response_text(resp_text: str):
    """
    Attempt to extract JSON from response string.
    Cleans common formatting issues like trailing commas before } or ].
    """
    try:
        # Extract first JSON array
        start = resp_text.index("[")
        end = resp_text.rindex("]") + 1
        snippet = resp_text[start:end]

        # Remove trailing commas before } or ]
        snippet = re.sub(r",\s*([}\]])", r"\1", snippet)

        # Parse JSON
        data = json.loads(snippet)
        if isinstance(data, list) and len(data) == 30:
            return data
        return data
    except Exception as e:
        raise ValueError(f"Failed to parse JSON from Gemini response: {e}\nRaw: {resp_text[:500]}")


# Function to list available models for debugging
def list_available_models():
    models = []
    for model in genai.list_models():
        if 'generateContent' in model.supported_generation_methods:
            models.append(model.name)
    return models

# Retry decorator for Gemini API calls
def call_gemini(prompt):
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        return response
    except Exception as e:
        if "NotFound" in str(e) or "404" in str(e):
            available = list_available_models()
            logging.error(f"Model not found. Available models with generateContent support: {available}")
        raise e

# Route: Home
@app.route("/")
def index():
    topics = [
    # Foundation / Must-prepare
    "Simplification",
    "Areas and Mensuration",
    "Percentage",
    "Averages",
    "Ratio & Proportions",
    "Ages",
    "Profit & Loss",
    "Statistics",
    "Time & Work",
    "Pipe & Cisterns",
    "Time, Speed & Distance",
    "Probability",
    "Permutations & Combinations",
    "Number System (HCF & LCM)",
    "Simple & Compound Interest",
    "Data Interpretation",
    
    # Reasoning
    "Syllogism",
    "Cubes / Cut Folds",
    "Blood Relations",
    "Direction",
    "Statement & Assumption",
    "Number Series",
    "Figure & Factual Analysis",
    "Decision Making",
    "Venn Diagrams",
    "Alphabet Series",
    "Coding Decoding",
    "Data Sufficiency",
    "Seating Arrangement",

    # Verbal Ability
    "Sentence Rearrangement",
    "Comprehension",
    "Error Detection",
    "Sentence Completion",
    "Tense",

    # Advanced / Optional
    "Advanced Aptitude",
    "Arrays",
    "Strings",
    "Patterns",
    "Basic Math Problems (Fibonacci, ARM, etc.)"
]

    return render_template("index.html", topics=topics)

# Route: create quiz page (form)
@app.route("/create", methods=["GET"])
def create_page():
    topics = [
    # Foundation / Must-prepare
    "Simplification",
    "Areas and Mensuration",
    "Percentage",
    "Averages",
    "Ratio & Proportions",
    "Ages",
    "Profit & Loss",
    "Statistics",
    "Time & Work",
    "Pipe & Cisterns",
    "Time, Speed & Distance",
    "Probability",
    "Permutations & Combinations",
    "Number System (HCF & LCM)",
    "Simple & Compound Interest",
    "Data Interpretation",
    
    # Reasoning
    "Syllogism",
    "Cubes / Cut Folds",
    "Blood Relations",
    "Direction",
    "Statement & Assumption",
    "Number Series",
    "Figure & Factual Analysis",
    "Decision Making",
    "Venn Diagrams",
    "Alphabet Series",
    "Coding Decoding",
    "Data Sufficiency",
    "Seating Arrangement",

    # Verbal Ability
    "Sentence Rearrangement",
    "Comprehension",
    "Error Detection",
    "Sentence Completion",
    "Tense",

    # Advanced / Optional
    "Advanced Aptitude",
    "Arrays",
    "Strings",
    "Patterns",
    "Basic Math Problems (Fibonacci, ARM, etc.)"
]

    return render_template("create.html", topics=topics)

# API: generate test
@app.route("/generate-test", methods=["POST"])
def generate_test():
    data = request.form or request.json or {}
    topic = data.get("topic")
    if not topic:
        return jsonify({"error": "topic is required"}), 400

    # Create quiz id
    quiz_id = uuid.uuid4().hex[:8]
    created_at = int(time.time())

    # Build prompt and call Gemini
    prompt = build_gemini_prompt(topic)

    try:
        response = call_gemini(prompt)
        resp_text = response.text
    except Exception as e:
        logging.error(f"Gemini API error: {str(e)}", exc_info=True)
        return jsonify({"error": "Failed to call Gemini API", "details": str(e)}), 500

    # Try parse JSON out of response text
    try:
        questions = parse_gemini_response_text(resp_text)
    except Exception as e:
        logging.error(f"Parse error: {str(e)}", exc_info=True)
        return jsonify({
            "error": "Failed to parse Gemini output",
            "details": str(e),
            "raw": resp_text[:1000]
        }), 500

    # Basic validation and normalization
    normalized = []
    for idx, q in enumerate(questions):
        try:
            question_text = q.get("question") if isinstance(q, dict) else q["question"]
            options = q.get("options") if isinstance(q, dict) else q["options"]
            answer = q.get("answer") if isinstance(q, dict) else q["answer"]
            explanation = q.get("explanation") if isinstance(q, dict) else q["explanation"]
            opts = {k: options.get(k, "") for k in ["A", "B", "C", "D"]}
            normalized.append({
                "id": idx + 1,
                "question": question_text,
                "options": opts,
                "answer": answer.strip().upper(),
                "explanation": explanation
            })
        except Exception as e:
            logging.error(f"Malformed question at index {idx}: {str(e)}", exc_info=True)
            return jsonify({"error": "Gemini returned malformed question structure", "index": idx, "item": q}), 500

    quiz_obj = {
        "quiz_id": quiz_id,
        "topic": topic,
        "created_at": created_at,
        "duration_minutes": 30,
        "time_limit_per_question_seconds": 60,
        "questions": normalized,
        "submissions": []
    }

    try:
        write_quiz_file(quiz_id, quiz_obj)
    except Exception as e:
        logging.error(f"Write quiz file error: {str(e)}", exc_info=True)
        return jsonify({"error": "Failed to write quiz file", "details": str(e)}), 500

    quiz_url = url_for("quiz_page", quiz_id=quiz_id, _external=True)
    return jsonify({"quiz_id": quiz_id, "quiz_url": quiz_url}), 200

# Route: show quiz (take test)
@app.route("/quiz/<quiz_id>", methods=["GET"])
def quiz_page(quiz_id):
    quiz = read_quiz_file(quiz_id)
    if not quiz:
        abort(404, description="Quiz not found.")
    # Do not leak answers to client — only send questions and options
    safe_questions = [
        {
            "id": q["id"],
            "question": q["question"],
            "options": q["options"]
        } for q in quiz["questions"]
    ]
    return render_template(
        "quiz.html",
        quiz_id=quiz_id,
        topic=quiz["topic"],
        duration=quiz.get("duration_minutes", 30),
        time_limit_per_question=quiz.get("time_limit_per_question_seconds", 60),
        questions=safe_questions
    )

# API: submit answers
@app.route("/submit/<quiz_id>", methods=["POST"])
def submit_quiz(quiz_id):
    quiz = read_quiz_file(quiz_id)
    if not quiz:
        return jsonify({"error": "Quiz not found"}), 404

    payload = request.get_json() or request.form or {}
    name = (payload.get("name") or "").strip() or "Anonymous"
    answers = payload.get("answers")
    time_seconds = payload.get("time_seconds")

    if not isinstance(answers, dict):
        return jsonify({"error": "answers must be provided as a dict"}), 400

    # Score calculation
    score = 0
    total = len(quiz["questions"])
    for q in quiz["questions"]:
        qid = str(q["id"])
        correct = q["answer"].strip().upper()
        submitted = (answers.get(qid, {}).get("choice") or "").strip().upper()
        if submitted == correct:
            score += 1

    # Save submission with per-question times
    submission = {
        "name": name,
        "score": score,
        "total": total,
        "time_seconds": float(time_seconds) if time_seconds else None,
        "submitted_at": int(time.time()),
        "answers": answers
    }
    quiz["submissions"].append(submission)
    try:
        write_quiz_file(quiz_id, quiz)
    except Exception as e:
        logging.error(f"Write submission error: {str(e)}", exc_info=True)
        return jsonify({"error": "Failed to save submission", "details": str(e)}), 500

    return jsonify({"result": "ok", "score": score, "total": total, "submission": submission}), 200

# Route: leaderboard
@app.route("/leaderboard/<quiz_id>", methods=["GET"])
def leaderboard(quiz_id):
    quiz = read_quiz_file(quiz_id)
    if not quiz:
        abort(404)
    subs = quiz.get("submissions", [])
    # sort by score desc, time asc (None times go after)
    def sort_key(s):
        t = s.get("time_seconds")
        if t is None:
            t = 10**9
        return (-int(s.get("score", 0)), float(t))
    ranked = sorted(subs, key=sort_key)
    return render_template("leaderboard.html", quiz=quiz, ranked=ranked)

# Utility: expose small API to get quiz meta (for front-end)
@app.route("/api/quiz/<quiz_id>/meta", methods=["GET"])
def quiz_meta(quiz_id):
    quiz = read_quiz_file(quiz_id)
    if not quiz:
        return jsonify({"error": "not found"}), 404
    return jsonify({
        "quiz_id": quiz["quiz_id"],
        "topic": quiz["topic"],
        "duration_minutes": quiz.get("duration_minutes", 30),
        "time_limit_per_question_seconds": quiz.get("time_limit_per_question_seconds", 60),
        "question_count": len(quiz["questions"])
    }), 200

# Run
if __name__ == "__main__":
    print(f"Quiz directory writable: {QUIZ_DIR.is_dir() and os.access(QUIZ_DIR, os.W_OK)}")
    print(f"GEMINI_API_KEY: {os.getenv('GEMINI_API_KEY')}")
    try:
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        print(f"Available Gemini models: {models[:5]}...")
    except Exception as e:
        print(f"Could not list models: {e}")
    app.run(debug=True)
