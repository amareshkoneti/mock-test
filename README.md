Aptitude Mock Test Generator (Flask + Gemini)

Setup:
1. python -m venv venv
2. source venv/bin/activate   # or venv\Scripts\activate on Windows
3. pip install -r requirements.txt
4. copy .env.example -> .env and set GEMINI_API_KEY
5. python app.py
6. Visit http://127.0.0.1:5000

Notes:
- Quizzes are stored as JSON files in /quizzes.
- Do not expose GEMINI_API_KEY in frontend.
- For production, secure the server and use a persistent store if desired.
