<<<<<<< HEAD
# 🤖 CodeAlpha FAQ Chatbot

I built this intelligent FAQ chatbot for the **CodeAlpha Artificial Intelligence Internship** as my **Task 2** project. It combines traditional NLP matching with an LLM fallback to provide useful, conversational answers.

---

## 📋 Project Overview

This chatbot leverages **Natural Language Processing (NLP)** and **Large Language Models (LLMs)** to answer user questions about the CodeAlpha AI Internship. It uses a hybrid approach:

1. **NLP Preprocessing** – Tokenizes, cleans, and lemmatizes user input using SpaCy.
2. **Intent Matching** – Uses **TF-IDF Vectorizer** and **Cosine Similarity** to match user queries against a curated FAQ dataset.
3. **LLM Fallback** – When no good FAQ match is found (similarity below threshold), it falls back to an LLM API for conversational responses.
4. **LLM Enhancement** – For high-similarity matches, the FAQ answer is optionally formatted politely by the LLM.

---

## 🎯 TASK 2 Requirements Met

| Requirement | Implementation |
|---|---|
| **FAQ Dataset** | 25 comprehensive FAQs in `faq_data.json` covering internship details, perks, tasks, submission, etc. |
| **NLP Preprocessing** | SpaCy pipeline for tokenization, stop word removal, punctuation cleaning, and lemmatization |
| **Intent Matching** | TF-IDF Vectorizer (with unigrams + bigrams) + Cosine Similarity scoring |
| **LLM Fallback** | OpenAI-compatible API (chatgpt-4o) when similarity < 0.60 threshold |
| **LLM Enhancement** | Matched FAQ answers are optionally reformatted politely by the LLM |
| **FastAPI Backend** | `/chat` endpoint with Pydantic validation and CORS support |
| **Modern Chat UI** | ChatGPT-style interface with message bubbles, typing indicator, and suggestion chips |
| **Environment Variables** | API key stored in `.env` file, loaded via `python-dotenv` |

---

## 📁 Project Structure

```
CodeAlpha_Chatbot_FAQ/
│
├── backend/
│   ├── main.py              # FastAPI server with /chat, /health, /faqs endpoints
│   ├── nlp_engine.py        # NLP preprocessing, TF-IDF, Cosine Similarity, LLM fallback
│   ├── faq_data.json        # FAQ dataset (25 questions & answers)
│   ├── requirements.txt     # Python dependencies
│   └── .env                 # Environment variables (API_KEY, BASE_URL, MODEL)
│
├── frontend/
│   ├── index.html           # Chat UI structure with welcome screen & suggestion chips
│   ├── style.css            # Modern ChatGPT-inspired responsive styling
│   └── script.js            # Fetch API communication, message handling, typing indicator
│
└── README.md                # This documentation file
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| **Backend Framework** | FastAPI (Python) |
| **NLP Library** | SpaCy (en_core_web_sm model) |
| **Vectorization** | Scikit-learn TF-IDF Vectorizer |
| **Similarity Metric** | Cosine Similarity (sklearn) |
| **LLM Integration** | OpenAI Python SDK (custom base URL) |
| **Frontend** | HTML5, CSS3, Vanilla JavaScript |
| **Environment Config** | python-dotenv |
| **Server** | Uvicorn (ASGI) |

---

## 🚀 Setup & Installation

### Prerequisites

- **Python 3.9+** installed on your system
- **pip** package manager
- A modern web browser (Chrome, Firefox, Edge, Safari)

### Step 1: Clone or Download the Project

```bash
# If using Git
git clone https://github.com/shreeyy18-git/codealpha_tasks.git
cd codealpha_tasks/CodeAlpha_FAQ_Chatbot
```

### Step 2: Set Up the Backend

```bash
# Navigate to the backend directory
cd backend

# Create a virtual environment (recommended)
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Download the SpaCy English language model
python -m spacy download en_core_web_sm
```

### Step 3: Configure Environment Variables

The `.env` file is already included in the `backend/` directory with the following configuration:

```env
API_KEY=
BASE_URL=
MODEL=chatgpt-4o
SIMILARITY_THRESHOLD=0.60
HOST=0.0.0.0
PORT=8000
```

> ⚠️ **Security Note**: Never commit your `.env` file to a public repository. Add it to `.gitignore`.

### Step 4: Start the Backend Server

```bash
# Make sure you're in the backend/ directory with venv activated
cd backend
python main.py
```

The server will start at **http://localhost:8000**. You should see:

```
============================================================
  CodeAlpha FAQ Chatbot - Starting Server
============================================================
  Host: 0.0.0.0
  Port: 8000
  Docs: http://0.0.0.0:8000/docs
============================================================
```

### Step 5: Open the Frontend

Simply open the `frontend/index.html` file in your web browser:

```bash
# Option 1: Double-click index.html in your file explorer

# Option 2: Open from terminal (macOS)
open frontend/index.html

# Option 3: Open from terminal (Linux)
xdg-open frontend/index.html

# Option 4: Open from terminal (Windows)
start frontend/index.html
```

You can also use a simple HTTP server:

```bash
# From the project root directory
cd frontend
python -m http.server 3000
# Then open http://localhost:3000 in your browser
```

---

## 💬 How It Works

### Architecture Flow

```
User Input
    │
    ▼
┌─────────────────────────┐
│   SpaCy Preprocessing   │  Tokenize → Clean → Lemmatize
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│   TF-IDF Vectorization  │  Convert text to numerical vectors
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│   Cosine Similarity     │  Find best matching FAQ
└──────────┬──────────────┘
           │
     ┌─────┴──────┐
     │ Score ≥ 0.60│  Score < 0.60
     ▼             ▼
┌──────────┐  ┌──────────────┐
│ Return   │  │ LLM Fallback │
│ FAQ      │  │ Generate new │
│ Answer   │  │ response     │
│ (+format)│  │              │
└──────────┘  └──────────────┘
```

### NLP Preprocessing Pipeline

1. **Tokenization** – Text is split into individual tokens using SpaCy's tokenizer.
2. **Lowercasing** – All text is converted to lowercase for uniformity.
3. **Stop Word Removal** – Common English stop words (the, is, at, etc.) are removed.
4. **Punctuation Removal** – Punctuation and whitespace tokens are filtered out.
5. **Lemmatization** – Each token is converted to its base dictionary form (e.g., "running" → "run", "interns" → "intern").

### Intent Matching

- The preprocessed FAQ questions are transformed into TF-IDF vectors using **unigrams and bigrams**.
- When a user sends a message, it is preprocessed and transformed using the same vectorizer.
- **Cosine Similarity** is computed between the user's vector and each FAQ vector.
- The FAQ with the highest similarity score is selected as the best match.

### LLM Fallback & Enhancement

- **Threshold**: If the best similarity score is **below 0.60**, the system falls back to the LLM API.
- **Fallback**: The user's question is sent to the LLM with a system prompt about CodeAlpha, generating a conversational response.
- **Enhancement**: If the score is **above 0.60**, the matched FAQ answer is sent to the LLM with a prompt to rephrase it politely and conversationally.

---

## 🔌 API Endpoints

### POST `/chat`

Send a message and receive a chatbot response.

**Request Body:**
```json
{
  "message": "What is the CodeAlpha AI Internship?",
  "use_llm_formatting": true
}
```

**Response:**
```json
{
  "response": "The CodeAlpha AI Internship is a fantastic virtual program...",
  "source": "faq_llm_formatted",
  "similarity_score": 0.85,
  "matched_question": "What is the CodeAlpha Artificial Intelligence Internship?"
}
```

**Response Sources:**
| Source | Description |
|---|---|
| `faq_direct` | Raw FAQ answer returned directly (when `use_llm_formatting` is false) |
| `faq_llm_formatted` | FAQ answer reformatted politely by the LLM |
| `llm_fallback` | LLM-generated response when no good FAQ match found |

### GET `/health`

Check API health and status.

### GET `/faqs`

List all FAQ questions with their IDs.

### GET `/stats`

Get engine statistics (FAQ count, TF-IDF matrix shape, threshold, etc.).

### GET `/docs`

Interactive Swagger UI documentation.

---

## 🧪 Testing

### Test the API Directly

```bash
# Health check
curl http://localhost:8000/health

# Send a chat message
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the CodeAlpha AI Internship?", "use_llm_formatting": true}'

# List all FAQs
curl http://localhost:8000/faqs
```

### Test via Swagger UI

Open **http://localhost:8000/docs** in your browser for an interactive API testing interface.

---

## 📝 Key Design Decisions

1. **SpaCy over NLTK**: SpaCy provides a more efficient and modern NLP pipeline with better lemmatization accuracy and faster processing compared to NLTK.

2. **TF-IDF with Bigrams**: Using both unigrams and bigrams captures more context in the FAQ matching process, improving accuracy for queries that match phrases rather than individual words.

3. **LLM Enhancement**: Instead of just returning raw FAQ answers, the LLM reformulates them conversationally, making the chatbot feel more natural and engaging.

4. **Singleton Pattern**: The NLP engine is initialized once and reused across all requests, avoiding the overhead of loading SpaCy models repeatedly.

5. **Vanilla JS Frontend**: A clean, dependency-free frontend ensures easy setup and no build tools required, while still providing a professional ChatGPT-style experience.

---

## 📜 License

This project is built as part of the **CodeAlpha Artificial Intelligence Internship** (TASK 2). Feel free to use and modify for educational purposes.

---

## 👤 Author

**Shreeyansh**
GitHub: [@shreeyy18-git](https://github.com/shreeyy18-git)
CodeAlpha Artificial Intelligence Intern
Task 2: FAQ Chatbot with NLP and LLM Fallback
=======
# 🤖 CodeAlpha FAQ Chatbot

I built this intelligent FAQ chatbot for the **CodeAlpha Artificial Intelligence Internship** as my **Task 2** project. It combines traditional NLP matching with an LLM fallback to provide useful, conversational answers.

---

## 📋 Project Overview

This chatbot leverages **Natural Language Processing (NLP)** and **Large Language Models (LLMs)** to answer user questions about the CodeAlpha AI Internship. It uses a hybrid approach:

1. **NLP Preprocessing** – Tokenizes, cleans, and lemmatizes user input using SpaCy.
2. **Intent Matching** – Uses **TF-IDF Vectorizer** and **Cosine Similarity** to match user queries against a curated FAQ dataset.
3. **LLM Fallback** – When no good FAQ match is found (similarity below threshold), it falls back to an LLM API for conversational responses.
4. **LLM Enhancement** – For high-similarity matches, the FAQ answer is optionally formatted politely by the LLM.

---

## 🎯 TASK 2 Requirements Met

| Requirement | Implementation |
|---|---|
| **FAQ Dataset** | 25 comprehensive FAQs in `faq_data.json` covering internship details, perks, tasks, submission, etc. |
| **NLP Preprocessing** | SpaCy pipeline for tokenization, stop word removal, punctuation cleaning, and lemmatization |
| **Intent Matching** | TF-IDF Vectorizer (with unigrams + bigrams) + Cosine Similarity scoring |
| **LLM Fallback** | OpenAI-compatible API (chatgpt-4o) when similarity < 0.60 threshold |
| **LLM Enhancement** | Matched FAQ answers are optionally reformatted politely by the LLM |
| **FastAPI Backend** | `/chat` endpoint with Pydantic validation and CORS support |
| **Modern Chat UI** | ChatGPT-style interface with message bubbles, typing indicator, and suggestion chips |
| **Environment Variables** | API key stored in `.env` file, loaded via `python-dotenv` |

---

## 📁 Project Structure

```
CodeAlpha_Chatbot_FAQ/
│
├── backend/
│   ├── main.py              # FastAPI server with /chat, /health, /faqs endpoints
│   ├── nlp_engine.py        # NLP preprocessing, TF-IDF, Cosine Similarity, LLM fallback
│   ├── faq_data.json        # FAQ dataset (25 questions & answers)
│   ├── requirements.txt     # Python dependencies
│   └── .env                 # Environment variables (API_KEY, BASE_URL, MODEL)
│
├── frontend/
│   ├── index.html           # Chat UI structure with welcome screen & suggestion chips
│   ├── style.css            # Modern ChatGPT-inspired responsive styling
│   └── script.js            # Fetch API communication, message handling, typing indicator
│
└── README.md                # This documentation file
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| **Backend Framework** | FastAPI (Python) |
| **NLP Library** | SpaCy (en_core_web_sm model) |
| **Vectorization** | Scikit-learn TF-IDF Vectorizer |
| **Similarity Metric** | Cosine Similarity (sklearn) |
| **LLM Integration** | OpenAI Python SDK (custom base URL) |
| **Frontend** | HTML5, CSS3, Vanilla JavaScript |
| **Environment Config** | python-dotenv |
| **Server** | Uvicorn (ASGI) |

---

## 🚀 Setup & Installation

### Prerequisites

- **Python 3.9+** installed on your system
- **pip** package manager
- A modern web browser (Chrome, Firefox, Edge, Safari)

### Step 1: Clone or Download the Project

```bash
# If using Git
git clone https://github.com/shreeyy18-git/codealpha_tasks.git
cd codealpha_tasks/CodeAlpha_FAQ_Chatbot
```

### Step 2: Set Up the Backend

```bash
# Navigate to the backend directory
cd backend

# Create a virtual environment (recommended)
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Download the SpaCy English language model
python -m spacy download en_core_web_sm
```

### Step 3: Configure Environment Variables

The `.env` file is already included in the `backend/` directory with the following configuration:

```env
API_KEY=
BASE_URL=
MODEL=chatgpt-4o
SIMILARITY_THRESHOLD=0.60
HOST=0.0.0.0
PORT=8000
```

> ⚠️ **Security Note**: Never commit your `.env` file to a public repository. Add it to `.gitignore`.

### Step 4: Start the Backend Server

```bash
# Make sure you're in the backend/ directory with venv activated
cd backend
python main.py
```

The server will start at **http://localhost:8000**. You should see:

```
============================================================
  CodeAlpha FAQ Chatbot - Starting Server
============================================================
  Host: 0.0.0.0
  Port: 8000
  Docs: http://0.0.0.0:8000/docs
============================================================
```

### Step 5: Open the Frontend

Simply open the `frontend/index.html` file in your web browser:

```bash
# Option 1: Double-click index.html in your file explorer

# Option 2: Open from terminal (macOS)
open frontend/index.html

# Option 3: Open from terminal (Linux)
xdg-open frontend/index.html

# Option 4: Open from terminal (Windows)
start frontend/index.html
```

You can also use a simple HTTP server:

```bash
# From the project root directory
cd frontend
python -m http.server 3000
# Then open http://localhost:3000 in your browser
```

---

## 💬 How It Works

### Architecture Flow

```
User Input
    │
    ▼
┌─────────────────────────┐
│   SpaCy Preprocessing   │  Tokenize → Clean → Lemmatize
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│   TF-IDF Vectorization  │  Convert text to numerical vectors
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│   Cosine Similarity     │  Find best matching FAQ
└──────────┬──────────────┘
           │
     ┌─────┴──────┐
     │ Score ≥ 0.60│  Score < 0.60
     ▼             ▼
┌──────────┐  ┌──────────────┐
│ Return   │  │ LLM Fallback │
│ FAQ      │  │ Generate new │
│ Answer   │  │ response     │
│ (+format)│  │              │
└──────────┘  └──────────────┘
```

### NLP Preprocessing Pipeline

1. **Tokenization** – Text is split into individual tokens using SpaCy's tokenizer.
2. **Lowercasing** – All text is converted to lowercase for uniformity.
3. **Stop Word Removal** – Common English stop words (the, is, at, etc.) are removed.
4. **Punctuation Removal** – Punctuation and whitespace tokens are filtered out.
5. **Lemmatization** – Each token is converted to its base dictionary form (e.g., "running" → "run", "interns" → "intern").

### Intent Matching

- The preprocessed FAQ questions are transformed into TF-IDF vectors using **unigrams and bigrams**.
- When a user sends a message, it is preprocessed and transformed using the same vectorizer.
- **Cosine Similarity** is computed between the user's vector and each FAQ vector.
- The FAQ with the highest similarity score is selected as the best match.

### LLM Fallback & Enhancement

- **Threshold**: If the best similarity score is **below 0.60**, the system falls back to the LLM API.
- **Fallback**: The user's question is sent to the LLM with a system prompt about CodeAlpha, generating a conversational response.
- **Enhancement**: If the score is **above 0.60**, the matched FAQ answer is sent to the LLM with a prompt to rephrase it politely and conversationally.

---

## 🔌 API Endpoints

### POST `/chat`

Send a message and receive a chatbot response.

**Request Body:**
```json
{
  "message": "What is the CodeAlpha AI Internship?",
  "use_llm_formatting": true
}
```

**Response:**
```json
{
  "response": "The CodeAlpha AI Internship is a fantastic virtual program...",
  "source": "faq_llm_formatted",
  "similarity_score": 0.85,
  "matched_question": "What is the CodeAlpha Artificial Intelligence Internship?"
}
```

**Response Sources:**
| Source | Description |
|---|---|
| `faq_direct` | Raw FAQ answer returned directly (when `use_llm_formatting` is false) |
| `faq_llm_formatted` | FAQ answer reformatted politely by the LLM |
| `llm_fallback` | LLM-generated response when no good FAQ match found |

### GET `/health`

Check API health and status.

### GET `/faqs`

List all FAQ questions with their IDs.

### GET `/stats`

Get engine statistics (FAQ count, TF-IDF matrix shape, threshold, etc.).

### GET `/docs`

Interactive Swagger UI documentation.

---

## 🧪 Testing

### Test the API Directly

```bash
# Health check
curl http://localhost:8000/health

# Send a chat message
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the CodeAlpha AI Internship?", "use_llm_formatting": true}'

# List all FAQs
curl http://localhost:8000/faqs
```

### Test via Swagger UI

Open **http://localhost:8000/docs** in your browser for an interactive API testing interface.

---

## 📝 Key Design Decisions

1. **SpaCy over NLTK**: SpaCy provides a more efficient and modern NLP pipeline with better lemmatization accuracy and faster processing compared to NLTK.

2. **TF-IDF with Bigrams**: Using both unigrams and bigrams captures more context in the FAQ matching process, improving accuracy for queries that match phrases rather than individual words.

3. **LLM Enhancement**: Instead of just returning raw FAQ answers, the LLM reformulates them conversationally, making the chatbot feel more natural and engaging.

4. **Singleton Pattern**: The NLP engine is initialized once and reused across all requests, avoiding the overhead of loading SpaCy models repeatedly.

5. **Vanilla JS Frontend**: A clean, dependency-free frontend ensures easy setup and no build tools required, while still providing a professional ChatGPT-style experience.

---

## 📜 License

This project is built as part of the **CodeAlpha Artificial Intelligence Internship** (TASK 2). Feel free to use and modify for educational purposes.

---

## 👤 Author

**Shreeyansh**
GitHub: [@shreeyy18-git](https://github.com/shreeyy18-git)
CodeAlpha Artificial Intelligence Intern
Task 2: FAQ Chatbot with NLP and LLM Fallback
>>>>>>> 767d4fe4fd49e129a8f58eedc3eddf932e98da9f
