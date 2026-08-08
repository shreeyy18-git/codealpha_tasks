"""
FastAPI Backend for CodeAlpha FAQ Chatbot
==========================================
Main server application that provides:
- POST /chat endpoint for user queries
- CORS middleware for frontend-backend communication
- Health check and FAQ listing endpoints

Author: Shreeyansh asati
Task: TASK 2 - FAQ Chatbot with NLP and LLM Fallback
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from nlp_engine import get_engine

# ─────────────────────────────────────────────
# Load environment variables
# ─────────────────────────────────────────────
load_dotenv()

# ─────────────────────────────────────────────
# Initialize FastAPI app
# ─────────────────────────────────────────────
app = FastAPI(
    title="CodeAlpha FAQ Chatbot API",
    description=(
        "An intelligent FAQ chatbot for the CodeAlpha AI Internship program. "
        "Uses NLP preprocessing (SpaCy), TF-IDF Vectorizer, Cosine Similarity "
        "for intent matching, and LLM fallback via OpenAI-compatible API."
    ),
    version="1.0.0",
)

# ─────────────────────────────────────────────
# Configure CORS middleware
# Allows the frontend to communicate with the backend
# ─────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],            # In production, restrict to specific origins
    allow_credentials=True,
    allow_methods=["*"],            # Allow all HTTP methods
    allow_headers=["*"],            # Allow all headers
)


# ─────────────────────────────────────────────
# Pydantic models for request/response validation
# ─────────────────────────────────────────────
class ChatRequest(BaseModel):
    """Request model for the /chat endpoint."""
    message: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="The user's question or message to the chatbot.",
    )
    use_llm_formatting: bool = Field(
        default=True,
        description="Whether to use LLM to format matched FAQ answers politely.",
    )


class ChatResponse(BaseModel):
    """Response model for the /chat endpoint."""
    response: str = Field(..., description="The chatbot's response text.")
    source: str = Field(
        ...,
        description=(
            "Source of the response: "
        "'faq_direct' (raw FAQ), "
        "'faq_llm_formatted' (FAQ formatted by LLM), "
        "'llm_fallback' (LLM-generated when no FAQ match)."
        ),
    )
    similarity_score: float = Field(
        ..., description="Cosine similarity score of the best FAQ match."
    )
    matched_question: str | None = Field(
        None, description="The matched FAQ question, if any."
    )


class HealthResponse(BaseModel):
    """Response model for the health check endpoint."""
    status: str
    faq_count: int
    model_loaded: bool


# ─────────────────────────────────────────────
# API Endpoints
# ─────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint that serves a simple welcome page."""
    return """
    <!DOCTYPE html>
    <html>
    <head><title>CodeAlpha FAQ Chatbot API</title></head>
    <body style="font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px;">
        <h1 style="color: #2563eb;">CodeAlpha FAQ Chatbot API</h1>
        <p>Welcome to the CodeAlpha AI Internship FAQ Chatbot backend!</p>
        <h3>Available Endpoints:</h3>
        <ul>
            <li><strong>POST /chat</strong> - Send a message and get a chatbot response</li>
            <li><strong>GET /health</strong> - Check API health and status</li>
            <li><strong>GET /faqs</strong> - List all FAQ questions</li>
            <li><strong>GET /docs</strong> - Interactive API documentation (Swagger UI)</li>
        </ul>
        <p>Open the <a href="/docs">API Docs</a> for more details.</p>
    </body>
    </html>
    """


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint that processes user messages.

    Workflow:
    1. Receives the user's message via POST request
    2. Preprocesses the query using SpaCy (tokenize, clean, lemmatize)
    3. Matches against FAQ dataset using TF-IDF + Cosine Similarity
    4. If similarity >= threshold: Returns FAQ answer (optionally LLM-formatted)
    5. If similarity < threshold: Falls back to LLM for conversational response

    Args:
        request: ChatRequest containing the user's message.

    Returns:
        ChatResponse with the chatbot's response, source, and similarity score.
    """
    try:
        engine = get_engine()
        result = engine.process_query(
            user_query=request.message,
            use_llm_formatting=request.use_llm_formatting,
        )
        return ChatResponse(**result)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while processing your query: {str(e)}",
        )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint to verify the API is running and the NLP engine is loaded.

    Returns:
        HealthResponse with status, FAQ count, and model load status.
    """
    try:
        engine = get_engine()
        return HealthResponse(
            status="healthy",
            faq_count=len(engine.faq_data),
            model_loaded=True,
        )
    except Exception as e:
        return HealthResponse(
            status="unhealthy",
            faq_count=0,
            model_loaded=False,
        )


@app.get("/faqs")
async def list_faqs():
    """
    List all FAQ questions in the dataset.

    Returns:
        JSON list of all FAQ questions with their IDs.
    """
    try:
        engine = get_engine()
        return JSONResponse(
            content=[
                {"id": item["id"], "question": item["question"]}
                for item in engine.faq_data
            ]
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error loading FAQs: {str(e)}",
        )


@app.get("/stats")
async def get_stats():
    """
    Get statistics about the NLP engine and FAQ dataset.

    Returns:
        JSON with engine statistics including FAQ count, TF-IDF matrix shape,
        and similarity threshold.
    """
    try:
        engine = get_engine()
        return JSONResponse(
            content={
                "faq_count": len(engine.faq_data),
                "tfidf_matrix_shape": list(engine.tfidf_matrix.shape),
                "similarity_threshold": float(os.getenv("SIMILARITY_THRESHOLD", "0.60")),
                "model_name": os.getenv("MODEL", "chatgpt-4o"),
                "base_url": os.getenv("BASE_URL", "https://api.gapgpt.app/v1"),
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting stats: {str(e)}",
        )


# ─────────────────────────────────────────────
# Run the server
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))

    print("=" * 60)
    print("  CodeAlpha FAQ Chatbot - Starting Server")
    print("=" * 60)
    print(f"  Host: {host}")
    print(f"  Port: {port}")
    print(f"  Docs: http://{host}:{port}/docs")
    print("=" * 60)

    uvicorn.run("main:app", host=host, port=port, reload=True)
