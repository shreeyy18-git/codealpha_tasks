"""
NLP Engine Module for CodeAlpha FAQ Chatbot
============================================
Handles all Natural Language Processing tasks:
- Text preprocessing (tokenization, cleaning, lemmatization)
- TF-IDF vectorization
- Cosine Similarity computation for intent matching
- LLM fallback integration using OpenAI-compatible API

Author: Shreeyansh
Task: TASK 2 - FAQ Chatbot with NLP and LLM Fallback
"""

import json
import os
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from dotenv import load_dotenv

# ─────────────────────────────────────────────
# Load environment variables from .env file
# ─────────────────────────────────────────────
load_dotenv()

API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL", "https://api.gapgpt.app/v1")
MODEL = os.getenv("MODEL", "chatgpt-4o")
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.60"))

# ─────────────────────────────────────────────
# Load SpaCy English language model
# ─────────────────────────────────────────────
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    raise OSError(
        "SpaCy model 'en_core_web_sm' not found. "
        "Please run: python -m spacy download en_core_web_sm"
    )

# ─────────────────────────────────────────────
# Initialize OpenAI client with custom base URL
# ─────────────────────────────────────────────
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


class NLPEngine:
    """
    NLP Engine that handles FAQ matching and LLM fallback.

    This class encapsulates all NLP operations:
    - Loading and preprocessing FAQ data
    - Building TF-IDF vectors for FAQ questions
    - Matching user queries against FAQs using cosine similarity
    - Falling back to LLM when no good match is found
    - Optionally using LLM to format matched answers politely
    """

    def __init__(self, faq_path: str = None):
        """
        Initialize the NLP Engine with FAQ data and TF-IDF vectorizer.

        Args:
            faq_path: Path to the FAQ JSON dataset file.
                      Defaults to 'faq_data.json' in the same directory.
        """
        if faq_path is None:
            faq_path = os.path.join(os.path.dirname(__file__), "faq_data.json")

        # Load FAQ dataset
        self.faq_data = self._load_faq_data(faq_path)
        self.faq_questions = [item["question"] for item in self.faq_data]
        self.faq_answers = [item["answer"] for item in self.faq_data]

        # Preprocess FAQ questions (tokenize, clean, lemmatize)
        self.preprocessed_questions = [
            self.preprocess_text(q) for q in self.faq_questions
        ]

        # Build TF-IDF vectorizer on preprocessed FAQ questions
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),       # Use unigrams and bigrams for better matching
            max_features=5000,        # Limit vocabulary size
            stop_words="english",     # Remove English stop words
            lowercase=True,           # Convert to lowercase
        )

        # Fit and transform the preprocessed FAQ questions
        self.tfidf_matrix = self.vectorizer.fit_transform(self.preprocessed_questions)

        print(f"[NLP Engine] Loaded {len(self.faq_data)} FAQs successfully.")
        print(f"[NLP Engine] TF-IDF matrix shape: {self.tfidf_matrix.shape}")
        print(f"[NLP Engine] Similarity threshold: {SIMILARITY_THRESHOLD}")

    @staticmethod
    def _load_faq_data(faq_path: str) -> list:
        """
        Load FAQ data from a JSON file.

        Args:
            faq_path: Path to the FAQ JSON file.

        Returns:
            List of FAQ dictionaries with 'id', 'question', and 'answer' keys.

        Raises:
            FileNotFoundError: If the FAQ file does not exist.
            json.JSONDecodeError: If the file is not valid JSON.
        """
        if not os.path.exists(faq_path):
            raise FileNotFoundError(f"FAQ data file not found: {faq_path}")

        with open(faq_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list) or len(data) == 0:
            raise ValueError("FAQ data must be a non-empty list of objects.")

        print(f"[NLP Engine] Loaded {len(data)} FAQs from {faq_path}")
        return data

    @staticmethod
    def preprocess_text(text: str) -> str:
        """
        Preprocess text using SpaCy NLP pipeline.

        Performs the following operations:
        1. Tokenization - Split text into individual tokens
        2. Cleaning - Remove punctuation, whitespace, and non-alphabetic tokens
        3. Lemmatization - Convert each token to its base/dictionary form
        4. Stop word removal - Remove common English stop words

        Args:
            text: Raw input text string.

        Returns:
            Preprocessed and lemmatized text string.
        """
        # Process text through SpaCy pipeline
        doc = nlp(text.lower())

        # Extract meaningful tokens: alphabetic, not stop words, not punctuation
        meaningful_tokens = []
        for token in doc:
            if (
                token.is_alpha              # Only alphabetic characters
                and not token.is_stop       # Remove stop words
                and not token.is_punct      # Remove punctuation
                and not token.is_space      # Remove whitespace tokens
                and len(token.text) > 1     # Remove single-character tokens
            ):
                # Use lemma (base form) of the token
                meaningful_tokens.append(token.lemma_)

        # Join tokens back into a single string
        return " ".join(meaningful_tokens)

    def find_best_match(self, user_query: str) -> dict:
        """
        Find the best matching FAQ for a user query using TF-IDF + Cosine Similarity.

        Args:
            user_query: The user's input question.

        Returns:
            Dictionary containing:
            - 'match_index': Index of the best matching FAQ (-1 if no match)
            - 'similarity_score': Cosine similarity score of the best match
            - 'matched_question': The matched FAQ question text
            - 'matched_answer': The matched FAQ answer text
            - 'is_above_threshold': Whether the score exceeds the threshold
        """
        # Preprocess the user's query
        preprocessed_query = self.preprocess_text(user_query)

        # Handle empty query after preprocessing
        if not preprocessed_query.strip():
            return {
                "match_index": -1,
                "similarity_score": 0.0,
                "matched_question": "",
                "matched_answer": "",
                "is_above_threshold": False,
            }

        # Transform query using the same TF-IDF vectorizer
        query_vector = self.vectorizer.transform([preprocessed_query])

        # Compute cosine similarity between query and all FAQ questions
        similarity_scores = cosine_similarity(query_vector, self.tfidf_matrix)[0]

        # Find the best matching FAQ
        best_match_index = similarity_scores.argmax()
        best_score = float(similarity_scores[best_match_index])

        result = {
            "match_index": int(best_match_index),
            "similarity_score": best_score,
            "matched_question": self.faq_questions[best_match_index],
            "matched_answer": self.faq_answers[best_match_index],
            "is_above_threshold": best_score >= SIMILARITY_THRESHOLD,
        }

        print(f"[NLP Engine] Query: '{user_query}'")
        print(f"[NLP Engine] Best match: '{result['matched_question']}'")
        print(f"[NLP Engine] Similarity score: {best_score:.4f}")
        print(f"[NLP Engine] Above threshold ({SIMILARITY_THRESHOLD}): {result['is_above_threshold']}")

        return result

    def get_llm_response(self, user_query: str, context: str = "") -> str:
        """
        Generate a response using the LLM API (fallback mechanism).

        This method is called when:
        1. The cosine similarity score is below the threshold (no good FAQ match)
        2. The similarity is high but we want the answer formatted politely

        Args:
            user_query: The user's input question.
            context: Optional context string (e.g., the matched FAQ answer).

        Returns:
            LLM-generated response string.
        """
        if context:
            # High similarity: ask LLM to format the FAQ answer politely
            system_prompt = (
                "You are a friendly and helpful AI assistant for the CodeAlpha "
                "Artificial Intelligence Internship program. Your task is to take "
                "the provided FAQ answer and rephrase it in a warm, conversational, "
                "and polite manner while keeping all the important information intact. "
                "Keep the response concise and engaging."
            )
            user_message = (
                f"The user asked: '{user_query}'\n\n"
                f"The best matching FAQ answer is: '{context}'\n\n"
                f"Please rephrase this answer politely and conversationally."
            )
        else:
            # Low similarity: generate a full response from the LLM
            system_prompt = (
                "You are a friendly and helpful AI assistant for the CodeAlpha "
                "Artificial Intelligence Internship program. You provide accurate, "
                "encouraging, and informative responses about the internship. "
                "If the question is not related to CodeAlpha's AI Internship, "
                "politely let the user know and suggest they ask about the internship. "
                "Keep responses concise but informative."
            )
            user_message = user_query

        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=500,
                temperature=0.7,
            )
            llm_response = response.choices[0].message.content.strip()
            print(f"[NLP Engine] LLM response generated successfully.")
            return llm_response

        except Exception as e:
            error_msg = f"I'm sorry, I'm having trouble connecting to my AI service right now. Please try again in a moment. (Error: {str(e)})"
            print(f"[NLP Engine] LLM API error: {str(e)}")
            return error_msg

    def process_query(self, user_query: str, use_llm_formatting: bool = True) -> dict:
        """
        Main entry point: Process a user query and return the best response.

        Workflow:
        1. Preprocess the user's query using SpaCy (tokenize, clean, lemmatize)
        2. Find the best matching FAQ using TF-IDF + Cosine Similarity
        3. If similarity >= threshold:
           a. If use_llm_formatting is True, ask LLM to format the answer politely
           b. Otherwise, return the raw FAQ answer
        4. If similarity < threshold:
           a. Fallback to LLM to generate a conversational response

        Args:
            user_query: The user's input question.
            use_llm_formatting: Whether to use LLM to format matched answers politely.

        Returns:
            Dictionary containing:
            - 'response': The final response text
            - 'source': Source of the response ('faq_direct', 'faq_llm_formatted', 'llm_fallback')
            - 'similarity_score': The cosine similarity score
            - 'matched_question': The matched FAQ question (if any)
        """
        # Step 1 & 2: Find best FAQ match
        match_result = self.find_best_match(user_query)

        # Step 3: High similarity - return FAQ answer (optionally formatted)
        if match_result["is_above_threshold"]:
            if use_llm_formatting:
                # Use LLM to format the FAQ answer politely
                formatted_response = self.get_llm_response(
                    user_query, context=match_result["matched_answer"]
                )
                return {
                    "response": formatted_response,
                    "source": "faq_llm_formatted",
                    "similarity_score": match_result["similarity_score"],
                    "matched_question": match_result["matched_question"],
                }
            else:
                # Return the raw FAQ answer directly
                return {
                    "response": match_result["matched_answer"],
                    "source": "faq_direct",
                    "similarity_score": match_result["similarity_score"],
                    "matched_question": match_result["matched_question"],
                }

        # Step 4: Low similarity - fallback to LLM
        llm_response = self.get_llm_response(user_query, context="")
        return {
            "response": llm_response,
            "source": "llm_fallback",
            "similarity_score": match_result["similarity_score"],
            "matched_question": match_result["matched_question"] if match_result["match_index"] >= 0 else None,
        }


# ─────────────────────────────────────────────
# Module-level singleton for reuse across requests
# ─────────────────────────────────────────────
_engine_instance = None


def get_engine() -> NLPEngine:
    """
    Get or create the singleton NLP Engine instance.

    This ensures the SpaCy model and TF-IDF vectorizer
    are loaded only once and reused across all requests.

    Returns:
        NLPEngine: The shared NLP Engine instance.
    """
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = NLPEngine()
    return _engine_instance
