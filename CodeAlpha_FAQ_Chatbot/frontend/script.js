/**
 * CodeAlpha FAQ Chatbot - Frontend JavaScript
 * =============================================
 * Handles:
 * - User input and message sending
 * - API communication with the FastAPI backend
 * - Message display with typing indicators
 * - Suggestion chip interactions
 * - Auto-scroll and textarea auto-resize
 *
 * Author: Shreeyansh
 * Task: TASK 2 - FAQ Chatbot with NLP and LLM Fallback
 */

// ─────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────
const API_BASE_URL = "http://localhost:8000";
const CHAT_ENDPOINT = `${API_BASE_URL}/chat`;

// ─────────────────────────────────────────────
// DOM Elements
// ─────────────────────────────────────────────
const chatMessages = document.getElementById("chatMessages");
const welcomeScreen = document.getElementById("welcomeScreen");
const messageInput = document.getElementById("messageInput");
const sendButton = document.getElementById("sendButton");
const clearChatBtn = document.getElementById("clearChat");
const suggestionChips = document.querySelectorAll(".chip");

// ─────────────────────────────────────────────
// State
// ─────────────────────────────────────────────
let isProcessing = false;
let messageCount = 0;

// ─────────────────────────────────────────────
// Initialize
// ─────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
    setupEventListeners();
    messageInput.focus();
});

/**
 * Set up all event listeners for user interactions.
 */
function setupEventListeners() {
    // Send button click
    sendButton.addEventListener("click", handleSendMessage);

    // Enter key to send (Shift+Enter for new line)
    messageInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            handleSendMessage();
        }
    });

    // Auto-resize textarea and toggle send button
    messageInput.addEventListener("input", () => {
        autoResizeTextarea();
        toggleSendButton();
    });

    // Clear chat button
    clearChatBtn.addEventListener("click", handleClearChat);

    // Suggestion chip clicks
    suggestionChips.forEach((chip) => {
        chip.addEventListener("click", () => {
            const query = chip.getAttribute("data-query");
            if (query) {
                messageInput.value = query;
                autoResizeTextarea();
                toggleSendButton();
                handleSendMessage();
            }
        });
    });
}

/**
 * Auto-resize the textarea based on content.
 * Grows up to a max height, then becomes scrollable.
 */
function autoResizeTextarea() {
    messageInput.style.height = "auto";
    messageInput.style.height = Math.min(messageInput.scrollHeight, 120) + "px";
}

/**
 * Enable/disable the send button based on input content.
 */
function toggleSendButton() {
    const hasContent = messageInput.value.trim().length > 0;
    sendButton.disabled = !hasContent || isProcessing;
}

/**
 * Handle sending a user message.
 */
async function handleSendMessage() {
    const message = messageInput.value.trim();
    if (!message || isProcessing) return;

    // Hide welcome screen on first message
    if (welcomeScreen) {
        welcomeScreen.style.display = "none";
    }

    // Add user message to chat
    addUserMessage(message);

    // Clear input and reset
    messageInput.value = "";
    autoResizeTextarea();
    toggleSendButton();

    // Show typing indicator
    isProcessing = true;
    toggleSendButton();
    const typingElement = showTypingIndicator();

    try {
        // Send message to backend API
        const response = await sendToBackend(message);

        // Remove typing indicator
        removeTypingIndicator(typingElement);

        // Add bot response to chat
        addBotMessage(
            response.response,
            response.source,
            response.similarity_score,
            response.matched_question
        );
    } catch (error) {
        // Remove typing indicator
        removeTypingIndicator(typingElement);

        // Show error message
        addErrorMessage(
            "Sorry, I couldn't connect to the server. Please make sure the backend is running and try again."
        );
        console.error("Chat error:", error);
    } finally {
        isProcessing = false;
        toggleSendButton();
        messageInput.focus();
    }
}

/**
 * Send a message to the FastAPI backend.
 *
 * @param {string} message - The user's message text.
 * @returns {Promise<Object>} The chatbot's response object.
 */
async function sendToBackend(message) {
    const response = await fetch(CHAT_ENDPOINT, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({
            message: message,
            use_llm_formatting: true,
        }),
    });

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(
            errorData.detail || `Server error: ${response.status}`
        );
    }

    return await response.json();
}

/**
 * Add a user message bubble to the chat.
 *
 * @param {string} text - The user's message text.
 */
function addUserMessage(text) {
    messageCount++;
    const messageDiv = document.createElement("div");
    messageDiv.className = "message user";
    messageDiv.innerHTML = `
        <div class="message-avatar">You</div>
        <div class="message-bubble">${escapeHtml(text)}</div>
    `;
    chatMessages.appendChild(messageDiv);
    scrollToBottom();
}

/**
 * Add a bot message bubble to the chat with metadata.
 *
 * @param {string} text - The bot's response text.
 * @param {string} source - Source of the response ('faq_direct', 'faq_llm_formatted', 'llm_fallback').
 * @param {number} similarityScore - The cosine similarity score.
 * @param {string|null} matchedQuestion - The matched FAQ question, if any.
 */
function addBotMessage(text, source, similarityScore, matchedQuestion) {
    messageCount++;
    const messageDiv = document.createElement("div");
    messageDiv.className = "message bot";

    // Determine source badge
    let sourceBadge = "";
    if (source === "faq_direct") {
        sourceBadge = `<span class="source-badge faq">FAQ Match</span>`;
    } else if (source === "faq_llm_formatted") {
        sourceBadge = `<span class="source-badge faq">FAQ + AI Enhanced</span>`;
    } else if (source === "llm_fallback") {
        sourceBadge = `<span class="source-badge llm">AI Generated</span>`;
    }

    // Similarity score display
    const scorePercent = (similarityScore * 100).toFixed(1);
    const similarityInfo = `<span class="similarity-info">Match confidence: ${scorePercent}%</span>`;

    messageDiv.innerHTML = `
        <div class="message-avatar">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M12 2a2 2 0 0 1 2 2c0 .74-.4 1.39-1 1.73V7h1a7 7 0 0 1 7 7h1a1 1 0 0 1 1 1v3a1 1 0 0 1-1 1h-1.27A7 7 0 0 1 7.27 19H6a1 1 0 0 1-1-1v-3a1 1 0 0 1 1-1h1a7 7 0 0 1 7-7h1V5.73A2 2 0 0 1 12 2z"/>
                <circle cx="9" cy="14" r="1" fill="currentColor"/>
                <circle cx="15" cy="14" r="1" fill="currentColor"/>
            </svg>
        </div>
        <div class="message-bubble">
            ${formatResponseText(text)}
            ${sourceBadge}
            ${similarityInfo}
        </div>
    `;

    chatMessages.appendChild(messageDiv);
    scrollToBottom();
}

/**
 * Add an error message bubble to the chat.
 *
 * @param {string} text - The error message text.
 */
function addErrorMessage(text) {
    messageCount++;
    const messageDiv = document.createElement("div");
    messageDiv.className = "message bot";
    messageDiv.innerHTML = `
        <div class="message-avatar">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <circle cx="12" cy="12" r="10"></circle>
                <line x1="12" y1="8" x2="12" y2="12"></line>
                <line x1="12" y1="16" x2="12.01" y2="16"></line>
            </svg>
        </div>
        <div class="message-bubble error-bubble">${escapeHtml(text)}</div>
    `;
    chatMessages.appendChild(messageDiv);
    scrollToBottom();
}

/**
 * Show a typing indicator while waiting for the bot response.
 *
 * @returns {HTMLElement} The typing indicator element.
 */
function showTypingIndicator() {
    const typingDiv = document.createElement("div");
    typingDiv.className = "typing-indicator";
    typingDiv.id = "typingIndicator";
    typingDiv.innerHTML = `
        <div class="message-avatar">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M12 2a2 2 0 0 1 2 2c0 .74-.4 1.39-1 1.73V7h1a7 7 0 0 1 7 7h1a1 1 0 0 1 1 1v3a1 1 0 0 1-1 1h-1.27A7 7 0 0 1 7.27 19H6a1 1 0 0 1-1-1v-3a1 1 0 0 1 1-1h1a7 7 0 0 1 7-7h1V5.73A2 2 0 0 1 12 2z"/>
                <circle cx="9" cy="14" r="1" fill="currentColor"/>
                <circle cx="15" cy="14" r="1" fill="currentColor"/>
            </svg>
        </div>
        <div class="typing-dots">
            <span></span>
            <span></span>
            <span></span>
        </div>
    `;
    chatMessages.appendChild(typingDiv);
    scrollToBottom();
    return typingDiv;
}

/**
 * Remove the typing indicator from the chat.
 *
 * @param {HTMLElement} element - The typing indicator element.
 */
function removeTypingIndicator(element) {
    if (element && element.parentNode) {
        element.parentNode.removeChild(element);
    }
}

/**
 * Handle clearing the chat history.
 */
function handleClearChat() {
    // Remove all messages
    const messages = chatMessages.querySelectorAll(".message, .typing-indicator");
    messages.forEach((msg) => msg.remove());

    // Show welcome screen again
    if (welcomeScreen) {
        welcomeScreen.style.display = "flex";
    }

    messageCount = 0;
    messageInput.focus();
}

/**
 * Scroll the chat to the bottom (latest message).
 */
function scrollToBottom() {
    requestAnimationFrame(() => {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    });
}

/**
 * Escape HTML characters to prevent XSS attacks.
 *
 * @param {string} text - Raw text that might contain HTML.
 * @returns {string} Escaped text safe for insertion into HTML.
 */
function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
}

/**
 * Format response text for display.
 * Converts newlines to <br> tags and preserves basic formatting.
 *
 * @param {string} text - The raw response text.
 * @returns {string} Formatted HTML string.
 */
function formatResponseText(text) {
    // Escape HTML first for safety
    let formatted = escapeHtml(text);
    // Convert newlines to <br>
    formatted = formatted.replace(/\n/g, "<br>");
    // Bold text between ** **
    formatted = formatted.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
    return formatted;
}
