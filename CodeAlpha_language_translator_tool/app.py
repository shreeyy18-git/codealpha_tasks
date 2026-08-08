import streamlit as st
from deep_translator import GoogleTranslator
from gtts import gTTS
import base64
import io

# ---------------- UI CONFIG ----------------
st.set_page_config(page_title="Language Translator", page_icon="🌍")

st.title("🌍 Language Translation Tool")
st.write("Translate text from ANY language to ANY language + Listen to audio 🔊")

# ---------------- LANGUAGE LIST ----------------
languages = {
    "English": "en",
    "Hindi": "hi",
    "Telugu": "te",
    "Tamil": "ta",
    "Kannada": "kn",
    "Malayalam": "ml",
    "French": "fr",
    "Spanish": "es",
    "German": "de",
    "Chinese": "zh-CN",
    "Japanese": "ja",
    "Korean": "ko",
    "Arabic": "ar"
}

# ---------------- INPUT ----------------
text = st.text_area("Enter text here")

col1, col2 = st.columns(2)

with col1:
    source_lang = st.selectbox("Source Language", list(languages.keys()))

with col2:
    target_lang = st.selectbox("Target Language", list(languages.keys()))

# ---------------- TRANSLATION ----------------
translated = ""

if st.button("Translate"):
    if text.strip() == "":
        st.warning("Please enter text to translate")
    else:
        try:
            translated = GoogleTranslator(
                source=languages[source_lang],
                target=languages[target_lang]
            ).translate(text)

            st.success("Translated Text:")
            st.write(translated)

            # Save in session (important for audio button)
            st.session_state["translated"] = translated

        except Exception as e:
            st.error(f"Translation Error: {e}")

# ---------------- GET TRANSLATED TEXT ----------------
translated = st.session_state.get("translated", "")

# ---------------- COPY BUTTON ----------------
if translated:
    if st.button("📋 Copy Text"):
        st.code(translated)
        st.success("Copy manually from above box")

# ---------------- BEST AUDIO SYSTEM ----------------
def autoplay_audio(audio_bytes):
    audio_base64 = base64.b64encode(audio_bytes).decode()
    audio_html = f"""
    <audio controls autoplay>
        <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
    </audio>
    """
    st.markdown(audio_html, unsafe_allow_html=True)
    st.audio(audio_bytes)
# ---------------- TEXT TO SPEECH ----------------
if translated:
    if st.button("🔊 Listen"):
        try:
            tts = gTTS(text=translated, lang="en")

            audio_file = "voice.mp3"
            tts.save(audio_file)

            with open(audio_file, "rb") as f:
                audio_bytes = f.read()

            st.audio(audio_bytes, format="audio/mp3")

        except Exception as e:
            st.error(f"Audio Error: {e}")