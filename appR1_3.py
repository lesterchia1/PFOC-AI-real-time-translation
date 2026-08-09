import streamlit as st
import os
import tempfile
import torch
import gc
import asyncio
import numpy as np
from faster_whisper import WhisperModel
import edge_tts
from groq import Groq
from openai import OpenAI

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="Real‑Time Conversation Translator", layout="wide")

# ============================================================
# 🚀 API KEY CONFIGURATION
# ============================================================
try:
    from google.colab import userdata
    GROQ_API_KEY = userdata.get('GROQ_API_KEY')
    SEALION_API_KEY = userdata.get('SEALION_API_KEY')
    HF_TOKEN = userdata.get('HF_TOKEN')
except ImportError:
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
    SEALION_API_KEY = os.environ.get("SEALION_API_KEY")
    HF_TOKEN = os.environ.get("HF_TOKEN")

if not GROQ_API_KEY or not SEALION_API_KEY:
    st.error("Missing API keys. Set GROQ_API_KEY and SEALION_API_KEY in environment or secrets.")
    st.stop()

if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN

groq_client = Groq(api_key=GROQ_API_KEY)
SEALION_BASE_URL = "https://api.sea-lion.ai/v1"
sealion_client = OpenAI(api_key=SEALION_API_KEY, base_url=SEALION_BASE_URL)

# ============================================================
# 🌍 LANGUAGE & MODEL REGISTRIES
# ============================================================
SUPPORTED_LANGUAGES = [
    "Auto",
    "English", "Chinese", "Thai",
    "Malaysian Malay", "Indonesian Malay",
    "Korean", "Japanese", "Spanish", "German",
    "Hindi", "Urdu", "French", "Russian",
    "Tagalog", "Arabic", "Myanmar", "Vietnamese",
    "Khmer"
]

LANGUAGE_CODES = {
    "English": "en", "Chinese": "zh", "Thai": "th",
    "Malaysian Malay": "ms", "Indonesian Malay": "id",
    "Korean": "ko", "Japanese": "ja", "Spanish": "es",
    "German": "de", "Hindi": "hi", "Urdu": "ur",
    "French": "fr", "Russian": "ru", "Tagalog": "tl",
    "Arabic": "ar", "Myanmar": "my", "Vietnamese": "vi",
    "Khmer": "km"
}
# Map display names to more LLM‑friendly labels
LANGUAGE_PROMPT_NAMES = {
    "Malaysian Malay": "Malay (ms)",
    "Indonesian Malay": "Indonesian (id)",
    "English": "English",
    "Chinese": "Chinese",
    "Thai": "Thai",
    "Korean": "Korean",
    "Japanese": "Japanese",
    "Spanish": "Spanish",
    "German": "German",
    "Hindi": "Hindi",
    "Urdu": "Urdu",
    "French": "French",
    "Russian": "Russian",
    "Tagalog": "Tagalog",
    "Arabic": "Arabic",
    "Myanmar": "Myanmar",
    "Vietnamese": "Vietnamese",
    "Khmer": "Khmer"
}

AVAILABLE_MODELS = {
    "SEA-LION v4 27B": "aisingapore/Gemma-SEA-LION-v4-27B-IT",
    "Qwen3 32B": "qwen/qwen3-32b",
    "kimi-k2": "moonshotai/kimi-k2-instruct-0905",
    "Llama-3.3 70B": "llama-3.3-70b-versatile",
    "Llama-3.1 instant 8B": "llama-3.1-8b-instant",
    "Llama-4 guard 12B": "meta-llama/llama-guard-4-12b"
}

# ============================================================
# 🎙️ FAST-WHISPER (GPU/CPU auto-detect) – cached
# ============================================================
@st.cache_resource
def load_whisper_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    st.write(f"🚀 Running Whisper on: {device} with {compute_type}")
    return WhisperModel("small", device=device, compute_type=compute_type)

whisper_model = load_whisper_model()

# ============================================================
# 🧹 MEMORY CLEANUP
# ============================================================
def cleanup_memory():
    temp_dir = tempfile.gettempdir()
    for f in os.listdir(temp_dir):
        if f.endswith(".mp3") or f.endswith(".wav"):
            try:
                os.remove(os.path.join(temp_dir, f))
            except:
                pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ============================================================
# 🗣️ TRANSCRIPTION – with auto‑detect
# ============================================================
def transcribe_audio(audio_bytes, input_lang_name):
    if audio_bytes is None:
        return None, None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        lang_code = None if input_lang_name == "Auto" else LANGUAGE_CODES.get(input_lang_name, "en")

        segments, info = whisper_model.transcribe(
            tmp_path,
            beam_size=1,
            vad_filter=True,
            language=lang_code,
            task="transcribe"
        )
        text = " ".join(seg.text for seg in segments).strip()
        if not text:
            segments, info = whisper_model.transcribe(
                tmp_path,
                beam_size=1,
                vad_filter=False,
                language=lang_code,
                task="transcribe"
            )
            text = " ".join(seg.text for seg in segments).strip()

        try:
            os.remove(tmp_path)
        except:
            pass

        detected_lang = None
        if input_lang_name == "Auto" and hasattr(info, 'language'):
            detected_lang = info.language

        return (text if text else None), detected_lang
    except Exception as e:
        st.error(f"Transcription error: {e}")
        return None, None

# ============================================================
# 🗣️ TRANSLATION + TTS (with detailed error reporting)
# ============================================================
def translate_and_speak(text, input_lang_name, reply_lang_name, model_choice):
    if not text:
        return None, "No text to translate."

    try:
        cleanup_memory()

        model_id = AVAILABLE_MODELS.get(model_choice, "llama-3.1-8b-instant")
        if model_choice == "SEA-LION v4 27B":
            client = sealion_client
        else:
            client = groq_client

        # Use more LLM‑friendly language names
        input_prompt = LANGUAGE_PROMPT_NAMES.get(input_lang_name, input_lang_name)
        reply_prompt = LANGUAGE_PROMPT_NAMES.get(reply_lang_name, reply_lang_name)

        system_prompt = "You are a translator. Output ONLY the translation, no explanations or extra text."
        user_prompt = f"Translate from {input_prompt} to {reply_prompt}: {text}"
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0
        )
        translation = response.choices[0].message.content.strip()

        if not translation:
            return None, "Translation was empty."

        # TTS
        reply_lang_code = LANGUAGE_CODES.get(reply_lang_name, "en")
        voice_map = {
            "en": "en-US-JennyNeural",
            "ms": "ms-MY-YasminNeural",
            "zh": "zh-CN-XiaoxiaoNeural",
            "id": "id-ID-GadisNeural",
            "th": "th-TH-PremwadeeNeural",
            "ja": "ja-JP-NanamiNeural",
            "ko": "ko-KR-SunHiNeural",
            "es": "es-ES-ElviraNeural",
            "fr": "fr-FR-DeniseNeural",
            "de": "de-DE-KatjaNeural",
            "ru": "ru-RU-SvetlanaNeural",
            "ar": "ar-EG-SalmaNeural",
            "vi": "vi-VN-HoaiMyNeural",
            "hi": "hi-IN-SwaraNeural",
            "ta": "ta-IN-PallaviNeural",
            "my": "my-MM-NilarNeural",
            "km": "km-KH-SreymomNeural",
            "tl": "fil-PH-AngeloNeural",
        }
        voice = voice_map.get(reply_lang_code, "en-US-JennyNeural")
        output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        output_file.close()

        async def tts_task():
            communicate = edge_tts.Communicate(translation, voice)
            await communicate.save(output_file.name)

        asyncio.run(tts_task())

        return translation, output_file.name

    except Exception as e:
        return None, f"API Error: {str(e)}"

# ============================================================
# 🖥️ STREAMLIT UI
# ============================================================
st.title("🎙️ R1.3 Real‑Time Conversation Translator")
st.markdown("**Auto‑detect source language** and **two‑way conversation** support.")

# Sidebar – settings
with st.sidebar:
    st.header("🌍 Settings")
    input_lang = st.selectbox("Input Language (or Auto)", SUPPORTED_LANGUAGES, index=0)
    reply_lang = st.selectbox("Reply Language", SUPPORTED_LANGUAGES, index=SUPPORTED_LANGUAGES.index("Malaysian Malay") if "Malaysian Malay" in SUPPORTED_LANGUAGES else 1)
    model_choice = st.selectbox("Translation Model", list(AVAILABLE_MODELS.keys()), index=3)

    two_way = st.checkbox("Two‑way Conversation", value=False,
                          help="Automatically swap languages after each reply.")
    if two_way:
        st.info("The app will swap Input and Reply languages after each turn.")

    # DEBUG: Show raw translation responses
    show_debug = st.checkbox("Show Debug Info", value=False)

    if st.button("Clear History", use_container_width=True):
        st.session_state.history = []
        st.session_state.reply_text = ""
        st.session_state.reply_audio = None
        st.session_state["transcription_edit"] = ""
        st.session_state.swap_flag = False
        st.session_state.last_error = ""
        st.rerun()

# Session state
if "history" not in st.session_state:
    st.session_state.history = []
if "reply_text" not in st.session_state:
    st.session_state.reply_text = ""
if "reply_audio" not in st.session_state:
    st.session_state.reply_audio = None
if "swap_flag" not in st.session_state:
    st.session_state.swap_flag = False
if "last_input" not in st.session_state:
    st.session_state.last_input = None
if "last_reply" not in st.session_state:
    st.session_state.last_reply = None
if "last_error" not in st.session_state:
    st.session_state.last_error = ""

col_left, col_right = st.columns([1, 1])

# ---- Left column ----
with col_left:
    st.subheader("🎤 Speak or Upload Audio")

    audio_data = st.audio_input("Record from microphone")
    uploaded_file = st.file_uploader("Or upload an audio file", type=["wav", "mp3", "m4a", "flac", "ogg"])

    # ----- MAIN BUTTON: Transcribe & Translate -----
    if st.button("🚀 Transcribe & Translate", use_container_width=True):
        audio_bytes = None
        if audio_data is not None:
            audio_bytes = audio_data.getvalue()
        elif uploaded_file is not None:
            audio_bytes = uploaded_file.read()

        if audio_bytes:
            with st.spinner("Processing..."):
                # Determine current languages (with swap logic)
                if two_way and st.session_state.swap_flag:
                    if st.session_state.last_input and st.session_state.last_reply:
                        current_input, current_reply = st.session_state.last_reply, st.session_state.last_input
                    else:
                        current_input, current_reply = reply_lang, input_lang
                else:
                    current_input, current_reply = input_lang, reply_lang

                # Transcribe (auto‑detect if needed)
                text, detected_lang = transcribe_audio(audio_bytes, current_input)
                if not text:
                    st.error("No speech detected. Please try again.")
                else:
                    # Determine source language display
                    if current_input == "Auto" and detected_lang:
                        for name, code in LANGUAGE_CODES.items():
                            if code == detected_lang:
                                source_display = name
                                break
                        else:
                            source_display = detected_lang
                    else:
                        source_display = current_input

                    # Translate
                    translation, audio_file = translate_and_speak(
                        text,
                        source_display,
                        current_reply,
                        model_choice
                    )
                    if translation and audio_file:
                        st.session_state.reply_text = translation
                        st.session_state.reply_audio = audio_file
                        st.session_state.history.append({
                            "role": "user",
                            "content": f"{source_display}: {text}"
                        })
                        st.session_state.history.append({
                            "role": "assistant",
                            "content": f"{current_reply}: {translation}"
                        })
                        st.session_state["transcription_edit"] = text
                        st.session_state.last_error = ""
                        st.success("Translation ready!")

                        if two_way:
                            st.session_state.last_input = current_input
                            st.session_state.last_reply = current_reply
                            st.session_state.swap_flag = not st.session_state.swap_flag
                            if st.session_state.swap_flag:
                                st.info("Swapped languages for next turn.")
                    else:
                        error_msg = audio_file if audio_file else "Unknown error"
                        st.session_state.last_error = error_msg
                        st.error(f"Translation failed: {error_msg}")
        else:
            st.warning("No audio to process. Record or upload first.")

    # Editable transcription
    st.subheader("📝 Transcription (Edit if needed)")
    transcribed_edit = st.text_area(
        "Transcription",
        height=100,
        key="transcription_edit",
        label_visibility="hidden"
    )

    # Manual translation button (for edited text)
    if st.button("🔄 Translate (edit) & Reply", type="primary", use_container_width=True):
        if not transcribed_edit.strip():
            st.warning("Please enter or speak some text.")
        else:
            with st.spinner("Translating..."):
                # Determine languages (same swap logic)
                if two_way and st.session_state.swap_flag:
                    if st.session_state.last_input and st.session_state.last_reply:
                        current_input, current_reply = st.session_state.last_reply, st.session_state.last_input
                    else:
                        current_input, current_reply = reply_lang, input_lang
                else:
                    current_input, current_reply = input_lang, reply_lang

                translation, audio_file = translate_and_speak(
                    transcribed_edit,
                    current_input,
                    current_reply,
                    model_choice
                )
                if translation and audio_file:
                    st.session_state.reply_text = translation
                    st.session_state.reply_audio = audio_file
                    st.session_state.history.append({
                        "role": "user",
                        "content": f"{current_input}: {transcribed_edit}"
                    })
                    st.session_state.history.append({
                        "role": "assistant",
                        "content": f"{current_reply}: {translation}"
                    })
                    st.session_state.last_error = ""
                    st.success("Translation ready!")
                    if two_way:
                        st.session_state.last_input = current_input
                        st.session_state.last_reply = current_reply
                        st.session_state.swap_flag = not st.session_state.swap_flag
                        if st.session_state.swap_flag:
                            st.info("Swapped languages for next turn.")
                else:
                    error_msg = audio_file if audio_file else "Unknown error"
                    st.session_state.last_error = error_msg
                    st.error(f"Translation failed: {error_msg}")

    # Show debug info if enabled
    if show_debug and st.session_state.last_error:
        st.warning(f"Debug: Last error = {st.session_state.last_error}")
    if show_debug and st.session_state.reply_text:
        st.info(f"Debug: Translation = {st.session_state.reply_text}")

# ---- Right column ----
with col_right:
    st.subheader("💬 Translation")
    if st.session_state.reply_text:
        st.markdown(f"**Translation**: {st.session_state.reply_text}")
        st.audio(st.session_state.reply_audio, format="audio/mp3")
    else:
        st.info("Translation and audio will appear here.")

    st.subheader("📜 Conversation History")
    if st.session_state.history:
        for msg in st.session_state.history:
            if msg["role"] == "user":
                st.chat_message("user").write(msg["content"])
            else:
                st.chat_message("assistant").write(msg["content"])
    else:
        st.info("No conversation yet.")
