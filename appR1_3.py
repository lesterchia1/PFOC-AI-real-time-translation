import streamlit as st
import os
import tempfile
import torch
import gc
import asyncio
import numpy as np
import io
from faster_whisper import WhisperModel
import edge_tts
from groq import Groq
from openai import OpenAI

# Pre-processing dependencies
import scipy.io.wavfile as wavfile
import noisereduce as nr
from pydub import AudioSegment, effects

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
LANGUAGE_PROMPT_NAMES = {
    "Malaysian Malay": "Malay (ms)",
    "Indonesian Malay": "Indonesian (id)",
    "English": "English",
    "Chinese": "Simplified Chinese",
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
    return WhisperModel("medium", device=device, compute_type=compute_type)

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
# 🔊 AUDIO PRE-PROCESSING (FIXED: NO SIGNAL CLIPPING)
# ============================================================
def preprocess_audio(audio_bytes):
    """
    Decodes audio to PCM WAV, normalizes volume, applies noise reduction,
    and safely clips 16-bit PCM float values to prevent digital distortion.
    """
    try:
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
        audio_segment = audio_segment.set_channels(1).set_frame_rate(16000)
        audio_segment = effects.normalize(audio_segment)
        
        samples = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
        rate = audio_segment.frame_rate

        # Moderate noise reduction (0.30 preserves vocal formants)
        reduced_noise_data = nr.reduce_noise(
            y=samples, 
            sr=rate, 
            prop_decrease=0.30,
            stationary=True
        )
        
        # Clamp bounds to [-32768, 32767] to avoid integer overflow static
        clipped_data = np.clip(reduced_noise_data, -32768, 32767).astype(np.int16)
        
        processed_segment = AudioSegment(
            clipped_data.tobytes(),
            frame_rate=rate,
            sample_width=2,
            channels=1
        )
        
        out_buffer = io.BytesIO()
        processed_segment.export(out_buffer, format="wav")
        return out_buffer.getvalue()
    except Exception as e:
        return audio_bytes

# ============================================================
# 🗣️ TRANSCRIPTION (FIXED: OPTIMIZED VAD & FALLBACK DECODE)
# ============================================================
def transcribe_audio(audio_bytes, input_lang_name):
    if audio_bytes is None:
        return None, None
    try:
        cleaned_audio_bytes = preprocess_audio(audio_bytes)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(cleaned_audio_bytes)
            tmp_path = tmp.name

        lang_code = None if input_lang_name == "Auto" else LANGUAGE_CODES.get(input_lang_name, "en")

        # Relaxed VAD parameters so soft/accented voice isn't cut off
        custom_vad_params = dict(
            threshold=0.20,             # Lowered from 0.30 to detect quiet speech
            min_speech_duration_ms=100, # Lowered from 200ms
            max_speech_duration_s=float("inf"),
            min_silence_duration_ms=500,
            speech_pad_ms=400           # Generous speech padding
        )

        segments, info = whisper_model.transcribe(
            tmp_path,
            beam_size=5,
            best_of=5,
            vad_filter=True,
            vad_parameters=custom_vad_params,
            language=lang_code,
            task="transcribe",
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
            compression_ratio_threshold=2.4
        )
        
        text = " ".join(seg.text for seg in segments).strip()

        # Fallback decode: If VAD filtered out all speech, retry without VAD
        if not text:
            segments, info = whisper_model.transcribe(
                tmp_path,
                beam_size=5,
                vad_filter=False,
                language=lang_code,
                task="transcribe",
                condition_on_previous_text=False
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
# 🗣️ TRANSLATION + TTS (with fallback and detailed errors)
# ============================================================
def translate_and_speak(text, input_lang_name, reply_lang_name, model_choice):
    if not text:
        return None, None, "No text to translate."

    try:
        cleanup_memory()

        model_id = AVAILABLE_MODELS.get(model_choice, "llama-3.1-8b-instant")
        if model_choice == "SEA-LION v4 27B":
            client = sealion_client
        else:
            client = groq_client

        input_prompt = LANGUAGE_PROMPT_NAMES.get(input_lang_name, input_lang_name)
        reply_prompt = LANGUAGE_PROMPT_NAMES.get(reply_lang_name, reply_lang_name)

        attempts = [
            f"Translate the following {input_prompt} text to {reply_prompt}. Do not add any extra text. Text: {text}",
            f"Translate this from {input_prompt} to {reply_prompt}: '{text}'",
            f"Translate '{text}' from {input_prompt} to {reply_prompt}"
        ]
        translation = None
        translation_error = None

        for prompt in attempts:
            response = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": "You are a translator. Output ONLY the translation, no explanations or extra text."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0
            )
            translation = response.choices[0].message.content.strip()
            if translation and translation != text:
                break
            elif translation == text:
                translation_error = "Model returned the same text."
            else:
                translation_error = "Empty translation."

        if not translation or translation == text:
            translation = text
            translation_error = "Translation failed; using original text."

        st.session_state.raw_translation = translation
        st.session_state.raw_prompt = attempts[0]

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

        tts_success = False
        tts_error = None

        try:
            async def tts_task():
                communicate = edge_tts.Communicate(translation, voice)
                await communicate.save(output_file.name)
            asyncio.run(tts_task())
            if os.path.exists(output_file.name) and os.path.getsize(output_file.name) > 0:
                tts_success = True
        except Exception as e:
            tts_error = str(e)
            try:
                fallback_voice = "en-US-JennyNeural"
                async def fallback_tts():
                    communicate = edge_tts.Communicate(translation, fallback_voice)
                    await communicate.save(output_file.name)
                asyncio.run(fallback_tts())
                if os.path.exists(output_file.name) and os.path.getsize(output_file.name) > 0:
                    tts_success = True
                    tts_error = f"Primary voice failed ({e}), fallback used."
            except Exception as e2:
                tts_error = f"Primary: {e}, Fallback: {e2}"

        if tts_success:
            return translation, output_file.name, None
        else:
            return translation, None, f"TTS failed: {tts_error}"

    except Exception as e:
        return text, None, f"API Error: {str(e)}"

# ============================================================
# 🖥️ STREAMLIT UI
# ============================================================
st.title("🎙️ R1.3 Real‑Time Conversation Translator_Advanced")
st.markdown("**Auto‑detect source language** and **two‑way conversation** support.")
st.markdown("**Best for Professional use and Advanced** | **SPEED: ⭐⭐⭐ , Accuracy: ⭐⭐⭐⭐⭐ , User Experience: ⭐⭐⭐⭐⭐** ")

with st.sidebar:
    st.header("🌍 Settings")
    input_lang = st.selectbox("Input Language (or Auto)", SUPPORTED_LANGUAGES, index=0)
    reply_lang = st.selectbox("Reply Language", SUPPORTED_LANGUAGES,
                              index=SUPPORTED_LANGUAGES.index("Malaysian Malay") if "Malaysian Malay" in SUPPORTED_LANGUAGES else 1)
    model_choice = st.selectbox("Translation Model", list(AVAILABLE_MODELS.keys()), index=3)

    two_way = st.checkbox("Two‑way Conversation", value=False,
                          help="Automatically swap languages after each reply.")
    if two_way:
        st.info("The app will swap Input and Reply languages after each turn.")

    show_debug = st.checkbox("Show Debug Info", value=False)

    if st.button("Clear History", use_container_width=True):
        st.session_state.history = []
        st.session_state.reply_text = ""
        st.session_state.reply_audio = None
        st.session_state["transcription_edit"] = ""
        st.session_state.swap_flag = False
        st.session_state.last_error = ""
        st.session_state.raw_translation = ""
        st.session_state.raw_prompt = ""
        st.session_state.last_tts_error = ""
        st.rerun()

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
if "raw_translation" not in st.session_state:
    st.session_state.raw_translation = ""
if "raw_prompt" not in st.session_state:
    st.session_state.raw_prompt = ""
if "last_tts_error" not in st.session_state:
    st.session_state.last_tts_error = ""

col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("🎤 Speak or Upload Audio")

    audio_data = st.audio_input("Record from microphone")
    uploaded_file = st.file_uploader("Or upload an audio file", type=["wav", "mp3", "m4a", "flac", "ogg"])

    if st.button("🚀 Transcribe & Translate", use_container_width=True):
        audio_bytes = None
        if audio_data is not None:
            audio_bytes = audio_data.getvalue()
        elif uploaded_file is not None:
            audio_bytes = uploaded_file.read()

        if audio_bytes:
            with st.spinner("Processing..."):
                if two_way and st.session_state.swap_flag:
                    if st.session_state.last_input and st.session_state.last_reply:
                        current_input, current_reply = st.session_state.last_reply, st.session_state.last_input
                    else:
                        current_input, current_reply = reply_lang, input_lang
                else:
                    current_input, current_reply = input_lang, reply_lang

                text, detected_lang = transcribe_audio(audio_bytes, current_input)
                if not text:
                    st.error("No speech detected. Please try again.")
                else:
                    if current_input == "Auto" and detected_lang:
                        for name, code in LANGUAGE_CODES.items():
                            if code == detected_lang:
                                source_display = name
                                break
                        else:
                            source_display = detected_lang
                    else:
                        source_display = current_input

                    translation, audio_file, tts_err = translate_and_speak(
                        text,
                        source_display,
                        current_reply,
                        model_choice
                    )
                    if translation:
                        st.session_state.reply_text = translation
                        if audio_file and os.path.exists(audio_file) and os.path.getsize(audio_file) > 0:
                            st.session_state.reply_audio = audio_file
                            st.session_state.last_tts_error = ""
                        else:
                            st.session_state.reply_audio = None
                            st.session_state.last_tts_error = tts_err or "Unknown TTS error"

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
                        error_msg = tts_err or "Unknown error"
                        st.session_state.last_error = error_msg
                        st.error(f"Translation failed: {error_msg}")
        else:
            st.warning("No audio to process. Record or upload first.")

    st.subheader("📝 Transcription (Edit if needed)")
    transcribed_edit = st.text_area(
        "Transcription",
        height=100,
        key="transcription_edit",
        label_visibility="hidden"
    )

    if st.button("🔄 Translate (edit) & Reply", type="primary", use_container_width=True):
        if not transcribed_edit.strip():
            st.warning("Please enter or speak some text.")
        else:
            with st.spinner("Translating..."):
                if two_way and st.session_state.swap_flag:
                    if st.session_state.last_input and st.session_state.last_reply:
                        current_input, current_reply = st.session_state.last_reply, st.session_state.last_input
                    else:
                        current_input, current_reply = reply_lang, input_lang
                else:
                    current_input, current_reply = input_lang, reply_lang

                translation, audio_file, tts_err = translate_and_speak(
                    transcribed_edit,
                    current_input,
                    current_reply,
                    model_choice
                )
                if translation:
                    st.session_state.reply_text = translation
                    if audio_file and os.path.exists(audio_file) and os.path.getsize(audio_file) > 0:
                        st.session_state.reply_audio = audio_file
                        st.session_state.last_tts_error = ""
                    else:
                        st.session_state.reply_audio = None
                        st.session_state.last_tts_error = tts_err or "Unknown TTS error"

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
                    error_msg = tts_err or "Unknown error"
                    st.session_state.last_error = error_msg
                    st.error(f"Translation failed: {error_msg}")

    if show_debug:
        if st.session_state.last_error:
            st.warning(f"Error: {st.session_state.last_error}")
        if st.session_state.raw_prompt:
            st.text_area("Last Prompt", st.session_state.raw_prompt, height=80)
        if st.session_state.raw_translation:
            st.text_area("Raw Translation", st.session_state.raw_translation, height=80)
        if st.session_state.last_tts_error:
            st.warning(f"TTS Error: {st.session_state.last_tts_error}")

with col_right:
    st.subheader("💬 Translation")
    if st.session_state.reply_text:
        st.markdown(f"**Translation**: {st.session_state.reply_text}")
        if st.session_state.reply_audio and os.path.exists(st.session_state.reply_audio):
            st.audio(st.session_state.reply_audio, format="audio/mp3")
        else:
            st.warning(f"Audio not available. Reason: {st.session_state.last_tts_error or 'Unknown error'}")
    else:
        st.info("Translation will appear here.")

    st.subheader("📜 Conversation History")
    if st.session_state.history:
        for msg in st.session_state.history:
            if msg["role"] == "user":
                st.chat_message("user").write(msg["content"])
            else:
                st.chat_message("assistant").write(msg["content"])
    else:
        st.info("No conversation yet.")
