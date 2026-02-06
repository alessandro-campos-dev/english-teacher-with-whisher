import streamlit as st
import whisper
import pyttsx3
import tempfile
import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np
import os
import time

# ===============================
# PATHS & DIRECTORIES
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR = os.path.join(BASE_DIR, "audio")
os.makedirs(AUDIO_DIR, exist_ok=True)

# ===============================
# STREAMLIT CONFIG
# ===============================
st.set_page_config(
    page_title="SpeakWise AI – English Teacher",
    page_icon="🎓",
    layout="centered"
)

# ===============================
# SESSION STATE
# ===============================
if "conversation_step" not in st.session_state:
    st.session_state.conversation_step = "idle"

if "level" not in st.session_state:
    st.session_state.level = None

if "last_lesson" not in st.session_state:
    st.session_state.last_lesson = ""

# ===============================
# LOAD WHISPER (CACHE)
# ===============================
@st.cache_resource
def load_whisper():
    return whisper.load_model("base")

whisper_model = load_whisper()

# ===============================
# AUDIO RECORDING
# ===============================
def record_audio(duration=5, fs=44100):
    st.info("🎤 Recording... Please speak clearly.")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()

    audio = recording.squeeze()
    filename = os.path.join(
        AUDIO_DIR, f"user_{int(time.time())}.wav"
    )
    wav.write(filename, fs, audio.astype(np.float32))
    return filename

# ===============================
# TEXT TO SPEECH (OFFLINE)
# ===============================
def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# ===============================
# SPEECH TO TEXT
# ===============================
def transcribe(audio_file):
    result = whisper_model.transcribe(audio_file, language="en")
    return result["text"].strip()

# ===============================
# PEDAGOGICAL ENGINE (OFFLINE)
# ===============================
def detect_level(text):
    words = len(text.split())
    if words < 5:
        return "Beginner"
    elif words < 12:
        return "Intermediate"
    else:
        return "Advanced"

def teacher_response(level):
    lessons = {
        "Beginner": "Let's start simple. Say: I am learning English.",
        "Intermediate": "Great! Tell me about your daily routine.",
        "Advanced": "Nice! What do you think about technology in education?"
    }
    return lessons[level]

# ===============================
# UI
# ===============================
st.title("🎓 SpeakWise AI – English Teacher (Offline)")
st.markdown(
    """
    **AI-powered English teacher**  
    ✔ 100% Offline  
    ✔ Voice-based  
    ✔ Continuous conversation  
    ✔ Level detection  
    """
)

st.divider()

# ===============================
# START PLACEMENT TEST
# ===============================
if st.session_state.conversation_step == "idle":
    if st.button("🎤 Start Placement Test"):
        st.session_state.conversation_step = "record"

# ===============================
# RECORD → TRANSCRIBE → TEACH
# ===============================
if st.session_state.conversation_step == "record":
    audio_path = record_audio()

    st.success("✅ Audio recorded successfully")

    student_text = transcribe(audio_path)

    st.markdown("### 🧑 Student said:")
    st.write(student_text)

    if st.session_state.level is None:
        st.session_state.level = detect_level(student_text)

    lesson = teacher_response(st.session_state.level)
    st.session_state.last_lesson = lesson

    st.markdown(f"### 🎯 Detected Level: **{st.session_state.level}**")
    st.markdown("### 👨‍🏫 Teacher:")
    st.write(lesson)

    speak(lesson)

    st.session_state.conversation_step = "waiting"

# ===============================
# CONTINUE CONVERSATION
# ===============================
if st.session_state.conversation_step == "waiting":
    if st.button("🎤 Continue Conversation"):
        st.session_state.conversation_step = "record"

# ===============================
# FOOTER
# ===============================
st.divider()
st.caption("Offline demo • No OpenAI • Educational prototype")
