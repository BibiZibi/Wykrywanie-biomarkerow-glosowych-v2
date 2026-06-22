import streamlit as st
import joblib
import tempfile
import os

from main import predict_from_audio

# =========================
# MODEL
# =========================

BASE_DIR = os.path.dirname(__file__)

model_path = os.path.join(BASE_DIR, "model.pkl")
scaler_path = os.path.join(BASE_DIR, "scaler.pkl")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# =========================
# OPISY CECH
# =========================
FEATURE_TRANSLATIONS = {
    "F0": {
        "name": "Wysokość głosu (pitch)",
        "desc": "Określa wysokość dźwięku generowanego przez struny głosowe.",
        "clinical": "U osób z zaburzeniami poznawczymi często występuje słabsza kontrola wysokości głosu.",
        "impact_pos": "Niestabilna wysokość głosu zwiększa ryzyko zaburzeń.",
        "impact_neg": "Stabilna wysokość głosu wskazuje na zdrową mowę."
    },
    "loudness": {
        "name": "Głośność mowy",
        "desc": "Określa siłę sygnału dźwięku (jak głośno mówi osoba).",
        "clinical": "Zmiany głośności mogą wynikać z problemów neurologicznych.",
        "impact_pos": "Nieregularna głośność zwiększa podejrzenie zaburzeń.",
        "impact_neg": "Stała głośność wskazuje na dobrą kontrolę mowy."
    },
    "jitter": {
        "name": "Niestabilność częstotliwości",
        "desc": "Mierzy zmiany częstotliwości między kolejnymi drganiami głosu.",
        "clinical": "Podwyższony jitter oznacza brak stabilności strun głosowych.",
        "impact_pos": "Duża niestabilność zwiększa ryzyko zaburzeń.",
        "impact_neg": "Stabilna częstotliwość oznacza zdrowy głos."
    },
    "shimmer": {
        "name": "Niestabilność amplitudy",
        "desc": "Opisuje zmienność głośności między kolejnymi fragmentami sygnału.",
        "clinical": "Wysoki shimmer oznacza drżenie głosu.",
        "impact_pos": "Zmienna amplituda zwiększa ryzyko.",
        "impact_neg": "Stała amplituda wskazuje na stabilny głos."
    },
    "formant": {
        "name": "Artykulacja (formanty)",
        "desc": "Opisuje sposób kształtowania dźwięków przez usta i język.",
        "clinical": "Zaburzenia artykulacji są typowe dla chorób neurodegeneracyjnych.",
        "impact_pos": "Nieprawidłowa artykulacja zwiększa ryzyko.",
        "impact_neg": "Poprawna artykulacja oznacza zdrową mowę."
    },
    "HNR": {
        "name": "Czystość głosu",
        "desc": "Stosunek dźwięku właściwego do szumu.",
        "clinical": "Większy szum oznacza gorszą jakość głosu.",
        "impact_pos": "Większa ilość szumu zwiększa ryzyko zaburzeń.",
        "impact_neg": "Czysty głos wskazuje na dobrą kontrolę."
    },
    "spectral": {
        "name": "Widmo dźwięku",
        "desc": "Rozkład energii w częstotliwościach.",
        "clinical": "Zmiany widma mogą wskazywać na nieprawidłową mowę.",
        "impact_pos": "Nietypowe widmo zwiększa podejrzenie.",
        "impact_neg": "Naturalne widmo wskazuje na zdrową mowę."
    }
}

def translate_feature(name):
    name_low = name.lower()

    for key, info in FEATURE_TRANSLATIONS.items():
        if key.lower() in name_low:
            return info

    return {
        "name": "Inna cecha akustyczna",
        "desc": "Model analizuje różne właściwości głosu.",
        "clinical": "Nie jest to główny wskaźnik diagnostyczny.",
        "impact_pos": "Może zwiększać ryzyko.",
        "impact_neg": "Może zmniejszać ryzyko."
    }

# =========================
# SESSION
# =========================
if "audio_path" not in st.session_state:
    st.session_state.audio_path = None

if "result" not in st.session_state:
    st.session_state.result = None

if "reset_counter" not in st.session_state:
    st.session_state.reset_counter = 0

# =========================
# UI
# =========================
st.title("Detekcja Alzheimera na podstawie głosu")

st.write("Opisz obrazek i nagraj swoją wypowiedź.")

# =========================
# 🖼 OBRAZ + INSTRUKCJA
# =========================
st.subheader("Zadanie")

st.write("""
Twoim zadaniem jest opisać dokładnie to, co widzisz na obrazku.
Spróbuj mówić pełnymi zdaniami i uwzględnij wszystkie elementy sceny.
""")


image_path = os.path.join(BASE_DIR, "cookie_theft.png")

st.image(image_path, caption="Opisz ten obrazek", use_container_width=True)


# =========================
# INPUT
# =========================
audio_input = st.audio_input(
    "Nagraj głos",
    key=f"a_{st.session_state.reset_counter}"
)

if audio_input is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_input.getbuffer())
        st.session_state.audio_path = tmp.name

uploaded_file = st.file_uploader(
    "Lub wgraj plik",
    type=["wav", "mp3"],
    key=f"f_{st.session_state.reset_counter}"
)

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.getbuffer())
        st.session_state.audio_path = tmp.name

# =========================
# PODGLĄD
# =========================
if st.session_state.audio_path is not None:
    st.audio(st.session_state.audio_path)

# =========================
# ANALIZA
# =========================
if st.button("🔍 Rozpocznij analizę"):

    if st.session_state.audio_path is None:
        st.warning("Najpierw dodaj nagranie.")
    else:
        with st.spinner("Analizuję nagranie..."):
            result = predict_from_audio(
                st.session_state.audio_path,
                model,
                scaler
            )

        st.session_state.result = result

# =========================
# WYNIK
# =========================
if st.session_state.result is not None:

    r = st.session_state.result

    st.subheader("Wynik analizy:")

    if r["prediction"] == 1:
        st.error("⚠️ WYKRYTO PODEJRZENIE ALZHEIMERA")
    else:
        st.success("✅ BRAK oznak Alzheimera")

    st.write(f"Prawdopodobieństwo: {r['probability']:.2f}")

    info = translate_feature(r["feature"])

    st.subheader("Najważniejsza cecha")
    st.markdown(f"### 👉 {info['name']}")
    st.write(info["desc"])
    st.write(info["clinical"])

    if r["impact"] > 0:
        st.warning(info["impact_pos"])
    else:
        st.success(info["impact_neg"])

    st.caption(f"🔬 Cecha: {r['feature']}")

    # =========================
    # TOP 5
    # =========================
    if "top_features" in r:
        with st.expander("📊 Zobacz szczegóły"):

            for i, f in enumerate(r["top_features"], 1):
                info = translate_feature(f["name"])

                if f["impact"] > 0:
                    direction = "⬆️ zwiększa ryzyko"
                else:
                    direction = "⬇️ zmniejsza ryzyko"

                st.write(f"{i}. **{info['name']}** — {direction}")
                st.caption(info["desc"])
                st.caption(f"🔬 Cecha: {f['name']}")

# =========================
# RESET
# =========================
if st.button("🔁 Nowe nagranie"):

    st.session_state.audio_path = None
    st.session_state.result = None
    st.session_state.reset_counter += 1

    st.rerun()
