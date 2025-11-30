# quiz-from-youtube.py
import os
import json
import copy
import uuid
import tempfile
from pathlib import Path
from zipfile import ZipFile
from urllib.parse import urlparse, parse_qs 

import streamlit as st
from openai import OpenAI


# ---------- Helper: OpenAI client ----------
@st.cache_resource 
def get_openai_client(api_key: str) -> OpenAI:
    """
    Cache de OpenAI client, zodat deze niet bij elke Streamlit herlading opnieuw wordt gemaakt.
    """
    return OpenAI(api_key=api_key)


# ---------- NIEUWE FUNCTIE: Transcriptie via chatgpt ----------

def get_transcript_from_youtube(url: str, client: OpenAI) -> str:
    """
    Gebruikt de GPT-4o API's multimodale capaciteiten om de
    inhoud van de YouTube-video-URL te analyseren en de transcriptie/inhoud terug te geven.
    """
    prompt = f"""
    Analyseer de inhoud van deze YouTube-video. 
    Het doel is om meerkeuzevragen te genereren. 
    Geef de volledige, uitgeschreven tekst (transcript) van de video terug, indien mogelijk, 
    of een zeer gedetailleerde samenvatting van de gehele inhoud. 
    Concentreer je op de feitelijke en educatieve inhoud. 
    De URL is: {url}
    """
    
    response = client.chat.completions.create(
        model="gpt-4o-mini", # Meest kosteneffectieve en snelle multimodale model
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    # Geef de URL door aan de AI (dit is hoe GPT-4o een URL leest)
                    {"type": "text", "text": url} 
                ]
            }
        ],
        temperature=0.0
    )
    
    # De AI geeft nu de transcriptie/samenvatting als tekst terug
    return response.choices[0].message.content

# ---------- Vragen genereren met gekozen taal (ONGEWIJZIGD) ----------
def generate_mc_from_text(
    text: str,
    n_questions: int = 5,
    question_language: str = "Nederlands",
    client: OpenAI = None,
):
    """
    Genereert n_questions meerkeuzevragen op basis van de aangeleverde tekst.
    question_language: taal waarin vragen en antwoorden moeten staan.
    """
    prompt = f"""
Je krijgt de uitgeschreven tekst van een video (transcript van de audio).
Maak {n_questions} meerkeuzevragen in het {question_language} over de inhoud.

Regels:
- Doelgroep: volwassen cursisten.
- Elke vraag:
  - 1 duidelijke vraagzin.
  - 4 antwoordmogelijkheden.
  - Slechts één juist antwoord.
- Maak inhoudelijke vragen (geen triviale details of losse woordjes).
- Schrijf ALLES in het {question_language} (zowel vragen als antwoorden).

Geef ALLEEN geldig JSON terug in dit formaat:

{{
  "questions": [
    {{
      "question": "vraagtekst",
      "answers": ["antwoord A", "antwoord B", "antwoord C", "antwoord D"],
      "correct_index": 0
    }}
  ]
}}

Tekst van de video:
\"\"\"{text}\"\"\"    
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": "Je maakt didactische meerkeuzevragen en geeft geldig JSON terug.",
            },
            {"role": "user", "content": prompt},
        ],
    )

    raw = response.choices[0].message.content
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        raise RuntimeError("Model antwoordde geen geldige JSON (kon niet parsen).")

    return data.get("questions", [])


# ---------- H5P helpers (ONGEWIJZIGD) ----------
def build_questions_from_mc(mc_questions, template_content):
    """
    Zet je eigen mc_questions om naar H5P.MultiChoice vragen,
    met behoud van de instellingen (feedback, knoppen, gedrag)
    uit de eerste vraag van de template.
    """
    base_q = template_content["questions"][0]  # eerste vraag als sjabloon
    base_params = base_q["params"]

    new_questions = []
    for i, q in enumerate(mc_questions, start=1):
        q_obj = copy.deepcopy(base_q)
        params = q_obj["params"]

        # Vraagtekst
        params["question"] = q["question"]

        # Antwoorden
        answers = []
        correct_idx = q["correct_index"]
        for idx, ans in enumerate(q["answers"]):
            ans_obj = copy.deepcopy(base_params["answers"][0])
            ans_obj["text"] = ans
            ans_obj["correct"] = bool(idx == correct_idx)
            answers.append(ans_obj)
        params["answers"] = answers

        # Metadata / ID
        q_obj["metadata"]["title"] = f"Vraag {i}"
        q_obj["metadata"]["extraTitle"] = f"Vraag {i}"
        q_obj["subContentId"] = str(uuid.uuid4())

        new_questions.append(q_obj)

    return new_questions


def create_h5p_from_template(template_h5p_path, output_h5p_path, mc_questions):
    """
    - Leest quiz-template.h5p
    - Vervangt de vragen door mc_questions
    - Schrijft een nieuw H5P-bestand weg.
    """
    template_h5p_path = Path(template_h5p_path)
    output_h5p_path = Path(output_h5p_path)

    with ZipFile(template_h5p_path, "r") as zin:
        # content.json inlezen
        content_json = json.loads(
            zin.read("content/content.json").decode("utf-8")
        )

        # vragen vervangen
        content_json["questions"] = build_questions_from_mc(
            mc_questions, content_json
        )

        new_content_bytes = json.dumps(
            content_json, ensure_ascii=False, indent=2
        ).encode("utf-8")

        # nieuwe .h5p schrijven
        with ZipFile(output_h5p_path, "w") as zout:
            for item in zin.infolist():
                data = zin.read(item.filename)
                if item.filename == "content/content.json":
                    data = new_content_bytes
                zout.writestr(item, data)


# ---------- Streamlit UI (AANGEPAST) ----------
st.set_page_config(page_title="YouTube → H5P-quiz", page_icon="🎬")

st.title("🎬 YouTube → 📚 H5P meerkeuzequiz")

# Logo bovenaan tonen (logo.png in dezelfde map als dit script)
logo_path = "logo.png"
if os.path.exists(logo_path):
    st.image(logo_path, width=400)
else:
    st.warning(f"Logo '{logo_path}' niet gevonden.")


st.markdown(
    """
1. Controleer of je OpenAI API-sleutel correct is ingesteld in Streamlit secrets.
2. Plak een YouTube-URL  
3. Kies aantal vragen en taal  
4. Upload (of gebruik) een H5P-template  
5. Genereer en download de nieuwe H5P-quiz
"""
)

# API-key: Gebruik st.secrets, met fallback naar invoerveld
try:
    api_key = st.secrets["OPENAI_API_KEY"]
    st.info("OpenAI API-sleutel is geladen vanuit Streamlit secrets. ✅")
except (KeyError, AttributeError):
    # Fallback: Als de key niet in secrets staat, vraag er dan om
    api_key = st.text_input(
        "OpenAI API-sleutel",
        type="password",
        help="OpenAI API-sleutel niet gevonden in `st.secrets`. Vul de sleutel hier handmatig in.",
    )
    if api_key:
        st.warning("Handmatige sleutel ingevoerd. Let op: beter om `st.secrets` te gebruiken.")
    
# YouTube URL
youtube_url = st.text_input("YouTube-URL", value="")

# Aantal vragen
aantal_vragen = st.number_input(
    "Hoeveel meerkeuzevragen wil je genereren?",
    min_value=1,
    max_value=30,
    value=5,
)

# Taalkeuze
taal_opties = [
    "Nederlands",
    "Engels",
    "Frans",
    "Duits",
    "Spaans",
    "Italiaans",
]
taal_vragen = st.selectbox(
    "Taal waarin de vragen moeten staan", options=taal_opties, index=0
)

# H5P template upload of default
st.markdown("#### H5P-template")
uploaded_template = st.file_uploader(
    "Upload een H5P-template (bv. quiz-template.h5p). "
    "Laat leeg om `quiz-template.h5p` uit deze map te gebruiken.",
    type="h5p",
)

if st.button("🚀 Genereer H5P-quiz"):
    if not api_key:
        st.error("Vul eerst je OpenAI API-sleutel in, of stel deze in via Streamlit secrets.")
    elif not youtube_url.strip():
        st.error("Vul een geldige YouTube-URL in.")
    else:
        try:
            # Client aanmaken (gebruikt cache)
            client = get_openai_client(api_key)

            # De tijdelijke map is nog steeds nodig voor het opslaan van de H5P output
            with tempfile.TemporaryDirectory() as tmpdir_str:
                tmpdir = Path(tmpdir_str)

                with st.status("Bezig met verwerken...", expanded=True) as status:
                    
                    # 1️⃣ Inhoud analyseren via GPT-4o-mini
                    status.write("1️⃣ Inhoud analyseren via OpenAI GPT-4o-mini...")
                    full_text = get_transcript_from_youtube(youtube_url, client)
                    status.write(f"✅ Transcript/Inhoud klaar (lengte: {len(full_text)} tekens)")

                    # 2️⃣ Vragen Genereren
                    status.write(
                        f"2️⃣ {aantal_vragen} meerkeuzevragen genereren in het {taal_vragen}..."
                    )
                    mc_questions = generate_mc_from_text(
                        full_text,
                        n_questions=aantal_vragen,
                        question_language=taal_vragen,
                        client=client,
                    )
                    if not mc_questions:
                        raise RuntimeError("Geen vragen teruggekregen van het model.")
                    status.write(f"✅ {len(mc_questions)} vragen ontvangen.")

                    # 3️⃣ H5P opbouwen
                    status.write("3️⃣ H5P-bestand opbouwen...")

                    # Template opslaan (als upload) of standaardbestand gebruiken
                    if uploaded_template is not None:
                        template_path = tmpdir / "template.h5p"
                        template_path.write_bytes(uploaded_template.read())
                    else:
                        template_path = Path("quiz-template.h5p")
                        if not template_path.exists():
                            raise FileNotFoundError(
                                "quiz-template.h5p niet gevonden in de huidige map "
                                "en er is geen template geüpload."
                            )

                    output_name = f"quiz-from-youtube-{uuid.uuid4().hex[:8]}.h5p"
                    output_path = tmpdir / output_name

                    create_h5p_from_template(template_path, output_path, mc_questions)
                    status.write(f"✅ H5P-quiz aangemaakt: {output_name}")

                    status.update(label="Klaar! ✅", state="complete", expanded=False)

                # Vragen tonen
                st.markdown("### Voorbeeld van de gegenereerde vragen")
                for i, q in enumerate(mc_questions, start=1):
                    with st.expander(f"Vraag {i}"):
                        st.write(q["question"])
                        for idx, ans in enumerate(q["answers"]):
                            label = chr(ord("A") + idx)
                            st.write(f"- **{label}.** {ans}")
                        correct_label = chr(ord("A") + q["correct_index"])
                        st.write(f"✅ Juiste antwoord: **{correct_label}**")

                # Downloadknop
                file_bytes = output_path.read_bytes()
                st.download_button(
                    "⬇️ Download H5P-quiz",
                    data=file_bytes,
                    file_name=output_name,
                    mime="application/zip",
                )

        except Exception as e:
            st.error(f"Er ging iets mis: {e}")