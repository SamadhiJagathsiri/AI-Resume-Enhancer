# app.py
import streamlit as st
import pandas as pd
import PyPDF2
import docx
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from textstat import flesch_reading_ease, gunning_fog, smog_index

# -------------------------------
# Helper Functions
# -------------------------------
SKILLS_LIST = ["Python", "SQL", "Excel", "TensorFlow", "AWS", "JavaScript", "Java", "R", "PowerBI", "Tableau"]

def extract_text(file):
    """Extract text from PDF, DOCX, or TXT file."""
    if file.type == "application/pdf":
        reader = PyPDF2.PdfReader(file)
        return "".join([page.extract_text() for page in reader.pages])
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(file)
        return " ".join([para.text for para in doc.paragraphs])
    elif file.type == "text/plain":
        return file.getvalue().decode("utf-8")
    else:
        return ""

def find_missing_skills(resume_text, job_text):
    job_skills = [s for s in SKILLS_LIST if s.lower() in job_text.lower()]
    resume_skills = [s for s in SKILLS_LIST if s.lower() in resume_text.lower()]
    return list(set(job_skills) - set(resume_skills))

def calculate_readability(text):
    return {
        "Flesch Reading Ease": round(flesch_reading_ease(text), 2),
        "Gunning Fog Index": round(gunning_fog(text), 2),
        "SMOG Index": round(smog_index(text), 2)
    }

def interpret_readability(scores):
    flesch = scores["Flesch Reading Ease"]
    if flesch >= 60:
        flesch_desc = "🟢 Easy to read (good for most audiences)"
    elif flesch >= 30:
        flesch_desc = "🟠 Fairly difficult (may need simplification)"
    else:
        flesch_desc = "🔴 Very difficult to read (simplify sentences)"

    fog = scores["Gunning Fog Index"]
    smog = scores["SMOG Index"]

    return (
        f"**Flesch Reading Ease:** {flesch} → {flesch_desc}\n\n"
        f"**Gunning Fog Index:** {fog} (years of education needed)\n\n"
        f"**SMOG Index:** {smog} (grade level for comprehension)"
    )

def compute_similarity(resume_text, job_text):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode([resume_text, job_text])
    return cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

# -------------------------------
# Streamlit App Config
# -------------------------------
st.set_page_config(page_title="AI Resume Enhancer", layout="wide", page_icon="📄")

# Theme Toggle
mode = st.radio("🌗 Choose Theme", ["Light", "Dark"], horizontal=True)
if mode == "Light":
    st.markdown(
        """
        <style>
        .stApp {background: #fdfdfd; color: #000000 !important;}
        .stFileUploader > div>div {background-color: #ffffff; border-radius: 10px;}
        h1,h2,h3,h4,h5,h6,p,div {color: #000000 !important;}
        </style>
        """, unsafe_allow_html=True)
else:
    st.markdown(
        """
        <style>
        .stApp {background: #121212; color: #ffffff !important;}
        .stFileUploader > div>div {background-color: #1e1e1e; border-radius: 10px;}
        h1,h2,h3,h4,h5,h6,p,div {color: #ffffff !important;}
        </style>
        """, unsafe_allow_html=True)

# -------------------------------
# App Title and Upload
# -------------------------------
st.title("📄 AI Resume Enhancer")
st.markdown("Upload your resume and a job description, then click **Analyse Resume** to see results.")

st.subheader("📑 Upload Resume & Job Description")
resume_file = st.file_uploader("Upload Resume (PDF, DOCX, TXT)", type=["pdf","docx","txt"])
job_file = st.file_uploader("Upload Job Description (PDF, DOCX, TXT)", type=["pdf","docx","txt"])

# -------------------------------
# Analyse Button Logic
# -------------------------------
if resume_file and job_file:
    if st.button("🔍 Analyse Resume"):
        with st.spinner("Processing files..."):
            resume_text = extract_text(resume_file)
            job_text = extract_text(job_file)

            match_score = compute_similarity(resume_text, job_text)
            missing_skills = find_missing_skills(resume_text, job_text)
            readability = calculate_readability(resume_text)

        # -------------------------------
        # Display Results
        # -------------------------------
        st.subheader("💼 Resume Analysis")
        col1, col2, col3 = st.columns(3)

        # Match Score
        col1.metric("Match Score", f"{match_score*100:.2f}%")

        # Missing Skills
        col2.subheader("Missing Skills")
        if missing_skills:
            col2.write(", ".join(missing_skills))
        else:
            col2.success("No missing skills! 🎉")

        # Readability with explanation
        col3.subheader("Readability")
        col3.markdown(interpret_readability(readability))

        # -------------------------------
        # Missing Skills Visualization
        # -------------------------------
        st.subheader("📊 Missing Skills Visualization")
        if missing_skills:
            skill_df = pd.DataFrame({"Skills": missing_skills, "Count": [1]*len(missing_skills)})
            st.bar_chart(skill_df.set_index("Skills"))
        else:
            st.success("🎉 Nothing to visualize — your skills match perfectly!")

        # -------------------------------
        # Text Preview Tabs
        # -------------------------------
        st.subheader("📝 Text Preview")
        tabs = st.tabs(["Resume Text", "Job Description Text"])
        with tabs[0]:
            st.text_area("Resume Content", resume_text, height=200)
        with tabs[1]:
            st.text_area("Job Description Content", job_text, height=200)

        st.info("✅ Tip: Add missing skills or simplify your resume for better readability!")
else:
    st.warning("⬆️ Please upload both files before analysis.")
