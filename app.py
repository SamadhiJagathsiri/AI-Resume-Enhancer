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
        text = "".join([page.extract_text() for page in reader.pages])
        return text
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(file)
        text = " ".join([para.text for para in doc.paragraphs])
        return text
    elif file.type == "text/plain":
        return file.getvalue().decode("utf-8")
    else:
        return ""

def find_missing_skills(resume_text, job_text):
    """Return missing skills compared to job description."""
    job_skills = [s for s in SKILLS_LIST if s.lower() in job_text.lower()]
    resume_skills = [s for s in SKILLS_LIST if s.lower() in resume_text.lower()]
    missing = list(set(job_skills) - set(resume_skills))
    return missing

def calculate_readability(text):
    """Return multiple readability scores."""
    return {
        "Flesch Reading Ease": round(flesch_reading_ease(text), 2),
        "Gunning Fog Index": round(gunning_fog(text), 2),
        "SMOG Index": round(smog_index(text), 2)
    }

def compute_similarity(resume_text, job_text):
    """Compute semantic similarity using Sentence-BERT."""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode([resume_text, job_text])
    sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return sim

# -------------------------------
# Streamlit App
# -------------------------------

# Page Config
st.set_page_config(
    page_title="AI Resume Enhancer",
    layout="wide",
    page_icon="📄"
)

# Background Color
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #f0f9ff, #e0f7fa);
    }
    .stFileUploader > div>div {
        background-color: #ffffff;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title
st.title("📄 AI Resume Enhancer")
st.markdown("Upload your resume and a job description to get a **match score**, **missing skills**, and **readability feedback**.")

# Upload Section
st.subheader("📑 Upload Resume & Job Description")
resume_file = st.file_uploader("Upload Resume (PDF, DOCX, TXT)", type=["pdf","docx","txt"])
job_file = st.file_uploader("Upload Job Description (PDF, DOCX, TXT)", type=["pdf","docx","txt"])

if resume_file and job_file:
    with st.spinner("Processing files..."):
        resume_text = extract_text(resume_file)
        job_text = extract_text(job_file)

        # Semantic match score
        match_score = compute_similarity(resume_text, job_text)

        # Missing skills
        missing_skills = find_missing_skills(resume_text, job_text)

        # Readability metrics
        readability = calculate_readability(resume_text)

    # -------------------------------
    # Layout: Columns for metrics
    # -------------------------------
    st.subheader("💼 Resume Analysis")

    col1, col2, col3 = st.columns(3)

    col1.metric("Match Score", f"{match_score*100:.2f}%")
    col2.subheader("Missing Skills")
    if missing_skills:
        col2.write(", ".join(missing_skills))
    else:
        col2.success("No missing skills! 🎉")
    col3.subheader("Readability")
    col3.json(readability)

    # -------------------------------
    # Bar Chart for Missing Skills Count
    # -------------------------------
    st.subheader("📊 Missing Skills Visualization")
    st.bar_chart([len(missing_skills)])

    # -------------------------------
    # Optional Tabs for Resume & Job Text Preview
    # -------------------------------
    st.subheader("📝 Text Preview")
    tabs = st.tabs(["Resume Text", "Job Description Text"])
    with tabs[0]:
        st.text_area("Resume Content", resume_text, height=200)
    with tabs[1]:
        st.text_area("Job Description Content", job_text, height=200)

    st.info("✅ Tip: Add missing skills to increase match score!")

else:
    st.warning("⬆️ Please upload both files to see the analysis.")
