import streamlit as st
import pandas as pd
import re, nltk, os
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textstat import flesch_reading_ease, gunning_fog, smog_index
from sentence_transformers import SentenceTransformer

# ---------------------------
# Load Kaggle Datasets
# ---------------------------
RESUME_CSV = "UpdatedResumeDataSet.csv"
JOB_CSV = "job_descriptions.csv"

@st.cache_data
def load_and_prepare():
    # Load CSVs
    resumes = pd.read_csv(RESUME_CSV)
    jobs = pd.read_csv(JOB_CSV)

    # Clean text
    nltk.download("stopwords")
    STOPWORDS = set(stopwords.words("english"))

    def clean_text(text):
        text = re.sub(r"[^a-zA-Z\s]", "", str(text))
        text = text.lower()
        return " ".join(w for w in text.split() if w not in STOPWORDS)

    resumes["Cleaned"] = resumes["Resume"].apply(clean_text)
    jobs.rename(columns={"Job Title": "JobTitle", "Job Description": "JobDescription"}, inplace=True)
    jobs["Cleaned"] = jobs["JobDescription"].apply(clean_text)

    # Build dynamic skills vocabulary (TF-IDF keywords)
    vectorizer = TfidfVectorizer(max_features=500)
    vectorizer.fit(list(resumes["Cleaned"]) + list(jobs["Cleaned"]))
    dynamic_skills = vectorizer.get_feature_names_out().tolist()

    return resumes, jobs, dynamic_skills

resumes_df, jobs_df, SKILLS_LIST = load_and_prepare()

# ---------------------------
# Helper Functions
# ---------------------------
def extract_text(file):
    from PyPDF2 import PdfReader
    import docx
    if file.type == "application/pdf":
        return "".join([page.extract_text() for page in PdfReader(file).pages])
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        d = docx.Document(file)
        return " ".join(p.text for p in d.paragraphs)
    else:
        return file.getvalue().decode("utf-8")

def find_missing_skills(resume_text, job_text):
    job_skills = [s for s in SKILLS_LIST if s in job_text.lower()]
    resume_skills = [s for s in SKILLS_LIST if s in resume_text.lower()]
    return list(set(job_skills) - set(resume_skills))

def calculate_readability(text):
    return {
        "Flesch": round(flesch_reading_ease(text),2),
        "Fog": round(gunning_fog(text),2),
        "SMOG": round(smog_index(text),2)
    }

def interpret_readability(scores):
    f = scores["Flesch"]
    if f>=60: desc = "🟢 Easy"
    elif f>=30: desc = "🟠 Moderate"
    else: desc = "🔴 Hard"
    return f"Flesch={f} {desc}, Fog={scores['Fog']}, SMOG={scores['SMOG']}"

def compute_similarity(resume_text, job_text):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    emb = model.encode([resume_text, job_text])
    return cosine_similarity([emb[0]],[emb[1]])[0][0]

# ---------------------------
# UI
# ---------------------------
st.set_page_config(page_title="AI Resume Enhancer (Dynamic Skills)", layout="wide")
st.title("📄 AI Resume Enhancer – Dynamic Skills Version")

option = st.radio("Choose Input Mode:", ["Upload Files","Use Kaggle Samples"])

if option == "Upload Files":
    resume_file = st.file_uploader("Upload Resume", type=["pdf","docx","txt"])
    job_file = st.file_uploader("Upload Job Description", type=["pdf","docx","txt"])

    if resume_file and job_file and st.button("🔍 Analyse Resume"):
        resume_text = extract_text(resume_file)
        job_text = extract_text(job_file)
        score = compute_similarity(resume_text, job_text)
        missing = find_missing_skills(resume_text, job_text)
        readability = calculate_readability(resume_text)

        st.metric("Match Score", f"{score*100:.2f}%")
        st.write("Missing Skills:", ", ".join(missing) if missing else "🎉 None")
        st.write(interpret_readability(readability))

elif option == "Use Kaggle Samples":
    resume_idx = st.selectbox("Choose a sample resume:", range(len(resumes_df)))
    job_idx = st.selectbox("Choose a job:", range(len(jobs_df)))
    if st.button("Analyse Sample"):
        resume_text = resumes_df.iloc[resume_idx]["Cleaned"]
        job_text = jobs_df.iloc[job_idx]["Cleaned"]
        score = compute_similarity(resume_text, job_text)
        missing = find_missing_skills(resume_text, job_text)
        readability = calculate_readability(resume_text)

        st.metric("Match Score", f"{score*100:.2f}%")
        st.write("Missing Skills:", ", ".join(missing) if missing else "🎉 None")
        st.write(interpret_readability(readability))
