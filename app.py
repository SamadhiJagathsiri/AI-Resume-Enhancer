import streamlit as st
import pandas as pd
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import textstat

# -------------------------------
# Setup
# -------------------------------
nltk.download("stopwords")
STOPWORDS = set(stopwords.words("english"))

st.set_page_config(page_title="AI Resume Enhancer", layout="wide")

st.title("📄 AI Resume Enhancer")
st.write("Upload your resume and a job description to get a **match score, missing skills, and readability feedback.**")

# -------------------------------
# Text cleaning
# -------------------------------
def clean_text(text):
    text = re.sub(r"[^a-zA-Z\s]", " ", str(text))
    text = text.lower()
    words = [w for w in text.split() if w not in STOPWORDS]
    return " ".join(words)

# Simple skills list (you can expand this)
SKILLS = [
    "python", "java", "c++", "sql", "excel", "tableau", "powerbi",
    "tensorflow", "pytorch", "nlp", "machine learning", "deep learning",
    "flask", "django", "react", "javascript", "html", "css", "aws", "azure"
]

def extract_missing_skills(resume_text, job_text):
    resume_words = set(resume_text.lower().split())
    job_words = set(job_text.lower().split())
    return [skill for skill in SKILLS if skill in job_words and skill not in resume_words]

# -------------------------------
# File upload section
# -------------------------------
resume_file = st.file_uploader("📑 Upload Resume (TXT only)", type=["txt"])
job_file = st.file_uploader("💼 Upload Job Description (TXT only)", type=["txt"])

if resume_file and job_file:
    # Read text
    resume_text = resume_file.read().decode("utf-8")
    job_text = job_file.read().decode("utf-8")

    # Clean text
    clean_resume = clean_text(resume_text)
    clean_job = clean_text(job_text)

    # TF-IDF similarity
    vectorizer = TfidfVectorizer(max_features=1000)
    vectors = vectorizer.fit_transform([clean_resume, clean_job])
    similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]

    # Missing skills
    missing_skills = extract_missing_skills(clean_resume, clean_job)

    # Readability
    readability = textstat.flesch_reading_ease(resume_text)

    # -------------------------------
    # Results Dashboard
    # -------------------------------
    st.subheader("✅ Results")

    col1, col2, col3 = st.columns(3)
    col1.metric("Match Score", f"{similarity*100:.2f}%")
    col2.metric("Missing Skills", len(missing_skills))
    col3.metric("Readability", f"{readability:.2f}")

    st.write("### 📌 Missing Skills / Keywords")
    if missing_skills:
        st.write(", ".join(missing_skills))
    else:
        st.success("🎉 No major skills missing!")

    st.write("### 📝 Resume Preview")
    st.text_area("Your Resume", resume_text, height=200)

    st.write("### 💼 Job Description Preview")
    st.text_area("Job Description", job_text, height=200)

else:
    st.info("⬆️ Please upload both files to see the analysis.")
