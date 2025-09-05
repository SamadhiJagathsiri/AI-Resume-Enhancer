import pandas as pd
import re
import os
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from textstat import flesch_reading_ease

print("Starting Day 2 script with Top 3 Matches, Missing Skills & Readability...")

# -------------------------------
# Step 0: File paths
# -------------------------------
resumes_file = r"C:\AI-Resume-Enhancer\processed_resumes.csv"
jobs_file = r"C:\AI-Resume-Enhancer\job_descriptions.csv"

# -------------------------------
# Step 1: Load processed resumes
# -------------------------------
if not os.path.exists(resumes_file):
    print(f"❌ File not found: {resumes_file}")
    exit()

resumes_df = pd.read_csv(resumes_file)
print("✅ Processed resumes loaded:", resumes_df.shape)

resume_text_col = None
for col in resumes_df.columns:
    if "resume" in col.lower():
        resume_text_col = col
        break
if not resume_text_col:
    print("❌ No column containing resume text found in processed_resumes.csv")
    exit()

# -------------------------------
# Step 2: Load job descriptions
# -------------------------------
if not os.path.exists(jobs_file):
    print(f"❌ File not found: {jobs_file}")
    exit()

jobs_df = pd.read_csv(jobs_file)
print("✅ Job descriptions loaded:", jobs_df.shape)

# Fix job columns and create JobID if missing
jobs_df = jobs_df.rename(columns={
    'Job Title': 'JobTitle',
    'Job Description': 'JobDescription'
})
if 'JobID' not in jobs_df.columns:
    jobs_df['JobID'] = range(1, len(jobs_df) + 1)

# -------------------------------
# Step 3: Text cleaning
# -------------------------------
nltk.download("stopwords")
STOPWORDS = set(stopwords.words("english"))

def clean_text(text):
    text = re.sub(r"[^a-zA-Z\s]", "", str(text))
    text = text.lower()
    words = text.split()
    words = [w for w in words if w not in STOPWORDS]
    return " ".join(words)

resumes_df["Cleaned_Resume"] = resumes_df[resume_text_col].apply(clean_text)
jobs_df["Cleaned_Job"] = jobs_df["JobDescription"].apply(clean_text)

# -------------------------------
# Step 4: Extract skills
# -------------------------------
SKILL_LIST = ["python", "sql", "excel", "tableau", "machine learning", "java", "r", "c++",
              "react", "django", "aws", "tensorflow", "keras", "javascript", "flutter", "sap"]

def extract_skills(text):
    text = text.lower()
    return [skill for skill in SKILL_LIST if skill in text]

resumes_df["Skills"] = resumes_df["Cleaned_Resume"].apply(extract_skills)
jobs_df["Skills"] = jobs_df["Cleaned_Job"].apply(extract_skills)

# -------------------------------
# Step 5: Compute Resume-Job Similarity
# -------------------------------
corpus = list(resumes_df["Cleaned_Resume"]) + list(jobs_df["Cleaned_Job"])
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(corpus)

resume_vectors = X[:len(resumes_df)]
job_vectors = X[len(resumes_df):]

similarity_matrix = cosine_similarity(resume_vectors, job_vectors)
print("✅ Similarity matrix shape:", similarity_matrix.shape)

# -------------------------------
# Step 6: Find Top 3 Job Matches + Missing Skills + Readability
# -------------------------------
def missing_skills(resume_idx, job_idx):
    resume_skills = set(resumes_df.iloc[resume_idx]["Skills"])
    job_skills = set(jobs_df.iloc[job_idx]["Skills"])
    return list(job_skills - resume_skills)

resumes_df["Readability"] = resumes_df["Cleaned_Resume"].apply(flesch_reading_ease)

top_n = 3
all_matches = []

for i, row in enumerate(similarity_matrix):
    top_indices = np.argsort(row)[-top_n:][::-1]  # top 3 indices
    for rank, job_idx in enumerate(top_indices, start=1):
        missing = missing_skills(i, job_idx)
        all_matches.append({
            "ResumeIndex": i,
            "Category": resumes_df.iloc[i]["Category"] if "Category" in resumes_df.columns else "",
            "JobRank": rank,
            "JobID": jobs_df.iloc[job_idx]["JobID"],
            "JobTitle": jobs_df.iloc[job_idx]["JobTitle"],
            "MatchScore": row[job_idx],
            "MissingSkills": ", ".join(missing) if missing else "",  # formatted
            "ResumeReadability": resumes_df.iloc[i]["Readability"]
        })

matches_df = pd.DataFrame(all_matches)
print("✅ Top 3 matches per resume with formatted MissingSkills:\n", matches_df.head(10))

# -------------------------------
# Step 7: Save Results
# -------------------------------
output_file = r"C:\AI-Resume-Enhancer\resume_job_matches_top3.csv"
matches_df.to_csv(output_file, index=False)
print(f"✅ Top 3 resume-job matches saved as {output_file}")
