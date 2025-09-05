import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import os

print("Starting Day 1 script...")

# -------------------------------
# Step 1: Load dataset
# -------------------------------
csv_file = r"C:\AI-Resume-Enhancer\UpdatedResumeDataSet.csv"

if not os.path.exists(csv_file):
    print(f"❌ File not found: {csv_file}")
    exit()
else:
    print(f"✅ Found dataset: {csv_file}")

df = pd.read_csv(csv_file)
print("Columns in CSV:", df.columns)

# Replace 'Resume' below with the actual column name from your CSV
resume_column = "Resume"  # <-- change if your column has a different name

if resume_column not in df.columns:
    print(f"❌ Column '{resume_column}' not found in CSV!")
    exit()

print("Dataset loaded successfully!")
print(df.head(2))  # show first 2 rows

# -------------------------------
# Step 2: Clean Resume Text
# -------------------------------
nltk.download("stopwords")
STOPWORDS = set(stopwords.words("english"))

def clean_text(text):
    text = re.sub(r"[^a-zA-Z\s]", "", str(text))
    text = text.lower()
    words = text.split()
    words = [w for w in words if w not in STOPWORDS]
    return " ".join(words)

df["Cleaned_Resume"] = df[resume_column].apply(clean_text)

print("\nSample cleaned text:")
print(df[["Resume", "Cleaned_Resume"]].head(2))

# -------------------------------
# Step 3: Extract Keywords (TF-IDF)
# -------------------------------
vectorizer = TfidfVectorizer(max_features=500)
X = vectorizer.fit_transform(df["Cleaned_Resume"])

print("\nVocabulary size:", len(vectorizer.get_feature_names_out()))
print("Some Keywords:", vectorizer.get_feature_names_out()[:20])

# -------------------------------
# Step 4: Save Processed Data
# -------------------------------
processed_file = r"C:\AI-Resume-Enhancer\processed_resumes.csv"
df.to_csv(processed_file, index=False)
print(f"\n✅ Processed dataset saved as {processed_file}")
