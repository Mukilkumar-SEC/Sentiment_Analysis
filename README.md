# Sentiment_Analysis
## Resume Job Category Prediction using NLP & Logistic Regression

This project classifies resumes into job categories (e.g., Data Science, HR, Mechanical Engineer) using NLP techniques and a Logistic Regression model.

---

## 📁 Dataset Info

- Dataset: `UpdatedResumeDataSet.csv`
- Records: 962 resumes
- Columns: 
  - `Category`: Job category label
  - `Resume`: Resume text

---

## 🔧 Full Code (Step-by-Step)

### 📌 1. Read the Dataset

```python
import pandas as pd

csv_path = "UpdatedResumeDataSet.csv"

try:
    df = pd.read_csv(csv_path)
except UnicodeDecodeError:
    df = pd.read_csv(csv_path, encoding="latin1")

print("Shape:", df.shape)
print("Columns:", list(df.columns))
```

### 🧼 2. Clean the Text

```python
import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['cleaned_resume'] = df['Resume'].apply(clean_text)
```

### 🔠 3. Tokenization & Lemmatization

```python
import spacy

nlp = spacy.load("en_core_web_sm")

def tokenize_text(text):
    doc = nlp(text)
    return [token.text for token in doc]

def remove_stopwords(tokens):
    return [token for token in tokens if token.lower() not in nlp.Defaults.stop_words]

def lemmatize_tokens(tokens):
    doc = nlp(" ".join(tokens))
    return [token.lemma_ for token in doc if token.is_alpha]

df['tokens'] = df['cleaned_resume'].apply(tokenize_text)
df['tokens_no_stop'] = df['tokens'].apply(remove_stopwords)
df['final_tokens'] = df['tokens_no_stop'].apply(lemmatize_tokens)
df['final_text'] = df['final_tokens'].apply(lambda x: " ".join(x))
```

### 📊 4. TF-IDF Vectorization

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['final_text'])
y = df['Category']
```

### 🤖 5. Train/Test Split & Logistic Regression

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
```

### ✅ 6. Model Evaluation

```python
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
plt.figure(figsize=(10,6))
sns.heatmap(cm, annot=True, cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

print(classification_report(y_test, y_pred))
```

### 🔍 7. Predict on New Resume

```python
import numpy as np

def preprocess_single_resume(text):
    cleaned = clean_text(text)
    tokens = tokenize_text(cleaned)
    tokens_no_stop = remove_stopwords(tokens)
    lemmas = lemmatize_tokens(tokens_no_stop)
    return " ".join(lemmas)

def predict_category(resume_text, top_k=3):
    final_text = preprocess_single_resume(resume_text)
    X_new = tfidf.transform([final_text])
    pred = model.predict(X_new)[0]
    proba = model.predict_proba(X_new)[0]
    classes = model.classes_
    top_idx = np.argsort(proba)[::-1][:top_k]
    return pred, [(classes[i], float(proba[i])) for i in top_idx]
```

### 🧪 8. Batch Prediction Test

```python
roles = {
    "HR": "Experienced HR professional with recruitment, onboarding, payroll, appraisal, and HRMS knowledge.",
    "Web Developer": "Skilled in HTML, CSS, JS, React, Node.js, Docker, SQL.",
    "Mechanical Engineer": "Expert in CAD, FEA, thermodynamics, and manufacturing processes."
}

for role_name, resume_txt in roles.items():
    pred, top3 = predict_category(resume_txt)
    print(f"\n=== True role: {role_name} ===")
    print("Predicted:", pred)
    print("Top-3 probabilities:")
    for lbl, p in top3:
        print(f"  {lbl}: {p:.4f}")
```

---

## ✅ Results Summary

- **Accuracy:** ~99% on test set
- **Features:** 5000 TF-IDF words
- **Classifier:** Logistic Regression
- **Tokenizer/Lemmatizer:** spaCy

---

## 📌 Future Improvements

- Use transformer models like BERT for semantic classification.
- Deploy with Flask or Streamlit for web-based classification.
- Add more job role categories and resume samples.
