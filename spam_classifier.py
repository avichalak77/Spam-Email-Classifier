import pandas as pd
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# 1. Load Dataset
data = pd.read_csv("spam.csv", encoding="latin-1")[["v1", "v2"]]
data.columns = ["label", "message"]

# 2. Convert labels to numbers
data["label"] = data["label"].map({"ham": 0, "spam": 1})

# 3. Text Cleaning Function
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

data["message"] = data["message"].apply(clean_text)

# 4. Feature Extraction (TF-IDF)
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(data["message"])
y = data["label"]

# 5. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Train Model (Naive Bayes)
model = MultinomialNB()
model.fit(X_train, y_train)

# 7. Evaluate Model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 8. Test with Custom Message
sample_email = ["Congratulations! You have won a free mobile recharge"]
sample_vector = vectorizer.transform(sample_email)
result = model.predict(sample_vector)

if result[0] == 1:
    print("\nPrediction: SPAM")
else:
    print("\nPrediction: NOT SPAM")
