import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Load your CSV
df = pd.read_csv('spam_data.csv')  # Make sure this file is in the same folder

# Rename columns to match what the model expects
df.rename(columns={'spam': 'label', 'text': 'message'}, inplace=True)

# Convert labels: if spam column contains text like "spam"/"ham", map to 1/0
if df['label'].dtype == 'object':
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

# TF-IDF vectorization
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train logistic regression
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Evaluate
y_pred = model.predict(X_test_tfidf)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Predict user input
while True:
    msg = input("\nEnter a message to classify (or type 'exit'): ")
    if msg.lower() == 'exit':
        break
    msg_vector = vectorizer.transform([msg])
    prediction = model.predict(msg_vector)[0]
    print("Prediction:", "Spam ❌" if prediction == 1 else "Not Spam ✅")
