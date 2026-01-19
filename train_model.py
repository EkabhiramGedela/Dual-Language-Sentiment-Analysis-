import pandas as pd
import numpy as np
import re
import joblib
import nltk

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.sentiment import SentimentIntensityAnalyzer

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer

from scipy.sparse import hstack, csr_matrix

nltk.download("stopwords")
nltk.download("vader_lexicon")


df = pd.read_csv("output (1).csv")
df = df.dropna(subset=["text", "label"]).reset_index(drop=True)


hinglish_map = {
    "accha": "good", "acha": "good", "achha": "good",
    "bakwas": "bad", "bekar": "bad", "bura": "bad",
    "mast": "excellent", "kamaal": "excellent",
    "pyaar": "love", "mohabbat": "love",
    "mehnga": "expensive", "sasta": "cheap",
    "faltu": "useless", "ghatiya": "bad",
    "timepass": "boring", "superb": "excellent"
}

stemmer = SnowballStemmer("english")
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = [stemmer.stem(hinglish_map.get(w, w)) for w in text.split() if w not in stop_words]
    return " ".join(words)

df["clean_text"] = df["text"].apply(clean_text)


le = LabelEncoder()
df["label_encoded"] = le.fit_transform(df["label"])
label_dict = dict(zip(le.transform(le.classes_), le.classes_))


sia = SentimentIntensityAnalyzer()
df["vader_score"] = df["clean_text"].apply(lambda x: sia.polarity_scores(x)["compound"])
df["label_text"] = df["label_encoded"].map(label_dict)


noisy_idx = df[((df["vader_score"] > 0.6) & (df["label_text"]=="negative")) |
               ((df["vader_score"] < -0.6) & (df["label_text"]=="positive"))].index
df = df.drop(noisy_idx).reset_index(drop=True)


augmentation_map = {
    "good": ["nice", "great", "awesome"],
    "bad": ["worst", "poor", "terrible"],
    "love": ["like", "enjoy"],
    "boring": ["dull", "slow"],
    "excellent": ["fantastic", "amazing"]
}

def augment_text(text):
    words = text.split()
    return " ".join([np.random.choice(augmentation_map[w]) if w in augmentation_map and np.random.rand() < 0.3 else w for w in words])

aug_texts = [augment_text(t) for t in df["clean_text"]]
aug_labels = df["label_encoded"].tolist()
aug_df = pd.DataFrame({"clean_text": aug_texts, "label_encoded": aug_labels})
df = pd.concat([df[["clean_text", "label_encoded"]], aug_df], ignore_index=True)

print(f"📈 Dataset size after augmentation: {len(df)}")


X = df["clean_text"]
y = df["label_encoded"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


char_vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3,5), max_features=9000)
word_vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1,2), max_features=9000, stop_words="english")

X_train_vec = hstack([char_vectorizer.fit_transform(X_train), word_vectorizer.fit_transform(X_train)])
X_test_vec = hstack([char_vectorizer.transform(X_test), word_vectorizer.transform(X_test)])


vader_train = np.array([sia.polarity_scores(t)["compound"] for t in X_train]).reshape(-1,1)
vader_test = np.array([sia.polarity_scores(t)["compound"] for t in X_test]).reshape(-1,1)

X_train_vec = hstack([X_train_vec, csr_matrix(vader_train)])
X_test_vec = hstack([X_test_vec, csr_matrix(vader_test)])


svm = CalibratedClassifierCV(LinearSVC(class_weight="balanced"), method="sigmoid")
lr = LogisticRegression(max_iter=5000, class_weight="balanced", C=1.5)

hybrid_model = VotingClassifier([("svm", svm), ("lr", lr)], voting="soft", n_jobs=-1)


hybrid_model.fit(X_train_vec, y_train)


y_pred = hybrid_model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred) * 100
f1 = f1_score(y_test, y_pred, average="weighted") * 100

print(f"\n✅ Accuracy: {accuracy:.2f}%")
print(f"✅ Weighted F1-score: {f1:.2f}%")


joblib.dump(hybrid_model, "sentiment_model.pkl")
joblib.dump(char_vectorizer, "char_vectorizer.pkl")
joblib.dump(word_vectorizer, "word_vectorizer.pkl")
joblib.dump(le, "label_encoder.pkl")

print("\n🎯 FINAL DATA-CENTRIC HYBRID MODEL SAVED")


print("\n💬 Enter your Hinglish sentence to predict sentiment (type 'exit' to quit):")
while True:
    user_input = input(">> ")
    if user_input.lower() == "exit":
        print("Exiting...")
        break

    clean_input = clean_text(user_input)


    X_input_char = char_vectorizer.transform([clean_input])
    X_input_word = word_vectorizer.transform([clean_input])
    X_input_vec = hstack([X_input_char, X_input_word])


    vader_feat = np.array([sia.polarity_scores(clean_input)["compound"]]).reshape(-1,1)
    X_input_vec = hstack([X_input_vec, csr_matrix(vader_feat)])

    pred_num = hybrid_model.predict(X_input_vec)[0]
    pred_label = label_dict[pred_num]  # <--- Correct mapping to string label

    print(f"🎯 Predicted Sentiment: {pred_label}")
