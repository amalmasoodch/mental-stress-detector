# import pandas as pd
# import numpy as np
# import re
# import nltk
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix)
# from sklearn.linear_model import LogisticRegression
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.svm import LinearSVC
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.decomposition import TruncatedSVD
# # For text preprocessing
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('omw-1.4')   # optional, helps with lemmatization
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
# import warnings
# warnings.filterwarnings('ignore')
# #load dataset
# df = pd.read_csv("D:\LAPTOP DATA\AI\model\Mental_Stress_Text.csv")
# print(df.head())
# print(df.info())
# print(df['status'].value_counts())
# #now data cleaning
# #Handle missing value
# df = df.dropna(subset=['statement', 'status'])
# df = df[df['statement'].str.strip() != '']  # remove empty strings
# #duplicate values
# df = df.drop_duplicates(subset=['statement', 'status'])
# #Text Preprocessing 
# stop_words = set(stopwords.words('english'))
# lemmatizer = WordNetLemmatizer()
# def clean_text(text):
#     text = text.lower()
#     text = re.sub(r'[^a-z\s]', '', text)          # keep only letters and spaces
#     text = re.sub(r'\s+', ' ', text).strip()      # remove extra spaces
#     tokens = text.split()
#     # remove stopwords and lemmatize
#     tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
#     return ' '.join(tokens)
# df['clean_statement'] = df['statement'].apply(clean_text)
# # Remove rows where text became empty after cleaning
# df = df[df['clean_statement'].str.strip() != '']
# print("Cleaning done. Shape:", df.shape)
# print(df['status'].value_counts())
# #label encoding
# label_encoder = LabelEncoder()
# df['label'] = label_encoder.fit_transform(df['status'])
# print(label_encoder.classes_)  # to see mapping
# print(df[['status', 'label']].head())
# print("Label distribution:\n", df['label'].value_counts())
# # Feature Extraction using TF-IDF
# vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
# X = vectorizer.fit_transform(df['clean_statement'])
# y = df['label']
# print("\n✅ Vectorization done")
# print("X shape:", X.shape)
# print("y shape:", y.shape)
# print("First 10 feature names:", vectorizer.get_feature_names_out()[:10])
# #Train/Test Split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42, stratify=y)
# print("Train shape:", X_train.shape, "Test shape:", X_test.shape)
# #reduce data for knn
# svd = TruncatedSVD(n_components=200, random_state=42)
# X_svd = svd.fit_transform(X)           # dense array of shape (n_samples, 200)
# X_train_svd, X_test_svd, y_train, y_test = train_test_split(X_svd, y, test_size=0.2, random_state=42, stratify=y)
# #Define Models
# models = {
#     'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced'),
#     'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
#     'Decision Tree': DecisionTreeClassifier(class_weight='balanced'),
#     'Random Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced', n_jobs=-1),
#     'Linear SVM': LinearSVC(class_weight='balanced', max_iter=2000),
#     'Naive Bayes': MultinomialNB(),
#     'Gradient Boosting': GradientBoostingClassifier(n_estimators=100)
# }
# for name, model in models.items():
#     print(f"\n--- {name} ---")
#     if name == 'K-Nearest Neighbors':
#         model.fit(X_train_svd, y_train)
#         y_pred = model.predict(X_test_svd)
#     else:
#         model.fit(X_train, y_train)
#         y_pred = model.predict(X_test)
# # Train and Evaluate Each Model
# results = []
# model_names = []
# accuracy_scores = []
# for name, model in models.items():
#     print(f"\n--- {name} ---")
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)  
#     acc = accuracy_score(y_test, y_pred)
#     prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
#     rec = recall_score(y_test, y_pred, average='macro')
#     f1 = f1_score(y_test, y_pred, average='macro')
#     results.append({
#         'Model': name,
#         'Accuracy': acc,
#         'Precision': prec,
#         'Recall': rec,
#         'F1-Score': f1
#     })
#     model_names.append(name)
#     accuracy_scores.append(acc) 
#     print(f"Accuracy: {acc:.4f}")
#     print(f"Precision (macro): {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
#     print(classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=0))
# # Visualization – Model Comparison
# results_df = pd.DataFrame(results).set_index('Model')
# print("\n All model results:\n", results_df.sort_values(by='Accuracy', ascending=False))
# # # Plot accuracy comparison
# plt.figure(figsize=(10,5))
# sns.barplot(x=model_names, y=accuracy_scores, palette='viridis')
# plt.title('Model Accuracy Comparison')
# plt.xticks(rotation=45)
# plt.ylabel('Accuracy')
# plt.ylim(0,1)
# for i, v in enumerate(accuracy_scores):
#     plt.text(i, v+0.01, f"{v:.3f}", ha='center')
# plt.tight_layout()
# plt.show()
# # # Plot precision/recall/f1
# results_df[['Precision','Recall','F1-Score']].plot(kind='bar', figsize=(10,6))
# plt.title('Precision, Recall, F1-Score per Model (macro avg)')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()
# #Confusion Matrix
# best_model_name = results_df['F1-Score'].idxmax()
# best_model = models[best_model_name]
# y_pred_best = best_model.predict(X_test)
# cm = confusion_matrix(y_test, y_pred_best)
# plt.figure(figsize=(8,6))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
# plt.title(f'Confusion Matrix - {best_model_name}')
# plt.ylabel('Actual')
# plt.xlabel('Predicted')
# plt.tight_layout()
# plt.show()


import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

import warnings
warnings.filterwarnings('ignore')

# --- 1. NLTK Downloads (run once) ---
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --- 2. Load Data (fix file path warning) ---
df = pd.read_csv(r"D:\LAPTOP DATA\AI\model\Mental_Stress_Text.csv")

print("Initial shape:", df.shape)
print(df['status'].value_counts())

# --- 3. Data Cleaning ---
df = df.drop(columns=['Unnamed: 0'], errors='ignore')   # drop index column
df = df.dropna(subset=['statement', 'status'])
df = df[df['statement'].str.strip() != '']
df = df.drop_duplicates(subset=['statement', 'status'])

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return ' '.join(tokens)

df['clean_statement'] = df['statement'].apply(clean_text)
df = df[df['clean_statement'].str.strip() != '']

print("Cleaned shape:", df.shape)

# --- 4. Label Encoding ---
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['status'])
print("Classes:", label_encoder.classes_)

# --- 5. TF‑IDF Feature Extraction ---
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X = vectorizer.fit_transform(df['clean_statement'])
y = df['label']

print("X shape:", X.shape)

# --- 6. Train/Test Split (sparse for most models) ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- 7. Dimensionality Reduction for KNN ---
svd = TruncatedSVD(n_components=200, random_state=42)
X_svd = svd.fit_transform(X)                # dense, shape (n, 200)
X_train_svd, X_test_svd, _, _ = train_test_split(
    X_svd, y, test_size=0.2, random_state=42, stratify=y
)
# --- 8. Define Models ---
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced'),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
    'Decision Tree': DecisionTreeClassifier(class_weight='balanced'),
    'Random Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced', n_jobs=-1),
    'Linear SVM': LinearSVC(class_weight='balanced', max_iter=2000),
    'Naive Bayes': MultinomialNB()
}
# --- 9. Single Evaluation Loop ---
results = []
model_names = []
accuracy_scores = []
for name, model in models.items():
    print(f"\n--- {name} ---")
    if name == 'K-Nearest Neighbors':
        # Use the reduced (SVD) versions for KNN
        model.fit(X_train_svd, y_train)
        y_pred = model.predict(X_test_svd)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    results.append({
        'Model': name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1
    })
    model_names.append(name)
    accuracy_scores.append(acc)
    print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=0))
    # --- 10. Results Table ---
results_df = pd.DataFrame(results).set_index('Model')
print("\n All model results (sorted by Accuracy):")
print(results_df.sort_values(by='Accuracy', ascending=False))

# --- 11. Visualizations ---
# Accuracy bar chart
plt.figure(figsize=(10,5))
bars = plt.bar(model_names, accuracy_scores, color='skyblue')
plt.title('Model Accuracy Comparison')
plt.xticks(rotation=45)
plt.ylabel('Accuracy')
plt.ylim(0,1)
for bar, val in zip(bars, accuracy_scores):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.01, f'{val:.3f}', ha='center')
plt.tight_layout()
plt.show()

# # Precision/Recall/F1 bar chart
results_df[['Precision','Recall','F1-Score']].plot(kind='bar', figsize=(10,6))
plt.title('Precision, Recall, F1-Score per Model (macro avg)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# # Confusion matrix for best model (by F1)
best_model_name = results_df['F1-Score'].idxmax()
best_model = models[best_model_name]
if best_model_name == 'K-Nearest Neighbors':
    y_pred_best = best_model.predict(X_test_svd)
else:
    y_pred_best = best_model.predict(X_test)

cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title(f'Confusion Matrix - {best_model_name}')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.show()