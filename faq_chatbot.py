# FAQ Chatbot using NLP and Cosine Similarity
# --------------------------------------
# This is a simple FAQ chatbot that:
# 1. Stores FAQs
# 2. Preprocesses text
# 3. Matches user questions using cosine similarity
# 4. Returns the best matching answer

import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download required NLTK data (run once)
nltk.download('punkt')
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# -----------------------------
# Step 1: Define FAQs
# -----------------------------
faqs = {
    "What is your return policy?": "You can return products within 30 days of purchase.",
    "How can I track my order?": "You can track your order using the tracking link sent to your email.",
    "Do you offer international shipping?": "Yes, we ship internationally with additional charges.",
    "How can I contact customer support?": "You can contact customer support via email or phone.",
    "What payment methods are accepted?": "We accept credit cards, debit cards, UPI, and net banking."
}

faq_questions = list(faqs.keys())

# -----------------------------
# Step 2: Text Preprocessing
# -----------------------------
def preprocess(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in string.punctuation]
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return " ".join(tokens)

processed_faqs = [preprocess(q) for q in faq_questions]

# -----------------------------
# Step 3: Vectorization
# -----------------------------
vectorizer = TfidfVectorizer()
faq_vectors = vectorizer.fit_transform(processed_faqs)

# -----------------------------
# Step 4: Chatbot Function
# -----------------------------
def chatbot(user_input):
    processed_input = preprocess(user_input)
    user_vector = vectorizer.transform([processed_input])
    similarities = cosine_similarity(user_vector, faq_vectors)
    best_match_index = similarities.argmax()

    if similarities[0][best_match_index] < 0.2:
        return "Sorry, I couldn't understand your question. Please try again."

    return faqs[faq_questions[best_match_index]]

# -----------------------------
# Step 5: Simple Chat Interface
# -----------------------------
print("FAQ Chatbot (type 'exit' to quit)")
while True:
    user_question = input("You: ")
    if user_question.lower() == 'exit':
        print("Bot: Thank you! Have a great day.")
        break
    response = chatbot(user_question)
    print("Bot:", response)