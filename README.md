# FAQ Chatbot using NLP and Cosine Similarity

## Internship
CodeAlpha Artificial Intelligence Internship

## Task
Task 2 – FAQ Chatbot

## Description
A simple FAQ chatbot that:
- Stores predefined FAQs
- Preprocesses text using NLP techniques
- Converts text to TF-IDF vectors
- Uses cosine similarity to find the best matching answer

## Technologies Used
- Python
- NLTK
- Scikit-learn
- TF-IDF Vectorization
- Cosine Similarity

## How It Works
1. User enters a question
2. Question is preprocessed (tokenization, stopword removal)
3. Cosine similarity is computed against stored FAQs
4. Best matching answer is returned

## How to Run

1. Install dependencies:
   pip install -r requirements.txt

2. Run the chatbot:
   python faq_chatbot.py

3. Type your question (type 'exit' to quit)

## Example
User: How can I track my order?
Bot: You can track your order using the tracking link sent to your email.