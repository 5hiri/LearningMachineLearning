import nltk
import re
import os
from nltk.corpus import stopwords
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


# nltk.download('punkt')
# nltk.download('stopwords')

os.chdir(os.path.dirname(__file__))

def preprocess_text(text):
    #convert to lowecase
    text = text.lower()
    #remove punctuation and numbers
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d', ' ', text)
    #Tokenize
    words = nltk.word_tokenize(text)
    #remove stop words
    words = [word for word in words if word not in set(stopwords.words('english'))]
    return words

#Example
# user_input = "Hello! How can I help you today?"
# print(preprocess_text(user_input))

# #Simple rule based responses
# def get_response(user_input):
#     # Preprocess the input
#     words = preprocess_text(user_input)
    
#     # Simple keyword matching
#     if'hello'in words:
#         return"Hi there! How can I assist you?"
#     elif'bye'in words:
#         return"Goodbye! Have a great day!"
#     elif'help'in words:
#         return"I'm here to help! What do you need assistance with?"
#     else:
#         return"Sorry, I don't understand that. Can you try rephrasing?"

# # Example
# user_input = "Hello! I need some help."
# print(get_response(user_input))

# #Enhancing the Chatbot
# def get_response2(user_input):
#     words = preprocess_text(user_input)
    
#     responses = {
#         'hello': ["Hi there!", "Hello!", "Hey! How can I help?"],
#         'bye': ["Goodbye!", "See you later!", "Take care!"],
#         'help': ["I'm here to help!", "How can I assist?", "What do you need?"]
#     }
    
#     if'hello'in words:
#         return random.choice(responses['hello'])
#     elif'bye'in words:
#         return random.choice(responses['bye'])
#     elif'help'in words:
#         return random.choice(responses['help'])
#     else:
#         return"Sorry, I don't understand that. Can you try rephrasing?"

# # Example
# user_input = "Hello"
# print(get_response2(user_input))

#Exploring Advanced NLP techniques
# Sample training data
training_data = pd.read_csv('Conversation.csv')
training_data2 = pd.read_csv('AI.csv')

training_data = pd.concat([training_data, training_data2], axis=1)
# # Preprocessing data
# training_data['question'] = [' '.join(preprocess_text(text)) for text in training_data['question']]
# print(training_data.head())

# Extracting features
tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=10000, min_df=2)
X_train = tfidf.fit_transform(training_data['question'])
y_train = training_data['answer']

# Inspect TF-IDF featuresprint("TF-IDF feature names:")
#print(tfidf.get_feature_names_out()[:10])  # Show some feature names# Train the model

# Training a Naive Bayes classifier
model = KNeighborsClassifier(n_neighbors=1000)
model.fit(X_train, y_train)

def classify(text):
    # Preprocess the input text
    text_tfidf = tfidf.transform([text])
    return model.predict(text_tfidf)[0]

# Example
print(classify("hi, how are you doing?"))