from flask import Flask, render_template, request
import pickle
import os
from nltk.tokenize import word_tokenize
from gensim import corpora
from transformers import pipeline
import torch

#summary libraries
from nltk.cluster.util import cosine_distance
import networkx as nx
from nltk.tokenize import sent_tokenize
import nltk
import numpy as np
import re


from nltk.corpus import stopwords, words
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

nltk.download('stopwords')
nltk.download('words')
all_stopwords = stopwords.words('english')
all_stopwords.remove('not')

valid_words = set(words.words())

# Initialize Flask app
app = Flask(__name__)

# Load the trained XGBoost model, CountVectorizer, and LabelEncoder from the 'models' folder
with open(os.path.join('models', 'xgb_model.pickle'), 'rb') as model_file:
    xgb_model = pickle.load(model_file)

with open(os.path.join('models', 'vectorizer.pickle'), 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

with open(os.path.join('models', 'label_encoder.pickle'), 'rb') as label_encoder_file:
    label_encoder = pickle.load(label_encoder_file)

# Load the pre-trained LDA model and dictionary from the 'models' folder
with open(os.path.join('models', 'lda_model.pickle'), 'rb') as lda_model_file:
    lda_model = pickle.load(lda_model_file)

with open(os.path.join('models', 'dictionary.pickle'), 'rb') as dictionary_file:
    id2word = pickle.load(dictionary_file)

# Topic Modeling Function using the loaded model
def perform_topic_modeling(text, num_topics=5):
    # Tokenize the input text without any lemmatization
    tokenized_text = word_tokenize(text)

    # Convert text to BoW format using the loaded dictionary
    bow = id2word.doc2bow(tokenized_text)

    # Get topics using the loaded LDA model
    topics = lda_model.get_document_topics(bow)
    
    # Example: Custom topic names based on the words in the topics
    topic_names = {
        0: "Business",
        1: "Entertaintment",
        2: "Sports",
        3: "Politics",
        4: "Technology",
       
    }

    # Extract readable topics with custom names
    formatted_topics = []
    for idx, topic_weight in topics:
        topic_words = [word for word, _ in lda_model.show_topic(idx, topn=5)]
        # Use custom topic names
        topic_name = topic_names.get(idx, f"Topic {idx+1}")
        formatted_topics.append(f"{topic_name}: {', '.join(topic_words)}")

    return formatted_topics


# Function to predict sentiment based on user input
def predict_sentiment(text):
    # Vectorize the input text using the trained CountVectorizer
    vectorized_text = vectorizer.transform([text]).toarray()
    
    # Predict sentiment using the loaded XGBoost model
    prediction = xgb_model.predict(vectorized_text)
    
    # Convert the prediction back to the sentiment label using LabelEncoder
    prediction_label = label_encoder.inverse_transform([int(prediction[0])])
    return prediction_label[0]

#Summary
def generate_summary(text):
    nltk.download('punkt')
    
    stop_words = stopwords.words('english')
    summarize_text = []
    
    # read text and tokenize
    sentences = read_input(text)
    
    # generate similarity matrix across sentences
    sentence_similarity_matrix = build_similarity_matrix(sentences,stop_words)
    
    # Rank sentences in similarirty matrix
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_matrix)
    scores = nx.pagerank(sentence_similarity_graph)
    
    # sort the rank and place top sentences
    ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)),reverse=True)
    
    senctence_count = len(ranked_sentences)
    count = int(senctence_count*0.3)

    if count == 0:
        top_n =1
    else:
        top_n = count

    # Get the top n number of sentences based on rank
    for i in range(min(top_n, len(ranked_sentences))):
        clean_sentence = clean_text(ranked_sentences[i][1])  # Clean the sentence
        summarize_text.append(clean_sentence + ".") 

    result = ' '.join(summarize_text)
    return result
        

    

def read_input(text):
    sentences = sent_tokenize(text)  # Tokenize the text into sentences
    cleaned_sentences = []
    for sentence in sentences:
        clean_sentence = clean_text(sentence)  # Clean each sentence
        cleaned_sentences.append(clean_sentence)
    return cleaned_sentences


def clean_text(text):
    
    # Remove special characters only
    text = re.sub(r"[^a-zA-Z0-9\s\-]", "", text)

    # Remove gibberish words not in valid words dictionary
    words_in_text = word_tokenize(text)
    filtered_words = [word for word in words_in_text if word.lower() in valid_words or word.lower() not in stopwords.words('english')]

    # Join the filtered words back into a sentence
    return ' '.join(filtered_words).strip()

def sentence_similarity(sent1,sent2,stopwords = None):
    if stopwords is None:
        stopwords = []
    
    # Tokenize sentences into words and convert to lowercase
    sent1 = [w.lower() for w in word_tokenize(sent1) if w.lower() not in stopwords]
    sent2 = [w.lower() for w in word_tokenize(sent2) if w.lower() not in stopwords]
    
    all_words = list(set(sent1+sent2))
    vector1 = [0]*len(all_words)
    vector2 = [0]*len(all_words)
    #build vector for 1st sentence
    for w in sent1:
        if not w in stopwords:
            vector1[all_words.index(w)]+=1
    for w in sent2:
        if not w in stopwords:
            vector2[all_words.index(w)]+=1
    return 1-cosine_distance(vector1,vector2)

def build_similarity_matrix(sentences,stop_words):
    #create an empty similarity matrix
    similarity_matrix = np.zeros((len(sentences),len(sentences)))
    
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1!=idx2:
                similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1],sentences[idx2],stop_words)
                
    return similarity_matrix

# Define Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if request.method == 'POST':
        news_text = request.form['news_text']  # Get the input text from the form
        
        # Perform sentiment analysis
        sentiment_result = predict_sentiment(news_text)
        
        # Perform text summarization
        summary_text = generate_summary(news_text)

        # Perform topic modeling using the pre-trained LDA model
        topics = perform_topic_modeling(news_text, num_topics=5)
        
        # Pass results back to the template to display sentiment, summary, and topics
        return render_template('index.html', sentiment=sentiment_result, summary=summary_text, topics=topics, news_text=news_text)

if __name__ == '__main__':
    app.run(debug=True)

