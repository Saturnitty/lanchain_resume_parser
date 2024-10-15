from flask import Flask, render_template, request
import os
from docx import Document
import pdfplumber
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from datetime import datetime
import re
from textblob import TextBlob
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Set OpenAI API key as an environment variable
os.environ['OPENAI_API_KEY'] = ''

app = Flask(__name__)

# Function to extract text from PDF
def extract_text_from_pdf(file_path):
    with pdfplumber.open(file_path) as pdf:
        text = ''.join([page.extract_text() for page in pdf.pages])
    return text

# Function to extract text from DOCX
def extract_text_from_docx(file_path):
    doc = Document(file_path)
    text = ''.join([paragraph.text for paragraph in doc.paragraphs])
    return text

# Function to preprocess text
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return ' '.join(tokens)

# Function to vectorize texts using LangChain with OpenAIEmbeddings
def vectorize_texts(texts):
    embedding_model = OpenAIEmbeddings(openai_api_key=os.getenv('OPENAI_API_KEY'))
    text_splitter = RecursiveCharacterTextSplitter()
    vectors = [embedding_model.embed(text_splitter.split_text(text)) for text in texts]
    return vectors, embedding_model

# Function to compute cosine similarity
def compute_cosine_similarity(vectors):
    similarity_matrix = cosine_similarity(vectors)
    return similarity_matrix

# Function to load resumes from directory
def load_resumes_from_directory(directory_path):
    resumes = []
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if filename.endswith('.pdf'):
            text = extract_text_from_pdf(file_path)
        elif filename.endswith('.docx'):
            text = extract_text_from_docx(file_path)
        else:
            continue
        resumes.append((filename, text))
    return resumes

# Function to extract years of experience
def extract_years_of_experience(text):
    current_year = datetime.now().year
    years = re.findall(r'\b(19|20)\d{2}\b', text)
    years = [int(year) for year in years if int(year) <= current_year]
    if not years:
        return 0
    experience_years = max(years) - min(years)
    return experience_years

# Function to extract skills
def extract_skills(text):
    # Placeholder for skill extraction logic
    return set()

# Function to rank resumes by experience
def rank_resumes_by_experience(resumes):
    experience_years = [extract_years_of_experience(text) for _, text in resumes]
    ranking_df = pd.DataFrame({
        'Resume Name': [name for name, _ in resumes],
        'Experience Years': experience_years
    })
    ranking_df['Experience Rank'] = ranking_df['Experience Years'].rank(ascending=False, method='min').astype(int)
    return ranking_df.sort_values(by='Experience Rank')

# Function to rank resumes by skills
def rank_resumes_by_skills(job_description, resumes):
    job_skills = extract_skills(preprocess_text(job_description))
    resume_skills = [extract_skills(preprocess_text(text)) for _, text in resumes]
    skill_match_scores = [len(job_skills.intersection(skills)) for skills in resume_skills]
    ranking_df = pd.DataFrame({
        'Resume Name': [name for name, _ in resumes],
        'Skill Match Score': skill_match_scores
    })
    ranking_df['Skill Rank'] = ranking_df['Skill Match Score'].rank(ascending=False, method='min').astype(int)
    return ranking_df.sort_values(by='Skill Rank')

# Function to analyze sentiment
def analyze_sentiment(text):
    sentiment = TextBlob(text).sentiment
    return sentiment.polarity

# Function to rank resumes
def rank_resumes(job_description, resumes_directory):
    resumes = load_resumes_from_directory(resumes_directory)
    texts = [job_description] + [text for _, text in resumes]
    preprocessed_texts = [preprocess_text(text) for text in texts]
    vectors, _ = vectorize_texts(preprocessed_texts)
    similarity_matrix = compute_cosine_similarity(vectors)
    similarity_scores = similarity_matrix[0, 1:]
    ranking_df = pd.DataFrame({
        'Resume Name': [name for name, _ in resumes],
        'Similarity Score': similarity_scores,
        'Sentiment Score': [analyze_sentiment(text) for text in preprocessed_texts[1:]]
    })
    ranking_df['Similarity Rank'] = ranking_df['Similarity Score'].rank(ascending=False, method='min').astype(int)
    ranking_df['Sentiment Rank'] = ranking_df['Sentiment Score'].rank(ascending=False, method='min').astype(int)
    experience_ranking = rank_resumes_by_experience(resumes)
    skill_ranking = rank_resumes_by_skills(job_description, resumes)
    combined_df = pd.merge(ranking_df, experience_ranking, on='Resume Name')
    combined_df = pd.merge(combined_df, skill_ranking, on='Resume Name')
    combined_df['Final Score'] = combined_df['Similarity Rank'] + combined_df['Experience Rank'] + combined_df['Skill Rank'] + combined_df['Sentiment Rank']
    combined_df['Final Rank'] = combined_df['Final Score'].rank(ascending=True, method='min').astype(int)
    return combined_df, ranking_df, experience_ranking, skill_ranking

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        job_description = request.form['job_description']
        resumes_directory = "C:\\Users\\rsank\\OneDrive\\Desktop\\2024internship\\langchain\\resumes"
        combined_ranking, similarity_ranking, experience_ranking, skill_ranking = rank_resumes(job_description, resumes_directory)
        combined_ranking.rename(columns={'Final Rank': 'Overall Rank'}, inplace=True)
        return render_template('results.html', overall_ranking=combined_ranking.to_html(), similarity_ranking=similarity_ranking.to_html(), experience_ranking=experience_ranking.to_html(), skill_ranking=skill_ranking.to_html())
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
