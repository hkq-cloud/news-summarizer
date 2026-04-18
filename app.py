from flask import Flask, request, jsonify
from flask_cors import CORS
from bs4 import BeautifulSoup
from transformers import pipeline
import textstat
from textblob import TextBlob
from openai import OpenAI
from dotenv import load_dotenv
import requests
import os

load_dotenv()
app = Flask(__name__)
CORS(app)

print("Loading summarization model, please wait...")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
print("Model loaded successfully!")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def scrape_article(url):
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers, timeout=10)
    if response.status_code != 200:
        raise Exception(f"Failed to access website, status code: {response.status_code}")
    soup = BeautifulSoup(response.text, 'html.parser')
    paragraphs = soup.find_all('p')
    text = ' '.join([p.get_text() for p in paragraphs])
    if len(text) < 100:
        raise Exception("Article content too short or could not be extracted")
    return text

def get_openai_summary(text):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a news summarizer. Summarize the article in 2-3 sentences."},
            {"role": "user", "content": text[:3000]}
        ]
    )
    return response.choices[0].message.content

def evaluate(text, summary):
    return {
        'summary': summary,
        'grade_level': round(textstat.flesch_kincaid_grade(summary), 1),
        'reading_ease': round(textstat.flesch_reading_ease(summary), 1),
        'original_sentiment': round(TextBlob(text).sentiment.polarity, 2),
        'summary_sentiment': round(TextBlob(summary).sentiment.polarity, 2)
    }

def process_url(url):
    url = url.strip()
    if not url.startswith('http'):
        return {'url': url, 'error': 'Invalid URL format'}
    try:
        article_text = scrape_article(url)
        trimmed = article_text[:1024]
        bart_result = summarizer(trimmed, max_length=150, min_length=40, do_sample=False)
        bart_summary = bart_result[0]['summary_text']
        gpt_summary = get_openai_summary(article_text)
        return {
            'url': url,
            'bart': evaluate(trimmed, bart_summary),
            'gpt': evaluate(trimmed, gpt_summary)
        }
    except Exception as e:
        return {'url': url, 'error': str(e)}

@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.get_json()
    if 'urls' in data:
        results = [process_url(url) for url in data['urls']]
        return jsonify({'results': results})
    elif 'url' in data:
        return jsonify(process_url(data['url']))
    else:
        return jsonify({'error': 'No URL provided'}), 400

if __name__ == '__main__':
    app.run(debug=True)