from flask import Flask, request, jsonify
import requests
from bs4 import BeautifulSoup
from transformers import pipeline
import textstat
from textblob import TextBlob

app = Flask(__name__)

print("Loading summarization model, please wait...")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
print("Model loaded successfully!")

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

def process_url(url):
    url = url.strip()
    if not url.startswith('http'):
        return {'url': url, 'error': 'Invalid URL format'}
    try:
        article_text = scrape_article(url)
        trimmed = article_text[:1024]
        result = summarizer(trimmed, max_length=150, min_length=40, do_sample=False)
        summary = result[0]['summary_text']
        return {
            'url': url,
            'summary': summary,
            'grade_level': round(textstat.flesch_kincaid_grade(summary), 1),
            'reading_ease': round(textstat.flesch_reading_ease(summary), 1),
            'original_sentiment': round(TextBlob(trimmed).sentiment.polarity, 2),
            'summary_sentiment': round(TextBlob(summary).sentiment.polarity, 2)
        }
    except Exception as e:
        return {'url': url, 'error': str(e)}

@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.get_json()

    if 'urls' in data:
        urls = data['urls']
        results = [process_url(url) for url in urls]
        return jsonify({'results': results})
    elif 'url' in data:
        return jsonify(process_url(data['url']))
    else:
        return jsonify({'error': 'No URL provided'}), 400

if __name__ == '__main__':
    app.run(debug=True)