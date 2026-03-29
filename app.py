from flask import Flask, request, jsonify
import requests
from bs4 import BeautifulSoup
from transformers import pipeline

app = Flask(__name__)
print("Loading the summary model... Please wait...")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
print("model loaded")

def scrape_article(url):
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers, timeout=10)
    soup = BeautifulSoup(response.text, 'html.parser')
    paragraphs = soup.find_all('p')
    text = ' '.join([p.get_text() for p in paragraphs])
    return text

@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.get_json()
    url = data.get('url', '')

    if not url:
        return jsonify({'error': 'No URL provided'}), 400

    try:
        article_text = scrape_article(url)
        trimmed = article_text[:1024]
        result = summarizer(trimmed, max_length=150, min_length=40, do_sample=False)
        summary = result[0]['summary_text']
        return jsonify({
            'url': url,
            'summary': summary
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
