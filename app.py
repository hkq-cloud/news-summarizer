from flask import Flask, request, jsonify
import requests
from bs4 import BeautifulSoup
from transformers import pipeline
import textstat
from textblob import TextBlob

app = Flask(__name__)
<<<<<<< HEAD

print("正在加载摘要模型，请稍等...")
=======
print("Loading the summary model... Please wait...")
>>>>>>> 3976faaa64916f83f7b0a8bae0264320496aaa7a
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
print("model loaded")

def scrape_article(url):
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers, timeout=10)
    if response.status_code != 200:
        raise Exception(f"无法访问网站，状态码：{response.status_code}")
    soup = BeautifulSoup(response.text, 'html.parser')
    paragraphs = soup.find_all('p')
    text = ' '.join([p.get_text() for p in paragraphs])
    if len(text) < 100:
        raise Exception("文章内容太短或无法提取")
    return text

@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.get_json()
    url = data.get('url', '').strip()

    if not url:
        return jsonify({'error': 'No URL provided'}), 400
    if not url.startswith('http'):
        return jsonify({'error': 'Invalid URL format'}), 400

    try:
        article_text = scrape_article(url)
        trimmed = article_text[:1024]
        result = summarizer(trimmed, max_length=150, min_length=40, do_sample=False)
        summary = result[0]['summary_text']

        # 评估指标
        grade_level = round(textstat.flesch_kincaid_grade(summary), 1)
        reading_ease = round(textstat.flesch_reading_ease(summary), 1)
        original_sentiment = round(TextBlob(trimmed).sentiment.polarity, 2)
        summary_sentiment = round(TextBlob(summary).sentiment.polarity, 2)

        return jsonify({
            'url': url,
            'summary': summary,
            'grade_level': grade_level,
            'reading_ease': reading_ease,
            'original_sentiment': original_sentiment,
            'summary_sentiment': summary_sentiment
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
