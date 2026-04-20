from flask import Flask, request, jsonify
from flask_cors import CORS
from bs4 import BeautifulSoup
from transformers import pipeline
import textstat
from textblob import TextBlob
from openai import OpenAI
from dotenv import load_dotenv
from rouge_score import rouge_scorer
import yake
import requests
import os

load_dotenv()
app = Flask(__name__)
CORS(app)

print("Loading summarization model, please wait...")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
print("Model loaded successfully!")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

PRESET_SETTINGS = {
    "general":      {"max_length": 60,  "min_length": 20, "level": "elementary"},
    "professional": {"max_length": 130, "min_length": 40, "level": "college"},
    "detailed":     {"max_length": 200, "min_length": 80, "level": "high school"},
}

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

def get_openai_summary(text, level):
    level_descriptions = {
        "elementary": "elementary school (simple words, short sentences)",
        "high school": "high school level",
        "college": "college level",
        "expert": "expert level with technical language",
    }
    level_desc = level_descriptions.get(level, "high school level")
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": f"You are a news summarizer. Summarize the article in 2-3 sentences at a {level_desc} reading level."},
            {"role": "user", "content": text[:3000]}
        ]
    )
    return response.choices[0].message.content

def get_rouge_scores(original, summary):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(original, summary)
    return {
        'rouge1': round(scores['rouge1'].fmeasure, 3),
        'rouge2': round(scores['rouge2'].fmeasure, 3),
        'rougeL': round(scores['rougeL'].fmeasure, 3),
    }

def get_keywords(text):
    kw_extractor = yake.KeywordExtractor(lan="en", n=2, top=5)
    keywords = kw_extractor.extract_keywords(text)
    return [kw[0] for kw in keywords]

def evaluate(original, summary):
    word_count_original = len(original.split())
    word_count_summary = len(summary.split())
    compression = round((1 - word_count_summary / word_count_original) * 100, 1)
    return {
        'summary': summary,
        'grade_level': round(textstat.flesch_kincaid_grade(summary), 1),
        'reading_ease': round(textstat.flesch_reading_ease(summary), 1),
        'original_sentiment': round(TextBlob(original).sentiment.polarity, 2),
        'summary_sentiment': round(TextBlob(summary).sentiment.polarity, 2),
        'rouge': get_rouge_scores(original, summary),
        'word_count_original': word_count_original,
        'word_count_summary': word_count_summary,
        'compression_ratio': compression,
    }

def process_url(url, preset="general"):
    url = url.strip()
    if not url.startswith('http'):
        return {'url': url, 'error': 'Invalid URL format'}
    try:
        article_text = scrape_article(url)
        trimmed = article_text[:1024]

        settings = PRESET_SETTINGS.get(preset, PRESET_SETTINGS["general"])
        bart_result = summarizer(trimmed, do_sample=False,
            max_length=settings["max_length"],
            min_length=settings["min_length"])
        bart_summary = bart_result[0]['summary_text']

        gpt_summary = get_openai_summary(article_text, settings["level"])
        keywords = get_keywords(trimmed)

        return {
            'url': url,
            'preset': preset,
            'keywords': keywords,
            'bart': evaluate(trimmed, bart_summary),
            'gpt': evaluate(trimmed, gpt_summary)
        }
    except Exception as e:
        return {'url': url, 'error': str(e)}

@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.get_json()
    preset = data.get('preset', 'general')

    if 'urls' in data:
        results = [process_url(url, preset) for url in data['urls']]
        return jsonify({'results': results})
    elif 'url' in data:
        return jsonify(process_url(data['url'], preset))
    else:
        return jsonify({'error': 'No URL provided'}), 400

if __name__ == '__main__':
    app.run(debug=True)