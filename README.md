---
title: News Summarizer
emoji: 📰
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# The Daily Brief - AI-Powered News Summarizer

A web application that scrapes news articles from user-provided URLs and summarizes them using four language models, with side-by-side comparison and automatic evaluation metrics.

**Live Demo:** https://huggingface.co/spaces/qq770119/news-summarizer

## Features

- Paste any news article URL and get summaries from 4 models simultaneously
- Compare BART, GPT-3.5-turbo, PEGASUS, and T5 side by side
- Choose reading level preset: General, Professional, or Detailed
- Select from 4 prompt variants (baseline, audience-adapted, chain-of-thought, constraint)
- View ROUGE scores, readability grade level, sentiment, and compression ratio for each summary
- Extract keywords from article content

## Models

| Model | Type | Notes |
|-------|------|-------|
| BART-large-CNN | Encoder-decoder | Fine-tuned on CNN/DailyMail |
| GPT-3.5-turbo | Generative (API) | OpenAI API |
| PEGASUS-xsum | Encoder-decoder | Fine-tuned on XSum |
| T5-small | Encoder-decoder | Lightweight local model |

## Setup (Local)

1. Clone the repo
   ```bash
   git clone https://github.com/hkq-cloud/news-summarizer.git
   cd news-summarizer
   ```

2. Create a virtual environment and install dependencies
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Create a `.env` file with your OpenAI API key
   ```
   OPENAI_API_KEY=your_key_here
   ```

4. Run the app
   ```bash
   python app.py
   ```

5. Open `http://localhost:5000` in your browser

## Evaluation

We evaluated all four models across 119 news articles (technology, sports, politics, business) using:
- **Automatic:** ROUGE-1, ROUGE-2, ROUGE-L against human-written reference summaries
- **Human:** Accuracy, Fluency, and Conciseness scored 1–5 across 96 sampled summaries

To reproduce the evaluation:
```bash
python evaluate.py          # Weiqi's dataset (tech + sports)
python evaluate_teammate.py # Teammate's dataset (politics + business)
```

## Dataset

| File | Description |
|------|-------------|
| `dataset.csv` | 60 articles (technology + sports) with reference summaries |
| `dataset_teammate.csv` | 59 usable articles (politics + business) with reference summaries |
| `results.csv` | Evaluation results for tech + sports |
| `results_teammate.csv` | Evaluation results for politics + business |
| `results_combined.csv` | Combined results (1,808 valid rows) |
| `human_eval_weiqi.csv` | Human evaluation scores (Weiqi Li, 48 articles) |

## Project Structure

```
news-summarizer/
├── app.py                  # Flask backend with 4 models + evaluation
├── evaluate.py             # Automated evaluation script
├── evaluate_teammate.py    # Evaluation for teammate's dataset
├── index.html              # Frontend UI
├── requirements.txt        # Python dependencies
├── Dockerfile              # HuggingFace Spaces deployment
├── dataset.csv             # Weiqi's article dataset
├── dataset_teammate.csv    # Teammate's article dataset
├── results*.csv            # Evaluation results
└── human_eval_weiqi.csv    # Human evaluation scores
```

## Tech Stack

- **Backend:** Python, Flask, Hugging Face Transformers
- **Frontend:** HTML, CSS, JavaScript
- **Models:** BART, PEGASUS, T5 (local), GPT-3.5-turbo (OpenAI API)
- **Evaluation:** rouge-score, textstat, TextBlob, YAKE
- **Deployment:** HuggingFace Spaces (Docker)

## Team

**Weiqi Li & Kshitija Kumbharkar**
ISE 547 — Generative AI, May 2026
