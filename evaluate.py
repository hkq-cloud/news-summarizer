"""
evaluate.py - Automated evaluation script for The Daily Brief
- BART, PEGASUS, T5: run once per article (no prompt variants)
- GPT-3.5: run 4 times per article (one per prompt variant)
Results saved to results.csv
"""

import requests
import csv
import time

# ============================================================
# CONFIG
# ============================================================
API_BASE = "http://127.0.0.1:5000"  # local
# API_BASE = "https://qq770119-news-summarizer.hf.space"  # HuggingFace

PROMPT_VARIANTS = ["v1_baseline", "v2_audience", "v3_cot", "v4_constraint"]

# ============================================================
# ADD YOUR ARTICLE URLs HERE (aim for 60 total)
# Tech + Sports articles
# ============================================================
ARTICLE_URLS = [
    # Already tested (keep these)
    "https://www.bbc.com/news/articles/cq6q66n86qyo",
    "https://www.cnn.com/2026/04/17/tech/anti-ai-attack-sam-altman",
    "https://www.cnbc.com/2026/04/17/netflix-mergers-m-a-strategy.html",
    "https://news.un.org/en/story/2026/04/1167318",

    # ADD MORE URLs BELOW
    # Tech:
    # "https://...",

    # Sports:
    # "https://...",
]

# ============================================================
# EVALUATION FUNCTIONS
# ============================================================

def call_api(url, prompt_variant="v1_baseline"):
    try:
        response = requests.post(
            f"{API_BASE}/summarize",
            json={"url": url, "preset": "general", "prompt_variant": prompt_variant},
            timeout=120
        )
        return response.json()
    except Exception as e:
        return {"url": url, "error": str(e)}

def extract_metrics(data, model_key):
    if model_key not in data:
        return None
    m = data[model_key]
    return {
        "rouge1": m.get("rouge", {}).get("rouge1", 0),
        "rouge2": m.get("rouge", {}).get("rouge2", 0),
        "rougeL": m.get("rouge", {}).get("rougeL", 0),
        "grade_level": m.get("grade_level", 0),
        "reading_ease": m.get("reading_ease", 0),
        "original_sentiment": m.get("original_sentiment", 0),
        "summary_sentiment": m.get("summary_sentiment", 0),
        "compression_ratio": m.get("compression_ratio", 0),
        "word_count_summary": m.get("word_count_summary", 0),
        "summary": m.get("summary", ""),
    }

def run_evaluation():
    print(f"Starting evaluation...")
    print(f"Articles: {len(ARTICLE_URLS)}")
    print(f"BART/PEGASUS/T5: run once per article")
    print(f"GPT-3.5: run {len(PROMPT_VARIANTS)} times per article")
    total_calls = len(ARTICLE_URLS) * (1 + len(PROMPT_VARIANTS))
    print(f"Total API calls: {total_calls}")
    print("-" * 60)

    rows = []
    count = 0

    for url in ARTICLE_URLS:
        print(f"\nArticle: {url[:70]}...")

        # Run BART, PEGASUS, T5 once
        count += 1
        print(f"  [{count}] Running BART/PEGASUS/T5...")
        result = call_api(url, "v1_baseline")

        if "error" in result and "bart" not in result:
            print(f"  ERROR: {result['error']}")
            rows.append({"url": url, "model": "ERROR", "prompt_variant": "N/A", "error": result.get("error", "")})
        else:
            for model in ["bart", "pegasus", "t5"]:
                metrics = extract_metrics(result, model)
                if metrics:
                    rows.append({
                        "url": url,
                        "model": model.upper(),
                        "prompt_variant": "N/A",
                        **metrics
                    })
                    print(f"    {model.upper()}: ROUGE-1={metrics['rouge1']}, Grade={metrics['grade_level']}, Ease={metrics['reading_ease']}")

        time.sleep(2)

        # Run GPT 4 times
        for prompt in PROMPT_VARIANTS:
            count += 1
            print(f"  [{count}] GPT | {prompt}...")
            result = call_api(url, prompt)

            if "error" in result and "gpt" not in result:
                print(f"  ERROR: {result['error']}")
                rows.append({"url": url, "model": "GPT", "prompt_variant": prompt, "error": result.get("error", "")})
            else:
                metrics = extract_metrics(result, "gpt")
                if metrics:
                    rows.append({
                        "url": url,
                        "model": "GPT",
                        "prompt_variant": prompt,
                        **metrics
                    })
                    print(f"    GPT [{prompt}]: ROUGE-1={metrics['rouge1']}, Grade={metrics['grade_level']}, Ease={metrics['reading_ease']}")

            time.sleep(2)

    # Save to CSV
    fieldnames = ["url", "model", "prompt_variant", "rouge1", "rouge2", "rougeL",
                  "grade_level", "reading_ease", "original_sentiment", "summary_sentiment",
                  "compression_ratio", "word_count_summary", "summary", "error"]

    with open("results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 60)
    print(f"Done! Results saved to results.csv")
    print(f"Total rows: {len(rows)}")

    # Summary stats
    print("\n=== AVERAGE METRICS BY MODEL ===")
    for model in ["BART", "PEGASUS", "T5"]:
        model_rows = [r for r in rows if r.get("model") == model and "error" not in r]
        if model_rows:
            avg_r1 = sum(r.get("rouge1", 0) for r in model_rows) / len(model_rows)
            avg_grade = sum(r.get("grade_level", 0) for r in model_rows) / len(model_rows)
            avg_ease = sum(r.get("reading_ease", 0) for r in model_rows) / len(model_rows)
            print(f"{model}: ROUGE-1={avg_r1:.3f}, Grade={avg_grade:.1f}, Ease={avg_ease:.1f}")

    print("\n=== GPT BY PROMPT VARIANT ===")
    for prompt in PROMPT_VARIANTS:
        gpt_rows = [r for r in rows if r.get("model") == "GPT" and r.get("prompt_variant") == prompt and "error" not in r]
        if gpt_rows:
            avg_r1 = sum(r.get("rouge1", 0) for r in gpt_rows) / len(gpt_rows)
            avg_grade = sum(r.get("grade_level", 0) for r in gpt_rows) / len(gpt_rows)
            avg_ease = sum(r.get("reading_ease", 0) for r in gpt_rows) / len(gpt_rows)
            print(f"GPT [{prompt}]: ROUGE-1={avg_r1:.3f}, Grade={avg_grade:.1f}, Ease={avg_ease:.1f}")

if __name__ == "__main__":
    run_evaluation()
