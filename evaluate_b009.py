"""
evaluate.py - Automated evaluation script for The Daily Brief
Tests 4 LLMs x 4 prompt variants across articles in a CSV dataset.
Computes ROUGE against reference summaries.

USAGE:
    1. Fill in dataset.csv with columns: id, category, url, reference_summary
    2. Make sure the Flask app is running locally (python app.py)
    3. Run: python evaluate.py

OUTPUT:
    results.csv — one row per (article x model x prompt)
"""

import requests
import csv
import time
import sys
from pathlib import Path
from rouge_score import rouge_scorer

API_BASE = "http://127.0.0.1:5000"
DATASET_FILE = "dataset_teammate.csv"
RESULTS_FILE = "results_teammate.csv"
PROMPT_VARIANTS = ["v1_baseline", "v2_audience", "v3_cot", "v4_constraint"]
MODELS = ["bart", "gpt", "pegasus", "t5"]
SKIP_REDUNDANT_LOCAL_MODELS = True
_rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

def load_dataset(path):
    if not Path(path).exists():
        sys.exit(f"ERROR: {path} not found.")
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            url = (r.get("url") or "").strip()
            ref = (r.get("reference_summary") or "").strip()
            if not url or not ref:
                continue
            rows.append({
                "id": (r.get("id") or "").strip(),
                "category": (r.get("category") or "").strip(),
                "url": url,
                "reference_summary": ref,
            })
    return rows

def call_api(url, prompt_variant, preset="general"):
    try:
        r = requests.post(
            f"{API_BASE}/summarize",
            json={"url": url, "preset": preset, "prompt_variant": prompt_variant},
            timeout=180,
        )
        return r.json()
    except Exception as e:
        return {"error": str(e)}

def rouge_against_reference(reference, summary):
    if not summary or not reference:
        return {"rouge1_ref": None, "rouge2_ref": None, "rougeL_ref": None}
    s = _rouge.score(reference, summary)
    return {
        "rouge1_ref": round(s["rouge1"].fmeasure, 4),
        "rouge2_ref": round(s["rouge2"].fmeasure, 4),
        "rougeL_ref": round(s["rougeL"].fmeasure, 4),
    }

def extract_row(article, model_key, prompt_variant, api_result):
    if model_key not in api_result:
        return None
    m = api_result[model_key]
    summary = m.get("summary", "") or ""
    if summary.startswith(("PEGASUS error", "T5 error")):
        return {
            "id": article["id"], "category": article["category"], "url": article["url"],
            "model": model_key.upper(), "prompt_variant": prompt_variant,
            "summary": summary, "error": summary,
        }
    ref_rouge = rouge_against_reference(article["reference_summary"], summary)
    src_rouge = m.get("rouge", {})
    sd_orig = m.get("original_sentiment") or 0
    sd_summ = m.get("summary_sentiment") or 0
    return {
        "id": article["id"],
        "category": article["category"],
        "url": article["url"],
        "model": model_key.upper(),
        "prompt_variant": prompt_variant,
        "summary": summary,
        "rouge1_ref": ref_rouge["rouge1_ref"],
        "rouge2_ref": ref_rouge["rouge2_ref"],
        "rougeL_ref": ref_rouge["rougeL_ref"],
        "rouge1_src": src_rouge.get("rouge1"),
        "rouge2_src": src_rouge.get("rouge2"),
        "rougeL_src": src_rouge.get("rougeL"),
        "grade_level": m.get("grade_level"),
        "reading_ease": m.get("reading_ease"),
        "original_sentiment": sd_orig,
        "summary_sentiment": sd_summ,
        "sentiment_drift": round(abs(sd_summ - sd_orig), 3),
        "compression_ratio": m.get("compression_ratio"),
        "word_count_summary": m.get("word_count_summary"),
        "word_count_original": m.get("word_count_original"),
    }

def main():
    articles = [a for a in load_dataset(DATASET_FILE) if a['id'] == 'b009']
    if not articles:
        sys.exit(f"No usable rows in {DATASET_FILE}.")
    print("=" * 60)
    print(f"Articles loaded:    {len(articles)}")
    print(f"Prompt variants:    {len(PROMPT_VARIANTS)}")
    print(f"Models:             {len(MODELS)}")
    total_rows = len(articles) * len(MODELS) * len(PROMPT_VARIANTS)
    print(f"Estimated CSV rows: {total_rows}")
    print("=" * 60)
    rows = []
    count = 0
    total = len(articles) * len(PROMPT_VARIANTS)
    for art in articles:
        cached_local = {}
        for variant in PROMPT_VARIANTS:
            count += 1
            print(f"[{count}/{total}] {art['id']} | {variant} | {art['url'][:60]}...")
            api_result = call_api(art["url"], variant)
            if "error" in api_result and "bart" not in api_result:
                print(f"   x ERROR: {api_result['error']}")
                rows.append({
                    "id": art["id"], "category": art["category"], "url": art["url"],
                    "model": "ALL", "prompt_variant": variant,
                    "error": api_result["error"],
                })
                continue
            for model_key in MODELS:
                if SKIP_REDUNDANT_LOCAL_MODELS and model_key != "gpt":
                    if model_key in cached_local:
                        cached = dict(cached_local[model_key])
                        cached["prompt_variant"] = variant
                        rows.append(cached)
                        continue
                row = extract_row(art, model_key, variant, api_result)
                if row:
                    rows.append(row)
                    if SKIP_REDUNDANT_LOCAL_MODELS and model_key != "gpt":
                        cached_local[model_key] = row
                    if "error" not in row:
                        r1 = row.get("rouge1_ref")
                        gl = row.get("grade_level")
                        print(f"   {model_key.upper():8s}  R1_ref={r1}  Grade={gl}")
            time.sleep(2)

    fieldnames = [
        "id", "category", "url", "model", "prompt_variant", "summary",
        "rouge1_ref", "rouge2_ref", "rougeL_ref",
        "rouge1_src", "rouge2_src", "rougeL_src",
        "grade_level", "reading_ease",
        "original_sentiment", "summary_sentiment", "sentiment_drift",
        "compression_ratio", "word_count_summary", "word_count_original",
        "error",
    ]
    with open(RESULTS_FILE, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writerows(rows)
    print()
    print("=" * 60)
    print(f"Saved {len(rows)} rows -> {RESULTS_FILE}")
    print("=" * 60)
    print("\n=== Average ROUGE-1 vs reference, by model ===")
    for model in ["BART", "GPT", "PEGASUS", "T5"]:
        model_rows = [r for r in rows if r.get("model") == model and r.get("rouge1_ref") is not None]
        if model_rows:
            avg = sum(r["rouge1_ref"] for r in model_rows) / len(model_rows)
            print(f"  {model}: {avg:.4f} (n={len(model_rows)})")
    print("\n=== GPT by prompt variant ===")
    for variant in PROMPT_VARIANTS:
        gpt_rows = [r for r in rows if r.get("model") == "GPT"
                    and r.get("prompt_variant") == variant
                    and r.get("rouge1_ref") is not None]
        if gpt_rows:
            avg = sum(r["rouge1_ref"] for r in gpt_rows) / len(gpt_rows)
            print(f"  {variant}: {avg:.4f} (n={len(gpt_rows)})")

if __name__ == "__main__":
    main()
