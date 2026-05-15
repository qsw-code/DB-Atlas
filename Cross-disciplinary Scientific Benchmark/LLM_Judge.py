
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

from openai import OpenAI



SYSTEM_PROMPT = """You are an expert in Information Retrieval and Recommender System Evaluation.
Your task is to score dataset recommendations for scientific research queries on TWO metrics.

OUTPUT RULES:
- Output ONLY valid JSON. No markdown, no explanation, no preamble, no code fences.
- All scores are integers between 0 and 100.
- Follow the exact schema provided. No extra fields."""

USER_PROMPT_TEMPLATE = """You will evaluate dataset recommendations for scientific research queries.

IMPORTANT:
This evaluation prioritizes scientific usefulness over strict keyword matching.
Datasets that are INDIRECTLY relevant, cross-domain, or provide complementary signals
(e.g., drivers, validation data, or auxiliary variables) should be considered relevant.

Do NOT overly penalize datasets that do not exactly match query keywords,
as long as they can contribute meaningfully to solving the research problem.

Think like a researcher designing a COMPLETE data workflow, not just retrieving exact matches.

*** CRITICAL RULE ON AUTHENTICITY ***
Check the "Recommendation Authenticity" flag for each dataset.
If a dataset is marked as "Invalid", you MUST immediately judge it as completely invalid and irrelevant.
Regardless of how relevant the dataset name appears to be, an "Invalid" dataset CANNOT be counted as useful and CANNOT contribute to the diversity score.

## Evaluation Scope
ONLY evaluate the TOP-10 recommended datasets.

## Scoring Criteria (each scored 0-100):

### 1. precision_score (Scientific Accuracy)
Evaluate how many of the TOP-10 recommended datasets are scientifically useful for the query.

Relevance includes:
- Direct relevance (explicitly matches query concepts)
- Indirect relevance (multi-hop or inferred connections)
- Complementary datasets
- ZERO TOLERANCE: Any dataset with "Recommendation Authenticity": "Invalid" is strictly NOT useful and counts as 0.

Scoring guide:
- 100: 10/10 datasets are useful
- 80-90: 8-9 datasets are useful
- 60-70: 6-7 datasets are useful
- 40-50: 4-5 datasets are useful
- 20-30: 2-3 datasets are useful
- 0-10: 0-1 dataset is useful

---

### 2. diversity_score (Scientific Diversity)
Evaluate diversity across THREE dimensions within TOP-10.
(NOTE: Only evaluate diversity among the VALID datasets. Datasets marked as "Invalid" contribute NOTHING to diversity.)

1) Source diversity
2) Data type diversity
3) Functional role diversity:
   - Core measurement
   - Driving variables
   - Validation / ground truth
   - Integration platforms

Scoring bands:
- 100: All 10 datasets are distinct across source, type, and role
- 80-90: Strong diversity, minor overlap
- 60-80: Moderate diversity, some redundancy
- 40-60: Limited diversity
- 20-40: Very limited diversity
- 0-20: Nearly homogeneous

Key rule:
Datasets from the same root source count as ONE source and LOW functional diversity if they serve the same role.

---

## Additional Outputs

For each query, also provide:

- "distinct_sources_count": integer (0-10)
  (Do NOT count sources from "Invalid" datasets)
- "relevant_count": integer
  Count datasets that are directly relevant, indirectly useful, or complementary. (MUST EXACTLY EXCLUDE all "Invalid" datasets)
- "top10_quality": one of ["excellent", "good", "fair", "poor"]
  Evaluate whether top 10 form a strong scientific foundation
- "score_rationale": a single concise sentence (max 30 words)
  Should reflect usefulness, diversity, and explicitly mention if invalid datasets dragged down the score.

---

Queries and their recommendations to evaluate:
{input_json}

---

Output ONLY a valid JSON array. Each element must follow this exact schema:
{{
  "query_id": "Mixed_001",
  "user_query": "the query text",
  "precision_score": 85,
  "diversity_score": 70,
  "distinct_sources_count": 7,
  "relevant_count": 8,
  "top10_quality": "good",
  "score_rationale": "Top-10 include core datasets with moderate diversity, though 2 datasets were penalized as Invalid."
}}
"""



def setup_logging(verbose: bool = False) -> None:

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def clean_json_response(raw: str) -> str:

    raw = raw.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        if len(parts) >= 2:
            raw = parts[1]
            if raw.lstrip().lower().startswith("json"):
                raw = raw.lstrip()[4:]
        raw = raw.strip("` \n")


    if not raw.startswith("["):
        start, end = raw.find("["), raw.rfind("]")
        if start != -1 and end != -1 and end > start:
            raw = raw[start : end + 1]
    return raw



def score_batch(
    client: OpenAI,
    batch: list[dict[str, Any]],
    model: str,
    temperature: float,
) -> list[dict[str, Any]]:
    """Single LLM call: score one batch of recommendations."""
    prompt = USER_PROMPT_TEMPLATE.format(
        input_json=json.dumps(batch, ensure_ascii=False, indent=2)
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
    )
    raw = response.choices[0].message.content or ""
    return json.loads(clean_json_response(raw))


def score_batch_with_retry(
    client: OpenAI,
    batch: list[dict[str, Any]],
    model: str,
    temperature: float,
    max_retries: int,
    retry_sleep: float,
    batch_label: str = "",
) -> list[dict[str, Any]]:

    attempt = 0
    last_error: Exception | None = None
    infinite = max_retries == -1

    while infinite or attempt <= max_retries:
        attempt += 1
        label = f"attempt {attempt}" + ("" if infinite else f"/{max_retries + 1}")

        try:
            result = score_batch(client, batch, model, temperature)
            if len(result) != len(batch):
                raise ValueError(
                    f"returned count mismatch (expected {len(batch)}, got {len(result)})"
                )
            if attempt > 1:
                logging.info("  %s succeeded on retry", label)
            return result

        except Exception as e:  # noqa: BLE001 - retry every error class
            last_error = e
            logging.warning("  %s failed [%s]: %s", label, type(e).__name__, e)

            if not infinite and attempt > max_retries:
                break

            wait = retry_sleep * attempt  # gentle linear backoff
            logging.info("  retrying batch %s in %.1fs...", batch_label, wait)
            time.sleep(wait)

    raise RuntimeError(
        f"Batch {batch_label} failed after {attempt} attempts. Last error: {last_error}"
    )



def compute_summary(results: list[dict[str, Any]]) -> dict[str, Any]:

    if not results:
        return {}

    n = len(results)
    precision_scores = [r["precision_score"] for r in results]
    diversity_scores = [r.get("diversity_score", 0) for r in results]
    relevant_counts = [r.get("relevant_count", 0) for r in results]
    distinct_counts = [r.get("distinct_sources_count", 0) for r in results]

    def avg(lst: list[float]) -> float:
        return round(sum(lst) / len(lst), 2)

    def pct_above(lst: list[float], threshold: float) -> float:
        return round(sum(1 for x in lst if x >= threshold) / len(lst) * 100, 1)

    return {
        "total_queries_evaluated": n,
        "average_scores": {
            "precision_score": avg(precision_scores),
            "diversity_score": avg(diversity_scores),
            "composite_score": avg(
                [(p + d) / 2 for p, d in zip(precision_scores, diversity_scores)]
            ),
        },
        "score_distribution": {
            "precision_above_80_pct": pct_above(precision_scores, 80),
            "diversity_above_80_pct": pct_above(diversity_scores, 80),
        },
        "average_relevant_datasets_per_query": avg(relevant_counts),
        "average_distinct_sources_per_query": avg(distinct_counts),
        "min_scores": {
            "precision_score": min(precision_scores),
            "diversity_score": min(diversity_scores),
        },
        "max_scores": {
            "precision_score": max(precision_scores),
            "diversity_score": max(diversity_scores),
        },
    }



def score_one_file(
    json_file: Path,
    client: OpenAI,
    output_folder: Path,
    model: str,
    batch_size: int,
    temperature: float,
    sleep_sec: float,
    max_retries: int,
    retry_sleep: float,
) -> tuple[dict[str, Any], int, list[int]]:

    logging.info("=" * 60)
    logging.info("Processing file: %s", json_file.name)

    with json_file.open(encoding="utf-8") as f:
        recommendations = json.load(f)

    total = len(recommendations)
    logging.info("  %d recommendations to score", total)

    all_scores: list[dict[str, Any]] = []
    failed_items: list[int] = []

    batches = [recommendations[i : i + batch_size] for i in range(0, total, batch_size)]
    total_batches = len(batches)

    for batch_idx, batch in enumerate(batches, start=1):
        start_no = (batch_idx - 1) * batch_size + 1
        end_no = min(batch_idx * batch_size, total)
        label = f"{batch_idx}/{total_batches} (items {start_no}-{end_no})"
        logging.info("  Batch %s ...", label)

        try:
            batch_scores = score_batch_with_retry(
                client=client,
                batch=batch,
                model=model,
                temperature=temperature,
                max_retries=max_retries,
                retry_sleep=retry_sleep,
                batch_label=label,
            )

            if len(batch_scores) != len(batch):
                logging.warning(
                    "    Returned count mismatch (expected %d, got %d). Keeping anyway.",
                    len(batch),
                    len(batch_scores),
                )

            all_scores.extend(batch_scores)
            avg_p = sum(s["precision_score"] for s in batch_scores) / len(batch_scores)
            avg_d = sum(s.get("diversity_score", 0) for s in batch_scores) / len(batch_scores)
            logging.info(
                "    Done. P=%.0f DIV=%.0f (cumulative %d)",
                avg_p, avg_d, len(all_scores),
            )

        except Exception as e:  # noqa: BLE001
            logging.error("    Batch failed permanently: %s", e)
            failed_items.extend(range(start_no - 1, end_no))

        if batch_idx < total_batches:
            time.sleep(sleep_sec)


    summary = compute_summary(all_scores)
    out_path = output_folder / f"{json_file.stem}_score.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(
            {"summary": summary, "details": all_scores},
            f, ensure_ascii=False, indent=2,
        )

    logging.info("  Saved: %s", out_path)
    if failed_items:
        logging.warning("  Failed indices: %s", failed_items)

    return summary, len(all_scores), failed_items



def run(
    input_folder: Path,
    output_folder: Path,
    api_key: str,
    base_url: str | None,
    model: str,
    batch_size: int,
    temperature: float,
    sleep_sec: float,
    max_retries: int,
    retry_sleep: float,
) -> None:
    output_folder.mkdir(parents=True, exist_ok=True)


    json_files = sorted(
        f for f in input_folder.glob("*.json") if "_score" not in f.stem
    )

    if not json_files:
        logging.warning("No input JSON files found in: %s", input_folder)
        return

    logging.info("Found %d recommendation file(s). Starting...", len(json_files))

    client_kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)

    all_summaries: dict[str, dict[str, Any]] = {}
    total_evaluated = 0
    total_failed: list[int] = []

    for json_file in json_files:
        try:
            summary, evaluated, failed = score_one_file(
                json_file=json_file,
                client=client,
                output_folder=output_folder,
                model=model,
                batch_size=batch_size,
                temperature=temperature,
                sleep_sec=sleep_sec,
                max_retries=max_retries,
                retry_sleep=retry_sleep,
            )
            all_summaries[json_file.name] = summary
            total_evaluated += evaluated
            total_failed.extend(failed)
        except Exception as e: 
            logging.error("Fatal error processing %s, skipping: %s", json_file.name, e)


    overview_path = output_folder / "_overview.json"
    with overview_path.open("w", encoding="utf-8") as f:
        json.dump(all_summaries, f, ensure_ascii=False, indent=2)


    logging.info("=" * 60)
    logging.info(
        "Finished. Processed %d file(s), scored %d query record(s).",
        len(json_files),
        total_evaluated,
    )
    logging.info("Results folder: %s", output_folder)
    logging.info("Overview saved to: %s", overview_path)
    if total_failed:
        logging.warning("Some items failed; check per-file logs above.")

    print(f"{'File':<50} {'Precision':>10} {'Diversity':>10} {'Composite':>10}")
    print("-" * 84)
    for fname, s in all_summaries.items():
        if s:
            avg = s["average_scores"]
            print(
                f"{fname:<50} "
                f"{avg['precision_score']:>10.2f} "
                f"{avg['diversity_score']:>10.2f} "
                f"{avg['composite_score']:>10.2f}"
            )
    print("-" * 84)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score scientific dataset recommendations (precision + diversity) "
                    "using an LLM as evaluator.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-folder", "-i", type=Path, required=True,
        help="Folder containing recommendation JSON files.",
    )
    parser.add_argument(
        "--output-folder", "-o", type=Path, required=True,
        help="Folder where per-file scores and the overview will be written.",
    )
    parser.add_argument(
        "--api-key", type=str, default=os.environ.get("OPENAI_API_KEY", ""),
        help="API key. Falls back to the OPENAI_API_KEY environment variable.",
    )
    parser.add_argument(
        "--base-url", type=str, default=os.environ.get("OPENAI_BASE_URL"),
        help="Custom base URL for the LLM endpoint (optional).",
    )
    parser.add_argument(
        "--model", type=str, default="gpt-5",
        help="Model name to use.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=2,
        help="Number of queries per API call.",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.2,
        help="Sampling temperature for the model (lower = more consistent scoring).",
    )
    parser.add_argument(
        "--sleep", type=float, default=3.0,
        help="Seconds to sleep between batches (rate-limit friendly).",
    )
    parser.add_argument(
        "--max-retries", type=int, default=-1,
        help="Maximum retry attempts per batch. Use -1 for infinite retries.",
    )
    parser.add_argument(
        "--retry-sleep", type=float, default=10.0,
        help="Base seconds to wait before each retry (scaled linearly per attempt).",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug-level logging.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(args.verbose)

    if not args.api_key:
        logging.error(
            "No API key provided. Use --api-key or set the OPENAI_API_KEY env variable."
        )
        return 1

    if not args.input_folder.exists() or not args.input_folder.is_dir():
        logging.error("Input folder not found or not a directory: %s", args.input_folder)
        return 1

    try:
        run(
            input_folder=args.input_folder,
            output_folder=args.output_folder,
            api_key=args.api_key,
            base_url=args.base_url,
            model=args.model,
            batch_size=args.batch_size,
            temperature=args.temperature,
            sleep_sec=args.sleep,
            max_retries=args.max_retries,
            retry_sleep=args.retry_sleep,
        )
    except KeyboardInterrupt:
        logging.warning("Interrupted by user.")
        return 130
    except Exception as e:  
        logging.exception("Fatal error: %s", e)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())