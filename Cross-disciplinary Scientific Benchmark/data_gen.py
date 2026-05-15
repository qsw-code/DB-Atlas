

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

from openai import OpenAI



SYSTEM_PROMPT = """You are an expert Scientific Researcher and Search Engine Evaluator.
Output ONLY valid JSON. No explanation, no markdown, no code blocks. \
Just the raw JSON array."""

USER_PROMPT_TEMPLATE = """I am evaluating the semantic understanding capabilities of a \
Scientific Dataset Retrieval System.
Your task is to generate {n} simulated search queries focused on fuzzy searches \
and broad research directions.

These queries should include BOTH:
1. Single-discipline research topics
2. Multi-disciplinary (cross-domain) research topics

IMPORTANT DISTRIBUTION REQUIREMENT:
- At least 4 queries must be single-discipline
- At least 4 queries must be cross-disciplinary
- The remaining can be either

CRITICAL Query Format Constraint:
Real users rarely type full questions into search bars. Therefore, the user_query \
MUST be formatted as:
1. Declarative statements
2. Descriptive noun phrases
3. Conceptual research topics
4. No more than seven words

Examples:
"single-cell RNA sequencing"
"urban heat islands and energy consumption"

DO NOT:
- Use interrogative sentences
- Use phrases like "Are there any...", "Where can I find...", "I need datasets for...", or "How does..."
- Only provide the core research concept.

Query Characteristics:
- Fuzzy & Exploratory: The query should describe a broad phenomenon, hypothesis, \
or variable relationship
- The user does not know specific dataset names
- Disciplinary Scope: Some queries should stay within a single domain
- Some queries should explicitly bridge multiple domains (e.g. Climate Science + Economics)

This is batch {batch_num} of {total_batches}. Use query_id format: \
"Mixed_{start_id:03d}" through "Mixed_{end_id:03d}".
Start from query_id "Mixed_{start_id:03d}".

Output Format:
Output ONLY a valid JSON array (no markdown, no extra text). Each element must \
follow this exact schema:
{{
    "query_id": "Mixed_001",
    "intersecting_domains": [
      "Climate Science",
      "Biogeochemistry"
    ],
    "user_query": "permafrost thaw methane emissions feedbacks",
    "fuzzy_intent_explanation": "The user studies links between permafrost thaw, \
methane release, and climate feedback strength across Arctic landscapes."
  }},
  {{
    "query_id": "BioMixed_002",
    "intersecting_domains": [
      "Neuroscience"
    ],
    "user_query": "synaptic plasticity dynamics",
    "fuzzy_intent_explanation": "The user aims to explore mechanisms and \
variability of synaptic strength changes across conditions and timescales."
}}"""



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
    return raw


def save_results(results: list[dict[str, Any]], filepath: Path) -> None:

    filepath.parent.mkdir(parents=True, exist_ok=True)
    with filepath.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)



def generate_batch(
    client: OpenAI,
    batch_num: int,
    total_batches: int,
    queries_per_batch: int,
    model: str,
    temperature: float,
    max_retries: int = 3,
) -> list[dict[str, Any]]:

    start_id = (batch_num - 1) * queries_per_batch + 1
    end_id = batch_num * queries_per_batch

    prompt = USER_PROMPT_TEMPLATE.format(
        n=queries_per_batch,
        batch_num=batch_num,
        total_batches=total_batches,
        start_id=start_id,
        end_id=end_id,
    )

    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
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

        except json.JSONDecodeError as e:
            last_exc = e
            logging.warning("Attempt %d: JSON parse failed (%s)", attempt, e)
        except Exception as e:  # noqa: BLE001 - we want to retry any API error
            last_exc = e
            logging.warning("Attempt %d: API call failed (%s)", attempt, e)

        if attempt < max_retries:
            backoff = 2 ** (attempt - 1)
            logging.info("Retrying in %.1fs...", backoff)
            time.sleep(backoff)

    assert last_exc is not None
    raise last_exc


def run(
    output_file: Path,
    api_key: str,
    base_url: str | None,
    model: str,
    total_batches: int,
    queries_per_batch: int,
    temperature: float,
    sleep_sec: float,
    max_retries: int,
) -> None:
    """Main pipeline: generate batches of queries and save them to JSON."""
    client_kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)

    all_queries: list[dict[str, Any]] = []
    failed_batches: list[int] = []

    logging.info(
        "Generating %d batches x %d queries (target: %d total).",
        total_batches,
        queries_per_batch,
        total_batches * queries_per_batch,
    )

    for batch_num in range(1, total_batches + 1):
        logging.info("Batch %d/%d...", batch_num, total_batches)
        try:
            batch = generate_batch(
                client=client,
                batch_num=batch_num,
                total_batches=total_batches,
                queries_per_batch=queries_per_batch,
                model=model,
                temperature=temperature,
                max_retries=max_retries,
            )

            if len(batch) != queries_per_batch:
                logging.warning(
                    "Returned count mismatch (expected %d, got %d). Keeping anyway.",
                    queries_per_batch,
                    len(batch),
                )

            all_queries.extend(batch)
            logging.info(
                "Batch %d done. Total queries: %d", batch_num, len(all_queries)
            )

        except Exception as e:  # noqa: BLE001
            logging.error("Batch %d failed permanently: %s", batch_num, e)
            failed_batches.append(batch_num)

        if batch_num < total_batches:
            time.sleep(sleep_sec)

    save_results(all_queries, output_file)


    logging.info("=" * 55)
    logging.info("Finished. Generated %d queries.", len(all_queries))
    logging.info("Results saved to: %s", output_file)

    if failed_batches:
        logging.warning("Failed batches (retry manually): %s", failed_batches)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic fuzzy research queries for evaluating "
                    "scientific dataset retrieval systems.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output", "-o", type=Path, required=True,
        help="Path where the output JSON file will be written.",
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
        "--total-batches", type=int, default=10,
        help="Number of batches to generate.",
    )
    parser.add_argument(
        "--queries-per-batch", type=int, default=10,
        help="Number of queries per batch.",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.85,
        help="Sampling temperature for the model (higher = more diverse).",
    )
    parser.add_argument(
        "--sleep", type=float, default=1.0,
        help="Seconds to sleep between batches (rate-limit friendly).",
    )
    parser.add_argument(
        "--max-retries", type=int, default=3,
        help="Maximum retry attempts per batch on failure.",
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

    try:
        run(
            output_file=args.output,
            api_key=args.api_key,
            base_url=args.base_url,
            model=args.model,
            total_batches=args.total_batches,
            queries_per_batch=args.queries_per_batch,
            temperature=args.temperature,
            sleep_sec=args.sleep,
            max_retries=args.max_retries,
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