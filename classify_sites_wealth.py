#!/usr/bin/env python3
import argparse
import csv
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

from tqdm import tqdm

ALLOWED_LABELS = {"Wealthy", "Medium Wealthy", "Poor"}
DEFAULT_MODEL = "gpt-4.1"
API_URL = "https://api.openai.com/v1/responses"


def load_env(env_path: Path) -> None:
    """Load key-value pairs from a `.env` file into process environment vars.

    Purpose:
        Provide local configuration loading for API credentials when running
        from the command line.

    Args:
        env_path: Path to the `.env` file.

    Returns:
        None. Updates `os.environ` in place for keys not already set.
    """
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def extract_output_text(response_json: dict) -> str:
    """Extract model text output from a Responses API JSON payload.

    Purpose:
        Normalize both `output_text` convenience responses and structured
        `output` message blocks into a single text string.

    Args:
        response_json: Parsed JSON object returned by the API.

    Returns:
        The stripped output text content. Returns an empty string if no text
        content exists.
    """
    if "output_text" in response_json:
        return str(response_json["output_text"]).strip()
    output = response_json.get("output", [])
    texts: list[str] = []
    for item in output:
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "output_text" and "text" in content:
                texts.append(content["text"])
    return "\n".join(texts).strip()


def call_openai(api_key: str, model: str, prompt: str, max_retries: int = 5) -> str:
    """Call the OpenAI Responses API and return the classifier label text.

    Purpose:
        Send a strict classification prompt, retrying transient API/network
        failures with exponential backoff.

    Args:
        api_key: OpenAI API key used for authorization.
        model: Model name to invoke for classification.
        prompt: Record-specific user input text to classify.
        max_retries: Maximum number of retry attempts for transient failures.

    Returns:
        Model output text containing the predicted class label.

    Raises:
        urllib.error.HTTPError: If a non-retriable HTTP error occurs.
        urllib.error.URLError: If network errors persist after retries.
        RuntimeError: If all retry attempts are exhausted.
    """
    payload = {
        "model": model,
        "instructions": (
            "You are a strict classifier. Your task is to classify countries or places "
            "according to modern-day economic standards into exactly one of these labels: "
            "Wealthy, Medium Wealthy, Poor.\n\n"
            "Guidelines:\n"
            "- Use contemporary, real-world standards (income, infrastructure, access to services, "
            "economic development, and overall prosperity).\n"
            "- If the record is not a country but a place within a country, infer the likely "
            "country/region from the available fields and classify based on that modern context.\n"
            "- If data is limited, make the best reasonable inference using any provided fields "
            "(e.g., province, modern name, rank, cost, connectivity).\n"
            "- Be consistent and conservative; prefer Medium Wealthy when evidence is mixed.\n"
            "- Do not add explanations, just output one label.\n\n"
            "Return exactly one label from: Wealthy, Medium Wealthy, Poor. "
            "Return only the label with no extra text."
        ),
        "input": prompt,
        "temperature": 0,
    }
    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    for attempt in range(max_retries):
        req = urllib.request.Request(API_URL, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                resp_json = json.loads(resp.read().decode("utf-8"))
            return extract_output_text(resp_json)
        except urllib.error.HTTPError as e:
            status = e.code
            if status in {429, 500, 502, 503, 504} and attempt < max_retries - 1:
                time.sleep(2**attempt)
                continue
            raise
        except urllib.error.URLError:
            if attempt < max_retries - 1:
                time.sleep(2**attempt)
                continue
            raise

    raise RuntimeError("Failed to get a response from the API after retries.")


def build_prompt(row: dict) -> str:
    """Build a deterministic classification prompt from one CSV record.

    Purpose:
        Serialize row attributes and ask for a single wealth class label.

    Args:
        row: Dictionary representation of one input CSV row.

    Returns:
        Prompt text to send to the model.
    """
    record = {k: row.get(k) for k in row.keys()}
    return (
        "Classify the following record into one of: Wealthy, Medium Wealthy, Poor, "
        "based on modern times and the record attributes.\n\n"
        f"Record JSON:\n{json.dumps(record, ensure_ascii=False)}\n\n"
        "Label:"
    )


def main() -> None:
    """Run the CLI workflow for classifying sites by modern wealth class.

    Purpose:
        Load configuration, iterate records in an input CSV, classify each row
        via OpenAI, and write the resulting labels to an output CSV file.

    Args:
        None.

    Returns:
        None. Writes an output CSV as a side effect.
    """
    parser = argparse.ArgumentParser(
        description="Classify sites into Wealthy/Medium Wealthy/Poor using OpenAI."
    )
    parser.add_argument(
        "--input",
        default="sites_extended.csv",
        help="Input CSV path (default: sites_extended.csv)",
    )
    parser.add_argument(
        "--output",
        default="sites_extended_with_wealth_class.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENAI_MODEL", DEFAULT_MODEL),
        help=f"OpenAI model (default: {DEFAULT_MODEL}, or OPENAI_MODEL env var)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for number of rows to process",
    )
    args = parser.parse_args()

    env_path = Path(__file__).with_name(".env")
    load_env(env_path)
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is missing. Add it to the .env file.")

    input_path = Path(args.input)
    output_path = Path(args.output)

    with input_path.open(newline="", encoding="utf-8") as f_in:
        reader = csv.DictReader(f_in)
        fieldnames = reader.fieldnames or []
        out_fieldnames = fieldnames + ["wealth_class"]
        with output_path.open("w", newline="", encoding="utf-8") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=out_fieldnames)
            writer.writeheader()

            rows_iter = tqdm(reader, desc="Classifying", unit="row")
            for idx, row in enumerate(rows_iter):
                if args.limit is not None and idx >= args.limit:
                    break
                prompt = build_prompt(row)
                label = call_openai(api_key, args.model, prompt)
                if label not in ALLOWED_LABELS:
                    # One retry with a stricter prompt
                    label = call_openai(
                        api_key,
                        args.model,
                        prompt + "\n\nReturn ONLY one of: Wealthy, Medium Wealthy, Poor.",
                    )
                if label not in ALLOWED_LABELS:
                    label = "Unknown"
                row["wealth_class"] = label
                writer.writerow(row)

    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
