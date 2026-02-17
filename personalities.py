import json
import os
import re
import time
from typing import List, Optional

from im_query import DEFAULT_MODEL, get_genai_client, types, genai, _is_token_limit_error


OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def send_text_prompt(
    prompt: str,
    history_contents: Optional[List[types.Content]] = None,
    system_instruction: Optional[str] = None,
    model: str = DEFAULT_MODEL,
) -> (str, List[types.Content]):
    """Send a text-only prompt to Gemini reusing `client` and `types` from `im_query`.

    Returns the response text and the updated history_contents list (appended with
    the user and model turn). History contents is a list of types.Content objects.
    """
    if history_contents is None:
        history_contents = []

    # Build the contents array starting from history (if any) and adding the user turn
    contents = list(history_contents)
    user_part = types.Part(text=prompt)
    contents.append(types.Content(role="user", parts=[user_part]))

    config = types.GenerateContentConfig(system_instruction=system_instruction) if system_instruction else None

    # Try until success on server error, or rethrow other exceptions after some retries
    retry = 0
    while True:
        try:
            request_kwargs = {
                "model": model,
                "contents": contents,
            }
            if config is not None:
                request_kwargs["config"] = config

            response = get_genai_client().models.generate_content(**request_kwargs)
            response_text = getattr(response, "text", "") or ""
            break
        except Exception as exc:
            # Handle token limit by trimming history one turn at a time
            if _is_token_limit_error(exc) and len(contents) >= 2:
                # drop the earliest user+model turn (2 contents) if present
                # find first user content index and drop it plus following model content if present
                # simplest: drop first two entries if length > 2
                contents = contents[2:]
                retry += 1
                continue
            # server errors -> backoff
            if isinstance(exc, genai.errors.ServerError):
                time.sleep(min(10 * (1 + retry), 60))
                retry += 1
                continue
            # otherwise re-raise
            raise

    # Append the user and model contents to history_contents and keep last 10 turns
    history_contents.extend([types.Content(role="user", parts=[user_part]), types.Content(role="model", parts=[types.Part(text=response_text)])])

    # Keep at most 10 user+assistant turns -> 20 Content items
    max_contents = 20
    if len(history_contents) > max_contents:
        history_contents = history_contents[-max_contents:]

    return response_text, history_contents


def extract_json_from_text(text: str) -> Optional[object]:
    """Try to parse JSON from text. If direct parse fails, attempt to extract the first
    JSON array or object found using a conservative bracket extraction.
    """
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    # Attempt to extract a JSON array or object by searching for balanced [] or {}
    # Find the first '[' and the matching closing ']' (simple heuristic)
    array_match = re.search(r"(\[).*", text)
    if array_match:
        start = array_match.start(1)
        # find the last closing bracket
        end = text.rfind("]")
        if end != -1 and end > start:
            candidate = text[start : end + 1]
            try:
                return json.loads(candidate)
            except Exception:
                pass

    # Try for object
    obj_match = re.search(r"(\{).*", text)
    if obj_match:
        start = obj_match.start(1)
        end = text.rfind("}")
        if end != -1 and end > start:
            candidate = text[start : end + 1]
            try:
                return json.loads(candidate)
            except Exception:
                pass

    return None


def main(total_personalities: int = 100, batch_size: int = 10):
    assert total_personalities % batch_size == 0, "total_personalities must be divisible by batch_size"
    batches = total_personalities // batch_size

    personalities: List[dict] = []
    history_contents: List[types.Content] = []

    system_instruction = (
        "You are an assistant that produces concise, creative, and diverse human personality descriptions. "
        "For each request, return strictly valid JSON: an array of objects (length exactly equal to the requested batch size). "
        "Each object must contain the following fields: id (string), name (string), age (integer), background (one-line string), "
        "personality (one-line string), profession (string), upbringing (one-line string), and a short_bio (1-3 sentences). "
        "Ensure each person is distinct across batches. Avoid repeating previously generated people. "
        "Do not include commentary or explanatory text outside the JSON array."
    )

    for batch_idx in range(1, batches + 1):
        prompt = (
            f"Generate {batch_size} unique human personality descriptions as a JSON array. "
            f"This is batch {batch_idx} of {batches}. "
            f"Return exactly {batch_size} objects with the fields: id, name, age, background, personality, profession, upbringing, short_bio. "
            f"Make each person believable and diverse in culture, age, socioeconomic background, and profession. "
        )

        print(f"Requesting batch {batch_idx}/{batches}...")
        response_text, history_contents = send_text_prompt(prompt, history_contents=history_contents, system_instruction=system_instruction)

        parsed = extract_json_from_text(response_text)
        batch_list = None
        if isinstance(parsed, list):
            batch_list = parsed
        elif isinstance(parsed, dict):
            # single object -> wrap
            batch_list = [parsed]

        if batch_list is None:
            # Save raw response for debugging and continue
            fallback_path = os.path.join(OUTPUT_DIR, f"personalities_batch_{batch_idx}_raw.txt")
            with open(fallback_path, "w", encoding="utf-8") as fh:
                fh.write(response_text)
            print(f"Warning: failed to parse JSON for batch {batch_idx}. Saved raw output to {fallback_path}")
            # Continue loop but do not add to personalities
            time.sleep(1)
            continue

        # Basic validation / normalization: ensure each item is a dict with required keys
        clean_items = []
        required = {"name", "age", "background", "personality", "profession", "upbringing", "short_bio"}
        for item in batch_list:
            if not isinstance(item, dict):
                continue
            # fill missing keys with placeholders
            for key in required:
                if key not in item:
                    item[key] = ""
            clean_items.append(item)

        personalities.extend(clean_items)

        # Save incremental output for safety
        out_path = os.path.join(OUTPUT_DIR, f"personalities_up_to_batch_{batch_idx}.json")
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(personalities, fh, indent=2, ensure_ascii=False)

        print(f"Saved {len(clean_items)} items from batch {batch_idx}. Total so far: {len(personalities)}")
        # small pause to avoid hitting rate limits
        time.sleep(1)

    # Final save
    final_path = os.path.join(OUTPUT_DIR, "personalities.json")
    with open(final_path, "w", encoding="utf-8") as fh:
        json.dump(personalities, fh, indent=2, ensure_ascii=False)

    print(f"Completed generation: wrote {len(personalities)} personalities to {final_path}")


if __name__ == "__main__":
    main(total_personalities=100, batch_size=10)
