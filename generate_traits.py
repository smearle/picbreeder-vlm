import json
import os
import re
import time
from typing import List, Optional

from im_query import client, DEFAULT_MODEL, types, genai, _is_token_limit_error
from prompts import DEFAULT_SYSTEM_INSTRUCTION

MODEL_NAME = "gemini-3-pro-preview"
TRAITS_FILE = "personality_traits.json"
SEED = 42

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

            response = client.models.generate_content(**request_kwargs)
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

def main(total_traits: int = 1_000):
    print(f"Generating {total_traits} personality traits using {MODEL_NAME}...")
    
    batch_size = 50
    batches = (total_traits + batch_size - 1) // batch_size
    
    traits: List[str] = []
    history_contents: List[types.Content] = []
    
    # We strip formatting placeholders from the system instruction to make it cleaner for the LLM
    # although leaving them is also fine.
    system_prompt_text = DEFAULT_SYSTEM_INSTRUCTION
    
    base_instruction = (
        "You are an assistant that produces creative and diverse personality traits for an AI agent. "
        "The agent will be performing a task described in the prompt provided. "
        "For each request, return strictly valid JSON: an array of strings (length exactly equal to the requested batch size). "
        "Each string should be a personality trait in the second person (e.g., 'You like driving at night'). "
        "The traits should be unique and distinct. "
        "Do not include commentary or explanatory text outside the JSON array."
    )
    
    task_context = (
        f"Here is the system prompt for the task the agent will be performing:\n\n"
        f"--- START SYSTEM PROMPT ---\n"
        f"{system_prompt_text}\n"
        f"--- END SYSTEM PROMPT ---\n\n"
        "Please generate a list of 100 unique personality traits, which you think may implicitly/indirectly affect the agent's behavior on the task at hand. "
        "They can be positive, negative, ambivalent, ambiguous; abstract or concrete; related or unrelated to the task at hand in a literal sense. "
        "Anything that might add some unique, more or less subtle quirk or quality to the agent's behavior in the given task. "
        "The traits should be written in second person 'You like driving at night', 'Your favorite ice cream flavor is rocky road.', 'Sunsets remind you of your ex' etc."
        "Avoid giving explicit goals or optimization objectives, focus on individual traits that might influence behavior in various ways."
    )
    
    full_system_instruction = f"{base_instruction}\n\n{task_context}"

    batch_idx = 0
    while len(traits) < total_traits:
        batch_idx += 1
        current_batch_size = min(batch_size, total_traits - len(traits))
            
        prompt = (
            f"Generate {current_batch_size} unique personality traits as a JSON array of strings. "
            f"This is batch {batch_idx} (target total: {total_traits}, current: {len(traits)}). "
            f"Ensure they are distinct from previous batches if any."
        )
        
        print(f"Requesting batch {batch_idx} (asking for {current_batch_size} traits)...")
        try:
            response_text, history_contents = send_text_prompt(
                prompt, 
                history_contents=history_contents, 
                system_instruction=full_system_instruction,
                model=MODEL_NAME
            )
        except Exception as e:
            print(f"Error calling model: {e}. Retrying in 5 seconds...")
            time.sleep(5)
            continue
        
        parsed = extract_json_from_text(response_text)
        batch_list = None
        if isinstance(parsed, list):
            batch_list = parsed
        
        if batch_list is None:
            print(f"Warning: failed to parse JSON for batch {batch_idx}. Retrying...")
            time.sleep(1)
            continue
            
        clean_items = [str(item) for item in batch_list if isinstance(item, str)]
        if not clean_items:
            print(f"Warning: parsed JSON contained no valid strings. Retrying...")
            time.sleep(1)
            continue

        traits.extend(clean_items)
        print(f"Received {len(clean_items)} traits. Total: {len(traits)}/{total_traits}")
        
        # Save progress after each batch
        output_data = {
            "metadata": {
                "model": MODEL_NAME,
                "seed": SEED,
                "timestamp": time.time()
            },
            "traits": traits
        }
        
        with open(TRAITS_FILE, "w") as f:
            json.dump(output_data, f, indent=2)
            
        time.sleep(1)

    print(f"Completed generation: wrote {len(traits)} traits to {TRAITS_FILE}")

if __name__ == "__main__":
    main()
