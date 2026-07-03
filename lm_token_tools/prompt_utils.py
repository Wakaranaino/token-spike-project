"""
prompt_utils.py

Prompt and response utilities for LM token / CoT-token experiments.

This file handles:
- building the user prompt sent to the model
- extracting final answer text
- extracting visible reasoning / CoT text
- extracting token usage fields

It does NOT handle:
- loading CSV files
- model/API setup
- experiment loops
- condition definitions
- saving results
"""

import re


# =======================
# PROMPT BUILDING
# =======================

def build_user_prompt(question_text: str, user_instruction: str | None = None) -> str:
    """
    Build the final user prompt.

    question_text:
        The actual question/prompt we want to test.

    user_instruction:
        Optional extra instruction, such as:
        - "Answer the question."
        - "Choose one option."
        - "Return only the final answer."

    If no user_instruction is provided, the question is sent as-is.
    """

    question_text = str(question_text).strip()

    if user_instruction is None or str(user_instruction).strip() == "":
        return question_text

    return f"{question_text}\n\n{str(user_instruction).strip()}"


def build_choice_prompt(question_text: str) -> str:
    """
    Optional helper for multiple-choice tests.

    This asks the model to return only one option letter.
    Use only when the prompt has options like A/B/C/D.
    """

    return build_user_prompt(
        question_text=question_text,
        user_instruction=(
            "Choose one option. "
            "Return only the option letter, such as A, B, C, or D. "
            "Do not explain."
        )
    )


# =======================
# REASONING / CoT EXTRACTION
# =======================

def extract_think_block(content: str) -> tuple[str, str]:
    """
    Extract reasoning text from <think>...</think> format.

    Some reasoning models return output like:

        <think>
        reasoning here
        </think>
        final answer here

    Returns:
        reasoning_text, final_answer

    If no <think> block is found:
        reasoning_text = ""
        final_answer = original content
    """

    content = str(content or "")

    pattern = r"<think>(.*?)</think>"
    match = re.search(pattern, content, flags=re.DOTALL | re.IGNORECASE)

    if not match:
        return "", content.strip()

    reasoning_text = match.group(1).strip()

    # Remove the think block from content to get final answer.
    final_answer = re.sub(pattern, "", content, flags=re.DOTALL | re.IGNORECASE).strip()

    return reasoning_text, final_answer


def extract_reasoning_from_field(reasoning_field) -> str:
    """
    Normalize reasoning text from provider-specific reasoning fields.

    Some APIs return reasoning as:
    - a plain string
    - a dictionary with a text/content field
    - a list of reasoning blocks

    This helper converts those shapes into one text string.
    """

    if reasoning_field is None:
        return ""

    if isinstance(reasoning_field, str):
        return reasoning_field.strip()

    if isinstance(reasoning_field, dict):
        for key in ["content", "text", "reasoning_content", "reasoning"]:
            value = reasoning_field.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        if isinstance(reasoning_field.get("content"), list):
            return extract_reasoning_from_field(reasoning_field.get("content"))

        return ""

    if isinstance(reasoning_field, list):
        text_parts = []

        for item in reasoning_field:
            if isinstance(item, str) and item.strip():
                text_parts.append(item.strip())
                continue

            if isinstance(item, dict):
                item_type = item.get("type")

                if item_type == "reasoning":
                    item_content = item.get("content")
                    if isinstance(item_content, str) and item_content.strip():
                        text_parts.append(item_content.strip())
                        continue

                for key in ["content", "text", "reasoning_content", "reasoning"]:
                    value = item.get(key)
                    if isinstance(value, str) and value.strip():
                        text_parts.append(value.strip())
                        break

        return "\n".join(text_parts).strip()

    return ""


def get_nested_value(data, path: str | None):
    """
    Read one nested value from a dictionary/list structure using a dot path.

    Examples:
    - choices.0.message.reasoning
    - output_tokens_details.reasoning_tokens

    Returns None if the path is missing.
    """

    if path is None or str(path).strip() == "":
        return None

    current_value = data

    for part in str(path).split("."):
        if isinstance(current_value, list):
            if not part.isdigit():
                return None

            index = int(part)
            if index < 0 or index >= len(current_value):
                return None

            current_value = current_value[index]
            continue

        if isinstance(current_value, dict):
            if part not in current_value:
                return None

            current_value = current_value[part]
            continue

        return None

    return current_value


def extract_token_usage(
    usage: dict | None,
    reasoning_token_path: str | None = None
) -> dict:
    """
    Extract token usage fields from API response.

    Handles common token fields across providers.

    If a field is missing, returns None for that field.
    """

    usage = usage or {}
    manual_reasoning_tokens = get_nested_value(usage, reasoning_token_path)

    if manual_reasoning_tokens is not None:
        return {
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "reasoning_tokens": manual_reasoning_tokens,
            "reasoning_tokens_source": reasoning_token_path
        }

    reasoning_token_paths = [
        ("completion_tokens_details", "reasoning_tokens"),
        ("output_tokens_details", "reasoning_tokens"),
        ("completion_tokens_details", "thinking_tokens"),
        ("output_tokens_details", "thinking_tokens"),
        ("usage_metadata", "thoughts_token_count"),
        ("usageMetadata", "thoughtsTokenCount"),
        ("reasoning_tokens",),
        ("thinking_tokens",),
        ("thoughts_token_count",),
        ("thoughtsTokenCount",),
    ]

    reasoning_tokens = None
    reasoning_tokens_source = None

    for path in reasoning_token_paths:
        current_value = usage
        path_found = True

        for key in path:
            if isinstance(current_value, dict) and key in current_value:
                current_value = current_value[key]
            else:
                path_found = False
                break

        if path_found and current_value is not None:
            reasoning_tokens = current_value
            reasoning_tokens_source = ".".join(path)
            break

    return {
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "reasoning_tokens": reasoning_tokens,
        "reasoning_tokens_source": reasoning_tokens_source
    }


# =======================
# MAIN RESPONSE PARSER
# =======================

def parse_model_message(
    message: dict,
    usage: dict | None = None,
    reasoning_text_path: str | None = None,
    reasoning_token_path: str | None = None
) -> dict:
    """
    Parse one model response message.

    This supports several possible reasoning-output formats:

    1. Qwen / LM Studio style:
        message["reasoning_content"]

    2. <think>...</think> style:
        message["content"] contains a think block

    3. Normal chat model:
        message["content"] only

    Returns a clean dictionary with:
    - final_answer
    - reasoning_text
    - has_reasoning
    - token usage fields
    """

    message = message or {}

    content = message.get("content") or ""
    reasoning_content = message.get("reasoning_content") or ""
    reasoning_field = message.get("reasoning")
    normalized_reasoning_field = extract_reasoning_from_field(reasoning_field)
    manual_reasoning_value = get_nested_value(message, reasoning_text_path)
    manual_reasoning_text = extract_reasoning_from_field(manual_reasoning_value)

    # Case 1: reasoning text loaded from a manual override path.
    if manual_reasoning_text:
        reasoning_text = manual_reasoning_text
        final_answer = content.strip()

    # Case 2: reasoning stored separately in reasoning_content.
    elif reasoning_content.strip():
        reasoning_text = reasoning_content.strip()
        final_answer = content.strip()

    # Case 3: reasoning stored separately in another reasoning field.
    elif normalized_reasoning_field:
        reasoning_text = normalized_reasoning_field
        final_answer = content.strip()

    # Case 4: reasoning embedded inside content as <think>...</think>.
    else:
        reasoning_text, final_answer = extract_think_block(content)

    token_usage = extract_token_usage(
        usage,
        reasoning_token_path=reasoning_token_path
    )

    return {
        "final_answer": final_answer,
        "reasoning_text": reasoning_text,
        "has_reasoning": bool(reasoning_text.strip()),

        # Token fields
        "prompt_tokens": token_usage["prompt_tokens"],
        "completion_tokens": token_usage["completion_tokens"],
        "reasoning_tokens": token_usage["reasoning_tokens"],
        "reasoning_tokens_source": token_usage["reasoning_tokens_source"],
        "total_tokens": token_usage["total_tokens"]
    }


# =======================
# QUICK TEST
# =======================

if __name__ == "__main__":
    # Example 1: Qwen / reasoning_content style
    sample_message_1 = {
        "content": "Hello!",
        "reasoning_content": "Thinking Process: I should greet the user briefly."
    }

    sample_usage_1 = {
        "prompt_tokens": 10,
        "completion_tokens": 30,
        "total_tokens": 40,
        "completion_tokens_details": {
            "reasoning_tokens": 25
        }
    }

    print("Test 1: reasoning_content style")
    print(parse_model_message(sample_message_1, sample_usage_1))

    # Example 2: <think>...</think> style
    sample_message_2 = {
        "content": "<think>I should answer briefly.</think>\nHello!"
    }

    sample_usage_2 = {
        "prompt_tokens": 8,
        "completion_tokens": 20,
        "total_tokens": 28
    }

    print("\nTest 2: <think> style")
    print(parse_model_message(sample_message_2, sample_usage_2))
