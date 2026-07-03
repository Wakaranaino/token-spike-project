"""
model_runner.py

Reusable experiment runner for LM token / CoT-token experiments.

This file handles the repeated experiment logic:
- loop through questions
- loop through experimental conditions
- optionally repeat trials
- send one independent request per question/condition/trial
- parse final answer, reasoning text, and token usage
- save results to CSV

It uses:
- config_utils.py for backend/model settings
- prompt_utils.py for building prompts and parsing model responses

It does NOT define:
- the question file path
- the experiment conditions
- the backend/model choice

Those should be defined in the final run script, such as:
    run_cot_token_test.py
"""

import time
import random
from pathlib import Path
from datetime import datetime

import requests
import pandas as pd

from lm_token_tools.prompt_utils import build_user_prompt, parse_model_message


def format_request_error(error: Exception, response_text: str | None = None) -> str:
    """
    Build a readable error message for failed model requests.

    If the server returned a response body, include it so API validation
    errors are easier to diagnose.
    """

    error_message = str(error)

    if response_text is None or response_text.strip() == "":
        return error_message

    response_preview = response_text.strip()

    if len(response_preview) > 2000:
        response_preview = response_preview[:2000] + "... [truncated]"

    return f"{error_message} | Response body: {response_preview}"


def resolve_output_file_path(output_file: str | None) -> str:
    """
    Choose a safe CSV output path.

    If no file name is provided, create a timestamped name.
    If the requested file already exists, add a numeric suffix so
    previous results are not overwritten.
    """

    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"cot_token_results_{timestamp}.csv"

    output_path = Path(output_file)

    if not output_path.exists():
        return str(output_path)

    stem = output_path.stem
    suffix = output_path.suffix or ".csv"
    parent = output_path.parent

    counter = 1
    while True:
        candidate = parent / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return str(candidate)
        counter += 1


def is_empty_error_value(value) -> bool:
    """
    Check whether an error field should be treated as empty.

    Successful saved rows usually store an empty string or a missing value.
    """

    if value is None or pd.isna(value):
        return True

    return str(value).strip() == ""


def build_completed_run_keys(df_existing: pd.DataFrame) -> set[tuple[int, str, str]]:
    """
    Build a set of successful run keys from an existing results file.

    Each completed run is identified by:
    - trial
    - question_id
    - condition
    """

    required_columns = {"trial", "question_id", "condition", "error"}
    if not required_columns.issubset(df_existing.columns):
        return set()

    completed_keys = set()

    for _, row in df_existing.iterrows():
        if not is_empty_error_value(row.get("error")):
            continue

        trial_value = row.get("trial")
        question_id = row.get("question_id")
        condition = row.get("condition")

        if pd.isna(trial_value) or pd.isna(question_id) or pd.isna(condition):
            continue

        completed_keys.add((int(trial_value), str(question_id), str(condition)))

    return completed_keys


def existing_results_match_run(
    df_existing: pd.DataFrame,
    config,
    user_instruction: str | None,
    conditions: dict
) -> bool:
    """
    Check whether an existing results file belongs to the same experiment setup.

    Resume is only allowed when the existing file matches the current:
    - backend
    - model
    - user instruction
    - condition/system prompt pairs
    """

    if df_existing.empty:
        return True

    required_columns = {
        "backend",
        "model",
        "user_instruction",
        "condition",
        "system_prompt"
    }

    if not required_columns.issubset(df_existing.columns):
        return False

    existing_backends = {
        str(value) for value in df_existing["backend"].dropna().unique()
    }
    if existing_backends and existing_backends != {config.backend_name}:
        return False

    existing_models = {
        str(value) for value in df_existing["model"].dropna().unique()
    }
    if existing_models and existing_models != {config.model}:
        return False

    normalized_user_instruction = (
        None if user_instruction is None else str(user_instruction)
    )
    existing_user_instructions = {
        None if pd.isna(value) else str(value)
        for value in df_existing["user_instruction"].unique()
    }
    if existing_user_instructions and existing_user_instructions != {normalized_user_instruction}:
        return False

    expected_condition_pairs = {
        (str(condition_name), str(system_prompt))
        for condition_name, system_prompt in conditions.items()
    }
    existing_condition_pairs = {
        (str(row["condition"]), str(row["system_prompt"]))
        for _, row in df_existing[["condition", "system_prompt"]].dropna().iterrows()
    }

    if existing_condition_pairs and not existing_condition_pairs.issubset(expected_condition_pairs):
        return False

    return True


# =======================
# SINGLE MODEL CALL
# =======================

def call_model_once(
    config,
    system_prompt: str,
    user_prompt: str,
    temperature: float | None = None,
    max_tokens: int | None = None,
    reasoning_text_path: str | None = None,
    reasoning_token_path: str | None = None
) -> dict:
    """
    Send one independent request to the model.

    One request = one question + one condition + one trial.

    max_tokens:
        None means no explicit output cap.
        Use a number only if you intentionally want to limit output length.
    """

    messages = []

    if system_prompt and str(system_prompt).strip():
        messages.append({
            "role": "system",
            "content": str(system_prompt).strip()
        })

    messages.append({
        "role": "user",
        "content": user_prompt
    })

    payload = {
        "model": config.model,
        "messages": messages
    }

    # Only add temperature if the model supports it and the user provides it.
    if temperature is not None:
        payload["temperature"] = temperature

    # Only add max_tokens if user explicitly provides it.
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    start_time = time.perf_counter()

    response = requests.post(
        config.chat_url,
        headers=config.headers,
        json=payload,
        timeout=config.timeout
    )

    end_time = time.perf_counter()
    latency_seconds = end_time - start_time

    try:
        response.raise_for_status()
    except requests.HTTPError as error:
        raise requests.HTTPError(
            format_request_error(error, response.text),
            response=response
        ) from error

    data = response.json()

    choice = data["choices"][0]
    message = choice.get("message", {})
    usage = data.get("usage", {})

    parsed = parse_model_message(
        message,
        usage,
        reasoning_text_path=reasoning_text_path,
        reasoning_token_path=reasoning_token_path
    )

    return {
        "latency_seconds": latency_seconds,
        "raw_response_id": data.get("id"),
        "raw_model": data.get("model"),
        "finish_reason": choice.get("finish_reason"),
        **parsed
    }


# =======================
# MAIN EXPERIMENT RUNNER
# =======================

def run_experiment(
    config,
    questions: list[dict],
    conditions: dict,
    output_file: str | None = None,
    user_instruction: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    trials: int = 1,
    randomize_conditions: bool = True,
    reasoning_text_path: str | None = None,
    reasoning_token_path: str | None = None,
    resume_run: bool = False,
    save_csv: bool = True
) -> pd.DataFrame:
    """
    Run a full question × condition × trial experiment.

    Parameters:
        config:
            BackendConfig from config_utils.py.

        questions:
            List of question dictionaries from data_utils.py.
            Each question should include:
                question_id
                question_text
                category_1
                category_2
                raw_question_text

        conditions:
            Dictionary of condition_name -> system_prompt.
            Example:
                {
                    "honest": "Answer truthfully.",
                    "dishonest": "Give an intentionally wrong answer."
                }

        output_file:
            CSV file name. If None, auto-generate one.

        user_instruction:
            Optional extra instruction added to every user prompt.
            Example:
                "Answer the question directly."

        temperature:
            Model sampling temperature.

        max_tokens:
            Optional output cap.
            For natural CoT-token measurement, keep this as None.

        trials:
            Number of repeated runs per question-condition pair.

        randomize_conditions:
            If True, shuffle condition order for each question/trial.

        reasoning_text_path:
            Optional manual path for reasoning text inside the message object.

        reasoning_token_path:
            Optional manual path for reasoning-token count inside the usage object.

        resume_run:
            If True, continue from an existing matching results file instead of
            creating a new numbered file.

        save_csv:
            If True, save results to CSV.

    Returns:
        pandas DataFrame containing all results.
    """

    results = []
    resolved_output_file = None
    completed_run_keys = set()
    core_question_fields = {
        "question_id",
        "question_text",
        "category_1",
        "category_2",
        "raw_question_text"
    }

    if save_csv:
        if resume_run and output_file is not None and Path(output_file).exists():
            existing_df = pd.read_csv(output_file)

            if existing_results_match_run(existing_df, config, user_instruction, conditions):
                resolved_output_file = output_file
                results = existing_df.to_dict(orient="records")
                completed_run_keys = build_completed_run_keys(existing_df)
                print(
                    f"Resume mode enabled. Continuing in existing file: "
                    f"{resolved_output_file}"
                )
                print(f"Found {len(completed_run_keys)} completed runs to skip.")
            else:
                resolved_output_file = resolve_output_file_path(output_file)
                print(
                    f"Resume file does not match the current run setup. "
                    f"Saving new results to: {resolved_output_file}"
                )
        else:
            resolved_output_file = resolve_output_file_path(output_file)

            if output_file is not None and resolved_output_file != output_file:
                print(
                    f"Output file already exists. "
                    f"Saving new results to: {resolved_output_file}"
                )

    total_runs = len(questions) * len(conditions) * trials
    run_count = 0

    for trial_index in range(1, trials + 1):

        for question in questions:

            condition_items = list(conditions.items())

            if randomize_conditions:
                random.shuffle(condition_items)

            for condition_name, system_prompt in condition_items:
                run_count += 1

                question_id = question.get("question_id")
                question_text = question.get("question_text")
                category_1 = question.get("category_1")
                category_2 = question.get("category_2")
                raw_question_text = question.get("raw_question_text")
                extra_question_fields = {
                    key: value
                    for key, value in question.items()
                    if key not in core_question_fields
                }
                run_key = (trial_index, str(question_id), str(condition_name))

                if run_key in completed_run_keys:
                    print(
                        f"Skipping completed run | "
                        f"trial={trial_index} | "
                        f"question={question_id} | "
                        f"condition={condition_name}"
                    )
                    continue

                print(
                    f"Running {run_count}/{total_runs} | "
                    f"trial={trial_index} | "
                    f"question={question_id} | "
                    f"condition={condition_name}"
                )

                user_prompt = build_user_prompt(
                    question_text=question_text,
                    user_instruction=user_instruction
                )

                try:
                    model_result = call_model_once(
                        config=config,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        reasoning_text_path=reasoning_text_path,
                        reasoning_token_path=reasoning_token_path
                    )

                    result_row = {
                        "trial": trial_index,
                        "question_id": question_id,
                        "category_1": category_1,
                        "category_2": category_2,
                        "condition": condition_name,
                        **extra_question_fields,

                        "backend": config.backend_name,
                        "model": config.model,

                        "system_prompt": system_prompt,
                        "user_instruction": user_instruction,
                        "question_text": question_text,
                        "raw_question_text": raw_question_text,
                        "user_prompt": user_prompt,

                        "final_answer": model_result.get("final_answer"),
                        "reasoning_text": model_result.get("reasoning_text"),
                        "has_reasoning": model_result.get("has_reasoning"),

                        "prompt_tokens": model_result.get("prompt_tokens"),
                        "completion_tokens": model_result.get("completion_tokens"),
                        "reasoning_tokens": model_result.get("reasoning_tokens"),
                        "reasoning_tokens_source": model_result.get("reasoning_tokens_source"),
                        "total_tokens": model_result.get("total_tokens"),

                        "latency_seconds": model_result.get("latency_seconds"),
                        "finish_reason": model_result.get("finish_reason"),
                        "raw_model": model_result.get("raw_model"),
                        "raw_response_id": model_result.get("raw_response_id"),

                        "error": None
                    }

                    results.append(result_row)
                    completed_run_keys.add(run_key)

                    if save_csv and resolved_output_file is not None:
                        pd.DataFrame(results).to_csv(resolved_output_file, index=False)

                    print(
                        f"Done | reasoning_tokens={result_row['reasoning_tokens']} | "
                        f"completion_tokens={result_row['completion_tokens']}"
                    )

                except Exception as e:
                    error_row = {
                        "trial": trial_index,
                        "question_id": question_id,
                        "category_1": category_1,
                        "category_2": category_2,
                        "condition": condition_name,
                        **extra_question_fields,

                        "backend": config.backend_name,
                        "model": config.model,

                        "system_prompt": system_prompt,
                        "user_instruction": user_instruction,
                        "question_text": question_text,
                        "raw_question_text": raw_question_text,
                        "user_prompt": user_prompt,

                        "final_answer": None,
                        "reasoning_text": None,
                        "has_reasoning": None,

                        "prompt_tokens": None,
                        "completion_tokens": None,
                        "reasoning_tokens": None,
                        "reasoning_tokens_source": None,
                        "total_tokens": None,

                        "latency_seconds": None,
                        "finish_reason": None,

                        "error": str(e),

                        "raw_model": None,
                        "raw_response_id": None
                    }

                    results.append(error_row)

                    if save_csv and resolved_output_file is not None:
                        pd.DataFrame(results).to_csv(resolved_output_file, index=False)

                    print(f"Error: {e}")

    df_results = pd.DataFrame(results)

    if save_csv:
        if resolved_output_file is not None:
            df_results.to_csv(resolved_output_file, index=False)
            print(f"\nSaved results to: {resolved_output_file}")

    return df_results


# =======================
# QUICK TEST PLACEHOLDER
# =======================

if __name__ == "__main__":
    print(
        "model_runner.py is a library file.\n"
        "Use it from a run script, such as run_cot_token_test.py."
    )
