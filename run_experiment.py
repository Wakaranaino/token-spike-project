"""
run_cot_token_test.py

Main experiment script for CoT / reasoning-token analysis.

This is the file most users should edit.

It uses the toolkit files in lm_token_tools/ to:
1. load model/backend settings
2. load question data
3. define experimental conditions
4. run one independent request per question × condition × trial
5. save results to CSV

Main outcome:
- reasoning_tokens
- completion_tokens
- final_answer
- reasoning_text
"""

from lm_token_tools.config_utils import get_backend_config
from lm_token_tools.data_utils import prepare_questions
from lm_token_tools.model_runner import run_experiment


# =======================
# 1. BACKEND / MODEL SETUP
# =======================

# Choose one backend:
# "local" = LM Studio local model
# "api"   = OpenAI-compatible cloud/server API
BACKEND = "local"


# ----- Option A: Local LM Studio -----
# Make sure LM Studio server is running.
# The model is auto-detected by default.
if BACKEND == "local":
    config = get_backend_config(
        backend="local",
        local_base_url="http://localhost:1234"
    )


# ----- Option B1: Azure-style API -----
# To use this, set BACKEND = "api" above and fill in the values below.
"""
if BACKEND == "api":
    config = get_backend_config(
        backend="api",
        api_base_url="i",
        api_model="",
        api_key="",
        auth_type="api-key"
    )
"""

# ----- Option B2: Groq / bearer-token API -----
# Use this block instead of Option B1 if you want a Groq-style API.

if BACKEND == "api":
    config = get_backend_config(
        backend="api",
        api_base_url="https://api.groq.com/openai",
        api_model="openai/gpt-oss-120b",
        api_key="",
        auth_type="bearer"
    )



print("Backend loaded:")
print(f"Backend: {config.backend_name}")
print(f"Model: {config.model}")
print(f"Chat URL: {config.chat_url}")


# =======================
# 2. QUESTION DATA SETUP
# =======================

# ----- Option A: Load questions from CSV -----
questions = prepare_questions(
    source="Data structure template.csv",
    source_type="csv",
    question_col="Prompt Text",
    id_col="Prompt ID",
    category_1_col="Top Level Category",
    category_2_col="Second Level Category",
    extra_cols=["Correct Answer Short"],

    # Default is full file.
    start_row=0,
    end_row= None,

    # Keep answer options by default.
    # Change to True only if you intentionally want to remove options.
    remove_options_flag=False
)

# ----- Option B: Manual pasted questions -----
# To use manual questions, comment out the CSV block above
# and uncomment this block.
"""
questions = prepare_questions(
    source=\"\"\"
    Are whales warm-blooded?
    (A) Yes
    (B) No
    (C) Only sometimes
    (D) Cannot be determined
    \"\"\",
    source_type="text",
    remove_options_flag=False
)
"""


print(f"Loaded questions: {len(questions)}")


# =======================
# 3. EXPERIMENT CONDITIONS
# =======================

# Conditions are system prompts.
# Each condition is tested separately for every question.
conditions = {
    "system": "You are completing a multiple-choice benchmark task."
}


# =======================
# 4. USER INSTRUCTION
# =======================

# This is added after every question as part of the user prompt.
# Keep None if you want to send the original question as-is.
USER_INSTRUCTION = "Select the option that best answers the question. Return only one letter: A, B, C, or D. Do not provide explanation."

# Example:
# USER_INSTRUCTION = "Answer the question directly."


# =======================
# 5. RUN SETTINGS
# =======================

TEMPERATURE = None

# For CoT-token measurement, keep this as None.
# Do not use small max_tokens, because that can cut off reasoning.
MAX_TOKENS = None

# Optional manual override paths for models that store reasoning fields in a provider-specific place.
# Keep these as None unless the inspection script shows a better path.
#
# Examples:
# REASONING_TEXT_PATH = "reasoning"
# REASONING_TOKEN_PATH = "output_tokens_details.reasoning_tokens"
REASONING_TEXT_PATH = None
REASONING_TOKEN_PATH = None

TRIALS = 1

RANDOMIZE_CONDITIONS = True

# If True, continue in the same output file when the existing file matches
# the same backend, model, system prompts, and user instruction.
RESUME_RUN = True

# Results are written to the CSV after each completed question-condition run.
# If the target file name already exists, the script saves to a new numbered file
# such as ..._1.csv or ..._2.csv instead of overwriting the existing file.
OUTPUT_FILE = "cot_token_test_results_gemma-4-e4b.csv"



# =======================
# 6. RUN EXPERIMENT
# =======================

df_results = run_experiment(
    config=config,
    questions=questions,
    conditions=conditions,
    output_file=OUTPUT_FILE,
    user_instruction=USER_INSTRUCTION,
    temperature=TEMPERATURE,
    max_tokens=MAX_TOKENS,
    trials=TRIALS,
    randomize_conditions=RANDOMIZE_CONDITIONS,
    reasoning_text_path=REASONING_TEXT_PATH,
    reasoning_token_path=REASONING_TOKEN_PATH,
    resume_run=RESUME_RUN,
    save_csv=True
)


# =======================
# 7. QUICK SUMMARY
# =======================

print("\nQuick summary:")
print(df_results[[
    "question_id",
    "condition",
    "final_answer",
    "reasoning_tokens",
    "completion_tokens",
    "total_tokens",
    "error"
]].head())

print(f"\nSaved to: {OUTPUT_FILE}")
