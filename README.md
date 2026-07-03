# Token Spike Project

This repository contains the code used for experiments on token consumption and non-truthful behavior in large language models.

Dataset files and experiment result files are not included in this GitHub repository. They will be shared separately through OSF for reproducibility and manuscript reference.

## Repository structure

- `run_experiment.py` — main experiment driver
- `lm_token_tools/` — reusable toolkit code
- `model_parameters.txt` — notes on model parameter settings by platform

## Experiment logic

- Each question-condition-trial combination is submitted as one separate API request.
- No chat history is carried across runs.
- The experiment driver defines the backend or model, condition prompts, user instruction, number of trials, and output settings.
- The toolkit saves results during the run and avoids overwriting an existing output file by creating a numbered file name when needed.
- The toolkit can also resume a matching interrupted run from an existing results file.

## Code files

- `run_experiment.py`
  - Defines the experiment configuration, including backend choice, question source, condition prompts, user instruction, number of trials, and output file name.

- `lm_token_tools/config_utils.py`
  - Prepares backend settings for either a local LM Studio model or an OpenAI-compatible API.
  - Builds the chat endpoint, request headers, model name, and timeout.
  - Can auto-detect the first available local model from LM Studio.

- `lm_token_tools/data_utils.py`
  - Loads question data from CSV, pasted text, or a structured dictionary list.
  - Supports row slicing, optional removal of multiple-choice options, and carry-through of extra metadata columns.
  - Stores both `question_text` and `raw_question_text` so that cleaned text and original text can both be retained.

- `lm_token_tools/prompt_utils.py`
  - Builds the final user prompt by combining the question text with an optional user instruction.
  - Parses returned responses into `final_answer` and `reasoning_text`.
  - Handles several reasoning formats, including separate `reasoning_content`, separate `reasoning` fields, and `<think>...</think>` blocks.
  - Extracts token fields such as `prompt_tokens`, `completion_tokens`, `reasoning_tokens`, `reasoning_tokens_source`, and `total_tokens`.

- `lm_token_tools/model_runner.py`
  - Iterates through all questions, conditions, and trials and sends one independent request for each run.
  - Records backend and prompt information for each row, including `system_prompt`, `user_instruction`, and `user_prompt`.
  - Saves returned outputs such as `final_answer`, `reasoning_text`, `has_reasoning`, `prompt_tokens`, `completion_tokens`, `reasoning_tokens`, `total_tokens`, `latency_seconds`, `finish_reason`, `raw_model`, `raw_response_id`, and `error`.
  - Saves partial results during the run, supports resume mode, and stores readable error messages when a request fails.


## Project information

- Developed by Vermut BH. Gao
- LUCID, Chapman University Brain Lab
- Lab page: [https://chapmanbrain.org](https://chapmanbrain.org)

## Installation

```bash
pip install -r requirements.txt
```

## Run

```bash
python run_experiment.py
```

## Notes

- API keys are not included in this repository.
- Some script settings, such as model endpoints, file paths, and output file names, should be adjusted locally before running.
- Data and result files will be provided separately through OSF.
