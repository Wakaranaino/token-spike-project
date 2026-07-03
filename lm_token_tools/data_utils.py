"""
data_utils.py

Question data utilities for LM token / CoT-token experiments.

This file prepares question data only:
- load questions from CSV
- use full file or selected row range
- optionally remove multiple-choice options
- create questions from pasted text
- create questions from advanced dictionary lists

It does NOT handle:
- model setup
- API calls
- system prompts / conditions
- CoT extraction
"""

import re
import pandas as pd


# =======================
# OPTION REMOVAL
# =======================

def remove_options(text: str) -> str:
    """
    Optionally remove multiple-choice options from a prompt.

    Examples removed:
    - Options: (A) ... (B) ...
    - Here are the options: A. ... B. ...
    - Choices: ...
    - (A) ...
    - A. ... / A) ...

    This is heuristic, so keep remove_options_flag=False unless needed.
    """

    text = str(text).strip().strip('"')

    patterns = [
        r"\n?\s*(Here are the\s+)?Options?:.*$",
        r"\n?\s*(Here are the\s+)?Choices?:.*$",
        r"\n?\s*\(A\).*$",
        r"\n?\s*A[\.\)].*$",
    ]

    for pattern in patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL)

    return text.strip()


# =======================
# ROW SLICING
# =======================

def slice_rows(df: pd.DataFrame, start_row: int = 0, end_row: int | None = None) -> pd.DataFrame:
    """
    Select a row range.

    Default:
        start_row=0, end_row=None
    means:
        use the full file.
    """

    if end_row is None:
        return df.iloc[start_row:].copy()

    return df.iloc[start_row:end_row].copy()


# =======================
# CSV LOADING
# =======================

def load_questions_from_csv(
    file_path: str,
    question_col: str,
    id_col: str | None = None,
    category_1_col: str | None = None,
    category_2_col: str | None = None,
    extra_cols: list[str] | None = None,
    start_row: int = 0,
    end_row: int | None = None,
    remove_options_flag: bool = False
) -> list[dict]:
    """
    Load question data from CSV.

    Required:
        file_path
        question_col

    Optional:
        id_col
        category_1_col
        category_2_col
        extra_cols
        start_row / end_row
        remove_options_flag

    Output format:
        [
            {
                "question_id": "...",
                "question_text": "...",
                "category_1": "...",
                "category_2": "...",
                "raw_question_text": "..."
            }
        ]
    """

    df = pd.read_csv(file_path)
    df = slice_rows(df, start_row=start_row, end_row=end_row)

    if question_col not in df.columns:
        raise ValueError(f"question_col '{question_col}' not found in CSV columns: {list(df.columns)}")

    extra_cols = extra_cols or []

    missing_extra_cols = [col for col in extra_cols if col not in df.columns]
    if missing_extra_cols:
        raise ValueError(
            f"extra_cols not found in CSV columns: {missing_extra_cols}. "
            f"Available columns: {list(df.columns)}"
        )

    questions = []

    for idx, row in df.iterrows():
        raw_text = row[question_col]
        question_text = remove_options(raw_text) if remove_options_flag else str(raw_text).strip()

        if id_col and id_col in df.columns:
            question_id = row[id_col]
        else:
            question_id = f"Q{idx + 1}"

        if category_1_col and category_1_col in df.columns:
            category_1 = row[category_1_col]
        else:
            category_1 = None

        if category_2_col and category_2_col in df.columns:
            category_2 = row[category_2_col]
        else:
            category_2 = None

        extra_data = {col: row[col] for col in extra_cols}

        question_record = {
            "question_id": question_id,
            "question_text": question_text,
            "category_1": category_1,
            "category_2": category_2,
            "raw_question_text": raw_text
        }
        question_record.update(extra_data)

        questions.append(question_record)

    return questions


# =======================
# MANUAL PASTED TEXT
# =======================

def questions_from_text(
    text_block: str,
    split_by_blank_line: bool = True,
    remove_options_flag: bool = False
) -> list[dict]:
    """
    Create questions from pasted text.

    Typical input:
        '''
        Question one?

        Question two?

        Question three?
        '''

    Default behavior:
        split questions by blank lines.

    If split_by_blank_line=False:
        split by each non-empty line.
    """

    text_block = str(text_block).strip()

    if split_by_blank_line:
        raw_items = re.split(r"\n\s*\n", text_block)
    else:
        raw_items = text_block.splitlines()

    raw_items = [item.strip() for item in raw_items if item.strip()]

    questions = []

    for idx, raw_text in enumerate(raw_items, start=1):
        question_text = remove_options(raw_text) if remove_options_flag else raw_text

        questions.append({
            "question_id": f"Q{idx}",
            "question_text": question_text,
            "category_1": None,
            "category_2": None,
            "raw_question_text": raw_text
        })

    return questions


# =======================
# ADVANCED DICTIONARY LIST
# =======================

def questions_from_dict_list(
    question_list: list[dict],
    remove_options_flag: bool = False
) -> list[dict]:
    """
    Advanced format for users who want to pass structured question data.

    Example:
        [
            {
                "question_id": "Q1",
                "question_text": "What is ...?",
                "category_1": "ethics",
                "category_2": "case-a"
            }
        ]
    """

    questions = []

    for idx, item in enumerate(question_list, start=1):
        raw_text = item.get("question_text") or item.get("prompt") or item.get("question")

        if raw_text is None:
            raise ValueError(f"Item {idx} has no question_text, prompt, or question field.")

        question_text = remove_options(raw_text) if remove_options_flag else str(raw_text).strip()

        question_record = {
            "question_id": item.get("question_id", f"Q{idx}"),
            "question_text": question_text,
            "category_1": item.get("category_1", item.get("category")),
            "category_2": item.get("category_2"),
            "raw_question_text": raw_text
        }

        for key, value in item.items():
            if key not in {
                "question_id",
                "question_text",
                "prompt",
                "question",
                "category",
                "category_1",
                "category_2"
            }:
                question_record[key] = value

        questions.append(question_record)

    return questions


# =======================
# CONVENIENCE FUNCTION
# =======================

def prepare_questions(
    source,
    source_type: str = "csv",
    question_col: str | None = None,
    id_col: str | None = None,
    category_1_col: str | None = None,
    category_2_col: str | None = None,
    extra_cols: list[str] | None = None,
    start_row: int = 0,
    end_row: int | None = None,
    remove_options_flag: bool = False,
    split_by_blank_line: bool = True
) -> list[dict]:
    """
    One main function for preparing questions.

    source_type options:
        "csv"       -> source is a CSV file path
        "text"      -> source is pasted text
        "dict_list" -> source is a list of dictionaries

    Examples:

        questions = prepare_questions(
            source="Prompts_token_latency.csv",
            source_type="csv",
            question_col="Question Prompt",
            id_col="ID #",
            category_1_col="Category",
            category_2_col="Second Category",
            extra_cols=["Condition"]
        )

        questions = prepare_questions(
            source=\"\"\"
            Question one?

            Question two?
            \"\"\",
            source_type="text"
        )

        questions = prepare_questions(
            source=[
                {"question_id": "Q1", "question_text": "Question one?", "category": "test"}
            ],
            source_type="dict_list"
        )
    """

    source_type = source_type.lower().strip()

    if source_type == "csv":
        if question_col is None:
            raise ValueError("question_col is required when source_type='csv'.")

        return load_questions_from_csv(
            file_path=source,
            question_col=question_col,
            id_col=id_col,
            category_1_col=category_1_col,
            category_2_col=category_2_col,
            extra_cols=extra_cols,
            start_row=start_row,
            end_row=end_row,
            remove_options_flag=remove_options_flag
        )

    if source_type == "text":
        return questions_from_text(
            text_block=source,
            split_by_blank_line=split_by_blank_line,
            remove_options_flag=remove_options_flag
        )

    if source_type == "dict_list":
        return questions_from_dict_list(
            question_list=source,
            remove_options_flag=remove_options_flag
        )

    raise ValueError("source_type must be 'csv', 'text', or 'dict_list'.")


# =======================
# QUICK TEST
# =======================

if __name__ == "__main__":
    sample_text = """
    What is the capital of France?

    What is 2 + 2? Options: (A) 3 (B) 4 (C) 5
    """

    questions = prepare_questions(
        source=sample_text,
        source_type="text",
        remove_options_flag=True
    )

    print("Prepared questions:")
    for q in questions:
        print(q)
