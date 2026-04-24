ROLE_PREFIX = {
    "none": "",
    "child": "As a child, reason about the following.\n",
    "teen": "As a teenager, reason about the following.\n",
    "adult": "As an adult, reason about the following.\n",
}

OPTIONS = ["Yes", "No", "Cannot determine"]

def build_prompt(item: dict, conclusion: dict) -> str:
    """
    Converts data into a natural-language prompt.

    Args:
        item (dict): A single record from the processed dataset (with premises, role, etc.).
        conclusion (dict): One conclusion entry with at least a "question" key.

    Returns:
        str: A text prompt ready for model input.
    """
    role_text = ROLE_PREFIX.get(item["role"], "")
    premises = item["premises"]

    # Build multiple-choice part
    option_text = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(OPTIONS)])

    prompt = (
        f"{role_text}"
        f"Premise 1: {premises[0]}\n"
        f"Premise 2: {premises[1]}\n"
        f"Question: {conclusion['question']}\n"
        f"Options:\n{option_text}\n"
        f"Answer with the letter only."
    )
    return prompt
