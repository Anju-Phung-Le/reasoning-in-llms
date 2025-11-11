ROLE_PREFIX = {
    "none": "",
    "child": "As a child, reason about the following.\n",
    "teen": "As a teenager, reason about the following.\n",
    "adult": "As an adult, reason about the following.\n",
}

def build_prompt(item: dict) -> str:
    """
    Converts data into a natural-language prompt. Sets the premises to item["premises"], etc. 

    Args:
        item from data.py (dict): A single example from the processed dataset.

    Returns:
        str: A text prompt ready for model input.
    """
    role_text = ROLE_PREFIX.get(item["role"], "")
    premises = item["premises"]
    options = item["options"]

    # Build multiple-choice part, add indexes to options (e.g. 0,"Yes", 1"No", 2"Cannot determine")
    option_text = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)]) # chr(65+i) converts int to ASCII character, 0:A, 1:B, 2:C

    prompt = (
        f"{role_text}"
        f"Premise 1: {premises[0]}\n"
        f"Premise 2: {premises[1]}\n"
        f"Question: {item['question']}\n"
        f"Options:\n{option_text}\n"
        f"Answer with the letter only."
    )
    return prompt
