import json
import os
from typing import List, Dict, Any, Callable
from PIL import Image
import datasets


# Base path for xchat data
XCHAT_DATA_BASE = os.path.join(os.path.dirname(__file__))


def xchat_load_dataset(data_files: List[str], test_split: str = "train") -> datasets.DatasetDict:
    """Load xchat dataset from local JSON files and add _data_dir to each doc.

    Args:
        data_files: List of JSON file paths
        test_split: The split to use (default: train)

    Returns:
        HuggingFace DatasetDict with _data_dir field added to each doc
    """
    # Extract base directory from the first data file
    # Format: lmms_eval/tasks/xchat/English/art_explanation/data.json
    first_file = data_files[0]
    first_file_dir = os.path.dirname(first_file)  # lmms_eval/tasks/xchat/English/art_explanation
    # Get language and task from path
    # lmms_eval/tasks/xchat/English/art_explanation -> English, art_explanation
    rel_path = os.path.relpath(first_file_dir, XCHAT_DATA_BASE)  # English/art_explanation
    parts = rel_path.split(os.sep)
    if len(parts) >= 2:
        language = parts[0]
        task = parts[1]
        data_dir_base = os.path.join(XCHAT_DATA_BASE, language, task)
    else:
        data_dir_base = first_file_dir

    # Load dataset using HuggingFace datasets
    ds = datasets.load_dataset(
        "json",
        data_files=data_files,
        split=test_split,
    )

    # Add _data_dir to each doc
    def add_data_dir(example):
        # Get the source file for this example
        # Since we can't get the exact file, we infer from the task field
        task_name = example.get("task", "")
        if task_name:
            example["_data_dir"] = os.path.join(XCHAT_DATA_BASE, language, task_name)
        else:
            example["_data_dir"] = data_dir_base
        return example

    ds = ds.map(add_data_dir)
    return datasets.DatasetDict({test_split: ds})


def xchat_process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    """Add _data_dir field to each document in the dataset.

    This function is called by the framework to preprocess the dataset
    before inference.

    Args:
        dataset: HuggingFace Dataset

    Returns:
        Dataset with _data_dir field added
    """
    # Try to infer language from the first document's _data_dir if it exists
    # Otherwise, we need to get it from the config or infer from task field
    # Since process_docs only receives the dataset, we need another approach

    # Get sample to check if _data_dir already exists
    if len(dataset) > 0:
        sample = dataset[0]
        if "_data_dir" in sample:
            return dataset

    # Try to infer from task field in the dataset
    # We'll add _data_dir based on the task field
    def add_data_dir(example):
        task = example.get("task", "")
        if task:
            # Try to find the language directory
            # This is a workaround - ideally we'd get language from config
            for lang in ["English", "Chinese", "Hindi", "Indonesian", "Japanese", "Kinyarwanda", "Korean", "Spanish"]:
                potential_dir = os.path.join(XCHAT_DATA_BASE, lang, task)
                if os.path.exists(potential_dir):
                    example["_data_dir"] = potential_dir
                    break
        return example

    return dataset.map(add_data_dir)


def get_all_categories(lang: str) -> List[str]:
    """Get all categories for a given language."""
    lang_path = os.path.join(XCHAT_DATA_BASE, lang)
    if not os.path.exists(lang_path):
        return []
    categories = []
    for item in os.listdir(lang_path):
        item_path = os.path.join(lang_path, item)
        if os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, "data.json")):
            categories.append(item)
    return sorted(categories)


def get_data_dir_for_category(lang: str, category: str) -> str:
    """Get the directory path for a specific category."""
    return os.path.join(XCHAT_DATA_BASE, lang, category)


def xchat_doc_to_visual(doc: Dict[str, Any]) -> List[Image.Image]:
    """Extract image from document.

    The dataset includes a '_data_dir' field that we add during processing.
    """
    # Get the image directory from the doc (added by dataset preprocessing)
    data_dir = doc.get("_data_dir", "")

    if not data_dir:
        # Fallback: try to infer from task field
        task = doc.get("task", "")
        if task:
            # Try to find the language directory by checking common locations
            for lang in ["English", "Chinese", "Hindi", "Indonesian", "Japanese", "Kinyarwanda", "Korean", "Spanish"]:
                potential_dir = os.path.join(XCHAT_DATA_BASE, lang, task)
                if os.path.exists(potential_dir):
                    data_dir = potential_dir
                    break

    if not data_dir:
        return []

    instance_idx = doc.get("instance_idx", 0)

    # Try different image extensions
    for ext in [".jpg", ".png", ".jpeg"]:
        img_path = os.path.join(data_dir, f"{instance_idx}{ext}")
        if os.path.exists(img_path):
            return [Image.open(img_path).convert("RGB")]

    # If no image found, return empty list
    return []


def xchat_doc_to_text(doc: Dict[str, Any], lmms_eval_specific_kwargs: Dict = None) -> str:
    """Convert document to text prompt (for simple models)."""
    system_prompt = doc.get("system_prompt", "")
    user_input = doc.get("input", "")

    # Combine system prompt and user input
    if system_prompt:
        return f"{system_prompt}\n\n{user_input}"
    return user_input


def xchat_doc_to_messages(doc: Dict[str, Any], lmms_eval_specific_kwargs: Dict = None) -> List[Dict[str, Any]]:
    """Convert document to chat messages format (for chat models).

    This function is called by the framework and should return a list of messages
    in the format expected by ChatMessages protocol.

    The return format should be:
    [
        {"role": "system", "content": [{"type": "text", "text": "..."}]},
        {"role": "user", "content": [
            {"type": "image", "url": pil_image},
            {"type": "text", "text": "..."}
        ]}
    ]
    """
    system_prompt = doc.get("system_prompt", "")
    user_input = doc.get("input", "")

    messages = []

    # Add system message if present
    if system_prompt:
        messages.append({
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}]
        })

    # Get images
    images = xchat_doc_to_visual(doc)

    # Build user message with text and images
    content = []
    for img in images:
        # For chat models, images are added as PIL Image objects
        content.append({"type": "image", "url": img})

    content.append({"type": "text", "text": user_input})

    messages.append({"role": "user", "content": content})

    return messages


def xchat_process_results(doc: Dict[str, Any], results: List[str]) -> Dict[str, Any]:
    """Process model results for xchat."""
    pred = results[0] if results else ""
    reference = doc.get("reference_answer", "")
    score_rubric = doc.get("score_rubric", {})
    task = doc.get("task", "")
    instance_idx = doc.get("instance_idx", 0)

    return {
        "pred": pred,
        "reference": reference,
        "score_rubric": score_rubric,
        "task": task,
        "instance_idx": instance_idx,
    }


def xchat_aggregate_results(results: List[Dict[str, Any]], args: Any) -> Dict[str, float]:
    """Aggregate results for xchat.

    Since xchat uses open-ended generation with LLM-as-judge evaluation,
    we simply save the results for manual evaluation or use an LLM judge.

    This function should compute metrics across all results.
    For now, we just return a placeholder since this requires LLM-as-judge.
    """
    # Since this is open-ended generation, we can't compute automatic metrics
    # The actual evaluation should be done with LLM-as-judge
    # For now, we'll save the predictions and let users evaluate manually
    return {"accuracy": 0.0}


# Language code mapping
LANGUAGE_CODES = {
    "English": "en",
    "Chinese": "zh",
    "Hindi": "hi",
    "Indonesian": "id",
    "Japanese": "ja",
    "Kinyarwanda": "rw",
    "Korean": "ko",
    "Spanish": "es",
}
