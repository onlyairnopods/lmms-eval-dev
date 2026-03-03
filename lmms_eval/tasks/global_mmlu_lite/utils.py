from lmms_eval.api.instance import Instance
from lmms_eval.protocol import ChatMessages


def global_mmlu_lite_doc_to_messages(doc, lmms_eval_specific_kwargs=None):
    """Convert a document to chat messages for chat models."""
    kwargs = lmms_eval_specific_kwargs or {}
    pre_prompt = kwargs.get("pre_prompt", "")
    post_prompt = kwargs.get("post_prompt", "")

    # Format the question with options
    question_text = pre_prompt + doc["question"] + "\n"
    question_text += f"A. {doc['option_a']}\n"
    question_text += f"B. {doc['option_b']}\n"
    question_text += f"C. {doc['option_c']}\n"
    question_text += f"D. {doc['option_d']}\n"
    question_text += post_prompt

    messages = [{"role": "user", "content": [{"type": "text", "text": question_text}]}]
    return messages


def global_mmlu_lite_process_results(doc, results):
    """Process the model results and compute accuracy."""
    pred = results[0].strip().upper()
    # Extract the answer letter (A, B, C, or D)
    answer = doc["answer"].strip().upper()

    # Try to find the answer letter in the prediction
    is_correct = False
    if answer in pred:
        is_correct = True
    # Also check if the prediction starts with the answer
    elif len(pred) > 0 and pred[0] in "ABCD":
        is_correct = pred[0] == answer

    return {"accuracy": 1.0 if is_correct else 0.0}


def global_mmlu_lite_aggregate_results(results):
    """Aggregate accuracy results."""
    return sum(results) / len(results) if results else 0.0
