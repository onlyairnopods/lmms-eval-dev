#!/usr/bin/env python3
"""
XChat LLM-as-Judge 批量评分脚本

使用 OpenAI API 对 XChat 任务的模型输出进行评分。

使用方法:
    python judge_xchat.py --input eval_outputs/samples_xchat_English.jsonl --output eval_outputs/scored.jsonl

环境变量:
    OPENAI_API_KEY: OpenAI API Key
    JUDGE_MODEL: 评分模型 (默认: gpt-4o)
"""

import argparse
import asyncio
import json
import os
import re
import sys
from typing import Any, Dict, List

from lmms_eval.llm_judge import ServerConfig, get_server


def parse_args():
    parser = argparse.ArgumentParser(description="XChat LLM-as-Judge 评分脚本")
    parser.add_argument("--input", "-i", required=True, help="输入文件 (模型输出 JSONL)")
    parser.add_argument("--output", "-o", required=True, help="输出文件 (评分结果 JSONL)")
    parser.add_argument(
        "--model",
        "-m",
        default=os.getenv("JUDGE_MODEL", "gpt-4o"),
        help="评分模型 (默认: gpt-4o)",
    )
    parser.add_argument(
        "--concurrent",
        "-c",
        type=int,
        default=5,
        help="并发数 (默认: 5)",
    )
    parser.add_argument(
        "--limit",
        "-n",
        type=int,
        default=None,
        help="限制评分的样本数量",
    )
    return parser.parse_args()


def load_samples(filepath: str, limit: int = None) -> List[Dict[str, Any]]:
    """加载模型输出样本"""
    samples = []
    with open(filepath, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Warning: Skip invalid JSON line {i + 1}")
    return samples


def build_xchat_prompt(doc: Dict[str, Any]) -> str:
    """构建 XChat 专用的评分提示词"""
    question = doc.get("input", "")
    reference = doc.get("reference_answer", "")
    rubric = doc.get("score_rubric", {})
    task = doc.get("task", "")

    prompt = f"""你是一个专业的视觉问答评分专家。请根据以下评分标准对模型回答进行评分。

任务类型: {task}
问题: {question}

参考答案: {reference}

模型回答: {doc.get('pred', '')}

评分标准: {rubric.get('criteria', '')}

评分选项:
"""

    for i in range(1, 6):
        desc = rubric.get(f"score{i}_description", "")
        if desc:
            prompt += f"- {i}分: {desc}\n"

    prompt += '\n请以以下JSON格式返回评分:\n{"score": <1-5的整数>, "reason": "<20字以内的评分理由>"}'

    return prompt


async def evaluate_single(judge, doc: Dict[str, Any]) -> Dict[str, Any]:
    """评估单个样本"""
    prompt = build_xchat_prompt(doc)

    result = await judge.evaluate_async(Request(messages=[{"role": "user", "content": prompt}]))

    # 解析结果
    try:
        json_match = re.search(r"\{[^}]+\}", result.content, re.DOTALL)
        if json_match:
            scores = json.loads(json_match.group())
            return {
                "doc_id": doc.get("doc_id", 0),
                "task": doc.get("task", ""),
                "prediction": doc.get("pred", ""),
                "reference": doc.get("reference_answer", ""),
                "score": scores.get("score", 0),
                "reason": scores.get("reason", "")[:50],  # 限制长度
            }
    except Exception as e:
        print(f"Parse error: {e}")

    return {
        "doc_id": doc.get("doc_id", 0),
        "task": doc.get("task", ""),
        "prediction": doc.get("pred", ""),
        "reference": doc.get("reference_answer", ""),
        "score": 0,
        "reason": "Parse failed",
        "raw_response": result.content[:200],
    }


async def batch_evaluate(samples: List[Dict[str, Any]], model: str, concurrent: int):
    """批量评估样本"""
    config = ServerConfig(
        model_name=model,
        temperature=0.0,
        max_tokens=2048,
        max_concurrent=concurrent,
    )
    judge = get_server("async_openai", config)

    results = []
    total = len(samples)

    for i, doc in enumerate(samples):
        print(f"Evaluating {i + 1}/{total}...", end="\r")
        result = await evaluate_single(judge, doc)
        results.append(result)
        await asyncio.sleep(0.2)  # 避免请求过快

    print(f"\nCompleted {total} samples")
    return results


def main():
    args = parse_args()

    # 检查 API Key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        sys.exit(1)

    # 加载样本
    print(f"Loading samples from {args.input}...")
    samples = load_samples(args.input, args.limit)
    print(f"Loaded {len(samples)} samples")

    if not samples:
        print("No samples to evaluate")
        sys.exit(1)

    # 批量评估
    print(f"Evaluating with {args.model}...")
    results = asyncio.run(batch_evaluate(samples, args.model, args.concurrent))

    # 保存结果
    with open(args.output, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Results saved to {args.output}")

    # 统计
    scores = [r["score"] for r in results if r["score"] > 0]
    if scores:
        avg = sum(scores) / len(scores)
        print(f"\nStatistics:")
        print(f"  Total samples: {len(results)}")
        print(f"  Valid scores: {len(scores)}")
        print(f"  Average score: {avg:.2f} / 5.0")
        print(f"  Pass rate (>=3): {len([s for s in scores if s >= 3]) / len(scores) * 100:.1f}%")


if __name__ == "__main__":
    main()
