"""Generate and evaluate KG path select-then-answer trajectories.

The script consumes candidate path sets produced by
``retrieve/sample_candidate_paths.py``.  It supports OpenAI-compatible chat
completion endpoints, including vLLM's OpenAI server, and records raw outputs,
parsed selections, answer metrics, and accept/reject decisions.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import string
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import openai


SELECT_SYS_PROMPT = """# Role
You are an expert in knowledge graph reasoning, evidence path selection, and faithful answer generation.

# Task Description
Given a question and a list of retrieved candidate knowledge-graph paths, first select the path indices that are necessary to answer the question, then generate the final answer using only the selected paths.

# Reasoning Requirements
- Understand the semantics of the question and each candidate path.
- Select only paths that provide necessary evidence for the final answer.
- During reasoning, enclose the selected path indices within <select> tags, e.g., <select>0,3,5</select>.
- The indices inside <select> refer to candidate path indices, not triple indices.
- Use only the selected paths to derive the final answer; do not answer from unsupported external knowledge.
- If the candidate paths are insufficient, output ans: not available.
- Do not select all paths unless every selected path is necessary.

# Output Format
Return:
1. A concise step-by-step reasoning process.
2. Exactly one <select>...</select> block containing selected path indices.
3. Final answers as a list, each line starting with "ans:".
"""


def normalize(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"\b(<pad>)\b", " ", text)
    return " ".join(text.split())


def is_match(prediction: str, answer: str) -> bool:
    pred_norm = normalize(prediction)
    ans_norm = normalize(answer)
    return bool(ans_norm) and ans_norm in pred_norm


def parse_answers(text: str) -> List[str]:
    answers = []
    for line in text.splitlines():
        if "ans:" not in line.lower():
            continue
        answer = line.strip()
        lowered = answer.lower()
        if "ans: not available" in lowered or "ans: no information available" in lowered:
            continue
        if answer not in answers:
            answers.append(answer)
    return answers


def parse_select_indices(text: str) -> Tuple[List[int], Optional[str]]:
    blocks = re.findall(r"<select>(.*?)</select>", text, flags=re.IGNORECASE | re.DOTALL)
    if len(blocks) != 1:
        return [], f"expected exactly one <select> block, found {len(blocks)}"
    content = blocks[0].strip()
    if not content or content.lower() in {"none", "n/a", "na"}:
        return [], None
    indices = []
    for piece in re.split(r"[,;\s]+", content):
        if not piece:
            continue
        if not re.fullmatch(r"\d+", piece):
            return [], f"non-integer selected index: {piece}"
        idx = int(piece)
        if idx not in indices:
            indices.append(idx)
    return indices, None


def compute_prf(predictions: Sequence[str], answers: Sequence[str]) -> Tuple[float, float, float, int]:
    if not predictions or not answers:
        return 0.0, 0.0, 0.0, 0
    unused_predictions = list(predictions)
    matched = 0
    for answer in answers:
        for prediction in list(unused_predictions):
            if is_match(prediction, answer) or is_match(answer, prediction.split("ans:")[-1].strip()):
                matched += 1
                unused_predictions.remove(prediction)
                break
    precision = matched / len(predictions) if predictions else 0.0
    recall = matched / len(answers) if answers else 0.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1, matched


def compute_hit(predictions: Sequence[str], answers: Sequence[str]) -> int:
    for answer in answers:
        for prediction in predictions:
            if is_match(prediction, answer) or is_match(answer, prediction.split("ans:")[-1].strip()):
                return 1
    return 0


def answer_in_selected_paths(predictions: Sequence[str], selected_paths: Sequence[Dict[str, Any]]) -> bool:
    if not predictions or not selected_paths:
        return False
    entities = []
    for path in selected_paths:
        for triple in path.get("triples", []):
            entities.append(str(triple.get("h", "")))
            entities.append(str(triple.get("t", "")))
    haystack = "\n".join(entities)
    return any(is_match(haystack, pred.split("ans:")[-1].strip()) for pred in predictions)


def evaluate_trajectory(
    response_text: str,
    gold_answers: Sequence[str],
    candidate_paths: Sequence[Dict[str, Any]],
    f1_threshold: float,
    require_hit: bool,
) -> Dict[str, Any]:
    selected_indices, select_error = parse_select_indices(response_text)
    answers = parse_answers(response_text)
    indices_ok = select_error is None and all(0 <= idx < len(candidate_paths) for idx in selected_indices)
    selected_paths = [candidate_paths[idx] for idx in selected_indices] if indices_ok else []
    precision, recall, f1, matched = compute_prf(answers, gold_answers)
    hit = compute_hit(answers, gold_answers)
    not_available = "ans: not available" in response_text.lower()
    answer_parse_ok = bool(answers) or not_available
    select_ok = indices_ok and (bool(selected_indices) or not_available)
    support_flag = answer_in_selected_paths(answers, selected_paths)
    accepted = bool(
        select_error is None
        and indices_ok
        and answer_parse_ok
        and (hit == 1 or not require_hit)
        and f1 >= f1_threshold
    )
    reject_reasons = []
    if select_error is not None:
        reject_reasons.append(select_error)
    if not indices_ok:
        reject_reasons.append("selected index out of range")
    if not answer_parse_ok:
        reject_reasons.append("no parseable ans: line")
    if require_hit and hit != 1:
        reject_reasons.append("hit != 1")
    if f1 < f1_threshold:
        reject_reasons.append(f"f1 < {f1_threshold}")
    return {
        "selected_path_indices": selected_indices,
        "answers": answers,
        "eval": {
            "hit": hit,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "matched": matched,
            "gold_count": len(gold_answers),
            "prediction_count": len(answers),
        },
        "validation": {
            "select_ok": select_ok,
            "indices_ok": indices_ok,
            "answer_parse_ok": answer_parse_ok,
            "answer_in_selected_paths": support_flag,
            "accepted": accepted,
            "reject_reasons": reject_reasons,
        },
    }


def format_path(path: Dict[str, Any]) -> str:
    lines = [f"[Path {path.get('path_index')} | score={path.get('path_score'):.6f} | method={path.get('method')}]" ]
    triples = path.get("triples", [])
    for hop_idx, triple in enumerate(triples):
        lines.append(
            f"({path.get('path_index')}.{hop_idx}) "
            f"{triple.get('h')} --{triple.get('r')}--> {triple.get('t')}"
        )
    return "\n".join(lines)


def build_user_prompt(record: Dict[str, Any], max_paths: Optional[int]) -> str:
    paths = record.get("candidate_paths", [])
    if max_paths is not None:
        paths = paths[:max_paths]
    path_block = "\n\n".join(format_path(path) for path in paths)
    # The [PATHS]/[QUERY] framing follows S-Path-RAG's prompt-mode fallback,
    # while preserving explicit path indices for select-then-generation.
    return (
        "[PATHS]\n"
        f"{path_block}\n\n"
        "[QUERY]\n"
        f"{record['question']}\n\n"
        "Please reason step by step, select useful path indices with <select>...</select>, "
        "and then provide final answers using ans: lines."
    )


def chat_completion(
    base_url: str,
    api_key: str,
    model: str,
    messages: Sequence[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    timeout: int,
) -> Dict[str, Any]:
    client = openai.OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
    response = client.chat.completions.create(
        model=model,
        messages=list(messages),
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.model_dump()


def extract_message_text(api_response: Dict[str, Any]) -> str:
    choices = api_response.get("choices", [])
    if not choices:
        return ""
    message = choices[0].get("message", {})
    return message.get("content", "") or choices[0].get("text", "") or ""


def iter_records(path: Path, split: Optional[str]) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fin:
        for line in fin:
            if not line.strip():
                continue
            record = json.loads(line)
            if split is not None and record.get("split") != split:
                continue
            yield record


def main(args: argparse.Namespace) -> None:
    input_path = Path(args.candidate_paths)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    accepted_path = Path(args.accepted_output) if args.accepted_output else output_path.with_suffix(".accepted.jsonl")
    rejected_path = Path(args.rejected_output) if args.rejected_output else output_path.with_suffix(".rejected.jsonl")
    sft_path = Path(args.sft_output) if args.sft_output else output_path.with_suffix(".sft.jsonl")
    accepted_path.parent.mkdir(parents=True, exist_ok=True)
    rejected_path.parent.mkdir(parents=True, exist_ok=True)
    sft_path.parent.mkdir(parents=True, exist_ok=True)

    if args.provider == "vllm":
        raise NotImplementedError("vLLM provider support is not implemented yet")

    api_key = args.api_key or os.environ.get(args.api_key_env, "")
    processed = accepted = rejected = 0
    with output_path.open("w", encoding="utf-8") as raw_fout, accepted_path.open("w", encoding="utf-8") as acc_fout, rejected_path.open("w", encoding="utf-8") as rej_fout, sft_path.open("w", encoding="utf-8") as sft_fout:
        for record in iter_records(input_path, args.split):
            if args.limit is not None and processed >= args.limit:
                break
            processed += 1
            candidate_paths = record.get("candidate_paths", [])
            if args.max_paths is not None:
                candidate_paths = candidate_paths[: args.max_paths]
            system_prompt = args.system_prompt or SELECT_SYS_PROMPT
            user_prompt = build_user_prompt(record, args.max_paths)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            attempts = []
            accepted_attempt = None
            for attempt_idx in range(args.max_retries):
                started = time.time()
                api_response: Dict[str, Any]
                response_text = ""
                api_error = None
                try:
                    api_response = chat_completion(
                        base_url=args.base_url,
                        api_key=api_key,
                        model=args.model,
                        messages=messages,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                        timeout=args.timeout,
                    )
                    response_text = extract_message_text(api_response)
                except (openai.OpenAIError, TimeoutError, json.JSONDecodeError) as exc:
                    api_response = {}
                    api_error = str(exc)

                parsed = evaluate_trajectory(
                    response_text,
                    gold_answers=record.get("gold_answers", []),
                    candidate_paths=candidate_paths,
                    f1_threshold=args.f1_threshold,
                    require_hit=args.require_hit,
                )
                attempt = {
                    "attempt": attempt_idx,
                    "raw_response": response_text,
                    "api_error": api_error,
                    "api_response": api_response if args.save_api_response else None,
                    "runtime_sec": time.time() - started,
                    **parsed,
                }
                attempts.append(attempt)
                if parsed["validation"]["accepted"]:
                    accepted_attempt = attempt_idx
                    break

            final_attempt = attempts[accepted_attempt] if accepted_attempt is not None else attempts[-1]
            out_record = {
                **record,
                "candidate_paths": candidate_paths,
                "prompt": {"system": system_prompt, "user": user_prompt},
                "llm_config": {
                    "provider": args.provider,
                    "model": args.model,
                    "base_url": args.base_url,
                    "temperature": args.temperature,
                    "max_tokens": args.max_tokens,
                },
                "attempts": attempts,
                "accepted": accepted_attempt is not None,
                "accepted_attempt": accepted_attempt,
                "final_eval": final_attempt.get("eval", {}),
                "final_validation": final_attempt.get("validation", {}),
            }
            raw_fout.write(json.dumps(out_record, ensure_ascii=False) + "\n")
            if accepted_attempt is not None:
                accepted += 1
                acc_fout.write(json.dumps(out_record, ensure_ascii=False) + "\n")
                sft_fout.write(json.dumps({
                    "messages": messages + [{"role": "assistant", "content": final_attempt["raw_response"]}],
                    "metadata": {
                        "trajectory_id": record.get("trajectory_id"),
                        "id": record.get("id"),
                        "path_set_id": record.get("path_set_id"),
                        "policy": record.get("policy"),
                        **final_attempt.get("eval", {}),
                    },
                }, ensure_ascii=False) + "\n")
            else:
                rejected += 1
                rej_fout.write(json.dumps(out_record, ensure_ascii=False) + "\n")

    print(f"Processed {processed} path sets; accepted={accepted}; rejected={rejected}")
    print(f"Raw trajectories: {output_path}")
    print(f"Accepted trajectories: {accepted_path}")
    print(f"Rejected trajectories: {rejected_path}")
    print(f"SFT messages: {sft_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-paths", required=True, help="Input candidate path-set JSONL")
    parser.add_argument("--output", required=True, help="Raw trajectory JSONL output")
    parser.add_argument("--accepted-output", default=None)
    parser.add_argument("--rejected-output", default=None)
    parser.add_argument("--sft-output", default=None)
    parser.add_argument("--split", default=None, help="Optional split filter")
    parser.add_argument("--provider", choices=["openai", "vllm"], default="openai")
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="https://api.openai.com/v1")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--max-paths", type=int, default=None)
    parser.add_argument("--f1-threshold", type=float, default=0.8)
    parser.add_argument("--require-hit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument("--save-api-response", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
