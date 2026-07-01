"""Sample S-Path-RAG-style candidate path sets with SubgraphRAG MLP scores.

This script does not use answer entities to build paths.  Gold answers are only
copied to the output metadata so that downstream trajectory generation can run
quality filtering without leaking labels into retrieval or prompts.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
from tqdm import tqdm

# Keep compatibility with the existing retrieve scripts, which expect imports
# and data_files relative to the retrieve directory.
RETRIEVE_DIR = Path(__file__).resolve().parent
if str(RETRIEVE_DIR) not in sys.path:
    sys.path.insert(0, str(RETRIEVE_DIR))
os.chdir(RETRIEVE_DIR)

from spathrag.path_sampler import CandidatePathSampler  # noqa: E402


def score_sample(model: Any, raw_sample: Dict[str, Any], device: Any, collate_retriever: Any, prepare_sample: Any) -> List[float]:
    sample = collate_retriever([raw_sample])
    (
        h_id_tensor,
        r_id_tensor,
        t_id_tensor,
        q_emb,
        entity_embs,
        num_non_text_entities,
        relation_embs,
        topic_entity_one_hot,
        _target_triple_probs,
        _a_entity_id_list,
    ) = prepare_sample(device, sample)

    if len(h_id_tensor) == 0:
        return []
    logits = model(
        h_id_tensor,
        r_id_tensor,
        t_id_tensor,
        q_emb,
        entity_embs,
        num_non_text_entities,
        relation_embs,
        topic_entity_one_hot,
    )
    return logits.reshape(-1).detach().cpu().tolist()


def resolve_device(device_arg: str) -> Any:
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_arg)


def build_output_record(
    raw_sample: Dict[str, Any],
    path_set_id: int,
    policy: str,
    candidate_paths: List[Any],
    path_pool_size: int,
    args: argparse.Namespace,
    runtime_sec: float,
) -> Dict[str, Any]:
    entity_list = raw_sample["text_entity_list"] + raw_sample["non_text_entity_list"]
    q_entity_in_graph = [entity_list[e_id] for e_id in raw_sample.get("q_entity_id_list", [])]
    a_entity_in_graph = [entity_list[e_id] for e_id in raw_sample.get("a_entity_id_list", [])]
    sample_id = str(raw_sample["id"])
    return {
        "trajectory_id": f"{sample_id}::pathset_{path_set_id}",
        "id": sample_id,
        "dataset": args.dataset,
        "split": args.split,
        "path_set_id": path_set_id,
        "policy": policy,
        "question": raw_sample["question"],
        "q_entity": raw_sample.get("q_entity", []),
        "q_entity_in_graph": q_entity_in_graph,
        "gold_answers": raw_sample.get("a_entity", []),
        "a_entity_in_graph": a_entity_in_graph,
        "candidate_paths": [path.to_dict() for path in candidate_paths],
        "path_sampling_config": {
            "checkpoint": args.checkpoint,
            "path_pool_size": args.path_pool_size,
            "paths_per_set": args.paths_per_set,
            "num_path_sets": args.num_path_sets,
            "max_path_length": args.max_path_length,
            "beam_width": args.beam_width,
            "expand_top_k": args.expand_top_k,
            "num_random_walks": args.num_random_walks,
            "path_score_agg": args.path_score_agg,
            "length_penalty": args.length_penalty,
            "policies": args.policies,
            "uses_answer_entities_for_sampling": False,
        },
        "trace": {
            "num_triples": len(raw_sample.get("h_id_list", [])),
            "num_entities": len(entity_list),
            "num_paths_in_pool": path_pool_size,
            "num_paths_in_set": len(candidate_paths),
            "runtime_sec": runtime_sec,
        },
    }


def main(args: argparse.Namespace) -> None:
    from src.dataset.retriever import RetrieverDataset, collate_retriever
    from src.model.retriever import Retriever
    from src.setup import prepare_sample, set_seed

    device = resolve_device(args.device)
    cpt = torch.load(args.checkpoint, map_location="cpu")
    config = cpt["config"]
    if args.dataset is not None:
        config["dataset"]["name"] = args.dataset
    set_seed(config["env"]["seed"])
    torch.set_num_threads(config["env"].get("num_threads", 1))

    dataset = RetrieverDataset(config=config, split=args.split, skip_no_path=False)
    emb_size = dataset[0]["q_emb"].shape[-1]
    model = Retriever(emb_size, **config["retriever"]).to(device)
    model.load_state_dict(cpt["model_state_dict"])
    model.eval()

    policies = [policy.strip() for policy in args.policies.split(",") if policy.strip()]
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = (Path.cwd() / output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_records = 0
    with output_path.open("w", encoding="utf-8") as fout:
        limit = len(dataset) if args.limit is None else min(args.limit, len(dataset))
        for sample_idx in tqdm(range(limit), desc="Sampling candidate path sets"):
            start_time = time.time()
            raw_sample = dataset[sample_idx]
            with torch.no_grad():
                triple_logits = score_sample(model, raw_sample, device, collate_retriever, prepare_sample)
            if not triple_logits:
                continue

            sampler = CandidatePathSampler(
                raw_sample,
                triple_logits,
                path_score_agg=args.path_score_agg,
                length_penalty=args.length_penalty,
            )
            source_ids = raw_sample.get("q_entity_id_list", [])
            path_pool = sampler.build_path_pool(
                source_ids=source_ids,
                max_path_length=args.max_path_length,
                beam_width=args.beam_width,
                expand_top_k=args.expand_top_k,
                num_random_walks=args.num_random_walks,
                random_seed=args.seed + sample_idx,
                pool_size=args.path_pool_size,
            )
            if not path_pool:
                continue

            for path_set_id in range(args.num_path_sets):
                policy = policies[path_set_id % len(policies)]
                candidate_paths = sampler.make_path_set(
                    path_pool,
                    paths_per_set=args.paths_per_set,
                    policy=policy,
                    seed=args.seed + sample_idx * 1009 + path_set_id,
                )
                if not candidate_paths:
                    continue
                record = build_output_record(
                    raw_sample,
                    path_set_id=path_set_id,
                    policy=policy,
                    candidate_paths=candidate_paths,
                    path_pool_size=len(path_pool),
                    args=args,
                    runtime_sec=time.time() - start_time,
                )
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_records += 1
    print(f"Saved {total_records} candidate path sets to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", "-p", required=True, help="Path to SubgraphRAG retriever checkpoint cpt.pth")
    parser.add_argument("--dataset", "-d", required=True, choices=["webqsp", "cwq"])
    parser.add_argument("--split", default="test", help="Dataset split to sample")
    parser.add_argument("--output", "-o", required=True, help="Output JSONL path")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of samples for debugging")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--path-pool-size", type=int, default=100)
    parser.add_argument("--paths-per-set", type=int, default=20)
    parser.add_argument("--num-path-sets", type=int, default=3)
    parser.add_argument("--policies", default="top_heavy,diverse,noisy_light")
    parser.add_argument("--max-path-length", type=int, default=4)
    parser.add_argument("--beam-width", type=int, default=16)
    parser.add_argument("--expand-top-k", type=int, default=32)
    parser.add_argument("--num-random-walks", type=int, default=32)
    parser.add_argument("--path-score-agg", default="mean_minus_len", choices=["mean", "sum", "min", "sum_sqrt_len", "mean_minus_len"])
    parser.add_argument("--length-penalty", type=float, default=0.05)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
