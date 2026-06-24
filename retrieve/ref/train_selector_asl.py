    """Train the second-stage selector with ASL over all candidate triples.

Unlike ``train_selector_sft.py``, this script does not use the frozen first-stage
retriever to rank/filter triples before selector scoring.  The standalone
``Selector`` builds SubgraphRAG-style triple representations internally, so every
triple in each question graph is passed to the selector and receives one
binary-classification logit.  Inference follows the fixed threshold rule:
select triples with ``logit > 0`` and discard triples with ``logit <= 0``.
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import swanlab
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent
RETRIEVE_ROOT = REPO_ROOT / "retrieve"
sys.path.insert(0, str(RETRIEVE_ROOT))

from src.dataset.retriever import RetrieverDataset, collate_retriever  # noqa: E402
from src.setup import prepare_sample, set_seed  # noqa: E402
from selector import Selector  # noqa: E402


def flatten_dict(input_dict, parent_key="", sep="/"):
    flat = {}
    for key, value in input_dict.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else str(key)
        if isinstance(value, dict):
            flat.update(flatten_dict(value, new_key, sep=sep))
        else:
            flat[new_key] = value
    return flat


def parse_k_list(k_list):
    if isinstance(k_list, str):
        return [int(k.strip()) for k in k_list.split(",") if k.strip()]
    if isinstance(k_list, int):
        return [k_list]
    return [int(k) for k in k_list]


def build_swanlab_config(args, retriever_config, triple_feature_size, emb_size):
    config = {f"args/{key}": value for key, value in vars(args).items()}
    config.update({f"retriever/{key}": value for key, value in flatten_dict(retriever_config).items()})
    config.update(
        {
            "selector/triple_feature_size": triple_feature_size,
            "selector/query_emb_size": emb_size,
        }
    )
    return config




def build_selector_config(args):
    """Build the retriever-compatible training config entirely from CLI args."""
    return {
        "env": {
            "num_threads": args.num_threads,
            "seed": args.seed,
        },
        "dataset": {
            "name": args.dataset,
            "text_encoder_name": args.text_encoder_name,
            "data_dir": args.data_dir,
        },
        "retriever": {
            "topic_pe": args.topic_pe,
            "DDE_kwargs": {
                "num_rounds": args.num_rounds,
                "num_reverse_rounds": args.num_reverse_rounds,
            },
        },
        "optimizer": {
            "lr": args.lr,
            "weight_decay": args.weight_decay,
        },
        "eval": {
            "k_list": parse_k_list(args.eval_k_list),
        },
        "train": {
            "num_epochs": args.num_epochs,
            "patience": args.patience,
            "save_prefix": args.save_prefix,
        },
    }

def asymmetric_loss_with_logits(logits, targets, gamma_pos=0.0, gamma_neg=2.0, eps=1e-8, neg_loss_weight=0.1):
    """Asymmetric Loss for sparse binary triple labels.

    Args:
        logits: Selector logits ``s_i`` for all triples in one sample.
        targets: Binary labels where positives are shortest-path/evidence triples.
        gamma_pos: Focusing factor for positives. Defaults to 0 to keep positive
            gradients unsuppressed.
        gamma_neg: Focusing factor for negatives. Defaults to 2 to down-weight
            easy negatives.
        eps: Numerical clamp for log probabilities.
        neg_loss_weight: Weight applied to the mean negative loss after
            positives and negatives are reduced separately.
    """
    targets = (targets > 0).to(dtype=logits.dtype)
    probs = torch.sigmoid(logits).clamp(eps, 1.0 - eps)

    pos_mask = targets > 0
    neg_mask = targets == 0
    pos_terms = -torch.pow(1.0 - probs, gamma_pos) * torch.log(probs)
    neg_terms = -torch.pow(probs, gamma_neg) * torch.log(1.0 - probs)

    pos_loss_mean = pos_terms[pos_mask].mean() if pos_mask.any() else logits.new_tensor(0.0)
    neg_loss_mean = neg_terms[neg_mask].mean() if neg_mask.any() else logits.new_tensor(0.0)
    loss = pos_loss_mean + neg_loss_weight * neg_loss_mean

    pos_count = int(pos_mask.sum().item())
    neg_count = int(neg_mask.sum().item())
    stats = {
        "pos_loss": float(pos_loss_mean.detach().item()),
        "neg_loss": float(neg_loss_mean.detach().item()),
        "pos_count": pos_count,
        "neg_count": neg_count,
        "pos_logit_mean": float(logits[pos_mask].detach().mean().item()) if pos_count > 0 else 0.0,
        "neg_logit_mean": float(logits[neg_mask].detach().mean().item()) if neg_count > 0 else 0.0,
        "selected_count": int((logits > 0).sum().item()),
    }
    return loss, stats


def pairwise_ranking_loss(selector_logits, target_labels):
    """Rank positive candidate triples above negative candidate triples."""
    pos_logits = selector_logits[target_labels > 0]
    neg_logits = selector_logits[target_labels <= 0]
    if pos_logits.numel() == 0 or neg_logits.numel() == 0:
        return selector_logits.new_zeros(())
    return -F.logsigmoid(pos_logits[:, None] - neg_logits[None, :]).mean()


@torch.no_grad()
def eval_epoch(args, config, device, data_loader, selector, k_list=None, desc="val"):
    selector.eval()
    metric_dict = defaultdict(list)
    total_tp = 0
    total_selected = 0
    total_target = 0

    eval_k_list = parse_k_list(k_list or config["eval"]["k_list"])

    for sample in tqdm(data_loader, desc=desc):
        prepared = prepare_sample(device, sample)
        h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs, num_non_text_entities, relation_embs, topic_entity_one_hot, target_triple_probs, a_entity_id_list = prepared
        if len(h_id_tensor) == 0:
            continue

        labels = (target_triple_probs.to(device).float() > 0).float()
        target_triple_ids = labels.nonzero().squeeze(-1)
        num_target_triples = int(target_triple_ids.numel())
        if num_target_triples == 0:
            continue

        selector_logits = selector(
            h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs,
            num_non_text_entities, relation_embs, topic_entity_one_hot
        ).reshape(-1)

        selected_mask = selector_logits > 0
        true_positive = int((selected_mask & (labels > 0)).sum().item())
        selected_count = int(selected_mask.sum().item())
        precision = true_positive / selected_count if selected_count > 0 else 0.0
        recall = true_positive / num_target_triples if num_target_triples > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        total_tp += true_positive
        total_selected += selected_count
        total_target += num_target_triples
        metric_dict["threshold_recall"].append(recall)
        metric_dict["threshold_precision"].append(precision)
        metric_dict["threshold_f1"].append(f1)
        metric_dict["threshold_selected_count"].append(selected_count)
        metric_dict["threshold_true_positive"].append(true_positive)

        sorted_triple_ids_pred = torch.argsort(selector_logits, descending=True)
        triple_ranks_pred = torch.empty_like(sorted_triple_ids_pred)
        triple_ranks_pred[sorted_triple_ids_pred] = torch.arange(len(triple_ranks_pred), device=device)

        num_total_entities = len(entity_embs) + num_non_text_entities
        for k in eval_k_list:
            k = min(int(k), len(triple_ranks_pred))
            if k <= 0:
                continue
            recall_k_sample = (triple_ranks_pred[target_triple_ids] < k).sum().item()
            metric_dict[f"triple_recall@{k}"].append(recall_k_sample / num_target_triples)

            triple_mask_k = triple_ranks_pred < k
            entity_mask_k = torch.zeros(num_total_entities, device=device)
            entity_mask_k[h_id_tensor[triple_mask_k]] = 1.0
            entity_mask_k[t_id_tensor[triple_mask_k]] = 1.0
            answer_ids = torch.tensor(a_entity_id_list, dtype=torch.long, device=device)
            recall_k_sample_ans = entity_mask_k[answer_ids].sum().item()
            metric_dict[f"ans_recall@{k}"].append(recall_k_sample_ans / len(a_entity_id_list))

    result = {key: float(np.mean(val)) for key, val in metric_dict.items()}
    micro_precision = total_tp / total_selected if total_selected > 0 else 0.0
    micro_recall = total_tp / total_target if total_target > 0 else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if micro_precision + micro_recall > 0
        else 0.0
    )
    result["threshold_micro_precision"] = micro_precision
    result["threshold_micro_recall"] = micro_recall
    result["threshold_micro_f1"] = micro_f1

    # Explicit selector-threshold aliases for SwanLab dashboards: these metrics
    # evaluate only the triples selected by the model's fixed inference rule
    # (logits > 0), independent of any top-k recall reference.
    result["selector_precision"] = result.get("threshold_precision", 0.0)
    result["selector_recall"] = result.get("threshold_recall", 0.0)
    result["selector_f1"] = result.get("threshold_f1", 0.0)
    result["selector_avg_selected_triples"] = result.get("threshold_selected_count", 0.0)
    result["selector_micro_precision"] = micro_precision
    result["selector_micro_recall"] = micro_recall
    result["selector_micro_f1"] = micro_f1
    return result


def train_epoch(args, device, train_loader, selector, optimizer):
    selector.train()
    loss_sums = defaultdict(float)
    num_updates = 0

    for sample in tqdm(train_loader, desc="train"):
        prepared = prepare_sample(device, sample)
        h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs, num_non_text_entities, relation_embs, topic_entity_one_hot, target_triple_probs, _ = prepared
        if len(h_id_tensor) == 0:
            continue

        target_labels = (target_triple_probs.to(device).float() > 0).float()
        selector_logits = selector(
            h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs,
            num_non_text_entities, relation_embs, topic_entity_one_hot
        ).reshape(-1)
        asl_loss, loss_stats = asymmetric_loss_with_logits(
            selector_logits,
            target_labels,
            gamma_pos=args.gamma_pos,
            gamma_neg=args.gamma_neg,
            eps=args.asl_eps,
            neg_loss_weight=args.neg_loss_weight,
        )
        rank_loss = pairwise_ranking_loss(selector_logits, target_labels)
        loss = asl_loss
        if args.rank_lambda != 0:
            loss = loss + args.rank_lambda * rank_loss

        optimizer.zero_grad()
        loss.backward()
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(selector.parameters(), args.max_grad_norm)
        optimizer.step()

        loss_sums["total_loss"] += float(loss.detach().item())
        loss_sums["asl_loss"] += float(asl_loss.detach().item())
        loss_sums["rank_loss"] += float(rank_loss.detach().item())
        for key, value in loss_stats.items():
            loss_sums[key] += float(value)
        num_updates += 1

    denom = max(num_updates, 1)
    return {f"train/{key}": value / denom for key, value in loss_sums.items()}


def main(args):
    config = build_selector_config(args)
    args.val_k_list = parse_k_list(args.val_k_list or args.eval_k_list)
    args.test_k_list = parse_k_list(args.test_k_list or args.eval_k_list)
    if args.target_val_k not in args.val_k_list:
        args.val_k_list.append(args.target_val_k)
    args.val_k_list = sorted(set(args.val_k_list))
    args.test_k_list = sorted(set(args.test_k_list))
    config["eval"]["k_list"] = args.val_k_list

    device = torch.device(args.device if torch.cuda.is_available() or "cuda" not in args.device else "cpu")
    torch.set_num_threads(config["env"].get("num_threads", 1))
    set_seed(args.seed if args.seed is not None else config["env"]["seed"])

    train_set = RetrieverDataset(config=config, split=args.split, skip_no_path=args.skip_no_path)
    val_set = RetrieverDataset(config=config, split=args.val_split, skip_no_path=args.skip_no_path)
    test_set = RetrieverDataset(config=config, split=args.test_split, split_type=False) if args.test_split else None
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, collate_fn=collate_retriever)
    test_loader = (
        DataLoader(test_set, batch_size=1, shuffle=False, collate_fn=collate_retriever)
        if test_set is not None
        else None
    )

    emb_size = train_set[0]["q_emb"].shape[-1]
    selector = Selector(
        emb_size=emb_size,
        **config["retriever"],
        global_hidden_size=args.global_hidden_size,
        num_heads=args.num_heads,
        global_layers=args.global_layers,
        dropout=args.dropout,
    ).to(device)
    triple_feature_size = selector.triple_feature_size

    optimizer_kwargs = dict(config.get("optimizer", {}))
    if args.lr is not None:
        optimizer_kwargs["lr"] = args.lr
    if args.weight_decay is not None:
        optimizer_kwargs["weight_decay"] = args.weight_decay
    optimizer = Adam(selector.parameters(), **optimizer_kwargs)

    ts = time.strftime("%b%d-%H:%M:%S", time.gmtime())
    output_dir = Path(args.output_dir or f"selector_asl_{args.dataset}_{ts}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    config_df = pd.json_normalize(build_swanlab_config(args, config, triple_feature_size, emb_size), sep="/")
    swanlab.init(
        project=args.swanlab_project or f"selector-asl-{args.dataset}",
        name=args.swanlab_name or output_dir.name,
        config=config_df.to_dict(orient="records")[0],
        mode=args.swanlab_mode,
    )

    num_patient_epochs = 0
    best_val_metric = 0.0
    best_state_dict = None
    for epoch in range(args.num_epochs):
        num_patient_epochs += 1
        val_eval_dict = eval_epoch(args, config, device, val_loader, selector, args.val_k_list, desc="val")
        if args.best_metric not in val_eval_dict:
            print(
                f"Warning: best_metric '{args.best_metric}' not found in validation metrics; "
                "falling back to 'threshold_f1'.",
                file=sys.stderr,
            )
            target_val_metric = val_eval_dict.get("threshold_f1", 0.0)
        else:
            target_val_metric = val_eval_dict.get(args.best_metric, 0.0)

        if target_val_metric > best_val_metric:
            num_patient_epochs = 0
            best_val_metric = target_val_metric
            best_state_dict = {
                "config": config,
                "selector_state_dict": selector.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "args": vars(args),
                "epoch": epoch,
                "best_val_metric": best_val_metric,
                "triple_feature_size": triple_feature_size,
                "query_emb_size": emb_size,
            }
            torch.save(best_state_dict, output_dir / "cpt.pth")

        swanlab.log({"val/epoch": epoch, **{f"val/{key}": val for key, val in val_eval_dict.items()}})

        if test_loader is not None:
            test_eval_dict = eval_epoch(args, config, device, test_loader, selector, args.test_k_list, desc="test")
            swanlab.log({"test/epoch": epoch, **{f"test/{key}": val for key, val in test_eval_dict.items()}})

        train_loader = DataLoader(train_set, batch_size=1, shuffle=True, collate_fn=collate_retriever)
        train_log_dict = train_epoch(args, device, train_loader, selector, optimizer)
        train_log_dict.update({"num_patient_epochs": num_patient_epochs, "epoch": epoch, "trainset_size": len(train_set)})
        swanlab.log(train_log_dict)

        last_state = {
            "config": config,
            "selector_state_dict": selector.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
            "epoch": epoch,
            "best_val_metric": best_val_metric,
            "triple_feature_size": triple_feature_size,
            "query_emb_size": emb_size,
        }
        torch.save(last_state, output_dir / "last.pth")
        if num_patient_epochs == args.patience:
            break

    if best_state_dict is None:
        torch.save(last_state if args.num_epochs > 0 else {
            "config": config,
            "selector_state_dict": selector.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
            "epoch": max(args.num_epochs - 1, 0),
            "best_val_metric": best_val_metric,
            "triple_feature_size": triple_feature_size,
            "query_emb_size": emb_size,
        }, output_dir / "cpt.pth")
    swanlab.finish()


def parse_args():
    parser = argparse.ArgumentParser(description="ASL training for all-triple second-stage selector scoring")
    parser.add_argument("-d", "--dataset", type=str, default="cwq", choices=["webqsp", "cwq"])
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--val_split", type=str, default="val")
    parser.add_argument("--test_split", type=str, default="test", help="Split used for test metrics. Set empty to disable.")
    parser.add_argument("--data_dir", type=str, default="retrieve")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_threads", type=int, default=16)
    parser.add_argument("--text_encoder_name", type=str, default="gte-large-en-v1.5")
    parser.add_argument("--topic_pe", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num_rounds", type=int, default=2)
    parser.add_argument("--num_reverse_rounds", type=int, default=2)
    parser.add_argument("--save_prefix", type=str, default=None)
    parser.add_argument("--num_epochs", type=int, default=10000)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--eval_k_list", type=str, default="100", help="Deprecated/default K values used when val/test K lists are unset.")
    parser.add_argument("--val_k_list", type=str, default=None, help="Comma-separated K values for validation metrics.")
    parser.add_argument("--test_k_list", "--k_list", dest="test_k_list", type=str, default=None, help="Comma-separated K values for test metrics.")
    parser.add_argument("--target_val_k", type=int, default=100)
    parser.add_argument("--gamma_pos", type=float, default=0.0)
    parser.add_argument("--gamma_neg", type=float, default=2.0)
    parser.add_argument("--asl_eps", type=float, default=1e-8)
    parser.add_argument("--neg_loss_weight", type=float, default=0.1)
    parser.add_argument("--rank_lambda", type=float, default=1.0)
    parser.add_argument(
        "--best_metric",
        type=str,
        default="threshold_f1",
        choices=["threshold_f1", "threshold_micro_f1", "triple_recall@100", "ans_recall@100"],
    )
    parser.add_argument("--global_hidden_size", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--global_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--skip_no_path", action="store_true", default=True)
    parser.add_argument("--keep_no_path", action="store_false", dest="skip_no_path")
    parser.add_argument("--swanlab_project", type=str, default=None)
    parser.add_argument("--swanlab_name", type=str, default=None)
    parser.add_argument("--swanlab_mode", type=str, default="cloud", choices=["cloud", "local", "disabled"])
    args = parser.parse_args()
    if args.save_prefix is None:
        args.save_prefix = args.dataset
    return args


if __name__ == "__main__":
    main(parse_args())
