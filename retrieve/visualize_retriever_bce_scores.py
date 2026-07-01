"""Visualize SubgraphRAG retriever BCE-logit distributions on triples.

This script loads a checkpoint produced by ``retrieve/train.py``, runs the
SubgraphRAG ``Retriever`` MLP over every triple in each requested split sample,
and plots the raw BCE logits (without applying sigmoid) for positive
(target/evidence) and negative triples as two continuous distributions.

It mirrors the plotting and summary workflow in ``visualize_selector_asl_scores.py``
while using the first-stage retriever checkpoint format and model.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

RETRIEVE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = RETRIEVE_ROOT.parent
sys.path.insert(0, str(RETRIEVE_ROOT))

from src.dataset.retriever import RetrieverDataset, collate_retriever  # noqa: E402
from src.model.retriever import Retriever  # noqa: E402
from src.setup import prepare_sample, set_seed  # noqa: E402


def parse_dataset_list(value):
    if isinstance(value, str):
        if value.lower() == "all":
            return ["webqsp", "cwq"]
        return [item.strip() for item in value.split(",") if item.strip()]
    return list(value)


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"Checkpoint {path} does not contain 'model_state_dict'.")
    if "config" not in checkpoint:
        raise KeyError(f"Checkpoint {path} does not contain 'config'.")
    return checkpoint


def build_dataset_config(args, checkpoint, dataset):
    config = dict(checkpoint["config"])
    config["dataset"] = dict(config.get("dataset", {}))
    config["dataset"]["name"] = dataset
    if args.data_dir:
        config["dataset"]["data_dir"] = args.data_dir
    if args.text_encoder_name:
        config["dataset"]["text_encoder_name"] = args.text_encoder_name
    config["env"] = dict(config.get("env", {}))
    config["env"]["num_threads"] = args.num_threads
    config["env"]["seed"] = args.seed
    return config


@torch.no_grad()
def collect_scores(args, checkpoint, dataset, device):
    config = build_dataset_config(args, checkpoint, dataset)
    infer_set = RetrieverDataset(config=config, split=args.split, skip_no_path=args.skip_no_path)
    data_loader = DataLoader(infer_set, batch_size=1, shuffle=False, collate_fn=collate_retriever)
    emb_size = infer_set[0]["q_emb"].shape[-1]

    model = Retriever(emb_size, **config["retriever"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    positive_scores = []
    negative_scores = []
    sample_count = 0
    triple_count = 0
    positive_count = 0

    for sample in tqdm(data_loader, desc=f"score {dataset}"):
        if args.max_samples is not None and sample_count >= args.max_samples:
            break
        prepared = prepare_sample(device, sample)
        (
            h_id_tensor,
            r_id_tensor,
            t_id_tensor,
            q_emb,
            entity_embs,
            num_non_text_entities,
            relation_embs,
            topic_entity_one_hot,
            target_triple_probs,
            _,
        ) = prepared
        if len(h_id_tensor) == 0:
            sample_count += 1
            continue

        logits = model(
            h_id_tensor,
            r_id_tensor,
            t_id_tensor,
            q_emb,
            entity_embs,
            num_non_text_entities,
            relation_embs,
            topic_entity_one_hot,
        ).reshape(-1)
        labels = (target_triple_probs.to(device).float() > 0).reshape(-1)

        positive_scores.extend(logits[labels].detach().cpu().tolist())
        negative_scores.extend(logits[~labels].detach().cpu().tolist())
        triple_count += int(logits.numel())
        positive_count += int(labels.sum().item())
        sample_count += 1

    return {
        "dataset": dataset,
        "positive": np.asarray(positive_scores, dtype=np.float64),
        "negative": np.asarray(negative_scores, dtype=np.float64),
        "num_samples": sample_count,
        "num_triples": triple_count,
        "num_positive": positive_count,
        "num_negative": triple_count - positive_count,
    }


def smooth_histogram(values, bins, sigma_bins):
    hist, edges = np.histogram(values, bins=bins, density=True)
    centers = (edges[:-1] + edges[1:]) / 2.0
    if sigma_bins > 0 and hist.size > 1:
        radius = max(1, int(round(3 * sigma_bins)))
        x = np.arange(-radius, radius + 1)
        kernel = np.exp(-0.5 * (x / sigma_bins) ** 2)
        kernel = kernel / kernel.sum()
        hist = np.convolve(hist, kernel, mode="same")
    return centers, hist


def score_range(all_results, args):
    arrays = [result[key] for result in all_results for key in ("positive", "negative") if result[key].size]
    if not arrays:
        raise ValueError("No logits were collected; check checkpoint, dataset, and split arguments.")
    if args.score_min is not None and args.score_max is not None:
        return args.score_min, args.score_max
    merged = np.concatenate(arrays)
    lower = float(np.percentile(merged, args.clip_percentile)) if args.score_min is None else args.score_min
    upper = float(np.percentile(merged, 100.0 - args.clip_percentile)) if args.score_max is None else args.score_max
    if lower == upper:
        lower -= 1.0
        upper += 1.0
    margin = 0.05 * (upper - lower)
    return lower - margin, upper + margin


def plot_results(results, args, output_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x_min, x_max = score_range(results, args)
    bins = np.linspace(x_min, x_max, args.num_bins + 1)
    score_label = "Retriever BCE logit (no sigmoid)"

    for result in results:
        fig, ax = plt.subplots(figsize=(8, 5), dpi=args.dpi)
        for label, color, key in (("Positive triples", "tab:blue", "positive"), ("Negative triples", "tab:orange", "negative")):
            values = result[key]
            if values.size == 0:
                continue
            centers, density = smooth_histogram(values, bins=bins, sigma_bins=args.smooth_sigma_bins)
            ax.plot(centers, density, label=f"{label} (n={values.size})", color=color, linewidth=2.0)
            ax.fill_between(centers, density, color=color, alpha=args.fill_alpha)
        ax.set_title(f"{result['dataset']} {args.split}: retriever BCE-logit distribution")
        ax.set_xlabel(score_label)
        ax.set_ylabel("Density")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"{result['dataset']}_{args.split}_bce_logit_distribution.pdf", bbox_inches="tight")
        plt.close(fig)

    if len(results) > 1:
        fig, axes = plt.subplots(1, len(results), figsize=(8 * len(results), 5), dpi=args.dpi, sharey=True)
        if len(results) == 1:
            axes = [axes]
        for ax, result in zip(axes, results):
            for label, color, key in (("Positive triples", "tab:blue", "positive"), ("Negative triples", "tab:orange", "negative")):
                values = result[key]
                if values.size == 0:
                    continue
                centers, density = smooth_histogram(values, bins=bins, sigma_bins=args.smooth_sigma_bins)
                ax.plot(centers, density, label=f"{label} (n={values.size})", color=color, linewidth=2.0)
                ax.fill_between(centers, density, color=color, alpha=args.fill_alpha)
            ax.set_title(f"{result['dataset']} {args.split}")
            ax.set_xlabel(score_label)
            ax.grid(alpha=0.25)
        axes[0].set_ylabel("Density")
        axes[-1].legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"all_{args.split}_bce_logit_distribution.png")
        plt.close(fig)


def write_summary(results, args, output_dir):
    rows = []
    for result in results:
        for key in ("positive", "negative"):
            values = result[key]
            rows.append({
                "dataset": result["dataset"],
                "label": key,
                "count": int(values.size),
                "mean": float(values.mean()) if values.size else np.nan,
                "std": float(values.std()) if values.size else np.nan,
                "median": float(np.median(values)) if values.size else np.nan,
                "p05": float(np.percentile(values, 5)) if values.size else np.nan,
                "p95": float(np.percentile(values, 95)) if values.size else np.nan,
                "num_samples": result["num_samples"],
                "num_triples": result["num_triples"],
            })
        np.savez_compressed(
            output_dir / f"{result['dataset']}_{args.split}_bce_logits.npz",
            positive=result["positive"],
            negative=result["negative"],
        )
    with (output_dir / f"summary_{args.split}_bce_logits.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "run_args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Plot positive/negative triple BCE-logit distributions for a SubgraphRAG retriever checkpoint.")
    parser.add_argument("-p", "--path", required=True, help="Path to retrieve/train.py checkpoint, e.g. webqsp_Nov08-01:14:47/cpt.pth")
    parser.add_argument("-d", "--datasets", default="all", help="Dataset(s) to evaluate: webqsp, cwq, comma-separated list, or all.")
    parser.add_argument("--split", default="test", help="Dataset split to visualize.")
    parser.add_argument("--data_dir", default="retrieve")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_threads", type=int, default=16)
    parser.add_argument("--text_encoder_name", default=None)
    parser.add_argument("--skip_no_path", action="store_true", default=False)
    parser.add_argument("--max_samples", type=int, default=None, help="Optional debug limit per dataset.")
    parser.add_argument("--num_bins", type=int, default=200)
    parser.add_argument("--smooth_sigma_bins", type=float, default=2.0)
    parser.add_argument("--fill_alpha", type=float, default=0.25)
    parser.add_argument("--clip_percentile", type=float, default=0.5, help="Clip each tail by this percentile when auto-scaling x-axis.")
    parser.add_argument("--score_min", type=float, default=None)
    parser.add_argument("--score_max", type=float, default=None)
    parser.add_argument("--dpi", type=int, default=160)
    return parser.parse_args()


def main(args):
    datasets = parse_dataset_list(args.datasets)
    invalid = sorted(set(datasets) - {"webqsp", "cwq"})
    if invalid:
        raise ValueError(f"Unsupported dataset(s): {invalid}. Expected webqsp, cwq, or all.")
    device = torch.device(args.device if torch.cuda.is_available() or "cuda" not in args.device else "cpu")
    torch.set_num_threads(args.num_threads)
    set_seed(args.seed)
    checkpoint = load_checkpoint(args.path)
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.path).resolve().parent / "bce_logit_distribution"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = [collect_scores(args, checkpoint, dataset, device) for dataset in datasets]
    plot_results(results, args, output_dir)
    write_summary(results, args, output_dir)
    for result in results:
        print(
            f"{result['dataset']}: samples={result['num_samples']} triples={result['num_triples']} "
            f"positive={result['num_positive']} negative={result['num_negative']}"
        )
    print(f"Saved plots and BCE-logit summaries to {output_dir}")


if __name__ == "__main__":
    main(parse_args())
