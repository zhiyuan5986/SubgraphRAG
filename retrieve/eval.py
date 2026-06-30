import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def smooth_histogram(values, bins, sigma_bins):
    """Return a smoothed density histogram for plotting as a curve."""
    hist, edges = np.histogram(values, bins=bins, density=True)
    centers = (edges[:-1] + edges[1:]) / 2.0
    if sigma_bins > 0 and hist.size > 1:
        radius = max(1, int(round(3 * sigma_bins)))
        x = np.arange(-radius, radius + 1)
        kernel = np.exp(-0.5 * (x / sigma_bins) ** 2)
        kernel = kernel / kernel.sum()
        hist = np.convolve(hist, kernel, mode="same")
    return centers, hist


def triple_count_range(values, args):
    if values.size == 0:
        raise ValueError(
            "No triple counts were collected; check retrieval result path."
        )
    lower = (
        float(values.min()) if args.triple_count_min is None else args.triple_count_min
    )
    upper = (
        float(values.max()) if args.triple_count_max is None else args.triple_count_max
    )
    if lower == upper:
        lower -= 1.0
        upper += 1.0
    margin = 0.05 * (upper - lower)
    return max(0.0, lower - margin), upper + margin


def visualize_triple_count_distribution(triple_counts, args):
    """Plot distribution curve for the number of retrieved triples per question."""
    counts = np.asarray(triple_counts, dtype=np.float64)
    if counts.size == 0:
        print("Skip visualization: no questions with scored triples were found.")
        return

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Skip visualization: matplotlib is not installed in this environment.")
        return

    output_dir = Path(args.path).resolve().parent
    x_min, x_max = triple_count_range(counts, args)
    bins = np.linspace(x_min, x_max, args.num_bins + 1)
    centers, density = smooth_histogram(
        counts, bins=bins, sigma_bins=args.smooth_sigma_bins
    )

    fig, ax = plt.subplots(figsize=(8, 5), dpi=args.dpi)
    ax.plot(
        centers,
        density,
        label=f"Retrieved triples (n={counts.size})",
        color="tab:blue",
        linewidth=2.0,
    )
    ax.fill_between(centers, density, color="tab:blue", alpha=args.fill_alpha)
    ax.set_title(f"{args.dataset}: retrieved triple-count distribution")
    ax.set_xlabel("Number of retrieved triples per question")
    ax.set_ylabel("Density")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()

    plot_path = output_dir / "question_retrieved_triple_count_distribution.pdf"
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)

    summary_path = output_dir / "question_retrieved_triple_count_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "num_questions",
                "mean",
                "std",
                "median",
                "min",
                "max",
                "p05",
                "p95",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "num_questions": int(counts.size),
                "mean": float(counts.mean()),
                "std": float(counts.std()),
                "median": float(np.median(counts)),
                "min": float(counts.min()),
                "max": float(counts.max()),
                "p05": float(np.percentile(counts, 5)),
                "p95": float(np.percentile(counts, 95)),
            }
        )
    print(f"Saved triple-count distribution plot to {plot_path}")
    print(f"Saved triple-count distribution summary to {summary_path}")


def resolve_gpt_triple_path(dataset, data_dir):
    candidates = [
        Path(data_dir) / "data_files" / dataset / "gpt_triples.pth",
        Path("data_files") / dataset / "gpt_triples.pth",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Cannot find gpt_triples.pth. Tried: "
        + ", ".join(str(path) for path in candidates)
    )


def main(args):
    pred_dict = torch.load(args.path, map_location='cpu')
    gpt_triple_dict = torch.load(
        resolve_gpt_triple_path(args.dataset, args.data_dir), map_location='cpu'
    )
    k_list = [int(k) for k in args.k_list.split(',')]
    
    metric_dict = dict()
    thres_metric_dict = None
    thres_triple_counts = []
    retrieved_triple_counts = []
    if args.thres is not None:
        thres_metric_dict = {
            'ans_recall': [],
            'shortest_path_triple_recall': [],
            'gpt_triple_recall': [],
        }

    for k in k_list:
        metric_dict[f'ans_recall@{k}'] = []
        metric_dict[f'shortest_path_triple_recall@{k}'] = []
        metric_dict[f'gpt_triple_recall@{k}'] = []
    
    shortest_path_triples_len_list = []
    for sample_id in pred_dict:
        if len(pred_dict[sample_id]['scored_triples']) == 0:
            continue

        retrieved_triple_counts.append(len(pred_dict[sample_id]['scored_triples']))
        
        h_list, r_list, t_list, score_list = zip(*pred_dict[sample_id]['scored_triples'])

        threshold_triples = []
        threshold_triple_set = set()
        threshold_entities = set()
        if args.thres is not None:
            threshold_triples = [
                (h, r, t)
                for h, r, t, score in zip(h_list, r_list, t_list, score_list)
                if float(score) > args.thres
            ]
            threshold_triple_set = set(threshold_triples)
            threshold_entities = {
                entity for h, _, t in threshold_triples for entity in (h, t)
            }
            thres_triple_counts.append(len(threshold_triples))
        
        a_entity_in_graph = set(pred_dict[sample_id]['a_entity_in_graph'])
        if len(a_entity_in_graph) > 0:
            for k in k_list:
                entities_k = set(h_list[:k] + t_list[:k])
                metric_dict[f'ans_recall@{k}'].append(
                    len(a_entity_in_graph & entities_k) / len(a_entity_in_graph)
                )
            if thres_metric_dict is not None:
                thres_metric_dict['ans_recall'].append(
                    len(a_entity_in_graph & threshold_entities) / len(a_entity_in_graph)
                )
        
        triples = list(zip(h_list, r_list, t_list))
        shortest_path_triples = set(pred_dict[sample_id]['target_relevant_triples'])
        if len(shortest_path_triples) > 0:
            for k in k_list:
                triples_k = set(triples[:k])
                metric_dict[f'shortest_path_triple_recall@{k}'].append(
                    len(shortest_path_triples & triples_k) / len(shortest_path_triples)
                )
            if thres_metric_dict is not None:
                thres_metric_dict['shortest_path_triple_recall'].append(
                    len(shortest_path_triples & threshold_triple_set)
                    / len(shortest_path_triples)
                )
            shortest_path_triples_len_list.append(len(shortest_path_triples))
        
        gpt_triples = set(gpt_triple_dict.get(sample_id, []))
        if len(gpt_triples) > 0:
            for k in k_list:
                triples_k = set(triples[:k])
                metric_dict[f'gpt_triple_recall@{k}'].append(
                    len(gpt_triples & triples_k) / len(gpt_triples)
                )
            if thres_metric_dict is not None:
                thres_metric_dict['gpt_triple_recall'].append(
                    len(gpt_triples & threshold_triple_set) / len(gpt_triples)
                )

    # draw histogram
    # import matplotlib.pyplot as plt
    # shortest_path_triples_len_list = [s if s < 100 else 100 for s in shortest_path_triples_len_list]
    # plt.hist(shortest_path_triples_len_list, bins=20)
    # plt.savefig(f'{args.dataset}_shortest_path_triples_len_hist.pdf', bbox_inches='tight')

    for k in k_list:
        print(len(metric_dict[f'gpt_triple_recall@{k}']))
    for metric, val in metric_dict.items():
        metric_dict[metric] = np.mean(val)
    
    
    table_dict = {
        'K': k_list,
        'ans_recall': [
            round(metric_dict[f'ans_recall@{k}'], 3) for k in k_list
        ],
        'shortest_path_triple_recall': [
            round(metric_dict[f'shortest_path_triple_recall@{k}'], 3) for k in k_list
        ],
        'gpt_triple_recall': [
            round(metric_dict[f'gpt_triple_recall@{k}'], 3) for k in k_list
        ]
    }
    df = pd.DataFrame(table_dict)
    print(df.to_string(index=False))

    if thres_metric_dict is not None:
        for metric, val in thres_metric_dict.items():
            thres_metric_dict[metric] = np.mean(val)

        thres_table_dict = {
            'thres': [args.thres],
            'ans_recall': [round(thres_metric_dict['ans_recall'], 3)],
            'shortest_path_triple_recall': [
                round(thres_metric_dict['shortest_path_triple_recall'], 3)
            ],
            'gpt_triple_recall': [round(thres_metric_dict['gpt_triple_recall'], 3)],
            'avg_triples': [round(float(np.mean(thres_triple_counts)), 3)],
        }
        thres_df = pd.DataFrame(thres_table_dict)
        print('\nThreshold-based metrics:')
        print(thres_df.to_string(index=False))

    if args.visualize:
        visualize_triple_count_distribution(retrieved_triple_counts, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, required=True, 
                        choices=['webqsp', 'cwq'], help='Dataset name')
    parser.add_argument('-p', '--path', type=str, required=True,
                        help='Path to retrieval result')
    parser.add_argument('--data_dir', type=str, default='.')
    parser.add_argument('--k_list', type=str, default='50,100,200,400',
                        help='Comma-separated list of K values for top-K recall evaluation')
    parser.add_argument(
        '--visualize',
        action='store_true',
        default=True,
        help=(
            'Plot the distribution of retrieved triple counts per question and save '
            'it under the parent directory of --path (enabled by default).'
        ),
    )
    parser.add_argument(
        '--no_visualize',
        action='store_false',
        dest='visualize',
        help='Disable triple-count distribution visualization.',
    )
    parser.add_argument('--num_bins', type=int, default=200,
                        help='Number of bins for the triple-count distribution plot.')
    parser.add_argument('--smooth_sigma_bins', type=float, default=2.0,
                        help='Gaussian smoothing sigma measured in histogram bins.')
    parser.add_argument('--fill_alpha', type=float, default=0.25,
                        help='Alpha for the filled area under the distribution curve.')
    parser.add_argument('--triple_count_min', type=float, default=None,
                        help='Optional minimum x-axis value for the distribution plot.')
    parser.add_argument('--triple_count_max', type=float, default=None,
                        help='Optional maximum x-axis value for the distribution plot.')
    parser.add_argument('--dpi', type=int, default=160,
                        help='DPI for the saved distribution plot.')
    parser.add_argument(
        '--thres',
        type=float,
        default=None,
        help=(
            'If set, select triples with score greater than this threshold and '
            'report recalls plus average selected triples.'
        ),
    )
    args = parser.parse_args()
    
    main(args)
