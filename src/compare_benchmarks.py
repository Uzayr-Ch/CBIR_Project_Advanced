"""Compare ResNet50 vs ViT-B/16 retrieval performance across datasets."""

from __future__ import annotations

import argparse
import os
import pickle
import time
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

DATASETS: Dict[str, str] = {
    "Corel-1K": "features/Corel-1K_features.pkl",
    "Corel-5K": "features/Corel-5K_features.pkl",
    "Corel-10K": "features/Corel-10K_features.pkl",
    "Caltech-101": "features/Caltech-101_features.pkl",
}

VIT_FEATURES = {name: path.replace("features/", "features/").replace("_features", "_vit_features") for name, path in DATASETS.items()}
RESULTS_DIR = "graphs"
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_features(features_path: str, paths_path: str) -> Tuple[np.ndarray, List[str]]:
    if not os.path.exists(features_path) or not os.path.exists(paths_path):
        raise FileNotFoundError(f"Missing features or paths pickle for {features_path}")
    with open(features_path, "rb") as f:
        feats = pickle.load(f)
    with open(paths_path, "rb") as f:
        paths = pickle.load(f)
    return np.asarray(feats, dtype=np.float32), list(paths)


def get_category(path: str) -> str:
    return os.path.basename(os.path.dirname(path))


def evaluate_model(features: np.ndarray, image_paths: List[str], k: int = 10, max_queries: int | None = None) -> Tuple[float, float]:
    indices = range(len(features)) if max_queries is None else range(min(len(features), max_queries))
    precisions: List[float] = []
    total_latency = 0.0

    for i in tqdm(indices, desc="Evaluating", leave=False):
        query_feat = features[i].reshape(1, -1)
        query_cat = get_category(image_paths[i])

        start = time.perf_counter()
        sims = cosine_similarity(query_feat, features).flatten()
        top_idx = np.argsort(sims)[::-1][1 : k + 1]  # skip the query itself
        latency_ms = (time.perf_counter() - start) * 1000

        hits = sum(1 for idx in top_idx if get_category(image_paths[idx]) == query_cat)
        precisions.append(hits / k)
        total_latency += latency_ms

    mAP = float(np.mean(precisions) * 100)
    avg_latency = (total_latency / len(precisions)) if precisions else 0.0
    return mAP, avg_latency


def format_markdown_table(results: Dict[str, Dict[str, Dict[str, float]]]) -> str:
    lines = ["| Dataset | Model | mAP (%) | Avg Retrieval Latency (ms) |", "|:-------|:------|--------:|---------------------------:|"]
    for dataset, models in results.items():
        for model_name, metrics in models.items():
            lines.append(
                f"| {dataset} | {model_name} | {metrics['mAP']:.2f} | {metrics['latency_ms']:.3f} |"
            )
    return "\n".join(lines)


def plot_accuracy_speed(results: Dict[str, Dict[str, Dict[str, float]]]) -> List[str]:
    """Generate multiple visually appealing comparison graphs."""
    
    # Modern style configuration
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.facecolor': 'white',
        'axes.facecolor': '#FAFBFC',
        'axes.edgecolor': '#E1E4E8',
        'axes.linewidth': 1.2,
        'grid.color': '#E1E4E8',
        'grid.linestyle': '-',
        'grid.linewidth': 0.8,
    })
    
    # Color palette
    RESNET_COLOR = '#3B82F6'  # Blue
    VIT_COLOR = '#10B981'     # Green
    ACCENT = '#F59E0B'        # Amber
    
    output_paths = []
    datasets = list(results.keys())
    
    # =========================================================================
    # GRAPH 1: Grouped Bar Chart - mAP Comparison
    # =========================================================================
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(datasets))
    width = 0.35
    
    resnet_maps = [results[d]["ResNet50"]["mAP"] for d in datasets]
    vit_maps = [results[d]["ViT-B/16"]["mAP"] for d in datasets]
    
    bars1 = ax.bar(x - width/2, resnet_maps, width, label='ResNet50', 
                   color=RESNET_COLOR, edgecolor='white', linewidth=1.5, zorder=3)
    bars2 = ax.bar(x + width/2, vit_maps, width, label='ViT-B/16',
                   color=VIT_COLOR, edgecolor='white', linewidth=1.5, zorder=3)
    
    # Value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 4), textcoords='offset points',
                        ha='center', va='bottom', fontweight='bold', fontsize=10,
                        color='#374151')
    
    # Winner indicators
    for i, d in enumerate(datasets):
        winner = "ViT" if vit_maps[i] > resnet_maps[i] else "ResNet"
        diff = abs(vit_maps[i] - resnet_maps[i])
        if diff > 0.5:
            max_y = max(resnet_maps[i], vit_maps[i])
            ax.annotate(f'{"↑" if winner == "ViT" else "←"} +{diff:.1f}%',
                        xy=(i, max_y + 5), ha='center', fontsize=9,
                        color=VIT_COLOR if winner == "ViT" else RESNET_COLOR,
                        fontweight='bold')
    
    ax.set_ylabel('Mean Average Precision (%)', fontweight='bold')
    ax.set_title('Accuracy Comparison: ResNet50 vs ViT-B/16', fontweight='bold', pad=15, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontweight='medium')
    ax.set_ylim(0, 115)
    ax.legend(loc='upper right', framealpha=0.95, edgecolor='#E1E4E8')
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    
    # Baseline reference
    ax.axhline(y=90, color='#EF4444', linestyle='--', linewidth=1.2, alpha=0.6, zorder=1)
    ax.text(len(datasets)-0.5, 91, '90% baseline', fontsize=9, color='#EF4444', style='italic')
    
    plt.tight_layout()
    path1 = os.path.join(RESULTS_DIR, "comparison_accuracy.png")
    plt.savefig(path1, dpi=200, facecolor='white', edgecolor='none')
    plt.close(fig)
    output_paths.append(path1)
    
    # =========================================================================
    # GRAPH 2: Grouped Bar Chart - Latency Comparison
    # =========================================================================
    fig, ax = plt.subplots(figsize=(12, 6))
    
    resnet_lats = [results[d]["ResNet50"]["latency_ms"] for d in datasets]
    vit_lats = [results[d]["ViT-B/16"]["latency_ms"] for d in datasets]
    
    bars1 = ax.bar(x - width/2, resnet_lats, width, label='ResNet50',
                   color=RESNET_COLOR, edgecolor='white', linewidth=1.5, zorder=3)
    bars2 = ax.bar(x + width/2, vit_lats, width, label='ViT-B/16',
                   color=VIT_COLOR, edgecolor='white', linewidth=1.5, zorder=3)
    
    # Value and speedup labels
    for i, d in enumerate(datasets):
        speedup = resnet_lats[i] / vit_lats[i] if vit_lats[i] > 0 else 1
        ax.annotate(f'{resnet_lats[i]:.1f}ms', xy=(i - width/2, resnet_lats[i]),
                    xytext=(0, 4), textcoords='offset points',
                    ha='center', fontweight='bold', fontsize=9, color='#374151')
        ax.annotate(f'{vit_lats[i]:.1f}ms', xy=(i + width/2, vit_lats[i]),
                    xytext=(0, 4), textcoords='offset points',
                    ha='center', fontweight='bold', fontsize=9, color='#374151')
        # Speedup annotation
        ax.annotate(f'⚡ {speedup:.1f}x faster', xy=(i, max(resnet_lats[i], vit_lats[i]) + 8),
                    ha='center', fontsize=9, color=VIT_COLOR, fontweight='bold')
    
    ax.set_ylabel('Retrieval Latency (ms)', fontweight='bold')
    ax.set_title('Speed Comparison: ResNet50 vs ViT-B/16', fontweight='bold', pad=15, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontweight='medium')
    ax.legend(loc='upper left', framealpha=0.95, edgecolor='#E1E4E8')
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    path2 = os.path.join(RESULTS_DIR, "comparison_latency.png")
    plt.savefig(path2, dpi=200, facecolor='white', edgecolor='none')
    plt.close(fig)
    output_paths.append(path2)
    
    # =========================================================================
    # GRAPH 3: Scatter Plot - Accuracy vs Speed Trade-off
    # =========================================================================
    fig, ax = plt.subplots(figsize=(11, 7))
    
    # Plot all points
    for i, dataset in enumerate(datasets):
        for model, color, marker in [("ResNet50", RESNET_COLOR, 'o'), ("ViT-B/16", VIT_COLOR, 's')]:
            m = results[dataset][model]
            size = 200 + i * 80  # Larger for larger datasets
            ax.scatter(m["latency_ms"], m["mAP"], s=size, c=color, marker=marker,
                       edgecolors='white', linewidths=2, alpha=0.85, zorder=3)
            
            # Label with dataset name
            offset_x = 8 if model == "ViT-B/16" else -8
            ha = 'left' if model == "ViT-B/16" else 'right'
            ax.annotate(f'{dataset}\n({model.split("-")[0]})', 
                        xy=(m["latency_ms"], m["mAP"]),
                        xytext=(offset_x, 0), textcoords='offset points',
                        ha=ha, va='center', fontsize=9,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                  edgecolor='#E1E4E8', alpha=0.9))
    
    # Connect same dataset points with arrows
    for dataset in datasets:
        r = results[dataset]["ResNet50"]
        v = results[dataset]["ViT-B/16"]
        ax.annotate('', xy=(v["latency_ms"], v["mAP"]), xytext=(r["latency_ms"], r["mAP"]),
                    arrowprops=dict(arrowstyle='->', color='#9CA3AF', lw=1.5, 
                                    connectionstyle='arc3,rad=0.1'))
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=RESNET_COLOR, 
               markersize=12, label='ResNet50'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=VIT_COLOR,
               markersize=12, label='ViT-B/16'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', framealpha=0.95)
    
    ax.set_xlabel('Retrieval Latency (ms) → Lower is Better', fontweight='bold')
    ax.set_ylabel('Mean Average Precision (%) → Higher is Better', fontweight='bold')
    ax.set_title('Accuracy vs Speed Trade-off Analysis', fontweight='bold', pad=15, fontsize=14)
    ax.grid(True, zorder=0)
    ax.set_axisbelow(True)
    
    # Ideal region annotation
    ax.annotate('← IDEAL REGION\n(High Accuracy, Low Latency)', 
                xy=(ax.get_xlim()[0] + 5, ax.get_ylim()[1] - 3),
                fontsize=10, color='#059669', fontweight='bold', style='italic')
    
    plt.tight_layout()
    path3 = os.path.join(RESULTS_DIR, "vit_vs_resnet_accuracy_speed.png")
    plt.savefig(path3, dpi=200, facecolor='white', edgecolor='none')
    plt.close(fig)
    output_paths.append(path3)
    
    # =========================================================================
    # GRAPH 4: Summary Dashboard
    # =========================================================================
    fig = plt.figure(figsize=(14, 8))
    
    # Create grid
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)
    ax1 = fig.add_subplot(gs[0, :2])  # Top left - wider
    ax2 = fig.add_subplot(gs[0, 2])   # Top right
    ax3 = fig.add_subplot(gs[1, :])   # Bottom - full width
    
    # Panel 1: mAP comparison
    x = np.arange(len(datasets))
    ax1.bar(x - 0.2, resnet_maps, 0.4, label='ResNet50', color=RESNET_COLOR, zorder=3)
    ax1.bar(x + 0.2, vit_maps, 0.4, label='ViT-B/16', color=VIT_COLOR, zorder=3)
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets)
    ax1.set_ylabel('mAP (%)')
    ax1.set_title('Accuracy by Dataset', fontweight='bold')
    ax1.legend()
    ax1.yaxis.grid(True, zorder=0)
    ax1.set_ylim(0, 110)
    
    # Panel 2: Overall Stats
    avg_resnet_map = np.mean(resnet_maps)
    avg_vit_map = np.mean(vit_maps)
    avg_resnet_lat = np.mean(resnet_lats)
    avg_vit_lat = np.mean(vit_lats)
    
    stats = [
        ('Avg mAP\nResNet50', f'{avg_resnet_map:.1f}%', RESNET_COLOR),
        ('Avg mAP\nViT-B/16', f'{avg_vit_map:.1f}%', VIT_COLOR),
        ('Avg Latency\nResNet50', f'{avg_resnet_lat:.1f}ms', RESNET_COLOR),
        ('Avg Latency\nViT-B/16', f'{avg_vit_lat:.1f}ms', VIT_COLOR),
    ]
    
    ax2.axis('off')
    for i, (label, value, color) in enumerate(stats):
        y = 0.85 - i * 0.22
        ax2.text(0.5, y, value, transform=ax2.transAxes, fontsize=20, 
                 fontweight='bold', ha='center', color=color)
        ax2.text(0.5, y - 0.08, label, transform=ax2.transAxes, fontsize=10,
                 ha='center', color='#6B7280')
    ax2.set_title('Overall Averages', fontweight='bold')
    
    # Panel 3: Speedup visualization
    speedups = [resnet_lats[i] / vit_lats[i] for i in range(len(datasets))]
    bars = ax3.barh(datasets, speedups, color=VIT_COLOR, edgecolor='white', linewidth=1.5, zorder=3)
    ax3.axvline(x=1, color='#9CA3AF', linestyle='--', linewidth=1.5)
    ax3.set_xlabel('Speedup Factor (ViT vs ResNet) → Higher is Better')
    ax3.set_title('ViT-B/16 Speedup Over ResNet50', fontweight='bold')
    ax3.xaxis.grid(True, zorder=0)
    
    for bar, speedup in zip(bars, speedups):
        ax3.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                 f'{speedup:.1f}x faster', va='center', fontweight='bold', fontsize=11)
    
    fig.suptitle('CBIR Model Comparison: ResNet50 vs ViT-B/16', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path4 = os.path.join(RESULTS_DIR, "comparison_dashboard.png")
    plt.savefig(path4, dpi=200, facecolor='white', edgecolor='none')
    plt.close(fig)
    output_paths.append(path4)
    
    return output_paths


def main(max_queries: int | None) -> None:
    results: Dict[str, Dict[str, Dict[str, float]]] = {}

    for dataset, resnet_feat_path in DATASETS.items():
        resnet_paths_path = resnet_feat_path.replace("_features.pkl", "_paths.pkl")
        vit_feat_path = VIT_FEATURES[dataset]
        vit_paths_path = vit_feat_path.replace("_features.pkl", "_paths.pkl")

        resnet_feats, resnet_paths = load_features(resnet_feat_path, resnet_paths_path)
        vit_feats, vit_paths = load_features(vit_feat_path, vit_paths_path)

        resnet_map, resnet_lat = evaluate_model(resnet_feats, resnet_paths, max_queries=max_queries)
        vit_map, vit_lat = evaluate_model(vit_feats, vit_paths, max_queries=max_queries)

        results[dataset] = {
            "ResNet50": {"mAP": resnet_map, "latency_ms": resnet_lat},
            "ViT-B/16": {"mAP": vit_map, "latency_ms": vit_lat},
        }

    table_md = format_markdown_table(results)
    graph_paths = plot_accuracy_speed(results)

    print("\n📋 Markdown Table (copy to docs):\n")
    print(table_md)
    print(f"\n📊 Graphs saved:")
    for path in graph_paths:
        print(f"   • {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare ResNet50 vs ViT-B/16 across datasets")
    parser.add_argument("--max-queries", type=int, default=None, help="Limit number of queries for a quicker run")
    args = parser.parse_args()
    main(max_queries=args.max_queries)
