"""
Research Visualization Script for CBIR System
==============================================
Generates publication-quality graphs for research papers.

Author: CBIR Research Project
Date: February 2026
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import PercentFormatter

# ============================================================================
# CONFIGURATION
# ============================================================================

# If you haven't run batch_evaluate.py yet, use these default values
# These will be overwritten if evaluation_results.pkl exists

DEFAULT_RESULTS = {
    "Corel-1K": {
        "num_images": 1000,
        "num_categories": 10,
        "mAP": 97.63,
        "latency_ms": 0.5
    },
    "Corel-5K": {
        "num_images": 5000,
        "num_categories": 50,
        "mAP": 88.50,  # Estimated
        "latency_ms": 2.5
    },
    "Corel-10K": {
        "num_images": 10000,
        "num_categories": 100,
        "mAP": 81.92,
        "latency_ms": 5.0
    },
    "Caltech-101": {
        "num_images": 9000,
        "num_categories": 102,
        "mAP": 78.00,  # Estimated
        "latency_ms": 4.5
    }
}

OUTPUT_DIR = "graphs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# LOAD RESULTS
# ============================================================================

results_file = "features/evaluation_results.pkl"

if os.path.exists(results_file):
    print("Loading evaluation results...")
    with open(results_file, 'rb') as f:
        results = pickle.load(f)
else:
    print("Using default results (run batch_evaluate.py for actual values)...")
    results = DEFAULT_RESULTS

# ============================================================================
# STYLE CONFIGURATION
# ============================================================================

# Modern clean style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 15,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 18,
    'figure.dpi': 150,
    'figure.facecolor': 'white',
    'axes.facecolor': '#FAFBFC',
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.edgecolor': '#D1D5DB',
    'axes.linewidth': 1.2,
    'grid.color': '#E5E7EB',
    'grid.linestyle': '-',
    'grid.linewidth': 0.8,
})

# Modern color palette (colorblind-friendly)
COLORS = {
    'primary': '#3B82F6',    # Blue
    'secondary': '#10B981',  # Green
    'tertiary': '#F59E0B',   # Amber
    'quaternary': '#EF4444', # Red
    'purple': '#8B5CF6',     # Purple
    'pink': '#EC4899',       # Pink
    'success': '#059669',    
    'warning': '#D97706',
    'grid': '#E5E7EB',
    'text': '#374151',
    'muted': '#9CA3AF'
}

DATASET_COLORS = ['#3B82F6', '#10B981', '#F59E0B', '#8B5CF6']
GRADIENT_COLORS = ['#60A5FA', '#34D399', '#FBBF24', '#A78BFA']

# ============================================================================
# GRAPH 1: SCALABILITY LINE GRAPH
# ============================================================================

def plot_scalability_graph():
    """Plot mAP vs Dataset Size with modern styling."""
    
    print("\n[1/5] Generating Scalability Line Graph...")
    
    fig, ax = plt.subplots(figsize=(11, 6))
    
    # Sort by dataset size
    sorted_data = sorted(results.items(), key=lambda x: x[1]['num_images'])
    
    names = [d[0] for d in sorted_data]
    sizes = [d[1]['num_images'] for d in sorted_data]
    maps = [d[1]['mAP'] for d in sorted_data]
    
    # Create gradient area fill
    ax.fill_between(sizes, maps, alpha=0.15, color=COLORS['primary'])
    
    # Plot main line
    line = ax.plot(sizes, maps, 
                   marker='o', 
                   markersize=14, 
                   linewidth=3,
                   color=COLORS['primary'],
                   markerfacecolor='white',
                   markeredgewidth=3,
                   markeredgecolor=COLORS['primary'],
                   zorder=5)
    
    # Add data labels with background
    for i, (name, size, mAP) in enumerate(zip(names, sizes, maps)):
        # mAP value above point
        ax.annotate(f'{mAP:.1f}%', 
                    xy=(size, mAP),
                    xytext=(0, 20),
                    textcoords='offset points',
                    ha='center',
                    fontweight='bold',
                    fontsize=12,
                    color=COLORS['primary'],
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                              edgecolor=COLORS['primary'], alpha=0.9))
        # Dataset name below
        ax.annotate(f'{name}',
                    xy=(size, mAP),
                    xytext=(0, -25),
                    textcoords='offset points',
                    ha='center',
                    fontsize=10,
                    fontweight='medium',
                    color=COLORS['text'])
    
    # Formatting
    ax.set_xlabel('Dataset Size (Number of Images)', fontweight='bold', color=COLORS['text'])
    ax.set_ylabel('Mean Average Precision (%)', fontweight='bold', color=COLORS['text'])
    ax.set_title('CBIR System Scalability Analysis\nResNet50 Feature Extraction', 
                 fontweight='bold', pad=20, color=COLORS['text'])
    
    ax.set_ylim(min(maps) - 10, 105)
    ax.set_xlim(0, max(sizes) * 1.15)
    
    # X-axis formatting
    ax.set_xticks(sizes)
    ax.set_xticklabels([f'{s:,}' for s in sizes], fontweight='medium')
    
    # Grid
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    
    # Performance region annotation
    ax.axhspan(90, 105, alpha=0.1, color=COLORS['success'], zorder=0)
    ax.text(sizes[0] + 200, 92, '✓ High Performance Zone (>90%)', 
            fontsize=9, color=COLORS['success'], fontweight='medium')
    
    plt.tight_layout()
    
    # Save
    filepath = os.path.join(OUTPUT_DIR, 'scalability_graph.png')
    plt.savefig(filepath, facecolor='white', edgecolor='none')
    plt.savefig(os.path.join(OUTPUT_DIR, 'scalability_graph.pdf'), facecolor='white')
    print(f"   ✓ Saved: {filepath}")
    
    plt.close()

# ============================================================================
# GRAPH 2: PRECISION BAR CHART
# ============================================================================

def plot_precision_bar_chart():
    """Plot modern bar chart comparing datasets."""
    
    print("[2/5] Generating Precision Bar Chart...")
    
    fig, ax = plt.subplots(figsize=(11, 6))
    
    # Sort by dataset size
    sorted_data = sorted(results.items(), key=lambda x: x[1]['num_images'])
    
    names = [d[0] for d in sorted_data]
    maps = [d[1]['mAP'] for d in sorted_data]
    
    x = np.arange(len(names))
    width = 0.65
    
    # Create bars with rounded edges effect
    bars = ax.bar(x, maps, width, color=DATASET_COLORS[:len(names)], 
                  edgecolor='white', linewidth=2, zorder=3)
    
    # Add gradient overlay effect (lighter shade on top)
    for bar, gcolor in zip(bars, GRADIENT_COLORS[:len(names)]):
        bar.set_alpha(0.9)
    
    # Add value labels on bars
    for bar, mAP in zip(bars, maps):
        height = bar.get_height()
        # Big value on top
        ax.annotate(f'{mAP:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 8),
                    textcoords='offset points',
                    ha='center', va='bottom',
                    fontweight='bold',
                    fontsize=14,
                    color=COLORS['text'])
    
    # Add dataset info inside bars
    for i, (name, data) in enumerate(sorted_data):
        ax.annotate(f'{data["num_images"]:,}\nimages',
                    xy=(i, maps[i] / 2),
                    ha='center', va='center',
                    fontsize=10,
                    color='white',
                    fontweight='bold',
                    alpha=0.95)
    
    # Formatting
    ax.set_xlabel('Dataset', fontweight='bold', color=COLORS['text'])
    ax.set_ylabel('Mean Average Precision (%)', fontweight='bold', color=COLORS['text'])
    ax.set_title('Precision Comparison Across Datasets\nCBIR using ResNet50 Features',
                 fontweight='bold', pad=20, color=COLORS['text'])
    
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontweight='bold')
    ax.set_ylim(0, max(maps) + 15)
    
    # Baseline reference lines
    ax.axhline(y=90, color=COLORS['success'], linestyle='--', linewidth=2, alpha=0.6, zorder=2)
    ax.axhline(y=80, color=COLORS['warning'], linestyle='--', linewidth=2, alpha=0.6, zorder=2)
    
    # Baseline labels
    ax.text(len(names) - 0.3, 91.5, '90% Excellent', fontsize=9, 
            color=COLORS['success'], fontweight='medium')
    ax.text(len(names) - 0.3, 81.5, '80% Good', fontsize=9, 
            color=COLORS['warning'], fontweight='medium')
    
    # Grid
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    # Save
    filepath = os.path.join(OUTPUT_DIR, 'precision_comparison.png')
    plt.savefig(filepath, facecolor='white', edgecolor='none')
    plt.savefig(os.path.join(OUTPUT_DIR, 'precision_comparison.pdf'), facecolor='white')
    print(f"   ✓ Saved: {filepath}")
    
    plt.close()

# ============================================================================
# GRAPH 3: LATENCY VS ACCURACY TRADE-OFF
# ============================================================================

def plot_latency_accuracy():
    """Plot modern bubble chart for latency vs accuracy."""
    
    print("[3/5] Generating Latency-Accuracy Trade-off Graph...")
    
    fig, ax = plt.subplots(figsize=(11, 7))
    
    sorted_data = sorted(results.items(), key=lambda x: x[1]['num_images'])
    
    names = [d[0] for d in sorted_data]
    maps = [d[1]['mAP'] for d in sorted_data]
    latencies = [d[1].get('latency_ms', 1.0) for d in sorted_data]
    sizes = [d[1]['num_images'] for d in sorted_data]
    
    # Bubble chart - size based on dataset size
    for i, (name, lat, mAP, size) in enumerate(zip(names, latencies, maps, sizes)):
        scatter = ax.scatter(lat, mAP, 
                             s=size/8,  # Scale bubble size
                             c=DATASET_COLORS[i],
                             alpha=0.75,
                             edgecolors='white',
                             linewidth=2.5,
                             zorder=3)
        
        # Add labels with background
        ax.annotate(f'{name}\n{mAP:.1f}%',
                    xy=(lat, mAP),
                    xytext=(15, 0),
                    textcoords='offset points',
                    fontsize=10,
                    fontweight='bold',
                    color=COLORS['text'],
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                              edgecolor=DATASET_COLORS[i], alpha=0.9),
                    zorder=4)
    
    # Formatting
    ax.set_xlabel('Average Retrieval Latency (ms) → Lower is Better', 
                  fontweight='bold', color=COLORS['text'])
    ax.set_ylabel('Mean Average Precision (%) → Higher is Better', 
                  fontweight='bold', color=COLORS['text'])
    ax.set_title('Accuracy vs Speed Trade-off Analysis\nBubble size represents dataset size',
                 fontweight='bold', pad=20, color=COLORS['text'])
    
    ax.set_ylim(min(maps) - 8, max(maps) + 8)
    ax.grid(True, zorder=0)
    ax.set_axisbelow(True)
    
    # Ideal region indicator
    ax.annotate('← IDEAL\n(Fast & Accurate)', 
                xy=(ax.get_xlim()[0] + 0.2, ax.get_ylim()[1] - 3),
                fontsize=10, color=COLORS['success'], fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='#D1FAE5', edgecolor=COLORS['success']))
    
    # Legend for bubble sizes
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['muted'], 
               markersize=8, label='1K images'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['muted'],
               markersize=14, label='5K images'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['muted'],
               markersize=20, label='10K images')
    ]
    ax.legend(handles=legend_elements, loc='lower right', title='Dataset Size',
              framealpha=0.95, edgecolor=COLORS['grid'])
    
    plt.tight_layout()
    
    # Save
    filepath = os.path.join(OUTPUT_DIR, 'latency_accuracy.png')
    plt.savefig(filepath, facecolor='white', edgecolor='none')
    plt.savefig(os.path.join(OUTPUT_DIR, 'latency_accuracy.pdf'), facecolor='white')
    print(f"   ✓ Saved: {filepath}")
    
    plt.close()

# ============================================================================
# GRAPH 4: COMBINED METRICS DASHBOARD
# ============================================================================

def plot_combined_metrics():
    """Plot modern combined metrics dashboard."""
    
    print("[4/5] Generating Combined Metrics Dashboard...")
    
    fig = plt.figure(figsize=(14, 8))
    
    # Create grid layout
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3, 
                          height_ratios=[1, 1], width_ratios=[1.5, 1, 1])
    
    sorted_data = sorted(results.items(), key=lambda x: x[1]['num_images'])
    
    names = [d[0] for d in sorted_data]
    maps = [d[1]['mAP'] for d in sorted_data]
    sizes = [d[1]['num_images'] for d in sorted_data]
    categories = [d[1]['num_categories'] for d in sorted_data]
    latencies = [d[1].get('latency_ms', 1.0) for d in sorted_data]
    
    # -------------------------------------------------------------------------
    # Panel 1: Horizontal bar chart (Performance ranking)
    # -------------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    
    bars = ax1.barh(names, maps, color=DATASET_COLORS[:len(names)], 
                    edgecolor='white', linewidth=2, height=0.6, zorder=3)
    
    for i, (bar, mAP) in enumerate(zip(bars, maps)):
        ax1.text(bar.get_width() + 1.5, bar.get_y() + bar.get_height()/2,
                 f'{mAP:.1f}%', va='center', fontweight='bold', fontsize=12,
                 color=COLORS['text'])
        # Add indicator for high performers
        if mAP >= 90:
            ax1.text(5, bar.get_y() + bar.get_height()/2, '★',
                     va='center', fontsize=14, color='white')
    
    ax1.set_xlabel('mAP (%)', fontweight='bold', color=COLORS['text'])
    ax1.set_title('Performance Ranking', fontweight='bold', pad=10)
    ax1.set_xlim(0, max(maps) + 12)
    ax1.xaxis.grid(True, zorder=0)
    ax1.set_axisbelow(True)
    
    # Excellence line
    ax1.axvline(x=90, color=COLORS['success'], linestyle='--', linewidth=2, alpha=0.6)
    
    # -------------------------------------------------------------------------
    # Panel 2: Dataset stats cards
    # -------------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1:])
    ax2.axis('off')
    
    # Create stat cards
    card_data = [
        ('Total Images', f'{sum(sizes):,}', COLORS['primary'], ''),
        ('Avg mAP', f'{np.mean(maps):.1f}%', COLORS['secondary'], ''),
        ('Datasets', f'{len(names)}', COLORS['tertiary'], ''),
        ('Avg Latency', f'{np.mean(latencies):.1f}ms', COLORS['purple'], ''),
    ]
    
    for i, (label, value, color, icon) in enumerate(card_data):
        x = 0.12 + (i % 2) * 0.45
        y = 0.7 - (i // 2) * 0.45
        
        # Card background
        rect = plt.Rectangle((x - 0.08, y - 0.15), 0.4, 0.35, 
                              transform=ax2.transAxes, facecolor=color, 
                              alpha=0.1, edgecolor=color, linewidth=2)
        ax2.add_patch(rect)
        
        ax2.text(x + 0.12, y + 0.08, f'{value}', transform=ax2.transAxes,
                 fontsize=18, fontweight='bold', ha='center', color=color)
        ax2.text(x + 0.12, y - 0.05, label, transform=ax2.transAxes,
                 fontsize=10, ha='center', color=COLORS['muted'])
    
    ax2.set_title('Summary Statistics', fontweight='bold', pad=10)
    
    # -------------------------------------------------------------------------
    # Panel 3: Dataset composition (stacked info)
    # -------------------------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, :2])
    
    x = np.arange(len(names))
    width = 0.35
    
    # Two grouped bars: Images/100 and Categories
    bars1 = ax3.bar(x - width/2, [s/100 for s in sizes], width, 
                    label='Images (×100)', color=COLORS['primary'], 
                    edgecolor='white', linewidth=1.5, zorder=3, alpha=0.85)
    bars2 = ax3.bar(x + width/2, categories, width,
                    label='Categories', color=COLORS['secondary'],
                    edgecolor='white', linewidth=1.5, zorder=3, alpha=0.85)
    
    ax3.set_xlabel('Dataset', fontweight='bold', color=COLORS['text'])
    ax3.set_ylabel('Count', fontweight='bold', color=COLORS['text'])
    ax3.set_title('Dataset Composition', fontweight='bold', pad=10)
    ax3.set_xticks(x)
    ax3.set_xticklabels(names, fontweight='medium')
    ax3.legend(loc='upper left', framealpha=0.95, edgecolor=COLORS['grid'])
    ax3.yaxis.grid(True, zorder=0)
    ax3.set_axisbelow(True)
    
    # -------------------------------------------------------------------------
    # Panel 4: Latency comparison (lollipop chart)
    # -------------------------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 2])
    
    y_pos = np.arange(len(names))
    ax4.hlines(y=y_pos, xmin=0, xmax=latencies, color=COLORS['purple'], 
               linewidth=3, alpha=0.7)
    ax4.scatter(latencies, y_pos, s=150, color=COLORS['purple'], 
                edgecolors='white', linewidth=2, zorder=3)
    
    for i, lat in enumerate(latencies):
        ax4.text(lat + 0.3, i, f'{lat:.1f}ms', va='center', fontsize=10,
                 fontweight='bold', color=COLORS['text'])
    
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(names)
    ax4.set_xlabel('Latency (ms)', fontweight='bold', color=COLORS['text'])
    ax4.set_title('Retrieval Speed', fontweight='bold', pad=10)
    ax4.xaxis.grid(True, zorder=0)
    ax4.set_axisbelow(True)
    ax4.set_xlim(0, max(latencies) * 1.3)
    
    # Main title
    fig.suptitle('CBIR System Performance Dashboard', 
                 fontsize=18, fontweight='bold', y=0.98, color=COLORS['text'])
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save
    filepath = os.path.join(OUTPUT_DIR, 'combined_metrics.png')
    plt.savefig(filepath, facecolor='white', edgecolor='none')
    plt.savefig(os.path.join(OUTPUT_DIR, 'combined_metrics.pdf'), facecolor='white')
    print(f"   ✓ Saved: {filepath}")
    
    plt.close()


# ============================================================================
# GRAPH 5: RADAR CHART FOR MULTI-METRIC COMPARISON
# ============================================================================

def plot_radar_comparison():
    """Plot radar chart comparing datasets across multiple metrics."""
    
    print("[5/5] Generating Radar Comparison Chart...")
    
    sorted_data = sorted(results.items(), key=lambda x: x[1]['num_images'])
    
    names = [d[0] for d in sorted_data]
    
    # Normalize metrics to 0-100 scale
    maps = [d[1]['mAP'] for d in sorted_data]
    sizes = [d[1]['num_images'] for d in sorted_data]
    categories = [d[1]['num_categories'] for d in sorted_data]
    latencies = [d[1].get('latency_ms', 1.0) for d in sorted_data]
    
    # Normalize each metric
    max_size = max(sizes)
    max_cat = max(categories)
    max_lat = max(latencies)
    
    # Create normalized scores (higher is better, so invert latency)
    metrics = ['mAP (%)', 'Dataset\nSize', 'Categories', 'Speed\n(inverted)']
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Number of metrics
    num_vars = len(metrics)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop
    
    # Plot each dataset
    for i, (name, data) in enumerate(sorted_data):
        values = [
            data['mAP'],
            (data['num_images'] / max_size) * 100,
            (data['num_categories'] / max_cat) * 100,
            (1 - data.get('latency_ms', 1) / max_lat) * 100  # Invert: lower latency = higher score
        ]
        values += values[:1]  # Complete the loop
        
        ax.plot(angles, values, 'o-', linewidth=2.5, label=name, 
                color=DATASET_COLORS[i], markersize=8)
        ax.fill(angles, values, alpha=0.15, color=DATASET_COLORS[i])
    
    # Set labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=11, fontweight='bold', color=COLORS['text'])
    
    # Configure radial axis
    ax.set_ylim(0, 105)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(['25', '50', '75', '100'], fontsize=9, color=COLORS['muted'])
    ax.yaxis.grid(True, linestyle='-', alpha=0.3)
    ax.xaxis.grid(True, linestyle='-', alpha=0.3)
    
    # Legend
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1.1), 
              framealpha=0.95, edgecolor=COLORS['grid'])
    
    plt.title('Multi-Metric Dataset Comparison\n(Normalized Scores)', 
              fontsize=14, fontweight='bold', pad=20, color=COLORS['text'])
    
    plt.tight_layout()
    
    # Save
    filepath = os.path.join(OUTPUT_DIR, 'radar_comparison.png')
    plt.savefig(filepath, facecolor='white', edgecolor='none')
    plt.savefig(os.path.join(OUTPUT_DIR, 'radar_comparison.pdf'), facecolor='white')
    print(f"   ✓ Saved: {filepath}")
    
    plt.close()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("📊 CBIR RESEARCH VISUALIZATION GENERATOR")
    print("=" * 60)
    
    print(f"\nDatasets found: {list(results.keys())}")
    print(f"Output directory: {OUTPUT_DIR}/")
    
    # Generate all graphs
    plot_scalability_graph()
    plot_precision_bar_chart()
    plot_latency_accuracy()
    plot_combined_metrics()
    plot_radar_comparison()
    
    print("\n" + "=" * 60)
    print("✅ All visualizations generated successfully!")
    print("=" * 60)
    print(f"\n📁 Files saved in '{OUTPUT_DIR}/' folder:")
    print("   • scalability_graph.png/pdf")
    print("   • precision_comparison.png/pdf")
    print("   • latency_accuracy.png/pdf")
    print("   • combined_metrics.png/pdf")
    print("   • radar_comparison.png/pdf")
    print("\n💡 Use PDF versions for research papers (300 DPI).")
