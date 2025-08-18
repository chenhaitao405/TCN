import json
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


def load_training_metrics(metrics_file):
    """Load training metrics from JSON file"""
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    return metrics


def create_training_report(checkpoint_dir, output_dir=None):
    """
    Create a comprehensive training report from saved metrics

    Args:
        checkpoint_dir: Directory containing training results
        output_dir: Directory to save report (default: same as checkpoint_dir)
    """
    if output_dir is None:
        output_dir = checkpoint_dir

    # Find metrics file
    metrics_file = os.path.join(checkpoint_dir, 'training_metrics.json')
    if not os.path.exists(metrics_file):
        print(f"Metrics file not found: {metrics_file}")
        return

    # Load metrics
    metrics = load_training_metrics(metrics_file)

    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 12))

    # Main title
    fig.suptitle(f'Training Report - {os.path.basename(checkpoint_dir)}',
                 fontsize=16, fontweight='bold')

    # Create subplots
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Loss curves (large plot)
    ax1 = fig.add_subplot(gs[0, :2])
    epochs = metrics['epochs']
    train_loss = metrics['train_loss']
    val_loss = metrics['val_loss']

    ax1.plot(epochs, train_loss, 'b-', label='Train Loss', linewidth=2, alpha=0.8)
    ax1.plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=2, alpha=0.8)
    ax1.fill_between(epochs, train_loss, val_loss,
                     where=(np.array(val_loss) > np.array(train_loss)),
                     alpha=0.2, color='green', label='Underfitting')
    ax1.fill_between(epochs, train_loss, val_loss,
                     where=(np.array(val_loss) <= np.array(train_loss)),
                     alpha=0.2, color='red', label='Overfitting')

    # Mark best epoch
    best_epoch = metrics.get('best_epoch', epochs[np.argmin(val_loss)])
    best_val = metrics.get('best_val_loss', min(val_loss))
    ax1.scatter([best_epoch], [best_val], color='gold', s=200,
                marker='*', zorder=5, label=f'Best: {best_val:.4f}')

    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('RMSE Loss', fontsize=12)
    ax1.set_title('Training Progress', fontsize=14)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # 2. Learning rate schedule
    ax2 = fig.add_subplot(gs[0, 2])
    lr = metrics['learning_rate']
    ax2.plot(epochs, lr, 'g-', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Learning Rate')
    ax2.set_title('Learning Rate Schedule')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)

    # 3. Loss reduction over time
    ax3 = fig.add_subplot(gs[1, 0])
    train_improvement = 100 * (1 - np.array(train_loss) / train_loss[0])
    val_improvement = 100 * (1 - np.array(val_loss) / val_loss[0])
    ax3.plot(epochs, train_improvement, 'b-', label='Train', linewidth=2)
    ax3.plot(epochs, val_improvement, 'r-', label='Validation', linewidth=2)
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Improvement (%)')
    ax3.set_title('Loss Reduction from Initial')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Smoothed loss curves (moving average)
    ax4 = fig.add_subplot(gs[1, 1])
    window = min(10, len(epochs) // 5)  # Window size for smoothing
    if window > 1:
        train_smooth = np.convolve(train_loss, np.ones(window) / window, mode='valid')
        val_smooth = np.convolve(val_loss, np.ones(window) / window, mode='valid')
        epochs_smooth = epochs[:len(train_smooth)]
        ax4.plot(epochs_smooth, train_smooth, 'b-', label='Train (smoothed)', linewidth=2)
        ax4.plot(epochs_smooth, val_smooth, 'r-', label='Val (smoothed)', linewidth=2)
    else:
        ax4.plot(epochs, train_loss, 'b-', label='Train', linewidth=2)
        ax4.plot(epochs, val_loss, 'r-', label='Val', linewidth=2)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('RMSE Loss')
    ax4.set_title(f'Smoothed Loss (window={window})')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. Convergence rate
    ax5 = fig.add_subplot(gs[1, 2])
    if len(epochs) > 1:
        train_rate = np.diff(train_loss)
        val_rate = np.diff(val_loss)
        ax5.plot(epochs[1:], train_rate, 'b-', label='Train', alpha=0.7)
        ax5.plot(epochs[1:], val_rate, 'r-', label='Val', alpha=0.7)
        ax5.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax5.set_xlabel('Epoch')
        ax5.set_ylabel('Loss Change')
        ax5.set_title('Convergence Rate (∆Loss)')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

    # 6. Loss statistics table
    ax6 = fig.add_subplot(gs[2, :2])
    ax6.axis('tight')
    ax6.axis('off')

    # Calculate statistics
    stats_data = []
    stats_data.append(['Metric', 'Train', 'Validation'])
    stats_data.append(['Initial Loss', f'{train_loss[0]:.4f}', f'{val_loss[0]:.4f}'])
    stats_data.append(['Final Loss', f'{train_loss[-1]:.4f}', f'{val_loss[-1]:.4f}'])
    stats_data.append(['Best Loss', f'{min(train_loss):.4f}', f'{min(val_loss):.4f}'])
    stats_data.append(['Mean Loss', f'{np.mean(train_loss):.4f}', f'{np.mean(val_loss):.4f}'])
    stats_data.append(['Std Dev', f'{np.std(train_loss):.4f}', f'{np.std(val_loss):.4f}'])

    # Calculate improvement
    train_improv = 100 * (train_loss[0] - train_loss[-1]) / train_loss[0]
    val_improv = 100 * (val_loss[0] - val_loss[-1]) / val_loss[0]
    stats_data.append(['Improvement', f'{train_improv:.1f}%', f'{val_improv:.1f}%'])

    # Add best epoch info
    stats_data.append(['Best Epoch', '-', str(best_epoch)])

    table = ax6.table(cellText=stats_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Style the header row
    for i in range(3):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    ax6.set_title('Training Statistics', fontsize=14, pad=20)

    # 7. Final vs Best comparison
    ax7 = fig.add_subplot(gs[2, 2])
    categories = ['Best\nVal Loss', 'Final\nVal Loss', 'Final\nTrain Loss']
    values = [min(val_loss), val_loss[-1], train_loss[-1]]
    colors = ['gold', 'red', 'blue']

    bars = ax7.bar(categories, values, color=colors, alpha=0.7, edgecolor='black')
    ax7.set_ylabel('RMSE Loss')
    ax7.set_title('Model Performance Comparison')

    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{val:.4f}', ha='center', va='bottom')

    ax7.grid(True, alpha=0.3, axis='y')

    # Save figure
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(output_dir, f'training_report_{timestamp}.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Training report saved to: {output_file}")

    plt.show()

    return output_file


def plot_loss_comparison(checkpoint_dirs, labels=None, output_file=None):
    """
    Compare training curves from multiple training runs

    Args:
        checkpoint_dirs: List of checkpoint directories
        labels: Optional labels for each run
        output_file: Path to save comparison plot
    """
    if labels is None:
        labels = [os.path.basename(d) for d in checkpoint_dirs]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Training Runs Comparison', fontsize=16, fontweight='bold')

    colors = plt.cm.tab10(np.linspace(0, 1, len(checkpoint_dirs)))

    for idx, (checkpoint_dir, label) in enumerate(zip(checkpoint_dirs, labels)):
        metrics_file = os.path.join(checkpoint_dir, 'training_metrics.json')
        if not os.path.exists(metrics_file):
            print(f"Skipping {label}: metrics file not found")
            continue

        metrics = load_training_metrics(metrics_file)
        epochs = metrics['epochs']
        train_loss = metrics['train_loss']
        val_loss = metrics['val_loss']
        color = colors[idx]

        # Plot train loss
        axes[0].plot(epochs, train_loss, color=color, label=label,
                     linewidth=2, alpha=0.8)

        # Plot validation loss
        axes[1].plot(epochs, val_loss, color=color, label=label,
                     linewidth=2, alpha=0.8)

        # Mark best validation
        best_idx = np.argmin(val_loss)
        axes[1].scatter(epochs[best_idx], val_loss[best_idx],
                        color=color, s=100, marker='*', zorder=5)

    # Configure subplots
    for ax, title in zip(axes, ['Training Loss', 'Validation Loss']):
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('RMSE Loss', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Comparison plot saved to: {output_file}")

    plt.show()


def analyze_convergence(checkpoint_dir):
    """
    Analyze convergence behavior and suggest improvements
    """
    metrics_file = os.path.join(checkpoint_dir, 'training_metrics.json')
    if not os.path.exists(metrics_file):
        print(f"Metrics file not found: {metrics_file}")
        return

    metrics = load_training_metrics(metrics_file)
    epochs = np.array(metrics['epochs'])
    train_loss = np.array(metrics['train_loss'])
    val_loss = np.array(metrics['val_loss'])
    lr = np.array(metrics['learning_rate'])

    print("\n" + "=" * 60)
    print("CONVERGENCE ANALYSIS")
    print("=" * 60)

    # 1. Overall improvement
    train_improve = 100 * (train_loss[0] - train_loss[-1]) / train_loss[0]
    val_improve = 100 * (val_loss[0] - val_loss[-1]) / val_loss[0]
    print(f"\nOverall Improvement:")
    print(f"  Train: {train_improve:.1f}%")
    print(f"  Validation: {val_improve:.1f}%")

    # 2. Best performance
    best_val_idx = np.argmin(val_loss)
    print(f"\nBest Performance:")
    print(f"  Epoch: {epochs[best_val_idx]}")
    print(f"  Validation Loss: {val_loss[best_val_idx]:.4f}")
    print(f"  Training Loss: {train_loss[best_val_idx]:.4f}")

    # 3. Overfitting analysis
    final_gap = val_loss[-1] - train_loss[-1]
    best_gap = val_loss[best_val_idx] - train_loss[best_val_idx]
    print(f"\nOverfitting Analysis:")
    print(f"  Final gap (val-train): {final_gap:.4f}")
    print(f"  Best epoch gap: {best_gap:.4f}")

    if final_gap > 0.1:
        print("  ⚠️ Significant overfitting detected")
    elif final_gap < -0.05:
        print("  ⚠️ Possible underfitting")
    else:
        print("  ✓ Good generalization")

    # 4. Convergence speed
    if len(epochs) > 10:
        early_improve = 100 * (val_loss[0] - val_loss[9]) / val_loss[0]
        late_improve = 100 * (val_loss[-10] - val_loss[-1]) / val_loss[-10] if val_loss[-10] != 0 else 0
        print(f"\nConvergence Speed:")
        print(f"  First 10 epochs improvement: {early_improve:.1f}%")
        print(f"  Last 10 epochs improvement: {late_improve:.1f}%")

        if late_improve < 1:
            print("  ℹ️ Training has plateaued")

    # 5. Learning rate analysis
    lr_changes = np.where(np.diff(lr) != 0)[0]
    print(f"\nLearning Rate:")
    print(f"  Initial: {lr[0]:.2e}")
    print(f"  Final: {lr[-1]:.2e}")
    print(f"  Number of reductions: {len(lr_changes)}")

    # 6. Recommendations
    print("\n" + "-" * 60)
    print("RECOMMENDATIONS:")
    print("-" * 60)

    recommendations = []

    if final_gap > 0.1:
        recommendations.append("• Add regularization (dropout, weight decay)")
        recommendations.append("• Reduce model complexity")
        recommendations.append("• Increase training data")

    if late_improve < 1 and lr[-1] > 1e-6:
        recommendations.append("• Continue training with lower learning rate")

    if val_improve < 10:
        recommendations.append("• Check data quality and preprocessing")
        recommendations.append("• Consider different model architecture")

    if best_val_idx < len(epochs) * 0.3:
        recommendations.append("• Implement early stopping")
        recommendations.append("• Reduce initial learning rate")

    if not recommendations:
        recommendations.append("✓ Training appears optimal")

    for rec in recommendations:
        print(rec)

    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Visualize training results')
    parser.add_argument('checkpoint_dir', type=str,
                        help='Path to checkpoint directory')
    parser.add_argument('--compare', type=str, nargs='+',
                        help='Compare with other checkpoint directories')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save visualizations')
    parser.add_argument('--analyze', action='store_true',
                        help='Perform convergence analysis')

    args = parser.parse_args()

    # Create training report
    print("Creating training report...")
    create_training_report(args.checkpoint_dir, args.output_dir)

    # Perform analysis if requested
    if args.analyze:
        analyze_convergence(args.checkpoint_dir)

    # Compare multiple runs if specified
    if args.compare:
        all_dirs = [args.checkpoint_dir] + args.compare
        print(f"\nComparing {len(all_dirs)} training runs...")
        output_file = os.path.join(
            args.output_dir or args.checkpoint_dir,
            'comparison.png'
        )
        plot_loss_comparison(all_dirs, output_file=output_file)


if __name__ == "__main__":
    main()