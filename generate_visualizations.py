"""
Generate PCA Scree Plot and SVM RBF Kernel Visualization
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.datasets import make_circles
from sklearn.preprocessing import StandardScaler


# ============================================================================
# 1. PCA CUMULATIVE VARIANCE PLOT (SCREE PLOT)
# ============================================================================

def load_dataset():
    """Load the brain tumor MRI dataset."""
    DATASET_LABELS = {
        'no_tumor': 0,
        'pituitary_tumor': 1,
        'meningioma_tumor': 2,
        'glioma_tumor': 3,
    }
    IMAGE_SIZE = (200, 200)
    
    images = []
    labels = []

    for class_name, class_id in DATASET_LABELS.items():
        class_path = f'./{class_name}'
        if not os.path.isdir(class_path):
            continue

        for filename in os.listdir(class_path):
            image_path = os.path.join(class_path, filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue

            image = cv2.resize(image, IMAGE_SIZE)
            images.append(image)
            labels.append(class_id)

    if not images:
        raise ValueError('No training images found.')

    return np.array(images), np.array(labels)


def generate_pca_plot():
    """Generate PCA cumulative variance (scree) plot."""
    print("Loading dataset for PCA analysis...")
    x, y = load_dataset()
    x_flat = x.reshape(len(x), -1) / 255.0
    
    print("Fitting PCA with 98% variance...")
    pca = PCA(0.98)
    pca.fit(x_flat)
    
    # Calculate cumulative variance
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    n_components = len(cumsum)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    
    # Plot cumulative variance line
    ax.plot(range(1, n_components + 1), cumsum, 'o-', 
            color='#0f766e', linewidth=2.5, markersize=4, label='Cumulative Variance')
    
    # Mark 98% threshold
    threshold_idx = np.argmax(cumsum >= 0.98)
    ax.axhline(y=0.98, color='#dc2626', linestyle='--', linewidth=2, label='98% Threshold')
    ax.axvline(x=threshold_idx + 1, color='#0284c7', linestyle=':', linewidth=2, 
               label=f'Optimal Components: {threshold_idx + 1}')
    
    # Highlight the cutoff point
    ax.plot(threshold_idx + 1, cumsum[threshold_idx], 'r*', markersize=20, 
            markeredgecolor='#0f766e', markeredgewidth=1.5)
    
    ax.set_xlabel('Number of PCA Components', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative Explained Variance Ratio', fontsize=12, fontweight='bold')
    ax.set_title('PCA Cumulative Variance Plot (Scree Plot)', fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.legend(loc='lower right', fontsize=10, framealpha=0.95)
    ax.set_ylim([0, 1.05])
    ax.set_xlim([0, min(n_components + 5, 150)])
    
    # Add annotation
    ax.annotate(f'{threshold_idx + 1} components\nexplain 98% variance',
                xy=(threshold_idx + 1, cumsum[threshold_idx]),
                xytext=(threshold_idx + 20, 0.88),
                fontsize=10,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#f1f5f9', edgecolor='#0f766e', linewidth=1.5),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3', 
                               color='#0f766e', lw=1.5))
    
    fig.tight_layout()
    fig.savefig('./static/pca_scree_plot.png', dpi=100, bbox_inches='tight', facecolor='white')
    print(f"✓ PCA plot saved to ./static/pca_scree_plot.png")
    print(f"  → {threshold_idx + 1} components capture 98% variance")
    plt.close(fig)


# ============================================================================
# 2. SVM RBF KERNEL VISUALIZATION
# ============================================================================

def generate_rbf_visualization():
    """Generate 3D SVM RBF kernel visualization."""
    print("\nGenerating RBF Kernel Visualization...")
    
    # Create non-linearly separable 2D data
    np.random.seed(42)
    X, y = make_circles(n_samples=300, noise=0.1, factor=0.3)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Train SVM with RBF kernel
    svm = SVC(kernel='rbf', gamma='scale', C=1.0)
    svm.fit(X, y)
    
    # Create 3D visualization
    fig = plt.figure(figsize=(12, 9), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    
    # RBF transformation: add a third dimension using RBF kernel trick
    # For visualization, we compute the RBF "height" for each point
    gamma = svm._gamma if hasattr(svm, '_gamma') else 1.0 / X.shape[1]
    
    # Compute RBF kernel to a reference point (origin)
    Z = np.exp(-gamma * np.sum(X**2, axis=1))
    
    # Plot the two classes in 3D space
    scatter1 = ax.scatter(X[y == 0, 0], X[y == 0, 1], Z[y == 0], 
                         c='#0f766e', s=80, alpha=0.7, label='Class 0', edgecolors='#115e59', linewidth=0.5)
    scatter2 = ax.scatter(X[y == 1, 0], X[y == 1, 1], Z[y == 1], 
                         c='#0284c7', s=80, alpha=0.7, label='Class 1', edgecolors='#0c4a6e', linewidth=0.5)
    
    # Create a mesh to visualize the separation surface
    h = 0.05
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Compute RBF height for mesh
    mesh_z = np.exp(-gamma * (xx**2 + yy**2))
    
    # Plot the surface
    ax.plot_surface(xx, yy, mesh_z, alpha=0.3, cmap='coolwarm', 
                   edgecolor='none', rstride=5, cstride=5)
    
    ax.set_xlabel('Feature 1', fontsize=11, fontweight='bold')
    ax.set_ylabel('Feature 2', fontsize=11, fontweight='bold')
    ax.set_zlabel('RBF Kernel Height\n(Mapped to Higher Dimension)', fontsize=11, fontweight='bold')
    ax.set_title('SVM RBF Kernel: 2D Non-linear Data Mapped to 3D Space', 
                fontsize=14, fontweight='bold', pad=20)
    
    ax.legend(loc='upper left', fontsize=10, framealpha=0.95)
    ax.view_init(elev=25, azim=45)
    
    fig.tight_layout()
    fig.savefig('./static/svm_rbf_kernel.png', dpi=100, bbox_inches='tight', facecolor='white')
    print(f"✓ RBF Kernel plot saved to ./static/svm_rbf_kernel.png")
    plt.close(fig)


# ============================================================================
# 3. PERFORMANCE COMPARISON BAR CHART
# ============================================================================

def generate_performance_chart():
    """Generate performance comparison bar chart."""
    print("\nGenerating Performance Comparison Chart...")
    
    # Model accuracy data
    models = ['Logistic Regression', 'SVM']
    accuracies = [95.36, 93.73]
    colors = ['#0f766e', '#0284c7']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    
    # Create bars
    bars = ax.bar(models, accuracies, color=colors, alpha=0.85, edgecolor='#0f172a', linewidth=2)
    
    # Customize bars with value labels on top
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{acc}%',
                ha='center', va='bottom', fontsize=13, fontweight='bold', color='#0f172a')
    
    # Set labels and title
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Comparison: Brain Tumor Detection', 
                fontsize=14, fontweight='bold', pad=20)
    
    # Set Y-axis to go from 0 to 100
    ax.set_ylim(0, 105)
    ax.set_yticks(np.arange(0, 101, 10))
    ax.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Add a horizontal line at 90% for reference
    ax.axhline(y=90, color='#dc2626', linestyle='--', linewidth=1.5, alpha=0.6, label='90% Threshold')
    
    # Enhance appearance
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='lower right', fontsize=10, framealpha=0.95)
    
    # Add difference annotation
    diff = accuracies[0] - accuracies[1]
    ax.text(0.5, 85, f'Difference: +{diff:.2f}%', 
           ha='center', fontsize=11, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.6', facecolor='#f1f5f9', edgecolor='#0f766e', linewidth=1.5))
    
    fig.tight_layout()
    fig.savefig('./static/performance_comparison.png', dpi=100, bbox_inches='tight', facecolor='white')
    print(f"✓ Performance chart saved to ./static/performance_comparison.png")
    plt.close(fig)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("Brain Tumor Detection: Generating Visualization Plots")
    print("=" * 70)
    
    try:
        generate_pca_plot()
        generate_rbf_visualization()
        generate_performance_chart()
        print("\n" + "=" * 70)
        print("✓ All visualizations generated successfully!")
        print("=" * 70)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
