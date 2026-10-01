"""Shared helpers for the face-recognition notebooks.

These functions were previously copied into each notebook; they are unchanged here.
"""
import numpy as np
import matplotlib.pyplot as plt


def compute_pca(data_matrix, num_components=None, variance_threshold=None):
    """
    Perform PCA on a data matrix (samples × features).
    
    Parameters:
        data_matrix: np.ndarray, shape (n_samples, n_features)
        num_components: int or None — number of components to keep
        variance_threshold: float or None — keep components that explain up to this cumulative variance (e.g., 0.95)
    
    Returns:
        pca_result: projected data, shape (n_samples, num_components)
        components: principal components (eigenvectors)
        explained_variance_ratio: array of variance explained by each component
        mean: mean of original data (for inverse transform if needed)
    """
    # Step 1: Center the data
    mean = np.mean(data_matrix, axis=0)
    centered_data = data_matrix - mean

    # Step 2: Covariance matrix
    cov_matrix = np.cov(centered_data, rowvar=False)

    # Step 3: Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Step 4: Sort in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Step 5: Compute explained variance
    explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
    cumulative_variance = np.cumsum(explained_variance_ratio)

    # Step 6: Determine number of components
    if variance_threshold is not None:
        num_components = np.argmax(cumulative_variance >= variance_threshold) + 1
        print(f"Using {num_components} components to explain {variance_threshold*100:.1f}% variance.")
    elif num_components is None:
        num_components = data_matrix.shape[1]  # Keep all

    # Step 7: Select components and project
    selected_components = eigenvectors[:, :num_components]
    pca_result = np.dot(centered_data, selected_components)

    return pca_result, selected_components, explained_variance_ratio[:num_components], mean


def compute_mda(data_matrix, labels, num_components=None):
    """
    Perform MDA (also known as LDA) on the given data.

    Parameters:
        data_matrix: np.ndarray of shape (n_samples, n_features)
        labels: array-like of shape (n_samples,)
        num_components: int or None — number of components to retain (must be ≤ n_classes - 1)

    Returns:
        mda_result: Projected data of shape (n_samples, num_components)
        components: Eigenvectors used for projection (n_features, num_components)
        eigenvalues: Corresponding eigenvalues
        overall_mean: Mean of the original data
    """
    n_samples, n_features = data_matrix.shape
    unique_classes = np.unique(labels)
    n_classes = len(unique_classes)

    # Step 1: Center the data
    overall_mean = np.mean(data_matrix, axis=0)
    centered_data = data_matrix - overall_mean

    # Step 2: Compute class means
    class_means = []
    for c in unique_classes:
        class_data = data_matrix[labels == c]
        class_mean = np.mean(class_data, axis=0)
        class_means.append(class_mean)

    # Step 3: Compute between-class scatter matrix (S_B)
    S_B = np.zeros((n_features, n_features))
    for i, c in enumerate(unique_classes):
        n_i = np.sum(labels == c)
        mean_diff = (class_means[i] - overall_mean).reshape(-1, 1)
        S_B += (n_i / n_samples) * (mean_diff @ mean_diff.T)

    # Step 4: Compute within-class scatter matrix (S_W)
    S_W = np.zeros((n_features, n_features))
    for i, c in enumerate(unique_classes):
        class_data = data_matrix[labels == c]
        n_i = class_data.shape[0]
        class_centered = class_data - class_means[i]
        S_W += (n_i / n_samples) * (class_centered.T @ class_centered) / n_i

    # Step 5: Solve generalized eigenvalue problem
    S_W_inv = np.linalg.pinv(S_W)
    eig_matrix = S_W_inv @ S_B
    eigenvalues, eigenvectors = np.linalg.eigh(eig_matrix)

    # Step 6: Sort eigenvectors by eigenvalue magnitude (descending)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Step 7: Limit number of components
    max_components = n_classes - 1
    if num_components is None or num_components > max_components:
        num_components = max_components
        print(f"Using max possible components for MDA: {num_components}")

    selected_components = eigenvectors[:, :num_components]
    mda_result = centered_data @ selected_components

    return mda_result, selected_components, eigenvalues[:num_components], overall_mean


def separate_train_test_manual(data, labels, sub_num_total, train_num, task_num, random_state=42):
    np.random.seed(random_state)
    
    if task_num == 1:  # Person identification with all 3 images
        selected_subjects = np.random.choice(sub_num_total, train_num, replace=False)
        train_indices = []
        test_indices = []

        # For each subject, put 2 images in training and 1 in testing
        for s in selected_subjects:
            base = 3 * s
            train_indices.extend([base, base + 1])  # First 2 images to training
            test_indices.append(base + 2)           # Last image to testing

        train_set = data[train_indices]
        train_labels = labels[train_indices]
        test_set = data[test_indices]
        test_labels = labels[test_indices]

    elif task_num == 2:
        # Get indices for neutral and expression images
        neutral_indices = list(range(0, 3 * sub_num_total, 3))     # Images at positions 0, 3, 6, ...
        expression_indices = list(range(1, 3 * sub_num_total, 3))  # Images at positions 1, 4, 7, ...
        
        # Create binary labels (0 for neutral, 1 for expression)
        neutral_labels = np.zeros(len(neutral_indices), dtype=int)
        expression_labels = np.ones(len(expression_indices), dtype=int)
        
        # Shuffle each set of indices separately
        np.random.shuffle(neutral_indices)
        np.random.shuffle(expression_indices)
        
        # Split each class into training (80%) and testing (20%)
        neutral_split = int(len(neutral_indices) * 0.8)
        expression_split = int(len(expression_indices) * 0.8)
        
        # Create training and testing sets for each class
        neutral_train = neutral_indices[:neutral_split]
        neutral_test = neutral_indices[neutral_split:]
        expression_train = expression_indices[:expression_split]
        expression_test = expression_indices[expression_split:]
        
        # Combine indices and labels
        train_indices = np.concatenate([neutral_train, expression_train])
        test_indices = np.concatenate([neutral_test, expression_test])
        
        # Create labels matching the indices
        train_labels = np.concatenate([np.zeros(len(neutral_train)), np.ones(len(expression_train))])
        test_labels = np.concatenate([np.zeros(len(neutral_test)), np.ones(len(expression_test))])
        
        # Extract the data
        train_set = data[train_indices]
        test_set = data[test_indices]
    
    else:
        raise ValueError("❌ Invalid task_num. Use 1 for person ID or 2 for expression classification.")
    return train_set, train_labels, test_set, test_labels


def plot_classification_results(y_pred, y_true, task_num=1):
    # Calculate accuracy
    accuracy = np.mean(y_pred == y_true) * 100
    
    # Create a figure
    plt.figure(figsize=(12, 6))
    
    # Create an array to represent if predictions were correct
    correct = y_pred == y_true
    incorrect = ~correct
    
    # Create indices for x-axis
    x = np.arange(len(y_true))
    
    # Plot correctly classified points at y=1
    plt.scatter(x[correct], np.ones_like(x[correct]), color='green', marker='o', s=80, 
                label=f'Correctly classified ({sum(correct)} samples)', alpha=0.7)
    
    # Plot incorrectly classified points at y=2
    plt.scatter(x[incorrect], np.ones_like(x[incorrect])*2, color='red', marker='x', s=80, 
                label=f'Incorrectly classified ({sum(incorrect)} samples)', alpha=0.7)
    
    # Add title and labels
    plt.title(f"Classification Results (Accuracy: {accuracy:.2f}%)")
    plt.xlabel("Test sample index")
    plt.ylabel("Classification result")
    
    # Set y-ticks to 1 and 2 with custom labels
    plt.yticks([1, 2], ['Correct', 'Incorrect'])
    
    # Set y-limits with some padding
    plt.ylim(0.5, 2.5)
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def rbf_kernel(x, y, sigma=1.0):
    return np.exp(-np.linalg.norm(x - y) ** 2 / (2 * sigma ** 2))


def poly_kernel(x, y, degree=3):
    return (np.dot(x, y) + 1) ** degree


def poly_kernel_reverse(x, y, degree=3):
    return (np.dot(x, y) + 1) ** (1/degree)


def linear_kernel (x,y):
    return np.dot (x,y)
