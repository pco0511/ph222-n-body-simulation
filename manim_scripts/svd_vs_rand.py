import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

fig_output_dir = os.path.dirname(os.path.abspath(__file__))

def generate_gaussian_mixture_samples(m, N, weights, means, covariances, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    
    k = len(weights)
    component_samples = np.random.multinomial(m, weights)
    samples = []
    for i in range(k):
        component_sample = np.random.multivariate_normal(
            mean=means[i],
            cov=covariances[i],
            size=component_samples[i]
        )
        samples.append(component_sample)
    samples = np.vstack(samples)
    indices = np.arange(m)
    np.random.shuffle(indices)
    samples = samples[indices]
    return samples

def get_top_right_singular_vectors(samples, num_vectors):
    centered_samples = samples - np.mean(samples, axis=0)
    U, S, Vt = np.linalg.svd(centered_samples, full_matrices=False)
    top_right_vectors = Vt[:num_vectors].T
    return top_right_vectors

def generate_random_orthonormal_vectors(d, n):
    np.random.seed(42)  # Ensure reproducibility of random vectors
    random_matrix = np.random.randn(d, n)
    q, _ = np.linalg.qr(random_matrix)  # QR decomposition to orthogonalize
    return q[:, :n]

def project_onto_subspace(samples, orthonormal_vectors):
    centered_samples = samples - np.mean(samples, axis=0)
    projected_samples = np.dot(centered_samples, orthonormal_vectors)
    return projected_samples

def visualize_samples(samples, title, name=None):
    kde = gaussian_kde(samples.T)
    density = kde(samples.T)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(samples[:, 0], samples[:, 1], c=density, s=10, cmap='Spectral', alpha=0.7)
    plt.colorbar(scatter, label='Density')  # Add colorbar
    plt.xticks([])
    plt.yticks([])
    plt.grid(True)
    plt.gca().set_frame_on(False)
    plt.title(title)
    if fig_output_dir:
        name = name or title.replace(' ', '_').lower()
        plt.savefig(os.path.join(fig_output_dir, f"{name}.png"))
    plt.show()

# Main logic
np.random.seed(42)  # Fix seed globally

N = 20
m = 5000  # Number of samples
k_mixture = 3

weights = [0.4, 0.35, 0.25]

# Generate means from a Gaussian with mean zero and covariance 18I
mean_covariance = np.eye(N) * 18  # Covariance matrix for generating means
generated_means = np.random.multivariate_normal(np.zeros(N), mean_covariance, k_mixture)

covariances = [
    np.eye(N) * 4,
    np.eye(N) * 9,
    np.eye(N) * 12
]

samples = generate_gaussian_mixture_samples(m, N, weights, generated_means, covariances, random_state=42)

# SVD-based projection
num_top_vectors = 2
top_vectors = get_top_right_singular_vectors(samples, num_top_vectors)
projected_svd_samples = project_onto_subspace(samples, top_vectors)

# Random orthonormal vector-based projection
random_vectors = generate_random_orthonormal_vectors(N, num_top_vectors)
projected_random_samples = project_onto_subspace(samples, random_vectors)

# Visualize and compare
visualize_samples(projected_svd_samples, title="Projection onto SVD Subspace", name="best-fit")
visualize_samples(projected_random_samples, title="Projection onto Random Orthonormal Subspace", name="random")
