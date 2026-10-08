# Pattern Recognition Projects (ENEE 633)

Two course projects from ENEE 633 (Statistical Pattern Recognition): image classification with classical machine learning and transfer learning, and face recognition with dimensionality reduction and kernel classifiers.

My KV-cache quantization project for LLaMA-2 is in its own repository: [kivi-kv-cache-quantization](https://github.com/kamirian/kivi-kv-cache-quantization).

---

## Projects

### 1. Image Classification — Classical ML and Transfer Learning

`image_classification/`

Benchmarks a progression of methods on MNIST (handwritten digits) and a monkey-species dataset (10 classes):

| Method | Dataset | Key result |
|--------|---------|-----------|
| SVM (RBF) + PCA (50 components) | MNIST | 97.1% test accuracy |
| Logistic regression from scratch (NumPy softmax) | MNIST | competitive with sklearn |
| Custom CNN | MNIST | 95.48% |
| VGG19 (frozen backbone) | Monkey species | 93.38% validation accuracy |
| VGG19 (fine-tuned) | Monkey species | 97.06% validation accuracy |

The notebook implements logistic regression from scratch using only NumPy (softmax activation, cross-entropy loss, gradient descent), providing a direct comparison against sklearn's L-BFGS solver. The VGG19 section demonstrates why fine-tuning outperforms frozen-feature extraction on a small domain-shift dataset.

**Data:** MNIST loads automatically. Monkey species dataset should be placed in `data/training/training/` and `data/validation/validation/`.

**Files:**
- `image_classification.ipynb` — full pipeline
- `report.pdf` — written report

---

### 2. Face Recognition — Dimensionality Reduction and Kernel Classifiers

`face_recognition/`

Evaluates classical face recognition methods on a 200-subject dataset (neutral, expression, illumination images; 24×21 pixels).

**Two tasks:**
- **Task 1:** 200-class person identification (2 training images per subject)
- **Task 2:** Binary neutral-vs-expression classification (80/20 split)

**Classifiers implemented from scratch:**

| Notebook | Method | Best Task 1 | Best Task 2 |
|----------|--------|------------|------------|
| `PCA.ipynb` | PCA projections of the data (visualization) | — | — |
| `MDA.ipynb` | MDA projections of the data (visualization) | — | — |
| `Bayes.ipynb` | Gaussian Bayes on raw, PCA and MDA features | 65.0% (MDA, 139 components) | 92.5% (MDA) |
| `K-NN.ipynb` | k-NN on raw, PCA and MDA features | 62.5% (MDA, k=3) | 91.25% (PCA, k=13; MDA, k=1) |
| `KernelSVM.ipynb` | Kernel SVM (CVXOPT and gradient descent) | — | 93.75% (RBF, CVXOPT); 91.25% (RBF, gradient descent) |
| `adaboost-svm.ipynb` | AdaBoost with SVM weak learners | — | 93.75% (PCA, 50 components) |

Shared helpers (`compute_pca`, `compute_mda`, `separate_train_test_manual`, `plot_classification_results`, and the SVM kernels) live in `face_utils.py`. Run each notebook from inside `face_recognition/` so it can import them.

**Custom SVM implementation:** CVXOPT (quadratic programming) showed numerical instability with high-degree polynomial kernels, motivating a custom **gradient-descent SVM** used throughout the AdaBoost experiments. `KernelSVM.ipynb` and `adaboost-svm.ipynb` keep the CVXOPT version for comparison; the AdaBoost results with the CVXOPT SVM are not in the report.

**Data:** `data.mat`, `pose.mat`, `illumination.mat` are included in the repository.

**Files:**
- `PCA.ipynb`, `Bayes.ipynb`, `K-NN.ipynb`, `KernelSVM.ipynb`, `MDA.ipynb`, `adaboost-svm.ipynb`
- `face_utils.py` — helper functions shared by the notebooks
- `report.pdf` — full written report

---

## Repository Structure

```
pattern-recognition-projects/
├── image_classification/
│   ├── image_classification.ipynb
│   └── report.pdf
├── face_recognition/
│   ├── PCA.ipynb
│   ├── Bayes.ipynb
│   ├── K-NN.ipynb
│   ├── KernelSVM.ipynb
│   ├── MDA.ipynb
│   ├── adaboost-svm.ipynb
│   ├── face_utils.py
│   ├── data.mat          (200 subjects × 3 images, 24×21 pixels)
│   ├── illumination.mat  (illumination variation subset)
│   ├── pose.mat          (pose variation subset)
│   └── report.pdf
└── requirements.txt
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Citation

```bibtex
@misc{amirian2025patternrecognition,
  author = {Kiyan Amirian},
  title  = {Pattern Recognition Projects: Image Classification and Face Recognition},
  year   = {2025},
  url    = {https://github.com/kamirian/pattern-recognition-projects}
}
```
