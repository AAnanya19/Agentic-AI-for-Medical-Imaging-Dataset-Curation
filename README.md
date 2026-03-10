# Agentic AI for Medical Imaging Dataset Curation

An agentic AI pipeline for automated quality control and curation of medical imaging datasets. The system deploys four specialised agents to detect mislabelled images, flag low-quality scans, and recommend dataset balancing strategies — addressing one of the most critical bottlenecks in clinical AI development.

---

## Motivation

Data quality is a foundational challenge in medical AI. Mislabelled images, low-quality scans, and class imbalance can silently degrade model performance in high-stakes clinical settings. This project automates the dataset curation process using a multi-agent architecture, combining lightweight computer vision models with LLM-generated audit reports to make the process scalable and interpretable.

---

## System Architecture

The pipeline consists of four specialised agents working in sequence:

| Agent | Role |
|---|---|
| **Data Inspector** | No-reference image quality assessment (sharpness, noise, contrast) |
| **Label Auditor** | Embedding-based outlier detection to flag potentially mislabelled images |
| **Dataset Optimizer** | Class-balance analysis with resampling and augmentation recommendations |
| **LLM Explainer** | Generates human-readable audit reports summarising findings |

---

## Datasets

The system is evaluated on publicly available medical imaging benchmarks:

- **BraTS** — Brain tumour MRI segmentation
- **ISIC** — Skin lesion dermoscopy images
- **NIH ChestX-ray14** — Chest X-ray multi-label classification
- **CheXpert** — Chest radiograph classification

---

## Features

- **Image Quality Assessment:** No-reference IQA metrics (sharpness via Laplacian variance, noise estimation, contrast analysis)
- **Outlier Detection:** Embedding-based methods — kNN, Mahalanobis distance, Isolation Forest — to surface mislabelled or anomalous samples
- **Class Imbalance Analysis:** Automated detection of underrepresented classes with resampling and augmentation strategy recommendations
- **LLM Reporting:** Small language models generate structured, human-readable audit summaries per dataset
- **Modular Design:** Each agent operates independently and can be applied to any image classification dataset

---

## Project Structure

```
Agentic-AI-for-Medical-Imaging-Dataset-Curation/
├── agents/
│   ├── data_inspector.py          # Image quality assessment agent
│   ├── label_auditor.py           # Outlier detection agent
│   ├── dataset_optimizer.py       # Class balance & resampling agent
│   └── llm_explainer.py           # LLM audit report agent
├── models/
│   ├── feature_extractor.py       # CNN backbone (MobileNetV3/EfficientNet/ResNet)
│   └── embeddings.py              # Embedding generation utilities
├── datasets/
│   ├── brats_loader.py
│   ├── isic_loader.py
│   ├── chestxray_loader.py
│   └── chexpert_loader.py
├── reports/                       # Generated audit reports
├── utils/
│   ├── iqa_metrics.py             # No-reference IQA utilities
│   └── visualisation.py          # Plotting and visualisation tools
├── main.py                        # Entry point — runs full pipeline
├── requirements.txt
└── README.md
```

---

## Getting Started

### 1. Install Requirements

```bash
pip install -r requirements.txt
```

### 2. Prepare a Dataset

Download one of the supported datasets and update the relevant loader in `datasets/`. Example for ISIC:

```bash
# Download ISIC dataset
# Update datasets/isic_loader.py with your local path
```

### 3. Run the Full Pipeline

```bash
python main.py --dataset isic --backbone efficientnet
```

### 4. Run Individual Agents

```bash
# Image quality assessment only
python agents/data_inspector.py --dataset brats

# Outlier detection only
python agents/label_auditor.py --dataset chestxray --method knn

# Generate audit report
python agents/llm_explainer.py --input reports/audit_results.json
```

---

## Methodology

### Vision Backbone
Lightweight CNN backbones (MobileNetV3, EfficientNet-B0, ResNet-18) are used for feature extraction. Pretrained ImageNet weights are fine-tuned where appropriate to generate discriminative embeddings for each image.

### Outlier Detection
Image embeddings are passed to the Label Auditor, which applies:
- **kNN distance** — flags images whose nearest neighbours belong to different classes
- **Mahalanobis distance** — accounts for feature correlations to detect distributional outliers
- **Isolation Forest** — unsupervised anomaly detection for unlabelled data

### LLM Reporting
A small language model processes structured JSON audit results and generates natural language summaries, highlighting the most critical quality issues and recommended remediation steps per dataset.

---

## Tech Stack

- **Language:** Python 3.8+
- **Framework:** PyTorch
- **Vision Models:** MobileNetV3, EfficientNet-B0, ResNet-18 (via `torchvision`)
- **Outlier Detection:** scikit-learn (kNN, Isolation Forest), SciPy (Mahalanobis)
- **LLM Reporting:** Small open-source LLMs
- **Datasets:** BraTS, ISIC, NIH ChestX-ray14, CheXpert

---

## Status

🚧 **This project is currently in progress** as part of the MSc Artificial Intelligence programme at the University of Surrey (Spring 2026). The repository will be updated as development progresses.
