# **Neuro-PARSDiff**

**Permutation-Aware Autoregressive Structured Diffusion for Brain Connectome Generation**:  
A ranking-driven, block-wise diffusion framework for robust, permutation-invariant generation and reconstruction of brain connectivity graphs.

---

## 🔍 **1. Overview**

**Neuro-PARSDiff** is a permutation-aware autoregressive diffusion framework for **brain connectome generation**.

Most existing graph diffusion and generative models implicitly assume a fixed or arbitrary node ordering.  
This breaks permutation invariance and leads to unstable diffusion dynamics, block inconsistency, and poor generalization across subjects, datasets, and graph resolutions.

Neuro-PARSDiff overcomes this limitation by introducing an explicit **ranking-driven autoregressive factorization** combined with **block-wise masked diffusion**, ensuring permutation consistency throughout both training and generation.

The framework integrates:
- permutation-consistent node ranking,
- adaptive block prediction,
- causal masking,
- graph-transformer–based denoising,

resulting in stable, scalable, and interpretable connectome generation.

---

**DCMD Framework**  
<img width="525" height="392" alt="image" src="https://github.com/user-attachments/assets/c9a666aa-d35b-4556-a6f8-1f78271aa309" />
---
## ✨ **2. Key Contributions**

1. **Permutation-Aware Autoregressive Diffusion**  
   Introduces a ranking-based factorization that preserves permutation consistency across diffusion steps.

2. **Block-Wise Graph Decomposition**  
   Decomposes large connectomes into adaptive blocks, enabling stable diffusion on high-dimensional graphs.

3. **Mask-Controlled Causal Denoising**  
   Enforces block-level causality via structured masks, preventing information leakage.

4. **Graph Transformer Denoiser (ℓφ)**  
   A graph-aware transformer that jointly models node and edge states under noise.

5. **Adaptive Block Predictor (gφ)**  
   Learns block sizes dynamically, enabling flexible generation across graph scales.

6. **Unified Training Objective**  
   Combines diffusion loss and categorical reconstruction loss for robust optimization.

---

## 🧠 **3. Method Overview**

### **Pipeline Summary**

1. **Input connectome** → Permutation-consistent ranking (ψ)  
2. **Ranked nodes** → Block extraction (Δₖ)  
3. **Block-masked graph** → Forward diffusion  
4. **Noisy graph** → Graph transformer denoiser (ℓφ)  
5. **Reverse diffusion** → Block-wise reconstruction  
6. **Autoregressive loop** → Full connectome generation  

---

### **Repository Structure**

```text
Neuro-PARSDiff/
│
├── configs/
│   ├── model.yaml
│   ├── diffusion.yaml
│   ├── dataset.yaml
│   └── train.yaml
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── splits/
│   └── README.md
│
├── neuro_parsdiff/
│   ├── models/
│   │   ├── denoiser.py
│   │   ├── block_predictor.py
│   │   ├── transformer.py
│   │   └── embeddings.py
│   │
│   ├── diffusion/
│   │   ├── forward.py
│   │   ├── reverse.py
│   │   ├── schedules.py
│   │   └── losses.py
│   │
│   ├── graph/
│   │   ├── ranking.py
│   │   ├── blocks.py
│   │   ├── masking.py
│   │   └── features.py
│   │
│   ├── generation/
│   │   ├── sample.py
│   │   └── reconstruct.py
│   │
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── mst.py
│   │   └── plots.py
│   │
│   ├── training/
│   │   ├── trainer.py
│   │   ├── engine.py
│   │   └── checkpoint.py
│   │
│   └── utils/
│       ├── io.py
│       ├── logger.py
│       ├── seed.py
│       └── config.py
│
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── generate.py
│   └── preprocess.py
│
├── experiments/
├── results/
├── notebooks/
├── tests/
│
├── environment.yml
├── LICENSE
└── README.md

```

## 📊 Supported Graph Domains

This implementation supports all brain graph domains once represented as adjacency-based connectomes:

```text
Functional connectomes     # fMRI-derived graphs
Structural connectomes     # DTI-based graphs
Multi-resolution graphs   # Different parcellations
Population-level graphs   # Subject cohorts
Attributed / unlabeled    # With or without node features
```

## ⚙️ Installation

### 1️⃣ Create environment (recommended)

```bash
conda env create -f environment.yml
conda activate neuro-parsdiff
```

### **2️⃣ Install dependencies**
```
pip install -r requirements.txt
```

### **Key libraries**

- **PyTorch**
- **NumPy**
- **OmegaConf**
- **PyYAML**
- **Matplotlib**

  ---

## 📂 **Dataset Preparation**

Expected graph batch structure:

```text
batch.x          # Node features
batch.edge_index # Graph connectivity
batch.batch      # Graph indices
```

✔ **Variable-size graphs supported**  
✔ **Permutation-invariant processing**  
✔ **Compatible with PyTorch Geometric–style batching**

---

## 🚀 **How to Run**

---

### 🔹 **Training**
```bash
python scripts/train.py --config configs/neuro_parsdiff_hcp.yaml
```

### **What happens:**
- **Sets deterministic random seeds** across Python, NumPy, and PyTorch
- **Applies permutation-consistent node ranking and block extraction**
- **Performs block-wise forward diffusion with causal masking**
- **Trains the graph transformer denoiser (ℓφ) and block predictor (gφ)**
- **Saves model checkpoints periodically**

### 🔹 Resume Training

```python
load_checkpoint(
    path="checkpoints/epoch_100.pt",
    model=model,
    optimizer=optimizer
)
```
**This allows training to resume exactly from the saved epoch**, including the optimizer state.
---

### 🔹 **Sampling / Inference**

```bash
python scripts/generate.py \
  --config configs/model.yaml \
  --checkpoint checkpoints/epoch_200.pt
```
### **This will:**
- **Run autoregressive reverse diffusion**
- **Generate full brain connectomes block by block**
- **Save generated graph samples to disk**
---

## 📈 **Evaluation**

- **Reconstruction error (MSE, Pearson correlation)**
- **Community and block consistency (NMI)**
- **Spectral similarity**
- **Minimum Spanning Tree (MST) similarity**

**Generated samples preserve graph size and structural statistics** to ensure **fair and consistent comparison** across datasets.

---

### 🔹 **Some of our Results**

<img width="594" height="227" alt="image" src="https://github.com/user-attachments/assets/d400e802-6d5a-49ee-9a1d-44c00804cf3b" />
<img width="611" height="339" alt="image" src="https://github.com/user-attachments/assets/18417dcc-6f59-4d7c-a876-3e00395f2e07" />
<img width="1008" height="350" alt="image" src="https://github.com/user-attachments/assets/92294a59-0da5-4f80-bb06-199fd3b3a207" />
<img width="849" height="406" alt="image" src="https://github.com/user-attachments/assets/f26e737c-3a5e-4be0-9c6c-f49727b43bfb" />
<img width="735" height="547" alt="image" src="https://github.com/user-attachments/assets/06720dc4-3227-411a-92e4-0dbc1ccf2d02" />
<img width="712" height="532" alt="image" src="https://github.com/user-attachments/assets/f91285c4-bdc0-4e95-aae5-d38db11e284e" />

## 🔬 **Reproducibility Notes**
- **Fixed random seeds** across **Python**, **NumPy**, and **PyTorch**
- **Deterministic CuDNN behavior** enabled
- **Config-driven experiments** (no hidden hyperparameters)
- **No dataset-specific logic inside model code**
- **Explicit permutation-aware modeling** to ensure stable diffusion


