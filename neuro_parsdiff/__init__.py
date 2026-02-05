Neuro-PARSDiff/
│
├── README.md
├── LICENSE
├── environment.yml        # or requirements.txt
├── setup.py               # optional (for pip install -e .)
│
├── configs/
│   ├── model.yaml         # architecture, layers, dims
│   ├── diffusion.yaml     # T, noise schedule, λ
│   ├── dataset.yaml       # HCP, OASIS paths & variants
│   └── train.yaml         # batch size, lr, epochs
│
├── data/
│   ├── raw/               # raw BrainGraph.org data (ignored)
│   ├── processed/         # adjacency tensors, masks
│   ├── splits/            # train/val/test splits
│   └── README.md          # how to download data
│
├── neuro_parsdiff/
│   ├── __init__.py
│   │
│   ├── models/
│   │   ├── denoiser.py        # ℓ_φ (Algo 4)
│   │   ├── block_predictor.py # g_φ (Algo 2)
│   │   ├── transformer.py    # hybrid graph transformer
│   │   └── embeddings.py
│   │
│   ├── diffusion/
│   │   ├── forward.py        # q(G_t | G_{t-1})
│   │   ├── reverse.py        # p_φ(G_{t-1} | G_t)
│   │   ├── schedules.py      # β_t, α_t
│   │   └── losses.py         # L_diff + CE
│   │
│   ├── graph/
│   │   ├── ranking.py        # Algorithm 1 (ψ)
│   │   ├── blocks.py         # block extraction Δ_k
│   │   ├── masking.py        # M_k (Lemma 1)
│   │   └── features.py       # degree, Laplacian, curvature
│   │
│   ├── training/
│   │   ├── trainer.py
│   │   ├── engine.py
│   │   └── checkpoint.py
│   │
│   ├── generation/
│   │   ├── sample.py         # Algo 5 (generation)
│   │   └── reconstruct.py
│   │
│   ├── evaluation/
│   │   ├── metrics.py        # MSE, Pearson, NMI, SPEC
│   │   ├── mst.py            # MST comparison
│   │   └── plots.py
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
│   ├── hcp_83/
│   ├── hcp_1015/
│   ├── oasis_3/
│   └── baselines/
│
├── results/
│   ├── figures/
│   ├── tables/
│   └── logs/
│
├── notebooks/
│   ├── visualization.ipynb
│   ├── ablation.ipynb
│   └── debug.ipynb
│
├── tests/
│   ├── test_ranking.py
│   ├── test_masking.py
│   └── test_diffusion.py
│
└── .gitignore
