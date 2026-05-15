# 🧠 NeuroFormer

**Transformer from Scratch → EEG Brain-Computer Interface Decoding**

A faithful PyTorch implementation of *"Attention Is All You Need"* (Vaswani et al., 2017), applied to EEG-based motor imagery classification for Brain-Computer Interfaces.

---

## Overview

This project implements the Transformer architecture **entirely from scratch** in PyTorch — no `nn.Transformer`, no HuggingFace — and then applies it to decode motor imagery from EEG brain signals using the [PhysioNet Motor Movement/Imagery Dataset](https://physionet.org/content/eegmmidb/1.0.0/).

The core idea: if someone imagines moving their left hand vs right hand, their brain produces distinguishable patterns in the EEG signal. The Transformer's self-attention mechanism can learn to identify these patterns by attending to the most informative time segments and spatial electrode combinations.

### Architecture

```
Raw EEG (64 channels × ~4s at 128 Hz)
  → CNN Tokenizer (temporal + spatial convolutions)
  → Patch Embedding → sequence of tokens
  → [CLS] token + Learnable Positional Encoding
  → Transformer Encoder (4 layers, 4 heads, d_model=128)
  → [CLS] output → Classification Head
  → 4-class prediction (left fist, right fist, both fists, both feet)
```

**Why CNN + Transformer?** Raw EEG isn't discrete tokens — the CNN extracts local time-frequency features (like learned bandpass filters) and spatial patterns (like Common Spatial Pattern), converting continuous signals into a token sequence the Transformer can process. The Transformer then models long-range temporal dependencies across these features.

---

## Project Structure

```
neuroformer/
├── src/
│   ├── transformer/          # Transformer from scratch (Paper Sections 3.1-3.5)
│   │   ├── attention.py      # Scaled dot-product & multi-head attention
│   │   ├── encoder.py        # Encoder layer + stack
│   │   ├── decoder.py        # Decoder layer + stack (for seq2seq tasks)
│   │   ├── embeddings.py     # Token embeddings + sinusoidal positional encoding
│   │   ├── feed_forward.py   # Position-wise feed-forward network
│   │   ├── transformer.py    # Full encoder-decoder model
│   │   └── utils.py          # Masking, LayerNorm, residual connections
│   │
│   ├── eeg/                  # EEG data pipeline
│   │   ├── dataset.py        # PhysioNet data loading & splitting
│   │   ├── preprocessing.py  # PSD, band power, signal quality
│   │   ├── tokenizer.py      # CNN-based EEG → token conversion
│   │   └── augmentation.py   # Noise, time-shift, channel dropout
│   │
│   ├── models/               # Application models
│   │   ├── eeg_transformer.py  # CNN-Transformer hybrid classifier
│   │   └── baselines.py        # EEGNet baseline for comparison
│   │
│   ├── training/             # Training infrastructure
│   │   ├── trainer.py        # Training loop + greedy decode
│   │   ├── scheduler.py      # Noam LR scheduler (Paper Section 5.3)
│   │   ├── losses.py         # Label smoothing (Paper Section 5.4)
│   │   └── metrics.py        # Accuracy, F1, Cohen's κ
│   │
│   └── visualization/        # Analysis & plots
│       ├── attention_maps.py     # Attention weight heatmaps
│       ├── training_curves.py    # Loss/accuracy curves, confusion matrices
│       └── eeg_plots.py          # EEG signals, PSD, class-averaged ERPs
│
├── configs/                  # YAML experiment configs
├── scripts/
│   ├── train_translation.py  # Transformer validation (copy task)
│   ├── train_eeg.py          # Main EEG training script
│   └── demo.py               # Interactive Gradio demo
│
├── tests/                    # Unit tests (44 tests)
└── notebooks/                # Jupyter exploration notebooks
```

---

## Getting Started

### Installation

```bash
git clone https://github.com/Wolverine-07/NeuroFormer-EEG-Motor-Imagery-Decoding-using-Transformers.git
cd NeuroFormer-EEG-Motor-Imagery-Decoding-using-Transformers

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e ".[all]"
# or
pip install -r requirements.txt
```

### Validate the Transformer (Copy Task)

First, verify the from-scratch Transformer works correctly:

```bash
python scripts/train_translation.py
```

This trains a small Transformer on a sequence-copying task. Expected result: **100% accuracy** and 10/10 perfect greedy decodes, confirming all components (attention, masking, positional encoding, etc.) are correctly implemented.

### Run Tests

```bash
pytest tests/ -v
```

All 44 tests should pass — covering attention mechanics, masking, encoder/decoder shapes, gradient flow, EEG dataset, augmentation, and both models.

### Train on EEG Data

```bash
# Subject-dependent evaluation (quick test with 5 subjects)
python scripts/train_eeg.py --subjects 1 2 3 4 5

# Full subject-dependent evaluation
python scripts/train_eeg.py --config configs/eeg_subject_dependent.yaml

# Cross-subject evaluation
python scripts/train_eeg.py --config configs/eeg_cross_subject.yaml --mode cross_subject

# Compare with EEGNet baseline
python scripts/train_eeg.py --model eegnet --subjects 1 2 3 4 5
```

### Interactive Demo

```bash
# With Gradio web UI
pip install gradio
python scripts/demo.py --demo

# CLI-only demo
python scripts/demo.py --cli
```

---

## Key Implementation Details

### Paper-Faithful Transformer

Every component maps directly to the paper:

| Paper Section | Component | File |
|---|---|---|
| 3.2.1 | Scaled Dot-Product Attention: `softmax(QK^T/√d_k)V` | `attention.py` |
| 3.2.2 | Multi-Head Attention: parallel heads with `W_Q, W_K, W_V, W_O` | `attention.py` |
| 3.3 | Position-wise FFN: `max(0, xW₁+b₁)W₂+b₂` with ReLU | `feed_forward.py` |
| 3.4 | Token embeddings scaled by `√d_model` | `embeddings.py` |
| 3.5 | Sinusoidal positional encoding | `embeddings.py` |
| 3.1 | Encoder/decoder stacks with residual connections + LayerNorm | `encoder.py`, `decoder.py` |
| 5.3 | Noam LR schedule with warmup | `scheduler.py` |
| 5.4 | Label smoothing `ε=0.1` | `losses.py` |

### EEG-Specific Adaptations

- **CNN Tokenizer**: Temporal conv → spatial conv (depthwise) → separable conv → patch embedding. Inspired by EEGNet architecture.
- **Classification via Encoder Only**: No decoder needed — we use a [CLS] token (BERT-style) with the encoder for classification.
- **Learnable Positional Encoding**: Fixed-length EEG trials work better with learned PE than sinusoidal.
- **Smaller Model**: `d_model=128, N=4, h=4` — EEG datasets are too small for the full 512-dim base model.

### Evaluation Protocols

1. **Subject-Dependent**: Train/test within each subject (80/20 split). Shows the model can learn individual brain patterns.
2. **Cross-Subject**: K-fold over subjects (leave-subjects-out). The harder but more realistic protocol for BCI deployment.

---

## Dataset

**PhysioNet EEG Motor Movement/Imagery Dataset**
- 109 subjects, 64 EEG channels, 160 Hz (resampled to 128 Hz)
- 4 motor imagery classes: left fist, right fist, both fists, both feet
- Preprocessing: 4-40 Hz bandpass (captures mu/beta rhythms), epoching, baseline correction, z-score normalization

The data is automatically downloaded via MNE-Python when you run the training script.

---

## References

- Vaswani, A., et al. (2017). *"Attention Is All You Need."* NeurIPS. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- Lawhern, V.J., et al. (2018). *"EEGNet: A Compact Convolutional Neural Network for EEG-Based Brain-Computer Interfaces."* Journal of Neural Engineering.
- Goldberger, A., et al. (2000). *"PhysioBank, PhysioToolkit, and PhysioNet."* Circulation.

---

## License

MIT
