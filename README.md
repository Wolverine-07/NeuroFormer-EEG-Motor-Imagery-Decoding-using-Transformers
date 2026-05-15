# NeuroFormer: Transformer from Scratch → EEG Brain-Computer Interface Decoding

> A faithful PyTorch implementation of "Attention Is All You Need" (Vaswani et al., 2017)
> applied to EEG-based motor imagery decoding for Brain-Computer Interfaces.

🚧 **Work in Progress** — Building phase by phase. See commit history for progress.

## Quick Start

```bash
# Clone and install
git clone https://github.com/YOUR_USERNAME/neuroformer.git
cd neuroformer
pip install -e ".[all]"

# Run tests
pytest tests/

# Train on copy task (validates transformer implementation)
python scripts/train_translation.py
```

## Project Structure

```
neuroformer/
├── src/
│   ├── transformer/     # Pure Transformer from scratch (Paper Sections 3.1-3.5)
│   ├── eeg/             # EEG data loading & preprocessing
│   ├── models/          # Application models (EEG-Transformer, baselines)
│   ├── training/        # Training loop, schedulers, losses
│   └── visualization/   # Attention maps, training curves, EEG plots
├── configs/             # Experiment configurations
├── scripts/             # Runnable training & demo scripts
├── notebooks/           # Jupyter exploration notebooks
└── tests/               # Unit tests
```

## References

- Vaswani, A., et al. (2017). "Attention Is All You Need." *NeurIPS*.
- Goldberger, A., et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet.

## License

MIT
