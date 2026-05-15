#!/usr/bin/env python3
"""
Interactive Gradio demo for EEG Motor Imagery Classification.

Provides a web interface to:
  - Select a subject and trial from the PhysioNet dataset
  - Run classification with the trained EEG-Transformer
  - Visualize attention maps and prediction confidence
  - Compare with the EEGNet baseline

Usage:
    python scripts/demo.py --checkpoint path/to/model.pt
    python scripts/demo.py --demo  # use random weights for UI testing
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.eeg.dataset import CLASS_NAMES


def create_demo_model(n_channels=64, n_times=513, n_classes=4):
    """Create a model (with random weights for demo purposes)."""
    from src.models.eeg_transformer import EEGTransformer
    model = EEGTransformer(
        n_channels=n_channels, n_times=n_times, n_classes=n_classes,
        d_model=128, num_heads=4, d_ff=256, num_layers=4,
    )
    model.eval()
    return model


def generate_sample_eeg(n_channels=64, n_times=513, class_idx=0):
    """Generate a synthetic EEG sample for demo purposes."""
    rng = np.random.RandomState(class_idx * 10 + 1)
    data = rng.randn(n_channels, n_times) * 0.5

    # Add some class-dependent patterns to make it interesting
    # Simulate mu-rhythm desynchronization in different hemispheres
    t = np.linspace(0, 4, n_times)

    if class_idx == 0:  # left fist — ERD in right hemisphere (C4 area, ~ch 30-35)
        for ch in range(30, 36):
            data[ch] += 0.3 * np.sin(2 * np.pi * 10 * t) * np.exp(-t / 2)
    elif class_idx == 1:  # right fist — ERD in left hemisphere (C3 area, ~ch 20-25)
        for ch in range(20, 26):
            data[ch] += 0.3 * np.sin(2 * np.pi * 10 * t) * np.exp(-t / 2)
    elif class_idx == 2:  # both fists — bilateral
        for ch in list(range(20, 26)) + list(range(30, 36)):
            data[ch] += 0.2 * np.sin(2 * np.pi * 12 * t) * np.exp(-t / 1.5)
    else:  # both feet — central (Cz area, ~ch 0-5)
        for ch in range(0, 6):
            data[ch] += 0.3 * np.sin(2 * np.pi * 8 * t) * np.exp(-t / 2)

    return data.astype(np.float32)


def classify_eeg(model, eeg_data, device="cpu"):
    """Run classification and return probabilities + attention maps."""
    model.eval()
    with torch.no_grad():
        x = torch.tensor(eeg_data, dtype=torch.float32).unsqueeze(0).to(device)
        logits, attn_weights = model(x, return_attention=True)
        probs = F.softmax(logits, dim=-1).squeeze().cpu().numpy()

    attn_np = [w.squeeze().cpu().numpy() for w in attn_weights]
    return probs, attn_np


def create_prediction_plot(probs):
    """Create a bar chart of class probabilities."""
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]
    bars = ax.barh(CLASS_NAMES, probs, color=colors, edgecolor="gray", alpha=0.85)

    for bar, prob in zip(bars, probs):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{prob:.1%}", va="center", fontsize=11, fontweight="bold")

    ax.set_xlim(0, 1.15)
    ax.set_xlabel("Confidence")
    ax.set_title("Classification Results")
    ax.grid(True, alpha=0.2, axis="x")
    plt.tight_layout()
    return fig


def create_attention_plot(attn_weights):
    """Create attention visualization for the last layer."""
    if not attn_weights:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No attention data", ha="center", va="center")
        return fig

    last_layer = attn_weights[-1]
    if last_layer.ndim == 3:
        avg_attn = last_layer.mean(axis=0)
    else:
        avg_attn = last_layer

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(avg_attn, cmap="viridis", aspect="auto")
    ax.set_xlabel("Key Position")
    ax.set_ylabel("Query Position")
    ax.set_title("Attention Map (Last Layer, Averaged)")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    return fig


def create_eeg_plot(eeg_data, sfreq=128.0, n_channels=8):
    """Create a simple EEG signal plot."""
    n_ch = min(n_channels, eeg_data.shape[0])
    times = np.arange(eeg_data.shape[1]) / sfreq

    fig, axes = plt.subplots(n_ch, 1, figsize=(10, 1.2 * n_ch), sharex=True)
    if n_ch == 1:
        axes = [axes]

    for i in range(n_ch):
        axes[i].plot(times, eeg_data[i], linewidth=0.5, color="navy")
        axes[i].set_ylabel(f"Ch{i}", fontsize=7, rotation=0, labelpad=20)
        axes[i].tick_params(labelsize=6)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Input EEG Signal", fontsize=11)
    plt.tight_layout()
    return fig


def launch_demo(model, device="cpu"):
    """Launch Gradio web interface."""
    try:
        import gradio as gr
    except ImportError:
        print("Gradio not installed. Install with: pip install gradio")
        print("Running CLI demo instead...\n")
        run_cli_demo(model, device)
        return

    def predict(class_label):
        class_idx = CLASS_NAMES.index(class_label)
        eeg_data = generate_sample_eeg(class_idx=class_idx)
        probs, attn = classify_eeg(model, eeg_data, device)

        pred_class = CLASS_NAMES[np.argmax(probs)]
        confidence = np.max(probs)

        pred_fig = create_prediction_plot(probs)
        attn_fig = create_attention_plot(attn)
        eeg_fig = create_eeg_plot(eeg_data)

        summary = f"**Predicted:** {pred_class} ({confidence:.1%} confidence)"
        return summary, eeg_fig, pred_fig, attn_fig

    demo = gr.Interface(
        fn=predict,
        inputs=gr.Dropdown(
            choices=CLASS_NAMES,
            value=CLASS_NAMES[0],
            label="Select Motor Imagery Class (ground truth)"
        ),
        outputs=[
            gr.Textbox(label="Prediction"),
            gr.Plot(label="Input EEG Signal"),
            gr.Plot(label="Classification Confidence"),
            gr.Plot(label="Attention Map"),
        ],
        title="🧠 NeuroFormer — EEG Motor Imagery Decoder",
        description=(
            "Interactive demo of a Transformer-based Brain-Computer Interface. "
            "Select a motor imagery class to generate a synthetic EEG trial, "
            "then see the model's classification and attention patterns."
        ),
        allow_flagging="never",
    )

    print("Launching Gradio demo at http://localhost:7860")
    demo.launch(share=False)


def run_cli_demo(model, device="cpu"):
    """Fallback CLI demo if Gradio isn't available."""
    print("=" * 50)
    print("NeuroFormer CLI Demo")
    print("=" * 50)

    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        eeg = generate_sample_eeg(class_idx=cls_idx)
        probs, _ = classify_eeg(model, eeg, device)
        pred = CLASS_NAMES[np.argmax(probs)]
        conf = np.max(probs)

        status = "✓" if pred == cls_name else "✗"
        print(f"  {status} True: {cls_name:12s} | Predicted: {pred:12s} ({conf:.1%})")
        print(f"    Probs: {' | '.join(f'{n}:{p:.2f}' for n, p in zip(CLASS_NAMES, probs))}")


def main():
    parser = argparse.ArgumentParser(description="NeuroFormer Demo")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to trained model checkpoint")
    parser.add_argument("--demo", action="store_true",
                        help="Run demo with random weights (for UI testing)")
    parser.add_argument("--cli", action="store_true",
                        help="Run CLI demo instead of Gradio")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_demo_model()

    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading checkpoint: {args.checkpoint}")
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state)

    model = model.to(device)
    model.eval()

    if args.cli:
        run_cli_demo(model, device)
    else:
        launch_demo(model, device)


if __name__ == "__main__":
    main()
