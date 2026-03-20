# Modular Variational Autoencoder Framework

## Overview

This repository provides a **modular and extensible framework** for building autoencoders and variational autoencoders (VAEs) in TensorFlow / Keras.

It supports multiple architectures:
- **Dense / MLP-based** autoencoders and VAEs  
- **Convolutional** (1D/2D) VAEs  
- **Temporal** (TimeVAE) models with trend and seasonality  
- **Sequential** (LSTM-based) VAEs  
- **U-Net-based** VAEs (1D or 2D) -> Not Working ATM
- **Transformer-based** VAEs for sequential data
- **Hybrid models** (supervised VAE for classification or forecasting)

The system separates:
- **Training logic** (AE, VAE, Hybrid)  
- **Behavioral mixins** (variational, sequence, supervision)  
- **Concrete architectures** (Dense, Conv, Time, U-Net, Seq)

This design enables easy composition of new architectures by combining mixins and base classes without rewriting training loops.

---

## Project structure

project_root/
│
├── base_vae.py
│ ├── BaseAutoencoder
│ ├── BaseVariationalAutoencoder
│ ├── SequenceMixin
│ ├── VariationalMixin
│ ├── HybridMixin
│
├── vae.py
│ ├── DenseAE / DenseVAE
│ ├── ConvAE / ConvVAE
│ ├── Encoder / Decoder factory functions
│
├── timevae.py
│ ├── TimeVAE
│ ├── TrendLayer / SeasonalLayer
│ ├── Encoder / Decoder builders
│
├── uvae.py
│ ├── UNetVAE
│ ├── (optional) UNetHybridVAE
│ ├── Encoder / Decoder builders
│
├── seq_vae.py
│ ├── SeqVAE
│ ├── HybridSequenceVAE
│
├── transformer_vae.py
│ ├── AETransformer
│ ├── VAETransformer
│ ├── Encoder / Decoder builders (Transformer)
│
└── README.md

## Core architecture

### 1. Base classes & mixins (`base_vae.py`)

| Component | Responsibility |
|------------|----------------|
| **BaseAutoencoder** | Generic Keras `Model` providing a unified training loop (`train_step`, `test_step`) that delegates loss computation to a single hook: `forward_and_losses()` |
| **BaseVariationalAutoencoder** | Extends `BaseAutoencoder` and `VariationalMixin` with default VAE loss handling (reconstruction + KL) and a KL tracker |
| **VariationalMixin** | Low-level API for variational logic: latent sampling, KL divergence computation, and helper methods |
| **SequenceMixin** | Adapts reconstruction loss for sequential/time-series data (`sum over time + mean over batch`) |
| **HybridMixin** | Adds supervised heads (classification or forecasting) and manages the supervised loss and accuracy metrics |

These components are fully composable.  
For instance:
```python
DenseVAE = VariationalMixin + BaseAutoencoder
HybridSeqVAE = HybridMixin + SequenceMixin + VariationalMixin + BaseAutoencoder

2. Model hierarchy
BaseAutoencoder
├── DenseAE
├── ConvAE
├── TimeAE
├── UNetAE
├── AETransformer
├── BaseVariationalAutoencoder
│   ├── DenseVAE
│   ├── ConvVAE
│   ├── TimeVAE
│   ├── UNetVAE
|   ├── VAETransformer
│   └── (custom VAE architectures)
│
├── SequenceMixin + VariationalMixin + BaseAutoencoder
│   ├── SeqVAE
│   └── HybridSequenceVAE
│
└── HybridMixin + (any of the above)
    ├── HybridAE
    ├── HybridVAE
    ├── HybridSequenceVAE
    └── UNetHybridVAE

Design philosophy
Mixins as orthogonal capabilities

Each Mixin provides one dimension of behavior:

VariationalMixin → latent sampling & KL divergence

SequenceMixin → time-aware reconstruction

HybridMixin → supervised objective (classification / forecasting)

They can be combined in any order to form new architectures without redefining the training loop.

Unified training loop

Every model inherits a single generic training routine from BaseAutoencoder:

def train_step(self, data):
    with tf.GradientTape() as tape:
        total_loss, logs = self.forward_and_losses(data)
    grads = tape.gradient(total_loss, self.trainable_weights)
    self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
    self._update_trackers(total_loss, logs)
    return self._collect_results(logs)


Models only need to implement forward_and_losses() to define how losses are computed:

For AE: reconstruction only

For VAE: reconstruction + KL

For HybridVAE: reconstruction + KL + supervised loss

###################################
Example: DenseVAE
from vae import DenseVAE
import tensorflow as tf

# Create a dense VAE
model = DenseVAE(seq_len=128, feat_dim=8, latent_dim=16, kl_weight=0.1)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3))

# Train on input-only data
model.fit(x_train, epochs=50, batch_size=64)

# Encode & decode
z_mean, z_log_var, z = model.encoder(x_train)
x_recon = model.decoder(z)
###################################


How to extend

-1 Create your architecture

-2 Define encoder and decoder (and optionally classifier).

-3 Make them standard tf.keras.Models.

-4 Pick the right base

	Simple AE → BaseAutoencoder

	VAE → BaseVariationalAutoencoder

	Sequential → add SequenceMixin

	Supervised → add HybridMixin

-5 Implement or inherit forward_and_losses()

	Usually already provided by BaseVariationalAutoencoder or Hybrid variant.

Example:

class MyCustomVAE(BaseVariationalAutoencoder):
    def __init__(self, ...):
        super().__init__(kl_weight=0.1)
        self.encoder = build_my_encoder(...)
        self.decoder = build_my_decoder(...)

🚀 Key advantages

✅ Unified training loop — same fit/evaluate/predict workflow for all models
✅ Composable mixins — easy to combine AE/VAE + Sequence + Hybrid
✅ Clean separation of concerns — training logic, architecture, and task objectives are independent
✅ Extensible — easy to add new backbones (e.g., Transformers, GraphVAEs, Diffusion decoders)

📘 Future extensions

TransformerMixin → self-attention encoder/decoder

DiffusionMixin → denoising/diffusion objectives

GraphVAE → graph neural VAE encoder/decoder

HybridForecastVAE → time-sequence forecasting head (multi-horizon)

✨ Credits

-> Partial inspiration from : https://github.com/abudesai/timeVAE/tree/main/src/vae