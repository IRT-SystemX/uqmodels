# Modular Autoencoder and Variational Autoencoder Framework

## Overview

This repository provides a modular TensorFlow / Keras framework for building autoencoders and variational autoencoders.

The architecture follows a layered decomposition:

```text
Layers
    ↓
Blocks
    ↓
SubNets
    ↓
Builder Functions
    ↓
Final AE / VAE Models
```

Each level has a clear responsibility:

* **Layers** implement elementary neural or structured operations.
* **Blocks** group reusable architectural patterns.
* **SubNets** assemble blocks into configurable encoder or decoder components.
* **Builder functions** wrap SubNets into standard `tf.keras.Model` encoders and decoders.
* **Final models** combine encoder, decoder, and shared AE/VAE training logic.

Supported architectures include:

* Dense / MLP
* Convolutional 1D and 2D
* LSTM
* Transformer
* Decomposition-based models with level, trend, seasonality, and optional CNN residual components
* Experimental U-Net-based models
* Hybrid supervised latent models

---

## Project Structure

```text
project_root/
│
├── base_vae.py
│   ├── BaseAutoencoder
│   ├── BaseVariationalAutoencoder
│   ├── VariationalMixin
│   ├── SequenceMixin
│   └── HybridMixin
│
├── layers.py
│   ├── MLPBlock
│   ├── DenseHeadBlock
│   ├── VariationalBlock
│   └── MLPSubNet
│
├── convlayers.py
│   ├── ConvBlock1D / ConvBlock2D
│   ├── TConvBlock1D / TConvBlock2D
│   └── CNNSubNet
│
├── seqlayers.py
│   ├── LstmBlock
│   └── LSTMSubNet
│
├── attlayers.py
│   ├── PositionalEmbedding
│   ├── TransformerEncoderBlock
│   ├── TransformerDecoderBlock
│   └── TransformerSubNet
│
├── trendseasonlayers.py
│   ├── LevelLayer
│   ├── TrendLayer
│   ├── SeasonalLayer
│   └── DecompositionSubNet
│
├── densevae.py
├── cnnvae.py
├── lstmvae.py
├── transformervae.py
├── decompositionvae.py
├── uvae.py
│
└── README.md
```

---

## Core Architecture

### Base Models

`BaseAutoencoder` centralizes deterministic AE training behavior.

`BaseVariationalAutoencoder` extends it with:

* latent sampling
* KL divergence
* KL loss tracking
* variational training logic

Final architecture classes therefore remain lightweight and mainly define:

```text
encoder
decoder
```

---

### Architectural Hierarchy

#### 1. Layers

Elementary neural or structured operations.

Examples:

```text
Dense
Dropout
PositionalEmbedding
LevelLayer
TrendLayer
SeasonalLayer
```

#### 2. Blocks

Reusable architectural patterns.

Examples:

```text
MLPBlock
ConvBlock1D
ConvBlock2D
TConvBlock1D
TConvBlock2D
LstmBlock
TransformerEncoderBlock
TransformerDecoderBlock
DenseHeadBlock
VariationalBlock
```

#### 3. SubNets

Configurable encoder or decoder components built from reusable blocks.

Available SubNets include:

```text
MLPSubNet
CNNSubNet
LSTMSubNet
TransformerSubNet
DecompositionSubNet
```

Typical structure:

```text
Backbone
    ↓
optional intermediate block
    ↓
DenseHeadBlock
```

#### 4. Builder Functions

Builders wrap SubNets into standard Keras models:

```text
Keras Input
    ↓
SubNet
    ↓
optional VariationalBlock
    ↓
tf.keras.Model
```

#### 5. Final Models

Final AE and VAE classes combine:

```text
Base training behavior
+
Encoder builder
+
Decoder builder
```

Examples:

```text
DenseAE / DenseVAE
ConvAE / ConvVAE
LstmAE / LstmVAE
TransformerAE / TransformerVAE
DecompositionAE / DecompositionVAE
```

---

## Model Hierarchy

```text
BaseAutoencoder
│
├── DenseAE
├── ConvAE
├── LstmAE
├── TransformerAE
├── DecompositionAE
├── UNetAE
│
└── BaseVariationalAutoencoder
    ├── DenseVAE
    ├── ConvVAE
    ├── LstmVAE
    ├── TransformerVAE
    ├── DecompositionVAE
    └── UNetVAE
```

Optional mixins provide additional behavior:

```text
VariationalMixin
SequenceMixin
HybridMixin
```

The primary architectural composition is nevertheless based on:

```text
Layer → Block → SubNet → Builder → Model
```

---

## Supported Architectures

### Dense

```text
MLPBlock
    ↓
MLPSubNet
    ↓
DenseAE / DenseVAE
```

### Convolutional

Supports 1D and 2D convolutional architectures.

```text
Conv / TConv Blocks
    ↓
CNNSubNet
    ↓
ConvAE / ConvVAE
```

### LSTM

Designed for sequential inputs of shape:

```text
(B, T, F)
```

```text
LstmBlock
    ↓
LSTMSubNet
    ↓
LstmAE / LstmVAE
```

### Transformer

```text
PositionalEmbedding
+
Transformer Blocks
    ↓
TransformerSubNet
    ↓
TransformerAE / TransformerVAE
```

### Decomposition-Based Models

Structured reconstruction based on:

```text
level
+
trend
+
seasonality
+
optional CNN residual
```

The structured components are aggregated by `DecompositionSubNet`.

Available models:

```text
DecompositionAE
DecompositionVAE
```

---

## Variational Modeling and Output Uncertainty

Two independent concepts are distinguished.

### Variational latent modeling

```python
variational=True
```

adds:

```text
latent representation
    ↓
VariationalBlock
    ↓
z_mean, z_log_var
```

Sampling and KL divergence are managed by `BaseVariationalAutoencoder`.

### Output formalization

`type_output` controls the output representation, for example:

```text
None
mc_dropout
Deep_ensemble
EDL
```

This logic is handled through `DenseHeadBlock` where applicable.

Variational latent modeling and output uncertainty are therefore independent mechanisms.

---

## Configuration-Driven Construction

Architectures are configured through dictionaries.

Example:

```python
cfg_encoder = {
    "dim_seq": 60,
    "dim_in": 52,
    "dim_z": 16,
    "variational": True,
    "cfg_subnet": CNNSubNet.make_config(
        mode="encoder",
        dim_seq=60,
        dim_in=52,
        dim_z=16,
    ),
}
```

Final model:

```python
model = ConvVAE(
    cfg_encoder=cfg_encoder,
    cfg_decoder=cfg_decoder,
    kl_weight=0.1,
)
```

This supports reproducibility, serialization, benchmarking, and systematic configuration management.

---

## How to Extend

Adding a new architecture follows the same pattern:

```text
1. Create primitive layers if needed
2. Create reusable blocks
3. Assemble them into a SubNet
4. Build encoder and decoder Keras models
5. Create final AE and VAE classes
```

Minimal model pattern:

```python
class MyCustomVAE(BaseVariationalAutoencoder):

    def __init__(
        self,
        cfg_encoder,
        cfg_decoder,
        kl_weight=1.0,
        **kwargs,
    ):
        super().__init__(
            kl_weight=kl_weight,
            **kwargs,
        )

        self.encoder = build_my_encoder(**cfg_encoder)
        self.decoder = build_my_decoder(**cfg_decoder)
```

No architecture-specific training loop is required.

---

## Design Principles

The framework follows four main principles:

* **Composition over duplication**
* **Clear separation of responsibilities**
* **Configuration-driven architecture construction**
* **Shared AE/VAE training behavior**

The central design is:

```text
Layer
    ↓
Block
    ↓
SubNet
    ↓
Builder
    ↓
Model
```

---

## Key Advantages

* Unified AE/VAE API
* Shared training logic
* Reusable architectural components
* Configurable and serializable SubNets
* Independent variational and output-uncertainty mechanisms
* Easy integration of new backbones
* Native TensorFlow / Keras compatibility

---

## Credits

The decomposition-based architecture is partially inspired by TimeVAE:

```text
https://github.com/abudesai/timeVAE/tree/main/src/vae
```
