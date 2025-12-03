# 🧭 Naming Conventions & Best Practices — UQModels

This document defines the **naming standards, argument order, and best design practices**
to follow in all model-building functions and classes within the **UQModels** project
(autoencoders, Transformers, MLPs, probabilistic heads, etc.).

---

## 📘 Purpose

Ensure:
- **Consistency** in signatures and docstrings across architectures.
- **Reliable serialization** through `get_config()` / `from_config()`.
- **Immediate readability** of argument roles.
- Natural **interoperability** between modules (Encoder ↔ Decoder ↔ Head).

---

## 🧱 1️⃣ General Naming Rules

| Variable Type | Recommended Prefix | Example | Description |
|----------------|--------------------|----------|--------------|
| Feature dimension (vectors) | `dim_` | `dim_in`, `dim_out`, `dim_model`, `dim_z` | Vector space dimensionality |
| Spatial / sequential shape | `shape_` | `shape_in=(H,W)`, `shape_out=(H_out,W_out)` | Tuple of 2D/3D dimensions |
| Number of elements | `num_` / `n_` | `num_heads`, `n_layers`, `n_blocks` | Count of repeated submodules |
| Dropout | `dp_` / `dropout_` | `dp`, `dp_attn`, `dp_ffn` | Dropout rate |
| Regularization | `reg_` | `reg_W`, `reg_bias` | L1/L2 regularization factors |
| Activation | `activation_` | `activation_hidden="gelu"`, `activation_output="softmax"` | Activation functions |
| Uncertainty (UQ) | `mc_` / `flag_` | `mc_dropout=True`, `flag_mc=True` | Stochastic behavior toggles |
| Random seed | `random_state` | `random_state=42` | Global random seed |
| Output heads | `head_` / `type_output` | `type_output="EDL"`, `head_type="MC_Dropout"` | Type of output/probabilistic head |
| System names | `name` | `name="Encoder_Block"` | Internal Keras name (always last) |

💡 **Key rule:**  
> ✅ *Names must describe functional purpose, not variable type.*

---

## ⚙️ 2️⃣ Standard Argument Order in `__init__`

All `layers.Layer` and `tf.keras.Model` classes follow the same argument order:

1. **Structure**: dimensions, sizes, ratios  
2. **Regularization / Activation**: dropout, activation, normalization  
3. **Stochastic / UQ**: Monte Carlo flags, seeds  
4. **System / Keras**: name, kwargs

### Example:

```python
class TransformerEncoderBlock(layers.Layer):
    def __init__(
        self,
        dim_model: int,
        num_heads: int,
        dim_ffn: int,
        dropout_attn: float = 0.1,
        dropout_ffn: float = 0.1,
        activation: str = "gelu",
        mc_dropout: bool = False,
        random_state: int | None = None,
        name: str | None = None,
        **kwargs,
    ):
        ...
```

✅ The same argument order applies to:
- `MLPLayer`
- `TransformerDecoderBlock`
- `AutoencoderHybrid`
- `VariationalEncoder`, etc.

---

## 🔁 3️⃣ UQModels-Specific Conventions

| Concept | Recommended Name | Example | Comment |
|----------|------------------|----------|----------|
| Latent dimension | `dim_z` | `dim_z=64` | Autoencoder latent space |
| Input dimension | `dim_x` | `dim_x=128` | Raw input dimension |
| Sequence length / horizon | `dim_horizon` or `seq_len` | `dim_horizon=20` | Temporal or predictive context |
| Monte Carlo dropout | `mc_dropout` | `mc_dropout=True` | Enables dropout during inference |
| Probabilistic head type | `type_output` | `"MC_Dropout"`, `"EDL"`, `"classif"` | Standardized output type |
| Log-variance bounds | `logvar_min`, `logvar_max` | `logvar_min=-10` | Numerical stability clip |

---

## 🧩 4️⃣ Best Practices for Layer Design

### ✅ Serialization
- Always implement `get_config()` and `from_config()`.
- Include **all** constructor hyperparameters (including seeds).
- Avoid manual serialization of sublayers (`Dense`, `Dropout`, …) if they are created in `__init__`.

### ✅ Layer naming
Use explicit, hierarchical names such as:
- `dense_hidden_1`, `dense_out`
- `attn_self`, `attn_cross`
- `dropout_ffn`, `norm_pre`
> Avoid redundancy like `Dense_1_1_1`.

### ✅ Docstring template
Every architecture class/function must include a structured docstring:

```python
"""
BlockName: short description.

Args:
    dim_in (int): Input dimension.
    dim_out (int): Output dimension.
    layers_size (list[int]): List of hidden layer widths.
    dp (float): Dropout rate.
    mc_dropout (bool): Enables dropout during inference (Monte Carlo).
    type_output (str): Output head type ('EDL', 'MC_Dropout', 'classif').
    random_state (int): Random seed for reproducibility.
"""
```

---

## 🧠 5️⃣ Conventions for **Model Builder Functions**

Model-building functions (`make_mlp_model`, `build_transformer_encoder`, etc.) must:

- Reuse the **same argument names** as their underlying layer classes.  
- Follow the **same logical order**.  
- Use a **prefix** `make_` or `build_`:
  - `make_mlp_model()`
  - `build_autoencoder()`
  - `make_transformer_decoder()`

### Example:

```python
def make_mlp_model(
    dim_in: int,
    dim_out: int,
    layers_size: list[int],
    dp: float = 0.1,
    mc_dropout: bool = False,
    type_output: str | None = None,
    name: str = "MLP",
) -> tf.keras.Model:
    """Builds a functional Keras model based on MLPLayer."""
    ...
```

---

## 🧩 6️⃣ Recommended Prefix Reference

| Prefix | Meaning | Inspired by |
|---------|----------|-------------|
| `dim_` | Scalar dimension (feature/latent) | Flax / JAX |
| `shape_` | Spatial shape (tuple) | Keras / TF |
| `num_` / `n_` | Count of elements | PyTorch |
| `dp_` / `dropout_` | Dropout rate | Internal |
| `mc_` / `flag_` | Boolean control flag | Internal UQ |
| `reg_` | Regularization factor | TensorFlow |
| `activation_` | Activation function | Keras |
| `head_` | Output head | HuggingFace |
| `cfg_` | Submodule configuration dict | PyTorch Lightning |

---

## 📦 7️⃣ Practical Recommendations

| Rule | Reason | Example |
|------|---------|----------|
| Always include `name` as the last argument | Consistent with Keras conventions | — |
| Always use **type hints** | Improves clarity and IDE support | `dim_in: int` |
| Prefer safe default values | Avoids config mistakes | `dp=0.1`, `activation="gelu"` |
| Avoid ambiguous abbreviations | Improves readability | `dp_rec` → `dropout_recurrent` |
| Centralize constants | Avoid magic numbers | `LOGVAR_MIN_DEFAULT = -10.0` |
| Preserve logical order | Input → Architecture → Output → System | — |

---

## ✅ 8️⃣ Example of a Harmonized Signature

```python
class MLPLayer(layers.Layer):
    def __init__(
        self,
        dim_in: int,
        dim_out: int | None = 1,
        layers_size: list[int] = (100, 50),
        dropout: float = 0.1,
        activation_hidden: str = "leaky_relu",
        mc_dropout: bool = False,
        type_output: str | None = None,
        logvar_min: float = -10.0,
        random_state: int | None = None,
        name: str | None = None,
        **kwargs,
    ):
        ...
```

---

## 🧩 9️⃣ Cross-Architecture Naming Consistency

| Class | Key Arguments |
|--------|----------------|
| `MLPLayer` | `dim_in`, `dim_out`, `layers_size`, `dropout`, `mc_dropout`, `type_output` |
| `TransformerEncoderBlock` | `dim_model`, `num_heads`, `dim_ffn`, `dropout_attn`, `dropout_ffn`, `mc_dropout` |
| `TransformerDecoderBlock` | `dim_model`, `num_heads`, `dim_ffn`, `use_cross_attention`, `mc_dropout` |
| `AutoencoderHybrid` | `dim_x`, `dim_z`, `encoder_config`, `decoder_config`, `head_type`, `flag_mc` |

---

## 📜 🔟 File Naming Conventions

| Module Type | Convention | Example |
|--------------|-------------|----------|
| Custom layers | `layers_<type>.py` | `layers_mlp.py`, `layers_transformer.py` |
| Model builders | `build_<type>.py` | `build_autoencoder.py`, `build_transformer.py` |
| Utilities | `utils_<scope>.py` | `utils_training.py`, `utils_seed.py` |
| Config/constants | `config_<scope>.py` | `config_models.py` |
| Tests | `test_<module>.py` | `test_layers_mlp.py` |

---

## 🧩 11️⃣ TL;DR — Summary

✅ **Argument order**:  
`dim_*, shape_*, num_*` → `dropout_*, activation_*` → `mc_*, random_state` → `name`

✅ **Standard argument names**:  
`dim_in`, `dim_out`, `dim_z`, `num_heads`, `dp`, `mc_dropout`,  
`random_state`, `activation_hidden`, `type_output`, `logvar_min`

✅ **Clear prefixes**:  
`dim_`, `shape_`, `num_`, `dp_`, `mc_`, `reg_`, `head_`

✅ **Last argument**: always `name`

---

## 🧩 12️⃣ Example: Standardized Builder Function

```python
def make_transformer_encoder(
    dim_model: int,
    num_heads: int,
    dim_ffn: int = 256,
    dropout_attn: float = 0.1,
    dropout_ffn: float = 0.1,
    activation: str = "gelu",
    mc_dropout: bool = False,
    random_state: int | None = None,
    name: str = "TransformerEncoder",
) -> tf.keras.Model:
    """Builds a Transformer Encoder following UQModels naming conventions."""
    ...
```

---

**Last updated:** November 2025
