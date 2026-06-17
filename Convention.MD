# 🧭 Naming Conventions & Best Practices — UQModels

This document defines the **naming standards, argument order, component hierarchy, configuration conventions, and best design practices**to follow in all model-building functions and classes within the **UQModels** project
(autoencoders, Transformers, MLPs, probabilistic heads, etc.).

---

## Purpose

Ensure:
- **Consistency** in signatures and docstrings across architectures.
- **Reliable serialization** through `get_config()` / `from_config()`.
- **Immediate readability** of argument roles.
- Natural **interoperability** between modules (Encoder ↔ Decoder ↔ Head).
- Clear separation between local computation blocks, structured sub-networks, complete models, and builder functions.
- Explicit validation of nested configuration dictionaries passed between components.

## 0 TL;DR — Summary

**Argument order:**  
`dim_*`, `shape_*`, `num_*` → `dropout_*`, `activation_*`, `reg_*` → `type_output`, `head_*` → `mc_*`, `random_state` → `*_kwargs` → `name`, `**kwargs`

**Standard argument names:**  
`dim_in`, `dim_out`, `dim_z`, `shape_in`, `shape_out`, `num_heads`, `dp`, `mc_dropout`,  
`random_state`, `activation`, `type_output`, `logvar_min`, `block_kwargs`

**Clear prefixes:**  
`dim_`, `shape_`, `num_`, `dp_`, `mc_`, `reg_`, `head_`, `*_kwargs`, `*_config`

**Component hierarchy:**  
`Block` → `SubNet` → `Model` → `Builder`


## 1️ General Naming Rules

| Variable Type | Recommended Prefix | Example | Description |
|----------------|--------------------|----------|--------------|
| Feature dimension (vectors) | `dim_` | `dim_in`, `dim_out`, `dim_model`, `dim_z` | Vector space dimensionality |
| Spatial / sequential shape | `shape_` | `shape_in=(H,W)`, `shape_out=(H_out,W_out)` | Tuple of 2D/3D dimensions |
| Number of elements | `num_` / `n_` | `num_heads`, `n_layers`, `n_blocks` | Count of repeated submodules |
| Dropout | `dp_` / `dropout_` | `dp`, `dp_attn`, `dp_ffn` | Dropout rate |
| Regularization | `reg_` | `reg_W`, `reg_bias` | L1/L2 regularization factors |
| Activation | `activation_` | `activation="gelu"`, `activation_output="softmax"` | Activation functions |
| Uncertainty (UQ) | `mc_` / `flag_` | `mc_dropout=True`, `flag_mc=True` | Stochastic behavior toggles |
| Random seed | `random_state` | `random_state=42` | Global random seed |
| Output heads | `head_` / `type_output` | `type_output="EDL"`, `head_type="MC_Dropout"` | Type of output/probabilistic head |
| Nested kwargs | `*cfg_` | `cfg_MLP_Block`, `cfg_CNNEncoder_SubNet` | user-facing or serialized configuration |
| System names | `name` | `name="Encoder_Block"` | Internal Keras name (always last) |



## 2️ Standard Argument Order in `__init__`

All `layers.Layer`, `tf.keras.Model`, `Block`, and `SubNet` classes should follow the same logical argument order:

1. **Structure**: dimensions, shapes, sizes, ratios  
2. **Regularization / Activation**: dropout, activation, normalization, regularization  
3. **Output / Head semantics**: output type, probabilistic head, numerical bounds  
4. **Stochastic / UQ**: Monte Carlo flags, seeds  
5. **component configuration**: `cfg_block`  
6. **System / Keras**: `name`, `**kwargs`

✅ The same argument order applies to:
- `MLPBlock`
- `MLPSubNet`
- `TransformerDecoderBlock`
- `AutoencoderHybrid`
- `VariationalEncoder`
- builder functions such as `make_mlp_model()` or `build_autoencoder()`.

---

## 3️ UQModels-Specific Conventions

| Concept | Recommended Name | Example | Comment |
|----------|------------------|----------|----------|
| Latent dimension | `dim_z` | `dim_z=64` | Autoencoder latent space |
| Input dimension | `dim_in` or `dim_x` | `dim_in=128`, `dim_x=128` | Raw or flattened input dimension |
| Output dimension | `dim_out` or `dim_y` | `dim_out=64` | Raw or flattened output dimension |
| Sequence length / horizon | `dim_horizon` or `seq_len` | `dim_horizon=20` | Temporal or predictive context |
| Input shape | `shape_in` | `shape_in=(T, C)` | Structured input shape excluding batch |
| Output shape | `shape_out` | `shape_out=(T, C)` | Structured output shape excluding batch |
| Monte Carlo dropout | `mc_dropout` | `mc_dropout=True` | Enables dropout during inference when explicitly designed for MC sampling |
| Probabilistic head type | `type_output` | `"MC_Dropout"`, `"EDL"`, `"classif"` | Standardized output type |
| Log-variance bounds | `logvar_min`, `logvar_max` | `logvar_min=-10` | Numerical stability clipping |
| Regularizer on weights | `reg_W` | `reg_W=(1e-5, 1e-5)` | L1/L2 regularization factors |

### Output type normalization

Components should internally normalize `type_output` to avoid casing errors:
Recommended canonical values: [None,"variational","mc_dropout","deep_ensemble","edl","classif"]
---

## 4️ Best Practices for Layer Design

### Serialization
- Always implement `get_config()` and `from_config()` for custom serializable layers and models.
- Include **all** constructor hyperparameters, including seeds and nested kwargs.
- Avoid manual serialization of sublayers (`Dense`, `Dropout`, etc.) if they are created in `__init__` from constructor parameters.
- Store nested kwargs after validation so that `get_config()` returns a clean and reproducible configuration.

### Shape validation

When both a flat dimension and a structured shape are provided, consistency must be checked explicitly.

### Docstring template
Every architecture class/function must include a structured docstring:

## 5️ Block / SubNet / Model / Builder Design Convention

UQModels distinguishes four architectural levels.

| Level | Role | Example |
|---|---|---|
| `Block` | Local reusable computational unit | `MLPBlock`, `TransformerEncoderBlock`, `CNNBlock` |
| `SubNet` | Structured sub-network wrapping one or more blocks | `MLPSubNet`, `CNNEncoderSubNet`, `DecoderSubNet` |
| `Model` | Complete Keras model exposed to the user | `MLPModel`, `AutoencoderModel` |
| `Builder` | Factory function assembling a model | `make_mlp_model`, `build_autoencoder` |

### Responsibility separation

A `Block` must only implement local transformations.
Example: Dense / Activation / Dropout stack

A `SubNet` may manage architectural semantics around blocks:
Example: input reshaping |MLPBlock call |output head |probabilistic or EDL processing |output reshaping 

A `Model` exposes a complete Keras object that can be trained, saved, loaded, and evaluated.

A `Builder` must only assemble components and expose a user-facing Keras model. It must not duplicate internal architectural logic.

## 6️ Configuration Dictionaries and Nested Kwargs

Use `cfg_**` for dictionaries forwarded directly to subcomponents and for higher-level, user-facing, or serialized configurations that are not directly passed as constructor arguments.

Each `SubNet` or composite component that forwards a dictionary to an internal component must explicitly declare the valid keys accepted by the nested component.

Each `SubNet` is responsible for checking the compatibility between its own explicit parameters and the `cfg_<component>` dictionaries that it forwards to internal components.

## 7 Recommended Prefix Reference

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

## 7️ Practical Recommendations

| Rule | Reason | Example |
|------|---------|----------|
| Always include `name` near the end of the signature | Consistent with Keras conventions | `name: str | None = None` |
| Always use **type hints** | Improves clarity and IDE support | `dim_in: int` |
| Prefer safe default values | Avoids configuration mistakes | `dp=0.1`, `activation="gelu"` |
| Avoid ambiguous abbreviations | Improves readability | `dp_rec` → `dropout_recurrent` |
| Centralize constants | Avoids magic numbers | `LOGVAR_MIN_DEFAULT = -10.0` |
| Preserve logical order | Input → Architecture → Output → Stochasticity → Nested kwargs → System | — |
| Validate shape/dimension consistency | Avoids silent reshape errors | `dim_in == np.prod(shape_in)` |
| Validate nested kwargs | Avoids misspelled or conflicting configuration | `BLOCK_ALLOWED_KEYS` |


## 8 File Naming Conventions

| Module Type | Convention | Example |
|--------------|-------------|----------|
| Custom layers / blocks | `layers_<type>.py` | `layers_mlp.py`, `layers_transformer.py` |
| Model builders | `build_<type>.py` | `build_autoencoder.py`, `build_transformer.py` |
| Utilities | `utils_<scope>.py` | `utils_training.py`, `utils_seed.py` |
| Config/constants | `config_<scope>.py` | `config_models.py` |
| Tests | `test_<module>.py` | `test_layers_mlp.py` |

---


**Last updated:** May 2026
