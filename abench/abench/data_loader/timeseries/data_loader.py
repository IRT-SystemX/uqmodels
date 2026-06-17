from abench.store.data_management import explore_csv_hierarchy, filter_paths_from_metadata, enrich_with_descriptors
from abench.data_loader.data_loader import ABLoader,ABDataExperiment
from abench.data_loader.timeseries.scaler import TemporalFeatureScaler
from sklearn.base import BaseEstimator
import os, json, hashlib, inspect, datetime
import numpy as np
import pandas as pd
import warnings
from tqdm import tqdm
from copy import deepcopy
from abench.utils import concat
from abench.data_loader.timeseries.chunk_utils import create_chunk_df,read_chunk,compute_chunk_info
from abench.data_loader.data_loader_utils import update_constraints
from abench.data_loader.timeseries.sampling import _apply_sampling
from typing import List, Tuple, Iterable

def is_scaler_class(obj):
    return isinstance(obj, type) and issubclass(obj, BaseEstimator)

def is_scaler_TemporalFeatureScaler(obj):
    return isinstance(obj, TemporalFeatureScaler) and not isinstance(obj, type)

def make_sequence_filter_cfg(
    segment_id_col: str | None = None,
    mode: str = "same_id_xy",
    enabled: bool = True,
) -> dict | None:
    """
    Build a sequence-window filtering configuration.

    Parameters
    ----------
    segment_id_col : str | None
        Column identifying independent sequence segments.
        If None, no filtering configuration is produced.

    mode : {"same_id_x", "same_id_xy"}
        Filtering strategy:
        - "same_id_x": all X timesteps must belong to the same segment.
        - "same_id_xy": all X and Y timesteps must belong to the same segment.

    enabled : bool
        If False, returns None to preserve the default behavior.

    Returns
    -------
    dict | None
        Sequence filtering configuration compatible with extract_sequence_dataset.
    """
    if not enabled or segment_id_col is None:
        return None

    allowed_modes = {"same_id_x", "same_id_xy"}
    if mode not in allowed_modes:
        raise ValueError(
            f"Invalid sequence filtering mode: {mode}. "
            f"Expected one of {sorted(allowed_modes)}."
        )

    return {
        "segment_id_col": segment_id_col,
        "mode": mode,
    }

def build_sequence_index_matrices(
    df: pd.DataFrame,
    window_size: int,
    horizon_start: int,
    prediction_number: int,
    y_step: int,
    sample_stride: int,
    sequence_filter_cfg: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build vectorized X/Y index matrices for sliding-window extraction.

    If `sequence_filter_cfg["segment_id_col"]` is provided, starts are generated
    independently within each contiguous segment. Otherwise, a global regular
    grid is used.

    Returns
    -------
    x_idx : np.ndarray
        Shape (n_windows, window_size).
    y_idx : np.ndarray
        Shape (n_windows, prediction_number).
    """
    n_rows = len(df)

    if n_rows == 0:
        return (
            np.empty((0, window_size), dtype=int),
            np.empty((0, prediction_number), dtype=int),
        )

    last_required_offset = max(
        window_size - 1,
        horizon_start + (prediction_number - 1) * y_step,
    )

    max_start = n_rows - last_required_offset - 1

    if max_start < 0:
        return (
            np.empty((0, window_size), dtype=int),
            np.empty((0, prediction_number), dtype=int),
        )

    segment_id_col = (
        None
        if sequence_filter_cfg is None
        else sequence_filter_cfg.get("segment_id_col", None)
    )

    if segment_id_col is None:
        starts = np.arange(0, max_start + 1, sample_stride, dtype=int)

    else:
        if segment_id_col not in df.columns:
            raise KeyError(f"Missing segment id column: {segment_id_col}")

        segment_ids = df[segment_id_col].to_numpy()

        is_start = np.empty(n_rows, dtype=bool)
        is_start[0] = True
        is_start[1:] = segment_ids[1:] != segment_ids[:-1]

        seg_starts = np.flatnonzero(is_start)
        seg_ends = np.r_[seg_starts[1:], n_rows]

        starts_parts = [
            seg_start + np.arange(
                0,
                seg_end - seg_start - last_required_offset,
                sample_stride,
                dtype=int,
            )
            for seg_start, seg_end in zip(seg_starts, seg_ends)
            if seg_end - seg_start > last_required_offset
        ]

        starts = (
            np.concatenate(starts_parts)
            if starts_parts
            else np.empty(0, dtype=int)
        )

    if len(starts) == 0:
        return (
            np.empty((0, window_size), dtype=int),
            np.empty((0, prediction_number), dtype=int),
        )

    x_offsets = np.arange(window_size)
    y_offsets = horizon_start + np.arange(prediction_number) * y_step

    x_idx = starts[:, None] + x_offsets[None, :]
    y_idx = starts[:, None] + y_offsets[None, :]

    # Defensive bounds mask.
    valid_bounds = (x_idx[:, -1] < n_rows) & (y_idx[:, -1] < n_rows)

    return x_idx[valid_bounds], y_idx[valid_bounds]

def extract_sequence_dataset(
    df: pd.DataFrame,
    window_size: int = 100,
    sampling: int = 1,
    x_features: list | None = None,
    horizon_start: int | None = None,
    prediction_number: int = 1,
    y_step: int = 1,
    y_features: list | None = None,
    sample_stride: int | None = None,
    context_features: list | None = None,
    seed: int = 0,
    drop_frac: float = 0.0,
    sampling_cfg: dict | None = None,
    feature_engineering: object | None = None,
    sequence_filter_cfg: dict | None = None):
    """
    Build supervised time-series samples ``(X, y, Context)`` from a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Source data ordered in time.
    window_size : int, default 100
        Number of consecutive rows constituting the input window *X*.
    sampling : int, default 1
        Sub-sampling factor; keeps one row every ``sampling`` rows.
    x_features : list, optional
        Columns used as input features *X*. Defaults to **all** columns.
    horizon_start : int, optional
        Offset between the last element of *X* and the first prediction step.
        Defaults to ``window_size`` (predict right after the window).
    prediction_number : int, default 1
        Number of prediction steps in *y*.
    y_step : int, default 1
        Stride (in rows) between successive prediction steps.
    y_features : list, optional
        Columns used as targets *y*. Defaults to ``x_features``.
    sample_stride : int, optional
        Sliding stride when moving the window along the series.
        Defaults to ``window_size``.
    context_features : list, optional
        Columns attached to each output step (e.g. calendar info).
    seed : int, default 0
        Random seed controlling both the initial sampling offset and
        the optional dropping phase.
    drop_frac : float, default 0.0
        Fraction (0 ≤ drop_frac < 1) of valid candidate windows to discard
        uniformly at random after consistency filtering and before tensor extraction.
    feature_engineering : None | Callable | Sequence[Callable] | dict, optional
        Optional feature-engineering step applied on the raw dataframe **before**
        sampling and sequence extraction.
        - If `dict`, it is passed as `enrich_with_descriptors(df, **feature_engineering)`.
        - If `Callable`, it is applied as `df = feature_engineering(df)`.
        - If `Sequence[Callable]`, each callable is applied in order.

    Returns
    -------
    X : np.ndarray  – shape (n_samples, window_size, n_x_features)
    Y : np.ndarray  – shape (n_samples, prediction_number, n_y_features)
    Context : np.ndarray  – shape (n_samples, prediction_number, n_context_features)

    Notes
    -----
    • The function is fully deterministic given the same ``seed``.  
    • If the DataFrame is too short to hold one full window + horizon,
      empty arrays are returned.
    """


    # Snapshot columns BEFORE feature engineering
    base_cols = list(df.columns)
    base_colset = set(base_cols)

    # -------- optional feature engineering (agnostic) ---------------------
    if feature_engineering is not None:
        if isinstance(feature_engineering, dict):
            # Delegates to data_management.enrich_with_descriptors
            df = enrich_with_descriptors(df, **feature_engineering)
        elif callable(feature_engineering):
            df = feature_engineering(df)
        else:
            # Assume iterable of callables
            try:
                for fn in feature_engineering:
                    if not callable(fn):
                        raise TypeError("feature_engineering pipeline contains a non-callable element.")
                    df = fn(df)
            except TypeError as e:
                raise TypeError(
                    "feature_engineering must be None, a dict, a callable, or an iterable of callables."
                    ) from e

    # Detect new columns created by feature engineering
    new_cols = [c for c in df.columns if c not in base_colset]

    # -------- feature fallbacks -------------------------------------------
    if x_features is None:
        x_features = df.columns.tolist()
    else:
        # Append newly created features while preserving order and removing duplicates
        x_features = list(dict.fromkeys(list(x_features) + new_cols))
    if y_features is None:
        y_features = x_features
    if horizon_start is None:
        horizon_start = window_size
    if sample_stride is None:
        sample_stride = window_size

    # -------- deterministic sub-sampling ----------------------------------
    df_sampled = _apply_sampling(df, seed=seed, sampling=sampling, sampling_cfg=sampling_cfg)
    

    # -------- sanity check: required columns exist ------------------------
    def _missing(cols):
        return [c for c in (cols or []) if c not in df_sampled.columns]
    
    missing_x = _missing(x_features)
    missing_y = _missing(y_features)
    missing_c = _missing(context_features)
    if missing_x or missing_y or missing_c:
        raise KeyError(
            "Missing columns after feature_engineering/sampling. "
            f"missing_x={missing_x}, missing_y={missing_y}, missing_context={missing_c}"
            )
    
    # -------- convert to NumPy --------------------------------------------
    x_data = df_sampled[x_features]

    x_data = x_data.drop(columns=x_data.select_dtypes(include=["datetime", "datetimetz"]).columns)
    x_data = x_data.to_numpy()
    y_data = df_sampled[y_features].to_numpy()
    
    context_data = (
        df_sampled.reindex(columns=context_features, fill_value=None).to_numpy()
        if context_features is not None
        else None
    )

    # -------- build vectorized index matrices ------------------------------
    x_idx, y_idx = build_sequence_index_matrices(
        df=df_sampled,
        window_size=window_size,
        horizon_start=horizon_start,
        prediction_number=prediction_number,
        y_step=y_step,
        sample_stride=sample_stride,
        sequence_filter_cfg=sequence_filter_cfg)


    # -------- sequence consistency mask ------------------------------------
    masking_func = build_sequence_window_masker(
        df_sampled,
        sequence_filter_cfg=sequence_filter_cfg)

    valid_sequence_mask = masking_func(x_idx, y_idx)

    x_idx = x_idx[valid_sequence_mask]
    y_idx = y_idx[valid_sequence_mask]

    # -------- random window drop on indices --------------------------------
    if 0.0 < drop_frac < 1.0 and len(x_idx):
        rng = np.random.RandomState(seed)
        keep_n = int(np.round(len(x_idx) * (1.0 - drop_frac)))
        keep_idx = rng.choice(len(x_idx), keep_n, replace=False)

        x_idx = x_idx[keep_idx]
        y_idx = y_idx[keep_idx]

    # -------- empty output guard -------------------------------------------
    if len(x_idx) == 0:
        print('Warning sequence length is 0')
        return (
            np.empty((0, window_size, x_data.shape[-1])),
            np.empty((0, prediction_number, len(y_features))),
            np.empty((0, prediction_number, len(context_features or []))),
        )

    # -------- vectorized extraction -> Stacking ----------------------------------------
    X = x_data[x_idx]
    Y = y_data[y_idx]

    Context = (
        context_data[y_idx]
        if context_features is not None
        else np.empty((len(X), prediction_number, 0))
)
    return X, Y, Context

def build_sequence_window_masker(
    df: pd.DataFrame,
    sequence_filter_cfg: dict | None = None):
    """
    Build a vectorized and memory-efficient window-level mask function.
    """
    if sequence_filter_cfg is None:
        return lambda x_idx, y_idx: np.ones(len(x_idx), dtype=bool)

    if not isinstance(sequence_filter_cfg, dict):
        raise TypeError("sequence_filter_cfg must be None or a dict.")

    segment_id_col = sequence_filter_cfg.get("segment_id_col", None)
    mode = sequence_filter_cfg.get("mode", "same_id_xy")

    if segment_id_col is None:
        return lambda x_idx, y_idx: np.ones(len(x_idx), dtype=bool)

    if segment_id_col not in df.columns:
        raise KeyError(f"Missing segment id column: {segment_id_col}")

    allowed_modes = {"same_id_x", "same_id_xy"}
    if mode not in allowed_modes:
        raise ValueError(
            f"Unknown sequence filtering mode: {mode}. "
            f"Expected one of {sorted(allowed_modes)}."
        )

    segment_ids = df[segment_id_col].to_numpy()

    boundary = np.empty(len(segment_ids), dtype=bool)
    boundary[0] = False
    boundary[1:] = segment_ids[1:] != segment_ids[:-1]

    def masking_func(x_idx: np.ndarray, y_idx: np.ndarray) -> np.ndarray:
        valid_x = ~boundary[x_idx[:, 1:]].any(axis=1)

        if mode == "same_id_x":
            return valid_x

        valid_y = ~boundary[y_idx[:, 1:]].any(axis=1)
        same_xy = segment_ids[x_idx[:, 0]] == segment_ids[y_idx[:, 0]]

        return valid_x & valid_y & same_xy

    return masking_func

class ABLoaderFromSequenceFolder(ABLoader):
    def __init__(self, 
                 path,
                 x_features,
                 y_features,
                 context_features,
                 constraint_selection_list = [], 
                 constraint_rejection_list = [],
                 w_size=120, 
                 sampling=1, 
                 sample_stride=10, 
                 horizon_start=60,
                 prediction_number=3,
                 y_step=60,
                 with_context=True,
                 with_metadata=True,
                 df_metadata = None,
                 depth_name_list=None,
                 Xscaler = None,
                 Yscaler = None,
                 seq_per_chunk=None,
                 shuffle=False,
                 all_data=True,
                 drop_frac=0.0,
                 name='set',
                 dir_cache=None,
                 cache_tag=None,
                 sampling_cfg=None,
                 feature_engineering=None,
                 sequence_filter_cfg=None):
        """
        Initializes a data loader for time series sequences stored in CSV files within a folder structure.
        This loader supports windowing, feature scaling, future target prediction, contextual features,
        and optional metadata filtering or augmentation.

        Parameters
        ----------
        path : str
            Path to the root folder containing CSV sequence files.
        
        x_features : list of str
            List of column names to use as input features (X).
        
        y_features : list of str
            List of column names to use as target outputs (Y).
        
        context_features : list of str
            List of column names to use as contextual (static or auxiliary) features.

        constraint_selection_list : list, optional
            List of selection constraints to filter sequences that meet specific conditions (default is empty).
        
        constraint_rejection_list : list, optional
            List of rejection constraints to exclude sequences that meet specific conditions (default is empty).

        w_size : int, optional
            Size of each input time window (default is 120).

        sampling : int, optional
            Temporal downsampling factor; `1` uses every time step, `2` every other step, etc. (default is 1).

        sample_stride : int, optional
            Step size between the start of consecutive input windows (default is 10).

        horizon_start : int, optional
            Number of time steps ahead from the window end to begin prediction (forecast horizon) (default is 60).

        prediction_number : int, optional
            Number of prediction time steps to generate per window (default is 3).

        y_step : int, optional
            Step between consecutive predicted time points (default is 60).

        with_context : bool, optional
            Whether to include context features as part of the output (default is True).

        with_metadata : bool, optional
            Whether to include file-level metadata along with each sample (default is True).

        df_metadata : pandas.DataFrame, optional
            Optional DataFrame containing metadata for the sequence files. If None, it will be automatically generated by scanning `path`.

        depth_name_list : list of str, optional
            Optional list to define folder hierarchy depth levels when scanning the CSV file tree.

        Xscaler : sklearn-compatible scaler or TemporalFeatureScaler obj, optional
            A scikit-learn-style scaler instance (e.g., `StandardScaler`, `MinMaxScaler`) to apply to input features.
            If TemporalFeatureScaler obj is already fit, it not will be fit again. 

        Yscaler : sklearn-compatible scaler or TemporalFeatureScaler obj, optional
            A scikit-learn-style scaler instance to apply to target values or a TemporalFeatureScaler obj.
            If TemporalFeatureScaler obj is already fit, it not will be fit again. 

        name : str, optional
            Name or label for the current data loader instance (e.g., "train", "val", "test") (default is 'set').

        sampling_cfg : Config form sampling mecanism (see extract sequence)
        
        feature_engineering : Config form features mecanism (see extract sequence)
        """

        if(df_metadata is None):
            df_metadata = explore_csv_hierarchy(path,depth_name_list)

        if df_metadata.empty:
            raise ValueError(path,"doesn't contains data")
            
        self.name = name
        self.sampling = sampling
        self.w_size = w_size
        self.x_features = x_features
        self.horizon_start = horizon_start
        self.prediction_number = prediction_number
        self.y_step = y_step
        self.y_features = y_features
        self.context_features = context_features
        self.sample_stride = sample_stride
        self.seq_per_chunk = seq_per_chunk
        self.shuffle = shuffle
        self.all_data = all_data
        self.drop_frac = drop_frac
        self.dir_cache = dir_cache
        self.cache_tag = cache_tag
        self.sampling_cfg = sampling_cfg
        self.feature_engineering = feature_engineering
        self.sequence_filter_cfg = sequence_filter_cfg

        self.dict_chunk_info = compute_chunk_info(window_size = self.w_size,
                                                  horizon_start = horizon_start,
                                                  prediction_number = prediction_number,
                                                  y_step= y_step,
                                                  sample_stride= sample_stride,
                                                  seq_per_chunk=seq_per_chunk)

        if(Xscaler is None):
            self.Xscaler = None
        elif is_scaler_class(Xscaler):
            self.Xscaler = TemporalFeatureScaler(Xscaler)
        elif is_scaler_TemporalFeatureScaler(Xscaler):
            self.Xscaler = Xscaler
        else:
            print(Xscaler,'is temporalFeatureScaler', isinstance(Xscaler, TemporalFeatureScaler), 'is object', not isinstance(Xscaler, type))
            raise(ValueError('Xscaler should be a Scaler class or a TemporalFeatureScaler obj'))

        if(Yscaler is None):
            self.Yscaler = None
        elif is_scaler_class(Yscaler):
            self.Yscaler = TemporalFeatureScaler(Yscaler)
        elif is_scaler_TemporalFeatureScaler(Yscaler):
            self.Yscaler = Yscaler
        else:
            raise(ValueError('Xscaler should be a Scaler class or a TemporalFeatureScaler obj'))

        self.constraint_selection_list = constraint_selection_list
        self.constraint_rejection_list = constraint_rejection_list
        self.list_path = filter_paths_from_metadata(df_metadata,
                                                    constraint_selection_list=constraint_selection_list,
                                                    constraint_rejection_list=constraint_rejection_list)
        
        self.chunk_dataframe = create_chunk_df(self.list_path,
                                                chunk_size=self.dict_chunk_info['chunk_size'],
                                                offset_before=self.dict_chunk_info['offset_before'],
                                                offset_after=self.dict_chunk_info['offset_after'])
        

        if self.shuffle:
            self.chunk_dataframe.sample(n=len(self.chunk_dataframe), random_state=42)

        metadata = None
        super().__init__(metadata=metadata,with_context=with_context,with_metadata=with_metadata,name=name)

    def get_setname(self):
        setname = super().get_setname()
        return setname
    
    def get_target_arg(self):
        """Required method for ABloader"""
        return self.metadata['target_arg']
        
    def chunk_process(self,n,drop_frac):
        """Process the 'n'th chunk using extract_sequence_dataset

        Args:
            n (_type_): _description_
            drop_frac (_type_): _description_
        """
        # Hold compatibilty bug
        if(hasattr(self,'chunk_dataframe')):
            chunk_dataframe = self.chunk_dataframe
        else:
            chunk_dataframe = self.chunck_dataframe

        chunk_load_info = chunk_dataframe.iloc[n]
        dataframe = read_chunk(chunk_load_info,
                                offset_before=self.dict_chunk_info['offset_before'],
                                offset_after=self.dict_chunk_info['offset_after'])
        
        if(not(hasattr(self,'sampling_cfg'))):
            self.sampling_cfg=None

        X_,Y_,Context_ = extract_sequence_dataset(dataframe,
                                                sampling=self.sampling,
                                                window_size=self.w_size,
                                                sample_stride=self.sample_stride,
                                                horizon_start=self.horizon_start,
                                                prediction_number=self.prediction_number,
                                                y_step=self.y_step,
                                                x_features=self.x_features,
                                                y_features=self.y_features,
                                                context_features=self.context_features,
                                                sampling_cfg=self.sampling_cfg,
                                                feature_engineering=getattr(self, "feature_engineering", None),
                                                sequence_filter_cfg=getattr(self, "sequence_filter_cfg", None),
                                                drop_frac=drop_frac)
        return(X_,Y_,Context_)
        
    def fit_scaler(self,X=None,y=None,drop_frac=0.50):

            # try cache
        if self.dir_cache and self._load_scalers():
            return

        if ((self.Xscaler is None) or (self.Xscaler.is_fitted())) and ((self.Yscaler is None) or (self.Yscaler.is_fitted())):
            pass

        else:
            X_list = []
            Y_list = []
            Context_list = []
            if ((X is None) or (y is None)):
                for n in tqdm(range(len(self.chunk_dataframe))):
                    X_,Y_,Context_ = self.chunk_process(n,drop_frac)
                    X_list.append(X_)
                    Y_list.append(Y_)
    
                X_list = concat(X_list,axis=0)
                Y_list = concat(Y_list,axis=0)
     
            
            self.__pack_and_scale_output__(X_list,Y_list,None)
            if self.dir_cache:
                self._save_scalers()

    def __pack_and_scale_output__(self,X,y,context):
        if(self.Xscaler is not None):
            if not self.Xscaler.is_fitted():
                self.Xscaler.fit(X)
            X = self.Xscaler.transform(X)

        if(self.Yscaler is not None):
            if not self.Yscaler.is_fitted():
                self.Yscaler.fit(y)
            y = self.Yscaler.transform(y)

        output = [(X, y)]
        if self.with_context:
            output.append(context)
        else:
            output.append(None)  # Ensure consistent structure
        if self.with_metadata:
            output.append(self.metadata)
        return(output)
    
    def __iter__(self):
        """
        Itère sur les chunks.
        - Si un cache global (post-scaling) existe, on le sert directement (un seul batch) puis on retourne.
        - Si self.all_data == False : streaming chunk par chunk (scaling à la volée).
        - Sinon : agrégation, shuffle optionnel, scaling, sauvegarde du cache et yield unique.
        """
        # --- Fast path : cache global ---
        if getattr(self, "dir_cache", None):
            cached = self._load_cache()
            if cached is not None:
                Xc, yc, cc = cached
                out = [(Xc, yc)]
                out.append(cc if self.with_context else None)
                if self.with_metadata:
                    out.append(self.metadata)
                yield out
                return

        def _take(obj, perm):
            # pandas DataFrame/Series
            if hasattr(obj, "iloc"):
                return obj.iloc[perm].reset_index(drop=True)
            # numpy array ou "array-like"
            return np.take(obj, perm, axis=0)

        # --- Mode streaming : chunk par chunk ---
        if not self.all_data:
            for n in tqdm(range(len(self.chunk_dataframe))):
                X_, y_, ctx_ = self.chunk_process(n, self.drop_frac)
                yield self.__pack_and_scale_output__(X_, y_, ctx_)
            return

        # --- Mode agrégé : on accumule puis on concatène ---
        X_parts, y_parts, ctx_parts = [], [], []

        # Hold compatibilty bug
        if(hasattr(self,'chunk_dataframe')):
            chunk_dataframe = self.chunk_dataframe
        else:
            chunk_dataframe = self.chunck_dataframe

        for n in tqdm(range(len(chunk_dataframe))):
            X_, y_, ctx_ = self.chunk_process(n, self.drop_frac)
            X_parts.append(X_)
            y_parts.append(y_)
            ctx_parts.append(ctx_)

        if not X_parts:
            return

        X = concat(X_parts, axis=0)
        y = concat(y_parts, axis=0)
        context = concat(ctx_parts, axis=0)

        # Shuffle cohérent si demandé
        if self.shuffle:
            rng = np.random.default_rng(42)
            n_rows = len(X) if hasattr(X, "__len__") else X.shape[0]
            perm = rng.permutation(n_rows)
            X = _take(X, perm)
            y = _take(y, perm)
            context = _take(context, perm)

        # Mise à l'échelle + empaquetage
        output = self.__pack_and_scale_output__(X, y, context)

        # Sauvegarde du cache global (post-scaling) + scalers
        if getattr(self, "dir_cache", None):
            scaled_X, scaled_y = output[0]
            cached_context = output[1] if self.with_context else context
            if cached_context is None:
                # contexte vide mais 3D, partageant la 1ère dimension
                cached_context = np.empty((scaled_X.shape[0], 0, 0))
            self._save_cache(scaled_X, scaled_y, cached_context)
            self._save_scalers()

        yield output
    # ================== CACHE COMPACT ==================
    @staticmethod
    def _stable_hash(obj) -> str:
        j = json.dumps(obj, sort_keys=True, default=str)
        return hashlib.blake2b(j.encode(), digest_size=16).hexdigest()

    def _paths_from_chunk_dataframe(self):
        # suppose une colonne 'path' dans self.chunk_dataframe
        paths = [os.path.abspath(os.path.realpath(str(p)))
                for p in self.chunk_dataframe['path'].tolist()]
        return sorted(set(paths))

    def _cache_key(self):
        fe = getattr(self, "feature_engineering", None)
        fe_fingerprint = None
        if isinstance(fe, dict):
            fe_fingerprint = fe
        elif fe is not None:
            # Best-effort fingerprint for callables/pipelines
            try:
                if callable(fe):
                    fe_fingerprint = {"callable": getattr(fe, "__qualname__", repr(fe))}
                else:
                    fe_fingerprint = {"pipeline": [getattr(f, "__qualname__", repr(f)) for f in fe]}
            except Exception:
                fe_fingerprint = {"feature_engineering": "unhashable"}
        
        payload = {
            "paths": self._paths_from_chunk_dataframe(),
            "feature_engineering": fe_fingerprint,
            "windowing": {
                "w_size": self.w_size,
                "sampling": self.sampling,
                "sample_stride": self.sample_stride,
                "horizon_start": self.horizon_start,
                "prediction_number": self.prediction_number,
                "y_step": self.y_step,
                "x_features": self.x_features,
                "y_features": self.y_features,
                "context_features": self.context_features,
                "sampling_cfg": getattr(self, "sampling_cfg", None),
                "sequence_filter_cfg": getattr(self, "sequence_filter_cfg", None),
},
                }
        return self._stable_hash(payload)

    def _base_cache(self):
        if not self.dir_cache: return None
        os.makedirs(self.dir_cache, exist_ok=True)
        key = self._cache_key()
        prefix = f"{self.cache_tag}_" if self.cache_tag else ""
        return os.path.join(self.dir_cache, f"{prefix}{key}")

    def _cache_paths(self):
        base = self._base_cache()
        if not base: return None, None
        return base + ".npz", base + ".json"

    def _scaler_cache_paths(self):
        base = self._base_cache()
        if not base: return None, None
        return base + ".xscaler.pkl", base + ".yscaler.pkl"

    def _save_cache(self, X, y, context):
        npz_path, meta_path = self._cache_paths()
        if not npz_path: return
        X = np.asarray(X); y = np.asarray(y); context = np.asarray(context)
        tmp = npz_path + ".tmp"
        with open(tmp, "wb") as f:
            np.savez_compressed(f, X=X, y=y, context=context)
        os.replace(tmp, npz_path)
        if meta_path:
            meta_tmp = meta_path + ".tmp"
            with open(meta_tmp, "w", encoding="utf-8") as f:
                json.dump({"created_utc": datetime.datetime.utcnow().isoformat()+"Z",
                        "key": self._cache_key()}, f)
            os.replace(meta_tmp, meta_path)

    def _load_cache(self):
        npz_path, _ = self._cache_paths()
        if not npz_path or not os.path.exists(npz_path): return None
        with np.load(npz_path, allow_pickle=True) as d:
            return d["X"], d["y"], d["context"]
        
    def _save_scalers(self):
        x_path, y_path = self._scaler_cache_paths()
        if x_path and self.Xscaler is not None:
            xt = x_path + ".tmp"
            with open(xt, "wb") as f: pickle.dump(self.Xscaler, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(xt, x_path)
        if y_path and self.Yscaler is not None:
            yt = y_path + ".tmp"
            with open(yt, "wb") as f: pickle.dump(self.Yscaler, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(yt, y_path)

    def _load_scalers(self):
        x_path, y_path = self._scaler_cache_paths()
        ok = False
        if x_path and os.path.exists(x_path) and self.Xscaler is not None:
            with open(x_path, "rb") as f: self.Xscaler = pickle.load(f); ok = True
        if y_path and os.path.exists(y_path) and self.Yscaler is not None:
            with open(y_path, "rb") as f: self.Yscaler = pickle.load(f); ok = True
        return ok

class ABCVDataExperiment(ABDataExperiment):
    def __init__(self,
                 ABloader,
                 ABloader_dict_params,
                 depth_name,
                 subjet_ids,
                 with_test=True,
                 validation_config=[],
                 name='cv_experiment'):
        
        ABtrainloader_list = []
        ABtestloader_sets_list = []
        for n_s,subject in enumerate(subjet_ids):
            print(subject)
            #Train set
            ABloader_dict_params_cur = deepcopy(ABloader_dict_params)
            ABloader_dict_params_cur['name'] = ABloader_dict_params_cur['name']+'_Train_LOSO_'+subject
            ABloader_dict_params_cur['constraint_rejection_list'].append((depth_name,[subject])) 
            ABtrainloader_list.append(ABloader(**ABloader_dict_params_cur))
            #Test set
            if(with_test):
                ABloader_dict_params_cur = deepcopy(ABloader_dict_params)
                ABloader_dict_params_cur['name'] = ABloader_dict_params_cur['name']+'_Test_LOSO_'+subject
                ABloader_dict_params_cur['constraint_selection_list'].append((depth_name,[subject]))
                ABtestloader_sets_list.append([ABloader(**ABloader_dict_params_cur)])
            else:
                ABtestloader_sets_list.append([])

            
            for name_valid_setup, dict_config in validation_config.items():
                #Other validation set
                ABloader_dict_params_cur = deepcopy(ABloader_dict_params)
                ABloader_dict_params_cur['name'] = ABloader_dict_params_cur['name']+'_Valid_LOSO_'+subject+"_"+name_valid_setup
                ABloader_dict_params_cur['constraint_selection_list'].append((depth_name,[subject]))

                constraint_selection_list = ABloader_dict_params_cur['constraint_selection_list']
                for new_constraint_selection in dict_config['constraint_selection']:
                    constraint_selection_list = update_constraints(constraint_selection_list,
                                                                   new_constraint_selection)
                    
                constraint_rejection_list = ABloader_dict_params_cur['constraint_rejection_list']
                for new_constraint_rejection in dict_config['constraint_rejection']:
                    constraint_rejection_list = update_constraints(constraint_rejection_list,
                                                                   new_constraint_rejection)

                ABloader_dict_params_cur['constraint_selection_list'] = constraint_selection_list
                ABloader_dict_params_cur['constraint_rejection_list'] = constraint_rejection_list
                ABtestloader_sets_list[n_s].append(ABloader(**ABloader_dict_params_cur))
        super().__init__(ABtrainloader_list,ABtestloader_sets_list,name=name)

class ABLosoDataExperiment(ABCVDataExperiment):
    def __init__(self, *args, **kwargs):

        if not("name" in kwargs):
            kwargs['name'] = 'loso_experiment'
        warnings.warn(
            "Class 'OldComponent' is deprecated and will be removed in a future release. "
            "Use 'Component' instead.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(*args, **kwargs)


# --------------- Fonction d'API Configuration ---------
        
