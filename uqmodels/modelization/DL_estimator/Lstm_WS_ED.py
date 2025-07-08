import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.layers import Input, Lambda, TimeDistributed
from keras.models import Model

from uqmodels.modelization.DL_estimator.metalayers import (
    Add_query_to_Z_Processing_with_state,
    Double_Moving_slice_layer,
    LSTM_DProcessing,
    LSTM_EProcessing,
    Moving_slice_layer,
    mlp,
)
from uqmodels.modelization.UQEstimator import UQEstimator
from uqmodels.utils import Extract_dict, apply_mask, cut, get_fold_nstep, stack_and_roll

tf.keras.backend.set_floatx("float32")

# Dev : Prototype of DEEP Neural network for dynamic capture


# LSTM ED class : Faire hériter LSTM ED de NNvar


class Lstm(UQEstimator):
    # LSTM E-D forecast on a sequence ensemble (day) of element (time step) multidimensional (station).
    # Model learning on daily sequence object by reccurent process.
    # y_encoder : (batch_size,dim_z)
    # y_decoder : (batch_size,size_subseq_dec,dim_target*n_parameters)

    # Init that not instanciate the NN (just store parameters&instanciator needed for instanciation)
    def __init__(
        self,
        model_parameters,
        model_specifications,
        architecture_parameters,
        training_parameters=dict(),
    ):
        # State variable

        super().__init__(
            model_specifications["name"], "var_A&E", model_specifications["rescale"]
        )

        self.initialized = False
        self.history = []
        self.type_output = model_specifications["type_output"]
        self.model_specifications = model_specifications
        self.model_parameters = model_parameters
        self.training_parameters = {
            "epochs": [1000, 1000],
            "b_s": [100, 20],
            "l_r": [0.01, 0.005],
            "verbose": 1,
            "list_loss": ["mse"],
            "metrics": None,
            "param_loss": None,
            "redundancy": 1,
        }

        for key in training_parameters.keys():
            self.training_parameters[key] = training_parameters[key]

        list_keys = [
            "size_window_past",
            "size_window_futur",
            "size_subseq_dec",
            "size_subseq_enc",
            "padding_past",
            "padding_futur",
        ]

        (
            size_window_past,
            size_window_futur,
            size_subseq_dec,
            size_subseq_enc,
            padding_past,
            padding_futur,
        ) = Extract_dict(model_parameters, list_keys)

        if not (
            ((size_window_past) % padding_past == 0)
            & ((size_window_futur - size_subseq_dec) % padding_futur == 0)
        ):
            raise Exception("size window, subsequence & padding incompatible")

        self.training_parameters = {
            "epochs": [1000, 1000],
            "b_s": [100, 20],
            "l_r": [0.01, 0.005],
            "verbose": 1,
            "list_loss": ["mse"],
            "metrics": None,
            "param_loss": None,
            "generator": True,
        }

        for key in training_parameters.keys():
            self.training_parameters[key] = training_parameters[key]

        # Parameter variable

        list_keys = ["dim_lt", "dim_target", "dim_target_in", "dim_z", "dp", "dp_r"]

        dim_lt, dim_target, dim_target_in, dim_z, dp, dp_r = Extract_dict(
            self.model_parameters, list_keys
        )

        dict_to_add = {
            "dyn_encoder": {},
            "ctx_decoder": {"dim_in": 10},  # Unused for the moment
            "ctx_encoder": {
                "dim_in": dim_lt,
                "dp": dp,
                "type_output": None,
                "dim_out": None,
            },
            "y_encoder": {
                "dp": dp,
                "type_output": None,
                "dim_out": None,
            },
            "y_decoder": {
                "dim_in": dim_z,
                "dp": dp,
                "type_output": self.type_output,
                "dim_out": size_subseq_dec * dim_target,
            },
        }
        # filling missing subpart
        for subpart in ["ctx_encoder", "y_encoder", "dyn_encoder", "y_decoder"]:
            if subpart not in architecture_parameters.keys():
                architecture_parameters[subpart] = {"builder": mlp, "name": subpart}

        for subpart in architecture_parameters.keys():
            for add_key in dict_to_add[subpart].keys():
                add_values = dict_to_add[subpart][add_key]
                architecture_parameters[subpart][add_key] = add_values

        self.architecture_parameters = architecture_parameters

        self.model = None
        self.model_enc = None
        self.model_pred = None

    # Model instantiation from parameters&instanciator.
    def model_initializer(self):
        # New functionality disabled : preform ctx + dynamic pred.
        with_ctx_predictor = False

        list_keys = [
            "dim_lt",
            "dim_target_in",
            "dim_target",
            "dim_z",
            "dim_st",
            "dp",
            "dp_r",
        ]
        flag_mc = None
        if (self.type_output == "MC_Dropout") or (self.type_output == "MCDP"):
            flag_mc = 1

        dim_lt, dim_target_in, _dim_target, dim_z, dim_st, dp, dp_r = Extract_dict(
            self.model_parameters, list_keys
        )
        if dp_r is None:
            dp = dp_r

        list_keys = [
            "with_loop",
            "only_futur",
            "with_dyn_feature",
            "with_static_feature",
            "with_query",
        ]

        (
            with_loop,
            only_futur,
            with_dyn_feature,
            with_static_feature,
            with_query,
        ) = Extract_dict(self.model_specifications, list_keys)

        list_keys = [
            "size_window_past",
            "size_window_futur",
            "size_subseq_dec",
            "size_subseq_enc",
            "padding_past",
            "padding_futur",
        ]

        (
            size_window_past,
            size_window_futur,
            size_subseq_dec,
            size_subseq_enc,
            padding_past,
            padding_futur,
        ) = Extract_dict(self.model_parameters, list_keys)

        # Instanciation of the y_encoder subpart
        y_encoder = self.architecture_parameters["y_encoder"]["builder"](
            **self.architecture_parameters["y_encoder"]
        )
        # Instanciation of the y_decoder subpart
        y_decoder = self.architecture_parameters["y_decoder"]["builder"](
            **self.architecture_parameters["y_decoder"]
        )

        n_step_futur = get_fold_nstep(size_window_futur, size_subseq_dec, padding_futur)

        n_step_past = get_fold_nstep(
            size_window_past + size_subseq_enc, size_subseq_enc, padding_past
        )

        if with_ctx_predictor:
            # ctx_decoder = self.architecture_parameters["ctx_decoder"]["builder"](
            #     **self.architecture_parameters["ctx_decoder"]
            # )
            self.architecture_parameters["dim_in"] = dim_z

        # Main layer definition.
        # Factory : Contextual preprocessing layers.

        z_embedding_past = []
        list_input = []

        # Contextual features (dim_t,dim_lt,dim_n) -> (dim_t,dim_lt,batch size).
        if with_static_feature or with_query:
            # Instanciation of the ctx_encoder subpart
            ctx_encoder = self.architecture_parameters["ctx_encoder"]["builder"](
                **self.architecture_parameters["ctx_encoder"]
            )

            full_query = size_window_past + 1
            ignore_futur = None

            if with_query:
                ignore_futur = full_query
                full_query = full_query + size_window_futur

            Input_query = Input(shape=(full_query, dim_lt), name="f_static_past")

            list_input.append(Input_query)
            # Defintiion of static encoder part
            if with_static_feature:
                # Select and fold pastZ_futur_query
                past_selected_query = Moving_slice_layer(
                    n_step_past, padding_past, ignore_futur=ignore_futur
                )(Input_query)
                # Encode pastZ_futur_query
                Z_past_query = TimeDistributed(ctx_encoder, name="TD_ctx_encoder")(
                    past_selected_query
                )
                z_embedding_past.append(Z_past_query)

            if with_query:
                # Select and double fold "futur"Z_futur_query for prediction
                futur_query = Double_Moving_slice_layer(
                    n_step_past,
                    n_step_futur,
                    padding_past,
                    padding_futur,
                    depth_slice=1,
                )(Input_query)

                Z_futur_query = TimeDistributed(
                    TimeDistributed(ctx_encoder), name="TD2_futur_query_encoder"
                )(futur_query)

        # Dynamic features -> (dim_t,size_window_futur,dim_target,dim_n) ->
        # (dim_t,size_window_futur,dim_target, batch size)
        if with_dyn_feature:
            dyn_encoder = self.architecture_parameters["dyn_encoder"]["builder"](
                **self.architecture_parameters["dyn_encoder"]
            )

            x_dyn_past = Input(shape=(size_window_past, dim_st), name="f_dyn_past")
            list_input.append(x_dyn_past)
            # Reformatting as a 3D tensor : (sequence of subsequence)
            T_dyn_past = Moving_slice_layer(
                n_step_past, padding_past, depth_slice=size_subseq_enc
            )(x_dyn_past)

            # Dynamic futur embedding
            z_dyn_past = TimeDistributed(dyn_encoder, name="TD_dyn_encoder")(T_dyn_past)
            z_embedding_past.append(z_dyn_past[:, 1:, :])

        x_lag_past = Input(
            shape=(size_window_past + size_subseq_enc, dim_target_in),
            name="f_lag_past",
        )

        list_input.append(x_lag_past)
        # Reformatting as a 3D tensor : (sequence of subsequence)

        x_lag_stacked = Moving_slice_layer(
            n_step_past, padding_past, depth_slice=size_subseq_enc
        )(x_lag_past)
        z_lag_past = TimeDistributed(y_encoder, name="TD_y_encoder")(x_lag_stacked)

        z_embedding_past.append(z_lag_past)
        z_lag_futur = z_lag_past

        # Dynamic features -> (dim_t,size_window_futur,dim_target,dim_n) ->
        # (dim_t,size_window_futur,dim_target, batch size)

        # Preprocessing interconnexion Concatenate Past ctx+dyn representation
        input_Processing = Lambda(lambda x: K.concatenate(x), name="Concat_past_z")(
            z_embedding_past
        )

        # Instanciation of the Encoder block
        EProcessing = LSTM_EProcessing(
            n_step_past,
            input_Processing.shape[-1],
            dim_z,
            flag_mc,
            dp=dp,
            dp_r=dp_r,
            l1_l2_reg=(0.0000001, 0.0000001),
        )

        # Instanciation of the Decoder block
        DProcessing = LSTM_DProcessing(
            n_step_futur,
            dim_z,
            flag_mc,
            dp=dp,
            dp_r=dp_r,
            l1_l2_reg=(0.0000001, 0.0000001),
        )

        # Improvement : Better Fusion / Attention mecanisme between feature type.
        # Actually : Done be Fusion are realized using basic dense layers

        # Encoding
        Z_processing = EProcessing(input_Processing)

        # If mode with-out explicit loop
        if not with_loop:
            # Duplication of Z_processing to feed DPprocessing block
            T_lstm_enc = Lambda(
                lambda x: K.repeat_elements(x[:, :, None, :], n_step_futur, 2),
                name="repeat_h1",
            )(Z_processing)

            print("log : mode_init", T_lstm_enc.shape, Z_futur_query.shape)

            # Add Z_query_information for each futur prediction
            if with_query:
                T_enc_futur = Add_query_to_Z_Processing_with_state(dim_z)(
                    T_lstm_enc, Z_futur_query
                )

            # No Z_query_information
            else:
                T_enc_futur = T_lstm_enc

            print(
                "log : mode_init",
                T_lstm_enc.shape,
                Z_futur_query.shape,
                T_enc_futur.shape,
            )

            # Apply DProcessing block : For each step_representation (including present-step)
            if not only_futur:
                z_lstm_dec = TimeDistributed(DProcessing, name="TD_lstm_dec")(
                    T_enc_futur
                )

                print("z_lstm_dec", z_lstm_dec.shape)
                # Interpret Z_DProcessing for past windows and futur window
                output = TimeDistributed(
                    TimeDistributed(y_decoder), name="TD_TD_y_decoder"
                )(z_lstm_dec)
                print("TD_TD_y_decoder", output.shape)
                # if with_ctx_predictor:
                #    print(Z_futur_query.shape)
                #    output_ctx = TimeDistributed(TimeDistributed(
                #        ctx_decoder), name='TD_TD_ctx_decoder')(Z_futur_query)
                #    output = output_ctx+output

            # Apply & DProcessing & y_decoder only for present-step_representation

            z_lstm_dec_futur = DProcessing(T_enc_futur[:, -1, :, :])
            print("z_lstm_dec_futur", z_lstm_dec_futur.shape)
            output_pred_NN = TimeDistributed(y_decoder)(z_lstm_dec_futur)
            print("output_pred_NN", output_pred_NN.shape)
            # if with_ctx_predictor:
            #    output_pred_NN_bis = TimeDistributed(
            #        ctx_decoder)(Z_futur_query[:, -1, :, :])
            #    output_pred_NN = output_pred_NN+output_pred_NN_bis

        # Loop learning
        else:
            prediction = []

            for i in range(1):  # Init loop
                if with_query:
                    T_enc_futur_next = Lambda(
                        lambda x: K.concatenate(x)[:, :, None, :], name="Concat_T_futur"
                    )(
                        [
                            Z_processing,
                            Z_futur_query[:, :, 0],
                            z_lag_futur[:, :, -1],
                        ]
                    )
                else:
                    T_enc_futur_next = Lambda(
                        lambda x: K.concatenate(x)[:, :, None, :], name="Concat_T_futur"
                    )([Z_processing, z_lag_futur[:, :, -1]])

                z_lstm_dec = DProcessing(T_enc_futur_next)
                pred_temp = TimeDistributed(y_decoder)(z_lstm_dec[:, :, 0])
                prediction.append(pred_temp[:, :, None])
                z_lag_futur = TimeDistributed(y_encoder)(pred_temp)

            for i in range(n_step_futur - 1):
                if with_query:
                    T_enc_futur_next = Lambda(
                        lambda x: K.concatenate(x)[:, :, None, :], name="Concat_T_futur"
                    )(
                        [
                            Z_processing[:, i : n_step_past + i],
                            z_lag_futur[:, :, 0, :],
                            z_lstm_dec[:, :, 0, :],
                        ]
                    )
                else:
                    T_enc_futur_next = Lambda(
                        lambda x: K.concatenate(x)[:, :, None, :], name="Concat_T_futur"
                    )([z_lag_futur[:, :, 0, :], z_lstm_dec[:, :, 0, :]])

                z_lstm_dec = TimeDistributed(DProcessing)(T_enc_futur_next)
                pred_temp = TimeDistributed(y_decoder)(z_lstm_dec[:, :, 0])
                prediction.append(pred_temp[:, :, None])
                z_lag_futur = TimeDistributed(y_encoder)(pred_temp)

            output = Lambda(
                lambda x: K.concatenate(x)[:, :, :], axis=2, name="Concat_output"
            )(prediction)

        # M : FULL model

        if size_subseq_dec == 0:
            output = output[:, :, :, :]
            output_pred_NN = output_pred_NN[:, :, :]

        if not (only_futur):
            self.model = Model(list_input, output)
            self.model_pred = Model(list_input, output_pred_NN)
        else:
            self.model = Model(list_input, output_pred_NN)
        self.model_enc = Model(list_input, Z_processing)

        # Mcut Partial model : Encoded embedding recovering.
        # self.modelcut = Model(inputs, c_h1)

    # Fit procedure Maybe partly redundant to NN fiting procedure : Refactoring needed
    def fit(
        self,
        Inputs,
        Targets=None,
        validation_data=None,
        epochs=None,
        steps_per_epoch=None,
        b_s=None,
        l_r=None,
        list_loss=None,
        param_loss=None,
        shuffle=True,
        sample_weight=None,
        verbose=None,
        metrics=None,
        callbacks=None,
        generator=None,
        validation_freq=1,
        **kwargs
    ):
        if generator is None:
            generator = self.training_parameters["generator"]

        if epochs is None:
            epochs = self.training_parameters["epochs"]

        if callbacks is None:
            callbacks = self.training_parameters["callbacks"]

        if b_s is None:
            b_s = self.training_parameters["b_s"]

        if l_r is None:
            l_r = self.training_parameters["l_r"]

        if list_loss is None:
            list_loss = self.training_parameters["list_loss"]

        if param_loss is None:
            param_loss = self.training_parameters["param_loss"]

        if metrics is None:
            metrics = self.training_parameters["metrics"]

        if verbose is None:
            verbose = self.training_parameters["verbose"]

        if not (self.initialized):
            self.model_initializer()
            self.initialized = True

        # Training function
        history = []

        if validation_data is None:
            np.random.seed(0)
            if isinstance(Inputs, list):
                train = np.random.rand(len(Inputs[0])) < 0.9
            else:
                train = np.random.rand(len(Inputs)) < 0.9

            test = np.invert(train)
            validation_data = (apply_mask(Inputs, test), apply_mask(Targets, test))
            Targets = apply_mask(Targets, train)
            Inputs = apply_mask(Inputs, train)

        if not (hasattr(self, "scaler")):
            _, _, _ = self.factory(Inputs, Targets, only_fit_scaler=True)

        for n, loss in enumerate(list_loss):
            for i, batch_size in enumerate(b_s):
                if param_loss is not None:
                    loss_ = loss(param_loss[n])
                else:
                    loss_ = loss
                print("l_r", l_r[i])
                self.model.compile(
                    optimizer=tf.optimizers.Nadam(learning_rate=l_r[i]),
                    loss=loss_,
                    metrics=metrics,
                )
                # Instanciate generator

                if generator:
                    In_ = self.Build_generator(
                        Inputs,
                        Targets,
                        batch_min=batch_size,
                        shuffle=shuffle,
                        train=True,
                    )
                    Tar_ = None
                    validation_data_ = self.Build_generator(
                        validation_data[0],
                        validation_data[1],
                        batch_min=batch_size,
                        shuffle=False,
                        train=False,
                    )
                    if steps_per_epoch is None:
                        steps_per_epoch = In_.__len__()
                    validation_steps = validation_data_.__len__()
                    batch_size = None
                else:
                    In_ = Inputs
                    Tar_ = Targets
                    validation_data_ = validation_data
                    validation_steps = None
                    steps_per_epoch = None
                self.history.append(
                    self.model.fit(
                        In_,
                        Tar_,
                        validation_data=validation_data_,
                        epochs=epochs[i],
                        steps_per_epoch=steps_per_epoch,
                        validation_steps=validation_steps,
                        batch_size=batch_size,
                        # sample_weight=sample_weight,
                        shuffle=shuffle,
                        callbacks=callbacks,
                        validation_freq=validation_freq,
                        verbose=verbose,
                    )
                )
                self.save("", "lstm_tmp")

        return history

    def predict(self, Inputs, n_ech=6, mask_h=0, mask_m=[0], generator=None, **kwargs):
        """Predict procedure Maybe partly redundant to NN fiting procedure : Refactoring needed
        Args:
            Inputs (List of NN features): Model inputs
            n_ech (int, optional): Number of MC-DP inferences.

        Returns:
            output: Meta-model output tuple of (Prediction,Var_A,Var_E)
        """
        if generator is None:
            generator = self.training_parameters["generator"]

        if generator:
            Inputs = self.Build_generator(
                Inputs, Inputs[1], batch_min=5000, shuffle=False, train=False
            )

        dim_out = self.model_parameters["dim_target"]
        only_futur = self.model_specifications["only_futur"]
        if only_futur:
            model = self.model
        else:
            model = self.model_pred

        var_a, var_e, output = None, None, None

        if self.type_output is None:
            pred = model.predict(Inputs)[:, mask_h, mask_m]
            var_a, var_e = pred * 0 + 1, pred * 0
            UQ = np.concatenate([var_a[None, :], var_e[None, :]], axis=0)

        elif self.type_output == "EDL":
            out = model.predict(Inputs)[:, mask_h, mask_m]
            gamma, vu, alpha, beta = tf.split(out, 4, -1)
            if hasattr(gamma, "numpy"):
                gamma = gamma.numpy()

            pred = gamma
            var_a = beta / (alpha - 1)
            var_e = beta / (vu * (alpha - 1))
            UQ = np.concatenate([var_a[None, :], var_e[None, :]], axis=0)

        elif self.type_output == "MC_Dropout":
            if generator:
                pred = []
                var_a = []
                var_e = []
                for Inputs_gen, _ in Inputs:  # by batch do n inf and aggreagate results
                    output = []
                    for i in range(n_ech):
                        output.append(model.predict(Inputs_gen)[:, mask_h, mask_m])

                    output = np.array(output)
                    pred.append(output[:, :, :, :dim_out].mean(axis=0))
                    var_a.append(np.exp(output[:, :, :, dim_out:]).mean(axis=0))
                    var_e.append(output[:, :, :, :dim_out].var(axis=0))

                pred = np.concatenate(pred, axis=0)
                var_a = np.concatenate(var_a, axis=0)
                var_e = np.concatenate(var_e, axis=0)
                UQ = np.concatenate([var_a[None, :], var_e[None, :]], axis=0)

            else:
                output = []
                for i in range(n_ech):
                    output.append(model.predict(Inputs)[:, mask_h, mask_m])
                output = np.array(output)
                pred = output[:, :, :, :dim_out].mean(axis=0)
                var_a = np.exp(output[:, :, :, dim_out:]).mean(axis=0)
                var_e = output[:, :, :, :dim_out].var(axis=0)
            UQ = np.concatenate([var_a[None, :], var_e[None, :]], axis=0)

        _, pred = self._format(None, pred, "inverse_transform")
        _, UQ = self._format(None, UQ, "inverse_transform", mode_UQ=True)
        return (pred, UQ)

    def _format(self, X, y, type_transform, mode_UQ=None):
        """Formatting features : Rework needed be : would be call to supermethod of NN_UQ

        Args:
            X (_type_): Input to reformate
            y (_type_): Target or output (Pred or var) to reforma
            fit (bool, optional): True -> fit scaler
            mode (_type_, optional): Specify the nature of y than impact reformatting (if pred or var)
            flag_inverse (bool, optional): Specify if normalisation or inverse_normalisation

        Returns:
            _type_: tupple (X,y)
        """
        if self.rescale:
            shape = y.shape
            if y is not None and len(y.shape) == 3:
                print("log format : strange_reshape :", y.shape)
                size_motif = y.shape[1]
                y_new = [
                    super(Lstm, self)._format(X, y[:, i, :], type_transform, mode_UQ)[
                        1
                    ][:, None, :]
                    for i in range(size_motif)
                ]
                y = np.concatenate(y_new, axis=1)
            else:
                X, y = super(Lstm, self)._format(X, y, type_transform, mode_UQ)
                if y.shape != shape:
                    y = y.reshape(shape)

        y = np.squeeze(y)
        return X, y

    def factory(
        self,
        X,
        y,
        mask=None,
        cut_param=None,
        fit_rescale=True,
        causality_remove=None,
        redundancy=None,
        only_fit_scaler=False,
    ):
        """Feature factory Reshape and redundundization (Moving window embedding representation)

        Args:
            X (_type_): X list contains (X_ctx, X_seq)
            y (_type_): Raw Y
            mask (_type_, optional): Mask of non-data
            cut_param (_type_, optional): cut paramaters on y distrubution

        Returns:
            (Inputs,targets,mask): model Inputs, Targets and mask.
        """

        X_ctx, X_seq = X

        with_static_feature = self.model_specifications["with_static_feature"]
        only_futur = self.model_specifications["only_futur"]
        with_query = self.model_specifications["with_query"]

        if redundancy is None:
            redundancy = self.training_parameters["redundancy"]

        if y is None:
            print("Factory log : Predict mode")
            redundancy = 1

        if cut_param is None:
            y_cut = y

        else:
            min_cut, max_cut = cut_param
            y_cut = cut(y, min_cut, max_cut)
            X_seq = cut(X_seq, min_cut, max_cut)

        X_ctx, y_cut = self._format(X_ctx, y_cut, type_transform="fit_transform")
        X_ctx, X_seq = self._format(X_ctx, X_seq, type_transform="fit_transform")

        if only_fit_scaler:
            return ([X_ctx, X_seq], y_cut, None)

        list_keys = [
            "size_window_past",
            "size_window_futur",
            "size_subseq_dec",
            "size_subseq_enc",
            "padding_past",
            "padding_futur",
        ]

        (
            size_window_past,
            size_window_futur,
            size_subseq_dec,
            size_subseq_enc,
            padding_past,
            padding_futur,
        ) = Extract_dict(self.model_parameters, list_keys)

        paramaters = self.model_parameters

        # Factory ctx_features !
        horizon = size_window_past + 1

        Inputs = []
        if with_static_feature:
            lag = 0
            if "factory_lag_lt" in paramaters.keys():
                lag = paramaters["factory_lag_lt"]

            if "factory_lag_target" in paramaters.keys():
                lag = +paramaters["factory_lag_target"]

            if not (with_query):
                horizon = size_window_past + 1
                x_lt = stack_and_roll(X_ctx, horizon, lag=lag)
                Inputs.append(x_lt)
            else:
                horizon_f = size_window_futur
                x_query = stack_and_roll(
                    X_ctx, horizon + horizon_f, lag=lag + horizon_f
                )
                if causality_remove is not None:
                    for i in range(1, horizon_f):
                        x_query[:, horizon + i, causality_remove] = x_query[
                            :, horizon, causality_remove
                        ]

                Inputs.append(x_query)

        # Factory lag_features :
        horizon = size_window_past + size_subseq_enc
        lag = 0
        if "factory_lag_st" in paramaters.keys():
            lag = paramaters["factory_lag_st"]

        x_lag_lstm = stack_and_roll(X_seq, horizon, lag=lag - 1)
        Inputs.append(x_lag_lstm)

        # Factory Target :
        lag = 0
        if "factory_lag_target" in paramaters.keys():
            lag = paramaters["factory_lag_target"]

        horizon = size_window_past + size_window_futur

        y_lstm = stack_and_roll(y_cut, horizon, lag=size_window_futur - lag - 1)

        y_deep = double_roll(
            y_lstm,
            size_window_past,
            size_window_futur,
            size_subseq_dec,
            padding_past,
            padding_futur,
            only_futur,
        )

        Targets = [y_deep.astype(np.float32)]

        if redundancy < 1:
            selection = np.ones(len(Inputs[0])) == 1
            removed = np.random.choice(
                np.nonzero(selection)[0],
                int(len(Inputs[0]) * (1 - redundancy)),
                replace=False,
            )
            selection[removed] = False
            Inputs = apply_mask(Inputs, selection)
            Targets = apply_mask(Targets, selection)
            # mask = apply_mask(mask, selection)
        return (Inputs, Targets, mask)

    def modify_dropout(self, dp):
        """Method to modify dp : have to be extend as API modification to model features.

        Args:
            dp (_type_): ddropout rate.
        """

        # For dropout, as rate is implictly saved in layers, we create new model with new dropout and transfer weight
        # Maybe be optimized in only modify layers paramters !
        self.model_parameters["dp"] = dp
        self.model.save_weights("test")
        self.model_initializer()
        self.model.load_weights("test")

    def get_params(self):
        # get parameters, refactoring needed to follows scikit standars.
        dict_param = {
            "model_parameters": self.model_parameters,
            "architecture_parameters": self.architecture_parameters,
            "model_specifications": self.model_specifications,
            "training_parameters": self.training_parameters,
        }
        return dict_param

    def save(self, path, name=None):
        # save model
        if name:
            self.model.save_weights(path + name)
        else:
            self.model.save_weights(path + "Lstm_ed_" + self.name)

    def load(self, path, name=None):
        # load saved model
        if name:
            self.model.load_weights(path + name)
        else:
            self.model.load_weights(path + "Lstm_ed_" + self.name)

    def reset(self):
        # reset model
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "model_enc"):
            del self.model_enc
        if hasattr(self, "model_pred"):
            del self.model_pred
        self.initialized = False

    def delete(self):
        # delete model
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "model_enc"):
            del self.model_enc
        if hasattr(self, "model_pred"):
            del self.model_pred
        del self.model_parameters
        del self.architecture_parameters
        del self.model_specifications
        del self.training_parameters

    def Build_generator(self, X, y, batch_min=32, shuffle=True, train=True):
        return LSTM_ED_Generator(
            X, y, self, batch_min=batch_min, shuffle=shuffle, train=train
        )

    def plot_metrics(self, name_loss="val_loss"):
        """Plot metrics values recovers form tensorflow metrics callback

        Args:
            name_loss (str, optional):metrics to visualisze.
        """
        metrics = [history.history[name_loss] for history in self.history]
        phase = [len(i) for i in metrics]
        metrics = np.concatenate(metrics)
        plt.figure()
        plt.plot(metrics)
        for i in phase:
            plt.vlines(i, metrics.min(), metrics.max(), ls="--", label="end phase")
        plt.ylim(
            np.quantile(metrics, 0.005) + 0.01 * np.abs(np.quantile(metrics, 0.99)),
            np.quantile(metrics, 0.995) - 0.01 * np.abs(np.quantile(metrics, 0.01)),
        )
        plt.show()


class LSTM_ED_Generator(tf.keras.utils.Sequence):
    def __init__(self, X, y, metamodel, batch_min=64, shuffle=True, train=True):
        self.X_ctx = X[0]
        self.X_seq = X[1]
        self.y = y
        self.lenX = self.X_ctx.shape[0]
        self.train = train
        self.seed = 0
        self.shuffle = shuffle
        self.redundancy = metamodel.training_parameters["redundancy"]
        self.batch_min = batch_min

        # self.scaler = metamodel.scaler
        self.factory = metamodel.factory
        self._format = metamodel._format
        self.rescale = metamodel.rescale
        self.causality_remove = None
        self.model_parameters = metamodel.model_parameters
        self.model_specifications = metamodel.model_specifications
        self.size_subseq_enc = metamodel.model_parameters["size_subseq_enc"]
        self.size_window_past = metamodel.model_parameters["size_window_past"]
        self.size_window_futur = metamodel.model_parameters["size_window_futur"]

        if not (self.train):
            set_off = 0
            div = self.lenX // self.batch_min
            mod = (self.lenX % self.batch_min) / self.batch_min
        else:
            set_off = self.size_window_futur
            div = (self.lenX - set_off) // self.batch_min
            mod = 0
        self.len_ = div + int(np.ceil(mod))

        if shuffle:
            self.indices = np.arange(self.len_)
            np.random.shuffle(self.indices)

    def load(self, idx):
        idx = idx * self.batch_min
        seuil_min = max(0, idx - (self.size_subseq_enc + self.size_window_past))

        size_full_seq = (
            self.size_subseq_enc
            + self.size_window_past
            + self.size_window_futur
            + self.batch_min
        )

        if not (self.train):
            seuil_max = max(
                self.size_subseq_enc
                + self.size_window_past
                + self.batch_min
                - seuil_min,
                idx + self.batch_min + self.size_window_futur,
            )

            seuil_max = min(self.lenX, seuil_max)

            if (seuil_max - seuil_min) < size_full_seq:
                diff = size_full_seq - (seuil_max - seuil_min)
                seuil_min = max(0, seuil_min - diff)
        else:
            seuil_max = max(
                size_full_seq + seuil_min, idx + self.batch_min + self.size_window_futur
            )

        return (
            [self.X_ctx[seuil_min:seuil_max], self.X_seq[seuil_min:seuil_max]],
            self.y[seuil_min:seuil_max],
        )

    def __len__(self):
        return self.len_

    def __getitem__(self, idx):
        if self.shuffle:
            idx = self.indices[idx]

        x, y = self.load(idx)

        redundancy = 1
        if self.train:
            redundancy = None
        Inputs, Ouputs, _ = self.factory(
            x, y, causality_remove=None, fit_rescale=False, redundancy=redundancy
        )

        if not (self.train):
            set_off = self.size_window_futur
        else:
            set_off = self.size_window_futur

        seq_len = len(Inputs[0])
        selection = np.zeros(len(Inputs[0])) == 1

        if idx * self.batch_min < (self.size_subseq_enc + self.size_window_past):
            selection[idx * self.batch_min : idx * self.batch_min + self.batch_min] = (
                True
            )

        else:
            corr = 0
            if (idx + 1) * self.batch_min + set_off > self.lenX:
                corr = ((idx + 1) * self.batch_min + set_off) - self.lenX
            selection[
                seq_len - self.batch_min - set_off + corr : seq_len - set_off + corr
            ] = True

        Inputs = apply_mask(Inputs, selection)
        Ouputs = apply_mask(Ouputs, selection)
        return Inputs, Ouputs

    # shuffles the dataset at the end of each epoch
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


def double_roll(
    y,
    size_window_past,
    size_window_futur,
    size_subseq_dec,
    padding_past,
    padding_futur,
    only_futur=False,
):
    n_step_Zdec = get_fold_nstep(size_window_futur, size_subseq_dec, padding_futur)

    n_step_Zenc = get_fold_nstep(size_window_past, n_step_Zdec, padding_past) + 1

    if (padding_futur == size_subseq_dec) & (padding_past == size_window_futur):
        y_n = y.reshape((y.shape[0], -1, size_subseq_dec, y.shape[-1]))
        y_n = y_n.reshape(
            (y.shape[0], n_step_Zenc, n_step_Zdec, size_subseq_dec, y.shape[-1])
        )
        if only_futur:
            y_n = y_n[:, -1]

    else:
        if only_futur:
            y_n = np.zeros((y.shape[0], n_step_Zdec, size_subseq_dec, y.shape[-1]))

            for j in range(n_step_Zdec):
                begin = (n_step_Zenc - 1) * padding_past + j * padding_futur
                end = begin + size_subseq_dec
                y_n[:, j, :, :] = y[:, begin:end]

        else:
            y_n = np.zeros(
                (y.shape[0], n_step_Zenc, n_step_Zdec, size_subseq_dec, y.shape[-1])
            )
            for i in range(n_step_Zenc):
                for j in range(n_step_Zdec):
                    begin = i * padding_past + j * padding_futur
                    end = begin + size_subseq_dec
                    y_n[:, i, j, :, :] = y[:, begin:end]
    return y_n
