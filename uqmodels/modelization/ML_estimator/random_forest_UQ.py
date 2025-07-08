import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble._forest import _generate_sample_indices, _get_n_samples_bootstrap

from uqmodels.modelization.UQEstimator import UQEstimator
from uqmodels.utils import EPSILON


class RF_UQEstimator(UQEstimator):
    """Uncertainty quantification approch based on "local" sub-sampling UQ estimation from Random forest
    neighboorhood extraction"""

    def __init__(
        self,
        estimator=RandomForestRegressor(),
        pretuned=False,
        type_UQ="var",
        use_biais=True,
        rescale=True,
        n_jobs=4,
        beta=0.1,
        var_min=0.00001,
        random_state=None,
    ):
        """Initialization

        Args:
            estimator (_type_, optional): RandomForestRegressor with meta-parameters
            pretuned (bool, optional): bool flag that freeze estimator. Defaults to False.
            type_UQ (str, optional): nature of UQmeasure. Defaults to 'var'.
            use_biais (bool, optional): use oob biais correction. Defaults to True.
            rescale (bool, optional): use rescale procedure. Defaults to True.
            n_jobs (int, optional): number of jobs used for parallelization purpose. Defaults to 4.
            beta (float, optional): miss coverage targets in case of type_UQ = quantile
            var_min (float, optional): minimal variance. Defaults to 0.00001.
            random_state (bool): handle experimental random using seed.
        """

        super().__init__(
            name="RF_UQEstimator",
            type_UQ=type_UQ,
            rescale=rescale,
            var_min=var_min,
            random_state=random_state,
        )

        self.beta = beta
        if type_UQ in ["quantile", "res_quantile"]:
            self.list_alpha = [beta / 2, (1 - beta / 2)]
            self.type_UQ_params = {"list_alpha": self.list_alpha}
        self.use_biais = use_biais
        self.list_statistics = [
            "pred",
            "n_obs",
            "aleatoric",
            "epistemic",
            "oob_aleatoric",
        ]
        if self.use_biais:
            self.list_statistics.append("biais")

        self.pretuned = pretuned
        self.estimator = estimator
        if self.random_state is not None:
            self.estimator.set_params(random_state=self.random_state)
        self.dict_leaves_statistics = dict()
        self.n_jobs = n_jobs
        if self.type_UQ in ["var", "res_var"]:
            self.list_statistics.append("var")

        elif self.type_UQ in ["2var", "res_2var"]:
            self.list_statistics.append("var_bot")
            self.list_statistics.append("var_top")

        elif self.type_UQ in ["quantile", "res_quantile"]:
            self.list_statistics.append("Q_bot")
            self.list_statistics.append("Q_top")

        self.Y_shape = None

    def _format(self, X, y, type_transform, mode_UQ=False, skip_format=False, **kwargs):
        """data & output standarization : see UQEstimator._format
        Args:
            X (np.array or None): Features
            y (np.array or None): Targets/observations or prediction or UQmeasure
            type_transform (str): _description_
            mode_UQ (bool, optional): True if normalise UQ. Defaults to False.
            var_min (int, optional): _description_. Defaults to 0.

        Returns:
            _type_: _description_
        """
        X, y = super()._format(
            X,
            y,
            type_transform=type_transform,
            mode_UQ=mode_UQ,
            skip_format=skip_format,
        )
        return X, y

    def fit(self, X, y, sample_weight=None, skip_format=False, **kwargs):
        """Train scikit RF model and then estimate and store leafs variance to uncertainty quantification purpose
        Args:
            X (array): Features of the training set
            y (array): Targets/Observations of the training set
        """

        def aux_leaf_statistics(Y_leaf, pred_leaf, Y_oob_leaf, list_statistics, beta):
            """Auxiliar function : Extract statistics of a leaf:

            Args:
                Y_leaf ([float np.array (?,dim)]): Target of train leaf's elements
                pred_leaf ([float np.array (?,dim)]): Predict of train leaf's elements
                Y_oob_leaf ([float np.array (?,dim)]): Target of oob(out-of-bound) leaf's elements
                list_statistics ([list]) List of statistics to compute.
                beta ([float]): miss-coverage target (used for quantile estimation)

            Ouput:
                dict_statistics : dict of extracted statistics statistics.
            """

            dict_statistics = dict()

            n_obs = len(Y_leaf)
            len(Y_oob_leaf)
            if "n_obs" in list_statistics:
                dict_statistics["n_obs"] = n_obs

            # Compute biais : Error on out of "bag" sample (Part of train sample leave beside for the Tree)
            biais = 0
            if "biais" in list_statistics:  # Moyenne du biais
                if len(Y_oob_leaf) > 4:
                    biais = (Y_oob_leaf - Y_leaf.mean(axis=0)).mean(axis=0)
                dict_statistics["biais"] = biais

            if "oob_aleatoric" in list_statistics:  # Variance du biais
                dict_statistics["oob_aleatoric"] = 0
                if len(Y_oob_leaf) > 1:
                    dict_statistics["oob_aleatoric"] = (Y_oob_leaf).var(axis=0, ddof=1)

            # Compute : the whole bias : Mean of leafs erros on both Used train sample and Non-Used train sample.
            # If val_pred is the tree forecast values (Y_train_leaf - val_pred) = 0 so biais = biais

            # Debias forecast values and compute residuals in order to perform variance estimation
            Residual = Y_leaf - Y_leaf.mean(axis=0)

            if "pred" in list_statistics:
                dict_statistics["pred"] = Y_leaf.mean(axis=0)

            # Estimation of leaf residuals variance.
            if "var" in list_statistics:
                dict_statistics["var"] = (np.power(Y_leaf, 2) / (n_obs)).sum(axis=0)

            # Estimation of exploratory statistics
            if "aleatoric" in list_statistics:
                dict_statistics["aleatoric"] = Y_leaf.var(
                    axis=0, ddof=1
                )  # partial E[Var[X]]

            # partial E[X] No biais because reducing with mean that doesn't take
            # account biais (native RF predict function)
            if "epistemic" in list_statistics:
                dict_statistics["epistemic"] = np.power(Y_leaf.mean(axis=0), 2)

            # Identify negative and positive residuals (for '2var' or 'quantile' estimation)
            Residual = Y_leaf - (Y_leaf.mean(axis=0) - biais)
            flag_res_bot = Residual <= 0
            flag_res_top = Residual >= 0

            # Estimation of positive and negative leaf residuals variance.
            if "var_bot" in list_statistics:
                # Identify negative and positive residuals
                flag_res_bot = Residual <= 0
                flag_res_top = Residual >= 0
                dict_statistics["var_bot"] = EPSILON
                dict_statistics["var_top"] = EPSILON
                if (flag_res_bot).sum() > 2:
                    dict_statistics["var_bot"] = Residual[flag_res_bot].var(
                        axis=0, ddof=1
                    )

                if (flag_res_top).sum() > 2:
                    dict_statistics["var_top"] = Residual[flag_res_top].var(
                        axis=0, ddof=1
                    )

            if "Q_bot" in list_statistics:
                # Identify negative and positive residuals
                flag_res_bot = Residual <= 0
                flag_res_top = Residual >= 0
                dict_statistics["Q_bot"] = EPSILON
                dict_statistics["Q_top"] = EPSILON
                if (flag_res_bot).sum() > 2:
                    dict_statistics["Q_bot"] = np.quantile(
                        Residual[flag_res_bot],
                        beta * (1 + 1 / flag_res_bot.sum()),
                        axis=0,
                    )
                if (flag_res_top).sum() > 2:
                    dict_statistics["Q_top"] = np.quantile(
                        Residual[flag_res_top],
                        np.minimum((1 - beta) * (1 + 1 / flag_res_top.sum()), 0.995),
                        axis=0,
                    )

            return dict_statistics

        def aux_tree_statistics(
            num_tree,
            n_samples,
            max_samples,
            y,
            pred,
            tree_affectation,
            list_statistics,
            beta,
            simple_m,
            bootstrap,
        ):
            """Extraction of statistics for each leaves of a tree

            Args:
                num_tree ([int]): ID of the tree
                n_samples ([float]): Random forest paramater (used to reproduce the trainning set)
                max_samples ([int]): Random forest parameter (used to reproduce the trainning set)
                y ([array]): Target of training set values
                pred ([array]): Forecast of training set values
                tree_affectation ([érray]): Array of element leaves affectations
                pred_true ([2D Array of float]): Forecast values !!! Non-used !!!
                list_statistics ([list]) List of statistics to compute.
                beta ([float]): miss-coverage target (used for quantile estimation)
                simple_m ([object]): scikit learn decision tree model

            Output:
                Side effect on dict_statistics
            """

            # Regenerate the bootstrat training sample thank to scikit function

            leaves = list(set(tree_affectation))
            if bootstrap:
                n_samples_bootstrap = _get_n_samples_bootstrap(n_samples, max_samples)
                re_sample = _generate_sample_indices(
                    simple_m.random_state, n_samples, n_samples_bootstrap
                )
                inv_draw = np.ones(n_samples)
                inv_draw[re_sample] = 0
                oob_sample = np.repeat(np.arange(inv_draw.size), inv_draw.astype(int))
            else:
                # Compute leaves affectation for the tree on bootstrap sample.

                # Identify non-used training data : oob data
                re_sample = np.arange(n_samples)
                inv_draw = np.ones(n_samples)
                inv_draw[re_sample] = 0
                oob_sample = np.repeat(np.arange(inv_draw.size), inv_draw.astype(int))
            # Compute leaves affectation for the oob sample.

            leaves_interest = []
            # add key : (num_tree,num_leaf) to save leaf values.

            # For each (visited) leaves :
            tree_statistics = dict()
            for num_leaf in leaves:
                # Identify concerned bootstrap and oob observations.
                Y_leaf = y[re_sample[tree_affectation[re_sample] == num_leaf]]
                pred_leaf = pred[re_sample[tree_affectation[re_sample] == num_leaf]]
                Y_oob_leaf = y[oob_sample[tree_affectation[oob_sample] == num_leaf]]

                # Extract leaf statistics
                tree_statistics[num_leaf] = aux_leaf_statistics(
                    Y_leaf, pred_leaf, Y_oob_leaf, list_statistics, beta
                )
                if (num_tree, num_leaf) in leaves_interest:
                    tree_statistics[num_leaf]["Y_leaf"] = Y_leaf
            return (num_tree, tree_statistics)

        beta = self.beta
        X, y = self._format(X, y, "fit_transform", skip_format=skip_format)
        # self.X_train = X
        # self.Y_train = y
        self.Y_shape = y.shape
        list_statistics = self.list_statistics
        model_rf = self.estimator
        # Fit the model using scikit method
        model_rf.fit(X, y, sample_weight=sample_weight)
        RF_affectation = model_rf.apply(X)
        n_estimators = int(model_rf.n_estimators)
        pred = model_rf.predict(X)

        # Extract subsample statistics
        parrallel_inputs = [
            (
                num_tree,
                len(y),
                model_rf.max_samples,
                y,
                pred,
                RF_affectation[:, num_tree],
                list_statistics,
                beta,
                model_rf.estimators_[num_tree],
                model_rf.bootstrap,
            )
            for num_tree in np.arange(n_estimators)
        ]

        Rf_leaves_statistics = Parallel(n_jobs=self.n_jobs)(
            delayed(aux_tree_statistics)(*inputs) for inputs in parrallel_inputs
        )
        # Store leaves statistics of each tree in a dict
        dict_leaves_statistics = dict()
        for num_tree, dict_tree_statistics in Rf_leaves_statistics:
            dict_leaves_statistics[num_tree] = dict_tree_statistics
        self.dict_leaves_statistics = dict_leaves_statistics
        self.is_fitted = True
        return

    def predict(self, X, beta=None, skip_format=False, **kwargs):
        """Predict both forecast and UQ estimations values
        Args:
            X ([type]): Features of the data to forecast
            beta ([type]): Miss-coverage target

        Returns:
        pred ([array]): Forecast values
        UQ ([type]): UQmeasure
        """

        if beta is None:
            beta = self.beta

        X, _ = self._format(X, None, "transform", skip_format=skip_format)

        # Call auxiliaire function that compute RF statistics from leaf subsampling.
        pred, biais, UQ, var_A, var_E, _ = self.RF_extraction(X)

        if self.use_biais:
            pred = pred - biais

        # Compute (top,bot) boundaries from (1 or 2)-var estimation

        _, UQ = self._format(None, UQ, "inverse_transform", mode_UQ=True)
        _, pred = self._format(None, pred, "inverse_transform")

        return (pred, UQ)

    def RF_extraction(self, X):
        """Random-forest subsampling statistics extraction "

        Args:
            X ([array]): Features of elements.

        Output:
            Statistics array of shape (n_obs,dim):
            'Pred' : Random forest forecast values
            'Biais' : RF Biais computed as the sum of Oob Tree biais

            'UQ' : Shape of UQ depends of the type_UQ !!!
                IF mode=var:
                    UQ = Var  RF variance computed as the sum of esiduals' variance of the leaves
                IF mode=2-var:
                    UQ = (Var_bot,Var_top), !!! 2-uple of 2D array !!!
                    Var_bot : bot variance stimation  (partial sum) as sum of negative residuals' variance of the leaves
                    Var_top : top variance stimation  (partial sum) as sum of positive residuals' variance of the leaves
                IF mode=quantile:
                    UQ = (Q_bot,Q_top), !!! 2-uple of 2D array !!!
                    'Q_bot' :  Bot quantile estimation (partial sum) as ponderation of leaves' bot quantile
                    'Q_top' :  Top quantile estimatuon (partial sum) as ponderation of leaves' top quantile

            Other advanced statistics.
            'Biais_oob' :  part of Biais
            'Var_E' : Part of total variance (Law of total variance)
            'E_Var' : Part of total variance (Law of total variance)
            'Var_oob' : Part of total variance (related to biais)
        """

        def aux_predict(
            shape,
            list_statistics,
            list_RF_affectation,
            list_dict_tree_statistics,
            prediction,
        ):
            """Aggregate statistics (partial sum) of serveral trees.
            Args:
                shape ([tupple]): shape of statistics
                list_statistics ([list]): list of statistcs to extract
                list_RF_affectation ([array]): array of elements affectations for serveral trees
                list_dict_tree_statistics ([dict]): pre-computed leaves statistics for the trees

            Returns:
                agg_statistics ([array]): Partials aggregated statistics (for serveral tree) of elmments to forecast
            """
            agg_statistics = np.zeros((shape))
            for num, tree_affectation in enumerate(list_RF_affectation):
                tree_statistic = list_dict_tree_statistics[num]
                tree_statistics = tree_predict(
                    shape, list_statistics, tree_affectation, tree_statistic
                )
                agg_statistics += np.array(tree_statistics)

            return agg_statistics

        def tree_predict(shape, list_statistics, tree_affectation, tree_statistic):
            """Compute extracted statistcs of a tree for the elements to forecast.

            Args:
                shape ([tupple]): shape of statistics
                list_statistics ([type]): [description]
                tree_affectation ([type]): array of elements affectations for the tree
                tree_statistic ([type]): pre-computed leaves statistics for the tree

            Returns:
                statistics ([array]): Partials statistics for (a tree) of elmments to forecast
            """
            leaves = list(set(tree_affectation))
            statistics = []
            for n, key in enumerate(list_statistics):
                statistics.append(np.zeros(shape[1::]))

            for num_leaf in leaves:
                mask = tree_affectation == num_leaf
                for n, key in enumerate(list_statistics):
                    statistics[n][mask] = tree_statistic[num_leaf][key]
            return statistics

        n_trees = self.estimator.n_estimators
        # Compute Leaves affectation array
        RF_affectation = self.estimator.apply(X)
        prediction = self.estimator.predict(X)
        list_statistics = self.list_statistics

        # Define shape of the statistic array.
        if len(self.Y_shape) == 1:
            shape = (len(list_statistics), len(X), 1)

        else:
            shape = (len(list_statistics), len(X), self.Y_shape[1])

        parallel_partition = np.array_split(range(n_trees), self.n_jobs * 2)

        # Split inputs of auxillar parralel tree statistics extraction
        parallel_input = []
        for partition in parallel_partition:
            parallel_input.append(
                (
                    shape,
                    list_statistics,
                    [RF_affectation[:, i] for i in partition],
                    [self.dict_leaves_statistics[i] for i in partition],
                    prediction,
                )
            )

        # Extract statistcs for each tree in RF in a parralel way :

        Predicted_statistics = Parallel(n_jobs=self.n_jobs)(
            delayed(aux_predict)(*inputs) for inputs in parallel_input
        )

        # Final aggregation and normalisation for each statistics
        Predicted_statistics = np.stack(Predicted_statistics).sum(axis=0)

        Pred = Predicted_statistics[list_statistics.index("pred")] / n_trees
        Biais = Pred * 0
        var_aleatoric, var_epistemic, var_aleatoric_oob = None, None, None
        if "biais" in list_statistics:
            Biais = Predicted_statistics[list_statistics.index("biais")] / n_trees

        if self.type_UQ in ["var", "res_var"]:
            var = (
                Predicted_statistics[list_statistics.index("var")] / (n_trees)
            ) - np.power(Pred, 2)
            UQ = var

        if self.type_UQ in ["res_2var", "2var"]:
            var_bot = Predicted_statistics[list_statistics.index("var_bot")] / n_trees

            var_top = Predicted_statistics[list_statistics.index("var_top")] / n_trees

            UQ = np.concatenate(
                [np.expand_dims(i, 0) for i in [var_bot, var_top]], axis=0
            )

        if self.type_UQ in ["res_quantile", "quantile"]:
            Q_bot = (
                Pred + Predicted_statistics[list_statistics.index("Q_bot")] / n_trees
            )
            Q_top = (
                Pred + Predicted_statistics[list_statistics.index("Q_top")] / n_trees
            )

            UQ = np.concatenate([np.expand_dims(i, 0) for i in [Q_bot, Q_top]], axis=0)

        if self.type_UQ == "var_A&E":
            if "aleatoric" in list_statistics:
                var_aleatoric = (
                    Predicted_statistics[list_statistics.index("aleatoric")] / n_trees
                )

            if "epistemic" in list_statistics:
                var_epistemic = Predicted_statistics[
                    list_statistics.index("epistemic")
                ] / (n_trees) - np.power(Pred, 2)

            if "oob_aleatoric" in list_statistics:
                var_aleatoric_oob = (
                    Predicted_statistics[list_statistics.index("oob_aleatoric")]
                    / n_trees
                )

            UQ = np.concatenate(
                [np.expand_dims(i, 0) for i in [var_aleatoric, var_epistemic]], axis=0
            )

        return (Pred, Biais, UQ, var_aleatoric, var_epistemic, var_aleatoric_oob)

    def _tuning(self, X, y, n_esti=100, folds=4, params=None, **kwarg):
        """Perform random search tuning using a given grid parameter"""
        if not (self.pretuned):
            if not isinstance(params, type(None)):
                X, y = self._format(X, y, "fit_transform")
                reg = RandomForestRegressor(random_state=0)
                score = "neg_mean_squared_error"
                self.estimator = super()._tuning(
                    reg, X, y, n_esti, folds, score, params
                )


def get_params_dict(
    estimator=None,
    pretuned=False,
    type_UQ="var_A&E",
    use_biais=True,
    rescale=True,
    n_jobs=4,
    beta=0.05,
    var_min=0.00001,
    n_estimators=125,
    max_depth=15,
    min_impurity_decrease=0.00001,
    ccp_alpha=1e-05,
    max_features=0.9,
    max_samples=0.7,
    random_state=None,
    min_samples_leaf=5,
    min_samples_split=5,
    **kwargs,
):
    """Provide a dict of paramaters to build an RF_UQEstimator
    Args:
        estimator (_type_, optional): RandomForestRegressor with meta-parameters
        pretuned (bool, optional): bool flag that freeze estimator. Defaults to False.
        type_UQ (str, optional): nature of UQmeasure. Defaults to 'var'.
        use_biais (bool, optional): use oob biais correction. Defaults to True.
        rescale (bool, optional): use rescale procedure. Defaults to True.
        n_jobs (int, optional): number of jobs used for parallelization purpose. Defaults to 4.
        beta (float, optional): miss coverage targets in case of type_UQ = quantile
        var_min (float, optional): minimal variance. Defaults to 0.00001.
    Returns:
        dict_parameters
    """

    if estimator is None:
        estimator = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            ccp_alpha=ccp_alpha,
            max_features=max_features,
            max_samples=max_samples,
            min_impurity_decrease=min_impurity_decrease,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            random_state=random_state,
            **kwargs,
        )

    # Specification of the UQestimator & Instanciation in a UQmodels wrapper that include post-processing
    dict_params = {
        "estimator": estimator,
        "pretuned": pretuned,
        "var_min": var_min,
        "n_jobs": n_jobs,
        "use_biais": use_biais,
        "type_UQ": type_UQ,
        "rescale": rescale,
        "beta": beta,
    }
    return dict_params
