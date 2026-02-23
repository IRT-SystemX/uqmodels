import numpy as np
import pandas as pd
import inspect
from sklearn.ensemble import RandomForestRegressor


def check_transform_input_to_panda(input, name=""):
    """Check if input is dataframe.
        if it's a np.ndarray turn it to dataframe
        else raise error.

    Args:
        input (_type_): input to check or tranforaam
        name (str, optional): name of input

    Raises:
        TypeError: Input have a wrong type

    Returns:
        input: pd.dataframe
    """
    if isinstance(input, np.ndarray):
        # print('Change ndarray in pd.dataframe')
        return pd.DataFrame(input)

    if not isinstance(input, pd.DataFrame):
        print("Type issues in " + inspect.stack()[1].function)
        print(name + "should be pd.DataFrame rather than " + type(input))
        raise TypeError
    return input


def select_features_from_FI(X, y, model="RF", threesold=0.01, **kwargs):
    if model == "RF":
        estimator = RandomForestRegressor(
            ccp_alpha=1e-05,
            max_depth=15,
            max_features=0.5,
            max_samples=0.5,
            min_impurity_decrease=0.0001,
            min_samples_leaf=2,
            min_samples_split=8,
            n_estimators=100,
            random_state=0,
        )

        estimator.fit(X, y)
        features_mask = estimator.feature_importances_ > threesold
    return features_mask


def select_data_and_context(
    data, context=None, ind_data=None, ind_context=None, **kwargs
):
    """Select data and context using ind_data & ind_context.

    Args:
        data (ndarray): data
        context (ndarray, optional): context_data. Defaults to None.
        n_components (int, optional): n_components of pca. Defaults to 3.
        ind_data (ind_array, optional): selected data. Defaults to None : all dim are pick
        ind_context (ind_array, optional): seletected data context.
        Defaults to None : all dim are pick if there is context
    Returns:
        data_selected : Ndarray that contains np.concatenation of all selected features
    """

    data = check_transform_input_to_panda(data, "data")

    if context is not None:
        context = check_transform_input_to_panda(context, "data")

    data_selected = []
    if ind_data is None:
        pass
    else:
        data_selected.append(data.loc[:, ind_data])

    if ind_context is None:
        pass
    else:
        data_selected.append(context.loc[:, ind_context])

    if len(data_selected) == 0:
        data_selected = data
    else:
        data_selected = pd.concat(data_selected, axis=1)
    return data_selected.values


def build_window_representation(y, step=1, window=10):
    if step > 1:
        mask = np.arange(len(y)) % step == step - 1
    else:
        mask = np.ones(len(y)) == 1

    list_df = []
    for i in range(window):
        dict_df_ts = {
            "id": np.arange(len(y[mask])),
            "time": np.roll(np.arange(len(y[mask])), i),
        }
        for k in range(y.shape[1]):
            dict_df_ts["value_" + str(k + 1)] = np.roll(y[:, k], i, axis=0)[mask]
        df_ts = pd.DataFrame(dict_df_ts)
        list_df.append(df_ts)
    df_ts = pd.concat(list_df)
    y_target = y[mask]
    if len(y) == window:
        df_ts = df_ts[df_ts["id"] == len(y) - 1]
        y_target = y_target[-1:]
    return (df_ts, y_target)
