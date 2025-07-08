import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import windows
from sklearn.preprocessing import StandardScaler

from uqmodels.utils import cut


def feature_augment(X, x_min, y_min):
    angle = np.angle(X[:, 0] + X[:, 1] * 1j)[:, None]
    norm = np.abs((X[:, 0] + X[:, 1] * 1j) * 0.001)[:, None]
    expX = np.exp(X[:, 0])[:, None]
    expY = np.exp(X[:, 1])[:, None]
    logX = np.log(X[:, 0] - x_min + 1)[:, None]
    logY = np.log(X[:, 1] - y_min + 1)[:, None]
    X = np.concatenate([X, angle, norm, expX, expY, logX, logY], axis=1)
    return X


def var_vec_matrix(i, mat, y, s=1):
    dist = (mat - mat[i]) ** 2
    dist = np.sum(dist, axis=1)
    dist = np.sqrt(dist)
    return y[dist < s].std()


def core_gen(
    n_samples=6000,
    n_mid=3000,
    n_mid_mid=1500,
    shuffle=True,
    noise_x=0.04,
    noise_target=0.15,
    random_state=0,
):
    outer_circ_x = np.cos(np.linspace(0, np.pi + (np.pi / 2.2), n_mid))[::-1]
    outer_line_x = np.linspace(0, 1, n_mid_mid)[::-1]
    outer_circ_y = np.sin(np.linspace(0 - (np.pi / 2.2), np.pi, n_mid))

    inner_circ_x = 1 - np.cos(np.linspace(0, np.pi + np.pi / 2.2, n_mid))
    inner_line_x = np.linspace(0.11, -0.11, n_mid_mid)[::-1]
    inner_circ_y = 1 - np.sin(np.linspace(0, np.pi + np.pi / 2.2, n_mid)) - 1

    X = np.vstack(
        [
            np.concatenate([outer_circ_x, outer_line_x, inner_circ_x]) - 0.5,
            np.concatenate([outer_circ_y, inner_line_x, inner_circ_y]),
        ]
    ).T

    target = np.arange(len(X)) / len(X)

    scale = np.concatenate(
        [
            (np.cos(np.linspace(0, np.pi * 10, n_mid * 2 + n_mid_mid)))[:, None],
            (np.cos(np.linspace(0, np.pi * 10, n_mid * 2 + n_mid_mid)))[:, None],
        ],
        axis=1,
    )
    scale_x = (-scale.min() + scale + 0.14) * noise_x
    X += np.random.normal(scale * 0, scale=scale_x, size=X.shape)
    # Add outliers
    ind_big_alea = np.random.choice(np.arange(len(X)), 200)
    X[ind_big_alea] += np.random.normal(
        scale[ind_big_alea] * 0, scale=0.18, size=X[ind_big_alea].shape
    )

    scale_target = (-scale.min() + scale + 0.03) * noise_target

    target = (
        np.cos((target * 17) - 0.8)
        + np.cos(target * 17 * 4) * 1 / 2
        + np.cos(target * 23) * 0.75
    )
    filter_ = windows.gaussian(100, std=2, sym=True)
    filter_ = filter_ / filter_.sum()
    print(target.shape)
    target = np.convolve(target, filter_, mode="same")
    print(target.shape)
    target += np.random.normal(
        scale[:, 0] * 0, scale=scale_target[:, 0], size=target.shape
    )
    target = cut((target - target.min()) / (target.max() - target.min()), 0.005, 0.995)
    target = (target - target.min()) / (target.max() - target.min())

    keep = ((np.arange(len(X)) < 600) | (np.arange(len(X)) > 800)) & (
        (np.arange(len(X)) < 5400) | (np.arange(len(X)) > 6000)
    )

    # Basé sur l'ordre initial sans prise en compte de la perturbation features
    # pseudo_var = np.array([y[max(0,i-100):i+100].var() for i in np.arange(len(X))])
    # Basé sur le voisnage après perturbation : prise en compte de la var X et Y

    pseudo_var = np.array(
        [var_vec_matrix(i, X[:, :2], target, s=0.1) for i in np.arange(len(X))]
    )

    var_max = np.quantile(pseudo_var, 0.98)
    var_min = np.quantile(pseudo_var, 0.2)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.scatter(
        X[:, 0], X[:, 1], c=target[:], vmin=0, vmax=1, s=3, cmap=plt.get_cmap("jet")
    )
    plt.subplot(1, 3, 2)
    plt.scatter(
        X[:, 0],
        X[:, 1],
        c=pseudo_var[:],
        vmin=var_min,
        vmax=var_max,
        s=3,
        cmap=plt.get_cmap("plasma"),
    )
    plt.subplot(1, 3, 3)
    plt.scatter(X[:, 0], X[:, 1], c=keep, s=3, cmap=plt.get_cmap("jet"))
    plt.show()

    X = feature_augment(X, X[:, 0].min() - 0.5, X[:, 1].min() - 0.5)
    plt.figure()

    plt.scatter(
        np.arange(len(target))[keep], target[keep], color=plt.get_cmap("jet", 2)(0)
    )
    plt.scatter(
        np.arange(len(target))[~keep], target[~keep], color=plt.get_cmap("jet", 2)(1)
    )
    plt.show()

    train = np.zeros(len(X))
    # Select ramdomly train point
    train[np.random.choice(np.arange(len(X)), int(len(X) / 2), replace=False)] = 1
    # Remove small part to create OOD subset
    train[np.invert(keep)] = 0
    train = train.astype(bool)
    test = np.invert(train)

    shape = (100, 100)
    x1 = np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, shape[0])
    x2 = np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, shape[1])
    # full coordinate arrays
    xx, yy = np.meshgrid(x1, x2)
    grid_sample = np.concatenate(
        [xx.reshape(-1)[:, None], yy.reshape(-1)[:, None]], axis=1
    )

    grid_sample = feature_augment(grid_sample, X[:, 0].min() - 0.5, X[:, 1].min() - 0.5)

    features_scaler = StandardScaler()
    X = features_scaler.fit_transform(X)
    grid_sample = features_scaler.transform(grid_sample)

    dict_data = {}
    dict_data["X"] = X
    dict_data["Y"] = target
    dict_data["context"] = None
    dict_data["train"] = train
    dict_data["test"] = test
    dict_data["X_split"] = train
    dict_data["aux"] = {
        "grid_sample": grid_sample,
        "pseudo_var": pseudo_var,
        "keep": keep,
    }

    return dict_data


def generate_default(dict_params=dict()):
    dict_data = core_gen(**dict_params)
    return dict_data
