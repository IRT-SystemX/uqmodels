import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

import uqmodels.utils as ut
from uqmodels.postprocessing.UQ_processing import (
    compute_Epistemic_score,
    fit_Epistemic_score,
    get_extremum_var_TOT_and_ndUQ_ratio,
    process_UQmeasure_to_TOT_and_E_sigma,
)


def feature_augment(X, x_min, y_min):
    angle = np.angle(X[:, 0] + X[:, 1] * 1j)[:, None]
    print(angle.min())
    norm = np.abs((X[:, 0] + X[:, 1] * 1j) * 0.001)[:, None]
    sqrtnorm = np.sqrt(np.abs((X[:, 0] + X[:, 1] * 1j) * 0.001)[:, None])
    pwdangle = np.power(angle, 2)
    sqrtangle = np.sqrt(angle + np.pi)
    expX = np.exp(X[:, 0])[:, None]
    expY = np.exp(X[:, 1])[:, None]
    logX = np.log(X[:, 0] - x_min + 1)[:, None]
    logY = np.log(X[:, 1] - y_min + 1)[:, None]
    if False:
        X = np.concatenate(
            [X, angle, pwdangle, sqrtangle, norm, sqrtnorm, expX, expY, logX, logY],
            axis=1,
        )
    else:
        X = np.concatenate([X, expX, expY, logX, logY], axis=1)

    return X


def var_vec_matrix(i, mat, y, s=1):
    dist = (mat - mat[i]) ** 2
    dist = np.sum(dist, axis=1)
    dist = np.sqrt(dist)
    return y[dist < s].std()


def gen_target(X):
    X = X
    a = (1 - np.cos(0.5 + X[:, 0] * 2 * np.pi)) + 1.5 * np.sin(
        -0.1 + X[:, 0] + X[:, 1] * 2 * np.pi
    )
    a = a + 2 * ((1 - np.power(X[:, 0], 2)) + (1 - np.power(X[:, 1], 2)))
    return a


def gen_UQ(X, noise_x):
    X = X
    # a =  2 *((1-np.power(0.5-X[:,0],2))+(1-np.power(X[:,1],2)))
    # b = 2*((1-np.power(0.7+X[:,0],2))+(1-np.power(0.7+X[:,1],2)))
    # a = np.power((a-a.min())/(a.max()-a.min()),400) + np.power((b-b.min())/(b.max()-b.min()),400)
    a = np.cos(0.3 + X[:, 0] * 2 * np.pi) + np.sin(X[:, 1] * 2 * np.pi)
    a = np.power((a - a.min()) / (a.max() - a.min()), 4)
    a = (a / a.max() + 0.05) * noise_x
    return a


def apply_UQ(X, noise_x):
    UQ_ = gen_UQ(X, noise_x)
    sigma = np.tile(UQ_[:, None], 2)
    X = X + np.random.normal(sigma * 0, sigma)
    return X


def core_gen(
    n_samples=6000,
    n_mid=3000,
    n_mid_mid=200,
    shuffle=True,
    noise_x=0.08,
    noise_outliers=0.14,
    keep_val=0.01,
    noise_target=0.05,
    random_state=0,
):
    outer_circ_x = -0.5 + np.cos(np.linspace(0, np.pi + (np.pi / 2.2), n_mid))[::-1]
    outer_line_x = np.linspace(0, 0.5, n_mid_mid)[::-1]
    outer_circ_y = np.sin(np.linspace(0 - (np.pi / 2.2), np.pi, n_mid))

    inner_circ_x = 1 - np.cos(np.linspace(0, np.pi + np.pi / 2.2, n_mid))
    inner_line_x = np.linspace(0, -0, n_mid_mid)[::-1]
    inner_circ_y = -np.sin(np.linspace(0, np.pi + np.pi / 2.2, n_mid))

    X = np.vstack(
        [
            np.concatenate([outer_circ_x, outer_line_x, inner_circ_x]) - 0.25,
            np.concatenate([outer_circ_y, inner_line_x, inner_circ_y]),
        ]
    ).T

    X += np.random.normal(X * 0, scale=noise_outliers / 10, size=X.shape)

    sigma = np.tile(gen_UQ(X, noise_x)[:, None], 2)
    # Add outliers
    ind_big_alea = np.random.choice(np.arange(len(X)), 300)
    X[ind_big_alea] += np.random.normal(
        X[ind_big_alea] * 0,
        scale=sigma[ind_big_alea] + noise_outliers,
        size=X[ind_big_alea].shape,
    )

    # Apply noise on target
    target = gen_target(apply_UQ(X, noise_x))
    # Add noise on input independtly of target
    X = apply_UQ(X, noise_x)
    target = ut.cut(target, 0.01, 0.99)
    target_min = target.min()
    target_max = target.max()
    target = (target - target_min) / (target_max - target_min)

    keep = (np.random.rand(len(X)) < keep_val) | (
        (np.arange(len(X)) < 2000) | (np.arange(len(X)) > 2700)
    )

    # Basé sur l'ordre initial sans prise en compte de la perturbation features
    # pseudo_var = np.array([y[max(0,i-100):i+100].var() for i in np.arange(len(X))])
    # Basé sur le voisnage après perturbation : prise en compte de la var X et Y

    pseudo_var = np.array(
        [var_vec_matrix(i, X[:, :2], target, s=0.05) for i in np.arange(len(X))]
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

    shape = (200, 200)
    x1 = np.linspace(X[:, 0].min() - 0.3, X[:, 0].max() + 0.3, shape[0])
    x2 = np.linspace(X[:, 1].min() - 0.3, X[:, 1].max() + 0.3, shape[1])
    # full coordinate arrays
    xx, yy = np.meshgrid(x1, x2)
    grid_X = np.concatenate([xx.reshape(-1)[:, None], yy.reshape(-1)[:, None]], axis=1)

    grid_X = feature_augment(grid_X, X[:, 0].min() - 0.5, X[:, 1].min() - 0.5)

    grid_sigma = gen_UQ(grid_X, noise_x)

    grid_target = gen_target(grid_X)
    grid_target = (grid_target - target_min) / (target_max - target_min)
    grid_target = ut.threshold(grid_target, min_val=0, max_val=1)

    print(target.min(), target.max())
    print(grid_target.min(), grid_target.max())

    x1 = grid_X[:, 0].reshape(shape)
    x2 = grid_X[:, 1].reshape(shape)
    plt.figure(figsize=(15, 15))
    plt.subplot(2, 2, 1)
    plt.title("Prediction")
    # h = plt.contourf(
    #     x1,
    #     x2,
    #     grid_target.reshape(shape),
    #     levels=np.arange(0, 1.05, 0.05),
    #     vmin=0,
    #     vmax=1,
    #     alpha=0.2,
    #     cmap=plt.get_cmap("jet"),
    # )
    # cbar = plt.colorbar(h, orientation="horizontal", ticks=[0, 1], label="target")
    plt.scatter(
        X[:, 0], X[:, 1], c=target, vmin=0, vmax=1, s=1, cmap=plt.get_cmap("jet")
    )
    plt.subplot(2, 2, 2)
    plt.title("UQ")
    # h = plt.contourf(
    #     x1,
    #     x2,
    #     grid_sigma.reshape(shape),
    #     levels=20,
    #     alpha=0.2,
    #     cmap=plt.get_cmap("jet"),
    # )
    # cbar = plt.colorbar(h, orientation="horizontal", ticks=[0, 1], label="target")
    plt.scatter(X[:, 0], X[:, 1], c=pseudo_var, s=1, cmap=plt.get_cmap("jet"))

    features_scaler = StandardScaler()
    X = features_scaler.fit_transform(X)
    grid_X = features_scaler.transform(grid_X)

    dict_data = {}
    dict_data["X"] = X
    dict_data["Y"] = target
    dict_data["context"] = None
    dict_data["train"] = train
    dict_data["test"] = test
    dict_data["X_split"] = train
    dict_data["aux"] = {
        "grid_X": grid_X,
        "grid_Y": grid_target,
        "pseudo_var": pseudo_var,
        "grid_sigma": grid_sigma,
        "keep": keep,
    }

    return dict_data


def generate_default(dict_params=dict()):
    dict_data = core_gen(**dict_params)
    return dict_data


def compute_val(pred, UQ, UQ_grid, train):
    params_ = fit_Epistemic_score(
        UQ,
        "var_A&E",
        pred=None,
        y=None,
        type_UQ_params=None,
        list_percent=[0.5, 0.8, 0.95, 0.98, 0.995, 1],
        var_min=0,
        var_max=None,
        min_cut=0.1,
        max_cut=0.97,
        q_var=1,
        q_Eratio=3,
        mode="score",
        reduc_filter=None,
    )

    extremum_var_TOT, ndUQ_ratio = get_extremum_var_TOT_and_ndUQ_ratio(
        UQ,
        type_UQ="var_A&E",
        pred=None,
        y=None,
        type_UQ_params=None,
        min_cut=0,
        max_cut=0.97,
        var_min=0,
        var_max=None,
        factor=1,
        q_var=1,
        q_Eratio=0.5,
        E_cut_in_var_nominal=True,
    )

    A, E = process_UQmeasure_to_TOT_and_E_sigma(
        UQ,
        "var_A&E",
        pred=None,
        y=None,
        type_UQ_params=None,
        var_min=0,
        var_max=None,
        min_cut=0,
        max_cut=1,
        q_var=1,
        q_var_e=1,
        k_var_e=1,
        ndUQ_ratio=ndUQ_ratio,
        extremum_var_TOT=extremum_var_TOT,
        reduc_filter=None,
        roll=0,
    )
    print(extremum_var_TOT)

    A, E = np.power(A, 2), np.power(E, 2)

    Amin = np.quantile(A, 0.01)
    Amax = np.quantile(A, 0.99)
    Emin = np.quantile(E, 0.01)
    Emax = np.quantile(E, 0.99)

    UQ = (A, E)
    dE, params_ = compute_Epistemic_score(
        UQ,
        "var_A&E",
        pred=None,
        y=None,
        type_UQ_params=None,
        list_percent=[0.5, 0.8, 0.95, 0.98, 0.995, 1],
        var_min=0,
        var_max=None,
        min_cut=0.1,
        max_cut=0.97,
        q_var=1,
        q_Eratio=1.01,
        mode="score",
        reduc_filter=None,
        params_=params_,
    )
    dE = np.log(dE)

    dEmin = np.quantile(dE, 0.01)
    dEmax = np.quantile(dE, 0.99)

    grid_A, grid_E = process_UQmeasure_to_TOT_and_E_sigma(
        UQ_grid,
        "var_A&E",
        pred=None,
        y=None,
        type_UQ_params=None,
        var_min=0,
        var_max=None,
        min_cut=0,
        max_cut=1,
        q_var=1,
        q_var_e=1,
        k_var_e=1,
        ndUQ_ratio=ndUQ_ratio,
        extremum_var_TOT=extremum_var_TOT,
        reduc_filter=None,
        roll=0,
    )

    grid_A, grid_E = np.power(grid_A, 2), np.power(grid_E, 2)

    UQ_grid = (grid_A, grid_E)
    grid_dE, params_ = compute_Epistemic_score(
        UQ_grid,
        "var_A&E",
        pred=None,
        y=None,
        type_UQ_params=None,
        list_percent=[0.5, 0.8, 0.95, 0.98, 0.995, 1],
        var_min=0,
        var_max=None,
        min_cut=0.1,
        max_cut=0.97,
        q_var=1,
        q_Eratio=1.01,
        mode="score",
        reduc_filter=None,
        params_=params_,
    )

    grid_dE = np.log(grid_dE)

    E[E > Emax] = Emax
    E[E < Emin] = Emin
    A[A > Amax] = Amax
    A[A < Amin] = Amin
    dE[dE > dEmax] = dEmax
    dE[dE < dEmin] = dEmin

    grid_E[grid_E > Emax] = Emax
    grid_E[grid_E < Emin] = Emin
    grid_A[grid_A > Amax] = Amax
    grid_A[grid_A < Amin] = Amin
    grid_dE[grid_dE > dEmax] = dEmax
    grid_dE[grid_dE < dEmin] = dEmin

    A = np.log(A)
    Amin, Amax = np.log(Amin), np.log(Amax)
    grid_A = np.log(grid_A)

    return (A, grid_A, Amin, Amax, E, grid_E, Emin, Emax, dE, grid_dE, dEmin, dEmax)


def two_dimensional_plot(
    y,
    pred,
    pred_grid,
    X,
    grid_X,
    A,
    Amin,
    Amax,
    dE,
    dEmin,
    dEmax,
    keep,
    grid_A,
    pseudo_var,
    var_min,
    var_max,
    grid_E,
    Emin,
    Emax,
    grid_dE,
    test,
    color_lvl=10,
):
    mask = keep

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 3, 1)
    plt.title("Cible à prédire (couleur)")
    h = plt.scatter(
        X[test, 0], X[test, 1], c=y[test], vmin=0, vmax=1, s=5, cmap=plt.get_cmap("jet")
    )
    cbar = plt.colorbar(h, orientation="horizontal", ticks=[0, 1], label="target")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.subplot(2, 3, 2)
    plt.title("Prediction")
    h = plt.scatter(
        X[test, 0],
        X[test, 1],
        c=pred[test],
        vmin=0,
        vmax=1,
        s=5,
        cmap=plt.get_cmap("jet"),
    )
    cbar = plt.colorbar(h, orientation="horizontal", ticks=[0, 1], label="pred_value")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.subplot(2, 3, 3)
    plt.title("Erreur modèle")
    res = y[test] - pred[test]
    h = plt.scatter(
        X[test, 0],
        X[test, 1],
        c=res,
        vmin=-1,
        vmax=1,
        cmap=plt.get_cmap("Spectral_r"),
        s=5,
    )
    cbar = plt.colorbar(
        h, orientation="horizontal", ticks=[res.min(), 0, res.max()], label="residu"
    )
    plt.xlabel("X1")
    plt.ylabel("X2")

    plt.subplot(2, 3, 4)
    plt.title("train_data")
    h = plt.scatter(X[:, 0], X[:, 1], c=keep, s=1, cmap=plt.get_cmap("jet"))
    cbar = plt.colorbar(h, orientation="horizontal", ticks=[0, 1], label="train data")
    cbar.ax.set_xticklabels(["Seen", "Unseen"])
    plt.xlabel("X1")
    plt.ylabel("X2")

    plt.subplot(2, 3, 5)
    plt.title("Aleatoric")
    h = plt.scatter(
        X[test, 0],
        X[test, 1],
        c=A[test],
        cmap=plt.get_cmap("plasma"),
        vmin=Amin,
        vmax=Amax,
        s=5,
    )
    cbar = plt.colorbar(
        h, orientation="horizontal", ticks=[Amin, Amax], label="Var Aleatoric"
    )
    cbar.ax.set_xticklabels(["Low", "High"])
    plt.xlabel("X1")
    plt.ylabel("X2")

    plt.subplot(2, 3, 6)
    plt.title("Epipstemic")

    h = plt.scatter(
        X[test, 0],
        X[test, 1],
        c=dE[test],
        cmap=plt.get_cmap("Spectral_r"),
        vmin=dEmin,
        vmax=dEmax,
        s=1,
    )
    cbar = plt.colorbar(h, orientation="horizontal", ticks=[dEmin, dEmax], label="Conf")
    cbar.ax.set_xticklabels(["Low risk (meduim of train)", "High risk"])
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.tight_layout()
    plt.show()

    shape = (200, 200)
    x1 = grid_X[:, 0].reshape(shape)
    x2 = grid_X[:, 1].reshape(shape)

    plt.figure(figsize=(15, 15))
    plt.subplot(2, 2, 1)
    plt.title("Prediction")
    h = plt.contourf(
        x1,
        x2,
        pred_grid.reshape(shape),
        levels=color_lvl,
        vmin=0,
        vmax=1,
        alpha=0.2,
        cmap=plt.get_cmap("jet"),
    )
    cbar = plt.colorbar(h, orientation="horizontal", ticks=[0, 1], label="target")
    plt.scatter(
        X[mask, 0], X[mask, 1], c=y[mask], vmin=0, vmax=1, s=1, cmap=plt.get_cmap("jet")
    )

    plt.subplot(2, 2, 2)
    plt.title("Incertitude Aleatoric")
    h = plt.contourf(
        x1,
        x2,
        grid_A.reshape(shape),
        levels=color_lvl,
        alpha=0.2,
        vmin=Amin,
        vmax=Amax,
        cmap=plt.get_cmap("plasma", 10),
    )
    plt.scatter(
        X[mask, 0],
        X[mask, 1],
        c=pseudo_var[mask],
        vmin=var_min,
        vmax=var_max,
        s=1,
        cmap=plt.get_cmap("plasma"),
    )
    cbar = plt.colorbar(
        h, orientation="horizontal", ticks=[Amin, Amax], label="Var Aleatoric"
    )

    plt.subplot(2, 2, 3)
    plt.title("Confiance Epistemic")
    h = plt.contourf(
        x1,
        x2,
        grid_E.reshape(shape),
        levels=color_lvl,
        alpha=0.2,
        vmin=Emin,
        vmax=Emax,
        cmap=plt.get_cmap("Spectral_r"),
    )
    cbar = plt.colorbar(h, orientation="horizontal", ticks=[Emin, Emax], label="Conf")
    cbar.ax.set_xticklabels(["Low risk", "High risk"])
    h = plt.scatter(
        X[mask, 0],
        X[mask, 1],
        c=np.invert(keep)[mask],
        s=1,
        cmap=plt.get_cmap("jet"),
        alpha=0.4,
    )

    plt.subplot(2, 2, 4)
    plt.title("Confiance D-Epistemic")
    h = plt.contourf(
        x1,
        x2,
        grid_dE.reshape(shape),
        levels=color_lvl,
        alpha=0.2,
        vmin=dEmin,
        vmax=dEmax,
        cmap=plt.get_cmap("Spectral_r"),
    )
    cbar = plt.colorbar(h, orientation="horizontal", ticks=[dEmin, dEmax], label="Conf")
    cbar.ax.set_xticklabels(["Low risk", "High risk"])
    h = plt.scatter(
        X[mask, 0],
        X[mask, 1],
        c=np.invert(keep)[mask],
        s=1,
        cmap=plt.get_cmap("jet"),
        alpha=0.4,
    )
    plt.tight_layout()
    plt.show()
