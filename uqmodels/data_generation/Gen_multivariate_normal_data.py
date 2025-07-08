import math

import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.signal import convolve2d, savgol_filter

from uqmodels.processing import write


def base_cos_freq(array, freq):
    list_ = []
    for i in freq:
        list_.append(np.cos(i * math.pi * array))
        list_.append(np.sin(i * math.pi * array))
    return np.concatenate(list_).reshape(len(freq) * 2, -1).T


def trunc(x):
    """Round function for values visualisation"""
    return np.round(x, 5)


"""Return gaussian constant link to the percentile"""


# Motifs generation
def gen_motif(verbose=0, dim=(400, 20, 3), seed=0):
    dim_n, dim_t, dim_g = dim
    dim_n, dim_t, dim_g = dim
    int(dim_g * 2)
    Pattern_day_mean = []
    Pattern_day_var = []
    Day_Motif = np.ones(dim_t)
    n_melange = 12

    for n, i in enumerate([0.25, 0.5, 0.70, 0.90]):
        Day_Motif[n] = i
        Day_Motif[-n] = i

    for g in range(dim_g):
        dist = []
        for i in range(n_melange):
            rng0 = np.random.RandomState(seed + g * n_melange * 4 + i * 4 + 0)
            rng1 = np.random.RandomState(seed + g * n_melange * 4 + i * 4 + 1)
            rng2 = np.random.RandomState(seed + g * n_melange * 4 + i * 4 + 2)
            rng3 = np.random.RandomState(seed + g * n_melange * 4 + i * 4 + 3)

            dist.append(
                rng0.normal(
                    rng1.uniform(0, 1),
                    rng2.uniform(0.02, 0.3),
                    np.maximum(10000, rng3.normal(2000, 1000)).astype(int),
                )
            )
        a = np.roll(np.histogram(dist, bins=dim_t)[0], rng0.randint(dim_t))
        Pattern_day_mean.append(0.03 + Day_Motif * savgol_filter((a) / (a).sum(), 9, 4))
        dist = []
        for i in range(n_melange):
            rng0 = np.random.RandomState(
                seed + g * n_melange * 4 + i * 4 + 0 + dim_g * n_melange * 4
            )
            rng1 = np.random.RandomState(
                seed + g * n_melange * 4 + i * 4 + 1 + dim_g * n_melange * 4
            )
            rng2 = np.random.RandomState(
                seed + g * n_melange * 4 + i * 4 + 2 + dim_g * n_melange * 4
            )
            rng3 = np.random.RandomState(
                seed + g * n_melange * 4 + i * 4 + 3 + dim_g * n_melange * 4
            )

            dist.append(
                rng0.normal(
                    rng1.uniform(0, 1),
                    rng2.uniform(0.02, 0.3),
                    np.maximum(10000, rng3.normal(2000, 1000)).astype(int),
                )
            )
        a = np.roll(np.histogram(dist, bins=dim_t)[0], rng0.randint(dim_t))
        var_aux = np.maximum(
            np.zeros(dim_t) + 0.0001, savgol_filter(dim_t * ((a) / (a).sum()), 9, 4)
        )
        Pattern_day_var.append(var_aux)
    if verbose != 0:
        plt.figure(figsize=(14, 4))
        plt.subplot(2, 1, 1)
        [plt.plot(i) for i in Pattern_day_mean]
        plt.subplot(2, 1, 2)
        [plt.plot(i) for i in Pattern_day_var]
        plt.show()
    return (Pattern_day_mean, Pattern_day_var)


# Seasonal partern generation
def gen_seas_contexte(seas=140, pond=3, dim=(400, 20, 3), seed=0):
    rng = np.random.RandomState(int(seas + seed))
    dim_n, dim_t, dim_g = dim
    seas = (
        np.cos(2 * math.pi * (rng.rand() + np.arange(dim_t * dim_n) / seas))
    ) * pond  # 6 : Seas2 magnitude ponderation
    return seas


def gen_permuts_contexte(seas=140, pond=3, dim=(400, 20, 3), seed=0):
    rng = np.random.RandomState(seed)
    dim_n, dim_t, dim_g = dim
    seas = (
        np.cos(2 * math.pi * (rng.rand() + np.arange(dim_t * dim_n) / seas))
    ) * pond  # 6 : Seas2 magnitude ponderation
    perm = rng.permutation(np.arange(dim_n * dim_t))
    seas = seas[perm]
    return seas


# Pattern catégorielle aléatoire.
def gen_cat_contexte(size=5, effect=0.2, per=0.1, dim=(400, 20, 3), seed=0):
    rng = np.random.RandomState(seed)
    dim_n, dim_t, dim_g = dim
    n_event = int((per * (dim_n * dim_t)) / size)
    impact_ctx_mean = np.zeros((dim_t * dim_n, dim_g))
    for n, i in enumerate(rng.choice(np.arange(dim_n * dim_t - size), n_event, False)):
        impact_ctx_mean[i : (i + size), :] += effect
    return impact_ctx_mean


# Anomaly pattern genaration :
# Experiment have been essentially realised with point anomaly.
# Detection based on non-point anomalies required specific methodology not defined in these implementations.
def gen_anom_motif(anom_point=True, verbose=False, dim=(400, 20, 3), seed=0):
    dim_t, dim_n, dim_g = dim
    rng = np.random.RandomState(seed)
    rng1 = np.random.RandomState(seed + 1)
    if anom_point:
        p2 = np.array([1, 0, 0, 0, 0])
        P_list = [p2]
        motif1 = np.stack([P_list[int(i)] for i in np.zeros(dim_g)])

        motif = [motif1, -motif1]
    # Non-point anomaly case.
    else:
        # Poisson law motifs.
        # p1=np.histogram(np.random.gamma(np.random.normal(11,2,1),
        #                                 scale=np.random.normal(0.8,0,1),
        #                                 size=10000),
        #                 bins=5)[0]/10000
        # p2=np.histogram(np.random.gamma(np.random.normal(11,2,1),
        #                                 scale=np.random.normal(0.8,0,1),
        #                                 size=10000),
        #                 bins=5)[0]/10000

        # Mannual motifs.
        # p0 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
        p0 = np.array([1.7, 1.7, 1.4, 1.4, 1, 1, 1, 0.2, 0.1])
        p1 = np.array([1.4, 1.1, 0.7, 0.4, 0.3, 0, 0, 0, 0])
        p2 = np.array([1, 1.5, 1.3, 0.8, -0.8, -1, -0.8, 0, 0])
        p3 = np.array([0.5, 1.1, -0.6, -0.7, -0.3, 0, 0, 0, 0])

        P_list = [p0, p1, p2, p3]

        motif1 = np.stack([p1 for i in np.ones(dim_g)])
        motif1 = np.stack([P_list[int(i)] for i in np.ones(dim_g)])
        motif2 = np.stack(
            [P_list[int(i)] for i in rng.choice(np.arange(len(P_list)), dim_g)]
        )
        motif3 = np.stack(
            [P_list[int(i)] for i in rng1.choice(np.arange(len(P_list)), dim_g)]
        )

        motif = [motif1, -motif1, motif2, motif3]
    txt_anom = [
        "Anom positive faible",
        "Anom negative forte",
        "Anom negative faible",
        "Anom negative forte",
    ]
    if verbose:
        fig = plt.figure(figsize=(6, 4))
        for n, i in enumerate(motif[0:4]):
            plt.subplot(2, 2, n + 1)
            plt.title(txt_anom[n])
            im = plt.imshow(i * (1 + (n // 2) * 0.4), vmin=-2, vmax=2, cmap="seismic")
            fig.colorbar(im, orientation="vertical", shrink=1)
            plt.xticks(np.arange(5), np.arange(5) - 1)
            plt.show()
    return motif


# Anomaly generation
def gen_fake_anom(n_anom, motif, dim=(400, 20, 3)):
    dim_n, dim_t, dim_g = dim
    len(motif)
    len_motif = motif[0].shape[1]
    anom = np.zeros((dim_n * dim_t))
    impact_anom = np.zeros((dim_t * dim_n, dim_g))
    anom_list = []
    ttype = [1, 2, 1, 1]
    for n, i in enumerate(
        [
            dim_t + dim_t / 3,
            2 * dim_t + dim_t / 4,
            2 * dim_t + dim_t * 4 / 5,
            3 * dim_t + dim_t * 1 / 2,
        ]
    ):
        i = int(i)
        anom[i] = i
        anom_list.append((i // dim_t, i % dim_t, i, 1))
        impact_anom[i : (i + len_motif), :] += (motif[ttype[n] - 1]).T
    return (anom_list, anom, impact_anom)


# Anomaly generation
def gen_anom(n_anom, motif, dim=(400, 20, 3), seed=0):
    rng = np.random.RandomState(seed)
    dim_n, dim_t, dim_g = dim
    n_motif = len(motif)
    len_motif = motif[0].shape[1]
    anom = np.zeros((dim_n * dim_t))
    impact_anom = np.zeros((dim_t * dim_n, dim_g))
    anom_list = []
    for n, i in enumerate(
        rng.choice(np.arange(dim_n * dim_t - len_motif), n_anom, False)
    ):
        rng_bis = np.random.RandomState(seed + n)
        type_ = rng_bis.randint(1, n_motif + 1)
        anom[i] = type_ * (1 + (n % 2))
        anom_list.append((i // dim_t, i % dim_t, type_, n % 2))
        # impact_anom[i:(i+len_motif),:]+=(motif[type_-1]*(1+(n%2)*0.5)).T
        impact_anom[i : (i + len_motif), :] += (motif[type_ - 1]).T
    return (anom_list, anom, impact_anom)


# Noise generation
def gen_noise(var, dim=(400, 20, 3), seed=0):
    rng = np.random.RandomState(seed)
    dim_n, dim_t, dim_g = dim
    noise = rng.normal(0, np.sqrt(var), (dim_t * dim_n, dim_g))
    return noise


# Convolution by dimension for g filters
def convolve_signal(filters, signal, dim):
    dim_n, dim_t, dim_g = dim
    signal_conv = []
    for g in range(dim_g):
        signal_conv.append(
            convolve2d(signal, filters[g][:, ::-1], mode="same", boundary="wrap")[
                :, g
            ].reshape((-1, 1))
        )
    signal_conv = np.concatenate(signal_conv, axis=1)
    return signal_conv


# Generation of dynamic pattern through an arbitrary autoregressive pattern + randomly daily magnitude.
def gen_data_dyn(
    data_stat,
    noise_stat,
    var_mult,
    var_add,
    c_dyn,
    dim=(400, 20, 3),
    verbose=True,
    seed=0,
):
    dim_n, dim_t, dim_g = dim

    autocor = int(dim_t / 4) + (int(dim_t / 4)) % 2
    # signal_dyn = [
    #     [],
    #     [],
    #     [],
    #     [],
    # ]  # Moyenne bruité, Moyenne non bruité, noise_mult, noise_add
    filters = []
    for g in range(dim_g):
        list_ = []
        for i in range(dim_g):  # Autocorelation pattern generation
            rng1 = np.random.RandomState(seed + 100 + g * dim_g * 4 + i * 4)
            rng2 = np.random.RandomState(seed + 100 + g * dim_g * 4 + i * 4 + 1)
            rng3 = np.random.RandomState(seed + 100 + g * dim_g * 4 + i * 4 + 2)
            rng4 = np.random.RandomState(seed + 100 + g * dim_g * 4 + i * 4 + 3)

            list_.append(
                rng1.normal(0.2, 0.5, 1)
                * np.histogram(
                    rng2.normal(
                        0 + rng3.normal(1, 3, 1), max(0, 1 + rng4.normal(1, 1, 1)), 5000
                    ),
                    bins=autocor,
                    range=(-5, 15),
                )[0]
                / 5000
            )

        filter_ = np.zeros((autocor * 2 + 1, dim_g))
        filter_[:autocor] = np.concatenate([list_], axis=1).T
        filter_ = filter_ / np.abs(filter_).sum()
        filters.append(filter_)

    np.zeros(dim_t)
    data_dyn = convolve_signal(filters, np.copy(data_stat), dim)
    noise_dyn = convolve_signal(filters, np.copy(noise_stat), dim)
    # Somme de bruit blanc : mu = Sum(mu) = Sum(0) var = sqrt(sum(var)^2))
    var_dyn_add = np.array(
        [np.sum(np.power(filt, 2) * var_add) for filt in filters]
    ) * np.ones((dim_n * dim_t, dim_g))

    # Somme de bruit blanc : mu = Sum(mu) = Sum(0) var= sqrt(sum(var)^2))
    var_dyn_mult = convolve_signal(
        [np.power(filt, 2) for filt in filters], var_mult, dim
    )

    return (data_dyn, noise_dyn, filters, var_dyn_mult, var_dyn_add)


# def : percentile


# Noise mult + Noise add + signal
def add_noise(data, noise_mult, noise_add):
    signal = (data * (1 + noise_mult)) + noise_add
    return signal


def add_noise_aux(
    data_dyn,
    data_stat,
    noise_dyn_mult,
    noise_dyn_add,
    noise_stat_mult,
    noise_stat_add,
    c_dyn,
):
    signal_stat = add_noise(data_stat, noise_stat_mult, noise_stat_add)
    signal_dyn = add_noise(data_dyn, noise_dyn_mult, noise_dyn_add)
    signal = (1 - c_dyn) * signal_stat + (c_dyn) * signal_dyn
    return signal


# Core Generation function.
def core_gen(
    c_dyn=0.4,  # Magnitude of dynamic pattern.
    var_jour=0.15,  # Magnitude of random daily.
    v_ratio=1,
    per_anom=0.05,  # Percentage of anomaly
    var_mult=0.075,  # mutliplicatife noise magnitude
    var_add=0.01,  # additive noise magnitude
    f_anom=1,  # General anomaly magnitude
    anom_seuil=0,  # Minimal flat impact of anomaly
    verbose=0,  # Display type
    anom_point=True,  # Anomaly type
    seasons_ctx=[],  # List of (Periodicity,Impact Ponderation)
    categorials_ctx=[],  # List of (
    permutations_ctx=[],
    v_mean=200,
    dim=(400, 20, 3),
    seed=None,
):  # Dimension
    if seed is None:
        seed = np.random.randint(2**31 - 1)

    dim_n, dim_t, dim_g = dim
    n_anom = int(per_anom * (dim_n * dim_t))
    # Generation of mean and variance basic pattern.
    Pattern_day_mean, Pattern_day_var = gen_motif(verbose, dim=dim, seed=seed)
    layer = []
    # Generation of seasonal influence
    seas_ctx = [
        gen_seas_contexte(seas=seas, pond=pond, dim=dim, seed=seed + n)
        for n, (seas, pond) in enumerate(seasons_ctx)
    ]
    seas_ctx.append(np.ones(dim_n * dim_t))
    seas_ctx = np.array(seas_ctx).sum(axis=0)
    influence_ctx = np.tile(seas_ctx, dim_g).reshape(dim_g, -1).T
    layer.append(seas_ctx)

    permuts_ctx = [
        gen_permuts_contexte(seas=seas, pond=pond, dim=dim, seed=seed + n)
        for n, (seas, pond) in enumerate(permutations_ctx)
    ]
    permuts_ctx.append(np.zeros((dim_n * dim_t)))
    permuts_ctx = np.array(permuts_ctx).sum(axis=0)
    influence_ctx += np.tile(permuts_ctx, dim_g).reshape(dim_g, -1).T
    layer.append(permuts_ctx)

    cats_ctx = [
        gen_cat_contexte(size=size, effect=effect, per=per, dim=dim, seed=seed + n)
        for n, (size, effect, per) in enumerate(categorials_ctx)
    ]
    cats_ctx.append(np.zeros((dim_n * dim_t, dim_g)))
    influence_ctx += np.array(cats_ctx).sum(axis=0)
    layer.append(cats_ctx)

    influence_ctx = np.sqrt(np.abs(1 + ((influence_ctx - influence_ctx.mean()))))
    influence_ctx[influence_ctx < 0.05] = 0.05
    inf_max = np.percentile(influence_ctx, 99.5)
    influence_ctx[influence_ctx > inf_max] = inf_max

    # Concatenation of the mean pattern of each spatial dimension
    Mean_pattern = np.concatenate(
        [np.tile((pattern), dim_n)[:, None] for pattern in Pattern_day_mean], axis=1
    )

    # Concatenation of the variance pattern of each spatial dimension
    variance = np.concatenate(
        [np.tile((pattern), dim_n)[:, None] for pattern in Pattern_day_var], axis=1
    ) * np.sqrt(influence_ctx)
    variance = np.array([savgol_filter(i, 5, 3) for i in variance.T]).T

    # ramdomly daily magnitude
    rng = np.random.RandomState(seed)
    Ampleur_day = np.repeat(
        rng.normal(1, np.sqrt(var_jour), (dim_n, dim_g)), dim_t, axis=0
    )

    # Signal composition
    data_stat = Mean_pattern / Mean_pattern.mean(axis=0) * influence_ctx * Ampleur_day
    data_stat = v_mean * data_stat
    layer.append(Ampleur_day)
    layer.append(Mean_pattern / Mean_pattern.mean(axis=0))

    # Generation of noises
    var_add_apply = var_add * v_mean
    data_bruit_add = gen_noise(var_add_apply, dim, seed=seed - 1)

    # Injection of variance (Combination of Variance pattern and mutliplicatif noise) and additif noise.
    var_mult_apply = np.power(np.abs(data_stat), v_ratio)
    var_mult_apply = (
        var_mult_apply / var_mult_apply.mean() * var_mult * variance
    )  # !!! chg ref sigma->var
    var_mult_apply[var_mult_apply <= 0] = 0

    data_bruit_mult = gen_noise(var_mult_apply, dim, seed=seed - 2)
    noise_stat = data_bruit_mult + data_bruit_add

    # Generation of dynamic pattern over the crossing of mean motifs with seasonal influences.
    data_dyn, noise_dyn, filters, var_dyn_mult, var_dyn_add = gen_data_dyn(
        data_stat, noise_stat, var_mult_apply, var_add_apply, c_dyn, dim, seed
    )

    data_full = add_noise_aux(data_dyn, data_stat, 0, 0, 0, 0, c_dyn)
    data_full_dyn_noised = add_noise_aux(
        data_dyn + noise_dyn, data_stat, 0, 0, 0, 0, c_dyn
    )

    data_full_stat_noised = add_noise_aux(
        data_dyn, data_stat + noise_stat, 0, 0, 0, 0, c_dyn
    )
    data_full_noised = add_noise_aux(
        data_dyn + noise_dyn, data_stat + noise_stat, 0, 0, 0, 0, c_dyn
    )

    # Injection of anomaly.
    # Anomaly magnitude taking into account the variance and the noise to be dicernable.

    # Var Referential -> coef ²
    var_dyn_mult = np.power(c_dyn, 2) * var_dyn_mult
    var_dyn_add = np.power(c_dyn, 2) * var_dyn_add  # Var Referential -> coef ²

    var_stat_mult = np.power((1 - c_dyn), 2) * var_mult_apply
    var_stat_add = np.power((1 - c_dyn), 2) * var_add_apply

    var_stat = (
        var_stat_mult + var_stat_add
    )  # Variance statiques (facteurs explicatifs non temporelles)

    var_dyn = (
        var_dyn_mult + var_dyn_add
    )  # Variance dynamique (Autocorélation de la variance)

    var_full = var_dyn + var_stat

    print("test", var_stat.mean(), var_full.mean())

    # Generation of anomaly
    motif = gen_anom_motif(anom_point, verbose=0, dim=dim, seed=seed)
    anom_list, anom, impact_anom = gen_anom(n_anom, motif, dim, seed=seed)
    # anom_list,anom,impact_anom=gen_fake_anom(n_anom,motif,dim)
    var_sum_noise = 0
    if per_anom != 0:
        var_sum_noise = scipy.stats.norm.ppf(1 - per_anom / 2, 0, np.sqrt(var_full))

    # Injection of anomaly.
    # Anomaly magnitude taking into account the variance and the noise to be dicernable.
    Y = data_full_noised + (2 * var_sum_noise * f_anom * impact_anom)

    print(
        "Var_noise_dyn_emp:",
        trunc((noise_dyn * (c_dyn)).var()),
        "var_noise_stat_emp:",
        trunc((noise_stat * (1 - c_dyn)).var()),
        "Var_emp",
        trunc((Y - data_full).var()),
    )

    per_list = [0.01, 1, 2.5, 10, 25, 75, 90, 97.5, 99, 99.99]
    per = []
    for n, i in enumerate(per_list):
        coverage = []
        per.append(
            add_noise(data_full, 0, scipy.stats.norm.ppf(i / 100, 0, np.sqrt(var_full)))
        )
        for g in range(dim_g):
            coverage.append(100 * ((Y[:, g] < per[n][:, g]).sum() / len(Y)))
        print(i, np.round(coverage, 2))

    # Features
    Features_name = []
    Features_list = []
    Features_list.append((np.arange(dim_t * dim_n) % dim_t)[:, None])
    Features_name.append("main_period")

    if len(seasons_ctx) > 0:
        for n, (i, _) in enumerate(seasons_ctx):
            Features_list.append((np.arange(dim_t * dim_n) % i)[:, None])
            Features_name.append("period_" + str(i))

    if len(permutations_ctx) > 0:
        for n, (i, _) in enumerate(permutations_ctx):
            permuts = np.random.RandomState(seed + n).permutation(
                np.arange(dim_n * dim_t)
            )
            Features_list.append((np.arange(dim_t * dim_n) % i)[permuts][:, None])
            Features_name.append("period_cat_" + str(i))

    if len(categorials_ctx) > 0:
        for n, (size, effect, per) in enumerate(categorials_ctx):
            # Flatten context impact : To modify if impact can be different by dimension
            X_cat = (
                gen_cat_contexte(
                    size=size, effect=effect, per=per, dim=dim, seed=seed + n
                )
                != 0
            ).astype(int)[:, 0][:, None]

            Features_list.append(X_cat)
            Features_name.append("cat_" + str(n) + "_" + str(np.max(X_cat)))

    X = np.concatenate(Features_list, axis=1)
    X_name = Features_name

    train_rf = np.arange(dim_t * dim_n) < (dim_t * dim_n * 2 / 3)
    test_rf = np.arange(dim_t * dim_n) >= (dim_t * dim_n * 2 / 3)

    print(X.shape, X_name)

    return (
        Y,
        data_full,
        data_full_stat_noised,
        data_full_dyn_noised,
        impact_anom,
        anom,
        anom_list,
        var_full,
        var_stat,
        var_dyn,
        X,
        X_name,
        train_rf,
        test_rf,
        seed,
    )


def moving_average(a, n):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return np.concatenate([ret[: n - 1] / (np.arange(n - 1) + 1), ret[n - 1 :] / n])


def factory(
    X_row, Y, Features_name, train, test, freq, dim=(400, 20, 3), lags=0, lags_mean=[]
):
    dim_n, dim_t, dim_g = dim
    Y = Y.reshape(dim_n * dim_t, dim_g)
    new_features_list = []
    new_features_name = []
    for i, name in enumerate(Features_name):
        if "period" in name:
            X_cur = X_row[:, i]
            seas = np.argmax(X_cur)
            new_features_list.append(base_cos_freq((X_cur % seas) / seas, freq))
            new_features_name.append("cos" + name)
            new_features_name.append("sin" + name)
        else:
            X_cur = X_row[:, i]
            new_features_list.append(X_cur[:, None])
            new_features_name.append(name)

    for g in range(dim_g):
        for i in lags:
            new_features_list.append(np.roll(Y[:, g], i)[:, None])
            new_features_name.append("dim:" + str(g) + " lag:" + str(i))
        for i, j in lags_mean:
            new_features_list.append(
                moving_average(np.roll(Y[:, g], i), j - i)[:, None]
            )
            new_features_name.append(
                "dim:" + str(g) + " lag_mean:" + str(i) + " " + str(j)
            )

    X = np.concatenate(new_features_list, axis=1)
    name = np.array(new_features_name)
    return (X, Y, name, train, test)


# Display of data curve with mean and variance.
def plot_var(
    Y,
    data_full,
    variance,
    impact_anom=None,
    anom=None,
    f_obs=None,
    dim=(400, 20, 3),
    g=0,
    res_flag=False,
    fig_s=(20, 3),
    title=None,
    ylim=None,
):
    dim_n, dim_t, dim_g = dim
    anom_pred = (np.abs(impact_anom).sum(axis=-1) > 0).astype(int) - (anom > 0).astype(
        int
    )

    if anom_pred.sum() < 1:
        anom_pred[0] = 1
        anom_pred[-1] = 1

    step = g
    plt.figure(figsize=fig_s)
    plt.title(title)
    # norm = Y.mean(axis=0)
    if anom_pred.sum() < 1:
        anom_pred[0] = 1
        anom_pred[-1] = 1

    ni = [100, 98, 95, 80, 50]
    color_full = [
        (0.5, 0.0, 0.5),
        (0.8, 0, 0),
        (0.8, 0.6, 0),
        (0, 0.8, 0),
        (0, 0.4, 0),
        (0, 0.8, 0),
        (0.8, 0.6, 0),
        (0.8, 0, 0),
        (0.5, 0.0, 0.5),
    ]
    color_full2 = [
        (0.5, 0.0, 0.5),
        (0.8, 0, 0),
        (0.8, 0.6, 0),
        (0, 0.8, 0),
        (0, 0.4, 0),
        (0, 0.4, 0),
        (0, 0.8, 0),
        (0.8, 0.6, 0),
        (0.8, 0, 0),
        (0.5, 0.0, 0.5),
    ]

    per_list = [0.01, 1, 2.5, 10, 25, 75, 90, 97.5, 99, 99.99]
    per = []

    res = data_full * 0
    if res_flag:
        res = data_full

    for i in per_list:
        per.append(
            add_noise(
                data_full - res,
                0,
                scipy.stats.norm.ppf((i / 100), 0, np.sqrt(variance)),
            ),
        )

    for i in range(len(per) - 1):
        plt.fill_between(
            np.arange(len(f_obs)),
            per[i][f_obs, step],
            per[i + 1][f_obs, step],
            color=color_full[i],
            alpha=0.20,
        )
    for i in range(len(per)):
        plt.plot(
            np.arange(len(f_obs)),
            per[i][f_obs, step],
            color=color_full2[i],
            linewidth=0.5,
            alpha=0.40,
        )
        if i > 4:
            plt.fill_between(
                [],
                [],
                [],
                color=color_full2[i],
                label=str(ni[9 - i]) + "% Coverage",
                alpha=0.20,
            )

    plt.plot(
        Y[f_obs, step] - res[f_obs, step],
        label="Series",
        color="black",
        linewidth=1.5,
        marker="o",
        ms=3,
    )
    flag = impact_anom[f_obs, step] != 0
    print(flag.sum())
    plt.plot(
        np.arange(len(f_obs))[flag],
        Y[f_obs, step][flag] - res[f_obs, step][flag],
        label="Anom",
        color="red",
        ls="",
        marker="X",
        ms=10,
        alpha=0.8,
    )

    if False:
        f_anom = np.repeat(np.abs(impact_anom)[f_obs, step] > 0, 7)
        for i in range(8):
            f_anom += np.roll(f_anom, i - 4)
        f_anom = f_anom > 0
        plt.fill_between(
            np.arange(len(f_obs) * 7) / 7 - (3 / 7),
            -1000,
            Y[f_obs, step].max() * 1.2,
            where=f_anom,
            facecolor="blue",
            alpha=0.2,
            label="Anomaly",
            zorder=-10,
        )

    if ylim is None:
        plt.ylim(per[0][f_obs, step].min() * 1.05, per[-1][f_obs, step].max() * 1.05)
    else:
        plt.ylim(ylim[0], ylim[1])
    plt.legend(ncol=7, fontsize=14)
    plt.xlim(0, len(f_obs))
    plt.tight_layout()
    plt.show()
    return (per, per_list)


def generate_default(
    Name_data="data_test", storing="", dict_param=dict(), seed=7885000
):
    pass

    # Multivariate time series génération of size : (Dim_n, dim_t, dim_g) : (Day, Hour, Station)
    n_year = 10
    dim_g = 3
    n_day = 3
    dim_t = 20
    dim_n = n_day * dim_t * n_year
    dim_g = 3
    dim = (dim_n, dim_t, dim_g)

    # Param generation
    dict_param_default = {
        "c_dyn": 0.6,  # 0.7
        "n_year": n_year,
        "n_day": n_day,
        "var_jour": 0.01,  # 0.08
        "per_anom": 0.01,
        "var_mult": 0.3,  # Pourcentage de bruit multiplicatifs. 0.30
        "var_add": 0.03,  # Pourcentage de bruit additifs. 0.12
        "f_anom": 0.8,
        "anom_seuil": 0.01,
        "verbose": 0,
        # [(dim_t*n_day,1.0),((dim_n*dim_t)/n_year,0.8)],
        "seas": [(dim_t * n_day, 2.0), ((dim_n * dim_t) / n_year, 1.5)],
        # [(dim_t*n_day,0.5),(255,0.2)],
        "permuts": [],
        # [(dim_t*n_day, 0.35), (255, 0.15)],
        # [(7,0.3,0.4),(3,0.8,0.2)],#[(7,0.4,0.3),(1,0.2,0.5),(3,0.8,0.2)],
        "cats": [],
        # [(7, 0.3, 0.4)],
        "v_mean": 2.5,
        "anom_point": True,
    }

    for key in dict_param_default.keys():
        if key not in dict_param.keys():
            dict_param[key] = dict_param_default[key]

    # seed=7885912

    # Génération d'un dataset
    gen_variable = core_gen(
        c_dyn=dict_param["c_dyn"],
        var_jour=dict_param["var_jour"],
        v_ratio=0.3,
        per_anom=dict_param["per_anom"],
        var_mult=dict_param["var_mult"],
        var_add=dict_param["var_add"],
        f_anom=dict_param["f_anom"],
        anom_seuil=dict_param["anom_seuil"],
        verbose=dict_param["verbose"],
        anom_point=dict_param["anom_point"],
        seasons_ctx=dict_param["seas"],
        categorials_ctx=dict_param["cats"],
        permutations_ctx=dict_param["permuts"],
        v_mean=dict_param["v_mean"],
        dim=dim,
        seed=seed,
    )

    (
        Y,
        data_mean,
        data_mean_stat,
        data_mean_dyn,
        impact_anom,
        anom,
        anom_list,
        var,
        var_stat,
        var_dyn,
        X_init,
        X_name_init,
        train,
        test,
        seed,
    ) = gen_variable

    # Remove multivariate other dimension

    X, Y, X_name, train, test = factory(
        X_init,
        Y,
        X_name_init,
        train,
        test,
        freq=[2],
        lags=[1, 2],
        lags_mean=[(1, int(dim_t / 2)), (1, n_day * dim_t)],
        dim=(dim_n, dim_t, dim_g),
    )

    Cat_flag = np.array(["cat" in i for i in X_name_init])
    X_cat = (
        (X_init[:, Cat_flag])
        * np.array([np.power(2, i) for i in range(0, Cat_flag.sum())])
    ).sum(axis=1)

    context = np.concatenate(
        [X_init[:, np.invert(Cat_flag)], X_cat[:, None], Y, var], axis=1
    )
    context_name = np.concatenate(
        [
            np.array(X_name_init)[np.invert(Cat_flag)].tolist(),
            ["Cat_contexts"],
            ["lag_Y_Numerical", "lag_dim1_Numerical", "lag_dim2_Numerical"],
            ["Var_Y_Numerical", "var_dim1_Numerical", "var_dim2_Numerical"],
        ]
    )

    context[:, 0] = np.floor(
        (context[:, 0] / (context[:, 0].max() / (dim_t - 0.000001)))
    ).astype(int)
    context[:, 1] = np.floor(
        (context[:, 1] / (context[:, 1].max() / (n_day - 0.000001)))
    ).astype(int)
    context[:, 2] = np.floor((context[:, 2] / (context[:, 2].max() / (5.99)))).astype(
        int
    )

    np.arange(dim_n) % dict_param["n_day"]
    anom_pred = (np.abs(impact_anom).sum(axis=-1) > 0).astype(int) - (anom > 0).astype(
        int
    )
    if anom_pred.sum() < 1:
        anom_pred[0] = 1
        anom_pred[-1] = 1

    f_obs = np.arange(len(Y))[test][0:200]

    # Env var
    for g in range(dim_g):
        per, per_info = plot_var(
            Y, data_mean, var, impact_anom, anom, f_obs=f_obs, dim=dim, g=g
        )

    f_obs = np.arange(len(Y))
    per, per_info = plot_var(
        Y, data_mean, var, impact_anom, anom, f_obs=f_obs, dim=dim, g=g
    )

    # Env full stat
    # per_bis,per_info=Gen.plot_var(Y,data_mean_dyn,var_stat,impact_anom,anom,f_obs=f_obs,dim=dim,g=0)

    # Env var dyn
    # per_bis,per_info=Gen.plot_var(Y,data_mean_stat,var_dyn,impact_anom,anom,f_obs=f_obs,dim=dim,g=0)

    plt.figure(figsize=(14, 2))
    plt.subplot(1, 2, 1)
    plt.scatter(
        data_mean[:, 0],
        np.abs(Y[:, 0] - data_mean[:, 0]),
        s=1,
        c=np.arange(dim_t * dim_n) % dim_t,
        cmap="jet",
    )
    plt.subplot(1, 2, 2)
    plt.scatter(
        data_mean[:, 0], var[:, 0], s=1, c=np.arange(dim_t * dim_n) % dim_t, cmap="jet"
    )
    plt.show()

    dict_data = {}
    dict_data["X"] = X
    dict_data["Y"] = Y
    dict_data["context"] = context
    dict_data["X_name"] = X_name
    dict_data["train"] = train
    dict_data["test"] = test
    dict_data["X_split"] = np.arange(len(X)) // (len(X) / 5)
    dict_data["context_name"] = context_name
    dict_data["aux"] = {
        "data_mean": data_mean,
        "data_mean_dyn": data_mean_dyn,
        "impact_anom": impact_anom,
        "anom": anom,
        "var": var,
        "var_dyn": var_dyn,
        "var_stat": var_stat,
        "f_obs": f_obs,
        "dim": dim,
    }

    # write(storing, [Name_data], dict_data)
    return dict_data
