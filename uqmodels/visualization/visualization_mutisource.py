"""
Visualization_multisource module.
"""

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

from uqmodels.preprocessing.structure import (
    get_regular_step_scale,
    get_step_mask,
    step_to_date,
    window_expansion,
)
from uqmodels.processing import read
from uqmodels.utils import Extract_dict, compute_born, expand_flag
from uqmodels.visualization.visualization import provide_cmap

mpl_date_format = mpl.dates.DateFormatter("%d/%m %H:%M")


def load_from_metadata(storing, str_keys_metadata):
    metadata = read(storing, ["metadata.json"])
    return Extract_dict(metadata, str_keys_metadata)


def load_and_select(storing, keys, x_min, x_max):
    str_keys_metadata = [
        "sources_name",
        "shift",
        "range_",
        "time_offset",
        "dtype",
        "date_init",
        "list_time_factor",
    ]
    (
        sources_name,
        shift,
        range_,
        time_offset,
        dtype,
        date_init,
        list_time_factor,
    ) = load_from_metadata(storing, str_keys_metadata)

    if keys[-1] == "raw_series":
        serie = read(storing, keys)
        time_mask = get_step_mask(serie["time"].values, x_min, x_max, out_of_mask=True)
        y = serie["data"].values[time_mask]
        x = serie["time"].values[time_mask] / shift
        dates = serie["timestamp"].values[time_mask]

    elif keys[-1] == "interpolated":
        serie = read(storing, keys)
        time_mask = get_step_mask(serie[0], x_min, x_max, out_of_mask=True)
        x = serie[0][time_mask]
        y = serie[1][time_mask]
        dates = step_to_date(
            serie[0][time_mask], delta=shift, dtype=dtype, date_init=date_init
        )

    elif keys[-1] == "regular_date":
        raw_id = sources_name.index(keys[0])
        t_scale = list_time_factor[raw_id]
        times = get_regular_step_scale(shift * t_scale, range_, time_offset)
        time_mask = get_step_mask(times, x_min, x_max, out_of_mask=True)
        x = times[time_mask] / shift
        y = None
        dates = step_to_date(x, delta=shift, dtype=dtype, date_init=date_init)

    elif keys[-1] == "channels":
        serie = read(storing, keys)
        # Size plateau affichage
        n_point = 10
        # Data recovering
        channels = read(storing, keys)
        # Time-scale recovering
        raw_id = sources_name.index(keys[0])
        t_scale = list_time_factor[raw_id]
        times = get_regular_step_scale(shift * t_scale, range_, time_offset=time_offset)
        mask_time = get_step_mask(times, x_min, x_max, out_of_mask=True)

        x = (times / shift)[mask_time]
        y = np.repeat(channels[mask_time], n_point, axis=0)
        dates = step_to_date(x, delta=shift, dtype=dtype, date_init=date_init)
    return (x, y, dates)


# Auxiliar minor-processing


def apply_cmap(val, vmin, vmax, cmap):
    """Transform valyes array into color values array using cmap and considering bound [vmin,vmax]

    Args:
        val (array): Values to turn to color
        vmin (float): min_val
        vmax (float): max_val
        cmap (cmap): matplotlib cmap

    Returns:
        _type_: Array of color
    """
    val = np.maximum(val, vmin)
    val = np.minimum(val, vmax)
    val = (val - vmin) / (vmax - vmin)
    return [cmap(i) for i in val]


def compute_dev_score(val, y, vmin, vmax):
    """Compute signed relative errors

    Args:
        val (array): Prediction
        y (array): Target
        vmin (float): negative minimum sensitivité
        vmax (float): positive minimun sensitivity

    Returns:
        r: signed relative errors.
    """

    r = y - val
    flag_res_neg = r > 0
    flag_res_pos = r < 0

    r[flag_res_neg] = r[flag_res_neg] / (y[flag_res_neg] - vmin[flag_res_neg])
    r[flag_res_pos] = r[flag_res_pos] / (vmax[flag_res_pos] - y[flag_res_pos])
    return r


# Main plot function


def plot_row_series(
    storing, sensors_mask, x_min, x_max, figsize=(12, 10), date_init=None, anom=None
):
    # Load metadata

    str_keys_metadata = [
        "sources_name",
        "step",
        "shift",
        "range_",
        "dtype",
        "date_init",
        "list_time_factor",
    ]

    (
        sensors_name,
        time_step,
        shift,
        range_,
        dtype,
        date_init,
        list_step_target,
    ) = load_from_metadata(storing, str_keys_metadata)

    plt.figure(figsize=figsize)
    for n, sensor_id in enumerate(sensors_mask):
        # Load data

        keys = [sensors_name[sensor_id], "raw_series"]
        x, y, dates = load_and_select(storing, keys, x_min, x_max)

        ax = plt.subplot(len(sensors_mask), 1, n + 1)

        ax.set_xlim(
            step_to_date(x_min, shift, dtype=dtype, date_init=date_init),
            step_to_date(x_max, shift, dtype=dtype, date_init=date_init),
        )

        plt.plot(dates, y, marker="o", lw=0.5, markersize=2)
        plt.legend(title=sensors_name[sensor_id], loc=2, ncol=4)

        if n != len(sensors_mask) - 1:
            plt.xticks([])

        if anom:
            if isinstance(anom, tuple):
                anom = [anom]

            for anom_ in anom:
                keys = [sensors_name[sensor_id], "regular_date"]
                x_anom, _, dates_anom = load_and_select(storing, keys, x_min, x_max)
                flag_anom = (x >= anom_[0]) & (x < anom_[1])

                plt.fill_between(
                    x=dates_anom,
                    y1=np.repeat(x_anom, 5).astype(float) * 0 + y.min(),
                    y2=np.repeat(x_anom, 5).astype(float) * 0 + y.max(),
                    where=expand_flag(np.repeat(flag_anom, 5)),
                    alpha=0.15,
                    fc="red",
                    ec="None",
                    label="Zone d'anomalie",
                )
        ax.xaxis.set_major_formatter(mpl_date_format)
    plt.tight_layout(pad=0, w_pad=0, h_pad=0)
    # plt.subplots_adjust(wspace=-0.5)
    plt.show()


def plot_row_series_statistics(
    storing,
    sensors_mask,
    x_min,
    x_max,
    figsize=(12, 10),
    interpolate_data=False,
    date_init=None,
):
    # Load metadata
    str_keys_metadata = [
        "sources_name",
        "step",
        "shift",
        "range_",
        "dtype",
        "date_init",
        "list_time_factor",
    ]

    (
        sensors_name,
        time_step,
        shift,
        range_,
        dtype,
        date_init,
        list_step_target,
    ) = load_from_metadata(storing, str_keys_metadata)

    x_min, x_max = x_min * time_step, x_max * time_step

    for sensor_id in sensors_mask:
        f, list_plt = plt.subplots(
            4,
            1,
            gridspec_kw={"height_ratios": [2, 1, 1, 1]},
            figsize=figsize,
            sharex=True,
        )

        if interpolate_data:
            raw_id = sensor_id
            keys = [sensors_name[raw_id], "interpolated"]
            x_inter, y_inter, dates_inter = load_and_select(storing, keys, x_min, x_max)

            list_plt[0].plot(
                dates_inter,
                y_inter,
                color="red",
                marker="x",
                lw=0.2,
                markersize=3,
                label="Données interpolées",
                zorder=10,
            )

        keys = [sensors_name[sensor_id], "raw_series"]
        x, y, dates = load_and_select(storing, keys, x_min, x_max)
        list_plt[0].plot(
            dates,
            y,
            color="black",
            marker="o",
            lw=0.4,
            markersize=5,
            label="Observations brutes",
        )

        list_plt[0].legend()

        list_plt[0].set_xlim(
            step_to_date(x_min, shift, dtype=dtype, date_init=date_init),
            step_to_date(x_max, shift, dtype=dtype, date_init=date_init),
        )

        # Time scale information recovering

        keys = [sensors_name[raw_id], "channels"]
        x_chan, y_chan, dates_chan = load_and_select(storing, keys, x_min, x_max)

        list_plt[1].plot(
            dates_chan,
            y_chan[:, 0],
            label="Moyenne",
            color="Gold",
            ls="--",
            lw=1,
        )

        list_plt[2].plot(
            dates_chan,
            y_chan[:, 1],
            label="Standard deviation",
            color="purple",
            lw=0.8,
        )

        list_plt[3].plot(
            dates_chan,
            y_chan[:, 2],
            label="Extremum",
            color="red",
            ls=":",
            lw=0.8,
        )

        if False:
            list_plt[4].plot(
                dates_chan,
                y_chan[:, 3],
                label="Etat d'activité",
                color="darkgreen",
                lw=0.8,
            )

        list_plt[0].legend()
        list_plt[1].legend()
        list_plt[2].legend()
        list_plt[3].legend()
        # list_plt[4].legend()
        f.tight_layout(pad=0.4, w_pad=0.5, h_pad=0.7)
        f.show()


def plot_channel(
    storing_data,
    storing_res,
    sensors_mask,
    x_min,
    x_max,
    mode=None,
    figsize=(20, 4),
    canal_nature=["Mean", "Std", "ExT"],
):
    # Recovering row data
    str_keys_metadata = [
        "sources_name",
        "step",
        "shift",
        "range_",
        "time_offset",
        "dtype",
        "date_init",
        "list_time_factor",
        "list_sensors_primaire",
    ]

    (
        sensors_name,
        time_step,
        shift,
        range_,
        time_offset,
        dtype,
        date_init,
        list_step_target,
        list_sensors_primaire,
    ) = load_from_metadata(storing_data, str_keys_metadata)
    x_min, x_max = x_min * time_step, x_max * time_step

    # recovering anomalies
    dict_res_anom = read(storing_res, ["dict_res_anom"])

    str_keys_data = ["S_anom_chan", "S_anom_sensor", "list_bot", "list_top"]
    S_anom_chan, S_anom_sensor, list_bot, list_top = Extract_dict(
        dict_res_anom, str_keys_data
    )
    n_chan = list_bot[0].shape[-1]

    channel_mask = np.arange(len(list_sensors_primaire) * n_chan)[
        [
            i // n_chan in sensors_mask
            for i in range(len(list_sensors_primaire) * n_chan)
        ]
    ]

    f, plot_list = plt.subplots(
        len(channel_mask),
        1,
        figsize=(figsize[0], figsize[1] * len(channel_mask)),
        sharex=True,
    )

    for n_ind, ind in enumerate(channel_mask):
        sensor_id = ind // n_chan
        channel_id = ind % n_chan
        raw_id = list_sensors_primaire[sensor_id]

        # Time scale information recovering
        t_scale = list_step_target[raw_id]

        times = get_regular_step_scale(shift * t_scale, range_, time_offset=time_offset)

        mask_time = get_step_mask(times, x_min, x_max, out_of_mask=True)

        y = read(storing_data, [sensors_name[raw_id], "channels.p"])
        output = read(storing_res, [sensors_name[raw_id], "output.p"])
        pred = output[0]

        x_new = times[mask_time]
        dates = step_to_date(times[mask_time], shift, dtype=dtype, date_init=date_init)

        Y_spline = y[mask_time, channel_id]
        pred_spline = pred[mask_time, channel_id]
        name_sensor = sensors_name[raw_id]
        top_spline = list_top[sensor_id][mask_time, channel_id]
        bot_spline = list_bot[sensor_id][mask_time, channel_id]

        plot_list[n_ind].plot(
            dates, Y_spline, marker="o", lw=0, markersize=2, color="black"
        )

        plot_list[n_ind].plot(dates, Y_spline, lw=0.5, markersize=2, color="black")

        plot_list[n_ind].plot(dates, pred_spline, color="darkgreen", lw=1)

        if mode is not None:
            var_A = output[1]
            var_E = output[2]

            bot_A, top_A = compute_born(
                y_pred=pred[mask_time, channel_id],
                sigma=np.sqrt(var_A)[mask_time, channel_id],
                alpha=0.001,
            )

            bot_E, top_E = compute_born(
                y_pred=pred[mask_time, channel_id],
                sigma=np.sqrt(var_E)[mask_time, channel_id],
                alpha=0.001,
            )

            bot_spline1 = bot_A + np.sqrt(var_E)[mask_time, channel_id]
            top_spline1 = top_A - np.sqrt(var_E)[mask_time, channel_id]

            bot, top = compute_born(
                y_pred=pred[mask_time, channel_id],
                sigma=np.sqrt(var_E + var_A)[mask_time, channel_id],
                alpha=0.001,
            )

            plot_list[n_ind].plot(
                dates, bot_A, color="blue", lw=1, ls="dotted", label="var_A"
            )
            plot_list[n_ind].plot(dates, top_A, color="blue", lw=1, ls="dotted")
            plot_list[n_ind].plot(dates, bot_E, color="darkorange", lw=1, ls="dotted")
            plot_list[n_ind].plot(
                dates, top_E, color="darkorange", lw=1, ls="dotted", label="var_E"
            )

            plot_list[n_ind].fill_between(
                x=dates,
                y1=bot,
                y2=top,
                alpha=0.2,
                fc="b",
                ec="None",
                color="lime",
                label="incertitude_total",
                lw=1,
            )  # ,label='Estimate IC'

            plot_list[n_ind].fill_between(
                x=dates,
                y1=bot_spline1,
                y2=top_spline1,
                alpha=0.1,
                fc="b",
                ec="None",
                color="red",
                label="IC_epistemic_penalized",
                lw=1,
            )  # ,label='Estimate IC'

        plot_list[n_ind].fill_between(
            x=dates,
            y1=top_spline,
            y2=bot_spline,
            alpha=0.2,
            fc="b",
            ec="None",
            color="orange",
            label="IC-99%",
            lw=1,
        )  # ,label='Estimate IC'

        dates_ = step_to_date(
            window_expansion(x_new, n_expand=5, delta=shift / 5),
            delta=shift,
            dtype=dtype,
            date_init=date_init,
        )

        plot_list[n_ind].fill_between(
            x=dates_,
            y1=np.repeat(bot_spline, 5).astype(float),
            y2=np.repeat(Y_spline, 5).astype(float),
            where=np.repeat(top_spline < Y_spline, 5),
            alpha=0.2,
            zorder=-1000,
            color="red",
            lw=1,
        )

        plot_list[n_ind].fill_between(
            x=dates_,
            y1=np.repeat(Y_spline, 5).astype(float),
            y2=np.repeat(top_spline, 5).astype(float),
            where=np.repeat(Y_spline < bot_spline, 5),
            alpha=0.2,
            label="alerte",
            zorder=-1000,
            color="red",
            lw=1,
        )

        plot_list[n_ind].legend(
            title=str(ind) + " : " + name_sensor + "_" + canal_nature[ind % n_chan],
            loc=4,
            fontsize=10,
        )

        plot_list[n_ind].set_xlim(
            step_to_date(x_min, shift, dtype=dtype, date_init=date_init),
            step_to_date(x_max, shift, dtype=dtype, date_init=date_init),
        )

        plot_list[n_ind].xaxis.set_major_formatter(
            mpl.dates.DateFormatter("%d/%m %H:%M")
        )

    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=0.8)
    plt.show()


def plot_anom_mat(
    storing_data,
    storing_res,
    sensors_mask,
    x_min,
    x_max,
    figsize=(20, 12),
    metadata=None,
):

    dict_res_anom = read(storing_res, ["dict_res_anom"])

    if metadata is None:
        metadata = read(storing_data, ["metadata.json"])
        str_keys_metadata = [
            "step",
            "shift",
            "range_",
            "time_offset",
            "date_init",
            "list_time_factor",
            "list_sensors_primaire",
        ]

        (
            time_step,
            shift,
            range_,
            time_offset,
            dtype,
            date_init,
            list_time_factor,
            list_sensors_primaire,
        ) = load_from_metadata(storing_data, str_keys_metadata)

    else:
        str_keys_metadata = [
            "step",
            "shift",
            "range_",
            "time_offset",
            "dtype" "date_init",
            "list_time_factor",
            "list_sensors_primaire",
        ]

        (
            time_step,
            shift,
            range_,
            time_offset,
            date_init,
            list_time_factor,
            list_sensors_primaire,
        ) = Extract_dict(dict_res_anom, str_keys_metadata)

    x_min, x_max = x_min * shift, x_max * shift

    # recovering anomalies
    dict_res_anom = read(storing_res, ["dict_res_anom"])

    str_keys_data = [
        "S_anom_chan",
        "S_anom_sensor",
        "S_anom_agg",
        "list_bot",
        "list_top",
    ]

    S_anom_channels, S_anom_sources, S_anom_agg, list_bot, list_top = Extract_dict(
        dict_res_anom, str_keys_data
    )

    n_sensor = S_anom_sources.shape[1]
    n_chan = list_bot[0].shape[-1]

    channel_mask = np.arange(n_sensor * n_chan)[
        [i // n_chan in sensors_mask for i in np.arange(n_sensor * n_chan)]
    ]

    f, (a0, a1, a2) = plt.subplots(
        3, 1, gridspec_kw={"height_ratios": [6, 1.2, 0.3]}, figsize=figsize, sharex=True
    )

    plt.tight_layout()

    times = get_regular_step_scale(shift, range_, time_offset)
    mask_time = get_step_mask(times, x_min, x_max, out_of_mask=True)

    x = (times / shift)[mask_time]
    dates = step_to_date(x, delta=shift, dtype=dtype, date_init=date_init)

    d0 = mdates.date2num(dates[0].to_pydatetime())
    d1 = mdates.date2num(dates[-1].to_pydatetime())
    a0.imshow(
        (S_anom_channels.T)[channel_mask][:, mask_time],
        vmin=-3,
        vmax=3,
        cmap=provide_cmap(mode="bluetored"),
        extent=[d0, d1, 0, len(channel_mask)],
        aspect="auto",
    )
    print(len(channel_mask))
    for i in range(len(channel_mask) * len(sensors_mask)):
        a0.hlines(i, d0, d1, color="grey", lw=0.5)

    for i in range(len(sensors_mask)):
        a0.hlines(i * n_chan, d0, d1, color="black", lw=1)
    a1.imshow(
        (S_anom_sources.T)[sensors_mask][:, mask_time],
        vmin=-2,
        vmax=2,
        cmap=plt.get_cmap("seismic"),
        extent=[d0, d1, 0, len(sensors_mask)],
        aspect="auto",
    )

    for i in range(len(sensors_mask))[1:]:
        a1.hlines(i, d0, d1, color="black", lw=1)

    a2.imshow(
        (S_anom_agg.T)[:, mask_time],
        vmin=-2,
        vmax=2,
        cmap=plt.get_cmap("seismic"),
        extent=[d0, d1, 0, 1],
        aspect="auto",
    )
    a2.set_ylim(0, S_anom_agg.shape[1])
    a1.set_ylim(0, S_anom_sources[:, sensors_mask].shape[1])
    a0.set_ylim(0, S_anom_channels[:, channel_mask].shape[1])

    a0.xaxis_date()
    a0.xaxis.set_major_formatter(mpl_date_format)

    a1.xaxis_date()
    a1.xaxis.set_major_formatter(mpl_date_format)

    a2.xaxis_date()
    a2.xaxis.set_major_formatter(mpl_date_format)
    # a2.set_yticks()

    a0.set_title("Score par statistiques")
    a1.set_title("Aggrégation par capteurs")
    a2.set_title("Aggégations globale")
    plt.show()


def plot_analysis(
    storing_data, storing_res, sensors_mask, x_min, x_max, figsize=(20, 4), matplot=True
):
    # Recovering row data

    read(storing_data, ["metadata.json"])
    str_keys_metadata = [
        "sources_name",
        "step",
        "shift",
        "range_",
        "time_offset",
        "dtype",
        "date_init",
        "list_time_factor",
        "list_sensors_primaire",
    ]

    (
        sources_name,
        time_step,
        shift,
        range_,
        time_offset,
        dtype,
        date_init,
        list_time_factor,
        list_sensors_primaire,
    ) = load_from_metadata(storing_data, str_keys_metadata)
    x_min, x_max = x_min * shift, x_max * shift

    # recovering anomalies
    dict_res_anom = read(storing_res, ["dict_res_anom"])

    str_keys_data = [
        "S_anom_chan",
        "S_anom_sensor",
        "S_anom_agg",
        "list_bot",
        "list_top",
    ]

    S_anom_channels, S_anom_sources, S_anom_agg, list_bot, list_top = Extract_dict(
        dict_res_anom, str_keys_data
    )

    n_chan = int(S_anom_channels.shape[1] / S_anom_sources.shape[1])
    sensors_mask_length = len(sensors_mask)
    plt.figure(figsize=(figsize[0], figsize[1] * sensors_mask_length))

    times_s_anom = get_regular_step_scale(shift, range_, 1)

    mask_time_s_anom = get_step_mask(times_s_anom, x_min, x_max, out_of_mask=True)

    for n, sensor_id in enumerate(sensors_mask):
        raw_id = list_sensors_primaire[sensor_id]
        source = sources_name[raw_id]
        ax = plt.subplot(sensors_mask_length, 1, n + 1)

        time_factor = list_time_factor[raw_id]

        y = read(storing_data, [source, "channels"])
        output = read(storing_res, [source, "output"])

        serie = read(storing_data, [source, "raw_series"])

        pred = output[0]
        bot = list_bot[sensor_id]
        top = list_top[sensor_id]

        times_row = serie["time"].values
        mask_time_serie = get_step_mask(times_row, x_min, x_max, out_of_mask=True)
        v_freq = []
        ind_time_reg = 0

        # Compute Freq anom indicator only if chan = 4 ie the 4'th chan is freq.

        if n_chan == 4:
            for i in serie["time"][mask_time_serie]:
                while i > times_s_anom[mask_time_s_anom][ind_time_reg + 1]:
                    if ind_time_reg + 2 < mask_time_s_anom.sum():
                        ind_time_reg += 1
                    else:
                        break
                v_freq.append(
                    S_anom_channels[mask_time_s_anom, sensor_id * n_chan + 4][
                        ind_time_reg
                    ]
                )
            c_freq = apply_cmap(
                np.array(v_freq), -1.7, 1.7, provide_cmap(mode="cyantopurple")
            )
        else:
            c_freq = "black"

        v_deviation = []

        times = get_regular_step_scale(shift * time_factor, range_, time_offset)
        mask_time = get_step_mask(times, x_min, x_max, out_of_mask=True)

        ind_time_reg = 0
        for loc, i in enumerate(serie["time"][mask_time_serie]):
            while i >= (times[mask_time][ind_time_reg + 1]):
                if ind_time_reg + 2 < mask_time.sum():
                    ind_time_reg += 1
                else:
                    break

            v = serie["data"][mask_time_serie].values[loc]
            yval = y[mask_time][ind_time_reg, 0]
            vmin = yval - top[mask_time][ind_time_reg, 2]  # ExT apply on bot
            vmax = yval + top[mask_time][ind_time_reg, 2]  # ExT apply on top
            v_deviation.append([v, yval, vmin, vmax])

        v_deviation = np.array(v_deviation)
        v_deviation = compute_dev_score(
            v_deviation[:, 0], v_deviation[:, 1], v_deviation[:, 2], v_deviation[:, 3]
        )
        c_deviation = apply_cmap(
            -np.array(v_deviation), -2, 2, provide_cmap(mode="bluetored")
        )

        # plot row series : dates = str_array_to_date(serie["timestamp"].values[time_mask])
        plt.plot(
            serie["timestamp"][mask_time_serie].values,
            serie["data"].values[mask_time_serie],
            marker="o",
            lw=0.3,
            markersize=1,
            label="real_serie",
            color="black",
        )

        plt.scatter(
            serie["timestamp"][mask_time_serie].values,
            serie["data"].values[mask_time_serie],
            marker="o",
            s=70 - min(50 * (len(serie["time"].values[mask_time_serie]) / 400), 50),
            c=c_deviation,
            edgecolors=c_freq,
            linewidth=1.8
            - min(1.3 * (len(serie["time"].values[mask_time_serie]) / 800), 1.3),
            zorder=100,
        )

        bounds = np.arange(-1, 1, 0.02)
        norm = mpl.colors.BoundaryNorm(bounds, provide_cmap(mode="bluetored").N)
        cbar1 = plt.colorbar(
            mpl.cm.ScalarMappable(cmap=provide_cmap(mode="bluetored"), norm=norm),
            orientation="horizontal",
            label="Val_anom",
            fraction=0.06,
            pad=0,
            shrink=0.30,
            anchor=(0.80, 1.3),
            aspect=80,
        )
        cbar1.set_ticks([-4 / 5, -2 / 5, 0, 2 / 5, 4 / 5])
        cbar1.set_ticklabels(["-2", "-1", "0", "1", "2"])

        norm = mpl.colors.BoundaryNorm(bounds, provide_cmap(mode="bluetored").N)
        cbar2 = plt.colorbar(
            mpl.cm.ScalarMappable(cmap=provide_cmap(mode="bluetored"), norm=norm),
            orientation="horizontal",
            label="Freq_anom",
            fraction=0.06,
            pad=0.14,
            shrink=0.30,
            anchor=(0.20, 1.3),
            aspect=80,
        )

        cbar2.set_ticks([-1 / 2, 0, 1 / 2])
        cbar2.set_ticklabels(
            [
                "-1",
                "0",
                "1",
            ]
        )

        # flag_no_obs = pred[:, 3] < -100
        flag_no_obs = pred[:, 0] * 0 > 1

        # Remove estimation when  predicted no observation
        bot[flag_no_obs, 1] = 0
        top[flag_no_obs, 1] = 0

        x = times[mask_time] / (time_step)
        dates = get_regular_step_scale(
            shift,
            ((x[-1] - x[0]) + 1) * shift,
            time_offset,
            date_init=date_init,
        )

        dates_w = step_to_date(
            x[0],
            shift,
            int(shift / 5),
            ((x[-1] - x[0]) + 1) * shift,
            date_init=date_init,
        )

        plt.scatter(
            dates[np.invert(flag_no_obs)[mask_time]],
            pred[mask_time & np.invert(flag_no_obs), 0],
            s=5,
            color="gold",
            alpha=0.3,
            edgecolors="olive",
            zorder=0,
        )

        plt.scatter(
            dates[flag_no_obs[mask_time]],
            pred[flag_no_obs & mask_time, 0],
            s=5,
            color="purple",
            zorder=10,
        )

        plt.plot(dates, pred[mask_time, 0], lw=2, markersize=0, alpha=0.1, color="gold")

        y1 = bot[mask_time, 0]  # IC_bot
        y2 = top[mask_time, 0]  # IC_top

        plt.fill_between(
            x=dates_w,
            y1=np.repeat(y1, 5).astype(float),
            y2=np.repeat(y2, 5).astype(float),
            alpha=0.15,
            fc="yellow",
            ec="None",
        )
        plt.plot(
            dates_w,
            np.repeat(y1, 5).astype(float),
            lw=1,
            ls=":",
            markersize=0,
            zorder=20,
            alpha=1,
            color="gold",
            label="mean_boundary",
        )

        plt.plot(
            dates_w,
            np.repeat(y2, 5).astype(float),
            lw=1,
            ls=":",
            zorder=20,
            markersize=0,
            alpha=1,
            color="gold",
        )

        # Plot Std_range
        # IC_bot = -2*val_max_std
        print(top[mask_time, 1].mean())
        y1 = y[mask_time, 0] - 2 * (top[mask_time, 1])
        # IC_top = *2*val_max_std
        y2 = y[mask_time, 0] + 2 * (top[mask_time, 1])

        plt.fill_between(
            x=dates_w,
            y1=np.repeat(y1, 5).astype(float),
            y2=np.repeat(y2, 5).astype(float),
            alpha=0.2,
            fc="green",
            ec="None",
            label="Std-dev_range",
        )

        # Plot Min/Max boundary
        y1 = y[mask_time, 0] - top[mask_time, 2]  # Ext estimation on bot
        y2 = y[mask_time, 0] + top[mask_time, 2]  # Ext estimation on top
        plt.plot(
            dates_w,
            np.repeat(y1, 5),
            lw=1,
            ls="--",
            color="orange",
        )

        plt.plot(
            dates_w,
            np.repeat(y2, 5),
            lw=1,
            ls="--",
            color="orange",
            label="Extremum boundary",
        )

        # Plot anom_area
        x = times_s_anom[mask_time_s_anom] / (time_step)
        dates_anom = step_to_date(
            x[0],
            shift,
            int(shift / 5),
            ((x[-1] - x[0]) + 1) * shift,
            date_init=date_init,
        )

        y1 = np.ones(len(x)) * min(y1)
        y2 = np.ones(len(x)) * max(y2)

        flag_anom = (
            np.max(
                np.abs(
                    S_anom_channels[mask_time_s_anom][
                        :, sensor_id * n_chan + np.arange(0, n_chan)
                    ]
                ),
                axis=1,
            )
            > 1
        )

        flag_anom = flag_anom | (S_anom_sources[mask_time_s_anom][:, sensor_id] > 1)

        plt.fill_between(
            x=dates_anom,
            y1=np.repeat(y1, 5).astype(float),
            y2=np.repeat(y2, 5).astype(float),
            where=expand_flag(np.repeat(flag_anom, 5)),
            alpha=0.1,
            fc="b",
            color="red",
            lw=1,
        )

        bot_env = y[mask_time, 0] - 2 * (bot[mask_time, 1])
        top_env = y[mask_time, 0] + 2 * (top[mask_time, 1])
        inter = (top_env - bot_env).max() * 0.2
        min_th = bot_env.min() - inter
        max_th = top_env.max() + inter
        min_emp = serie["data"].values[mask_time_serie].min() - 2 * inter
        max_emp = serie["data"].values[mask_time_serie].max() + 2 * inter
        plt.ylim(min(min_th, min_emp), max(max_th, max_emp))
        plt.xlim(
            step_to_date(x_min / time_step, shift, date_init=date_init),
            step_to_date(x_max / time_step, shift, date_init=date_init),
        )
        ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%d/%m %Hh"))
        plt.legend(
            fontsize=15 * figsize[1] / 7, ncol=2, loc=0, title=sources_name[raw_id]
        )
        plt.xticks(rotation=5)
    plt.tight_layout()
    plt.savefig("cur.svg")
    plt.show()
