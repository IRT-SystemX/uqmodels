import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import callbacks


def build_MSE_loss(split=2, metric=False, var=1):
    def MSE_loss(true, pred):
        if true.shape[-1] == pred.shape[-1]:
            mu = pred
        else:
            pred_split = tf.split(pred, split, -1)
            mu = pred_split[0]

        if metric:  # Only consider 1 prediction (t+1)
            if len(mu.shape) == 3:
                loss = K.pow(true[:, 0] - mu[:, 0], 2)

            elif (
                len(mu.shape) > 3
            ):  # Only consider 1 prediction of last (futur) segment = t+1
                loss = K.pow(true[:, -1, 0] - mu[:, -1, 0], 2)

            else:
                loss = K.pow(true - mu, 2)

            reduce_loss = K.sqrt(K.mean(loss)) * np.sqrt(var)
        else:
            loss = K.pow(true - mu, 2)
            reduce_loss = K.mean(loss, axis=0)
        return reduce_loss

    return MSE_loss


def build_CCE_loss():
    def CCE_loss(true, pred):
        loss = -K.mean(K.log(true * pred))
        reduce_loss = K.sum(loss, axis=0)
        return reduce_loss

    return CCE_loss


@tf.keras.utils.register_keras_serializable(
    package="UQModels_loss", name="BNN_loss_gaussian"
)
def BNN_loss_gaussian(true, pred, alpha=0.95):
    mu = pred
    mu, logvar = tf.split(pred, 2, -1)
    loss = K.pow(true - mu, 2) * K.exp(-logvar) + alpha * logvar
    reduce_loss = K.mean(loss, axis=0)
    return reduce_loss


@tf.keras.utils.register_keras_serializable(
    package="UQModels_loss", name="BNN_loss_edl"
)
def BNN_loss_edl(true, pred, alpha=0.95):
    gamma, vu, alpha_edl, beta = tf.split(pred, 4, -1)
    mu = gamma
    logvar = K.log(beta / (alpha_edl - 1))
    loss = K.pow(true - mu, 2) * K.exp(-logvar) + alpha * logvar
    reduce_loss = K.mean(loss, axis=0)
    return reduce_loss


@tf.keras.utils.register_keras_serializable(
    package="UQModels_loss", name="BNN_metric_gaussian"
)
def BNN_metric_gaussian(true, pred):
    mu = pred
    mu, logvar = tf.split(pred, 2, -1)
    if len(mu.shape) == 3:
        mu_ = mu[:, 0]
        logvar_ = logvar[:, 0]
        true_ = true[:, 0]

    elif len(mu.shape) > 3:
        mu_ = mu[:, -1, 0]
        logvar_ = logvar[:, -1, 0]
        true_ = true[:, -1, 0]
    else:
        mu_ = mu
        logvar_ = logvar
        true_ = true

    loss = K.greater(2 * K.sqrt(K.exp(logvar_)), K.abs(true_ - mu_))
    reduce_loss = K.mean(loss)
    return reduce_loss


@tf.keras.utils.register_keras_serializable(
    package="UQModels_loss", name="BNN_metric_edl"
)
def BNN_metric_edl(true, pred):
    gamma, vu, alpha_edl, beta = tf.split(pred, 4, -1)
    mu = gamma
    logvar = K.log(beta / (alpha_edl - 1))

    if len(mu.shape) == 3:
        mu_ = mu[:, 0]
        logvar_ = logvar[:, 0]
        true_ = true[:, 0]

    elif len(mu.shape) > 3:
        mu_ = mu[:, -1, 0]
        logvar_ = logvar[:, -1, 0]
        true_ = true[:, -1, 0]
    else:
        mu_ = mu
        logvar_ = logvar
        true_ = true

    loss = K.greater(2 * K.sqrt(K.exp(logvar_)), K.abs(true_ - mu_))
    reduce_loss = K.mean(loss)
    return reduce_loss


def build_BNN_loss(alpha=0.95, metric=False, type_output="MC_Dropout"):
    def BNN_loss(true, pred):
        mu = pred
        if (type_output == "MC_Dropout") or (type_output == "Deep_ensemble"):
            mu, logvar = tf.split(pred, 2, -1)

        elif type_output == "EDL":
            gamma, vu, alpha_edl, beta = tf.split(pred, 4, -1)
            mu = gamma
            logvar = K.log(beta / (alpha_edl - 1))
        else:
            mu = pred
            np.array([[1], [1]])

        if metric:
            if len(mu.shape) == 3:
                mu_ = mu[:, 0]
                logvar_ = logvar[:, 0]
                true_ = true[:, 0]

            if len(mu.shape) > 3:
                mu_ = mu[:, -1, 0]
                logvar_ = logvar[:, -1, 0]
                true_ = true[:, -1, 0]
            else:
                mu_ = mu
                logvar_ = logvar
                true_ = true

            loss = K.greater(2 * K.sqrt(K.exp(logvar_)), K.abs(true_ - mu_))
            reduce_loss = K.mean(loss)

        else:
            loss = K.pow(true - mu, 2) * K.exp(-logvar) + alpha * logvar
            reduce_loss = K.mean(loss, axis=0)
        return reduce_loss

    return BNN_loss


# EDL LOSS
# https://github.com/aamini/evidential-deep-learning/blob/main/evidential_deep_learning/layers/dense.py


def build_EDL_loss(coeff_reg=0.95, coeff_var_pen=1):
    def NIG_NLL(y, gamma, v, alpha, beta, reduce=True):
        twoBlambda = 2 * beta * (coeff_var_pen + v)
        nll = (
            0.5 * tf.math.log(np.pi / v)
            - alpha * tf.math.log(twoBlambda)
            + (alpha + 0.5) * tf.math.log(v * (y - gamma) ** 2 + twoBlambda)
            + tf.math.lgamma(alpha)
            - tf.math.lgamma(alpha + 0.5)
        )
        return tf.reduce_mean(nll) if reduce else nll

    def KL_NIG(mu1, v1, a1, b1, mu2, v2, a2, b2):
        KL = (
            0.5 * (a1 - 1) / b1 * (v2 * tf.square(mu2 - mu1))
            + 0.5 * v2 / v1
            - 0.5 * tf.math.log(tf.abs(v2) / tf.abs(v1))
            - 0.5
            + a2 * tf.math.log(b1 / b2)
            - (tf.math.lgamma(a1) - tf.math.lgamma(a2))
            + (a1 - a2) * tf.math.digamma(a1)
            - (b1 - b2) * a1 / b1
        )
        return KL

    def NIG_Reg(y, gamma, v, alpha, beta, omega=0.01, reduce=True, kl=False):
        # error = tf.stop_gradient(tf.abs(y-gamma))
        error = tf.abs(y - gamma)
        if kl:
            kl = KL_NIG(gamma, v, alpha, beta, gamma, omega, 1 + omega, beta)
            reg = error * kl
        else:
            evi = 2 * v + (alpha)
            reg = error * evi

        return tf.reduce_mean(reg) if reduce else reg

    def EvidentialRegressionLoss(y_true, pred):
        gamma, v, alpha, beta = tf.split(pred, 4, axis=-1)
        loss_NLL = NIG_NLL(y_true, gamma, v, alpha, beta, reduce=False)
        loss_Reg = NIG_Reg(y_true, gamma, v, alpha, beta, reduce=False)
        reduce_loss = K.mean(loss_NLL + coeff_reg * loss_Reg, axis=0)
        return reduce_loss

    return EvidentialRegressionLoss


# Call back procedure
def default_callbacks(
    min_delta=0.0001,
    earlystop_patience=60,
    reducelr_patience=30,
    reducelr_factor=0.3,
    reduce_lr_min_lr=0.000001,
    verbose=0,
):
    call2 = callbacks.TerminateOnNaN()
    call0 = callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=min_delta,
        patience=earlystop_patience,  # 60/ 30
        verbose=verbose,
        mode="min",
        restore_best_weights=False,
    )
    call1 = callbacks.ReduceLROnPlateau(
        monitor="loss",
        min_delta=min_delta,
        factor=reducelr_factor,
        patience=reducelr_patience,  # 10
        verbose=verbose,
        mode="min",
        cooldown=0,
        min_lr=reduce_lr_min_lr,  # 0.00001
    )
    return [call0, call1, call2]


# Basic Loss
# class InterruptingCallback(tf.keras.callbacks.Callback):
#  def on_epoch_begin(self, epoch, logs=None):
#    if epoch == 4:
#      raise RuntimeError('Interrupting!')
# callback = tf.keras.callbacks.BackupAndRestore(backup_dir="/tmp/backup")
# model = tf.keras.models.Sequential([tf.keras.layers.Dense(10)])
# model.compile(tf.keras.optimizers.SGD(), loss='mse')
# try:
#  model.fit(np.arange(100).reshape(5, 20), np.zeros(5), epochs=10,
#            batch_size=1, callbacks=[callback, InterruptingCallback()],
#            verbose=0)
# except:
#  pass
# history = model.fit(np.arange(100).reshape(5, 20), np.zeros(5),
#                    epochs=10, batch_size=1, callbacks=[callback],
#                    verbose=0)
# Only 6 more epochs are run, since first trainning got interrupted at
# zero-indexed epoch 4, second training will continue from 4 to 9.
