import os
from copy import deepcopy
from typing import Dict, Any, List, Tuple, Optional
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import callbacks
from uqmodels.modelization.TF_estimator.base.train_config import add_callback

class UniversalKLScheduler(tf.keras.callbacks.Callback):
    """
    Scheduler unifié pour ajuster dynamiquement le poids KL d'un VAE.
    Stratégies : 'linear', 'cyclic', 'free_bits', 'tf_schedule'
    
    necessite :
      self.beta = tf.Variable(0.0,trainable=False) (all stratégie) et 
   	 self.free_bits_lambda = tf.Variable(0.2,trainable=False)
    
    """

    def __init__(self,
                 strategy="linear",
                 max_beta=1.0,
                 warmup_epochs=50,
                 cycle_length=20,
                 free_bits_lambda=0.2,
                 tf_steps=5000,
                 global_step=None):

        super().__init__()

        self.strategy = strategy
        self.max_beta = max_beta
        self.warmup_epochs = warmup_epochs
        self.cycle_length = cycle_length
        self.free_bits_lambda = free_bits_lambda
        self.tf_steps = tf_steps

        # pour la stratégie tf_schedule
        self.global_step = global_step or tf.Variable(0, trainable=False)

    def on_epoch_begin(self, epoch, logs=None):
        """Met à jour model.beta et éventuellement free bits"""
        # -------- STRATEGIE 1 : LINEAR ----------
        if self.strategy == "linear":
            epoch = epoch - self.no_beta_step
            if(epoch<0):
                beta=0
            else:
                beta = min(self.max_beta, self.max_beta * (epoch / self.warmup_epochs))

        # -------- STRATEGIE 2 : CYCLIC ----------
        elif self.strategy == "cyclic":
            cycle_pos = (epoch % self.cycle_length) / self.cycle_length
            beta = cycle_pos * self.max_beta

        # -------- STRATEGIE 3 : FREE BITS ---------
        elif self.strategy == "free_bits":
            beta = self.max_beta
            if hasattr(self.model, "free_bits_lambda"):
                self.model.free_bits_lambda.assign(self.free_bits_lambda)

        # -------- STRATEGIE 4 : TF SCHEDULE -------
        elif self.strategy == "tf_schedule":
            beta = tf.minimum(
                1.0,
                tf.cast(self.global_step, tf.float32) / self.tf_steps
            ).numpy()
            self.global_step.assign_add(1)

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        # appliquer beta au modèle
        self.model.beta.assign(beta)

        print(f"[KL Scheduler] Strategy={self.strategy} | Beta={beta:.4f}")


    def on_train_batch_end(self, batch, logs=None):
        if self.strategy == "tf_schedule":
            self.global_step.assign_add(1)

DEFAULT_CALLBACKS_CONFIG = {"UniversalKLScheduler":{ 
    "enabled": False,    
    "class": UniversalKLScheduler,    
    "params": {"strategy": "linear",
               "max_beta": 0.1,
               "no_beta_step":30,
               "warmup_epochs": 200,
               "cycle_length": 20,
               "free_bits_lambda": 0.2,
               "tf_steps": 5000,
               "global_step": None}}}

def add_vae_callbacks(callbacks_list,kl_weight=0.05):
    callback_class = UniversalKLScheduler
    callback_params = {"strategy": "linear",
                       "max_beta": kl_weight,
                       "no_beta_step":30,
                       "warmup_epochs": 100}
    callbacks_list = add_callback(callbacks_list,
                                  callback_class=callback_class,    
                                  callback_params=callback_params)
    return(callbacks_list)