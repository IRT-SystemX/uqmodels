import numpy as np
import pytest
import uqmodels.data_generation.Gen_two_dimension_uncertainty_data as gen_twoD
import uqmodels.data_generation.Gen_multivariate_normal_data as gen_mult
import uqmodels.data_generation.Gen_basic_times_series as gen_basic


def test_gen_2D_UQ_data():
    dict_data = gen_twoD.generate_default()
 
def test_gen_mult_data():
    dict_data = gen_mult.generate_default()

def test_gen_basic_data():
    dict_data = gen_basic.generate_default()
    