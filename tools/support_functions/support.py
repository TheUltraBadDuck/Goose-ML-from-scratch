import numpy as np



def set_colours(sample_count, colour_val):
    return np.full(sample_count, colour_val)
    

def get_errors(feature_var_errors, shape):
    if feature_var_errors is None:
        feature_var_errors = np.full(shape[1], 1.0, dtype=float)

    error = np.zeros(shape, dtype=float)
    for i, e in enumerate(feature_var_errors):
        error[:, i] = np.random.normal(0, e, shape[0])

    return error


def get_zero_scale(shape):
    return np.zeros(shape, dtype=float)


def get_fill_value(shape, value):
    return np.full(shape, value, dtype=float)


def get_rotation_matrix(angle):
    return np.array([[np.cos(angle * np.pi / 180), np.sin(angle * np.pi / 180)],
                     [-np.sin(angle * np.pi / 180), np.cos(angle * np.pi / 180)]])


