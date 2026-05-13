import numpy as np
def feature_scaling(data):
    data_min = np.min(data, axis=0)
    data_max = np.max(data, axis=0)
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)

    standardized_data = np.round(((data - mean)/std), 4).tolist()
    normalized_data = np.round(((data-data_min)/(data_max-data_min)), 4).tolist()
    return standardized_data, normalized_data