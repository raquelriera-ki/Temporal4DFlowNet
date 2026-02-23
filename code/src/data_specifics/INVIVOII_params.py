import numpy as np

volunteer_plane_normal_ascending_orig = {
    'v3_origin': [198.75, 166.096, 90.2740], 
    'v3_normal': [-0.9993, -0.03153, 0.019410], 
    'v4_origin': [182.09, 147.1993, 88.808613], 
    'v4_normal': [-0.98131, -0.176387, 0.076911],
    'v5_origin': [184.8120, 141.065, 82.21268],
    'v5_normal': [-0.92945, -0.07347, 0.3615],
    'v6_origin': [198.75, 165.3485, 85.68],
    'v6_normal': [0.98590, 0.055869, -0.15771],
    'v7_origin': [175.760, 167.830, 84.6532],
    'v7_normal': [-0.9899, -0.11857, 0.07666],
}
volunteer_plane_normal_descending_orig = {
    'v3_origin': [212.00, 222.761, 112.845],
    'v3_normal': [0.71023, -0.074448, -0.7000],
    'v4_origin': [202.46253, 201.068, 113.0036],
    'v4_normal': [0.7341784, 0.0059383, -0.67893],
    'v5_origin': [201.3359, 205.441, 116.173],
    'v5_normal': [0.658955, -0.14523, -0.738026],
    'v6_origin': [199.579, 216.9952277, 88.363],
    'v6_normal': [0.710689, -0.151932, -0.68690],
    'v7_origin': [204.60304, 216.851, 92.217],
    'v7_normal': [0.74243, -0.112015, -0.66048],
}

volunteer_plot_settings_orig = {
    'v3': {
        'order_normal': [2, 1, 0],
        'factor_plane_normal': [1, 1, -1],
        'idxs_nonflow_area_ascending': [np.index_exp[:, 80:, :]],
        'idxs_nonflow_area_descending': [np.index_exp[:, :80, :]],
        'thickness_ascending': 30,
        'thickness_descending': 2,
    }, 
    'v4': {
        'order_normal': [2, 1, 0],
        'factor_plane_normal': [1, 1, -1],
        'idxs_nonflow_area_ascending': [np.index_exp[:, 70:, :]],
        'idxs_nonflow_area_descending': [np.index_exp[:, :70, :]],
        'thickness_ascending': 10,
        'thickness_descending': 2,
    }, 
    'v5': {
        'order_normal': [2, 1, 0],
        'factor_plane_normal': [1, 1, -1],
        'idxs_nonflow_area_ascending': [np.index_exp[:, 65:, :]],
        'idxs_nonflow_area_descending': [np.index_exp[:, :65, :]],
        'thickness_ascending': 2,
        'thickness_descending': 2,
    }, 
    'v6': {
        'order_normal': [2, 1, 0],
        'factor_plane_normal': [1, 1, -1],
        'idxs_nonflow_area_ascending': [np.index_exp[:, 73:, :]],
        'idxs_nonflow_area_descending': [np.index_exp[:, :73, :]],
        'thickness_ascending': 2,
        'thickness_descending': 2,
    }, 
    'v7': {
        'order_normal': [2, 1, 0],
        'factor_plane_normal': [1, 1, -1],
        'idxs_nonflow_area_ascending': [np.index_exp[:, 80:, :]],
        'idxs_nonflow_area_descending': [np.index_exp[:, :80, :]],
        'thickness_ascending': 30,
        'thickness_descending': 2,
    }, 
}

volunteer_plane_normal_ascending_transformed = {
    'v3_origin': [198.7, 166.096, 90.2740], 
    'v3_normal': [0.9993, -0.03153, 0.019410], 
    'v4_origin': [215.41, 147.1993, 88.808613], 
    'v4_normal': [0.98131, -0.176387, 0.076911],
    'v5_origin': [212.688, 141.065, 82.21268],
    'v5_normal': [0.92945, -0.07347, 0.3615],
    'v6_origin': [198.75, 165.3485, 85.68],
    'v6_normal': [0.98590, 0.055869, -0.15771],
    'v7_origin': [221.74, 167.83, 84.6532],
    'v7_normal': [0.9899, -0.11857, 0.07666],
}
volunteer_plane_normal_descending_transformed = {
    'v3_origin': [185.5, 222.761, 112.845],
    'v3_normal': [-0.71023, -0.074448, -0.7000],
    'v4_origin': [195.03747, 201.068, 113.0036],
    'v4_normal': [-0.7341784, 0.0059383, -0.67893],
    'v5_origin': [196.1641, 205.441, 116.173],
    'v5_normal': [-0.658955, -0.14523, -0.738026],
    'v6_origin': [197.921, 216.9952277, 88.363],
    'v6_normal': [-0.710689, -0.151932, -0.68690],
    'v7_origin': [192.89696, 216.851, 92.217],
    'v7_normal': [-0.74243, -0.112015, -0.66048],
}

volunteer_plot_settings_transformed = {
    'v3': {
        'order_normal': [0, 1, 2],
        'factor_plane_normal': [1, 1, 1],
        'idxs_nonflow_area_ascending': [np.index_exp[:, 80:, :]],
        'idxs_nonflow_area_descending': [np.index_exp[:, :80, :]],
        'thickness_ascending': 30,
        'thickness_descending': 2,
    }, 
    'v4': {
        'order_normal': [0, 1, 2],
        'factor_plane_normal': [1, 1, 1],
        'idxs_nonflow_area_ascending': [np.index_exp[:, 70:, :]],
        'idxs_nonflow_area_descending': [np.index_exp[:, :70, :]],
        'thickness_ascending': 10,
        'thickness_descending': 2,
    }, 
    'v5': {
        'order_normal': [0, 1, 2],
        'factor_plane_normal': [1, 1, 1],
        'idxs_nonflow_area_ascending': [np.index_exp[:, 65:, :]],
        'idxs_nonflow_area_descending': [np.index_exp[:, :65, :]],
        'thickness_ascending': 2,
        'thickness_descending': 2,
    }, 
    'v6': {
        'order_normal': [0, 1, 2],
        'factor_plane_normal': [1, 1, 1],
        'idxs_nonflow_area_ascending': [np.index_exp[:, 73:, :]],
        'idxs_nonflow_area_descending': [np.index_exp[:, :73, :]],
        'thickness_ascending': 2,
        'thickness_descending': 2,
    }, 
    'v7': {
        'order_normal': [0, 1, 2],
        'factor_plane_normal': [1, 1, 1],
        'idxs_nonflow_area_ascending': [np.index_exp[:, 80:, :]],
        'idxs_nonflow_area_descending': [np.index_exp[:, :80, :]],
        'thickness_ascending': 30,
        'thickness_descending': 2,
    }, 
}