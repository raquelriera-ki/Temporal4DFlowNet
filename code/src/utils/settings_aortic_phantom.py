import numpy as np 

AORTIC_PHANTOM_SETTINGS = {
        'mc1_4DFlowWIP_Ao_v120_2.5': {
            't_frames_diastole_systole': [5, 12], # TODO update
            'order_normal': [2, 0, 1],
            'factor_plane_normal': [1, 1, -1],
            'idxs_nonflow_area_ascending': [np.index_exp[:, 50:, :]],
            'idxs_nonflow_area_descending': [np.index_exp[:, :50, :]], 
            'thickness_ascending': 3,
            'thickness_descending': 3,
            'idx_anim_slice': np.index_exp[:, :, 20],
            'inlet':{
                'origin': [133.40354131, 98.6511, 42.11284],
                'normal': [1, 0., 0.],
                'aortic_side': 'ascending',
            },
            'outlet':{
                'origin': [333.92501, 173.7371, 52.670234],
                'normal': [1., 0., 0.],
                'aortic_side': 'descending',
            },
            'AAo':{
                'origin': [115.858067, 99.44150, 42.93138], 
                'normal': [-0.9566418, 0.281483641, -0.0748552],
                'aortic_side': 'ascending'
            },
            'BCT':{
                'origin': [89.589735, 109.666033, 42.47488],
                'normal': [-0.804879482095, 0.589382, 0.06926],
                'aortic_side': 'ascending'
            },
            'LSA':{
                'origin': [83.8472978261, 158.148, 52.521],
                'normal': [0.200723, 0.979639, 0.008],
                'aortic_side': 'descending'
            },
            'DAo':{
                'origin': [170.215, 177.725, 45.8437 ],
                'normal': [1., 0., 0.],
                'aortic_side': 'descending'
            }
        }, 
        'mc2_4DFlowWIP_Ao_v120_2.5': {
        't_frames_diastole_systole': [5, 12], # TODO update
            'order_normal': [2, 0, 1],
            'factor_plane_normal': [1, 1, -1],
            'idxs_nonflow_area_ascending': [np.index_exp[:, 50:, :]],
            'idxs_nonflow_area_descending': [],#np.index_exp[:, :50, :]
            'thickness_ascending': 3,
            'thickness_descending': 3,
            'idx_anim_slice': np.index_exp[:, :, 20],
            'inlet':{
                'origin': [133.40354131, 98.6511, 42.11284],
                'normal': [1, 0., 0.],
                'aortic_side': 'ascending',
            },
            'outlet':{
                'origin': [333.92501, 173.7371, 52.670234],
                'normal': [1., 0., 0.],
                'aortic_side': 'descending',
            },
            'AAo':{
                'origin': [115.858067, 99.44150, 42.93138], 
                'normal': [-0.9566418, 0.281483641, -0.0748552],
                'aortic_side': 'ascending'
            },
            'BCT':{
                'origin': [89.589735, 109.666033, 42.47488],
                'normal': [-0.804879482095, 0.589382, 0.06926],
                'aortic_side': 'ascending'
            },
            'LSA':{
                'origin': [83.8472978261, 158.148, 52.521],
                'normal': [0.200723, 0.979639, 0.008],
                'aortic_side': 'descending'
            },
            'DAo':{
                'origin': [170.215, 177.725, 45.8437 ],
                'normal': [1., 0., 0.],
                'aortic_side': 'descending'
            }
        }, 
        'mr_4DFlowWIP_Ao_v120_2.5': {
            't_frames_diastole_systole': [5, 12], # TODO update
            'order_normal': [2, 0, 1],
            'factor_plane_normal': [1, 1, -1],
            'idxs_nonflow_area_ascending': [np.index_exp[:, 50:, :]],
            'idxs_nonflow_area_descending': [],#np.index_exp[:, :50, :]
            'thickness_ascending': 3,
            'thickness_descending': 3,
            'idx_anim_slice': np.index_exp[:, :, 20],
            'inlet':{
                'origin': [133.40354131, 98.6511, 42.11284],
                'normal': [1, 0., 0.],
                'aortic_side': 'ascending',
            },
            'outlet':{
                'origin': [333.92501, 173.7371, 52.670234],
                'normal': [1., 0., 0.],
                'aortic_side': 'descending',
            },
            'AAo':{
                'origin': [115.858067, 99.44150, 42.93138], 
                'normal': [-0.9566418, 0.281483641, -0.0748552],
                'aortic_side': 'ascending'
            },
            'BCT':{
                'origin': [89.589735, 109.666033, 42.47488],
                'normal': [-0.804879482095, 0.589382, 0.06926],
                'aortic_side': 'ascending'
            },
            'LSA':{
                'origin': [83.8472978261, 158.148, 52.521],
                'normal': [0.200723, 0.979639, 0.008],
                'aortic_side': 'descending'
            },
            'DAo':{
                'origin': [170.215, 177.725, 45.8437 ],
                'normal': [1., 0., 0.],
                'aortic_side': 'descending'
            }
    }
}