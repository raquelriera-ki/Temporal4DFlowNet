import numpy as np
import time
import os
from matplotlib import pyplot as plt
import h5py
from collections import defaultdict
import argparse
from Network.PatchGenerator import PatchGenerator
from utils import prediction_utils
from utils.evaluate_utils import *

# from utils.vtkwriter_per_dir import uvw_mask_to_vtk


plt.rcParams['figure.figsize'] = [10, 8]



def load_vel_data(gt_filepath, lr_filepath, pred_filepath,  vel_colnames = ['u', 'v', 'w'],res_colnames = ['u_combined', 'v_combined', 'w_combined'], threshold = 0.5, offset = 0, factor = 2):
    

    gt = {}
    lr = {}
    pred = {}

    with h5py.File(pred_filepath, mode = 'r' ) as h5_pred:
        with h5py.File(gt_filepath, mode = 'r' ) as h5_gt:
            with h5py.File(lr_filepath, mode = 'r' ) as h5_lr:
                
                # load mask
                gt["mask"] = np.asarray(h5_gt["mask"]).squeeze()
                gt["mask"][np.where(gt["mask"] >= threshold)] = 1
                gt["mask"][np.where(gt["mask"] <  threshold)] = 0

                if len(gt['mask'].shape) == 3 : # check for dynamical mask, otherwise create one
                    gt["mask"] = create_dynamic_mask(gt["mask"], h5_gt['u'].shape[0])
                
                # check for LR dimension, two options: 
                # 1. LR has same temporal resolution as HR: downsampling is done here (on the fly)
                # 2. LR is already downsampled: only load dataset
                if h5_gt[vel_colnames[0]].shape[0] == h5_lr[vel_colnames[0]].shape[0]:
                    downsample_lr = True
                else:
                    downsample_lr = False

                if 'mask' in h5_lr.keys():
                    print('Load mask from low resolution file')
                    lr['mask'] = np.asarray(h5_lr['mask']).squeeze()
                else:
                    print('Create LR mask from HR mask')
                    lr['mask'] = gt["mask"][offset::factor, :, :, :].copy()

                # load velocity fields
                for vel, r_vel in zip(vel_colnames, res_colnames):
                    
                    gt[vel] = np.asarray(h5_gt[vel]).squeeze()
                    pred[vel] = np.asarray(h5_pred[r_vel]).squeeze()
                    #TODO remake
                    # if pred[vel].shape[0] != gt[vel].shape[0]:
                    #     print('Cut prediction to GT shape, from ', pred[vel].shape, 'to', gt[vel].shape)
                    #     # t_gt = gt[vel].shape[0]
                    #     pred[vel] = pred[vel][:gt[vel].shape[0], :, :, :]
                    if downsample_lr:
                        lr[vel] = np.asarray(h5_lr[vel])[offset::factor, :, :, :]
                    else:
                        lr[vel] = np.asarray(h5_lr[vel]).squeeze()  

                    print('Shapes', gt[vel].shape, pred[vel].shape, lr[vel].shape, gt['mask'].shape)
                    # take away background outside mask
                    pred[f'{vel}_fluid'] =np.multiply(pred[vel], gt["mask"])
                    lr[f'{vel}_fluid'] =  np.multiply(lr[vel], lr['mask'])
                    gt[f'{vel}_fluid'] =  np.multiply(gt[vel], gt["mask"])

                    # Check that shapes match
                    # assert gt[vel].shape == pred[vel].shape, f"Shape mismatch HR/SR: {gt[vel].shape} != {pred[vel].shape}"
                    
                # include speed calculations
                gt['speed']   = np.sqrt(gt["u"]**2 + gt["v"]**2 + gt["w"]**2)
                lr['speed']   = np.sqrt(lr["u"]**2 + lr["v"]**2 + lr["w"]**2)
                pred['speed'] = np.sqrt(pred["u"]**2 + pred["v"]**2 + pred["w"]**2)

                gt['speed_fluid']   = np.multiply(gt['speed'], gt["mask"])
                lr['speed_fluid']   = np.multiply(lr['speed'], lr['mask'])
                pred['speed_fluid'] = np.multiply(pred['speed'], gt["mask"])
    
    return gt, lr, pred

def evaluate_model(pred, gt ):

    rel_error = calculate_relative_error_normalized(pred['u'], pred['v'], pred['w'], gt['u'], gt['v'], gt['w'], gt['mask'])
    cos_sim = cosine_similarity(pred['u'], pred['v'], pred['w'], gt['u'], gt['v'], gt['w'], gt['mask'])
    k = np.zeros((gt['mask'].shape[0], 3))
    r2 = np.zeros((gt['mask'].shape[0], 3))
    mae = np.zeros((gt['mask'].shape[0], 3))
    rmse = np.zeros((gt['mask'].shape[0], 3))
    for i, vel in enumerate(['u', 'v', 'w']):
        rmse[:, i] = calculate_rmse(pred[vel], gt[vel], gt['mask'])
        mae[:, i] = np.mean(np.abs(pred[vel] - gt[vel]), axis=(1, 2, 3), where= gt['mask'] != 0)
        k[:, i], r2[:, i] = calculate_k_R2_timeseries(pred, gt, gt['mask'])

    dict_metrics = {
        'rmse_u': np.mean(rmse[:, 0]),
        'rmse_v': np.mean(rmse[:, 1]),
        'rmse_w': np.mean(rmse[:, 2]),
        'mae_u': np.mean(mae[:, 0]),
        'mae_v': np.mean(mae[:, 1]),
        'mae_w': np.mean(mae[:, 2]),
        're': rel_error,
        'cos_sim': cos_sim,
        'k_u': k[:, 0],
        'k_v': k[:, 1],
        'k_w': k[:, 2],
        'r2_u': r2[:, 0],
        'r2_v': r2[:, 1],
        'r2_w': r2[:, 2]
    }

    return dict_metrics

def plot_RE_comparison(results_all):
    for nn, data in results_all.items():
        plt.plot(data['RE'].flatten(), label=nn)
    plt.legend()
    plt.title('Relative Error Comparison')
    plt.savefig(f"{eval_dir}/RE_comparison.png")


def generate_summary_table(results_all, t_peak):
    summary = []
    for nn, data in results_all.items():
        row = {
            'Network': nn,
            # Mean values
            'RMSE u (mean)': np.mean(data['rmse_u']),
            'RMSE v (mean)': np.mean(data['rmse_v']),
            'RMSE w (mean)': np.mean(data['rmse_w']),
            'MAE u (mean)': np.mean(data['mae_u']),
            'MAE v (mean)': np.mean(data['mae_v']),
            'MAE w (mean)': np.mean(data['mae_w']),
            'RE (mean)': np.mean(data['re']),
            'CosSim (mean)': np.mean(data['cos_sim']),
            'k u (mean)': np.mean(data['k_u']),
            'k v (mean)': np.mean(data['k_v']),
            'k w (mean)': np.mean(data['k_w']),
            'R² u (mean)': np.mean(data['r2_u']),
            'R² v (mean)': np.mean(data['r2_v']),
            'R² w (mean)': np.mean(data['r2_w']),

            # Peak frame values
            'RMSE u (peak)': data['rmse_u'][t_peak],
            'RMSE v (peak)': data['rmse_v'][t_peak],
            'RMSE w (peak)': data['rmse_w'][t_peak],
            'MAE u (peak)': data['mae_u'][t_peak],
            'MAE v (peak)': data['mae_v'][t_peak],
            'MAE w (peak)': data['mae_w'][t_peak],
            'RE (peak)': data['re'][t_peak],
            'CosSim (peak)': data['cos_sim'][t_peak],
            'k u (peak)': data['k_u'][t_peak],
            'k v (peak)': data['k_v'][t_peak],
            'k w (peak)': data['k_w'][t_peak],
            'R² u (peak)': data['r2_u'][t_peak],
            'R² v (peak)': data['r2_v'][t_peak],
            'R² w (peak)': data['r2_w'][t_peak],
        }
        summary.append(row)

    df = pd.DataFrame(summary)
    df.to_csv(f"{eval_dir}/network_comparison_summary.csv", index=False)
    return df


if __name__ == "__main__":

    network_names = ['20241018-1552', '20241018-1553', '20241018-1554']

    data_model= '4'
    step = 2
    ups_factor = 2

    # settings
    vel_colnames=['u', 'v', 'w']
    t_range_in_ms = True
    exclude_tbounds = False
    use_peak_systole = False
    range_systole = np.arange(0, 25)

    eval_dir = f'results/comparison/angular_term'
    os.makedirs(eval_dir, exist_ok=True)

    
    set_name = 'Test'   
    add_description = '_VENC3'
    # directories
    # filenames
    data_dir = 'Temporal4DFlowNet/data/CARDIAC'
    gt_filename = f'M{data_model}_2mm_step{step}_cs_invivoP02_hr.h5'
    lr_filename = f'M{data_model}_2mm_step{step}_cs_invivoP02_lr{add_description}.h5'
    gt_filepath = '{}/{}'.format(data_dir, gt_filename)
    lr_filepath = '{}/{}'.format(data_dir, lr_filename)
    gt, lr, _ = load_vel_data(gt_filepath, lr_filepath, pred_filepath=None, vel_colnames=vel_colnames)

    # calculate velocity values in 1% and 99% quantile for plotting 
    min_v = {}
    max_v = {}
    for vel in vel_colnames:
        min_v[vel] = np.quantile(gt[vel][np.where(gt['mask'] !=0)].flatten(), 0.01)
        max_v[vel] = np.quantile(gt[vel][np.where(gt['mask'] !=0)].flatten(), 0.99)

    min_v_global = np.min([min_v[vel] for vel in vel_colnames])
    max_v_global = np.max([max_v[vel] for vel in vel_colnames])
   
    N_frames = gt['u'].shape[0]


    hr_mean_speed = calculate_mean_speed(gt['u'], gt['v'] , gt['v'] , gt['mask'] )
    T_peak_flow_frame = np.argmax(hr_mean_speed)
    synthesized_peak_flow_frame = T_peak_flow_frame.copy() 

    # define peak flow frame
    if use_peak_systole:
        synthesized_peak_flow_frame = np.argmax(hr_mean_speed[range_systole])
        print('Restricting peak flow frame to systole!')

    if synthesized_peak_flow_frame % 2 == 0: 
        if hr_mean_speed[synthesized_peak_flow_frame-1] > hr_mean_speed[synthesized_peak_flow_frame+1]:
            synthesized_peak_flow_frame -=1
        else:
            synthesized_peak_flow_frame +=1
    
    print('Synthesized peak flow frame:', synthesized_peak_flow_frame, 'Peak flow frame:', T_peak_flow_frame)
   
    results_all = {}
    for nn_name in network_names:
        print(f'Evaluating model {nn_name}...')

        # load prediction
        pred_dir = f'Temporal4DFlowNet/results/Temporal4DFlowNet_{nn_name}'
        pred_filename = f'{set_name}set_result_model{data_model}_2mm_step{step}_{nn_name[-4::]}_temporal{add_description}.h5'
        pred_filepath = '{}/{}'.format(pred_dir, pred_filename)

        _,_, pred = load_vel_data(gt_filepath, lr_filepath, pred_filepath, vel_colnames = vel_colnames)
        
        # check that dimension fits
        assert(gt["u"].shape == pred["u"].shape)  ,str(pred["u"].shape) + str(gt["u"].shape) # dimensions of HR and SR need to be the same
        assert(gt["u"].shape[1::] == lr["u"].shape[1::])    ,str(lr["u"].shape) + str(gt["u"].shape) # spatial dimensions need to be the same
        
        # evaluate model
        results_all[nn_name] = evaluate_model(pred, gt)

    # plot RE
    plot_RE_comparison(results_all)
    # generate summary table
    summary_table = generate_summary_table(results_all, synthesized_peak_flow_frame)

    # make qualitative plot


 



        