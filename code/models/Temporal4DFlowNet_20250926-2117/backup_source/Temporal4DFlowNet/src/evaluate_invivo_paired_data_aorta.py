import numpy as np
import time
import h5py
import pandas as pd

import argparse
import matplotlib.pyplot as plt
from utils.evaluate_utils import *
from scipy.ndimage import map_coordinates
# from utils.vtkwriter_per_dir import vectors_and_mask_to_vtk
from mpl_toolkits.axes_grid1 import make_axes_locatable
from collections import defaultdict
from prepare_data.h5functions import save_to_h5
import matplotlib
import scipy
from utils.colors import *
import matplotlib.animation as animation
import matplotlib.cm as cm
from utils.settings_aortic_phantom import *
plt.rcParams['figure.figsize'] = [10, 8]

def generate_plane_coordinates(plane_origin, plane_normal, dx, shape, grid_res=50):
    """
    Generate a 2D sampling grid along a plane in a 3D volume.
    """
    normal = plane_normal / np.linalg.norm(plane_normal)
    ref_vector = np.array([1, 0, 0]) if abs(normal[0]) < 0.9 else np.array([0, 1, 0])
    u = np.cross(normal, ref_vector)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)

    lin = np.linspace(-shape[0] * dx[0] / 2, shape[0] * dx[0] / 2, grid_res)
    X, Y = np.meshgrid(lin, lin)

    plane_points = plane_origin[:, None, None] + u[:, None, None] * X + v[:, None, None] * Y
    dx = np.array(dx)
    dx = dx[:, None, None]
    print(f"Shape of plane points: {plane_points.shape}, {dx}")
    coords = plane_points / dx  # Convert to voxel indices
    print(f"Shape of plane points: {plane_points.shape}, shape of coords: {coords.shape}")
    print(f"Shape of X: {X.shape}, shape of Y: {Y.shape}")
    return coords, X, Y, plane_points[2]

def extract_velocity_on_plane(velocity, coords, mask):
    """
    Extract velocity values along a plane using interpolation.
    """
    T, _, _, _, _ = velocity.shape
    grid_res = coords.shape[1]
    
    velocity_plane = np.zeros((T, grid_res, grid_res))
    mask_plane = np.zeros((grid_res, grid_res))
    mean_velocity = np.zeros(T)
    
    for t in range(T):
        v_interp = np.zeros((3, grid_res, grid_res))
        for d in range(3):  
            v_interp[d] = map_coordinates(velocity[t, ..., d], coords, order=1, mode='nearest')

        velocity_mag = np.linalg.norm(v_interp, axis=0)
        
        # Mask projection (only need to compute once)
        if t == 0:
            mask_interp = map_coordinates(mask.astype(float), coords, order=1, mode='nearest')
            mask_plane = mask_interp > 0.5  

        velocity_plane[t] = velocity_mag
        mean_velocity[t] = np.mean(velocity_mag[mask_plane]) if np.any(mask_plane) else 0
    
    return velocity_plane, mask_plane, mean_velocity



def define_plane(cp_plane, plane_normal, size=100):
    plane_normal = plane_normal / np.linalg.norm(plane_normal)  # Normalize

    # Choose any vector not parallel to the normal
    if np.abs(plane_normal[0]) > np.abs(plane_normal[1]):
        tangent1 = np.array([-plane_normal[2], 0, plane_normal[0]])  # Cross with [0,1,0]
    else:
        tangent1 = np.array([0, plane_normal[2], -plane_normal[1]])  # Cross with [1,0,0]
    
    tangent1 /= np.linalg.norm(tangent1)  # Normalize
    tangent2 = np.cross(plane_normal, tangent1)  # Get second tangent vector

    # Create mesh grid along these tangent directions
    u = np.linspace(-size//2, size//2, size)
    v = np.linspace(-size//2, size//2, size)
    uu, vv = np.meshgrid(u, v)

    # Compute plane points
    plane_points = cp_plane[:, None, None] + tangent1[:, None, None] * uu + tangent2[:, None, None] * vv
    
    return plane_points


def plot_plane_flows(u_hr, v_hr, w_hr, u_lr, v_lr, w_lr, u_sr, v_sr, w_sr, 
                     cp_plane, plane_normal, idxs_nonflow_area, 
                     order_normal = [0, 1, 2], 
                     show_mask = True, save_as = None, 
                    thickness=2, factor_plane_normal = [1, 1, 1], lr_hb_duration=1, hr_hb_duration=1):

    print("Plot plane flow..")

    plane_normal /= np.linalg.norm(plane_normal)

    # calculate the plane
    d = -np.dot(cp_plane, plane_normal)
    xx, yy = np.meshgrid(np.arange(0, u_hr.shape[1]), np.arange(0, u_hr.shape[2]))
    print(f'xx.shape: {xx.shape}, yy.shape: {yy.shape}')
    zz = (-plane_normal[0] * xx - plane_normal[1] * yy - d) * 1. / plane_normal[2] # adapt to oder of normal ? 
    # if plane_normal[2] != 0:
    # # General case (normal has a z-component)
    #     zz = (-plane_normal[0] * xx - plane_normal[1] * yy - d) / plane_normal[2]
    
    # elif plane_normal[1] != 0:
    #     # Normal is in the XZ plane (e.g., [0,1,0]), solve for y instead
    #     yy = (-plane_normal[0] * xx - d) / plane_normal[1]
    #     zz = np.zeros_like(xx)  # Placeholder, z is arbitrary in this case

    # elif plane_normal[0] != 0:
    # # Normal is in the YZ plane (e.g., [1,0,0]), solve for x instead
    #     xx = (-plane_normal[1] * yy - d) / plane_normal[0]
    #     zz = np.zeros_like(xx)  # Placeholder, z is arbitrary in this case

    # plane_points = define_plane(cp_plane, plane_normal, size=100)
    # print(f'plane_points1.shape: {plane_points.shape}')
    # velocity = np.stack([u_hr, v_hr, w_hr], axis=-1)
    # coords, X, Y, plane_points = generate_plane_coordinates(cp_plane, plane_normal, [1, 1, 1], u_hr.shape, grid_res=50)
    # velocity_plane, mask_plane, mean_velocity = extract_velocity_on_plane(velocity, coords, mask)
    # # plt.plot(m)
    # print(f'plane_points.shape: {plane_points.shape}')
    # exit()

    # print(f'zz.shape: {zz.shape}')
    # print(f'zz: {zz}')
    
    # Initialize the volume region
    volume_region = np.zeros_like(mask)
    
    # Iterate over a range of values to create a thickness around the plane
    for t in range(-thickness, thickness + 1):
        zz_t = zz + t  # Shift plane points by t voxels
        # set points outside the volume to the closest point inside the volume
        zz_t[np.where(zz_t < 0)] = 0
        zz_t[np.where(zz_t >= u_hr.shape[3])] = u_hr.shape[3] - 1
        
        # Get points within the volume
        points_in_thick_plane = np.zeros_like(mask)
        points_in_thick_plane[xx.flatten().astype(int), yy.flatten().astype(int), zz_t.flatten().astype(int)] = 1
        # Add these points to the volume region
        volume_region += points_in_thick_plane
    # Ensure the region values are binary (1 if inside the volume, 0 if outside)
    print(f'Unique values in volume region: {np.unique(volume_region, return_counts=True)}')
    volume_region = np.clip(volume_region, 0, 1)
    print(f'Unique values in volume region: {np.unique(volume_region, return_counts=True)}')
    # Restrict the volume to fluid region
    volume_core = volume_region.copy()
    print(f'Number of points in the plane: {np.sum(volume_core[np.where(mask == 1)])}')
    volume_core[np.where(mask == 0)] = 0

    

    # Adjust to different models
    volume_selected_region = volume_core.copy()
    print(f'Number of points in the plane before cutting: {np.sum(volume_selected_region)}')
    for idx_nonflow_area in idxs_nonflow_area:
        volume_selected_region[idx_nonflow_area] = 0

    print('----------------')

    # plot only the selected region in the thin slice plane
    print(f'Number of points in the plane: {np.sum(volume_selected_region)}')

    if show_mask:
        # Always adjust to different models
        zz[np.where(zz < 0)] = 0
        zz[np.where(zz >= u_hr.shape[3])] = u_hr.shape[3] - 1
        points_in_plane = np.zeros_like(mask)
        points_in_plane[xx.flatten().astype(int), yy.flatten().astype(int), zz.flatten().astype(int)] = 1
        idx_plane                = np.where(points_in_plane>0)
        points_MV = points_in_plane.copy()
        points_MV[np.where(mask==0)] = 0
        for idx_nonflow_area in idxs_nonflow_area:
            points_MV[idx_nonflow_area] = 0
        img_mask = mask[idx_plane].reshape(xx.shape[1], -1)
        img_MV_mask = points_MV[idx_plane].reshape(xx.shape[1], -1)
        plt.imshow(mask[idx_plane].reshape(xx.shape[1], -1))
        plt.imshow(img_MV_mask+img_mask)
        if save_as is not None:
            plt.savefig(f'{eval_dir}/{os.path.basename(save_as).replace("vel", "mask_of_plane")}')
        plt.show()

    if True: 
        #-----plot MV 2; 3D plot with plane and intersection----- 
        boundary_mask, _ = get_boundaries(mask)

        x_bounds, y_bounds, z_bounds = np.where(boundary_mask==1)
        x_MV, y_MV, z_MV = np.where(volume_selected_region==1)

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        # ax.plot_surface(xx, yy, zz, alpha = 0.33, color = KI_colors['Grey']) # plot plane
        ax.scatter3D(x_bounds, y_bounds, z_bounds, s= 3, alpha = 0.1) #plot boundary points
        ax.scatter3D(cp_plane[0], cp_plane[1], cp_plane[2],'x', color = 'red') #show point in plane
        ax.scatter3D(cp_plane[0], cp_plane[1] ,cp_plane[2] , s = 3, color = 'black') # plot normal point
        ax.scatter3D(x_MV, y_MV, z_MV, alpha = 0.2, s = 3, color = 'red') #plot MV points
        plt.xlabel('x')
        plt.ylabel('y')
        ax.set_zlabel('z')
        ax.set_zlim(0, hr_u.shape[3])
        plt.title(f'3D plot of plane and intersection, in total {np.sum(volume_selected_region)} points')
        plt.savefig(f'{eval_dir}/{os.path.basename(save_as).replace("vel", "3D_plot")}')
        plt.show()
        print(f'SAVED {eval_dir}/{os.path.basename(save_as).replace("vel", "3D_plot")}')

    if True:
        # inpt and idx plane
        zz[np.where(zz < 0)] = 0
        zz[np.where(zz >= u_hr.shape[3])] = u_hr.shape[3] - 1
        points_in_plane = np.zeros_like(mask)
        points_in_plane[xx.flatten().astype(int), yy.flatten().astype(int), zz.flatten().astype(int)] = 1
        idx_plane                = np.where(points_in_plane>0)
        lr_vel   = velocity_through_plane_paired_invivo(idx_plane, u_lr, v_lr, w_lr, plane_normal, order_normal = order_normal).reshape(u_lr.shape[0], xx.shape[1], -1)
        hr_vel   = velocity_through_plane_paired_invivo(idx_plane, u_hr, v_hr, w_hr, plane_normal, order_normal = order_normal).reshape(u_hr.shape[0], xx.shape[1], -1)
        pred_vel = velocity_through_plane_paired_invivo(idx_plane, u_sr, v_sr, w_sr, plane_normal, order_normal = order_normal).reshape(u_sr.shape[0], xx.shape[1], -1)

        #-----plot MV 1; Qualitave plot----- 
        # idx_crop = np.index_exp[:, 17:37, 5:25]
        # idx_crop2 = np.index_exp[17:37, 5:25]
        idx_crop = np.index_exp[:, :, :]

        # crop to important region
        lr_vel_crop = lr_vel[idx_crop]
        hr_vel_crop = hr_vel[idx_crop ]
        pred_vel_crop = pred_vel[idx_crop]

        timepoints_hr = [6, 7, 8, 9]
        timepoints_lr = [3, 4, 5, 6]
        plot_qual_comparsion(hr_vel_crop[timepoints_hr[0]:timepoints_hr[-1]+1], lr_vel_crop[timepoints_lr[0]:timepoints_lr[-1]+1], pred_vel_crop[timepoints_hr[0]:timepoints_hr[-1]+1], img_MV_mask,None,  [], [], 
                         min_v=None, max_v = None,  timepoints = timepoints_hr,figsize=(8, 5),  save_as = save_as.replace("vel", "qualitative_timeseries_LRHRSR"))
        plt.show()
    
    plt.close('all')
    #-----plot MV 3; Plot Flow profile within mask----- 

    # project velocities on plane normal
    vel_hr_plane_projection = vel_projection_plane(u_hr, v_hr, w_hr, plane_normal, volume_selected_region, order_normal = order_normal, factor_plane_normal = factor_plane_normal)
    vel_sr_plane_projection = vel_projection_plane(u_sr, v_sr, w_sr, plane_normal, volume_selected_region, order_normal = order_normal, factor_plane_normal = factor_plane_normal)
    vel_lr_plane_projection = vel_projection_plane(u_lr, v_lr, w_lr, plane_normal, volume_selected_region, order_normal = order_normal, factor_plane_normal = factor_plane_normal)

    # plot mean velocities over time:
    t_range_in_s_hr = np.linspace(0, hr_hb_duration, vel_hr_plane_projection.shape[0])
    t_range_in_s_lr = np.linspace(0, lr_hb_duration, vel_lr_plane_projection.shape[0])
    t_range_in_s_sr = np.linspace(0, lr_hb_duration, vel_sr_plane_projection.shape[0])
    plt.figure(figsize=(6, 3))
    plt.plot(t_range_in_s_hr, vel_hr_plane_projection,'-o', label='HR', color='black', markersize = 3)
    plt.plot(t_range_in_s_lr, vel_lr_plane_projection,'-o', label='LR', color='forestgreen', markersize = 3)
    plt.plot(t_range_in_s_sr, vel_sr_plane_projection, '--o', label='SR', color=KI_colors['Plum'], markersize = 3)
    plt.legend(fontsize=16)
    plt.xlabel('time [s]', fontsize=16)
    plt.ylabel('Velocity (m/s)', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize = 16)
    plt.locator_params(axis='y', nbins=3) 
    if save_as is not None:
        plt.savefig(save_as, bbox_inches='tight')#, transparent=True)
    plt.show()
    plt.close('all')
    return vel_hr_plane_projection, vel_sr_plane_projection, vel_lr_plane_projection

def velocity_through_plane_paired_invivo(idx_plane, u, v, w, plane_normal, order_normal = [0, 1, 2]):

    N_frames = u.shape[0]
    vx_in_plane = u[:, idx_plane[0], idx_plane[1], idx_plane[2]].reshape(N_frames, -1)
    vy_in_plane = v[:, idx_plane[0], idx_plane[1], idx_plane[2]].reshape(N_frames, -1)
    vz_in_plane = w[:, idx_plane[0], idx_plane[1], idx_plane[2]].reshape(N_frames, -1)
    return vx_in_plane*plane_normal[order_normal[0]]+ vy_in_plane*plane_normal[order_normal[1]]+ vz_in_plane*plane_normal[order_normal[2]]
    

def vel_projection_plane(u, v, w, plane_normal, mask_volume, order_normal = [0, 1, 2], factor_plane_normal = [1, 1, 1] ):
    assert mask_volume.shape == u.shape[1:], 'Mask and velocity field must have the same shape'
    assert mask_volume.ndim == 3, 'Mask must be 3D'

    idx_x, idx_y, idx_z = np.where(mask_volume)
    idx_timeseries = np.index_exp[:, idx_x, idx_y, idx_z]

    # project velocities on plane normal
    vel_plane_projection = np.mean((u[idx_timeseries]*plane_normal[order_normal[0]]*factor_plane_normal[0]+
                                    v[idx_timeseries]*plane_normal[order_normal[1]]* factor_plane_normal[1]+
                                    w[idx_timeseries]*plane_normal[order_normal[2]]* factor_plane_normal[2]).reshape(u.shape[0], -1),
                                        where = mask_volume[idx_x, idx_y, idx_z].astype(bool), axis = 1)

    return vel_plane_projection


def plot_correlation_paired_invivo(v_hr, v_sr, mask, frame_idx, p=0.1, fontsize=16, save_as=None, title_y='SR', direction='x', ax=None):
    """
    Plot correlation plot between high-resolution (v_hr) and super-resolution (v_sr) data for a specified frame.
    
    Parameters:
        v_hr: High-resolution data.
        v_sr: Super-resolution data.
        mask: Mask for valid data points.
        frame_idx: Frame index for plotting.
        p: Proportion of random samples to take from mask.
        fontsize: Font size for labels and title.
        save_as: File path to save the plot.
        title_y: Title for the Y-axis.
        direction: Title for the plot (X or Y direction).
        ax: Optional Matplotlib axes object for plotting (for subplots).
    """
    
    # Set up random sampling and shared figure settings
    mask = np.asarray(mask).squeeze()
    if v_hr.shape[0] != v_sr.shape[0]:
        frame_idx_sr = frame_idx // int(v_hr.shape[0] / v_sr.shape[0])
    else:
        frame_idx_sr = frame_idx
    
    # Determine core and random sample points
    idx_core = np.where(mask == 1)
    x_idx, y_idx, z_idx = random_indices3D(mask, n=int(p * np.count_nonzero(mask)))
    
    # Extract random samples and compute limits
    hr_samples = v_hr[frame_idx, x_idx, y_idx, z_idx]
    sr_samples = v_sr[frame_idx_sr, x_idx, y_idx, z_idx]
    abs_max = max(abs(hr_samples).max(), abs(sr_samples).max())
    
    # Generate correlation line data
    x_range = np.linspace(-abs_max, abs_max, 100)
    corr_line, text = get_corr_line_and_r2(v_hr[frame_idx][idx_core], v_sr[frame_idx_sr][idx_core], x_range)

    # Create a new figure and axes if none provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))
        single_plot = True
    else:
        single_plot = False
    
    # Plotting
    ax.plot(x_range, x_range, color='gray', label='Diagonal Line')   # Diagonal
    ax.plot(x_range, corr_line, 'k--', label='Fit Line')             # Regression
    
    # Scatter plot
    ax.scatter(hr_samples, sr_samples, s=30, color='black', label='Core Voxels')
    
    # Set labels and title
    # ax.set_title(rf"{direction}", fontsize=fontsize)  # Ensure proper formatting for direction
    ax.set_xlabel(r"$V_{HR}$ [m/s]", fontsize=fontsize)        # Consistent formatting for HR
    if single_plot:
        ax.set_ylabel(rf"$V_{{{title_y}}}$ [m/s]", fontsize=fontsize)  # Ensure proper formatting for title_y
    
    # Customize limits, ticks, and spines
    ax.set_xlim(-abs_max, abs_max)
    ax.set_ylim(-abs_max, abs_max)
    ax.locator_params(axis='both', nbins=3)
    ax.tick_params(axis='both', which='minor', color='lightgray', labelsize=fontsize//2+2, labelcolor='gray')
    ax.tick_params(axis='both', which='major', color='gray', labelcolor='gray', labelsize=fontsize//2+2)
    for spine in ax.spines.values():
        spine.set_color('gray')
        spine.set_linewidth(0.5)

    # Add text for R² and other fit metrics if needed
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=fontsize, verticalalignment='top')
    
    # Adjust layout and save if specified
    if ax is None:  # Only apply tight layout if a new figure was created
        plt.tight_layout()
        if save_as:
            plt.savefig(save_as, transparent=True)
        plt.show()

    return ax  # Return the axes object instead of the figure



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="My script description")
    parser.add_argument("--model", type=str, help="Optional argument to pass the name of the model")
    args = parser.parse_args()

    # Define directories and filenames
    if args.model is not None:
        network_model = args.model
    else:
        network_model = '20240709-2057' 
    # for one network evluation on multiple invivo datasets
    
    # set directories 
    # input_dir = 'data/paired_invivo/'
    lr_dir = 'Temporal4DFlowNet/data/Aortic_Phantom_Stanford'
    hr_dir = 'Temporal4DFlowNet/data/Aortic_Phantom_Stanford'
    sr_dir = 'Temporal4DFlowNet/results/in_vivo/Aortic_Phantom_Stanford'
    
    show_animation = False
    show_qual_timeseries = False
    show_plane_velocities = True
    show_bland_altman = False
    show_qual_peak_frames = False
    tabulate_results = False
    include_interpolation = False
    save_as_vti = False
    # plot_mean_speed = True
    # plot_correlation = True
    planes_to_plot = ['AAo', 'BCT', 'LSA', 'DAo','inlet', 'outlet'] #'AAo', 'BCT', 'LSA', 'DAo', 

    
    eval_dir_base = f'Temporal4DFlowNet/results/in_vivo/Aortic_Phantom_Stanford/plots_3d_slicer_18march/{network_model}'
    os.makedirs(eval_dir_base, exist_ok=True)

    phantom_models = [
             'mc1_4DFlowWIP_Ao_v120_2.5', ]
            #  'mc2_4DFlowWIP_Ao_v120_2.5',
            #  'mr_4DFlowWIP_Ao_v120_2.5']

    results_aorta_lv = {}
    results_aorta = {}
    results_lv = {}
    results_planes_aorta = {}

    for phantom_model in phantom_models:
        print(f'Evaluate volunteer {phantom_model}...')

        lr_filename = f'{phantom_model}_25frames.h5'
        hr_filename = f'{phantom_model}_50frames.h5'
        sr_filename = f'{phantom_model}_25frames/{phantom_model}_25frames_{network_model}.h5'
        eval_dir = f'{eval_dir_base}/{phantom_model}'
        os.makedirs(eval_dir, exist_ok=True)

        with h5py.File(f'{lr_dir}/{lr_filename}', 'r') as f:
            lr_u = np.array(f['u'])/100 #m/s
            lr_v = np.array(f['v'])/100 #m/s
            lr_w = np.array(f['w'])/100 #m/s
            print(f.keys())
            mask = np.array(f['mask_aorta_phantom'])[0, :, :, :]
            
            # TODO exchnage this of we want to consider other regions
            mask_aorta = np.array(f['mask_aorta_phantom'])[0, :, :, :]
            mask_outflow = np.array(f['mask_outflow'])[0, :, :, :]
            lr_hb_duration = 1#np.array(f['hb_duration']).astype(float)/1000 #s

        with h5py.File(f'{hr_dir}/{hr_filename}', 'r') as f:
            hr_u = np.array(f['u'])/100 #m/s
            hr_v = np.array(f['v'])/100 #m/s
            hr_w = np.array(f['w'])/100 #m/s
            hr_hb_duration = 1#np.array(f['hb_duration']).astype(float)/1000 #s
            if 'dx' in f:
                dx = np.array(f['dx'])
                if dx.shape[0] == 1:
                    dx = dx[0]
            else:
                dx = 2.5


        with h5py.File(f'{sr_dir}/{sr_filename}', 'r') as f:
            sr_u = np.array(f['u_combined'])/100 #m/s
            sr_v = np.array(f['v_combined'])/100 #m/s
            sr_w = np.array(f['w_combined'])/100 #m/s

        T_lr = lr_u.shape[0]
        T_hr = hr_u.shape[0]
        # print('hb_duration', lr_hb_duration, hr_hb_duration)

        if include_interpolation:
            # interpolate low resolution data to high resolution
            interpolate_linear_u = temporal_linear_interpolation_np(lr_u, hr_u.shape)
            interpolate_linear_v = temporal_linear_interpolation_np(lr_v, hr_v.shape)
            interpolate_linear_w = temporal_linear_interpolation_np(lr_w, hr_w.shape)

            # load sinc interpolation
            hr_range = np.linspace(0,1,  hr_u.shape[0])
            ups_factor = hr_u.shape[0]//lr_u.shape[0]
            lr_range = hr_range[::ups_factor] # downsamplie like this to get exact same evaluation points

            interpolate_sinc_u = temporal_sinc_interpolation_ndarray(lr_u, lr_range, hr_range)
            interpolate_sinc_v = temporal_sinc_interpolation_ndarray(lr_v, lr_range, hr_range)
            interpolate_sinc_w = temporal_sinc_interpolation_ndarray(lr_w, lr_range, hr_range)

        hr_mean_speed = np.mean(np.sqrt(hr_u**2 + hr_v**2 + hr_w**2), axis=(1, 2, 3), where=mask.astype(bool))
        T_peak_flow_frame = np.argmax(hr_mean_speed)
        synthesized_peak_flow_frame = T_peak_flow_frame.copy() 

        # take next frame if peak flow frame included in lr data
        if synthesized_peak_flow_frame % 2 == 0: 
            if hr_mean_speed[synthesized_peak_flow_frame-1] > hr_mean_speed[synthesized_peak_flow_frame+1]:
                synthesized_peak_flow_frame -=1
            else:
                synthesized_peak_flow_frame +=1


        #------ whole heart evaluation (aorta + lv) ------
        if tabulate_results:
            # calculate rmse 
            rmse_u = np.sqrt(np.mean((hr_u - sr_u)**2, axis=(1, 2, 3), where=mask.astype(bool)))
            rmse_v = np.sqrt(np.mean((hr_v - sr_v)**2, axis=(1, 2, 3), where=mask.astype(bool)))
            rmse_w = np.sqrt(np.mean((hr_w - sr_w)**2, axis=(1, 2, 3), where=mask.astype(bool)))

            # calculate k and r2 values for each direction over time

            k_u_srhr, r2_u_srhr = calculate_k_R2_timeseries(sr_u, hr_u, np.repeat(mask[np.newaxis, ...], T_hr, axis=0))
            k_v_srhr, r2_v_srhr = calculate_k_R2_timeseries(sr_v, hr_v, np.repeat(mask[np.newaxis, ...], T_hr, axis=0))
            k_w_srhr, r2_w_srhr = calculate_k_R2_timeseries(sr_w, hr_w, np.repeat(mask[np.newaxis, ...], T_hr, axis=0))

            results_aorta_lv_volunteer = {}
            results_aorta_lv_volunteer['Volunteer'] = phantom_model
            results_aorta_lv_volunteer['RMSE_u'] = rmse_u
            results_aorta_lv_volunteer['RMSE_v'] = rmse_v
            results_aorta_lv_volunteer['RMSE_w'] = rmse_w
            results_aorta_lv_volunteer['k_u'] = k_u_srhr
            results_aorta_lv_volunteer['k_v'] = k_v_srhr
            results_aorta_lv_volunteer['k_w'] = k_w_srhr
            results_aorta_lv_volunteer['r2_u'] = r2_u_srhr
            results_aorta_lv_volunteer['r2_v'] = r2_v_srhr
            results_aorta_lv_volunteer['r2_w'] = r2_w_srhr


            #------ aorta evaluation ------
            # calculate rmse
            rmse_u_aorta = np.sqrt(np.mean((hr_u - sr_u)**2, axis=(1, 2, 3), where=mask_aorta.astype(bool)))
            rmse_v_aorta = np.sqrt(np.mean((hr_v - sr_v)**2, axis=(1, 2, 3), where=mask_aorta.astype(bool)))
            rmse_w_aorta = np.sqrt(np.mean((hr_w - sr_w)**2, axis=(1, 2, 3), where=mask_aorta.astype(bool)))

            # calculate k and r2 values for each direction over time

            k_u_srhr_aorta, r2_u_srhr_aorta = calculate_k_R2_timeseries(sr_u, hr_u, np.repeat(mask_aorta[np.newaxis, ...], T_hr, axis=0))
            k_v_srhr_aorta, r2_v_srhr_aorta = calculate_k_R2_timeseries(sr_v, hr_v, np.repeat(mask_aorta[np.newaxis, ...], T_hr, axis=0))
            k_w_srhr_aorta, r2_w_srhr_aorta = calculate_k_R2_timeseries(sr_w, hr_w, np.repeat(mask_aorta[np.newaxis, ...], T_hr, axis=0))

            results_aorta_volunteer = {}
            results_aorta_volunteer['Volunteer'] = phantom_model
            results_aorta_volunteer['RMSE_u'] = rmse_u_aorta
            results_aorta_volunteer['RMSE_v'] = rmse_v_aorta
            results_aorta_volunteer['RMSE_w'] = rmse_w_aorta
            results_aorta_volunteer['k_u'] = k_u_srhr_aorta
            results_aorta_volunteer['k_v'] = k_v_srhr_aorta
            results_aorta_volunteer['k_w'] = k_w_srhr_aorta
            results_aorta_volunteer['r2_u'] = r2_u_srhr_aorta
            results_aorta_volunteer['r2_v'] = r2_v_srhr_aorta
            results_aorta_volunteer['r2_w'] = r2_w_srhr_aorta

            #------ left ventricle evaluation ------
            # calculate rmse
            rmse_u_lv = np.sqrt(np.mean((hr_u - sr_u)**2, axis=(1, 2, 3), where=mask_outflow.astype(bool)))
            rmse_v_lv = np.sqrt(np.mean((hr_v - sr_v)**2, axis=(1, 2, 3), where=mask_outflow.astype(bool)))
            rmse_w_lv = np.sqrt(np.mean((hr_w - sr_w)**2, axis=(1, 2, 3), where=mask_outflow.astype(bool)))

            # calculate k and r2 values for each direction over time
            
            k_u_srhr_lv, r2_u_srhr_lv = calculate_k_R2_timeseries(sr_u, hr_u, np.repeat(mask_outflow[np.newaxis, ...], T_hr, axis=0))
            k_v_srhr_lv, r2_v_srhr_lv = calculate_k_R2_timeseries(sr_v, hr_v, np.repeat(mask_outflow[np.newaxis, ...], T_hr, axis=0))
            k_w_srhr_lv, r2_w_srhr_lv = calculate_k_R2_timeseries(sr_w, hr_w, np.repeat(mask_outflow[np.newaxis, ...], T_hr, axis=0))

            results_lv_volunteer = {}
            results_lv_volunteer['Volunteer'] = phantom_model
            results_lv_volunteer['RMSE_u'] = rmse_u_lv
            results_lv_volunteer['RMSE_v'] = rmse_v_lv
            results_lv_volunteer['RMSE_w'] = rmse_w_lv
            results_lv_volunteer['k_u'] = k_u_srhr_lv
            results_lv_volunteer['k_v'] = k_v_srhr_lv
            results_lv_volunteer['k_w'] = k_w_srhr_lv
            results_lv_volunteer['r2_u'] = r2_u_srhr_lv
            results_lv_volunteer['r2_v'] = r2_v_srhr_lv
            results_lv_volunteer['r2_w'] = r2_w_srhr_lv

            # save results in dictionary
            results_aorta_lv[phantom_model] = results_aorta_lv_volunteer
            results_aorta[phantom_model] = results_aorta_volunteer
            results_lv[phantom_model] = results_lv_volunteer

        # plot desecning and asceing aorta
        T_lr = lr_u.shape[0]
        T_hr = hr_u.shape[0]
        
        print('shapes u:', lr_u.shape, hr_u.shape, sr_u.shape)
        print('shapes v:', lr_v.shape, hr_v.shape, sr_v.shape)
        print('shapes w:', lr_w.shape, hr_w.shape, sr_w.shape)

        if show_plane_velocities:
            results_planes_aorta[phantom_model] = {}
            for plane in planes_to_plot:
                print(f'Plotting plane {plane}..')
                plane_normal = np.array(AORTIC_PHANTOM_SETTINGS[phantom_model][plane]['normal'])
                cp_plane = np.array(AORTIC_PHANTOM_SETTINGS[phantom_model][plane]['origin'])/dx
                aortic_side = AORTIC_PHANTOM_SETTINGS[phantom_model][plane]['aortic_side']
                thickness = AORTIC_PHANTOM_SETTINGS[phantom_model]['thickness_ascending'] if aortic_side == 'ascending' else AORTIC_PHANTOM_SETTINGS[phantom_model]['thickness_descending']
                factor_plane_normal = AORTIC_PHANTOM_SETTINGS[phantom_model]['factor_plane_normal']

                hr_plane, sr_plane, lr_plane = plot_plane_flows(hr_u, hr_v, hr_w, lr_u, lr_v, lr_w, sr_u, sr_v, sr_w, cp_plane, plane_normal, 
                                                                AORTIC_PHANTOM_SETTINGS[phantom_model][f'idxs_nonflow_area_{aortic_side}'], 
                                                                order_normal = AORTIC_PHANTOM_SETTINGS[phantom_model]['order_normal'], thickness=thickness, 
                                                                factor_plane_normal = factor_plane_normal, save_as = f'{eval_dir}/{phantom_model}_{plane}_vel.png', 
                                                                lr_hb_duration=lr_hb_duration, hr_hb_duration=hr_hb_duration)
                
                results_planes_aorta[phantom_model][plane] = {
                    'hr': hr_plane,
                    'sr': sr_plane,
                    'lr': lr_plane,
                }
                results_planes_aorta[phantom_model]['hr_hb_duration'] = hr_hb_duration
                results_planes_aorta[phantom_model]['lr_hb_duration'] = lr_hb_duration   

        if show_qual_timeseries:
            time_points = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
            idx_cube = np.index_exp[time_points[0]:time_points[-1]+1, :, :, 20]
            idx_cube_lr = np.index_exp[time_points[0]//2:time_points[-1]//2+1, :, :, 20]
            idx_mask = np.index_exp[:, :, 20]

            # without interpolation
            plot_qual_comparsion(hr_u[idx_cube], lr_u[idx_cube_lr], sr_u[idx_cube], mask[idx_mask], None, [], [], time_points, None, None, include_error = False,  
                                figsize = (12, 3), save_as = f"{eval_dir}/{phantom_model}_qual_comparison_u_frames{time_points[0]}-{time_points[-1]}.png", colormap='viridis')
            plot_qual_comparsion(hr_v[idx_cube], lr_v[idx_cube_lr], sr_v[idx_cube], mask[idx_mask], None, [], [], time_points, None, None, include_error = False,  
                                figsize = (12, 3), save_as = f"{eval_dir}/{phantom_model}_qual_comparison_v_frames{time_points[0]}-{time_points[-1]}.png", colormap='viridis')
            plot_qual_comparsion(hr_w[idx_cube], lr_w[idx_cube_lr], sr_w[idx_cube], mask[idx_mask], None, [], [], time_points, None, None, include_error = False,  
                                figsize = (12, 3), save_as = f"{eval_dir}/{phantom_model}_qual_comparison_w_frames{time_points[0]}-{time_points[-1]}.png", colormap='viridis')
            
            
            for time_point in time_points:
                if time_point % 2 != 0:
                    plt.figure(figsize=(4, 4))
                    plot_correlation_paired_invivo(hr_u, sr_u, mask, time_point, p=0.1, fontsize = 16, direction = r'$V_x$', save_as = f'{eval_dir}/{phantom_model}_correlation_frame{time_point}_u.png')
                    plt.savefig(f'{eval_dir}/{phantom_model}_correlation_frame{time_point}_Vx.png', bbox_inches='tight', transparent=True)
                    plt.close('all')
                    plot_correlation_paired_invivo(hr_v, sr_v, mask, time_point, p=0.1, fontsize = 16, direction = r'$V_y$', save_as = f'{eval_dir}/{phantom_model}_correlation_frame{time_point}_v.png')
                    plt.savefig(f'{eval_dir}/{phantom_model}_correlation_frame{time_point}_Vy.png', bbox_inches='tight', transparent=True)
                    plt.close('all')
                    plot_correlation_paired_invivo(hr_w, sr_w, mask, time_point, p=0.1, fontsize = 16, direction = r'$V_z$', save_as = f'{eval_dir}/{phantom_model}_correlation_frame{time_point}_w.png')
                    plt.savefig(f'{eval_dir}/{phantom_model}_correlation_frame{time_point}_Vz.png', bbox_inches='tight', transparent=True)

                    # make subplots for all directions
                    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                    plot_correlation_paired_invivo(hr_u, sr_u, mask, time_point, p=0.1, fontsize = 16, direction = r'$V_x$', ax = axes[0])
                    plot_correlation_paired_invivo(hr_v, sr_v, mask, time_point, p=0.1, fontsize = 16, direction = r'$V_y$', ax = axes[1])
                    plot_correlation_paired_invivo(hr_w, sr_w, mask, time_point, p=0.1, fontsize = 16, direction = r'$V_z$', ax = axes[2])
                    axes[0].set_ylabel(r'V$_{SR}$ [m/s]', fontsize=16)
                    plt.tight_layout()
                    plt.savefig(f'{eval_dir}/{phantom_model}_correlation_frame{time_point}_VxVyVz.png', bbox_inches='tight', transparent=True)
                    plt.close('all')
            if show_qual_peak_frames:

                idx_cube = np.index_exp[synthesized_peak_flow_frame, 67, 45:95, 20:70]
                mask_slice = np.index_exp[67, 45:95, 20:70]
                if include_interpolation:
                    comparison_data = [
                        (interpolate_linear_u[idx_cube], interpolate_linear_v[idx_cube], interpolate_linear_w[idx_cube]),
                        (interpolate_sinc_u[idx_cube], interpolate_sinc_v[idx_cube], interpolate_sinc_w[idx_cube])
                    ]
                    comparison_names = ['linear', 'sinc']
                else:
                    comparison_data = []
                    comparison_names = []

                plot_qual_comparison_peak(
                    hr_u[idx_cube], hr_v[idx_cube], hr_w[idx_cube],
                    sr_u[idx_cube], sr_v[idx_cube], sr_w[idx_cube], 
                    mask[mask_slice], None, comparison_data, comparison_names, synthesized_peak_flow_frame, None, None, colormap='plasma',
                    figsize=(7.2, 4.6), save_as=f"{eval_dir}/{phantom_model}_qual_peak_timepoint{synthesized_peak_flow_frame}_VxVyVz.png"
                )
                plt.show()
                plt.close('all')

        if show_bland_altman:
            # plot bland altman
            

            print(f'Plot bland altman for volunteer {phantom_model}...')
            # plot bland altman
            bland_altman_plot(sr_u, hr_u, mask,timepoint=synthesized_peak_flow_frame, save_as=f'{eval_dir}/{phantom_model}_bland_altman_u_peaksyn{synthesized_peak_flow_frame}.png')
            plt.show()
            plt.close('all')
            bland_altman_plot(sr_v, hr_v, mask,timepoint=synthesized_peak_flow_frame, save_as=f'{eval_dir}/{phantom_model}_bland_altman_v_peaksyn{synthesized_peak_flow_frame}.png')
            plt.show()
            plt.close('all')
            bland_altman_plot(sr_w, hr_w, mask,timepoint=synthesized_peak_flow_frame, save_as=f'{eval_dir}/{phantom_model}_bland_altman_w_peaksyn{synthesized_peak_flow_frame}.png')
            plt.show()

            fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
            max_diff = np.max([np.max(hr_u[synthesized_peak_flow_frame][np.where(mask> 0.5)] - sr_u[synthesized_peak_flow_frame][np.where(mask> 0.5)]),
                                np.max(hr_v[synthesized_peak_flow_frame][np.where(mask> 0.5)] - sr_v[synthesized_peak_flow_frame][np.where(mask> 0.5)]),
                                np.max(hr_w[synthesized_peak_flow_frame][np.where(mask> 0.5)] - sr_w[synthesized_peak_flow_frame][np.where(mask> 0.5)])])
            bland_altman_plot(sr_u, hr_u, mask, timepoint=synthesized_peak_flow_frame,y_lim=(-max_diff, max_diff),  ax=axes[0],  fontsize=16)
            axes[0].tick_params(axis='y', labelleft=True)
            axes[0].set_ylabel(r'V$_{HR}$ - V$_{SR}$ [m/s]', fontsize=16)

            bland_altman_plot(sr_v, hr_v, mask, timepoint=synthesized_peak_flow_frame,y_lim=(-max_diff, max_diff), ax=axes[1],fontsize=16)
            axes[1].tick_params(axis='y', labelleft=True)

            bland_altman_plot(sr_w, hr_w, mask, timepoint=synthesized_peak_flow_frame,y_lim=(-max_diff, max_diff), ax=axes[2],fontsize=16)
            axes[2].tick_params(axis='y', labelleft=True)
            # set background color
            mean_diff_u = np.mean(hr_u[synthesized_peak_flow_frame][np.where(mask> 0.5)] - sr_u[synthesized_peak_flow_frame][np.where(mask> 0.5)])
            mean_diff_v = np.mean(hr_v[synthesized_peak_flow_frame][np.where(mask> 0.5)] - sr_v[synthesized_peak_flow_frame][np.where(mask> 0.5)])
            mean_diff_w = np.mean(hr_w[synthesized_peak_flow_frame][np.where(mask> 0.5)] - sr_w[synthesized_peak_flow_frame][np.where(mask> 0.5)])

            
            print(f'Mean bias u: {mean_diff_u}, v: {mean_diff_v}, w: {mean_diff_w}')    

            plt.tight_layout()
            plt.savefig(f'{eval_dir}/{phantom_model}_bland_altman_peaksyn{synthesized_peak_flow_frame}_VxVyVz_centered.png', transparent=True)
            plt.show()
        
            
            fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharey='row') #15 10
            vel_plotname = [r'$V_x$', r'$V_y$', r'$V_z$']
            
            x_rnd, y_rnd, z_rnd = random_indices3D(mask, n = int(0.1*np.sum(mask)))
            idx_core_t = np.where(mask > 0.5)

            abs_max_u = np.maximum(np.max(np.abs(hr_u[synthesized_peak_flow_frame][idx_core_t])), np.max(np.abs(sr_u[synthesized_peak_flow_frame][idx_core_t])))
            abs_max_v = np.maximum(np.max(np.abs(hr_v[synthesized_peak_flow_frame][idx_core_t])), np.max(np.abs(sr_v[synthesized_peak_flow_frame][idx_core_t])))
            abs_max_w = np.maximum(np.max(np.abs(hr_w[synthesized_peak_flow_frame][idx_core_t])), np.max(np.abs(sr_w[synthesized_peak_flow_frame][idx_core_t])))

            abs_max = np.maximum(abs_max_u, np.maximum(abs_max_v, abs_max_w))

            plot_regression_points_new(axes[0, 0],hr_u[synthesized_peak_flow_frame, x_rnd, y_rnd, z_rnd], sr_u[synthesized_peak_flow_frame, x_rnd, y_rnd, z_rnd], 
                                    hr_u[synthesized_peak_flow_frame][idx_core_t], sr_u[synthesized_peak_flow_frame][idx_core_t], abs_max, direction= r'V$_x$', color='black', show_text=True)
            bland_altman_plot(sr_u, hr_u, mask, timepoint=synthesized_peak_flow_frame, ax=axes[1, 0], fontsize=18, centered_ylim=True, y_lim=(-max_diff, max_diff))

            plot_regression_points_new(axes[0, 1], hr_v[synthesized_peak_flow_frame, x_rnd, y_rnd, z_rnd], sr_v[synthesized_peak_flow_frame, x_rnd, y_rnd, z_rnd], 
                                    hr_v[synthesized_peak_flow_frame][idx_core_t], sr_v[synthesized_peak_flow_frame][idx_core_t], abs_max, direction=r'V$_y$', color='black', show_text=True)
            bland_altman_plot(sr_v, hr_v, mask, timepoint=synthesized_peak_flow_frame, ax=axes[1, 1], fontsize=18, centered_ylim=True, y_lim=(-max_diff, max_diff))

            plot_regression_points_new(axes[0, 2], hr_w[synthesized_peak_flow_frame, x_rnd, y_rnd, z_rnd], sr_w[synthesized_peak_flow_frame, x_rnd, y_rnd, z_rnd], 
                                    hr_w[synthesized_peak_flow_frame][idx_core_t], sr_w[synthesized_peak_flow_frame][idx_core_t], abs_max, direction=r'V$_z$', color='black', show_text=True)
            bland_altman_plot(sr_w, hr_w, mask, timepoint=synthesized_peak_flow_frame, ax=axes[1, 2], fontsize=18, centered_ylim=True, y_lim=(-max_diff, max_diff))
            axes[0, 0].tick_params(axis='y', labelleft=True)
            axes[1, 0].tick_params(axis='y', labelleft=True)
            axes[0, 1].tick_params(axis='y', labelleft=True)
            axes[1, 1].tick_params(axis='y', labelleft=True)
            axes[0, 2].tick_params(axis='y', labelleft=True)
            axes[1, 2].tick_params(axis='y', labelleft=True)
            
            axes[1, 0].set_ylabel(r'V$_{HR}$ - V$_{SR}$ [m/s]', fontsize=16)

            plt.tight_layout()
            plt.savefig(f'{eval_dir}/{phantom_model}_correlation_and_blandaltman_COMBINED_synpeakframe{synthesized_peak_flow_frame}_core.png')
            plt.show()

        if show_animation:
            include_orig_data = True
            eval_gifs = f'{eval_dir}/gifs'
            os.makedirs(eval_gifs, exist_ok = True)
            
            min_V = np.min([np.min(hr_u[AORTIC_PHANTOM_SETTINGS[phantom_model]['idx_anim_slice']]), np.min(hr_v), np.min(hr_w)])
            max_V = np.max([np.max(hr_u[AORTIC_PHANTOM_SETTINGS[phantom_model]['idx_anim_slice']]), np.max(hr_v), np.max(hr_w)])

            fps_anim = 10
            fps_pred = fps_anim
            if include_orig_data: 
                
                print('Create video of original data..')
                # animate_data_over_time_gif(volunteer_plot_settings[volunteer]['idx_anim_slice'], mag, 0, np.quantile(magn, 0.99),      fps = fps_anim , save_as = f'{eval_gifs}/{volunteer}_animate_mag', colormap='Greys_r' )
                # animate_data_over_time_gif(volunteer_plot_settings[volunteer]['idx_anim_slice'], mask, 0, 1,           fps = fps_anim , save_as = f'{eval_gifs}/{volunteer}_animate_mask', colormap='Greys' )
                animate_data_over_time_gif(AORTIC_PHANTOM_SETTINGS[phantom_model]['idx_anim_slice'], hr_u, min_V, max_V,      fps = fps_anim , save_as = f'{eval_gifs}/{phantom_model}_animate_u_gt')
                animate_data_over_time_gif(AORTIC_PHANTOM_SETTINGS[phantom_model]['idx_anim_slice'], hr_v, min_V, max_V,      fps = fps_anim , save_as = f'{eval_gifs}/{phantom_model}_animate_v_gt')
                animate_data_over_time_gif(AORTIC_PHANTOM_SETTINGS[phantom_model]['idx_anim_slice'], hr_w, min_V, max_V,      fps = fps_anim , save_as = f'{eval_gifs}/{phantom_model}_animate_w_gt')
                animate_data_over_time_gif(AORTIC_PHANTOM_SETTINGS[phantom_model]['idx_anim_slice'], hr_u*mask, min_V, max_V,fps = fps_anim , save_as = f'{eval_gifs}/{phantom_model}_animate_u_gt_fluid')
                animate_data_over_time_gif(AORTIC_PHANTOM_SETTINGS[phantom_model]['idx_anim_slice'], hr_v*mask, min_V, max_V,fps = fps_anim , save_as = f'{eval_gifs}/{phantom_model}_animate_v_gt_fluid')
                animate_data_over_time_gif(AORTIC_PHANTOM_SETTINGS[phantom_model]['idx_anim_slice'], hr_w*mask, min_V, max_V,fps = fps_anim , save_as = f'{eval_gifs}/{phantom_model}_animate_w_gt_fluid')

            animate_data_over_time_gif(AORTIC_PHANTOM_SETTINGS[phantom_model]['idx_anim_slice'], lr_u, min_V, max_V, fps = fps_anim//2 , save_as = f'{eval_gifs}/{phantom_model}_animate_u_lr')
            animate_data_over_time_gif(AORTIC_PHANTOM_SETTINGS[phantom_model]['idx_anim_slice'], lr_v, min_V, max_V, fps = fps_anim//2 , save_as = f'{eval_gifs}/{phantom_model}_animate_v_lr')
            animate_data_over_time_gif(AORTIC_PHANTOM_SETTINGS[phantom_model]['idx_anim_slice'], lr_w, min_V, max_V, fps = fps_anim//2 , save_as = f'{eval_gifs}/{phantom_model}_animate_w_lr')

            animate_data_over_time_gif(AORTIC_PHANTOM_SETTINGS[phantom_model]['idx_anim_slice'], sr_u, min_V, max_V, fps = fps_anim , save_as = f'{eval_gifs}/{phantom_model}_animate_u_sr')
            animate_data_over_time_gif(AORTIC_PHANTOM_SETTINGS[phantom_model]['idx_anim_slice'], sr_v, min_V, max_V, fps = fps_anim , save_as = f'{eval_gifs}/{phantom_model}_animate_v_sr')
            animate_data_over_time_gif(AORTIC_PHANTOM_SETTINGS[phantom_model]['idx_anim_slice'], sr_w, min_V, max_V, fps = fps_anim , save_as = f'{eval_gifs}/{phantom_model}_animate_w_sr')

    # results in df
    if tabulate_results:
        # plot results
        plt.figure(figsize=(15, 10))

        # RMSE
        plt.subplot(2, 3, 1)
        for phantom_model in phantom_models:
            plt.plot(results_aorta_lv[phantom_model]['RMSE_u'], label=phantom_model)
        plt.xlabel('Frame', fontsize=12)
        plt.ylabel('RMSE u', fontsize=12)
        plt.legend(fontsize=10)

        plt.subplot(2, 3, 2)
        for phantom_model in phantom_models:
            plt.plot(results_aorta_lv[phantom_model]['RMSE_v'], label=phantom_model)
        plt.xlabel('Frame', fontsize=12)
        plt.ylabel('RMSE v', fontsize=12)
        plt.legend(fontsize=10)

        plt.subplot(2, 3, 3)
        for phantom_model in phantom_models:
            plt.plot(results_aorta_lv[phantom_model]['RMSE_w'], label=phantom_model)
        plt.xlabel('Frame', fontsize=12)
        plt.ylabel('RMSE w', fontsize=12)
        plt.legend(fontsize=10)

        # k
        plt.subplot(2, 3, 4)
        for phantom_model in phantom_models:
            plt.plot(results_aorta_lv[phantom_model]['k_u'], label=phantom_model)
        plt.xlabel('Frame', fontsize=12)
        plt.ylabel('k u', fontsize=12)
        plt.legend(fontsize=10)

        plt.subplot(2, 3, 5)
        for phantom_model in phantom_models:
            plt.plot(results_aorta_lv[phantom_model]['k_v'], label=phantom_model)
        plt.xlabel('Frame', fontsize=12)
        plt.ylabel('k v', fontsize=12)
        plt.legend(fontsize=10)

        plt.subplot(2, 3, 6)
        for phantom_model in phantom_models:
            plt.plot(results_aorta_lv[phantom_model]['k_w'], label=phantom_model)
        plt.xlabel('Frame', fontsize=12)
        plt.ylabel('k w', fontsize=12)
        plt.legend(fontsize=10)

        plt.tight_layout()
        plt.savefig(f'{eval_dir}/rmse_k_whole_domain.png')
        plt.show()

        # r2
        plt.figure(figsize=(10, 3))
        plt.subplot(1, 3, 1)
        for phantom_model in phantom_models:
            plt.plot(results_aorta_lv[phantom_model]['r2_u'], label=phantom_model)
        plt.xlabel('Frame', fontsize=12)
        plt.ylabel('r2 u', fontsize=12)
        plt.legend(fontsize=10)

        plt.subplot(1, 3, 2)
        for phantom_model in phantom_models:
            plt.plot(results_aorta_lv[phantom_model]['r2_v'], label=phantom_model)
        plt.xlabel('Frame', fontsize=12)
        plt.ylabel('r2 v', fontsize=12)
        plt.legend(fontsize=10)

        plt.subplot(1, 3, 3)
        for phantom_model in phantom_models:
            plt.plot(results_aorta_lv[phantom_model]['r2_w'], label=phantom_model)
        plt.xlabel('Frame', fontsize=12)
        plt.ylabel('r2 w', fontsize=12)
        plt.legend(fontsize=10)

        plt.tight_layout()
        plt.savefig(f'{eval_dir}/R2_whole_domain.png')
        plt.show()


        # now create a dataframe with the results, by taking the mean over all timesteps
        results_aorta_lv_mean = defaultdict(list)
        results_aorta_mean = defaultdict(list)
        results_lv_mean = defaultdict(list)

        for phantom_model in phantom_models:
            for key in results_aorta_lv[phantom_model].keys():
                if key != 'Volunteer':
                    results_aorta_lv_mean[key].append(np.mean(results_aorta_lv[phantom_model][key]))
                    results_aorta_mean[key].append(np.mean(results_aorta[phantom_model][key]))
                    results_lv_mean[key].append(np.mean(results_lv[phantom_model][key]))
                else:
                    results_aorta_lv_mean[key].append(phantom_model)
                    results_aorta_mean[key].append(phantom_model)
                    results_lv_mean[key].append(phantom_model)

        # to dataframe
        df_aorta_lv = pd.DataFrame(results_aorta_lv_mean)
        df_aorta = pd.DataFrame(results_aorta_mean)
        df_lv = pd.DataFrame(results_lv_mean)



        # display results
        print('Results aorta and left ventricle:')
        print(df_aorta_lv)
        print('Results aorta:')
        print(df_aorta)
        print('Results left ventricle:')
        print(df_lv)

        # save results
        df_aorta_lv.to_csv(f'{eval_dir_base}/results_aorta_lv.csv')
        df_aorta.to_csv(f'{eval_dir_base}/results_aorta.csv')
        df_lv.to_csv(f'{eval_dir_base}/results_lv.csv')

        df_aorta_lv_frames = pd.DataFrame()
        df_aorta_peak_systole = pd.DataFrame()
        df_aorta_peak_diastole = pd.DataFrame()
        df_lv_peak_systole = pd.DataFrame()
        df_lv_peak_diastole = pd.DataFrame()

        for phantom_model in phantom_models:
            t_sys = AORTIC_PHANTOM_SETTINGS[phantom_model]['t_frames_diastole_systole'][0]
            t_dia = AORTIC_PHANTOM_SETTINGS[phantom_model]['t_frames_diastole_systole'][1]
            df_aorta_lv_frames = pd.concat([df_aorta_lv_frames, pd.DataFrame(results_aorta_lv[phantom_model]).iloc[t_sys, :].to_frame().T])
            df_aorta_lv_frames = pd.concat([df_aorta_lv_frames, pd.DataFrame(results_aorta_lv[phantom_model]).iloc[t_dia, :].to_frame().T])
            df_aorta_peak_systole = pd.concat([df_aorta_peak_systole, pd.DataFrame(results_aorta[phantom_model]).iloc[t_sys, :].to_frame().T])
            df_aorta_peak_diastole = pd.concat([df_aorta_peak_diastole, pd.DataFrame(results_aorta[phantom_model]).iloc[t_dia, :].to_frame().T])
            df_lv_peak_systole = pd.concat([df_lv_peak_systole, pd.DataFrame(results_lv[phantom_model]).iloc[t_sys, :].to_frame().T])
            df_lv_peak_diastole = pd.concat([df_lv_peak_diastole, pd.DataFrame(results_lv[phantom_model]).iloc[t_dia, :].to_frame().T])

        # put in one dataframe
        df_aorta_lv_frames['Cardiac Phase'] = ['Systole', 'Diastole'] * len(phantom_models)
        df_aorta_peak_systole['Cardiac Phase'] = ['Systole'] * len(phantom_models)
        df_aorta_peak_diastole['Cardiac Phase'] = ['Diastole'] * len(phantom_models)
        df_lv_peak_systole['Cardiac Phase'] = ['Systole'] * len(phantom_models)
        df_lv_peak_diastole['Cardiac Phase'] = ['Diastole'] * len(phantom_models)

        df_aorta_peak_systole['Region'] = ['Aorta'] * len(phantom_models)
        df_aorta_peak_diastole['Region'] = ['Aorta'] * len(phantom_models)
        df_lv_peak_systole['Region'] = ['LV'] * len(phantom_models)
        df_lv_peak_diastole['Region'] = ['LV'] * len(phantom_models)


        df_aorta_lv_frames = df_aorta_lv_frames.reset_index(drop=True)
        df_aorta_peak_systole = df_aorta_peak_systole.reset_index(drop=True)
        df_aorta_peak_diastole = df_aorta_peak_diastole.reset_index(drop=True)
        df_lv_peak_systole = df_lv_peak_systole.reset_index(drop=True)
        df_lv_peak_diastole = df_lv_peak_diastole.reset_index(drop=True)

        print(df_aorta_lv_frames)



        # now make a table with aorta peak systole and lv peak diastole
        # df_peak_systole_diastole = pd.concat([df_aorta_peak_systole.reset_index(drop=True), df_lv_peak_diastole.reset_index(drop=True)], axis=1)
        df_peak_systole_diastole = pd.concat([df_aorta_peak_systole, df_lv_peak_diastole], axis=0)
        # Scale RMSE columns by 1/100
        rmse_columns = [col for col in df_peak_systole_diastole.columns if 'RMSE' in col]
        df_peak_systole_diastole[rmse_columns] = df_peak_systole_diastole[rmse_columns]
        print(df_peak_systole_diastole)

        #print in latex
        # print(df_peak_systole_diastole.to_latex(index=False, float_format="%.2f"))
        # save as csv
        df_peak_systole_diastole.to_csv(f'{eval_dir_base}/results_peak_systole_diastole.csv')


    if show_plane_velocities:
        fig, axes = plt.subplots(len(planes_to_plot), len(phantom_models), figsize=(4*len(phantom_models), 2*len(planes_to_plot)))  # Shared y-axis for each row , sharey='row'
        viridis = cm.get_cmap('viridis', 100)  # Viridis colormap

        N_frames = results_planes_aorta[phantom_model][planes_to_plot[0]]['hr'].shape[0]
        N_frames_lr = results_planes_aorta[phantom_model][planes_to_plot[0]]['lr'].shape[0]

        axes[0, 0].set_ylabel('velocity [m/s]', fontsize=12)
        axes[-1, 0].set_ylabel('velocity [m/s]', fontsize=12)
        for i, phantom_model in enumerate(phantom_models): 
            t_range_in_s_hr = np.linspace(0, results_planes_aorta[phantom_model]['hr_hb_duration'], N_frames)
            t_range_in_s_lr = np.linspace(0, results_planes_aorta[phantom_model]['lr_hb_duration'], N_frames_lr)
            t_range_in_s_sr = np.linspace(0, results_planes_aorta[phantom_model]['lr_hb_duration'], N_frames)

            for j, plane in enumerate(planes_to_plot):
                # Plane is row j
                axes[j, i].plot(t_range_in_s_hr, results_planes_aorta[phantom_model][plane]['hr'], '.-',  markersize=3 , label='HR', color='black')
                axes[j, i].plot(t_range_in_s_lr, results_planes_aorta[phantom_model][plane]['lr'], '.-',  markersize=3 , label='LR', color=viridis(0.25))
                axes[j, i].plot(t_range_in_s_sr, results_planes_aorta[phantom_model][plane]['sr'], '.--', markersize=3 , label='SR', color=viridis(0.85))


            # Set custom ticks
            # axes[0, i].set_yticks([0, 1])  # Example: Two ticks on the y-axis
            # axes[1, i].set_yticks([0, 1])  # Same for descending aorta
            axes[0, i].yaxis.set_tick_params(labelleft=True, labelright=False)
            axes[1, i].yaxis.set_tick_params(labelleft=True, labelright=False)

            # Set x-ticks for all plots (example: 3 ticks)
            axes[0, i].set_xticks(np.linspace(0, 1, 3))
            axes[1, i].set_xticks(np.linspace(0, 1, 3))

            # Set x-tick labels for all plots (example: 3 ticks)
            axes[1, i].set_xlabel('Time [s]', fontsize=12)

            # Customize ticks: smaller and gray
            for ax in [axes[0, i], axes[1, i]]:
                ax.tick_params(axis='both', which='major', labelsize=8, colors='grey')

            # Grey borders for all subplots
            for ax in [axes[0, i], axes[1, i]]:
                for spine in ax.spines.values():
                    spine.set_edgecolor('grey')

        # Add a shared legend below the second row
        # handles, labels = axes[1, 0].get_legend_handles_labels()  # Retrieve legend info
        
        # fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=10, frameon=False, bbox_to_anchor=(0.5, -0.2))

        # One y-label for each row, in gray
        # fig.text(0.01, 0.75, 'Ascending Aorta', va='center', ha='center', rotation='vertical', fontsize=12, color='grey')
        # fig.text(0.01, 0.25, 'Descending Aorta', va='center', ha='center', rotation='vertical', fontsize=12, color='grey')
        
        
        # axes[1, 1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=2, fontsize=9)
        plt.tight_layout()
        # plt.subplots_adjust(bottom=0.1)  # Adjust to make space for the legend
        plt.savefig(f'{eval_dir_base}/All_volunteers_subplots_planes_desc_asc_aorta.png')
        plt.savefig(f'{eval_dir_base}/All_volunteers_subplots_planes_desc_asc_aorta.svg')
        plt.show()
        plt.close('all')

    # save as vti files: 
    if save_as_vti:
        for phantom_model in phantom_models:
            spacing = [dx, dx, dx]
            os.makedirs(f'{eval_dir}/vti', exist_ok = True)

            for t in range(N_frames):
                
                output_filepath = f'{eval_dir}/vti/{phantom_model}_SR_{network_model}_frame{t}_uvw.vti'

                if os.path.isfile(output_filepath):
                        print(f'File {output_filepath} already exists')
                else:
                    vectors_and_mask_to_vtk((sr_u[t],sr_u[t],sr_u[t]),mask[t], spacing, output_filepath, include_mask = True)
    #----------------------

#-------------------------------------------------------------------------------------------------------------------------
