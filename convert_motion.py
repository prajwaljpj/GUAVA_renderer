import torch
import numpy as np
import pickle
import os
from scipy.interpolate import interp1d

def resample_sequence(data, from_fps, to_fps):
    """Resamples a numpy array from one FPS to another using linear interpolation."""
    if from_fps == to_fps:
        return data
    
    num_frames_from = data.shape[0]
    duration = (num_frames_from - 1) / from_fps
    
    # Create the time axes
    time_from = np.linspace(0, duration, num=num_frames_from)
    
    num_frames_to = int(np.ceil(duration * to_fps)) + 1
    time_to = np.linspace(0, duration, num=num_frames_to)
    
    # Create the interpolation function
    # We assume the data is structured [frames, features]
    interp_func = interp1d(time_from, data, axis=0, kind='linear', fill_value='extrapolate')
    
    # Apply the interpolation
    resampled_data = interp_func(time_to)
    
    print(f"Resampled sequence from {num_frames_from} frames ({from_fps} FPS) to {resampled_data.shape[0]} frames ({to_fps} FPS).")
    return resampled_data

def convert_motion_data():
    """
    Loads ARTalk (FLAME) and EMAGE (SMPL-X) motion files, resamples them
    to a matching framerate, and reformats them into the per-frame 
    dictionary structure expected by the GUAVA model.
    """
    artalk_path = '/home/prajwaljpj/projects/talking_video/ARTalk/render_results/motion_data/ElevenLabs_TTS_E_M_long_motion.pt'
    emage_path = '/home/prajwaljpj/projects/talking_video/PantoMatrix/examples/motion_zoom/ElevenLabs_TTS_E_M_long_output.npz'
    
    output_motion_path = 'assets/example/custom_video/my_animation/optim_tracking_custom.pkl'
    output_identity_path = 'assets/example/custom_video/my_animation/id_share_params.pkl'

    # --- Define FPS ---
    flame_fps = 25
    smplx_fps = 30
    target_fps = 25 # We will downsample SMPL-X to match FLAME

    print("--- Loading source motion files ---")
    artalk_data_torch = torch.load(artalk_path, map_location='cpu')
    artalk_data = artalk_data_torch.numpy() # Convert to numpy for consistency
    emage_data = np.load(emage_path)

    # --- 1. Resample SMPL-X data to match FLAME FPS ---
    print("\n--- Resampling SMPL-X data ---")
    smplx_poses_orig = emage_data['poses']
    smplx_expr_orig = emage_data['expressions']
    smplx_trans_orig = emage_data['trans']

    smplx_poses_resampled = resample_sequence(smplx_poses_orig, smplx_fps, target_fps)
    smplx_expr_resampled = resample_sequence(smplx_expr_orig, smplx_fps, target_fps)
    smplx_trans_resampled = resample_sequence(smplx_trans_orig, smplx_fps, target_fps)

    # --- 2. Handle Frame Count Mismatch (Post-Resampling) ---
    num_frames_artalk = artalk_data.shape[0]
    num_frames_smplx = smplx_poses_resampled.shape[0]
    
    min_frames = min(num_frames_artalk, num_frames_smplx)
    
    print(f"\nFLAME frames: {num_frames_artalk}, Resampled SMPL-X frames: {num_frames_smplx}")
    if num_frames_artalk != num_frames_smplx:
        print(f"WARNING: Frame counts still mismatch after resampling. Truncating to the minimum: {min_frames} frames.")

    # --- 3. Create Identity File ---
    identity_params = {
        'smplx_coeffs': {
            'shape': torch.from_numpy(emage_data['betas'][:10]).float(),
            'joints_offset': torch.zeros(24, 3).float(),
            'head_scale': torch.ones(1).float(),
            'hand_scale': torch.ones(1).float(),
        },
        'flame_coeffs': {
            'shape_params': torch.zeros(100).float(),
        }
    }
    
    # --- 4. Convert Motion Data ---
    converted_motion = {}
    for i in range(min_frames):
        frame_key = f'frame_{i:06d}'
        
        # --- Process FLAME Data ---
        artalk_frame = artalk_data[i]
        flame_coeffs = {
            'expression_params': torch.from_numpy(artalk_frame[0:50]),
            'pose_params': torch.from_numpy(artalk_frame[50:53]),
            'jaw_params': torch.from_numpy(artalk_frame[53:56]),
            'neck_pose_params': torch.zeros(3),
            'eye_pose_params': torch.zeros(6),
            'eyelid_params': torch.zeros(2),
            'cam': torch.zeros(3),
            'camera_RT_params': torch.eye(4)[:3]
        }

        # --- Process Resampled SMPL-X Data ---
        smplx_poses = smplx_poses_resampled[i].reshape(55, 3)
        
        smplx_coeffs = {
            'exp': torch.from_numpy(smplx_expr_resampled[i, :50]),
            'global_pose': torch.from_numpy(smplx_poses[0]),
            'body_pose': torch.from_numpy(smplx_poses[1:22]),
            'body_cam': torch.from_numpy(smplx_trans_resampled[i]),
            'left_hand_pose': torch.from_numpy(smplx_poses[25:40]),
            'right_hand_pose': torch.from_numpy(smplx_poses[40:55]),
            'camera_RT_params': torch.eye(4)[:3]
        }

        converted_motion[frame_key] = {
            'flame_coeffs': flame_coeffs,
            'smplx_coeffs': smplx_coeffs
        }

    # --- 5. Save Files ---
    os.makedirs(os.path.dirname(output_motion_path), exist_ok=True)
    print(f"\n--- Saving converted files ---")
    with open(output_motion_path, 'wb') as f:
        pickle.dump(converted_motion, f)
    print(f"Saved converted motion data to: {output_motion_path}")

    with open(output_identity_path, 'wb') as f:
        pickle.dump(identity_params, f)
    print(f"Saved identity parameters to: {output_identity_path}")


if __name__ == "__main__":
    convert_motion_data()