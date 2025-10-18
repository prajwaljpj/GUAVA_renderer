import torch
import numpy as np
from scipy.interpolate import interp1d
import os

def resample_flame_to_smplx_rate():
    """
    Loads 25 FPS FLAME data and 30 FPS SMPL-X data, and resamples the
    FLAME motion to match the SMPL-X frame rate and duration.
    """
    flame_path = '/home/prajwaljpj/projects/talking_video/ARTalk/render_results/motion_data/ElevenLabs_TTS_E_M_long_motion.pt'
    smplx_path = '/home/prajwaljpj/projects/talking_video/PantoMatrix/examples/motion_zoom/ElevenLabs_TTS_E_M_long_output.npz'
    output_path = '/home/prajwaljpj/projects/talking_video/ARTalk/render_results/motion_data/ElevenLabs_TTS_E_M_long_motion_30fps.npy'

    print("--- Loading Motion Data ---")
    try:
        flame_data_25fps = torch.load(flame_path, map_location='cpu').numpy()
        smplx_data = np.load(smplx_path)
        smplx_poses = smplx_data['poses']
        print(f"Original FLAME data shape: {flame_data_25fps.shape}")
        print(f"Original SMPL-X poses shape: {smplx_poses.shape}")
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    # --- Timeline Calculation ---
    flame_fps = 25
    smplx_fps = 30
    
    num_frames_flame = flame_data_25fps.shape[0]
    num_frames_smplx = smplx_poses.shape[0]

    # Original timestamps for the FLAME data (x-axis for interpolation)
    # e.g., [0.0, 0.04, 0.08, ...]
    original_flame_timestamps = np.linspace(0, (num_frames_flame - 1) / flame_fps, num_frames_flame)

    # Target timestamps based on the SMPL-X timeline (new x-axis)
    # e.g., [0.0, 0.033, 0.066, ...]
    target_smplx_timestamps = np.linspace(0, (num_frames_smplx - 1) / smplx_fps, num_frames_smplx)

    print(f"\nFLAME duration: {original_flame_timestamps[-1]:.2f} seconds")
    print(f"SMPL-X duration: {target_smplx_timestamps[-1]:.2f} seconds")
    
    print("\n--- Resampling FLAME data from 25 FPS to 30 FPS ---")

    # --- Interpolation ---
    # Create an interpolation function. We do this for each of the 106 FLAME parameters.
    # 'linear' interpolation is fast and usually sufficient for motion data.
    # 'axis=0' tells scipy to interpolate along the time dimension (the frames).
    # `bounds_error=False` and `fill_value='extrapolate'` handle cases where
    # the target timeline might be slightly longer than the source.
    interp_func = interp1d(
        original_flame_timestamps, 
        flame_data_25fps, 
        kind='linear', 
        axis=0, 
        bounds_error=False, 
        fill_value='extrapolate'
    )

    # Apply the function to the target timestamps to get the new, resampled data
    flame_data_30fps = interp_func(target_smplx_timestamps)

    print(f"New resampled FLAME data shape: {flame_data_30fps.shape}")

    # --- Verification and Saving ---
    if flame_data_30fps.shape[0] == num_frames_smplx:
        print("✅ Resampling successful. Frame counts now match.")
        np.save(output_path, flame_data_30fps)
        print(f"Resampled data saved to: {output_path}")
    else:
        print("❌ Error: Resampled frame count does not match.")

if __name__ == "__main__":
    resample_flame_to_smplx_rate()
