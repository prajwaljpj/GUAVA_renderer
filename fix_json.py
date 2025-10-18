import pickle
import os
import copy

def fix_motion_file():
    """
    Loads a reference motion file and a custom motion file, then
    injects the custom motion into the reference data structure,
    preserving all other data like camera parameters.
    """
    reference_motion_path = 'assets/example/tracked_video/6gvP8f5WQyo__056/optim_tracking_ehm.pkl'
    custom_motion_path = 'assets/example/custom_video/my_animation2/optim_tracking_custom.pkl'
    output_path = 'assets/example/custom_video/my_animation2/optim_tracking_ehm_fixed.pkl'

    print("--- Loading Motion Files ---")
    try:
        with open(reference_motion_path, 'rb') as f:
            reference_data = pickle.load(f)
        print(f"Loaded reference motion from: {reference_motion_path}")

        with open(custom_motion_path, 'rb') as f:
            custom_data = pickle.load(f)
        print(f"Loaded custom motion from: {custom_motion_path}")

    except FileNotFoundError as e:
        print(f"Error: Could not find file - {e}")
        return
    except Exception as e:
        print(f"An error occurred while loading files: {e}")
        return

    # Create a deep copy to avoid modifying the original data
    fixed_data = copy.deepcopy(reference_data)

    print("\n--- Injecting Custom Motion ---")
    custom_frames = sorted(custom_data.keys())
    reference_frames = sorted(reference_data.keys())
    
    num_frames_to_process = min(len(custom_frames), len(reference_frames))
    print(f"Reference has {len(reference_frames)} frames. Custom motion has {len(custom_frames)} frames.")
    print(f"Processing {num_frames_to_process} frames.")

    for i in range(num_frames_to_process):
        frame_key = custom_frames[i]
        ref_frame_key = reference_frames[i] # Use the corresponding key from the reference

        if ref_frame_key in fixed_data:
            # Replace only the smplx and flame motion coefficients
            fixed_data[ref_frame_key]['smplx_coeffs'] = custom_data[frame_key]['smplx_coeffs']
            fixed_data[ref_frame_key]['flame_coeffs'] = custom_data[frame_key]['flame_coeffs']
        else:
            print(f"Warning: Frame key '{ref_frame_key}' not found in reference data. Skipping.")

    # --- Save the new motion file ---
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(fixed_data, f)
        print(f"\nSuccessfully saved fixed motion data to: {output_path}")
    except Exception as e:
        print(f"An error occurred while saving the file: {e}")


if __name__ == "__main__":
    fix_motion_file()
