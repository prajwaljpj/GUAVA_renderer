import torch
import numpy as np
import pickle

def inspect_dict(d, indent=0, name="root"):
    """Recursively prints the structure of a dictionary, including tensor/array shapes."""
    print('  ' * indent + f"Inspecting dict: '{name}'")
    for key, value in d.items():
        print('  ' * (indent + 1) + f"Key: '{key}'", end='')
        if isinstance(value, dict):
            print()
            inspect_dict(value, indent + 2, name=key)
        elif hasattr(value, 'shape'):
            # Squeeze to remove batch dimensions of 1 for easier comparison
            squeezed_shape = np.squeeze(value).shape
            print(f", Shape: {value.shape} -> Squeezed: {squeezed_shape}, Dtype: {getattr(value, 'dtype', 'N/A')}")
        else:
            print(f", Type: {type(value)}")

def detailed_verification():
    """
    Loads user's motion data and a GUAVA reference file to perform a
    detailed comparison of keys, shapes, and data structures.
    """
    # --- File Paths ---
    smplx_user_path = '/home/prajwaljpj/projects/talking_video/PantoMatrix/examples/motion_zoom/ElevenLabs_TTS_E_M_long_output.npz'
    flame_user_path = '/home/prajwaljpj/projects/talking_video/ARTalk/render_results/motion_data/ElevenLabs_TTS_E_M_long_motion_30fps.npy'
    guava_ref_path = 'assets/example/tracked_video/6gvP8f5WQyo__056/optim_tracking_ehm.pkl'

    print("=================================================")
    print("--- 1. Analyzing GUAVA Reference Data ---")
    print("=================================================")
    try:
        with open(guava_ref_path, 'rb') as f:
            guava_ref_data = pickle.load(f)
        
        # Data is a dict of frames. Let's inspect the first frame.
        first_frame_key = list(guava_ref_data.keys())[0]
        guava_frame_data = guava_ref_data[first_frame_key]
        
        print(f"Reference file loaded. Inspecting structure of first frame ('{first_frame_key}'):\n")
        inspect_dict(guava_frame_data)
        
        # Extract reference shapes for key parameters
        ref_smplx = guava_frame_data.get('smplx_coeffs', {})
        ref_flame = guava_frame_data.get('flame_coeffs', {})

    except Exception as e:
        print(f"Error loading or inspecting GUAVA reference file: {e}")
        return

    print("\n=================================================")
    print("--- 2. Analyzing User's Motion Data ---")
    print("=================================================")
    
    # --- User's SMPL-X Data ---
    print("\n--- User's SMPL-X data (PantoMatrix) ---")
    try:
        smplx_user_data = np.load(smplx_user_path)
        print(f"Keys: {list(smplx_user_data.keys())}")
        for key in smplx_user_data.keys():
            print(f"Key: '{key}', Shape: {smplx_user_data[key].shape}, Dtype: {smplx_user_data[key].dtype}")
        
        # --- Analyze Translation Data ---
        if 'trans' in smplx_user_data:
            translations = smplx_user_data['trans']
            min_trans = np.min(translations, axis=0)
            max_trans = np.max(translations, axis=0)
            print("\n--- Translation Analysis ---")
            print(f"Min translation (X, Y, Z): {min_trans}")
            print(f"Max translation (X, Y, Z): {max_trans}")
            print("--------------------------")

    except Exception as e:
        print(f"Error loading SMPL-X file: {e}")
        smplx_user_data = None

    # --- User's FLAME Data ---
    print("\n--- User's FLAME data (ARTalk, Resampled) ---")
    try:
        flame_user_data = np.load(flame_user_path)
        print(f"Data is a single array with Shape: {flame_user_data.shape}")
    except Exception as e:
        print(f"Error loading FLAME file: {e}")
        flame_user_data = None

    print("\n=================================================")
    print("--- 3. Detailed Comparison and Verdict ---")
    print("=================================================")

    # --- SMPL-X Comparison ---
    print("\n[SMPL-X CHECKLIST]")
    if smplx_user_data is not None:
        # global_orient + body_pose
        if 'poses' in smplx_user_data and smplx_user_data['poses'].shape[1] == 165:
             print("✅ Found 'poses' (shape 165), which contains:")
             print("   - global_orient (first 3 values)")
             print("   - body_pose (next 63 values)")
             print("   - hand poses, etc. (remaining values)")
        else:
            print("❌ CRITICAL: Missing 'poses' data or incorrect shape.")

        # transl
        if 'trans' in smplx_user_data and smplx_user_data['trans'].shape[1] == 3:
            print("✅ Found 'transl' (shape 3) for global translation.")
        else:
            print("❌ CRITICAL: Missing 'trans' data for translation.")
            
        # shape (betas)
        if 'betas' in smplx_user_data and smplx_user_data['betas'].shape[0] == 300:
             print("✅ Found 'betas' (shape 300) for body shape. We will use the first 10 as per SMPL-X standard.")
        else:
            print("⚠️ WARNING: Missing 'betas' for body shape. Will have to use a zero-vector.")
    else:
        print("❌ FAILED: Could not load user's SMPL-X data.")

    # --- FLAME Comparison ---
    print("\n[FLAME CHECKLIST]")
    if flame_user_data is not None:
        ref_expr_shape = np.squeeze(ref_flame.get('expression_params')).shape
        ref_jaw_shape = np.squeeze(ref_flame.get('jaw_pose')).shape
        
        if flame_user_data.shape[1] == 106:
            print(f"✅ Found FLAME data array (shape 106). Based on reference and standards, we will assume the following mapping:")
            print(f"   - expression_params (first 100 values) -> Target shape: {ref_expr_shape}")
            print(f"   - jaw_pose (next 3 values) -> Target shape: {ref_jaw_shape}")
            print(f"   - neck_pose (next 3 values)")
        else:
            print(f"❌ CRITICAL: User's FLAME data has shape {flame_user_data.shape}, which is not the expected (num_frames, 106).")
    else:
        print("❌ FAILED: Could not load user's FLAME data.")
        
    # --- Missing Components ---
    print("\n[OTHER ESSENTIALS CHECKLIST]")
    print("✅ Camera Parameters: Missing, as expected. We will create a static camera during rendering.")
    print("✅ FLAME Shape Parameters: Missing, as expected. We will use a zero-vector as a placeholder for an average face shape.")


if __name__ == "__main__":
    detailed_verification()