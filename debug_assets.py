import os
import torch
import numpy as np
import argparse
from plyfile import PlyData, PlyElement

# --- Model and Utility Imports ---
from models.UbodyAvatar import Ubody_Gaussian
from utils.general_utils import ConfigDict, add_extra_cfgs
from models.modules.ehm import EHM

def construct_debug_ply_attributes():
    """Helper function to create the list of attributes for a debug PLY file."""
    l = ['x', 'y', 'z']
    l.extend([f'red', 'green', 'blue'])
    l.append('opacity')
    l.extend([f'scale_0', 'scale_1', 'scale_2'])
    l.extend([f'rot_0', 'rot_1', 'rot_2', 'rot_3'])
    return l

def save_assets_to_ply(save_path, deformed_assets):
    """Saves the deformed Gaussian splat data to a .ply file for debugging."""
    print("Saving deformed Gaussian assets to .ply file for inspection...")
    
    with torch.no_grad():
        xyz = deformed_assets['xyz'][0].cpu().numpy()
        # It's crucial to apply sigmoid to get opacity in a human-readable [0,1] range
        opacities = torch.sigmoid(deformed_assets['opacity'][0]).cpu().numpy()
        scales = deformed_assets['scaling'][0].cpu().numpy()
        rotations = deformed_assets['rotation'][0].cpu().numpy()
        
        # The first 3 SH coefficients are the DC component (base color)
        # We apply sigmoid to bring them into a [0,1] range for visualization
        colors = torch.sigmoid(deformed_assets['features_color'][0, :, :3]).cpu().numpy()

    # --- Sanity Checks ---
    print("\n--- Sanity Checking Deformed Assets ---")
    print(f"XYZ min: {xyz.min():.3f}, max: {xyz.max():.3f}, mean: {xyz.mean():.3f}")
    print(f"Opacity min: {opacities.min():.3f}, max: {opacities.max():.3f}, mean: {opacities.mean():.3f}")
    print(f"Scale min: {scales.min():.3f}, max: {scales.max():.3f}, mean: {scales.mean():.3f}")
    print(f"Color min: {colors.min():.3f}, max: {colors.max():.3f}, mean: {colors.mean():.3f}")
    print("-------------------------------------\n")

    # Convert to structured array for PLY saving
    num_points = xyz.shape[0]
    vertices = np.empty(num_points, dtype=[
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
        ('opacity', 'f4'),
        ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
        ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4')
    ])

    vertices['x'] = xyz[:, 0]
    vertices['y'] = xyz[:, 1]
    vertices['z'] = xyz[:, 2]
    vertices['red'] = (colors[:, 0] * 255).astype('u1')
    vertices['green'] = (colors[:, 1] * 255).astype('u1')
    vertices['blue'] = (colors[:, 2] * 255).astype('u1')
    vertices['opacity'] = opacities[:, 0]
    vertices['scale_0'] = scales[:, 0]
    vertices['scale_1'] = scales[:, 1]
    vertices['scale_2'] = scales[:, 2]
    vertices['rot_0'] = rotations[:, 0]
    vertices['rot_1'] = rotations[:, 1]
    vertices['rot_2'] = rotations[:, 2]
    vertices['rot_3'] = rotations[:, 3]

    el = PlyElement.describe(vertices, 'vertex')
    PlyData([el]).write(save_path)
    print(f"✅ Success! Debug PLY saved to: {save_path}")


def debug_assets(args):
    """
    Loads an avatar, deforms it for the first frame, and saves the raw 
    Gaussian splat assets to a PLY file for external inspection.
    """
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"--- 1. Setup: Loading Models ---")
    print(f"Using device: {device}")

    # --- Load Config ---
    model_config_path = os.path.join(args.model_path, 'config.yaml')
    meta_cfg = ConfigDict(model_config_path=model_config_path)
    meta_cfg = add_extra_cfgs(meta_cfg)

    # --- Load Saved Avatar ---
    num_verts = 7922
    num_uv_points = 512
    placeholder_vertex_assets = {k: torch.zeros(1, num_verts, v, device=device) for k,v in {'scales':3, 'rotations':4, 'opacities':1, 'positions':3, 'static_offsets':3, 'colors':48}.items()}
    placeholder_uv_assets = {k: torch.zeros(1, num_uv_points, v, device=device) for k,v in {'scales':3, 'rotations':4, 'opacities':1, 'local_pos':3, 'face_bary':3, 'colors':48}.items()}
    placeholder_uv_assets['binding_face'] = torch.zeros(1, num_uv_points, 1, dtype=torch.long, device=device)

    ubody_gaussians = Ubody_Gaussian(meta_cfg.MODEL, placeholder_vertex_assets, placeholder_uv_assets, pruning=False)
    print(f"Loading pre-built avatar from: {args.avatar_path}")
    ubody_gaussians.load_state_dict(torch.load(args.avatar_path, map_location='cpu'), strict=False)
    
    # Re-initialize scales to a non-zero value
    with torch.no_grad():
        ubody_gaussians._smplx_scaling.data.fill_(0.01)
        ubody_gaussians._uv_scaling.data.fill_(0.01)

    print("Initializing EHM model...")
    ehm_model = EHM(meta_cfg.MODEL.flame_assets_dir, meta_cfg.MODEL.smplx_assets_dir, add_teeth=meta_cfg.MODEL.add_teeth, uv_size=meta_cfg.MODEL.uvmap_size)
    ubody_gaussians.init_ehm(ehm_model) 
    ubody_gaussians.to(device)
    ubody_gaussians.eval()

    print("\n--- 2. Loading Motion Data for First Frame ---")
    smplx_data = np.load(args.smplx_path)
    flame_data = np.load(args.flame_path)
    
    smplx_shape = torch.tensor(smplx_data['betas'][:10], dtype=torch.float32, device=device).unsqueeze(0)
    flame_shape_params = torch.zeros(1, 300, dtype=torch.float32, device=device)

    # --- Assemble the `target_info` for frame 0 ---
    i = 0
    global_orient = torch.tensor(smplx_data['poses'][i, :3], dtype=torch.float32, device=device).reshape(1, 1, 3)
    body_pose = torch.tensor(smplx_data['poses'][i, 3:66], dtype=torch.float32, device=device).reshape(1, 21, 3)
    left_hand_pose = torch.tensor(smplx_data['poses'][i, 66:111], dtype=torch.float32, device=device).reshape(1, 15, 3)
    right_hand_pose = torch.tensor(smplx_data['poses'][i, 111:156], dtype=torch.float32, device=device).reshape(1, 15, 3)
    transl = torch.zeros(1, 3, dtype=torch.float32, device=device) # Force to origin

    expression_params = torch.tensor(flame_data[i, :50], dtype=torch.float32, device=device).unsqueeze(0)
    jaw_pose = torch.tensor(flame_data[i, 100:103], dtype=torch.float32, device=device).unsqueeze(0)
    neck_pose = torch.tensor(flame_data[i, 103:106], dtype=torch.float32, device=device).unsqueeze(0)
    
    target_info = {
        'smplx_coeffs': {
            'shape': smplx_shape, 'global_orient': global_orient, 'body_pose': body_pose,
            'left_hand_pose': left_hand_pose, 'right_hand_pose': right_hand_pose, 'transl': transl,
            'exp': torch.zeros(1, 50, dtype=torch.float32, device=device),
            'head_scale': torch.ones(1, 1, dtype=torch.float32, device=device),
            'hand_scale': torch.ones(1, 1, dtype=torch.float32, device=device),
            'joints_offset': torch.zeros(1, 55, 3, dtype=torch.float32, device=device),
        },
        'flame_coeffs': {
            'shape_params': flame_shape_params, 'expression_params': expression_params,
            'jaw_params': jaw_pose, 'neck_pose_params': neck_pose,
            'eye_pose_params': torch.zeros(1, 6, dtype=torch.float32, device=device),
            'pose_params': torch.zeros(1, 3, dtype=torch.float32, device=device), 
            'eyelid_params': torch.zeros(1, 2, dtype=torch.float32, device=device),
        }
    }
    print("Motion data loaded.")

    print("\n--- 3. Deforming Avatar ---")
    with torch.no_grad():
        deform_gaussian_assets = ubody_gaussians(target_info)
    print("Deformation complete.")

    # --- 4. Save for Inspection ---
    save_assets_to_ply(args.output_path, deform_gaussian_assets)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug animation by saving the raw Gaussian splat assets to a PLY file.")
    
    parser.add_argument('--avatar_path', type=str, default='my_avatar.pt')
    parser.add_argument('--model_path', type=str, default='assets/GUAVA')
    parser.add_argument('--smplx_path', type=str, required=True)
    parser.add_argument('--flame_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, default='debug_assets.ply')
    
    args = parser.parse_args()
    
    torch.set_float32_matmul_precision('high')
    debug_assets(args)