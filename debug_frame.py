import os
import torch
import numpy as np
import argparse
from plyfile import PlyData, PlyElement

# --- Model and Utility Imports ---
from models.UbodyAvatar import Ubody_Gaussian
from utils.general_utils import ConfigDict, add_extra_cfgs
from utils.camera_utils import get_full_proj_matrix
from models.modules.ehm import EHM
from utils.general_utils import inverse_sigmoid

def save_deformed_ply(save_path, deformed_assets):
    """Saves the deformed Gaussian splat data to a .ply file for debugging."""
    print("Saving deformed assets to .ply file...")
    
    xyz = deformed_assets['xyz'][0].detach().cpu().numpy()
    colors = deformed_assets['features_color'][0, :, :3].detach().cpu().numpy()

    print(f"Shape of xyz: {xyz.shape}")
    print(f"Shape of colors: {colors.shape}")

    if xyz.shape[0] != colors.shape[0]:
        print("!!! ERROR: Mismatch in number of vertices and colors.")
        min_len = min(xyz.shape[0], colors.shape[0])
        print(f"Truncating to the smaller size: {min_len}")
        xyz = xyz[:min_len]
        colors = colors[:min_len]

    # Create a simple PLY file with just vertices and colors
    verts_tuple = np.zeros((xyz.shape[0],), dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    colors_tuple = np.zeros((colors.shape[0],), dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    for i in range(xyz.shape[0]):
        verts_tuple[i] = tuple(xyz[i])
        colors_tuple[i] = tuple((colors[i] * 255).astype('u1'))

    vert_all = np.empty(xyz.shape[0], verts_tuple.dtype.descr + colors_tuple.dtype.descr)
    for prop in verts_tuple.dtype.names:
        vert_all[prop] = verts_tuple[prop]
    for prop in colors_tuple.dtype.names:
        vert_all[prop] = colors_tuple[prop]
        
    el = PlyElement.describe(vert_all, 'vertex')
    PlyData([el]).write(save_path)
    print(f"✅ Success! Deformed PLY saved to: {save_path}")


def debug_first_frame(args):
    """
    Loads the avatar and motion data for the first frame, deforms the avatar,
    and saves the result to a .ply file for inspection.
    """
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"--- 1. Setup: Loading Config and Models ---")
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
    save_deformed_ply(args.output_path, deform_gaussian_assets)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug the first frame of a motion sequence by saving its deformed geometry.")
    
    parser.add_argument('--avatar_path', type=str, default='my_avatar.pt')
    parser.add_argument('--model_path', type=str, default='assets/GUAVA')
    parser.add_argument('--smplx_path', type=str, required=True)
    parser.add_argument('--flame_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, default='deformed_frame_0.ply')
    
    args = parser.parse_args()
    
    torch.set_float32_matmul_precision('high')
    debug_first_frame(args)
