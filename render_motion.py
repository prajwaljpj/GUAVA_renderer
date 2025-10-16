import os
import torch
import numpy as np
import imageio
import argparse
from tqdm import tqdm

# --- Model and Utility Imports ---
from models.UbodyAvatar import Ubody_Gaussian, GaussianRenderer
from utils.general_utils import ConfigDict, add_extra_cfgs, to8b
from models.modules.ehm import EHM
from utils.camera_utils import get_full_proj_matrix
from pytorch3d.renderer import look_at_view_transform

def render_motion(args):
    """
    Loads a pre-saved avatar, applies motion from SMPL-X and FLAME files,
    and renders the result to a video file.
    """
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"--- 1. Setup: Loading Config and Models ---")
    print(f"Using device: {device}")

    # --- Load Config ---
    model_config_path = os.path.join(args.model_path, 'config.yaml')
    meta_cfg = ConfigDict(model_config_path=model_config_path)
    meta_cfg = add_extra_cfgs(meta_cfg)

    # --- Load Renderer ---
    render_model = GaussianRenderer(meta_cfg.MODEL)
    ckpt_path = os.path.join(args.model_path, 'checkpoints', 'best_160000.pt')
    print(f"Loading renderer weights from: {ckpt_path}")
    _state = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    render_model.load_state_dict(_state['render_model'], strict=False)
    render_model.to(device)
    render_model.eval()

    # Conditionally disable the neural refiner
    if args.no_refiner:
        render_model.neural_refine = False
        print("--- Neural Refiner DISABLED ---")

    # --- Load Saved Avatar ---
    print(f"Loading pre-built avatar from: {args.avatar_path}")
    ubody_gaussians = torch.load(args.avatar_path, map_location='cpu')
    ubody_gaussians.to(device)
    
    print("Initializing EHM model...")
    ehm_model = EHM(meta_cfg.MODEL.flame_assets_dir, meta_cfg.MODEL.smplx_assets_dir, add_teeth=meta_cfg.MODEL.add_teeth, uv_size=meta_cfg.MODEL.uvmap_size).to(device)
    ubody_gaussians.init_ehm(ehm_model.state_dict())
    ubody_gaussians.eval()

    print("\n--- 2. Loading Motion Data ---")
    smplx_data = np.load(args.smplx_path)
    flame_data = np.load(args.flame_path)
    
    smplx_shape = torch.tensor(smplx_data['betas'][:10], dtype=torch.float32, device=device).unsqueeze(0)
    flame_shape_params = torch.zeros(1, 300, dtype=torch.float32, device=device)
    num_frames = smplx_data['poses'].shape[0]

    print("\n--- 3. Setting up Static Camera ---")
    # Use the correct, user-provided method to create a renderer-compatible camera.
    camera_pos = (0, 0.2, 8.0) # Lowered camera height and set distance to 8.0
    look_at = (0, 0.2, 0)      # Lowered look_at point to keep camera level
    up_vector = (0, 1, 0)
    image_size = int(args.resolution)
    tanfov = 1.0 / meta_cfg.MODEL.invtanfov

    # 1. Use look_at_view_transform to get camera extrinsics (R, T)
    R, T = look_at_view_transform(
        dist=torch.tensor([camera_pos[2]], dtype=torch.float32),
        elev=torch.tensor([0], dtype=torch.float32), # Keep camera level with look_at
        azim=torch.tensor([0], dtype=torch.float32),
        at=torch.tensor([look_at], dtype=torch.float32),
        up=torch.tensor([up_vector], dtype=torch.float32),
        device=device
    )

    # 2. Combine R and T into a world-to-camera matrix (w2c_cam)
    w2c_cam_pytorch3d = torch.eye(4, device=device)
    w2c_cam_pytorch3d[:3, :3] = R
    w2c_cam_pytorch3d[:3, 3] = T

    # 3. Apply the critical coordinate system flip
    c2c_mat = torch.tensor([
        [-1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=torch.float32, device=device)
    w2c_cam_renderer = torch.matmul(c2c_mat, w2c_cam_pytorch3d)

    # 4. Generate the final view and projection matrices
    view_matrix, full_proj_matrix = get_full_proj_matrix(w2c_cam_renderer, tanfov)

    # 5. Get the camera center in world space
    c2w_cam_renderer = torch.inverse(w2c_cam_renderer)
    camera_center = c2w_cam_renderer[:3, 3]

    # 6. Assemble the final dictionary with batch dimensions
    render_cam_params = {
        "world_view_transform": view_matrix.unsqueeze(0),
        "full_proj_transform": full_proj_matrix.unsqueeze(0),
        'tanfovx': torch.tensor([tanfov], device=device),
        'tanfovy': torch.tensor([tanfov], device=device),
        'image_height': torch.tensor([image_size], device=device),
        'image_width': torch.tensor([image_size], device=device),
        'camera_center': camera_center.unsqueeze(0)
    }
    print("Camera created.")

    print("\n--- 4. Rendering Animation Loop ---")
    video_frames = []

    for i in tqdm(range(num_frames)):
        global_orient = torch.tensor(smplx_data['poses'][i, :3], dtype=torch.float32, device=device).reshape(1, 1, 3)
        body_pose = torch.tensor(smplx_data['poses'][i, 3:66], dtype=torch.float32, device=device).reshape(1, 21, 3)
        left_hand_pose = torch.tensor(smplx_data['poses'][i, 66:111], dtype=torch.float32, device=device).reshape(1, 15, 3)
        right_hand_pose = torch.tensor(smplx_data['poses'][i, 111:156], dtype=torch.float32, device=device).reshape(1, 15, 3)
        transl = torch.tensor(smplx_data['trans'][i], dtype=torch.float32, device=device).unsqueeze(0)

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

        with torch.no_grad():
            deform_gaussian_assets = ubody_gaussians(target_info)
            render_results = render_model(deform_gaussian_assets, render_cam_params, bg=0.0)
        
        render_image = render_results['renders'][0]
        video_frames.append(to8b(render_image.permute(1, 2, 0).detach().cpu().numpy()))

    print("\n--- 5. Saving Video ---")
    imageio.mimwrite(args.output_path, video_frames, fps=30, quality=8)
    print(f"✅ Success! Animation saved to: {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render an animation from a pre-built avatar and motion data.")
    
    parser.add_argument('--avatar_path', type=str, default='my_avatar.pt', help="Path to the pre-saved avatar file.")
    parser.add_argument('--model_path', type=str, default='assets/GUAVA', help="Path to the GUAVA model assets directory.")
    
    parser.add_argument('--smplx_path', type=str, required=True, help="Path to the SMPL-X motion data (.npz file).")
    parser.add_argument('--flame_path', type=str, required=True, help="Path to the resampled FLAME motion data (.npy file).")
    
    parser.add_argument('--output_path', type=str, default='render_output.mp4', help="Path to save the final rendered video.")
    parser.add_argument('--resolution', type=str, default='512', choices=['512', '720', '1080', '2048'], help="Output video resolution. Options: 512, 720 (HD), 1080 (FHD), 2048 (2K).")
    parser.add_argument('--no_refiner', action='store_true', help="Disable the neural refiner post-processing step.")
    
    args = parser.parse_args()
    
    torch.set_float32_matmul_precision('high')
    render_motion(args)