import os
import torch
import numpy as np
import argparse

# --- Model and Utility Imports ---
from models.UbodyAvatar import Ubody_Gaussian_inferer
from dataset import TrackedData_infer
from utils.general_utils import ConfigDict, add_extra_cfgs, find_pt_file
import copy
from omegaconf import OmegaConf

def debug_inferer(args):
    """
    Loads the Ubody_Gaussian_inferer, runs it on a single source image,
    and inspects the raw output to check for valid scale and color data.
    """
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"--- 1. Setup: Loading Inferer Model ---")
    print(f"Using device: {device}")

    # --- Load Config and Model ---
    model_config_path = os.path.join(args.model_path, 'config.yaml')
    meta_cfg = ConfigDict(model_config_path=model_config_path)
    meta_cfg = add_extra_cfgs(meta_cfg)
    
    infer_model = Ubody_Gaussian_inferer(meta_cfg.MODEL)
    
    ckpt_path = os.path.join(args.model_path, 'checkpoints')
    base_model_path = find_pt_file(ckpt_path, 'best')
    print(f"Loading checkpoint from: {base_model_path}")
    _state = torch.load(base_model_path, map_location='cpu', weights_only=True)
    infer_model.load_state_dict(_state['model'], strict=False)
    infer_model.to(device).eval()

    # --- 2. Load Source Data ---
    print("\n--- 2. Loading Source Identity from Dataset ---")
    meta_cfg_dataset = copy.deepcopy(meta_cfg)
    OmegaConf.set_readonly(meta_cfg_dataset.DATASET, False)
    meta_cfg_dataset.DATASET.data_path = args.data_path
    dataset = TrackedData_infer(cfg=meta_cfg_dataset, split='test', device=device, test_full=True)
    
    print(f"Loading source info for video ID: '{args.source_video_id}'")
    source_info = dataset._load_source_info(args.source_video_id)
    dataset._lmdb_engine.close()

    # --- 3. Run Inference ---
    print("\n--- 3. Running Inference ---")
    with torch.no_grad():
        vertex_gs_dict, up_point_gs_dict, _ = infer_model(source_info)
    print("Inference complete.")

    # --- 4. Sanity Check Raw Output ---
    print("\n--- 4. Sanity Checking Raw Inferer Output ---")
    scales = vertex_gs_dict['scales'][0].detach().cpu().numpy()
    colors = vertex_gs_dict['colors'][0].detach().cpu().numpy()
    
    print(f"Output Scales Shape: {scales.shape}")
    print(f"Output Colors Shape: {colors.shape}")
    
    print(f"\nScale min: {scales.min():.4f}, max: {scales.max():.4f}, mean: {scales.mean():.4f}")
    print(f"Color min: {colors.min():.4f}, max: {colors.max():.4f}, mean: {colors.mean():.4f}")
    
    # Check for variance
    if np.std(scales) < 1e-6:
        print("‼️ WARNING: Scales are all nearly identical.")
    else:
        print("✅ Scales have variance.")
        
    if np.std(colors) < 1e-6:
        print("‼️ WARNING: Colors are all nearly identical.")
    else:
        print("✅ Colors have variance.")
    print("---------------------------------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug the Ubody_Gaussian_inferer model.")
    
    parser.add_argument('--model_path', type=str, default='assets/GUAVA')
    parser.add_argument('--data_path', type=str, default='assets/example/tracked_video/6gvP8f5WQyo__056')
    parser.add_argument('--source_video_id', type=str, default='6gvP8f5WQyo__056')
    
    args = parser.parse_args()
    
    torch.set_float32_matmul_precision('high')
    debug_inferer(args)
