import os
import torch
import argparse
import lightning
import copy
from dataset import TrackedData_infer
from models.UbodyAvatar import Ubody_Gaussian_inferer, Ubody_Gaussian
from utils.general_utils import (
    ConfigDict, device_parser, 
    find_pt_file, add_extra_cfgs
)
from omegaconf import OmegaConf

def create_and_save_avatar(args):
    """
    Loads a pre-trained model, creates a Ubody_Gaussian avatar from a 
    single source image, and saves it to a file. Based on the original test.py.
    """
    print("--- 1. Loading Configuration and Models ---")
    
    # Use the config loading logic from the original script
    model_config_path = os.path.join(args.model_path, 'config.yaml')
    meta_cfg = ConfigDict(model_config_path=model_config_path)
    meta_cfg = add_extra_cfgs(meta_cfg)
    
    lightning.fabric.seed_everything(10)
    device = f'cuda:{args.devices[0]}'
    print(f"Using device: {device}")

    # This deepcopy is the key to making the config mutable
    meta_cfg = copy.deepcopy(meta_cfg)
    
    # Initialize the inference model
    infer_model = Ubody_Gaussian_inferer(meta_cfg.MODEL)
    infer_model.to(device)
    infer_model.eval()

    # Find and load the pre-trained checkpoint
    ckpt_path = os.path.join(args.model_path, 'checkpoints')
    base_model_path = find_pt_file(ckpt_path, 'best')
    if base_model_path is None:
        base_model_path = find_pt_file(ckpt_path, 'latest')
    
    assert os.path.exists(base_model_path), f"Checkpoint not found at {base_model_path}"
    print(f"Loading checkpoint from: {base_model_path}")
    _state = torch.load(base_model_path, map_location='cpu', weights_only=True)
    infer_model.load_state_dict(_state['model'], strict=False)

    print("--- 2. Loading Source Identity from Dataset ---")
    
    # Configure and load the dataset
    OmegaConf.set_readonly(meta_cfg.DATASET, False)
    meta_cfg.DATASET.data_path = args.data_path
    dataset = TrackedData_infer(cfg=meta_cfg, split='test', device=device, test_full=True)
    
    print(f"Loading source info for video ID: '{args.source_video_id}'")
    source_info = dataset._load_source_info(args.source_video_id)
    dataset._lmdb_engine.close()

    print("--- 3. Creating the Ubody Gaussian Avatar ---")
    with torch.no_grad():
        vertex_gs_dict, up_point_gs_dict, _ = infer_model(source_info)

    ubody_gaussians = Ubody_Gaussian(meta_cfg.MODEL, vertex_gs_dict, up_point_gs_dict, pruning=True)
    ubody_gaussians.init_ehm(infer_model.ehm.state_dict())
    ubody_gaussians.eval()

    print("--- 4. Saving Avatar to File ---")
    ubody_gaussians.to('cpu')
    torch.save(ubody_gaussians, args.output_avatar_path)
    
    print(f"\n✅ Success! Avatar has been saved to: {args.output_avatar_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Hardcoded defaults for our specific use case
    parser.add_argument('--model_path', '-m', type=str, default='assets/GUAVA')
    parser.add_argument('--data_path', type=str, default='assets/example/tracked_video/6gvP8f5WQyo__056')
    parser.add_argument('--source_video_id', type=str, default='6gvP8f5WQyo__056')
    parser.add_argument('--output_avatar_path', type=str, default='my_avatar.pt')
    parser.add_argument('--devices', '-d', default='0', type=str)
    
    args = parser.parse_args()
    
    torch.set_float32_matmul_precision('high')
    create_and_save_avatar(args)
