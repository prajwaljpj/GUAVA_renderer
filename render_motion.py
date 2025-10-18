import os
import torch
import numpy as np
import imageio
import argparse
import subprocess
from scipy.interpolate import interp1d
from rich.progress import Progress
from rich.console import Console
from rich.panel import Panel

# --- Model and Utility Imports ---
from models.UbodyAvatar import Ubody_Gaussian, GaussianRenderer
from utils.general_utils import ConfigDict, add_extra_cfgs, to8b
from models.modules.ehm import EHM
from utils.camera_utils import get_full_proj_matrix
from pytorch3d.renderer import look_at_view_transform


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
    interp_func = interp1d(
        time_from, data, axis=0, kind="linear", fill_value="extrapolate"
    )

    # Apply the interpolation
    resampled_data = interp_func(time_to)

    return resampled_data


def synchronize_motion_data(
    console, smplx_data, flame_data, smplx_fps=30, flame_fps=25
):
    """
    Resamples FLAME data to match SMPL-X FPS and returns synchronized data.
    """
    console.print(
        Panel(
            "Synchronizing Motion Data",
            title="[bold cyan]Step 2.5[/bold cyan]",
            expand=False,
        )
    )
    target_fps = smplx_fps

    flame_data_resampled = {}
    for key, value in flame_data.items():
        # Only resample arrays that have a frame dimension
        if len(value.shape) > 1 and value.shape[0] > 1:
            resampled_value = resample_sequence(
                value, from_fps=flame_fps, to_fps=target_fps
            )
            console.print(
                f"  - Resampled [magenta]'{key}'[/magenta] from {value.shape[0]} frames to {resampled_value.shape[0]} frames."
            )
            flame_data_resampled[key] = resampled_value
        else:
            flame_data_resampled[key] = value  # Keep non-sequence data as is

    # Determine the number of frames to render (the minimum of the two sequences)
    num_frames = min(
        smplx_data["poses"].shape[0], flame_data_resampled["expression"].shape[0]
    )
    console.print(
        f"[bold green]✓[/bold green] Data synchronized. Rendering [yellow]{num_frames}[/yellow] frames."
    )

    return smplx_data, flame_data_resampled, num_frames


def render_motion(args):
    """
    Loads a pre-saved avatar, applies motion from SMPL-X and FLAME files,
    and renders the result to a video file.
    """
    console = Console()

    # --- Step 1: Setup ---
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    setup_info = f"Using device: [bold yellow]{device}[/bold yellow]"
    console.print(
        Panel(
            setup_info,
            title="[bold cyan]Step 1: Setup & Model Loading[/bold cyan]",
            expand=False,
        )
    )

    model_config_path = os.path.join(args.model_path, "config.yaml")
    meta_cfg = ConfigDict(model_config_path=model_config_path)
    meta_cfg = add_extra_cfgs(meta_cfg)
    console.print(f"  - Loaded config from [magenta]{model_config_path}[/magenta]")

    render_model = GaussianRenderer(meta_cfg.MODEL)
    ckpt_path = os.path.join(args.model_path, "checkpoints", "best_160000.pt")
    _state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    render_model.load_state_dict(_state["render_model"], strict=False)
    render_model.to(device)
    render_model.eval()
    console.print(f"  - Loaded renderer weights from [magenta]{ckpt_path}[/magenta]")

    if args.no_refiner:
        render_model.neural_refine = False
        console.print("  - [yellow]Neural Refiner DISABLED[/yellow]")

    ubody_gaussians = torch.load(args.avatar_path, map_location="cpu")
    ubody_gaussians.to(device)
    console.print(
        f"  - Loaded pre-built avatar from [magenta]{args.avatar_path}[/magenta]"
    )

    ehm_model = EHM(
        meta_cfg.MODEL.flame_assets_dir,
        meta_cfg.MODEL.smplx_assets_dir,
        add_teeth=meta_cfg.MODEL.add_teeth,
        uv_size=meta_cfg.MODEL.uvmap_size,
    ).to(device)
    ubody_gaussians.init_ehm(ehm_model.state_dict())
    ubody_gaussians.eval()
    console.print("  - Initialized EHM model.")
    console.print("[bold green]✓[/bold green] Setup complete.")

    # --- Step 2: Loading Motion Data ---
    console.print(
        Panel(
            f"SMPL-X: [magenta]{args.smplx_path}[/magenta]\nFLAME:  [magenta]{args.flame_path}[/magenta]",
            title="[bold cyan]Step 2: Loading Motion Data[/bold cyan]",
            expand=False,
        )
    )
    smplx_data = np.load(args.smplx_path)
    flame_data = np.load(args.flame_path)

    if args.sync_fps:
        smplx_data, flame_data, num_frames = synchronize_motion_data(
            console, smplx_data, flame_data
        )
    else:
        num_frames_smplx = smplx_data["poses"].shape[0]
        num_frames_flame = flame_data["expression"].shape[0]
        if num_frames_smplx != num_frames_flame:
            console.print(
                f"[yellow]WARNING:[/] Frame counts mismatch ([bold]{num_frames_smplx}[/] vs [bold]{num_frames_flame}[/]). Use --sync_fps or fix data."
            )
        num_frames = min(num_frames_smplx, num_frames_flame)
        console.print(
            f"FPS synchronization disabled. Rendering [yellow]{num_frames}[/yellow] frames."
        )

    smplx_shape = torch.tensor(
        smplx_data["betas"][:10], dtype=torch.float32, device=device
    ).unsqueeze(0)
    flame_shape_params = torch.zeros(1, 300, dtype=torch.float32, device=device)

    # --- Step 3: Camera Setup ---
    console.print(
        Panel(
            "Creating static camera",
            title="[bold cyan]Step 3: Setting up Static Camera[/bold cyan]",
            expand=False,
        )
    )
    camera_pos = (0, 0.2, 8.0)
    look_at = (0, 0.2, 0)
    up_vector = (0, 1, 0)
    image_size = int(args.resolution)
    tanfov = 1.0 / meta_cfg.MODEL.invtanfov

    R, T = look_at_view_transform(
        dist=torch.tensor([camera_pos[2]], dtype=torch.float32),
        elev=torch.tensor([0], dtype=torch.float32),
        azim=torch.tensor([0], dtype=torch.float32),
        at=torch.tensor([look_at], dtype=torch.float32),
        up=torch.tensor([up_vector], dtype=torch.float32),
        device=device,
    )

    w2c_cam_pytorch3d = torch.eye(4, device=device)
    w2c_cam_pytorch3d[:3, :3] = R
    w2c_cam_pytorch3d[:3, 3] = T

    c2c_mat = torch.tensor(
        [[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        dtype=torch.float32,
        device=device,
    )
    w2c_cam_renderer = torch.matmul(c2c_mat, w2c_cam_pytorch3d)
    view_matrix, full_proj_matrix = get_full_proj_matrix(w2c_cam_renderer, tanfov)
    c2w_cam_renderer = torch.inverse(w2c_cam_renderer)
    camera_center = c2w_cam_renderer[:3, 3]

    render_cam_params = {
        "world_view_transform": view_matrix.unsqueeze(0),
        "full_proj_transform": full_proj_matrix.unsqueeze(0),
        "tanfovx": torch.tensor([tanfov], device=device),
        "tanfovy": torch.tensor([tanfov], device=device),
        "image_height": torch.tensor([image_size], device=device),
        "image_width": torch.tensor([image_size], device=device),
        "camera_center": camera_center.unsqueeze(0),
    }
    console.print("[bold green]✓[/bold green] Camera created.")

    # --- Step 4: Rendering ---
    console.print(
        Panel(
            f"Rendering [yellow]{num_frames}[/yellow] frames...",
            title="[bold cyan]Step 4: Rendering Animation[/bold cyan]",
            expand=False,
        )
    )
    video_frames = []

    with Progress() as progress:
        task = progress.add_task("[green]Processing Frames...", total=num_frames)

        for i in range(num_frames):
            global_orient = torch.tensor(
                smplx_data["poses"][i, :3], dtype=torch.float32, device=device
            ).reshape(1, 1, 3)
            body_pose = torch.tensor(
                smplx_data["poses"][i, 3:66], dtype=torch.float32, device=device
            ).reshape(1, 21, 3)

            # Apply forearm motion scaling if requested
            if args.forearm_motion_scale != 1.0:
                # SMPL-X body_pose forearm joint indices:
                # 14: L-Elbow, 15: R-Elbow, 18: L-Wrist, 19: R-Wrist
                forearm_joint_indices = [14, 15, 18, 19]
                body_pose[0, forearm_joint_indices] *= args.forearm_motion_scale

            left_hand_pose = (
                torch.tensor(
                    smplx_data["poses"][i, 66:111], dtype=torch.float32, device=device
                ).reshape(1, 15, 3)
                * args.hand_motion_scale
            )
            right_hand_pose = (
                torch.tensor(
                    smplx_data["poses"][i, 111:156], dtype=torch.float32, device=device
                ).reshape(1, 15, 3)
                * args.hand_motion_scale
            )
            transl = torch.tensor(
                smplx_data["trans"][i], dtype=torch.float32, device=device
            ).unsqueeze(0)
            expression_params = torch.tensor(
                flame_data["expression"][i, :50], dtype=torch.float32, device=device
            ).unsqueeze(0)
            full_pose_params = flame_data["pose"][i]
            head_pose = torch.tensor(
                full_pose_params[:3], dtype=torch.float32, device=device
            ).unsqueeze(0)
            jaw_pose = torch.tensor(
                full_pose_params[3:6], dtype=torch.float32, device=device
            ).unsqueeze(0)
            neck_pose = torch.zeros(1, 3, dtype=torch.float32, device=device)

            target_info = {
                "smplx_coeffs": {
                    "shape": smplx_shape,
                    "global_orient": global_orient,
                    "body_pose": body_pose,
                    "left_hand_pose": left_hand_pose,
                    "right_hand_pose": right_hand_pose,
                    "transl": transl,
                    "exp": torch.zeros(1, 50, dtype=torch.float32, device=device),
                    "head_scale": torch.ones(1, 1, dtype=torch.float32, device=device),
                    "hand_scale": torch.ones(1, 1, dtype=torch.float32, device=device),
                    "joints_offset": torch.zeros(
                        1, 55, 3, dtype=torch.float32, device=device
                    ),
                },
                "flame_coeffs": {
                    "shape_params": flame_shape_params,
                    "expression_params": expression_params,
                    "jaw_params": jaw_pose,
                    "neck_pose_params": neck_pose,
                    "eye_pose_params": torch.zeros(
                        1, 6, dtype=torch.float32, device=device
                    ),
                    "pose_params": head_pose,
                    "eyelid_params": torch.zeros(
                        1, 2, dtype=torch.float32, device=device
                    ),
                },
            }

            with torch.no_grad():
                deform_gaussian_assets = ubody_gaussians(target_info)
                render_results = render_model(
                    deform_gaussian_assets, render_cam_params, bg=0.0
                )

            render_image = render_results["renders"][0]
            video_frames.append(
                to8b(render_image.permute(1, 2, 0).detach().cpu().numpy())
            )
            progress.update(task, advance=1)

    # --- Step 5: Saving Video ---
    console.print(
        Panel(
            "Saving final video...",
            title="[bold cyan]Step 5: Saving Video[/bold cyan]",
            expand=False,
        )
    )

    if args.audio_path:
        silent_video_path = args.output_path.replace(".mp4", "_silent.mp4")
        console.print(
            f"  - Saving silent video to temporary file: [magenta]{silent_video_path}[/magenta]"
        )
        imageio.mimwrite(silent_video_path, video_frames, fps=30, quality=8)

        console.print(f"  - Merging audio from: [magenta]{args.audio_path}[/magenta]")

        cmd = [
            "ffmpeg",
            "-i",
            silent_video_path,
            "-i",
            args.audio_path,
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            "-y",
            "-loglevel",
            "error",
            args.output_path,
        ]

        try:
            subprocess.run(cmd, check=True)
            console.print("  - [bold green]✓[/bold green] Audio merge successful.")
            os.remove(silent_video_path)
            console.print(
                f"  - Removed temporary file: [magenta]{silent_video_path}[/magenta]"
            )
        except FileNotFoundError:
            console.print(
                "[bold red]ERROR:[/] ffmpeg not found. Please install ffmpeg to merge audio."
            )
            console.print(
                f"The silent video was saved here: [magenta]{silent_video_path}[/magenta]"
            )
        except subprocess.CalledProcessError:
            console.print("[bold red]ERROR:[/] ffmpeg failed to merge the audio.")
            console.print(
                f"The silent video was saved here: [magenta]{silent_video_path}[/magenta]"
            )

    else:
        imageio.mimwrite(args.output_path, video_frames, fps=30, quality=8)

    console.print(
        Panel(
            f"[bold green]✅ Success! Final video saved to: [magenta]{args.output_path}[/magenta]",
            expand=False,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Render an animation from a pre-built avatar and motion data."
    )

    parser.add_argument(
        "--avatar_path",
        type=str,
        default="my_avatar.pt",
        help="Path to the pre-saved avatar file.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="assets/GUAVA",
        help="Path to the GUAVA model assets directory.",
    )

    parser.add_argument(
        "--smplx_path",
        type=str,
        required=True,
        help="Path to the SMPL-X motion data (.npz file).",
    )
    parser.add_argument(
        "--flame_path",
        type=str,
        required=True,
        help="Path to the resampled FLAME motion data (.npy file).",
    )
    parser.add_argument(
        "--audio_path",
        type=str,
        default=None,
        help="Path to the audio file to merge with the final video.",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default="render_output.mp4",
        help="Path to save the final rendered video.",
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default="512",
        choices=["512", "720", "1080", "2048"],
        help="Output video resolution. Options: 512, 720 (HD), 1080 (FHD), 2048 (2K).",
    )
    parser.add_argument(
        "--no_refiner",
        action="store_true",
        help="Disable the neural refiner post-processing step.",
    )
    parser.add_argument(
        "--sync_fps",
        action="store_true",
        help="Resample FLAME data from 25fps to match SMPL-X 30fps.",
    )
    parser.add_argument(
        "--hand_motion_scale",
        type=float,
        default=1.0,
        help="Scale factor to reduce hand motion (e.g., 0.5 for 50% reduction).",
    )
    parser.add_argument(
        "--forearm_motion_scale",
        type=float,
        default=1.0,
        help="Scale factor to reduce elbow and wrist motion (e.g., 0.5 for 50% reduction).",
    )

    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")
    render_motion(args)
