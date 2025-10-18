#!/usr/bin/env python3

import pickle


def inspect_pkl(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)

    first_frame_key = sorted(data.keys())[0]
    print(f"\n--- Inspecting {file_path} ---")
    print(f"Total frames: {len(data)}")
    print(f"First frame key: '{first_frame_key}'")

    flame_coeffs = data[first_frame_key].get("flame_coeffs", {})
    print("FLAME Coeffs Keys:", flame_coeffs.keys())

    if "jaw_params" in flame_coeffs:
        print("'jaw_params' shape:", flame_coeffs["jaw_params"].shape)
        print("'jaw_params' values:", flame_coeffs["jaw_params"])
    if "jaw_pose" in flame_coeffs:  # The reference file might use this key
        print("'jaw_pose' shape:", flame_coeffs["jaw_pose"].shape)
        print("'jaw_pose' values:", flame_coeffs["jaw_pose"])
    if "expression_params" in flame_coeffs:
        print("'expression_params' shape:", flame_coeffs["expression_params"].shape)
        # print just a slice to keep it readable
        print(
            "'expression_params' values (first 5):",
            flame_coeffs["expression_params"][:5],
        )


# --- Paths ---
your_file = "assets/example/custom_video/my_animation/optim_tracking_ehm_fixed.pkl"
reference_file = "assets/example/tracked_video/6gvP8f5WQyo__056/optim_tracking_ehm.pkl"

inspect_pkl(your_file)
inspect_pkl(reference_file)
