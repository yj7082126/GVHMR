import numpy as np
import matplotlib.pyplot as plt
import torch
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle

def plot_angle_changes(global_orient_aa,     
    fps=30,
    angle_unit="deg",
    threshold_deg=30.0,
    title="SMPL-X Global Orientation Abrupt Changes"):

    aa = global_orient_aa * 180.0 / np.pi
    aa = aa.cpu().numpy()

    R = axis_angle_to_matrix(global_orient_aa)  # (T,3,3)
    R_rel = torch.matmul(R[:-1].transpose(-1, -2), R[1:])  # (T-1,3,3)
    aa_rel = matrix_to_axis_angle(R_rel)  # (T-1,3)
    angles = torch.norm(aa_rel, dim=1)  # radians

    if angle_unit == "deg":
        angles = angles * 180.0 / np.pi
        threshold = threshold_deg
        y_label = "Rotation Change (degrees)"
    else:
        threshold = np.deg2rad(threshold_deg)
        y_label = "Rotation Change (radians)"
    angles = angles.cpu().numpy()
    
    fig, ax = plt.subplots(2, 1, figsize=(12, 9))

    time = np.arange(len(angles)) / fps
    ax[0].plot(time, angles, label="Δ Global Orient", linewidth=2)
    ax[0].axhline(threshold, color="r", linestyle="--", label=f"Threshold ({30}°)")

    # Highlight abrupt frames
    abrupt_idx = np.where(angles > threshold)[0]
    ax[0].scatter(time[abrupt_idx], angles[abrupt_idx], color="red", zorder=5)

    ax[0].set_xlabel("Time (seconds)")
    ax[0].set_ylabel("Rotation Change (degrees)")
    ax[0].set_title("Abrupt Rotation Changes")
    ax[0].legend()
    ax[0].grid(True)

    time = np.arange(len(aa)) / fps
    ax[1].plot(time, aa[:, 0], label="X")
    ax[1].plot(time, aa[:, 1], label="Y")
    ax[1].plot(time, aa[:, 2], label="Z")

    ax[1].set_xlabel("Time (seconds)")
    ax[1].set_ylabel("Axis-Angle Value (radians)")
    ax[1].set_title("SMPL-X Global Orientation Components")
    ax[1].legend()
    ax[1].grid(True)

    fig.tight_layout()
    plt.show()

def save_plot(path, **kw):
    """Save the current figure without any white margin."""
    plt.savefig(path, bbox_inches="tight", pad_inches=0, **kw)
