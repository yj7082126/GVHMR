import numpy as np
import torch
from pytorch3d.transforms import (
    axis_angle_to_matrix,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
)

from hmr4d.utils.geo.hmr_cam import create_camera_sensor
from hmr4d.utils.geo_transform import transform_mat, apply_T_on_points
from hmr4d.utils.geo.transforms import axis_rotate_to_matrix

def create_camera(w_root, width, focal, pitch_mean=5.0, pitch_std=22.5, 
                  roll_std=7.5, tz_range1=[3.0, 6.0]):

    # algo
    yaw = np.random.rand() * 2 * np.pi  # Look at any direction in xz-plane
    pitch = np.clip(np.random.randn() * pitch_std + pitch_mean, -90., 90)
    roll = np.clip(np.random.randn() * roll_std, -90., 90)  # Normal-dist

    # Note we use OpenCV's camera system by first applying R_y_upsidedown
    R_y_upsidedown = torch.tensor([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]).float()
    yaw_rm = axis_rotate_to_matrix(yaw, axis="y", use_deg=True)
    pitch_rm = axis_rotate_to_matrix(pitch, axis="x", use_deg=True)
    roll_rm = axis_rotate_to_matrix(roll, axis="z", use_deg=True)
    R_w2c = (roll_rm @ pitch_rm @ yaw_rm @ R_y_upsidedown).squeeze(0)  # (3, 3)

    # Place people in the scene
    tz = np.random.rand() * (tz_range1[1] - tz_range1[0]) + tz_range1[0]
    max_dist_in_fov = (width / 2) / focal * tz
    tx = (np.random.rand() * 2 - 1) * 0.7 * max_dist_in_fov
    ty = (np.random.rand() * 2 - 1) * 0.5 * max_dist_in_fov

    dist = torch.tensor([tx, ty, tz], dtype=torch.float)
    t_w2c = dist - torch.matmul(R_w2c, w_root)

    return R_w2c, t_w2c

def adjust_camera_for_rel_area(
    smplx_joints_w,  # (F, J, 3) or (J, 3)
    R0_w2c,          # (3, 3)
    t0_w2c,          # (3,)
    K,               # (3, 3)
    min_rel_area=0.005,  # 0.1%
    max_rel_area=0.50,   # 50%
    joint_indices=[9,12,13,14,15,16,17],   # list/1d tensor of joint ids
    is_backward_facing=-0.1,
    l_margin=0.3,
    t_margin=0.5,
    r_margin=0.7,
    b_margin=0.8,
    max_loops=3,
    **kwargs
):
    if smplx_joints_w.dim() == 2:
        smplx_joints_w = smplx_joints_w.unsqueeze(0)  # (1, J, 3)

    cx, cy = float(K[0, 2]), float(K[1, 2])
    img_w, img_h = int(round(cx * 2)), int(round(cy * 2))
    joint_indices = torch.as_tensor(joint_indices, device=smplx_joints_w.device)
    joints_w = smplx_joints_w[:, joint_indices]  # (B, M, 3)

    def project(R, t):
        joints_c = torch.einsum("ij,bkj->bki", R, joints_w) + t[None, None, :]
        z = joints_c[..., 2].clamp(min=1e-6)
        u = K[0, 0] * (joints_c[..., 0] / z) + K[0, 2]
        v = K[1, 1] * (joints_c[..., 1] / z) + K[1, 2]
        # print(torch.cat([u,v], dim=-1)[0].v)
        u_min, u_max = u.min(dim=1).values[0], u.max(dim=1).values[0]
        v_min, v_max = v.min(dim=1).values[0], v.max(dim=1).values[0]
        return u_min, u_max, v_min, v_max, z
    
    def human_look_direction_cam(
        joints_w,         # (J, 3) or (B, J, 3) in world coords
        R_w2c,            # (3, 3) world -> camera
    ):
        # Default indices for your 127-joint layout
        if joints_w.dim() == 2:
            joints_w = joints_w.unsqueeze(0)  # (1, J, 3)

        neck, jaw, re, le, nose = 12, 22, 56, 57, 55
        # Head direction in world space: from neck to face center
        face = (joints_w[:, nose] + joints_w[:, re] + joints_w[:, le] + joints_w[:, jaw]) / 4.0
        head_dir_w = face - joints_w[:, neck]
        head_dir_w = head_dir_w / (head_dir_w.norm(dim=-1, keepdim=True) + 1e-6)
        # Convert to camera space
        head_dir_c = torch.einsum("ij,bj->bi", R_w2c, head_dir_w)
        head_dir_c = head_dir_c / (head_dir_c.norm(dim=-1, keepdim=True) + 1e-6)

        return head_dir_c

    # 1) scale z to keep area in range
    for _ in range(max_loops):
        u_min, u_max, v_min, v_max, z = project(R0_w2c, t0_w2c)
        joint_w, joint_h = (u_max - u_min) , (v_max - v_min)
        percentage = (joint_w * joint_h) / float(img_w * img_h)
        print(f"Percentage : {percentage.item()*100:.2f}%")
        if percentage >= min_rel_area and percentage < max_rel_area:
            break
        
        scale = torch.sqrt(percentage / torch.clamp(percentage, min_rel_area, max_rel_area)).item()
        # scale > 1: too large -> move away; scale < 1: too small -> move closer
        t0_w2c[2] = t0_w2c[2] * scale
        
    # 2) if head faces away, rotate camera 180° around Y (camera space)
    head_dir_c = human_look_direction_cam(smplx_joints_w, R0_w2c)
    print(f"Head looking backwards camera: {head_dir_c[..., 2] > is_backward_facing} ({head_dir_c[..., 2].mean().item():.2f})")
    if head_dir_c[..., 2].mean() > is_backward_facing:
        print(f"Backward facing detected (z={head_dir_c[..., 2].mean().item():.2f}), flip camera")
        R_world = axis_angle_to_matrix(
            torch.tensor([0.0, np.pi, 0.0], device=R0_w2c.device)
        ).squeeze(0)
        # subject root in world (frame 0)
        P = smplx_joints_w[0, 0]  # pelvis/root joint in world
        # camera center in world
        C = -R0_w2c.transpose(0, 1) @ t0_w2c
        # rotate camera center about subject
        C_new = R_world @ (C - P) + P
        # rotate camera orientation (world rotation)
        R0_w2c = R0_w2c @ R_world.transpose(0, 1)
        # recompute translation from new camera center
        t0_w2c = -R0_w2c @ C_new
        head_dir_c = human_look_direction_cam(smplx_joints_w, R0_w2c)
        print(f"Head looking backwards camera: {head_dir_c[..., 2] > is_backward_facing} ({head_dir_c[..., 2].mean().item():.2f})")

    # 3) keep all joints within margin box (l/t/r/b), iterate up to max_loops
    left_bound = l_margin * img_w
    right_bound = r_margin * img_w
    top_bound = t_margin * img_h
    bottom_bound = b_margin * img_h
    for _ in range(max_loops):
        u_min, u_max, v_min, v_max, z = project(R0_w2c, t0_w2c)
    
        dx_px = 0.0
        if u_min < left_bound:
            dx_px = left_bound - u_min
        elif u_max > right_bound:
            dx_px = right_bound - u_max
        dy_px = 0.0
        if v_min < top_bound:
            dy_px = top_bound - v_min
        elif v_max > bottom_bound:
            dy_px = bottom_bound - v_max

        print(f"Margin adjusting: ({u_min:.2f}, {v_min:.2f}, {u_max:.2f}, {v_max:.2f}) dx_px={dx_px:.1f}, dy_px={dy_px:.1f}")
        if dx_px == 0.0 and dy_px == 0.0:
            break
        
        z_mean = z.mean()
        t0_w2c = t0_w2c.clone()
        t0_w2c[0] += dx_px * z_mean / K[0, 0]
        t0_w2c[1] += dy_px * z_mean / K[1, 1]

    return R0_w2c, t0_w2c

def noisy_interpolation(x, length, linspace=None, step_noise_perc=0.2):
    assert x.shape[0] == 2 and len(x.shape) == 2
    dim = x.shape[-1]
    output = np.zeros((length, dim))

    # Use linsapce(0, 1) +- noise as reference
    if linspace is None:
        linspace = np.linspace(0.0, 1.0, length).reshape(1, -1).repeat(dim, axis=0)  # (D, L)
        noise = (linspace[0, 1] - linspace[0, 0]) * step_noise_perc
        space_noise = np.random.uniform(-noise, noise, (dim, length - 2))  # (D, L-2)
        linspace[:, 1:-1] = linspace[:, 1:-1] + space_noise

    # Do 1d interp
    for i in range(dim):
        output[:, i] = np.interp(linspace[i], np.array([0.0, 1.0]), x[:, i])
    return output, linspace

def create_rotation_track(R, nframe, yaw=0, pitch=0, roll=0, linspace=None, step_noise_perc=0.2):
    r_xyz = np.deg2rad([pitch, yaw, roll])
    Rf = R @ axis_angle_to_matrix(torch.from_numpy(r_xyz).float())
    
    # Inbetweening two poses
    Rs = torch.stack((R, Rf))
    rs = matrix_to_rotation_6d(Rs).numpy()
    rs_move, linspace = noisy_interpolation(rs, nframe, linspace=linspace, step_noise_perc=step_noise_perc)
    R_move = rotation_6d_to_matrix(torch.from_numpy(rs_move).float())
    return R_move, linspace

def create_translation_track(R_w2c, t_w2c, nframe, tx=0, ty=0, tz=0, linspace=None, step_noise_perc=0.2):
    Ts = np.array([[0, 0, 0], [tx, ty, tz]])
    T_move, linspace = noisy_interpolation(Ts, nframe, linspace=linspace, step_noise_perc=step_noise_perc)
    T_move = torch.from_numpy(T_move).float()
    
    t_move = t_w2c + torch.einsum("ij,lj->li", R_w2c, T_move)
    return t_move, linspace

    
class CameraAugmenterV20:
    
    def __init__(self, w, h, f_fullframe=24):
        self.width, self.height, self.K_fullimg = create_camera_sensor(w, h, f_fullframe)
        self.yaw_std = 10.0
        self.pitch_std = 5.0
        self.roll_std = 2.5
        self.tx_std = 0.5
        self.ty_std = 0.2
        self.tz_std = 0.5
        self.step_noise_perc = 0.2
        
    def __call__(self, w_j3d, seed=None, **kwargs):
        nframe = w_j3d.shape[0]
        if seed is not None:
            np.random.seed(seed)
        
        R0_w2c, t0_w2c = create_camera(w_j3d[0,0], self.width, self.K_fullimg[0,0])
        R0_new_w2c, t0_new_w2c = adjust_camera_for_rel_area(
            w_j3d[:1], R0_w2c, t0_w2c, self.K_fullimg, **kwargs)

        yaw, pitch, roll = kwargs.get("yaw", None), kwargs.get("pitch", None), kwargs.get("roll", None)
        if yaw is None:
            yaw = (np.random.rand() * 2 - 1) * self.yaw_std # +/- 60°
        if pitch is None:
            pitch = (np.random.rand() * 2 - 1) * self.pitch_std  # +/- 30°
        if roll is None:
            roll = (np.random.rand() * 2 - 1) * self.roll_std  # +/- 30°
        print(f"Camera rotation deltas (yaw, pitch, roll): ({yaw:.1f}, {pitch:.1f}, {roll:.1f})")
        R_w2c, R_linspace = create_rotation_track(R0_new_w2c, nframe, 
            yaw=yaw, pitch=pitch, roll=roll, step_noise_perc=self.step_noise_perc)

        tx, ty, tz = kwargs.get("tx", None), kwargs.get("ty", None), kwargs.get("tz", None)
        if tx is None:
            tx = (np.random.rand() * 2 - 1) * self.tx_std # +/- 1.0
        if ty is None:
            ty = (np.random.rand() * 2 - 1) * self.ty_std  # +/- 0.25
        if tz is None:
            tz = (np.random.rand() * 2 - 1) * self.tz_std  # +/- 1.0
        print(f"Camera translation deltas (tx, ty, tz): ({tx:.2f}, {ty:.2f}, {tz:.2f})")
        t_w2c, t_linspace = create_translation_track(R0_new_w2c, t0_new_w2c, nframe, 
            tx=tx, ty=ty, tz=tz, step_noise_perc=self.step_noise_perc)
        T_w2c = transform_mat(R_w2c, t_w2c)

        meta = {
            'seed': seed,
            'R0_new_w2c': R0_new_w2c,
            't0_new_w2c': t0_new_w2c,
            'yaw': yaw, 'pitch': pitch, 'roll': roll,
            'tx': tx, 'ty': ty, 'tz': tz,
            'step_noise_perc': self.step_noise_perc,
        }
        return T_w2c, R_linspace, t_linspace, meta