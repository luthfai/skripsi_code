import numpy as np
import json

def compute_angles_and_save(pose_file, pose_file2, output_file):
    pose_sequence = np.load(pose_file)  # (T, 42)
    pose_sequence2 = np.load(pose_file2)  # (T, 42)
    smoothed = np.copy(pose_sequence2)
    window_size = 10
    half_window = window_size // 2

    num_frames = pose_sequence2.shape[0]

    # For each frame
    for i in range(num_frames):
        start = max(0, i - half_window)
        end = min(num_frames, i + half_window + 1)
        smoothed[i] = np.mean(pose_sequence2[start:end], axis=0)
        
    num_frames = pose_sequence.shape[0]
    pose_sequence = pose_sequence.reshape(num_frames, 14, 3)
    pose_sequence2 = smoothed.reshape(num_frames, 19, 3)

    def convert_numpy(obj):
        if isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(i) for i in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        else:
            return obj
    
    def clamp(value, min_value, max_value):
        return max(min_value, min(max_value, value))

    def compute_arm_left(P_shoulder, P_elbow, P_wrist):
        aup_L = P_elbow - P_shoulder
        alow_L = P_wrist - P_elbow

        LShoulderPitch = np.arctan2(-aup_L[0], -aup_L[2])  # (X, Z)
        LShoulderRoll = np.arctan2(np.sqrt(aup_L[0]**2 + aup_L[2]**2), aup_L[1])
        Ry = np.array([
            [np.cos(-LShoulderPitch), 0, np.sin(-LShoulderPitch)],
            [0, 1, 0],
            [-np.sin(-LShoulderPitch), 0, np.cos(-LShoulderPitch)]
        ])

        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(-LShoulderRoll), np.sin(-LShoulderRoll)],
            [0, -np.sin(-LShoulderRoll),  np.cos(-LShoulderRoll)]
        ])
        alow_L_prime = Rx @ Ry @ alow_L
        LElbowPitch = np.arctan2(-alow_L_prime[0], -alow_L_prime[2])
        LElbowRoll = np.arctan2(
            -np.sqrt(alow_L_prime[0]**2 + alow_L_prime[2]**2),
            alow_L_prime[1]
        )
        return {
            "ShoulderPitch": clamp(np.degrees(LShoulderPitch), -180, 180),
            "ShoulderRoll": clamp(np.degrees(LShoulderRoll), -100, 100),
            "ElbowTwist": clamp(np.degrees(LElbowPitch), -180, 180),
            "ElbowFlexion": clamp(np.degrees(LElbowRoll), -180, 110)  # Flexion positif ke bawah
        }

    def compute_arm_right(P_shoulder, P_elbow, P_wrist):
        aup_L = P_elbow - P_shoulder
        alow_L = P_wrist - P_elbow
        aup_L[1] = -aup_L[1]
        alow_L[1] = -alow_L[1]

        LShoulderPitch = np.arctan2(-aup_L[0], -aup_L[2])  # (X, Z)
        LShoulderRoll = np.arctan2(np.sqrt(aup_L[0]**2 + aup_L[2]**2), aup_L[1])
        Ry = np.array([
            [np.cos(-LShoulderPitch), 0, np.sin(-LShoulderPitch)],
            [0, 1, 0],
            [-np.sin(-LShoulderPitch), 0, np.cos(-LShoulderPitch)]
        ])

        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(-LShoulderRoll), np.sin(-LShoulderRoll)],
            [0, -np.sin(-LShoulderRoll),  np.cos(-LShoulderRoll)]
        ])
        alow_L_prime = Rx @ Ry @ alow_L
        LElbowPitch = np.arctan2(-alow_L_prime[0], -alow_L_prime[2])
        LElbowRoll = np.arctan2(
            -np.sqrt(alow_L_prime[0]**2 + alow_L_prime[2]**2),
            alow_L_prime[1]
        )
        
        return {
            "ShoulderPitch": clamp(-np.degrees(LShoulderPitch), -180, 180),
            "ShoulderRoll": clamp(-np.degrees(LShoulderRoll), -100, 100),
            "ElbowTwist": clamp(-np.degrees(LElbowPitch), -180, 180),
            "ElbowFlexion": clamp(np.degrees(LElbowRoll), -180, 110)
        }
        
    def compute_head(P_head, P_neck, P_lear, P_rear, p_eye):
        ahead = P_head - P_neck
        print(ahead)
        # Compute local coordinate frame
        z_axis = P_head - P_neck
        z_axis /= np.linalg.norm(z_axis)

        y_axis = P_lear - P_rear
        y_axis /= np.linalg.norm(y_axis)

        x_axis = np.cross(y_axis, z_axis)
        x_axis /= np.linalg.norm(x_axis)

        y_axis = np.cross(z_axis, x_axis)

        # Roll = arcsin of how much the ear line tilts in Z (local frame)
        HeadRoll = np.arcsin(y_axis[2])
        ear_to_ear = P_rear - P_lear

        HeadRoll = np.arctan2(-ear_to_ear[2],-ear_to_ear[1])
        
        nose_to_ear = P_head - ((P_lear + P_rear) / 2)
        HeadTilt = np.arctan2(-np.sqrt(nose_to_ear[0]**2 + nose_to_ear[1]**2), nose_to_ear[2]) +np.radians(92)
        HeadPan = np.arctan2(nose_to_ear[1], -nose_to_ear[0])

        return {
            "HeadPan": clamp(np.degrees(HeadPan), -120, 120),
            "HeadRoll": clamp(np.degrees(HeadRoll), -90, 90),
            "HeadTilt": clamp(np.degrees(HeadTilt), -90, 90)
        }
        
    results = {}

    for frame_idx in range(num_frames):
        pose = pose_sequence[frame_idx][:, [2, 0, 1]]  # swap
        pose_head = pose_sequence2[frame_idx][:, [2, 0, 1]]  # swap
        
        # Extract keypoints
        P1 = pose[9]
        P4 = pose[8]
        P7 = (pose[2] + pose[3]) / 2

        z_axis = (P1 + P4) / 2 - P7  # Up direction (torso)
        z_axis = z_axis / np.linalg.norm(z_axis)
        y_axis = P1 - P4             # Front direction (if shoulders are aligned front-back)
        y_axis = y_axis / np.linalg.norm(y_axis)
        x_axis = np.cross(y_axis, z_axis)  # Left to right
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)  # Re-orthogonalize y_axis
        y_axis = y_axis / np.linalg.norm(y_axis)
        A = np.stack([x_axis, y_axis, z_axis], axis=1)

        P1_L = A.T @ (pose[9] - P1)
        P2_L = A.T @ (pose[10] - P1)
        P3_L = A.T @ (pose[11] - P1)
        P4_R = A.T @ (pose[8] - P4)
        P5_R = A.T @ (pose[7] - P4)
        P6_R = A.T @ (pose[6] - P4)

        ### HEAD ###
        # === Key joints ===
        P_neck = pose_head[0]
        P_pelv = pose_head[2]
        P_lsho = pose_head[3]
        P_rsho = pose_head[9]

        # === Define coordinate axes ===
        z_axis = P_neck - P_pelv  # Up: torso direction
        z_axis = z_axis / np.linalg.norm(z_axis)

        y_axis = P_lsho - P_rsho  # Left to right: shoulder line
        y_axis = y_axis / np.linalg.norm(y_axis)

        x_axis = np.cross(z_axis, y_axis)  # Forward: perpendicular to up & left-right
        x_axis = x_axis / np.linalg.norm(x_axis)

        # Re-orthogonalize x_axis to ensure perfect right-handed frame
        y_axis = np.cross(x_axis, z_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)

        # === Build transformation matrix ===
        A_head = np.stack([x_axis, y_axis, z_axis], axis=1)  # shape: (3, 3)

        # === Transform points into head coordinate frame ===
        P_neck = A_head.T @ pose_head[0]
        P_head = A_head.T @ pose_head[1]
        P_lear = A_head.T @ pose_head[16]
        P_rear = A_head.T @ pose_head[18]
        P_eye = A_head.T @ pose_head[15]  # fixed: no unnecessary scaling

        angles_L = compute_arm_left(P1_L, P2_L, P3_L)
        angles_R = compute_arm_right(P4_R, P5_R, P6_R)
        angles_head= compute_head(P_head, P_neck, P_lear, P_rear, P_eye)
        
        frame_result = {
            "r_sho_pitch": angles_R["ShoulderPitch"],
            "l_sho_pitch": angles_L["ShoulderPitch"],
            "r_sho_roll": angles_R["ShoulderRoll"],
            "l_sho_roll": angles_L["ShoulderRoll"],
            "r_el_pitch": angles_R["ElbowTwist"],
            "l_el_pitch": angles_L["ElbowTwist"],
            "r_el_roll": angles_R["ElbowFlexion"],
            "l_el_roll": angles_L["ElbowFlexion"],
            "head_pan": angles_head["HeadPan"],
            "head_pitch": angles_head["HeadTilt"],
            "head_roll": angles_head["HeadRoll"],
        }

        results[str(frame_idx)] = frame_result

    with open(output_file, "w") as f:
        json.dump(convert_numpy(results), f, indent=2)

    print(f"Saved angles for {num_frames} frames to {output_file}")
