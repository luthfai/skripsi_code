#!/usr/bin/env python3

import cv2
import tensorflow as tf
import numpy as np
import json
import time
import requests

# === Load model ===
model = tf.saved_model.load('/home/luthfai/Downloads/metrabs_eff2l_y4_384px_800k_28ds')

# === Use specific camera ===
cam_path = "/dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920_11573E6F-video-index0"
cap = cv2.VideoCapture(cam_path)
# === Robot endpoint ===
ROBOT_IP = "172.20.10.2"  # Replace with your robot's IP
ROBOT_URL = f"http://{ROBOT_IP}:5000/realtime"

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
        "ShoulderPitch": (np.degrees(LShoulderPitch)),
        "ShoulderRoll": (np.degrees(LShoulderRoll)),
        "ElbowTwist": (np.degrees(LElbowPitch)),
        "ElbowFlexion": (np.degrees(LElbowRoll))
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
        "ShoulderPitch": (-np.degrees(LShoulderPitch)),
        "ShoulderRoll": (-np.degrees(LShoulderRoll)),
        "ElbowTwist": (-np.degrees(LElbowPitch)),
        "ElbowFlexion": (np.degrees(LElbowRoll))
    }
    
def compute_head(P_head, P_neck, P_lear, P_rear, p_eye):
    ahead = P_head - P_neck
    # print(ahead)
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
        "HeadPan": (np.degrees(HeadPan)),
        "HeadRoll": (np.degrees(HeadRoll)),
        "HeadTilt": (np.degrees(HeadTilt)),
    }

print("Starting real-time pose estimation...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image")
        break

    # Convert to TensorFlow format
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_tensor = tf.convert_to_tensor(image_rgb)

    # Run inference
    preds = model.detect_poses(image_tensor, skeleton='lsp_14')
    preds_head = model.detect_poses(image_tensor, skeleton='coco_19')

    poses3d = preds['poses3d'].numpy()
    poses3d_head = preds_head['poses3d'].numpy()

    if len(poses3d) == 0 or len(poses3d_head) == 0:
        print("No person detected.")
        continue

    pose = poses3d[0][:, [2, 0, 1]]
    pose_head = poses3d_head[0][:, [2, 0, 1]]

    # === Arm IK ===
    P1 = pose[9]
    P4 = pose[8]
    P7 = (pose[2] + pose[3]) / 2

    z_axis = (P1 + P4) / 2 - P7
    z_axis = z_axis / np.linalg.norm(z_axis)
    y_axis = P1 - P4
    y_axis = y_axis / np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)

    A = np.stack([x_axis, y_axis, z_axis], axis=1)

    P1_L = A.T @ (pose[9] - P1)
    P2_L = A.T @ (pose[10] - P1)
    P3_L = A.T @ (pose[11] - P1)

    P4_R = A.T @ (pose[8] - P4)
    P5_R = A.T @ (pose[7] - P4)
    P6_R = A.T @ (pose[6] - P4)

    angles_L = compute_arm_left(P1_L, P2_L, P3_L)
    angles_R = compute_arm_right(P4_R, P5_R, P6_R)

    # === Head IK ===
    P_neck = pose_head[0]
    P_pelv = pose_head[2]
    P_lsho = pose_head[3]
    P_rsho = pose_head[9]

    z_axis = P_neck - P_pelv
    z_axis /= np.linalg.norm(z_axis)

    y_axis = P_lsho - P_rsho
    y_axis /= np.linalg.norm(y_axis)

    x_axis = np.cross(z_axis, y_axis)
    x_axis /= np.linalg.norm(x_axis)

    y_axis = np.cross(x_axis, z_axis)

    A_head = np.stack([x_axis, y_axis, z_axis], axis=1)

    P_neck = A_head.T @ pose_head[0]
    P_head = A_head.T @ pose_head[1]
    P_lear = A_head.T @ pose_head[16]
    P_rear = A_head.T @ pose_head[18]
    P_eye = A_head.T @ pose_head[15]

    angles_head = compute_head(P_head, P_neck, P_lear, P_rear, P_eye)

    # === Result ===
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

    # === Send to robot ===
    try:
        response = requests.post(ROBOT_URL, json=convert_numpy(frame_result), timeout=0.2)
        if response.status_code != 200:
            print(f"Robot error: {response.text}")
    except Exception as e:
        print(f"Failed to send to robot: {e}")

    # === Show webcam ===
    cv2.imshow("Webcam", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
