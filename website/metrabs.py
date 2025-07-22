from fastapi import FastAPI, UploadFile
import uvicorn
import tensorflow as tf
import numpy as np
import cv2
import os

MODEL_PATH = "/home/luthfai/Downloads/metrabs_eff2l_y4_384px_800k_28ds"
model = tf.saved_model.load(MODEL_PATH)

app = FastAPI()

@app.post("/estimate")
async def estimate_pose(file: UploadFile):
    import numpy as np

    contents = await file.read()
    tmp_path = "tmp_video.mp4"
    with open(tmp_path, "wb") as f:
        f.write(contents)

    cap = cv2.VideoCapture(tmp_path)

    pose_sequence = []
    pose_sequence2 = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_tensor = tf.convert_to_tensor(frame_rgb, dtype=tf.uint8)
        preds = model.detect_poses(image_tensor, skeleton='lsp_14')
        preds1 = model.detect_poses(image_tensor, skeleton='coco_19')
        poses3d = preds['poses3d'].numpy()
        posehead = preds1['poses3d'].numpy()
        if poses3d.shape[0] > 0:
            pose_sequence.append(poses3d[0])
        if posehead.shape[0] > 0:
            pose_sequence2.append(posehead[0])
            
    cap.release()
    os.remove(tmp_path)

    if len(pose_sequence) == 0:
        return {"status": "no poses detected"}

    # Save as .npy
    pose_sequence = np.stack(pose_sequence)
    save_path = "visualization/latest_pose.npy"
    np.save(save_path, pose_sequence)
    save_path2 = "visualization/latest_pose_head.npy"
    np.save(save_path2, pose_sequence2)
    
    return {
        "status": "ok",
        "frames_detected": len(pose_sequence),
        "pose_file1": save_path,
        "pose_file2": save_path2
    }


