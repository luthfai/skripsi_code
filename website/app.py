import subprocess
from flask import Flask, render_template, request, jsonify, send_file
import os
import time
import matplotlib.pyplot as plt
import requests
import json
from pose_angles import compute_angles_and_save

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['video']
    if file:
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        return jsonify({"message": f"Uploaded {file.filename} successfully.", "filename": file.filename})
    return jsonify({"message": "No file uploaded."})


@app.route('/estimate', methods=['POST'])
def estimate_pose():
    filename = request.json.get('filename')
    filepath = os.path.join(UPLOAD_FOLDER, filename)

    
    with open(filepath, "rb") as f:
        response = requests.post(
            "http://127.0.0.1:8000/estimate",
            files={"file": f}
        )
    result = response.json()
    print(result)
    subprocess.run(
        [
            "/home/luthfai/miniconda3/envs/cv/bin/python3",
            "/home/luthfai/Devel/skripsi/DeciWatch/demo_test.py"
        ],
        check=True
    )
    
    subprocess.run(
        [
            "/home/luthfai/miniconda3/envs/cv/bin/python3",
            "pose_visualization.py",
            "--pose_file", "visualization/latest_pose_smooth.npy",
            "--fps", "3",
            "--skip", "10",
            "--output", "visualization/pose.gif"
        ]
    )   
    
    
    return jsonify({"message": "Pose estimation, smoothing, and visualization completed. "})

@app.route('/save_pose', methods=['POST'])
def save_pose():
    pose_file = "visualization/latest_pose_smooth.npy"
    output_json = "/home/luthfai/Devel/skripsi/website/pose_angles.json"
    pose_file2 = "visualization/latest_pose_head.npy"
    
    compute_angles_and_save(
        pose_file=pose_file,
        pose_file2=pose_file2,
        output_file=output_json
    )

    return jsonify({"message": f"Pose angles saved to {output_json}"})

@app.route('/move_robot', methods=['POST'])
def move_robot():
    robot_ip = "192.168.0.101"  # Replace with your robot's actual IP
    robot_port = 5000

    motion_json = "/home/luthfai/Devel/skripsi/website/pose_angles.json"

    with open(motion_json, 'r') as f:
        motion_data = json.load(f)

    try:
        response = requests.post(
            "http://{}:{}/play".format(robot_ip, robot_port),
            json=motion_data,
            timeout=5
        )

        if response.status_code == 200:
            return jsonify({"message": "Robot started playing motion."})
        elif response.status_code == 409:
            return jsonify({"message": "Robot is busy. Try again later."}), 409
        else:
            return jsonify({"message": "Unexpected response: {}".format(response.text)}), 500

    except requests.exceptions.RequestException as e:
        return jsonify({"error": str(e)}), 500


@app.route('/get_visualization')
def get_visualization():
    return send_file("visualization/pose.gif", mimetype='image/gif')

if __name__ == '__main__':
    app.run(debug=True)