import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import os
import argparse
import imageio

parser = argparse.ArgumentParser()
parser.add_argument("--pose_file", required=True, help="Path to .npy pose file")
parser.add_argument("--output", default="visualization/pose.gif", help="Output GIF file")
parser.add_argument("--fps", type=int, default=5, help="Frames per second")
parser.add_argument("--skip", type=int, default=5, help="Skip every N frames (1 = no skipping)")
args = parser.parse_args()

# Load pose sequence
pose_sequence = np.load(args.pose_file)
num_frames = pose_sequence.shape[0]

# Reshape if needed
if pose_sequence.ndim == 3 and pose_sequence.shape[1:] == (14, 3):
    pass
elif pose_sequence.ndim == 2 and pose_sequence.shape[1] == 42:
    pose_sequence = pose_sequence.reshape(num_frames, 14, 3)
else:
    raise ValueError(f"Unexpected pose shape: {pose_sequence.shape}")

# Define skeleton connectivity (lsp_14)
connections = [
    (0, 1), (1, 2), (5, 4), (4, 3),
    (2, 3), (8, 9), (2, 8), (3, 9),
    (6, 7), (7, 8), (11, 10), (10, 9), (12, 13)
]

# Compute global axis limits
all_coords = pose_sequence.reshape(-1, 3)
all_coords = np.stack([
    all_coords[:, 2],       # X
    all_coords[:, 0],       # Z -> Y
    -all_coords[:, 1]       # -Y -> Z
], axis=-1)

x_center = np.mean(all_coords[:,0])
y_center = np.mean(all_coords[:,1])
z_center = np.mean(all_coords[:,2])

max_range = np.max(np.ptp(all_coords, axis=0)) / 2
xlim = (x_center - max_range, x_center + max_range)
ylim = (y_center - max_range, y_center + max_range)
zlim = (z_center - max_range, z_center + max_range)

frames = []
selected_indices = list(range(0, num_frames, args.skip))

print(f"Rendering {len(selected_indices)} frames for GIF...")

for i in selected_indices:
    fig = plt.figure(figsize=(4,4))
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=30, azim=135)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)
    ax.set_box_aspect([1,1,1])
    ax.set_title(f"Frame {i+1}/{num_frames}")

    joints = pose_sequence[i]
    for (j1, j2) in connections:
        x = [joints[j1,2], joints[j2,2]]
        y = [joints[j1,0], joints[j2,0]]
        z = [-joints[j1,1], -joints[j2,1]]
        ax.plot(x, y, z, c='blue', lw=2)
    ax.scatter(joints[:,2], joints[:,0], -joints[:,1], c='red', s=20)

    canvas.draw()
    img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    frames.append(img)

    plt.close(fig)

# Save GIF
print(f"Saving GIF to {args.output}")
imageio.mimsave(
    args.output,
    frames,
    fps=args.fps,
    loop=0,
    duration=1.0 / args.fps
)
print("GIF saved successfully.")
