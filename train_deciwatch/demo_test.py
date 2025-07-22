import os
import torch
import numpy as np
from lib.models.deciwatch import DeciWatch


class NPYPoseDataset(torch.utils.data.Dataset):
    def __init__(self, pose_path, device='cuda'):
        pose = np.load(pose_path)  # (T, 14, 3)
        assert pose.shape[1:] == (14, 3), f"Expected shape (T, 14, 3), got {pose.shape}"
        self.data = torch.tensor(pose.reshape(pose.shape[0], -1), dtype=torch.float32).to(device)  # (T, 42)
        self.input_dimension = self.data.shape[1]

    def __getitem__(self, idx):
        return self.data[idx], 0  # dummy label

    def __len__(self):
        return len(self.data)

def run_deciwatch(model, x, device):
    """
    Run inference using the actual DeciWatch.forward(x, device).
    x: torch.Tensor, shape (1, L, C)
    Returns: numpy array, shape (L, C)
    """
    with torch.no_grad():
        recover, _ = model(x, device)  # only pass sequence and device
    return recover.squeeze(0).cpu().numpy()  # shape (L, C)

def run_full_sequence(model, full_sequence, sample_interval, slide_window_q, device):
    """
    Apply DeciWatch to a full pose sequence using sliding window inference.

    Args:
        model: DeciWatch model
        full_sequence: torch.Tensor of shape (T, C)
        sample_interval: int, e.g. 10
        slide_window_q: int, e.g. 10
        device: 'cuda' or 'cpu'

    Returns:
        final_output: np.ndarray of shape (T, C)
    """
    model.eval()
    T, C = full_sequence.shape
    window_size = sample_interval * slide_window_q + 1
    step = 1

    results = torch.zeros((T, C), dtype=torch.float32).to(device)
    counts = torch.zeros((T, C), dtype=torch.float32).to(device)

    for start in range(0, T - window_size + 1, step):
        input_window = full_sequence[start:start + window_size].unsqueeze(0)  # (1, W, C)
        with torch.no_grad():
            output_window = model(input_window, device)[0].squeeze(0)  # (W, C)
        results[start:start + window_size] += output_window
        counts[start:start + window_size] += 1

    # Handle division safely
    counts[counts == 0] = 1
    final_output = (results / counts).cpu().numpy()
    return final_output

def main():
    # === CONFIGURATION ===
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    PRETRAINED_MODEL = '/home/luthfai/Devel/skripsi/DeciWatch/results/21-06-2025_07-20-42_aist_spin/checkpoint.pth.tar'
    POSE_FILE = '/home/luthfai/Devel/skripsi/website/visualization/latest_pose.npy'
    OUTPUT_FILE = '/home/luthfai/Devel/skripsi/website/visualization/latest_pose_smooth.npy'


    # DeciWatch architecture config (from AIST baseline)
    config = {
        'sample_interval': 5,
        'slide_window_q': 10,
        'encoder_embedding_dim': 128,
        'decoder_embedding_dim': 128,
        'dropout': 0.1,
        'encoder_head': 4,
        'encoder_blocks': 5,
        'decoder': 'transformer',
        'decoder_interp': 'linear',
        'pre_norm': True
    }

    # === LOAD DATA ===
    dataset = NPYPoseDataset(POSE_FILE, device=DEVICE)

    # === BUILD MODEL ===
    model = DeciWatch(
        input_dim=dataset.input_dimension,
        sample_interval=config['sample_interval'],
        encoder_hidden_dim=config['encoder_embedding_dim'],
        decoder_hidden_dim=config['decoder_embedding_dim'],
        dropout=config['dropout'],
        nheads=config['encoder_head'],
        dim_feedforward=256,
        enc_layers=config['encoder_blocks'],
        dec_layers=config['encoder_blocks'],
        activation="leaky_relu",
        pre_norm=config['pre_norm'],
        recovernet_interp_method=config['decoder_interp'],
        recovernet_mode=config['decoder']
    ).to(DEVICE)

    # === LOAD WEIGHTS ===
    if os.path.isfile(PRETRAINED_MODEL):
        checkpoint = torch.load(PRETRAINED_MODEL)
        model.load_state_dict(checkpoint['state_dict'])
        print(f"✅ Loaded model from {PRETRAINED_MODEL}")
    else:
        raise FileNotFoundError(f"❌ Checkpoint not found at {PRETRAINED_MODEL}")

    # === RUN INFERENCE ===
    model.eval()
    # required_window = config['sample_interval'] * config['slide_window_q'] + 1
    # T = dataset.data.shape[0]

    # if T < required_window:
    #     raise ValueError(f"❌ Input sequence must be at least {required_window} frames long, got {T}.")
    
    # # Select first valid window
    # input_seq = dataset.data[:required_window].unsqueeze(0)  # shape: (1, L, C)

    # Inference
    output_seq = run_full_sequence(
        model,
        dataset.data,  # shape (T, 42)
        sample_interval=config['sample_interval'],
        slide_window_q=config['slide_window_q'],
        device=DEVICE
    )
    # Save
    np.save(OUTPUT_FILE, output_seq)
    print(f"✅ Saved smoothed output to {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
