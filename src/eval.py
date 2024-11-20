import os
import numpy as np
import torch
from models import ASTModel
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, recall_score
import new_dataloader
from scipy.stats import norm


def calculate_d_prime(hit_rate, false_alarm_rate):
    """Calculate d-prime based on hit rate and false alarm rate."""
    # Avoid extremes for hit_rate and false_alarm_rate
    hit_rate = min(max(hit_rate, 1e-6), 1 - 1e-6)
    false_alarm_rate = min(max(false_alarm_rate, 1e-6), 1 - 1e-6)

    # Z-scores for hit rate and false alarm rate
    z_hit = norm.ppf(hit_rate)
    z_fa = norm.ppf(false_alarm_rate)

    # d-prime formula
    return z_hit - z_fa


def evaluate(audio_model_path, eval_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running evaluation on {device}")

    # Model input dimension
    input_tdim = 512

    # Initialize the model
    ast_mdl = ASTModel(label_dim=2, input_tdim=input_tdim, imagenet_pretrain=False, audioset_pretrain=False)

    # Load checkpoint
    if not os.path.exists(audio_model_path):
        raise FileNotFoundError(f"Checkpoint not found: {audio_model_path}")
    print(f'[*INFO] Loading checkpoint: {audio_model_path}')
    checkpoint = torch.load(audio_model_path, map_location=device)

    # Wrap model with DataParallel
    audio_model = torch.nn.DataParallel(ast_mdl)
    audio_model.load_state_dict(checkpoint)

    # Move to device
    audio_model = audio_model.to(device)
    audio_model.eval()

    # Initialize storage for predictions and targets
    A_predictions = []
    A_targets = []

    # No gradient computation during evaluation
    with torch.no_grad():
        for i, (audio_input, labels) in enumerate(eval_loader):
            audio_input = audio_input.to(device)
            labels = labels.to(device)

            # Forward pass
            audio_output = audio_model(audio_input)
            audio_output = torch.sigmoid(audio_output)

            # Collect predictions and labels
            A_predictions.append(audio_output.to('cpu').detach())
            A_targets.append(labels.to('cpu').detach())

    # Concatenate all batches
    predictions = torch.cat(A_predictions).numpy()
    targets = torch.cat(A_targets).numpy()

    # Convert predictions to binary and calculate metrics
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(targets, axis=1)

    acc = accuracy_score(true_labels, predicted_labels)
    auc = roc_auc_score(targets, predictions, multi_class='ovr')
    avg_precision = average_precision_score(targets, predictions)
    avg_recall = recall_score(true_labels, predicted_labels, average="macro")

    # d-prime calculation
    hit_rate = avg_recall
    false_alarm_rate = 1 - acc
    d_prime = calculate_d_prime(hit_rate, false_alarm_rate)

    # Print results
    print(f"Accuracy: {acc:.6f}")
    print(f"AUC: {auc:.6f}")
    print(f"Avg Precision: {avg_precision:.6f}")
    print(f"Avg Recall: {avg_recall:.6f}")
    print(f"d_prime: {d_prime:.6f}")

    return {
        "accuracy": acc,
        "AUC": auc,
        "Avg Precision": avg_precision,
        "Avg Recall": avg_recall,
        "d_prime": d_prime,
    }

def main():
    base = os.getcwd()
    data_path = os.path.join('..','egs','asvspoof2021' ,'datafiles', 'la_train_fold_1.csv')

    # Define audio configuration
    audio_conf = {
        'num_mel_bins': 128,
        'target_length': 512,
        'freqm': 24,
        'timem': 96,
        'mean': -3.8588,
        'std': 5.2843,
        'mixup': 0,
        'skip_norm': False,
        'mode': 'eval',
        'dataset': 'asvspoof2021',
        'noise': False
    }

    print("[INFO] Initializing DataLoader...")
    eval_loader = torch.utils.data.DataLoader(
        new_dataloader.AudiosetDataset(data_path, audio_conf=audio_conf),
        batch_size=24,  # Reduced batch size to manage memory
        shuffle=False, num_workers=8, pin_memory=True
    )

    print("[INFO] DataLoader ready.")

    # Path to the saved model
    audio_model_path = os.path.join(
        '..',
        'egs',
        'asvspoof2021',
        'exp',
        'test-asvspoof2021-f10-t10-impTrue-aspTrue-b12-lr1e-5',
        'fold1',
        'models',
        'best_audio_model.pth'
    )

    # Perform evaluation
    evaluate(audio_model_path, eval_loader)


if __name__ == '__main__':
    main()
