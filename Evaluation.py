import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import (roc_auc_score, accuracy_score,
                             f1_score, confusion_matrix)
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os


# ------------------------------------------------------------------
# Main Evaluation Function
# ------------------------------------------------------------------
def evaluate(target_net, ccs, cos, dataloader,
             num_classes, device, mode='test'):
    """
    Evaluate model on target domain test set.

    target_net: TargetNet feature extractor
    ccs:        ClosedSetClassifier
    cos:        OpenSetClassifier
    dataloader: target test dataloader
    num_classes: number of known classes C
    device:     torch device
    mode:       'test' or 'val'

    Returns dict with ACC, AUC, HOS, F1
    """
    target_net.eval()
    ccs.eval()
    cos.eval()

    all_features = []
    all_ood_scores = []   # p(C+1) for AUC
    all_predictions = []  # argmax for ACC
    all_true_labels = []  # ground truth

    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            # extract features
            zt = target_net(imgs)             # [N, 512]

            # open-set classifier output
            pos_probs = cos(zt)               # [N, C+1]

            # OOD score: p(C+1) — probability of being unknown
            ood_score = pos_probs[:, -1]      # [N]

            # final prediction: argmax over C+1 outputs
            predictions = torch.argmax(pos_probs, dim=1)  # [N]

            all_features.append(zt.cpu())
            all_ood_scores.append(ood_score.cpu())
            all_predictions.append(predictions.cpu())
            all_true_labels.append(labels.cpu())

    # concatenate all batches
    all_features = torch.cat(all_features).numpy()
    all_ood_scores = torch.cat(all_ood_scores).numpy()
    all_predictions = torch.cat(all_predictions).numpy()
    all_true_labels = torch.cat(all_true_labels).numpy()

    # binary labels for AUC: 1=unknown, 0=known
    binary_true = (all_true_labels == num_classes).astype(int)

    # ------------------------------------------------------------------
    # ACC: closed-set accuracy on known samples only
    # ------------------------------------------------------------------
    known_mask = all_true_labels < num_classes
    if known_mask.sum() > 0:
        acc = accuracy_score(
            all_true_labels[known_mask],
            all_predictions[known_mask]
        ) * 100
    else:
        acc = 0.0

    # ------------------------------------------------------------------
    # AUC: open-set detection using p(C+1) as continuous OOD score
    # Note: threshold swept internally by roc_auc_score for evaluation
    # only — final predictions still use argmax (no threshold needed)
    # ------------------------------------------------------------------
    if binary_true.sum() > 0 and (1 - binary_true).sum() > 0:
        auc = roc_auc_score(binary_true, all_ood_scores) * 100
    else:
        auc = 0.0

    # ------------------------------------------------------------------
    # HOS: harmonic mean of ACC and AUC
    # ------------------------------------------------------------------
    if acc + auc > 0:
        hos = 2 * acc * auc / (acc + auc)
    else:
        hos = 0.0

    # ------------------------------------------------------------------
    # F1: for unknown class detection
    # ------------------------------------------------------------------
    pred_binary = (all_predictions == num_classes).astype(int)
    f1 = f1_score(binary_true, pred_binary,
                  zero_division=0) * 100

    results = {
        'ACC':  round(acc, 2),
        'AUC':  round(auc, 2),
        'HOS':  round(hos, 2),
        'F1':   round(f1, 2),
        'features':    all_features,
        'true_labels': all_true_labels,
        'predictions': all_predictions,
        'ood_scores':  all_ood_scores
    }

    return results


# ------------------------------------------------------------------
# t-SNE Visualization
# ------------------------------------------------------------------
def plot_tsne(features_before, features_after,
              true_labels, num_classes,
              class_names, save_path=None):
    """
    Plot t-SNE visualization comparing before and after adaptation.

    features_before: [N, D] source model features on target domain
    features_after:  [N, D] adapted model features on target domain
    true_labels:     [N] ground truth labels
    num_classes:     number of known classes C
    class_names:     list of class names
    save_path:       path to save figure
    """
    print("Computing t-SNE embeddings...")

    tsne = TSNE(n_components=2, random_state=42,
                perplexity=30, max_iter=1000)

    # fit t-SNE on both feature sets
    emb_before = tsne.fit_transform(features_before)
    emb_after = tsne.fit_transform(features_after)

    # color map
    colors = plt.cm.tab10(np.linspace(0, 1, num_classes + 1))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    titles = ['Ground Truth',
              'Before Adaptation',
              'After Adaptation (SADG-SODA)']

    embeddings = [emb_before, emb_before, emb_after]

    for ax, emb, title in zip(axes, embeddings, titles):
        for c in range(num_classes):
            mask = true_labels == c
            ax.scatter(emb[mask, 0], emb[mask, 1],
                      c=[colors[c]], label=class_names[c],
                      alpha=0.6, s=10)
        # unknown samples
        unknown_mask = true_labels == num_classes
        if unknown_mask.sum() > 0:
            ax.scatter(emb[unknown_mask, 0], emb[unknown_mask, 1],
                      c='black', label='Unknown',
                      alpha=0.6, s=10, marker='x')

        ax.set_title(title, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])

    # shared legend
    handles = [mpatches.Patch(color=colors[c], label=class_names[c])
               for c in range(num_classes)]
    handles.append(mpatches.Patch(color='black', label='Unknown'))
    fig.legend(handles=handles, loc='lower center',
               ncol=num_classes + 1, fontsize=10,
               bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"t-SNE saved to {save_path}")
    else:
        plt.show()

    plt.close()


# ------------------------------------------------------------------
# Print Results
# ------------------------------------------------------------------
def print_results(results, split, direction):
    """
    Print evaluation results in a clean format.
    """
    print(f"\n{'='*50}")
    print(f"Split {split} | {direction}")
    print(f"{'='*50}")
    print(f"  ACC:  {results['ACC']:.2f}%")
    print(f"  AUC:  {results['AUC']:.2f}%")
    print(f"  HOS:  {results['HOS']:.2f}%")
    print(f"  F1:   {results['F1']:.2f}%")
    print(f"{'='*50}\n")


# ------------------------------------------------------------------
# Save Results
# ------------------------------------------------------------------
def save_results(results, save_path):
    """
    Save results to a text file.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        f.write(f"ACC:  {results['ACC']:.2f}\n")
        f.write(f"AUC:  {results['AUC']:.2f}\n")
        f.write(f"HOS:  {results['HOS']:.2f}\n")
        f.write(f"F1:   {results['F1']:.2f}\n")
    print(f"Results saved to {save_path}")


# ------------------------------------------------------------------
# Test
# ------------------------------------------------------------------
if __name__ == '__main__':
    from Settings import get_args
    from Network import build_models

    args = get_args()
    device = torch.device(args.device)

    models = build_models(args)
    target_net = models['target_net'].to(device)
    ccs = models['ccs'].to(device)
    cos = models['cos'].to(device)

    # simulate dummy dataloader
    from torch.utils.data import DataLoader, TensorDataset

    N = 100
    C = args.num_class
    dummy_imgs = torch.randn(N, 3, 224, 224)
    # mix of known (0..C-1) and unknown (C) labels
    dummy_labels = torch.cat([
        torch.randint(0, C, (int(N * 0.8),)),  # 80% known
        torch.full((int(N * 0.2),), C)          # 20% unknown
    ])
    dummy_ds = TensorDataset(dummy_imgs, dummy_labels)
    dummy_loader = DataLoader(dummy_ds, batch_size=16)

    # evaluate
    results = evaluate(
        target_net, ccs, cos,
        dummy_loader, C, device
    )

    print_results(results, split=args.split,
                  direction='K19->K16')

    # test t-SNE with dummy features
    N_tsne = 200
    features_before = np.random.randn(N_tsne, 50)
    features_after = np.random.randn(N_tsne, 50)
    true_labels = np.concatenate([
        np.random.randint(0, C, int(N_tsne * 0.8)),
        np.full(int(N_tsne * 0.2), C)
    ])
    class_names = ['TUM', 'STR', 'LYM', 'NORM']

    plot_tsne(features_before, features_after,
              true_labels, C, class_names,
              save_path='results/tsne.png')

    print("Evaluation tested successfully.")
