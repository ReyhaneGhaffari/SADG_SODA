import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import random

from Settings import get_args
from Network import build_models
from GMM import GMM
from Losses import (semantic_alignment_loss, adversarial_loss,
                    kl_divergence_loss, calibration_loss,
                    select_high_confidence, total_loss)
from Evaluation import evaluate, print_results, save_results, plot_tsne
from DataLoader import build_dataloaders, SPLIT_CONFIG


# ------------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------------
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# ------------------------------------------------------------------
# Source Training
# ------------------------------------------------------------------
def source_training(source_net, ccs, dataloader_train,
                    dataloader_val, args, device):
    """
    Train source network on labeled source domain data.
    Trains: SourceNet + ClosedSetClassifier (Ccs)

    source_net: SourceNet feature extractor
    ccs:        ClosedSetClassifier
    """
    print("\n" + "="*50)
    print("SOURCE TRAINING")
    print("="*50)

    # determine epochs based on dataset
    if 'kather16' in args.source.lower():
        num_epochs = args.source_epochs_k16
    else:
        num_epochs = args.source_epochs_k19

    # optimizer for source network + classifier
    optimizer = optim.Adam(
        list(source_net.parameters()) + list(ccs.parameters()),
        lr=args.source_lr
    )
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state = None

    for epoch in range(num_epochs):
        source_net.train()
        ccs.train()

        total_loss_val = 0.0
        correct = 0
        total = 0

        for imgs, labels in tqdm(dataloader_train,
                                  desc=f"Source Epoch {epoch+1}/{num_epochs}",
                                  leave=False):
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # forward pass
            zst = source_net(imgs)           # [N, 512]
            logits = ccs.fc(zst)             # [N, C] raw logits
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            total_loss_val += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = 100 * correct / total
        avg_loss = total_loss_val / len(dataloader_train)

        # validation
        val_acc = validate_source(source_net, ccs,
                                   dataloader_val, device)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {
                'source_net': source_net.state_dict(),
                'ccs': ccs.state_dict()
            }

        if (epoch + 1) % args.log_interval == 0:
            print(f"Epoch {epoch+1}/{num_epochs} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Train ACC: {train_acc:.2f}% | "
                  f"Val ACC: {val_acc:.2f}%")

    # restore best model
    if best_state:
        source_net.load_state_dict(best_state['source_net'])
        ccs.load_state_dict(best_state['ccs'])
        print(f"\nBest source model restored. Val ACC: {best_val_acc:.2f}%")

    # save checkpoint
    os.makedirs(args.save_path, exist_ok=True)
    torch.save(best_state, os.path.join(args.save_path,
                                         'source_model.pth'))
    print("Source model saved.")

    return source_net, ccs


def validate_source(source_net, ccs, dataloader, device):
    """Validate source model on closed-set validation data."""
    source_net.eval()
    ccs.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            zst = source_net(imgs)
            logits = ccs.fc(zst)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return 100 * correct / total if total > 0 else 0.0


# ------------------------------------------------------------------
# Target Adaptation
# ------------------------------------------------------------------
def target_adaptation(source_net, target_net, ccs, cos,
                      dataloader_train, dataloader_val,
                      args, device):
    """
    Adapt target network to unlabeled target domain.

    Step A: forward pass + semantic alignment loss Lsa
    Step B: prototype estimation + OOD scores + Ladv + Lkl + Lcal
    Step C: EM update at M/2 mini-batches

    source_net: frozen SourceNet (provides zst)
    target_net: TargetNet to be adapted (provides zt)
    ccs:        frozen ClosedSetClassifier
    cos:        OpenSetClassifier (trained)
    """
    print("\n" + "="*50)
    print("TARGET ADAPTATION")
    print("="*50)

    # determine epochs based on dataset
    if 'kather16' in args.target.lower():
        num_epochs = args.target_epochs_k16
    else:
        num_epochs = args.target_epochs_k19

    # freeze source net and closed-set classifier
    source_net.eval()
    for param in source_net.parameters():
        param.requires_grad = False
    for param in ccs.parameters():
        param.requires_grad = False

    # optimizer for target net feature extractor + open-set classifier
    optimizer_feat = optim.Adam(
        target_net.parameters(), lr=args.target_lr)
    optimizer_cos = optim.Adam(
        list(cos.parameters()), lr=args.target_lr)

    # initialize GMM
    gmm = GMM(K=args.K,
              em_iterations=args.em_iterations,
              purity_threshold=args.purity_threshold,
              beta=args.beta)

    # collect all target features for GMM initialization
    print("Initializing GMM with K-means warm-start...")
    all_zt = _collect_features(target_net, dataloader_train, device)
    gmm.initialize(all_zt)
    print(f"GMM initialized with K={args.K} components.")

    global_step = 0
    best_val_hos = 0.0
    best_state = None

    for epoch in range(num_epochs):
        target_net.train()
        cos.train()

        M = len(dataloader_train)
        half_M = M // 2

        epoch_loss = 0.0

        for b, (imgs, _) in enumerate(
                tqdm(dataloader_train,
                     desc=f"Adapt Epoch {epoch+1}/{num_epochs}",
                     leave=False)):

            imgs = imgs.to(device)
            N = imgs.size(0)

            # ------------------------------------------------
            # Step A: forward pass + semantic alignment Lsa
            # ------------------------------------------------
            optimizer_feat.zero_grad()
            optimizer_cos.zero_grad()

            # source features (frozen)
            with torch.no_grad():
                zst = source_net(imgs)          # [N, 512]
                pcs = ccs(zst)                  # [N, C]

            # target features
            zt = target_net(imgs)               # [N, 512]

            # semantic alignment loss
            Lsa = semantic_alignment_loss(zst, zt)

            # ------------------------------------------------
            # Step B: prototypes + OOD + losses
            # ------------------------------------------------
            # class prototypes
            with torch.no_grad():
                mu_c = gmm.compute_class_prototypes(
                    zt.detach(), zst, pcs)       # [C, 512]

            # openness weight
            w, Egm, Eop = gmm.openness_weight(
                zst, pcs, mu_c)                  # [N]

            # density posterior
            pg = gmm.density_posterior(
                zst, mu_c)                       # [N, C]

            # open-set classifier output
            pos_probs = cos(zt)                  # [N, C+1]
            pos_known = pos_probs[:, :-1]        # [N, C]
            pos_unknown = pos_probs[:, -1]       # [N]

            # KL divergence loss
            Lkl = kl_divergence_loss(pos_known, pg)

            # adversarial loss
            Ladv = adversarial_loss(pos_unknown, w)

            # high confidence sample selection
            high_conf_mask, TC = select_high_confidence(
                pcs, pos_probs, args.rho)

            # calibration loss
            logits = cos.get_logits(zt)[:, :-1]  # [N, C]
            Lcal = calibration_loss(
                logits, pg, cos.tau, high_conf_mask)

            # total loss
            L_feat, L_cos = total_loss(
                Lsa, Lkl, Ladv, args.lambda_adv)

            # update feature extractor
            L_feat.backward(retain_graph=True)
            optimizer_feat.step()

            # update open-set classifier
            optimizer_cos.zero_grad()
            L_cos_new = kl_divergence_loss(
                cos(zt.detach())[:, :-1], pg) + \
                args.lambda_adv * adversarial_loss(
                    cos(zt.detach())[:, -1], w)
            L_cos_new.backward()
            optimizer_cos.step()

            # update temperature
            optimizer_tau = optim.Adam([cos.tau], lr=args.target_lr)
            optimizer_tau.zero_grad()
            Lcal.backward()
            optimizer_tau.step()

            epoch_loss += L_feat.item()
            global_step += 1

            # ------------------------------------------------
            # Step C: EM update at M/2
            # ------------------------------------------------
            if b == half_M:
                with torch.no_grad():
                    all_zt = _collect_features(
                        target_net, dataloader_train, device)
                    all_zst = _collect_source_features(
                        source_net, dataloader_train, device)
                    all_pcs = _collect_pcs(
                        source_net, ccs,
                        dataloader_train, device)

                gmm.run_em(all_zt)
                gmm.refine_priors(all_zt, all_pcs)

        avg_loss = epoch_loss / M
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Loss: {avg_loss:.4f} | "
              f"tau: {cos.tau.item():.4f}")

    # save final adapted model
    os.makedirs(args.save_path, exist_ok=True)
    torch.save({
        'target_net': target_net.state_dict(),
        'cos': cos.state_dict()
    }, os.path.join(args.save_path, 'adapted_model.pth'))
    print("Adapted model saved.")

    return target_net, cos, gmm


# ------------------------------------------------------------------
# Helper: collect features
# ------------------------------------------------------------------
def _collect_features(target_net, dataloader, device):
    """Collect all target features zt from dataloader."""
    target_net.eval()
    all_zt = []
    with torch.no_grad():
        for imgs, _ in dataloader:
            imgs = imgs.to(device)
            zt = target_net(imgs)
            all_zt.append(zt.cpu())
    target_net.train()
    return torch.cat(all_zt).to(device)


def _collect_source_features(source_net, dataloader, device):
    """Collect all source features zst from dataloader."""
    all_zst = []
    with torch.no_grad():
        for imgs, _ in dataloader:
            imgs = imgs.to(device)
            zst = source_net(imgs)
            all_zst.append(zst.cpu())
    return torch.cat(all_zst).to(device)


def _collect_pcs(source_net, ccs, dataloader, device):
    """Collect all closed-set probabilities pcs."""
    all_pcs = []
    with torch.no_grad():
        for imgs, _ in dataloader:
            imgs = imgs.to(device)
            zst = source_net(imgs)
            pcs = ccs(zst)
            all_pcs.append(pcs.cpu())
    return torch.cat(all_pcs).to(device)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    args = get_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    print(f"\nSADG-SODA: Style-Aware Density-Guided SF-OSDA")
    print(f"Backbone: {args.backbone}")
    print(f"Source: {args.source} -> Target: {args.target}")
    print(f"Split: {args.split}")
    print(f"Device: {device}\n")

    # build models
    models = build_models(args)
    source_net = models['source_net'].to(device)
    target_net = models['target_net'].to(device)
    ccs = models['ccs'].to(device)
    cos = models['cos'].to(device)

    # build dataloaders
    dataloaders = build_dataloaders(args)

    # Stage 1: source training
    source_net, ccs = source_training(
        source_net, ccs,
        dataloaders['source_train'],
        dataloaders['source_val'],
        args, device
    )

    # copy source net weights to target net backbone
    # (target net starts from same backbone as source)
    target_net.backbone.load_state_dict(
        source_net.backbone.state_dict())
    target_net.fc_layers.load_state_dict(
        source_net.fc_layers.state_dict())
    print("Source weights transferred to target network.")

    # collect features before adaptation for t-SNE
    features_before = _collect_features(
        target_net, dataloaders['target_test'], device).cpu().numpy()

    # Stage 2: target adaptation
    target_net, cos, gmm = target_adaptation(
        source_net, target_net, ccs, cos,
        dataloaders['target_train'],
        dataloaders['target_val'],
        args, device
    )

    # collect features after adaptation for t-SNE
    features_after = _collect_features(
        target_net, dataloaders['target_test'], device).cpu().numpy()

    # evaluation on test set
    print("\nEvaluating on test set...")
    results = evaluate(
        target_net, ccs, cos,
        dataloaders['target_test'],
        args.num_class, device
    )

    direction = f"{args.source}→{args.target}"
    print_results(results, args.split, direction)

    # save results
    save_results(results, os.path.join(
        args.save_path, f'results_split{args.split}.txt'))

    # t-SNE visualization
    split_config = SPLIT_CONFIG[args.split]
    class_names = split_config['closed']
    true_labels = results['true_labels']

    plot_tsne(
        features_before, features_after,
        true_labels, args.num_class, class_names,
        save_path=os.path.join(
            args.save_path,
            f'tsne_split{args.split}.png')
    )

    print("\nDone.")
    return results


if __name__ == '__main__':
    main()
