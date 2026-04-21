import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


# ------------------------------------------------------------------
# Gradient Reversal Layer
# ------------------------------------------------------------------
class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.save_for_backward(torch.tensor(lambd))
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        lambd = ctx.saved_tensors[0].item()
        return grad_output.neg() * lambd, None


def grad_reverse(x, lambd):
    return GradReverse.apply(x, lambd)


def inverse_decay_scheduler(step, initial_lambda=1.0,
                             gamma=10, power=0.75, max_iter=1000):
    step = min(step, max_iter)
    return initial_lambda * ((1 + gamma * step / max_iter) ** (-power))


# ------------------------------------------------------------------
# Semantic Alignment Loss (Eq. 20)
# ------------------------------------------------------------------
def semantic_alignment_loss(Fs, Ft):
    """
    AdaIN-based semantic alignment loss Lsa.
    Transfers source style to target features via
    first and second order statistics.

    Fs: [N, D] source features (zst)
    Ft: [N, D] target features (zt)

    returns: Lsa scalar
    """
    # compute statistics
    mu_s = Fs.mean(dim=0, keepdim=True)       # [1, D]
    mu_t = Ft.mean(dim=0, keepdim=True)       # [1, D]
    sigma_s = Fs.std(dim=0, keepdim=True) + 1e-8   # [1, D]
    sigma_t = Ft.std(dim=0, keepdim=True) + 1e-8   # [1, D]

    # AdaIN: normalize target, re-scale with source statistics
    Ft_normalized = (Ft - mu_t) / sigma_t
    Ft_sm = sigma_s * Ft_normalized + mu_s    # [N, D]

    # cosine similarity constraint (Eq. 20)
    cos_sim = F.cosine_similarity(Ft_sm, Fs, dim=1)  # [N]
    Lsa = (1 - cos_sim).mean()
    return Lsa


# ------------------------------------------------------------------
# Adversarial Loss (Eq. 10)
# ------------------------------------------------------------------
def adversarial_loss(pos, w):
    """
    Weighted adversarial loss Ladv for open-set classification.
    Eq. 10 from paper.

    pos: [N] probability of unknown class p(C+1|zst)
    w:   [N] openness weight from GMM

    returns: Ladv scalar
    """
    eps = 1e-8
    pos = pos.clamp(eps, 1 - eps)
    Ladv = (-w * torch.log(1 - pos) - (1 - w) * torch.log(pos)).mean()
    return Ladv


# ------------------------------------------------------------------
# KL Divergence Loss (Eq. 16)
# ------------------------------------------------------------------
def kl_divergence_loss(pos_probs, pg):
    """
    KL divergence alignment loss Lkl.
    Aligns open-set classifier posterior with density-induced posterior.
    Eq. 16 from paper.

    pos_probs: [N, C] open-set classifier probabilities
                      (excluding unknown class — only known classes)
    pg:        [N, C] density-induced posterior from GMM

    returns: Lkl scalar
    """
    # KL(pos || pg)
    pos_probs = pos_probs.clamp(1e-8, 1.0)
    pg = pg.clamp(1e-8, 1.0)
    Lkl = F.kl_div(pos_probs.log(), pg, reduction='batchmean')
    return Lkl


# ------------------------------------------------------------------
# Calibration Loss (Eq. 18)
# ------------------------------------------------------------------
def calibration_loss(logits, pseudo_labels, tau, high_conf_mask):
    """
    Temperature-based calibration loss Lcal.
    Updated on high-confidence samples only.
    Eq. 18 from paper.

    logits:          [N, C] raw logits from open-set classifier
    pseudo_labels:   [N, C] soft pseudo labels
    tau:             learnable temperature parameter
    high_conf_mask:  [N] boolean mask for high-confidence samples

    returns: Lcal scalar
    """
    if high_conf_mask.sum() == 0:
        return torch.tensor(0.0, requires_grad=True)

    # select high confidence samples
    logits_hc = logits[high_conf_mask]           # [Nh, C]
    labels_hc = pseudo_labels[high_conf_mask]    # [Nh, C]

    # density-normalized calibrated logits (Eq. 17)
    logits_norm = F.normalize(logits_hc, dim=1)
    calibrated = logits_norm * tau

    # MSE loss between calibrated logits and pseudo labels
    Lcal = F.mse_loss(calibrated, labels_hc)
    return Lcal


# ------------------------------------------------------------------
# High Confidence Sample Selection
# ------------------------------------------------------------------
def select_high_confidence(pcs, pos_probs, rho=0.5):
    """
    Select high-confidence samples for calibration update.
    Uses threshold TC computed from accuracy estimate.

    pcs:       [N, C] closed-set classifier probabilities
    pos_probs: [N, C+1] open-set classifier probabilities
    rho:       threshold sensitivity parameter

    returns:
        high_conf_mask: [N] boolean mask
        TC: confidence threshold
    """
    # predicted class probabilities
    max_probs, _ = pcs.max(dim=1)               # [N]

    # estimate accuracy A as fraction above mean confidence
    A = (max_probs > max_probs.mean()).float().mean()
    A = A.clamp(1e-8, 1.0)

    # compute threshold TC
    TC = 1 / (1 + torch.exp(-rho * A))

    # select samples where max probability exceeds TC
    high_conf_mask = max_probs >= TC
    return high_conf_mask, TC


# ------------------------------------------------------------------
# NLL Loss wrapper (for reference — minimized via EM not gradient)
# ------------------------------------------------------------------
def nll_loss(zt, gmm):
    """
    Negative log-likelihood loss Lnll (Eq. 4).
    NOTE: In our framework this is minimized via EM algorithm,
    not gradient descent. This function is provided for reference only.

    zt:  [N, D] target features
    gmm: GMM instance

    returns: Lnll scalar
    """
    p_zk = gmm._p_z_given_k(zt)               # [N, K]
    p_z = (gmm.pi * p_zk).sum(dim=1)          # [N]
    Lnll = -torch.log(p_z + 1e-8).mean()
    return Lnll


# ------------------------------------------------------------------
# Total Loss
# ------------------------------------------------------------------
def total_loss(Lsa, Lkl, Ladv, lambda_adv=0.1):
    """
    Total training objective for feature extractor.
    Eq. 21 from paper.

    Feature extractor: min(Lsa + Lkl - lambda_adv * Ladv)
    Open-set classifier: min(Lkl + lambda_adv * Ladv)
    Temperature: min(Lcal) — handled separately

    returns: L_feat, L_cos
    """
    L_feat = Lsa + Lkl - lambda_adv * Ladv
    L_cos = Lkl + lambda_adv * Ladv
    return L_feat, L_cos


if __name__ == '__main__':
    from Settings import get_args
    args = get_args()

    N, D, C = 64, 512, 4

    # simulate features and probabilities
    Fs = torch.randn(N, D)
    Ft = torch.randn(N, D)
    pcs = torch.softmax(torch.randn(N, C), dim=1)
    pos_full = torch.softmax(torch.randn(N, C + 1), dim=1)
    pos_known = pos_full[:, :C]
    pos_unknown = pos_full[:, -1]
    pg = torch.softmax(torch.randn(N, C), dim=1)
    w = torch.sigmoid(torch.randn(N))
    tau = nn.Parameter(torch.tensor(args.tau_init))
    logits = torch.randn(N, C)
    pseudo_labels = torch.softmax(torch.randn(N, C), dim=1)

    # test each loss
    Lsa = semantic_alignment_loss(Fs, Ft)
    print(f"Lsa: {Lsa.item():.4f}")

    Ladv = adversarial_loss(pos_unknown, w)
    print(f"Ladv: {Ladv.item():.4f}")

    Lkl = kl_divergence_loss(pos_known, pg)
    print(f"Lkl: {Lkl.item():.4f}")

    high_conf_mask, TC = select_high_confidence(pcs, pos_full)
    print(f"TC: {TC.item():.4f}, high conf samples: {high_conf_mask.sum().item()}")

    Lcal = calibration_loss(logits, pseudo_labels, tau, high_conf_mask)
    print(f"Lcal: {Lcal.item():.4f}")

    L_feat, L_cos = total_loss(Lsa, Lkl, Ladv, args.lambda_adv)
    print(f"L_feat: {L_feat.item():.4f}")
    print(f"L_cos: {L_cos.item():.4f}")

    print("\nAll losses tested successfully.")
