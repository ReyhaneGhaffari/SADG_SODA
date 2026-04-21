import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
import numpy as np


class GMM:
    """
    Gaussian Mixture Model with EM algorithm.
    Used to discover intrinsic target feature space structure.

    Parameters:
        K: number of mixture components
        em_iterations: number of EM iterations per update
        purity_threshold: cluster purity threshold t
        beta: GMM prior refinement weight
    """

    def __init__(self, K=16, em_iterations=5,
                 purity_threshold=0.5, beta=0.5):
        self.K = K
        self.em_iterations = em_iterations
        self.purity_threshold = purity_threshold
        self.beta = beta

        # GMM parameters
        self.mu = None       # [K, D] cluster means
        self.sigma = None    # [D, D] shared covariance
        self.pi = None       # [K] cluster priors
        self.initialized = False

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def initialize(self, zt):
        """
        K-means warm-start initialization.
        zt: [N, D] target features
        """
        N, D = zt.shape
        device = zt.device

        # K-means warm start
        kmeans = KMeans(n_clusters=self.K, n_init=10, random_state=42)
        cluster_idx = kmeans.fit_predict(zt.detach().cpu().numpy())
        cluster_idx = torch.tensor(cluster_idx, device=device)

        # compute initial mu and sigma
        self._compute_mu_sigma(zt, cluster_idx)

        # uniform priors
        self.pi = torch.ones(self.K, device=device) / self.K
        self.initialized = True

        return cluster_idx

    # ------------------------------------------------------------------
    # EM Algorithm
    # ------------------------------------------------------------------
    def e_step(self, zt):
        """
        E-step: compute soft cluster responsibilities r_ik.
        zt: [N, D]
        returns: responsibilities [N, K]
        """
        p_zk = self._p_z_given_k(zt)          # [N, K]
        weighted = p_zk * self.pi              # [N, K]
        denom = weighted.sum(dim=1, keepdim=True) + 1e-8
        r = weighted / denom                   # [N, K]
        return r

    def m_step(self, zt, r):
        """
        M-step: update mu, sigma from responsibilities.
        zt: [N, D]
        r: [N, K] responsibilities
        """
        N, D = zt.shape
        device = zt.device

        Nk = r.sum(dim=0)  # [K]

        # update means
        mu = torch.zeros(self.K, D, device=device)
        for k in range(self.K):
            if Nk[k] > 0:
                mu[k] = (r[:, k].unsqueeze(1) * zt).sum(dim=0) / Nk[k]
        self.mu = mu

        # update shared covariance
        sigma = torch.zeros(D, D, device=device)
        for k in range(self.K):
            diff = zt - self.mu[k]             # [N, D]
            weighted_outer = (r[:, k].unsqueeze(1) * diff).T @ diff
            sigma += weighted_outer
        sigma /= N
        self.sigma = sigma + 1e-6 * torch.eye(D, device=device)

    def run_em(self, zt):
        """
        Run EM for em_iterations steps.
        Triggered every M/2 mini-batches.
        zt: [N, D] target features
        """
        if not self.initialized:
            self.initialize(zt)
            return

        for _ in range(self.em_iterations):
            r = self.e_step(zt)
            self.m_step(zt, r)

    # ------------------------------------------------------------------
    # Class-Conditional Prototype Computation
    # ------------------------------------------------------------------
    def compute_class_prototypes(self, zt, zst, pcs):
        """
        Compute class-conditional cluster prototypes mu_c.
        Eq. 5 and Eq. 6 from paper.

        zt:  [N, D] target features (clustering branch)
        zst: [N, D] source features (classifier branch)
        pcs: [N, C] closed-set classifier probabilities

        returns: mu_c [C, D] class prototypes
        """
        device = zt.device
        N, D = zt.shape
        C = pcs.shape[1]

        # get cluster assignments from current GMM
        r = self.e_step(zt)                    # [N, K]
        cluster_idx = torch.argmax(r, dim=1)   # [N]

        # compute mu_kc: [K, C, D]
        mu_kc = torch.zeros(self.K, C, D, device=device)

        for k in range(self.K):
            idx_k = (cluster_idx == k).nonzero(as_tuple=True)[0]
            if len(idx_k) == 0:
                continue

            zt_k = zt[idx_k]          # [Nk, D]
            pcs_k = pcs[idx_k]        # [Nk, C]
            argmax_c = torch.argmax(pcs_k, dim=1)  # [Nk]

            for c in range(C):
                mask = (argmax_c == c).float()
                weights = mask * pcs_k[:, c]
                denom = weights.sum() + 1e-8
                mu_kc[k, c] = (weights.unsqueeze(1) * zt_k).sum(dim=0) / denom

        # average over clusters: mu_c [C, D]  (Eq. 6)
        mu_c = mu_kc.mean(dim=0)
        return mu_c

    # ------------------------------------------------------------------
    # GMM Prior Refinement (Eq. 14)
    # ------------------------------------------------------------------
    def refine_priors(self, zt, pcs):
        """
        Class-aware refinement of GMM priors based on cluster purity.
        Eq. 11-14 from paper.

        zt:  [N, D] target features
        pcs: [N, C] closed-set classifier probabilities
        """
        device = zt.device
        r = self.e_step(zt)
        cluster_idx = torch.argmax(r, dim=1)
        C = pcs.shape[1]

        # Step 1: compute cluster purities A_k,c (Eq. 11)
        A = torch.zeros(self.K, C, device=device)
        for k in range(self.K):
            idx_k = (cluster_idx == k).nonzero(as_tuple=True)[0]
            if len(idx_k) == 0:
                continue
            pcs_k = pcs[idx_k]
            pred_k = torch.argmax(pcs_k, dim=1)
            for c in range(C):
                A[k, c] = (pred_k == c).float().mean()

        # max purity per cluster
        mk = A.max(dim=1).values  # [K]

        # Step 2: compute delta_k (Eq. 12)
        delta = torch.clamp(mk - self.purity_threshold, min=0)  # [K]

        # identify strong clusters S
        S_mask = mk > self.purity_threshold
        S_size = S_mask.sum().item()

        # compute weights w_k^p (Eq. 13)
        wp = torch.ones(self.K, device=device)
        if S_size > 0:
            avg_delta_S = delta[S_mask].mean()
            for k in range(self.K):
                if S_mask[k]:
                    wp[k] = 1 + self.beta * avg_delta_S
                else:
                    wp[k] = 1 + self.beta * delta[k]
        else:
            wp = 1 + self.beta * delta

        # Step 3: update priors (Eq. 14)
        new_pi = wp * self.pi
        self.pi = new_pi / (new_pi.sum() + 1e-8)

    # ------------------------------------------------------------------
    # OOD Scoring
    # ------------------------------------------------------------------
    def gem_score(self, zst, mu_c):
        """
        Gaussian Mixture Energy Measurement (GEM) score.
        Eq. 7 from paper.

        zst:  [N, D]
        mu_c: [C, D] class prototypes

        returns: Egm [N] energy scores (higher = more OOD)
        """
        device = zst.device
        N = zst.shape[0]
        C = mu_c.shape[0]

        sigma_inv = torch.linalg.inv(self.sigma)  # [D, D]

        energies = []
        for c in range(C):
            diff = zst - mu_c[c]                   # [N, D]
            mahal = (diff @ sigma_inv * diff).sum(dim=1)  # [N]
            energies.append(-0.5 * mahal)

        energies = torch.stack(energies, dim=1)    # [N, C]
        Egm = -torch.logsumexp(energies, dim=1)    # [N]
        return Egm

    def entropy_score(self, pcs):
        """
        Entropy-based OOD score.
        Eq. 8 from paper.

        pcs: [N, C] closed-set classifier probabilities

        returns: Eop [N] entropy scores (higher = more OOD)
        """
        Eop = -(pcs * torch.log(pcs + 1e-8)).sum(dim=1)  # [N]
        return Eop

    def openness_weight(self, zst, pcs, mu_c):
        """
        Combined openness weight w(zst).
        Eq. 9 from paper.

        returns: w [N] sigmoid of Egm * Eop
        """
        Egm = self.gem_score(zst, mu_c)    # [N]
        Eop = self.entropy_score(pcs)      # [N]
        d = Egm * Eop
        w = torch.sigmoid(d)               # [N]
        return w, Egm, Eop

    def density_posterior(self, zst, mu_c):
        """
        Density-induced class posterior pg(c|zst).
        Eq. 15 from paper.

        zst:  [N, D]
        mu_c: [C, D]

        returns: pg [N, C]
        """
        device = zst.device
        N = zst.shape[0]
        C = mu_c.shape[0]

        sigma_inv = torch.linalg.inv(self.sigma)

        log_probs = []
        for c in range(C):
            diff = zst - mu_c[c]
            mahal = (diff @ sigma_inv * diff).sum(dim=1)
            log_probs.append(-0.5 * mahal)

        log_probs = torch.stack(log_probs, dim=1)  # [N, C]
        pg = F.softmax(log_probs, dim=1)           # [N, C]
        return pg

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _compute_mu_sigma(self, zt, cluster_idx):
        """
        Compute cluster means and shared covariance from hard assignments.
        """
        N, D = zt.shape
        device = zt.device

        mu = torch.zeros(self.K, D, device=device)
        for k in range(self.K):
            members = zt[cluster_idx == k]
            if members.shape[0] > 0:
                mu[k] = members.mean(dim=0)
            else:
                mu[k] = zt.mean(dim=0)
        self.mu = mu

        sigma = torch.zeros(D, D, device=device)
        for k in range(self.K):
            members = zt[cluster_idx == k]
            if members.shape[0] == 0:
                continue
            diff = members - self.mu[k]
            sigma += diff.T @ diff
        self.sigma = sigma / N + 1e-6 * torch.eye(D, device=device)

    def _p_z_given_k(self, zt):
        """
        Compute p(z|k) for all clusters using current mu and sigma.
        returns: [N, K]
        """
        N, D = zt.shape
        device = zt.device

        sigma_inv = torch.linalg.inv(self.sigma)

        p = []
        for k in range(self.K):
            diff = zt - self.mu[k]
            exponent = -0.5 * (diff @ sigma_inv * diff).sum(dim=1)
            p.append(exponent)

        log_p = torch.stack(p, dim=1)     # [N, K]
        p = torch.softmax(log_p, dim=1)   # [N, K] normalized
        return p


if __name__ == '__main__':
    from Settings import get_args
    args = get_args()

    # simulate target features
    N, D, C = 100, 512, 4
    zt = torch.randn(N, D)
    zst = torch.randn(N, D)
    pcs = torch.softmax(torch.randn(N, C), dim=1)

    # initialize GMM
    gmm = GMM(K=args.K,
              em_iterations=args.em_iterations,
              purity_threshold=args.purity_threshold,
              beta=args.beta)

    # initialization
    gmm.initialize(zt)
    print(f"mu shape: {gmm.mu.shape}")       # [16, 512]
    print(f"sigma shape: {gmm.sigma.shape}") # [512, 512]
    print(f"pi shape: {gmm.pi.shape}")       # [16]

    # run EM
    gmm.run_em(zt)
    print("EM update completed")

    # class prototypes
    mu_c = gmm.compute_class_prototypes(zt, zst, pcs)
    print(f"mu_c shape: {mu_c.shape}")       # [4, 512]

    # prior refinement
    gmm.refine_priors(zt, pcs)
    print(f"refined pi: {gmm.pi}")

    # OOD scores
    w, Egm, Eop = gmm.openness_weight(zst, pcs, mu_c)
    print(f"w shape: {w.shape}")             # [100]
    print(f"Egm range: [{Egm.min():.3f}, {Egm.max():.3f}]")
    print(f"Eop range: [{Eop.min():.3f}, {Eop.max():.3f}]")

    # density posterior
    pg = gmm.density_posterior(zst, mu_c)
    print(f"pg shape: {pg.shape}")           # [100, 4]

    print("\nGMM tested successfully.")
