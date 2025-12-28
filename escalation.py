import numpy as np
import cv2

# ================= BASIC CROWD PHYSICS =================

def velocity_variance(flow):
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return float(np.var(mag))


def directional_entropy(flow, bins=16):
    _, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    ang = ang.flatten()

    bin_edges = np.linspace(0, 2 * np.pi, bins + 1)
    hist, _ = np.histogram(ang, bins=bin_edges)

    hist = hist / (hist.sum() + 1e-8)
    return float(-np.sum(hist * np.log(hist + 1e-8)))


def density_rate(current, prev=[0]):
    rate = current - prev[0]
    prev[0] = current
    return float(rate)

# ================= FINAL ESCALATION FORMULATION (CONTRAST-BASED) =================

def escalation_contrast(
    local_scores,
    alpha=0.2,      # weight for variance term
    beta=0.8,       # weight for max/mean term
    eps=1e-6
):
    """
    Escalation risk based on SPATIAL CONTRAST of local instability.

    local_scores: list of local instability energies e_i >= 0

    Idea:
    - Stable crowd -> all e_i similar -> low variance -> low risk
    - Local outbreak -> one e_i >> others -> high contrast -> high risk
    """

    e = np.asarray(local_scores, dtype=np.float32)

    if e.size == 0:
        return 0.0

    mean_e = np.mean(e)
    var_e = np.var(e)

    # --- contrast terms ---
    variance_term = var_e / (mean_e + eps)
    maxmean_term = (np.max(e) - mean_e) / (mean_e + eps)

    # --- combined contrast energy ---
    contrast_energy = alpha * variance_term + beta * maxmean_term

    return float(contrast_energy)


def squash_escalation(x, tau=0.5, gamma=6.0):
    """
    Map contrast energy to [0,1] escalation risk.

    tau   : stability threshold
    gamma : sharpness of transition
    """
    return float(1.0 / (1.0 + np.exp(-gamma * (x - tau))))