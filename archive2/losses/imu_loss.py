#!/usr/bin/env python3
"""L_imu: IMU-supervised motion routing for the room/body Sem-MixIT separator.

WHAT IT REPLACES
  train_roombody.py currently names the body slots with L_dop, built on
  motion_stat() = coherent short-lag autocovariance. That is a PROXY:
  "anything with 25-200 ms coherence is body". It cannot tell the body from a
  fan, a slow gain drift, or a person outside the link.

  L_imu is the same routing objective with the proxy replaced by a
  MEASUREMENT: "anything whose energy rises and falls WHEN THE BODY MOVES is
  body". The IMU is worn, so it is immune to every RF nuisance in the room.

DESIGN PROPERTIES PRESERVED FROM L_dop (deliberately, do not break these)
  * scale-blind    : correlation is invariant to slot gain, so the loss never
                     prices total energy -- it only says WHERE motion routes.
  * static-blind   : a slot with a flat envelope has zero variance, its
                     attribution is defined as 0, and it contributes to
                     neither numerator nor denominator. The body's STATIC
                     reflection is therefore never pushed to room by this term
                     (the same guarantee motion_stat() gives by mean-removal).
  * same sign/shape: L_imu in [0,1], minimized, drop-in for L_dop.

ABLATION
  Keep BOTH terms and weight them:  DOPW * L_dop + IMUW * L_imu.
  IMUW=0 reproduces the current model exactly, so the A/B is clean and the
  IMU arm is only ever an addition.

NOT DECIDED HERE (needs the inspector output first)
  how the IMU stream is read and clock-aligned to the CSI window; that lives
  in imu_stream.py once 00_inspect_imu.py has been run on rosebyte.
"""
import torch
import torch.nn.functional as F


def energy_envelope(c, frame, hop):
    """(B, M, 114, T) complex slots -> (B, M, F) non-negative motion envelope.

    Time-mean removed per window first, so static content contributes nothing,
    exactly as motion_stat() does."""
    d = c - c.mean(-1, keepdim=True)
    p = d.abs().pow(2).sum(-2)                       # (B, M, T)  over subcarriers
    B, M, T = p.shape
    e = F.avg_pool1d(p.reshape(B * M, 1, T), kernel_size=frame, stride=hop)
    return e.reshape(B, M, -1)


def _zscore(x, eps=1e-8):
    x = x - x.mean(-1, keepdim=True)
    return x / (x.pow(2).mean(-1, keepdim=True).sqrt() + eps)


def attribution(env, imu, eps=1e-8):
    """Per-slot IMU attribution in [0,1].

    env : (B, M, F) slot envelopes
    imu : (B, F)    body motion envelope on the SAME frame grid
    Returns clamped Pearson correlation; a flat slot or a flat IMU window
    yields ~0, which is what makes the loss static-blind."""
    sv = env.pow(2).mean(-1) - env.mean(-1).pow(2)
    iv = imu.pow(2).mean(-1) - imu.mean(-1).pow(2)
    live = ((sv > eps) & (iv[:, None] > eps)).float()          # gate dead windows
    r = (_zscore(env) * _zscore(imu)[:, None, :]).mean(-1)
    return (r.clamp_min(0.0) * live)                            # (B, M)


def imu_route_loss(cs, imu, room_slots=(0, 2), frame=32, hop=16, eps=1e-8):
    """Fraction of IMU-attributable motion that lands in the ROOM slots.

    cs   : (B, M, 114, T) complex separator outputs
    imu  : (B, F) body motion envelope, frame grid must match energy_envelope
    Minimize. Range [0, 1]. Drop-in replacement for L_dop.
    """
    env = energy_envelope(cs, frame, hop)                       # (B, M, F)
    Fn = env.shape[-1]
    if imu.shape[-1] != Fn:                                     # tolerate off-by-a-frame
        imu = F.interpolate(imu[:, None, :], size=Fn, mode="linear",
                            align_corners=False)[:, 0, :]
    a = attribution(env, imu)                                   # (B, M)
    tot = a.sum(1)
    keep = tot > eps                                            # windows with no IMU motion
    if keep.sum() == 0:
        return env.new_zeros(())
    frac = a[:, list(room_slots)].sum(1) / (tot + eps)
    return frac[keep].mean()


def imu_pair_loss(cs, imu1, imu2, a, room=(0, 2), body=(1, 3),
                  frame=32, hop=16, eps=1e-8):
    """Paired form. Uses the fact that the mixture-of-mixtures holds TWO bodies.

    train_roombody.py already computes `a` (B,) bool = "group1 reconstructs x2".
    So group1's slots belong to whichever recording group1 reconstructs, and
    that recording has its OWN IMU. Each body slot therefore gets a specific
    target instead of a generic 'motion' label -- this is the semantic part.

    Per group g: frac_g = attr(room_g, imu_g) / (attr(room_g, imu_g)
                                                 + attr(body_g, imu_g))
    i.e. of the motion that THIS recording's wearer actually made, how much
    landed in THIS group's room slot. Minimize. Range [0, 1], like L_dop.

    cs        : (B, M, 114, T) complex slots
    imu1,imu2 : (B, F) envelopes of the two source recordings
    a         : (B,) bool from the MixIT assignment search
    """
    env = energy_envelope(cs, frame, hop)                       # (B, M, F)
    Fn = env.shape[-1]
    def fit(x):
        if x.shape[-1] == Fn: return x
        return F.interpolate(x[:, None, :], size=Fn, mode="linear",
                             align_corners=False)[:, 0, :]
    i1, i2 = fit(imu1), fit(imu2)
    tgt1 = torch.where(a[:, None], i2, i1)                      # group1's own wearer
    tgt2 = torch.where(a[:, None], i1, i2)                      # group2's own wearer
    out, n = env.new_zeros(()), 0
    for (r, b), tgt in zip(zip(room, body), (tgt1, tgt2)):
        ar = attribution(env[:, r:r + 1], tgt)[:, 0]
        ab = attribution(env[:, b:b + 1], tgt)[:, 0]
        tot = ar + ab
        keep = tot > eps
        if keep.sum() == 0: continue
        out = out + (ar / (tot + eps))[keep].mean(); n += 1
    return out / max(n, 1)


if __name__ == "__main__":
    # shape + gradient check only. NOT a validation -- validation happens on
    # OctoNet, never on made-up tensors.
    torch.manual_seed(0)
    B, M, T = 4, 4, 400
    cs = torch.randn(B, M, 114, T, dtype=torch.cfloat, requires_grad=True)
    env = energy_envelope(cs, 32, 16)
    imu = torch.rand(B, env.shape[-1])
    L = imu_route_loss(cs, imu)
    L.backward()
    print(f"env {tuple(env.shape)}  L_imu {L.item():.4f}  "
          f"grad finite: {bool(torch.isfinite(cs.grad).all())}")
    print(f"range check: {'OK' if 0.0 <= L.item() <= 1.0 else 'OUT OF RANGE'}")
    i1, i2 = torch.rand(B, env.shape[-1]), torch.rand(B, env.shape[-1])
    aa = torch.rand(B) > 0.5
    cs2 = cs.detach().clone().requires_grad_(True)
    Lp = imu_pair_loss(cs2, i1, i2, aa); Lp.backward()
    print(f"L_imu_pair {Lp.item():.4f}  grad finite: {bool(torch.isfinite(cs2.grad).all())}  "
          f"range: {'OK' if 0.0 <= Lp.item() <= 1.0 else 'OUT OF RANGE'}")
