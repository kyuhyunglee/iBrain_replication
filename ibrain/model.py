"""iBrain Replication
Encoders (spike MLP, iEEG conv + residual linear), shared criss-cross ST backbone, decoders, SimSiam heads, losses.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

SPIKE, IEEG = 0, 1  # e_type index (Eq. 2). Same as the index into the enc/dec/type_emb lists


class SpikeEncoder(nn.Module):
    """f_spike: R^P -> R^d. The 5 counts of a patch to a token (Eq. 1, U2)."""

    def __init__(self, P, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(P, d), nn.GELU(), nn.Linear(d, d))

    def forward(self, x):  # (B, C, S, P) -> (B, C, S, d)
        return self.net(x)


class IEEGEncoder(nn.Module):
    """f_iEEG: R^P -> R^d. Temporal CNN adapter + residual linear path (SPEC 1.2, U1).
    conv path: local waveform patterns within the patch, mean-pooled over time. linear path: samples -> token space."""

    def __init__(self, P, d, k=5):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, k, padding=k // 2), nn.GELU(),
            nn.Conv1d(64, 128, k, padding=k // 2), nn.GELU(),
            nn.Conv1d(128, d, k, padding=k // 2),
        )
        self.lin = nn.Linear(P, d)

    def forward(self, x):  # (B, C, S, P) -> (B, C, S, d)
        B, C, S, P = x.shape
        h = self.conv(x.reshape(B * C * S, 1, P)).mean(-1)  # (B*C*S, d)
        return h.view(B, C, S, -1) + self.lin(x)


class PatchDecoder(nn.Module):
    """g_m: R^d -> R^P, 2-layer MLP (Eq. 7). Same structure for both signal types, hidden width d (U25)."""

    def __init__(self, d, P):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, P))

    def forward(self, u):  # (B, C, S, d) -> (B, C, S, P)
        return self.net(u)


class MHA(nn.Module):
    """Multi-head attention. mask is a float tensor added to the logits, broadcast to (N, H, L, L)."""

    def __init__(self, d, H, p):
        super().__init__()
        self.H, self.p = H, p
        self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)

    def forward(self, x, mask=None):  # x (N, L, d)
        N, L, d = x.shape
        q, k, v = self.qkv(x).view(N, L, 3, self.H, d // self.H).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=self.p if self.training else 0.0
        )
        return self.o(y.transpose(1, 2).reshape(N, L, d))


class CrissCrossBlock(nn.Module):
    """Channel attention (Eq. 3) then temporal attention (Eq. 4). pre-norm, residual, FFN."""

    def __init__(self, d, H, ffn, S, p):
        super().__init__()
        self.n1, self.n2, self.n3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.ch_attn = MHA(d, H, p)
        self.t_attn = MHA(d, H, p)
        self.ffn = nn.Sequential(nn.Linear(d, ffn), nn.GELU(), nn.Linear(ffn, d))
        self.drop = nn.Dropout(p)
        # Learned relative temporal bias: one scalar per head for each offset s-s' (U3)
        self.rel = nn.Parameter(torch.zeros(H, 2 * S - 1))
        idx = torch.arange(S)[:, None] - torch.arange(S)[None, :] + (S - 1)
        self.register_buffer("rel_idx", idx, persistent=False)

    def forward(self, z, valid):  # z (B, C, S, d), valid (B, C) bool
        B, C, S, d = z.shape

        # Channel attention: independent per time s, over the channel axis. Padded channels excluded from keys (V_c)
        x = z.permute(0, 2, 1, 3).reshape(B * S, C, d)
        km = torch.zeros(B, C, device=z.device, dtype=z.dtype).masked_fill(~valid, float("-inf"))
        km = km[:, None, :].expand(B, S, C).reshape(B * S, 1, 1, C)
        x = x + self.drop(self.ch_attn(self.n1(x), km))
        z = x.view(B, S, C, d).permute(0, 2, 1, 3)

        # Temporal attention: independent per channel c, over the time axis, with the relative bias added
        x = z.reshape(B * C, S, d)
        bias = self.rel[:, self.rel_idx][None]  # (1, H, S, S)
        x = x + self.drop(self.t_attn(self.n2(x), bias))
        x = x + self.drop(self.ffn(self.n3(x)))
        return x.view(B, C, S, d)


class Backbone(nn.Module):
    """F_ST: L blocks (Eq. 5). Shared by both signal types. The last LayerNorm is ours, not in the paper (U23);
    final_norm=False leaves it out (collapse ablation, 2026-10-08)."""

    def __init__(self, d, H, ffn, S, L, p, final_norm=True):
        super().__init__()
        self.blocks = nn.ModuleList([CrissCrossBlock(d, H, ffn, S, p) for _ in range(L)])
        self.norm = nn.LayerNorm(d) if final_norm else nn.Identity()

    def forward(self, z, valid, ckpt=False):
        """ckpt: activation checkpointing per block (U37), recomputes each block in backward instead of storing it."""
        for b in self.blocks:
            z = checkpoint(b, z, valid, use_reentrant=False) if ckpt else b(z, valid)
        return self.norm(z)


class GroupBatchNorm1d(nn.BatchNorm1d):
    """BatchNorm1d whose training statistics come from consecutive groups of `group` rows, as BN on each of several
    GPUs without SyncBN (the paper's 8 GPUs x 32). Batches are drawn at random, so the groups are random subsets.
    Running statistics are updated from the first group only, like the rank-0 copy a DDP checkpoint keeps. A last
    group of one row joins the previous one. group None, or a batch no larger than one group: plain BatchNorm1d."""

    def __init__(self, num_features, group=None, **kw):
        super().__init__(num_features, **kw)
        self.group = group

    def forward(self, x):
        if not self.training or self.group is None or len(x) <= self.group:
            return super().forward(x)
        chunks = list(x.split(self.group))
        if len(chunks[-1]) < 2:
            chunks[-2:] = [torch.cat(chunks[-2:])]
        out = [super().forward(chunks[0])]  # also updates the running statistics
        out += [F.batch_norm(c, None, None, self.weight, self.bias, True, 0.0, self.eps) for c in chunks[1:]]
        return torch.cat(out)


class IBrain(nn.Module):
    def __init__(self, d=256, H=8, ffn=1024, L=6, S=10, P_spike=5, P_ieeg=50, p=0.1, d_proj=128, final_norm=True,
                 pool="mean", head_bn=False, head_bn_group=None):
        """final_norm, pool: variants from the collapse ablation (2026-10-08); the decided model keeps the defaults,
        final LayerNorm (U23) and mean pooling over valid tokens (U6). head_bn: BatchNorm in the SimSiam head, decided
        on (U7) and set in configs/paper.yaml; the default False keeps checkpoints from before 2026-10-08 loadable.
        head_bn_group: BN statistics per group of that many windows of a step (per GPU, U7: 32), None = whole step."""
        super().__init__()
        if pool not in ("mean", "max"):
            raise ValueError(f"pool must be 'mean' or 'max', got {pool!r}")
        self.d, self.pool_mode, self.head_bn = d, pool, head_bn
        # e_type (Eq. 2): one parameter per signal type, so a step of one type leaves the other's row
        # without a gradient and AdamW skips it (no decay, no momentum) (U24)
        self.type_emb = nn.ParameterList([nn.Parameter(torch.randn(d) * 0.02) for _ in range(2)])  # U22 init
        self.time_emb = nn.Parameter(torch.randn(S, d) * 0.02)  # e_time (Eq. 2)
        self.enc = nn.ModuleList([SpikeEncoder(P_spike, d), IEEGEncoder(P_ieeg, d)])  # index = SPIKE, IEEG
        self.dec = nn.ModuleList([PatchDecoder(d, P_spike), PatchDecoder(d, P_ieeg)])
        self.backbone = Backbone(d, H, ffn, S, L, p, final_norm)
        # channel-view alignment head (Eq. 12, U7). projection d -> 128, predictor 128 -> 64 -> 128
        if head_bn:  # SimSiam's placement: BN after each hidden layer and on the projection output (no affine there)
            g = head_bn_group
            self.proj = nn.Sequential(nn.Linear(d, d, bias=False), GroupBatchNorm1d(d, g), nn.GELU(),
                                      nn.Linear(d, d_proj, bias=False), GroupBatchNorm1d(d_proj, g, affine=False))
            self.pred = nn.Sequential(nn.Linear(d_proj, d_proj // 2, bias=False), GroupBatchNorm1d(d_proj // 2, g),
                                      nn.GELU(), nn.Linear(d_proj // 2, d_proj))
        else:
            self.proj = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d_proj))
            self.pred = nn.Sequential(nn.Linear(d_proj, d_proj // 2), nn.GELU(), nn.Linear(d_proj // 2, d_proj))
        # Activation checkpointing of the encoder and each backbone block while training (U37). A runtime switch,
        # not a weight: it changes memory and time only, the outputs and gradients are the same
        self.grad_checkpoint = False

    def compile_parts(self):
        """torch.compile the encoders, decoders and backbone blocks in place (pretrain.compile, U40): the many small
        ops of a block (LayerNorm, residual, dropout, reshapes, dtype casts) are fused into few kernels. In place, so
        parameter names and checkpoints do not change. dynamic=True, because micro-batches differ in windows and
        units. Dropout draws from the eager RNG (fallback_random), so the second pass of _bn_align replays it."""
        import torch._dynamo
        import torch._inductor.config

        torch._inductor.config.fallback_random = True
        torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
        for m in (*self.enc, *self.dec, *self.backbone.blocks):
            m.compile(dynamic=True)
        return self

    def encode(self, x, valid, sig=SPIKE):  # (B, C, S, P_sig), (B, C) -> U (B, C, S, d)
        ckpt = self.grad_checkpoint and self.training and torch.is_grad_enabled()
        h = checkpoint(self.enc[sig], x, use_reentrant=False) if ckpt else self.enc[sig](x)  # Eq. 1
        z = h + self.type_emb[sig] + self.time_emb  # Eq. 2
        return self.backbone(z, valid, ckpt)  # Eq. 5

    def reconstruct(self, u, sig=SPIKE):  # Eq. 7
        return self.dec[sig](u)

    def pool(self, u, valid):  # over valid tokens (U6): mean, or max for the ablation. (B, C, S, d), (B, C) -> (B, d)
        if self.pool_mode == "max":
            return u.masked_fill(~valid[:, :, None, None], float("-inf")).amax((1, 2))
        w = valid[..., None, None].to(u.dtype)
        return (u * w).sum((1, 2)) / (w.sum((1, 2)) * u.shape[2])

    def align(self, r):  # r (B, d) -> q, p (B, d_proj)
        q = self.proj(r)
        return q, self.pred(q)


def _masked_mean(per_patch, mask, valid):
    """Numerator/denominator of Eq. (9)(11): mean over masked valid patches. per_patch, mask (B, C, S), valid (B, C)."""
    w = (mask & valid[..., None]).to(per_patch.dtype)
    return (per_patch * w).sum() / w.sum().clamp(min=1)


def masked_mse(xhat, x, mask, valid):
    """Eq. (8)(9). MSE over the samples within a patch, then mean over masked valid patches.
    x is the channel-normalized waveform."""
    return _masked_mean(((xhat - x) ** 2).mean(-1), mask, valid)


def masked_poisson_nll(xhat, x, mask, valid, eps=1e-6):
    """Eq. (10)(11). mask (B, C, S) 1 = masked patch, valid (B, C).
    Same formula as PoissonNLLLoss(log_input=False, full=False).
    Mean over the bins within a patch, then mean over masked valid patches."""
    lam = F.softplus(xhat) + eps
    return _masked_mean((lam - x * torch.log(lam)).mean(-1), mask, valid)


def simsiam_loss(p1, q2, p2, q1):
    """Eq. (12). D = negative cosine, sg = stop-gradient."""
    def D(p, q):
        return -F.cosine_similarity(p, q.detach(), dim=-1).mean()
    return 0.5 * D(p1, q2) + 0.5 * D(p2, q1)
