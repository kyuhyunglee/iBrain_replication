"""M1 verification: shape, parameter count, padding invariance, loss decrease."""
import torch
from ibrain.model import IBrain, masked_poisson_nll, SPIKE

B, C, S, P, d = 2, 6, 10, 5, 256


def make(seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.poisson(torch.full((B, C, S, P), 2.0), generator=g)
    valid = torch.ones(B, C, dtype=torch.bool)
    valid[:, 4:] = False  # channels 4, 5 are padding
    return x, valid


def test_shapes_and_params():
    m = IBrain()
    x, valid = make()
    u = m.encode(x, valid, SPIKE)
    assert u.shape == (B, C, S, d)  # Eq. 5
    assert m.reconstruct(u).shape == (B, C, S, P)  # Eq. 7
    r = m.pool(u, valid)
    assert r.shape == (B, d)
    q, p = m.align(r)
    assert q.shape == p.shape == (B, 128)
    n = sum(t.numel() for t in m.parameters())
    print(f"\nparams = {n / 1e6:.2f}M")
    assert 5e6 < n < 8e6


def test_padding_invariance():
    """Padding channels must not affect the outputs of valid channels (V_c)."""
    m = IBrain().eval()
    x, valid = make()
    u6 = m.encode(x, valid, SPIKE)[:, :4]
    u4 = m.encode(x[:, :4], valid[:, :4], SPIKE)
    assert torch.allclose(u6, u4, atol=1e-5)


def test_loss_decreases():
    torch.manual_seed(0)
    m = IBrain()
    x, valid = make()
    mask = torch.rand(B, C, S) < 0.5  # 50% (SPEC 1.5)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)

    def step():
        u = m.encode(x * (~mask[..., None]).to(x.dtype), valid, SPIKE)  # Eq. 6: masking in raw-signal space
        return masked_poisson_nll(m.reconstruct(u), x, mask, valid)

    l0 = step().item()
    for _ in range(30):
        opt.zero_grad()
        l = step()
        l.backward()
        opt.step()
    print(f"\nloss {l0:.4f} -> {l.item():.4f}")
    assert l.item() < l0
