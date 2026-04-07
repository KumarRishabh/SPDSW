import torch
import time
from spdsw.spdsw import SPDSW

# Small test case
ds = 10
n_samples = 1000
device = "cpu"
dtype = torch.float64

# Generate random SPD matrices
eye = torch.eye(ds, dtype=dtype)
x0 = torch.rand(n_samples, ds, ds, dtype=dtype)
x0 = torch.bmm(x0, x0.transpose(1, 2)) + 0.1*eye.unsqueeze(0).repeat(n_samples, 1, 1)
x1 = torch.rand(n_samples, ds, ds, dtype=dtype)
x1 = torch.bmm(x1, x1.transpose(1, 2)) + 0.1*eye.unsqueeze(0).repeat(n_samples, 1, 1)

# Test SPDSW
t0 = time.time()
n_proj = 50
spdsw = SPDSW(ds, n_proj, device=device, dtype=dtype, sampling="spdsw")
dist = spdsw.spdsw(x0, x1, p=2)
elapsed = time.time() - t0

print(f"SPDSW test complete in {elapsed:.3f} seconds")
print(f"Distance: {dist.item():.6f}")