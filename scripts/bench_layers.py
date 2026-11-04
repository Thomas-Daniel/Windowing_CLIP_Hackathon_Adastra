"""Per-layer timing of the channels-last GemmConv3d convs of the Atlas stem (one GCD, bf16)."""
import time
import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mcw.gemm_conv3d import GemmConv3d  # noqa: E402


def t(fn, reps=3):
    fn(); torch.cuda.synchronize(); t0 = time.time()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize(); return (time.time() - t0) / reps


for name, cin, cout, s, p, n, bias, need_x in (("stem s3 11->64", 11, 64, 3, 0, 384, False, False),
                                                ("stem s3 11->64 +gx", 11, 64, 3, 0, 384, False, True),
                                                ("64->192 @128", 64, 192, 1, 1, 128, False, True),
                                                ("192->192 @128", 192, 192, 1, 1, 128, True, True),
                                                ("down s2 192->384", 192, 384, 2, 1, 128, False, True)):
    conv = GemmConv3d(cin, cout, 3, stride=s, padding=p, bias=bias).cuda()
    x = torch.randn(1, cin, n, n, n, device="cuda", dtype=torch.bfloat16).contiguous(memory_format=torch.channels_last_3d)
    x.requires_grad_(need_x)
    def f():
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            conv(x)
    def fb():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = conv(x)
        y.backward(torch.ones_like(y))
    fl = 2 * cin * cout * 27 * (n // s) ** 3
    tf, tfb = t(f), t(fb)
    print(f"{name:20s} fwd {tf * 1e3:7.1f} ms ({fl / tf / 1e12:5.1f} TF/s) | fwd+bwd {tfb * 1e3:7.1f} ms | bwd {(tfb - tf) * 1e3:7.1f} ms", flush=True)
