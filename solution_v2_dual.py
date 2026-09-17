# made by - Karthik
import os, sys, time, math, random, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
T0 = time.time()

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

SEED = int(os.environ.get("ERIS_SEED", "1337"))
FOLDS = int(os.environ.get("ERIS_FOLDS", "5"))
EPOCHS = int(os.environ.get("ERIS_EPOCHS", "12"))
RES_DEFAULT = "160"
BS = int(os.environ.get("ERIS_BS", "32"))
LR = float(os.environ.get("ERIS_LR", "3e-4"))
WD = float(os.environ.get("ERIS_WD", "1e-4"))
RES = int(os.environ.get("ERIS_RES", RES_DEFAULT))
ARCH = os.environ.get("ERIS_ARCH", "resnet18")
NTTA = int(os.environ.get("ERIS_NTTA", "4"))
RANK_W = float(os.environ.get("ERIS_RANKW", "0.0"))
DEADLINE = float(os.environ.get("ERIS_DEADLINE", "4900"))
NFINE = int(os.environ.get("ERIS_NFINE", "40"))
NREG = int(os.environ.get("ERIS_NREG", "6"))
SMOKE = os.environ.get("ERIS_SMOKE", "0") == "1"
if SMOKE:
    FOLDS, EPOCHS, DEADLINE = 2, 1, 600

NTHREAD = max(1, min(10, (os.cpu_count() or 4)))
torch.set_num_threads(NTHREAD)
DEV = torch.device(os.environ.get("ERIS_DEVICE", "") or ("cuda" if torch.cuda.is_available() else "cpu"))


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


seed_all(SEED)


def find_data_root(argv):
    cands = []
    if len(argv) > 1:
        cands.append(Path(argv[1]))
    cands += [Path("."), Path("dataset/public"), Path("../dataset/public"),
              Path("/kaggle/input"), Path(__file__).resolve().parent]
    for c in cands:
        try:
            if (c / "test.csv").exists() and (c / "train.csv").exists():
                return c
        except Exception:
            pass
    raise SystemExit("data root not found")


ROOT = find_data_root(sys.argv)
if len(sys.argv) > 2:
    SUB_OUT = Path(sys.argv[2])
else:
    SUB_OUT = Path("working/submission.csv")
SUB_OUT.parent.mkdir(parents=True, exist_ok=True)
log("root", ROOT, "sub", SUB_OUT, "dev", DEV, "threads", NTHREAD)

train_df = pd.read_csv(ROOT / "train.csv")
test_df = pd.read_csv(ROOT / "test.csv")
if SMOKE:
    train_df = train_df.sample(400, random_state=0).reset_index(drop=True)
    test_df = test_df.head(200).reset_index(drop=True)


def write_submission(probs):
    p = np.clip(np.asarray(probs, dtype=np.float64), 1e-6, 1 - 1e-6)
    p = np.where(np.isfinite(p), p, 0.5)
    out = pd.DataFrame({"sample_id": test_df.sample_id.values, "encounter_probability": p})
    out.to_csv(SUB_OUT, index=False)


write_submission(np.full(len(test_df), float(train_df.encounter_risk.mean())))
log("wrote fallback submission")


def split_panels(arr):
    h, w = arr.shape[:2]
    half = (w - 4) // 2
    return arr[:, :half], arr[:, w - half:]


def native_acq(gb, ring):
    from scipy import ndimage
    g = gb.astype(np.float32)
    p05, p50, p92, p98, p02 = np.percentile(g, [5, 50, 92, 98, 2])
    dark = g < (p92 - 45)
    dark &= ~ndimage.binary_dilation(ring > 0, iterations=2)
    lab, k = ndimage.label(dark)
    if k:
        ar = np.bincount(lab.ravel())[1:]
        ar = ar[ar >= 12]
    else:
        ar = np.array([])
    return np.array([p92, p05, p50, p98 - p02,
                     float(np.median(ar)) if len(ar) else 0.0,
                     float(ar.max()) if len(ar) else 0.0, float(len(ar)),
                     float(dark.mean()), g.mean(), g.std()], dtype=np.float32)


def load_split(df):
    n = len(df)
    A = np.zeros((n, RES, RES), dtype=np.uint8)
    B = np.zeros((n, RES, RES), dtype=np.uint8)
    ACQ = np.zeros((n, 10), dtype=np.float32)
    DSC = np.zeros((n, 1600), dtype=np.float32)
    for i, rp in enumerate(df.image_path.values):
        a = np.asarray(Image.open(ROOT / rp).convert("RGB"), dtype=np.uint8)
        pa, pb = split_panels(a)
        r = pb[..., 0].astype(np.int16)
        g = pb[..., 1].astype(np.int16)
        b = pb[..., 2].astype(np.int16)
        ring = (((g - r) > 25) & ((b - r) > 25)).astype(np.uint8) * 255
        ga = np.asarray(Image.fromarray(pa).convert("L"), dtype=np.uint8)
        gb = np.asarray(Image.fromarray(pb).convert("L"), dtype=np.uint8)
        ACQ[i] = native_acq(gb, ring)
        DSC[i] = np.asarray(Image.fromarray(gb).resize((40, 40), Image.BILINEAR), dtype=np.float32).ravel()
        if ga.shape[0] != RES:
            ga = np.asarray(Image.fromarray(ga).resize((RES, RES), Image.BILINEAR), dtype=np.uint8)
            gb = np.asarray(Image.fromarray(gb).resize((RES, RES), Image.BILINEAR), dtype=np.uint8)
        A[i], B[i] = ga, gb
    return A, B, ACQ, DSC


log("loading images ...")
trA, trB, trF, trD = load_split(train_df)
teA, teB, _, _ = load_split(test_df)
log("loaded", trA.shape, teA.shape)


def norm_params(A, B):
    n = len(A)
    lo = np.zeros(n, dtype=np.float32)
    hi = np.zeros(n, dtype=np.float32)
    for i in range(n):
        both = np.concatenate([A[i].ravel(), B[i].ravel()])
        lo[i] = np.percentile(both, 5)
        hi[i] = np.percentile(both, 95)
    return lo, np.maximum(hi - lo, 8.0)


trLO, trSC = norm_params(trA, trB)
teLO, teSC = norm_params(teA, teB)


def make_groups(feats, desc):
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.neighbors import NearestNeighbors
    n = len(feats)
    X = StandardScaler().fit_transform(feats)
    kf = max(2, min(NFINE, n // 20))
    fine = KMeans(n_clusters=kf, n_init=10, random_state=SEED).fit_predict(X)
    kr = max(2, min(NREG, n // 60))
    reg = KMeans(n_clusters=kr, n_init=10, random_state=SEED).fit_predict(X)

    d = (desc - desc.mean(1, keepdims=True)) / (desc.std(1, keepdims=True) + 1e-6)
    nn = NearestNeighbors(n_neighbors=min(6, n)).fit(d)
    dist, idx = nn.kneighbors(d)
    thr = float(np.percentile(dist[:, 1], 15))

    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for i in range(n):
        for j in range(1, idx.shape[1]):
            if dist[i, j] <= thr:
                union(i, int(idx[i, j]))
    track = np.array([find(i) for i in range(n)])

    group = fine.copy()
    for t in np.unique(track):
        mem = np.where(track == t)[0]
        if len(mem) > 1:
            group[mem] = np.bincount(fine[mem]).argmax()
    _, group = np.unique(group, return_inverse=True)
    log(f"groups: {len(np.unique(group))} (fine k={kf}, dup thr={thr:.2f}, "
        f"tracks={len(np.unique(track))}), regimes={len(np.unique(reg))}")
    return group, reg


groups, regimes = make_groups(trF, trD)
y = train_df.encounter_risk.values.astype(np.float32)


def assign_folds(group, nfold):
    sizes = pd.Series(group).value_counts()
    fold_of = {}
    load = np.zeros(nfold)
    for g, s in sizes.items():
        f = int(np.argmin(load))
        fold_of[g] = f
        load[f] += s
    return np.array([fold_of[g] for g in group])


fold_id = assign_folds(groups, FOLDS)
log("fold sizes", np.bincount(fold_id, minlength=FOLDS).tolist(),
    "fold posrate", [round(float(y[fold_id == f].mean()), 3) for f in range(FOLDS)])


def average_precision(yt, p):
    yt = np.asarray(yt, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    if yt.sum() == 0:
        return float("nan")
    o = np.argsort(-p, kind="mergesort")
    yt = yt[o]
    tp = np.cumsum(yt)
    prec = tp / np.arange(1, len(yt) + 1)
    return float((prec * yt).sum() / yt.sum())


def regime_utility(yt, p):
    ap = average_precision(yt, p)
    brier = float(np.mean((np.asarray(p, dtype=np.float64) - np.asarray(yt, dtype=np.float64)) ** 2))
    return 0.70 * ap + 0.30 * (1.0 - brier)


def macro_score(yt, p, reg):
    vals = []
    for r in np.unique(reg):
        m = reg == r
        if m.sum() >= 10 and 0 < yt[m].sum() < m.sum():
            vals.append(regime_utility(yt[m], p[m]))
    if not vals:
        return regime_utility(yt, p)
    return float(np.mean(vals))


YY, XX = np.mgrid[0:RES, 0:RES].astype(np.float32)
cc = (RES - 1) / 2.0
RAD = np.sqrt((XX - cc) ** 2 + (YY - cc) ** 2) / (RES / 2.0)
RAD_T = torch.from_numpy((RAD - 0.7).astype(np.float32))[None, None]

POLAR = os.environ.get("ERIS_POLAR", "0") == "1"
DUAL = os.environ.get("ERIS_DUAL", "1") == "1"
PR = int(os.environ.get("ERIS_PR", "112"))
PA = int(os.environ.get("ERIS_PA", "160"))
RMAX = float(os.environ.get("ERIS_RMAX", "1.35"))


def build_spec(name, polar, pr=0, pa=0, rmax=1.0):
    s = dict(name=name, polar=polar, pr=pr, pa=pa, rmax=rmax)
    if polar:
        s["th"] = torch.linspace(0, 2 * math.pi, pa + 1)[:pa]
        s["rr"] = torch.linspace(0.0, rmax, pr)
        s["rad"] = (torch.linspace(0.0, 1.0, pr).view(1, 1, pr, 1) - 0.5).expand(1, 1, pr, pa).contiguous()
        s["w"] = (pr * pa) / float(RES * RES)
    else:
        s["rad"] = RAD_T
        s["w"] = 1.0
    return s


if DUAL:
    SPECS = [build_spec("polar", True, PR, PA, RMAX), build_spec("cart", False)]
elif POLAR:
    SPECS = [build_spec("polar", True, PR, PA, RMAX)]
else:
    SPECS = [build_spec("cart", False)]
log("specs:", [(s["name"], round(s["w"], 3)) for s in SPECS])


def polar_warp(x, aug, gen, spec):
    n = x.shape[0]
    pr, pa = spec["pr"], spec["pa"]
    if aug:
        ang = torch.rand(n, generator=gen) * 2 * math.pi
        scl = 1.0 + (torch.rand(n, generator=gen) - 0.5) * 0.16
        flip = torch.where(torch.rand(n, generator=gen) < 0.5, -1.0, 1.0)
    else:
        ang = torch.zeros(n); scl = torch.ones(n); flip = torch.ones(n)
    th = spec["th"].view(1, 1, pa) * flip.view(n, 1, 1) + ang.view(n, 1, 1)
    rr = spec["rr"].view(1, pr, 1) * scl.view(n, 1, 1)
    grid = torch.stack([rr * torch.cos(th), rr * torch.sin(th)], dim=-1)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


def make_batch(A, B, LO, SC, idx, aug, gen=None, spec=None):
    a = torch.from_numpy(A[idx].astype(np.float32))
    b = torch.from_numpy(B[idx].astype(np.float32))
    lo = torch.from_numpy(LO[idx])[:, None, None]
    sc = torch.from_numpy(SC[idx])[:, None, None]
    ga = (a - lo) / sc
    gb = (b - lo) / sc
    if aug:
        n = len(idx)
        cj = 1.0 + (torch.rand(n, 1, 1, generator=gen) - 0.5) * 0.30
        bj = (torch.rand(n, 1, 1, generator=gen) - 0.5) * 0.20
        ga = ga * cj + bj
        gb = gb * cj + bj
    x = torch.stack([ga - 1.0, gb - 1.0, (gb - ga) * 2.0], dim=1)
    if spec["polar"]:
        p = polar_warp(x, aug, gen, spec)
        return torch.cat([p, spec["rad"].expand(p.shape[0], 1, spec["pr"], spec["pa"])], dim=1)
    if aug:
        n = x.shape[0]
        ang = torch.rand(n, generator=gen) * 2 * math.pi
        scl = 1.0 + (torch.rand(n, generator=gen) - 0.5) * 0.16
        flip = torch.where(torch.rand(n, generator=gen) < 0.5, -1.0, 1.0)
        cos, sin = torch.cos(ang) / scl, torch.sin(ang) / scl
        theta = torch.zeros(n, 2, 3)
        theta[:, 0, 0] = cos * flip
        theta[:, 0, 1] = -sin
        theta[:, 1, 0] = sin * flip
        theta[:, 1, 1] = cos
        grid = F.affine_grid(theta, x.shape, align_corners=False)
        x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    r = RAD_T.expand(x.shape[0], 1, RES, RES)
    return torch.cat([x, r], dim=1)


def ring_masks(s, nring):
    yy, xx = np.mgrid[0:s, 0:s].astype(np.float32)
    c = (s - 1) / 2.0
    r = np.sqrt((xx - c) ** 2 + (yy - c) ** 2)
    edges = np.linspace(0.0, r.max() + 1e-6, nring + 1)
    M = np.zeros((nring, s, s), dtype=np.float32)
    for k in range(nring):
        m = (r >= edges[k]) & (r < edges[k + 1])
        if m.sum() == 0:
            m = r >= edges[k]
        M[k] = m.astype(np.float32) / max(1.0, float(m.sum()))
    return torch.from_numpy(M)


NRING = int(os.environ.get("ERIS_NRING", "5"))


class Net(nn.Module):
    def __init__(self, spec, arch=ARCH):
        super().__init__()
        self.polar = spec["polar"]
        try:
            m = getattr(torchvision.models, arch)(weights="IMAGENET1K_V1")
            log("pretrained weights loaded")
        except Exception as e:
            log("pretrained unavailable, from scratch:", repr(e)[:120])
            m = getattr(torchvision.models, arch)(weights=None)
        w = m.conv1.weight.data
        c1 = nn.Conv2d(4, 64, 7, 2, 3, bias=False)
        c1.weight.data[:, :3] = w
        c1.weight.data[:, 3:] = 0.0
        m.conv1 = c1
        nf = m.fc.in_features
        m.fc = nn.Identity()
        self.backbone = m
        self.reduce2 = nn.Conv2d(nf // 4, 32, 1)
        self.reduce3 = nn.Conv2d(nf // 2, 64, 1)
        if spec["polar"]:
            d2, d3 = 32 * max(1, spec["pr"] // 8), 64 * max(1, spec["pr"] // 16)
        else:
            nr2, nr3 = NRING + 3, NRING + 1
            d2, d3 = 32 * nr2, 64 * nr3
            self.register_buffer("rings2", ring_masks(max(2, RES // 8), nr2))
            self.register_buffer("rings3", ring_masks(max(2, RES // 16), nr3))
        self.bn = nn.BatchNorm1d(d2 + d3)
        self.drop = nn.Dropout(0.3)
        self.head = nn.Linear(nf + d2 + d3, 1)

    def forward(self, x):
        b = self.backbone
        x = b.maxpool(b.relu(b.bn1(b.conv1(x))))
        l2 = b.layer2(b.layer1(x))
        l3 = b.layer3(l2)
        l4 = b.layer4(l3)
        g = l4.mean(dim=(2, 3))
        if self.polar:
            r2 = self.reduce2(l2).mean(dim=3).flatten(1)
            r3 = self.reduce3(l3).mean(dim=3).flatten(1)
        else:
            r2 = torch.einsum("bchw,khw->bck", self.reduce2(l2), self.rings2).flatten(1)
            r3 = torch.einsum("bchw,khw->bck", self.reduce3(l3), self.rings3).flatten(1)
        rp = self.bn(torch.cat([r2, r3], dim=1))
        return self.head(self.drop(torch.cat([g, rp], dim=1))).squeeze(1)


@torch.no_grad()
def predict(model, A, B, LO, SC, spec, ntta=NTTA, bs=64):
    model.eval()
    n = len(A)
    acc = np.zeros(n, dtype=np.float64)
    for t in range(max(1, ntta)):
        for s in range(0, n, bs):
            idx = np.arange(s, min(s + bs, n))
            xb = make_batch(A, B, LO, SC, idx, aug=False, spec=spec)
            if t:
                xb = (torch.roll(xb, t * (spec["pa"] // max(1, ntta)), dims=3) if spec["polar"]
                      else torch.rot90(xb, t, dims=(2, 3)))
            acc[idx] += torch.sigmoid(model(xb.to(DEV))).float().cpu().numpy()
    return acc / max(1, ntta)


EP_T = {}


def train_fold(f, spec, fold_end):
    nm = spec["name"]
    EP_T.setdefault(nm, [])
    seed_all(SEED + f)
    gen = torch.Generator().manual_seed(SEED + f)
    tri = np.where(fold_id != f)[0]
    vai = np.where(fold_id == f)[0]
    model = Net(spec).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    steps = max(1, math.ceil(len(tri) / BS)) * EPOCHS
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps,
                                                pct_start=0.25, div_factor=8, final_div_factor=40)
    done = 0
    for ep in range(EPOCHS):
        est = float(np.median(EP_T[nm])) if EP_T[nm] else 0.0
        if ep > 0 and (time.time() - T0) + est > fold_end:
            log(f"{nm} fold {f}: budget stop at epoch {ep} (est epoch {est:.0f}s)")
            break
        ep_t0 = time.time()
        model.train()
        perm = np.random.permutation(tri)
        tot = 0.0
        for s in range(0, len(perm), BS):
            idx = perm[s:s + BS]
            if len(idx) < 2:
                continue
            xb = make_batch(trA, trB, trLO, trSC, idx, aug=True, gen=gen, spec=spec).to(DEV)
            yb = torch.from_numpy(y[idx]).to(DEV)
            opt.zero_grad(set_to_none=True)
            out = model(xb)
            loss = F.binary_cross_entropy_with_logits(out, yb)
            if RANK_W > 0:
                sp, sn = out[yb > 0.5], out[yb < 0.5]
                if len(sp) and len(sn):
                    loss = loss + RANK_W * F.softplus(-(sp[:, None] - sn[None, :])).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            if done < steps:
                sched.step(); done += 1
            tot += float(loss) * len(idx)
        EP_T[nm].append(time.time() - ep_t0)
        log(f"{nm} fold {f} ep {ep} loss {tot / len(perm):.4f} ({EP_T[nm][-1]:.0f}s)")
    vp = predict(model, trA[vai], trB[vai], trLO[vai], trSC[vai], spec)
    log(f"{nm} fold {f} val util {regime_utility(y[vai], vp):.4f} ap {average_precision(y[vai], vp):.4f}")
    return model, vai, vp


oof_by = {s["name"]: np.full(len(train_df), np.nan) for s in SPECS}
test_by = {s["name"]: np.zeros(len(test_df)) for s in SPECS}
cnt_by = {s["name"]: 0 for s in SPECS}
NP = min(64, len(test_df))
T_TEST_EST = {}
for _s in SPECS:
    _pm = Net(_s).to(DEV)
    _p0 = time.time()
    predict(_pm, teA[:NP], teB[:NP], teLO[:NP], teSC[:NP], _s)
    T_TEST_EST[_s["name"]] = (time.time() - _p0) * (len(test_df) / NP) * 1.25
    del _pm
    log(f"probe {_s['name']}: est {T_TEST_EST[_s['name']]:.0f}s per model")


def blended_test():
    parts = [test_by[k] / cnt_by[k] for k in cnt_by if cnt_by[k] > 0]
    return np.mean(parts, axis=0) if parts else np.full(len(test_df), float(y.mean()))


def blended_oof():
    parts = [oof_by[k] for k in cnt_by if cnt_by[k] > 0]
    if not parts:
        return np.full(len(train_df), np.nan)
    S = np.stack(parts)
    with np.errstate(invalid="ignore"):
        return np.nanmean(S, axis=0)


T_TEST = {s["name"]: [] for s in SPECS}


def est_epoch(nm):
    return float(np.median(EP_T[nm])) if EP_T.get(nm) else None


def est_test(nm):
    return float(np.median(T_TEST[nm])) if T_TEST[nm] else T_TEST_EST[nm]


def unit_cost(nm):
    e = est_epoch(nm)
    return EPOCHS * (e if e else est_test(nm) * 0.7) + est_test(nm)


done_folds = 0
for f in range(FOLDS):
    now = time.time() - T0
    if done_folds >= 1 and all(now + unit_cost(s["name"]) + 40.0 > DEADLINE for s in SPECS):
        log(f"stopping before fold {f}: cheapest unit needs "
            f"{min(unit_cost(s['name']) for s in SPECS):.0f}s, only {DEADLINE - now:.0f}s left")
        break
    for spec in SPECS:
        nm = spec["name"]
        now = time.time() - T0
        if done_folds >= 1 and now + unit_cost(nm) + 40.0 > DEADLINE:
            log(f"skipping {nm} fold {f}: needs {unit_cost(nm):.0f}s, {DEADLINE - now:.0f}s left")
            continue
        e = est_epoch(nm)
        want = EPOCHS * e * 1.10 if e else (DEADLINE - now - est_test(nm) - 60.0)
        fold_end = min(now + want, DEADLINE - est_test(nm) - 30.0)
        if fold_end <= now and min(cnt_by.values()) >= 1:
            log(f"skipping {nm} fold {f}: out of budget")
            continue
        log(f"{nm} fold {f}: budget until {fold_end:.0f}s (est epoch {e if e else -1:.0f}s)")
        model, vai, vp = train_fold(f, spec, fold_end)
        oof_by[nm][vai] = vp
        tt0 = time.time()
        test_by[nm] += predict(model, teA, teB, teLO, teSC, spec)
        T_TEST[nm].append(time.time() - tt0)
        cnt_by[nm] += 1
        write_submission(blended_test())
        log(f"{nm} fold {f} done, submission updated (counts {cnt_by})")
        del model
    done_folds += 1

test_p = blended_test()
oof = blended_oof()
have = ~np.isnan(oof)
for k in cnt_by:
    if cnt_by[k]:
        h = ~np.isnan(oof_by[k])
        log(f"single {k}: n={h.sum()} macro={macro_score(y[h], oof_by[k][h], regimes[h]):.4f} "
            f"AP={average_precision(y[h], oof_by[k][h]):.4f}")
log(f"OOF covered {have.sum()}/{len(oof)} from {cnt_by}")
raw_macro = macro_score(y[have], oof[have], regimes[have])
log(f"OOF macro utility (raw)  {raw_macro:.4f}  | pooled {regime_utility(y[have], oof[have]):.4f} "
    f"| AP {average_precision(y[have], oof[have]):.4f}")
for r in np.unique(regimes[have]):
    m = (regimes == r) & have
    if m.sum() >= 10 and 0 < y[m].sum() < m.sum():
        log(f"  regime {r}: n={m.sum():4d} pos={y[m].mean():.3f} "
            f"AP={average_precision(y[m], oof[m]):.4f} "
            f"brier={np.mean((oof[m] - y[m]) ** 2):.4f} util={regime_utility(y[m], oof[m]):.4f}")

try:
    from sklearn.linear_model import LogisticRegression
    z = np.log(np.clip(oof[have], 1e-6, 1 - 1e-6) / (1 - np.clip(oof[have], 1e-6, 1 - 1e-6)))
    pl = LogisticRegression(C=1e6, max_iter=1000).fit(z.reshape(-1, 1), y[have].astype(int))
    cal_oof = pl.predict_proba(z.reshape(-1, 1))[:, 1]
    cal_macro = macro_score(y[have], cal_oof, regimes[have])
    log(f"OOF macro utility (platt) {cal_macro:.4f}  coef {pl.coef_[0][0]:.3f} int {pl.intercept_[0]:.3f}")
    zt = np.log(np.clip(test_p, 1e-6, 1 - 1e-6) / (1 - np.clip(test_p, 1e-6, 1 - 1e-6)))
    cal_test = pl.predict_proba(zt.reshape(-1, 1))[:, 1]
    final = 0.5 * (test_p + cal_test)
    log(f"OOF macro utility (blend) {macro_score(y[have], 0.5 * (oof[have] + cal_oof), regimes[have]):.4f}")
except Exception as e:
    log("calibration skipped:", repr(e)[:120])
    final = test_p

_tag = os.environ.get("ERIS_OOFTAG", "")
if _tag:
    np.save(SUB_OUT.parent / f"oof_{_tag}.npy", oof)
    np.save(SUB_OUT.parent / f"tst_{_tag}.npy", test_p)
    np.save(SUB_OUT.parent / "y_true.npy", y)
    np.save(SUB_OUT.parent / "regimes.npy", regimes)

write_submission(final)
log("final submission written", SUB_OUT, "rows", len(test_df),
    "mean", float(np.mean(final)), "min", float(np.min(final)), "max", float(np.max(final)))
log("total runtime", round(time.time() - T0, 1))
