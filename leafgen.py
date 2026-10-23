# leafgen.py — synthetic leaf generator for the Week 5 teaching simulation (no real photos; no real dataset).
# Deterministic: the same seeds give the same images on any machine (numpy Generator PCG64).
import numpy as np, cv2
SZ = 96

def background(kind, r):
    if kind == "lab_light":      # healthy leaves in the 'lab' were shot on a light board
        v = int(r.integers(195, 225)); bg = np.full((SZ, SZ, 3), v, np.uint8)
    elif kind == "lab_dark":     # diseased leaves were shot on a darker board (the shortcut)
        v = int(r.integers(120, 150)); bg = np.full((SZ, SZ, 3), v, np.uint8)
    elif kind == "lab_any":
        v = int(r.integers(120, 225)); bg = np.full((SZ, SZ, 3), v, np.uint8)
    else:                         # field: soil / grass texture, uneven light
        if r.random() < 0.5:   # grass
            base = np.array([r.integers(30, 70), r.integers(90, 150), r.integers(40, 90)])     # BGR green
        else:                  # soil
            base = np.array([r.integers(30, 60), r.integers(60, 100), r.integers(90, 140)])    # BGR brown
        bg = np.clip(base[None, None, :] + r.normal(0, 22, (SZ, SZ, 3)), 0, 255).astype(np.uint8)
        bg = cv2.GaussianBlur(bg, (0, 0), 1.2)
        # a few grass strokes / stones
        for _ in range(12):
            p1 = tuple(int(x) for x in r.integers(0, SZ, 2)); p2 = tuple(int(x) for x in r.integers(0, SZ, 2))
            k = float(r.uniform(0.6, 1.4)); col = tuple(int(min(255, x * k)) for x in base)   # same hue, different shade
            cv2.line(bg, p1, p2, col, int(r.integers(1, 3)))
        # lighting gradient
        gx = np.linspace(r.uniform(0.7, 1.0), r.uniform(0.9, 1.4), SZ)
        bg = np.clip(bg * gx[None, :, None], 0, 255).astype(np.uint8)
    return bg

def leaf_mask_and_color(r, diseased, field):
    mask = np.zeros((SZ, SZ), np.uint8)
    cx, cy = int(r.integers(36, 60)), int(r.integers(36, 60))
    ax, ay = int(r.integers(22, 34)), int(r.integers(12, 20))
    ang = float(r.uniform(0, 180))
    cv2.ellipse(mask, (cx, cy), (ax, ay), ang, 0, 360, 255, -1)
    g = int(r.integers(120, 190)); leaf = np.zeros((SZ, SZ, 3), np.uint8)
    leaf[:] = (int(r.integers(30, 70)), g, int(r.integers(30, 80)))            # BGR green
    # vein
    cv2.ellipse(leaf, (cx, cy), (ax, 1), ang, 0, 360, (40, g - 40, 40), 1)
    if diseased:
        n = int(r.integers(3, 8))
        for _ in range(n):
            t = r.uniform(0, 2 * np.pi); rad = r.uniform(0, 0.8)
            px = int(cx + rad * ax * np.cos(t) * np.cos(np.radians(ang)) - rad * ay * np.sin(t) * np.sin(np.radians(ang)))
            py = int(cy + rad * ax * np.cos(t) * np.sin(np.radians(ang)) + rad * ay * np.sin(t) * np.cos(np.radians(ang)))
            cv2.circle(leaf, (px, py), int(r.integers(2, 5)), (int(r.integers(10, 40)), int(r.integers(50, 90)), int(r.integers(90, 140))), -1)   # brown spot
    if field:   # field: partial occlusion by another leaf, slight blur, noise
        if r.random() < 0.5:
            m2 = np.zeros_like(mask); cv2.ellipse(m2, (int(r.integers(0, SZ)), int(r.integers(0, SZ))), (int(r.integers(15, 30)), int(r.integers(8, 16))), float(r.uniform(0, 180)), 0, 360, 255, -1)
            mask[m2 > 0] = 0
            leaf[m2 > 0] = (int(r.integers(30, 70)), int(r.integers(100, 170)), int(r.integers(30, 80)))
            mask = mask | m2
        leaf = cv2.GaussianBlur(leaf, (0, 0), float(r.uniform(0.3, 1.0)))
    return mask, leaf

def compose(bg, mask, leaf, r, noise=4):
    m = (mask > 0)[:, :, None]
    img = np.where(m, leaf, bg).astype(np.uint8)
    img = np.clip(img + r.normal(0, noise, img.shape), 0, 255).astype(np.uint8)
    return img

def make_set(n, cond, r, seed_shift=0):
    X, y, masks, leaves, bgs = [], [], [], [], []
    for i in range(n):
        d = i % 2
        field = cond == "field"
        if cond == "lab":
            bg = background("lab_dark" if d else "lab_light", r)
        elif cond == "lab_random":
            bg = background("lab_any", r)
        else:
            bg = background("field", r)
        mask, leaf = leaf_mask_and_color(r, d, field)
        X.append(compose(bg, mask, leaf, r)); y.append(d); masks.append(mask); leaves.append(leaf); bgs.append(bg)
    return np.array(X), np.array(y), masks, leaves, bgs

def feats(X):
    """colour histogram features (what a small classifier 'sees')"""
    out = []
    for im in X:
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        h = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256]).flatten()
        out.append(h / h.sum())
    return np.array(out)


def make_all(seed=5):
    """The four sets used in the lecture: lab train, lab test (+masks, leaf layers), field, lab-random train."""
    rng = np.random.default_rng(seed)
    Xtr, ytr, _, _, _ = make_set(600, "lab", rng)
    Xte, yte, mte, lte, _ = make_set(300, "lab", rng)
    Xf, yf, _, _, _ = make_set(300, "field", rng)
    Xtr2, ytr2, _, _, _ = make_set(600, "lab_random", rng)
    return dict(Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte, masks=np.array(mte), leaves=np.array(lte), Xf=Xf, yf=yf, Xtr2=Xtr2, ytr2=ytr2)
