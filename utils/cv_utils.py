"""
cv_utils.py
Bilgisayarlı Görü kursu için paylaşılan yardımcılar.
Bağımlılığı az: numpy + matplotlib (+ notebook'ların içinde isteğe bağlı scipy/skimage).

Örnek görüntüler sentetik olarak üretilir, böylece notebook'lar harici indirme gerektirmez.
"""

import numpy as np
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------------- görüntüleme
def show(img, title=None, ax=None, cmap="gray", vmin=None, vmax=None):
    """Tek bir görüntüyü (gri tonlamalı veya RGB) makul varsayılanlarla görüntüle."""
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))
    if img.ndim == 2:
        ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    else:
        ax.imshow(np.clip(img, 0, 1) if img.dtype != np.uint8 else img)
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=11)
    return ax


def show_row(images, titles=None, cmap="gray", figsize=None):
    """Yan yana karşılaştırma için bir satır görüntü göster."""
    n = len(images)
    figsize = figsize or (3.2 * n, 3.2)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]
    for i, (ax, im) in enumerate(zip(axes, images)):
        t = titles[i] if titles else None
        show(im, title=t, ax=ax, cmap=cmap)
    plt.tight_layout()
    return axes


# ----------------------------------------------------------------------------- sentetik görüntüler
def sample_gray(n=128, seed=0):
    """Sentetik bir gri tonlamalı sahne: gradyanlar, şekiller, kenarlar ve doku.
    [0, 1] aralığında float bir görüntü döndürür. Tohum (seed) verildiğinde deterministiktir."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    img = 0.35 + 0.25 * (xx / n)                      # düzgün yatay rampa
    # parlak bir kare
    img[n//6:n//2, n//6:n//2] = 0.85
    # koyu bir disk
    cy, cx, r = int(0.68 * n), int(0.68 * n), n // 6
    img[(yy - cy) ** 2 + (xx - cx) ** 2 < r ** 2] = 0.15
    # ince bir çapraz çizgi (güçlü bir kenar)
    for t in range(n):
        j = int(0.15 * n + 0.5 * t)
        if 0 <= j < n:
            img[t, j] = 0.95
    # ince dokulu bir yama
    img[n//12:n//4, int(0.6*n):int(0.85*n)] += 0.15 * rng.standard_normal((n//4 - n//12, int(0.85*n) - int(0.6*n)))
    return np.clip(img, 0, 1)


def sample_rgb(n=128, seed=0):
    """Bir gradyan üzerinde üç renkli bölgeye sahip sentetik bir RGB sahne."""
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    img = np.zeros((n, n, 3))
    img[..., 0] = 0.2 + 0.6 * (xx / n)               # kırmızı rampa
    img[..., 1] = 0.2 + 0.6 * (yy / n)               # yeşil rampa
    img[..., 2] = 0.5
    img[n//6:n//2, n//6:n//2] = [0.9, 0.2, 0.2]      # kırmızı kare
    cy, cx, r = int(0.68 * n), int(0.35 * n), n // 7
    img[(yy - cy) ** 2 + (xx - cx) ** 2 < r ** 2] = [0.2, 0.4, 0.9]   # mavi disk
    return np.clip(img, 0, 1)


def checkerboard(n=128, squares=8):
    """Temiz bir satranç tahtası — köşe ve kenar demoları için ideal."""
    s = n // squares
    board = (np.indices((n, n)) // s).sum(axis=0) % 2
    return board.astype(float)


def to_gray(rgb):
    """Parlaklık (luminance) dönüşümü (Rec. 601 ağırlıkları)."""
    return rgb @ np.array([0.299, 0.587, 0.114])


# ----------------------------------------------------------------------------- doğrulama
def check(name, mine, reference, atol=1e-6):
    """Sıfırdan elde edilen bir sonuç ile bir referans arasındaki karşılaştırmayı düzgünce yazdır."""
    mine = np.asarray(mine, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if mine.shape != reference.shape:
        print(f"[BAŞARISIZ] {name:<34} biçim {mine.shape} vs {reference.shape}")
        return False
    ok = np.allclose(mine, reference, atol=atol)
    err = float(np.max(np.abs(mine - reference)))
    print(f"[{'GEÇTİ' if ok else 'BAŞARISIZ'}] {name:<34} maks|Δ| = {err:.2e}")
    return ok
