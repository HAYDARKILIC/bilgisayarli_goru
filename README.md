# Bilgisayarlı Görü: Piksellerden Derin Ağlara

Modern bilgisayarlı görüyü **saf NumPy ile sıfırdan** inşa eden, altı haftalık ve ilk ilkelere
dayanan bir kurs — görüntü işlemenin klasik matematiğini, bugün görüyü güçlendiren evrişimli
ağlara bağlıyor. Gauss bulanıklığından geri yayılımı yapılmış bir evrişim katmanına kadar her
operatör, herhangi bir üst düzey kütüphaneye izin verilmeden önce elle türetilip uygulanıyor.

Her ders, kendi kendine yeten bir Jupyter notebook'udur: matematiksel türetme, sıfırdan bir
uygulama, gerçek görüntüler üzerinde görselleştirmeler, ilgili yerlerde `scipy`/`skimage` ile
doğrulama ve çözümleriyle birlikte alıştırmalar.

---

## Felsefe

Çoğu bilgisayarlı görü kursu `import torch` ile başlar ve evrişimi bir kara kutu gibi ele alır. Bu
kurs ise evrişimi tanımlayan integralle başlar ve her gradyanını kendiniz türettiğiniz bir CNN ile
biter. Ana eksen her zaman şudur:

> **matematiği türet → NumPy'da uygula → gerçek bir görüntüde görselleştir → bir referansa karşı doğrula → modern bir modele bağla.**

Kursun sonunda şunları elle yazmış olacaksınız: 2B evrişim, Gauss ve türev filtreleri, Canny
kenar dedektörü, Harris köşeleri, Hough dönüşümü, Lucas–Kanade optik akış denklemleri, k-ortalamalar
ve çizge tabanlı bölütleme, geri yayılımlı tam bir evrişim katmanı ve eğitilmiş küçük bir CNN — ve
her derin öğrenme bileşeninin hangi klasik fikri genellediğini tam olarak anlayacaksınız.

---

## Önkoşullar

- Rahat Python ve NumPy bilgisi (diziler, dilimleme, yayınlama/broadcasting)
- `linear_algebra_for_ml` eşlik eden kursu düzeyinde doğrusal cebir (vektörler, matrisler,
  doğrusal bir operatör olarak evrişim, Harris dedektörü ve PCA için özdeğerler)
- Temel kalkülüs (gradyanlar, geri yayılım için zincir kuralı)

---

## Müfredat

### Hafta 1 — Diziler Olarak Görüntüler ve Nokta İşlemleri
Bir dijital görüntünün *ne olduğu*: bir sayı ızgarasına örneklenmiş bir fonksiyon. Pikseller,
kanallar, renk uzayları, histogramlar ve nokta işlemleri (parlaklık, kontrast, gama, eşikleme,
histogram eşitleme). Geri kalan her şeyin üzerine inşa edildiği temel.
- `01_images_as_arrays.ipynb`
- `02_point_operations_histograms.ipynb`

### Hafta 2 — Evrişim ve Doğrusal Filtreleme
Görüdeki en önemli tek işlem. Evrişimi ilk ilkelerden türetin, sıfırdan uygulayın ve klasik
filtreleri inşa edin: kutu, Gauss, keskinleştirme. Ayrılabilir çekirdekler, kenarlar ve evrişimin
neden doğrusal ötelemeye değişmez (linear shift-invariant) bir sistem olduğu.
- `03_convolution_from_scratch.ipynb`
- `04_smoothing_gaussian_filters.ipynb`

### Hafta 3 — Gradyanlar, Kenarlar ve Canny Dedektörü
Bir görüntüde bilgi nerede yaşar? Kenarlarında. Görüntü gradyanları, Sobel operatörleri,
Laplas operatörü ve elle inşa edilen çok aşamalı tam **Canny kenar dedektörü** — maksimum olmayanın
bastırılması (non-maximum suppression) ve histerezis dahil.
- `05_image_gradients_edges.ipynb`
- `06_canny_edge_detector.ipynb`

### Hafta 4 — Öznitelikler, Köşeler ve Eşleştirme
Ayırt edici noktaları bulma ve betimleme. **Harris köşe dedektörü** (bir özdeğer problemi —
doğrudan doğrusal cebirden), ölçek boyunca damla (blob) tespiti, basit betimleyiciler ve iki görünüm
arasında öznitelik eşleştirme — panoramaların ve hareketten yapı (structure-from-motion)
çıkarımının temeli.
- `07_harris_corners.ipynb`
- `08_blobs_features_matching.ipynb`

### Hafta 5 — Bölütleme ve Hareket
Pikselleri gruplama ve zaman içinde izleme. Eşikleme, k-ortalamalar renk bölütlemesi, doğrular için
Hough dönüşümü ve parlaklık sabitliği varsayımından türetilip uygulanan **Lucas–Kanade optik akış**
denklemleri.
- `09_segmentation_hough_kmeans.ipynb`
- `10_optical_flow_lucas_kanade.ipynb`

### Hafta 6 — Sıfırdan Evrişimli Sinir Ağları
Modern görüye köprü. NumPy'da tam ileri ve geri geçişli bir evrişim katmanı inşa edin, havuzlama
ve bir sınıflandırıcı başlığı ekleyin, gerçek rakamlar üzerinde küçük bir CNN eğitin ve öğrenilen
filtreleri görselleştirin — bunları ta Hafta 3'ün elle tasarlanmış kenar dedektörlerine kadar
bağlayarak.
- `11_cnn_layer_from_scratch.ipynb`
- `12_training_a_cnn_capstone.ipynb`

---

## Depo yapısı

```
computer_vision/
├── README.md
├── requirements.txt
├── notebooks/          # 12 ders notebook'u
├── utils/              # paylaşılan görüntü & çizim yardımcıları
│   └── cv_utils.py
├── data/               # örnek görüntüler notebook içinde üretilir (büyük ikili dosya yok)
└── assets/             # dışa aktarılan figürler
```

## Başlarken

```bash
git clone https://github.com/HAYDARKILIC/computer_vision.git
cd computer_vision
pip install -r requirements.txt
jupyter lab
```

`notebooks/01_images_as_arrays.ipynb` dosyasını açın ve yukarıdan aşağıya çalışın. Her notebook
bağımsız çalışır ve kendi örnek görüntülerini üretir, dolayısıyla büyük görüntü indirmeleri yoktur.

---

## Bu kurs nasıl kullanılır

1. **Türetmeyi okuyun** — her operatör kod olarak sunulmaz, matematiksel olarak gerekçelendirilir.
2. Verileni okumadan önce **uygulamayı kendiniz yazın**.
3. **Görselleştirmeleri çalıştırın** — çekirdeği, eşiği, görüntüyü değiştirin; neyin bozulduğunu izleyin.
4. **Alıştırmaları yapın** — çözümler her notebook'un en altında katlanmış halde bulunur.
5. **İleriye bağlayın** — her notebook "Bunun modern görüde nerede karşımıza çıktığı" ile biter.

---

## Lisans

MIT — öğretim ve bireysel çalışma için serbestçe kullanılabilir.
