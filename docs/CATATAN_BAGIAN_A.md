# Catatan Belajar Bagian A — bahan, bukan jawaban final

> **Status:** catatan konsep dan referensi yang disusun dengan bantuan AI (Claude) sebagai *klarifikasi konsep*, sesuai §7.3 PDF tugas.
> Bagian A diberi judul *"Analisis Konseptual"*, dan §7.3 mewajibkan analisis ditulis secara mandiri. Karena itu, **jawaban 100–150 kata ditulis sendiri** dari catatan ini. Jangan menyalin kalimat dari sini ke laporan.
> Semua referensi di bawah sudah dicek keberadaannya (judul, venue, arXiv ID). Baca minimal abstraknya sebelum mengutip.

## Peta slide (sudah dicocokkan dengan isi PPT)

| Topik | Slide | Isi yang benar-benar ada di slide |
|---|---|---|
| BatchNorm | 6 | normalisasi aktivasi per mini-batch, *reduces internal covariate shift*, LR lebih tinggi, *acts as regularizer* |
| | 7 | rumus μ, σ², x̂ = (x−μ)/√(σ²+ε), y = γx̂ + … (**suku β terpotong di slide**) |
| | 8 | *acts like mild regularization*, *often eliminates the need for Dropout*, *enables deeper networks* |
| Residual | 10 | *deeper networks sometimes perform worse*, *gradient vanishing or explosion*, *information loss across layers* |
| | 11 | diagram blok residual, `y = F(x) + x`, *helps gradients flow through network* |
| | 12 | ResNet-18/34/50/101, *very stable for transfer learning* |
| Transfer learning | 14 | pretrained pada dataset besar, transfer ke dataset kecil, *fine-tune or freeze some layers* |
| | 15 | kapan dipakai: data terbatas, domain mirip ImageNet, training dari nol terlalu mahal |
| | 16 | **Feature Extraction:** bekukan semua layer pretrained, ganti head. **Fine-Tuning:** mulai dari bobot pretrained, *unfreeze deeper layers gradually*, *lower learning rate* |
| | 17–18 | kode freeze dan unfreeze `layer4` (slide 17 memakai API lama `pretrained=True`) |
| Augmentasi | 20 | transformasi pada *training images*, simulasi data tambahan, generalisasi pada dataset kecil |
| | 21 | flip (horizontal, **vertical**), rotation ±10–30°, crop & zoom, color jitter, CutMix/MixUp |
| | 22 | kode augmentasi PyTorch (tanpa `Normalize`) |
| | **23** | **hanya judul bagian "MODERN ARCHITECTURES"**. PDF menyebut slide 23 untuk A4; kemungkinan yang dimaksud slide 22 |
| EfficientNet | 24 | compound scaling (depth, width, resolution), B0→B7 |
| Workflow | 27 | pilih model → tentukan freeze/unfreeze → ganti classifier → augmentasi → train/eval → tuning |
| Challenges | 28 | ukuran input tidak cocok, *overfitting if pretrained model is too big*, *catastrophic forgetting if fine-tuned too fast* |
| Best practices | 29 | *always start with feature extraction*, *lr_scheduler for fine-tuning phases*, *EarlyStopping / validation checkpoints*, *visualize feature maps before/after fine-tuning* |

Saran penulisan: tandai sumber setiap klaim, misalnya "(Slide 8)" untuk isi slide dan "(Li et al., 2019)" untuk pengetahuan eksternal. Rubrik menilai ketepatan rujukan slide.

---

## A1 — Efek regularisasi BatchNorm (Slide 6–8)

**Mekanisme yang perlu dijelaskan:**
- Saat training, μ dan σ² dihitung dari *mini-batch* (Slide 7). Akibatnya, output ternormalisasi satu sampel **bergantung pada sampel lain** yang kebetulan berada di batch yang sama.
- Komposisi batch acak tiap iterasi menjadi semacam **noise multiplikatif dan aditif** pada aktivasi. Model tidak bisa mengandalkan nilai aktivasi yang persis sama untuk sampel yang sama, dan efeknya mirip *noise injection* (regularisasi).
- Besar noise turun ketika ukuran batch membesar (estimasi μ, σ² makin stabil). Saat inferensi, BN memakai *running statistics* sehingga deterministik. Luo et al. (2019) menganalisis efek ini secara teoretis.

**Kondisi ketika klaim "menghilangkan Dropout" tidak berlaku:**
- **Batch sangat kecil:** estimasi statistik terlalu berisik, dan ada ketidaksesuaian statistik train vs inferensi. Error naik tajam (Wu & He, 2018). Ioffe (2017) mengusulkan *Batch Renormalization* untuk masalah ini.
- **Batch besar:** noise kecil, sehingga efek regularisasinya lemah. Overfitting tetap bisa terjadi pada data kecil.
- **Dropout ditempatkan sebelum BN:** terjadi *variance shift*. Variansi aktivasi saat train (dengan dropout) berbeda dengan saat inferensi, sehingga running stats BN tidak cocok dan performa bisa menurun atau tidak stabil (Li et al., 2019).
- **Head fully-connected besar atau data sangat sedikit:** BN tidak menggantikan regularisasi eksplisit.
- **Kaitan dengan eksperimen:** pada B2/B3, BN di bagian yang dibekukan dijalankan dalam mode eval (statistik tetap), jadi **tidak ada efek regularisasi BN dari bagian itu**. Ini layak disebut sebagai nilai tambah.

## A2 — Degradation problem dan residual (Slide 10–12; Sesi 02)

**Poin kunci:**
- Degradation **bukan overfitting**. *Training error* model yang lebih dalam juga lebih tinggi (He et al., 2016a). Masalahnya ada pada **optimisasi**: secara teori, solusi identitas untuk layer tambahan ada, tetapi solver berbasis gradien sulit menemukannya melalui tumpukan layer nonlinier.
- Slide 10 menyebut penyebab *gradient vanishing/explosion* dan *information loss*. Nuansa yang perlu diketahui: He et al. (2016a) mencatat bahwa dengan BN, sinyal maju dan mundur tidak hilang, dan mereka menduga laju konvergensi yang sangat lambat sebagai penyebab. Sampaikan nuansa ini secara jujur.
- **Formulasi:** y = F(x) + x (Slide 11). Layer cukup belajar residual F(x) = H(x) − x. Mendorong F → 0 lebih mudah daripada mempelajari identitas dari nol.
- **Aliran gradien:** ∂y/∂x = I + ∂F/∂x. Untuk banyak blok berturut-turut (He et al., 2016b):
  ∂L/∂x_l = ∂L/∂x_L · (1 + ∂/∂x_l Σ_{i=l}^{L−1} F(x_i, W_i)).
  Suku "1" berarti gradien dari layer dalam mengalir **langsung** ke layer dangkal tanpa harus melalui perkalian banyak Jacobian.
- **Kaitan Sesi 02:** vanishing gradient muncul dari perkalian berantai turunan yang bernilai < 1 (contoh: turunan sigmoid ≤ 0,25). Jalur identitas mengganti perkalian murni menjadi penjumlahan, sehingga gradien tidak menyusut secara eksponensial terhadap kedalaman.

## A3 — Feature extraction vs fine-tuning (Slide 14–16)

Gunakan angka dari notebook. Angka berikut sudah diverifikasi oleh *assert* di kode.

| Dimensi | Feature extraction (B2) | Fine-tuning `layer4` (B3) |
|---|---|---|
| Trainable params | 5.130 (≈0,05%) | 8.398.858 (≈75%) |
| Kebutuhan data | kecil; hanya head linear yang dipelajari | lebih besar; jutaan parameter berisiko overfit pada ±1.400 citra |
| Komputasi/memori | backward berhenti di `fc`; tanpa *state* AdamW untuk backbone; fitur bahkan bisa diekstrak sekali saja | backward melewati `layer4`; aktivasi disimpan; AdamW menyimpan 2 *moment* per parameter (≈2 × 8,4 juta nilai) |
| Risiko *destructive update* | nol, karena fitur tidak berubah | ada: gradien besar dari head yang masih acak di awal training mengalir ke `layer4` (Kumar et al., 2022) |

**Kriteria praktis:** gunakan Slide 15 (data terbatas, domain mirip ImageNet, biaya) dan Slide 29 (*always start with feature extraction*). Yosinski et al. (2014): fitur layer awal bersifat umum, sedangkan fitur layer akhir lebih spesifik terhadap tugas. Semakin jauh domain target dari ImageNet, semakin besar manfaat membuka layer akhir.
**Catatan:** dataset kita berisi ras yang sebagian juga merupakan kelas ImageNet (lihat `imagenet_name_overlap.csv`). Hal ini memengaruhi seberapa kuat B2.

## A4 — Kaidah augmentasi (Slide 20–22; "Slide 23" di PDF kemungkinan salah ketik)

**Alasan augmentasi hanya pada train:**
- Augmentasi adalah regularisasi: memperluas distribusi *training* dengan variasi yang **mempertahankan label** (Slide 20).
- Val/test berfungsi sebagai **estimator performa pada distribusi data nyata**. Transformasi acak di val akan (a) membuat val loss berisik, padahal val loss dipakai untuk early stopping dan pemilihan checkpoint, (b) membuat perbandingan antar-epoch dan antar-model tidak setara, dan (c) mengukur performa pada data yang bukan distribusi deployment.
- Bedakan dengan *test-time augmentation* (TTA), yaitu teknik inferensi yang disengaja, diterapkan konsisten, dan dilaporkan terpisah. TTA bukan augmentasi acak pada evaluasi.

**Kandidat contoh transformasi yang tidak layak (pilih satu dan bangun argumennya sendiri):**
- **Hue shift kuat / ColorJitter hue besar:** warna bulu adalah fitur pembeda ras, misalnya bulu abu-kebiruan Russian Blue atau pola *colorpoint* Siamese. Mengubah hue bisa membuat citra kehilangan bukti visual labelnya, jadi label tidak lagi invarian terhadap transformasi ini.
- **Vertical flip:** slide 21 mencantumkannya sebagai teknik umum. Untuk foto hewan peliharaan, citra terbalik hampir tidak pernah muncul saat deployment. Argumennya lebih ke ketidaksesuaian distribusi daripada perubahan label, jadi argumen invariansi semantiknya lebih lemah. Jika memilih ini, jelaskan nuansa tersebut.
- **Crop sangat agresif** (mis. `RandomResizedCrop` scale 0,08): potongan bisa hanya berisi bulu atau latar, sehingga bukti ras (wajah, telinga) hilang.
- Referensi umum: Shorten & Khoshgoftaar (2019).

## A5 — Catastrophic forgetting (Slide 28–29)

**Mekanisme:**
- Di awal fine-tuning, head masih acak sehingga loss dan gradien besar. Dengan LR tinggi ("fine-tuned too fast", Slide 28), bobot pretrained bergeser jauh dari titik awalnya.
- Dengan data kecil, pergeseran itu mengikuti pola spesifik ±1.400 citra (bahkan noise), sehingga fitur umum hasil ImageNet tertimpa. Gejalanya: akurasi train naik, tetapi generalisasi turun. Fenomena ini pertama dideskripsikan sebagai *catastrophic interference* oleh McCloskey & Cohen (1989).

**Strategi mitigasi (pilih dua):**
1. **Differential (discriminative) learning rate:** LR kecil untuk layer pretrained (1e-4) dan LR besar untuk head (1e-3). Ukuran langkah kecil menjaga bobot backbone dekat dengan titik pretrained. Ini sudah diterapkan di B3 (Slide 16 *lower learning rate*; Howard & Ruder, 2018).
2. **Gradual unfreezing / feature extraction dulu:** latih head sampai stabil, baru buka layer dalam secara bertahap. Gradien yang masuk ke backbone jadi berasal dari head yang sudah bermakna, bukan acak (Slide 16 *unfreeze gradually*, Slide 29; Howard & Ruder, 2018; Kumar et al., 2022 menyebutnya LP-FT).
3. **Linear warmup / LR scheduler:** LR dinaikkan bertahap di awal agar pembaruan pertama tidak terlalu besar (Goyal et al., 2017; Slide 29 *lr_scheduler*).
4. **Regularisasi ke bobot awal:** penalti terhadap jarak dari bobot pretrained, misalnya EWC (Kirkpatrick et al., 2017).

---

## Daftar referensi (keberadaan sudah dicek; baca sebelum mengutip)
- Goyal, P., et al. (2017). Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour. arXiv:1706.02677.
- He, K., Zhang, X., Ren, S., & Sun, J. (2016a). Deep Residual Learning for Image Recognition. *CVPR*. arXiv:1512.03385.
- He, K., Zhang, X., Ren, S., & Sun, J. (2016b). Identity Mappings in Deep Residual Networks. *ECCV*. arXiv:1603.05027.
- Howard, J., & Ruder, S. (2018). Universal Language Model Fine-tuning for Text Classification. *ACL*. arXiv:1801.06146.
- Ioffe, S., & Szegedy, C. (2015). Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. *ICML*. arXiv:1502.03167.
- Ioffe, S. (2017). Batch Renormalization: Towards Reducing Minibatch Dependence in Batch-Normalized Models. *NeurIPS*. arXiv:1702.03275.
- Kirkpatrick, J., et al. (2017). Overcoming catastrophic forgetting in neural networks. *PNAS*, 114(13). arXiv:1612.00796.
- Kumar, A., Raghunathan, A., Jones, R., Ma, T., & Liang, P. (2022). Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution. *ICLR*. arXiv:2202.10054.
- Li, X., Chen, S., Hu, X., & Yang, J. (2019). Understanding the Disharmony between Dropout and Batch Normalization by Variance Shift. *CVPR*. arXiv:1801.05134.
- Luo, P., Wang, X., Shao, W., & Peng, Z. (2019). Towards Understanding Regularization in Batch Normalization. *ICLR*. arXiv:1809.00846.
- McCloskey, M., & Cohen, N. J. (1989). Catastrophic Interference in Connectionist Networks: The Sequential Learning Problem. *Psychology of Learning and Motivation*, 24.
- Shorten, C., & Khoshgoftaar, T. M. (2019). A survey on Image Data Augmentation for Deep Learning. *Journal of Big Data*, 6, 60.
- Wu, Y., & He, K. (2018). Group Normalization. *ECCV*. arXiv:1803.08494.
- Yosinski, J., Clune, J., Bengio, Y., & Lipson, H. (2014). How transferable are features in deep neural networks? *NeurIPS*. arXiv:1411.1792.
