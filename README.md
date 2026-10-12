# Tugas 05 — Transfer Learning untuk Klasifikasi Citra dengan Data Terbatas

**Nama:** Nicholas Salim · **NIM:** 2802420776 · **Mata kuliah:** Deep Learning, Sesi 05 *Advanced CNNs and Transfer Learning* (BINUS University)

Studi eksperimental yang membandingkan pelatihan *from scratch*, *feature extraction*, dan *progressive fine-tuning* ResNet-18, ditambah uji kontrol *data augmentation*, pada subset 10 kelas Oxford-IIIT Pet (±2.000 citra). Bonus: perbandingan dengan EfficientNet-B0.

## Dataset
* **Oxford-IIIT Pet** (Parkhi et al., CVPR 2012) melalui `torchvision.datasets.OxfordIIITPet`, diunduh otomatis oleh notebook.
* Split resmi `trainval` dan `test` digabung, lalu dipilih **10 kelas (5 ras kucing + 5 ras anjing)** secara acak dengan `np.random.default_rng(42)`.
* Split internal **stratified 70/15/15** (train/val/test) di level indeks dengan `random_state=42`. Daftar citra per split disimpan di `outputs/results/split.csv`, beserta hash MD5-nya.

## Protokol eksperimen
| Komponen | Nilai |
|---|---|
| Arsitektur | `torchvision.models.resnet18` (B5: `efficientnet_b0`) |
| Bobot pretrained | `ResNet18_Weights.IMAGENET1K_V1` (B2–B4) |
| Input | 224×224, normalisasi ImageNet |
| Optimizer / loss | AdamW (`weight_decay=1e-2`) / CrossEntropyLoss |
| Batch size | 32 |
| Early stopping | monitor *validation loss*, patience 5, `max_epochs` 40 |
| Seed | 42 (Python, NumPy, PyTorch, DataLoader generator; cuDNN deterministik) |
| DataLoader | `num_workers=2`, konteks `fork`; hasil hanya dapat diulang dengan `num_workers` dan jenis GPU yang sama |

| Kode | Kondisi | Parameter trainable | LR |
|---|---|---|---|
| B1 | from scratch (`weights=None`) | 11.181.642 | 1e-3 |
| B2 | feature extraction (backbone beku, `fc` 512→10) | 5.130 | fc 1e-3 |
| B3 | progressive fine-tuning (`layer4` + `fc`) | 8.398.858 | layer4 1e-4, fc 1e-3 |
| B4 | B3 tanpa vs dengan augmentasi | 8.398.858 | sama dengan B3 |
| B5 | EfficientNet-B0, konfigurasi terbaik dari validation | dicatat di log | dicatat di log |

Augmentasi (hanya pada train): `Resize(256)`, `RandomResizedCrop(224, scale=(0.8, 1.0))`, `RandomHorizontalFlip`, `RandomRotation(15)`, `ColorJitter(0.2, 0.2)`. Val dan test memakai `Resize(256)` + `CenterCrop(224)`.

## Cara menjalankan (Google Colab)
1. Buka `Tugas05_TransferLearning.ipynb` di Colab, lalu pilih `Runtime > Change runtime type > T4 GPU`.
2. Jalankan `Runtime > Run all` dan izinkan akses Google Drive. Artefak disimpan ke `MyDrive/Tugas05_DL/outputs/`.
3. Jika runtime terputus, jalankan `Run all` lagi. Kondisi yang sudah selesai dimuat dari Drive (`resume`), dan kondisi yang sedang berjalan saat terputus dilatih ulang dari epoch 1. Untuk run final yang bersih, hapus folder `outputs` di Drive terlebih dahulu.
4. B5 (bonus) dan C5 (opsional) aktif secara default. Jika gagal, keduanya dicatat (`b5_status.json`, `c5_status.json`) lalu dilewati tanpa menghentikan sel wajib. Untuk mematikannya, set `"run_b5": False` / `"run_c5": False` di sel konfigurasi.
5. Unduh notebook beserta output (`File > Download > .ipynb`) dan `outputs_bundle.zip`.

Untuk menjalankan di luar Colab: `pip install -r requirements.txt`, lalu `jupyter nbconvert --to notebook --execute Tugas05_TransferLearning.ipynb`. Lokasi data dan output bisa diatur lewat `TUGAS05_DATA_ROOT` dan `TUGAS05_OUT_DIR`.

### Smoke test (opsional, tanpa GPU dan tanpa internet)
`tools/make_fake_pets.py` membuat dataset tiruan dengan struktur folder dan anotasi persis seperti Oxford-IIIT Pet. Dataset ini dipakai untuk menguji bahwa seluruh sel notebook berjalan:
```bash
python tools/make_fake_pets.py ./fake_data 24
TUGAS05_SMOKE=1 TUGAS05_DATA_ROOT=./fake_data TUGAS05_OUT_DIR=./smoke_outputs \
  jupyter nbconvert --to notebook --execute Tugas05_TransferLearning.ipynb --output smoke.ipynb
```
Dalam mode ini bobot pretrained tidak diunduh dan training hanya 2 epoch. **Angka smoke test bukan hasil eksperimen.**

## Hasil eksekusi
Notebook dijalankan satu kali dari awal sampai akhir di Google Colab (Tesla T4, PyTorch 2.11.0+cu130, torchvision 0.26.0+cu130); selesai 2026-10-12 03:47:09 UTC. Pemeriksaan otomatis: 14/14 PASS (`outputs/results/quality_checks.csv`). Model terbaik berdasarkan validation: **B5**.

Tabel berikut disalin otomatis dari `outputs/results/summary_table.md`:

| Kode     | Strategi           | Augmentasi   |   Trainable params | Learning rate                                                             |   Epoch konvergen (best) |   Epoch dijalankan |   Durasi (s) |   Val acc |   Test Macro F1 | Test Macro F1 CI95   |
|:---------|:-------------------|:-------------|-------------------:|:--------------------------------------------------------------------------|-------------------------:|-------------------:|-------------:|----------:|----------------:|:---------------------|
| B1       | scratch            | aug          |           11181642 | all=0.001                                                                 |                       15 |                 20 |        168.5 |    0.532  |          0.4957 | [0.438, 0.546]       |
| B2       | feature_extraction | aug          |               5130 | fc=0.001                                                                  |                       22 |                 27 |        188.8 |    0.9428 |          0.9254 | [0.894, 0.952]       |
| B3       | finetune_last      | aug          |            8398858 | layer4=0.0001, fc=0.001                                                   |                        3 |                  8 |         55.2 |    0.9428 |          0.9385 | [0.909, 0.964]       |
| B4_noaug | finetune_last      | noaug        |            8398858 | layer4=0.0001, fc=0.001                                                   |                        8 |                 13 |         42.6 |    0.9596 |          0.9552 | [0.930, 0.976]       |
| B4_aug   | finetune_last      | aug          |            8398858 | layer4=0.0001, fc=0.001                                                   |                        3 |                  8 |         55.6 |    0.9428 |          0.9385 | [0.909, 0.964]       |
| B5       | finetune_last      | noaug        |            3168550 | features.6=0.0001, features.7=0.0001, features.8=0.0001, classifier=0.001 |                        5 |                 10 |         43   |    0.963  |          0.9762 | [0.957, 0.992]       |

## Struktur berkas
```
Tugas05_TransferLearning.ipynb   # notebook utama (dikumpulkan dalam keadaan sudah dieksekusi)
requirements.txt
README.md
tools/make_fake_pets.py          # pembuat dataset tiruan untuk smoke test
tools/finalize.py                # report: isi angka laporan dari log -> PDF | facts: docs/FAKTA_HASIL.md | zip: susun & periksa ZIP
docs/LAPORAN_Tugas05_final.pdf   # laporan (angka faktual diisi otomatis dari outputs/results)
outputs/
  results/   summary.csv|json, summary_<kode>.json, history_<kode>.csv,
             test_predictions_<kode>.csv, split.csv, config.json, failure_cases_*.csv, ...
  figures/   loss_curves_B1_B2_B3.png, B4_augmentation_gap.png, confusion_matrix_*.png,
             failure_cases_*.png, gradcam_layer4_B2_vs_B3.png, ...
```
Checkpoint model (`outputs/checkpoints/*.pt`, ±45 MB per model) tidak dimasukkan ke repositori.

## Integritas data
Semua angka di laporan berasal dari `outputs/results/summary.csv` / `summary.json`. Kurva berasal dari `history_<kode>.csv`, dan failure analysis dari `test_predictions_<kode>.csv`. Notebook melakukan pemeriksaan otomatis (lihat `quality_checks.csv`) terhadap kebocoran split, jumlah parameter trainable, tensor beku yang tidak berubah (termasuk buffer BatchNorm), konsistensi Macro F1, dan asal sampel failure analysis.

## Referensi
* He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. *CVPR*.
* Parkhi, O. M., Vedaldi, A., Zisserman, A., & Jawahar, C. V. (2012). Cats and Dogs. *CVPR*.
* Tan, M., & Le, Q. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. *ICML*.
* Selvaraju, R. R., et al. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. *ICCV*.
* Kornblith, S., Norouzi, M., Lee, H., & Hinton, G. (2019). Similarity of Neural Network Representations Revisited. *ICML*.
* Dokumentasi torchvision: `torchvision.datasets.OxfordIIITPet`, `torchvision.models`.

## Pernyataan penggunaan AI
Lihat lampiran *AI Disclosure Statement* di laporan (`docs/`).
