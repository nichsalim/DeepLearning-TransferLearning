# Lembar fakta hasil eksperimen (dibangkitkan otomatis dari outputs/results)

> Isi berkas ini hanya fakta dan perbandingan deskriptif dari log. **Interpretasi (mengapa) ditulis sendiri di laporan.**

## Ringkasan per kondisi

| Kode | Best/jalan | Val loss min | Val acc | Train acc (clean) | Gap acc | Gap loss (val−train) | Test acc | Test F1 | s/epoch | Durasi (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| B1 | 15/20 | 1.3495 | 0.5320 | 0.5871 | +0.0551 | +0.1867 | 0.5017 | 0.4957 | 7.66 | 168.5 |
| B2 | 22/27 | 0.1713 | 0.9428 | 0.9516 | +0.0088 | +0.0323 | 0.9259 | 0.9254 | 6.29 | 188.8 |
| B3 | 3/8 | 0.1744 | 0.9428 | 0.9783 | +0.0355 | +0.0966 | 0.9394 | 0.9385 | 6.22 | 55.2 |
| B4_noaug | 8/13 | 0.1554 | 0.9596 | 1.0000 | +0.0404 | +0.1544 | 0.9562 | 0.9552 | 2.59 | 42.6 |
| B4_aug | 3/8 | 0.1744 | 0.9428 | 0.9783 | +0.0355 | +0.0966 | 0.9394 | 0.9385 | 6.30 | 55.6 |
| B5 | 5/10 | 0.1032 | 0.9630 | 1.0000 | +0.0370 | +0.0980 | 0.9764 | 0.9762 | 3.58 | 43.0 |

**Catatan durasi (fakta terukur):** B4_noaug dan B4_aug memakai model, optimizer, dan data identik; satu-satunya beda adalah pipeline augmentasi train. s/epoch: B4_noaug = 2.59, B4_aug = 6.30. Waktu per epoch karenanya mencakup biaya pipeline data di CPU, bukan hanya komputasi GPU. Bandingkan durasi antar-kondisi yang memakai pipeline transform yang sama.

## Perbandingan berpasangan (Macro F1 test, bootstrap 95%)

- B2 − B1 = +0.4297, CI95 [+0.3740, +0.4900] → CI tidak memuat 0
- B3 − B1 = +0.4429, CI95 [+0.3899, +0.5008] → CI tidak memuat 0
- B3 − B2 = +0.0132, CI95 [-0.0114, +0.0409] → CI memuat 0 (tidak dapat dibedakan dari noise ukuran test set)
- B4_aug − B4_noaug = -0.0167, CI95 [-0.0412, +0.0071] → CI memuat 0 (tidak dapat dibedakan dari noise ukuran test set)
- B4_aug − B3 = +0.0000, CI95 [+0.0000, +0.0000] → CI memuat 0 (tidak dapat dibedakan dari noise ukuran test set)
- B5 − B4_noaug = +0.0210, CI95 [+0.0003, +0.0444] → CI tidak memuat 0

## Pola kurva (dari history_*.csv)

- B1: val_loss epoch 1 = 2.2997, minimum = 1.3495 (epoch 15), terakhir = 1.6908; train_loss epoch 1 = 2.3696, terakhir = 1.0332
- B2: val_loss epoch 1 = 0.6497, minimum = 0.1713 (epoch 22), terakhir = 0.1742; train_loss epoch 1 = 1.3405, terakhir = 0.1116
- B3: val_loss epoch 1 = 0.2186, minimum = 0.1744 (epoch 3), terakhir = 0.1749; train_loss epoch 1 = 0.7871, terakhir = 0.0297
- B4_noaug: val_loss epoch 1 = 0.2268, minimum = 0.1554 (epoch 8), terakhir = 0.1958; train_loss epoch 1 = 0.7038, terakhir = 0.0028
- B4_aug: val_loss epoch 1 = 0.2186, minimum = 0.1744 (epoch 3), terakhir = 0.1749; train_loss epoch 1 = 0.7871, terakhir = 0.0297
- B5: val_loss epoch 1 = 0.2328, minimum = 0.1032 (epoch 5), terakhir = 0.1378; train_loss epoch 1 = 1.0134, terakhir = 0.0135

## Failure cases model terbaik (B5, dipilih via validation)

- 7 misklasifikasi ditampilkan; 7 di antaranya terjadi di dalam spesies yang sama (kucing→kucing / anjing→anjing).
- american_pit_bull_terrier_43: American Pit Bull Terrier → Staffordshire Bull Terrier (conf 0.967; p(GT) 0.033; top-3: Staffordshire Bull Terrier 0.97; American Pit Bull Terrier 0.03; Maine Coon 0.00)
- american_pit_bull_terrier_53: American Pit Bull Terrier → Staffordshire Bull Terrier (conf 0.923; p(GT) 0.077; top-3: Staffordshire Bull Terrier 0.92; American Pit Bull Terrier 0.08; Maine Coon 0.00)
- american_pit_bull_terrier_65: American Pit Bull Terrier → Staffordshire Bull Terrier (conf 0.842; p(GT) 0.158; top-3: Staffordshire Bull Terrier 0.84; American Pit Bull Terrier 0.16; Abyssinian 0.00)
- British_Shorthair_34: British Shorthair → Russian Blue (conf 0.836; p(GT) 0.127; top-3: Russian Blue 0.84; British Shorthair 0.13; Staffordshire Bull Terrier 0.02)
- Egyptian_Mau_155: Egyptian Mau → Abyssinian (conf 0.757; p(GT) 0.239; top-3: Abyssinian 0.76; Egyptian Mau 0.24; Maine Coon 0.00)
- British_Shorthair_190: British Shorthair → Russian Blue (conf 0.631; p(GT) 0.324; top-3: Russian Blue 0.63; British Shorthair 0.32; Staffordshire Bull Terrier 0.03)
- Egyptian_Mau_209: Egyptian Mau → Abyssinian (conf 0.364; p(GT) 0.268; top-3: Abyssinian 0.36; Maine Coon 0.33; Egyptian Mau 0.27)

## C5: linear CKA fitur pooled layer4, B2 vs B3 (test) = 0.8375

## Pertanyaan pemandu (jawab sendiri, kaitkan ke slide)

- C2.1 (Slide 10–12, 14–16): apa yang membedakan representasi awal B1 vs B2/B3, dan apakah B1 dibatasi data, epoch, atau keduanya?
- C2.2 (Slide 16, 28): selisih B3−B2 signifikan atau tidak? Bandingkan biaya (s/epoch, trainable params) dan kecepatan konvergensi.
- C2.3 (Slide 20–22): augmentasi mengubah gap acc dan gap loss ke arah mana? Apakah itu sejalan dengan perubahan val/test acc?
- C3: pasangan ras mana yang tertukar, dan ciri visual apa yang terlihat pada citranya (pose, warna, pencahayaan, crop)?
- C4: dengan fakta di atas, strategi apa yang direkomendasikan, dan dalam batasan apa (data, domain, biaya, single seed)?
