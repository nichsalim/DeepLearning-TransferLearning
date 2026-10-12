"""Finalisasi pengumpulan Tugas 05.

  python tools/finalize.py report   -> docs/LAPORAN_Tugas05_final.md + .pdf (angka diisi dari outputs/results)
  python tools/finalize.py zip      -> Tugas05_2802420776_NicholasSalim.zip + daftar pemeriksaan wajib

Semua angka faktual di laporan diambil dari berkas log; skrip ini tidak pernah membuat angka.
"""
import json, re, subprocess, sys, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES, FIG, DOCS = ROOT / "outputs" / "results", ROOT / "outputs" / "figures", ROOT / "docs"
NB = ROOT / "Tugas05_TransferLearning.ipynb"
TEMPLATE, FINAL_MD, FINAL_PDF = DOCS / "LAPORAN_Tugas05.md", DOCS / "LAPORAN_Tugas05_final.md", DOCS / "LAPORAN_Tugas05_final.pdf"
ZIP_NAME = "Tugas05_2802420776_NicholasSalim.zip"
MANDATORY = ["B1", "B2", "B3", "B4_noaug", "B4_aug"]


def need(path):
    if not path.exists():
        sys.exit(f"[GAGAL] berkas tidak ditemukan: {path.relative_to(ROOT)}")
    return path


def read_csv(path):
    import csv
    with open(need(path), newline="") as f:
        return list(csv.DictReader(f))


def fill_tokens():
    cfg = json.loads(need(RES / "config.json").read_text())
    if cfg["env"].get("smoke_test"):
        sys.exit("[GAGAL] outputs berasal dari SMOKE TEST — jalankan notebook dengan data asli dulu.")
    split = read_csv(RES / "split.csv")
    classes = read_csv(RES / "selected_classes.csv")
    best = json.loads(need(RES / "best_model.json").read_text())["best_code"]
    fsum = json.loads(need(RES / f"failure_summary_{best}.json").read_text())
    cases = read_csv(RES / f"failure_cases_{best}.csv")
    dropped = read_csv(RES / "dropped_images.csv") if (RES / "dropped_images.csv").exists() else []
    count = lambda s: sum(r["split"] == s for r in split)
    env = cfg["env"]

    # Hipotesis disimpan terpisah (docs/C3_hipotesis.csv) agar CSV log mentah tidak diubah.
    hyp_file = DOCS / "C3_hipotesis.csv"
    hyps = {r["image_id"]: r["hipotesis"] for r in read_csv(hyp_file)} if hyp_file.exists() else {}
    # jumlah tanda '-' menentukan lebar relatif kolom di pandoc (kolom hipotesis dibuat paling lebar)
    rows = ["| # | image_id | Ground truth | Prediksi | Conf. | p(GT) | Hipotesis penyebab |",
            "|-|------------|--------|--------|---|---|------------------------------------------------|"]
    for i, r in enumerate(cases, 1):
        hyp = hyps.get(r["image_id"], "").strip() or r.get("hipotesis_penyebab", "").strip() or "[DIISI MAHASISWA]"
        if r["jenis"] != "misklasifikasi":
            hyp = f"(pelengkap, prediksi benar) {hyp}"
        rows.append(f"| {i} | {r['image_id']} | {r['true_name']} | {r['pred_name']} | "
                    f"{float(r['confidence']):.3f} | {float(r['prob_true']):.3f} | {hyp} |")

    return {
        "CLASS_LIST": "; ".join(f"{c['class_name']} ({'kucing' if c['species'] == 'cat' else 'anjing'})" for c in classes),
        "N_TOTAL": str(len(split)), "N_TRAIN": str(count("train")), "N_VAL": str(count("val")), "N_TEST": str(count("test")),
        "DROPPED_NOTE": (f" Sebanyak {len(dropped)} berkas citra tidak terbaca dan dibuang sebelum split (`dropped_images.csv`)."
                         if dropped else ""),
        "ENV": f"{env.get('gpu_name') or env['device']}, PyTorch {env['torch']}, torchvision {env['torchvision']}, Python {env['python']}",
        "C1_TABLE": need(RES / "summary_table.md").read_text().strip(),
        "BEST_CODE": best, "N_MISCLASSIFIED": str(fsum["n_misclassified"]),
        "FAILURE_TABLE": "\n".join(rows),
    }


def numbers_not_in_logs(text):
    """Angka desimal di laporan yang tidak ditemukan di log (perlu dicek manual)."""
    vals = set()
    def walk(x):
        if isinstance(x, dict): [walk(v) for v in x.values()]
        elif isinstance(x, list): [walk(v) for v in x]
        elif isinstance(x, (int, float)) and not isinstance(x, bool):
            for d in (1, 2, 3, 4):
                vals.add(f"{round(abs(x), d):.{d}f}"); vals.add(f"{round(100 * abs(x), d):.{d}f}")
    for f in RES.glob("*.json"):
        walk(json.loads(f.read_text()))
    for f in RES.glob("*.csv"):
        for v in re.findall(r"-?\d+\.\d+", f.read_text()):
            x = abs(float(v))
            for d in (1, 2, 3, 4):
                vals.add(f"{round(x, d):.{d}f}"); vals.add(f"{round(100 * x, d):.{d}f}")
    ints = set()
    for f in RES.glob("*.json"):
        ints.update(re.findall(r"(?<![\d.])\d{4,}(?![\d.])", f.read_text()))
    for f in RES.glob("*.csv"):
        ints.add(str(sum(1 for _ in open(f)) - 1))
    if (RES / "split.csv").exists():                                   # jumlah citra per split
        import csv as _csv
        from collections import Counter
        ints.update(str(v) for v in Counter(r["split"] for r in _csv.DictReader(open(RES / "split.csv"))).values())
    body = re.sub(r"\(\s*\d{4}[a-z]?\)|arXiv:\S+|\d+\(\d+\)", "", text)
    bad = set()
    for n in re.findall(r"(?<![\w.,])\d{1,3}(?:\.\d{3})+(?![\d,])", body):
        # bisa ribuan gaya Indonesia (8.398.858) atau desimal 3 digit (0.963): terima bila salah satu cocok dengan log
        if n.replace(".", "") not in ints and n not in vals:
            bad.add(n)
    body = re.sub(r"(?<![\w.,])\d{1,3}(?:\.\d{3})+(?![\d,])", "", body)
    found = re.findall(r"(?<![\w.])\d+[.,]\d+", body)
    bad.update(n for n in found if n.replace(",", ".") not in vals)
    return sorted(bad)


PLACEHOLDER_RE = re.compile(r"\[(?:DIISI|TEMPEL|Tambahkan|hapus jika)[^\]]*\]")
CSS = ("body{font-family:'DejaVu Serif',serif;font-size:10pt;line-height:1.35;max-width:none;margin:0}"
       "h1{font-size:15pt}h2{font-size:12pt;margin-top:14px}table{border-collapse:collapse;font-size:7.5pt;width:100%}"
       "td,th{border:1px solid #999;padding:2px 4px}img{max-width:100%;max-height:330px;display:block;margin:6px auto}")


def pdf_pages(path):
    out = subprocess.run(["pdfinfo", str(path)], capture_output=True, text=True, check=True).stdout
    return int(re.search(r"Pages:\s+(\d+)", out).group(1))


def render_pdf(text, out):
    html, css = DOCS / "_report.html", DOCS / "_report.css"
    css.write_text(CSS)
    try:
        subprocess.run(["pandoc", "-f", "markdown", "-t", "html", "-s", "--css", str(css), "--metadata", "title= ",
                        "-o", str(html)], input=text.encode(), check=True)
        subprocess.run(["wkhtmltopdf", "-q", "--enable-local-file-access", "-s", "A4", "-T", "18mm", "-B", "18mm",
                        "-L", "18mm", "-R", "18mm", str(html), str(out)], check=True)
    finally:
        html.unlink(missing_ok=True); css.unlink(missing_ok=True)
    return pdf_pages(out)


def part_a_word_counts(text):
    counts = {}
    for num, body in re.findall(r"\*\*A(\d)\.[^*]*\*\*(.*?)(?=\*\*A\d\.|## C1)", text, flags=re.S):
        counts[f"A{num}"] = None if PLACEHOLDER_RE.search(body) else len(re.findall(r"\S+", body))
    return counts


def analyze(text):
    """Pemeriksaan isi laporan final (tanpa menilai benar/tidaknya interpretasi)."""
    imgs = re.findall(r"!\[[^\]]*\]\(([^)\s]+)\)", text)
    return {
        "token_tersisa": re.findall(r"\{\{\w+\}\}", text),
        "placeholder_tersisa": PLACEHOLDER_RE.findall(text),
        "gambar_hilang": [i for i in imgs if not (DOCS / i).resolve().exists()],
        "kata_bagian_A": part_a_word_counts(text),
        "checkbox_belum_dicentang": text.count("- [ ]"),
        "angka_tidak_ditemukan_di_log": numbers_not_in_logs(text),
    }


def make_failure_grid(best, n_rows):
    """Susun ulang panel citra TEST (kolom kiri) dari failure_cases_<best>.png menjadi grid ringkas untuk laporan.
    Murni memotong gambar keluaran notebook; tidak ada citra/label baru."""
    from PIL import Image
    src = FIG / f"failure_cases_{best}.png"
    if not src.exists() or n_rows == 0:
        return None
    im = Image.open(src).convert("RGB")
    W, H = im.size
    h, w = H // n_rows, int(W * 0.535)
    tiles = [im.crop((0, r * h, w, (r + 1) * h)) for r in range(n_rows)]
    cols = 4
    rows = -(-n_rows // cols)
    grid = Image.new("RGB", (cols * w, rows * h), "white")
    for i, t in enumerate(tiles):
        grid.paste(t, ((i % cols) * w, (i // cols) * h))
    out = FIG / f"failure_cases_{best}_grid.png"
    grid.save(out)
    return out


def report():
    tokens = fill_tokens()
    best = tokens["BEST_CODE"]
    fs = json.loads((RES / f"failure_summary_{best}.json").read_text())
    make_failure_grid(best, fs["n_misclassified_shown"] + fs["n_supplementary_correct_shown"])
    diffs = read_csv(RES / "paired_bootstrap_f1_diff.csv")
    tokens["C2_DIFFS"] = "\n".join(
        ["| A vs B | Selisih Macro F1 (A−B) | CI95 bootstrap | CI memuat 0? |", "|---|---|---|---|"] +
        [f"| {r['A']} vs {r['B']} | {float(r['F1(A)-F1(B) titik']):+.4f} | [{float(r['CI95 bawah']):+.3f}, "
         f"{float(r['CI95 atas']):+.3f}] | {'ya' if r['CI memuat 0?'] == 'True' else 'tidak'} |" for r in diffs])
    text = re.sub(r"<!--.*?-->", "", need(TEMPLATE).read_text(), flags=re.S)
    c5 = RES / "c5_status.json"
    if not (c5.exists() and json.loads(c5.read_text()).get("ok") and (FIG / "gradcam_layer4_B2_vs_B3.png").exists()):
        text = re.sub(r"## C5\..*?(?=## Referensi)", "", text, flags=re.S)   # C5 opsional & tidak dijalankan
        print("  C5 tidak tersedia -> bagian C5 dihapus dari laporan")
    for k, v in tokens.items():
        text = text.replace("{{" + k + "}}", v)
    FINAL_MD.write_text(text)

    abs_text = text.replace("../outputs/figures/", str(FIG) + "/")
    body = abs_text.split("## Lampiran")[0]
    body_pages = render_pdf(body, DOCS / "_body_only.pdf")
    (DOCS / "_body_only.pdf").unlink()
    total_pages = render_pdf(abs_text, FINAL_PDF)

    a = analyze(text)
    a.update({"halaman_isi_tanpa_lampiran": body_pages, "halaman_total": total_pages})
    (DOCS / "_report_checks.json").write_text(json.dumps(a, indent=2, ensure_ascii=False))
    print(f"Laporan: {FINAL_MD.relative_to(ROOT)} + {FINAL_PDF.relative_to(ROOT)}")
    print(f"  halaman isi (tanpa lampiran AI): {body_pages} | total: {total_pages}")
    for k, v in a.items():
        print(f"  {k}: {v if v not in ([], {}, 0) else 'tidak ada'}")
    print("  Catatan: cek angka hanya mencocokkan string ke log (heuristik); kecocokan BUKAN bukti interpretasi benar.")


def facts():
    """docs/FAKTA_HASIL.md: angka & perbandingan deskriptif dari log untuk menulis C2–C4 (tanpa interpretasi)."""
    import csv
    S = json.loads(need(RES / "summary.json").read_text())
    H = {c: list(csv.DictReader(open(RES / f"history_{c}.csv"))) for c in S}
    D = read_csv(RES / "paired_bootstrap_f1_diff.csv")
    best = json.loads(need(RES / "best_model.json").read_text())["best_code"]
    cases = read_csv(RES / f"failure_cases_{best}.csv")
    cls = {r["class_name"]: r["species"] for r in read_csv(RES / "selected_classes.csv")}
    cka = json.loads((RES / "c5_cka.json").read_text()) if (RES / "c5_cka.json").exists() else None
    f = lambda x: f"{x:.4f}"
    L = ["# Lembar fakta hasil eksperimen (dibangkitkan otomatis dari outputs/results)",
         "", "> Isi berkas ini hanya fakta dan perbandingan deskriptif dari log. **Interpretasi (mengapa) ditulis sendiri di laporan.**", "",
         "## Ringkasan per kondisi", "",
         "| Kode | Best/jalan | Val loss min | Val acc | Train acc (clean) | Gap acc | Gap loss (val−train) | Test acc | Test F1 | s/epoch | Durasi (s) |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for c, v in S.items():
        L.append(f"| {c} | {v['best_epoch']}/{v['epochs_run']} | {f(v['best_val_loss'])} | {f(v['val_acc'])} | {f(v['train_clean_acc_at_best'])} | "
                 f"{v['gap_acc_clean_train_minus_val']:+.4f} | {v['gap_loss_val_minus_clean_train']:+.4f} | {f(v['test_acc'])} | "
                 f"{f(v['test_macro_f1'])} | {v['sec_per_epoch_train']:.2f} | {v['duration_s']:.1f} |")
    if "B4_noaug" in S and "B4_aug" in S:
        L += ["", "**Catatan durasi (fakta terukur):** B4_noaug dan B4_aug memakai model, optimizer, dan data identik; "
              f"satu-satunya beda adalah pipeline augmentasi train. s/epoch: B4_noaug = {S['B4_noaug']['sec_per_epoch_train']:.2f}, "
              f"B4_aug = {S['B4_aug']['sec_per_epoch_train']:.2f}. Waktu per epoch karenanya mencakup biaya pipeline data di CPU, "
              "bukan hanya komputasi GPU. Bandingkan durasi antar-kondisi yang memakai pipeline transform yang sama."]
    L += ["", "## Perbandingan berpasangan (Macro F1 test, bootstrap 95%)", ""]
    for r in D:
        lo, hi = float(r["CI95 bawah"]), float(r["CI95 atas"])
        L.append(f"- {r['A']} − {r['B']} = {float(r['F1(A)-F1(B) titik']):+.4f}, CI95 [{lo:+.4f}, {hi:+.4f}] → "
                 + ("CI memuat 0 (tidak dapat dibedakan dari noise ukuran test set)" if lo <= 0 <= hi else "CI tidak memuat 0"))
    L += ["", "## Pola kurva (dari history_*.csv)", ""]
    for c, h in H.items():
        vl = [float(r["val_loss"]) for r in h]
        L.append(f"- {c}: val_loss epoch 1 = {vl[0]:.4f}, minimum = {min(vl):.4f} (epoch {vl.index(min(vl)) + 1}), "
                 f"terakhir = {vl[-1]:.4f}; train_loss epoch 1 = {float(h[0]['train_loss']):.4f}, terakhir = {float(h[-1]['train_loss']):.4f}")
    L += ["", f"## Failure cases model terbaik ({best}, dipilih via validation)", ""]
    same = sum(cls[r["true_name"]] == cls[r["pred_name"]] for r in cases if r["jenis"] == "misklasifikasi")
    nmis = sum(r["jenis"] == "misklasifikasi" for r in cases)
    L.append(f"- {nmis} misklasifikasi ditampilkan; {same} di antaranya terjadi di dalam spesies yang sama (kucing→kucing / anjing→anjing).")
    for r in cases:
        L.append(f"- {r['image_id']}: {r['true_name']} → {r['pred_name']} (conf {float(r['confidence']):.3f}; p(GT) {float(r['prob_true']):.3f}; top-3: {r['top3']})")
    if cka:
        L += ["", f"## C5: linear CKA fitur pooled layer4, B2 vs B3 (test) = {cka['linear_cka_layer4_B2_vs_B3_test']:.4f}"]
    L += ["", "## Pertanyaan pemandu (jawab sendiri, kaitkan ke slide)", "",
          "- C2.1 (Slide 10–12, 14–16): apa yang membedakan representasi awal B1 vs B2/B3, dan apakah B1 dibatasi data, epoch, atau keduanya?",
          "- C2.2 (Slide 16, 28): selisih B3−B2 signifikan atau tidak? Bandingkan biaya (s/epoch, trainable params) dan kecepatan konvergensi.",
          "- C2.3 (Slide 20–22): augmentasi mengubah gap acc dan gap loss ke arah mana? Apakah itu sejalan dengan perubahan val/test acc?",
          "- C3: pasangan ras mana yang tertukar, dan ciri visual apa yang terlihat pada citranya (pose, warna, pencahayaan, crop)?",
          "- C4: dengan fakta di atas, strategi apa yang direkomendasikan, dan dalam batasan apa (data, domain, biaya, single seed)?"]
    (DOCS / "FAKTA_HASIL.md").write_text("\n".join(L) + "\n")
    print(f"Ditulis: {(DOCS / 'FAKTA_HASIL.md').relative_to(ROOT)}")


def check_notebook():
    nb = json.loads(need(NB).read_text())
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    out_text = json.dumps([c.get("outputs", []) for c in code])          # hanya OUTPUT, bukan kode sumber
    cfg = json.loads(need(RES / "config.json").read_text())
    return {
        "Notebook: semua sel kode dieksekusi": all(c.get("execution_count") for c in code),
        "Notebook: output sel B1–B4 tersimpan": all(c.get("outputs") for c in code if "_ = run_experiment(" in "".join(c["source"])),
        "Notebook: tanpa error": not any(o.get("output_type") == "error" for c in code for o in c.get("outputs", [])),
        "Notebook & log: bukan smoke test": "SMOKE TEST MODE" not in out_text and not cfg["env"].get("smoke_test"),
    }


def make_zip():
    checks = check_notebook()
    for c in MANDATORY:
        checks[f"Log {c} (summary + history)"] = (RES / f"summary_{c}.json").exists() and (RES / f"history_{c}.csv").exists()
    checks["Log konsolidasi summary.csv / summary.json"] = (RES / "summary.csv").exists() and (RES / "summary.json").exists()
    qc = read_csv(RES / "quality_checks.csv") if (RES / "quality_checks.csv").exists() else []
    checks["quality_checks.csv tanpa FAIL"] = bool(qc) and not any(r["status"] == "FAIL" for r in qc)
    checks["Laporan PDF & MD final ada"] = FINAL_PDF.exists() and FINAL_MD.exists()
    if FINAL_MD.exists():
        text = FINAL_MD.read_text()
        a = analyze(text)
        rc = json.loads((DOCS / "_report_checks.json").read_text()) if (DOCS / "_report_checks.json").exists() else {}
        checks["Laporan: tanpa token/placeholder tersisa"] = not a["token_tersisa"] and not a["placeholder_tersisa"]
        checks["Laporan: semua gambar yang dirujuk ada"] = not a["gambar_hilang"]
        for k, n in a["kata_bagian_A"].items():
            checks[f"Bagian {k}: 100–150 kata (sekarang {n})"] = n is not None and 100 <= n <= 150
        checks["Bagian A: kelima nomor ada"] = len(a["kata_bagian_A"]) == 5
        checks["AI Disclosure: tidak ada checkbox kosong (centang atau hapus)"] = a["checkbox_belum_dicentang"] == 0
        bp = rc.get("halaman_isi_tanpa_lampiran")
        checks[f"Laporan isi 3–5 halaman (sekarang {bp}; total {rc.get('halaman_total')})"] = bp is not None and 3 <= bp <= 5
        if a["angka_tidak_ditemukan_di_log"]:
            print(f"  PERINGATAN (cek manual): angka tidak ditemukan di log: {a['angka_tidak_ditemukan_di_log']}")
    link = (ROOT / "GITHUB_LINK.txt")
    checks["GITHUB_LINK.txt berisi tautan repo"] = link.exists() and "github.com/" in link.read_text()
    checks["README.md & requirements.txt"] = (ROOT / "README.md").exists() and (ROOT / "requirements.txt").exists()

    for k, v in checks.items():
        print(f"  [{'OK' if v else '!!'}] {k}")
    if not all(checks.values()):
        sys.exit("\nZIP TIDAK dibuat: perbaiki butir [!!] di atas.")

    files = [NB, FINAL_PDF, FINAL_MD, ROOT / "README.md", ROOT / "requirements.txt", link]
    files += sorted(p for p in RES.iterdir() if p.is_file()) + sorted(p for p in FIG.iterdir() if p.is_file())
    out = ROOT.parent / ZIP_NAME
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, Path(ZIP_NAME[:-4]) / p.relative_to(ROOT))
    print(f"\nZIP dibuat: {out} ({out.stat().st_size / 1e6:.1f} MB, {len(files)} berkas)")


if __name__ == "__main__":
    {"report": report, "zip": make_zip, "facts": facts}[sys.argv[1] if len(sys.argv) > 1 else "report"]()
