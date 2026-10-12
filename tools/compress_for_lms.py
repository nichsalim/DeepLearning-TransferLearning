"""Buat ZIP pengumpulan <= batas ukuran LMS dengan mengompres GAMBAR saja.

  python tools/compress_for_lms.py [--max-mb 10]

- Gambar foto (banyak warna) -> JPEG q85; grafik/kurva -> PNG palet 256 warna (teks tetap tajam).
- Berlaku untuk outputs/figures/*.png dan gambar yang tertanam di output notebook.
- Teks output notebook, log CSV/JSON, dan angka laporan TIDAK diubah (diverifikasi otomatis).
- Repo GitHub tetap menyimpan versi resolusi asli.
"""
import base64, copy, hashlib, io, json, shutil, subprocess, sys, tempfile
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
MAX_MB = float(sys.argv[sys.argv.index("--max-mb") + 1]) if "--max-mb" in sys.argv else 10.0


def is_photo(im):
    small = im.convert("RGB").resize((min(400, im.width), min(400, im.height)))
    return len(set(small.get_flattened_data() if hasattr(small, 'get_flattened_data') else small.getdata())) > 20000


def compress(raw: bytes, allow_jpeg=True):
    """Kembalikan (bytes, mime). Hanya dipakai bila hasilnya lebih kecil.
    allow_jpeg=False untuk berkas .png di figures/ (format harus sesuai ekstensi): foto -> PNG palet + dithering."""
    im = Image.open(io.BytesIO(raw))
    im = im.convert("RGB") if im.mode in ("RGBA", "LA", "P") else im
    buf = io.BytesIO()
    if is_photo(im) and not allow_jpeg:
        # palet adaptif (median cut) + dithering Floyd–Steinberg: menghindari posterization pada foto
        im.convert("RGB").convert("P", palette=Image.Palette.ADAPTIVE, colors=256, dither=Image.Dither.FLOYDSTEINBERG).save(buf, "PNG", optimize=True)
        mime = "image/png"
    elif is_photo(im):
        im.save(buf, "JPEG", quality=85, optimize=True, progressive=True)
        mime = "image/jpeg"
    else:
        im.quantize(colors=256, method=Image.Quantize.FASTOCTREE, dither=Image.Dither.NONE).save(buf, "PNG", optimize=True)
        mime = "image/png"
    out = buf.getvalue()
    return (out, mime) if len(out) < len(raw) else (raw, "image/png")


def main():
    stage = Path(tempfile.mkdtemp()) / "repo"
    shutil.copytree(ROOT, stage, ignore=shutil.ignore_patterns("checkpoints", "__pycache__", "_*"))
    fig = stage / "outputs" / "figures"

    # 1) figures/: simpan dengan nama sama (.png) agar rujukan laporan tidak berubah; isi bisa JPEG atau PNG palet
    before = sum(p.stat().st_size for p in fig.glob("*.png"))
    for p in fig.glob("*.png"):
        data, _ = compress(p.read_bytes(), allow_jpeg=False)
        p.write_bytes(data)
    after = sum(p.stat().st_size for p in fig.glob("*.png"))
    print(f"figures/: {before/1e6:.2f} MB -> {after/1e6:.2f} MB")

    # 2) notebook: ganti hanya data gambar di output; semua output non-gambar harus identik
    nbp = stage / "Tugas05_TransferLearning.ipynb"
    nb = json.loads(nbp.read_text())
    orig = copy.deepcopy(nb)
    for c in nb["cells"]:
        for o in c.get("outputs", []):
            d = o.get("data", {})
            if "image/png" in d:
                data, mime = compress(base64.b64decode(d["image/png"]))
                del d["image/png"]
                d[mime] = base64.b64encode(data).decode()
    strip = lambda n: [[{k: v for k, v in o.items() if k != "data"} | {"data": {m: x for m, x in o.get("data", {}).items()
                         if not m.startswith("image/")}} for o in c.get("outputs", [])] for c in n["cells"]]
    assert strip(nb) == strip(orig), "Output non-gambar berubah!"
    assert [c["source"] for c in nb["cells"]] == [c["source"] for c in orig["cells"]]
    nbp.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
    print(f"notebook: {(ROOT / nbp.name).stat().st_size/1e6:.2f} MB -> {nbp.stat().st_size/1e6:.2f} MB (teks output identik)")

    # 3) log mentah harus byte-identik
    for p in (ROOT / "outputs" / "results").iterdir():
        assert hashlib.md5(p.read_bytes()).digest() == hashlib.md5((stage / "outputs/results" / p.name).read_bytes()).digest()
    print("outputs/results/: byte-identik dengan aslinya")

    # 4) render ulang PDF dari gambar terkompres, lalu bangun & validasi ZIP (aturan sama dengan finalize.py zip)
    run = lambda *a: subprocess.run([sys.executable, str(stage / "tools/finalize.py"), *a], check=True, capture_output=True, text=True)
    run("report")
    t_orig = subprocess.run(["pdftotext", str(ROOT / "docs/LAPORAN_Tugas05_final.pdf"), "-"], capture_output=True, text=True).stdout
    t_new = subprocess.run(["pdftotext", str(stage / "docs/LAPORAN_Tugas05_final.pdf"), "-"], capture_output=True, text=True).stdout
    assert t_orig == t_new, "Teks PDF berubah!"
    print("PDF: teks identik dengan versi asli")
    out = run("zip").stdout
    print("\n".join(l for l in out.splitlines() if "[!!]" in l or "ZIP dibuat" in l))
    z = stage.parent / "Tugas05_2802420776_NicholasSalim.zip"
    mb = z.stat().st_size / 1e6
    dest = ROOT.parent / "LMS" / z.name
    dest.parent.mkdir(exist_ok=True)
    shutil.copy(z, dest)
    print(f"ZIP LMS: {dest} ({mb:.2f} MB; batas {MAX_MB} MB) -> {'OK' if mb <= MAX_MB else 'MASIH TERLALU BESAR'}")


if __name__ == "__main__":
    main()
