# Image IO Overview

This document explains the **image I/O layer** for your Martial Arts OCR stack. It covers what each module does, the key APIs, conventions, common recipes, and edge-case behavior.

---

## Goals

* **Predictable & safe** loading/saving across cameras/scanners (EXIF, DPI, ICC aware).
* **Explicit color spaces** (BGR/RGB/GRAY) to avoid silent bugs.
* **Validation first**: catch corrupt/unsupported inputs early.
* **Metadata fast-path**: read headers without decoding full pixels.
* **Bytes or files**: same APIs work for disk, HTTP/S3 buffers, etc.

---

## Module Map

```
utils/image/io/
  read.py      # load from path/bytes with EXIF-aware orientation (Pillow) + cv2 fallback
  write.py     # save/encode with format-aware params, atomic writes, color-space handling
  validate.py  # extensions, decodability, min-size; bytes & file paths; structured reports
  meta.py      # fast metadata (format, size, DPI, EXIF/ICC) and ndarray probing
```

> High-level convenience imports are re-exported via `utils/image/api.py`.

---

## Conventions

* **OpenCV default** is **BGR**. Readers return BGR by default; writers assume BGR input by default.
* Color mode parameters: `mode` / `input_mode` ∈ `{"bgr","rgb","gray"}`.
* Grayscale returns a **2D array**; color returns **HxWx3** (alpha dropped).
* All APIs accept `str | Path`; byte APIs accept `bytes`.

---

## `read.py` — Loading

**Responsibilities**

* Load from **filesystem** or **bytes**.
* Apply **EXIF orientation** (Pillow `ImageOps.exif_transpose`).
* Return in requested **color mode** (BGR/RGB/GRAY).
* Fall back to `cv2` if PIL fails.

**Key APIs**

* `load_image(path, mode="bgr", use_exif=True) -> np.ndarray`
* `imread_bytes(data, mode="bgr", use_exif=True) -> np.ndarray`

**Notes**

* If EXIF step fails, logs a warning and falls back to `cv2.imread`.
* Drops alpha channel to keep 3-channel output (consistent downstream).

---

## `write.py` — Saving & Encoding

**Responsibilities**

* Save to path with **format-aware parameters**.
* **Atomic writes** by default to avoid partial files.
* In-memory **encoding** for HTTP/S3.

**Key APIs**

* `save_image(image, output_path, *, quality=95, input_mode="bgr", progressive_jpeg=True, png_compression=None, tiff_lzw=True, atomic=True) -> bool`
* `imencode(ext, image, *, input_mode="bgr", quality=95, ...) -> (ok, buf)`

**Format knobs**

* **JPEG**: `IMWRITE_JPEG_QUALITY`, progressive/optimize when available.
* **PNG**: compression `0..9` (auto-mapped from `quality` if not provided).
* **TIFF**: LZW attempt (`IMWRITE_TIFF_COMPRESSION=1`) when available.
* **WEBP**: quality `1..100`.

**Color space**

* `input_mode` converts RGB→BGR as needed; 2D arrays written unchanged.

---

## `validate.py` — Validation

**Responsibilities**

* Check **existence, type, extension**.
* Verify **decodability** with `Pillow.verify()`.
* Enforce **minimum dimensions** (default 10×10).
* **Bytes** and **files** supported.

**Key APIs**

* `validate_image_file(path, *, allowed_extensions=None, require_extension_match=True, min_size=(10,10)) -> ValidationReport`
* `validate_image_bytes(data, *, min_size=(10,10)) -> ValidationReport`
* `is_supported_extension(ext, allowed_extensions=None) -> bool`
* `sniff_image_format(source: path|bytes) -> Optional[str]`

**ValidationReport**

```
is_valid: bool
reason: str
width, height: Optional[int]
format, mode: Optional[str]
size_bytes: Optional[int]
dpi: Optional[int]
```

---

## `meta.py` — Metadata

**Responsibilities**

* Fast **header-level** metadata via Pillow (no full decode).
* Optional full **decode** to capture `dtype` and `channels`.
* **EXIF** subset (orientation, datetime), **DPI**, **ICC** description.
* Convert to your existing `ImageInfo`.

**Key APIs**

* `get_image_meta(path, *, decode_for_dtype=False, decode_mode="bgr") -> ImageMeta`
* `probe_array(arr) -> ImageMeta`
* `export_image_info(meta) -> ImageInfo`

**ImageMeta fields (selected)**

```
width, height, channels, aspect_ratio, megapixels
path, file_size
dtype, pil_mode, color_space, has_alpha
fmt, dpi, exif_orientation, exif_datetime, icc_profile_desc
exif (small dict)
```

---

## Typical Recipes

### 1) Validate → Load (EXIF-aware)

```python
from utils.image import api as img

arr = img.ensure_valid_then_load("page.jpg", mode="bgr")
```

### 2) Save with atomic write

```python
ok = img.save_image(arr, "out/page.jpg", quality=90, input_mode="bgr", atomic=True)
```

### 3) Bytes in/out (e.g., S3/HTTP)

```python
from utils.image import api as img

arr = img.imread_bytes(blob, mode="rgb")
ok, buf = img.imencode(".webp", arr, input_mode="rgb", quality=92)
if ok: send(buf.tobytes())
```

### 4) Quick metadata (no full decode)

```python
from utils.image import api as img
m = img.get_image_meta("scan.tif")
print(m.fmt, m.width, m.height, m.dpi, m.exif_orientation)
```

### 5) Thumbnail pipeline (file → file)

```python
from utils.image import api as img
img.quick_thumb_to_file("scan.jpg", "thumbs/scan_320.jpg", (320, 320))
```

---

## Error Handling & Logging

* All modules log **debug** details (sizes, modes) and **warnings** on fallbacks (e.g., EXIF load failure).
* `read.py`/`meta.py` **raise** on missing files or irrecoverable decode errors.
* `save_image` returns **False** on failure and logs the reason; atomic temp files are cleaned up on best effort.

---

## Performance Notes

* Prefer `meta.get_image_meta(..., decode_for_dtype=False)` to query size/format/DPI cheaply.
* For large batch loads, leave `use_exif=True` unless you know images lack EXIF (saves re-rotations later).
* PNG high compression is slow; set `png_compression=3..6` for balanced throughput.

---

## Interop with `ops` & Pipelines

* **Resizing/Thumbs**: use `utils.image.ops.resize` and `ops.thumb` after `read.load_image`.
* **Tone/Contrast**: `ops.tone` (CLAHE, gamma, unsharp) before OCR.
* **OCR Prep**: `pipelines/ocr_prep.py` provides `jp_text_prep`, `en_text_prep`, and `adaptive_binarize`.

---

## Quick CLI (optional, if you included `utils/image/cli.py`)

```
python -m utils.image.cli validate input.jpg
python -m utils.image.cli info input.jpg
python -m utils.image.cli thumb input.jpg out.jpg --size 320x320 --pad
python -m utils.image.cli resize input.jpg out.jpg --long 1664
```

---

## Edge Cases & Gotchas

* **Alpha channels**: writers drop alpha to standardize 3 channels; keep a copy earlier if you need it.
* **Unsupported TIFF compression** in some OpenCV builds: `tiff_lzw=True` is best-effort.
* **Tiny images** (<10×10): rejected by default; adjust `min_size` if you truly need icons.
* **Weird modes** (e.g., `P`, `CMYK`): `read.py` normalizes via PIL to RGB before converting to BGR.

---

## Minimal End-to-End Example

```python
from utils.image import api as img
from utils.image.pipelines.ocr_prep import ocr_prep

# 1) Validate & load
arr = img.ensure_valid_then_load("samples/jp_page.jpg", mode="bgr")

# 2) Prep for OCR (Japanese)
g = ocr_prep(arr, lang="jp", target_long_edge=1664, binarize=True, bin_method="otsu")

# 3) Save result (grayscale PNG)
img.save_image(g, "out/jp_page_ocr.png", input_mode="gray")
```
