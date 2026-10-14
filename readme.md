# Format Compare Visual

A modular Python & Tkinter application that compares how different image codecs (**JPEG**, **WebP Lossy**, **WebP Lossless**, **PNG**, **PNG-8 Palette**, **JPEG 2000**, **GIF**, and **TIFF**) visually render an input image when constrained to an exact **target file size budget**.

## Project Architecture

| File | Responsibility |
| :--- | :--- |
| `main.py` | Entry point and Tkinter GUI (`FormatCompareApp`) with a top file size slider (`0` to input image file size), direct byte/KB entry (`e.g. 2.5 KB`), responsive comparison grid, side-by-side zoom inspector modal, and headless `--benchmark` CLI mode. |
| `formats.py` | Standardized, extensible codec definitions (`ParameterSpec`, `FormatSpec`, `FORMAT_REGISTRY`, `register_format`) specifying fixed options (such as `optimize=True`) and tunable parameters (`scale` in `(0, 1]`, `quality`, `subsampling`, `colors`, `dither`, `target_psnr_db`). |
| `optimizer.py` | Parallel optimization engine (`optimize_format`, `ParallelFormatOptimizer`) implementing **1D Binary Search** for single-parameter formats and **Hybrid Coordinate-Sliced Binary Search + Iso-Byte Pareto Frontier + Local Coordinate Polish** for multi-parameter formats. |

## Built-In Formats & Tunable Parameters

| Format ID | Display Name | Fixed Options | Tunable Parameters | Optimizer Strategy |
| :--- | :--- | :--- | :--- | :--- |
| `jpeg` | **JPEG** | `optimize=True` | `scale ∈ (0, 1]`, `quality ∈ [1, 95]`, `subsampling ∈ {'4:2:0', '4:2:2', '4:4:4'}` | Multi-Param Frontier + Coordinate Polish |
| `webp_lossy` | **WebP (Lossy)** | `lossless=False, method=4` | `scale ∈ (0, 1]`, `quality ∈ [1, 100]` | Multi-Param Frontier + Coordinate Polish |
| `webp_lossless` | **WebP (Lossless)** | `lossless=True, quality=100, method=4` | `scale ∈ (0, 1]` | 1D Binary Search |
| `png` | **PNG (Lossless)** | `optimize=True` | `scale ∈ (0, 1]` | 1D Binary Search |
| `png_quantized` | **PNG-8 (Palette)** | `optimize=True` | `scale ∈ (0, 1]`, `colors ∈ [2, 256]`, `dither ∈ {0, 1}` | Multi-Param Frontier + Coordinate Polish |
| `jpeg2000` | **JPEG 2000** | `quality_mode='dB', irreversible=True` | `scale ∈ (0, 1]`, `target_psnr_db ∈ [18.0, 58.0]` | Multi-Param Frontier + Coordinate Polish |
| `gif` | **GIF** | `optimize=True` | `scale ∈ (0, 1]`, `colors ∈ [2, 256]` | Multi-Param Frontier + Coordinate Polish |
| `tiff` | **TIFF (Deflate)** | `compression='tiff_adobe_deflate'` | `scale ∈ (0, 1]` | 1D Binary Search |

## Plugging In a New Format

New formats can be added in `formats.py` (or dynamically at runtime) with `register_format`:

```python
from formats import DEFAULT_SCALE_PARAMETER, FormatSpec, ParameterSpec, register_format

register_format(
    FormatSpec(
        id="bmp",
        name="BMP (Uncompressed)",
        pil_format="BMP",
        extension=".bmp",
        supports_alpha=False,
        parameters=(DEFAULT_SCALE_PARAMETER,),
        description="Uncompressed bitmap; tunes resolution scale via 1D binary search.",
    )
)
```

## Running the Project

### 1. Interactive Tkinter GUI

```bash
# Launch with a specific input image:
python3 main.py path/to/input_image.png

# Or launch directly with the built-in high-detail test image:
python3 main.py
```

### 2. Headless CLI Benchmark Mode

```bash
# Optimize all formats in parallel for a 2.5 KB (2,560 bytes) budget:
python3 main.py --benchmark --budget 2560
```
