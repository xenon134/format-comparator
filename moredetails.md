main.py: Launches the Tkinter application (FormatCompareApp) with a top file size slider ranging from 0 to the input image's byte size, a direct byte/KB input box (e.g., 2.5 KB or 15%), quick preset buttons, a responsive multi-column codec comparison grid, a side-by-side Original vs. Compressed zoom inspector modal (with file export), and a headless --benchmark CLI mode.

formats.py: Defines a standardized, plug-and-play codec registry (ParameterSpec, FormatSpec, FORMAT_REGISTRY, and register_format()) with built-in definitions for JPEG, WebP (Lossy), WebP (Lossless), PNG (Lossless), PNG-8 (Palette), JPEG 2000, JPEG XL, GIF, and TIFF (Deflate).

optimizer.py: Implements the parallel optimization engine (optimize_format() and ParallelFormatOptimizer) supporting both ThreadPoolExecutor and ProcessPoolExecutor with cooperative cancellation (threading.Event) when the slider moves.

README.md: Documents the module architecture, codec parameter specifications, plugin registration API, and CLI commands.

Codec Specifications & Optimization Strategies

Format,Fixed Options,Tunable Parameters,Optimizer Strategy
JPEG,optimize=True,"scale ∈ (0, 1], quality ∈ [1, 95], subsampling ∈ {'4:2:0', '4:2:2', '4:4:4'}",Multi-Param Frontier + Coordinate Polish
WebP (Lossy),"lossless=False, method=2","scale ∈ (0, 1], quality ∈ [1, 100]",Multi-Param Frontier + Coordinate Polish
WebP (Lossless),"lossless=True, quality=80, method=1","scale ∈ (0, 1]",1D Binary Search
PNG (Lossless),optimize=True,"scale ∈ (0, 1]",1D Binary Search
PNG-8 (Palette),optimize=True,"scale ∈ (0, 1], colors ∈ [2, 256], dither ∈ {0, 1}",Multi-Param Frontier + Coordinate Polish
JPEG 2000,"quality_mode='dB', irreversible=True","scale ∈ (0, 1], target_psnr_db ∈ [18.0, 58.0]",Multi-Param Frontier + Coordinate Polish
JPEG XL,effort=3,"scale ∈ (0, 1], quality ∈ [1, 100]",Multi-Param Frontier + Coordinate Polish
GIF,optimize=True,"scale ∈ (0, 1], colors ∈ [2, 256]",Multi-Param Frontier + Coordinate Polish
TIFF (Deflate),compression='tiff_adobe_deflate',"scale ∈ (0, 1]",1D Binary Search

Key Algorithmic Highlights
Single-Parameter 1D Binary Search: Formats with only scale as a tunable parameter (PNG, WebP Lossless, TIFF) use a warm-started secant + bisection search over scale ∈ (0, 1] with pixel-dimension (width, height) memoization and a ±1–2 px local neighborhood polish to handle minor DEFLATE/VP8L predictor non-monotonicity.

Multi-Parameter Frontier & Coordinate Polish: Formats with multiple parameters (JPEG, WebP Lossy, PNG-8, GIF, JPEG 2000, JPEG XL) evaluate full-scale feasibility first, then slice the iso-byte Pareto frontier across anchor levels using monotonic upper-bound propagation on scale, cross-evaluate categorical choices (subsampling, dither), and finish with coordinate descent steps (+1 px scale, +1 quality/color step) and lazy PSNR tie-breaking.

Sample 2.5 KB (2,560 B) Benchmark: On the built-in 384×256 test image (53,466 B raw), WebP (Lossy) retains 212×141 resolution (scale=0.551, quality=66, 2,544 B, 24.86 dB PSNR) and JPEG retains 330×220 (scale=0.859, quality=12, subsampling=4:2:0, 2,560 B, 24.42 dB PSNR), whereas lossless PNG must downscale to 45×30 (scale=0.117, 2,536 B, 19.73 dB PSNR) to fit the same budget.

Run python3 main.py path/to/image.png to open the interactive Tkinter comparator on any image, or run python3 main.py --benchmark --budget 2560 to print a parallel benchmark table in your terminal.
