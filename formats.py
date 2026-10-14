"""
formats.py — Extensible Image Format & Codec Parameter Definitions
==================================================================

Defines a standard, plug-and-play structure (`ParameterSpec` and `FormatSpec`)
for image codecs supported by PIL/Pillow. Each format specifies:
  - Fixed codec options that are always optimal (e.g., `optimize=True` for PNG/JPEG)
  - Tunable parameters (`scale` in (0, 1], plus format-specific knobs such as
    `quality`, `subsampling`, `colors`, `dither`, `target_psnr_db`)
  - Optional preprocessing hook to map abstract parameters into PIL save kwargs
    or image transformations (such as palette quantization or RGB conversion).

Adding a new format requires only calling `register_format(FormatSpec(...))`.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple
from PIL import Image


ParameterKind = Literal["float", "int", "choice"]


@dataclass(frozen=True)
class ParameterSpec:
    """
    Standard specification for a single tunable codec or scaling parameter.

    Attributes:
        name: Parameter identifier (e.g. 'scale', 'quality', 'subsampling').
        kind: Parameter domain type: 'float', 'int', or 'choice'.
        min_val: Minimum value for 'float' or 'int' parameters.
        max_val: Maximum value for 'float' or 'int' parameters.
        choices: Ordered sequence of allowed values for 'choice' parameters,
                 ordered from smallest expected file size / lowest fidelity
                 to largest file size / highest fidelity.
        default: Default/maximum-fidelity value for this parameter.
        monotonic_with_size: True if increasing this parameter (or advancing
                             along `choices`) generally increases file size.
        description: Human-readable explanation of the parameter.
    """

    name: str
    kind: ParameterKind
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    choices: Optional[Tuple[Any, ...]] = None
    default: Any = None
    monotonic_with_size: bool = True
    description: str = ""

    def __post_init__(self) -> None:
        if self.kind in ("float", "int"):
            if self.min_val is None or self.max_val is None:
                raise ValueError(
                    f"Parameter '{self.name}' of kind '{self.kind}' requires min_val and max_val."
                )
            if self.min_val > self.max_val:
                raise ValueError(
                    f"Parameter '{self.name}' has min_val ({self.min_val}) > max_val ({self.max_val})."
                )
        elif self.kind == "choice":
            if not self.choices or len(self.choices) == 0:
                raise ValueError(
                    f"Parameter '{self.name}' of kind 'choice' requires a non-empty choices tuple."
                )
        else:
            raise ValueError(f"Unsupported parameter kind: {self.kind}")

    def clamp_or_normalize(self, value: Any) -> Any:
        """Clamp numeric values to [min_val, max_val] and cast to the proper type."""
        if self.kind == "float":
            v = float(value)
            return max(float(self.min_val), min(float(self.max_val), v))
        if self.kind == "int":
            v = int(round(float(value)))
            return max(int(self.min_val), min(int(self.max_val), v))
        if self.kind == "choice":
            assert self.choices is not None
            if value in self.choices:
                return value
            if isinstance(value, (int, float)):
                idx = max(0, min(len(self.choices) - 1, int(round(value))))
                return self.choices[idx]
            return self.choices[-1]
        return value

    def to_normalized(self, value: Any) -> float:
        """Map a parameter value into normalized [0.0, 1.0] coordinate space."""
        if self.kind in ("float", "int"):
            span = float(self.max_val) - float(self.min_val)
            if span <= 0:
                return 1.0
            return max(0.0, min(1.0, (float(value) - float(self.min_val)) / span))
        if self.kind == "choice":
            assert self.choices is not None
            if len(self.choices) <= 1:
                return 1.0
            idx = self.choices.index(value) if value in self.choices else len(self.choices) - 1
            return idx / float(len(self.choices) - 1)
        return 1.0

    def from_normalized(self, u: float) -> Any:
        """Map a normalized coordinate u in [0.0, 1.0] back to parameter domain."""
        u_clamped = max(0.0, min(1.0, float(u)))
        if self.kind == "float":
            val = float(self.min_val) + u_clamped * (float(self.max_val) - float(self.min_val))
            return self.clamp_or_normalize(val)
        if self.kind == "int":
            val = float(self.min_val) + u_clamped * (float(self.max_val) - float(self.min_val))
            return self.clamp_or_normalize(int(round(val)))
        if self.kind == "choice":
            assert self.choices is not None
            idx = int(round(u_clamped * (len(self.choices) - 1)))
            idx = max(0, min(len(self.choices) - 1, idx))
            return self.choices[idx]
        return self.default


# Standard resolution scale parameter shared across all formats: scale in (0, 1]
DEFAULT_SCALE_PARAMETER = ParameterSpec(
    name="scale",
    kind="float",
    min_val=0.01,
    max_val=1.0,
    default=1.0,
    monotonic_with_size=True,
    description=(
        "Resolution scaling factor in (0, 1]. Scales image dimensions to "
        "(round(width * scale), round(height * scale)) before encoding."
    ),
)


PreprocessHook = Callable[
    [Image.Image, Dict[str, Any]], Tuple[Image.Image, Dict[str, Any]]
]


@dataclass(frozen=True)
class FormatSpec:
    """
    Complete specification for an image format/codec and its tunable parameters.

    Attributes:
        id: Unique machine identifier (e.g. 'jpeg', 'png', 'webp_lossy').
        name: Display name in the UI grid (e.g. 'JPEG', 'PNG (Lossless)').
        pil_format: Format string passed to `PIL.Image.save(..., format=pil_format)`.
        extension: Standard file extension (e.g. '.jpg', '.png', '.webp').
        parameters: Tuple of tunable `ParameterSpec` definitions (always includes `scale`).
        fixed_options: Dict of non-tunable PIL save options that are fixed to their
                       optimal values (e.g., `{'optimize': True}`).
        supports_alpha: Whether the codec natively preserves an alpha channel.
        preprocess_hook: Optional hook `(img, params) -> (prepared_img, pil_save_kwargs)`
                         for formats needing custom parameter translation or color quantization.
        description: Short summary of the codec's characteristics.
    """

    id: str
    name: str
    pil_format: str
    extension: str
    parameters: Tuple[ParameterSpec, ...] = field(
        default_factory=lambda: (DEFAULT_SCALE_PARAMETER,)
    )
    fixed_options: Dict[str, Any] = field(default_factory=dict)
    supports_alpha: bool = True
    preprocess_hook: Optional[PreprocessHook] = None
    description: str = ""

    @property
    def is_single_parameter(self) -> bool:
        """Return True if this format has only a single tunable parameter (e.g. `scale`)."""
        return len(self.parameters) == 1

    @property
    def non_scale_parameters(self) -> Tuple[ParameterSpec, ...]:
        """Return all tunable parameters other than `scale`."""
        return tuple(p for p in self.parameters if p.name != "scale")

    def get_parameter(self, name: str) -> Optional[ParameterSpec]:
        """Look up a parameter specification by name."""
        for p in self.parameters:
            if p.name == name:
                return p
        return None

    def max_fidelity_params(self) -> Dict[str, Any]:
        """Return the highest-quality / full-scale parameter dictionary."""
        result: Dict[str, Any] = {}
        for p in self.parameters:
            if p.default is not None:
                result[p.name] = p.default
            elif p.kind in ("float", "int"):
                result[p.name] = p.max_val
            elif p.kind == "choice" and p.choices:
                result[p.name] = p.choices[-1]
        return result

    def min_fidelity_params(self) -> Dict[str, Any]:
        """Return the lowest-size / minimum-fidelity parameter dictionary."""
        result: Dict[str, Any] = {}
        for p in self.parameters:
            if p.kind in ("float", "int"):
                result[p.name] = p.min_val
            elif p.kind == "choice" and p.choices:
                result[p.name] = p.choices[0]
            else:
                result[p.name] = p.default
        return result

    def sanitize_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clamp a dictionary of candidate parameters."""
        clean: Dict[str, Any] = {}
        defaults = self.max_fidelity_params()
        for p in self.parameters:
            raw_val = params.get(p.name, defaults[p.name])
            clean[p.name] = p.clamp_or_normalize(raw_val)
        return clean

    def compute_scaled_dimensions(
        self, width: int, height: int, scale: float
    ) -> Tuple[int, int]:
        """
        Compute target `(width, height)` in pixels for a given `scale` in (0, 1].
        """
        s = max(1e-4, min(1.0, float(scale)))
        new_w = max(1, int(round(width * s)))
        new_h = max(1, int(round(height * s)))
        return new_w, new_h

    def encode(
        self,
        image: Image.Image,
        params: Dict[str, Any],
        resampling: Image.Resampling = Image.Resampling.LANCZOS,
    ) -> Tuple[bytes, Tuple[int, int], Dict[str, Any]]:
        """
        Scale and encode `image` using `params` into an in-memory byte buffer.

        Returns:
            Tuple of `(encoded_bytes, (scaled_w, scaled_h), sanitized_params)`.
        """
        clean_params = self.sanitize_params(params)
        scale = float(clean_params.get("scale", 1.0))
        scaled_w, scaled_h = self.compute_scaled_dimensions(
            image.width, image.height, scale
        )

        if (scaled_w, scaled_h) != (image.width, image.height):
            work_img = image.resize((scaled_w, scaled_h), resample=resampling)
        else:
            work_img = image

        if not self.supports_alpha and work_img.mode in ("RGBA", "LA", "PA", "P"):
            rgba = work_img.convert("RGBA")
            bg = Image.new("RGB", rgba.size, (255, 255, 255))
            bg.paste(rgba, mask=rgba.split()[3])
            work_img = bg
        elif not self.supports_alpha and work_img.mode != "RGB":
            work_img = work_img.convert("RGB")

        codec_params = {k: v for k, v in clean_params.items() if k != "scale"}
        if self.preprocess_hook is not None:
            work_img, save_kwargs = self.preprocess_hook(work_img, codec_params)
        else:
            save_kwargs = dict(codec_params)

        final_save_kwargs = {**self.fixed_options, **save_kwargs}
        buf = io.BytesIO()
        work_img.save(buf, format=self.pil_format, **final_save_kwargs)
        return buf.getvalue(), (scaled_w, scaled_h), clean_params


# ============================================================================
# Preprocessing Hooks for Specific Formats
# ============================================================================


def _preprocess_png_quantized(
    img: Image.Image, codec_params: Dict[str, Any]
) -> Tuple[Image.Image, Dict[str, Any]]:
    """Quantize image to N palette colors using FastOctree before saving as PNG-8."""
    colors = int(codec_params.get("colors", 256))
    dither_flag = int(codec_params.get("dither", 1))
    dither = Image.Dither.FLOYDSTEINBERG if dither_flag else Image.Dither.NONE
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGBA" if "A" in img.mode else "RGB")
    quantized = img.quantize(
        colors=colors, method=Image.Quantize.FASTOCTREE, dither=dither
    )
    return quantized, {}


def _preprocess_gif(
    img: Image.Image, codec_params: Dict[str, Any]
) -> Tuple[Image.Image, Dict[str, Any]]:
    """Quantize image to N palette colors using FastOctree before saving as GIF."""
    colors = int(codec_params.get("colors", 256))
    rgb = img.convert("RGB")
    quantized = rgb.quantize(
        colors=colors,
        method=Image.Quantize.FASTOCTREE,
        dither=Image.Dither.FLOYDSTEINBERG,
    )
    return quantized, {}


def _preprocess_jpeg2000(
    img: Image.Image, codec_params: Dict[str, Any]
) -> Tuple[Image.Image, Dict[str, Any]]:
    """Translate `target_psnr_db` into Pillow's `quality_layers` list for JPEG 2000."""
    target_db = float(codec_params.get("target_psnr_db", 45.0))
    if img.mode not in ("RGB", "RGBA", "L"):
        img = img.convert("RGB")
    return img, {"quality_layers": [target_db]}


# ============================================================================
# Format Registry & Built-In Format Definitions
# ============================================================================

FORMAT_REGISTRY: Dict[str, FormatSpec] = {}


def register_format(spec: FormatSpec) -> FormatSpec:
    """
    Register a `FormatSpec` in the global format registry.
    Can be called by user plugins to add new codecs dynamically.
    """
    FORMAT_REGISTRY[spec.id] = spec
    return spec


def get_registered_formats() -> List[FormatSpec]:
    """Return all registered `FormatSpec` objects in registration order."""
    return list(FORMAT_REGISTRY.values())


def get_format_by_id(format_id: str) -> FormatSpec:
    """Return a registered `FormatSpec` by its ID, or raise KeyError."""
    if format_id not in FORMAT_REGISTRY:
        raise KeyError(
            f"Unknown format ID '{format_id}'. Available: {list(FORMAT_REGISTRY.keys())}"
        )
    return FORMAT_REGISTRY[format_id]


def _register_builtin_formats() -> None:
    """Populate `FORMAT_REGISTRY` with common image formats and codecs."""

    # 1. JPEG — quality in [1, 95], subsampling in {'4:2:0', '4:2:2', '4:4:4'}, scale in (0, 1]
    register_format(
        FormatSpec(
            id="jpeg",
            name="JPEG",
            pil_format="JPEG",
            extension=".jpg",
            fixed_options={"optimize": True},
            supports_alpha=False,
            parameters=(
                DEFAULT_SCALE_PARAMETER,
                ParameterSpec(
                    name="quality",
                    kind="int",
                    min_val=1,
                    max_val=95,
                    default=95,
                    monotonic_with_size=True,
                    description="DCT quantization quality factor [1, 95].",
                ),
                ParameterSpec(
                    name="subsampling",
                    kind="choice",
                    choices=("4:2:0", "4:2:2", "4:4:4"),
                    default="4:4:4",
                    monotonic_with_size=True,
                    description="Chroma subsampling mode ('4:2:0', '4:2:2', '4:4:4').",
                ),
            ),
            description="Lossy DCT codec; tunes scale, quality [1..95], and chroma subsampling.",
        )
    )

    # 2. WebP (Lossy) — quality in [1, 100], scale in (0, 1]
    register_format(
        FormatSpec(
            id="webp_lossy",
            name="WebP (Lossy)",
            pil_format="WEBP",
            extension=".webp",
            fixed_options={"lossless": False, "method": 2},
            supports_alpha=True,
            parameters=(
                DEFAULT_SCALE_PARAMETER,
                ParameterSpec(
                    name="quality",
                    kind="int",
                    min_val=1,
                    max_val=100,
                    default=100,
                    monotonic_with_size=True,
                    description="VP8 lossy compression quality factor [1, 100].",
                ),
            ),
            description="Modern VP8 lossy codec; tunes scale and quality [1..100].",
        )
    )

    # 3. WebP (Lossless) — single tunable parameter: scale in (0, 1]
    register_format(
        FormatSpec(
            id="webp_lossless",
            name="WebP (Lossless)",
            pil_format="WEBP",
            extension=".webp",
            fixed_options={"lossless": True, "quality": 80, "method": 1},
            supports_alpha=True,
            parameters=(DEFAULT_SCALE_PARAMETER,),
            description="Lossless VP8L codec; tunes resolution scale via 1D binary search.",
        )
    )

    # 4. PNG (24/32-bit Lossless) — single tunable parameter: scale in (0, 1]
    register_format(
        FormatSpec(
            id="png",
            name="PNG (Lossless)",
            pil_format="PNG",
            extension=".png",
            fixed_options={"optimize": True},
            supports_alpha=True,
            parameters=(DEFAULT_SCALE_PARAMETER,),
            description="Lossless DEFLATE PNG with optimize=True; tunes scale via 1D binary search.",
        )
    )

    # 5. PNG-8 (Indexed Palette) — scale in (0, 1], colors in [2, 256], dither in {0, 1}
    register_format(
        FormatSpec(
            id="png_quantized",
            name="PNG-8 (Palette)",
            pil_format="PNG",
            extension=".png",
            fixed_options={"optimize": True},
            supports_alpha=True,
            preprocess_hook=_preprocess_png_quantized,
            parameters=(
                DEFAULT_SCALE_PARAMETER,
                ParameterSpec(
                    name="colors",
                    kind="int",
                    min_val=2,
                    max_val=256,
                    default=256,
                    monotonic_with_size=True,
                    description="Palette size [2, 256] colors.",
                ),
                ParameterSpec(
                    name="dither",
                    kind="choice",
                    choices=(0, 1),
                    default=1,
                    monotonic_with_size=True,
                    description="Floyd-Steinberg dithering (0=None, 1=Enabled).",
                ),
            ),
            description="Palette-quantized PNG; tunes scale, palette size [2..256], and dithering.",
        )
    )

    # 6. JPEG 2000 — scale in (0, 1], target_psnr_db in [18, 58]
    register_format(
        FormatSpec(
            id="jpeg2000",
            name="JPEG 2000",
            pil_format="JPEG2000",
            extension=".jp2",
            fixed_options={"quality_mode": "dB", "irreversible": True},
            supports_alpha=True,
            preprocess_hook=_preprocess_jpeg2000,
            parameters=(
                DEFAULT_SCALE_PARAMETER,
                ParameterSpec(
                    name="target_psnr_db",
                    kind="float",
                    min_val=18.0,
                    max_val=58.0,
                    default=58.0,
                    monotonic_with_size=True,
                    description="Wavelet target PSNR layer quality in dB [18, 58].",
                ),
            ),
            description="Wavelet-based JPEG 2000 codec; tunes scale and target dB layer.",
        )
    )

    # 7. GIF — scale in (0, 1], colors in [2, 256]
    register_format(
        FormatSpec(
            id="gif",
            name="GIF",
            pil_format="GIF",
            extension=".gif",
            fixed_options={"optimize": True},
            supports_alpha=False,
            preprocess_hook=_preprocess_gif,
            parameters=(
                DEFAULT_SCALE_PARAMETER,
                ParameterSpec(
                    name="colors",
                    kind="int",
                    min_val=2,
                    max_val=256,
                    default=256,
                    monotonic_with_size=True,
                    description="LZW palette color count [2, 256].",
                ),
            ),
            description="8-bit LZW palette GIF; tunes scale and color count [2..256].",
        )
    )

    # 8. TIFF (Adobe Deflate) — single tunable parameter: scale in (0, 1]
    register_format(
        FormatSpec(
            id="tiff",
            name="TIFF (Deflate)",
            pil_format="TIFF",
            extension=".tiff",
            fixed_options={"compression": "tiff_adobe_deflate"},
            supports_alpha=True,
            parameters=(DEFAULT_SCALE_PARAMETER,),
            description="Lossless Deflate-compressed TIFF; tunes scale via 1D binary search.",
        )
    )


_register_builtin_formats()