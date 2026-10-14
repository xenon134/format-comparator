"""
optimizer.py — Parallel Single- & Multi-Parameter Codec Optimizer
=================================================================

Finds the optimal parameter configuration (including resolution `scale` in (0, 1]
and format-specific parameters such as `quality`, `subsampling`, `colors`, etc.)
whose encoded byte size is `<= target_bytes` and as close as possible to
`target_bytes`.

Optimization strategies:
  1. Single-Parameter Formats (`len(parameters) == 1`, e.g., PNG, Lossless WebP,
     TIFF):
     - Exact 1D Binary Search over `scale` with pixel-dimension memoization and
       local pixel-step neighborhood refinement.
  2. Multi-Parameter Formats (`len(parameters) > 1`, e.g., JPEG, Lossy WebP,
     PNG-8, GIF, JPEG 2000):
     - Hybrid Coordinate-Sliced Binary Search with Monotonic Upper-Bound
       Propagation + Iso-Byte Pareto Frontier Search + Local Coordinate Polish.
  3. Parallel Execution:
     - Concurrent execution across all registered codecs using
       `ThreadPoolExecutor` or `ProcessPoolExecutor` with cooperative
       cancellation (`threading.Event`) for responsive UI slider updates.
"""

from __future__ import annotations

import io
import itertools
import math
import os
import threading
import time
from concurrent.futures import (
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from formats import FormatSpec, ParameterSpec, get_registered_formats


ObjectiveMode = Literal["closest_size", "perceptual_frontier"]


@dataclass
class OptimizerConfig:
    """
    Configuration knobs for the format parameter optimizer.

    Attributes:
        objective_mode:
            - 'closest_size': Strictly maximizes `actual_bytes <= target_bytes`
              (minimizes `target_bytes - actual_bytes`), breaking ties within
              `tie_tolerance_bytes` by higher perceptual PSNR.
            - 'perceptual_frontier': Among configurations using at least 92% of
              `target_bytes`, selects the one with highest perceptual PSNR.
        binary_search_steps: Number of bisection iterations for 1D binary search.
        multi_param_anchors: Number of iso-size frontier slices per continuous knob.
        local_polish_steps: Number of coordinate descent steps around top candidates.
        tie_tolerance_bytes: Byte window within which higher PSNR breaks ties in
                             'closest_size' mode.
        compute_psnr: Whether to decode top candidates and compute PSNR (dB)
                      against the source image.
    """

    objective_mode: ObjectiveMode = "closest_size"
    binary_search_steps: int = 12
    multi_param_anchors: int = 4
    local_polish_steps: int = 3
    tie_tolerance_bytes: int = 10
    compute_psnr: bool = True


@dataclass
class OptimizationResult:
    """
    Result returned by optimizing a single `FormatSpec` for a target byte budget.
    """

    format_id: str
    format_name: str
    target_bytes: int
    actual_bytes: int
    within_budget: bool
    utilization_pct: float
    params: Dict[str, Any]
    scaled_size: Tuple[int, int]
    original_size: Tuple[int, int]
    psnr_db: float
    encoded_bytes: bytes
    decoded_image: Optional[Image.Image]
    evaluations: int
    elapsed_ms: float
    strategy_used: str

    def format_params_summary(self) -> str:
        """Return a concise human-readable string of the optimized parameters."""
        parts: List[str] = []
        for k, v in self.params.items():
            if k == "scale":
                parts.append(f"scale={float(v):.3f}")
            elif isinstance(v, float):
                parts.append(f"{k}={v:.1f}")
            else:
                parts.append(f"{k}={v}")
        return ", ".join(parts)


def compute_psnr(
    original_rgb: np.ndarray,
    encoded_bytes: bytes,
    target_size: Tuple[int, int],
) -> Tuple[float, Image.Image]:
    """
    Decode `encoded_bytes`, upscale back to `target_size` if scaled down,
    and compute Peak Signal-to-Noise Ratio (PSNR in dB) against `original_rgb`.
    """
    decoded = Image.open(io.BytesIO(encoded_bytes))
    decoded.load()
    decoded_rgb_img = decoded.convert("RGB")
    if decoded_rgb_img.size != target_size:
        upscaled = decoded_rgb_img.resize(
            target_size, resample=Image.Resampling.BILINEAR
        )
    else:
        upscaled = decoded_rgb_img

    arr_cmp = np.asarray(upscaled, dtype=np.float32)
    mse = float(np.mean((original_rgb - arr_cmp) ** 2))
    if mse <= 1e-10:
        return 99.9, decoded
    psnr = 10.0 * math.log10((255.0 * 255.0) / mse)
    return max(0.0, min(99.9, psnr)), decoded


class _EvaluationSession:
    """
    Memoized evaluation helper for a single `(image, format_spec, target_bytes)`
    optimization run. Caches results keyed by `(scaled_w, scaled_h, codec_params)`
    so identical discrete configurations are never encoded twice, and computes
    PSNR lazily only when comparing frontier winners.
    """

    def __init__(
        self,
        image: Image.Image,
        spec: FormatSpec,
        target_bytes: int,
        config: OptimizerConfig,
        cancel_event: Optional[threading.Event] = None,
        original_rgb: Optional[np.ndarray] = None,
    ) -> None:
        self.image = image
        self.spec = spec
        self.target_bytes = max(0, int(target_bytes))
        self.config = config
        self.cancel_event = cancel_event
        self.original_rgb = (
            original_rgb
            if original_rgb is not None
            else np.asarray(image.convert("RGB"), dtype=np.float32)
        )

        self.cache: Dict[
            Tuple[Any, ...], Tuple[bytes, Tuple[int, int], Dict[str, Any], float]
        ] = {}
        self.resized_cache: Dict[Tuple[int, int], Image.Image] = {
            (image.width, image.height): image
        }
        self.evaluations = 0

        self.best_valid: Optional[
            Tuple[bytes, Tuple[int, int], Dict[str, Any], float]
        ] = None
        self.smallest_overall: Optional[
            Tuple[bytes, Tuple[int, int], Dict[str, Any], float]
        ] = None

    @property
    def is_cancelled(self) -> bool:
        return self.cancel_event is not None and self.cancel_event.is_set()

    def _get_resized(self, w: int, h: int) -> Image.Image:
        key = (w, h)
        if key not in self.resized_cache:
            self.resized_cache[key] = self.image.resize(
                key, resample=Image.Resampling.BILINEAR
            )
        return self.resized_cache[key]

    def _ensure_psnr(
        self, entry: Tuple[bytes, Tuple[int, int], Dict[str, Any], float]
    ) -> Tuple[bytes, Tuple[int, int], Dict[str, Any], float]:
        enc_bytes, dims, params, psnr_val = entry
        if psnr_val < 0.0 and self.config.compute_psnr:
            psnr_val, _ = compute_psnr(
                self.original_rgb, enc_bytes, self.image.size
            )
            cache_key = (
                dims[0],
                dims[1],
                tuple(
                    (p.name, params[p.name])
                    for p in self.spec.non_scale_parameters
                ),
            )
            updated = (enc_bytes, dims, params, psnr_val)
            self.cache[cache_key] = updated
            return updated
        return entry

    def evaluate(
        self, raw_params: Dict[str, Any]
    ) -> Tuple[int, Tuple[int, int], Dict[str, Any]]:
        """
        Encode the image with `raw_params`, update best candidate tracking,
        and return `(byte_size, (scaled_w, scaled_h), sanitized_params)`.
        """
        clean_params = self.spec.sanitize_params(raw_params)
        scale = float(clean_params.get("scale", 1.0))
        scaled_w, scaled_h = self.spec.compute_scaled_dimensions(
            self.image.width, self.image.height, scale
        )
        effective_scale = round(
            0.5 * (scaled_w / self.image.width + scaled_h / self.image.height), 6
        )
        clean_params["scale"] = max(0.001, min(1.0, effective_scale))

        cache_key = (
            scaled_w,
            scaled_h,
            tuple(
                (p.name, clean_params[p.name])
                for p in self.spec.non_scale_parameters
            ),
        )

        if cache_key in self.cache:
            enc_bytes, dims, saved_params, _ = self.cache[cache_key]
            return len(enc_bytes), dims, saved_params

        work_img = self._get_resized(scaled_w, scaled_h)
        if not self.spec.supports_alpha and work_img.mode in ("RGBA", "LA", "PA", "P"):
            rgba = work_img.convert("RGBA")
            bg = Image.new("RGB", rgba.size, (255, 255, 255))
            bg.paste(rgba, mask=rgba.split()[3])
            work_img = bg
        elif not self.spec.supports_alpha and work_img.mode != "RGB":
            work_img = work_img.convert("RGB")

        codec_params = {k: v for k, v in clean_params.items() if k != "scale"}
        if self.spec.preprocess_hook is not None:
            work_img, save_kwargs = self.spec.preprocess_hook(work_img, codec_params)
        else:
            save_kwargs = dict(codec_params)

        final_save_kwargs = {**self.spec.fixed_options, **save_kwargs}
        buf = io.BytesIO()
        work_img.save(buf, format=self.spec.pil_format, **final_save_kwargs)
        enc_bytes = buf.getvalue()
        self.evaluations += 1

        entry = (enc_bytes, (scaled_w, scaled_h), clean_params, -1.0)
        self.cache[cache_key] = entry
        self._update_best(entry)
        return len(enc_bytes), (scaled_w, scaled_h), clean_params

    def _is_better_valid(
        self,
        new_entry: Tuple[bytes, Tuple[int, int], Dict[str, Any], float],
        cur_entry: Tuple[bytes, Tuple[int, int], Dict[str, Any], float],
    ) -> Tuple[bool, Tuple[bytes, Tuple[int, int], Dict[str, Any], float]]:
        new_sz = len(new_entry[0])
        cur_sz = len(cur_entry[0])

        if self.config.objective_mode == "perceptual_frontier":
            floor_bytes = 0.90 * self.target_bytes
            if new_sz >= floor_bytes and cur_sz >= floor_bytes:
                new_entry = self._ensure_psnr(new_entry)
                cur_entry = self._ensure_psnr(cur_entry)
                self.best_valid = cur_entry
                if abs(new_entry[3] - cur_entry[3]) > 0.15:
                    return (new_entry[3] > cur_entry[3]), new_entry
            return (new_sz > cur_sz), new_entry

        tol = max(4, self.config.tie_tolerance_bytes)
        if abs(new_sz - cur_sz) <= tol:
            if self.config.compute_psnr:
                new_entry = self._ensure_psnr(new_entry)
                cur_entry = self._ensure_psnr(cur_entry)
                self.best_valid = cur_entry
                if abs(new_entry[3] - cur_entry[3]) > 0.05:
                    return (new_entry[3] > cur_entry[3]), new_entry
            new_px = new_entry[1][0] * new_entry[1][1]
            cur_px = cur_entry[1][0] * cur_entry[1][1]
            return (new_px > cur_px), new_entry
        return (new_sz > cur_sz), new_entry

    def _update_best(
        self,
        entry: Tuple[bytes, Tuple[int, int], Dict[str, Any], float],
    ) -> None:
        sz = len(entry[0])
        if self.smallest_overall is None or sz < len(self.smallest_overall[0]):
            self.smallest_overall = entry

        if sz <= self.target_bytes:
            if self.best_valid is None:
                self.best_valid = entry
            else:
                is_better, updated_new = self._is_better_valid(
                    entry, self.best_valid
                )
                if is_better:
                    self.best_valid = updated_new


# ============================================================================
# Strategy 1: 1D Binary Search for Single-Parameter Formats (e.g. PNG, WebP-LL)
# ============================================================================


def _optimize_single_param_binary_search(
    session: _EvaluationSession,
) -> str:
    """
    Perform exact 1D binary search over the single tunable parameter (`scale`)
    followed by local pixel-step neighborhood refinement.
    """
    param = session.spec.parameters[0]
    lo = float(param.min_val if param.min_val is not None else 0.01)
    hi = float(param.max_val if param.max_val is not None else 1.0)

    # 1. Check upper bound (full scale = 1.0) first
    hi_size, _, _ = session.evaluate({param.name: hi})
    if hi_size <= session.target_bytes or session.is_cancelled:
        return "1D Binary Search (Full Scale Fits)"

    # 2. Check lower bound (minimum scale)
    lo_size, _, _ = session.evaluate({param.name: lo})
    if lo_size > session.target_bytes or session.is_cancelled:
        min_px_scale = 1.0 / max(session.image.width, session.image.height)
        session.evaluate({param.name: min_px_scale})
        return "1D Binary Search (Min Floor)"

    # 3. Binary search on [lo, hi] with secant initial jump
    best_lo = lo
    for step_idx in range(session.config.binary_search_steps):
        if session.is_cancelled:
            break
        if step_idx == 0 and hi_size > lo_size:
            ratio = max(
                0.02,
                min(0.95, (session.target_bytes - lo_size) / max(1.0, hi_size - lo_size)),
            )
            mid = lo + (hi - lo) * (ratio ** 0.65)
        else:
            mid = 0.5 * (lo + hi)

        mid_size, _, _ = session.evaluate({param.name: mid})
        if mid_size <= session.target_bytes:
            best_lo = mid
            lo = mid
            if session.target_bytes - mid_size <= 2:
                break
        else:
            hi = mid

    # 4. Local pixel-step polish around best_lo to overcome tiny predictor ripples
    max_dim = max(session.image.width, session.image.height)
    px_step = 1.0 / max(1, max_dim)
    for delta in (-1, 1, 2):
        if session.is_cancelled:
            break
        cand_scale = max(0.001, min(1.0, best_lo + delta * px_step))
        session.evaluate({param.name: cand_scale})

    return "1D Binary Search"


# ============================================================================
# Strategy 2: Hybrid Multi-Parameter Optimizer (Sliced Binary Search + Polish)
# ============================================================================


def _binary_search_1d_slice(
    session: _EvaluationSession,
    fixed_params: Dict[str, Any],
    target_param: ParameterSpec,
    max_iters: int = 7,
    custom_hi: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """
    Given `fixed_params` for all other parameters, run a 1D binary search over
    `target_param` (continuous float or discrete int) to land as close as
    possible to `session.target_bytes` from below (`size <= target_bytes`).
    """
    lo = float(target_param.min_val)
    hi = float(custom_hi if custom_hi is not None else target_param.max_val)
    hi = max(lo, min(float(target_param.max_val), hi))

    p_hi = {**fixed_params, target_param.name: hi}
    sz_hi, _, clean_hi = session.evaluate(p_hi)
    if sz_hi <= session.target_bytes:
        return clean_hi

    p_lo = {**fixed_params, target_param.name: lo}
    sz_lo, _, clean_lo = session.evaluate(p_lo)
    if sz_lo > session.target_bytes:
        return None

    best_params = clean_lo

    for step_idx in range(max_iters):
        if session.is_cancelled:
            break
        if target_param.kind == "int" and int(round(hi)) - int(round(lo)) <= 1:
            break

        if step_idx == 0 and target_param.name == "scale" and sz_hi > sz_lo:
            ratio = max(
                0.02,
                min(0.95, (session.target_bytes - sz_lo) / max(1.0, sz_hi - sz_lo)),
            )
            mid = lo + (hi - lo) * (ratio ** 0.65)
            mid = max(lo + 0.05 * (hi - lo), min(hi - 0.05 * (hi - lo), mid))
        else:
            mid = 0.5 * (lo + hi)

        p_mid = {**fixed_params, target_param.name: mid}
        sz_mid, _, clean_mid = session.evaluate(p_mid)
        if sz_mid <= session.target_bytes:
            best_params = clean_mid
            lo = mid
            if session.target_bytes - sz_mid <= 3:
                break
        else:
            hi = mid

    return best_params


def _optimize_multi_param(session: _EvaluationSession) -> str:
    """
    Multi-parameter optimizer for formats with `scale` + codec knobs
    (e.g., JPEG with `scale`, `quality`, `subsampling`; Lossy WebP; PNG-8; GIF; JP2).

    Uses monotonic upper-bound propagation on `scale` across increasing quality
    anchors so expensive full-scale encodings are never repeated unnecessarily.
    """
    spec = session.spec

    # Stage 1: Check maximum fidelity parameters first
    max_params = spec.max_fidelity_params()
    max_sz, _, _ = session.evaluate(max_params)
    if max_sz <= session.target_bytes or session.is_cancelled:
        return "Multi-Param (Max Fidelity Fits)"

    # Check absolute minimum floor
    min_params = spec.min_fidelity_params()
    min_px_scale = 1.0 / max(session.image.width, session.image.height)
    min_params["scale"] = min(float(min_params.get("scale", 0.01)), min_px_scale)
    session.evaluate(min_params)

    choice_params = [p for p in spec.non_scale_parameters if p.kind == "choice"]
    numeric_params = [
        p for p in spec.non_scale_parameters if p.kind in ("int", "float")
    ]
    scale_param = spec.get_parameter("scale")
    assert scale_param is not None

    if choice_params:
        choice_grids = [list(p.choices or ()) for p in choice_params]
        choice_combos = [
            dict(zip([p.name for p in choice_params], combo))
            for combo in itertools.product(*choice_grids)
        ]
    else:
        choice_combos = [{}]

    frontier_candidates: List[Dict[str, Any]] = []
    primary_choice = choice_combos[0]

    # Stage 2: Full-Resolution (`scale=1.0`) Search on primary choice combo
    if numeric_params:
        primary_num = numeric_params[0]
        other_num_defaults = {p.name: p.default for p in numeric_params[1:]}
        fixed_full = {"scale": 1.0, **primary_choice, **other_num_defaults}
        found_full = _binary_search_1d_slice(
            session, fixed_full, primary_num, max_iters=6
        )
        if found_full is not None:
            frontier_candidates.append(found_full)

    # Stage 3: Iso-Byte Frontier Search in (scale, numeric_param, choice_param) space
    # Since anchor_u_values are increasing in quality/colors, the valid scale
    # upper bound `scale_hi_bound` shrinks monotonically!
    n_anchors = max(3, session.config.multi_param_anchors)
    anchor_u_values = [
        0.12 + 0.78 * (i / float(n_anchors - 1)) for i in range(n_anchors)
    ]

    scale_hi_bound = 1.0
    if numeric_params:
        primary_num = numeric_params[0]
        other_num_defaults = {p.name: p.default for p in numeric_params[1:]}
        for u in anchor_u_values:
            if session.is_cancelled:
                break
            val = primary_num.from_normalized(u)
            fixed = {primary_num.name: val, **primary_choice, **other_num_defaults}
            found_scale_cfg = _binary_search_1d_slice(
                session,
                fixed,
                scale_param,
                max_iters=7,
                custom_hi=scale_hi_bound,
            )
            if found_scale_cfg is not None:
                frontier_candidates.append(found_scale_cfg)
                # Propagate upper bound with a small 8% margin for non-monotonicity
                scale_hi_bound = min(
                    1.0, max(0.05, float(found_scale_cfg["scale"]) * 1.08)
                )
                # Hold discovered scale fixed and refine primary numeric parameter
                refined_num_cfg = _binary_search_1d_slice(
                    session,
                    {
                        "scale": found_scale_cfg["scale"],
                        **primary_choice,
                        **other_num_defaults,
                    },
                    primary_num,
                    max_iters=5,
                )
                if refined_num_cfg is not None:
                    frontier_candidates.append(refined_num_cfg)

        # Evaluate secondary choice combinations (e.g. 4:2:2, 4:4:4) at the best frontier scale
        if len(choice_combos) > 1 and frontier_candidates and not session.is_cancelled:
            ref_cfg = (
                dict(session.best_valid[2])
                if session.best_valid is not None
                else frontier_candidates[-1]
            )
            for alt_choice in choice_combos[1:]:
                if session.is_cancelled:
                    break
                alt_found = _binary_search_1d_slice(
                    session,
                    {
                        "scale": ref_cfg["scale"],
                        **alt_choice,
                        **other_num_defaults,
                    },
                    primary_num,
                    max_iters=5,
                )
                if alt_found is not None:
                    frontier_candidates.append(alt_found)
    else:
        for c_dict in choice_combos:
            if session.is_cancelled:
                break
            found_scale_cfg = _binary_search_1d_slice(
                session, c_dict, scale_param, max_iters=9
            )
            if found_scale_cfg is not None:
                frontier_candidates.append(found_scale_cfg)

    # Stage 4: Coordinate Descent / Local Polish around the winning candidate
    if session.best_valid is not None and not session.is_cancelled:
        max_dim = max(session.image.width, session.image.height)
        px_step = 1.0 / max(1, max_dim)

        cur = dict(session.best_valid[2])
        for _ in range(session.config.local_polish_steps):
            if session.is_cancelled:
                break
            improved = False
            prev_best_sz = len(session.best_valid[0])

            for d_px in (1, 2):
                cand = dict(cur)
                cand["scale"] = max(
                    0.001, min(1.0, float(cur["scale"]) + d_px * px_step)
                )
                sz, _, clean_cand = session.evaluate(cand)
                if sz <= session.target_bytes and sz > prev_best_sz:
                    cur = clean_cand
                    prev_best_sz = sz
                    improved = True

            for p in numeric_params:
                step = (
                    1.0
                    if p.kind == "int"
                    else 0.02 * (float(p.max_val) - float(p.min_val))
                )
                for mult in (1, 2):
                    cand = dict(cur)
                    cand[p.name] = p.clamp_or_normalize(
                        float(cur[p.name]) + mult * step
                    )
                    sz, _, clean_cand = session.evaluate(cand)
                    if sz <= session.target_bytes and sz > prev_best_sz:
                        cur = clean_cand
                        prev_best_sz = sz
                        improved = True

            if not improved:
                break

    return "Multi-Param Frontier + Coordinate Polish"


# ============================================================================
# Public Entry Points for Single-Format & Parallel Multi-Format Optimization
# ============================================================================


def optimize_format(
    image: Image.Image,
    spec: FormatSpec,
    target_bytes: int,
    config: Optional[OptimizerConfig] = None,
    cancel_event: Optional[threading.Event] = None,
    original_rgb: Optional[np.ndarray] = None,
) -> OptimizationResult:
    """
    Find the optimal configuration of `spec.parameters` (options + `scale`)
    for `image` whose encoded size is `<= target_bytes` and closest to `target_bytes`.

    Automatically dispatches to:
      - 1D Binary Search when `spec.is_single_parameter` is True
      - Hybrid Multi-Parameter Frontier + Coordinate Polish when `len(spec.parameters) > 1`
    """
    cfg = config or OptimizerConfig()
    t0 = time.perf_counter()

    session = _EvaluationSession(
        image=image,
        spec=spec,
        target_bytes=target_bytes,
        config=cfg,
        cancel_event=cancel_event,
        original_rgb=original_rgb,
    )

    if spec.is_single_parameter:
        strategy = _optimize_single_param_binary_search(session)
    else:
        strategy = _optimize_multi_param(session)

    within_budget = session.best_valid is not None
    chosen = session.best_valid if within_budget else session.smallest_overall
    assert chosen is not None

    enc_bytes, scaled_dims, best_params, psnr_val = chosen
    if psnr_val < 0.0:
        psnr_val, decoded_img = compute_psnr(
            session.original_rgb, enc_bytes, image.size
        )
    else:
        decoded_img = Image.open(io.BytesIO(enc_bytes))
        decoded_img.load()

    actual_sz = len(enc_bytes)
    utilization = (
        (100.0 * actual_sz / target_bytes) if target_bytes > 0 else 100.0
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return OptimizationResult(
        format_id=spec.id,
        format_name=spec.name,
        target_bytes=target_bytes,
        actual_bytes=actual_sz,
        within_budget=within_budget,
        utilization_pct=utilization,
        params=best_params,
        scaled_size=scaled_dims,
        original_size=image.size,
        psnr_db=psnr_val,
        encoded_bytes=enc_bytes,
        decoded_image=decoded_img,
        evaluations=session.evaluations,
        elapsed_ms=elapsed_ms,
        strategy_used=strategy,
    )


def _process_worker_entry(
    png_source_bytes: bytes,
    format_id: str,
    target_bytes: int,
    config: OptimizerConfig,
) -> OptimizationResult:
    """Top-level picklable worker function for `ProcessPoolExecutor` mode."""
    from formats import get_format_by_id

    img = Image.open(io.BytesIO(png_source_bytes))
    img.load()
    spec = get_format_by_id(format_id)
    res = optimize_format(img, spec, target_bytes, config=config)
    res.decoded_image = None
    return res


class ParallelFormatOptimizer:
    """
    Manages parallel optimization across all formats using a worker pool
    (`ThreadPoolExecutor` by default, or `ProcessPoolExecutor`).

    Supports cooperative cancellation (`cancel_current()`) so moving the
    UI slider immediately aborts in-flight searches and starts the new target.
    """

    def __init__(
        self,
        formats: Optional[Sequence[FormatSpec]] = None,
        max_workers: Optional[int] = None,
        executor_mode: Literal["thread", "process"] = "thread",
        config: Optional[OptimizerConfig] = None,
    ) -> None:
        self.formats: List[FormatSpec] = (
            list(formats) if formats is not None else get_registered_formats()
        )
        self.max_workers = max_workers or min(
            len(self.formats), max(4, (os.cpu_count() or 4))
        )
        self.executor_mode = executor_mode
        self.config = config or OptimizerConfig()
        self._cancel_event = threading.Event()
        self._lock = threading.Lock()
        self._executor = (
            ProcessPoolExecutor(max_workers=self.max_workers)
            if executor_mode == "process"
            else ThreadPoolExecutor(
                max_workers=self.max_workers, thread_name_prefix="CodecOpt"
            )
        )

    def cancel_current(self) -> threading.Event:
        """Signal any currently running optimization tasks to stop early."""
        with self._lock:
            self._cancel_event.set()
            self._cancel_event = threading.Event()
            return self._cancel_event

    def optimize_all(
        self,
        image: Image.Image,
        target_bytes: int,
        on_format_done: Optional[Callable[[OptimizationResult], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> List[OptimizationResult]:
        """
        Optimize all registered formats in parallel for `target_bytes`.
        """
        ev = cancel_event or self.cancel_current()
        original_rgb = np.asarray(image.convert("RGB"), dtype=np.float32)

        futures_map: Dict[Future[OptimizationResult], FormatSpec] = {}
        if self.executor_mode == "process":
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            raw_png = buf.getvalue()
            for spec in self.formats:
                fut = self._executor.submit(
                    _process_worker_entry,
                    raw_png,
                    spec.id,
                    target_bytes,
                    self.config,
                )
                futures_map[fut] = spec
        else:
            for spec in self.formats:
                img_copy = image.copy()
                fut = self._executor.submit(
                    optimize_format,
                    img_copy,
                    spec,
                    target_bytes,
                    self.config,
                    ev,
                    original_rgb,
                )
                futures_map[fut] = spec

        results_by_id: Dict[str, OptimizationResult] = {}
        for fut in as_completed(futures_map):
            spec = futures_map[fut]
            res = fut.result()
            if res.decoded_image is None and res.encoded_bytes:
                dec = Image.open(io.BytesIO(res.encoded_bytes))
                dec.load()
                res.decoded_image = dec
            results_by_id[spec.id] = res
            if on_format_done is not None and not ev.is_set():
                on_format_done(res)

        return [
            results_by_id[s.id]
            for s in self.formats
            if s.id in results_by_id
        ]

    def shutdown(self) -> None:
        """Cleanly shut down the underlying worker pool."""
        self._cancel_event.set()
        self._executor.shutdown(wait=False, cancel_futures=True)