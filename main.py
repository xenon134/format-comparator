"""
main.py — Format Compare Visual (Tkinter GUI & CLI Entry Point)
===============================================================

Opens a Tkinter window with:
  1. A top control bar featuring a file size slider ranging from `0` to the
     input image's file size (plus direct byte/KB entry and quick presets).
  2. A large single-format display showing one codec at a time (all codecs
     still optimized in parallel in the background via `formats.py` +
     `optimizer.py`), with a bottom tab bar of equal-sized codec tabs.
     Tabs support click-to-view and Left/Right arrow navigation (tabs checked
     as `skip` in `disabledForNavigation` are skipped by arrows only).
  3. Interactive click-to-inspect modal for side-by-side zoom comparison
     against the original image and one-click export.

Usage:
  python main.py [path/to/image.jpg]
  python main.py --benchmark --budget 2560 [path/to/image.jpg]
"""

from __future__ import annotations

import argparse
import io
import os
import queue
import sys
import threading
import time
from typing import Any, Dict, List, Literal, Optional, Tuple

from PIL import Image

from formats import FormatSpec, get_registered_formats
from optimizer import (
    ObjectiveMode,
    OptimizationResult,
    OptimizerConfig,
    ParallelFormatOptimizer,
)


def format_byte_size(num_bytes: float) -> str:
    """Format a byte count into a clean human-readable string (B, KB, MB)."""
    b = max(0.0, float(num_bytes))
    if b < 1024:
        return f"{int(round(b))} B"
    if b < 1024 * 1024:
        return f"{b / 1024.0:.2f} KB"
    return f"{b / (1024.0 * 1024.0):.2f} MB"


def format_shortfall(num_bytes: float) -> str:
    """Format how far `num_bytes` is short of a target as B / KB / MB."""
    b = max(0.0, float(num_bytes))
    if b < 1024:
        return f"{int(round(b))} B"
    if b < 1024 * 1024:
        return f"{b / 1024.0:.2f} KB"
    return f"{b / (1024.0 * 1024.0):.2f} MB"


def parse_size_input(text: str, max_bytes: int) -> Optional[int]:
    """
    Parse user size string like '2.5 KB', '2500', '15%', '0.5 MB' into bytes.
    """
    s = text.strip().upper()
    if not s:
        return None
    try:
        if s.endswith("%"):
            pct = float(s[:-1].strip())
            return max(0, min(max_bytes, int(round(max_bytes * (pct / 100.0)))))
        if s.endswith("KB") or s.endswith("K"):
            num = float(s.rstrip("KB").strip())
            return max(0, min(max_bytes, int(round(num * 1024.0))))
        if s.endswith("MB") or s.endswith("M"):
            num = float(s.rstrip("MB").strip())
            return max(0, min(max_bytes, int(round(num * 1024.0 * 1024.0))))
        if s.endswith("B"):
            num = float(s[:-1].strip())
            return max(0, min(max_bytes, int(round(num))))
        num = float(s)
        return max(0, min(max_bytes, int(round(num))))
    except ValueError:
        return None


def load_random_image_from_folder() -> Optional[Tuple[Image.Image, int, str]]:
    import PILutils
    img = PILutils.randomPicture()
    return img, max(256, os.path.getsize(img.filename)), os.path.basename(img.filename)


def load_image_and_size(path: Optional[str]) -> Tuple[Image.Image, int, str]:
    """
    Load an input image from `path`. If `path` is None or missing, pick a
    random image from the placeholder folder (see `load_random_image_from_folder`).
    Returns `(image, input_file_size_bytes, display_label)`.
    """
    if path and os.path.isfile(path):
        file_bytes = os.path.getsize(path)
        img = Image.open(path)
        img.load()
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGBA" if "A" in img.mode else "RGB")
        return img, max(256, file_bytes), os.path.basename(path)
    random_img = load_random_image_from_folder()
    if random_img is not None:
        return random_img
    raise FileNotFoundError(
        "No input image supplied and no random image found in "
        f"{PLACEHOLDER_IMAGE_FOLDER}. Pass an image path explicitly."
    )


# ============================================================================
# Tkinter Application UI
# ============================================================================


class FormatCompareApp:
    """
    Main Tkinter GUI window for Format Compare Visual.

    Features:
      - Top control bar with a File Size Slider (`0` .. `input_file_size`),
        direct size entry box (e.g., `2.5 KB`), quick percentage buttons,
        objective selector, and upscale filter toggle.
      - Single-codec display showing one optimized result at a time with actual
        size, budget utilization, effective resolution (`scale`), PSNR (dB),
        and chosen codec options.
      - Bottom tab bar with equal-sized codec tabs (click to view) plus a
        `skip` checkbox per tab and Left/Right arrow navigation. Checked tabs
        in `disabledForNavigation` are skipped by arrows only (still optimized
        and clickable).
      - Background worker pool with cooperative cancellation so dragging the
        slider triggers immediate parallel re-optimization without freezing UI.
    """

    BG_MAIN = "#0f172a"
    BG_PANEL = "#1e293b"
    BG_CARD = "#1e293b"
    BG_CARD_HEADER = "#334155"
    FG_PRIMARY = "#f8fafc"
    FG_SECONDARY = "#94a3b8"
    ACCENT_BLUE = "#38bdf8"
    ACCENT_GREEN = "#4ade80"
    ACCENT_AMBER = "#fbbf24"
    ACCENT_RED = "#f87171"

    def __init__(self, root: Any, initial_image_path: Optional[str] = None) -> None:
        import tkinter as tk
        from tkinter import ttk
        from PIL import ImageTk

        self.tk = tk
        self.ttk = ttk
        self.ImageTk = ImageTk
        self.root = root
        self.root.title("Format Compare Visual — Multi-Codec File Size Optimizer")
        self.root.geometry("1280x860")
        self.root.minsize(920, 640)
        self.root.configure(bg=self.BG_MAIN)

        # Load initial image & determine slider upper bound
        self.source_image, self.input_file_size, self.image_label = load_image_and_size(
            initial_image_path
        )

        # Default target budget: 15% of input file size (or ~10 KB)
        default_budget = max(512, min(self.input_file_size, int(self.input_file_size * 0.15)))

        self.formats: List[FormatSpec] = get_registered_formats()
        self.optimizer_config = OptimizerConfig(objective_mode="closest_size")
        self.parallel_optimizer = ParallelFormatOptimizer(
            formats=self.formats,
            executor_mode="thread",
            config=self.optimizer_config,
        )

        # State variables
        self.slider_var = tk.IntVar(value=default_budget)
        self.entry_var = tk.StringVar(value=f"{default_budget / 1024.0:.2f} KB")
        self.objective_var = tk.StringVar(value="closest_size")
        self.status_var = tk.StringVar(value="Ready.")

        self._result_queue: "queue.Queue[Tuple[int, OptimizationResult, bool]]" = queue.Queue()
        self._latest_results: Dict[str, OptimizationResult] = {}
        self._photo_refs: Dict[str, Any] = {}
        self._card_widgets: Dict[str, Dict[str, Any]] = {}
        # Single-view selection state (UI only; backend still optimizes all formats).
        self.active_format_id: Optional[str] = (
            self.formats[0].id if self.formats else None
        )
        # When True for a format_id, Left/Right arrow navigation skips that tab.
        # The format itself is still optimized and still viewable by clicking its tab.
        self.disabledForNavigation: Dict[str, Any] = {}
        self._tab_widgets: Dict[str, Dict[str, Any]] = {}
        self._single_widgets: Dict[str, Any] = {}
        self._debounce_after_id: Optional[str] = None
        self._run_generation_id: int = 0
        self._completed_in_run: int = 0
        self._run_start_time: float = 0.0
        # Dynamic preview sizing: no hardcoded pixel box. The preview photo is
        # re-rendered to fit whatever space the image label currently has.
        self._last_preview_box: Optional[Tuple[int, int]] = None
        self._preview_resize_after_id: Optional[str] = None

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Start polling queue and kick off initial optimization
        self.root.after(30, self._poll_result_queue)
        self.trigger_optimization(default_budget)

    def _build_ui(self) -> None:
        tk = self.tk
        ttk = self.ttk

        # --------------------------------------------------------------------
        # 1. Top Control Panel (Header + File Size Slider + Quick Controls)
        # --------------------------------------------------------------------
        top_frame = tk.Frame(self.root, bg=self.BG_PANEL, padx=16, pady=12)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=12, pady=(12, 6))

        # Row 1: Title, Image Info, and Open Image Button
        header_row = tk.Frame(top_frame, bg=self.BG_PANEL)
        header_row.pack(side=tk.TOP, fill=tk.X, pady=(0, 8))

        title_lbl = tk.Label(
            header_row,
            text="Format Compare Visual",
            font=("Helvetica", 15, "bold"),
            fg=self.FG_PRIMARY,
            bg=self.BG_PANEL,
        )
        title_lbl.pack(side=tk.LEFT)

        self.info_lbl = tk.Label(
            header_row,
            text=self._format_source_info(),
            font=("Helvetica", 10),
            fg=self.ACCENT_BLUE,
            bg=self.BG_PANEL,
            padx=14,
        )
        self.info_lbl.pack(side=tk.LEFT)

        open_btn = tk.Button(
            header_row,
            text="Open Image...",
            command=self._on_open_image,
            bg="#0284c7",
            fg=self.FG_PRIMARY,
            activebackground="#0369a1",
            activeforeground=self.FG_PRIMARY,
            relief=tk.FLAT,
            padx=12,
            pady=4,
            font=("Helvetica", 10, "bold"),
        )
        open_btn.pack(side=tk.RIGHT)

        # Row 2: File Size Slider (0 .. input_file_size) + Direct Input
        slider_row = tk.Frame(top_frame, bg=self.BG_PANEL)
        slider_row.pack(side=tk.TOP, fill=tk.X, pady=(4, 6))

        slider_title = tk.Label(
            slider_row,
            text="Target File Size Budget:",
            font=("Helvetica", 10, "bold"),
            fg=self.FG_PRIMARY,
            bg=self.BG_PANEL,
        )
        slider_title.pack(side=tk.LEFT, padx=(0, 10))

        self.size_scale = tk.Scale(
            slider_row,
            from_=0,
            to=self.input_file_size,
            orient=tk.HORIZONTAL,
            variable=self.slider_var,
            command=self._on_slider_moved,
            showvalue=False,
            bg=self.BG_PANEL,
            fg=self.FG_PRIMARY,
            troughcolor="#0f172a",
            highlightthickness=0,
            bd=0,
            length=480,
        )
        self.size_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 12))

        self.readout_lbl = tk.Label(
            slider_row,
            text=format_byte_size(self.slider_var.get()),
            font=("Courier", 11, "bold"),
            fg=self.ACCENT_GREEN,
            bg="#0f172a",
            padx=10,
            pady=4,
            width=22,
            anchor="e",
        )
        self.readout_lbl.pack(side=tk.LEFT, padx=(0, 8))

        size_entry = tk.Entry(
            slider_row,
            textvariable=self.entry_var,
            width=11,
            font=("Courier", 10),
            bg="#0f172a",
            fg=self.FG_PRIMARY,
            insertbackground=self.FG_PRIMARY,
            relief=tk.FLAT,
        )
        size_entry.pack(side=tk.LEFT, padx=(0, 4), ipady=4)
        size_entry.bind("<Return>", self._on_entry_submitted)

        apply_btn = tk.Button(
            slider_row,
            text="Set Size",
            command=self._on_entry_submitted,
            bg="#334155",
            fg=self.FG_PRIMARY,
            activebackground="#475569",
            activeforeground=self.FG_PRIMARY,
            relief=tk.FLAT,
            padx=8,
            pady=3,
            font=("Helvetica", 9),
        )
        apply_btn.pack(side=tk.LEFT)

        # Row 3: Quick Presets + Optimizer Mode + Display Upscale Filter + Status
        controls_row = tk.Frame(top_frame, bg=self.BG_PANEL)
        controls_row.pack(side=tk.TOP, fill=tk.X, pady=(4, 0))

        tk.Label(
            controls_row,
            text="Presets:",
            font=("Helvetica", 9),
            fg=self.FG_SECONDARY,
            bg=self.BG_PANEL,
        ).pack(side=tk.LEFT, padx=(0, 6))

        for label, byte_fn in [
            ("1 KB", lambda m: min(m, 1024)),
            ("2.5 KB", lambda m: min(m, 2560)),
            ("5 KB", lambda m: min(m, 5120)),
            ("5%", lambda m: int(m * 0.05)),
            ("15%", lambda m: int(m * 0.15)),
            ("35%", lambda m: int(m * 0.35)),
            ("75%", lambda m: int(m * 0.75)),
            ("100%", lambda m: m),
        ]:
            btn = tk.Button(
                controls_row,
                text=label,
                command=lambda fn=byte_fn: self._set_budget(fn(self.input_file_size)),
                bg="#0f172a",
                fg=self.FG_SECONDARY,
                activebackground="#334155",
                activeforeground=self.FG_PRIMARY,
                relief=tk.FLAT,
                padx=7,
                pady=2,
                font=("Helvetica", 9),
            )
            btn.pack(side=tk.LEFT, padx=2)

        # Objective selector
        tk.Label(
            controls_row,
            text="  Objective:",
            font=("Helvetica", 9),
            fg=self.FG_SECONDARY,
            bg=self.BG_PANEL,
        ).pack(side=tk.LEFT, padx=(10, 4))

        obj_menu = ttk.Combobox(
            controls_row,
            textvariable=self.objective_var,
            values=["closest_size", "perceptual_frontier"],
            width=16,
            state="readonly",
        )
        obj_menu.pack(side=tk.LEFT)
        obj_menu.bind(
            "<<ComboboxSelected>>",
            lambda _e: self._on_objective_changed(),
        )

        # Live status indicator on the right
        self.status_lbl = tk.Label(
            controls_row,
            textvariable=self.status_var,
            font=("Helvetica", 9),
            fg=self.ACCENT_AMBER,
            bg=self.BG_PANEL,
        )
        self.status_lbl.pack(side=tk.RIGHT)

        # --------------------------------------------------------------------
        # 2. Single-Format Display Area (one codec visible at a time)
        # --------------------------------------------------------------------
        display_outer = tk.Frame(self.root, bg=self.BG_MAIN)
        display_outer.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=12, pady=(4, 6))

        self._create_single_view(display_outer)

        # --------------------------------------------------------------------
        # 3. Bottom Codec Tab Bar (equal-sized tabs, one per codec)
        # --------------------------------------------------------------------
        tab_outer = tk.Frame(self.root, bg=self.BG_PANEL, padx=8, pady=8)
        tab_outer.pack(side=tk.BOTTOM, fill=tk.X, padx=12, pady=(0, 12))

        self.tab_bar = tk.Frame(tab_outer, bg=self.BG_PANEL)
        self.tab_bar.pack(side=tk.TOP, fill=tk.X, expand=True)

        self._create_tab_bar()

        # Arrow-key navigation between tabs (skips disabledForNavigation tabs).
        self.root.bind("<Left>", self._on_left_arrow)
        self.root.bind("<Right>", self._on_right_arrow)

    def _create_single_view(self, parent: Any) -> None:
        """Create the large single-codec display area."""
        tk = self.tk

        card = tk.Frame(
            parent,
            bg=self.BG_CARD,
            highlightbackground="#334155",
            highlightthickness=1,
            padx=14,
            pady=10,
        )
        card.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Header: format name + strategy badge
        hdr = tk.Frame(card, bg=self.BG_CARD)
        hdr.pack(side=tk.TOP, fill=tk.X, pady=(0, 6))

        name_lbl = tk.Label(
            hdr,
            text="",
            font=("Helvetica", 13, "bold"),
            fg=self.FG_PRIMARY,
            bg=self.BG_CARD,
        )
        name_lbl.pack(side=tk.LEFT)

        badge_lbl = tk.Label(
            hdr,
            text="",
            font=("Helvetica", 8),
            fg=self.ACCENT_BLUE,
            bg="#0f172a",
            padx=6,
            pady=2,
        )
        badge_lbl.pack(side=tk.RIGHT)

        # Large image area (click to inspect). Fills all free card space;
        # the photo itself is re-rendered on <Configure> to match the label size.
        img_lbl = tk.Label(
            card,
            bg="#090d16",
            text="Optimizing...",
            fg=self.FG_SECONDARY,
            font=("Helvetica", 10),
            cursor="hand2",
            anchor="center",
        )
        img_lbl.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=4)
        img_lbl.bind("<Button-1>", lambda _e: self._open_active_inspector())
        img_lbl.bind("<Configure>", self._on_preview_resize)

        # Size & budget utilization readout
        size_lbl = tk.Label(
            card,
            text="Size: —",
            font=("Courier", 11, "bold"),
            fg=self.ACCENT_GREEN,
            bg=self.BG_CARD,
            anchor="w",
        )
        size_lbl.pack(side=tk.TOP, fill=tk.X, pady=(4, 1))

        # Resolution & PSNR readout
        metrics_lbl = tk.Label(
            card,
            text="Res: — | PSNR: —",
            font=("Helvetica", 10),
            fg=self.FG_PRIMARY,
            bg=self.BG_CARD,
            anchor="w",
        )
        metrics_lbl.pack(side=tk.TOP, fill=tk.X, pady=1)

        # Tuned parameters summary
        params_lbl = tk.Label(
            card,
            text="Params: —",
            font=("Courier", 9),
            fg=self.FG_SECONDARY,
            bg=self.BG_CARD,
            anchor="w",
            justify=tk.LEFT,
        )
        params_lbl.pack(side=tk.TOP, fill=tk.X, pady=(2, 0))

        hint_lbl = tk.Label(
            card,
            text="Click image to inspect • Use tabs below or ← / → arrow keys to switch codec",
            font=("Helvetica", 8),
            fg=self.FG_SECONDARY,
            bg=self.BG_CARD,
            anchor="w",
        )
        hint_lbl.pack(side=tk.TOP, fill=tk.X, pady=(6, 0))

        self._single_widgets = {
            "card": card,
            "name_lbl": name_lbl,
            "badge_lbl": badge_lbl,
            "img_lbl": img_lbl,
            "size_lbl": size_lbl,
            "metrics_lbl": metrics_lbl,
            "params_lbl": params_lbl,
        }

    def _create_tab_bar(self) -> None:
        """Create equal-sized bottom tabs, one per codec, each with a nav checkbox."""
        tk = self.tk
        for idx, spec in enumerate(self.formats):
            # uniform="codec_tabs" keeps every tab cell exactly equal width.
            self.tab_bar.grid_columnconfigure(idx, weight=1, uniform="codec_tabs")

            cell = tk.Frame(self.tab_bar, bg=self.BG_PANEL, padx=2, pady=2)
            cell.grid(row=0, column=idx, sticky="nsew")

            tab_btn = tk.Button(
                cell,
                text=spec.name,
                command=lambda fid=spec.id: self.select_format(fid),
                bg="#0f172a",
                fg=self.FG_SECONDARY,
                activebackground="#334155",
                activeforeground=self.FG_PRIMARY,
                relief=tk.FLAT,
                padx=4,
                pady=8,
                font=("Helvetica", 9, "bold"),
                wraplength=140,
            )
            tab_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))

            # Checked => this tab is skipped by Left/Right navigation only.
            # The format itself stays fully optimized and clickable.
            skip_var = tk.BooleanVar(value=False)
            self.disabledForNavigation[spec.id] = skip_var
            skip_chk = tk.Checkbutton(
                cell,
                text="skip",
                variable=skip_var,
                bg=self.BG_PANEL,
                fg=self.FG_SECONDARY,
                selectcolor="#0f172a",
                activebackground=self.BG_PANEL,
                activeforeground=self.FG_PRIMARY,
                relief=tk.FLAT,
                padx=2,
                font=("Helvetica", 8),
            )
            skip_chk.pack(side=tk.RIGHT)

            self._tab_widgets[spec.id] = {
                "cell": cell,
                "button": tab_btn,
                "skip_check": skip_chk,
            }

        self._refresh_tab_styles()
        # Show placeholder header for the initially selected format.
        self._show_placeholder_for_active()

    def select_format(self, format_id: str) -> None:
        """Show the optimized result for `format_id` in the single-view area."""
        known_ids = {s.id for s in self.formats}
        if format_id not in known_ids:
            return
        self.active_format_id = format_id
        self._refresh_tab_styles()
        res = self._latest_results.get(format_id)
        if res is not None:
            self._render_single_result(res)
        else:
            self._show_placeholder_for_active()

    def _refresh_tab_styles(self) -> None:
        """Highlight the active tab; leave all others in the idle style."""
        for fid, widgets in self._tab_widgets.items():
            btn = widgets["button"]
            if fid == self.active_format_id:
                btn.configure(bg="#0284c7", fg=self.FG_PRIMARY)
            else:
                btn.configure(bg="#0f172a", fg=self.FG_SECONDARY)

    def _show_placeholder_for_active(self) -> None:
        spec = next(
            (s for s in self.formats if s.id == self.active_format_id),
            self.formats[0] if self.formats else None,
        )
        if spec is None or not self._single_widgets:
            return
        badge_text = (
            "1D Binary Search"
            if spec.is_single_parameter
            else f"{len(spec.parameters)}D Multi-Param"
        )
        self._single_widgets["name_lbl"].configure(text=spec.name)
        self._single_widgets["badge_lbl"].configure(text=badge_text)
        self._single_widgets["img_lbl"].configure(image="", text="Optimizing...")
        self._single_widgets["size_lbl"].configure(text="Size: —")
        self._single_widgets["metrics_lbl"].configure(text="Res: — | PSNR: —")
        self._single_widgets["params_lbl"].configure(text="Params: —")

    def _is_navigation_skipped(self, format_id: str) -> bool:
        var = self.disabledForNavigation.get(format_id)
        if var is None:
            return False
        try:
            return bool(var.get())
        except Exception:
            return bool(var)

    def _navigate_tabs(self, direction: int) -> None:
        """Move selection left (-1) or right (+1), skipping skipped tabs, wrapping."""
        if not self.formats:
            return
        ids = [s.id for s in self.formats]
        try:
            cur_idx = ids.index(self.active_format_id)  # type: ignore[arg-type]
        except ValueError:
            cur_idx = 0 if direction > 0 else len(ids) - 1
        for step in range(1, len(ids) + 1):
            nxt_idx = (cur_idx + direction * step) % len(ids)
            candidate = ids[nxt_idx]
            if not self._is_navigation_skipped(candidate):
                self.select_format(candidate)
                return
        # Every tab is skipped: keep the current selection.

    def _on_left_arrow(self, event: Any = None) -> Optional[str]:
        if self._arrow_key_in_text_entry():
            return None
        self._navigate_tabs(-1)
        return "break"

    def _on_right_arrow(self, event: Any = None) -> Optional[str]:
        if self._arrow_key_in_text_entry():
            return None
        self._navigate_tabs(1)
        return "break"

    def _arrow_key_in_text_entry(self) -> bool:
        try:
            focused = self.root.focus_get()
            if focused is not None and focused.winfo_class() in (
                "Entry",
                "TEntry",
                "TCombobox",
                "Text",
            ):
                return True
        except Exception:
            pass
        return False

    def _open_active_inspector(self) -> None:
        if self.active_format_id is not None:
            self._open_inspector_modal(self.active_format_id)

    def _format_source_info(self) -> str:
        w, h = self.source_image.size
        return (
            f"Source: {self.image_label}  |  {w}×{h} px  |  "
            f"Input Size: {format_byte_size(self.input_file_size)}"
        )

    def _on_slider_moved(self, raw_val: str) -> None:
        target = int(float(raw_val))
        self.readout_lbl.configure(text=format_byte_size(target))
        self.entry_var.set(f"{target / 1024.0:.2f} KB")

        # Debounce rapid slider motion (65ms) while cancelling stale runs immediately
        self.parallel_optimizer.cancel_current()
        if self._debounce_after_id is not None:
            self.root.after_cancel(self._debounce_after_id)
        self._debounce_after_id = self.root.after(
            65, lambda: self.trigger_optimization(target)
        )

    def _on_entry_submitted(self, _event: Any = None) -> None:
        parsed = parse_size_input(self.entry_var.get(), self.input_file_size)
        if parsed is not None:
            self._set_budget(parsed)

    def _set_budget(self, target_bytes: int) -> None:
        clamped = max(0, min(self.input_file_size, int(target_bytes)))
        self.slider_var.set(clamped)
        self.readout_lbl.configure(text=format_byte_size(clamped))
        self.entry_var.set(f"{clamped / 1024.0:.2f} KB")
        self.trigger_optimization(clamped)

    def _on_objective_changed(self) -> None:
        mode: ObjectiveMode = (
            "perceptual_frontier"
            if self.objective_var.get() == "perceptual_frontier"
            else "closest_size"
        )
        self.optimizer_config.objective_mode = mode
        self.trigger_optimization(self.slider_var.get())

    def _on_open_image(self) -> None:
        from tkinter import filedialog

        path = filedialog.askopenfilename(
            title="Select Input Image",
            filetypes=[
                ("Image Files", "*.png *.jpg *.jpeg *.webp *.bmp *.gif *.tiff *.jp2"),
                ("All Files", "*.*"),
            ],
        )
        if not path:
            return
        img, fsize, label = load_image_and_size(path)
        self.source_image = img
        self.input_file_size = fsize
        self.image_label = label
        self.info_lbl.configure(text=self._format_source_info())
        self.size_scale.configure(to=self.input_file_size)
        new_budget = max(512, int(self.input_file_size * 0.15))
        self._set_budget(new_budget)

    def trigger_optimization(self, target_bytes: int) -> None:
        """
        Cancel any in-flight optimization and launch parallel optimization
        across all formats in a background controller thread.
        """
        self._run_generation_id += 1
        gen_id = self._run_generation_id
        self._completed_in_run = 0
        self._run_start_time = time.perf_counter()

        cancel_ev = self.parallel_optimizer.cancel_current()
        self.status_var.set(
            f"Optimizing {len(self.formats)} codecs in parallel for {format_byte_size(target_bytes)}..."
        )

        source_copy = self.source_image.copy()

        def _bg_runner() -> None:
            def _on_one_done(res: OptimizationResult) -> None:
                if not cancel_ev.is_set():
                    self._result_queue.put((gen_id, res, False))

            self.parallel_optimizer.optimize_all(
                image=source_copy,
                target_bytes=target_bytes,
                on_format_done=_on_one_done,
                cancel_event=cancel_ev,
            )
            if not cancel_ev.is_set():
                dummy = self._latest_results.get(self.formats[0].id)
                if dummy is not None:
                    self._result_queue.put((gen_id, dummy, True))

        threading.Thread(target=_bg_runner, daemon=True).start()

    def _poll_result_queue(self) -> None:
        """Drain completed `OptimizationResult` objects and update single view."""
        try:
            while True:
                gen_id, res, is_done_sentinel = self._result_queue.get_nowait()
                if gen_id != self._run_generation_id:
                    continue
                if is_done_sentinel:
                    elapsed = (time.perf_counter() - self._run_start_time) * 1000.0
                    self.status_var.set(
                        f"All {len(self.formats)} codecs optimized in {elapsed:.0f} ms (click image to inspect, use tabs or ←/→)."
                    )
                    continue
                self._latest_results[res.format_id] = res
                self._completed_in_run += 1
                self._update_card(res)
                if self._completed_in_run == len(self.formats):
                    elapsed = (time.perf_counter() - self._run_start_time) * 1000.0
                    self.status_var.set(
                        f"All {len(self.formats)} codecs optimized in {elapsed:.0f} ms (click image to inspect, use tabs or ←/→)."
                    )
        except queue.Empty:
            pass
        finally:
            self.root.after(25, self._poll_result_queue)

    def _get_available_preview_size(self) -> Tuple[int, int]:
        """Return the current pixel box available for the preview photo.

        Measured live from the image label so the preview fills whatever
        window size the user gives it. Falls back to the card / window size
        while the layout is still settling (winfo_width/height == 1).
        """
        pad = 8
        w, h = 0, 0
        try:
            lbl = self._single_widgets.get("img_lbl")
            if lbl is not None and lbl.winfo_exists():
                w, h = lbl.winfo_width(), lbl.winfo_height()
        except Exception:
            w, h = 0, 0
        if w < 50 or h < 50:
            try:
                card = self._single_widgets.get("card")
                if card is not None and card.winfo_exists():
                    w, h = card.winfo_width() - pad, card.winfo_height() - 140
            except Exception:
                pass
        if w < 50 or h < 50:
            try:
                w, h = self.root.winfo_width() - 60, self.root.winfo_height() - 380
            except Exception:
                pass
        return (max(80, int(w - pad)), max(80, int(h - pad)))

    def _on_preview_resize(self, event: Any = None) -> None:
        """Debounced re-render of the active result when the view size changes."""
        if self._preview_resize_after_id is not None:
            try:
                self.root.after_cancel(self._preview_resize_after_id)
            except Exception:
                pass
            self._preview_resize_after_id = None

        def _do_resize() -> None:
            self._preview_resize_after_id = None
            res = self._latest_results.get(self.active_format_id or "")
            if res is None or res.decoded_image is None:
                return
            box = self._get_available_preview_size()
            last = self._last_preview_box
            # Ignore tiny <8px jitter so re-rendering the photo can't loop.
            if last is not None and abs(box[0] - last[0]) < 8 and abs(box[1] - last[1]) < 8:
                return
            self._render_single_result(res)

        self._preview_resize_after_id = self.root.after(80, _do_resize)

    def _render_preview_photo(
        self, decoded_img: Image.Image, max_w: Optional[int] = None, max_h: Optional[int] = None
    ) -> Any:
        """
        Fit a decoded codec image into the currently available display space
        (preserving the source aspect ratio) so resolution loss and compression
        artifacts can be compared across codecs at any window size.
        """
        if max_w is None or max_h is None:
            avail_w, avail_h = self._get_available_preview_size()
            max_w = avail_w if max_w is None else max_w
            max_h = avail_h if max_h is None else max_h
        max_w, max_h = max(80, int(max_w)), max(80, int(max_h))
        self._last_preview_box = (max_w, max_h)

        src_w, src_h = self.source_image.size
        fit_ratio = min(max_w / max(1, src_w), max_h / max(1, src_h))
        disp_w = max(1, int(round(src_w * fit_ratio)))
        disp_h = max(1, int(round(src_h * fit_ratio)))

        resample = Image.Resampling.BICUBIC
        disp_img = decoded_img.convert("RGB").resize((disp_w, disp_h), resample=resample)
        return self.ImageTk.PhotoImage(disp_img)

    def _update_card(self, res: OptimizationResult) -> None:
        # Backend-compat alias: single-view mode only re-renders when the
        # result belongs to the currently selected tab.
        if res.format_id != self.active_format_id:
            return
        self._render_single_result(res)

    def _render_single_result(self, res: OptimizationResult) -> None:
        widgets = self._single_widgets
        if not widgets:
            return

        spec = next((s for s in self.formats if s.id == res.format_id), None)
        badge_text = ""
        if spec is not None:
            badge_text = (
                "1D Binary Search"
                if spec.is_single_parameter
                else f"{len(spec.parameters)}D Multi-Param"
            )
        widgets["name_lbl"].configure(text=res.format_name)
        widgets["badge_lbl"].configure(text=badge_text)

        if res.decoded_image is not None:
            photo = self._render_preview_photo(res.decoded_image)
            self._photo_refs[res.format_id] = photo
            widgets["img_lbl"].configure(image=photo, text="")

        if res.within_budget:
            shortfall = res.target_bytes - res.actual_bytes
            size_text = (
                f"{format_byte_size(res.actual_bytes)} / {format_byte_size(res.target_bytes)} "
                f"({format_shortfall(shortfall)} short of target)"
            )
            size_color = (
                self.ACCENT_GREEN
                if shortfall <= max(4, int(res.target_bytes * 0.15))
                else self.ACCENT_AMBER
            )
        else:
            size_text = (
                f"{format_byte_size(res.actual_bytes)} (Min floor > {format_byte_size(res.target_bytes)})"
            )
            size_color = self.ACCENT_RED

        widgets["size_lbl"].configure(text=size_text, fg=size_color)

        sw, sh = res.scaled_size
        widgets["metrics_lbl"].configure(
            text=(
                f"Res: {sw}×{sh} px  |  PSNR: {res.psnr_db:.2f} dB  |  "
                f"{res.evaluations} evals ({res.elapsed_ms:.0f}ms)"
            )
        )
        widgets["params_lbl"].configure(
            text=f"Params: {res.format_params_summary()}"
        )

    def _refresh_all_previews(self) -> None:
        res = self._latest_results.get(self.active_format_id or "")
        if res is not None:
            self._render_single_result(res)

    def _open_inspector_modal(self, format_id: str) -> None:
        """Open a side-by-side Original vs Compressed zoom inspector dialog."""
        res = self._latest_results.get(format_id)
        if res is None or res.decoded_image is None:
            return

        tk = self.tk
        from tkinter import filedialog

        modal = tk.Toplevel(self.root)
        modal.title(f"Inspect: {res.format_name} vs Original")
        modal.geometry("920x620")
        modal.configure(bg=self.BG_MAIN)

        hdr = tk.Frame(modal, bg=self.BG_PANEL, padx=14, pady=10)
        hdr.pack(side=tk.TOP, fill=tk.X)

        tk.Label(
            hdr,
            text=(
                f"{res.format_name}  —  {format_byte_size(res.actual_bytes)} / {format_byte_size(res.target_bytes)}  "
                f"|  Scaled: {res.scaled_size[0]}×{res.scaled_size[1]}  "
                f"|  PSNR: {res.psnr_db:.2f} dB  |  {res.format_params_summary()}"
            ),
            font=("Helvetica", 10, "bold"),
            fg=self.FG_PRIMARY,
            bg=self.BG_PANEL,
        ).pack(side=tk.LEFT)

        def _save_encoded() -> None:
            ext = next(
                (f.extension for f in self.formats if f.id == format_id), ".bin"
            )
            out_path = filedialog.asksaveasfilename(
                parent=modal,
                title=f"Save {res.format_name} Output",
                defaultextension=ext,
                initialfile=f"optimized_{format_id}_{res.actual_bytes}B{ext}",
            )
            if out_path:
                with open(out_path, "wb") as f:
                    f.write(res.encoded_bytes)

        tk.Button(
            hdr,
            text="Export Compressed File...",
            command=_save_encoded,
            bg="#0284c7",
            fg=self.FG_PRIMARY,
            relief=tk.FLAT,
            padx=10,
            pady=4,
            font=("Helvetica", 9, "bold"),
        ).pack(side=tk.RIGHT)

        body = tk.Frame(modal, bg=self.BG_MAIN, padx=12, pady=12)
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        modal.update_idletasks()

        # Fit each pane's image to half the available modal width (minus padding).
        body_w = max(300, body.winfo_width() - 20)
        pane_w = max(200, int(body_w / 2) - 20)
        src_w, src_h = self.source_image.size
        scale_factor = min(pane_w / max(1, src_w), pane_w / max(1, src_h))
        disp_size = (
            max(1, int(round(src_w * scale_factor))),
            max(1, int(round(src_h * scale_factor))),
        )

        orig_disp = self.source_image.convert("RGB").resize(
            disp_size, resample=Image.Resampling.LANCZOS
        )
        comp_disp = res.decoded_image.convert("RGB").resize(
            disp_size,
            resample=Image.Resampling.BICUBIC,
        )

        orig_photo = self.ImageTk.PhotoImage(orig_disp)
        comp_photo = self.ImageTk.PhotoImage(comp_disp)
        modal._photos = (orig_photo, comp_photo)  # type: ignore[attr-defined]

        for col_idx, (title, photo) in enumerate(
            [
                (
                    f"Original ({src_w}×{src_h}, {format_byte_size(self.input_file_size)})",
                    orig_photo,
                ),
                (
                    f"{res.format_name} ({res.scaled_size[0]}×{res.scaled_size[1]}, {format_byte_size(res.actual_bytes)})",
                    comp_photo,
                ),
            ]
        ):
            pane = tk.Frame(body, bg=self.BG_PANEL, padx=10, pady=10)
            pane.grid(row=0, column=col_idx, padx=8, pady=4, sticky="nsew")
            body.grid_columnconfigure(col_idx, weight=1)
            tk.Label(
                pane,
                text=title,
                font=("Helvetica", 10, "bold"),
                fg=self.FG_PRIMARY,
                bg=self.BG_PANEL,
            ).pack(side=tk.TOP, pady=(0, 8))
            tk.Label(pane, image=photo, bg="#090d16").pack(
                side=tk.TOP, expand=True
            )

    def _on_close(self) -> None:
        self.parallel_optimizer.shutdown()
        self.root.destroy()


# ============================================================================
# CLI Benchmark & Entry Point
# ============================================================================


def run_cli_benchmark(
    image_path: Optional[str],
    budget_bytes: int,
    objective: ObjectiveMode = "closest_size",
    executor_mode: Literal["thread", "process"] = "thread",
) -> List[OptimizationResult]:
    """
    Run parallel optimization across all registered formats from the command line
    and print a formatted comparison table.
    """
    img, input_size, label = load_image_and_size(image_path)
    target = max(1, min(input_size, budget_bytes))
    print(f"Source Image : {label} ({img.width}x{img.height} px, {input_size:,} bytes)")
    print(f"Target Budget: {target:,} bytes ({target / 1024.0:.2f} KB) | Objective: {objective}")
    print("-" * 104)

    cfg = OptimizerConfig(objective_mode=objective)
    opt = ParallelFormatOptimizer(executor_mode=executor_mode, config=cfg)
    t0 = time.perf_counter()
    results = opt.optimize_all(img, target_bytes=target)
    total_ms = (time.perf_counter() - t0) * 1000.0
    opt.shutdown()

    header = (
        f"{'Format':<18} | {'Size (B)':>9} | {'Util %':>7} | "
        f"{'Scaled Res':<11} | {'PSNR (dB)':>9} | {'Evals':>5} | {'Parameters'}"
    )
    print(header)
    print("-" * 104)
    for r in sorted(results, key=lambda x: x.psnr_db, reverse=True):
        res_str = f"{r.scaled_size[0]}x{r.scaled_size[1]}"
        print(
            f"{r.format_name:<18} | {r.actual_bytes:>9,} | {r.utilization_pct:>6.1f}% | "
            f"{res_str:<11} | {r.psnr_db:>9.2f} | {r.evaluations:>5} | {r.format_params_summary()}"
        )
    print("-" * 104)
    print(f"Completed parallel optimization across {len(results)} codecs in {total_ms:.1f} ms.")
    return results


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Format Compare Visual — Interactive Tkinter & Parallel Codec Optimizer"
    )
    parser.add_argument(
        "image_path",
        nargs="?",
        default=None,
        help="Optional path to an input image file (uses a synthetic test image if omitted).",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run headless parallel optimization benchmark and print table to stdout.",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=2560,
        help="Target file size in bytes for --benchmark mode (default: 2560 bytes / 2.5 KB).",
    )
    parser.add_argument(
        "--objective",
        choices=["closest_size", "perceptual_frontier"],
        default="closest_size",
        help="Optimization objective mode.",
    )
    parser.add_argument(
        "--executor",
        choices=["thread", "process"],
        default="thread",
        help="Parallel worker pool backend ('thread' or 'process').",
    )
    args = parser.parse_args(argv)

    if args.benchmark:
        run_cli_benchmark(
            args.image_path,
            budget_bytes=args.budget,
            objective=args.objective,
            executor_mode=args.executor,
        )
        return 0

    import tkinter as tk

    root = tk.Tk()
    _app = FormatCompareApp(root, initial_image_path=args.image_path)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())