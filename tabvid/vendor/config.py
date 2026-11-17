"""Baseline detection constants. Values preserve the accepted screenshot pipeline.

LINE_THR deliberately includes faint staff lines; lowering it to 215 caused clipping.
Fixed morphology kernels and small safety pads remain local to their operations.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class ProcessingConfig:
    """Baseline screenshot heuristics. Change values only with golden regression evidence."""
    whole_page_aspect: float = 0.5
    minimum_row_height: int = 30
    annotation_reach_fraction: float = 0.12
    crop_line_fraction: float = 0.1
    ink_threshold: int = 140
    detached_header_gap_fraction: float = 0.06
    bold_header_erosion_fraction: float = 0.25
    bracket_height_fraction: float = 0.07
    tab_bottom_spacing: int = 3
    system_bottom_fraction: float = 0.05
    system_max_top_fraction: float = 0.9
    tab_top_max_spacing: int = 9
    system_top_gap_fraction: float = 0.105
    tab_top_gap_spacing: float = 1.1
    staff_cluster_gap: float = 2.5
    min_image_width: int = 300
    min_image_height: int = 60
    barline_ink_threshold: int = 110
    remaining_barline_fraction: float = 0.5
    cursor_column_fraction: float = 0.3
    cursor_fringe_pixels: int = 2
    cursor_white_percentile: float = 90
    cursor_black_percentile: float = 0  # Even a one-pixel visible stroke must survive.
    cursor_ink_point: float = 120
    cursor_paper_point: float = 225
    cursor_min_contrast: float = 25
    card_edge_rows: int = 2
    card_rule_coverage: float = .65
    card_rule_staff_clearance: float = 2.5
    red_hue_end: int = 15
    yellow_hue_end: int = 40
    red_hue_start: int = 165
    saturation_threshold: int = 110
    color_value_threshold: int = 70
    chroma_fringe_min_extent: int = 3  # JPEG fringes are typically 1-2 raster pixels thick.
    chroma_fringe_ink_value: int = 80
    chroma_fringe_gray_value: int = 180
    chroma_fringe_gray_saturation: int = 100
    long_overlay_fraction: float = 0.45
    vertical_overlay_width: float = 0.06
    horizontal_overlay_height: float = 0.12
    green_hue_start: int = 45
    green_hue_end: int = 85
    blue_hue_start: int = 100
    blue_hue_end: int = 135
    teal_value_threshold: int = 60
    teal_saturation_threshold: int = 16
    teal_hue_start: int = 66
    teal_hue_end: int = 100
    light_text_threshold: int = 235
    text_vertical_min_height: int = 25
    text_gamma: float = 2.2
    text_vertical_fraction: float = 0.08
    broad_gray_width_divisor: int = 200
    broad_gray_low: int = 190
    broad_gray_high: int = 252
    gray_box_min_height: int = 14
    gray_box_low: int = 90
    gray_box_high: int = 225
    gray_box_min_width_fraction: float = 0.02
    gray_box_min_aspect: float = 1.8
    gray_box_max_width_fraction: float = 0.2
    thumbnail_min_width: int = 50
    thumbnail_width_fraction: float = 0.045
    thumbnail_max_width_fraction: float = 0.2
    thumbnail_max_height_fraction: float = 0.3
    thumbnail_edge_coverage: float = 0.8
    edge_speck_fraction: float = 0.06
    background_sample_width: int = 450
    edge_speck_area: int = 150
    white_point: float = 235.0
    black_point: float = 60.0


CFG = ProcessingConfig()

LINE_THR = 238



def validate_layout(args):
    """Reject impossible page layouts before producing misleading PDFs."""
    import math
    if args.page not in ("letter", "a4"):
        raise ValueError("page must be letter or a4")
    if args.rows is not None and (not isinstance(args.rows, int) or args.rows <= 0):
        raise ValueError("rows must be a positive integer")
    width, height = (595.28, 841.89) if args.page == "a4" else (612, 792)
    if not isinstance(args.margin, (int, float)) or not math.isfinite(args.margin) or args.margin < 0 or args.margin * 144 >= min(width, height):
        raise ValueError("margin must be nonnegative and leave usable page space")
    if not isinstance(args.gap, (int, float)) or not math.isfinite(args.gap) or args.gap < 0:
        raise ValueError("gap must be finite and nonnegative")
    if args.rows and args.rows * args.gap + 24 >= height - args.margin * 144:
        raise ValueError("rows and gap leave no vertical space for music")
    if args.split not in ("auto", "always", "never"):
        raise ValueError("split must be auto, always or never")
