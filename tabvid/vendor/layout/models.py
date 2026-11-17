"""Pixel geometry, without interpreting musical notes or repeat semantics."""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Staff:
    lines: tuple[float, ...]
    spacing: float


@dataclass(frozen=True)
class Barline:
    x: int
    left: int
    right: int
    confidence: float


@dataclass(frozen=True)
class Measure:
    left: int
    right: int
    left_cut: bool
    right_cut: bool


@dataclass
class SystemStructure:
    staves: list[Staff] = field(default_factory=list)
    barlines: list[Barline] = field(default_factory=list)
    measures: list[Measure] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
