"""Grid warehouse layout: racks, zones, spawn/dock points."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Warehouse:
    width: int
    height: int
    racks: frozenset[tuple[int, int]]
    zone_size: tuple[int, int]
    spawn_points: tuple[tuple[int, int], ...]
    dropoff_points: tuple[tuple[int, int], ...]
    pickup_points: tuple[tuple[int, int], ...]

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def is_free(self, x: int, y: int) -> bool:
        return self.in_bounds(x, y) and (x, y) not in self.racks

    def zone_id(self, x: int, y: int) -> str:
        zw, zh = self.zone_size
        return f"Z_{x // zw}_{y // zh}"

    def zone_ids(self) -> list[str]:
        zw, zh = self.zone_size
        n_x = -(-self.width // zw)  # ceil
        n_y = -(-self.height // zh)
        return [f"Z_{gx}_{gy}" for gy in range(n_y) for gx in range(n_x)]

    def neighbors(self, x: int, y: int) -> list[tuple[int, int]]:
        candidates = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        return [c for c in candidates if self.is_free(*c)]


def generate_default_layout(
    width: int = 24,
    height: int = 16,
    zone_size: tuple[int, int] = (6, 4),
) -> Warehouse:
    """A fixed, deterministic small-warehouse floor plan.

    Three horizontal rack bands with vertical cross-aisle gaps, a dock line
    along the left edge (robot spawn / order drop-off), and pickup points on
    the aisle cells bordering each rack band (order origins).
    """
    rack_bands = [3, 7, 11]
    rack_height = 2
    aisle_gap_cols = {6, 12, 18}
    rack_x_range = range(2, width - 2)

    racks: set[tuple[int, int]] = set()
    pickup_points: list[tuple[int, int]] = []
    for band_y in rack_bands:
        for dy in range(rack_height):
            y = band_y + dy
            for x in rack_x_range:
                if x in aisle_gap_cols:
                    continue
                racks.add((x, y))
        # pickup points: aisle cell directly above the band
        pickup_points.extend((x, band_y - 1) for x in rack_x_range if x not in aisle_gap_cols)

    dock_ys = [1, 4, 7, 10, 13]
    dock_ys = [y for y in dock_ys if y < height]
    dropoff_points = tuple((0, y) for y in dock_ys)
    spawn_points = dropoff_points

    return Warehouse(
        width=width,
        height=height,
        racks=frozenset(racks),
        zone_size=zone_size,
        spawn_points=spawn_points,
        dropoff_points=dropoff_points,
        pickup_points=tuple(pickup_points),
    )
