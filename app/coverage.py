"""Satellite overpass footprint coverage check against a compartment map.

Semantics
---------
* A *check* is one overpass footprint evaluated against exactly one immutable
  ``map_versions`` row.  The footprint is the union of every uploaded polygon,
  so overlapping scenes are counted once.
* Areas are geodesic areas on the WGS84 ellipsoid (pyproj/Geod), never planar
  degree areas.  Every polygon is oriented before measuring so the sign of the
  geodesic area cannot depend on ring winding.
* For each compartment of the pinned version we store the total area, the
  covered/uncovered area and the uncovered fraction.  The exported GeoJSON is
  produced from the same difference geometry that produced the uncovered area,
  so the file and the numbers cannot disagree.
* A check is never invalidated by a map swap.  It keeps its original
  ``map_version_id`` and is reported as stale when that version is no longer
  active; only a new check (or a re-check) reads the newly active version.
* Invalid input is rejected before anything is written.  Results are inserted
  in one transaction, so a failed request never leaves a partial result set.
"""

from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Sequence

from pyproj import Geod
from shapely.geometry.base import BaseGeometry
from shapely.geometry.polygon import orient
from shapely.validation import make_valid

from . import db
from .parsing import (
    Compartment,
    Footprint,
    InputError,
    geometry_to_wkb,
    parse_footprint_geojson,
    sha256_bytes,
    wkb_to_geometry,
)

# Slivers below one square metre are numerical noise from clipping, not a
# missing swath; they are reported as fully covered and not exported.
AREA_EPSILON_M2 = 1.0

_GEOD = Geod(ellps="WGS84")


def _polygons(geometry: BaseGeometry):
    if geometry.is_empty:
        return
    if geometry.geom_type == "Polygon":
        yield geometry
    elif geometry.geom_type in {"MultiPolygon", "GeometryCollection"}:
        for part in geometry.geoms:
            yield from _polygons(part)


def geodesic_area_m2(geometry: BaseGeometry) -> float:
    """Area in square metres on the WGS84 ellipsoid.

    Rings are normalised (exterior counter-clockwise, holes clockwise) before
    the geodesic measurement, because ``Geod.geometry_area_perimeter`` derives
    the sign from winding order.
    """

    total = 0.0
    for polygon in _polygons(geometry):
        area, _ = _GEOD.geometry_area_perimeter(orient(polygon, 1.0))
        total += abs(area)
    return total


@dataclass
class CompartmentCoverage:
    seq: int
    code: str
    area_m2: float
    covered_area_m2: float
    uncovered_area_m2: float
    uncovered_ratio: float
    uncovered_geometry: BaseGeometry | None


@dataclass
class CoverageSummary:
    total_area_m2: float
    covered_area_m2: float
    uncovered_area_m2: float
    uncovered_ratio: float
    compartments: list[CompartmentCoverage] = field(default_factory=list)


def compute_coverage(
    compartments: Sequence[Compartment], footprint: BaseGeometry
) -> CoverageSummary:
    """Evaluate one footprint against an already-pinned compartment list.

    The aggregate is an area-weighted rollup over compartments.  If the map
    itself has overlapping compartments, the overlap is counted in each of
    them (per-compartment responsibility), so the aggregate total can exceed
    the union area of the map.  Each per-compartment ratio is unaffected.
    """

    rows: list[CompartmentCoverage] = []
    total_sum = uncovered_sum = 0.0
    for item in compartments:
        area = geodesic_area_m2(item.geometry)
        if area <= 0:
            continue
        difference = item.geometry.difference(footprint)
        if not difference.is_valid:
            # Clipping valid inputs can still emit a self-touching ring at a
            # tangency; repairing only the *difference* keeps the footprint
            # untouched and the area meaningful.
            difference = make_valid(difference)
        uncovered = geodesic_area_m2(difference)
        if uncovered < AREA_EPSILON_M2:
            uncovered = 0.0
            difference = None
        elif uncovered > area:  # numeric guard; cannot exceed the compartment
            uncovered = area
        covered = max(area - uncovered, 0.0)
        ratio = min(max(uncovered / area, 0.0), 1.0) if area > 0 else 0.0
        rows.append(
            CompartmentCoverage(
                seq=item.seq,
                code=item.code,
                area_m2=area,
                covered_area_m2=covered,
                uncovered_area_m2=uncovered,
                uncovered_ratio=ratio,
                uncovered_geometry=difference,
            )
        )
        total_sum += area
        uncovered_sum += uncovered
    covered_sum = max(total_sum - uncovered_sum, 0.0)
    ratio_sum = uncovered_sum / total_sum if total_sum > 0 else 0.0
    return CoverageSummary(
        total_area_m2=total_sum,
        covered_area_m2=covered_sum,
        uncovered_area_m2=uncovered_sum,
        uncovered_ratio=ratio_sum,
        compartments=rows,
    )


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------


@dataclass
class PreparedCoverage:
    filename: str
    sha256: str
    footprint: Footprint
    map_version_id: int
    map_filename: str
    compartments: list[Compartment]


def _load_active_snapshot(conn: sqlite3.Connection) -> tuple[int, str, list[Compartment]]:
    """Read the active map version and its compartments in one read snapshot."""

    active = conn.execute(
        "SELECT id, filename FROM map_versions WHERE is_active = 1"
    ).fetchone()
    if active is None:
        raise InputError("尚未上传林班底图，无法进行覆盖核查")
    map_version_id = int(active["id"])
    compartments = [
        Compartment(
            seq=int(item["seq"]),
            code=str(item["code"]),
            geometry=wkb_to_geometry(item["geom"]),
            minx=float(item["minx"]),
            miny=float(item["miny"]),
            maxx=float(item["maxx"]),
            maxy=float(item["maxy"]),
        )
        for item in conn.execute(
            "SELECT seq, code, geom, minx, miny, maxx, maxy "
            "FROM compartments WHERE map_version_id = ? ORDER BY seq",
            (map_version_id,),
        )
    ]
    return map_version_id, str(active["filename"]), compartments


def prepare_coverage(db_path: str, data: bytes, filename: str) -> PreparedCoverage:
    """Parse the footprint and snapshot the active map version.

    Parsing happens first: an invalid footprint fails before a map version is
    even read, so the request cannot show partial results.
    """

    footprint = parse_footprint_geojson(data, filename)
    with db.connection(db_path) as conn:
        with db.read_tx(conn):
            map_version_id, map_filename, compartments = _load_active_snapshot(conn)
    return PreparedCoverage(
        filename=filename,
        sha256=sha256_bytes(data),
        footprint=footprint,
        map_version_id=map_version_id,
        map_filename=map_filename,
        compartments=compartments,
    )


def prepare_recheck(db_path: str, check_id: int) -> PreparedCoverage:
    """Build a check from a stored footprint, pinned to the *current* map.

    Re-checking is the only way to move a footprint onto a newly activated map
    version; it does not modify the old check.
    """

    with db.connection(db_path) as conn:
        with db.read_tx(conn):
            check = conn.execute(
                "SELECT filename, sha256, feature_count, footprint_geom "
                "FROM coverage_checks WHERE id = ?",
                (check_id,),
            ).fetchone()
            if check is None:
                raise InputError("覆盖核查记录不存在")
            footprint_geom = wkb_to_geometry(check["footprint_geom"])
            map_version_id, map_filename, compartments = _load_active_snapshot(conn)
    return PreparedCoverage(
        filename=f"{check['filename']}（重新检查）",
        sha256=str(check["sha256"]),
        footprint=Footprint(
            geometry=footprint_geom, part_count=int(check["feature_count"])
        ),
        map_version_id=map_version_id,
        map_filename=map_filename,
        compartments=compartments,
    )


def execute_coverage(db_path: str, prepared: PreparedCoverage) -> tuple[int, bool]:
    """Compute and store a check atomically.

    Returns ``(check_id, stale)`` where ``stale`` means the active map version
    changed while the check was running: the result is still stored against —
    and labelled with — the version it actually read.
    """

    summary = compute_coverage(prepared.compartments, prepared.footprint.geometry)
    footprint = prepared.footprint.geometry
    footprint_area = geodesic_area_m2(footprint)
    now = db.utc_now()
    with db.connection(db_path) as conn:
        with db.write_tx(conn):
            active = conn.execute(
                "SELECT id FROM map_versions WHERE is_active = 1"
            ).fetchone()
            stale = active is None or int(active["id"]) != prepared.map_version_id
            cursor = conn.execute(
                """
                INSERT INTO coverage_checks
                    (filename, sha256, map_version_id, status, error, created_at,
                     finished_at, feature_count, footprint_geom, footprint_area_m2,
                     total_area_m2, covered_area_m2, uncovered_area_m2, uncovered_ratio)
                VALUES (?, ?, ?, 'succeeded', NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    prepared.filename,
                    prepared.sha256,
                    prepared.map_version_id,
                    now,
                    now,
                    prepared.footprint.part_count,
                    geometry_to_wkb(footprint),
                    footprint_area,
                    summary.total_area_m2,
                    summary.covered_area_m2,
                    summary.uncovered_area_m2,
                    summary.uncovered_ratio,
                ),
            )
            check_id = int(cursor.lastrowid)
            conn.executemany(
                """
                INSERT INTO coverage_results
                    (check_id, compartment_seq, code, area_m2, covered_area_m2,
                     uncovered_area_m2, uncovered_ratio, uncovered_geom)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        check_id,
                        row.seq,
                        row.code,
                        row.area_m2,
                        row.covered_area_m2,
                        row.uncovered_area_m2,
                        row.uncovered_ratio,
                        (
                            geometry_to_wkb(row.uncovered_geometry)
                            if row.uncovered_geometry is not None
                            else None
                        ),
                    )
                    for row in summary.compartments
                ],
            )
    return check_id, stale


def create_coverage(db_path: str, data: bytes, filename: str) -> tuple[int, bool]:
    return execute_coverage(db_path, prepare_coverage(db_path, data, filename))


# --------------------------------------------------------------------------
# Queries / export
# --------------------------------------------------------------------------


def get_check(conn: sqlite3.Connection, check_id: int) -> dict[str, Any] | None:
    row = conn.execute(
        """
        SELECT cc.*, m.filename AS map_filename, m.sha256 AS map_sha256,
               m.created_at AS map_created_at
        FROM coverage_checks cc
        JOIN map_versions m ON m.id = cc.map_version_id
        WHERE cc.id = ?
        """,
        (check_id,),
    ).fetchone()
    return dict(row) if row else None


def list_checks(conn: sqlite3.Connection, limit: int = 30) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT cc.*, m.filename AS map_filename
        FROM coverage_checks cc
        JOIN map_versions m ON m.id = cc.map_version_id
        ORDER BY cc.id DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return [dict(row) for row in rows]


def get_results(conn: sqlite3.Connection, check_id: int) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT compartment_seq, code, area_m2, covered_area_m2, uncovered_area_m2,
               uncovered_ratio, uncovered_geom
        FROM coverage_results
        WHERE check_id = ?
        ORDER BY compartment_seq
        """,
        (check_id,),
    ).fetchall()
    return [dict(row) for row in rows]


def is_stale(check: dict[str, Any], active_map_id: int | None) -> bool:
    return active_map_id is None or int(check["map_version_id"]) != active_map_id


def _round_coord(value: float) -> float:
    return round(value, 9)


def _ring_coords(ring) -> list[list[float]]:
    return [[_round_coord(x), _round_coord(y)] for x, y in ring.coords]


def _polygon_doc(polygon) -> dict[str, Any]:
    return {
        "type": "Polygon",
        "coordinates": [_ring_coords(polygon.exterior)]
        + [_ring_coords(interior) for interior in polygon.interiors],
    }


def _geometry_doc(geometry: BaseGeometry) -> dict[str, Any]:
    # Clipping can leave collapsed line/point pieces in a GeometryCollection;
    # export only the polygonal parts, since this file drives re-flight.
    polygons = list(_polygons(geometry))
    if len(polygons) == 1:
        return _polygon_doc(polygons[0])
    return {
        "type": "MultiPolygon",
        "coordinates": [_polygon_doc(polygon)["coordinates"] for polygon in polygons],
    }


def uncovered_geojson(check: dict[str, Any], results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """FeatureCollection of the uncovered part of every compartment.

    Compartments whose uncovered area is zero are omitted: there is nothing to
    re-fly there.
    """

    features = []
    for row in results:
        if not row["uncovered_geom"]:
            continue
        if row["uncovered_area_m2"] <= AREA_EPSILON_M2:
            continue
        geometry = wkb_to_geometry(row["uncovered_geom"])
        features.append(
            {
                "type": "Feature",
                "id": f"{check['id']}-{row['code']}",
                "properties": {
                    "check_id": int(check["id"]),
                    "compartment_code": row["code"],
                    "map_version_id": int(check["map_version_id"]),
                    "compartment_area_m2": round(row["area_m2"], 3),
                    "covered_area_m2": round(row["covered_area_m2"], 3),
                    "uncovered_area_m2": round(row["uncovered_area_m2"], 3),
                    "uncovered_ratio": round(row["uncovered_ratio"], 6),
                },
                "geometry": _geometry_doc(geometry),
            }
        )
    return {
        "type": "FeatureCollection",
        "name": f"uncovered_check_{check['id']}_map_v{check['map_version_id']}",
        "features": features,
    }


def ratio_to_percent(value: float | None) -> float:
    if value is None or not math.isfinite(value):
        return 0.0
    return round(value * 100.0, 4)
