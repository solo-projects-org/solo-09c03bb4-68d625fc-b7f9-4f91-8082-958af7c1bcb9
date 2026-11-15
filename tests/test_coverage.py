"""Tests for the overpass-footprint coverage check.

The properties that matter to the remote-sensing analyst:

* overlapping footprint scenes are unioned, never double counted;
* areas are geodesic (m² on the WGS84 ellipsoid), not degree/planar areas;
* a check is pinned to the map version it read; a swap on the other web
  instance leaves the stored result labelled with the old version and marks it
  stale, and only a re-check reads the new version;
* an invalid footprint produces no partial result and does not disturb the
  hotspot attribution or its download.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from shapely.geometry import box

from app import coverage, db, service
from app.coverage import compute_coverage, geodesic_area_m2
from app.main import create_app
from app.parsing import Compartment, Footprint, InputError, parse_footprint_geojson

SAMPLES = Path(__file__).resolve().parent.parent / "samples"
M2_PER_DEG2_AT_25 = 111_320.0**2 * math.cos(math.radians(25.0))  # rough planar


def make_app(tmp_path):
    app = create_app(db_path=str(tmp_path / "forest.db"), start_worker=False)
    return TestClient(app), app


def geojson(polygons, crs=None):
    features = [
        {
            "type": "Feature",
            "properties": {"code": code, "name": code},
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [minx, miny],
                        [maxx, miny],
                        [maxx, maxy],
                        [minx, maxy],
                        [minx, miny],
                    ]
                ],
            },
        }
        for code, (minx, miny, maxx, maxy) in polygons.items()
    ]
    doc = {"type": "FeatureCollection", "features": features}
    if crs is not None:
        doc["crs"] = crs
    return json.dumps(doc).encode()


def footprint(*rects):
    data = geojson({f"F{i}": rect for i, rect in enumerate(rects)})
    return data


def compartment(code, rect, seq=0):
    return Compartment(
        seq=seq,
        code=code,
        geometry=box(*rect),
        minx=rect[0],
        miny=rect[1],
        maxx=rect[2],
        maxy=rect[3],
    )


def upload_map(client, name, payload):
    return client.post(
        "/maps",
        files={"file": (name, payload, "application/geo+json")},
        follow_redirects=False,
    )


def upload_footprint(client, name, payload):
    return client.post(
        "/coverage",
        files={"file": (name, payload, "application/geo+json")},
        follow_redirects=False,
    )


def csv_bytes(rows):
    lines = ["hotspot_no,lon,lat,observed_at"]
    lines += [",".join(map(str, row)) for row in rows]
    return ("\n".join(lines) + "\n").encode()


# --------------------------------------------------------------------------
# Geometry / area
# --------------------------------------------------------------------------


def test_area_is_geodesic_not_planar():
    square = box(100.0, 25.0, 100.1, 25.1)
    geodesic = geodesic_area_m2(square)
    planar_degrees = square.area  # 0.01 "square degrees"
    naive_m2 = planar_degrees * M2_PER_DEG2_AT_25
    # The naive degree*metres conversion is several percent off; the geodesic
    # value must not match it.
    assert abs(geodesic - naive_m2) / geodesic > 0.002
    # 0.01° x 0.01° at 25°N ~= 111.78 km² (known value).
    assert 110_000_000 < geodesic < 113_000_000


def test_area_does_not_depend_on_ring_winding():
    from shapely.geometry import Polygon

    outer = [(100.0, 25.0), (100.2, 25.0), (100.2, 25.2), (100.0, 25.2), (100.0, 25.0)]
    reversed_outer = list(reversed(outer))
    hole = [(100.05, 25.05), (100.1, 25.05), (100.1, 25.1), (100.05, 25.1), (100.05, 25.05)]
    a = geodesic_area_m2(Polygon(outer, [hole]))
    b = geodesic_area_m2(Polygon(reversed_outer, [list(reversed(hole))]))
    assert a == pytest.approx(b, rel=1e-12)
    assert a > 0


def test_geodesic_area_of_multipolygon_is_sum_of_parts():
    from shapely.geometry import MultiPolygon

    one = box(100.0, 25.0, 100.1, 25.1)
    two = box(100.5, 25.5, 100.6, 25.6)
    assert geodesic_area_m2(MultiPolygon([one, two])) == pytest.approx(
        geodesic_area_m2(one) + geodesic_area_m2(two), rel=1e-12
    )


def test_overlapping_footprint_scenes_are_counted_once():
    single = parse_footprint_geojson(footprint((100.0, 25.0, 100.2, 25.2)), "a.geojson")
    overlapping = parse_footprint_geojson(
        footprint((100.0, 25.0, 100.15, 25.2), (100.05, 25.0, 100.2, 25.2)), "b.geojson"
    )
    assert overlapping.part_count == 2
    # Union inserts vertices along shared edges; geodesic edge integration is
    # vertex-dependent at the sub-metre level, so allow 1e-6 relative.
    assert geodesic_area_m2(overlapping.geometry) == pytest.approx(
        geodesic_area_m2(single.geometry), rel=1e-6
    )
    # A duplicated scene must not inflate the area either.
    duplicate = parse_footprint_geojson(
        footprint((100.0, 25.0, 100.2, 25.2), (100.0, 25.0, 100.2, 25.2)), "c.geojson"
    )
    assert geodesic_area_m2(duplicate.geometry) == pytest.approx(
        geodesic_area_m2(single.geometry), rel=1e-9
    )


def test_compute_coverage_uncovered_equals_total_minus_covered():
    compartments = [
        compartment("A", (100.0, 25.0, 100.1, 25.1), 0),
        compartment("B", (100.1, 25.0, 100.2, 25.1), 1),
    ]
    fp = parse_footprint_geojson(footprint((100.0, 25.0, 100.05, 25.1)), "f.geojson")
    summary = compute_coverage(compartments, fp.geometry)
    by_code = {row.code: row for row in summary.compartments}
    assert by_code["A"].uncovered_ratio == pytest.approx(0.5, rel=1e-6)
    assert by_code["B"].uncovered_ratio == pytest.approx(1.0, rel=1e-9)
    assert by_code["A"].covered_area_m2 + by_code["A"].uncovered_area_m2 == pytest.approx(
        by_code["A"].area_m2, rel=1e-9
    )
    assert summary.uncovered_area_m2 == pytest.approx(
        by_code["A"].uncovered_area_m2 + by_code["B"].uncovered_area_m2, rel=1e-9
    )
    assert summary.covered_area_m2 == pytest.approx(summary.total_area_m2 - summary.uncovered_area_m2, rel=1e-9)
    # Uncovered geometry area must agree with the reported number (geodesic).
    assert geodesic_area_m2(by_code["A"].uncovered_geometry) == pytest.approx(
        by_code["A"].uncovered_area_m2, rel=1e-9
    )


def test_sliver_below_one_square_metre_is_treated_as_covered():
    compartments = [compartment("A", (100.0, 25.0, 100.1, 25.1), 0)]
    # Footprint covers all but a ~0.5 m² strip at the eastern edge.
    almost = box(100.0, 25.0, 100.1 - 5e-10, 25.1)
    summary = compute_coverage(compartments, almost)
    assert summary.compartments[0].uncovered_area_m2 == 0.0
    assert summary.compartments[0].uncovered_geometry is None


# --------------------------------------------------------------------------
# Footprint validation
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "payload, message",
    [
        (b"{}", "Feature"),
        (b'{"type":"FeatureCollection","features":[]}', "没有要素"),
        (
            json.dumps(
                {
                    "type": "Feature",
                    "properties": {},
                    "geometry": {"type": "Point", "coordinates": [100, 25]},
                }
            ).encode(),
            "面",
        ),
        (
            json.dumps(
                {
                    "type": "Feature",
                    "properties": {},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [[100, 25], [100.1, 25.2], [100.1, 25], [100, 25.2], [100, 25]]
                        ],
                    },
                }
            ).encode(),
            "无效",
        ),
        (
            geojson({"A": (100, 25, 100.1, 25.1)}, crs={"properties": {"name": "EPSG:3857"}}),
            "坐标系",
        ),
        (b'{"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": []}}', "空"),
    ],
)
def test_invalid_footprints_are_rejected(payload, message):
    with pytest.raises(InputError, match=message):
        parse_footprint_geojson(payload, "bad.geojson")


def test_crs84_declaration_is_accepted():
    fp = parse_footprint_geojson(
        geojson({"A": (100, 25, 100.1, 25.1)}, crs={"properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}}),
        "ok.geojson",
    )
    assert not fp.geometry.is_empty


def test_antimeridian_span_is_rejected():
    with pytest.raises(InputError, match="180"):
        parse_footprint_geojson(geojson({"A": (-179, 25, 179, 26)}), "wide.geojson")


# --------------------------------------------------------------------------
# API flow
# --------------------------------------------------------------------------


def test_coverage_api_flow_and_geojson_export(tmp_path):
    client, app = make_app(tmp_path)
    with client:
        upload_map(client, "v1.geojson", geojson({"A": (100.0, 25.0, 100.1, 25.1), "B": (100.1, 25.0, 100.2, 25.1)}))
        response = upload_footprint(client, "f.geojson", footprint((100.0, 25.0, 100.05, 25.1)))
        assert response.status_code == 303
        assert "/coverage/1" in response.headers["location"]

        payload = client.get("/api/coverage/1").json()
        assert payload["map_version_id"] == 1
        assert payload["stale"] is False
        assert payload["feature_count"] == 1
        by_code = {row["code"]: row for row in payload["compartments"]}
        assert by_code["A"]["uncovered_percent"] == pytest.approx(50.0, abs=0.01)
        assert by_code["B"]["uncovered_percent"] == pytest.approx(100.0, abs=1e-6)
        assert payload["uncovered_percent"] == pytest.approx(75.0, abs=0.01)

        page = client.get("/coverage/1")
        assert page.status_code == 200
        assert "未覆盖" in page.text and "50.00%" in page.text
        assert "已过期" not in page.text

        exported = client.get("/coverage/1/uncovered.geojson")
        assert exported.status_code == 200
        assert exported.headers["content-type"].startswith("application/geo+json")
        document = json.loads(exported.content)
        assert document["type"] == "FeatureCollection"
        assert {feature["properties"]["compartment_code"] for feature in document["features"]} == {"A", "B"}
        for feature in document["features"]:
            assert feature["properties"]["map_version_id"] == 1
        # Exported geometry reproduces the reported area (geodesic, not planar).
        from shapely.geometry import shape

        total_exported = sum(
            geodesic_area_m2(shape(feature["geometry"])) for feature in document["features"]
        )
        assert total_exported == pytest.approx(payload["uncovered_area_m2"], rel=1e-6)


def test_overlapping_scenes_produce_same_result_as_single_scene(tmp_path):
    client, app = make_app(tmp_path)
    with client:
        upload_map(client, "v1.geojson", geojson({"A": (100.0, 25.0, 100.2, 25.2)}))
        upload_footprint(client, "single.geojson", footprint((100.0, 25.0, 100.1, 25.2)))
        upload_footprint(
            client,
            "split.geojson",
            footprint((100.0, 25.0, 100.06, 25.2), (100.04, 25.0, 100.1, 25.2)),
        )
        single = client.get("/api/coverage/1").json()
        split = client.get("/api/coverage/2").json()
        assert split["uncovered_area_m2"] == pytest.approx(single["uncovered_area_m2"], rel=1e-6)
        assert split["covered_area_m2"] == pytest.approx(single["covered_area_m2"], rel=1e-6)
        # Union inserts extra vertices on shared edges; geodesic edge
        # integration is vertex-dependent at the sub-metre level.
        assert split["footprint_area_m2"] == pytest.approx(single["footprint_area_m2"], rel=1e-6)


def test_samples_footprint_coverage(tmp_path):
    client, app = make_app(tmp_path)
    with client:
        upload_map(
            client,
            "compartments_v1.geojson",
            (SAMPLES / "compartments_v1.geojson").read_bytes(),
        )
        response = upload_footprint(
            client, "footprint_v1.geojson", (SAMPLES / "footprint_v1.geojson").read_bytes()
        )
        assert response.status_code == 303
        payload = client.get("/api/coverage/1").json()
        assert payload["feature_count"] == 3
        by_code = {row["code"]: row for row in payload["compartments"]}
        assert set(by_code) == {"A-01", "B-02", "C-03", "D-04", "E-05"}
        # A-01 entirely inside the merged footprint; C-03 only partly.
        assert by_code["A-01"]["uncovered_percent"] == pytest.approx(0.0, abs=1e-6)
        assert by_code["C-03"]["uncovered_percent"] > 0
        for row in payload["compartments"]:
            assert row["uncovered_area_m2"] <= row["area_m2"] * (1 + 1e-9)
        assert 0 < payload["uncovered_ratio"] < 1


def test_invalid_footprint_leaves_no_partial_result_and_hotspots_work(tmp_path):
    client, app = make_app(tmp_path)
    with client:
        upload_map(client, "v1.geojson", geojson({"A": (100.0, 25.0, 100.1, 25.1)}))
        client.post(
            "/batches",
            files={
                "file": (
                    "h.csv",
                    csv_bytes([("H1", 100.05, 25.05, "2026-01-01T00:00:00Z")]),
                    "text/csv",
                )
            },
        )
        # Run the queued attribution deterministically.
        while True:
            claimed = service.claim_next_run(app.state.settings["db_path"])
            if claimed is None:
                break
            service.execute_run(app.state.settings["db_path"], claimed[0], attempt_token=claimed[1])
        before = client.get("/batches/1/download.csv").content

        bad = client.post(
            "/coverage",
            files={"file": ("bad.geojson", b'{"type":"FeatureCollection","features":[]}', "application/geo+json")},
            follow_redirects=False,
        )
        assert bad.status_code == 303
        assert "err=" in bad.headers["location"]
        with db.connection(app.state.settings["db_path"]) as conn:
            assert conn.execute("SELECT COUNT(*) AS n FROM coverage_checks").fetchone()["n"] == 0
            assert conn.execute("SELECT COUNT(*) AS n FROM coverage_results").fetchone()["n"] == 0

        # Existing hotspot attribution and download are unaffected.
        assert client.get("/api/batches/1").json()["matched"] == 1
        assert client.get("/batches/1/download.csv").content == before
        # And the coverage page still renders without a check.
        assert client.get("/coverage").status_code == 200


def test_coverage_requires_active_map(tmp_path):
    client, app = make_app(tmp_path)
    with client:
        response = upload_footprint(client, "f.geojson", footprint((100.0, 25.0, 100.1, 25.1)))
        assert response.status_code == 303
        assert "err=" in response.headers["location"]
        assert client.get("/coverage").status_code == 200


# --------------------------------------------------------------------------
# Map swap semantics
# --------------------------------------------------------------------------


def test_map_swapped_mid_check_keeps_old_label_and_flags_stale(tmp_path, monkeypatch):
    """Another instance swaps the map while the check is computing."""

    client_a, app = make_app(tmp_path)
    client_b = TestClient(create_app(db_path=app.state.settings["db_path"], start_worker=False))
    with client_a, client_b:
        upload_map(client_a, "v1.geojson", geojson({"A": (100.0, 25.0, 100.1, 25.1)}))
        original = coverage.compute_coverage
        swapped = {"done": False}

        def swap_during_check(compartments, footprint_geometry):
            summary = original(compartments, footprint_geometry)
            if not swapped["done"]:
                swapped["done"] = True
                # The other web instance publishes a v2 map mid-check.
                upload_map(
                    client_b,
                    "v2.geojson",
                    geojson({"B": (101.0, 26.0, 101.1, 26.1)}),
                )
            return summary

        monkeypatch.setattr(coverage, "compute_coverage", swap_during_check)
        response = upload_footprint(client_a, "f.geojson", footprint((100.0, 25.0, 100.05, 25.1)))
        assert response.status_code == 303
        assert "/coverage/1" in response.headers["location"]
        assert "msg=" in response.headers["location"]  # stale message present
        monkeypatch.setattr(coverage, "compute_coverage", original)

        payload = client_a.get("/api/coverage/1").json()
        # Result stays labelled with, and computed against, the version it read.
        assert payload["map_version_id"] == 1
        assert payload["map_filename"] == "v1.geojson"
        assert payload["active_map_version"] == 2
        assert payload["stale"] is True
        assert {row["code"] for row in payload["compartments"]} == {"A"}

        detail = client_a.get("/coverage/1")
        assert "已过期" in detail.text
        assert "v1" in detail.text and "v2" in detail.text

        exported = client_a.get("/coverage/1/uncovered.geojson")
        document = json.loads(exported.content)
        assert document["name"].endswith("map_v1")
        assert all(f["properties"]["map_version_id"] == 1 for f in document["features"])

        # A re-check reads the (new) active version and produces a fresh label.
        response = client_a.post("/coverage/1/recheck", follow_redirects=False)
        assert response.status_code == 303
        assert "/coverage/2" in response.headers["location"]
        fresh = client_a.get("/api/coverage/2").json()
        assert fresh["map_version_id"] == 2
        assert fresh["stale"] is False
        assert {row["code"] for row in fresh["compartments"]} == {"B"}
        assert fresh["uncovered_percent"] == pytest.approx(100.0, abs=1e-6)

        # The old check is immutable and still labelled v1.
        assert client_a.get("/api/coverage/1").json()["map_version_id"] == 1


def test_result_flagged_stale_after_later_swap_but_still_downloadable(tmp_path):
    client, app = make_app(tmp_path)
    with client:
        upload_map(client, "v1.geojson", geojson({"A": (100.0, 25.0, 100.1, 25.1)}))
        upload_footprint(client, "f.geojson", footprint((100.0, 25.0, 100.05, 25.1)))
        assert client.get("/api/coverage/1").json()["stale"] is False
        payload_before = client.get("/api/coverage/1").json()

        upload_map(client, "v2.geojson", geojson({"A": (100.0, 25.0, 100.1, 25.1), "Z": (102, 27, 102.1, 27.1)}))
        after = client.get("/api/coverage/1").json()
        assert after["stale"] is True
        assert after["map_version_id"] == 1
        assert after["uncovered_area_m2"] == payload_before["uncovered_area_m2"]

        listing = client.get("/coverage")
        assert "已过期" in listing.text
        exported = client.get("/coverage/1/uncovered.geojson")
        assert exported.status_code == 200
        assert json.loads(exported.content)["name"].endswith("map_v1")


def test_recheck_missing_check_is_404(tmp_path):
    client, app = make_app(tmp_path)
    with client:
        upload_map(client, "v1.geojson", geojson({"A": (100.0, 25.0, 100.1, 25.1)}))
        assert client.get("/coverage/999").status_code == 404
        assert client.get("/api/coverage/999").status_code == 404
        assert client.get("/coverage/999/uncovered.geojson").status_code == 404


def test_check_pins_map_even_if_swap_lands_between_prepare_and_execute(tmp_path):
    """Direct service-level check of the pinning contract."""

    client, app = make_app(tmp_path)
    db_path = app.state.settings["db_path"]
    with client:
        upload_map(client, "v1.geojson", geojson({"A": (100.0, 25.0, 100.1, 25.1)}))
        prepared = coverage.prepare_coverage(
            db_path, footprint((100.0, 25.0, 100.05, 25.1)), "f.geojson"
        )
        assert prepared.map_version_id == 1
        # Swap before execution starts: the prepared check must still use v1
        # and be flagged stale instead of silently reading v2.
        service.publish_map(
            db_path, geojson({"B": (101, 26, 101.1, 26.1)}), "v2.geojson"
        )
        check_id, stale = coverage.execute_coverage(db_path, prepared)
        assert stale is True
        with db.connection(db_path) as conn:
            check = coverage.get_check(conn, check_id)
            results = coverage.get_results(conn, check_id)
            active = service.get_active_map(conn)
        assert check["map_version_id"] == 1
        assert active.id == 2
        assert [row["code"] for row in results] == ["A"]
