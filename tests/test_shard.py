"""
test_shard.py
=============
Core pytest suite for metatron_shard.py (tested in isolation; TinyMetatron
never imports this module — contract §R9).

Covers the contract fixes (IMPLEMENTATION_CONTRACT.md §1):
  * diameter_reduction(base) for hexahedron vs dodecahedron is >= 0.333
    (5 vs 3 -> 0.4).
  * compute_diameter honours the topology argument (does NOT always use
    self.topology).
  * _balance_load recomputes ALL shard boundaries from new per-shard token
    counts — no overlapping ranges, no gaps.
  * bytes_per_token == 4 (float32) and comm events carry it.
  * ICOSA_DISTRIBUTED exists in the Strategy enum.
  * __main__ smoke runs clean (ASCII OK/FAIL — UTF-8 safe).
"""

from __future__ import annotations

import subprocess
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest

from config import CONFIG
from metatron_shard import (
    MetatronShardPlanner,
    Shard,
    ShardSchedule,
    CommEvent,
    Phase,
    Strategy,
    BYTES_PER_TOKEN,
    DIAMETERS,
)


# ── bytes_per_token == 4 ─────────────────────────────────────────────────────

def test_bytes_per_token_is_four():
    """BYTES_PER_TOKEN must be 4 (float32)."""
    assert BYTES_PER_TOKEN == 4


def test_comm_events_carry_bytes_per_token():
    """Every generated CommEvent must carry bytes_per_token == 4."""
    planner = MetatronShardPlanner(topology='dodecahedron', n_devices=4)
    schedule = planner.plan(seq_len=256)
    assert len(schedule.comm_events) > 0
    for ev in schedule.comm_events:
        assert ev.bytes_per_token == 4


# ── Diameter table ───────────────────────────────────────────────────────────

def test_diameter_table_correct():
    """Diameter table per contract §1: tetra=1, octa=2, cube=3, icosa=3,
    dodeca=5."""
    assert DIAMETERS['tetrahedron'] == 1
    assert DIAMETERS['octahedron'] == 2
    assert DIAMETERS['hexahedron'] == 3
    assert DIAMETERS['icosahedron'] == 3
    assert DIAMETERS['dodecahedron'] == 5


def test_compute_diameter_honours_topology_arg():
    """compute_diameter(topology=X) must use X, not self.topology."""
    planner = MetatronShardPlanner(topology='hexahedron', n_devices=4)
    # self.topology is hexahedron -> 3
    assert planner.compute_diameter() == 3
    # Explicit arg overrides self.topology.
    assert planner.compute_diameter(topology='dodecahedron') == 5
    assert planner.compute_diameter(topology='tetrahedron') == 1
    assert planner.compute_diameter(topology='octahedron') == 2
    assert planner.compute_diameter(topology='icosahedron') == 3


# ── diameter_reduction: hexa vs dodeca >= 0.333 ──────────────────────────────

def test_diameter_reduction_hexa_vs_dodeca():
    """diameter_reduction('dodecahedron') for a hexahedron planner must be
    >= 0.333 (base 5 vs this 3 -> 0.4)."""
    planner = MetatronShardPlanner(topology='hexahedron', n_devices=4)
    reduction = planner.diameter_reduction('dodecahedron')
    assert reduction >= 0.333, f"reduction={reduction} < 0.333"
    # Exact value check: (5-3)/5 = 0.4
    assert reduction == pytest.approx(0.4, abs=1e-6)


def test_diameter_reduction_uses_separate_base_planner():
    """diameter_reduction must instantiate a SEPARATE planner for the base
    topology — i.e. it must NOT use self.topology for both. We verify by
    using a base topology whose diameter differs from self.topology's."""
    planner = MetatronShardPlanner(topology='icosahedron', n_devices=4)
    # icosa (3) vs dodeca (5) -> (5-3)/5 = 0.4
    assert planner.diameter_reduction('dodecahedron') == pytest.approx(0.4, abs=1e-6)
    # icosa (3) vs tetra (1) -> (1-3)/1 = -2.0 (negative: this topology is worse)
    # The function should still compute it correctly (not clamp to 0 unless
    # base_d <= 0).
    assert planner.diameter_reduction('tetrahedron') == pytest.approx(-2.0, abs=1e-6)


# ── ICOSA_DISTRIBUTED exists ─────────────────────────────────────────────────

def test_icosa_distributed_strategy_exists():
    """Strategy enum must include ICOSA_DISTRIBUTED (renamed from
    ICOSADISTRIBUTED)."""
    assert hasattr(Strategy, 'ICOSA_DISTRIBUTED')
    assert hasattr(Strategy, 'DODECA_DISTRIBUTED')
    assert hasattr(Strategy, 'HEXA_DISTRIBUTED')
    assert hasattr(Strategy, 'OCTA_DISTRIBUTED')
    assert hasattr(Strategy, 'TETRA_DISTRIBUTED')
    # The OLD name must NOT exist.
    assert not hasattr(Strategy, 'ICOSADISTRIBUTED')


def test_strategy_name_mapping_includes_icosa():
    """_strategy_name('icosahedron') must return 'ICOSA_DISTRIBUTED'."""
    assert MetatronShardPlanner._strategy_name('icosahedron') == 'ICOSA_DISTRIBUTED'
    assert MetatronShardPlanner._strategy_name('dodecahedron') == 'DODECA_DISTRIBUTED'


def test_effective_shards_for_icosa():
    """ICOSA_DISTRIBUTED yields 12 effective shards (icosahedron: 12 vertices)."""
    planner = MetatronShardPlanner(topology='icosahedron', n_devices=4)
    assert planner._effective_shards(Strategy.ICOSA_DISTRIBUTED) == 12


# ── _balance_load: no overlapping ranges, no gaps ────────────────────────────

def test_balance_load_no_overlaps_no_gaps():
    """After _balance_load, shard token_ranges must form an exact partition of
    [0, total_tokens): no overlaps, no gaps."""
    planner = MetatronShardPlanner(topology='dodecahedron', n_devices=4)
    seq_len = 1024
    schedule = planner.plan(seq_len)
    shards = schedule.shards

    # Telemetry: vary load so rebalancing is non-trivial.
    n = len(shards)
    telemetry = {
        shards[i].shard_id: {'tokens_per_sec': 100.0 * (i + 1)}
        for i in range(n)
    }
    planner.total_tokens = seq_len
    rebalanced = planner._balance_load(shards, telemetry)

    # Collect (start, end) for each shard, skip empty ranges.
    ranges = [(s.token_range.start, s.token_range.stop) for s in rebalanced]
    # Non-overlapping: sorted starts must be >= previous stop.
    sorted_ranges = sorted(ranges)
    for i in range(1, len(sorted_ranges)):
        assert sorted_ranges[i][0] >= sorted_ranges[i - 1][1], (
            f"overlapping ranges: {sorted_ranges}"
        )
    # No gaps: union covers [0, seq_len).
    covered = sum(end - start for start, end in ranges)
    assert covered == seq_len, (
        f"coverage {covered} != seq_len {seq_len}; ranges={ranges}"
    )
    # The first shard starts at 0 and the last ends at seq_len.
    starts = [r[0] for r in ranges]
    stops = [r[1] for r in ranges]
    assert min(starts) == 0
    assert max(stops) == seq_len


def test_balance_load_respects_total_tokens_zero():
    """With total_tokens=0, balance produces empty ranges (no crash)."""
    planner = MetatronShardPlanner(topology='tetrahedron', n_devices=1)
    planner.total_tokens = 0
    shards = [Shard(shard_id=i, device_id=0, token_range=range(0, 0),
                    poly_face=(0,), phase=Phase.RADIAL) for i in range(4)]
    telemetry = {i: {'tokens_per_sec': 10.0} for i in range(4)}
    rebalanced = planner._balance_load(shards, telemetry)
    for s in rebalanced:
        assert s.token_range.start == s.token_range.stop


def test_total_tokens_init_to_zero():
    """__init__ must initialise total_tokens to 0 (no stale state)."""
    planner = MetatronShardPlanner(topology='dodecahedron', n_devices=2)
    assert planner.total_tokens == 0


# ── plan(): phases present ───────────────────────────────────────────────────

def test_plan_has_three_phases():
    """plan() must produce comm events in all three phases (radial, chord,
    ring) — Patent Claim 27."""
    planner = MetatronShardPlanner(topology='dodecahedron', n_devices=4)
    schedule = planner.plan(seq_len=1024)
    phases_present = {ev.phase for ev in schedule.comm_events}
    assert Phase.RADIAL in phases_present
    assert Phase.CHORD in phases_present
    assert Phase.RING in phases_present


def test_plan_shards_cover_seq_len():
    """Shard token_ranges from plan() must partition [0, seq_len)."""
    seq_len = 1024
    planner = MetatronShardPlanner(topology='dodecahedron', n_devices=4)
    schedule = planner.plan(seq_len)
    ranges = [(s.token_range.start, s.token_range.stop) for s in schedule.shards]
    # No overlaps.
    sorted_ranges = sorted(ranges)
    for i in range(1, len(sorted_ranges)):
        assert sorted_ranges[i][0] >= sorted_ranges[i - 1][1]
    # Full coverage.
    covered = sum(end - start for start, end in ranges)
    assert covered == seq_len
    assert min(r[0] for r in ranges) == 0
    assert max(r[1] for r in ranges) == seq_len


def test_plan_total_tokens_field():
    """ShardSchedule.total_tokens must equal the requested seq_len."""
    planner = MetatronShardPlanner(topology='dodecahedron', n_devices=4)
    schedule = planner.plan(seq_len=512)
    assert schedule.total_tokens == 512


# ── _est_latency uses total bytes (token_count * 4) ──────────────────────────

def test_est_latency_uses_total_bytes():
    """_est_latency must compute from total bytes (token_count * 4 * 8 bits /
    bandwidth)."""
    planner = MetatronShardPlanner(topology='dodecahedron', n_devices=1,
                                   bandwidth_bps=1e9)
    # 1000 tokens * 4 bytes = 4000 bytes = 32000 bits; /1e9 = 3.2e-5 s =
    # 0.032 ms.
    latency = planner._est_latency(4000)
    assert latency == pytest.approx(0.032, abs=1e-6)


def test_zero_bytes_zero_latency():
    """Empty payload -> zero latency (no division issues)."""
    planner = MetatronShardPlanner(topology='dodecahedron', n_devices=1)
    assert planner._est_latency(0) == 0.0


# ── __main__ smoke runs clean ───────────────────────────────────────────────

def test_main_smoke_runs_clean():
    """`python metatron_shard.py` must exit 0 with no traceback (ASCII
    OK/FAIL output — UTF-8 safe)."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run(
        [sys.executable, "metatron_shard.py"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"metatron_shard.py exited {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "Traceback" not in result.stderr
    # The smoke output must include the OK marker for the reduction check.
    assert "OK" in result.stdout


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    pytest.main([__file__, "-q"])