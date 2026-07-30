"""
metatron_shard.py
=================
MetatronShard — distributed sharding planner for polyhedral topologies.
Assigns sequence shards to faces/edges of the polyhedral graph and
generates deterministic 3-phase communication schedules
(radial → chord → ring) with dynamic load balancing.

Matches TMT Patent Draft §7.2 / Claims 10–14:
    "planificador de sharding geométrico"
    "cronograma de comunicación por fases (radial→chord→ring)"
    "balanceador dinámico con telemetría de throughput/latencia"
    "DODECA_DISTRIBUTED: maximiza throughput bajo ancho de banda limitado"

NOTE (contract §R9): tested in isolation only; TinyMetatron never imports this.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum

import torch

from config import CONFIG

# ── Types ────────────────────────────────────────────────────────────────────
Phase = Enum('Phase', 'RADIAL CHORD RING')
Strategy = Enum('Strategy', 'TETRA_DISTRIBUTED HEXA_DISTRIBUTED '
                            'OCTA_DISTRIBUTED DODECA_DISTRIBUTED '
                            'ICOSA_DISTRIBUTED')

# Bytes per token for the communication model (float32 = 4 bytes).
# Centralised so radial / chord / ring all agree on the unit token weight.
BYTES_PER_TOKEN: int = 4

# Graph diameter (longest shortest path in hops) per polyhedral topology.
# tetra=1, octa=2, cube(hexa)=3, icosa=3, dodeca=5 (contract §1).
DIAMETERS: Dict[str, int] = {
    'tetrahedron':  1,
    'octahedron':   2,
    'hexahedron':   3,
    'icosahedron':  3,
    'dodecahedron': 5,
}


@dataclass
class Shard:
    """One shard — a sub-sequence mapped to a face/edge of the polyhedron."""
    shard_id: int
    device_id: int
    token_range: range          # [start, end) token indices
    poly_face: Tuple[int, ...]  # polyhedron vertex indices
    phase: Phase
    load: float = 0.0           # tokens/s measured
    latency_p95: float = 0.0


@dataclass
class CommEvent:
    """One collective communication event in the schedule."""
    phase: Phase
    src_shards: List[int]
    dst_shards: List[int]
    collective_type: str        # 'all-gather' | 'reduce-scatter' | 'all-reduce'
    bytes_per_token: int
    estimated_latency_ms: float


@dataclass
class ShardSchedule:
    """
    Deterministic communication schedule for a distributed run.
    Groups collectives into 3 phases: radial → chord → ring
    (Patent Claim 27: "pipeline de tres fases radial–chord–ring")
    """
    shards: List[Shard]
    comm_events: List[CommEvent] = field(default_factory=list)
    strategy: Strategy = Strategy.DODECA_DISTRIBUTED
    total_tokens: int = 0
    bandwidth_bps: float = 1e9     # 1 Gbps default
    n_devices: int = 1

    def phase_events(self, phase: Phase) -> List[CommEvent]:
        return [e for e in self.comm_events if e.phase == phase]

    def total_comm_latency_ms(self) -> float:
        return sum(e.estimated_latency_ms for e in self.comm_events)

    def print_summary(self):
        print(f"Strategy: {self.strategy.name}")
        print(f"Devices  : {self.n_devices}")
        print(f"Shards   : {len(self.shards)}")
        print(f"Comm evts: {len(self.comm_events)}")
        print(f"Comm latency: {self.total_comm_latency_ms():.2f} ms")


class MetatronShardPlanner:
    """
    Geometric sharding planner for polyhedral cluster topologies.

    Pipeline (Patent §7.2):
      1. Partition sequence into sub-sequences mapped to faces/edges of G
      2. Build 3-phase communication schedule (radial→chord→ring)
      3. Balance online with telemetry (tokens/s, latency, HBM/PCIe)
      4. Group collectives by phase

    Usage:
        planner = MetatronShardPlanner(topology='icosahedron', n_devices=4)
        schedule = planner.plan(seq_len=1024)
        schedule.print_summary()
    """

    # Approximate vertex counts per solid
    SOLID_VERTICES = {
        'tetrahedron':  4,
        'hexahedron':   8,
        'octahedron':   6,
        'dodecahedron': 20,
        'icosahedron':  12,
    }

    def __init__(self,
                 topology: Optional[str] = None,
                 n_devices: Optional[int] = None,
                 bandwidth_bps: Optional[float] = None):
        self.topology   = topology if topology is not None else CONFIG['shard_topology']
        self.n_devices  = max(1, n_devices if n_devices is not None
                               else CONFIG['shard_n_devices'])
        self.bandwidth  = (bandwidth_bps if bandwidth_bps is not None
                           else CONFIG['shard_bandwidth_bps'])
        self.V = self.SOLID_VERTICES[self.topology]
        # Total tokens covered by the current plan; updated in plan().
        # Initialised to 0 so _balance_load can never read stale state.
        self.total_tokens: int = 0

    def _shard_to_face(self, seq_len: int,
                       n_shards: int) -> List[Tuple[int, Tuple[int, ...]]]:
        """
        Map each shard to a face of the polyhedron.
        Returns: [(shard_id, face_vertex_tuple), ...]
        """
        from metatron_sparse_attention import POLYHEDRA
        poly = POLYHEDRA[self.topology]
        faces = poly['faces']
        n_faces = len(faces)

        shard_faces = []
        for sid in range(n_shards):
            face = faces[sid % n_faces]
            shard_faces.append((sid, face))
        return shard_faces

    def _radial_phase(self, shards: List[Shard]) -> List[CommEvent]:
        """
        RADIAL phase: each shard communicates with its radial parent.
        Patent: "fase radial — comunica al padre topológico"
        """
        events = []
        for i, shard in enumerate(shards):
            if i == 0:
                continue  # root shard has no parent
            parent = shards[(i - 1) % len(shards)]
            token_count = shard.token_range.stop - shard.token_range.start
            total_bytes = token_count * BYTES_PER_TOKEN
            ev = CommEvent(
                phase=Phase.RADIAL,
                src_shards=[shard.shard_id],
                dst_shards=[parent.shard_id],
                collective_type='all-gather',
                bytes_per_token=BYTES_PER_TOKEN,
                estimated_latency_ms=self._est_latency(total_bytes),
            )
            events.append(ev)
        return events

    def _chord_phase(self, shards: List[Shard]) -> List[CommEvent]:
        """
        CHORD phase: shards exchange data along chord (shortcut) connections.
        Patent: "fase chord — salto geométrico entre niveles no adyacentes"
        Uses the golden-ratio and √2 chord ratios from the attention mask.
        """
        events = []
        n = len(shards)
        # φ-chord hops: skip ~61.8% around the ring
        phi_hop = max(1, int(n * CONFIG['phi']))
        for i in range(n):
            target = (i + phi_hop) % n
            if target == i:
                continue
            ev = CommEvent(
                phase=Phase.CHORD,
                src_shards=[i],
                dst_shards=[target],
                collective_type='all-reduce',
                bytes_per_token=BYTES_PER_TOKEN,   # float32
                estimated_latency_ms=self._est_latency(BYTES_PER_TOKEN),
            )
            events.append(ev)
        return events

    def _ring_phase(self, shards: List[Shard]) -> List[CommEvent]:
        """
        RING phase: all shards synchronise in a ring barrier.
        Patent: "fase ring — sweep circular global"
        """
        events = []
        n = len(shards)
        for i in range(n):
            src = i
            dst = (i + 1) % n
            token_count = shards[src].token_range.stop - shards[src].token_range.start
            total_bytes = token_count * BYTES_PER_TOKEN
            ev = CommEvent(
                phase=Phase.RING,
                src_shards=[src],
                dst_shards=[dst],
                collective_type='reduce-scatter',
                bytes_per_token=BYTES_PER_TOKEN,
                estimated_latency_ms=self._est_latency(total_bytes),
            )
            events.append(ev)
        return events

    def _est_latency(self, total_bytes: int, n_hops: int = 1) -> float:
        """Estimate one-hop latency in ms from link bandwidth.

        ``total_bytes`` is the full payload size for the collective (NOT a
        per-token figure); radial/ring pass ``token_count * BYTES_PER_TOKEN``,
        chord passes a single token's ``BYTES_PER_TOKEN``.
        """
        if total_bytes <= 0:
            return 0.0
        bit_time_s = (total_bytes * 8) / self.bandwidth
        return bit_time_s * 1000 * n_hops

    def _balance_load(self, shards: List[Shard],
                     telemetry: Dict[int, Dict[str, float]]) -> List[Shard]:
        """
        Online load balancing based on telemetry (Claim 10, 13).
        Recomputes ALL shard boundaries from new per-shard token counts
        derived from the load ratios, so the tiling of [0, total_tokens)
        never develops overlaps or gaps.

        Per-shard token count is proportional to the measured tokens/s
        (slower shard → fewer tokens; faster shard → more tokens, balancing
        wall-clock time). Boundaries are then assigned cumulatively,
        guaranteeing a partition of the token range.

        Patent Claim 13: "SLA configurables por experto"
        """
        n = len(shards)
        if n == 0:
            return shards

        # ── Gather per-shard load telemetry ──────────────────────────
        loads: List[float] = []
        for shard in shards:
            telem = telemetry.get(shard.shard_id, {})
            load = float(telem.get('tokens_per_sec', 0.0))
            shard.load = load
            # floor to a tiny positive value so proportional weighting is finite
            loads.append(load if load > 0 else 1e-9)

        # ── Derive new per-shard token counts (throughput-proportional) ─
        # Faster shards (higher tokens/s) get a larger share so every shard
        # finishes its slice in roughly the same wall-clock time. A shard with
        # zero reported load keeps the tiny floor above (effectively no tokens)
        # rather than being handed the whole budget.
        total_load = sum(loads)
        if total_load <= 0:
            # All-zero telemetry → split evenly as a safe default.
            raw = [self.total_tokens / n] * n
        else:
            raw = [self.total_tokens * (l / total_load) for l in loads]
        counts = [int(math.floor(r)) for r in raw]

        # Distribute the rounding remainder to the largest fractional parts
        # so the counts sum exactly to self.total_tokens.
        remainder = self.total_tokens - sum(counts)
        if remainder > 0:
            fracs = sorted(range(n), key=lambda k: raw[k] - counts[k], reverse=True)
            for k in range(remainder):
                counts[fracs[k % n]] += 1
        elif remainder < 0:
            # Over-allocation: trim from the smallest fractional parts.
            fracs = sorted(range(n), key=lambda k: raw[k] - counts[k])
            for k in range(-remainder):
                if counts[fracs[k % n]] > 1:
                    counts[fracs[k % n]] -= 1

        # ── Recompute ALL boundaries cumulatively (no overlaps/gaps) ──
        cur = 0
        for shard, c in zip(shards, counts):
            start = cur
            end = min(cur + max(0, c), self.total_tokens)
            shard.token_range = range(start, end)
            cur = end
        # Final shard absorbs any leftover so the tiling is exact.
        last = shards[-1]
        if last.token_range.stop < self.total_tokens:
            last.token_range = range(last.token_range.start, self.total_tokens)
        return shards

    def plan(self, seq_len: int,
             telemetry: Optional[Dict[int, Dict[str, float]]] = None,
             strategy: Strategy = Strategy.DODECA_DISTRIBUTED) -> ShardSchedule:
        """
        Generate a complete shard + communication schedule.

        Args:
            seq_len:   total sequence length
            telemetry: optional per-shard telemetry for load balancing
                       {shard_id: {'tokens_per_sec': float, 'sla_target': float}}
            strategy:  distribution strategy (default DODECA_DISTRIBUTED)

        Returns: ShardSchedule
        """
        self.total_tokens = seq_len
        n_shards = self._effective_shards(strategy)
        tokens_per_shard = math.ceil(seq_len / n_shards) if n_shards > 0 else seq_len

        # ── Step 1: Partition sequence to faces/edges ─────────────
        shard_faces = self._shard_to_face(seq_len, n_shards)
        shards: List[Shard] = []
        for sid, (s_id, face) in enumerate(shard_faces):
            device = s_id % self.n_devices
            start  = s_id * tokens_per_shard
            end    = min(start + tokens_per_shard, seq_len)
            shard = Shard(
                shard_id=s_id,
                device_id=device,
                token_range=range(start, end),
                poly_face=face,
                phase=Phase.RADIAL,
            )
            shards.append(shard)

        # ── Step 2: Apply telemetry-based load balancing ──────────
        if telemetry:
            shards = self._balance_load(shards, telemetry)

        # ── Step 3: Build 3-phase communication schedule ──────────
        # Patent Claim 27: "pipeline de tres fases radial–chord–ring"
        comm_events: List[CommEvent] = []
        comm_events += self._radial_phase(shards)
        comm_events += self._chord_phase(shards)
        comm_events += self._ring_phase(shards)

        return ShardSchedule(
            shards=shards,
            comm_events=comm_events,
            strategy=strategy,
            total_tokens=seq_len,
            bandwidth_bps=self.bandwidth,
            n_devices=self.n_devices,
        )

    def _effective_shards(self, strategy: Strategy) -> int:
        """Return the effective number of shards for a given strategy."""
        if strategy == Strategy.DODECA_DISTRIBUTED:
            return 20        # dodecahedron: 20 faces ≈ 20 shards
        elif strategy == Strategy.ICOSA_DISTRIBUTED:
            return 12        # icosahedron: 12 vertices
        elif strategy == Strategy.HEXA_DISTRIBUTED:
            return 8         # cube: 8 vertices
        elif strategy == Strategy.OCTA_DISTRIBUTED:
            return 6         # octahedron: 6 vertices
        elif strategy == Strategy.TETRA_DISTRIBUTED:
            return 4         # tetrahedron: 4 faces/vertices
        return max(1, self.n_devices)

    def compute_diameter(self,
                         topology: Optional[str] = None) -> int:
        """
        Compute the graph diameter (longest shortest path in hops) for the
        given topology, falling back to ``self.topology`` when ``topology``
        is None.

        Patent Claim 28: "diámetro en saltos se reduce ≥33% frente a la
        topología base".

        Diameter table (contract §1):
            tetrahedron=1, octahedron=2, hexahedron(cube)=3,
            icosahedron=3, dodecahedron=5.
        """
        topo = topology if topology is not None else self.topology
        return DIAMETERS.get(topo, 3)

    def diameter_reduction(self, base_topology: str = 'dodecahedron') -> float:
        """
        Compute diameter reduction of ``self.topology`` vs ``base_topology``.

        A SEPARATE planner is instantiated for the base topology so that
        both diameters are computed against their own topology (never
        ``self.topology`` for both).

        reduction = (base_diameter - this_diameter) / base_diameter

        Example (contract §1): base=dodecahedron (5), this=hexahedron (3)
        → 0.4 ≥ 0.333.
        """
        base_planner = MetatronShardPlanner(
            topology=base_topology,
            n_devices=self.n_devices,
            bandwidth_bps=self.bandwidth,
        )
        base_d = base_planner.compute_diameter()      # base topology's diameter
        this_d = self.compute_diameter()              # self.topology's diameter
        if base_d <= 0:
            return 0.0
        return (base_d - this_d) / base_d

    @staticmethod
    def _strategy_name(topology: str) -> str:
        m = {'tetrahedron': 'TETRA_DISTRIBUTED', 'hexahedron': 'HEXA_DISTRIBUTED',
             'octahedron': 'OCTA_DISTRIBUTED', 'dodecahedron': 'DODECA_DISTRIBUTED',
             'icosahedron': 'ICOSA_DISTRIBUTED'}
        return m.get(topology, 'DODECA_DISTRIBUTED')


if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8')

    torch.manual_seed(42)

    # ── Diameter-reduction demo ───────────────────────────────────────
    # Base = dodecahedron (diameter 5); this = hexahedron (diameter 3).
    # reduction = (5 - 3) / 5 = 0.4 ≥ 0.333  (Patent Claim 28).
    demo = MetatronShardPlanner(topology='hexahedron', n_devices=4)
    base_d = demo.compute_diameter(topology='dodecahedron')   # 5
    this_d = demo.compute_diameter()                         # 3
    reduction = demo.diameter_reduction('dodecahedron')      # 0.4
    print(f"Base (dodecahedron) diameter : {base_d} hops")
    print(f"Target (hexahedron) diameter : {this_d} hops")
    print(f"Diameter reduction           : {reduction:.1%}  "
          f"({'OK' if reduction >= 0.333 else 'FAIL'})")

    # ── Plan a real run on the default dodecahedron topology ──────────
    planner = MetatronShardPlanner(topology='dodecahedron', n_devices=4)
    schedule = planner.plan(seq_len=1024)
    schedule.print_summary()

    print("\nPhase breakdown:")
    for phase in Phase:
        evs = schedule.phase_events(phase)
        print(f"  {phase.name:8s}: {len(evs)} events")