export type RepoKind = "dir" | "file";
export type RepoLang = "md" | "yaml" | "json";

export interface RepoNode {
  name: string;
  path: string;
  kind: RepoKind;
  lang?: RepoLang;
  children?: RepoNode[];
  content?: string;
}

export const REPO: RepoNode = {
  name: "quorum",
  path: "quorum",
  kind: "dir",
  children: [
    {
      name: "README.md",
      path: "quorum/README.md",
      kind: "file",
      lang: "md",
      content: `# QUORUM

Seventeen agents. One lattice.

QUORUM is a research environment for studying coordination in a distributed
quantum control plane. Agents do not share a heap. They share entanglement,
a clock, and an append-only ledger.

## Layout

- \`agents/\` — the 17 agent specifications (identity, duties, neighbors)
- \`protocols/\` — GHZ, teleport, surface code, sensing, consensus
- \`fabric/\` — topology, entanglement graph, channel model
- \`experiments/\` — run cards and baselines
- \`telemetry/\` — metric definitions
- \`papers/\` — system notes
- \`datasets/\` — synthetic run logs

## Invariants

1. IRIS is read-only against the fabric.
2. Only ORACLE may commit a global rewrite of the coordination graph.
3. LEDGER will not store a run without a PROOF signature.
4. A WATCH fault pauses QUEUE. It does not auto-resume.
`,
    },
    {
      name: "CITATION.cff",
      path: "quorum/CITATION.cff",
      kind: "file",
      lang: "yaml",
      content: `cff-version: 1.2.0
title: "QUORUM: a 17-agent quantum coordination lattice"
message: If you use this lattice, please cite it.
authors:
  - name: QUORUM Research
version: 0.1.0
license: CC-BY-4.0
`,
    },
    {
      name: "agents",
      path: "quorum/agents",
      kind: "dir",
      children: [
        {
          name: "_index.yaml",
          path: "quorum/agents/_index.yaml",
          kind: "file",
          lang: "yaml",
          content: `lattice: quorum
count: 17
layers:
  control: [oracle, clock, policy]
  fabric: [bond, wave, relay, shield, drift]
  sensing: [tomo, read, map, proof]
  ops: [pool, queue, watch, ledger, iris]
invariants:
  - iris_readonly: true
  - oracle_unique_commit: true
  - proof_required_for_ledger: true
`,
        },
        {
          name: "q00-oracle.yaml",
          path: "quorum/agents/q00-oracle.yaml",
          kind: "file",
          lang: "yaml",
          content: `id: oracle
code: Q-00
callsign: ORACLE
layer: control
role: consensus_and_dispatch
qubits: 8
writes:
  - coordination_graph
  - protocol_admission
reads:
  - policy.budget
  - clock.phase
  - queue.tape
neighbors: [clock, policy, bond, queue, iris]
`,
        },
        {
          name: "q03-bond.yaml",
          path: "quorum/agents/q03-bond.yaml",
          kind: "file",
          lang: "yaml",
          content: `id: bond
code: Q-03
callsign: BOND
layer: fabric
role: entanglement_broker
qubits: 16
mint:
  - bell_pair
  - ghz
  - cluster
retire_below_fidelity: 0.92
neighbors: [oracle, wave, relay, proof, pool]
`,
        },
        {
          name: "q06-shield.yaml",
          path: "quorum/agents/q06-shield.yaml",
          kind: "file",
          lang: "yaml",
          content: `id: shield
code: Q-06
callsign: SHIELD
layer: fabric
role: error_correction
code_family: rotated_surface
distance: 5
qubits: 24
cycle:
  - x_stabilizers
  - z_stabilizers
  - decode
  - declare_logical
neighbors: [policy, relay, drift, proof, pool]
`,
        },
        {
          name: "q16-iris.yaml",
          path: "quorum/agents/q16-iris.yaml",
          kind: "file",
          lang: "yaml",
          content: `id: iris
code: Q-16
callsign: IRIS
layer: ops
role: observer_interface
qubits: 0
permissions:
  fabric: read
  ledger: read
  oracle: request
notes: >
  The only agent a human may touch. Writes are requests, never commands.
  Holding a protocol qubit is a specification violation.
neighbors: [oracle, policy, read, watch, ledger]
`,
        },
      ],
    },
    {
      name: "protocols",
      path: "quorum/protocols",
      kind: "dir",
      children: [
        {
          name: "ghz-broadcast.md",
          path: "quorum/protocols/ghz-broadcast.md",
          kind: "file",
          lang: "md",
          content: `# GHZ broadcast

Goal: mint a certified n-partite GHZ across the live lattice.

## Tape

1. ORACLE admits the run against POLICY's T2 budget.
2. POOL leases pairs to BOND.
3. BOND + WAVE purify Bell pairs on the inner ring.
4. Fusion walks the ring until all 17 share one state.
5. READ takes sacrificial shots; TOMO estimates reduced states.
6. PROOF signs. LEDGER commits.

## Failures

- Pair fidelity below 0.92 before fusion → abort, return qubits to POOL.
- WATCH burst during fusion → QUEUE pauses; SHIELD may attempt recovery.
`,
        },
        {
          name: "teleport-relay.md",
          path: "quorum/protocols/teleport-relay.md",
          kind: "file",
          lang: "md",
          content: `# Teleport relay

Goal: move an unknown payload around the ring without a physical carrier.

MAP compiles a decoherence-weighted path. RELAY consumes one Bell pair
per hop. SHIELD may wrap the payload in a distance-3 code for the
longest hop. PROOF compares the reconstructed state to the prepared
payload on a sacrificial copy — never on the payload itself.
`,
        },
        {
          name: "surface-code.md",
          path: "quorum/protocols/surface-code.md",
          kind: "file",
          lang: "md",
          content: `# Surface-code round

One full stabilizer cycle on the logical qubit owned by SHIELD.

X then Z, decode, declare. DRIFT updates the noise model between
rounds. A correlated burst from WATCH during the Z round is treated
as a likely logical error; PROOF must be shown the tape.
`,
        },
        {
          name: "distributed-sensing.md",
          path: "quorum/protocols/distributed-sensing.md",
          kind: "file",
          lang: "md",
          content: `# Distributed sensing

Outer-ring agents accumulate phase against an unknown field using
pairs minted by BOND. TOMO fuses classical shadows. MAP places the
estimate. IRIS is shown a picture; the fabric qubits are retired
before the picture is drawn.
`,
        },
        {
          name: "consensus-lock.md",
          path: "quorum/protocols/consensus-lock.md",
          kind: "file",
          lang: "md",
          content: `# Consensus lock

A lattice-wide phase lock. Inner ring first, outer ring second,
WATCH veto window last. A single missed ack aborts. There is no
partial lock — ORACLE either commits all 17 or none.
`,
        },
      ],
    },
    {
      name: "fabric",
      path: "quorum/fabric",
      kind: "dir",
      children: [
        {
          name: "topology.json",
          path: "quorum/fabric/topology.json",
          kind: "file",
          lang: "json",
          content: `{
  "center": "oracle",
  "inner": ["clock", "policy", "bond", "wave", "relay", "shield"],
  "outer": ["drift", "tomo", "read", "map", "proof", "pool", "queue", "watch", "ledger", "iris"],
  "rule": "inner star + inner cycle + outer-to-inner + sparse outer cycle"
}
`,
        },
        {
          name: "channel-model.md",
          path: "quorum/fabric/channel-model.md",
          kind: "file",
          lang: "md",
          content: `# Channel model

Inner-ring links: loss 0.4 dB, T2 28 µs, purification every 4 ticks.
Outer links: loss 1.1 dB, T2 18 µs, no purification unless WAVE is
explicitly leased.

Classical control is assumed lossless and ordered against CLOCK.
A reordering is a WATCH fault, not a retry.
`,
        },
      ],
    },
    {
      name: "experiments",
      path: "quorum/experiments",
      kind: "dir",
      children: [
        {
          name: "run-schema.json",
          path: "quorum/experiments/run-schema.json",
          kind: "file",
          lang: "json",
          content: `{
  "id": "uuid",
  "protocol": "ghz | teleport | surface | sensing | consensus",
  "admitted_by": "oracle",
  "tape": [{ "at": 0, "label": "string", "agents": [] }],
  "proof": { "pass": true, "fidelity": 0.0 },
  "watch": { "fault": false },
  "ledger_commit": "hex"
}
`,
        },
        {
          name: "baselines.md",
          path: "quorum/experiments/baselines.md",
          kind: "file",
          lang: "md",
          content: `# Baselines

Idle lattice: mean fidelity ≥ 0.97, event rate < 4 Hz, no bonds.
GHZ broadcast: certified fidelity ≥ 0.94 on n=17.
Teleport relay (3 hops): end-to-end ≥ 0.91.
Surface-code round (d=5): logical error < 1e-3 per round under the
idle noise model.
`,
        },
      ],
    },
    {
      name: "telemetry",
      path: "quorum/telemetry",
      kind: "dir",
      children: [
        {
          name: "metrics.md",
          path: "quorum/telemetry/metrics.md",
          kind: "file",
          lang: "md",
          content: `# Metrics

- **Mean fidelity** — unweighted mean of per-agent process fidelity.
- **Bonds** — live entanglement edges with fidelity ≥ 0.92.
- **T2** — minimum coherence time reported by DRIFT.
- **Event rate** — LEDGER commits plus WATCH faults per second.

All four are sampled at 10 Hz into a 96-sample ring for the strip.
`,
        },
      ],
    },
    {
      name: "papers",
      path: "quorum/papers",
      kind: "dir",
      children: [
        {
          name: "00-system-overview.md",
          path: "quorum/papers/00-system-overview.md",
          kind: "file",
          lang: "md",
          content: `# System overview

QUORUM splits a quantum control plane into 17 named agents so that
coordination can be studied as a protocol, not as a monolith.

The control plane (ORACLE, CLOCK, POLICY) is small on purpose. It
admits work and refuses work. It does not hold payload qubits.

The fabric (BOND, WAVE, RELAY, SHIELD, DRIFT) is where entanglement
lives. Sensing (TOMO, READ, MAP, PROOF) is allowed to look, not to
steer. Operations (POOL, QUEUE, WATCH, LEDGER, IRIS) keep inventory,
time, faults, history, and the human out of the way.

The interesting failure is not a dead qubit. It is two agents that
both believe they hold the lock.
`,
        },
        {
          name: "01-coordination-calculus.md",
          path: "quorum/papers/01-coordination-calculus.md",
          kind: "file",
          lang: "md",
          content: `# Coordination calculus

A run is a tape. A tape is a sequence of (time, agents, edges).
Time is CLOCK. Agents are a subset of the 17. Edges are a subset
of the live topology.

Admission is a function of POLICY's budget and POOL's inventory.
Certification is a function of PROOF. Persistence is LEDGER.
Nothing else is allowed to look like a commit.

The calculus has one reduction rule: if WATCH raises, QUEUE stops.
Resume is a new admission, not a continuation.
`,
        },
        {
          name: "02-decoherence-budget.md",
          path: "quorum/papers/02-decoherence-budget.md",
          kind: "file",
          lang: "md",
          content: `# Decoherence budget

Every protocol spends T2. POLICY tracks a lattice-wide envelope:
the sum of expected hold times, weighted by the number of live
bonds, must fit inside the worst T2 DRIFT has published in the
last second.

GHZ broadcast is expensive because fusion holds all pairs at once.
Teleport is cheaper: only one hop is live. Consensus is cheapest:
it holds phase, not payload.
`,
        },
      ],
    },
    {
      name: "datasets",
      path: "quorum/datasets",
      kind: "dir",
      children: [
        {
          name: "synthetic-runs.md",
          path: "quorum/datasets/synthetic-runs.md",
          kind: "file",
          lang: "md",
          content: `# Synthetic runs

The live lab generates synthetic run cards against the idle noise
model. They are not experimental data. They exist so the repository
browser, the lattice, and the telemetry strip share one clock.

To replace them with hardware traces, point LEDGER at a shot store
and keep the run schema unchanged.
`,
        },
      ],
    },
  ],
};

export function flattenRepo(node: RepoNode = REPO): RepoNode[] {
  const out: RepoNode[] = [node];
  for (const child of node.children ?? []) out.push(...flattenRepo(child));
  return out;
}

export function findRepo(path: string, node: RepoNode = REPO): RepoNode | null {
  if (node.path === path) return node;
  for (const child of node.children ?? []) {
    const hit = findRepo(path, child);
    if (hit) return hit;
  }
  return null;
}

export const DEFAULT_FILE = "quorum/README.md";
