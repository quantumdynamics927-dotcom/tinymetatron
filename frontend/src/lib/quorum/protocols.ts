import type { ProtocolDef, ProtocolId } from "./types";

export const PROTOCOLS: Record<ProtocolId, ProtocolDef> = {
  ghz: {
    id: "ghz",
    name: "GHZ broadcast",
    short: "Mint a 17-partite GHZ and certify it.",
    duration: 11,
    summary:
      "ORACLE admits a global GHZ. BOND fuses Bell pairs around the inner ring, WAVE purifies, TOMO/READ estimate the reduced states, and PROOF signs the run.",
    outcome: "A certified n-partite GHZ with process fidelity logged to LEDGER.",
    steps: [
      {
        at: 0,
        label: "Admit the run",
        note: "ORACLE checks the T2 budget with POLICY and CLOCK.",
        active: ["oracle", "policy", "clock"],
        edges: [
          ["oracle", "policy"],
          ["oracle", "clock"],
        ],
      },
      {
        at: 0.12,
        label: "Lease qubits",
        note: "POOL hands BOND a calibrated pair inventory.",
        active: ["pool", "bond", "queue"],
        edges: [
          ["pool", "bond"],
          ["oracle", "queue"],
        ],
      },
      {
        at: 0.28,
        label: "Mint Bell pairs",
        note: "BOND and WAVE purify pairs on the inner ring.",
        active: ["bond", "wave", "clock"],
        edges: [
          ["bond", "wave"],
          ["clock", "wave"],
        ],
      },
      {
        at: 0.46,
        label: "Fuse to GHZ",
        note: "Pairwise fusion walks the ring until all 17 share one state.",
        active: ["bond", "relay", "shield", "wave"],
        edges: [
          ["bond", "relay"],
          ["relay", "shield"],
          ["bond", "wave"],
        ],
      },
      {
        at: 0.64,
        label: "Shadow estimate",
        note: "READ takes sacrificial shots; TOMO reconstructs reduced states.",
        active: ["read", "tomo", "proof", "clock"],
        wait: ["bond"],
        edges: [
          ["read", "tomo"],
          ["tomo", "proof"],
        ],
      },
      {
        at: 0.82,
        label: "Certify & commit",
        note: "PROOF signs. LEDGER commits. IRIS is shown the result.",
        active: ["proof", "ledger", "iris", "oracle"],
        edges: [
          ["proof", "ledger"],
          ["ledger", "iris"],
          ["oracle", "iris"],
        ],
      },
    ],
  },
  teleport: {
    id: "teleport",
    name: "Teleport relay",
    short: "Move an unknown state around the ring.",
    duration: 10,
    summary:
      "MAP compiles a path. RELAY consumes one Bell pair per hop. SHIELD protects the logical payload. PROOF checks the reconstructed state at IRIS.",
    outcome: "Payload reconstructed at the far node; hop fidelities in LEDGER.",
    steps: [
      {
        at: 0,
        label: "Compile the path",
        note: "MAP weights edges by live decoherence; QUEUE writes the tape.",
        active: ["map", "queue", "oracle"],
        edges: [
          ["map", "queue"],
          ["oracle", "queue"],
        ],
      },
      {
        at: 0.16,
        label: "Reserve pairs",
        note: "BOND and POOL reserve one pair per hop.",
        active: ["bond", "pool", "relay"],
        edges: [
          ["bond", "pool"],
          ["bond", "relay"],
        ],
      },
      {
        at: 0.34,
        label: "Hop 1 — inner ring",
        note: "RELAY teleports CLOCK → WAVE.",
        active: ["relay", "clock", "wave"],
        edges: [
          ["clock", "relay"],
          ["relay", "wave"],
        ],
      },
      {
        at: 0.52,
        label: "Hop 2 — fabric",
        note: "WAVE → SHIELD, with a stabilizer round in flight.",
        active: ["relay", "wave", "shield"],
        edges: [
          ["wave", "relay"],
          ["relay", "shield"],
        ],
      },
      {
        at: 0.7,
        label: "Hop 3 — observer",
        note: "Final hop onto a sacrificial qubit READ can measure.",
        active: ["relay", "read", "iris"],
        edges: [
          ["relay", "queue"],
          ["read", "iris"],
        ],
      },
      {
        at: 0.86,
        label: "Verify reconstruction",
        note: "TOMO and PROOF compare against the prepared payload.",
        active: ["tomo", "proof", "ledger"],
        edges: [
          ["tomo", "proof"],
          ["proof", "ledger"],
        ],
      },
    ],
  },
  surface: {
    id: "surface",
    name: "Surface-code round",
    short: "One full stabilizer cycle on the logical qubit.",
    duration: 9,
    summary:
      "SHIELD measures X and Z syndromes. DRIFT updates the noise model. WATCH looks for a burst. PROOF declares the logical live or dead.",
    outcome: "Logical qubit held or killed; syndrome tape committed.",
    steps: [
      {
        at: 0,
        label: "Arm the code",
        note: "POLICY authorizes syndrome spend; POOL leases ancillas.",
        active: ["policy", "pool", "shield"],
        edges: [
          ["policy", "shield"],
          ["shield", "pool"],
        ],
      },
      {
        at: 0.18,
        label: "X stabilizers",
        note: "SHIELD measures the X lattice against CLOCK.",
        active: ["shield", "clock", "drift"],
        edges: [
          ["shield", "drift"],
          ["clock", "drift"],
        ],
      },
      {
        at: 0.4,
        label: "Z stabilizers",
        note: "Z round. DRIFT watches for a correlated burst.",
        active: ["shield", "drift", "watch"],
        edges: [
          ["shield", "drift"],
          ["drift", "watch"],
        ],
      },
      {
        at: 0.62,
        label: "Decode",
        note: "Minimum-weight matching against the live error model.",
        active: ["shield", "proof", "watch"],
        edges: [
          ["shield", "proof"],
          ["watch", "queue"],
        ],
      },
      {
        at: 0.82,
        label: "Declare logical",
        note: "PROOF signs live or dead. LEDGER stores the tape.",
        active: ["proof", "ledger", "oracle"],
        edges: [
          ["proof", "ledger"],
          ["oracle", "policy"],
        ],
      },
    ],
  },
  sensing: {
    id: "sensing",
    name: "Distributed sensing",
    short: "Fuse a field estimate across the outer ring.",
    duration: 12,
    summary:
      "Outer-ring agents sample a shared field. TOMO fuses the shadows. MAP places the estimate on the lattice. IRIS is shown a classical picture only.",
    outcome: "A field map with uncertainty, no protocol qubit leaked to IRIS.",
    steps: [
      {
        at: 0,
        label: "Phase lock sensors",
        note: "CLOCK locks the outer ring; POLICY keeps IRIS read-only.",
        active: ["clock", "policy", "iris"],
        edges: [
          ["clock", "drift"],
          ["policy", "iris"],
        ],
      },
      {
        at: 0.14,
        label: "Share probe pairs",
        note: "BOND distributes sensing pairs to the outer ring.",
        active: ["bond", "drift", "tomo", "read"],
        edges: [
          ["bond", "wave"],
          ["drift", "tomo"],
          ["tomo", "read"],
        ],
      },
      {
        at: 0.34,
        label: "Sample the field",
        note: "Outer agents accumulate phase against the unknown field.",
        active: ["drift", "tomo", "read", "map", "watch"],
        edges: [
          ["drift", "tomo"],
          ["tomo", "map"],
          ["read", "map"],
        ],
      },
      {
        at: 0.54,
        label: "Fuse shadows",
        note: "TOMO combines shots. MAP weights by local T2.",
        active: ["tomo", "map", "proof"],
        edges: [
          ["tomo", "proof"],
          ["tomo", "map"],
        ],
      },
      {
        at: 0.72,
        label: "Place the estimate",
        note: "MAP writes a classical field onto the topology.",
        active: ["map", "ledger", "queue"],
        edges: [
          ["map", "queue"],
          ["read", "ledger"],
        ],
      },
      {
        at: 0.88,
        label: "Show, don't touch",
        note: "IRIS receives a picture. The fabric qubits are retired.",
        active: ["iris", "ledger", "oracle", "pool"],
        edges: [
          ["ledger", "iris"],
          ["oracle", "iris"],
          ["pool", "bond"],
        ],
      },
    ],
  },
  consensus: {
    id: "consensus",
    name: "Consensus lock",
    short: "Lock a single global phase across all 17 agents.",
    duration: 8,
    summary:
      "The control plane publishes a phase. Every agent acknowledges against CLOCK. WATCH vetoes stragglers. ORACLE commits the lock.",
    outcome: "A lattice-wide phase lock, or an abort if any agent misses the window.",
    steps: [
      {
        at: 0,
        label: "Publish intent",
        note: "ORACLE proposes a lock; POLICY checks the budget.",
        active: ["oracle", "policy", "clock"],
        edges: [
          ["oracle", "policy"],
          ["oracle", "clock"],
        ],
      },
      {
        at: 0.2,
        label: "Inner ring ack",
        note: "Fabric agents phase-lock to CLOCK.",
        active: ["clock", "bond", "wave", "relay", "shield"],
        edges: [
          ["clock", "wave"],
          ["clock", "relay"],
          ["oracle", "bond"],
        ],
      },
      {
        at: 0.45,
        label: "Outer ring ack",
        note: "Sensing and ops agents join the lock window.",
        active: ["clock", "drift", "tomo", "read", "map", "proof", "pool", "queue", "watch", "ledger", "iris"],
        edges: [
          ["clock", "drift"],
          ["queue", "watch"],
          ["ledger", "iris"],
        ],
      },
      {
        at: 0.7,
        label: "Veto window",
        note: "WATCH scores stragglers. A single miss aborts.",
        active: ["watch", "policy", "queue"],
        edges: [
          ["watch", "queue"],
          ["policy", "watch"],
        ],
      },
      {
        at: 0.86,
        label: "Commit lock",
        note: "ORACLE commits. LEDGER records the lock card.",
        active: ["oracle", "ledger", "iris"],
        edges: [
          ["oracle", "iris"],
          ["ledger", "iris"],
        ],
      },
    ],
  },
};

export const PROTOCOL_LIST = Object.values(PROTOCOLS);
