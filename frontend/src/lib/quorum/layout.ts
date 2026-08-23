import { INNER_RING, OUTER_RING } from "./agents";
import type { AgentId } from "./types";

export interface NodePos {
  x: number;
  y: number;
  ring: 0 | 1 | 2;
}

const TAU = Math.PI * 2;

function ringPos(count: number, index: number, radius: number, turn: number, ring: 1 | 2): NodePos {
  const a = turn + (TAU * index) / count;
  return { x: Math.cos(a) * radius, y: Math.sin(a) * radius, ring };
}

function build(): Record<AgentId, NodePos> {
  const pos = { oracle: { x: 0, y: 0, ring: 0 as const } } as Record<AgentId, NodePos>;
  INNER_RING.forEach((id, i) => {
    pos[id] = ringPos(INNER_RING.length, i, 0.42, -Math.PI / 2, 1);
  });
  OUTER_RING.forEach((id, i) => {
    pos[id] = ringPos(OUTER_RING.length, i, 0.82, -Math.PI / 2 + Math.PI / 10, 2);
  });
  return pos;
}

export const NODE_POS: Record<AgentId, NodePos> = build();

export function toCanvas(pos: NodePos, w: number, h: number, pad = 28): { x: number; y: number } {
  const s = Math.min(w, h) / 2 - pad;
  return { x: w / 2 + pos.x * s, y: h / 2 + pos.y * s };
}
