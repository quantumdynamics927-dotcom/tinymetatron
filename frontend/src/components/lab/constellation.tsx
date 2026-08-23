import { useEffect, useRef } from "react";
import { AGENTS, AGENT_IDS, BASE_EDGES } from "@/lib/quorum/agents";
import { engine } from "@/lib/quorum/engine";
import { selectAgent, hoverAgent } from "@/lib/quorum/engine";
import { NODE_POS, toCanvas } from "@/lib/quorum/layout";
import type { AgentRole } from "@/lib/quorum/types";

const COLORS = {
  bg: "#08090b",
  fg: "#e6e8eb",
  muted: "#8b919a",
  subtle: "#6a7078",
  line: "rgba(230,232,235,0.10)",
  lineHot: "rgba(197,204,214,0.55)",
  nodeIdle: "#3a3f46",
  nodeSync: "#8b919a",
  nodeActive: "#e6e8eb",
  nodeWait: "#c4b59a",
  nodeFault: "#c48b84",
  packet: "#c5ccd6",
  ring: "rgba(230,232,235,0.06)",
};

function nodeColor(id: AgentRole): string {
  const s = engine.runtime[id]?.status ?? "idle";
  if (s === "active") return COLORS.nodeActive;
  if (s === "sync") return COLORS.nodeSync;
  if (s === "wait") return COLORS.nodeWait;
  if (s === "fault") return COLORS.nodeFault;
  return COLORS.nodeIdle;
}

function hitTest(x: number, y: number, w: number, h: number): AgentRole | null {
  let best: AgentRole | null = null;
  let bestD = 22;
  for (const id of AGENT_IDS) {
    const p = toCanvas(NODE_POS[id], w, h);
    const d = Math.hypot(p.x - x, p.y - y);
    const r = id === "strategic" ? 18 : 14; // oracle = strategic (center)
    if (d < r && d < bestD) {
      best = id;
      bestD = d;
    }
  }
  return best;
}

export function Constellation() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const wrap = wrapRef.current;
    if (!canvas || !wrap) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let raf = 0;

    const resize = () => {
      const rect = wrap.getBoundingClientRect();
      const w = Math.max(1, Math.floor(rect.width));
      const h = Math.max(1, Math.floor(rect.height));
      const dpr = Math.min(2, window.devicePixelRatio || 1);
      canvas.width = Math.floor(w * dpr);
      canvas.height = Math.floor(h * dpr);
      canvas.style.width = `${w}px`;
      canvas.style.height = `${h}px`;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };

    const ro = new ResizeObserver(resize);
    ro.observe(wrap);
    resize();

    const draw = () => {
      const w = parseInt(canvas.style.width) || canvas.width;
      const h = parseInt(canvas.style.height) || canvas.height;
      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = COLORS.bg;
      ctx.fillRect(0, 0, w, h);

      const cx = w / 2;
      const cy = h / 2;
      const s = Math.min(w, h) / 2 - 28;

      ctx.strokeStyle = COLORS.ring;
      ctx.lineWidth = 1;
      for (const r of [0.42, 0.82]) {
        ctx.beginPath();
        ctx.arc(cx, cy, r * s, 0, Math.PI * 2);
        ctx.stroke();
      }
      ctx.beginPath();
      ctx.arc(cx, cy, 0.14 * s, 0, Math.PI * 2);
      ctx.stroke();

      // Draw edges
      ctx.lineWidth = 1;
      for (const [a, b] of BASE_EDGES) {
        const pa = toCanvas(NODE_POS[a], w, h);
        const pb = toCanvas(NODE_POS[b], w, h);
        ctx.strokeStyle = COLORS.line;
        ctx.beginPath();
        ctx.moveTo(pa.x, pa.y);
        ctx.lineTo(pb.x, pb.y);
        ctx.stroke();
      }

      // Draw agents
      for (const id of AGENT_IDS) {
        const p = toCanvas(NODE_POS[id], w, h);
        const def = AGENTS[id];
        if (!def) continue;
        const selected = engine.selectedId === id;
        const hovered = engine.hoveredId === id;
        const r = id === "strategic" ? 7.5 : 5.2;
        const color = nodeColor(id);
        const t = engine.time / 1000;
        const pulse = engine.runtime[id]?.status === "active" ? 1 + Math.sin(t * 5) * 0.08 : 1;

        if (selected || hovered) {
          ctx.strokeStyle = COLORS.fg;
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.arc(p.x, p.y, r * pulse + 6, 0, Math.PI * 2);
          ctx.stroke();
        }

        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(p.x, p.y, r * pulse, 0, Math.PI * 2);
        ctx.fill();

        ctx.font = "500 10px 'IBM Plex Mono', ui-monospace, monospace";
        ctx.textAlign = "center";
        ctx.textBaseline = "top";
        ctx.fillStyle = selected || hovered ? COLORS.fg : COLORS.muted;
        ctx.fillText(def.callsign, p.x, p.y + r + 6);
      }

      raf = requestAnimationFrame(draw);
    };

    raf = requestAnimationFrame(draw);

    const onMove = (ev: PointerEvent) => {
      const rect = canvas.getBoundingClientRect();
      const id = hitTest(ev.clientX - rect.left, ev.clientY - rect.top, parseInt(canvas.style.width), parseInt(canvas.style.height));
      hoverAgent(id);
      canvas.style.cursor = id ? "pointer" : "default";
    };
    const onLeave = () => hoverAgent(null);
    const onClick = (ev: PointerEvent) => {
      const rect = canvas.getBoundingClientRect();
      const id = hitTest(ev.clientX - rect.left, ev.clientY - rect.top, parseInt(canvas.style.width), parseInt(canvas.style.height));
      selectAgent(id);
    };

    canvas.addEventListener("pointermove", onMove);
    canvas.addEventListener("pointerleave", onLeave);
    canvas.addEventListener("click", onClick);

    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      canvas.removeEventListener("pointermove", onMove);
      canvas.removeEventListener("pointerleave", onLeave);
      canvas.removeEventListener("click", onClick);
    };
  }, []);

  return (
    <div ref={wrapRef} className="relative h-full min-h-[320px] w-full overflow-hidden rounded-xl bg-background">
      <canvas ref={canvasRef} className="block h-full w-full" aria-label="17-agent constellation" />
    </div>
  );
}
