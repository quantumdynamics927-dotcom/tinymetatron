import { useEffect, useState } from "react";
import { initStore } from "@/lib/quorum/store";
import { tick } from "@/lib/quorum/engine";
import { useQuorum } from "@/lib/quorum/store";

export function EngineHost() {
  const bump = useQuorum((s) => s.actions.bump);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    initStore();
    setReady(true);
  }, []);

  // Animation loop
  useEffect(() => {
    if (!ready) return;
    let raf = 0;
    let last = performance.now();

    const loop = (now: number) => {
      const dt = Math.min(0.05, (now - last) / 1000);
      last = now;
      tick(dt);
      raf = requestAnimationFrame(loop);
    };

    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [ready]);

  // Space bar toggle
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement | null)?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA") return;
      if (e.key === " ") {
        e.preventDefault();
        const { actions } = useQuorum.getState() as any;
        actions.toggleRunning();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  return null;
}
