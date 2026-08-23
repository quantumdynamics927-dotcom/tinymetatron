import { useState } from "react";
import { createFileRoute } from "@tanstack/react-router";
import { ChevronRight, FileText, Folder } from "lucide-react";
import { Doc } from "@/components/lab/markdown";
import { Shell } from "@/components/lab/shell";
import { ScrollArea } from "@/components/ui/scroll-area";
import { cn } from "@/lib/utils";

export const Route = createFileRoute("/repository")({ component: RepositoryPage });

// TinyMetatron Copilot documentation tree
const TMT_REPO = {
  name: "tinymetatron",
  path: "tinymetatron",
  kind: "dir" as const,
  children: [
    {
      name: "README.md",
      path: "tinymetatron/README.md",
      kind: "file" as const,
      lang: "md" as const,
      content: `# TinyMetatron Copilot

17-agent quantum orchestration layer over TinyMetatron loops.

## Architecture

- **copilot/** — orchestration engine (agents, routing, topology, benchmark)
- **loops/** — training, corpus, evaluation, generalization, feedback, promote
- **quantum_corpus/** — RAG backbone with BM25 retrieval

## Agents

Four layers: INPUT, PROCESSING, INTEGRATION, OUTPUT.
Each agent is defined in \`copilot/agents/*.yaml\` with φ-score, resonance
frequency, and fitness. Agents are invoked via \`POST /copilot/agents/{id}/invoke\`.

## Execution Modes

- **simulation** — dry-run, no real loop execution
- **live** — real training/corpus/evaluation loops
- **hybrid** — live + IBM Quantum hardware fallback
`,
    },
    {
      name: "AGENTS.md",
      path: "tinymetatron/AGENTS.md",
      kind: "file" as const,
      lang: "md" as const,
      content: `# Agent Roster

| Role | Layer | φ-Score | Resonance |
|---|---|---|---|
| strategic | processing | 0.929 | 682 Hz |
| synthesizer | integration | 0.879 | 630 Hz |
| federation | processing | 0.873 | 644 Hz |
| fractal | integration | 0.871 | 668 Hz |
| archivist | output | 0.888 | 612 Hz |
| validator | processing | — | — |
| observer | input | — | — |
| workflow | processing | — | — |
| auditor | output | — | — |
| bronze | processing | — | — |
| bitnet | processing | — | — |
| harmonic | processing | — | — |
| mirror | integration | — | — |
| wormhole | input | — | — |
| stealth | output | — | — |
| visual | integration | — | — |
| bio | input | — | — |
`,
    },
    {
      name: "copilot",
      path: "tinymetatron/copilot",
      kind: "dir" as const,
      children: [
        {
          name: "orchestration",
          path: "tinymetatron/copilot/orchestration",
          kind: "dir" as const,
          children: [
            {
              name: "orchestrator.py",
              path: "tinymetatron/copilot/orchestration/orchestrator.py",
              kind: "file" as const,
              lang: "md" as const,
              content: `# AgentOrchestrator

Central dispatcher. Routes task types to agent roles, executes via
loop adapters, and collects coordination traces.

\`\`\`python
orch = AgentOrchestrator(vault_path=Path("copilot/agents"))
trace = orch.execute("validation", "Run gate validation",
                     context={"run_id": "exp-42"},
                     execution_mode=ExecutionMode.SIMULATION)
\`\`\`
`,
            },
            {
              name: "loop_adapters.py",
              path: "tinymetatron/copilot/orchestration/loop_adapters.py",
              kind: "file" as const,
              lang: "md" as const,
              content: `# Loop Adapters

17 functions bridging agents to real TinyMetatron loops.

Key adapters:
- \`call_workflow\` → \`train_loop.run_training()\` / \`corpus_loop.run_corpus_pipeline()\`
- \`call_validator\` → \`generalize_loop.run_gate()\`
- \`call_observer\` → \`db.get_evaluations()\`, \`db.get_gate_results()\`
- \`call_synthesizer\` → aggregates upstream outputs
- \`call_mirror\` → stall detection (val_ce plateau)
- \`call_fractal\` → Sierpinski circuit spec generation
`,
            },
          ],
        },
      ],
    },
  ],
};

type RepoNode = {
  name: string;
  path: string;
  kind: "dir" | "file";
  lang?: "md" | "yaml" | "json";
  children?: RepoNode[];
  content?: string;
};

function findRepo(path: string, node: RepoNode): RepoNode | null {
  if (node.path === path) return node;
  for (const child of node.children ?? []) {
    const hit = findRepo(path, child);
    if (hit) return hit;
  }
  return null;
}

const DEFAULT_FILE = "tinymetatron/README.md";

function RepositoryPage() {
  const [path, setPath] = useState(DEFAULT_FILE);
  const file = findRepo(path, TMT_REPO as RepoNode);

  return (
    <Shell>
      <div className="grid min-h-0 flex-1 grid-cols-1 lg:grid-cols-[260px_minmax(0,1fr)]">
        <aside className="border-b border-border lg:border-r lg:border-b-0">
          <ScrollArea className="h-full max-h-[40dvh] lg:max-h-none">
            <div className="p-4">
              <p className="mb-3 font-mono text-[10px] tracking-[0.16em] text-muted-foreground uppercase">
                Tree
              </p>
              <Tree node={TMT_REPO as RepoNode} selected={path} onSelect={setPath} depth={0} />
            </div>
          </ScrollArea>
        </aside>
        <ScrollArea className="h-full">
          <div className="mx-auto w-full max-w-2xl px-4 py-6 lg:px-10">
            <p className="font-mono text-[11px] text-muted-foreground">{path}</p>
            <div className="mt-6">
              {file?.kind === "file" && file.content ? (
                <Doc content={file.content} lang={file.lang} />
              ) : (
                <p className="text-sm text-muted-foreground">Select a file from the tree.</p>
              )}
            </div>
          </div>
        </ScrollArea>
      </div>
    </Shell>
  );
}

function Tree({
  node,
  selected,
  onSelect,
  depth,
}: {
  node: RepoNode;
  selected: string;
  onSelect: (path: string) => void;
  depth: number;
}) {
  const [open, setOpen] = useState(depth < 2);
  if (node.kind === "file") {
    const active = selected === node.path;
    return (
      <button
        type="button"
        onClick={() => onSelect(node.path)}
        className={cn(
          "flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-left text-sm",
          active ? "bg-secondary text-foreground" : "text-muted-foreground hover:text-foreground",
        )}
        style={{ paddingLeft: 8 + depth * 12 }}
      >
        <FileText className="size-3.5 shrink-0" />
        <span className="truncate">{node.name}</span>
      </button>
    );
  }
  return (
    <div>
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-left text-sm text-foreground/90 hover:bg-accent"
        style={{ paddingLeft: 8 + depth * 12 }}
      >
        <ChevronRight className={cn("size-3.5 shrink-0 transition-transform duration-150", open && "rotate-90")} />
        <Folder className="size-3.5 shrink-0" />
        <span className="truncate">{node.name}</span>
      </button>
      {open
        ? node.children?.map((child) => (
            <Tree key={child.path} node={child} selected={selected} onSelect={onSelect} depth={depth + 1} />
          ))
        : null}
    </div>
  );
}
