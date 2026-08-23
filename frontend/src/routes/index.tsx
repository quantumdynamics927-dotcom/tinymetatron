import { createFileRoute } from "@tanstack/react-router";
import { AgentList } from "@/components/lab/agent-list";
import { Constellation } from "@/components/lab/constellation";
import { Inspector } from "@/components/lab/inspector";
import { Shell } from "@/components/lab/shell";
import { ScrollArea } from "@/components/ui/scroll-area";

export const Route = createFileRoute("/")({ component: Home });

function Home() {
  return (
    <Shell>
      <div className="grid min-h-0 flex-1 grid-cols-1 lg:grid-cols-[220px_minmax(0,1fr)_300px]">
        <aside className="hidden min-h-0 border-r border-border lg:block">
          <ScrollArea className="h-full">
            <div className="p-4">
              <AgentList compact />
            </div>
          </ScrollArea>
        </aside>
        <section className="min-h-[52dvh] p-3 lg:min-h-0 lg:p-4">
          <Constellation />
        </section>
        <aside className="min-h-0 border-t border-border lg:border-t-0 lg:border-l">
          <ScrollArea className="h-full">
            <div className="p-5">
              <Inspector />
            </div>
          </ScrollArea>
        </aside>
      </div>
    </Shell>
  );
}
