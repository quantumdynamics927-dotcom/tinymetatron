export function Doc({ content, lang }: { content: string; lang?: string }) {
  if (lang && lang !== "md") {
    return (
      <pre className="overflow-x-auto rounded-lg bg-secondary p-4 font-mono text-[12px] leading-relaxed text-foreground/90">
        {content}
      </pre>
    );
  }

  const blocks = splitBlocks(content);
  return (
    <article className="space-y-4">
      {blocks.map((b, i) => {
        if (b.type === "h1")
          return (
            <h1 key={i} className="font-display text-3xl leading-tight tracking-tight">
              {b.text}
            </h1>
          );
        if (b.type === "h2")
          return (
            <h2 key={i} className="font-display text-xl leading-snug">
              {b.text}
            </h2>
          );
        if (b.type === "code")
          return (
            <pre
              key={i}
              className="overflow-x-auto rounded-lg bg-secondary p-4 font-mono text-[12px] leading-relaxed"
            >
              {b.text}
            </pre>
          );
        if (b.type === "list")
          return (
            <ul key={i} className="space-y-1.5 text-sm leading-relaxed text-foreground/85">
              {b.items.map((item) => (
                <li key={item} className="pl-4 -indent-4 before:mr-2 before:text-subtle before:content-['—']">
                  {item}
                </li>
              ))}
            </ul>
          );
        return (
          <p key={i} className="text-sm leading-relaxed text-foreground/85">
            {b.text}
          </p>
        );
      })}
    </article>
  );
}

type Block =
  | { type: "h1" | "h2" | "p" | "code"; text: string }
  | { type: "list"; items: string[] };

function splitBlocks(src: string): Block[] {
  const lines = src.replace(/\r\n/g, "\n").split("\n");
  const out: Block[] = [];
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    if (line.startsWith("```")) {
      const buf: string[] = [];
      i += 1;
      while (i < lines.length && !lines[i].startsWith("```")) {
        buf.push(lines[i]);
        i += 1;
      }
      i += 1;
      out.push({ type: "code", text: buf.join("\n") });
      continue;
    }
    if (line.startsWith("# ")) {
      out.push({ type: "h1", text: line.slice(2) });
      i += 1;
      continue;
    }
    if (line.startsWith("## ")) {
      out.push({ type: "h2", text: line.slice(3) });
      i += 1;
      continue;
    }
    if (line.startsWith("- ")) {
      const items: string[] = [];
      while (i < lines.length && lines[i].startsWith("- ")) {
        items.push(lines[i].slice(2));
        i += 1;
      }
      out.push({ type: "list", items });
      continue;
    }
    if (line.trim() === "") {
      i += 1;
      continue;
    }
    const buf: string[] = [line];
    i += 1;
    while (i < lines.length && lines[i].trim() && !/^#{1,2} |^- |```/.test(lines[i])) {
      buf.push(lines[i]);
      i += 1;
    }
    out.push({ type: "p", text: buf.join(" ") });
  }
  return out;
}
