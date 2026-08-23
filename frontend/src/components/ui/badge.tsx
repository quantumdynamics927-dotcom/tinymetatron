import type { HTMLAttributes } from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const badgeVariants = cva(
  "inline-flex items-center rounded-full px-2 py-0.5 font-mono text-[10px] font-medium uppercase tracking-[0.12em]",
  {
    variants: {
      variant: {
        default: "bg-primary/12 text-primary",
        muted: "bg-secondary text-muted-foreground",
        ok: "bg-ok/15 text-ok",
        warn: "bg-warn/15 text-warn",
        fault: "bg-fault/15 text-fault",
        outline: "shadow-[var(--shadow-border)] text-muted-foreground",
      },
    },
    defaultVariants: { variant: "default" },
  },
);

export function Badge({
  className,
  variant,
  ...props
}: HTMLAttributes<HTMLSpanElement> & VariantProps<typeof badgeVariants>) {
  return <span className={cn(badgeVariants({ variant }), className)} {...props} />;
}
