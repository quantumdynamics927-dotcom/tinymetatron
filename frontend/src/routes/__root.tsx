import { createRootRoute, HeadContent, Outlet, Scripts } from "@tanstack/react-router";
import { EngineHost } from "@/components/lab/engine-host";
import { TooltipProvider } from "@/components/ui/tooltip";
import appCss from "../styles.css?url";

const APP_NAME = "TinyMetatron Copilot";

export const Route = createRootRoute({
  head: () => ({
    meta: [
      { charSet: "utf-8" },
      { name: "viewport", content: "width=device-width, initial-scale=1" },
      { title: APP_NAME },
      {
        name: "description",
        content: "TinyMetatron Copilot — 17-agent quantum orchestration lab.",
      },
      { name: "theme-color", content: "#08090b" },
    ],
    links: [
      { rel: "icon", type: "image/svg+xml", href: "/favicon.svg" },
      { rel: "stylesheet", href: appCss },
      { rel: "preconnect", href: "https://fonts.googleapis.com" },
      { rel: "preconnect", href: "https://fonts.gstatic.com", crossOrigin: "anonymous" },
      {
        rel: "stylesheet",
        href: "https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;1,400&family=Instrument+Serif:ital@0;1&display=swap",
      },
    ],
  }),
  component: () => (
    <html lang="en" className="dark antialiased" suppressHydrationWarning>
      <head>
        <HeadContent />
      </head>
      <body>
        <TooltipProvider delayDuration={200}>
          <EngineHost />
          <Outlet />
        </TooltipProvider>
        <Scripts />
      </body>
    </html>
  ),
});
