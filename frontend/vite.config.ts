import { defineConfig } from "vite";
import viteReact from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  server: {
    host: "0.0.0.0",
    port: 5173,
    proxy: {
      "/copilot": {
        target: "http://localhost:8000",
        changeOrigin: true,
        ws: true,
      },
    },
  },
  resolve: { tsconfigPaths: true },
  plugins: [
    tailwindcss(),
    viteReact(),
  ],
});
