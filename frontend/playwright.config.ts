import { defineConfig } from "@playwright/test";
import path from "node:path";
import { randomBytes } from "node:crypto";
// Each browser run uses its own credential, never a repository default.
process.env.TEAM_PASSWORD ||= randomBytes(24).toString("hex");
const root = path.resolve(__dirname, "..");
const python =
  process.env.PYTHON ||
  (process.platform === "win32"
    ? path.join(root, ".venv", "Scripts", "python.exe")
    : "python");
export default defineConfig({
  testDir: "./tests",
  timeout: 90000,
  workers: 1,
  use: {
    baseURL: "http://127.0.0.1:3101",
    browserName: "chromium",
    channel:
      process.env.BROWSER_CHANNEL ||
      (process.platform === "win32" ? "msedge" : undefined),
    trace: process.env.CI ? "retain-on-failure" : "off",
    screenshot: "only-on-failure",
  },
  webServer: [
    {
      command:
        '"' +
        python +
        '" -m uvicorn app:app --app-dir backend --host 127.0.0.1 --port 8101',
      cwd: root,
      url: "http://127.0.0.1:8101/health",
      reuseExistingServer: false,
      env: {
        DATABASE_URL: "sqlite:///./runtime/e2e-" + process.pid + ".db",
        DEMO_MODE: "true",
        TEAM_USERNAME: "demo",
        TEAM_PASSWORD: process.env.TEAM_PASSWORD!,
        LLM_PROVIDER: "ollama",
        // Exercise model-outage usability deterministically; real Qwen has a separate benchmark.
        OLLAMA_URL: "http://127.0.0.1:11435",
        ALLOWED_ORIGINS: "http://127.0.0.1:3101,http://localhost:3101",
      },
    },
    {
      command:
        "node node_modules/next/dist/bin/next start --hostname 127.0.0.1 -p 3101",
      url: "http://127.0.0.1:3101",
      reuseExistingServer: false,
      env: { BACKEND_URL: "http://127.0.0.1:8101" },
    },
  ],
});
