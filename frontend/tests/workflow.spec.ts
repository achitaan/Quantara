import { expect, test } from "@playwright/test";

test("import, analyze, simulate, paper trade, forecast, research and explain", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("/");
  await page.getByRole("button", { name: "Enter workspace" }).click();
  await expect(
    page.getByRole("heading", { name: "Your research, connected." }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Load research demo" }).click();
  await expect(page.getByText(/Across 1 portfolios/)).toBeVisible();
  await page.screenshot({ path: "../runtime/overview.png", fullPage: true });
  await page
    .getByRole("navigation")
    .getByRole("button", { name: "Market data", exact: true })
    .click();
  await page
    .getByRole("textbox", { name: "Watchlist symbols", exact: true })
    .fill("AAPL,MSFT");
  await page
    .getByRole("button", { name: "Save watchlist", exact: true })
    .click();
  await expect(
    page.getByRole("textbox", { name: "Watchlist symbols", exact: true }),
  ).toHaveValue("AAPL,MSFT");
  await expect
    .poll(
      async () =>
        (await (await page.request.get("/api/settings")).json()).watchlist,
    )
    .toEqual(["AAPL", "MSFT"]);
  await page
    .getByRole("navigation")
    .getByRole("button", { name: "Portfolio", exact: true })
    .click();
  await page.getByRole("button", { name: "Run analysis", exact: true }).click();
  await expect(page.locator(".result")).toContainText("historical_var", {
    timeout: 30000,
  });
  await page
    .getByRole("combobox", { name: "Analysis", exact: true })
    .selectOption("optimize");
  await page
    .getByRole("combobox", { name: "Allocation method", exact: true })
    .selectOption("risk_parity");
  await page.getByRole("button", { name: "Run analysis", exact: true }).click();
  await expect(page.locator(".result")).toContainText("weights", {
    timeout: 30000,
  });
  await page
    .getByRole("navigation")
    .getByRole("button", { name: "Simulator", exact: true })
    .click();
  await page.getByRole("button", { name: "Run simulation" }).click();
  await expect(page.locator(".result")).toContainText("dataset_version", {
    timeout: 30000,
  });
  await page.screenshot({ path: "../runtime/simulator.png", fullPage: true });
  await page
    .getByRole("navigation")
    .getByRole("button", { name: "Paper trading", exact: true })
    .click();
  await page
    .getByRole("button", { name: "Create account", exact: true })
    .click();
  const paper = page.locator(".panel").filter({
    has: page.getByRole("heading", {
      name: "Strategy paper account",
      exact: true,
    }),
  });
  await paper.getByRole("button", { name: "Start", exact: true }).click();
  await paper
    .getByRole("button", { name: "Replay 20 bars", exact: true })
    .click();
  await expect(page.locator(".result")).toContainText('"steps": 20', {
    timeout: 30000,
  });
  await page
    .getByRole("navigation")
    .getByRole("button", { name: "Cash flow", exact: true })
    .click();
  await page
    .getByRole("button", { name: "Load synthetic cash-flow fixture" })
    .click();
  await page
    .getByRole("button", { name: "Forecast cash flow", exact: true })
    .click();
  await expect(page.locator(".result")).toContainText("shortfall_alerts", {
    timeout: 30000,
  });
  await page
    .getByRole("navigation")
    .getByRole("button", { name: "Tax research", exact: true })
    .click();
  await page.getByRole("button", { name: "Research tax positions" }).click();
  await expect(page.locator(".result")).toContainText("harvesting_proposals", {
    timeout: 30000,
  });
  await page
    .getByRole("navigation")
    .getByRole("button", { name: "Assistant", exact: true })
    .click();
  await page
    .getByPlaceholder("Ask about your recorded results…")
    .fill("Explain the recorded simulation and its limitations.");
  await page.getByRole("button", { name: "Send", exact: false }).click();
  await expect(page.locator(".message.assistant")).toBeVisible({
    timeout: 30000,
  });
  await expect(page.locator(".message.assistant")).toContainText(
    "calculations run independently",
  );
  expect(errors).toEqual([]);
});

test("mobile navigation and safe text rendering", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");
  await page.getByRole("button", { name: "Enter workspace" }).click();
  await expect(
    page.getByRole("heading", { name: "Your research, connected." }),
  ).toBeVisible();
  await page
    .getByRole("navigation")
    .getByRole("button", { name: "Assistant", exact: true })
    .click();
  await expect(
    page.getByRole("heading", { name: "Research assistant", exact: true }),
  ).toBeVisible();
  await page.screenshot({ path: "../runtime/mobile.png", fullPage: true });
});

test("appearance persists across reloads and preserves the watchlist", async ({
  page,
}) => {
  await page.goto("/");
  await page.getByRole("button", { name: "Enter workspace" }).click();
  await expect(
    page.getByRole("heading", { name: "Your research, connected." }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: /Switch to (light|dark) mode/ }),
  ).toBeEnabled();
  const before = (await (await page.request.get("/api/settings")).json())
    .watchlist;
  const light = page.getByRole("button", { name: "Switch to light mode" });
  if (await light.isVisible()) await light.click();
  await expect
    .poll(
      async () =>
        (await (await page.request.get("/api/settings")).json()).theme,
    )
    .toBe("light");
  await page.screenshot({ path: "../runtime/glass-light.png", fullPage: true });
  await page.getByRole("button", { name: "Switch to dark mode" }).click();
  await expect
    .poll(
      async () =>
        (await (await page.request.get("/api/settings")).json()).theme,
    )
    .toBe("dark");
  await page.screenshot({ path: "../runtime/glass-dark.png", fullPage: true });
  await page.reload();
  await expect(
    page.getByRole("button", { name: "Switch to light mode" }),
  ).toBeVisible();
  expect(
    (await (await page.request.get("/api/settings")).json()).watchlist,
  ).toEqual(before);
  await page.setViewportSize({ width: 390, height: 844 });
  await page.getByRole("button", { name: "Switch to light mode" }).click();
  await expect
    .poll(
      async () =>
        (await (await page.request.get("/api/settings")).json()).theme,
    )
    .toBe("light");
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth,
    ),
  ).toBeTruthy();
  await page.screenshot({
    path: "../runtime/glass-mobile.png",
    fullPage: true,
  });
});
