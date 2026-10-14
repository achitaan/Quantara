import { expect, test, type Page } from "@playwright/test";

async function mockWorkspace(page: Page) {
  await page.route("**/api/**", (route) => {
    const path = new URL(route.request().url()).pathname.replace("/api/", "");
    const data: Record<string, unknown> = {
      "auth/me": { username: "demo" },
      settings: { theme: "light", watchlist: ["NVDA"] },
      health: {},
      "cashflow/transactions": { transactions: [] },
      reports: [
        {
          id: "snapshot-report",
          name: "saved_snapshot",
          result: {
            metrics: { fixture_metric: 42 },
            marker: "retained evidence",
          },
        },
      ],
    };
    return route.fulfill({ json: data[path] ?? [] });
  });
}

test("manual refresh loads once and retains recorded evidence across success and failure", async ({
  page,
}) => {
  await mockWorkspace(page);
  let portfolioLoads = 0;
  let fail = false;
  let releaseFailure: (() => void) | undefined;
  const failureGate = new Promise<void>((resolve) => {
    releaseFailure = resolve;
  });
  await page.route("**/api/portfolios", async (route) => {
    portfolioLoads++;
    if (fail) {
      await failureGate;
      await route.fulfill({
        status: 503,
        contentType: "text/html",
        body: "<h1>Unavailable</h1>",
      });
    } else await route.fulfill({ json: [] });
  });
  await page.goto("/");
  await expect(page.locator(".current-hero h1")).toBeVisible();
  await page
    .getByRole("navigation", { name: "Workspace" })
    .getByRole("button", { name: "Portfolio", exact: true })
    .click();
  await page.getByText("Review saved analyses", { exact: true }).click();
  await page.getByRole("button", { name: "Saved snapshot · snapsh" }).click();
  const result = page.locator(".panel").filter({
    has: page.getByRole("heading", { name: "Recorded result", exact: true }),
  });
  await result
    .getByText("Recorded result and provenance", { exact: true })
    .click();
  await expect(result).toContainText("retained evidence");
  expect(portfolioLoads).toBe(1);
  await page.getByLabel("Account menu", { exact: true }).click();
  const refresh = page.getByRole("button", { name: /Refresh/ });
  await refresh.focus();
  await refresh.press("Enter");
  await expect.poll(() => portfolioLoads).toBe(2);
  await expect(refresh).toBeEnabled();
  expect(portfolioLoads).toBe(2);
  await expect(result).toContainText("retained evidence");

  fail = true;
  await refresh.click();
  await expect(refresh).toBeDisabled();
  await expect(result).toContainText("retained evidence");
  releaseFailure!();
  await expect(page.locator("main").getByRole("alert")).toContainText(
    "HTTP 503",
  );
  await expect(refresh).toBeEnabled();
  expect(portfolioLoads).toBe(3);
  await expect(result).toContainText("retained evidence");
});

test("authenticated bootstrap reports a workspace load failure and can recover", async ({
  page,
}) => {
  await mockWorkspace(page);
  let unavailable = true;
  await page.route("**/api/portfolios", (route) =>
    unavailable
      ? route.fulfill({
          status: 503,
          json: { detail: "Workspace storage unavailable" },
        })
      : route.fulfill({ json: [] }),
  );
  await page.goto("/");
  await expect(page.locator("main").getByRole("alert")).toContainText(
    "Workspace storage unavailable",
  );
  await expect(page.locator(".current-hero h1")).toBeVisible();
  await expect(page.getByLabel("Password", { exact: true })).toHaveCount(0);
  unavailable = false;
  await page.getByLabel("Account menu", { exact: true }).click();
  await page.getByRole("button", { name: /Refresh/ }).click();
  await expect(page.getByRole("button", { name: /Refresh/ })).toBeEnabled();
  await expect(page.locator("main").getByRole("alert")).toHaveCount(0);
});

test("expired authentication returns to usable sign-in without loading the workspace", async ({
  page,
}) => {
  await mockWorkspace(page);
  let portfolioLoads = 0;
  await page.route("**/api/portfolios", (route) => {
    portfolioLoads++;
    return route.fulfill({ json: [] });
  });
  await page.route("**/api/auth/me", (route) =>
    route.fulfill({ status: 401, json: { detail: "Not authenticated" } }),
  );
  await page.goto("/");
  await expect(page.getByLabel("Password", { exact: true })).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Enter workspace" }),
  ).toBeEnabled();
  await expect(page.locator("main").getByRole("alert")).toHaveCount(0);
  expect(portfolioLoads).toBe(0);
});
