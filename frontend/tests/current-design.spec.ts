import { expect, test, type Page } from "@playwright/test";
import type { Portfolio } from "../lib/client";

const portfolios: Portfolio[] = [
  {
    id: "current-usd",
    name: "Core portfolio",
    currency: "USD",
    balance: {
      cash: 63500,
      currency: "USD",
      holdings: { AAPL: 100.125, MSFT: 50 },
      accounts: {},
    },
    transactions: [
      {
        external_id: "usd-buy",
        timestamp: "2024-01-03T14:30:00Z",
        type: "buy",
        symbol: "AAPL",
        quantity: 100.125,
        price: 180,
        amount: 0,
        fee: 0,
        currency: "USD",
        account: "main",
        account_type: "taxable",
        affiliated: false,
      },
    ],
  },
  {
    id: "current-cad",
    name: "Canadian portfolio",
    currency: "CAD",
    balance: {
      cash: 18250.37,
      currency: "CAD",
      holdings: { RY: 12.75 },
      accounts: {},
    },
    transactions: [
      {
        external_id: "cad-buy",
        timestamp: "2024-01-04T14:30:00Z",
        type: "buy",
        symbol: "RY",
        quantity: 12.75,
        price: 130,
        amount: 0,
        fee: 0,
        currency: "CAD",
        account: "main",
        account_type: "taxable",
        affiliated: false,
      },
    ],
  },
];

async function signIn(page: Page) {
  await page.goto("/");
  await page
    .getByRole("textbox", { name: "Username", exact: true })
    .fill("demo");
  await page
    .getByLabel("Password", { exact: true })
    .fill(process.env.TEAM_PASSWORD!);
  await page.getByRole("button", { name: "Enter workspace" }).click();
  await expect(page.locator(".current-hero h1")).toBeVisible();
}

test("Current uses the active portfolio currency and fractional holdings", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.route("**/api/portfolios", (route) =>
    route.fulfill({ json: portfolios }),
  );
  await page.route("**/api/settings", (route) =>
    route.fulfill({ json: { theme: "light", watchlist: ["NVDA", "GOOGL"] } }),
  );
  await page.route("**/api/backtests", (route) =>
    route.fulfill({
      json: [
        {
          id: "current-study",
          name: "Recorded fixture study",
          engine_version: 2,
          state: {
            equity: [
              { timestamp: "2024-01-01T14:30:00Z", equity: 100000 },
              { timestamp: "2024-02-01T14:30:00Z", equity: 101000 },
              { timestamp: "2024-03-01T14:30:00Z", equity: 98000 },
              { timestamp: "2024-04-01T14:30:00Z", equity: 102000 },
            ],
          },
          metrics: { total_return: 0.02, max_drawdown: 0.0297, sharpe: 0.4 },
        },
      ],
    }),
  );
  await signIn(page);
  await expect(page.locator(".current-hero h1")).toHaveText("Core portfolio");
  await expect(page.locator(".current-cash")).toContainText("63,500");
  await expect(page.locator(".current-cash")).toContainText("USD");
  await expect(page.locator(".current-cash")).toContainText(
    "Excludes holdings value",
  );
  const holdings = page.getByRole("table", {
    name: "Recorded holdings for Core portfolio",
  });
  await expect(
    holdings.getByRole("row").filter({ hasText: "AAPL" }),
  ).toContainText("100.125");
  await holdings.getByRole("button", { name: "Shares", exact: true }).click();
  await expect(
    holdings.getByRole("columnheader", { name: "Shares", exact: true }),
  ).toHaveAttribute("aria-sort", "ascending");
  await expect(holdings.locator("tbody tr").first()).toContainText("MSFT");
  await holdings.getByRole("button", { name: "Shares", exact: true }).click();
  await expect(holdings.locator("tbody tr").first()).toContainText("AAPL");
  await holdings.getByRole("button", { name: "Asset", exact: true }).click();
  await expect(
    page.getByRole("img", { name: "Simulated equity", exact: true }),
  ).toBeVisible();
  const axisSize = () =>
    page
      .getByRole("img", { name: "Simulated equity", exact: true })
      .evaluate((element) => {
        const svg = element as unknown as SVGSVGElement;
        return (
          (parseFloat(getComputedStyle(svg.querySelector("text")!).fontSize) *
            svg.getBoundingClientRect().width) /
          svg.viewBox.baseVal.width
        );
      });
  await expect.poll(axisSize).toBeGreaterThanOrEqual(10.5);
  await page.getByRole("slider", { name: "Inspect data point" }).press("Home");
  await expect(page.locator(".current-simulation")).toContainText("100,000");
  await page.locator(".current-hero h1").click();
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.screenshot({
    path: "../runtime/current-desktop.png",
    fullPage: true,
  });

  await page
    .getByRole("combobox", { name: "Active portfolio", exact: true })
    .selectOption("current-cad");
  await expect(page.locator(".current-hero h1")).toHaveText(
    "Canadian portfolio",
  );
  await expect(page.locator(".current-cash")).toContainText("18,250.37");
  await expect(page.locator(".current-cash")).toContainText("CAD");
  await expect(
    page.getByRole("table", {
      name: "Recorded holdings for Canadian portfolio",
    }),
  ).toContainText("12.75");
  await expect(page.locator(".holdings-table")).not.toContainText("AAPL");
  await page
    .locator(".current-section-tabs")
    .getByRole("button", { name: "Activity", exact: true })
    .click();
  await expect(page.locator("#current-view")).toContainText("RY");
  await page
    .locator(".current-section-tabs")
    .getByRole("button", { name: "Watchlist", exact: true })
    .click();
  await expect(page.locator("#current-view")).toContainText("NVDA");
  await page
    .locator(".current-section-tabs")
    .getByRole("button", { name: "Holdings", exact: true })
    .click();
  await page
    .getByRole("combobox", { name: "Active portfolio", exact: true })
    .selectOption("current-usd");
  await page.setViewportSize({ width: 375, height: 812 });
  await expect(page.locator(".current-hero h1")).toBeVisible();
  await expect
    .poll(() =>
      page.evaluate(
        () => document.documentElement.scrollWidth <= window.innerWidth,
      ),
    )
    .toBeTruthy();
  await expect.poll(axisSize).toBeGreaterThanOrEqual(10.5);
  await page.locator(".current-hero h1").click();
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.screenshot({
    path: "../runtime/current-mobile.png",
    fullPage: true,
  });
  await page
    .getByRole("navigation", { name: "Workspace" })
    .getByRole("button", { name: "Portfolio", exact: true })
    .click();
  await expect(
    page.getByRole("heading", { name: "Your portfolios", exact: true }),
  ).toBeVisible();
  await expect(
    page.getByRole("combobox", { name: "Active portfolio", exact: true }),
  ).toHaveValue("current-usd");
  expect(errors).toEqual([]);
});

test("Current empty state offers a working portfolio creation path", async ({
  page,
}) => {
  await page.route("**/api/portfolios", (route) => route.fulfill({ json: [] }));
  await signIn(page);
  await expect(page.locator(".current-cash")).not.toContainText("$0");
  await expect(
    page.getByRole("heading", {
      name: "Make room for your portfolio",
      exact: true,
    }),
  ).toBeVisible();
  await page
    .getByRole("button", { name: "Create portfolio", exact: true })
    .click();
  await expect(
    page.getByRole("heading", { name: "Create a portfolio", exact: true }),
  ).toBeVisible();
});

test("all research tools remain reachable on desktop and mobile", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await signIn(page);
  const sections = [
    "Overview",
    "Portfolio",
    "Market data",
    "Strategies",
    "Simulator",
    "Paper trading",
    "Signals",
    "Cash flow",
    "Tax research",
    "Models",
    "Assistant",
  ];
  for (const width of [1440, 375]) {
    await page.setViewportSize({ width, height: width === 1440 ? 1000 : 812 });
    for (const section of sections) {
      await page
        .getByRole("navigation", { name: "Workspace" })
        .getByRole("button", { name: section, exact: true })
        .click();
      await expect(page.locator("main h1")).toBeVisible();
      await expect
        .poll(() =>
          page.evaluate(
            () => document.documentElement.scrollWidth <= window.innerWidth,
          ),
        )
        .toBeTruthy();
      if (width === 1440)
        await page.screenshot({
          path:
            "../runtime/current-" +
            section.toLowerCase().replaceAll(" ", "-") +
            "-screen.png",
          fullPage: true,
        });
    }
  }
  expect(errors).toEqual([]);
});

test("successful sign out returns to login without refreshing protected data", async ({
  page,
}) => {
  await signIn(page);
  await page.getByLabel("Account menu", { exact: true }).click();
  let postLogoutRefreshes = 0;
  page.on("request", (request) => {
    if (request.url().endsWith("/api/portfolios")) postLogoutRefreshes++;
  });
  await page.getByRole("button", { name: "Sign out", exact: true }).click();
  await expect(page.getByLabel("Password", { exact: true })).toBeVisible();
  await expect(page.locator("main").getByRole("alert")).toHaveCount(0);
  expect(postLogoutRefreshes).toBe(0);
});

test("failed sign out keeps the workspace available and reports the failure", async ({
  page,
}) => {
  await signIn(page);
  await page.route("**/api/auth/logout", (route) =>
    route.fulfill({
      status: 503,
      json: { detail: "Sign out temporarily unavailable" },
    }),
  );
  await page.getByLabel("Account menu", { exact: true }).click();
  await page.getByRole("button", { name: "Sign out", exact: true }).click();
  await expect(page.locator("main").getByRole("alert")).toContainText(
    "Sign out temporarily unavailable",
  );
  await expect(page.locator(".current-hero h1")).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Sign out", exact: true }),
  ).toBeEnabled();
});

for (const failure of ["network", "server"] as const) {
  test(`appearance stays usable after a ${failure} save failure`, async ({
    page,
  }) => {
    let unavailable = true;
    let settings = { theme: "light", watchlist: ["NVDA"] };
    await page.route("**/api/settings", async (route) => {
      if (route.request().method() === "PUT") {
        if (unavailable) {
          if (failure === "network") await route.abort("failed");
          else
            await route.fulfill({
              status: 503,
              json: { detail: "Service unavailable" },
            });
          return;
        }
        settings = route.request().postDataJSON();
      }
      await route.fulfill({ json: settings });
    });
    await signIn(page);
    await page.getByLabel("Account menu", { exact: true }).click();
    await page
      .getByRole("button", { name: "Switch to dark mode", exact: true })
      .click();
    await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
    await expect(page.locator("main").getByRole("alert")).toContainText(
      "Appearance changed on this device",
    );
    await expect(
      page.getByRole("button", { name: "Switch to light mode", exact: true }),
    ).toBeEnabled();

    await page.getByRole("button", { name: /Refresh/, exact: false }).click();
    await expect(
      page.getByRole("button", { name: /Refresh/, exact: false }),
    ).toBeEnabled();
    await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
    await page.reload();
    await expect(page.locator(".current-hero h1")).toBeVisible();
    await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
    await expect
      .poll(() => page.evaluate(() => localStorage.getItem("quantara-theme")))
      .toBe("dark");

    unavailable = false;
    await page.getByLabel("Account menu", { exact: true }).click();
    await page
      .getByRole("button", { name: "Switch to light mode", exact: true })
      .click();
    await expect.poll(() => settings.theme).toBe("light");
    await expect(
      page.getByRole("button", { name: "Switch to dark mode", exact: true }),
    ).toBeEnabled();
    await expect(page.locator("html")).toHaveAttribute("data-theme", "light");
    await expect(page.locator("main").getByRole("alert")).toHaveCount(0);
    expect(settings.watchlist).toEqual(["NVDA"]);
  });
}
