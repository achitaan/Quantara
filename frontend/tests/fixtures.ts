import {
  test as base,
  expect,
  type APIRequestContext,
  type Page,
} from "@playwright/test";

type AuthState = Awaited<ReturnType<APIRequestContext["storageState"]>>;

// Each test gets a fresh browser context, while its worker logs in only once.
// UI sign-in/sign-out tests create separate sessions so revocation stays isolated.
export const test = base.extend<{}, { authenticatedState: AuthState }>({
  authenticatedState: [
    async ({ playwright }, use, workerInfo) => {
      const request = await playwright.request.newContext({
        baseURL: workerInfo.project.use.baseURL,
      });
      try {
        const response = await request.post("/api/auth/login", {
          data: { username: "demo", password: process.env.TEAM_PASSWORD! },
        });
        expect(response.ok(), "Test session login must succeed").toBeTruthy();
        await use(await request.storageState());
      } finally {
        await request.dispose();
      }
    },
    { scope: "worker" },
  ],
  storageState: async ({ authenticatedState }, use) => use(authenticatedState),
});

export { expect };

export async function openWorkspace(page: Page) {
  await page.goto("/");
  await expect(page.locator(".current-hero h1")).toBeVisible();
}

export async function signIn(page: Page) {
  await page.context().clearCookies();
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
