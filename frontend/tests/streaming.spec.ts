import { expect, test } from "@playwright/test";

test("assistant shows a streamed preview and then cited document evidence", async ({
  page,
}) => {
  await page.goto("/");
  await page.getByRole("button", { name: "Enter workspace" }).click();
  await expect(
    page.getByRole("heading", { name: "Your research, connected." }),
  ).toBeVisible();
  await page
    .getByRole("navigation")
    .getByRole("button", { name: "Assistant", exact: true })
    .click();
  const job = {
    id: "stream-preview-test",
    operation: "chat",
    arguments: {},
    status: "running",
    progress: 0.2,
    message: "Requesting local explanation",
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
    error: null,
    result: null,
    details: {
      draft: "The document says",
      phase: "generating",
      retrieval_mode: "local_embeddings",
      source_count: 1,
    },
  };
  const result = {
    conversation_id: "preview-conversation",
    role: "assistant",
    content: "The document states a reserve target of CAD 100 [doc:test:p4].",
    retrieval_mode: "local_embeddings",
    latency_seconds: 2,
    sources: [
      {
        document_id: "test",
        name: "Reserve research note",
        page: 4,
        text: "The research cash reserve target is CAD 100.",
        citation: "doc:test:p4",
      },
    ],
    tools: [],
  };
  let finished = false;
  await page.route("**/api/conversations/chat", async (route) =>
    route.fulfill({ json: job }),
  );
  await page.route("**/api/jobs?include_results=false", async (route) =>
    route.fulfill({
      json: [{ ...job, status: finished ? "complete" : "running" }],
    }),
  );
  await page.route(
    "**/api/jobs/stream-preview-test/events?include_results=false",
    async (route) =>
      route.fulfill({
        contentType: "text/event-stream",
        body: "data: " + JSON.stringify(job) + "\n\n",
      }),
  );
  await page.route("**/api/jobs/stream-preview-test", async (route) =>
    route.fulfill({ json: { ...job, status: "complete", result } }),
  );
  await page
    .getByPlaceholder("Ask about your recorded results…")
    .fill("According to the document, what is the reserve target?");
  await page.getByRole("button", { name: /Send/ }).click();
  await expect(page.locator(".streaming-reply")).toContainText(
    "Reply preview — citations checked when complete",
  );
  await expect(page.locator(".streaming-reply")).toContainText(
    "The document says",
  );
  finished = true;
  const reply = page.locator(".message.assistant:not(.streaming-reply)");
  await expect(reply).toContainText("CAD 100");
  await expect(page.locator(".streaming-reply")).toHaveCount(0);
  await expect(reply).toContainText("Retrieval: local embeddings");
  await reply.getByText("Document sources used", { exact: true }).click();
  await expect(
    reply.getByText("Reserve research note · page 4", { exact: true }),
  ).toBeVisible();
  await expect(
    reply.getByText("The research cash reserve target is CAD 100.", {
      exact: true,
    }),
  ).toBeVisible();
});
