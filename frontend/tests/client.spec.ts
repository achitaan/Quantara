import { expect, test } from "@playwright/test";
import { api, upload } from "../lib/client";

for (const transport of ["api", "upload"] as const) {
  const send = () =>
    transport === "api"
      ? api("fixture", { name: "research" })
      : upload("fixture", new File(["symbol,quantity\nAAPL,1"], "ledger.csv"), {
          currency: "CAD",
        });

  test(`${transport} preserves JSON errors and handles text, HTML and empty HTTP failures`, async () => {
    const originalFetch = globalThis.fetch;
    const cases = [
      {
        status: 400,
        body: JSON.stringify({ detail: "Invalid ledger" }),
        contentType: "application/json",
        message: "Invalid ledger",
      },
      {
        status: 422,
        body: JSON.stringify({
          detail: [{ loc: ["body", "symbol"], msg: "Required" }],
        }),
        contentType: "application/json",
        message: JSON.stringify([{ loc: ["body", "symbol"], msg: "Required" }]),
      },
      {
        status: 502,
        body: "Upstream service unavailable",
        contentType: "text/plain",
        message: "HTTP 502 Bad Gateway: Upstream service unavailable",
      },
      {
        status: 503,
        body: "<!doctype html><h1>Service unavailable</h1>",
        contentType: "text/html",
        message: "HTTP 503 Service Unavailable",
      },
      {
        status: 503,
        body: "",
        contentType: "application/json",
        message: "HTTP 503 Service Unavailable",
      },
    ];
    try {
      for (const fixture of cases) {
        globalThis.fetch = async () =>
          new Response(fixture.body, {
            status: fixture.status,
            statusText:
              fixture.status === 502 ? "Bad Gateway" : "Service Unavailable",
            headers: { "Content-Type": fixture.contentType },
          });
        await expect(send()).rejects.toThrow(fixture.message);
      }
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  test(`${transport} keeps successful data and its request encoding`, async () => {
    const originalFetch = globalThis.fetch;
    let request: RequestInit | undefined;
    try {
      globalThis.fetch = async (_url, init) => {
        request = init;
        return Response.json({ id: "saved", metrics: { value: 42 } });
      };
      await expect(send()).resolves.toEqual({
        id: "saved",
        metrics: { value: 42 },
      });
      expect(request?.method).toBe("POST");
      if (transport === "api") {
        expect(request?.headers).toEqual({
          "Content-Type": "application/json",
        });
        expect(JSON.parse(String(request?.body))).toEqual({ name: "research" });
      } else {
        expect(request?.headers).toBeUndefined();
        expect(request?.body).toBeInstanceOf(FormData);
        const form = request?.body as FormData;
        expect((form.get("file") as File).name).toBe("ledger.csv");
        expect(JSON.parse(String(form.get("options")))).toEqual({
          currency: "CAD",
        });
      }
      globalThis.fetch = async () => new Response(null, { status: 204 });
      await expect(send()).resolves.toBeUndefined();
    } finally {
      globalThis.fetch = originalFetch;
    }
  });
}
