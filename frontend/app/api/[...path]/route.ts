import { NextRequest } from "next/server";
export const dynamic = "force-dynamic";
async function proxy(
  request: NextRequest,
  context: { params: Promise<{ path: string[] }> },
) {
  const { path } = await context.params;
  if (path.some((p) => !/^[a-zA-Z0-9._:-]+$/.test(p) || p === "..")) {
    return Response.json({ detail: "Invalid API path" }, { status: 400 });
  }
  const base = process.env.BACKEND_URL || "http://127.0.0.1:8000";
  const headers = new Headers();
  for (const key of [
    "content-type",
    "cookie",
    "origin",
    "authorization",
    "accept",
  ]) {
    const value = request.headers.get(key);
    if (value) headers.set(key, value);
  }
  try {
    const response = await fetch(
      base + "/api/v1/" + path.join("/") + request.nextUrl.search,
      {
        method: request.method,
        headers,
        body: ["GET", "HEAD"].includes(request.method)
          ? undefined
          : await request.arrayBuffer(),
        cache: "no-store",
        signal: request.signal,
      },
    );
    const output = new Headers();
    for (const key of [
      "content-type",
      "set-cookie",
      "content-disposition",
      "cache-control",
    ]) {
      const value = response.headers.get(key);
      if (value) output.set(key, value);
    }
    return new Response(response.body, {
      status: response.status,
      headers: output,
    });
  } catch {
    return Response.json(
      {
        detail:
          "The research backend is unavailable. Start the API service and retry.",
      },
      { status: 503 },
    );
  }
}
export { proxy as GET, proxy as POST, proxy as PUT, proxy as DELETE };
