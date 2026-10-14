import type { components } from "./api-types";
export type Portfolio = components["schemas"]["PortfolioView"];
export type Market = components["schemas"]["MarketInfo"];
export type Strategy = components["schemas"]["StrategyView"];
export type Job = components["schemas"]["JobView"];
export type Json =
  | null
  | boolean
  | number
  | string
  | Json[]
  | { [key: string]: Json };
export type RecordData = { id: string; [key: string]: Json };
export function record(value: unknown): Record<string, Json> {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, Json>)
    : {};
}
export function rows(value: unknown): RecordData[] {
  return Array.isArray(value) ? (value as RecordData[]) : [];
}
async function decodeResponse<T>(response: Response): Promise<T> {
  const body = (await response.text()).trim();
  let value: unknown;
  let isJson = false;
  if (body) {
    try {
      value = JSON.parse(body);
      isJson = true;
    } catch {
      // Proxies and unavailable services can return text or HTML instead of JSON.
    }
  }
  const status = `HTTP ${response.status}${response.statusText ? " " + response.statusText : ""}`;
  if (!response.ok) {
    if (isJson) {
      const detail = record(value).detail ?? value;
      if (typeof detail === "string" && detail) throw new Error(detail);
      if (detail !== null && detail !== undefined)
        throw new Error(JSON.stringify(detail));
    }
    const isHtml =
      response.headers.get("content-type")?.includes("text/html") ||
      body.startsWith("<");
    const message =
      !isHtml && body ? body.replace(/\s+/g, " ").slice(0, 500) : "";
    throw new Error(status + (message ? ": " + message : ""));
  }
  if (body && !isJson) throw new Error(status + ": Expected a JSON response.");
  return value as T;
}
export async function api<T>(
  path: string,
  body?: unknown,
  method = body ? "POST" : "GET",
): Promise<T> {
  const response = await fetch("/api/" + path, {
    method,
    headers: body ? { "Content-Type": "application/json" } : {},
    body: body ? JSON.stringify(body) : undefined,
  });
  return decodeResponse<T>(response);
}
export async function upload<T>(
  path: string,
  file: File,
  options?: object,
): Promise<T> {
  const form = new FormData();
  form.set("file", file);
  if (options) form.set("options", JSON.stringify(options));
  const response = await fetch("/api/" + path, { method: "POST", body: form });
  return decodeResponse<T>(response);
}
