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
  const value = await response.json();
  if (!response.ok)
    throw new Error(
      typeof value.detail === "string"
        ? value.detail
        : JSON.stringify(value.detail || value),
    );
  return value as T;
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
  const value = await response.json();
  if (!response.ok)
    throw new Error(
      typeof value.detail === "string"
        ? value.detail
        : JSON.stringify(value.detail || value),
    );
  return value as T;
}
