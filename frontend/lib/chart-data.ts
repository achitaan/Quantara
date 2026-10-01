export { record, rows } from "./client";
import type { Json } from "./client";
export const number = (v: Json | undefined) => (typeof v === "number" ? v : 0);
export const text = (v: Json | undefined) => (typeof v === "string" ? v : "");
