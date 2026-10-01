import type { Metadata } from "next";
import "./globals.css";
export const metadata: Metadata = {
  title: "Quantara · Research workspace",
  description: "Local-first portfolio research and strategy simulation",
};
export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
