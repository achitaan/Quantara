import type { NextConfig } from "next";
const config: NextConfig = {
  output: process.env.QUANTARA_STANDALONE === "true" ? "standalone" : undefined,
  webpack(configuration) {
    if (process.env.QUANTARA_DISABLE_BUILD_CACHE === "true")
      configuration.cache = false;
    return configuration;
  },
  poweredByHeader: false,
  async headers() {
    return [
      {
        source: "/:path*",
        headers: [
          { key: "X-Content-Type-Options", value: "nosniff" },
          { key: "Referrer-Policy", value: "same-origin" },
          { key: "X-Frame-Options", value: "DENY" },
        ],
      },
    ];
  },
};
export default config;
