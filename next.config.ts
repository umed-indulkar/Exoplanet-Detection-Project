import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: 'export',  // Add this line
  images: {
    unoptimized: true, // Required for static export if you use Next/Image
  },
};

export default nextConfig;
