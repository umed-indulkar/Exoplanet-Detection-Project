import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "ExoVision AI — Discover New Worlds",
  description:
    "An AI-powered exoplanet detection platform. Upload stellar data, analyze light curves, and reveal hidden signals from distant star systems.",
  keywords: ["exoplanet", "AI", "space", "stellar data", "light curves", "transit detection"],
  openGraph: {
    title: "ExoVision AI — Discover New Worlds",
    description: "AI-powered exoplanet detection from stellar light curves.",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="antialiased">
      <body className="min-h-full">{children}</body>
    </html>
  );
}
