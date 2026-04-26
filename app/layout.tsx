import type { Metadata } from "next";
import type { ReactNode } from "react";
import { IBM_Plex_Mono, IBM_Plex_Sans, JetBrains_Mono } from "next/font/google";
import "./globals.css";

const jetBrains = JetBrains_Mono({
  variable: "--font-display",
  subsets: ["latin"],
  weight: ["500", "700"],
});

const ibmSans = IBM_Plex_Sans({
  variable: "--font-body",
  subsets: ["latin"],
  weight: ["400", "500", "600"],
});

const ibmMono = IBM_Plex_Mono({
  variable: "--font-mono-custom",
  subsets: ["latin"],
  weight: ["400", "500", "600"],
});

export const metadata: Metadata = {
  title: "ML-Labs Command Shell",
  description: "Autonomous machine-learning lab with source resolution, experiment graph, and exportable artifacts.",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html
      lang="en"
      className={`${jetBrains.variable} ${ibmSans.variable} ${ibmMono.variable}`}
    >
      <body>{children}</body>
    </html>
  );
}
