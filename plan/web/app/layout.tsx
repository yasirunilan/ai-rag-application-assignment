import "./globals.css";
import type { ReactNode } from "react";

export const metadata = { title: "Manual Finder", description: "Multimodal retrieval over technical manuals" };

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
