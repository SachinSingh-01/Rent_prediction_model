import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Rentwise — House Rent Predictor",
  description:
    "Explore a machine learning project for estimating monthly house rent across Indian cities.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
