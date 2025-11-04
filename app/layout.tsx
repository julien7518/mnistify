import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import { Analytics } from "@vercel/analytics/next";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "MNISTify",
  description: "Draw a digit, let MNISTify predict it. AI meets handwritten recognition in a simple, interactive demo.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <Analytics />
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased bg-gray-50`}
      >
        {children}
        <footer className="flex w-full justify-around text-sm text-gray-500 pb-6">
          <p>Made by <a className="underline" target="_blank" href="https://github.com/julien7518">me</a></p>
        </footer>
      </body>
    </html>
  );
}
