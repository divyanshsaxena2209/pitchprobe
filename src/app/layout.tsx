import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  display: "swap",
  variable: "--font-inter",
});

export const metadata: Metadata = {
  title: "PitchProbe — Know what an investor might question",
  description:
    "Upload your startup pitch deck to spot vague or exaggerated claims and see the questions an investor might ask.",
  icons: {
    icon: "/favicon.ico",
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={`${inter.variable} h-full antialiased`}>
      <body className="min-h-full flex flex-col bg-[#F3F6F1] text-[#202922] font-sans selection:bg-[#315B41]/15 selection:text-[#202922]">
        {children}
      </body>
    </html>
  );
}
