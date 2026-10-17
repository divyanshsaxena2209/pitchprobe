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
      <body className="min-h-full flex flex-col bg-[#FAF9F6] text-[#20211F] font-sans selection:bg-[#426B50]/15 selection:text-[#20211F]">
        {children}
      </body>
    </html>
  );
}
