"use client";

import { useState } from "react";
import UploadArea from "@/components/UploadArea";
import ProductExample from "@/components/ProductExample";

export default function Home() {
  const [hasResults, setHasResults] = useState(false);

  return (
    <div className="app-shell min-h-screen flex flex-col justify-between bg-[#F3F6F1] text-[#202922]">
      {/* A. Small Top Navigation */}
      <header className="site-header w-full border-b border-[#DCE4D9] bg-[#F3F6F1]">
        <div className="site-header-inner mx-auto w-full px-4 sm:px-6 flex items-center justify-between">
          <div className="brand-lockup flex items-center">
            <span className="brand-name text-base font-semibold tracking-tight text-[#202922]">
              PitchProbe
            </span>
            <span className="product-tag text-[11px] font-medium text-[#6F7970] tracking-wide border border-[#DCE4D9] bg-white px-1.5 py-0.5 rounded">
              NLP
            </span>
          </div>
          <span className="header-descriptor text-xs sm:text-sm text-[#6F7970] font-normal">
            Pitch deck analysis
          </span>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="main-stage flex-1 flex flex-col justify-center items-center px-4 sm:px-6">
        <div className={`home-content w-full ${hasResults ? "home-content-results" : ""}`}>
          {/* B. Main Heading & Supporting Text */}
          <div className="home-intro">
            <p className="hero-kicker">Pitch deck due diligence</p>
            <h1 className="hero-title text-2xl sm:text-4xl font-semibold tracking-tight text-[#202922] mb-3 leading-tight sm:leading-tight">
              Know what an investor might question.
            </h1>
            <p className="hero-description text-sm sm:text-base text-[#6F7970] max-w-lg mx-auto leading-relaxed">
              Upload your startup pitch deck to spot vague or exaggerated claims
              and see the questions an investor might ask.
            </p>
          </div>

          {/* C. Pitch Deck Upload Area */}
          <UploadArea onResultsChange={setHasResults} />

          {/* D. Small Product Example (preview shown when idle, hidden when live results are rendered) */}
          {!hasResults && <ProductExample />}
        </div>
      </main>

      {/* Minimal Footer */}
      <footer className="site-footer w-full border-t border-[#DCE4D9] py-5 px-4 sm:px-6 bg-[#F3F6F1]">
        <div className="site-footer-inner max-w-4xl mx-auto w-full flex flex-col sm:flex-row items-center justify-between text-xs text-[#6F7970] gap-2">
          <span>PitchProbe-NLP · First Version</span>
          <span>PDF text analysis for startup pitch decks</span>
        </div>
      </footer>
    </div>
  );
}
