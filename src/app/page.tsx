"use client";

import { useState } from "react";
import UploadArea from "@/components/UploadArea";
import ProductExample from "@/components/ProductExample";

export default function Home() {
  const [hasResults, setHasResults] = useState(false);

  return (
    <div className="min-h-screen flex flex-col justify-between bg-[#FAF9F6] text-[#20211F]">
      {/* A. Small Top Navigation */}
      <header className="w-full border-b border-[#E6E4DE] bg-[#FAF9F6]">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 h-14 flex items-center justify-between">
          <div className="flex items-center gap-1.5">
            <span className="text-base font-semibold tracking-tight text-[#20211F]">
              PitchProbe
            </span>
            <span className="text-[11px] font-medium text-[#73746F] tracking-wide border border-[#E6E4DE] bg-white px-1.5 py-0.5 rounded">
              NLP
            </span>
          </div>
          <span className="text-xs sm:text-sm text-[#73746F] font-normal">
            Pitch deck analysis
          </span>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 flex flex-col justify-center items-center px-4 sm:px-6 py-12 sm:py-16">
        <div className="w-full max-w-2xl text-center">
          {/* B. Main Heading & Supporting Text */}
          <div className="mb-8 sm:mb-10">
            <h1 className="text-2xl sm:text-4xl font-semibold tracking-tight text-[#20211F] mb-3 leading-tight sm:leading-tight">
              Know what an investor might question.
            </h1>
            <p className="text-sm sm:text-base text-[#73746F] max-w-lg mx-auto leading-relaxed">
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
      <footer className="w-full border-t border-[#E6E4DE] py-5 px-4 sm:px-6 bg-[#FAF9F6]">
        <div className="max-w-4xl mx-auto flex flex-col sm:flex-row items-center justify-between text-xs text-[#73746F] gap-2">
          <span>PitchProbe-NLP · First Version</span>
          <span>PDF text analysis for startup pitch decks</span>
        </div>
      </footer>
    </div>
  );
}
