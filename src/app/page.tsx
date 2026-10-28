"use client";

import { useState } from "react";
import UploadArea from "@/components/UploadArea";
import ProductExample from "@/components/ProductExample";

export default function Home() {
  const [hasResults, setHasResults] = useState(false);
  const [showUploader, setShowUploader] = useState(false);

  return (
    <div className="app-shell">
      <header className="site-header">
        <div className="site-header-inner">
          <div className="brand-lockup" aria-label="PitchProbe">
            <svg className="brand-logo" viewBox="0 0 40 40" role="img" aria-label="PitchProbe logo">
              <rect x="1" y="1" width="38" height="38" rx="10" fill="#315B41" />
              <path d="M10 26.5 16.4 19l4.4 4.2L30 12.5" fill="none" stroke="#FFFFFF" strokeWidth="2.3" strokeLinecap="round" strokeLinejoin="round" />
              <path d="M23.8 12.5H30v6.2" fill="none" stroke="#FFFFFF" strokeWidth="2.3" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
            <span className="brand-name">PitchProbe</span>
          </div>
        </div>
      </header>

      <main className={`main-stage ${showUploader ? "main-stage--workflow" : ""}`}>
        {!showUploader ? (
          <div className="landing-shell">
            <section className="hero-section" aria-labelledby="hero-title">
              <div className="hero-copy">
                <p className="hero-kicker">Investor-readiness review</p>
                <h1 className="hero-title" id="hero-title">Know what an investor might question.</h1>
                <p className="hero-description">
                  A sharper read on your pitch deck. Surface vague claims, see where the risk sits,
                  and get the questions an investor is likely to ask.
                </p>
              </div>

              <div className="process-strip" aria-label="How PitchProbe works">
                <div className="process-step">
                  <div>
                    <p className="process-title">Scan the claims</p>
                    <p className="process-copy">Pull out the statements that need a closer look.</p>
                  </div>
                </div>
                <div className="process-step">
                  <div>
                    <p className="process-title">Rank by risk</p>
                    <p className="process-copy">Separate high-risk claims from more grounded ones.</p>
                  </div>
                </div>
                <div className="process-step">
                  <div>
                    <p className="process-title">Ask sharper questions</p>
                    <p className="process-copy">Prepare for the evidence an investor may request.</p>
                  </div>
                </div>
              </div>

              <div className="hero-action">
                <button type="button" className="landing-cta" onClick={() => setShowUploader(true)}>
                  Analyse pitch deck
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" aria-hidden="true">
                    <path d="M5 12h14m-6-6 6 6-6 6" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                </button>
              </div>
            </section>

            <ProductExample />
          </div>
        ) : (
          <div className={`workflow-shell ${hasResults ? "workflow-shell--results" : ""}`}>
            <div className="workflow-heading">
              <p className="section-kicker">{hasResults ? "ANALYSIS REPORT" : "PITCH DECK REVIEW"}</p>
              <h1>{hasResults ? "Your claims, reviewed." : "Analyse your pitch deck"}</h1>
              <p>
                {hasResults
                  ? "Browse claims by topic and risk. Higher-risk claims appear first within each section."
                  : "Upload a PDF to see which claims may raise questions and what evidence an investor may ask for."}
              </p>
            </div>
            <UploadArea onResultsChange={setHasResults} />
          </div>
        )}
      </main>
    </div>
  );
}
