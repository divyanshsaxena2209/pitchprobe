"use client";

import { useState, useRef, ChangeEvent, DragEvent } from "react";

function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export interface SentenceAnalysis {
  id: number;
  text: string;
  risk: "Low" | "Moderate" | "High";
  score: number;
  question: string | null;
}

export interface RiskSummary {
  total_sentences: number;
  low_risk: number;
  moderate_risk: number;
  high_risk: number;
}

export interface PipelineInfo {
  classifier: string;
  generator: string;
  mode: string;
  note: string;
  document_warning?: string | null;
}

export interface AnalysisResult {
  filename: string;
  total_sentences: number;
  summary: RiskSummary;
  sentences: SentenceAnalysis[];
  pipeline_info: PipelineInfo;
}

interface UploadAreaProps {
  onResultsChange?: (hasResults: boolean) => void;
}

export default function UploadArea({ onResultsChange }: UploadAreaProps) {
  const [file, setFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [riskFilter, setRiskFilter] = useState<"ALL" | "High" | "Moderate" | "Low">("ALL");

  const fileInputRef = useRef<HTMLInputElement>(null);

  const validateAndSetFile = (selectedFile: File) => {
    setErrorMessage(null);

    // Validate MIME type or file extension
    const isPdf =
      selectedFile.type === "application/pdf" ||
      selectedFile.name.toLowerCase().endsWith(".pdf");

    if (!isPdf) {
      setErrorMessage(
        "Please select a PDF file. Other file formats are not supported."
      );
      setFile(null);
      return;
    }

    // Strict 20 MB size limit per MVP requirements
    if (selectedFile.size > 20 * 1024 * 1024) {
      setErrorMessage("File is too large. PitchProbe limits uploads to 20 MB.");
      setFile(null);
      return;
    }

    setFile(selectedFile);
  };

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    const selected = e.target.files?.[0];
    if (selected) {
      validateAndSetFile(selected);
    }
  };

  const handleDragEnter = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    if (!isDragging) setIsDragging(true);
  };

  const handleDragLeave = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.currentTarget.contains(e.relatedTarget as Node)) return;
    setIsDragging(false);
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const droppedFile = e.dataTransfer.files[0];
      validateAndSetFile(droppedFile);
    }
  };

  const handleRemove = () => {
    setFile(null);
    setErrorMessage(null);
    setAnalysisResult(null);
    onResultsChange?.(false);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const handleChooseClick = () => {
    fileInputRef.current?.click();
  };

  const handleAnalyzeClick = async () => {
    if (!file) return;

    setIsAnalyzing(true);
    setErrorMessage(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      // Send long-running analysis directly to FastAPI. The Next.js rewrite
      // proxy can reset very slow requests (CPU analysis of a full deck may
      // take over two minutes). The launcher writes this port to .env.local;
      // restart Next.js after the backend selects a different port.
      const backendBaseUrl = process.env.NEXT_PUBLIC_BACKEND_URL?.replace(/\/+$/, "");
      const analyzeUrl = backendBaseUrl
        ? `${backendBaseUrl}/api/analyze`
        : "/api/analyze";

      const res = await fetch(analyzeUrl, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        let errDetail = "Analysis request failed.";
        try {
          const errData = await res.json();
          if (errData.detail) errDetail = errData.detail;
        } catch {
          errDetail = `Server returned error (${res.status} ${res.statusText})`;
        }
        throw new Error(errDetail);
      }

      const data: AnalysisResult = await res.json();
      setAnalysisResult(data);
      onResultsChange?.(true);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "An unexpected error occurred.";
      if (msg.includes("Failed to fetch") || msg.includes("NetworkError")) {
        setErrorMessage(
          "Could not reach the PitchProbe backend. Check that it is running and that NEXT_PUBLIC_BACKEND_URL in .env.local matches BACKEND_URL, then restart Next.js."
        );
      } else {
        setErrorMessage(msg);
      }
    } finally {
      setIsAnalyzing(false);
    }
  };

  const filteredSentences = analysisResult
    ? analysisResult.sentences
        .filter((s) => (riskFilter === "ALL" ? true : s.risk === riskFilter))
        .sort((a, b) => {
          // Put question-bearing claims first, then order each group by
          // the displayed confidence percentage from highest to lowest.
          if (a.question && !b.question) return -1;
          if (!a.question && b.question) return 1;
          return b.score - a.score;
        })
    : [];

  return (
    <div className="upload-area w-full max-w-2xl mx-auto text-left">
      {/* Hidden native file input */}
      <input
        ref={fileInputRef}
        type="file"
        id="pitch-deck-input"
        accept="application/pdf,.pdf"
        className="sr-only"
        onChange={handleFileChange}
        aria-label="Upload pitch deck PDF"
      />

      {/* Upload Box (Hidden when results are actively showing) */}
      {!analysisResult ? (
        <div
          onDragEnter={handleDragEnter}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          className={`upload-dropzone relative w-full rounded-2xl bg-white border transition-all duration-200 p-8 sm:p-10 text-center ${
            isDragging
              ? "border-[#426B50] bg-[#F4F7F4] ring-2 ring-[#426B50]/20"
              : errorMessage
              ? "border-[#E3B5AF] bg-white"
              : "border-[#E6E4DE] shadow-[0_1px_3px_rgba(0,0,0,0.03)] hover:border-[#D1CFC7]"
          }`}
        >
          {!file ? (
            /* Initial State */
            <div className="flex flex-col items-center">
              <div
                className={`w-14 h-14 rounded-full flex items-center justify-center mb-4 transition-colors ${
                  isDragging
                    ? "bg-[#426B50]/10 text-[#426B50]"
                    : "bg-[#FAF9F6] text-[#73746F] border border-[#E6E4DE]"
                }`}
                aria-hidden="true"
              >
                <svg
                  className="w-6 h-6"
                  fill="none"
                  viewBox="0 0 24 24"
                  strokeWidth={1.75}
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M19.5 14.25v-2.625a3.375 3.375 0 0 0-3.375-3.375h-1.5A1.125 1.125 0 0 1 13.5 7.125v-1.5a3.375 3.375 0 0 0-3.375-3.375H8.25m6.75 12-3-3m0 0-3 3m3-3v8.25M6 10.5h.008v.008H6V10.5Z"
                  />
                </svg>
              </div>

              <h2 className="text-lg font-semibold text-[#20211F] mb-1 tracking-tight">
                {isDragging ? "Drop your PDF file here" : "Upload your pitch deck"}
              </h2>
              <p className="text-sm text-[#73746F] mb-6">
                Choose a PDF up to 20 MB from your computer.
              </p>

              <button
                type="button"
                id="choose-pdf-button"
                onClick={handleChooseClick}
                className="primary-action inline-flex items-center justify-center px-5 py-2.5 rounded-lg bg-[#426B50] hover:bg-[#365741] text-white text-sm font-medium transition-colors shadow-sm cursor-pointer focus-visible:ring-2 focus-visible:ring-[#426B50] focus-visible:ring-offset-2"
              >
                <svg
                  className="w-4 h-4 mr-2"
                  fill="none"
                  viewBox="0 0 24 24"
                  strokeWidth={2}
                  stroke="currentColor"
                  aria-hidden="true"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M12 4.5v15m7.5-7.5h-15"
                  />
                </svg>
                Choose PDF
              </button>

              <p className="mt-6 text-xs text-[#73746F]">
                PDF only · Evaluates claim vagueness &amp; investor questions
              </p>
            </div>
          ) : (
            /* Selected File State */
            <div className="flex flex-col items-center">
              <div className="w-14 h-14 rounded-xl bg-[#FAF9F6] border border-[#E6E4DE] text-[#426B50] flex items-center justify-center mb-4">
                <svg
                  className="w-7 h-7"
                  fill="none"
                  viewBox="0 0 24 24"
                  strokeWidth={1.5}
                  stroke="currentColor"
                  aria-hidden="true"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M19.5 14.25v-2.625a3.375 3.375 0 0 0-3.375-3.375h-1.5A1.125 1.125 0 0 1 13.5 7.125v-1.5a3.375 3.375 0 0 0-3.375-3.375H8.25m0 12.75h7.5m-7.5 3H12M10.5 2.25H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 0 0-9-9Z"
                  />
                </svg>
              </div>

              <div className="max-w-full px-2 mb-2 text-center">
                <h3
                  className="text-base font-medium text-[#20211F] truncate"
                  title={file.name}
                >
                  {file.name}
                </h3>
                <p className="text-xs text-[#73746F] mt-0.5">
                  {formatFileSize(file.size)} · PDF Pitch Deck
                </p>
              </div>

              <div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-[#FAF9F6] border border-[#E6E4DE] text-xs font-medium text-[#426B50] my-3">
                <span className="w-1.5 h-1.5 rounded-full bg-[#426B50]" />
                File ready for analysis
              </div>

              <div className="flex flex-col sm:flex-row items-center gap-2.5 mt-2 w-full max-w-xs">
                <button
                  type="button"
                  id="analyze-deck-button"
                  onClick={handleAnalyzeClick}
                  disabled={isAnalyzing}
                  className="primary-action w-full sm:w-auto flex-1 inline-flex items-center justify-center px-4 py-2.5 rounded-lg bg-[#426B50] hover:bg-[#365741] text-white text-sm font-medium transition-colors shadow-sm cursor-pointer disabled:opacity-60 disabled:cursor-not-allowed focus-visible:ring-2 focus-visible:ring-[#426B50] focus-visible:ring-offset-2"
                >
                  {isAnalyzing ? (
                    <>
                      <svg
                        className="animate-spin -ml-1 mr-2 h-4 w-4 text-white"
                        fill="none"
                        viewBox="0 0 24 24"
                      >
                        <circle
                          className="opacity-25"
                          cx="12"
                          cy="12"
                          r="10"
                          stroke="currentColor"
                          strokeWidth="4"
                        />
                        <path
                          className="opacity-75"
                          fill="currentColor"
                          d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z"
                        />
                      </svg>
                      Analyzing deck...
                    </>
                  ) : (
                    "Analyze pitch deck"
                  )}
                </button>
                <button
                  type="button"
                  id="remove-file-button"
                  onClick={handleRemove}
                  disabled={isAnalyzing}
                  className="secondary-action w-full sm:w-auto inline-flex items-center justify-center px-3.5 py-2.5 rounded-lg border border-[#E6E4DE] bg-white hover:bg-[#FAF9F6] text-[#73746F] hover:text-[#20211F] text-sm font-medium transition-colors cursor-pointer disabled:opacity-50 focus-visible:ring-2 focus-visible:ring-[#426B50] focus-visible:ring-offset-2"
                >
                  Remove
                </button>
              </div>
            </div>
          )}
        </div>
      ) : (
        /* Results View */
        <div className="w-full space-y-6">
          {/* Top Summary Bar */}
          <div className="results-overview rounded-2xl bg-white border border-[#E6E4DE] p-6 shadow-sm">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-5 border-b border-[#E6E4DE]">
              <div>
                <span className="text-[11px] font-semibold tracking-wider text-[#73746F] uppercase">
                  ANALYSIS RESULTS
                </span>
                <h2 className="text-xl font-semibold text-[#20211F] mt-0.5 truncate max-w-md">
                  {analysisResult.filename}
                </h2>
                <p className="text-xs text-[#73746F] mt-1">
                  Detected and risk-scored {analysisResult.total_sentences} investor-facing claims via {analysisResult.pipeline_info.mode} mode
                </p>
              </div>

              <button
                type="button"
                id="reset-analysis-button"
                onClick={handleRemove}
                className="self-start sm:self-auto inline-flex items-center gap-1.5 px-3.5 py-2 rounded-lg border border-[#E6E4DE] bg-white hover:bg-[#FAF9F6] text-xs font-medium text-[#20211F] transition-colors cursor-pointer"
              >
                <svg
                  className="w-3.5 h-3.5 text-[#73746F]"
                  fill="none"
                  viewBox="0 0 24 24"
                  strokeWidth={2}
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M16.023 9.348h4.992v-.001M2.985 19.644v-4.992m0 0h4.992m-4.993 0 3.181 3.183a8.25 8.25 0 0 0 13.803-3.7M4.031 9.865a8.25 8.25 0 0 1 13.803-3.7l3.181 3.182m0-4.991v4.99"
                  />
                </svg>
                Analyze another deck
              </button>
            </div>

            {/* Metric Counters */}
            <div className="grid grid-cols-3 gap-3 pt-5 text-center">
              <div className="metric-card rounded-xl p-3 bg-[#FAF0EE] border border-[#F2CDC7]">
                <div className="text-2xl font-bold text-[#9E3628]">
                  {analysisResult.summary.high_risk}
                </div>
                <div className="text-xs font-medium text-[#9E3628] mt-0.5">
                  High Risk Claims
                </div>
              </div>

              <div className="metric-card rounded-xl p-3 bg-[#FFF9EB] border border-[#FCE3AC]">
                <div className="text-2xl font-bold text-[#975A16]">
                  {analysisResult.summary.moderate_risk}
                </div>
                <div className="text-xs font-medium text-[#975A16] mt-0.5">
                  Moderate Risk Claims
                </div>
              </div>

              <div className="metric-card rounded-xl p-3 bg-[#F4F7F4] border border-[#CDE0D2]">
                <div className="text-2xl font-bold text-[#2D6A4F]">
                  {analysisResult.summary.low_risk}
                </div>
                <div className="text-xs font-medium text-[#2D6A4F] mt-0.5">
                  Grounded / Low Risk
                </div>
              </div>
            </div>
          </div>

          {/* Filter Pills */}
          {analysisResult.total_sentences > 0 && (
          <div className="risk-filter flex items-center gap-2 overflow-x-auto pb-1 text-xs">
            <span className="text-[#73746F] font-medium mr-1">Filter:</span>
            <button
              type="button"
              onClick={() => setRiskFilter("ALL")}
              className={`risk-filter-button px-3 py-1.5 rounded-lg font-medium transition-colors cursor-pointer ${
                riskFilter === "ALL"
                  ? "bg-[#20211F] text-white"
                  : "bg-white border border-[#E6E4DE] text-[#73746F] hover:text-[#20211F]"
              }`}
            >
              All ({analysisResult.total_sentences})
            </button>
            <button
              type="button"
              onClick={() => setRiskFilter("High")}
              className={`px-3 py-1.5 rounded-lg font-medium transition-colors cursor-pointer ${
                riskFilter === "High"
                  ? "bg-[#9E3628] text-white"
                  : "bg-white border border-[#E6E4DE] text-[#73746F] hover:text-[#9E3628]"
              }`}
            >
              High Risk ({analysisResult.summary.high_risk})
            </button>
            <button
              type="button"
              onClick={() => setRiskFilter("Moderate")}
              className={`px-3 py-1.5 rounded-lg font-medium transition-colors cursor-pointer ${
                riskFilter === "Moderate"
                  ? "bg-[#975A16] text-white"
                  : "bg-white border border-[#E6E4DE] text-[#73746F] hover:text-[#975A16]"
              }`}
            >
              Moderate Risk ({analysisResult.summary.moderate_risk})
            </button>
            <button
              type="button"
              onClick={() => setRiskFilter("Low")}
              className={`px-3 py-1.5 rounded-lg font-medium transition-colors cursor-pointer ${
                riskFilter === "Low"
                  ? "bg-[#2D6A4F] text-white"
                  : "bg-white border border-[#E6E4DE] text-[#73746F] hover:text-[#2D6A4F]"
              }`}
            >
              Low Risk ({analysisResult.summary.low_risk})
            </button>
          </div>
          )}

          {/* Sentence Claims List */}
          <div className="space-y-4">
            {analysisResult.total_sentences === 0 ? (
              <div className="rounded-xl bg-white border border-[#E6E4DE] p-8 text-center text-[#73746F] text-sm leading-relaxed">
                {analysisResult.pipeline_info.document_warning ??
                  "No investor-facing claims were detected. Upload a startup pitch deck with product, market, traction, customer, or financial assertions."}
              </div>
            ) : filteredSentences.length === 0 ? (
              <div className="rounded-xl bg-white border border-[#E6E4DE] p-8 text-center text-[#73746F] text-sm">
                No claims match the selected filter.
              </div>
            ) : (
              filteredSentences.map((item) => (
                <div
                  key={item.id}
                  className="claim-card rounded-xl bg-white border border-[#E6E4DE] p-5 sm:p-6 text-left shadow-[0_1px_2px_rgba(0,0,0,0.02)] transition-all hover:border-[#D1CFC7]"
                >
                  {/* Claim Header */}
                  <div className="flex items-center justify-between gap-3 mb-2">
                    <span className="text-xs font-medium text-[#73746F]">
                      Claim #{item.id}
                    </span>
                    <div className="flex items-center gap-2">
                      <span className="text-[11px] text-[#73746F]">
                        {Math.round(item.score * 100)}% relative support
                      </span>
                      {item.risk === "High" && (
                        <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-[#FAF0EE] text-[#9E3628] border border-[#F2CDC7]">
                          Risk: High
                        </span>
                      )}
                      {item.risk === "Moderate" && (
                        <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-[#FFF9EB] text-[#975A16] border border-[#FCE3AC]">
                          Risk: Moderate
                        </span>
                      )}
                      {item.risk === "Low" && (
                        <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-[#F4F7F4] text-[#2D6A4F] border border-[#CDE0D2]">
                          Risk: Low
                        </span>
                      )}
                    </div>
                  </div>

                  {/* Sentence Text */}
                  <p className="text-sm sm:text-base font-medium text-[#20211F] leading-relaxed mb-4">
                    &ldquo;{item.text}&rdquo;
                  </p>

                  {/* Gated Investor Counter-Question (Moderate & High only) */}
                  <div className="pt-3.5 border-t border-[#F0EFEB]">
                    {item.question ? (
                      <div>
                        <div className="flex items-center gap-1.5 mb-1 text-xs font-medium text-[#73746F]">
                          <svg
                            className="w-3.5 h-3.5 text-[#426B50]"
                            fill="none"
                            viewBox="0 0 24 24"
                            strokeWidth={2}
                            stroke="currentColor"
                          >
                            <path
                              strokeLinecap="round"
                              strokeLinejoin="round"
                              d="M9.879 7.519c1.171-1.025 3.071-1.025 4.242 0 1.172 1.025 1.172 2.687 0 3.712-.203.179-.43.326-.67.442-.745.361-1.45.999-1.45 1.827v.75M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Zm-9 5.25h.008v.008H12v-.008Z"
                            />
                          </svg>
                          Investor Counter-Question
                        </div>
                        <p className="text-xs sm:text-sm text-[#20211F] font-normal leading-relaxed italic">
                          &ldquo;{item.question}&rdquo;
                        </p>
                      </div>
                    ) : (
                      <div className="flex items-center gap-1.5 text-xs text-[#2D6A4F] font-normal">
                        <svg
                          className="w-3.5 h-3.5"
                          fill="none"
                          viewBox="0 0 24 24"
                          strokeWidth={2}
                          stroke="currentColor"
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            d="m4.5 12.75 6 6 9-13.5"
                          />
                        </svg>
                        Grounded claim · No counter-question required
                      </div>
                    )}
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      )}

      {/* Error Message */}
      {errorMessage && (
        <div
          role="alert"
          className="error-panel mt-4 p-3.5 rounded-xl bg-[#FDF4F3] border border-[#F2C5C0] flex items-start gap-2.5 text-left"
        >
          <svg
            className="w-5 h-5 text-[#B94A38] shrink-0 mt-0.5"
            fill="none"
            viewBox="0 0 24 24"
            strokeWidth={1.75}
            stroke="currentColor"
            aria-hidden="true"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M12 9v3.75m9-.75a9 9 0 1 1-18 0 9 9 0 0 1 18 0Zm-9 3.75h.008v.008H12v-.008Z"
            />
          </svg>
          <div className="flex-1 text-xs sm:text-sm text-[#872D20]">
            <p className="font-medium">{errorMessage}</p>
          </div>
          <button
            type="button"
            onClick={() => setErrorMessage(null)}
            className="text-[#872D20] hover:text-[#5E1E14] text-xs font-semibold px-1 py-0.5 rounded cursor-pointer"
            aria-label="Dismiss error"
          >
            Dismiss
          </button>
        </div>
      )}
    </div>
  );
}
