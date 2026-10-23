export default function ProductExample() {
  return (
    <section
      aria-labelledby="example-output-heading"
      className="sample-preview w-full max-w-xl mx-auto mt-14 sm:mt-16 pt-10 border-t border-[#DCE4D9]"
    >
      {/* Section Label */}
      <div className="preview-heading flex items-center justify-between mb-4">
        <span
          id="example-output-heading"
          className="text-[11px] font-semibold tracking-wider text-[#6F7970] uppercase"
        >
          WHAT YOU&apos;LL GET
        </span>
        <span className="text-xs text-[#6F7970] bg-[#F3F6F1] border border-[#DCE4D9] px-2 py-0.5 rounded">
          Sample Output Preview
        </span>
      </div>

      {/* Compact Example Card */}
      <div className="preview-card rounded-xl bg-white border border-[#DCE4D9] p-5 sm:p-6 text-left">
        {/* Claim Block */}
        <div className="mb-4">
          <div className="flex items-center justify-between gap-3 mb-1.5">
            <span className="text-xs font-medium text-[#6F7970]">
              Detected Claim
            </span>
            {/* Risk Badge */}
            <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-[#FAF0EE] text-[#9E3628] border border-[#F2CDC7]">
              Risk: High
            </span>
          </div>
          <p className="text-sm sm:text-base font-medium text-[#202922] leading-relaxed mb-3">
            &ldquo;Our platform will capture a huge share of the global market.&rdquo;
          </p>

          {/* Static confidence bar for the preview */}
          <div className="confidence-bar-track">
            <div
              className="confidence-bar-fill confidence-bar-fill--high"
              style={{ width: "82%" }}
            />
          </div>
        </div>

        {/* Counter Question Block */}
        <div className="pt-3.5 border-t border-[#F0EFEB]">
          <div className="flex items-center gap-1.5 mb-1 text-xs font-medium text-[#6F7970]">
            <svg
              className="w-3.5 h-3.5 text-[#315B41]"
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
          <p className="text-xs sm:text-sm text-[#383935] leading-relaxed italic">
            &ldquo;What evidence supports your expected market share, and how did you calculate it?&rdquo;
          </p>
        </div>
      </div>
    </section>
  );
}
