export default function ProductExample() {
  return (
    <section
      aria-labelledby="example-output-heading"
      className="sample-preview"
    >
      {/* Section Label */}
      <div className="preview-heading">
        <span
          id="example-output-heading"
          className="text-[11px] font-semibold tracking-wider text-[#777064] uppercase"
        >
          WHAT YOU&apos;LL GET
        </span>
        <span className="text-xs text-[#777064] bg-[#F4EEE3] border border-[#E4D8C6] px-2 py-0.5 rounded">
          Sample Output Preview
        </span>
      </div>

      {/* Compact Example Card */}
      <div className="preview-card rounded-xl bg-white border border-[#E4D8C6] p-5 sm:p-6 text-left">
        {/* Claim Block */}
        <div className="mb-4">
          <div className="flex items-center justify-between gap-3 mb-1.5">
            <span className="text-xs font-medium text-[#777064]">
              Detected Claim
            </span>
            {/* Risk Badge */}
            <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-[#F8F3E9] text-[#9E3628] border border-[#E4D8C6]">
              Risk: High
            </span>
          </div>
          <p className="text-sm sm:text-base font-medium text-[#29251E] leading-relaxed mb-3">
            &ldquo;Our platform will capture a huge share of the global market.&rdquo;
          </p>

          {/* Static confidence bar for the preview */}
          <div className="confidence-timeline">
            <span className="confidence-bar-value" style={{ left: "82%" }}>82% support</span>
            <div className="confidence-bar-track" role="meter" aria-valuemin={0} aria-valuemax={100} aria-valuenow={82} aria-label="Sample relative model support">
              <div
                className="confidence-bar-fill confidence-bar-fill--high"
                style={{ width: "82%" }}
              />
            </div>
          </div>
        </div>

        {/* Counter Question Block */}
        <div className="pt-3.5 border-t border-[#EEE5D8]">
          <div className="flex items-center gap-1.5 mb-1 text-xs font-medium text-[#777064]">
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
          <p className="text-xs sm:text-sm text-[#4C453A] leading-relaxed italic">
            &ldquo;What evidence supports your expected market share, and how did you calculate it?&rdquo;
          </p>
        </div>
      </div>
    </section>
  );
}
