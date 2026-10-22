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
      <div className="preview-card rounded-xl bg-white border border-[#DCE4D9] p-5 sm:p-6 text-left shadow-[0_1px_2px_rgba(0,0,0,0.02)]">
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
          <p className="text-sm sm:text-base font-medium text-[#202922] leading-relaxed">
            &ldquo;Our platform will capture a huge share of the global market.&rdquo;
          </p>
        </div>

        {/* Counter Question Block */}
        <div className="pt-3.5 border-t border-[#F0EFEB]">
          <span className="text-xs font-medium text-[#6F7970] block mb-1">
            Investor Counter-Question
          </span>
          <p className="text-xs sm:text-sm text-[#383935] leading-relaxed">
            &ldquo;What evidence supports your expected market share, and how did you calculate it?&rdquo;
          </p>
        </div>
      </div>
    </section>
  );
}
