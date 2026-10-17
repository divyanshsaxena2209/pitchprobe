export default function ProductExample() {
  return (
    <section
      aria-labelledby="example-output-heading"
      className="w-full max-w-xl mx-auto mt-14 sm:mt-16 pt-10 border-t border-[#E6E4DE]"
    >
      {/* Section Label */}
      <div className="flex items-center justify-between mb-4">
        <span
          id="example-output-heading"
          className="text-[11px] font-semibold tracking-wider text-[#73746F] uppercase"
        >
          WHAT YOU&apos;LL GET
        </span>
        <span className="text-xs text-[#73746F] bg-[#FAF9F6] border border-[#E6E4DE] px-2 py-0.5 rounded">
          Sample Output Preview
        </span>
      </div>

      {/* Compact Example Card */}
      <div className="rounded-xl bg-white border border-[#E6E4DE] p-5 sm:p-6 text-left shadow-[0_1px_2px_rgba(0,0,0,0.02)]">
        {/* Claim Block */}
        <div className="mb-4">
          <div className="flex items-center justify-between gap-3 mb-1.5">
            <span className="text-xs font-medium text-[#73746F]">
              Detected Claim
            </span>
            {/* Risk Badge */}
            <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-[#FAF0EE] text-[#9E3628] border border-[#F2CDC7]">
              Risk: High
            </span>
          </div>
          <p className="text-sm sm:text-base font-medium text-[#20211F] leading-relaxed">
            &ldquo;Our platform will capture a huge share of the global market.&rdquo;
          </p>
        </div>

        {/* Counter Question Block */}
        <div className="pt-3.5 border-t border-[#F0EFEB]">
          <span className="text-xs font-medium text-[#73746F] block mb-1">
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
