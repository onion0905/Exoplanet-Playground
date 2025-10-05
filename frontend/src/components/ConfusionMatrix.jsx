import React from "react";

/**
 * ConfusionMatrix
 * @param {number[][]} data  3x3 confusion matrix: rows = Actual, cols = Predicted
 * @param {string[]} [labels] Optional class labels length=3
 * @param {string} [title]    Optional title
 * @param {string} [className]
 */
export default function ConfusionMatrix({
  data = [
    [128, 7, 9],
    [8, 52, 6],
    [6, 10, 157],
  ],
  labels = ["Not-Exoplanet", "Candidate", "Exoplanet"],
  title = "Exoplanet Classification — 3-Class Confusion Matrix",
  className = "",
  resultCounts = null, // 新增：結果數量數據
}) {
  // ---- basic guards ---------------------------------------------------------
  const n = 3;
  const A =
    Array.isArray(data) && data.length === n && data.every(r => Array.isArray(r) && r.length === n)
      ? data
      : Array.from({ length: n }, () => Array(n).fill(0));

  // ---- totals & metrics -----------------------------------------------------
  const totals = { row: [0, 0, 0], col: [0, 0, 0], all: 0 };
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      totals.row[i] += A[i][j];
      totals.col[j] += A[i][j];
      totals.all += A[i][j];
    }
  }
  const diag = A[0][0] + A[1][1] + A[2][2];
  const accuracy = safe(div(diag, totals.all));
  const perClass = Array.from({ length: n }, (_, k) => {
    const TP = A[k][k];
    const FP = totals.col[k] - TP;
    const FN = totals.row[k] - TP;
    const precision = safe(div(TP, TP + FP));
    const recall = safe(div(TP, TP + FN));
    const f1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;
    return { precision, recall, f1 };
  });
  const macroF1 = safe(perClass.reduce((s, x) => s + x.f1, 0) / n);

  // ---- helpers --------------------------------------------------------------
  function div(a, b) {
    return b > 0 ? a / b : 0;
  }
  function safe(x) {
    return Number.isFinite(x) ? x : 0;
  }
  const fmt = (x) => (Number.isFinite(x) ? (Math.round(x * 100) / 100).toFixed(2) : "—");

  const cellBg = (i, j) => {
    // 對角線：正確→綠；中心 candidate：琥珀色；其餘：紅
    if (i === j) return "bg-emerald-500/5";
    if (i === 1 && j === 1) return "bg-amber-400/5";
    return "bg-rose-500/5";
  };
  const cellTag = (i, j) => {
    if (i === j && i === 0) return { tag: "TN", cls: "text-emerald-300/90" };
    if (i === j && i === 1) return { tag: "CAND", cls: "text-amber-300/90" };
    if (i === j && i === 2) return { tag: "TP", cls: "text-emerald-300/90" };
    return { tag: "Misclass", cls: "text-rose-300/90" };
  };

  return (
    <div className={`starfield min-h-screen text-slate-200 ${className}`}>
      {/* Inline 星空 / 霓虹 / 格線樣式（搬到全域 CSS 也可以） */}
      <style>{`
        body:has(.starfield){
          background:
            radial-gradient(1200px 600px at 10% 10%, #0b1231 0%, transparent 60%),
            radial-gradient(900px 500px at 90% 20%, #1b2b78 0%, transparent 60%),
            radial-gradient(800px 600px at 30% 80%, #141b4b 0%, transparent 60%),
            #050816;
        }
        .starfield::before, .starfield::after {
          content: ""; position: fixed; inset: 0; pointer-events:none;
          background-image:
            radial-gradient(2px 2px at 20% 30%, rgba(255,255,255,.8) 50%, transparent 51%),
            radial-gradient(1.5px 1.5px at 70% 60%, rgba(255,255,255,.7) 50%, transparent 51%),
            radial-gradient(1.5px 1.5px at 40% 80%, rgba(255,255,255,.6) 50%, transparent 51%),
            radial-gradient(1px 1px at 85% 35%, rgba(255,255,255,.6) 50%, transparent 51%),
            radial-gradient(1px 1px at 15% 70%, rgba(255,255,255,.5) 50%, transparent 51%);
          animation: twinkle 6s linear infinite; opacity:.7;
        }
        .starfield::after { animation-delay:-3s; opacity:.5; }
        @keyframes twinkle { 50% { opacity:.3; } }
        .neon { box-shadow:
          0 0 0 1px rgba(137,176,255,.25) inset,
          0 0 20px rgba(56,189,248,.25),
          0 0 40px rgba(59,130,246,.12);
        }
        .grid-lines {
          background-image:
            linear-gradient(to right, rgba(255,255,255,0.08) 1px, transparent 1px),
            linear-gradient(to bottom, rgba(255,255,255,0.08) 1px, transparent 1px);
          background-size: calc(100%/3) calc(100%/3);
        }
      `}</style>

      {/* Header */}
      <header className="max-w-6xl mx-auto px-6 pt-10">
        <div className="flex items-center gap-3">
          <div className="h-8 w-8 rounded-full bg-gradient-to-tr from-cyan-400 to-indigo-500 shadow-[0_0_20px_rgba(34,211,238,0.6)]" />
          <h1 className="text-2xl md:text-3xl font-semibold tracking-wide">
            {title}
          </h1>
        </div>
        <p className="mt-2 text-slate-400">
          Classes：{labels.join(" · ")}
        </p>
      </header>

      {/* Main */}
      <main className="max-w-6xl mx-auto px-6 py-8 grid md:grid-cols-5 gap-6">
        {/* Matrix */}
        <section className="md:col-span-3 bg-slate-900/40 rounded-2xl p-6 neon">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-medium tracking-wide">Confusion Matrix (3×3)</h2>
            <div className="flex items-center gap-3 text-xs">
              <span className="inline-flex items-center gap-2">
                <span className="h-3 w-3 rounded-sm bg-emerald-500/70" /> 正確(對角)
              </span>
              <span className="inline-flex items-center gap-2">
                <span className="h-3 w-3 rounded-sm bg-amber-400/70" /> 候選 candidate
              </span>
              <span className="inline-flex items-center gap-2">
                <span className="h-3 w-3 rounded-sm bg-rose-500/70" /> 錯誤
              </span>
            </div>
          </div>

          {/* 軸標籤 */}
          <div className="grid grid-cols-12 gap-2 items-center">
            <div className="col-span-1" />
            <div className="col-span-11">
              <div className="text-sm text-slate-400 mb-1 pl-2">Predicted →</div>
            </div>
            <div className="col-span-1 flex items-center rotate-180 [writing-mode:vertical-rl] text-sm text-slate-400">
              Actual →
            </div>

            <div className="col-span-11">
              <div className="grid grid-cols-3 rounded-xl overflow-hidden grid-lines">
                {A.map((row, i) =>
                  row.map((val, j) => {
                    const tag = cellTag(i, j);
                    return (
                      <div key={`${i}-${j}`} className="p-5 relative">
                        <div className={`absolute inset-0 ${cellBg(i, j)}`} />
                        <div className="relative">
                          <div className={`text-xs uppercase tracking-widest ${tag.cls}`}>{tag.tag}</div>
                          <div
                            className="mt-1 text-2xl font-semibold"
                            aria-label={`Actual ${labels[i]}, Pred ${labels[j]}`}
                          >
                            {val}
                          </div>
                        </div>
                      </div>
                    );
                  })
                )}
              </div>

              {/* column labels */}
              <div className="grid grid-cols-3 text-center mt-3 text-xs text-slate-400">
                <div>Pred: {labels[0]}</div>
                <div>Pred: {labels[1]}</div>
                <div>Pred: {labels[2]}</div>
              </div>
              {/* row labels */}
              <div className="grid grid-cols-3 mt-2 text-xs text-slate-400">
                <div>Actual: {labels[0]}</div>
                <div className="text-center">Actual: {labels[1]}</div>
                <div className="text-right">Actual: {labels[2]}</div>
              </div>
            </div>
          </div>
        </section>

        {/* Metrics */}
        <aside className="md:col-span-2 space-y-6">
          <div className="bg-slate-900/40 rounded-2xl p-6 neon">
            <h3 className="text-sm font-medium tracking-wide mb-4">Overall Metrics</h3>
            <div className="grid grid-cols-2 gap-3 text-slate-200">
              <div className="rounded-xl bg-slate-800/60 p-4">
                <div className="text-xs text-slate-400">Accuracy</div>
                <div className="text-2xl font-semibold">{fmt(accuracy)}</div>
              </div>
              <div className="rounded-xl bg-slate-800/60 p-4">
                <div className="text-xs text-slate-400">Macro F1</div>
                <div className="text-2xl font-semibold">{fmt(macroF1)}</div>
              </div>
            </div>
          </div>

          <div className="bg-slate-900/40 rounded-2xl p-6 neon">
            <h3 className="text-sm font-medium tracking-wide mb-4">
              {resultCounts ? "Prediction Results" : "Per-Class (Precision / Recall / F1)"}
            </h3>
            {resultCounts ? (
              <div className="grid gap-3">
                {resultCounts.map((count, i) => (
                  <div key={i} className="rounded-xl bg-slate-800/60 p-4">
                    <div className="text-xs text-slate-400 mb-1">{labels[i]}</div>
                    <div className="text-2xl font-semibold text-slate-200">{count}</div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="grid gap-3">
                {perClass.map((m, i) => (
                  <div key={i} className="rounded-xl bg-slate-800/60 p-4">
                    <div className="text-xs text-slate-400 mb-1">{labels[i]}</div>
                    <div className="text-lg">
                      {fmt(m.precision)} / {fmt(m.recall)} / {fmt(m.f1)}
                    </div>
                  </div>
                ))}
              </div>
            )}
            {!resultCounts && (
              <p className="mt-3 text-xs text-slate-400">
                格式：Precision / Recall / F1（四捨五入到小數點二位）
              </p>
            )}
          </div>
        </aside>
      </main>

      <footer className="max-w-6xl mx-auto px-6 pb-12 text-xs text-slate-500">
        © Exoplanet Prediction Playground
      </footer>
    </div>
  );
}