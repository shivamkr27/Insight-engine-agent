export default function InsightCard({ answer, sources: sourcesProp, judgebadge, judgereason }) {
  const sources = typeof sourcesProp === "string" ? JSON.parse(sourcesProp) : (sourcesProp || []);
  const C = {
    accent:       "#F0BA72",
    accentDim:    "rgba(240,186,114,0.10)",
    accentBorder: "rgba(240,186,114,0.25)",
    border:       "rgba(255,255,255,0.10)",
    borderFaint:  "rgba(255,255,255,0.05)",
    text:         "#E2E8F0",
    textSub:      "#8B949E",
    textDim:      "#4A5568",
    green:        "#3FB950",
    yellow:       "#F59E0B",
    red:          "#EF4444",
  };

  const badgeColor = judgebadge?.includes("🟢") ? C.green
    : judgebadge?.includes("🔴") ? C.red
    : C.yellow;

  const handleCopy = () => {
    navigator.clipboard.writeText(answer || "").catch(() => {});
  };

  const handleExport = () => {
    const parts = [answer || ""];
    if (sources?.length) {
      parts.push("\n## Sources");
      sources.forEach(s => parts.push(`- ${s}`));
    }
    if (judgebadge) {
      parts.push(`\n*AI Judge: ${judgebadge}${judgereason ? " — " + judgereason : ""}*`);
    }
    const blob = new Blob([parts.join("\n")], { type: "text/markdown" });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement("a");
    a.href     = url;
    a.download = "insight-note.md";
    a.click();
    URL.revokeObjectURL(url);
  };

  const btnStyle = {
    fontSize:     "11px",
    color:        C.textSub,
    background:   "rgba(255,255,255,0.04)",
    border:       `1px solid ${C.border}`,
    borderRadius: "6px",
    padding:      "4px 10px",
    cursor:       "pointer",
    transition:   "border-color 0.15s, color 0.15s",
  };

  return (
    <div style={{ marginTop: "12px", borderTop: `1px solid ${C.borderFaint}`, paddingTop: "10px" }}>

      {/* ── Judge badge ── */}
      {judgebadge && (
        <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "10px", flexWrap: "wrap" }}>
          <span style={{
            fontSize: "11px", fontWeight: 600, color: badgeColor,
            background: `${badgeColor}18`, border: `1px solid ${badgeColor}35`,
            borderRadius: "20px", padding: "3px 10px",
          }}>
            {judgebadge}
          </span>
          {judgereason && (
            <span style={{ fontSize: "11px", color: C.textDim }}>{judgereason}</span>
          )}
        </div>
      )}

      {/* ── Source chips ── */}
      {sources?.length > 0 && (
        <div style={{ marginBottom: "10px" }}>
          <span style={{ fontSize: "11px", color: C.textDim, display: "block", marginBottom: "4px" }}>
            📄 Sources
          </span>
          <div style={{ display: "flex", flexWrap: "wrap", gap: "5px" }}>
            {sources.map((s, i) => (
              <span key={i} style={{
                fontSize: "11px", color: C.accent,
                background: C.accentDim, border: `1px solid ${C.accentBorder}`,
                borderRadius: "4px", padding: "2px 8px",
              }}>
                {s}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* ── Action buttons ── */}
      <div style={{ display: "flex", gap: "8px" }}>
        <button onClick={handleCopy}   style={btnStyle}>📋 Copy Summary</button>
        <button onClick={handleExport} style={btnStyle}>📥 Export to Note</button>
      </div>

    </div>
  );
}
