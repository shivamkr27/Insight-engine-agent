export default function DocLibraryCard({ docs }) {
  const C = {
    accent:       "#F0BA72",
    accentDim:    "rgba(240,186,114,0.10)",
    accentBorder: "rgba(240,186,114,0.20)",
    border:       "rgba(255,255,255,0.10)",
    borderFaint:  "rgba(255,255,255,0.04)",
    text:         "#E2E8F0",
    textDim:      "#4A5568",
  };

  const icon = (name) =>
    name.endsWith(".pdf") ? "📕" : name.endsWith(".docx") ? "📘" : "📄";

  const badge = (name) =>
    name.endsWith(".pdf") ? "PDF" : name.endsWith(".docx") ? "DOCX" : "TXT";

  if (!docs?.length) {
    return (
      <div style={{ padding: "14px", color: C.textDim, fontSize: "13px", textAlign: "center" }}>
        No documents yet. Drop a PDF, Word doc, or text file to get started.
      </div>
    );
  }

  return (
    <div style={{
      borderRadius: "10px",
      border:       `1px solid ${C.border}`,
      overflow:     "hidden",
    }}>
      <div style={{
        padding:       "8px 14px",
        borderBottom:  `1px solid ${C.border}`,
        fontSize:      "10px",
        fontWeight:    600,
        color:         C.textDim,
        textTransform: "uppercase",
        letterSpacing: "0.07em",
      }}>
        {docs.length} document{docs.length !== 1 ? "s" : ""}
      </div>

      {docs.map((doc, i) => (
        <div key={i} style={{
          display:     "flex",
          alignItems:  "center",
          gap:         "10px",
          padding:     "9px 14px",
          borderBottom: i < docs.length - 1 ? `1px solid ${C.borderFaint}` : "none",
        }}>
          <span style={{ fontSize: "16px", flexShrink: 0 }}>{icon(doc)}</span>
          <span style={{
            flex: 1, fontSize: "13px", color: C.text,
            overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
          }}>
            {doc}
          </span>
          <span style={{
            fontSize: "10px", fontWeight: 600,
            color:      C.accent,
            background: C.accentDim,
            border:     `1px solid ${C.accentBorder}`,
            borderRadius: "3px",
            padding:    "1px 6px",
            flexShrink: 0,
          }}>
            {badge(doc)}
          </span>
        </div>
      ))}
    </div>
  );
}
