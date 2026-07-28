function resolveStyles(clone: SVGSVGElement, original: SVGSVGElement) {
  const origElements = original.querySelectorAll("*");
  const cloneElements = clone.querySelectorAll("*");

  cloneElements.forEach((el, i) => {
    if (!(el instanceof SVGElement) || !origElements[i]) return;
    const computed = window.getComputedStyle(origElements[i]);

    const stroke = computed.stroke;
    if (stroke && stroke !== "none") {
      el.setAttribute("stroke", stroke);
    }

    const fill = computed.fill;
    if (fill && fill !== "none") {
      el.setAttribute("fill", fill);
    }

    if (el instanceof SVGTextElement) {
      el.setAttribute("fill", computed.fill || "#333");
      el.style.fontSize = computed.fontSize;
      el.style.fontFamily = computed.fontFamily;
    }
  });
}

function applyExportTheme(clone: SVGSVGElement) {
  // White background
  const bg = document.createElementNS("http://www.w3.org/2000/svg", "rect");
  bg.setAttribute("width", "100%");
  bg.setAttribute("height", "100%");
  bg.setAttribute("fill", "#ffffff");
  clone.insertBefore(bg, clone.firstChild);

  // Grid lines → light gray
  clone.querySelectorAll("line[stroke-dasharray]").forEach((el) => {
    el.setAttribute("stroke", "#e2e8f0");
  });

  // Axis lines → dark
  clone.querySelectorAll(".recharts-cartesian-axis line, .recharts-cartesian-axis-line").forEach((el) => {
    el.setAttribute("stroke", "#94a3b8");
  });

  // All text → dark for readability
  clone.querySelectorAll("text").forEach((el) => {
    const currentFill = el.getAttribute("fill") || "";
    // Keep colored legend/tooltip text, but make axis ticks dark
    if (!currentFill || currentFill.startsWith("var(") ||
        currentFill === "none" || currentFill === "#94a3b8" ||
        currentFill.includes("chart")) {
      el.setAttribute("fill", "#334155");
    }
    // If fill is a very light color (from dark theme), make it dark
    if (currentFill.match(/^#[c-f][c-f]/i) || currentFill.match(/^rgb\((1[5-9]\d|2\d\d)/)) {
      el.setAttribute("fill", "#334155");
    }
  });

  // Legend text
  clone.querySelectorAll(".recharts-legend-item-text").forEach((el) => {
    el.setAttribute("fill", "#475569");
  });
}

export function downloadSvgAsPng(container: HTMLElement, filename: string) {
  const svg = container.querySelector("svg");
  if (!svg) return;

  const clone = svg.cloneNode(true) as SVGSVGElement;
  clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");

  // Copy the exact dimensions
  const rect = svg.getBoundingClientRect();
  clone.setAttribute("width", String(rect.width));
  clone.setAttribute("height", String(rect.height));

  // Resolve computed styles from the live DOM into inline attributes
  resolveStyles(clone, svg);

  // Override to a clean export-friendly theme (white bg, dark text)
  applyExportTheme(clone);

  const svgData = new XMLSerializer().serializeToString(clone);
  const svgBlob = new Blob([svgData], { type: "image/svg+xml;charset=utf-8" });
  const url = URL.createObjectURL(svgBlob);

  const img = new window.Image();
  img.onload = () => {
    const scale = 3;
    const canvas = document.createElement("canvas");
    canvas.width = rect.width * scale;
    canvas.height = rect.height * scale;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.scale(scale, scale);
    ctx.drawImage(img, 0, 0);
    URL.revokeObjectURL(url);

    canvas.toBlob((blob) => {
      if (!blob) return;
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = filename;
      a.click();
      URL.revokeObjectURL(a.href);
    }, "image/png");
  };
  img.src = url;
}

export function downloadCsv(
  data: Record<string, number | undefined>[],
  columns: string[],
  filename: string,
) {
  const header = ["step", ...columns].join(",");
  const rows = data.map((row) =>
    ["step", ...columns].map((k) => row[k] ?? "").join(","),
  );
  const csv = [header, ...rows].join("\n");
  const blob = new Blob([csv], { type: "text/csv" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  a.click();
  URL.revokeObjectURL(a.href);
}
