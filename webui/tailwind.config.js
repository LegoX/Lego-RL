/** @type {import('tailwindcss').Config} */

// Every palette the components reach for is redefined here on top of CSS
// variables (see src/index.css), so `text-indigo-400` renders the LegoFlow
// terracotta rather than Tailwind's stock indigo. Keeping the stock names means
// no class churn across the ~30 panels; the variables carry the theme flip.
const ramp = (name, shades) =>
  Object.fromEntries(
    Object.entries(shades).map(([shade, varShade]) => [
      shade,
      `rgb(var(--c-${name}-${varShade}) / <alpha-value>)`,
    ]),
  );

// Shades the panels actually use get their own variable; the rest alias to the
// nearest one so a stray class never falls back to a cool stock colour.
const accentShades = { 100: 200, 200: 200, 300: 300, 400: 400, 500: 500, 600: 600, 700: 600 };
const fiveShades = { 100: 300, 200: 300, 300: 300, 400: 400, 500: 500, 600: 600, 700: 600 };
const tealShades = { 100: 200, 200: 200, 300: 300, 400: 400, 500: 500, 600: 600, 700: 600 };
const blueShades = { 100: 400, 200: 400, 300: 400, 400: 400, 500: 500, 600: 500, 700: 500 };

export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        slate: {
          50: "rgb(var(--c-slate-50) / <alpha-value>)",
          100: "rgb(var(--c-slate-100) / <alpha-value>)",
          200: "rgb(var(--c-slate-200) / <alpha-value>)",
          300: "rgb(var(--c-slate-300) / <alpha-value>)",
          400: "rgb(var(--c-slate-400) / <alpha-value>)",
          500: "rgb(var(--c-slate-500) / <alpha-value>)",
          600: "rgb(var(--c-slate-600) / <alpha-value>)",
          700: "rgb(var(--c-slate-700) / <alpha-value>)",
          800: "rgb(var(--c-slate-800) / <alpha-value>)",
          900: "rgb(var(--c-slate-900) / <alpha-value>)",
          950: "rgb(var(--c-slate-950) / <alpha-value>)",
        },
        indigo: ramp("indigo", accentShades),
        emerald: ramp("emerald", fiveShades),
        green: ramp("emerald", fiveShades),
        rose: ramp("rose", fiveShades),
        red: ramp("rose", fiveShades),
        amber: ramp("amber", fiveShades),
        yellow: ramp("amber", fiveShades),
        orange: ramp("amber", fiveShades),
        violet: ramp("violet", fiveShades),
        purple: ramp("violet", fiveShades),
        teal: ramp("teal", tealShades),
        cyan: ramp("teal", tealShades),
        blue: ramp("blue", blueShades),
        sky: ramp("blue", blueShades),
      },
    },
  },
  plugins: [require("@tailwindcss/typography")],
};
