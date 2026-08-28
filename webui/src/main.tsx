import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import { installStaticMode } from "./staticMode";
import "./index.css";

// Must settle before the first render: App fetches on mount, and a request that
// goes out before the shim is installed would hit /api on a host that has no
// backend. Probing costs one 404 on a live deployment.
installStaticMode().finally(() => {
  createRoot(document.getElementById("root")!).render(
    <StrictMode>
      <App />
    </StrictMode>,
  );
});
