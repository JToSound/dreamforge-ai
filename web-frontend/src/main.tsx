import React, { Component, type ErrorInfo, type ReactNode } from "react";
import ReactDOM from "react-dom/client";
import App from "./App";

class RootErrorBoundary extends Component<{ children: ReactNode }, { error: Error | null }> {
  override state: { error: Error | null } = { error: null };

  override componentDidCatch(error: Error, _info: ErrorInfo): void {
    this.setState({ error });
  }

  override render(): ReactNode {
    if (this.state.error) {
      return (
        <div style={{ padding: "24px", color: "#e2e8f0", fontFamily: "system-ui, sans-serif" }}>
          <h1 style={{ marginTop: 0 }}>DreamForge frontend failed to render</h1>
          <p style={{ lineHeight: 1.6 }}>{this.state.error.message}</p>
        </div>
      );
    }
    return this.props.children;
  }
}

const rootEl = document.getElementById("root") as HTMLElement;

const renderBootstrapError = (message: string): void => {
  rootEl.innerHTML = `
    <div style="padding:24px;color:#e2e8f0;font-family:system-ui,sans-serif;background:#0b1220;min-height:100vh;">
      <h1 style="margin-top:0;">DreamForge frontend bootstrap error</h1>
      <pre style="white-space:pre-wrap;line-height:1.6;">${message}</pre>
    </div>
  `;
};

window.addEventListener("error", (event) => {
  if (!rootEl.childElementCount) {
    renderBootstrapError(event.error?.message ?? event.message);
  }
});

window.addEventListener("unhandledrejection", (event) => {
  if (!rootEl.childElementCount) {
    const reason =
      event.reason instanceof Error ? event.reason.message : String(event.reason ?? "Unknown rejection");
    renderBootstrapError(reason);
  }
});

ReactDOM.createRoot(rootEl).render(
  <React.StrictMode>
    <RootErrorBoundary>
      <App />
    </RootErrorBoundary>
  </React.StrictMode>,
);
