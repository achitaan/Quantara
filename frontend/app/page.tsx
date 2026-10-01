"use client";
import {
  useCallback,
  useEffect,
  useState,
  type FormEvent,
  type ReactNode,
} from "react";
import ReactMarkdown from "react-markdown";
import {
  api,
  record,
  rows,
  upload,
  type Job,
  type Json,
  type Market,
  type Portfolio,
  type RecordData,
  type Strategy,
} from "@/lib/client";

const sections = [
  "Overview",
  "Portfolio",
  "Market data",
  "Strategies",
  "Simulator",
  "Paper trading",
  "Signals",
  "Cash flow",
  "Tax research",
  "Models",
  "Assistant",
] as const;
type Section = (typeof sections)[number];
const money = (n: number, currency = "USD") =>
  new Intl.NumberFormat("en-CA", {
    style: "currency",
    currency,
    maximumFractionDigits: 0,
  }).format(n);
const number = (value: Json | undefined) =>
  typeof value === "number" ? value : 0;
const text = (value: Json | undefined) =>
  typeof value === "string" ? value : "";
const methods = [
  "equal_weight",
  "mean_variance",
  "min_volatility",
  "risk_parity",
  "black_litterman",
  "cvar",
];
const label = (s: string) =>
  s.replaceAll("_", " ").replace(/^\w/, (c) => c.toUpperCase());
function ThemeSwitch({
  theme,
  onChange,
  disabled = false,
}: {
  theme: "light" | "dark";
  onChange: () => void;
  disabled?: boolean;
}) {
  const next = theme === "light" ? "dark" : "light";
  return (
    <button
      className="theme-toggle"
      type="button"
      onClick={onChange}
      disabled={disabled}
      aria-label={"Switch to " + next + " mode"}
      title={"Switch to " + next + " mode"}
    >
      <svg
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.7"
        strokeLinecap="round"
        aria-hidden
      >
        {next === "dark" ? (
          <path d="M20.6 13.2A8.8 8.8 0 0 1 10.8 3.4a8.8 8.8 0 1 0 9.8 9.8Z" />
        ) : (
          <>
            <circle cx="12" cy="12" r="4" />
            <path d="M12 2v2m0 16v2M2 12h2m16 0h2M5 5l1.4 1.4m11.2 11.2L19 19M5 19l1.4-1.4M17.6 6.4 19 5" />
          </>
        )}
      </svg>
      <span className="theme-label">{label(next)} mode</span>
    </button>
  );
}
function Icon({ name }: { name: string }) {
  const paths: Record<string, string> = {
    Overview: "M3 3h7v7H3z M14 3h7v7h-7z M3 14h7v7H3z M14 14h7v7h-7z",
    Portfolio: "M3 7h18v14H3z M8 7V3h8v4 M3 12h18 M10 12v3h4v-3",
    "Market data": "M3 20V4 M3 20h18 M6 15l4-5 4 3 6-8",
    Strategies: "M4 4h6v6H4z M14 14h6v6h-6z M10 7h7v7 M7 10v7h7",
    Simulator: "M4 3l17 9L4 21z",
    "Paper trading": "M4 3h16v18H4z M8 7h8 M8 12h8 M8 17h4",
    Signals: "M3 12h4l3-8 4 16 3-8h4",
    "Cash flow": "M3 8h18 M17 4l4 4-4 4 M21 16H3 M7 12l-4 4 4 4",
    "Tax research": "M5 3h14v18H5z M9 7h6 M9 11h6 M9 15h6 M9 19h3",
    Models: "M12 3l9 5v8l-9 5-9-5V8z M3 8l9 5 9-5 M12 13v8",
    Assistant: "M3 4h18v13H9l-6 4z M7 8h10 M7 12h6",
  };
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
    >
      <path d={paths[name] || paths.Overview} />
    </svg>
  );
}
function Field({
  name,
  title,
  value,
  options,
  type = "text",
  min,
  max,
  step,
  required = true,
}: {
  name: string;
  title: string;
  value?: string | number;
  options?: (string | { value: string; label: string })[];
  type?: string;
  min?: number;
  max?: number;
  step?: number | string;
  required?: boolean;
}) {
  return (
    <label className="field">
      <span>{title}</span>
      {options ? (
        <select name={name} defaultValue={value} required={required}>
          {options.map((o) => {
            const v = typeof o === "string" ? o : o.value;
            return (
              <option key={v} value={v}>
                {typeof o === "string" ? label(o) : o.label}
              </option>
            );
          })}
        </select>
      ) : (
        <input
          name={name}
          type={type}
          defaultValue={value}
          min={min}
          max={max}
          step={step}
          required={required}
        />
      )}
    </label>
  );
}
function ExecutionCosts() {
  return (
    <>
      <Field
        name="commission"
        title="Commission per fill"
        type="number"
        min={0}
        step={0.01}
        value={0}
      />
      <Field
        name="slippage"
        title="Slippage (basis points)"
        type="number"
        min={0}
        max={1000}
        step={0.1}
        value={5}
      />
      <Field
        name="spread"
        title="Bid/ask spread (basis points)"
        type="number"
        min={0}
        max={1000}
        step={0.1}
        value={0}
      />
      <Field
        name="participation"
        title="Maximum share of bar volume"
        type="number"
        min={0.000001}
        max={1}
        step="any"
        value={0.01}
      />
      <Field name="benchmark" title="Benchmark symbol" value="SPY" />
    </>
  );
}
function Panel({
  title,
  subtitle,
  children,
  className = "",
}: {
  title: string;
  subtitle?: string;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={"panel " + className}>
      <div className="panel-heading">
        <h2>{title}</h2>
        {subtitle && <p>{subtitle}</p>}
      </div>
      {children}
    </section>
  );
}
function Table({
  data,
  keys,
}: {
  data: Record<string, Json>[];
  keys: string[];
}) {
  if (!data.length) return <p className="empty">No records yet.</p>;
  return (
    <div className="table-scroll">
      <table>
        <thead>
          <tr>
            {keys.map((k) => (
              <th key={k}>{label(k)}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.slice(-100).map((row, i) => (
            <tr key={i}>
              {keys.map((k) => (
                <td key={k}>
                  {typeof row[k] === "number"
                    ? (row[k] as number).toLocaleString("en", {
                        maximumFractionDigits: 4,
                      })
                    : typeof row[k] === "object"
                      ? JSON.stringify(row[k])
                      : String(row[k] ?? "—")}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      {data.length > 100 && (
        <small>
          Showing the most recent 100 records. Download the complete log for all
          trades.
        </small>
      )}
    </div>
  );
}
function Chart({
  data,
  title = "Portfolio equity",
}: {
  data: Record<string, Json>[];
  title?: string;
}) {
  if (data.length < 2)
    return <p className="empty">Run a simulation to see its equity curve.</p>;
  const values = data.map((d) => number(d.equity ?? d.balance));
  const min = Math.min(...values),
    max = Math.max(...values),
    range = Math.max(1, max - min);
  const points = values.map((v, i) => [
    58 + (i / (values.length - 1)) * 670,
    225 - ((v - min) / range) * 190,
  ]);
  const path = points.map((p, i) => (i ? "L" : "M") + p.join(",")).join(" ");
  return (
    <div className="chart">
      <div className="chart-title">
        <span>{title}</span>
        <strong>{money(values.at(-1) || 0)}</strong>
      </div>
      <svg
        viewBox="0 0 760 270"
        role="img"
        aria-label={
          title +
          " from " +
          money(values[0]) +
          " to " +
          money(values.at(-1) || 0)
        }
      >
        <defs>
          <linearGradient id="area" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0" stopColor="var(--purple)" stopOpacity=".25" />
            <stop offset="1" stopColor="var(--purple)" stopOpacity="0" />
          </linearGradient>
        </defs>
        {[0, 0.5, 1].map((v) => (
          <g key={v}>
            <line
              x1="58"
              x2="728"
              y1={225 - v * 190}
              y2={225 - v * 190}
              stroke="var(--line)"
              strokeDasharray="4 6"
            />
            <text x="0" y={230 - v * 190} fill="var(--muted)" fontSize="11">
              {money(min + v * range)}
            </text>
          </g>
        ))}
        <path d={path + " L728,225 L58,225 Z"} fill="url(#area)" />
        <path d={path} fill="none" stroke="var(--purple)" strokeWidth="2.5" />
        <text x="58" y="255" fill="var(--muted)" fontSize="11">
          {text(data[0].timestamp ?? data[0].date).slice(0, 10)}
        </text>
        <text
          x="728"
          y="255"
          textAnchor="end"
          fill="var(--muted)"
          fontSize="11"
        >
          {text(data.at(-1)?.timestamp ?? data.at(-1)?.date).slice(0, 10)}
        </text>
      </svg>
    </div>
  );
}
function Result({ value }: { value: unknown }) {
  const obj = record(value);
  const metrics = record(obj.metrics ?? obj.risk ?? obj.evaluation ?? obj);
  const state = record(obj.state);
  const modelEvaluation = record(obj.model_evaluation);
  const changedCosts = modelEvaluation.cost_settings_changed;
  return (
    <div className="result">
      {obj.metrics && state.equity && obj.engine_version !== 2 && (
        <p className="notice">
          This result was saved with an earlier simulator. Rerun it with the
          corrected accounting engine.
        </p>
      )}
      {obj.checkpoint && obj.observation_version !== 2 && (
        <p className="notice">
          Retrain this checkpoint to use the corrected share valuations.
        </p>
      )}
      {obj.model_evaluation && (
        <p className="notice">
          {text(modelEvaluation.scope)}.{" "}
          {Array.isArray(changedCosts) && changedCosts.length > 0
            ? "Execution settings differ from training."
            : ""}
        </p>
      )}
      {obj.execution_assumptions && (
        <details>
          <summary>Execution assumptions</summary>
          {Object.entries(record(obj.execution_assumptions)).map(
            ([key, value]) => (
              <p key={key}>
                <strong>{label(key)}: </strong>
                {text(value)}
              </p>
            ),
          )}
        </details>
      )}
      {Object.keys(metrics).length > 0 && (
        <div className="metric-strip">
          {Object.entries(metrics)
            .filter(([, v]) => typeof v === "number" || v === null)
            .map(([k, v]) => (
              <div key={k}>
                <small>{label(k)}</small>
                <strong>
                  {typeof v === "number"
                    ? [
                        "total_return",
                        "volatility",
                        "max_drawdown",
                        "historical_var",
                        "expected_shortfall",
                        "annualized_return",
                      ].includes(k)
                      ? (v * 100).toFixed(2) + "%"
                      : v.toFixed(3)
                    : "—"}
                </strong>
              </div>
            ))}
        </div>
      )}
      {obj.weights && (
        <Table
          data={Object.entries(record(obj.weights)).map(([symbol, w]) => ({
            symbol,
            weight: number(w) * 100 + "%",
          }))}
          keys={["symbol", "weight"]}
        />
      )}
      {state.equity && <Chart data={rows(state.equity)} />}
      {obj.forecast && (
        <>
          <Chart data={rows(obj.forecast)} title="Projected cash balance" />
          <Table
            data={rows(obj.forecast)}
            keys={[
              "date",
              "net_flow",
              "balance",
              "lower",
              "upper",
              "shortfall",
            ]}
          />
        </>
      )}
      {state.fills && (
        <Table
          data={rows(state.fills)}
          keys={[
            "timestamp",
            "symbol",
            "side",
            "quantity",
            "price",
            "commission",
          ]}
        />
      )}
      {state.orders && (
        <details>
          <summary>Order audit log</summary>
          <Table
            data={rows(state.orders)}
            keys={[
              "submitted_at",
              "symbol",
              "side",
              "quantity",
              "remaining",
              "limit",
              "status",
            ]}
          />
        </details>
      )}
      {obj.benchmark_metrics && (
        <Table
          data={[
            { series: "Strategy", ...record(obj.metrics) },
            { series: "Benchmark", ...record(obj.benchmark_metrics) },
          ]}
          keys={[
            "series",
            "total_return",
            "volatility",
            "max_drawdown",
            "sharpe",
          ]}
        />
      )}
      {obj.pools && (
        <Table
          data={Object.entries(record(obj.pools)).map(([symbol, pool]) => ({
            symbol,
            ...record(pool),
          }))}
          keys={["symbol", "quantity", "acb_cad"]}
        />
      )}
      {obj.dispositions && (
        <Table
          data={rows(obj.dispositions)}
          keys={[
            "timestamp",
            "symbol",
            "quantity",
            "gain_cad",
            "allowable_loss_cad",
            "superficial_loss",
          ]}
        />
      )}
      {obj.harvesting_proposals && (
        <Table
          data={rows(obj.harvesting_proposals)}
          keys={["symbol", "quantity", "estimated_loss_cad", "status"]}
        />
      )}
      {obj.dispositions && obj.report_id && (
        <a href={"/api/reports/" + text(obj.report_id) + "/tax.csv"}>
          Download disposition research CSV ↓
        </a>
      )}
      {obj.provisional === true && (
        <p className="notice">
          This tax result is provisional. Review incomplete records and
          replacement windows.
        </p>
      )}
      {obj.warning && <p className="notice">{text(obj.warning)}</p>}
      {obj.content && (
        <div className="markdown">
          <ReactMarkdown skipHtml>{text(obj.content)}</ReactMarkdown>
        </div>
      )}
      <details>
        <summary>Recorded result and provenance</summary>
        <pre>{JSON.stringify(value, null, 2)}</pre>
      </details>
    </div>
  );
}
const titles: Record<Section, string> = {
  Overview: "Your research, connected.",
  Portfolio: "Every position starts with a ledger.",
  "Market data": "Know the history behind the numbers.",
  Strategies: "Turn an idea into a repeatable rule.",
  Simulator: "Test the idea before the trade.",
  "Paper trading": "Follow a strategy as events arrive.",
  Signals: "Put the headline in context.",
  "Cash flow": "See what your cash can support.",
  "Tax research": "Research losses with Canadian rules.",
  Models: "Train, then challenge the result.",
  Assistant: "Ask about the evidence.",
};

export default function Workspace() {
  const [user, setUser] = useState<string | null>(null),
    [checking, setChecking] = useState(true);
  const [section, setSection] = useState<Section>("Overview");
  const [portfolios, setPortfolios] = useState<Portfolio[]>([]),
    [datasets, setDatasets] = useState<Market[]>([]),
    [strategies, setStrategies] = useState<Strategy[]>([]);
  const [backtests, setBacktests] = useState<RecordData[]>([]),
    [papers, setPapers] = useState<RecordData[]>([]);
  const [signals, setSignals] = useState<RecordData[]>([]),
    [models, setModels] = useState<RecordData[]>([]),
    [documents, setDocuments] = useState<RecordData[]>([]);
  const [reports, setReports] = useState<RecordData[]>([]),
    [jobs, setJobs] = useState<Job[]>([]),
    [cashbook, setCashbook] = useState<Json[]>([]);
  const [health, setHealth] = useState<Record<string, Json>>({}),
    [result, setResult] = useState<unknown>(null),
    [error, setError] = useState("");
  const [busy, setBusy] = useState(false),
    [selectedPortfolio, setPortfolio] = useState(""),
    [selectedDataset, setDataset] = useState("");
  const [messages, setMessages] = useState<
      { role: string; content: string; warning?: string }[]
    >([]),
    [conversation, setConversation] = useState<string>();
  const [conversationList, setConversationList] = useState<RecordData[]>([]);
  const [pendingJobs, setPendingJobs] = useState<string[]>([]);
  const [preferences, setPreferences] = useState<{
    watchlist: string[];
    theme: "dark" | "light";
  }>({ watchlist: [], theme: "light" });
  const [themeReady, setThemeReady] = useState(false);
  const [savingTheme, setSavingTheme] = useState(false);
  useEffect(() => {
    try {
      const stored = localStorage.getItem("quantara-theme");
      if (stored === "light" || stored === "dark")
        setPreferences((v) => ({ ...v, theme: stored }));
    } catch {
      /* Appearance also works when browser storage is unavailable. */
    }
    setThemeReady(true);
  }, []);
  useEffect(() => {
    if (!themeReady) return;
    document.documentElement.dataset.theme = preferences.theme;
    try {
      localStorage.setItem("quantara-theme", preferences.theme);
    } catch {
      /* Optional browser cache. */
    }
  }, [preferences.theme, themeReady]);
  async function toggleTheme() {
    const previous = preferences;
    const next = {
      watchlist: preferences.watchlist,
      theme:
        preferences.theme === "light" ? ("dark" as const) : ("light" as const),
    };
    setPreferences(next);
    if (!user) return;
    setSavingTheme(true);
    setBusy(true);
    try {
      const saved = await api<typeof next>("settings", next, "PUT");
      setPreferences({ watchlist: saved.watchlist, theme: saved.theme });
    } catch (e) {
      setPreferences(previous);
      setError(
        e instanceof Error ? e.message : "Appearance could not be saved.",
      );
    } finally {
      setSavingTheme(false);
      setBusy(false);
    }
  }
  const refresh = useCallback(async () => {
    const [p, d, s, b, a, n, m, r, j, doc, c, cb, h, prefs] = await Promise.all(
      [
        api<Portfolio[]>("portfolios"),
        api<Market[]>("market"),
        api<Strategy[]>("strategies"),
        api<RecordData[]>("backtests"),
        api<RecordData[]>("paper-accounts"),
        api<RecordData[]>("signals"),
        api<RecordData[]>("models"),
        api<RecordData[]>("reports"),
        api<Job[]>("jobs"),
        api<RecordData[]>("documents"),
        api<RecordData[]>("conversations"),
        api<{ transactions: Json[] }>("cashflow/transactions"),
        api<Record<string, Json>>("health"),
        api<{ watchlist: string[]; theme: "dark" | "light" }>("settings"),
      ],
    );
    setPortfolios(p);
    setDatasets(d);
    setStrategies(s);
    setBacktests(b);
    setPapers(a);
    setSignals(n);
    setModels(m);
    setReports(r);
    setJobs(j);
    setDocuments(doc);
    setConversationList(c);
    setCashbook(cb.transactions);
    setHealth(h);
    setPreferences(prefs);
    setPortfolio((v) => v || p[0]?.id || "");
    setDataset((v) => v || d[0]?.id || "");
  }, []);
  useEffect(() => {
    api<{ username: string }>("auth/me")
      .then((v) => {
        setUser(v.username);
        return refresh();
      })
      .catch(() => {})
      .finally(() => setChecking(false));
  }, [refresh]);
  useEffect(() => {
    if (
      !user ||
      (!pendingJobs.length &&
        !jobs.some((j) => j.status === "running" || j.status === "queued"))
    )
      return;
    const timer = setInterval(async () => {
      try {
        const next = await api<Job[]>("jobs");
        const completed = next.filter(
          (j) =>
            j.status === "complete" &&
            (pendingJobs.includes(j.id) ||
              jobs.some((old) => old.id === j.id && old.status !== "complete")),
        );
        const failed = next.find(
          (j) =>
            j.status === "failed" &&
            (pendingJobs.includes(j.id) ||
              jobs.some((old) => old.id === j.id && old.status !== "failed")),
        );
        setJobs(next);
        setPendingJobs((ids) =>
          ids.filter(
            (id) =>
              !next.some(
                (j) =>
                  j.id === id &&
                  ["complete", "failed", "cancelled"].includes(j.status),
              ),
          ),
        );
        if (failed) setError(failed.error || "The operation failed.");
        for (const j of completed) {
          setResult(j.result);
          if (j.operation === "chat") {
            const reply = record(j.result);
            setConversation(text(reply.conversation_id));
            setMessages((v) => [
              ...v,
              {
                role: "assistant",
                content: text(reply.content),
                warning: text(reply.warning),
              },
            ]);
          }
        }
        if (completed.length || failed) await refresh();
      } catch (e) {
        setError(e instanceof Error ? e.message : "Connection lost");
      }
    }, 1000);
    return () => clearInterval(timer);
  }, [user, jobs, pendingJobs, refresh]);
  async function act(action: () => Promise<unknown>) {
    setError("");
    setBusy(true);
    try {
      const value = await action();
      const submitted = record(value);
      if (
        typeof submitted.operation === "string" &&
        typeof submitted.id === "string"
      ) {
        setPendingJobs((ids) => [...new Set([...ids, text(submitted.id)])]);
        setJobs((existing) => [
          ...existing.filter((j) => j.id !== submitted.id),
          value as Job,
        ]);
        setResult(null);
      } else {
        setResult(value);
      }
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Operation failed");
    } finally {
      setBusy(false);
    }
  }
  async function submit(path: string, body: unknown) {
    return act(() => api(path, body));
  }
  function form(action: (data: FormData) => Promise<unknown>) {
    return (event: FormEvent<HTMLFormElement>) => {
      event.preventDefault();
      const data = new FormData(event.currentTarget);
      void act(() => action(data));
    };
  }
  const get = (d: FormData, k: string) => String(d.get(k) || "");
  const num = (d: FormData, k: string) => Number(d.get(k));
  const split = (s: string) =>
    s
      .toUpperCase()
      .split(/[\s,]+/)
      .filter(Boolean);
  const portfolio = portfolios.find((p) => p.id === selectedPortfolio),
    dataset = datasets.find((d) => d.id === selectedDataset);
  const datasetOptions = datasets.map((d) => ({ value: d.id, label: d.name }));
  const strategyOptions = strategies.map((s) => ({
    value: s.id,
    label: s.name,
  }));
  const equalWeights = Object.fromEntries(
    (Object.keys(portfolio?.balance.holdings || {}).length
      ? Object.keys(portfolio!.balance.holdings)
      : dataset?.symbols || []
    ).map((s, _, a) => [s, 1 / a.length]),
  );
  const active = jobs.filter(
    (j) => j.status === "queued" || j.status === "running",
  );
  const lastTest = backtests.at(-1),
    lastEquity = rows(record(lastTest?.state).equity);
  async function fileChange(
    path: string,
    file: File | undefined,
    options?: object,
  ) {
    if (file) await act(() => upload(path, file, options));
  }
  async function connectSandbox() {
    return act(async () => {
      const value = await api<{ link_token: string }>(
        "cashflow/plaid/link-token",
        {},
      );
      if (!window.Plaid)
        await new Promise<void>((resolve, reject) => {
          const script = document.createElement("script");
          script.src =
            "https://cdn.plaid.com/link/v2/stable/link-initialize.js";
          script.onload = () => resolve();
          script.onerror = () => reject(new Error("Plaid Link could not load"));
          document.head.appendChild(script);
        });
      return new Promise((resolve) => {
        const handler = window.Plaid!.create({
          token: value.link_token,
          onSuccess: (token) => {
            void api<{ item_id: string }>("cashflow/plaid/exchange", {
              public_token: token,
            })
              .then((item) =>
                api("cashflow/plaid/" + item.item_id + "/sync", {}),
              )
              .then(resolve)
              .catch((e) => {
                setError(
                  e instanceof Error ? e.message : "Sandbox sync failed",
                );
                resolve(null);
              })
              .finally(() => handler.destroy());
          },
          onExit: () => {
            handler.destroy();
            resolve(null);
          },
        });
        handler.open();
      });
    });
  }
  function selectSection(next: Section) {
    setSection(next);
    setResult(null);
    setError("");
  }
  if (checking)
    return (
      <main className="login">
        <div className="brand">
          Q<span>QUANTARA</span>
        </div>
        <p>Opening your workspace…</p>
      </main>
    );
  if (!user)
    return (
      <main className="login">
        <div className="login-theme">
          <ThemeSwitch
            theme={preferences.theme}
            onChange={() => void toggleTheme()}
          />
        </div>
        <div className="login-card">
          <div className="brand">
            <b>Q</b>
            <span>Quantara</span>
          </div>
          <p className="eyebrow">LOCAL-FIRST RESEARCH</p>
          <h1>
            Make room for
            <br />
            better questions.
          </h1>
          <p>
            Portfolios, strategy simulation and grounded AI in one research
            workspace.
          </p>
          <form
            onSubmit={async (e) => {
              e.preventDefault();
              const data = new FormData(e.currentTarget);
              setBusy(true);
              setError("");
              try {
                const v = await api<{ username: string }>("auth/login", {
                  username: get(data, "username"),
                  password: get(data, "password"),
                });
                setUser(v.username);
                await refresh();
              } catch (e) {
                setError(e instanceof Error ? e.message : "Sign-in failed");
              } finally {
                setBusy(false);
              }
            }}
          >
            <Field name="username" title="Username" value="demo" />
            <Field
              name="password"
              title="Password"
              type="password"
              value="quantara-local-demo"
            />
            {error && (
              <p role="alert" className="error">
                {error}
              </p>
            )}
            <button disabled={busy} className="primary">
              {busy ? "Signing in…" : "Enter workspace →"}
            </button>
          </form>
          <small>
            Local demo account. Configure team credentials before sharing
            access.
          </small>
        </div>
        <div className="login-art">
          <span className="orbit one" />
          <span className="orbit two" />
          <span className="orbit three" />
          <div>
            Ideas.
            <br />
            Evidence.
            <br />
            <em>Perspective.</em>
          </div>
        </div>
      </main>
    );
  return (
    <div className="workspace">
      <aside className="sidebar">
        <div className="brand">
          <b>Q</b>
          <span>
            Quantara<small>RESEARCH WORKSPACE</small>
          </span>
        </div>
        <p className="nav-caption">WORKSPACE</p>
        <nav aria-label="Workspace">
          {sections.map((s) => (
            <button
              key={s}
              aria-label={s}
              className={section === s ? "nav-item selected" : "nav-item"}
              onClick={() => selectSection(s)}
            >
              <Icon name={s} />
              <span>{s}</span>
              {s === "Paper trading" &&
                papers.some((p) => p.status === "running") && <i />}
            </button>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <div className="model-status">
            <i className={health.model_ready ? "online" : ""} />
            <div>
              <strong>Local intelligence</strong>
              <small>
                {health.model_ready
                  ? text(health.model)
                  : "Model offline · calculations ready"}
              </small>
            </div>
          </div>
          <button
            className="user"
            onClick={() => {
              void api("auth/logout", {}).then(() => setUser(null));
            }}
            title="Sign out"
          >
            <span className="avatar">{user[0].toUpperCase()}</span>
            <span>
              {user}
              <small>Research account</small>
            </span>
            <span>↪</span>
          </button>
        </div>
      </aside>
      <main className="main">
        <header className="topbar">
          <div>
            <span>Workspace</span>
            <span className="slash">/</span>
            <strong>{section}</strong>
          </div>
          <div className="top-actions">
            <span className="tag">VIRTUAL ORDERS ONLY</span>
            <ThemeSwitch
              theme={preferences.theme}
              onChange={() => void toggleTheme()}
              disabled={savingTheme || busy}
            />
            <button
              className="quiet"
              disabled={busy}
              onClick={() => void act(refresh)}
            >
              ↻ Refresh
            </button>
          </div>
        </header>
        <div className="content">
          <div className="page-heading">
            <div>
              <p className="eyebrow">
                {section === "Overview"
                  ? "RESEARCH HOME"
                  : section.toUpperCase()}
              </p>
              <h1>{titles[section]}</h1>
              <p>
                {section === "Overview"
                  ? "A clear view of your portfolio, experiments and next questions."
                  : "Run the calculation. Keep the evidence. Compare what changes."}
              </p>
            </div>
            {section === "Overview" && (
              <button
                className="primary"
                disabled={busy}
                onClick={() => void submit("demo", {})}
              >
                Load research demo <span>↗</span>
              </button>
            )}
          </div>
          {error && (
            <div className="error" role="alert">
              {error}
              <button onClick={() => setError("")} aria-label="Dismiss error">
                ×
              </button>
            </div>
          )}
          {active.length > 0 && (
            <div className="job-list">
              {active.map((j) => (
                <div className="job" key={j.id}>
                  <div>
                    <strong>{label(j.operation)}</strong>
                    <span>{j.message}</span>
                  </div>
                  <progress max="1" value={j.progress} />
                  <button
                    onClick={() => void submit("jobs/" + j.id + "/cancel", {})}
                  >
                    Cancel
                  </button>
                </div>
              ))}
            </div>
          )}
          {!datasets.length && section !== "Overview" && (
            <p className="notice">
              Start with “Load research demo” on Overview, or import your own
              records and market data.
            </p>
          )}
          {section === "Overview" && (
            <>
              <div className="stats">
                <div className="stat">
                  <span>Recorded cash</span>
                  <strong>
                    {money(
                      portfolios
                        .filter((p) => p.currency === "USD")
                        .reduce((n, p) => n + p.balance.cash, 0),
                    )}
                  </strong>
                  <small>
                    Across {portfolios.length} portfolios · USD shown
                    {portfolios.some((p) => p.currency === "CAD") &&
                      " · CAD " +
                        money(
                          portfolios
                            .filter((p) => p.currency === "CAD")
                            .reduce((n, p) => n + p.balance.cash, 0),
                          "CAD",
                        )}
                  </small>
                </div>
                <div className="stat">
                  <span>Research datasets</span>
                  <strong>{datasets.length.toString().padStart(2, "0")}</strong>
                  <small>
                    {datasets.filter((d) => d.fixture).length} synthetic
                    fixtures
                  </small>
                </div>
                <div className="stat">
                  <span>Completed simulations</span>
                  <strong>
                    {backtests.length.toString().padStart(2, "0")}
                  </strong>
                  <small>{strategies.length} saved strategy definitions</small>
                </div>
                <div className="stat">
                  <span>Paper accounts</span>
                  <strong>
                    {papers
                      .filter((p) => p.status === "running")
                      .length.toString()
                      .padStart(2, "0")}
                  </strong>
                  <small>{papers.length} virtual accounts in total</small>
                </div>
              </div>
              <div className="grid wide">
                <Panel
                  title="Latest experiment"
                  subtitle={
                    lastTest
                      ? text(lastTest.name)
                      : "Your first result starts with a question."
                  }
                >
                  {lastTest && lastTest.engine_version !== 2 && (
                    <p className="notice">
                      Earlier simulator result. Rerun this experiment with the
                      corrected engine.
                    </p>
                  )}
                  <Chart data={lastEquity} />
                  {lastTest && (
                    <div className="metric-strip">
                      {["total_return", "max_drawdown", "sharpe"].map((k) => (
                        <div key={k}>
                          <small>{label(k)}</small>
                          <strong>
                            {typeof record(lastTest.metrics)[k] === "number"
                              ? k === "sharpe"
                                ? number(record(lastTest.metrics)[k]).toFixed(3)
                                : (
                                    number(record(lastTest.metrics)[k]) * 100
                                  ).toFixed(2) + "%"
                              : "—"}
                          </strong>
                        </div>
                      ))}
                    </div>
                  )}
                </Panel>
                <Panel
                  title="Continue your research"
                  subtitle="From records to a reproducible result."
                >
                  <div className="workflow">
                    {(
                      [
                        "Portfolio",
                        "Market data",
                        "Strategies",
                        "Simulator",
                        "Paper trading",
                        "Assistant",
                      ] as Section[]
                    ).map((s, i) => (
                      <button key={s} onClick={() => selectSection(s)}>
                        <span>{String(i + 1).padStart(2, "0")}</span>
                        <div>
                          <strong>{s}</strong>
                          <small>
                            {
                              [
                                "Import and reconcile transactions",
                                "Check source and coverage",
                                "Define a repeatable rule",
                                "Compare costs and benchmark",
                                "Replay or follow arriving bars",
                                "Explain the recorded evidence",
                              ][i]
                            }
                          </small>
                        </div>
                        <b>↗</b>
                      </button>
                    ))}
                  </div>
                </Panel>
              </div>
              <Panel
                title="Research activity"
                subtitle="Saved work stays available for review."
              >
                <Table
                  data={[
                    ...backtests.slice(-3).map((b) => ({
                      name: b.name,
                      type: "Simulation",
                      status: "Complete",
                    })),
                    ...reports.slice(-3).map((r) => ({
                      name: r.name,
                      type: "Analysis",
                      status: "Complete",
                    })),
                  ]}
                  keys={["name", "type", "status"]}
                />
              </Panel>
            </>
          )}
          {section === "Portfolio" && (
            <>
              <div className="grid">
                <Panel title="Your portfolios">
                  <label className="field">
                    <span>Active portfolio</span>
                    <select
                      value={selectedPortfolio}
                      onChange={(e) => setPortfolio(e.target.value)}
                    >
                      {portfolios.map((p) => (
                        <option key={p.id} value={p.id}>
                          {p.name}
                        </option>
                      ))}
                    </select>
                  </label>
                  {portfolio && (
                    <>
                      <div className="balance">
                        <small>{portfolio.currency} cash</small>
                        <strong>
                          {money(portfolio.balance.cash, portfolio.currency)}
                        </strong>
                      </div>
                      <Table
                        data={Object.entries(portfolio.balance.holdings).map(
                          ([symbol, quantity]) => ({ symbol, quantity }),
                        )}
                        keys={["symbol", "quantity"]}
                      />
                      <label className="file">
                        Import transaction CSV
                        <input
                          type="file"
                          accept=".csv"
                          onChange={(e) =>
                            void fileChange(
                              "portfolios/" + portfolio.id + "/csv",
                              e.target.files?.[0],
                            )
                          }
                        />
                      </label>
                      <small>
                        Columns: external_id, timestamp (with timezone), type,
                        symbol, quantity, price, amount, fee, currency, fx_cad,
                        account, account_type, affiliated.
                      </small>
                    </>
                  )}
                </Panel>
                <Panel
                  title="Create a portfolio"
                  subtitle="Start with a cash deposit, then import your transactions."
                >
                  <form
                    onSubmit={form((d) =>
                      api("portfolios", {
                        name: get(d, "name"),
                        currency: get(d, "currency"),
                        transactions: [
                          {
                            external_id: crypto.randomUUID(),
                            timestamp: new Date().toISOString(),
                            type: "deposit",
                            amount: num(d, "capital"),
                            currency: get(d, "currency"),
                          },
                        ],
                      }),
                    )}
                  >
                    <Field
                      name="name"
                      title="Portfolio name"
                      value="My research portfolio"
                    />
                    <Field
                      name="currency"
                      title="Account currency"
                      value="USD"
                      options={["USD", "CAD"]}
                    />
                    <Field
                      name="capital"
                      title="Initial cash"
                      type="number"
                      min={1}
                      value={100000}
                    />
                    <button className="primary" disabled={busy}>
                      Create portfolio
                    </button>
                  </form>
                </Panel>
              </div>
              {portfolio && (
                <Panel title="Transaction ledger">
                  <Table
                    data={
                      portfolio.transactions as unknown as Record<
                        string,
                        Json
                      >[]
                    }
                    keys={[
                      "timestamp",
                      "type",
                      "symbol",
                      "quantity",
                      "price",
                      "amount",
                      "fee",
                      "fx_cad",
                    ]}
                  />
                </Panel>
              )}
              <Panel
                title="Risk and allocation"
                subtitle="Weights are calculated in Python from the selected history."
              >
                <form
                  className="form-grid"
                  onSubmit={form((d) =>
                    api(
                      get(d, "analysis") === "risk"
                        ? "risk"
                        : "portfolios/optimize",
                      get(d, "analysis") === "risk"
                        ? { dataset_id: selectedDataset, weights: equalWeights }
                        : {
                            dataset_id: selectedDataset,
                            method: get(d, "method"),
                            holdings: equalWeights,
                            max_weight: num(d, "cap"),
                            views: Object.fromEntries(
                              get(d, "views")
                                .split(",")
                                .filter(Boolean)
                                .map((v) => {
                                  const [s, w] = v.trim().split("=");
                                  return [s.toUpperCase(), Number(w)];
                                }),
                            ),
                          },
                    ),
                  )}
                >
                  <label className="field">
                    <span>Market history</span>
                    <select
                      value={selectedDataset}
                      onChange={(e) => setDataset(e.target.value)}
                    >
                      {datasetOptions.map((o) => (
                        <option key={o.value} value={o.value}>
                          {o.label}
                        </option>
                      ))}
                    </select>
                  </label>
                  <Field
                    name="analysis"
                    title="Analysis"
                    options={["risk", "optimize"]}
                  />
                  <Field
                    name="method"
                    title="Allocation method"
                    options={methods}
                  />
                  <Field
                    name="cap"
                    title="Maximum asset weight"
                    type="number"
                    min={0.01}
                    max={1}
                    step={0.01}
                    value={1}
                  />
                  <Field
                    name="views"
                    title="Annual return views (Black–Litterman)"
                    value=""
                    required={false}
                  />
                  <button className="primary" disabled={busy || !dataset}>
                    Run analysis
                  </button>
                </form>
                <small>
                  Risk uses equal weights over recorded holdings (or dataset
                  symbols when no holdings exist). Return views use
                  AAPL=0.08,MSFT=0.06.
                </small>
              </Panel>
            </>
          )}
          {section === "Market data" && (
            <>
              <Panel
                title="Research watchlist"
                subtitle="Saved symbols for your research workspace."
              >
                <form
                  className="form-grid"
                  onSubmit={form((d) =>
                    api(
                      "settings",
                      {
                        watchlist: [...new Set(split(get(d, "watchlist")))],
                        theme: preferences.theme,
                      },
                      "PUT",
                    ),
                  )}
                >
                  <Field
                    key={preferences.watchlist.join(",")}
                    name="watchlist"
                    title="Watchlist symbols"
                    value={preferences.watchlist.join(",")}
                    required={false}
                  />
                  <button className="primary" disabled={busy}>
                    Save watchlist
                  </button>
                </form>
              </Panel>
              <Panel
                title="History library"
                subtitle="Fixture labels and provider coverage travel with every result."
              >
                <Table
                  data={datasets as unknown as Record<string, Json>[]}
                  keys={[
                    "name",
                    "interval",
                    "source",
                    "fixture",
                    "start",
                    "end",
                    "bars_count",
                    "coverage",
                  ]}
                />
              </Panel>
              <div className="grid">
                <Panel title="Fetch research history">
                  <form
                    className="form-grid"
                    onSubmit={form((d) =>
                      api("market/fetch", {
                        symbols: split(get(d, "symbols")),
                        start: get(d, "start") + "T00:00:00Z",
                        end: get(d, "end") + "T00:00:00Z",
                        interval: get(d, "interval"),
                        provider: get(d, "provider"),
                      }),
                    )}
                  >
                    <Field
                      name="symbols"
                      title="Symbols"
                      value="AAPL,MSFT,SPY"
                    />
                    <Field
                      name="provider"
                      title="Provider"
                      options={["yahoo", "alpaca"]}
                    />
                    <Field
                      name="start"
                      title="From"
                      type="date"
                      value="2024-01-01"
                    />
                    <Field
                      name="end"
                      title="Until (exclusive)"
                      type="date"
                      value="2025-01-01"
                    />
                    <Field
                      name="interval"
                      title="Bar interval"
                      options={["1d", "1m", "5m"]}
                    />
                    <button className="primary" disabled={busy}>
                      Fetch history
                    </button>
                  </form>
                  <small>
                    Yahoo intraday coverage is limited. Alpaca needs backend
                    credentials and uses IEX coverage.
                  </small>
                </Panel>
                <Panel
                  title="Import an archive"
                  subtitle="Raw, aligned OHLCV bars and explicit corporate actions."
                >
                  <form
                    onSubmit={form(async (d) => {
                      const file = d.get("file") as File;
                      return upload("market/csv", file, {
                        name: get(d, "name"),
                        interval: get(d, "interval"),
                        source: get(d, "source"),
                        coverage: get(d, "coverage"),
                      });
                    })}
                  >
                    <Field
                      name="name"
                      title="Dataset name"
                      value="Imported history"
                    />
                    <Field
                      name="interval"
                      title="Interval"
                      options={["1d", "1m", "5m"]}
                    />
                    <Field name="source" title="Source" value="csv-research" />
                    <Field
                      name="coverage"
                      title="Coverage notes"
                      value="User-supplied raw bars"
                    />
                    <Field name="file" title="Bar CSV" type="file" />
                    <button className="primary" disabled={busy}>
                      Validate and import
                    </button>
                  </form>
                  <small>
                    Columns: timestamp,symbol,open,high,low,close,volume. Import
                    actions with the full dataset JSON API.
                  </small>
                </Panel>
              </div>
            </>
          )}
          {section === "Strategies" && (
            <div className="grid">
              <Panel title="Strategy library">
                <Table
                  data={strategies as unknown as Record<string, Json>[]}
                  keys={[
                    "name",
                    "type",
                    "symbols",
                    "fast",
                    "slow",
                    "rebalance_every",
                  ]}
                />
              </Panel>
              <Panel
                title="Create a strategy"
                subtitle="Rules are validated before they can execute."
              >
                <form
                  className="form-grid"
                  onSubmit={form((d) =>
                    api("strategies", {
                      name: get(d, "name"),
                      type: get(d, "type"),
                      symbols: split(get(d, "symbols")),
                      fast: num(d, "fast"),
                      slow: num(d, "slow"),
                      rebalance_every: num(d, "rebalance"),
                      method: get(d, "method"),
                      rsi_buy: num(d, "rsi_buy"),
                      rsi_sell: num(d, "rsi_sell"),
                      sentiment_threshold: num(d, "threshold"),
                      ...(get(d, "limit")
                        ? { limit_offset_bps: num(d, "limit") }
                        : {}),
                      ...(get(d, "model_id")
                        ? { model_id: get(d, "model_id") }
                        : {}),
                    }),
                  )}
                >
                  <Field
                    name="name"
                    title="Strategy name"
                    value="Moving average research"
                  />
                  <Field
                    name="type"
                    title="Rule template"
                    value="sma"
                    options={[
                      "buy_hold",
                      "sma",
                      "rsi",
                      "momentum",
                      "rebalance",
                      "sentiment",
                      "rl",
                    ]}
                  />
                  <Field name="symbols" title="Symbols" value="AAPL,MSFT" />
                  <Field
                    name="fast"
                    title="Fast / RSI window (bars)"
                    type="number"
                    min={2}
                    value={10}
                  />
                  <Field
                    name="slow"
                    title="Slow / momentum window (bars)"
                    type="number"
                    min={3}
                    value={30}
                  />
                  <Field
                    name="rebalance"
                    title="Rebalance every (bars)"
                    type="number"
                    min={1}
                    value={20}
                  />
                  <Field
                    name="method"
                    title="Rebalance allocation"
                    options={methods.filter((m) => m !== "black_litterman")}
                  />
                  <Field
                    name="rsi_buy"
                    title="RSI buy threshold"
                    type="number"
                    min={0}
                    max={100}
                    value={30}
                  />
                  <Field
                    name="rsi_sell"
                    title="RSI sell threshold"
                    type="number"
                    min={0}
                    max={100}
                    value={70}
                  />
                  <Field
                    name="threshold"
                    title="Sentiment threshold"
                    type="number"
                    min={-1}
                    max={1}
                    step={0.01}
                    value={0.2}
                  />
                  <Field
                    name="limit"
                    title="Limit offset (bps, blank for market)"
                    type="number"
                    min={0}
                    required={false}
                  />
                  <Field
                    name="model_id"
                    title="Trained policy"
                    required={false}
                    options={[
                      { value: "", label: "No policy" },
                      ...models.map((m) => ({
                        value: m.id,
                        label:
                          text(m.name) +
                          (m.observation_version !== 2
                            ? " · Retrain required"
                            : ""),
                      })),
                    ]}
                  />
                  <button className="primary" disabled={busy}>
                    Save strategy
                  </button>
                </form>
              </Panel>
            </div>
          )}
          {section === "Simulator" && (
            <>
              <Panel
                title="Historical simulation"
                subtitle="Long-only • next-bar execution • reproducible dataset versions"
              >
                <form
                  className="form-grid"
                  onSubmit={form((d) =>
                    api("backtests", {
                      dataset_id: get(d, "dataset_id"),
                      strategy_id: get(d, "strategy_id"),
                      capital: num(d, "capital"),
                      commission: num(d, "commission"),
                      slippage_bps: num(d, "slippage"),
                      spread_bps: num(d, "spread"),
                      participation: num(d, "participation"),
                      benchmark: get(d, "benchmark"),
                      evaluation: get(d, "evaluation"),
                    }),
                  )}
                >
                  <Field
                    name="dataset_id"
                    title="History"
                    options={datasetOptions}
                  />
                  <Field
                    name="strategy_id"
                    title="Strategy"
                    options={strategyOptions}
                  />
                  <Field
                    name="capital"
                    title="Initial USD capital"
                    type="number"
                    min={1}
                    value={100000}
                  />
                  <Field
                    name="commission"
                    title="Commission per fill"
                    type="number"
                    min={0}
                    step={0.01}
                    value={0}
                  />
                  <Field
                    name="slippage"
                    title="Slippage (basis points)"
                    type="number"
                    min={0}
                    step={0.1}
                    value={5}
                  />
                  <Field
                    name="spread"
                    title="Bid/ask spread (basis points)"
                    type="number"
                    min={0}
                    max={1000}
                    step={0.1}
                    value={0}
                  />
                  <Field
                    name="participation"
                    title="Maximum share of bar volume"
                    type="number"
                    min={0.000001}
                    max={1}
                    step="any"
                    value={0.01}
                  />
                  <Field
                    name="benchmark"
                    title="Benchmark symbol"
                    value="SPY"
                  />
                  <Field
                    name="evaluation"
                    title="Evaluation period"
                    options={["full", "test", "walk_forward"]}
                  />
                  <button
                    className="primary"
                    disabled={busy || !strategies.length}
                  >
                    Run simulation →
                  </button>
                </form>
                <small>
                  Chronological split: 60% training, 20% validation, 20% test.
                  Walk-forward summarizes consecutive test windows using fixed
                  rules, without retraining.
                </small>
              </Panel>
              <Panel title="Saved experiments">
                <div className="experiment-list">
                  {backtests.map((b) => (
                    <button key={b.id} onClick={() => setResult(b)}>
                      <Icon name="Simulator" />
                      <span>
                        <strong>{text(b.name)}</strong>
                        <small>
                          {b.fixture ? "Synthetic fixture" : text(b.source)} ·{" "}
                          {text(b.evaluation)}
                          {b.engine_version !== 2 && " · Rerun required"}
                        </small>
                      </span>
                      <b>
                        {(number(record(b.metrics).total_return) * 100).toFixed(
                          2,
                        )}
                        %
                      </b>
                      <a
                        href={"/api/backtests/" + b.id + "/trades.csv"}
                        onClick={(e) => e.stopPropagation()}
                      >
                        Trades ↓
                      </a>
                    </button>
                  ))}
                </div>
              </Panel>
            </>
          )}
          {section === "Paper trading" && (
            <>
              <Panel
                title="Create a virtual account"
                subtitle="Replay imported bars or poll arriving provider bars."
              >
                <form
                  className="form-grid"
                  onSubmit={form((d) =>
                    api("paper-accounts", {
                      name: get(d, "name"),
                      mode: get(d, "mode"),
                      feed: get(d, "feed"),
                      config: {
                        dataset_id: get(d, "dataset_id"),
                        strategy_id: get(d, "strategy_id"),
                        capital: num(d, "capital"),
                        commission: num(d, "commission"),
                        slippage_bps: num(d, "slippage"),
                        spread_bps: num(d, "spread"),
                        participation: num(d, "participation"),
                        benchmark: get(d, "benchmark"),
                      },
                    }),
                  )}
                >
                  <Field
                    name="name"
                    title="Account name"
                    value="Strategy paper account"
                  />
                  <Field
                    name="dataset_id"
                    title="History / warmup"
                    options={datasetOptions}
                  />
                  <Field
                    name="strategy_id"
                    title="Strategy"
                    options={strategyOptions}
                  />
                  <Field
                    name="mode"
                    title="Mode"
                    options={["replay", "forward"]}
                  />
                  <Field
                    name="feed"
                    title="Forward provider"
                    options={["alpaca", "yahoo"]}
                  />
                  <Field
                    name="capital"
                    title="Initial USD cash"
                    type="number"
                    min={1}
                    value={100000}
                  />
                  <ExecutionCosts />
                  <button className="primary" disabled={busy}>
                    Create account
                  </button>
                </form>
                <small>
                  Forward mode requires raw provider history. In native
                  development, use Poll arriving bars. Docker's worker polls
                  running accounts every minute.
                </small>
              </Panel>
              <div className="grid">
                {papers.map((p) => (
                  <Panel
                    key={p.id}
                    title={text(p.name)}
                    subtitle={
                      label(text(p.mode)) + " · " + label(text(p.status))
                    }
                  >
                    <div className="balance">
                      <small>Cash remaining</small>
                      <strong>{money(number(record(p.state).cash))}</strong>
                    </div>
                    {p.stale && (
                      <p className="notice">
                        Feed is stale. Review the most recent provider
                        timestamp.
                      </p>
                    )}
                    <p className="muted">
                      Last event:{" "}
                      {text(p.last_feed_timestamp) || "Waiting to start"}
                    </p>
                    <div className="button-row">
                      {(["start", "pause", "resume", "stop"] as const).map(
                        (action) => (
                          <button
                            key={action}
                            disabled={
                              busy ||
                              (action === "start"
                                ? p.status !== "created"
                                : action === "pause"
                                  ? p.status !== "running"
                                  : action === "resume"
                                    ? p.status !== "paused"
                                    : p.status === "stopped")
                            }
                            onClick={() =>
                              void submit(
                                "paper-accounts/" + p.id + "/control",
                                { action },
                              )
                            }
                          >
                            {label(action)}
                          </button>
                        ),
                      )}
                      <button
                        className="primary"
                        disabled={busy || p.status !== "running"}
                        onClick={() =>
                          void submit(
                            "paper-accounts/" +
                              p.id +
                              (p.mode === "replay" ? "/step" : "/poll"),
                            p.mode === "replay" ? { bars: 20 } : {},
                          )
                        }
                      >
                        {p.mode === "replay"
                          ? "Replay 20 bars"
                          : "Poll arriving bars"}
                      </button>
                      <button onClick={() => setResult(p)}>
                        Inspect orders
                      </button>
                    </div>
                  </Panel>
                ))}
              </div>
            </>
          )}
          {section === "Signals" && (
            <>
              <div className="grid">
                <Panel
                  title="Classify a headline"
                  subtitle="Local classifiers remain separate from Qwen."
                >
                  <form
                    onSubmit={form((d) =>
                      api("signals/import", {
                        method: get(d, "method"),
                        items: [
                          {
                            external_id: crypto.randomUUID(),
                            timestamp: get(d, "timestamp") + "Z",
                            available_at: get(d, "timestamp") + "Z",
                            symbols: split(get(d, "symbols")),
                            text: get(d, "text"),
                            source: "manual-research",
                          },
                        ],
                      }),
                    )}
                  >
                    <Field name="symbols" title="Symbols" value="AAPL" />
                    <Field
                      name="timestamp"
                      title="Available at (UTC)"
                      type="datetime-local"
                      value="2024-06-03T14:30"
                    />
                    <label className="field">
                      <span>Headline / public post</span>
                      <textarea
                        name="text"
                        required
                        placeholder="Paste the timestamped source text…"
                      />
                    </label>
                    <Field
                      name="method"
                      title="Classifier"
                      options={["baseline", "distilbert"]}
                    />
                    <button className="primary" disabled={busy}>
                      Classify and record
                    </button>
                  </form>
                  <label className="file">
                    Import timestamped news CSV
                    <input
                      type="file"
                      accept=".csv"
                      onChange={(e) =>
                        void fileChange("signals/csv", e.target.files?.[0], {
                          method: "baseline",
                        })
                      }
                    />
                  </label>
                </Panel>
                <Panel
                  title="Evaluate the signal"
                  subtitle="Labels crossing the train/test boundary are purged."
                >
                  <form
                    onSubmit={form((d) =>
                      api("signals/evaluate", {
                        dataset_id: get(d, "dataset_id"),
                        horizon_bars: num(d, "horizon"),
                      }),
                    )}
                  >
                    <Field
                      name="dataset_id"
                      title="Price history"
                      options={datasetOptions}
                    />
                    <Field
                      name="horizon"
                      title="Prediction horizon (bars)"
                      type="number"
                      min={1}
                      value={5}
                    />
                    <button className="primary" disabled={busy}>
                      Evaluate news impact
                    </button>
                  </form>
                  <form
                    onSubmit={form((d) =>
                      api("signals/fetch", {
                        symbols: split(get(d, "symbols")),
                        start: get(d, "start") + "T00:00:00Z",
                        end: get(d, "end") + "T00:00:00Z",
                        method: get(d, "method"),
                      }),
                    )}
                  >
                    <h3>Provider news</h3>
                    <Field name="symbols" title="Symbols" value="AAPL,MSFT" />
                    <Field
                      name="start"
                      title="From"
                      type="date"
                      value="2024-01-01"
                    />
                    <Field
                      name="end"
                      title="Until"
                      type="date"
                      value="2025-01-01"
                    />
                    <Field
                      name="method"
                      title="Classifier"
                      options={["baseline", "distilbert"]}
                    />
                    <button disabled={busy}>Fetch Alpaca news</button>
                  </form>
                </Panel>
              </div>
              <Panel title="Recorded signals">
                <Table
                  data={signals}
                  keys={[
                    "available_at",
                    "symbols",
                    "text",
                    "score",
                    "label",
                    "model",
                    "source",
                    "fixture",
                  ]}
                />
              </Panel>
            </>
          )}
          {section === "Cash flow" && (
            <div className="grid">
              <Panel
                title="Cash-flow history"
                subtitle="Import income and spending in one currency."
              >
                <label className="file">
                  Import cash transaction CSV
                  <input
                    type="file"
                    accept=".csv"
                    onChange={(e) =>
                      void fileChange("cashflow/csv", e.target.files?.[0])
                    }
                  />
                </label>
                <small>
                  Columns: external_id,timestamp,amount,category,currency.
                  Income is positive, spending negative.
                </small>
                <button
                  onClick={() => void submit("cashflow/demo", {})}
                  disabled={busy}
                >
                  Load synthetic cash-flow fixture
                </button>
                <button onClick={() => void connectSandbox()} disabled={busy}>
                  Connect Plaid Sandbox
                </button>
                <Table
                  data={cashbook as Record<string, Json>[]}
                  keys={["timestamp", "category", "amount", "currency"]}
                />
              </Panel>
              <Panel
                title="Forecast and shortfall alerts"
                subtitle="Evaluate a chronological holdout, then refit on recorded history."
              >
                <form
                  onSubmit={form((d) =>
                    api("cashflow", {
                      transactions: cashbook,
                      currency: get(d, "currency"),
                      balance: num(d, "balance"),
                      days: num(d, "days"),
                      threshold: num(d, "threshold"),
                      method: get(d, "method"),
                    }),
                  )}
                >
                  <Field
                    name="balance"
                    title="Current cash balance"
                    type="number"
                    value={10000}
                  />
                  <Field
                    name="currency"
                    title="Currency"
                    options={["CAD", "USD"]}
                  />
                  <Field
                    name="days"
                    title="Forecast horizon (days)"
                    type="number"
                    min={1}
                    max={365}
                    value={30}
                  />
                  <Field
                    name="threshold"
                    title="Alert below this balance"
                    type="number"
                    value={0}
                  />
                  <Field
                    name="method"
                    title="Forecast model"
                    options={["baseline", "prophet", "lstm"]}
                  />
                  <button
                    className="primary"
                    disabled={busy || !cashbook.length}
                  >
                    Forecast cash flow
                  </button>
                </form>
                <small>
                  Prophet and LSTM require the optional model dependencies and
                  at least 90 days of history.
                </small>
              </Panel>
            </div>
          )}
          {section === "Tax research" && (
            <>
              <Panel
                title="Canadian loss research"
                subtitle="CAD pooled adjusted cost base across supplied taxable accounts."
              >
                <form
                  className="form-grid"
                  onSubmit={form((d) =>
                    api("tax/research", {
                      portfolio_id: get(d, "portfolio_id"),
                      as_of: get(d, "as_of") + "T23:59:59Z",
                      affiliated_records_complete: d.get("complete") === "on",
                      marks_cad: Object.fromEntries(
                        get(d, "marks")
                          .split(",")
                          .filter(Boolean)
                          .map((v) => {
                            const [s, p] = v.trim().split("=");
                            return [s.toUpperCase(), Number(p)];
                          }),
                      ),
                    }),
                  )}
                >
                  <Field
                    name="portfolio_id"
                    title="Transaction ledger"
                    options={portfolios.map((p) => ({
                      value: p.id,
                      label: p.name,
                    }))}
                  />
                  <Field
                    name="as_of"
                    title="As of date"
                    type="date"
                    value={new Date().toISOString().slice(0, 10)}
                  />
                  <Field
                    name="marks"
                    title="Current marks in CAD (for harvesting)"
                    value="AAPL=200,MSFT=500"
                    required={false}
                  />
                  <label className="check">
                    <input type="checkbox" name="complete" />
                    All own and affiliated account activity is included
                  </label>
                  <button
                    className="primary"
                    disabled={busy || !portfolios.length}
                  >
                    Research tax positions
                  </button>
                </form>
                <p className="muted">
                  USD trades require their transaction-date CAD exchange rate.
                  Registered accounts inform replacement checks. Unfinished
                  30-day windows and missing records stay provisional.
                </p>
              </Panel>
              <Panel title="Review the assumptions">
                <p>
                  Include all accounts that hold identical property, including
                  affiliated and registered activity. Complex overlapping
                  disposals or affiliated ACB transfers are marked for
                  individual review.
                </p>
                <a
                  href="https://www.canada.ca/en/revenue-agency/services/tax/individuals/topics/about-your-tax-return/tax-return/completing-a-tax-return/personal-income/line-12700-capital-gains/capital-losses-deductions.html/1000"
                  target="_blank"
                  rel="noreferrer"
                >
                  CRA superficial-loss guidance ↗
                </a>
              </Panel>
            </>
          )}
          {section === "Models" && (
            <>
              <Panel
                title="Train a research policy"
                subtitle="The environment shares the simulator's costs and accounting."
              >
                <form
                  className="form-grid"
                  onSubmit={form((d) =>
                    api("models/train", {
                      dataset_id: get(d, "dataset_id"),
                      symbols: split(get(d, "symbols")),
                      algorithm: get(d, "algorithm"),
                      timesteps: num(d, "steps"),
                      seed: num(d, "seed"),
                      capital: num(d, "capital"),
                      commission: num(d, "commission"),
                      slippage_bps: num(d, "slippage"),
                      spread_bps: num(d, "spread"),
                      participation: num(d, "participation"),
                      benchmark: get(d, "benchmark"),
                    }),
                  )}
                >
                  <Field
                    name="dataset_id"
                    title="Training history"
                    options={datasetOptions}
                  />
                  <Field
                    name="symbols"
                    title="Policy symbols"
                    value="AAPL,MSFT"
                  />
                  <Field
                    name="algorithm"
                    title="Algorithm"
                    options={["PPO", "DDPG"]}
                  />
                  <Field
                    name="steps"
                    title="Training steps"
                    type="number"
                    min={100}
                    max={1000000}
                    value={2000}
                  />
                  <Field
                    name="seed"
                    title="Random seed"
                    type="number"
                    min={0}
                    value={42}
                  />
                  <Field
                    name="capital"
                    title="Initial USD capital"
                    type="number"
                    min={1}
                    value={100000}
                  />
                  <ExecutionCosts />
                  <button className="primary" disabled={busy}>
                    Train and evaluate
                  </button>
                </form>
                <small>
                  Local CPU training requires optional model dependencies. A
                  saved checkpoint is compared with buy-and-hold, SMA and
                  momentum on held-out history.
                </small>
              </Panel>
              <Panel title="Saved checkpoints">
                <div className="experiment-list">
                  {models.map((m) => (
                    <button key={m.id} onClick={() => setResult(m)}>
                      <Icon name="Models" />
                      <span>
                        <strong>{text(m.name)}</strong>
                        <small>
                          Seed {number(m.seed)} · test starts{" "}
                          {text(m.test_start)}
                          {m.observation_version !== 2 && " · Retrain required"}
                        </small>
                      </span>
                      <b>Inspect ↗</b>
                    </button>
                  ))}
                </div>
              </Panel>
            </>
          )}
          {section === "Assistant" && (
            <div className="grid wide">
              <Panel
                title="Research assistant"
                subtitle={
                  health.model_ready
                    ? "Qwen is ready to explain recorded evidence."
                    : "Qwen is offline. Your research tools remain available."
                }
                className="chat-panel"
              >
                <label className="field">
                  <span>Saved conversation</span>
                  <select
                    value={conversation || ""}
                    onChange={(e) => {
                      const id = e.target.value;
                      setConversation(id || undefined);
                      const c = conversationList.find((c) => c.id === id);
                      setMessages(
                        rows(c?.messages).map((m) => ({
                          role: text(m.role),
                          content: text(m.content),
                          warning: text(m.warning),
                        })),
                      );
                    }}
                  >
                    <option value="">New conversation</option>
                    {conversationList.map((c) => (
                      <option key={c.id} value={c.id}>
                        {text(c.title)}
                      </option>
                    ))}
                  </select>
                </label>
                <div className="chat-messages">
                  {!messages.length && (
                    <div className="chat-empty">
                      <div className="assistant-mark">✦</div>
                      <h3>Bring a question. Start with evidence.</h3>
                      <p>
                        Ask Qwen to analyze your portfolio, compare a simulation
                        or explain an imported document. Calculations are run by
                        registered Python tools.
                      </p>
                    </div>
                  )}
                  {messages.map((m, i) => (
                    <div key={i} className={"message " + m.role}>
                      <small>
                        {m.role === "user" ? user : "Quantara · Qwen"}
                      </small>
                      <ReactMarkdown skipHtml>{m.content}</ReactMarkdown>
                      {m.warning && <p className="notice">{m.warning}</p>}
                    </div>
                  ))}
                </div>
                <form
                  className="chat-compose"
                  onSubmit={(e) => {
                    e.preventDefault();
                    const form = e.currentTarget;
                    const data = new FormData(form);
                    const message = get(data, "message");
                    setMessages((v) => [
                      ...v,
                      { role: "user", content: message },
                    ]);
                    form.reset();
                    void submit("conversations/chat", {
                      message,
                      conversation_id: conversation,
                      portfolio_id: selectedPortfolio || null,
                      dataset_id: selectedDataset || null,
                    });
                  }}
                >
                  <textarea
                    name="message"
                    required
                    maxLength={8000}
                    placeholder="Ask about your recorded results…"
                  />
                  <button
                    className="primary"
                    disabled={
                      busy || active.some((j) => j.operation === "chat")
                    }
                  >
                    Send ↑
                  </button>
                </form>
                <small>
                  {text(health.model)} · {text(health.provider)} · Tool calls
                  and citations are recorded.
                </small>
              </Panel>
              <Panel
                title="Document library"
                subtitle="Give the assistant sources it can cite by page."
              >
                <label className="file">
                  Import a research PDF
                  <input
                    type="file"
                    accept=".pdf"
                    onChange={(e) =>
                      void fileChange("documents/pdf", e.target.files?.[0])
                    }
                  />
                </label>
                <form
                  onSubmit={form((d) =>
                    api("documents", {
                      name: get(d, "name"),
                      text: get(d, "text"),
                    }),
                  )}
                >
                  <Field
                    name="name"
                    title="Source title"
                    value="Research notes"
                  />
                  <label className="field">
                    <span>Source text</span>
                    <textarea name="text" required />
                  </label>
                  <button disabled={busy}>Save source</button>
                </form>
                <div className="source-list">
                  {documents.map((d) => (
                    <div key={d.id}>
                      <span>▤</span>
                      <div>
                        <strong>{text(d.name)}</strong>
                        <small>Page {number(d.page)}</small>
                      </div>
                    </div>
                  ))}
                </div>
                <button
                  onClick={() => void submit("documents/reindex", {})}
                  disabled={busy || !documents.length}
                >
                  Build local embedding index
                </button>
                <small>
                  Text search works before indexing. Embeddings require the
                  optional local model dependencies.
                </small>
              </Panel>
            </div>
          )}
          {result !== null && !record(result).operation && (
            <Panel title="Recorded result">
              <Result value={result} />
            </Panel>
          )}
          {section !== "Overview" && reports.length > 0 && (
            <details className="saved-reports">
              <summary>Review saved analyses</summary>
              <div className="button-row">
                {reports.map((r) => (
                  <button key={r.id} onClick={() => setResult(r.result)}>
                    {label(text(r.name))} · {r.id.slice(0, 6)}
                  </button>
                ))}
              </div>
            </details>
          )}
          <footer>
            <span>Quantara / Research workspace</span>
            <span>Local-first · Reproducible history · Virtual execution</span>
          </footer>
        </div>
      </main>
    </div>
  );
}
