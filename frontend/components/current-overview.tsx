"use client";

import { useState, type ReactNode } from "react";
import type { Portfolio, RecordData } from "@/lib/client";

export type CurrentSection =
  | "Overview"
  | "Portfolio"
  | "Market data"
  | "Strategies"
  | "Simulator"
  | "Paper trading"
  | "Signals"
  | "Cash flow"
  | "Tax research"
  | "Models"
  | "Assistant";

const tabs = [
  "Holdings",
  "Research",
  "Simulation",
  "Watchlist",
  "Activity",
] as const;

export function CurrentHero({
  portfolios,
  portfolio,
  onSelect,
}: {
  portfolios: Portfolio[];
  portfolio?: Portfolio;
  onSelect: (id: string) => void;
}) {
  const cash = portfolio
    ? new Intl.NumberFormat("en-CA", {
        style: "currency",
        currency: portfolio.currency,
        currencyDisplay: "narrowSymbol",
        minimumFractionDigits: Number.isInteger(portfolio.balance.cash) ? 0 : 2,
        maximumFractionDigits: 2,
      }).format(portfolio.balance.cash)
    : "—";
  return (
    <section className="current-hero" aria-labelledby="current-title">
      <div className="current-hero-copy">
        <p className="eyebrow">CURRENT</p>
        <h1 id="current-title">{portfolio?.name || "Your portfolio"}</h1>
        <p className="current-intro">
          A more considered approach
          <br />
          to what’s next.
        </p>
        {portfolios.length > 1 && (
          <label className="current-portfolio-selector">
            <span>Active portfolio</span>
            <select
              value={portfolio?.id || ""}
              onChange={(e) => onSelect(e.target.value)}
            >
              {portfolios.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.name} · {p.currency}
                </option>
              ))}
            </select>
          </label>
        )}
        <div className="current-cash" aria-live="polite" aria-atomic="true">
          <p>
            Recorded cash{" "}
            <span
              className="cash-info"
              title="Cash recorded in the transaction ledger. Excludes holdings value."
              aria-hidden="true"
            >
              i
            </span>
          </p>
          <strong>
            {cash}
            {portfolio && <span> {portfolio.currency}</span>}
          </strong>
          <small>
            {portfolio
              ? "Excludes holdings value"
              : "Create a portfolio to start your ledger"}
          </small>
        </div>
      </div>
      <div className="current-manifesto" aria-hidden="true">
        <p>
          DISCIPLINE
          <br />
          CLARITY
          <br />A BRIGHTER TOMORROW
        </p>
        <span>—</span>
        <blockquote>
          Same currents.
          <br />A calmer you.
        </blockquote>
      </div>
    </section>
  );
}

export function CurrentOverview({
  portfolio,
  watchlist,
  backtests,
  reports,
  datasetCount,
  strategyCount,
  simulation,
  busy,
  onNavigate,
  onPortfolioAction,
  onDemo,
}: {
  portfolio?: Portfolio;
  watchlist: string[];
  backtests: RecordData[];
  reports: RecordData[];
  datasetCount: number;
  strategyCount: number;
  simulation: ReactNode;
  busy: boolean;
  onNavigate: (section: CurrentSection) => void;
  onPortfolioAction: (action: "records" | "risk") => void;
  onDemo: () => void;
}) {
  const [tab, setTab] = useState<(typeof tabs)[number]>("Holdings");
  const [sort, setSort] = useState<{
    key: "asset" | "shares";
    direction: "ascending" | "descending";
  }>({ key: "asset", direction: "ascending" });
  const transactions = portfolio?.transactions || [];
  const holdings = Object.entries(portfolio?.balance.holdings || {}).sort(
    ([a, qa], [b, qb]) => {
      const comparison = sort.key === "asset" ? a.localeCompare(b) : qa - qb;
      return sort.direction === "ascending" ? comparison : -comparison;
    },
  );
  const sortBy = (key: "asset" | "shares") =>
    setSort((previous) => ({
      key,
      direction:
        previous.key === key && previous.direction === "ascending"
          ? "descending"
          : "ascending",
    }));
  return (
    <section className="current-overview" aria-label="Portfolio overview">
      <div className="current-section-bar">
        <div className="current-section-tabs" aria-label="Overview views">
          {tabs.map((name) => (
            <button
              type="button"
              key={name}
              aria-pressed={tab === name}
              aria-controls="current-view"
              onClick={() => setTab(name)}
            >
              {name}
            </button>
          ))}
        </div>
        <span className="current-data-note">
          Recorded ledger · Virtual execution
        </span>
      </div>
      <div
        id="current-view"
        className={tab === "Holdings" ? "current-dashboard" : "current-view"}
      >
        {tab === "Holdings" && (
          <>
            <div className="current-holdings">
              {holdings.length ? (
                <div className="table-scroll">
                  <table className="holdings-table">
                    <caption className="sr-only">
                      Recorded holdings for {portfolio?.name}
                    </caption>
                    <thead>
                      <tr>
                        <th
                          scope="col"
                          aria-sort={
                            sort.key === "asset" ? sort.direction : "none"
                          }
                        >
                          <button type="button" onClick={() => sortBy("asset")}>
                            Asset{" "}
                            <span aria-hidden="true">
                              {sort.key === "asset"
                                ? sort.direction === "ascending"
                                  ? "↑"
                                  : "↓"
                                : "↕"}
                            </span>
                          </button>
                        </th>
                        <th
                          scope="col"
                          aria-sort={
                            sort.key === "shares" ? sort.direction : "none"
                          }
                        >
                          <button
                            type="button"
                            onClick={() => sortBy("shares")}
                          >
                            Shares{" "}
                            <span aria-hidden="true">
                              {sort.key === "shares"
                                ? sort.direction === "ascending"
                                  ? "↑"
                                  : "↓"
                                : "↕"}
                            </span>
                          </button>
                        </th>
                        <th scope="col">Record</th>
                      </tr>
                    </thead>
                    <tbody>
                      {holdings.map(([symbol, quantity]) => (
                        <tr key={symbol}>
                          <th scope="row">
                            <span
                              className="current-asset-mark"
                              aria-hidden="true"
                            >
                              {symbol.slice(0, 2)}
                            </span>
                            <span>
                              {symbol}
                              <small>Recorded holding</small>
                            </span>
                          </th>
                          <td>
                            {new Intl.NumberFormat("en-CA", {
                              maximumFractionDigits: 6,
                              minimumFractionDigits: 2,
                            }).format(quantity)}
                          </td>
                          <td>Transaction ledger</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div className="current-empty">
                  <h2>
                    {portfolio
                      ? "A clear starting point"
                      : "Make room for your portfolio"}
                  </h2>
                  <p>
                    {portfolio
                      ? "No holdings recorded yet. Import your transactions to build a view of your positions."
                      : "Create a portfolio, then import your transactions to keep your research connected."}
                  </p>
                </div>
              )}
              <div className="current-holdings-actions">
                <button
                  type="button"
                  className="text-action"
                  onClick={() => onPortfolioAction("records")}
                >
                  <span aria-hidden="true">＋</span>
                  {portfolio ? "Add / import holdings" : "Create portfolio"}
                </button>
                <button
                  type="button"
                  className="text-action"
                  onClick={() => onPortfolioAction("risk")}
                >
                  <span aria-hidden="true">⇄</span>Risk & allocation
                </button>
                <span>
                  {holdings.length}{" "}
                  {holdings.length === 1 ? "security" : "securities"}
                </span>
              </div>
              <p className="current-record-note">
                Prices, holdings value and allocation are unavailable from the
                recorded ledger. Risk analysis uses equal weights over recorded
                symbols.
              </p>
            </div>
            <aside
              className="current-simulation"
              aria-labelledby="current-simulation-title"
            >
              <div className="current-panel-title">
                <h2 id="current-simulation-title">Simulation</h2>
                <button
                  type="button"
                  className="icon-action"
                  aria-label="Open simulator"
                  onClick={() => onNavigate("Simulator")}
                >
                  ↗
                </button>
              </div>
              <p className="current-simulation-note">
                Latest recorded experiment
                <br />
                Hypothetical results · excludes live performance
              </p>
              {simulation}
            </aside>
          </>
        )}
        {tab === "Research" && (
          <div className="current-research-view">
            <h2>Keep the evidence. Follow the question.</h2>
            <p>
              {datasetCount} saved datasets · {strategyCount} strategy
              definitions
            </p>
            <div className="current-research-links">
              {(
                [
                  ["Market data", "Inspect history and source coverage"],
                  ["Strategies", "Define a repeatable rule"],
                  ["Signals", "Put new information in context"],
                  ["Models", "Train and evaluate a model"],
                  ["Assistant", "Ask about recorded evidence"],
                ] as [CurrentSection, string][]
              ).map(([name, detail]) => (
                <button
                  type="button"
                  key={name}
                  onClick={() => onNavigate(name)}
                >
                  <strong>{name}</strong>
                  <small>{detail}</small>
                  <span aria-hidden="true">↗</span>
                </button>
              ))}
            </div>
          </div>
        )}
        {tab === "Simulation" && (
          <div className="current-research-view">
            <h2>Your recorded simulations</h2>
            <p>
              {backtests.length} saved experiments. Results describe simulated
              strategy behavior.
            </p>
            {simulation}
            <button
              type="button"
              className="text-action"
              onClick={() => onNavigate("Simulator")}
            >
              Open simulator <span aria-hidden="true">↗</span>
            </button>
          </div>
        )}
        {tab === "Watchlist" && (
          <div className="current-research-view">
            <h2>Your watchlist</h2>
            {watchlist.length ? (
              <ul className="current-watchlist">
                {watchlist.map((symbol) => (
                  <li key={symbol}>{symbol}</li>
                ))}
              </ul>
            ) : (
              <p>No saved symbols yet. Add securities from Market data.</p>
            )}
            <button
              type="button"
              className="text-action"
              onClick={() => onNavigate("Market data")}
            >
              Manage watchlist <span aria-hidden="true">↗</span>
            </button>
          </div>
        )}
        {tab === "Activity" && (
          <div className="current-research-view">
            <h2>Portfolio activity</h2>
            <p>
              Recent transactions for{" "}
              {portfolio?.name || "your active portfolio"}.
            </p>
            {transactions.length ? (
              <div className="table-scroll">
                <table>
                  <caption className="sr-only">
                    Recent transactions for {portfolio?.name}
                  </caption>
                  <thead>
                    <tr>
                      {[
                        "Date",
                        "Type",
                        "Asset",
                        "Shares",
                        "Amount",
                        "Currency",
                      ].map((title) => (
                        <th key={title} scope="col">
                          {title}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {transactions
                      .slice(-8)
                      .reverse()
                      .map((transaction, index) => (
                        <tr key={transaction.external_id || index}>
                          <td>{transaction.timestamp}</td>
                          <td>{transaction.type}</td>
                          <td>{transaction.symbol || "—"}</td>
                          <td>{transaction.quantity ?? "—"}</td>
                          <td>{transaction.amount ?? "—"}</td>
                          <td>{transaction.currency}</td>
                        </tr>
                      ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <p>No transactions recorded yet.</p>
            )}
            <h2 className="current-activity-heading">Research activity</h2>
            {backtests.length || reports.length ? (
              <div className="table-scroll">
                <table>
                  <thead>
                    <tr>
                      <th scope="col">Name</th>
                      <th scope="col">Type</th>
                      <th scope="col">Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {[
                      ...backtests.slice(-3).map((b) => ({
                        id: b.id,
                        name: b.name,
                        type: "Simulation",
                      })),
                      ...reports.slice(-3).map((r) => ({
                        id: r.id,
                        name: r.name,
                        type: "Analysis",
                      })),
                    ].map((r) => (
                      <tr key={r.type + r.id}>
                        <td>
                          {typeof r.name === "string"
                            ? r.name
                            : "Recorded research"}
                        </td>
                        <td>{r.type}</td>
                        <td>Saved</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <p>
                No saved research yet. Run a simulation or analysis to record
                your first result.
              </p>
            )}
          </div>
        )}
      </div>
      <div className="current-demo-row">
        <p>Explore the workflow with synthetic research data.</p>
        <button
          type="button"
          className="text-action"
          disabled={busy}
          onClick={onDemo}
        >
          {busy ? "Working…" : "Load research demo"}
          <span aria-hidden="true">↗</span>
        </button>
      </div>
    </section>
  );
}
