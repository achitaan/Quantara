"use client";
import { useEffect, useRef, useState } from "react";
import { number, record, rows, text } from "@/lib/chart-data";

type Point = { x: string | number; y: number };
type Series = { name: string; points: Point[] };
const colors = [
  "var(--accent)",
  "var(--muted)",
  "#b7674b",
  "#4a8e9c",
  "#ba863d",
];
const format = (v: number, unit: string) =>
  unit === "USD"
    ? new Intl.NumberFormat("en-US", {
        style: "currency",
        currency: "USD",
        maximumFractionDigits: 0,
      }).format(v)
    : unit === "%"
      ? (v * 100).toFixed(2) + "%"
      : v.toFixed(6);
const xValue = (x: string | number) =>
  typeof x === "number" ? x : Date.parse(x);
const metric = (value: unknown, key: string, unit = "%") => {
  const v = record(value)[key];
  return typeof v === "number"
    ? unit === "%"
      ? format(v, unit)
      : v.toFixed(3)
    : "—";
};
const xLabel = (x: string | number) =>
  typeof x === "number" ? "Step " + x : x.slice(0, 16).replace("T", " ");

export function SeriesChart({
  title,
  series,
  unit = "USD",
}: {
  title: string;
  series: Series[];
  unit?: string;
}) {
  const [hidden, setHidden] = useState<string[]>([]);
  const [selected, setSelected] = useState<number | null>(null);
  const plotRef = useRef<SVGSVGElement>(null);
  const [plotWidth, setPlotWidth] = useState(760);
  const available = series.filter((s) => s.points.length > 1);
  const visible = available.filter((s) => !hidden.includes(s.name));
  const all = visible
    .flatMap((s) => s.points)
    .filter((p) => Number.isFinite(p.y) && Number.isFinite(xValue(p.x)));
  const hasPoints = all.length > 0;
  useEffect(() => {
    const plot = plotRef.current;
    if (!plot) return;
    const measure = () => {
      const width = Math.round(plot.getBoundingClientRect().width);
      if (width > 0) setPlotWidth(width);
    };
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(plot);
    return () => observer.disconnect();
  }, [hasPoints]);
  if (!all.length)
    return (
      <p className="empty">
        No recorded data is available for {title.toLowerCase()}.
      </p>
    );
  const minX = all.reduce((a, p) => Math.min(a, xValue(p.x)), Infinity),
    maxX = all.reduce((a, p) => Math.max(a, xValue(p.x)), -Infinity);
  const minY = all.reduce((a, p) => Math.min(a, p.y), Infinity),
    maxY = all.reduce((a, p) => Math.max(a, p.y), -Infinity);
  const yRange = Math.max(maxY - minY, unit === "USD" ? 1 : 0.000001);
  const compact = plotWidth < 520;
  const axisFormat = (value: number) =>
    compact && unit === "USD"
      ? new Intl.NumberFormat("en-US", {
          style: "currency",
          currency: "USD",
          notation: "compact",
          maximumFractionDigits: 1,
        }).format(value)
      : format(value, unit);
  const left = Math.max(
    64,
    ...[minY, minY + yRange / 2, minY + yRange].map(
      (v) => axisFormat(v).length * 7 + 12,
    ),
  );
  const right = plotWidth - 14;
  const height = compact ? 206 : 255;
  const top = 20;
  const bottom = height - 36;
  const tickLabel = (value: string | number) => {
    if (!compact || typeof value === "number") return xLabel(value);
    const sameDay =
      reference[0].x.toString().slice(0, 10) ===
      reference.at(-1)!.x.toString().slice(0, 10);
    return new Intl.DateTimeFormat(
      "en-US",
      sameDay
        ? { hour: "2-digit", minute: "2-digit", hour12: false, timeZone: "UTC" }
        : { month: "short", day: "2-digit", timeZone: "UTC" },
    ).format(new Date(value));
  };
  const x = (v: Point["x"]) =>
    left + ((xValue(v) - minX) / Math.max(1, maxX - minX)) * (right - left);
  const y = (v: number) => bottom - ((v - minY) / yRange) * (bottom - top);
  const reference = visible[0].points;
  const index = Math.min(
    selected ?? reference.length - 1,
    reference.length - 1,
  );
  const point = reference[index];
  return (
    <div className="chart research-chart">
      <div className="chart-title">
        <strong>{title}</strong>
        <span>{xLabel(point.x)}</span>
      </div>
      <div className="chart-legend" aria-label={title + " series"}>
        {available.map((s, i) => (
          <button
            type="button"
            key={s.name}
            aria-pressed={!hidden.includes(s.name)}
            onClick={() =>
              setHidden((old) =>
                old.includes(s.name)
                  ? old.filter((name) => name !== s.name)
                  : visible.length > 1
                    ? [...old, s.name]
                    : old,
              )
            }
          >
            <i style={{ background: colors[i % colors.length] }} />
            {s.name}
          </button>
        ))}
      </div>
      <svg
        ref={plotRef}
        viewBox={`0 0 ${plotWidth} ${height}`}
        role="img"
        aria-label={title}
        onPointerMove={(e) => {
          const bounds = e.currentTarget.getBoundingClientRect();
          const value =
            minX +
            Math.max(
              0,
              Math.min(
                1,
                (e.clientX - bounds.left - (bounds.width * left) / plotWidth) /
                  ((bounds.width * (right - left)) / plotWidth),
              ),
            ) *
              (maxX - minX);
          let nearest = 0;
          for (let i = 1; i < reference.length; i++)
            if (
              Math.abs(xValue(reference[i].x) - value) <
              Math.abs(xValue(reference[nearest].x) - value)
            )
              nearest = i;
          setSelected(nearest);
        }}
      >
        {[0, 0.5, 1].map((v) => (
          <g key={v}>
            <line
              x1={left}
              x2={right}
              y1={bottom - v * (bottom - top)}
              y2={bottom - v * (bottom - top)}
              stroke="var(--line)"
              strokeDasharray="4 6"
            />
            <text
              x={left - 9}
              y={bottom + 4 - v * (bottom - top)}
              textAnchor="end"
              fill="var(--muted)"
              fontSize="11"
            >
              {axisFormat(minY + v * yRange)}
            </text>
          </g>
        ))}
        {visible.map((s) => {
          const stride = Math.max(1, Math.ceil(s.points.length / 600));
          const points = s.points.filter(
            (_, i) => i % stride === 0 || i === s.points.length - 1,
          );
          const color = colors[available.indexOf(s) % colors.length];
          return (
            <path
              key={s.name}
              aria-label={s.name + " series"}
              d={points
                .map((p, i) => (i ? "L" : "M") + x(p.x) + "," + y(p.y))
                .join(" ")}
              fill="none"
              stroke={color}
              strokeWidth="2.5"
              strokeDasharray={s.name === "Benchmark" ? "7 5" : undefined}
            />
          );
        })}
        <line
          x1={x(point.x)}
          x2={x(point.x)}
          y1={top}
          y2={bottom}
          stroke="var(--muted)"
          strokeDasharray="2 5"
        />
        <text x={left} y={height - 10} fill="var(--muted)" fontSize="11">
          {tickLabel(reference[0].x)}
        </text>
        <text
          x={right}
          y={height - 10}
          textAnchor="end"
          fill="var(--muted)"
          fontSize="11"
        >
          {tickLabel(reference.at(-1)!.x)}
        </text>
      </svg>
      <label className="chart-inspector">
        Inspect data point
        <input
          type="range"
          min={0}
          max={reference.length - 1}
          value={index}
          onChange={(e) => setSelected(Number(e.target.value))}
        />
      </label>
      <div className="metric-strip">
        {visible.map((s) => {
          const matched = s.points.find((p) => xValue(p.x) === xValue(point.x));
          return (
            <div key={s.name}>
              <small>{s.name}</small>
              <strong>
                {matched ? format(matched.y, unit) : "No matching observation"}
              </strong>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export function EquityCharts({
  data,
  benchmark = [],
  title = "Portfolio equity",
}: {
  data: unknown;
  benchmark?: unknown;
  title?: string;
}) {
  const [mode, setMode] = useState("equity");
  const points = (value: unknown, drawdown: boolean) => {
    let peak = -Infinity;
    return rows(value).map((p) => {
      const equity = number(p.equity ?? p.balance);
      peak = Math.max(peak, equity);
      return {
        x: text(p.timestamp ?? p.date),
        y: drawdown ? (peak > 0 ? equity / peak - 1 : 0) : equity,
      };
    });
  };
  return (
    <div>
      <div className="chart-tabs">
        <button
          type="button"
          aria-pressed={mode === "equity"}
          onClick={() => setMode("equity")}
        >
          Portfolio value
        </button>
        <button
          type="button"
          aria-pressed={mode === "drawdown"}
          onClick={() => setMode("drawdown")}
        >
          Drawdown
        </button>
      </div>
      <SeriesChart
        title={mode === "equity" ? title : "Drawdown from previous peak"}
        unit={mode === "equity" ? "USD" : "%"}
        series={[
          { name: "Strategy", points: points(data, mode === "drawdown") },
          { name: "Benchmark", points: points(benchmark, mode === "drawdown") },
        ]}
      />
      {rows(benchmark).length > 0 && (
        <small>Solid line: strategy · Dashed line: benchmark</small>
      )}
    </div>
  );
}

export function TrainingCharts({
  value,
  live = false,
}: {
  value: unknown;
  live?: boolean;
}) {
  const obj = record(value),
    evaluation = record(obj.evaluation),
    curves = record(evaluation.curves);
  const [mode, setMode] = useState("reward");
  const history = rows(obj.training_history),
    episodes = rows(obj.training_episodes ?? obj.episodes);
  return (
    <div className="training-charts">
      <p>
        {live
          ? "Live training observations; these are not test results."
          : text(obj.training_metric_scope) ||
            "This checkpoint predates training telemetry. Retrain it to record learning curves."}
      </p>
      <div className="chart-tabs">
        <button
          type="button"
          aria-pressed={mode === "reward"}
          onClick={() => setMode("reward")}
        >
          Step reward
        </button>
        <button
          type="button"
          aria-pressed={mode === "episodes"}
          onClick={() => setMode("episodes")}
        >
          Episode returns
        </button>
      </div>
      <SeriesChart
        title={
          mode === "reward" ? "Training reward" : "Training episode returns"
        }
        unit={mode === "reward" ? "log return" : "%"}
        series={[
          {
            name: "Training only",
            points: (mode === "reward" ? history : episodes).map((p) => ({
              x: number(p.step),
              y: number(
                mode === "reward" ? p.mean_step_reward : p.episode_return,
              ),
            })),
          },
        ]}
      />
      {!live && (
        <>
          <h3>Held-out performance</h3>
          <p>
            {text(evaluation.scope)}. {text(evaluation.curve_scope)}
          </p>
          <SeriesChart
            title="Held-out portfolio value"
            series={Object.entries(curves).map(([name, values]) => ({
              name:
                name === "benchmark"
                  ? "Benchmark"
                  : name === "policy"
                    ? "Trained policy"
                    : name.replaceAll("_", " "),
              points: rows(values).map((p) => ({
                x: text(p.timestamp),
                y: number(p.equity),
              })),
            }))}
          />
          {obj.evaluation && (
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Strategy</th>
                    <th>Return</th>
                    <th>Max drawdown</th>
                    <th>Sharpe</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries({
                    "Trained policy": evaluation.policy,
                    ...record(evaluation.comparisons),
                    ...(evaluation.benchmark
                      ? { "Benchmark (before costs)": evaluation.benchmark }
                      : {}),
                  }).map(([name, metrics]) => (
                    <tr key={name}>
                      <td>{name.replaceAll("_", " ")}</td>
                      <td>{metric(metrics, "total_return")}</td>
                      <td>{metric(metrics, "max_drawdown")}</td>
                      <td>{metric(metrics, "sharpe", "ratio")}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </>
      )}
      {!live && (
        <details>
          <summary>Checkpoint and evaluation details</summary>
          <pre>{JSON.stringify(value, null, 2)}</pre>
        </details>
      )}
    </div>
  );
}
