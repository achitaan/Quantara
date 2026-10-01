"""Real Ollama evaluation; records latency, model residency and tool/citation validity."""

import argparse
from getpass import getpass
import os

from dotenv import load_dotenv
import json
import re
from pathlib import Path
import time
from uuid import uuid4

import httpx


def main(url, ollama_url, output):
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    credential = os.getenv("TEAM_PASSWORD") or getpass("Password: ")
    with httpx.Client(base_url=url, timeout=150) as client:
        response = client.post(
            "/api/v1/auth/login",
            json={
                "username": os.getenv("TEAM_USERNAME", "demo"),
                "password": credential,
            },
        )
        response.raise_for_status()
        health = client.get("/api/v1/health").json()
        if not health["model_ready"]:
            raise ValueError(
                "The configured local Qwen model is not ready; no benchmark was fabricated."
            )
        d = client.post("/api/v1/demo", json={}).json()
        marker = "benchmark" + uuid4().hex
        source_name = "Qwen benchmark source " + marker
        source = client.post(
            "/api/v1/documents",
            json={
                "name": source_name,
                "page": 4,
                "text": marker
                + ": This synthetic validation note sets the research cash reserve target at CAD 12,345. This is a test fixture, not a recommendation.",
            },
        )
        source.raise_for_status()
        source_id = source.json()["id"]
        cases = [
            {
                "name": "risk",
                "prompt": "Use analyze_risk to calculate VaR and expected shortfall for equal AAPL/MSFT weights using dataset "
                + d["dataset_id"],
                "tool": "analyze_risk",
            },
            {
                "name": "optimization",
                "prompt": "Use optimize_portfolio with equal_weight and holdings AAPL=0.5, MSFT=0.5 on dataset "
                + d["dataset_id"],
                "tool": "optimize_portfolio",
            },
            {
                "name": "tax",
                "prompt": "Use research_tax for portfolio "
                + d["portfolio_id"]
                + " as of 2025-02-01T00:00:00Z. Explain pooled ACB and record completeness.",
                "tool": "research_tax",
            },
            {
                "name": "document",
                "prompt": "According to "
                + source_name
                + ", what is the cash reserve target and currency? Cite the document page and identify it as a synthetic fixture.",
                "tool": None,
                "citation": "doc:" + source_id + ":p4",
            },
        ]
        evaluations = []
        for case in cases:
            start = time.perf_counter()
            job = client.post(
                "/api/v1/conversations/chat",
                json={
                    "message": case["prompt"],
                    "dataset_id": d["dataset_id"],
                    "portfolio_id": d["portfolio_id"],
                },
            ).json()
            for _ in range(1680):
                saved = client.get("/api/v1/jobs/" + job["id"]).json()
                if saved["status"] in ("complete", "failed"):
                    break
                time.sleep(0.25)
            if saved["status"] != "complete":
                raise ValueError("Evaluation did not complete")
            value = saved["result"]
            tools = value["tools"]
            citations = value["sources"]
            cited = re.findall(
                r"\[((?:doc|report|backtest):[^\]]+)\]", value["content"]
            )
            reports = {
                r["id"]: r["result"] for r in client.get("/api/v1/reports").json()
            }
            residency = httpx.get(
                ollama_url + "/api/ps", timeout=10, trust_env=False
            ).json()
            evaluations.append(
                {
                    "case": case["name"],
                    "latency_seconds": time.perf_counter() - start,
                    "expected_tool": case["tool"],
                    "selected_expected_tool": any(
                        t["name"] == case["tool"] for t in tools
                    )
                    if case["tool"]
                    else None,
                    "arguments_valid": all("error" not in t for t in tools),
                    "explanation_available": value["mode"] == "explanation",
                    "warning": value["warning"],
                    "timing": value.get("timing"),
                    "citation_validation": value.get("citation_validation"),
                    "source_citation_present": bool(cited),
                    "expected_document_cited": case["citation"] in cited
                    if case.get("citation")
                    else None,
                    "citation_identifiers_valid": not (
                        value["warning"] and "unverified citation" in value["warning"]
                    ),
                    "tools": tools,
                    "calculated_reports": {
                        t["report_id"]: reports[t["report_id"]]
                        for t in tools
                        if t.get("report_id") in reports
                    },
                    "document_sources": citations,
                    "explanation": value["content"],
                    "model_residency": residency,
                    "numerical_agreement": "Review explanation against saved tool reports; identifier validation alone is insufficient",
                }
            )
            print(
                case["name"]
                + ": "
                + value["mode"]
                + ", "
                + str(round(evaluations[-1]["latency_seconds"], 2))
                + " seconds",
                flush=True,
            )
        result = {
            "model": health["model"],
            "provider": health["provider"],
            "cases": evaluations,
            "cold_warm_note": "First call may include model loading; subsequent calls may be warm.",
            "memory_note": "Ollama residency fields describe loaded model/VRAM allocation, not total host peak RSS.",
        }
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(target.resolve())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--output", default="runtime/qwen-evaluation.json")
    args = parser.parse_args()
    main(args.url, args.ollama_url, args.output)
