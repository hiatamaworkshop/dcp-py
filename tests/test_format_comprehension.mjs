/**
 * Format Comprehension Test: NL vs JSON Object vs DCP Array
 *
 * Tests whether lightweight LLMs can read structured data at all,
 * or if DCP positional arrays are specifically the problem.
 *
 * 3 formats × 4 tasks × 3 models × 3 runs
 *
 * Requires: ollama running locally (http://localhost:11434)
 * Run: node tests/test_format_comprehension.mjs
 */

import { writeFileSync } from "fs";

const OLLAMA_URL = "http://localhost:11434";
const ALL_MODELS = ["phi3:mini", "gemma2:2b", "llama3.2:1b"];
const RUNS = 3;

// Allow single model run via CLI arg: node test_format_comprehension.mjs phi3:mini
const cliModel = process.argv[2];
const MODELS = cliModel ? ALL_MODELS.filter((m) => m.includes(cliModel)) : ALL_MODELS;

async function generate(model, prompt, timeoutMs = 120_000) {
  try {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    const resp = await fetch(`${OLLAMA_URL}/api/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model,
        prompt,
        stream: false,
        options: { temperature: 0, num_predict: 128 },
      }),
      signal: controller.signal,
    });
    clearTimeout(timer);
    const data = await resp.json();
    return (data.response || "").trim();
  } catch (e) {
    return `[ERROR: ${e.message}]`;
  }
}

// ── Test definitions: same question, 3 formats ──────────────

const TESTS = [
  {
    name: "field_lookup",
    question: "What is the score of auth-service? Answer with just the number.",
    formats: {
      nl: `Here are monitoring results:

Service: api-gateway, Status: healthy, Score: 0.95, Uptime: 720h
Service: auth-service, Status: degraded, Score: 0.42, Uptime: 48h
Service: db-primary, Status: healthy, Score: 0.88, Uptime: 2160h

What is the score of auth-service? Answer with just the number.`,

      json: `Here are monitoring results:

[
  {"service": "api-gateway", "status": "healthy", "score": 0.95, "uptime_hrs": 720},
  {"service": "auth-service", "status": "degraded", "score": 0.42, "uptime_hrs": 48},
  {"service": "db-primary", "status": "healthy", "score": 0.88, "uptime_hrs": 2160}
]

What is the score of auth-service? Answer with just the number.`,

      dcp: `Here are monitoring results:

Fields: [service, status, score, uptime_hrs]
["api-gateway","healthy",0.95,720]
["auth-service","degraded",0.42,48]
["db-primary","healthy",0.88,2160]

What is the score of auth-service? Answer with just the number.`,
    },
    check: (r) => r.includes("0.42"),
  },
  {
    name: "count_filter",
    question: "How many entries have level 'error'? Answer with just the number.",
    formats: {
      nl: `Log entries:

Entry 1: Level=error, Service=auth, Time=1711284600, Code=E_TIMEOUT
Entry 2: Level=warn, Service=payment, Time=1711284700, Code=null
Entry 3: Level=error, Service=db, Time=1711284800, Code=E_CONN
Entry 4: Level=info, Service=auth, Time=1711284900, Code=null
Entry 5: Level=error, Service=cache, Time=1711285000, Code=E_MEM

How many entries have level "error"? Answer with just the number.`,

      json: `Log entries:

[
  {"level": "error", "service": "auth", "timestamp": 1711284600, "code": "E_TIMEOUT"},
  {"level": "warn", "service": "payment", "timestamp": 1711284700, "code": null},
  {"level": "error", "service": "db", "timestamp": 1711284800, "code": "E_CONN"},
  {"level": "info", "service": "auth", "timestamp": 1711284900, "code": null},
  {"level": "error", "service": "cache", "timestamp": 1711285000, "code": "E_MEM"}
]

How many entries have level "error"? Answer with just the number.`,

      dcp: `Log entries:

Fields: [level, service, timestamp, code]
["error","auth",1711284600,"E_TIMEOUT"]
["warn","payment",1711284700,null]
["error","db",1711284800,"E_CONN"]
["info","auth",1711284900,null]
["error","cache",1711285000,"E_MEM"]

How many entries have level "error"? Answer with just the number.`,
    },
    check: (r) => r.includes("3"),
  },
  {
    name: "max_value",
    question: "Which endpoint has the highest latency? Answer with just the endpoint path.",
    formats: {
      nl: `API response times:

Endpoint: /v1/users, Method: GET, Latency: 42ms, Status: 200
Endpoint: /v1/orders, Method: POST, Latency: 187ms, Status: 201
Endpoint: /v1/auth, Method: POST, Latency: 95ms, Status: 200
Endpoint: /v1/search, Method: GET, Latency: 312ms, Status: 200

Which endpoint has the highest latency? Answer with just the endpoint path.`,

      json: `API response times:

[
  {"endpoint": "/v1/users", "method": "GET", "latency_ms": 42, "status": 200},
  {"endpoint": "/v1/orders", "method": "POST", "latency_ms": 187, "status": 201},
  {"endpoint": "/v1/auth", "method": "POST", "latency_ms": 95, "status": 200},
  {"endpoint": "/v1/search", "method": "GET", "latency_ms": 312, "status": 200}
]

Which endpoint has the highest latency? Answer with just the endpoint path.`,

      dcp: `API response times:

Fields: [endpoint, method, latency_ms, status]
["/v1/users","GET",42,200]
["/v1/orders","POST",187,201]
["/v1/auth","POST",95,200]
["/v1/search","GET",312,200]

Which endpoint has the highest latency? Answer with just the endpoint path.`,
    },
    check: (r) => r.toLowerCase().includes("/v1/search"),
  },
  {
    name: "cross_reference",
    question: "What is the status of the service with the lowest weight? Answer: service name and status.",
    formats: {
      nl: `Service registry:

Name: gateway, Status: active, Weight: 5, Region: us-east
Name: worker, Status: paused, Weight: 1, Region: eu-west
Name: scheduler, Status: active, Weight: 3, Region: us-east
Name: indexer, Status: active, Weight: 8, Region: ap-south

What is the status of the service with the lowest weight? Answer: service name and status.`,

      json: `Service registry:

[
  {"name": "gateway", "status": "active", "weight": 5, "region": "us-east"},
  {"name": "worker", "status": "paused", "weight": 1, "region": "eu-west"},
  {"name": "scheduler", "status": "active", "weight": 3, "region": "us-east"},
  {"name": "indexer", "status": "active", "weight": 8, "region": "ap-south"}
]

What is the status of the service with the lowest weight? Answer: service name and status.`,

      dcp: `Service registry:

Fields: [name, status, weight, region]
["gateway","active",5,"us-east"]
["worker","paused",1,"eu-west"]
["scheduler","active",3,"us-east"]
["indexer","active",8,"ap-south"]

What is the status of the service with the lowest weight? Answer: service name and status.`,
    },
    check: (r) => r.toLowerCase().includes("worker") && r.toLowerCase().includes("paused"),
  },
];

// ── Runner ──────────────────────────────────────────────────

async function runTests() {
  const results = {};

  for (const model of MODELS) {
    console.log(`\n${"=".repeat(60)}`);
    console.log(`MODEL: ${model}`);
    console.log("=".repeat(60));

    console.log("  Warming up (loading model, may take ~30s)...");
    await generate(model, "Hello", 180_000);
    console.log("  Model ready.");

    results[model] = [];

    for (const test of TESTS) {
      const row = { test: test.name };

      for (const fmt of ["nl", "json", "dcp"]) {
        let passes = 0;
        const samples = [];
        for (let i = 0; i < RUNS; i++) {
          process.stdout.write(`    ${test.name}/${fmt} run ${i + 1}/${RUNS}...\r`);
          const r = await generate(model, test.formats[fmt]);
          samples.push(r);
          if (test.check(r)) passes++;
        }
        row[fmt] = { pass: `${passes}/${RUNS}`, sample: samples[0].slice(0, 100) };
      }

      console.log(`  ${test.name}: NL=${row.nl.pass}  JSON=${row.json.pass}  DCP=${row.dcp.pass}`);
      results[model].push(row);
    }
  }

  return results;
}

function printSummary(results) {
  console.log(`\n${"=".repeat(70)}`);
  console.log("FORMAT COMPREHENSION COMPARISON — SUMMARY");
  console.log("=".repeat(70));

  console.log("\n" + ["Model".padEnd(18), "Test".padEnd(18), "NL".padEnd(8), "JSON".padEnd(8), "DCP".padEnd(8)].join(""));
  console.log("-".repeat(60));

  for (const model of MODELS) {
    for (const row of results[model]) {
      console.log([
        model.padEnd(18),
        row.test.padEnd(18),
        row.nl.pass.padEnd(8),
        row.json.pass.padEnd(8),
        row.dcp.pass.padEnd(8),
      ].join(""));
    }
    console.log("");
  }

  // Per-format totals
  console.log("\n── Totals (pass/total) ──");
  for (const model of MODELS) {
    const totals = { nl: 0, json: 0, dcp: 0, max: results[model].length * RUNS };
    for (const row of results[model]) {
      for (const fmt of ["nl", "json", "dcp"]) {
        totals[fmt] += parseInt(row[fmt].pass);
      }
    }
    console.log(`  ${model.padEnd(18)} NL=${totals.nl}/${totals.max}  JSON=${totals.json}/${totals.max}  DCP=${totals.dcp}/${totals.max}`);
  }
}

async function main() {
  try {
    await fetch(`${OLLAMA_URL}/api/tags`);
  } catch {
    console.error("ERROR: ollama not available at", OLLAMA_URL);
    process.exit(1);
  }

  const results = await runTests();

  writeFileSync(
    "reports/format_comprehension_results.json",
    JSON.stringify(results, null, 2),
    "utf-8",
  );
  console.log("\nRaw results saved to reports/format_comprehension_results.json");

  printSummary(results);
}

main();