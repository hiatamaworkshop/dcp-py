/**
 * Shadow Level LLM Comprehension Test
 *
 * Tests how lightweight LLMs handle different DCP header densities:
 *   L0: fields only       — ["source","page","section","score","chunk_index"]
 *   L2: full protocol     — ["$S","rag-chunk-meta:v1",5,"source","page","section","score","chunk_index"]
 *   L4: NL fallback       — Source: docs/auth.md, Page: 12, Section: JWT Config, Score: 0.92
 *
 * Same data, same questions, different presentation.
 * 3 levels × 3 tasks × 3 models × 3 runs
 *
 * Requires: ollama running locally (http://localhost:11434)
 * Run: node tests/test_shadow_level_llm.mjs [model-filter]
 */

import { writeFileSync } from "fs";

const OLLAMA_URL = "http://localhost:11434";
const ALL_MODELS = ["phi3:mini", "gemma2:2b", "llama3.2:1b"];
const RUNS = 3;

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

// ── Test definitions: same data, 3 shadow levels ────────────

const TESTS = [
  {
    name: "field_lookup",
    question: "What is the score of auth-service? Answer with just the number.",
    levels: {
      L0: `Here are monitoring results:

[service, status, score, uptime_hrs]
["api-gateway","healthy",0.95,720]
["auth-service","degraded",0.42,48]
["db-primary","healthy",0.88,2160]

What is the score of auth-service? Answer with just the number.`,

      L2: `Here are monitoring results:

["$S","service-status:v1",4,"service","status","score","uptime_hrs"]
["api-gateway","healthy",0.95,720]
["auth-service","degraded",0.42,48]
["db-primary","healthy",0.88,2160]

What is the score of auth-service? Answer with just the number.`,

      L4: `Here are monitoring results:

service: api-gateway, status: healthy, score: 0.95, uptime_hrs: 720
service: auth-service, status: degraded, score: 0.42, uptime_hrs: 48
service: db-primary, status: healthy, score: 0.88, uptime_hrs: 2160

What is the score of auth-service? Answer with just the number.`,
    },
    check: (r) => r.includes("0.42"),
  },
  {
    name: "count_filter",
    question: "How many entries have level 'error'? Answer with just the number.",
    levels: {
      L0: `Log entries:

[level, service, timestamp, code]
["error","auth",1711284600,"E_TIMEOUT"]
["warn","payment",1711284700,null]
["error","db",1711284800,"E_CONN"]
["info","auth",1711284900,null]
["error","cache",1711285000,"E_MEM"]

How many entries have level "error"? Answer with just the number.`,

      L2: `Log entries:

["$S","log-entry:v1",4,"level","service","timestamp","code"]
["error","auth",1711284600,"E_TIMEOUT"]
["warn","payment",1711284700,null]
["error","db",1711284800,"E_CONN"]
["info","auth",1711284900,null]
["error","cache",1711285000,"E_MEM"]

How many entries have level "error"? Answer with just the number.`,

      L4: `Log entries:

level: error, service: auth, timestamp: 1711284600, code: E_TIMEOUT
level: warn, service: payment, timestamp: 1711284700, code: null
level: error, service: db, timestamp: 1711284800, code: E_CONN
level: info, service: auth, timestamp: 1711284900, code: null
level: error, service: cache, timestamp: 1711285000, code: E_MEM

How many entries have level "error"? Answer with just the number.`,
    },
    check: (r) => r.includes("3"),
  },
  {
    name: "max_value",
    question: "Which endpoint has the highest latency? Answer with just the endpoint path.",
    levels: {
      L0: `API response times:

[endpoint, method, latency_ms, status]
["/v1/users","GET",42,200]
["/v1/orders","POST",187,201]
["/v1/auth","POST",95,200]
["/v1/search","GET",312,200]

Which endpoint has the highest latency? Answer with just the endpoint path.`,

      L2: `API response times:

["$S","api-response:v1",4,"endpoint","method","latency_ms","status"]
["/v1/users","GET",42,200]
["/v1/orders","POST",187,201]
["/v1/auth","POST",95,200]
["/v1/search","GET",312,200]

Which endpoint has the highest latency? Answer with just the endpoint path.`,

      L4: `API response times:

endpoint: /v1/users, method: GET, latency_ms: 42, status: 200
endpoint: /v1/orders, method: POST, latency_ms: 187, status: 201
endpoint: /v1/auth, method: POST, latency_ms: 95, status: 200
endpoint: /v1/search, method: GET, latency_ms: 312, status: 200

Which endpoint has the highest latency? Answer with just the endpoint path.`,
    },
    check: (r) => r.toLowerCase().includes("/v1/search"),
  },
];

// ── Runner ──────────────────────────────────────────────────

async function runTests() {
  const results = {};

  for (const model of MODELS) {
    console.log(`\n${"=".repeat(60)}`);
    console.log(`MODEL: ${model}`);
    console.log("=".repeat(60));

    console.log("  Warming up (loading model)...");
    await generate(model, "Hello", 180_000);
    console.log("  Model ready.");

    results[model] = [];

    for (const test of TESTS) {
      const row = { test: test.name };

      for (const level of ["L0", "L2", "L4"]) {
        let passes = 0;
        const samples = [];
        for (let i = 0; i < RUNS; i++) {
          process.stdout.write(`    ${test.name}/${level} run ${i + 1}/${RUNS}...\r`);
          const r = await generate(model, test.levels[level]);
          samples.push(r);
          if (test.check(r)) passes++;
        }
        row[level] = { pass: `${passes}/${RUNS}`, sample: samples[0].slice(0, 100) };
      }

      console.log(`  ${test.name}: L0=${row.L0.pass}  L2=${row.L2.pass}  L4=${row.L4.pass}`);
      results[model].push(row);
    }
  }

  return results;
}

function printSummary(results) {
  console.log(`\n${"=".repeat(70)}`);
  console.log("SHADOW LEVEL COMPREHENSION — SUMMARY");
  console.log("=".repeat(70));
  console.log("  L0 = fields only (no protocol)");
  console.log("  L2 = full $S header");
  console.log("  L4 = NL key-value fallback");

  console.log("\n" + ["Model".padEnd(18), "Test".padEnd(18), "L0".padEnd(8), "L2".padEnd(8), "L4".padEnd(8)].join(""));
  console.log("-".repeat(60));

  for (const model of MODELS) {
    for (const row of results[model]) {
      console.log([
        model.padEnd(18),
        row.test.padEnd(18),
        row.L0.pass.padEnd(8),
        row.L2.pass.padEnd(8),
        row.L4.pass.padEnd(8),
      ].join(""));
    }
    console.log("");
  }

  // Per-level totals
  console.log("── Totals (pass/total) ──");
  for (const model of MODELS) {
    const totals = { L0: 0, L2: 0, L4: 0, max: results[model].length * RUNS };
    for (const row of results[model]) {
      for (const lvl of ["L0", "L2", "L4"]) {
        totals[lvl] += parseInt(row[lvl].pass);
      }
    }
    console.log(`  ${model.padEnd(18)} L0=${totals.L0}/${totals.max}  L2=${totals.L2}/${totals.max}  L4=${totals.L4}/${totals.max}`);
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
    "reports/shadow_level_results.json",
    JSON.stringify(results, null, 2),
    "utf-8",
  );
  console.log("\nRaw results saved to reports/shadow_level_results.json");

  printSummary(results);
}

main();