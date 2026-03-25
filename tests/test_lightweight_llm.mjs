/**
 * DCP Lightweight LLM Compatibility Tests
 *
 * Runs 5 test categories across all available ollama models to measure
 * DCP comprehension, generation, and education effectiveness.
 *
 * Requires: ollama running locally (http://localhost:11434)
 * Run: node tests/test_lightweight_llm.mjs
 */

import { writeFileSync } from "fs";

const OLLAMA_URL = "http://localhost:11434";
const MODELS = ["phi3:mini", "gemma2:2b", "qwen2.5:1.5b", "llama3.2:1b", "qwen2.5:0.5b"];
const RUNS = 3;

async function generate(model, prompt) {
  try {
    const resp = await fetch(`${OLLAMA_URL}/api/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model,
        prompt,
        stream: false,
        options: { temperature: 0, num_predict: 256 },
      }),
    });
    const data = await resp.json();
    return (data.response || "").trim();
  } catch (e) {
    return `[ERROR: ${e.message}]`;
  }
}

function extractJsonArray(text) {
  for (const line of [text.trim(), ...text.trim().split("\n")]) {
    const s = line.trim();
    if (s.startsWith("[")) {
      try { return JSON.parse(s); } catch {}
    }
  }
  const m = text.match(/\[.*?\]/s);
  if (m) { try { return JSON.parse(m[0]); } catch {} }
  return null;
}

function checkFieldOrder(resp, expected) {
  const arr = extractJsonArray(resp);
  if (!arr || arr.length !== expected.length) return false;
  return expected.every((e, i) => {
    if (typeof e === "string") return String(arr[i]).toLowerCase() === e.toLowerCase();
    if (typeof e === "number") return Number(arr[i]) === e;
    return false;
  });
}

// ── Test 1: DCP Read Comprehension ─────────────────────────

const TEST1 = [
  {
    name: "basic_field_lookup",
    prompt: `Given this DCP (Data Cost Protocol) data:
["$S","rag-chunk-meta:v1",5,"source","page","section","score","chunk_index"]
["docs/auth.md",12,"JWT Config",0.92,3]
["docs/api.md",5,"Rate Limiting",0.87,1]

What is the score of the second entry? Answer with just the number.`,
    check: (r) => r.includes("0.87"),
  },
  {
    name: "field_by_position",
    prompt: `Given this DCP data where the header defines field positions:
["$S","log-entry:v1",4,"level","service","timestamp","error_code"]
["error","auth-service",1711284600,"E_TIMEOUT"]
["warn","payment-service",1711284700,null]

What is the service name in the first data row? Answer with just the value.`,
    check: (r) => r.toLowerCase().includes("auth-service"),
  },
  {
    name: "count_and_filter",
    prompt: `Given this DCP data:
["$S","status:v1",3,"component","state","uptime_hrs"]
["api-gateway","healthy",720]
["auth-service","degraded",48]
["db-primary","healthy",2160]
["cache","healthy",168]
["worker","stopped",0]

How many components are in "healthy" state? Answer with just the number.`,
    check: (r) => r.includes("3"),
  },
];

async function runTest1(model) {
  const results = [];
  for (const t of TEST1) {
    let passes = 0;
    const responses = [];
    for (let i = 0; i < RUNS; i++) {
      const r = await generate(model, t.prompt);
      responses.push(r);
      if (t.check(r)) passes++;
    }
    results.push({
      test: t.name,
      pass_rate: `${passes}/${RUNS}`,
      passed: passes === RUNS,
      sample: responses[0].slice(0, 120),
    });
  }
  return results;
}

// ── Test 2: DCP Generation ─────────────────────────────────

const TEST2 = [
  {
    name: "simple_generation",
    prompt: `DCP uses positional arrays where field order is defined by a schema header.
Schema: ["$S","log-entry:v1",4,"level","service","timestamp","error_code"]

Convert this to a single DCP data row (JSON array, no extra text):
An error in auth-service at timestamp 1711284600 with code E_TIMEOUT

Output only the JSON array:`,
    expected: ["error", "auth-service", 1711284600, "E_TIMEOUT"],
  },
  {
    name: "from_kv_pairs",
    prompt: `Schema: ["$S","api-response:v1",4,"status","latency_ms","endpoint","method"]

Convert these key-value pairs to a DCP row (positional JSON array, field order must match schema):
method: GET, endpoint: /v1/users, status: 200, latency_ms: 42

Output only the JSON array:`,
    expected: [200, 42, "/v1/users", "GET"],
  },
];

async function runTest2(model) {
  const results = [];
  for (const t of TEST2) {
    let validJson = 0, correctOrder = 0, hasFields = 0;
    const responses = [];
    for (let i = 0; i < RUNS; i++) {
      const r = await generate(model, t.prompt);
      responses.push(r);
      const arr = extractJsonArray(r);
      if (arr) { validJson++; if (arr.length === t.expected.length) hasFields++; }
      if (checkFieldOrder(r, t.expected)) correctOrder++;
    }
    results.push({
      test: t.name,
      checks: {
        valid_json: `${validJson}/${RUNS}`,
        correct_order: `${correctOrder}/${RUNS}`,
        has_all_fields: `${hasFields}/${RUNS}`,
      },
      expected: JSON.stringify(t.expected),
      sample: responses[0].slice(0, 120),
    });
  }
  return results;
}

// ── Test 3: NL vs DCP Accuracy ─────────────────────────────

const TEST3 = [
  {
    name: "highest_score",
    nl: `Here are search results:

[Result 1]
Source: docs/auth.md
Page: 12
Section: JWT Config
Relevance Score: 0.92
Chunk Index: 3

[Result 2]
Source: docs/api.md
Page: 5
Section: Rate Limiting
Relevance Score: 0.87
Chunk Index: 1

[Result 3]
Source: docs/deploy.md
Page: 28
Section: Docker Setup
Relevance Score: 0.95
Chunk Index: 7

Which document has the highest relevance score? Answer with just the filename.`,
    dcp: `Here are search results in DCP format:
["$S","rag-chunk-meta:v1",5,"source","page","section","score","chunk_index"]
["docs/auth.md",12,"JWT Config",0.92,3]
["docs/api.md",5,"Rate Limiting",0.87,1]
["docs/deploy.md",28,"Docker Setup",0.95,7]

Which document has the highest score? Answer with just the filename.`,
    check: (r) => r.toLowerCase().includes("deploy"),
  },
  {
    name: "filter_by_value",
    nl: `Log entries:

Entry 1: Level=error, Service=auth-service, Time=1711284600, Code=E_TIMEOUT
Entry 2: Level=warn, Service=payment-service, Time=1711284700, Code=null
Entry 3: Level=error, Service=db-service, Time=1711284800, Code=E_CONN
Entry 4: Level=info, Service=auth-service, Time=1711284900, Code=null

How many entries have level "error"? Answer with just the number.`,
    dcp: `Log entries in DCP format:
["$S","log-entry:v1",4,"level","service","timestamp","error_code"]
["error","auth-service",1711284600,"E_TIMEOUT"]
["warn","payment-service",1711284700,null]
["error","db-service",1711284800,"E_CONN"]
["info","auth-service",1711284900,null]

How many entries have level "error"? Answer with just the number.`,
    check: (r) => r.includes("2"),
  },
];

async function runTest3(model) {
  const results = [];
  for (const t of TEST3) {
    let nlPass = 0, dcpPass = 0;
    for (let i = 0; i < RUNS; i++) {
      if (t.check(await generate(model, t.nl))) nlPass++;
      if (t.check(await generate(model, t.dcp))) dcpPass++;
    }
    results.push({
      test: t.name,
      nl_pass: `${nlPass}/${RUNS}`,
      dcp_pass: `${dcpPass}/${RUNS}`,
      dcp_advantage: dcpPass >= nlPass,
    });
  }
  return results;
}

// ── Test 4: Density Level Understanding ────────────────────

const TEST4 = {
  abbreviated: `Data with schema reference:
$S:knowledge:v1#fcbc [expand:GET /schemas/knowledge:v1]
["add","auth","jwt migration fix",0.8]
["flag","payment","outdated gateway config",0.3]

What action is being performed in the first row and in which domain? Answer in format: action=X, domain=Y`,

  expanded: `Data with schema hint:
$S:knowledge:v1#fcbc [action(add|replace|flag|remove) domain detail confidence:0-1] [expand:GET /schemas/knowledge:v1]
["add","auth","jwt migration fix",0.8]
["flag","payment","outdated gateway config",0.3]

What action is being performed in the first row and in which domain? Answer in format: action=X, domain=Y`,

  full: `Data with full schema definition:
Schema: {"id":"knowledge:v1","fields":["action","domain","detail","confidence"],"types":{"action":{"type":"string","enum":["add","replace","flag","remove"]},"domain":{"type":"string"},"detail":{"type":"string"},"confidence":{"type":"number","min":0,"max":1}}}

["add","auth","jwt migration fix",0.8]
["flag","payment","outdated gateway config",0.3]

What action is being performed in the first row and in which domain? Answer in format: action=X, domain=Y`,
};

const test4Check = (r) => r.toLowerCase().includes("add") && r.toLowerCase().includes("auth");

async function runTest4(model) {
  const results = [];
  for (const [density, prompt] of Object.entries(TEST4)) {
    let passes = 0;
    const responses = [];
    for (let i = 0; i < RUNS; i++) {
      const r = await generate(model, prompt);
      responses.push(r);
      if (test4Check(r)) passes++;
    }
    results.push({
      density,
      pass_rate: `${passes}/${RUNS}`,
      passed: passes > 0,
      sample: responses[0].slice(0, 120),
    });
  }
  return results;
}

// ── Test 5: Passive Education ──────────────────────────────

const TEST5 = [
  {
    name: "turn1_no_schema",
    prompt: `Structure this information as a compact data array (JSON array format):
Action: add, Domain: auth, Detail: jwt migration fix, Confidence: 0.8

Output only the JSON array:`,
  },
  {
    name: "turn2_with_expanded_hint",
    prompt: `Schema hint: $S:knowledge:v1 [action(add|replace|flag|remove) domain detail confidence:0-1]

Structure this information as a DCP row matching the schema above (JSON array, positional):
Action: replace, Domain: payment, Detail: gateway timeout increased, Confidence: 0.6

Output only the JSON array:`,
    expected: ["replace", "payment", "gateway timeout increased", 0.6],
  },
  {
    name: "turn3_abbreviated_only",
    prompt: `$S:knowledge:v1#fcbc

Structure this as a DCP row for the schema above (JSON array, positional):
Action: flag, Domain: db, Detail: connection pool exhausted, Confidence: 0.9

Output only the JSON array:`,
    expected: ["flag", "db", "connection pool exhausted", 0.9],
  },
];

async function runTest5(model) {
  const results = [];
  for (const t of TEST5) {
    const responses = [];
    for (let i = 0; i < RUNS; i++) {
      responses.push(await generate(model, t.prompt));
    }
    const entry = { turn: t.name, sample: responses[0].slice(0, 120) };
    if (t.expected) {
      const vj = responses.filter((r) => extractJsonArray(r) !== null).length;
      const co = responses.filter((r) => checkFieldOrder(r, t.expected)).length;
      entry.checks = { valid_json: `${vj}/${RUNS}`, correct_order: `${co}/${RUNS}` };
    }
    results.push(entry);
  }
  return results;
}

// ── Runner ─────────────────────────────────────────────────

function printSummary(all) {
  console.log("\n" + "=".repeat(80));
  console.log("DCP LIGHTWEIGHT LLM COMPATIBILITY — SUMMARY");
  console.log("=".repeat(80));

  console.log("\n── Test 1: DCP Read Comprehension ──");
  const t1names = TEST1.map((t) => t.name);
  console.log(["Model".padEnd(20), ...t1names.map((n) => n.padEnd(25))].join(" "));
  for (const m of MODELS) {
    const cols = all[m].test1.map((t) => t.pass_rate.padEnd(25));
    console.log([m.padEnd(20), ...cols].join(" "));
  }

  console.log("\n── Test 2: DCP Generation ──");
  for (const m of MODELS) {
    console.log(`\n  ${m}:`);
    for (const t of all[m].test2) {
      const cs = Object.entries(t.checks).map(([k, v]) => `${k}=${v}`).join(", ");
      console.log(`    ${t.test}: ${cs}`);
      console.log(`      sample: ${t.sample}`);
    }
  }

  console.log("\n── Test 3: NL vs DCP Accuracy ──");
  console.log(["Model".padEnd(20), "Test".padEnd(20), "NL".padEnd(10), "DCP".padEnd(10), "DCP≥NL"].join(""));
  for (const m of MODELS) {
    for (const t of all[m].test3) {
      const adv = t.dcp_advantage ? "✓" : "✗";
      console.log([m.padEnd(20), t.test.padEnd(20), t.nl_pass.padEnd(10), t.dcp_pass.padEnd(10), adv].join(""));
    }
  }

  console.log("\n── Test 4: Schema Density Understanding ──");
  console.log(["Model".padEnd(20), "abbreviated".padEnd(15), "expanded".padEnd(15), "full".padEnd(15)].join(""));
  for (const m of MODELS) {
    const cols = all[m].test4.map((t) => t.pass_rate.padEnd(15));
    console.log([m.padEnd(20), ...cols].join(""));
  }

  console.log("\n── Test 5: Passive Education ──");
  for (const m of MODELS) {
    console.log(`\n  ${m}:`);
    for (const t of all[m].test5) {
      if (t.checks) {
        const cs = Object.entries(t.checks).map(([k, v]) => `${k}=${v}`).join(", ");
        console.log(`    ${t.turn}: ${cs}`);
      } else {
        console.log(`    ${t.turn}: (baseline)`);
        console.log(`      sample: ${t.sample}`);
      }
    }
  }
}

async function main() {
  // Check ollama
  try {
    await fetch(`${OLLAMA_URL}/api/tags`);
  } catch {
    console.error("ERROR: ollama not available at", OLLAMA_URL);
    process.exit(1);
  }

  const all = {};

  for (const model of MODELS) {
    all[model] = {};
    console.log(`\n${"=".repeat(60)}`);
    console.log(`MODEL: ${model}`);
    console.log("=".repeat(60));

    console.log(`  Warming up...`);
    await generate(model, "Hello");

    console.log(`  Test 1: Read Comprehension...`);
    all[model].test1 = await runTest1(model);

    console.log(`  Test 2: Generation...`);
    all[model].test2 = await runTest2(model);

    console.log(`  Test 3: NL vs DCP...`);
    all[model].test3 = await runTest3(model);

    console.log(`  Test 4: Density Levels...`);
    all[model].test4 = await runTest4(model);

    console.log(`  Test 5: Passive Education...`);
    all[model].test5 = await runTest5(model);
  }

  writeFileSync(
    "reports/lightweight_llm_results.json",
    JSON.stringify(all, null, 2),
    "utf-8",
  );
  console.log("\nRaw results saved to reports/lightweight_llm_results.json");

  printSummary(all);
}

main();
