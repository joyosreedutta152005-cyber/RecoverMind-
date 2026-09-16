// RecoverMind recovery simulation - frontend.
//
// Everything animated here comes from one real POST /api/simulate response: two real
// EpisodeReport objects (bare agent, RecoverMind), produced by actually calling
// RecoverMind(enabled=False/True).run(injected=...) on the unmodified released package.
// This file only sequences and renders that data - it computes nothing about failure
// detection, diagnosis, or recovery itself.

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

const state = {
  scenarios: [],
  selected: null,      // fault code or null
  result: null,         // {bare, recovermind}
  playing: false,
  paused: false,
  timer: null,
  attemptCount: 0,
};

const PIPELINE_ORDER = ["task", "agent", "tool", "failure", "diagnosis", "recovery", "verify", "success"];

function setPill(kind, text) {
  const el = $("#statusPill");
  el.className = "status-pill " + kind;
  el.textContent = text;
}

function setNode(stage, cls) {
  const n = document.querySelector(`.pnode[data-stage="${stage}"]`);
  if (!n) return;
  n.classList.remove("active", "done", "bad", "skip");
  if (cls) n.classList.add(cls);
}

function resetPipeline() {
  PIPELINE_ORDER.forEach((s) => setNode(s, null));
}

function nowTime() {
  const d = new Date();
  return d.toTimeString().slice(0, 8) + "." + String(d.getMilliseconds()).padStart(3, "0");
}

function logLine(text, cls) {
  const box = $("#eventLog");
  const row = document.createElement("div");
  row.className = "ln" + (cls ? " " + cls : "");
  row.innerHTML = `<span class="ts">${nowTime()}</span>${escapeHtml(text)}`;
  box.appendChild(row);
  box.scrollTop = box.scrollHeight;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
}

function delay() {
  const speed = parseFloat($("#speedRange").value) || 1;
  return 650 / speed;
}

// ------------------------------------------------------------------ scenario loading
async function loadScenarios() {
  const res = await fetch("/api/scenarios");
  state.scenarios = await res.json();
  const grid = $("#scenarioCards");
  grid.innerHTML = "";
  state.scenarios.forEach((s, i) => {
    const card = document.createElement("div");
    card.className = "scenario-card" + (i === 0 ? " selected" : "");
    card.dataset.code = s.code === null ? "" : s.code;
    const expectedLine = s.code
      ? `<div class="sc-expected">Expected injection point: plan position ${s.expected_at} (<code>${s.expected_tool}</code>)</div>`
      : "";
    card.innerHTML = `
      <span class="sc-label">${escapeHtml(s.label)}</span>
      ${s.code ? `<span class="sc-code">${s.code} · ${escapeHtml(s.family_name || "")}</span>` : `<span class="sc-code">healthy control</span>`}
      <div class="sc-note">${escapeHtml(s.note)}</div>
      ${expectedLine}`;
    card.addEventListener("click", () => {
      $$(".scenario-card").forEach((c) => c.classList.remove("selected"));
      card.classList.add("selected");
      state.selected = s.code;
    });
    grid.appendChild(card);
  });
  state.selected = state.scenarios[0] ? state.scenarios[0].code : null;
}

// ------------------------------------------------------------------ run
async function startSimulation() {
  if (state.playing) return;
  resetVisualState();
  setPill("running", "● Running");
  $("#startBtn").disabled = true;
  $("#pauseBtn").disabled = false;
  logLine("Task initialized: " + (state.scenarios.find(s => s.code === state.selected)?.label || "Healthy control"));

  let payload;
  try {
    const res = await fetch("/api/simulate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ fault: state.selected }),
    });
    payload = await res.json();
    if (payload.error) throw new Error(payload.error);
  } catch (e) {
    logLine("ERROR calling backend: " + e.message, "bad");
    setPill("failed", "● Error");
    $("#startBtn").disabled = false;
    return;
  }
  state.result = payload;
  logLine("Real episode computed by the backend (both arms). Beginning playback…");
  playEvents(payload.recovermind, payload.bare);
}

function resetVisualState() {
  resetPipeline();
  $("#eventLog").innerHTML = "";
  ["groundTruthCard", "failureCard", "diagnosisCard", "recoveryCard", "verifyCard", "resultCard", "compareCard"]
    .forEach((id) => $("#" + id).classList.add("hidden"));
  $("#locExpected").textContent = "—";
  $("#locActual").textContent = "—";
  $("#locMatch").textContent = "—";
  $("#locMatch").className = "loc-match";
  $("#detectionLag").textContent = "—";
  $("#recoveryExecuting").classList.add("hidden");
  $("#dxCandidates").innerHTML = "";
  $("#verifyChecklist").innerHTML = "";
  $("#stTask").textContent = "—";
  $("#stState").textContent = "IDLE";
  $("#stAction").textContent = "—";
  $("#stToolStatus").textContent = "—";
  $("#stFailureType").textContent = "—";
  $("#stAttempt").textContent = "0";
  $("#stFinal").textContent = "—";
  state.attemptCount = 0;
  state.paused = false;
  $("#pauseBtn").textContent = "Pause";
}

function resetSimulation() {
  clearTimeout(state.timer);
  state.playing = false;
  state.result = null;
  resetVisualState();
  setPill("idle", "● Simulation Ready");
  $("#startBtn").disabled = false;
  $("#pauseBtn").disabled = true;
}

// ------------------------------------------------------------------ playback
function playEvents(rmReport, bareReport) {
  state.playing = true;
  $("#stTask").textContent = rmReport.task;
  setNode("task", "done");

  const queue = [];
  queue.push(() => { setNode("agent", "active"); $("#stState").textContent = "EXECUTING"; });

  let sawAlarm = false, sawDiagnosis = false, sawApply = false, sawRecovered = false, sawEscalate = false;

  rmReport.events.forEach((ev) => {
    if (ev.kind === "inject") {
      queue.push(() => {
        const scenario = state.scenarios.find((s) => s.code === state.selected);
        const expectedAt = scenario ? scenario.expected_at : null;
        const expectedTool = scenario ? scenario.expected_tool : null;
        $("#groundTruthCard").classList.remove("hidden");
        $("#locExpected").textContent = `plan position ${expectedAt} (${expectedTool})`;
        $("#locActual").textContent = `plan position ${ev.at} (${ev.tool})`;
        const matchEl = $("#locMatch");
        const matches = expectedAt === ev.at;
        matchEl.textContent = matches
          ? "✓ Matches — the fault landed exactly where env/faults.py declared it would"
          : "✗ Mismatch — actual injection point differs from the declared position";
        matchEl.className = "loc-match " + (matches ? "match" : "mismatch");
        logLine(`Fault injected into environment at plan position ${ev.at} (${ev.tool}) — ground truth: ${ev.fault} — not yet visible to the agent`, "warn");
      });
    } else if (ev.kind === "step") {
      queue.push(() => {
        setNode("tool", ev.ok ? "active" : "bad");
        $("#stAction").textContent = "Calling tool: " + ev.tool;
        $("#stToolStatus").textContent = ev.ok ? "OK" : "FAILED";
        logLine(`t=${ev.t}  ${ev.ok ? "ok " : "ERR"}  ${ev.tool}  α=${ev.alpha.toFixed(2)}` + (ev.fired.length ? `  [${ev.fired.join(",")}]` : ""), ev.ok ? null : "bad");
      });
    } else if (ev.kind === "alarm") {
      sawAlarm = true;
      queue.push(() => {
        setNode("failure", "bad");
        setNode("tool", "done");
        $("#stState").textContent = "FAILURE DETECTED";
        $("#failureCard").classList.remove("hidden");
        $("#failureMsg").textContent = `Anomaly score α=${ev.alpha.toFixed(2)} exceeded the detection threshold τ_det.`;
        const lag = rmReport.detection_lag;
        $("#detectionLag").textContent = (lag === null || lag === undefined)
          ? "—" : `${lag} step${lag === 1 ? "" : "s"} after the fault was injected (injected at position ${rmReport.injected_at}, detected at position ${rmReport.detected_at})`;
        logLine(`⚠ ALARM — α=${ev.alpha.toFixed(2)} > τ_det  (detection lag: ${lag} step${lag === 1 ? "" : "s"})`, "warn");
      });
    } else if (ev.kind === "diagnosis") {
      sawDiagnosis = true;
      queue.push(() => {
        setNode("failure", "done");
        setNode("diagnosis", "active");
        $("#stState").textContent = "DIAGNOSING";
        const sub = ev.subclass;
        $("#stFailureType").textContent = sub;
        $("#failureType").textContent = sub;
        $("#diagnosisCard").classList.remove("hidden");
        $("#dxObserved").textContent = "Anomaly signals: " + (ev.evidence.join(", ") || "—");
        $("#dxCategory").textContent = sub.split(".")[0];
        $("#dxRootCause").textContent = sub;
        $("#dxConfidence").textContent = ev.confidence.toFixed(2) + "  (posterior margin, top-2 hypotheses)";
        const box = $("#dxCandidates");
        box.innerHTML = "<div class='hint' style='margin:6px 0 2px'>Ranked hypotheses (real Diagnoser output):</div>";
        ev.candidates.forEach((c) => {
          const row = document.createElement("div");
          row.className = "cand-row";
          row.innerHTML = `<span>${c.subclass}</span><span>p=${c.p.toFixed(2)}</span>`;
          box.appendChild(row);
        });
        logLine(`DIAGNOSIS ${sub}  confidence=${ev.confidence.toFixed(2)}  probes=${ev.probes}  evidence=[${ev.evidence.join(",")}]`);
      });
    } else if (ev.kind === "abstain") {
      queue.push(() => logLine("Diagnoser abstained (margin below threshold) — issuing a safe probe and continuing", "warn"));
    } else if (ev.kind === "rank") {
      queue.push(() => logLine(`  planner ranked ${ev.op} (${ev.name}): U=${ev.utility.toFixed(3)} cost=${ev.cost} rev=${ev.reversibility}`));
    } else if (ev.kind === "apply") {
      sawApply = true;
      state.attemptCount += 1;
      queue.push(() => {
        setNode("diagnosis", "done");
        setNode("recovery", "active");
        $("#stState").textContent = "RECOVERING";
        $("#stAttempt").textContent = String(state.attemptCount);
        $("#recoveryCard").classList.remove("hidden");
        $("#recoveryStrategy").textContent = `${ev.op} — ${ev.name}`;
        $("#recoveryExecuting").classList.remove("hidden");
        logLine(`APPLY ${ev.op} (${ev.name}) …`);
      });
      queue.push(() => {
        $("#recoveryExecuting").classList.add("hidden");
        setNode("recovery", "done");
        setNode("verify", "active");
        $("#stState").textContent = "VERIFYING";
        $("#verifyCard").classList.remove("hidden");
        const list = $("#verifyChecklist");
        list.innerHTML = "";
        if (ev.verification) {
          const rows = [
            ["φ_s — repair achieved its declared effect", ev.verification.phi_s],
            ["No new anomaly introduced", ev.verification.no_new_anomaly],
            ["Trajectory demonstrably advanced", ev.verification.advanced],
          ];
          rows.forEach(([label, ok]) => {
            const li = document.createElement("li");
            li.className = ok ? "pass" : "fail";
            li.textContent = (ok ? "✓ " : "✗ ") + label;
            list.appendChild(li);
          });
        }
        const overall = document.createElement("li");
        overall.className = ev.verified ? "pass" : "fail";
        overall.textContent = (ev.verified ? "✓ " : "✗ ") + "Overall: " + (ev.verified ? "Recovery verified" : "Recovery rejected — rolled back to checkpoint");
        list.appendChild(overall);
        logLine(`APPLY ${ev.op} -> ${ev.verified ? "VERIFIED" : "rejected"} | ${ev.detail}`, ev.verified ? "good" : "bad");
      });
    } else if (ev.kind === "recovered") {
      sawRecovered = true;
      queue.push(() => {
        setNode("verify", "done");
        setNode("success", "done");
        $("#stState").textContent = "RECOVERED";
        $("#stFinal").textContent = "RECOVERED";
        logLine("== RECOVERED, resuming from verified checkpoint", "good");
      });
    } else if (ev.kind === "escalate") {
      sawEscalate = true;
      queue.push(() => {
        setNode("verify", "bad");
        setNode("success", "bad");
        $("#stState").textContent = "ESCALATED";
        $("#stFinal").textContent = "ESCALATED TO HUMAN";
        logLine("== BUDGET EXHAUSTED -> escalate to human (S9)", "bad");
      });
    }
  });

  queue.push(() => finishPlayback(rmReport, bareReport, { sawAlarm, sawDiagnosis, sawApply, sawRecovered, sawEscalate }));

  runQueue(queue);
}

function runQueue(queue) {
  let i = 0;
  function step() {
    if (state.paused) { state.timer = setTimeout(step, 150); return; }
    if (i >= queue.length) { state.playing = false; return; }
    queue[i++]();
    state.timer = setTimeout(step, delay());
  }
  step();
}

function finishPlayback(rmReport, bareReport, flags) {
  if (!flags.sawAlarm) {
    // healthy control run - no fault injected, nothing to detect
    ["failure", "diagnosis", "recovery", "verify"].forEach((s) => setNode(s, "skip"));
    setNode("success", "done");
    $("#stState").textContent = "COMPLETE";
    $("#stFinal").textContent = "TASK COMPLETED — NO FAULT INJECTED";
    logLine("No alarm raised on this healthy run — matches the paper's 0 false-alarms result.", "good");
  }

  const success = rmReport.task_success;
  setPill(success ? "success" : "failed", success ? "● Recovery Successful" : "● Recovery Failed");

  const rc = $("#resultCard");
  rc.classList.remove("hidden");
  rc.classList.toggle("success", success);
  rc.classList.toggle("failed", !success);
  $("#resultTitle").textContent = success ? "RECOVERY SUCCESSFUL ✓" : "RECOVERY FAILED ✕";
  $("#resFailureType").textContent = rmReport.injected || "none (healthy control)";
  $("#resDiagnosis").textContent = rmReport.diagnosed
    ? `${rmReport.diagnosed} (${rmReport.diagnosis_correct ? "correct" : "incorrect"})`
    : "n/a (no alarm)";
  $("#resStrategy").textContent = rmReport.attempts.length
    ? rmReport.attempts.map((a) => `${a.strategy} (${a.name})`).join(" → ")
    : "n/a";
  $("#resAttempts").textContent = String(rmReport.attempts.length);
  $("#resFinalState").textContent = success ? "Task completed successfully" : (rmReport.escalated ? "Escalated to human" : "Task not completed");
  $("#resTime").textContent = (rmReport.elapsed_seconds * 1000).toFixed(2) + " ms (scripted host, no LLM inference latency)";

  // before / after
  $("#compareCard").classList.remove("hidden");
  $("#compareBare").textContent =
    `Task → ${bareReport.injected ? "Failure (undetected)" : "Healthy run"} → ` +
    (bareReport.task_success ? "Task Succeeded" : "Task FAILED ❌") +
    `\nsteps used: ${bareReport.steps_used}/${bareReport.max_steps}`;
  $("#compareRM").textContent =
    `Task → ${rmReport.injected ? "Failure" : "Healthy run"}` +
    (rmReport.diagnosed ? ` → Diagnose (${rmReport.diagnosed})` : "") +
    (rmReport.attempts.length ? ` → Recover (${rmReport.attempts[rmReport.attempts.length - 1].strategy})` : "") +
    ` → ${rmReport.task_success ? "Success ✓" : "Failed ✕"}` +
    `\nsteps used: ${rmReport.steps_used}/${rmReport.max_steps}  ·  recovery cost: ${rmReport.recovery_cost}/${rmReport.recovery_budget}`;

  logLine(`Episode complete. task_success=${success}  steps=${rmReport.steps_used}  recovery_cost=${rmReport.recovery_cost}`, success ? "good" : "bad");
  $("#startBtn").disabled = false;
  $("#pauseBtn").disabled = true;
}

// ------------------------------------------------------------------ wiring
$("#startBtn").addEventListener("click", startSimulation);
$("#resetBtn").addEventListener("click", resetSimulation);
$("#pauseBtn").addEventListener("click", () => {
  state.paused = !state.paused;
  $("#pauseBtn").textContent = state.paused ? "Resume" : "Pause";
});
$("#archBtn").addEventListener("click", () => $("#archOverlay").classList.remove("hidden"));
$("#archClose").addEventListener("click", () => $("#archOverlay").classList.add("hidden"));
$("#archOverlay").addEventListener("click", (e) => { if (e.target.id === "archOverlay") e.currentTarget.classList.add("hidden"); });

$("#themeBtn").addEventListener("click", () => {
  const html = document.documentElement;
  const next = html.getAttribute("data-theme") === "dark" ? "light" : "dark";
  html.setAttribute("data-theme", next);
  try { localStorage.setItem("rm-theme", next); } catch (e) {}
});
(function initTheme() {
  try {
    const saved = localStorage.getItem("rm-theme");
    if (saved) document.documentElement.setAttribute("data-theme", saved);
  } catch (e) {}
})();

loadScenarios();
