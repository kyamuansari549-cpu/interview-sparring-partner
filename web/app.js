// Sparring Partner frontend. Plain JS on purpose — the page only does
// four things (upload, start, record, report); a framework would just watch.
// State
const state = {
  profile: null,
  questions: [],
  sessionId: null,
  qi: 0,
};

// DOM helpers
const $ = (id) => document.getElementById(id);
const views = ["view-upload", "view-ready", "view-interview", "view-report"];

function showView(id) {
  views.forEach((v) => $(v).classList.toggle("hidden", v !== id));
}

function showError(msg) {
  const banner = $("error-banner");
  banner.textContent = msg;
  banner.classList.remove("hidden");
}

function clearError() {
  const banner = $("error-banner");
  banner.textContent = "";
  banner.classList.add("hidden");
}

function esc(s) {
  return String(s == null ? "" : s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

// resume upload
$("btn-upload").addEventListener("click", async () => {
  clearError();
  const file = $("resume-file").files[0];
  if (!file) {
    $("upload-status").textContent = "Pehle ek PDF file chuno.";
    return;
  }
  $("upload-status").textContent = "Resume padh raha hoon — sawal bana raha hoon…";
  const form = new FormData();
  form.append("file", file);
  try {
    const res = await fetch("/api/resume/upload", { method: "POST", body: form });
    if (!res.ok) throw new Error("Upload failed (" + res.status + ")");
    const data = await res.json();
    state.profile = data.profile;
    state.questions = data.questions || [];
    renderReady();
    $("upload-status").textContent = "";
    showView("view-ready");
  } catch (err) {
    $("upload-status").textContent = "";
    showError("Resume upload failed: " + err.message);
  }
});

function renderReady() {
  $("ready-name").textContent = (state.profile && state.profile.name) || "Unknown";
  $("ready-initial").textContent =
    (((state.profile && state.profile.name) || "?").trim().charAt(0) || "?").toUpperCase();
  $("ready-skills").innerHTML = ((state.profile && state.profile.skills) || [])
    .map((s) => `<span class="chip">${esc(s)}</span>`).join("");
  $("ready-projects").innerHTML = ((state.profile && state.profile.projects) || [])
    .map((p) => `<span class="chip">${esc(p.title || p)}</span>`).join("");
}

// start the bout
$("btn-start").addEventListener("click", async () => {
  clearError();
  $("ready-status").textContent = "Bout shuru ho raha hai…";
  try {
    const res = await fetch("/api/session/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ profile: state.profile, questions: state.questions }),
    });
    if (!res.ok) throw new Error("Session start failed (" + res.status + ")");
    const data = await res.json();
    state.sessionId = data.session_id;
    state.qi = 0;
    $("ready-status").textContent = "";
    showView("view-interview");
    renderQuestion();
  } catch (err) {
    $("ready-status").textContent = "";
    showError("Could not start session: " + err.message);
  }
});

// play the question aloud
function currentQuestion() {
  return state.questions[state.qi];
}

function renderQuestion() {
  const q = currentQuestion();
  $("result-panel").classList.add("hidden");
  $("q-category").textContent = q.category || "general";
  $("q-difficulty").textContent = q.difficulty || "";
  $("q-count").textContent = "Question " + (state.qi + 1) + " of " + state.questions.length;
  $("q-text").textContent = q.question;
  $("q-progress-fill").style.width =
    ((state.qi + 1) / state.questions.length) * 100 + "%";
  $("interview-status").textContent = "";
  const audio = $("q-audio");
  audio.pause();
  audio.classList.add("hidden");
  audio.removeAttribute("src");
  resetRecordButton();
}

$("btn-play").addEventListener("click", () => {
  clearError();
  const q = currentQuestion();
  const audio = $("q-audio");
  audio.src = "/api/session/" + state.sessionId + "/question/" + q.id + "/audio";
  audio.classList.remove("hidden");
  audio.play().catch((err) => showError("Audio playback failed: " + err.message));
});

// recording — the heart of the whole app
let mediaRecorder = null;
let audioChunks = [];

function resetRecordButton() {
  const btn = $("btn-record");
  $("record-label").textContent = "Jawab record karo";
  btn.classList.remove("recording");
}

$("btn-record").addEventListener("click", async () => {
  clearError();
  const btn = $("btn-record");
  if (mediaRecorder && mediaRecorder.state === "recording") {
    mediaRecorder.stop(); // -> onstop submits
    return;
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    audioChunks = [];
    mediaRecorder = new MediaRecorder(stream, { mimeType: "audio/webm" });
    mediaRecorder.ondataavailable = (e) => { if (e.data.size > 0) audioChunks.push(e.data); };
    mediaRecorder.onstop = submitAnswer;
    mediaRecorder.start();
    $("record-label").textContent = "Stop karo";
    btn.classList.add("recording");
    $("interview-status").textContent = "Recording chal rahi hai — jawab bolo, phir Stop dabao.";
  } catch (err) {
    showError("Microphone access failed: " + err.message + ". Please allow mic permission in the browser.");
  }
});

async function submitAnswer() {
  resetRecordButton();
  $("interview-status").textContent = "Jawab sun liya — score lag raha hai…";
  const q = currentQuestion();
  const blob = new Blob(audioChunks, { type: "audio/webm" });
  const form = new FormData();
  form.append("qid", q.id);
  form.append("audio", blob, "answer.webm");
  try {
    const res = await fetch("/api/session/" + state.sessionId + "/answer", {
      method: "POST",
      body: form,
    });
    if (!res.ok) throw new Error("Answer submission failed (" + res.status + ")");
    const data = await res.json();
    $("interview-status").textContent = "";
    renderResult(data);
  } catch (err) {
    $("interview-status").textContent = "";
    showError("Answer submission failed: " + err.message);
  }
}

function renderResult(data) {
  const ev = data.evaluation || {};
  $("r-transcript").textContent = data.transcript || "(no transcript)";
  $("r-score").textContent = ev.score_1_10 != null ? ev.score_1_10 : "-";
  $("r-bar").style.width = (ev.score_1_10 != null ? ev.score_1_10 * 10 : 0) + "%";
  $("r-fillers").textContent = ev.filler_count != null ? ev.filler_count : "-";
  $("r-content").textContent = ev.content_feedback || "";
  $("r-structure").textContent = ev.structure_feedback || "";
  $("r-clarity").textContent = ev.clarity_feedback || "";
  $("r-improvements").innerHTML = (ev.improvements || [])
    .map((i) => `<li>${esc(i)}</li>`).join("");
  $("result-panel").classList.remove("hidden");
  $("btn-next").textContent =
    state.qi + 1 < state.questions.length ? "Agla sawal" : "Poori report dekho";
}

// next round, or the final scorecard
$("btn-next").addEventListener("click", async () => {
  clearError();
  if (state.qi + 1 < state.questions.length) {
    state.qi += 1;
    renderQuestion();
  } else {
    await loadReport();
  }
});

async function loadReport() {
  $("interview-status").textContent = "Scorecard ban raha hai…";
  try {
    const res = await fetch("/api/session/" + state.sessionId + "/report");
    if (!res.ok) throw new Error("Report fetch failed (" + res.status + ")");
    const data = await res.json();
    $("interview-status").textContent = "";
    renderReport(data);
    showView("view-report");
  } catch (err) {
    $("interview-status").textContent = "";
    showError("Could not load report: " + err.message);
  }
}

function renderReport(data) {
  $("rep-avg").textContent = data.avg_score != null ? data.avg_score : "-";
  $("rep-fillers").textContent = data.total_fillers != null ? data.total_fillers : "-";
  $("rep-per-question").innerHTML = (data.per_question || [])
    .map((p) => `<li><span>${esc(p.question)}</span><span class="s">${esc(String(p.score))}/10</span></li>`).join("");
  $("rep-top-improvements").innerHTML = (data.top_improvements || [])
    .map((i) => `<li>${esc(i)}</li>`).join("");
  $("rep-model-answers").innerHTML = (data.weakest_model_answers || [])
    .map((w) => `<div class="model-answer"><p><strong>${esc(w.question)}</strong></p><p>${esc(w.model_answer_2lines)}</p></div>`).join("");
}

// back to the upload view
$("btn-restart").addEventListener("click", () => {
  clearError();
  state.profile = null;
  state.questions = [];
  state.sessionId = null;
  state.qi = 0;
  $("resume-file").value = "";
  $("upload-status").textContent = "";
  showView("view-upload");
});

$("resume-file").addEventListener("change", () => {
  const f = $("resume-file").files[0];
  if (f) $("dz-sub").textContent = f.name;
});

// first paint
showView("view-upload");
