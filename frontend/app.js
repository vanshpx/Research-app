/**
 * ResearchAI — Frontend Application Logic
 * Handles PDF upload, chat Q&A, and citation display.
 */

const API_BASE = "http://localhost:8000/api";

// ─── State ─────────────────────────────────────────────────────────────────
const state = {
  documents: [],          // { filename, numChunks }
  isUploading: false,
  isQuerying: false,
  currentCitations: [],
};

// ─── DOM References ─────────────────────────────────────────────────────────
const uploadZone      = document.getElementById("uploadZone");
const fileInput       = document.getElementById("fileInput");
const uploadToast     = document.getElementById("uploadToast");
const uploadProgressWrap = document.getElementById("uploadProgressWrap");
const docList         = document.getElementById("docList");
const docEmpty        = document.getElementById("docEmpty");
const docCount        = document.getElementById("docCount");
const messagesWrap    = document.getElementById("messagesWrap");
const welcomeState    = document.getElementById("welcomeState");
const questionInput   = document.getElementById("questionInput");
const sendBtn         = document.getElementById("sendBtn");
const citationsPanel  = document.getElementById("citationsPanel");
const citationsList   = document.getElementById("citationsList");
const closePanelBtn   = document.getElementById("closePanelBtn");
const appLayout       = document.querySelector(".app-layout");
const statusDot       = document.getElementById("statusDot");
const statusText      = document.getElementById("statusText");

// Pipeline stage elements
const stageChunking  = document.getElementById("stageChunking");
const stageEmbedding = document.getElementById("stageEmbedding");
const stageIndexing  = document.getElementById("stageIndexing");

// ─── Health Check ──────────────────────────────────────────────────────────
async function checkHealth() {
  try {
    statusDot.className = "status-dot loading";
    statusText.textContent = "Connecting...";
    const res = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(5000) });
    if (!res.ok) throw new Error("Server error");
    const data = await res.json();
    statusDot.className = "status-dot ready";
    statusText.textContent = `Ready · ${data.documents_indexed} chunks indexed`;
  } catch (e) {
    statusDot.className = "status-dot error";
    statusText.textContent = "Server offline";
  }
}

// ─── Upload Handling ───────────────────────────────────────────────────────
function setupUpload() {
  // Click to open file picker
  uploadZone.addEventListener("click", () => fileInput.click());
  uploadZone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); }
  });

  // Drag and drop
  uploadZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadZone.classList.add("drag-over");
  });
  uploadZone.addEventListener("dragleave", () => uploadZone.classList.remove("drag-over"));
  uploadZone.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadZone.classList.remove("drag-over");
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files[0]) handleFile(e.target.files[0]);
  });
}

async function handleFile(file) {
  if (state.isUploading) return;
  if (!file.name.toLowerCase().endsWith(".pdf")) {
    showToast("Only PDF files are supported.", "error");
    return;
  }

  state.isUploading = true;
  hideToast();
  pipelineReset();
  uploadProgressWrap.classList.remove("hidden");

  const formData = new FormData();
  formData.append("file", file);

  // Stage 1: CHUNKING — starts immediately (fast, ~0.3s real time)
  pipelineActivate("stageChunking");

  // Stage 2: CONVERTING TO VECTOR — starts after a brief delay
  // Embedding is the slowest step; we start showing it early so
  // the UI feels in sync with the server.
  const embeddingTimer = setTimeout(() => {
    pipelineComplete("stageChunking");
    pipelineActivate("stageEmbedding");
  }, 700);

  // Stage 3: INDEXING — shown after embedding animation
  const indexingTimer = setTimeout(() => {
    pipelineComplete("stageEmbedding");
    pipelineActivate("stageIndexing");
  }, 2200);

  try {
    const res = await fetch(`${API_BASE}/upload`, { method: "POST", body: formData });

    // Clear timers — response arrived, take control of stage transitions
    clearTimeout(embeddingTimer);
    clearTimeout(indexingTimer);

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Upload failed");
    }

    const data = await res.json();

    // Complete any stages that haven't finished yet, then show done state
    pipelineComplete("stageChunking");
    pipelineComplete("stageEmbedding");
    pipelineComplete("stageIndexing");

    setTimeout(() => {
      uploadProgressWrap.classList.add("hidden");
      showToast(`✓ "${data.filename}" — ${data.num_chunks} chunks indexed`, "success");
      addDocumentToList(data.filename, data.num_chunks);
      checkHealth();
      state.isUploading = false;
      fileInput.value = "";
    }, 600);

  } catch (e) {
    clearTimeout(embeddingTimer);
    clearTimeout(indexingTimer);
    uploadProgressWrap.classList.add("hidden");
    showToast(e.message || "Upload failed. Please try again.", "error");
    state.isUploading = false;
  }
}

// ─── Pipeline Stage Controller ─────────────────────────────────────────────

/** Reset all stages to idle state */
function pipelineReset() {
  [stageChunking, stageEmbedding, stageIndexing].forEach(el => {
    el.classList.remove("active", "done");
    const fill = el.querySelector(".connector-fill");
    if (fill) fill.style.width = "0%";
  });
}

/** Set a stage to active (pulsing) */
function pipelineActivate(id) {
  const el = document.getElementById(id);
  if (!el) return;
  el.classList.remove("done");
  el.classList.add("active");
}

/** Mark a stage as complete (checkmark) and animate its connector */
function pipelineComplete(id) {
  const el = document.getElementById(id);
  if (!el) return;
  el.classList.remove("active");
  el.classList.add("done");
  // Animate the connector line filling up
  const fill = el.querySelector(".connector-fill");
  if (fill) {
    fill.style.transition = "width 0.4s ease";
    fill.style.width = "100%";
  }
}

function showProgress(show) {
  if (show) {
    uploadProgressWrap.classList.remove("hidden");
  } else {
    uploadProgressWrap.classList.add("hidden");
  }
}

function showToast(msg, type) {
  uploadToast.textContent = msg;
  uploadToast.className = `upload-toast ${type}`;
  uploadToast.classList.remove("hidden");
  setTimeout(() => uploadToast.classList.add("hidden"), 5000);
}

function hideToast() {
  uploadToast.classList.add("hidden");
}

function addDocumentToList(filename, numChunks) {
  // Remove empty state
  docEmpty?.remove();

  const item = document.createElement("li");
  item.className = "doc-item";
  item.setAttribute("role", "listitem");
  item.innerHTML = `
    <div class="doc-icon" aria-hidden="true">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
        <polyline points="14 2 14 8 20 8"/>
      </svg>
    </div>
    <div class="doc-info">
      <p class="doc-name" title="${escapeHtml(filename)}">${escapeHtml(filename)}</p>
      <p class="doc-meta">PDF Document</p>
    </div>
    <span class="doc-chunks-badge">${numChunks}</span>
  `;
  docList.prepend(item);

  state.documents.push({ filename, numChunks });
  docCount.textContent = state.documents.length;
}

// ─── Chat Handling ─────────────────────────────────────────────────────────
function setupChat() {
  sendBtn.addEventListener("click", sendQuestion);
  questionInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendQuestion(); }
  });

  // Auto-resize textarea
  questionInput.addEventListener("input", () => {
    questionInput.style.height = "auto";
    questionInput.style.height = Math.min(questionInput.scrollHeight, 160) + "px";
  });

  // Example chips
  document.querySelectorAll(".chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      questionInput.value = chip.dataset.query;
      questionInput.focus();
    });
  });

  // Close citations panel
  closePanelBtn.addEventListener("click", closeCitationsPanel);
}

async function sendQuestion() {
  const question = questionInput.value.trim();
  if (!question || state.isQuerying) return;

  state.isQuerying = true;
  sendBtn.disabled = true;

  // Hide welcome state
  if (welcomeState) welcomeState.style.display = "none";

  // Add user message
  addMessage("user", question);
  questionInput.value = "";
  questionInput.style.height = "auto";

  // Add AI typing indicator
  const typingId = addTypingIndicator();

  try {
    const res = await fetch(`${API_BASE}/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, top_k: 5 }),
    });

    removeTypingIndicator(typingId);

    if (!res.ok) {
      const err = await res.json();
      addMessage("ai", `⚠️ ${err.detail || "Something went wrong. Please try again."}`, [], 0);
    } else {
      const data = await res.json();
      addMessage("ai", data.answer, data.citations, data.retrieval_steps);
    }
  } catch (e) {
    removeTypingIndicator(typingId);
    addMessage("ai", "⚠️ Could not reach the server. Make sure the backend is running.", [], 0);
  }

  state.isQuerying = false;
  sendBtn.disabled = false;
  scrollToBottom();
}

function addMessage(role, text, citations = [], retrievalSteps = 0) {
  const msgEl = document.createElement("div");
  msgEl.className = `message ${role}`;

  const avatarLabel = role === "user" ? "You" : "AI";
  const avatarContent = role === "user"
    ? `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>`
    : `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="3"/><path d="M12 1v4M12 19v4M4.22 4.22l2.83 2.83M16.95 16.95l2.83 2.83M1 12h4M19 12h4M4.22 19.78l2.83-2.83M16.95 7.05l2.83-2.83"/></svg>`;

  const timeStr = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

  const hasCitations = citations && citations.length > 0;
  const citationsBtn = hasCitations
    ? `<button class="citations-btn" data-citations='${JSON.stringify(citations)}' aria-label="View ${citations.length} source citations">
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M9 11l3 3L22 4"/><path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11"/></svg>
        Sources <span class="citations-badge">${citations.length}</span>
       </button>`
    : "";

  const stepsInfo = role === "ai" && retrievalSteps > 0
    ? `<span class="retrieval-steps" title="ReAct retrieval steps">
        <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="23 4 23 10 17 10"/><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/></svg>
        ${retrievalSteps} retrieval step${retrievalSteps !== 1 ? "s" : ""}
      </span>`
    : "";

  msgEl.innerHTML = `
    <div class="msg-avatar" aria-label="${avatarLabel}">${avatarContent}</div>
    <div class="msg-content">
      <div class="msg-bubble">${formatText(text)}</div>
      <div class="msg-meta">
        <span class="msg-time">${timeStr}</span>
        ${stepsInfo}
        ${citationsBtn}
      </div>
    </div>
  `;

  // Attach citation button handler
  const btn = msgEl.querySelector(".citations-btn");
  if (btn) {
    btn.addEventListener("click", () => {
      const cits = JSON.parse(btn.dataset.citations);
      showCitationsPanel(cits);
    });
  }

  messagesWrap.appendChild(msgEl);
  scrollToBottom();
}

function addTypingIndicator() {
  const id = `typing-${Date.now()}`;
  const el = document.createElement("div");
  el.className = "message ai";
  el.id = id;
  el.innerHTML = `
    <div class="msg-avatar" aria-label="AI">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="3"/><path d="M12 1v4M12 19v4M4.22 4.22l2.83 2.83M16.95 16.95l2.83 2.83M1 12h4M19 12h4M4.22 19.78l2.83-2.83M16.95 7.05l2.83-2.83"/></svg>
    </div>
    <div class="msg-content">
      <div class="msg-bubble">
        <div class="typing-indicator" aria-label="AI is thinking">
          <div class="typing-dot"></div>
          <div class="typing-dot"></div>
          <div class="typing-dot"></div>
        </div>
      </div>
    </div>
  `;
  messagesWrap.appendChild(el);
  scrollToBottom();
  return id;
}

function removeTypingIndicator(id) {
  document.getElementById(id)?.remove();
}

// ─── Citations Panel ───────────────────────────────────────────────────────
function showCitationsPanel(citations) {
  state.currentCitations = citations;
  citationsList.innerHTML = "";

  citations.forEach((cit, i) => {
    const card = document.createElement("div");
    card.className = "citation-card";
    card.setAttribute("role", "listitem");
    card.innerHTML = `
      <div class="citation-header">
        <div class="citation-num" aria-hidden="true">${i + 1}</div>
        <span class="citation-source" title="${escapeHtml(cit.source)}">${escapeHtml(cit.source)}</span>
        <span class="citation-page">Page ${cit.page}</span>
      </div>
      <div class="citation-snippet">${escapeHtml(cit.snippet)}</div>
    `;
    citationsList.appendChild(card);
  });

  appLayout.classList.add("citations-open");
  citationsPanel.removeAttribute("aria-hidden");
}

function closeCitationsPanel() {
  appLayout.classList.remove("citations-open");
  citationsPanel.setAttribute("aria-hidden", "true");
}

// ─── Utilities ─────────────────────────────────────────────────────────────
function scrollToBottom() {
  requestAnimationFrame(() => {
    messagesWrap.scrollTo({ top: messagesWrap.scrollHeight, behavior: "smooth" });
  });
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function formatText(text) {
  // Basic markdown-like formatting
  return escapeHtml(text)
    .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
    .replace(/\*(.*?)\*/g, "<em>$1</em>")
    .replace(/`(.*?)`/g, `<code style="font-family:var(--font-mono);background:var(--bg-base);padding:1px 5px;border-radius:4px;">$1</code>`)
    .replace(/\n/g, "<br>");
}

// ─── Init ──────────────────────────────────────────────────────────────────
function init() {
  setupUpload();
  setupChat();
  checkHealth();
  // Poll health every 30 seconds
  setInterval(checkHealth, 30000);
}

document.addEventListener("DOMContentLoaded", init);
