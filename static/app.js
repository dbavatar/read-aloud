const urlInput = document.getElementById("urlInput");
const fetchBtn = document.getElementById("fetchBtn");
const openBtn = document.getElementById("openBtn");
const pasteBtn = document.getElementById("pasteBtn");
const textInput = document.getElementById("textInput");
const readingView = document.getElementById("readingView");
const editModeBtn = document.getElementById("editModeBtn");
const modelSelect = document.getElementById("modelSelect");
const downloadModelBtn = document.getElementById("downloadModelBtn");
const modelMeta = document.getElementById("modelMeta");
const modelWarning = document.getElementById("modelWarning");
const backendPill = document.getElementById("backendPill");
const voiceSelect = document.getElementById("voiceSelect");
const speedSlider = document.getElementById("speedSlider");
const speedValue = document.getElementById("speedValue");
const playBtn = document.getElementById("playBtn");
const pauseBtn = document.getElementById("pauseBtn");
const stopBtn = document.getElementById("stopBtn");
const fetchMeta = document.getElementById("fetchMeta");
const fetchWarning = document.getElementById("fetchWarning");
const textMeta = document.getElementById("textMeta");
const playbackStatus = document.getElementById("playbackStatus");
const chunkCounter = document.getElementById("chunkCounter");
const progressFill = document.getElementById("progressFill");
const modelStatus = document.getElementById("modelStatus");
const steadyReadingToggle = document.getElementById("steadyReadingToggle");
const chunkPipeline = document.getElementById("chunkPipeline");

let sessionId = null;
let totalChunks = 0;
let currentChunk = 0;
let currentAudio = null;
let isPaused = false;
let playbackGeneration = 0;
let playAllPromise = null;
let fetchAbortController = null;
let prefetchCache = new Map();
let chunkTexts = [];
let chunkStarts = [];
let chunkElements = [];
let chunkStates = [];
let serverSynthesizingChunk = null;
let isReadingMode = false;
let modelCatalog = [];
let statusPollTimer = null;
let chunkStatusPollTimer = null;
let playAllLock = false;
let blobDurationCache = new WeakMap();
let startOffset = 0;
let lastScrollAt = 0;

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function setStatus(message) {
  playbackStatus.textContent = message;
}

function setButtons({ playing = false, paused = false } = {}) {
  playBtn.disabled = playing && !paused;
  pauseBtn.disabled = !playing;
  stopBtn.disabled = !playing && !paused;
}

function isPlaybackActive(generation) {
  return generation === playbackGeneration;
}

function detachAudio(audio) {
  if (!audio) {
    return;
  }
  audio.ontimeupdate = null;
  audio.onended = null;
  audio.onerror = null;
  audio.pause();
  audio.removeAttribute("src");
  audio.load();
}

function abortPendingFetches() {
  if (fetchAbortController) {
    fetchAbortController.abort();
    fetchAbortController = null;
  }
}

async function cancelServerSession(activeSessionId) {
  if (!activeSessionId) {
    return;
  }
  try {
    await fetch(`/api/session/${activeSessionId}/stop`, { method: "POST" });
  } catch (_error) {
    // Ignore network errors during stop.
  }
}

function getCursorOffsetInTrimmedText() {
  const raw = textInput.value;
  const leading = raw.length - raw.trimStart().length;
  const trimmedLength = raw.trim().length;
  return Math.max(0, Math.min(trimmedLength, textInput.selectionStart - leading));
}

function findChunkIndexForOffset(offset) {
  if (!chunkStarts.length) {
    return 0;
  }
  for (let index = chunkStarts.length - 1; index >= 0; index -= 1) {
    if (offset >= chunkStarts[index]) {
      return index;
    }
  }
  return 0;
}

function updateTextMeta() {
  const count = textInput.value.trim().length;
  let hint = "Click in the text to set start position";
  if (totalChunks > 0) {
    hint = `Start: section ${currentChunk + 1} of ${totalChunks}`;
  } else if (startOffset > 0) {
    hint = `Start: character ${startOffset.toLocaleString()}`;
  }
  textMeta.textContent = `${count.toLocaleString()} characters · ${hint}`;
}

function scrollChunkIntoView(element, force = false) {
  if (!element || readingView.classList.contains("hidden")) {
    return;
  }
  const now = Date.now();
  if (!force && now - lastScrollAt < 700) {
    return;
  }
  lastScrollAt = now;

  const top =
    element.offsetTop - readingView.clientHeight / 2 + element.clientHeight / 2;
  readingView.scrollTo({
    top: Math.max(0, top),
    behavior: force ? "auto" : "smooth",
  });
}

function applyStartPosition(index) {
  currentChunk = Math.max(0, Math.min(index, totalChunks - 1));
  chunkElements.forEach((element, chunkIndex) => {
    element.classList.remove("start-marker");
    if (chunkIndex < currentChunk) {
      chunkStates[chunkIndex] = "done";
      element.className = "chunk done";
      element.innerHTML = escapeHtml(chunkTexts[chunkIndex]);
      return;
    }
    if (chunkIndex === currentChunk) {
      if (chunkStates[chunkIndex] !== "ready" && chunkStates[chunkIndex] !== "generating") {
        chunkStates[chunkIndex] = "pending";
      }
      element.className = "chunk pending start-marker";
      if (chunkStates[chunkIndex] === "generating") {
        element.classList.add("generating");
      }
      if (chunkStates[chunkIndex] === "ready") {
        element.classList.add("audio-ready");
      }
      element.innerHTML = `<span class="read-part"></span><span class="unread-part">${escapeHtml(chunkTexts[chunkIndex])}</span>`;
      return;
    }
    if (chunkStates[chunkIndex] === "playing" || chunkStates[chunkIndex] === "done") {
      chunkStates[chunkIndex] = prefetchCache.has(`${sessionId}:${chunkIndex}`) ? "ready" : "pending";
    }
    setChunkState(chunkIndex, chunkStates[chunkIndex] || "pending");
    element.innerHTML = `<span class="read-part"></span><span class="unread-part">${escapeHtml(chunkTexts[chunkIndex])}</span>`;
  });
  updateProgress();
  updateTextMeta();
  updateChunkPipeline();
}

function jumpToChunk(index) {
  if (!totalChunks) {
    startOffset = 0;
    updateTextMeta();
    return;
  }

  if (playAllPromise || currentAudio) {
    stopPlayback();
  }
  isPaused = false;
  applyStartPosition(index);
  if (chunkElements[index]) {
    scrollChunkIntoView(chunkElements[index], true);
  }
  setButtons({ playing: false, paused: false });
  setStatus(`Start set at section ${index + 1} of ${totalChunks}. Press Play to continue.`);
}

function playbackSpeed() {
  return Number(speedSlider.value);
}

function applyAudioSpeed(audio) {
  const speed = playbackSpeed();
  audio.defaultPlaybackRate = speed;
  audio.playbackRate = speed;
}

async function getBlobDuration(blob) {
  if (blobDurationCache.has(blob)) {
    return blobDurationCache.get(blob);
  }
  const buffer = await blob.slice(0).arrayBuffer();
  const ctx = new AudioContext();
  try {
    const audioBuffer = await ctx.decodeAudioData(buffer);
    blobDurationCache.set(blob, audioBuffer.duration);
    return audioBuffer.duration;
  } finally {
    await ctx.close();
  }
}

function updateSpeedLabel() {
  const speed = playbackSpeed().toFixed(1);
  speedValue.textContent = `${speed}x`;
  if (currentAudio) {
    applyAudioSpeed(currentAudio);
  }
}

function updateChunkPipeline() {
  if (!chunkPipeline || !totalChunks) {
    if (chunkPipeline) {
      chunkPipeline.classList.add("hidden");
    }
    return;
  }

  const parts = [];
  const playingIndex = chunkStates.findIndex((state) => state === "playing");
  if (playingIndex >= 0) {
    parts.push(`Playing section ${playingIndex + 1}`);
  }

  const generatingIndex = chunkStates.findIndex((state) => state === "generating");
  if (generatingIndex >= 0) {
    parts.push(`Building section ${generatingIndex + 1}`);
  } else if (serverSynthesizingChunk !== null) {
    parts.push(`Building section ${serverSynthesizingChunk + 1}`);
  }

  const readyCount = chunkStates.filter((state) => state === "ready").length;
  if (readyCount > 0) {
    parts.push(`${readyCount} ready`);
  }

  chunkPipeline.textContent = parts.join(" · ");
  chunkPipeline.classList.remove("hidden");
}

function setChunkState(index, state) {
  if (index < 0 || index >= totalChunks) {
    return;
  }
  chunkStates[index] = state;
  const element = chunkElements[index];
  if (!element || state === "playing") {
    updateChunkPipeline();
    return;
  }

  const classes = ["chunk"];
  if (state === "done") {
    classes.push("done");
  } else {
    classes.push("pending");
    if (state === "generating") {
      classes.push("generating");
    }
    if (state === "ready") {
      classes.push("audio-ready");
    }
    if (element.classList.contains("start-marker")) {
      classes.push("start-marker");
    }
  }
  element.className = classes.join(" ");
  updateChunkPipeline();
}

function applyChunkClasses(element, index, primary) {
  const state = chunkStates[index] || "pending";
  const classes = ["chunk", primary];
  if (state === "generating" && primary !== "done") {
    classes.push("generating");
  }
  if (state === "ready" && primary !== "done" && primary !== "current") {
    classes.push("audio-ready");
  }
  if (element.classList.contains("start-marker") && primary !== "done") {
    classes.push("start-marker");
  }
  element.className = classes.join(" ");
}

function startChunkStatusPoll() {
  stopChunkStatusPoll();
  chunkStatusPollTimer = window.setInterval(async () => {
    if (!sessionId) {
      return;
    }
    try {
      const response = await fetch(`/api/session/${sessionId}/status`);
      if (!response.ok) {
        return;
      }
      const data = await response.json();
      serverSynthesizingChunk = data.synthesizing_chunk;

      for (const index of data.cached_chunks || []) {
        if (chunkStates[index] === "pending" || chunkStates[index] === "generating") {
          const cacheKey = `${sessionId}:${index}`;
          if (prefetchCache.has(cacheKey) || data.cached_chunks.includes(index)) {
            setChunkState(index, "ready");
          }
        }
      }

      if (
        serverSynthesizingChunk !== null &&
        chunkStates[serverSynthesizingChunk] !== "playing" &&
        chunkStates[serverSynthesizingChunk] !== "done" &&
        chunkStates[serverSynthesizingChunk] !== "ready"
      ) {
        setChunkState(serverSynthesizingChunk, "generating");
      }
      updateChunkPipeline();
    } catch (_error) {
      // Ignore polling errors.
    }
  }, 800);
}

function stopChunkStatusPoll() {
  if (chunkStatusPollTimer) {
    window.clearInterval(chunkStatusPollTimer);
    chunkStatusPollTimer = null;
  }
  serverSynthesizingChunk = null;
}

function updateProgress() {
  if (!totalChunks) {
    progressFill.style.width = "0%";
    chunkCounter.textContent = "";
    return;
  }
  const pct = Math.min(100, (currentChunk / totalChunks) * 100);
  progressFill.style.width = `${pct}%`;
  chunkCounter.textContent = `${currentChunk} / ${totalChunks}`;
}

function getReadableText() {
  return textInput.value.trim();
}

function showReadingMode() {
  isReadingMode = true;
  textInput.classList.add("hidden");
  readingView.classList.remove("hidden");
  editModeBtn.classList.remove("hidden");
}

function showEditMode() {
  isReadingMode = false;
  textInput.classList.remove("hidden");
  readingView.classList.add("hidden");
  editModeBtn.classList.add("hidden");
  clearReadingHighlight();
}

function buildReadingView(chunks) {
  chunkTexts = chunks;
  chunkElements = [];
  chunkStates = chunks.map(() => "pending");
  readingView.innerHTML = "";

  chunks.forEach((chunk, index) => {
    const span = document.createElement("span");
    span.className = "chunk pending";
    span.dataset.index = String(index);
    span.title = "Click to start reading here";
    span.innerHTML = `<span class="read-part"></span><span class="unread-part">${escapeHtml(chunk)}</span>`;
    readingView.appendChild(span);
    chunkElements.push(span);
  });
  updateChunkPipeline();
}

function clearReadingHighlight() {
  chunkElements.forEach((element) => {
    element.className = "chunk pending";
    const text = chunkTexts[Number(element.dataset.index)] || "";
    element.innerHTML = `<span class="read-part"></span><span class="unread-part">${escapeHtml(text)}</span>`;
  });
}

function setChunkProgress(index, ratio) {
  const element = chunkElements[index];
  const text = chunkTexts[index] || "";
  if (!element || !text) {
    return;
  }

  const clampedRatio = Math.max(0, Math.min(1, ratio));

  for (let chunkIndex = 0; chunkIndex < index; chunkIndex += 1) {
    if (chunkStates[chunkIndex] === "done") {
      continue;
    }
    chunkStates[chunkIndex] = "done";
    chunkElements[chunkIndex].className = "chunk done";
    chunkElements[chunkIndex].innerHTML = escapeHtml(chunkTexts[chunkIndex]);
  }

  const readChars = Math.max(
    0,
    Math.min(text.length, Math.ceil(text.length * clampedRatio)),
  );
  const readPart = escapeHtml(text.slice(0, readChars));
  const unreadPart = escapeHtml(text.slice(readChars));
  chunkStates[index] = clampedRatio >= 1 ? "done" : "playing";
  applyChunkClasses(element, index, clampedRatio >= 1 ? "done" : "current");
  element.innerHTML = `<span class="read-part">${readPart}</span><span class="unread-part">${unreadPart}</span>`;
  updateChunkPipeline();
  if (clampedRatio <= 0.02 || clampedRatio >= 0.98) {
    scrollChunkIntoView(element, clampedRatio <= 0.02);
  }
}

function markChunkDone(index) {
  const element = chunkElements[index];
  if (!element) {
    return;
  }
  setChunkState(index, "done");
  element.innerHTML = escapeHtml(chunkTexts[index] || "");
}

function formatModelLabel(model) {
  const badge = model.backend === "mlx" ? "MLX" : model.backend.toUpperCase();
  const status =
    model.status === "ready"
      ? "✓"
      : model.status === "downloading"
        ? "…"
        : model.status === "partial"
          ? "partial"
          : "download";
  return `${badge} · ${model.name} (${status})`;
}

function selectedModel() {
  return modelCatalog.find((model) => model.id === modelSelect.value) || null;
}

function updateModelControls() {
  const model = selectedModel();
  if (!model) {
    return;
  }

  modelMeta.textContent = `${model.description} · ${model.size_label}`;
  backendPill.textContent = model.backend === "mlx" ? "MLX · Apple Silicon" : model.backend.toUpperCase();

  if (model.status === "ready") {
    downloadModelBtn.textContent = model.selected ? "Loaded" : "Use model";
    downloadModelBtn.disabled = model.selected;
  } else if (model.status === "downloading") {
    downloadModelBtn.textContent = "Downloading…";
    downloadModelBtn.disabled = true;
  } else {
    downloadModelBtn.textContent = model.status === "partial" ? "Finish download" : "Download";
    downloadModelBtn.disabled = false;
  }
}

async function loadModels() {
  const response = await fetch("/api/models");
  const data = await response.json();
  modelCatalog = data.models || [];
  const previous = modelSelect.value;

  modelSelect.innerHTML = "";
  for (const model of modelCatalog) {
    const option = document.createElement("option");
    option.value = model.id;
    option.textContent = formatModelLabel(model);
    option.selected = model.selected;
    modelSelect.appendChild(option);
  }

  if (previous) {
    modelSelect.value = previous;
  }
  updateModelControls();
}

async function loadModelStatus() {
  const response = await fetch("/api/status");
  const data = await response.json();

  if (data.ready) {
    modelStatus.innerHTML = `<strong>${data.backend_label}</strong><br>${data.model_name}<br><span class="meta-inline">Ready on ${data.device}</span>`;
    playBtn.disabled = false;
  } else if (data.loading) {
    modelStatus.innerHTML = `<strong>${data.backend_label}</strong><br>Switching model…`;
    playBtn.disabled = true;
  } else {
    modelStatus.innerHTML = `<strong>${data.backend_label}</strong><br>Loading model…`;
    playBtn.disabled = true;
  }

  backendPill.textContent = `${data.backend_label || "MLX"} · Apple Silicon`;
  return data;
}

async function selectCurrentModel() {
  const model = selectedModel();
  if (!model) {
    return;
  }
  if (model.status !== "ready" && model.status !== "partial") {
    throw new Error("Download this MLX model before using it.");
  }

  modelWarning.classList.add("hidden");
  setStatus("Switching model…");
  playBtn.disabled = true;

  const response = await fetch("/api/models/select", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model_id: model.id }),
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || "Could not switch model");
  }

  resetSession();
  await Promise.all([loadModels(), loadModelStatus()]);
  setStatus("Ready");
}

async function downloadCurrentModel() {
  const model = selectedModel();
  if (!model) {
    return;
  }
  if (model.status === "ready") {
    await selectCurrentModel();
    return;
  }

  modelWarning.classList.add("hidden");
  downloadModelBtn.disabled = true;
  downloadModelBtn.textContent = "Downloading…";

  const response = await fetch("/api/models/download", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model_id: model.id }),
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || "Download failed");
  }

  modelWarning.textContent = data.message || "Download started. This may take a few minutes.";
  modelWarning.classList.remove("hidden");
  startModelPolling();
}

function startModelPolling() {
  if (statusPollTimer) {
    return;
  }
  statusPollTimer = window.setInterval(async () => {
    await loadModels();
    const status = await loadModelStatus();
    const model = selectedModel();
    if (model && (model.status === "ready" || !status.loading)) {
      window.clearInterval(statusPollTimer);
      statusPollTimer = null;
      updateModelControls();
    }
  }, 2500);
}

async function loadVoices() {
  const response = await fetch("/api/voices");
  const data = await response.json();
  voiceSelect.innerHTML = "";
  for (const voice of data.voices) {
    const option = document.createElement("option");
    option.value = voice.id;
    option.textContent = voice.label;
    if (voice.default) {
      option.selected = true;
    }
    voiceSelect.appendChild(option);
  }
}

async function fetchUrl() {
  const url = urlInput.value.trim();
  if (!url) {
    return;
  }

  fetchBtn.disabled = true;
  fetchMeta.textContent = "Fetching…";
  fetchWarning.classList.add("hidden");
  setStatus("Fetching page…");

  try {
    const response = await fetch("/api/fetch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url }),
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "Fetch failed");
    }

    fetchMeta.textContent = data.title
      ? `${data.title}${data.source_url ? ` · ${data.source_url}` : ""}`
      : data.source_url || "Fetched";

    if (data.text) {
      textInput.value = data.text;
      updateTextMeta();
      resetSession();
    }

    if (data.warning) {
      fetchWarning.textContent = data.warning;
      fetchWarning.classList.remove("hidden");
    } else {
      fetchWarning.classList.add("hidden");
    }

    setStatus(data.text ? "Ready" : "Paste text to continue");
  } catch (error) {
    fetchMeta.textContent = "Fetch failed";
    fetchWarning.textContent = error.message;
    fetchWarning.classList.remove("hidden");
    setStatus("Fetch failed");
  } finally {
    fetchBtn.disabled = false;
  }
}

async function pasteClipboard() {
  try {
    const text = await navigator.clipboard.readText();
    if (text) {
      textInput.value = text;
      updateTextMeta();
      resetSession();
      setStatus("Pasted from clipboard");
    }
  } catch (error) {
    fetchWarning.textContent = "Clipboard paste failed. Use Cmd+V in the text box.";
    fetchWarning.classList.remove("hidden");
  }
}

async function prepareSession(generation = playbackGeneration) {
  const text = getReadableText();
  if (!text) {
    throw new Error("Add some text first. If URL fetch failed, paste the article text.");
  }

  const controller = new AbortController();
  const timeoutId = window.setTimeout(() => controller.abort(), 15000);

  const response = await fetch("/api/prepare", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      text,
      speaker: voiceSelect.value,
      chunk_chars: 600,
      steady_reading: steadyReadingToggle.checked,
    }),
    signal: controller.signal,
  }).finally(() => window.clearTimeout(timeoutId));

  if (!isPlaybackActive(generation)) {
    throw new DOMException("Playback stopped", "AbortError");
  }

  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || "Could not prepare audio");
  }

  sessionId = data.session_id;
  totalChunks = data.chunk_count;
  chunkStarts = data.chunk_starts || [];
  prefetchCache.clear();
  buildReadingView(data.chunks || []);

  const startIndex =
    totalChunks > 0 ? Math.min(currentChunk, data.chunk_count - 1) : findChunkIndexForOffset(startOffset);
  currentChunk = startIndex;
  showReadingMode();
  applyStartPosition(startIndex);
  if (chunkElements[startIndex]) {
    scrollChunkIntoView(chunkElements[startIndex], true);
  }
  updateProgress();
  return data;
}

async function fetchChunkBlob(index, generation) {
  if (!isPlaybackActive(generation)) {
    throw new DOMException("Playback stopped", "AbortError");
  }

  const cacheKey = `${sessionId}:${index}`;
  if (prefetchCache.has(cacheKey)) {
    if (chunkStates[index] !== "playing" && chunkStates[index] !== "done") {
      setChunkState(index, "ready");
    }
    return prefetchCache.get(cacheKey);
  }

  if (chunkStates[index] !== "playing" && chunkStates[index] !== "done") {
    setChunkState(index, "generating");
  }

  const controller = new AbortController();
  fetchAbortController = controller;

  try {
    const response = await fetch(`/api/chunk/${sessionId}/${index}`, {
      signal: controller.signal,
    });
    if (!isPlaybackActive(generation)) {
      throw new DOMException("Playback stopped", "AbortError");
    }
    if (!response.ok) {
      const data = await response.json().catch(() => ({}));
      if (response.status === 499) {
        throw new DOMException(data.error || "Session cancelled", "AbortError");
      }
      throw new Error(data.error || `Failed to load chunk ${index + 1}`);
    }

    const blob = await response.blob();
    if (!isPlaybackActive(generation)) {
      throw new DOMException("Playback stopped", "AbortError");
    }
    prefetchCache.set(cacheKey, blob);
    if (chunkStates[index] !== "playing" && chunkStates[index] !== "done") {
      setChunkState(index, "ready");
    }
    return blob;
  } finally {
    if (fetchAbortController === controller) {
      fetchAbortController = null;
    }
  }
}

function prefetchChunk(index, generation) {
  if (index >= totalChunks || !isPlaybackActive(generation)) {
    return;
  }
  fetchChunkBlob(index, generation).catch(() => {});
}

async function playBlob(blob, chunkIndex, generation) {
  if (!isPlaybackActive(generation) || isPaused) {
    return false;
  }

  let mediaDuration = 0;
  try {
    mediaDuration = await getBlobDuration(blob);
  } catch (_error) {
    throw new Error("Could not decode audio for playback");
  }

  if (!isPlaybackActive(generation) || mediaDuration <= 0) {
    return false;
  }

  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(blob);
    const audio = new Audio();
    currentAudio = audio;
    let rafId = 0;
    let playStartedAt = 0;
    let completed = false;

    const cleanup = () => {
      cancelAnimationFrame(rafId);
      audio.ontimeupdate = null;
      URL.revokeObjectURL(url);
    };

    const getRatio = () => {
      const speed = playbackSpeed();
      const audioRatio =
        audio.currentTime > 0 ? audio.currentTime / mediaDuration : 0;
      const wallRatio =
        playStartedAt > 0
          ? (performance.now() - playStartedAt) / 1000 / (mediaDuration / speed)
          : 0;
      return Math.min(1, Math.max(audioRatio, wallRatio));
    };

    const updateHighlight = () => {
      if (!isPlaybackActive(generation) || currentAudio !== audio) {
        return;
      }
      setChunkProgress(chunkIndex, getRatio());
    };

    const tick = () => {
      updateHighlight();
      if (!audio.paused && !audio.ended) {
        rafId = requestAnimationFrame(tick);
      }
    };

    const finishChunk = (success) => {
      if (completed) {
        return;
      }
      completed = true;
      cleanup();
      if (currentAudio === audio) {
        currentAudio = null;
      }
      if (success) {
        setChunkProgress(chunkIndex, 1);
        markChunkDone(chunkIndex);
      }
      resolve(success);
    };

    audio.onended = () => {
      finishChunk(isPlaybackActive(generation));
    };

    audio.onerror = () => {
      cleanup();
      if (currentAudio === audio) {
        currentAudio = null;
      }
      reject(new Error("Audio playback failed"));
    };

    audio.ontimeupdate = () => updateHighlight();

    const beginPlay = () => {
      if (!isPlaybackActive(generation)) {
        finishChunk(false);
        return;
      }
      applyAudioSpeed(audio);
      setChunkProgress(chunkIndex, 0);
      playStartedAt = performance.now();
      audio
        .play()
        .then(() => {
          if (!isPlaybackActive(generation)) {
            audio.pause();
            finishChunk(false);
            return;
          }
          applyAudioSpeed(audio);
          playStartedAt = performance.now();
          rafId = requestAnimationFrame(tick);
        })
        .catch((error) => {
          cleanup();
          reject(error);
        });
    };

    audio.addEventListener("loadedmetadata", () => applyAudioSpeed(audio), { once: true });
    audio.src = url;
    if (audio.readyState >= HTMLMediaElement.HAVE_FUTURE_DATA) {
      beginPlay();
    } else {
      audio.addEventListener("canplay", beginPlay, { once: true });
    }
  });
}

async function playAll() {
  if (!getReadableText()) {
    fetchWarning.textContent = "Add some text first. Open the page, copy the text, and paste it here.";
    fetchWarning.classList.remove("hidden");
    setStatus("Ready");
    return;
  }

  if (playAllPromise || playAllLock) {
    if (isPaused && playAllPromise) {
      isPaused = false;
      setStatus(`Playing chunk ${currentChunk + 1} of ${totalChunks}`);
      setButtons({ playing: true, paused: false });
      if (currentAudio) {
        applyAudioSpeed(currentAudio);
        currentAudio.play().catch(() => {});
      }
    }
    return;
  }

  playAllLock = true;
  const generation = playbackGeneration;
  isPaused = false;
  setButtons({ playing: true, paused: false });
  startChunkStatusPoll();

  playAllPromise = (async () => {
    try {
      if (!sessionId) {
        setStatus("Preparing…");
        await prepareSession(generation);
        if (!isPlaybackActive(generation)) {
          return;
        }
      } else {
        if (!isReadingMode) {
          showReadingMode();
        }
        applyStartPosition(currentChunk);
      }

      for (let index = currentChunk; index < totalChunks; index += 1) {
        if (!isPlaybackActive(generation)) {
          break;
        }

        while (isPaused && isPlaybackActive(generation)) {
          await new Promise((resolve) => setTimeout(resolve, 120));
        }
        if (!isPlaybackActive(generation)) {
          break;
        }

        setStatus(`Synthesizing chunk ${index + 1} of ${totalChunks}…`);
        let blob;
        try {
          blob = await fetchChunkBlob(index, generation);
        } catch (error) {
          if (error.name === "AbortError" || !isPlaybackActive(generation)) {
            break;
          }
          throw error;
        }

        prefetchChunk(index + 1, generation);

        setStatus(`Playing chunk ${index + 1} of ${totalChunks}`);
        let finished = false;
        try {
          finished = await playBlob(blob, index, generation);
        } catch (error) {
          if (!isPlaybackActive(generation)) {
            break;
          }
          throw error;
        }

        if (!isPlaybackActive(generation)) {
          currentChunk = finished ? index + 1 : index;
          updateProgress();
          break;
        }

        if (!finished) {
          if (isPaused) {
            currentChunk = index;
            updateProgress();
            break;
          }
          throw new Error(`Playback stopped unexpectedly at section ${index + 1}`);
        }

        currentChunk = index + 1;
        updateProgress();
      }

      if (!isPlaybackActive(generation)) {
        applyStartPosition(currentChunk);
        setStatus("Stopped");
        return;
      }

      if (currentChunk >= totalChunks) {
        setStatus("Finished");
        sessionId = null;
        totalChunks = 0;
        currentChunk = 0;
        updateProgress();
        showEditMode();
      } else if (isPaused) {
        setStatus("Paused");
      }
    } catch (error) {
      if (!isPlaybackActive(generation)) {
        return;
      }
      setStatus("Playback failed");
      fetchWarning.textContent =
        error.name === "AbortError"
          ? "Preparing timed out. Try again or restart the server."
          : error.message;
      fetchWarning.classList.remove("hidden");
      showEditMode();
    } finally {
      playAllPromise = null;
      playAllLock = false;
      stopChunkStatusPoll();
      if (!isPlaybackActive(generation)) {
        return;
      }
      if (isPaused) {
        setButtons({ playing: true, paused: true });
      } else {
        setButtons({ playing: false, paused: false });
      }
      updateChunkPipeline();
    }
  })();

  await playAllPromise;
}

function pausePlayback() {
  if (!currentAudio) {
    isPaused = true;
    setStatus("Paused");
    setButtons({ playing: true, paused: true });
    return;
  }
  currentAudio.pause();
  isPaused = true;
  setStatus("Paused");
  setButtons({ playing: true, paused: true });
}

function stopPlayback() {
  const activeSessionId = sessionId;
  playbackGeneration += 1;
  playAllLock = false;
  isPaused = false;
  stopChunkStatusPoll();
  abortPendingFetches();
  if (currentAudio) {
    detachAudio(currentAudio);
    currentAudio = null;
  }
  prefetchCache.clear();
  cancelServerSession(activeSessionId);
  sessionId = null;
  if (totalChunks > 0) {
    applyStartPosition(currentChunk);
  } else {
    updateProgress();
  }
  setStatus("Stopped");
  setButtons({ playing: false, paused: false });
}

function resetSession() {
  if (playAllPromise || currentAudio) {
    stopPlayback();
  }
  sessionId = null;
  totalChunks = 0;
  currentChunk = 0;
  chunkTexts = [];
  chunkStarts = [];
  chunkElements = [];
  chunkStates = [];
  prefetchCache.clear();
  if (chunkPipeline) {
    chunkPipeline.classList.add("hidden");
  }
  updateProgress();
  updateTextMeta();
  showEditMode();
}

function rememberCursorStart() {
  startOffset = getCursorOffsetInTrimmedText();
  if (sessionId && chunkStarts.length) {
    jumpToChunk(findChunkIndexForOffset(startOffset));
    return;
  }
  updateTextMeta();
}

fetchBtn.addEventListener("click", fetchUrl);
openBtn.addEventListener("click", () => {
  const url = urlInput.value.trim();
  if (url) {
    window.open(url, "_blank", "noopener");
  }
});
pasteBtn.addEventListener("click", pasteClipboard);
editModeBtn.addEventListener("click", () => {
  stopPlayback();
  showEditMode();
});
textInput.addEventListener("input", () => {
  resetSession();
  updateTextMeta();
});
textInput.addEventListener("click", rememberCursorStart);
textInput.addEventListener("keyup", rememberCursorStart);
readingView.addEventListener("click", (event) => {
  const chunk = event.target.closest(".chunk");
  if (!chunk) {
    return;
  }
  jumpToChunk(Number(chunk.dataset.index));
});
voiceSelect.addEventListener("change", resetSession);
steadyReadingToggle.addEventListener("change", resetSession);
modelSelect.addEventListener("change", updateModelControls);
downloadModelBtn.addEventListener("click", async () => {
  try {
    const model = selectedModel();
    if (!model) {
      return;
    }
    if (model.status === "ready" && !model.selected) {
      await selectCurrentModel();
      return;
    }
    await downloadCurrentModel();
  } catch (error) {
    modelWarning.textContent = error.message;
    modelWarning.classList.remove("hidden");
    updateModelControls();
  }
});
playBtn.addEventListener("click", playAll);
pauseBtn.addEventListener("click", pausePlayback);
stopBtn.addEventListener("click", stopPlayback);
speedSlider.addEventListener("input", updateSpeedLabel);

document.querySelectorAll("[data-speed]").forEach((button) => {
  button.addEventListener("click", () => {
    speedSlider.value = button.dataset.speed;
    updateSpeedLabel();
  });
});

async function bootstrap() {
  const params = new URLSearchParams(window.location.search);
  const initialUrl = params.get("url");
  if (initialUrl) {
    urlInput.value = initialUrl;
    await fetchUrl();
  }

  await Promise.all([loadVoices(), loadModels(), loadModelStatus()]);
  updateTextMeta();
  updateSpeedLabel();
  setStatus("Ready");
  const model = selectedModel();
  if (model && model.status === "downloading") {
    startModelPolling();
  }
}

bootstrap();