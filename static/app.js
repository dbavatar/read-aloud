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
const buildInfo = document.getElementById("buildInfo");
const pageLoadedAt = new Date();
const steadyReadingToggle = document.getElementById("steadyReadingToggle");
const chunkPipeline = document.getElementById("chunkPipeline");

let sessionId = null;
let totalChunks = 0;
let currentChunk = 0;
let currentAudio = null;
let activeBlobTiming = null;
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
let cancelActiveBlobPlayback = null;
let blobAudioBufferCache = new WeakMap();
let startOffset = 0;
let lastScrollAt = 0;
const SPEED_STORAGE_KEY = "readAloud.playbackSpeed";
const MIN_PLAYBACK_SPEED = 0.5;
const MAX_PLAYBACK_SPEED = 4;
let preferredPlaybackSpeed = 2;
let pendingReswitchChunk = null;
let pendingResumeRatio = 0;
let readingScrollLocked = false;
let playbackReswitchActive = false;

function wasPlaybackInProgress() {
  return (
    !!(playAllPromise || playAllLock || currentAudio) ||
    chunkStates.some((state) => state === "playing" || state === "generating") ||
    serverSynthesizingChunk !== null
  );
}

function clearReadingSurface() {
  chunkTexts = [];
  chunkElements = [];
  chunkStates = [];
  chunkStarts = [];
  readingView.innerHTML = "";
  if (chunkPipeline) {
    chunkPipeline.classList.add("hidden");
  }
}

function endReadingSession() {
  sessionId = null;
  totalChunks = 0;
  currentChunk = 0;
  prefetchCache.clear();
  clearReadingSurface();
  updateProgress();
  updateTextMeta();
  showEditMode();
}

function endPlaybackReswitch() {
  pendingReswitchChunk = null;
  playbackReswitchActive = false;
  readingScrollLocked = false;
}

function hasActiveReadingSession() {
  return !!(
    sessionId ||
    totalChunks > 0 ||
    chunkElements.length > 0 ||
    playAllPromise ||
    playAllLock ||
    currentAudio
  );
}

async function rebindSessionKeepingPosition(resumeChunk, savedTotalChunks) {
  if (savedTotalChunks <= 0) {
    return null;
  }
  const data = await bindNewSession(resumeChunk);
  const chunks = data.chunks || [];
  if (chunks.length > 0 && !chunksMatch(chunks, chunkTexts)) {
    buildReadingView(chunks);
    applyStartPosition(clampChunkIndex(resumeChunk));
  }
  clearAudioCacheStates(resumeChunk);
  return data;
}

async function reswitchPlayback({
  kind,
  applyChange,
  disablePlayBtn = false,
} = {}) {
  const resumeChunk = getResumeChunkIndex();
  pendingResumeRatio = getChunkReadRatio(resumeChunk);
  const shouldResume = wasPlaybackInProgress() && !isPaused;
  const pausedSection =
    Math.max(totalChunks, chunkElements.length, chunkTexts.length) > 0 ? resumeChunk + 1 : null;
  const scrollTop = captureReadingScroll();
  const savedTotalChunks = Math.max(totalChunks, chunkElements.length, chunkTexts.length);

  pendingReswitchChunk = resumeChunk;
  readingScrollLocked = savedTotalChunks > 0;
  playbackReswitchActive = true;

  if (disablePlayBtn) {
    playBtn.disabled = true;
  }

  try {
    setStatus(
      pausedSection !== null
        ? `Switching ${kind} (keeping position)… section ${pausedSection}`
        : `Switching ${kind}…`,
    );
    await interruptPlaybackForReswitch();
    currentChunk = clampChunkIndex(resumeChunk);
    await applyChange({ resumeChunk, savedTotalChunks, currentChunk });

    syncChunkSessionState(savedTotalChunks);
    showReadingMode();
    restoreReadingScroll(scrollTop);
    if (pendingResumeRatio > 0) {
      setChunkProgress(resumeChunk, pendingResumeRatio);
    }
    updateSidebarIndicators();

    if (savedTotalChunks > 0 && shouldResume) {
      readingScrollLocked = false;
      const resumeRatio = pendingResumeRatio;
      endPlaybackReswitch();
      pendingResumeRatio = resumeRatio;
      setStatus(`Resuming section ${currentChunk + 1} with new ${kind}…`);
      void beginPlayback({ force: true, skipPositionReset: true });
      return;
    }

    endPlaybackReswitch();
    if (savedTotalChunks > 0) {
      setStatus(`Ready — press Play to continue from section ${currentChunk + 1}`);
    } else {
      setStatus("Ready");
    }
  } catch (error) {
    currentChunk = clampChunkIndex(resumeChunk);
    syncChunkSessionState(savedTotalChunks);
    showReadingMode();
    restoreReadingScroll(scrollTop);
    updateSidebarIndicators();
    endPlaybackReswitch();
    throw error;
  } finally {
    if (disablePlayBtn) {
      playBtn.disabled = false;
    }
  }
}

function updateSidebarIndicators() {
  syncChunkSessionState();
  updateProgress();
  updateTextMeta();
  updateChunkPipeline();
}

function syncChunkSessionState(minChunks = 0) {
  const domChunks = chunkElements.length;
  if (domChunks > 0 && totalChunks === 0) {
    totalChunks = domChunks;
  }
  if (minChunks > 0) {
    totalChunks = Math.max(totalChunks, minChunks);
  }
  if (chunkPipeline && totalChunks > 0) {
    chunkPipeline.classList.remove("hidden");
  }
}

function clearAudioCacheStates(fromChunk) {
  if (!totalChunks) {
    return;
  }
  const from = clampChunkIndex(fromChunk);
  currentChunk = from;
  for (let index = from; index < totalChunks; index += 1) {
    if (chunkStates[index] === "ready" || chunkStates[index] === "generating") {
      chunkStates[index] = "pending";
      const element = chunkElements[index];
      if (element) {
        element.classList.remove("audio-ready", "generating");
      }
    }
  }
}

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function prepareText(value) {
  let text = String(value ?? "").replace(/\r\n?/g, "\n");
  text = text.replace(/```[\s\S]*?```/g, " ");
  text = text.replace(/`([^`]+)`/g, "$1");
  text = text.replace(/\[([^\]]+)\]\([^)]+\)/g, "$1");
  text = text.replace(/[*_#>]+/g, " ");
  text = text.replace(/[^\S\n]+/g, " ");
  text = text.replace(/\n{3,}/g, "\n\n");
  return text.trim();
}

function getPlaybackStartOffset() {
  const raw = textInput.value;
  const selection = textInput.selectionStart ?? 0;
  const preparedBefore = prepareText(raw.slice(0, selection));
  const preparedFull = prepareText(raw);
  return Math.min(preparedBefore.length, preparedFull.length);
}

function getOffsetForChunkIndex(chunkIndex, ratio = 0) {
  if (!chunkStarts.length || !chunkTexts.length) {
    return 0;
  }
  const index = clampChunkIndex(chunkIndex);
  const start = chunkStarts[index] ?? 0;
  const body = chunkTexts[index] || "";
  const clampedRatio = Math.max(0, Math.min(1, ratio));
  return start + Math.floor(body.length * clampedRatio);
}

function resolvePrepareStartOffset({ hadReadingView = false, startChunk = null } = {}) {
  if (!hadReadingView || !chunkTexts.length) {
    return getPlaybackStartOffset();
  }
  const index = startChunk ?? currentChunk;
  return getOffsetForChunkIndex(index, getChunkReadRatio(index));
}

function preferredChunkChars() {
  const model = selectedModel();
  // Qwen CustomVoice maps best with moderate chunks: long 600-char clips
  // amplify highlight drift at 2× and risk end truncation under steady tone.
  if (model?.family === "qwen" || model?.supports_steady_reading) {
    return 420;
  }
  return 600;
}

function buildPreparePayload({ startOffset = null } = {}) {
  return {
    text: getReadableText(),
    speaker: voiceSelect.value,
    chunk_chars: preferredChunkChars(),
    steady_reading: steadyReadingToggle.checked,
    start_offset: startOffset ?? getPlaybackStartOffset(),
  };
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
  return getPlaybackStartOffset();
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

function captureReadingScroll() {
  if (readingView.classList.contains("hidden")) {
    return 0;
  }
  return readingView.scrollTop;
}

function restoreReadingScroll(scrollTop) {
  if (readingView.classList.contains("hidden")) {
    return;
  }
  readingView.scrollTop = scrollTop;
  requestAnimationFrame(() => {
    readingView.scrollTop = scrollTop;
  });
}

function chunksMatch(left, right) {
  return left.length === right.length && left.every((chunk, index) => chunk === right[index]);
}

function scrollChunkIntoView(element, force = false) {
  if (readingScrollLocked || !element || readingView.classList.contains("hidden")) {
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

function getResumeChunkIndex() {
  const playingIndex = chunkStates.findIndex((state) => state === "playing");
  if (playingIndex >= 0) {
    return playingIndex;
  }
  const generatingIndex = chunkStates.findIndex((state) => state === "generating");
  if (generatingIndex >= 0) {
    return generatingIndex;
  }
  return currentChunk;
}

function getChunkReadRatio(index) {
  const text = chunkTexts[index] || "";
  if (!text.length) {
    return 0;
  }

  if (currentAudio && chunkStates[index] === "playing") {
    const speechStart = currentAudio._speechStart ?? 0;
    const speechEnd =
      currentAudio._speechEnd ?? currentAudio._mediaDuration ?? currentAudio.duration;
    const speechDuration = Math.max(0, speechEnd - speechStart);
    if (Number.isFinite(speechDuration) && speechDuration > 0) {
      const t = Number.isFinite(currentAudio.currentTime) ? currentAudio.currentTime : speechStart;
      return Math.max(0, Math.min(1, (t - speechStart) / speechDuration));
    }
  }

  const readLen = chunkElements[index]?.querySelector(".read-part")?.textContent?.length ?? 0;
  if (readLen <= 0) {
    return 0;
  }
  return Math.max(0, Math.min(1, readLen / text.length));
}

function getPlaybackStartRatio(chunkIndex) {
  if (chunkIndex === currentChunk && pendingResumeRatio > 0) {
    const ratio = pendingResumeRatio;
    pendingResumeRatio = 0;
    return ratio;
  }
  return getChunkReadRatio(chunkIndex);
}

function clampChunkIndex(index) {
  if (!totalChunks) {
    return 0;
  }
  return Math.max(0, Math.min(index, totalChunks - 1));
}

function restoreChunkPosition(index, { preserveScroll = false } = {}) {
  if (!totalChunks) {
    return;
  }
  const scrollTop = preserveScroll ? captureReadingScroll() : null;
  currentChunk = clampChunkIndex(index);
  showReadingMode();
  applyStartPosition(currentChunk);
  updateProgress();
  updateTextMeta();
  updateChunkPipeline();
  if (chunkPipeline) {
    chunkPipeline.classList.remove("hidden");
  }
  if (preserveScroll && scrollTop !== null) {
    restoreReadingScroll(scrollTop);
  }
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

async function jumpToChunk(index) {
  if (!totalChunks) {
    startOffset = 0;
    updateTextMeta();
    return;
  }

  if (playAllPromise || playAllLock || currentAudio) {
    await stopPlayback();
  }
  isPaused = false;
  pendingResumeRatio = 0;
  applyStartPosition(index);
  if (chunkElements[index]) {
    scrollChunkIntoView(chunkElements[index], true);
  }
  setButtons({ playing: false, paused: false });
  setStatus(`Start set at section ${index + 1} of ${totalChunks}. Press Play to continue.`);
}

function clampPlaybackSpeed(speed) {
  const value = Number(speed);
  if (!Number.isFinite(value)) {
    return preferredPlaybackSpeed;
  }
  return Math.min(MAX_PLAYBACK_SPEED, Math.max(MIN_PLAYBACK_SPEED, value));
}

function playbackSpeed() {
  return preferredPlaybackSpeed;
}

function loadStoredPlaybackSpeed() {
  const stored = Number(localStorage.getItem(SPEED_STORAGE_KEY));
  if (Number.isFinite(stored)) {
    return clampPlaybackSpeed(stored);
  }
  return clampPlaybackSpeed(speedSlider.value);
}

function persistPlaybackSpeed(speed) {
  localStorage.setItem(SPEED_STORAGE_KEY, String(speed));
}

function setPlaybackSpeed(speed) {
  preferredPlaybackSpeed = clampPlaybackSpeed(speed);
  speedSlider.value = String(preferredPlaybackSpeed);
  persistPlaybackSpeed(preferredPlaybackSpeed);
  reflectSpeedUi();
  applySpeedToCurrentAudio();
}

function reflectSpeedUi() {
  speedSlider.value = String(preferredPlaybackSpeed);
  speedValue.textContent = `${preferredPlaybackSpeed.toFixed(1)}x`;
}

function applySpeedToCurrentAudio() {
  if (!currentAudio) {
    return;
  }
  enforceAudioSpeed(currentAudio, { force: true });
  if (typeof activeBlobTiming?.reanchorWallClock === "function") {
    activeBlobTiming.reanchorWallClock();
  }
  if (typeof activeBlobTiming?.scheduleEndWatchdog === "function") {
    activeBlobTiming.scheduleEndWatchdog();
  }
}

function applyPreservesPitch(audio) {
  if (!audio) {
    return;
  }
  audio.preservesPitch = true;
  if ("webkitPreservesPitch" in audio) {
    audio.webkitPreservesPitch = true;
  }
  if ("mozPreservesPitch" in audio) {
    audio.mozPreservesPitch = true;
  }
}

function enforceAudioSpeed(audio, { force = false, afterSeek = false } = {}) {
  if (!audio) {
    return;
  }
  const speed = playbackSpeed();
  applyPreservesPitch(audio);
  if (force || audio.defaultPlaybackRate !== speed) {
    audio.defaultPlaybackRate = speed;
  }
  if (force || audio.playbackRate !== speed) {
    audio.playbackRate = speed;
  }
  if (afterSeek) {
    window.requestAnimationFrame(() => {
      applyPreservesPitch(audio);
      if (audio.defaultPlaybackRate !== speed) {
        audio.defaultPlaybackRate = speed;
      }
      if (audio.playbackRate !== speed) {
        audio.playbackRate = speed;
      }
    });
  }
}

function applyAudioSpeed(audio) {
  enforceAudioSpeed(audio, { force: true });
}

async function getBlobAudioBuffer(blob) {
  if (blobAudioBufferCache.has(blob)) {
    return blobAudioBufferCache.get(blob);
  }
  const arrayBuffer = await blob.slice(0).arrayBuffer();
  const ctx = new AudioContext();
  try {
    const audioBuffer = await ctx.decodeAudioData(arrayBuffer);
    blobAudioBufferCache.set(blob, audioBuffer);
    return audioBuffer;
  } finally {
    await ctx.close();
  }
}

/**
 * Locate the speech-active window inside a decoded buffer.
 * Qwen (and similar) WAVs often include lead-in / trail-out silence; mapping
 * highlight progress over the full file makes the voice run ahead of the cursor.
 */
function analyzeSpeechWindow(audioBuffer) {
  const duration = audioBuffer.duration;
  if (!Number.isFinite(duration) || duration <= 0) {
    return { duration: 0, speechStart: 0, speechEnd: 0 };
  }

  const channel = audioBuffer.numberOfChannels > 0 ? audioBuffer.getChannelData(0) : null;
  if (!channel || channel.length === 0) {
    return { duration, speechStart: 0, speechEnd: duration };
  }

  const sampleRate = audioBuffer.sampleRate || 24000;
  const win = Math.max(1, Math.floor(sampleRate * 0.02));
  const frameCount = Math.floor(channel.length / win);
  if (frameCount <= 0) {
    return { duration, speechStart: 0, speechEnd: duration };
  }

  let peak = 0;
  const rms = new Float32Array(frameCount);
  for (let frame = 0; frame < frameCount; frame += 1) {
    const offset = frame * win;
    let sum = 0;
    for (let i = 0; i < win; i += 1) {
      const sample = channel[offset + i];
      sum += sample * sample;
    }
    const value = Math.sqrt(sum / win);
    rms[frame] = value;
    if (value > peak) {
      peak = value;
    }
  }

  // Relative to peak, with a small absolute floor so near-silent clips still map.
  const threshold = Math.max(peak * 0.06, 0.004);
  let first = 0;
  let last = frameCount - 1;
  for (let frame = 0; frame < frameCount; frame += 1) {
    if (rms[frame] >= threshold) {
      first = frame;
      break;
    }
  }
  for (let frame = frameCount - 1; frame >= 0; frame -= 1) {
    if (rms[frame] >= threshold) {
      last = frame;
      break;
    }
  }

  // Small pad so the first/last phonemes are not clipped from the map.
  const pad = 0.04;
  let speechStart = Math.max(0, (first * win) / sampleRate - pad);
  let speechEnd = Math.min(duration, ((last + 1) * win) / sampleRate + pad);
  if (speechEnd - speechStart < duration * 0.2) {
    // Degenerate detection — fall back to the full clip.
    speechStart = 0;
    speechEnd = duration;
  }

  return { duration, speechStart, speechEnd };
}

async function getBlobTiming(blob) {
  const audioBuffer = await getBlobAudioBuffer(blob);
  return analyzeSpeechWindow(audioBuffer);
}

async function getBlobDuration(blob) {
  const timing = await getBlobTiming(blob);
  return timing.duration;
}

function onSpeedControlInput() {
  preferredPlaybackSpeed = clampPlaybackSpeed(speedSlider.value);
  persistPlaybackSpeed(preferredPlaybackSpeed);
  reflectSpeedUi();
  applySpeedToCurrentAudio();
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
  return prepareText(textInput.value);
}

function showReadingMode() {
  isReadingMode = true;
  textInput.classList.add("hidden");
  readingView.classList.remove("hidden");
  editModeBtn.classList.remove("hidden");
}

function showEditMode() {
  if (playbackReswitchActive) {
    return;
  }
  if (chunkElements.length > 0 || chunkTexts.length > 0) {
    return;
  }
  isReadingMode = false;
  textInput.classList.remove("hidden");
  readingView.classList.add("hidden");
  editModeBtn.classList.add("hidden");
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
    delete chunkElements[chunkIndex].dataset.readChars;
  }

  // Prefer word boundaries so a slightly drifted ratio does not sit mid-word.
  let readChars = Math.max(
    0,
    Math.min(text.length, Math.floor(text.length * clampedRatio)),
  );
  if (clampedRatio > 0 && clampedRatio < 1 && readChars > 0 && readChars < text.length) {
    const atBoundary =
      /[\s.,;:!?…"'”)\]]/.test(text[readChars - 1]) || /\s/.test(text[readChars]);
    if (!atBoundary) {
      const nextBreak = text.indexOf(" ", readChars);
      const prevBreak = text.lastIndexOf(" ", readChars - 1);
      if (nextBreak !== -1 && nextBreak - readChars <= 12) {
        readChars = nextBreak + 1;
      } else if (prevBreak !== -1 && readChars - prevBreak <= 12) {
        readChars = prevBreak + 1;
      }
    }
  }
  if (clampedRatio >= 0.995) {
    readChars = text.length;
  }

  // Skip DOM rewrites when the visible caret has not moved (large Qwen
  // chunks thrash innerHTML every frame otherwise, which lags the rAF loop).
  const prevChars = Number(element.dataset.readChars || -1);
  const nextState = clampedRatio >= 1 ? "done" : "playing";
  if (prevChars === readChars && chunkStates[index] === nextState) {
    if (clampedRatio > 0.05) {
      readingScrollLocked = false;
    }
    return;
  }

  const readPart = escapeHtml(text.slice(0, readChars));
  const unreadPart = escapeHtml(text.slice(readChars));
  chunkStates[index] = nextState;
  applyChunkClasses(element, index, nextState === "done" ? "done" : "current");
  element.innerHTML = `<span class="read-part">${readPart}</span><span class="unread-part">${unreadPart}</span>`;
  element.dataset.readChars = String(readChars);
  updateChunkPipeline();
  if (clampedRatio > 0.05) {
    readingScrollLocked = false;
  }
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

function formatTimestamp(value) {
  if (!value) {
    return "unknown";
  }
  const date = value instanceof Date ? value : new Date(value);
  if (Number.isNaN(date.getTime())) {
    return "unknown";
  }
  return date.toLocaleString();
}

function updateBuildInfo(build) {
  if (!buildInfo || !build) {
    return;
  }

  const commitLabel = build.git_dirty ? `${build.git_commit} (dirty)` : build.git_commit;
  const parts = [
    build.version,
    commitLabel,
    `js ${build.app_js_version || "?"}`,
    `commit ${formatTimestamp(build.git_commit_date)}`,
    `server ${formatTimestamp(build.server_started_at)}`,
    `page ${formatTimestamp(pageLoadedAt)}`,
  ];
  buildInfo.textContent = parts.join(" · ");
  buildInfo.title = build.git_describe || build.version;
}

function formatModelSize(model) {
  if (model.size_gb_on_disk != null) {
    return `${Number(model.size_gb_on_disk).toFixed(2)} GB`;
  }
  if (model.size_gb_estimate != null) {
    return `~${Number(model.size_gb_estimate).toFixed(2)} GB`;
  }
  return model.size_label || "";
}

function formatMemoryLine(data) {
  const active = data.memory_active_gb;
  const peak = data.memory_peak_gb;
  if (active == null && peak == null) {
    return "";
  }
  const parts = [];
  if (active != null) {
    parts.push(`${Number(active).toFixed(2)} GB active`);
  }
  if (peak != null && peak > 0) {
    parts.push(`${Number(peak).toFixed(2)} GB peak`);
  }
  return parts.join(" · ");
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
  const size = formatModelSize(model);
  return `${badge} · ${model.name} · ${size} (${status})`;
}

function selectedModel() {
  return modelCatalog.find((model) => model.id === modelSelect.value) || null;
}

function updatePlaybackControls() {
  const model = selectedModel();
  const supportsSteady = Boolean(model?.supports_steady_reading);
  steadyReadingToggle.disabled = !supportsSteady;
  const steadyRow = steadyReadingToggle.closest(".toggle-row");
  if (steadyRow) {
    steadyRow.classList.toggle("disabled", !supportsSteady);
  }
  if (!supportsSteady) {
    steadyReadingToggle.checked = false;
  }

  const singleVoice = model?.voice_type === "single";
  voiceSelect.disabled = singleVoice;
}

function updateModelControls() {
  const model = selectedModel();
  if (!model) {
    return;
  }

  const sizeText = formatModelSize(model);
  const sizeDetail =
    model.size_gb_on_disk != null ? `${sizeText} on disk` : `${sizeText} download`;
  modelMeta.textContent = `${model.description} · ${sizeDetail}`;
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

  updatePlaybackControls();
}

async function loadModels({ preferSelection = null } = {}) {
  const response = await fetch("/api/models");
  const data = await response.json();
  modelCatalog = data.models || [];
  const previous = preferSelection || modelSelect.value;
  const serverSelected = modelCatalog.find((model) => model.selected)?.id || "";

  modelSelect.innerHTML = "";
  for (const model of modelCatalog) {
    const option = document.createElement("option");
    option.value = model.id;
    option.textContent = formatModelLabel(model);
    option.selected = model.selected;
    modelSelect.appendChild(option);
  }

  // Keep the user's dropdown choice when browsing/downloading; after an
  // explicit switch, preferSelection pins the newly loaded model.
  if (preferSelection && modelCatalog.some((model) => model.id === preferSelection)) {
    modelSelect.value = preferSelection;
  } else if (previous && modelCatalog.some((model) => model.id === previous)) {
    modelSelect.value = previous;
  } else if (serverSelected) {
    modelSelect.value = serverSelected;
  }
  updateModelControls();
}

async function loadModelStatus() {
  const response = await fetch("/api/status");
  const data = await response.json();

  const memoryLine = formatMemoryLine(data);
  const sizeLine = data.size_gb_on_disk != null
    ? `${Number(data.size_gb_on_disk).toFixed(2)} GB loaded`
    : data.size_gb_estimate != null
      ? `~${Number(data.size_gb_estimate).toFixed(2)} GB`
      : "";
  const detailParts = [`Ready on ${data.device}`];
  if (sizeLine) {
    detailParts.push(sizeLine);
  }
  if (memoryLine) {
    detailParts.push(`VRAM ${memoryLine}`);
  }

  if (data.ready) {
    modelStatus.innerHTML = `<strong>${data.backend_label}</strong><br>${data.model_name}<br><span class="meta-inline">${detailParts.join(" · ")}</span>`;
    playBtn.disabled = false;
  } else if (data.loading) {
    modelStatus.innerHTML = `<strong>${data.backend_label}</strong><br>Switching model…`;
    playBtn.disabled = true;
  } else {
    modelStatus.innerHTML = `<strong>${data.backend_label}</strong><br>Loading model…`;
    playBtn.disabled = true;
  }

  backendPill.textContent = `${data.backend_label || "MLX"} · Apple Silicon`;
  updateBuildInfo(data.build);
  return data;
}

async function selectCurrentModel() {
  const model = selectedModel();
  if (!model) {
    return;
  }
  if (model.status !== "ready") {
    throw new Error(
      model.status === "partial"
        ? "This model is incomplete. Finish the download before using it."
        : "Download this MLX model before using it.",
    );
  }

  modelWarning.classList.add("hidden");
  await reswitchPlayback({
    kind: "model",
    disablePlayBtn: true,
    applyChange: async ({ resumeChunk, savedTotalChunks }) => {
      const response = await fetch("/api/models/select", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model_id: model.id }),
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || "Could not switch model");
      }
      // Prefer server selection after a successful switch.
      await Promise.all([
        loadModels({ preferSelection: model.id }),
        loadModelStatus(),
        loadVoices({ preferVoice: voiceSelect.value }),
      ]);
      updatePlaybackControls();
      await rebindSessionKeepingPosition(resumeChunk, savedTotalChunks);
    },
  });
}

async function selectCurrentVoice() {
  await reswitchPlayback({
    kind: "voice",
    applyChange: async ({ resumeChunk, savedTotalChunks }) => {
      await rebindSessionKeepingPosition(resumeChunk, savedTotalChunks);
    },
  });
}

async function applySteadyReadingChange() {
  await reswitchPlayback({
    kind: "reading style",
    applyChange: async ({ resumeChunk, savedTotalChunks }) => {
      await rebindSessionKeepingPosition(resumeChunk, savedTotalChunks);
    },
  });
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
    const previousId = modelSelect.value;
    await loadModels({ preferSelection: previousId });
    await loadModelStatus();
    const model = selectedModel();
    // status.loading is MLX load state, not Hugging Face download progress.
    // Keep polling until the selected model leaves the downloading state.
    if (!model || model.status !== "downloading") {
      window.clearInterval(statusPollTimer);
      statusPollTimer = null;
      updateModelControls();
      if (model?.status === "ready" && !model.selected) {
        modelWarning.textContent = "Download complete. Click “Use model” to load it.";
        modelWarning.classList.remove("hidden");
      } else if (model?.status === "error" || model?.download_message) {
        if (model.status !== "ready") {
          modelWarning.textContent = model.download_message || "Download failed.";
          modelWarning.classList.remove("hidden");
        }
      }
    }
  }, 2500);
}

async function loadVoices({ preferVoice = null } = {}) {
  const previous = preferVoice || voiceSelect.value;
  const response = await fetch("/api/voices");
  const data = await response.json();
  const voices = data.voices || [];
  voiceSelect.innerHTML = "";
  let matchedPrevious = false;
  for (const voice of voices) {
    const option = document.createElement("option");
    option.value = voice.id;
    option.textContent = voice.label;
    if (previous && voice.id === previous) {
      option.selected = true;
      matchedPrevious = true;
    } else if (!matchedPrevious && voice.default) {
      option.selected = true;
    }
    voiceSelect.appendChild(option);
  }
  if (matchedPrevious) {
    voiceSelect.value = previous;
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

async function prepareSession(
  generation = playbackGeneration,
  startChunk = null,
  { preserveScroll = false } = {},
) {
  const text = getReadableText();
  if (!text) {
    throw new Error("Add some text first. If URL fetch failed, paste the article text.");
  }

  const hadReadingView = chunkTexts.length > 0;
  const resumeAt = startChunk ?? (hadReadingView ? currentChunk : null);
  const scrollTop = preserveScroll || hadReadingView ? captureReadingScroll() : null;

  const controller = new AbortController();
  const timeoutId = window.setTimeout(() => controller.abort(), 15000);

  const response = await fetch("/api/prepare", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(
      buildPreparePayload({
        startOffset: resolvePrepareStartOffset({ hadReadingView, startChunk: resumeAt }),
      }),
    ),
    signal: controller.signal,
  }).finally(() => window.clearTimeout(timeoutId));

  if (!isPlaybackActive(generation)) {
    throw new DOMException("Playback stopped", "AbortError");
  }

  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || "Could not prepare audio");
  }

  const chunks = data.chunks || [];
  const reuseView = hadReadingView && chunksMatch(chunks, chunkTexts);

  sessionId = data.session_id;
  totalChunks = data.chunk_count;
  chunkStarts = data.chunk_starts || [];
  prefetchCache.clear();

  if (!reuseView) {
    buildReadingView(chunks);
  }

  const startIndex = clampChunkIndex(resumeAt ?? data.start_chunk ?? 0);
  const startRatio = resumeAt === null ? Number(data.start_ratio) || 0 : 0;
  pendingResumeRatio = startRatio;
  restoreChunkPosition(startIndex, { preserveScroll: preserveScroll || reuseView });
  if (startRatio > 0) {
    setChunkProgress(startIndex, startRatio);
  }

  if (scrollTop !== null) {
    restoreReadingScroll(scrollTop);
  } else if (!hadReadingView && chunkElements[startIndex]) {
    scrollChunkIntoView(chunkElements[startIndex], true);
  }
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

  let timing;
  try {
    timing = await getBlobTiming(blob);
  } catch (_error) {
    throw new Error("Could not decode audio for playback");
  }

  const mediaDuration = timing.duration;
  if (!isPlaybackActive(generation) || mediaDuration <= 0) {
    return false;
  }

  // Map highlight over the speech-active window, not lead/trail silence.
  // Large Qwen clips otherwise leave the voice well ahead of the cursor.
  const speechStart = Math.max(0, Math.min(timing.speechStart ?? 0, mediaDuration));
  const speechEnd = Math.max(
    speechStart + 0.05,
    Math.min(timing.speechEnd ?? mediaDuration, mediaDuration),
  );
  const speechDuration = Math.max(0.05, speechEnd - speechStart);

  const startRatio = getPlaybackStartRatio(chunkIndex);

  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(blob);
    const audio = new Audio();
    currentAudio = audio;
    let rafId = 0;
    let completed = false;
    let earlyEndRetries = 0;
    let endTimer = 0;
    // Wall-clock anchor: (perf ms, estimated media seconds).
    let wallAnchorPerf = 0;
    let wallAnchorMedia = 0;
    let wallAnchored = false;
    const clampedStart = Math.max(0, Math.min(1, startRatio));
    // Media time within the speech window for mid-chunk resume.
    const mediaStart = speechStart + clampedStart * speechDuration;
    const nearEndEpsilon = Math.min(0.15, Math.max(0.05, speechDuration * 0.02));

    audio._mediaDuration = mediaDuration;
    audio._speechStart = speechStart;
    audio._speechEnd = speechEnd;
    audio._mediaStart = mediaStart;
    audio._startRatio = clampedStart;

    const estimatedMediaTime = () => {
      const reported = Number.isFinite(audio.currentTime) ? audio.currentTime : 0;
      if (!wallAnchored || audio.paused || isPaused) {
        return reported;
      }
      const speed = Math.max(0.25, playbackSpeed());
      const wallMedia = wallAnchorMedia + ((performance.now() - wallAnchorPerf) / 1000) * speed;
      // Browser currentTime often lags at 2–4× on long Qwen clips; DESIGN.md
      // uses max(currentTime, wallClockElapsed). Cap wall at speechEnd so we
      // never invent progress past the clip while waiting on trail silence.
      return Math.max(reported, Math.min(wallMedia, speechEnd));
    };

    const reanchorWallClock = () => {
      wallAnchorPerf = performance.now();
      wallAnchorMedia = Number.isFinite(audio.currentTime) ? audio.currentTime : mediaStart;
      wallAnchored = true;
    };

    const getRatio = () => {
      if (speechDuration <= 0) {
        return 1;
      }
      const mediaTime = estimatedMediaTime();
      if (mediaTime <= speechStart) {
        return clampedStart > 0 ? clampedStart : 0;
      }
      if (mediaTime >= speechEnd - nearEndEpsilon * 0.25) {
        // Close enough to speech end — snap remaining text so we don't
        // "skip" the tail when the last words land during trail silence.
        const raw = (mediaTime - speechStart) / speechDuration;
        return Math.min(1, Math.max(clampedStart, raw));
      }
      const progressInSpeech = Math.max(
        0,
        Math.min(1, (mediaTime - speechStart) / speechDuration),
      );
      // progressInSpeech is absolute within the speech window (includes start ratio).
      return Math.min(1, Math.max(clampedStart, progressInSpeech));
    };

    const isNearEnd = () => {
      if (audio.ended) {
        return true;
      }
      const mediaTime = estimatedMediaTime();
      return mediaTime >= speechEnd - nearEndEpsilon || mediaTime >= mediaDuration - nearEndEpsilon;
    };

    const seekToStart = () => {
      if (clampedStart <= 0 && speechStart <= 0.02) {
        return;
      }
      audio.currentTime = mediaStart;
      enforceAudioSpeed(audio, { force: true, afterSeek: true });
      setChunkProgress(chunkIndex, clampedStart);
    };

    const cleanup = () => {
      window.clearTimeout(endTimer);
      cancelAnimationFrame(rafId);
      audio.onended = null;
      audio.ontimeupdate = null;
      audio.onerror = null;
      audio.onplaying = null;
      audio.pause();
      URL.revokeObjectURL(url);
      if (activeBlobTiming?.audio === audio) {
        activeBlobTiming = null;
      }
    };

    const updateHighlight = () => {
      if (!isPlaybackActive(generation) || currentAudio !== audio) {
        return;
      }
      // When currentTime catches up to the wall estimate, re-sync the anchor
      // so small rate errors do not accumulate across long Qwen chunks.
      if (wallAnchored && !audio.paused && !isPaused) {
        const speed = Math.max(0.25, playbackSpeed());
        const wallMedia =
          wallAnchorMedia + ((performance.now() - wallAnchorPerf) / 1000) * speed;
        const reported = audio.currentTime;
        if (Number.isFinite(reported) && reported + 0.05 >= wallMedia) {
          reanchorWallClock();
        }
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
      if (cancelActiveBlobPlayback === finishChunk) {
        cancelActiveBlobPlayback = null;
      }
      cleanup();
      if (currentAudio === audio) {
        currentAudio = null;
      }
      if (success) {
        setChunkProgress(chunkIndex, 1);
        markChunkDone(chunkIndex);
      }
      resolve(Boolean(success) && isPlaybackActive(generation));
    };

    const scheduleEndWatchdog = () => {
      window.clearTimeout(endTimer);
      if (completed || isPaused) {
        return;
      }
      const speed = Math.max(0.25, playbackSpeed());
      const remainingMedia = Math.max(0, speechEnd - estimatedMediaTime());
      const wallMs = (remainingMedia / speed) * 1000;
      // Complete shortly after the speech window should end (don't wait on trail silence).
      const ms = Math.max(800, wallMs + 600);
      endTimer = window.setTimeout(() => {
        if (completed) {
          return;
        }
        if (!isPlaybackActive(generation) || currentAudio !== audio) {
          finishChunk(false);
          return;
        }
        if (isPaused) {
          scheduleEndWatchdog();
          return;
        }
        if (isNearEnd() || getRatio() >= 0.97) {
          finishChunk(isPlaybackActive(generation));
          return;
        }
        // Still mid-chunk with no ended event — try resume once, then re-arm.
        if (earlyEndRetries < 3 && !audio.paused) {
          earlyEndRetries += 1;
          enforceAudioSpeed(audio, { force: true });
          audio.play().catch(() => {});
        }
        scheduleEndWatchdog();
      }, ms);
    };

    const handleEnded = () => {
      if (completed) {
        return;
      }
      if (isNearEnd() || getRatio() >= 0.9) {
        finishChunk(isPlaybackActive(generation));
        return;
      }
      // Premature ended: retry resume instead of marking the chunk done.
      if (isPlaybackActive(generation) && !isPaused && earlyEndRetries < 5) {
        earlyEndRetries += 1;
        enforceAudioSpeed(audio, { force: true });
        audio
          .play()
          .then(() => {
            reanchorWallClock();
            scheduleEndWatchdog();
            rafId = requestAnimationFrame(tick);
          })
          .catch(() => finishChunk(false));
        return;
      }
      if (getRatio() >= 0.85) {
        finishChunk(isPlaybackActive(generation));
      } else {
        finishChunk(false);
      }
    };

    cancelActiveBlobPlayback = finishChunk;
    activeBlobTiming = { audio, scheduleEndWatchdog, reanchorWallClock };

    audio.onended = handleEnded;
    audio.onerror = () => {
      if (cancelActiveBlobPlayback === finishChunk) {
        cancelActiveBlobPlayback = null;
      }
      cleanup();
      if (currentAudio === audio) {
        currentAudio = null;
      }
      reject(new Error("Audio playback failed"));
    };
    audio.ontimeupdate = updateHighlight;
    // Prefer the real "playing" event for wall-clock anchor — play() may
    // resolve before the first sample is emitted on large first chunks.
    audio.onplaying = () => {
      enforceAudioSpeed(audio, { force: true });
      reanchorWallClock();
      scheduleEndWatchdog();
    };

    const beginPlay = () => {
      if (!isPlaybackActive(generation)) {
        finishChunk(false);
        return;
      }
      applyPreservesPitch(audio);
      enforceAudioSpeed(audio, { force: true });
      if (clampedStart > 0) {
        setChunkProgress(chunkIndex, clampedStart);
      } else if (!readingScrollLocked) {
        setChunkProgress(chunkIndex, 0);
      }
      audio
        .play()
        .then(() => {
          if (!isPlaybackActive(generation)) {
            audio.pause();
            finishChunk(false);
            return;
          }
          enforceAudioSpeed(audio, { force: true });
          // Fallback anchor if "playing" already fired or never does.
          if (!wallAnchored) {
            reanchorWallClock();
          }
          scheduleEndWatchdog();
          rafId = requestAnimationFrame(tick);
        })
        .catch((error) => {
          if (cancelActiveBlobPlayback === finishChunk) {
            cancelActiveBlobPlayback = null;
          }
          cleanup();
          reject(error);
        });
    };

    let playbackPrepared = false;
    const prepareAndPlay = () => {
      if (playbackPrepared || completed) {
        return;
      }
      playbackPrepared = true;
      applyPreservesPitch(audio);
      enforceAudioSpeed(audio, { force: true });
      seekToStart();
      beginPlay();
    };

    audio.addEventListener("loadedmetadata", prepareAndPlay, { once: true });
    applyPreservesPitch(audio);
    audio.defaultPlaybackRate = playbackSpeed();
    audio.playbackRate = playbackSpeed();
    audio.src = url;
    if (audio.readyState >= HTMLMediaElement.HAVE_METADATA) {
      prepareAndPlay();
    }
  });
}

async function runPlaybackLoop(generation, { skipPositionReset = false } = {}) {
  if (!sessionId) {
    setStatus("Preparing…");
    if (chunkTexts.length > 0) {
      await refreshSessionOnly(generation, currentChunk);
    } else {
      await prepareSession(generation, null, { preserveScroll: readingScrollLocked });
    }
    if (!isPlaybackActive(generation)) {
      return;
    }
    showReadingMode();
  } else if (!skipPositionReset && !readingScrollLocked && !playbackReswitchActive) {
    restoreChunkPosition(currentChunk, { preserveScroll: false });
  } else {
    showReadingMode();
    updateSidebarIndicators();
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
      currentChunk = index;
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
    if (pendingReswitchChunk === null && !playbackReswitchActive) {
      restoreChunkPosition(currentChunk);
      setStatus("Stopped");
    }
    return;
  }

  if (totalChunks > 0 && currentChunk >= totalChunks) {
    setStatus("Finished");
    endReadingSession();
  } else if (isPaused) {
    setStatus("Paused");
  }
}

async function beginPlayback({ force = false, skipPositionReset = false } = {}) {
  if (!getReadableText()) {
    fetchWarning.textContent = "Add some text first. Open the page, copy the text, and paste it here.";
    fetchWarning.classList.remove("hidden");
    setStatus("Ready");
    return;
  }

  if (!isReadingMode || chunkTexts.length === 0) {
    startOffset = getPlaybackStartOffset();
    updateTextMeta();
  }

  if (!force && (playAllPromise || playAllLock)) {
    if (isPaused && playAllPromise) {
      isPaused = false;
      setStatus(`Playing chunk ${currentChunk + 1} of ${totalChunks}`);
      setButtons({ playing: true, paused: false });
      if (currentAudio) {
        applyAudioSpeed(currentAudio);
        currentAudio
          .play()
          .then(() => {
            if (typeof activeBlobTiming?.reanchorWallClock === "function") {
              activeBlobTiming.reanchorWallClock();
            }
            if (typeof activeBlobTiming?.scheduleEndWatchdog === "function") {
              activeBlobTiming.scheduleEndWatchdog();
            }
          })
          .catch(() => {});
      }
    }
    return;
  }

  if (force && (playAllPromise || playAllLock)) {
    playbackGeneration += 1;
    if (cancelActiveBlobPlayback) {
      cancelActiveBlobPlayback(false);
      cancelActiveBlobPlayback = null;
    }
    abortPendingFetches();
    if (currentAudio) {
      detachAudio(currentAudio);
      currentAudio = null;
    }
    await waitForActivePlaybackWorker();
    playAllPromise = null;
    playAllLock = false;
    stopChunkStatusPoll();
  }

  const generation = playbackGeneration;
  isPaused = false;
  setButtons({ playing: true, paused: false });
  startChunkStatusPoll();
  playAllLock = true;

  const worker = runPlaybackLoop(generation, { skipPositionReset }).catch((error) => {
    if (!isPlaybackActive(generation)) {
      return;
    }
    setStatus("Playback failed");
    fetchWarning.textContent =
      error.name === "AbortError"
        ? "Preparing timed out. Try again or restart the server."
        : error.message;
    fetchWarning.classList.remove("hidden");
    showReadingMode();
    updateSidebarIndicators();
    setButtons({ playing: false, paused: false });
  });

  playAllPromise = worker;
  try {
    await worker;
  } finally {
    if (playAllPromise !== worker) {
      return;
    }
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
}

async function playAll(options = {}) {
  return beginPlayback(options);
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
  // Freeze wall-clock tracking while paused so resume doesn't skip ahead.
  if (typeof activeBlobTiming?.reanchorWallClock === "function") {
    activeBlobTiming.reanchorWallClock();
  }
}

async function stopPlayback({ statusMessage = "Stopped", resumeChunk = null } = {}) {
  const activeSessionId = sessionId;
  const audio = currentAudio;
  playbackGeneration += 1;
  pendingResumeRatio = 0;
  isPaused = false;
  stopChunkStatusPoll();
  if (cancelActiveBlobPlayback) {
    cancelActiveBlobPlayback(false);
    cancelActiveBlobPlayback = null;
  }
  abortPendingFetches();
  if (audio) {
    detachAudio(audio);
    if (currentAudio === audio) {
      currentAudio = null;
    }
  }
  prefetchCache.clear();
  cancelServerSession(activeSessionId);
  sessionId = null;
  await waitForActivePlaybackWorker();
  playAllPromise = null;
  playAllLock = false;
  if (resumeChunk !== null) {
    restoreChunkPosition(resumeChunk, { preserveScroll: readingScrollLocked });
  } else if (totalChunks > 0) {
    restoreChunkPosition(currentChunk, { preserveScroll: readingScrollLocked });
  } else {
    updateProgress();
  }
  setStatus(statusMessage);
  setButtons({ playing: false, paused: false });
}

async function bindNewSession(startChunk, { checkGeneration = null } = {}) {
  const text = getReadableText();
  if (!text) {
    throw new Error("Add some text first. If URL fetch failed, paste the article text.");
  }

  const controller = new AbortController();
  const timeoutId = window.setTimeout(() => controller.abort(), 15000);

  const response = await fetch("/api/prepare", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(
      buildPreparePayload({
        startOffset:
          startChunk != null
            ? getOffsetForChunkIndex(startChunk, getChunkReadRatio(startChunk))
            : getPlaybackStartOffset(),
      }),
    ),
    signal: controller.signal,
  }).finally(() => window.clearTimeout(timeoutId));

  if (checkGeneration !== null && !isPlaybackActive(checkGeneration)) {
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
  currentChunk = clampChunkIndex(startChunk ?? currentChunk);
  return data;
}

async function waitForActivePlaybackWorker() {
  while (playAllLock && !playAllPromise) {
    await new Promise((resolve) => setTimeout(resolve, 20));
  }
  if (playAllPromise) {
    await playAllPromise.catch(() => {});
  }
}

async function interruptPlaybackForReswitch() {
  const activeSessionId = sessionId;
  const audio = currentAudio;
  playbackGeneration += 1;
  isPaused = false;
  stopChunkStatusPoll();
  if (cancelActiveBlobPlayback) {
    cancelActiveBlobPlayback(false);
    cancelActiveBlobPlayback = null;
  }
  abortPendingFetches();
  if (audio) {
    detachAudio(audio);
    if (currentAudio === audio) {
      currentAudio = null;
    }
  }
  prefetchCache.clear();
  sessionId = null;
  cancelServerSession(activeSessionId);
  setButtons({ playing: false, paused: false });
  await waitForActivePlaybackWorker();
  playAllPromise = null;
  playAllLock = false;
  stopChunkStatusPoll();
}

async function refreshSessionOnly(generation = playbackGeneration, startChunk = null) {
  const data = await bindNewSession(startChunk, { checkGeneration: generation });
  clearAudioCacheStates(currentChunk);
  return data;
}

async function resetSession() {
  if (playbackReswitchActive) {
    return;
  }
  if (playAllPromise || playAllLock || currentAudio) {
    await stopPlayback();
  }
  sessionId = null;
  totalChunks = 0;
  currentChunk = 0;
  prefetchCache.clear();
  clearReadingSurface();
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
editModeBtn.addEventListener("click", async () => {
  await stopPlayback();
  sessionId = null;
  totalChunks = 0;
  currentChunk = 0;
  prefetchCache.clear();
  clearReadingSurface();
  updateProgress();
  updateTextMeta();
  showEditMode();
});
textInput.addEventListener("input", () => {
  resetSession();
  updateTextMeta();
});
textInput.addEventListener("click", rememberCursorStart);
textInput.addEventListener("mouseup", rememberCursorStart);
textInput.addEventListener("keyup", rememberCursorStart);
textInput.addEventListener("select", rememberCursorStart);
document.addEventListener("selectionchange", () => {
  if (document.activeElement === textInput) {
    rememberCursorStart();
  }
});
readingView.addEventListener("click", async (event) => {
  const chunk = event.target.closest(".chunk");
  if (!chunk) {
    return;
  }
  await jumpToChunk(Number(chunk.dataset.index));
});
voiceSelect.addEventListener("pointerdown", () => {
  voiceSelect.dataset.previousVoice = voiceSelect.value;
});

voiceSelect.addEventListener("change", async () => {
  const previousVoice = voiceSelect.dataset.previousVoice || voiceSelect.value;
  if (!hasActiveReadingSession()) {
    voiceSelect.dataset.previousVoice = voiceSelect.value;
    return;
  }
  try {
    fetchWarning.classList.add("hidden");
    await selectCurrentVoice();
    voiceSelect.dataset.previousVoice = voiceSelect.value;
  } catch (error) {
    voiceSelect.value = previousVoice;
    fetchWarning.textContent = error.message;
    fetchWarning.classList.remove("hidden");
    setStatus("Ready");
  }
});

steadyReadingToggle.addEventListener("change", async () => {
  const previousValue = !steadyReadingToggle.checked;
  if (!hasActiveReadingSession()) {
    return;
  }
  try {
    fetchWarning.classList.add("hidden");
    await applySteadyReadingChange();
  } catch (error) {
    steadyReadingToggle.checked = previousValue;
    fetchWarning.textContent = error.message;
    fetchWarning.classList.remove("hidden");
    setStatus("Ready");
  }
});
modelSelect.addEventListener("pointerdown", () => {
  modelSelect.dataset.previousModel = modelSelect.value;
});

modelSelect.addEventListener("change", async () => {
  const previousModel = modelSelect.dataset.previousModel || modelSelect.value;
  const model = selectedModel();
  updateModelControls();

  if (!model || model.selected) {
    modelSelect.dataset.previousModel = modelSelect.value;
    return;
  }

  // Not downloaded yet — leave selection so the user can hit Download.
  if (model.status !== "ready") {
    modelWarning.classList.add("hidden");
    if (model.status === "partial") {
      modelWarning.textContent = "This model is incomplete. Finish the download before using it.";
      modelWarning.classList.remove("hidden");
    }
    modelSelect.dataset.previousModel = modelSelect.value;
    return;
  }

  try {
    modelWarning.classList.add("hidden");
    fetchWarning.classList.add("hidden");
    await selectCurrentModel();
    modelSelect.dataset.previousModel = modelSelect.value;
  } catch (error) {
    modelSelect.value = previousModel;
    modelWarning.textContent = error.message;
    modelWarning.classList.remove("hidden");
    updateModelControls();
    setStatus("Ready");
  }
});

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
playBtn.addEventListener("click", () => {
  void playAll();
});
pauseBtn.addEventListener("click", pausePlayback);
stopBtn.addEventListener("click", stopPlayback);
speedSlider.addEventListener("input", onSpeedControlInput);
speedSlider.addEventListener("change", onSpeedControlInput);

document.querySelectorAll("[data-speed]").forEach((button) => {
  button.addEventListener("click", () => {
    setPlaybackSpeed(button.dataset.speed);
  });
});

async function bootstrap() {
  const params = new URLSearchParams(window.location.search);
  const initialUrl = params.get("url");
  if (initialUrl) {
    urlInput.value = initialUrl;
    await fetchUrl();
  }

  preferredPlaybackSpeed = loadStoredPlaybackSpeed();
  reflectSpeedUi();

  await Promise.all([loadVoices(), loadModels(), loadModelStatus()]);
  updatePlaybackControls();
  updateTextMeta();
  setStatus("Ready");
  const model = selectedModel();
  if (model && model.status === "downloading") {
    startModelPolling();
  }
}

bootstrap();