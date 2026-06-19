import { chromium } from "playwright";

const BASE = process.env.READ_ALOUD_URL || "http://127.0.0.1:8765";
const TEXT =
  "This is section one of a voice switch test. ".repeat(20) +
  "This is section two with more words to read aloud. ".repeat(20) +
  "This is section three near the end of the sample article. ".repeat(20);

async function state(page) {
  return page.evaluate(() => ({
    status: document.getElementById("playbackStatus")?.textContent,
    textHidden: document.getElementById("textInput")?.classList.contains("hidden"),
    readingHidden: document.getElementById("readingView")?.classList.contains("hidden"),
    chunkElements: window.chunkElements?.length ?? "no-global",
    totalChunks: window.totalChunks ?? "no-global",
    playAllLock: window.playAllLock ?? "no-global",
    playAllPromise: !!window.playAllPromise,
    sessionId: window.sessionId ?? "no-global",
    warning: document.getElementById("fetchWarning")?.textContent,
  }));
}

// Expose globals for debugging
const expose = `
  window.chunkElements = chunkElements;
  window.totalChunks = totalChunks;
  window.playAllLock = playAllLock;
  window.playAllPromise = playAllPromise;
  window.sessionId = sessionId;
`;

const browser = await chromium.launch({ headless: true });
const page = await browser.newPage();

page.on("console", (msg) => {
  if (msg.type() === "error") console.log("browser error:", msg.text());
});

await page.goto(`${BASE}/?t=${Date.now()}`, { waitUntil: "networkidle" });
await page.evaluate(expose);

// Paste text
await page.fill("#textInput", TEXT);
await page.evaluate(expose);

console.log("before play", await state(page));

await page.click("#playBtn");

// Wait for reading mode + synthesis
await page.waitForFunction(
  () =>
    !document.getElementById("readingView")?.classList.contains("hidden") &&
    (document.getElementById("playbackStatus")?.textContent || "").includes("Playing"),
  { timeout: 120000 },
);

await page.waitForTimeout(1500);
await page.evaluate(expose);
console.log("during play", await state(page));

const voices = await page.$$eval("#voiceSelect option", (opts) =>
  opts.map((o) => ({ value: o.value, label: o.textContent })),
);
const currentVoice = await page.inputValue("#voiceSelect");
const nextVoice = voices.find((v) => v.value !== currentVoice) || voices[1];
console.log("switching voice", currentVoice, "->", nextVoice?.value);

await page.selectOption("#voiceSelect", nextVoice.value);

await page.waitForFunction(
  () => {
    const s = document.getElementById("playbackStatus")?.textContent || "";
    return (
      s.includes("Switching voice") ||
      s.includes("Resuming section") ||
      s.includes("Synthesizing chunk") ||
      s.includes("Playing chunk") ||
      s.includes("Playback failed") ||
      s.includes("Ready — press Play")
    );
  },
  { timeout: 60000 },
);

await page.waitForTimeout(8000);
await page.evaluate(expose);
console.log("after voice change (+8s)", await state(page));

// Stop any stuck worker, then try play again
await page.click("#stopBtn").catch(() => {});
await page.waitForTimeout(1000);
await page.evaluate(expose);
console.log("after stop", await state(page));

const playDisabled = await page.isDisabled("#playBtn");
console.log("playBtn disabled:", playDisabled);

if (!playDisabled) {
  await page.click("#playBtn");
  await page.waitForFunction(
    () => (document.getElementById("playbackStatus")?.textContent || "").includes("Playing"),
    { timeout: 120000 },
  );
  await page.evaluate(expose);
  console.log("after second play", await state(page));
} else {
  console.log("FAIL: play still disabled after stop");
}

await browser.close();