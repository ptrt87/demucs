const input = document.querySelector("#audioInput");
const dropZone = document.querySelector("#dropZone");
const fileName = document.querySelector("#fileName");
const processButton = document.querySelector("#processButton");
const resetButton = document.querySelector("#resetButton");
const uploadBar = document.querySelector("#uploadBar");
const uploadPercent = document.querySelector("#uploadPercent");
const processBar = document.querySelector("#processBar");
const processPercent = document.querySelector("#processPercent");
const statusText = document.querySelector("#statusText");
const stageList = Array.from(document.querySelectorAll("#stageList li"));
const vocalsDownload = document.querySelector("#vocalsDownload");
const instrumentalDownload = document.querySelector("#instrumentalDownload");
const errorBox = document.querySelector("#errorBox");
const waveCanvas = document.querySelector("#waveCanvas");
const waveContext = waveCanvas.getContext("2d");

let selectedFile = null;
let currentJobId = null;
let pollTimer = null;

const stages = [
  "Separating audio...",
  "Checking background noise...",
  "Removing noise...",
  "Checking bleed and artifacts...",
  "Enhancing audio...",
  "Finalizing files..."
];

drawIdleWave();

input.addEventListener("change", () => {
  setSelectedFile(input.files[0] || null);
});

dropZone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropZone.classList.add("dragover");
});

dropZone.addEventListener("dragleave", () => {
  dropZone.classList.remove("dragover");
});

dropZone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropZone.classList.remove("dragover");
  const file = event.dataTransfer.files[0];
  if (file) {
    input.files = event.dataTransfer.files;
    setSelectedFile(file);
  }
});

processButton.addEventListener("click", () => {
  if (!selectedFile) {
    return;
  }
  uploadAndProcess(selectedFile);
});

resetButton.addEventListener("click", () => {
  resetApp(true);
});

function setSelectedFile(file) {
  selectedFile = file;
  clearError();
  setDownloadLinks(null);
  if (!file) {
    fileName.textContent = "No file selected";
    processButton.disabled = true;
    drawIdleWave();
    return;
  }
  fileName.textContent = file.name;
  processButton.disabled = false;
  renderWaveform(file);
}

function uploadAndProcess(file) {
  clearError();
  processButton.disabled = true;
  setProgress(uploadBar, uploadPercent, 0);
  setProgress(processBar, processPercent, 0);
  statusText.textContent = "Uploading audio";
  markStage("");

  const formData = new FormData();
  formData.append("audio", file);
  const request = new XMLHttpRequest();
  request.open("POST", "/api/jobs");
  request.upload.addEventListener("progress", (event) => {
    if (event.lengthComputable) {
      setProgress(uploadBar, uploadPercent, event.loaded / event.total);
    }
  });
  request.addEventListener("load", () => {
    if (request.status < 200 || request.status >= 300) {
      showError(readError(request.responseText));
      processButton.disabled = false;
      return;
    }
    setProgress(uploadBar, uploadPercent, 1);
    const payload = JSON.parse(request.responseText);
    currentJobId = payload.id;
    pollJob();
  });
  request.addEventListener("error", () => {
    showError("Upload failed. Check the file and try again.");
    processButton.disabled = false;
  });
  request.send(formData);
}

async function pollJob() {
  if (!currentJobId) {
    return;
  }
  window.clearTimeout(pollTimer);
  try {
    const response = await fetch(`/api/jobs/${currentJobId}`);
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || "Could not read job status.");
    }
    updateJob(payload);
    if (payload.status === "processing" || payload.status === "queued") {
      pollTimer = window.setTimeout(pollJob, 900);
    }
  } catch (error) {
    showError(error.message);
    processButton.disabled = false;
  }
}

function updateJob(job) {
  statusText.textContent = job.stage || "Processing";
  setProgress(processBar, processPercent, job.progress || 0);
  markStage(job.stage);
  if (job.status === "complete") {
    statusText.textContent = "Files are ready";
    markStage("done");
    setDownloadLinks(job.id);
  }
  if (job.status === "failed") {
    showError(job.error || "Processing failed.");
    processButton.disabled = false;
  }
}

function markStage(stage) {
  let activeIndex = stages.indexOf(stage);
  if (stage === "done") {
    activeIndex = stages.length;
  }
  stageList.forEach((item, index) => {
    item.classList.toggle("active", index === activeIndex);
    item.classList.toggle("done", activeIndex > index);
  });
}

function setDownloadLinks(jobId) {
  if (!jobId) {
    vocalsDownload.href = "#";
    instrumentalDownload.href = "#";
    vocalsDownload.classList.add("disabled");
    instrumentalDownload.classList.add("disabled");
    return;
  }
  vocalsDownload.href = `/api/jobs/${jobId}/download/vocals`;
  instrumentalDownload.href = `/api/jobs/${jobId}/download/instrumental`;
  vocalsDownload.classList.remove("disabled");
  instrumentalDownload.classList.remove("disabled");
}

async function resetApp(deleteJob) {
  window.clearTimeout(pollTimer);
  if (deleteJob && currentJobId) {
    fetch(`/api/jobs/${currentJobId}`, { method: "DELETE" }).catch(() => {});
  }
  currentJobId = null;
  selectedFile = null;
  input.value = "";
  fileName.textContent = "No file selected";
  processButton.disabled = true;
  statusText.textContent = "Waiting for audio";
  setProgress(uploadBar, uploadPercent, 0);
  setProgress(processBar, processPercent, 0);
  markStage("");
  setDownloadLinks(null);
  clearError();
  drawIdleWave();
}

function setProgress(bar, label, value) {
  const percent = Math.max(0, Math.min(100, Math.round(value * 100)));
  bar.style.width = `${percent}%`;
  label.textContent = `${percent}%`;
}

function showError(message) {
  errorBox.textContent = message;
  errorBox.hidden = false;
  statusText.textContent = "Needs attention";
}

function clearError() {
  errorBox.textContent = "";
  errorBox.hidden = true;
}

function readError(text) {
  try {
    return JSON.parse(text).error || "Something went wrong.";
  } catch {
    return "Something went wrong.";
  }
}

async function renderWaveform(file) {
  drawLoadingWave();
  try {
    const arrayBuffer = await file.arrayBuffer();
    const AudioContext = window.AudioContext || window.webkitAudioContext;
    const audioContext = new AudioContext();
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer.slice(0));
    const data = audioBuffer.getChannelData(0);
    drawWave(data);
    await audioContext.close();
  } catch {
    drawIdleWave();
  }
}

function drawWave(data) {
  resizeCanvas();
  const { width, height } = waveCanvas;
  waveContext.clearRect(0, 0, width, height);
  waveContext.fillStyle = "#101512";
  waveContext.fillRect(0, 0, width, height);
  const step = Math.max(1, Math.floor(data.length / width));
  const center = height / 2;
  const gradient = waveContext.createLinearGradient(0, 0, width, 0);
  gradient.addColorStop(0, "#0f8b7b");
  gradient.addColorStop(0.55, "#c79728");
  gradient.addColorStop(1, "#dc6b4d");
  waveContext.strokeStyle = gradient;
  waveContext.lineWidth = 2;
  waveContext.beginPath();
  for (let x = 0; x < width; x += 1) {
    let min = 1;
    let max = -1;
    for (let j = 0; j < step; j += 1) {
      const sample = data[(x * step) + j] || 0;
      min = Math.min(min, sample);
      max = Math.max(max, sample);
    }
    waveContext.moveTo(x, center + min * center * 0.86);
    waveContext.lineTo(x, center + max * center * 0.86);
  }
  waveContext.stroke();
}

function drawIdleWave() {
  resizeCanvas();
  const { width, height } = waveCanvas;
  waveContext.clearRect(0, 0, width, height);
  waveContext.fillStyle = "#101512";
  waveContext.fillRect(0, 0, width, height);
  waveContext.strokeStyle = "#2f4f47";
  waveContext.lineWidth = 2;
  waveContext.beginPath();
  for (let x = 0; x < width; x += 10) {
    const y = height / 2 + Math.sin(x / 28) * 24 + Math.sin(x / 73) * 16;
    if (x === 0) {
      waveContext.moveTo(x, y);
    } else {
      waveContext.lineTo(x, y);
    }
  }
  waveContext.stroke();
}

function drawLoadingWave() {
  resizeCanvas();
  const { width, height } = waveCanvas;
  waveContext.clearRect(0, 0, width, height);
  waveContext.fillStyle = "#101512";
  waveContext.fillRect(0, 0, width, height);
  waveContext.fillStyle = "#dbe7dd";
  waveContext.fillRect(width * 0.1, height / 2 - 4, width * 0.8, 8);
}

function resizeCanvas() {
  const rect = waveCanvas.getBoundingClientRect();
  const scale = window.devicePixelRatio || 1;
  const width = Math.max(320, Math.floor(rect.width * scale));
  const height = Math.max(140, Math.floor(rect.height * scale));
  if (waveCanvas.width !== width || waveCanvas.height !== height) {
    waveCanvas.width = width;
    waveCanvas.height = height;
  }
}

window.addEventListener("resize", () => {
  if (!selectedFile) {
    drawIdleWave();
  }
});
