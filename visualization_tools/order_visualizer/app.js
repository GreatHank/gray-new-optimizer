const DEFAULTS = Object.freeze({
  wavelength: 480,
  period: 2000,
  channels: 9,
  mStart: -3,
  nStart: -3,
  theta: 42.7517,
  phi: 45,
  nIn: 1,
  nOut: 1,
  na: 1,
  opacity: 34,
});

const MAX_FULL_FIELD_SCAN = 2500;
const PARAMETER_INPUT_DELAY_MS = 180;
const CHANNEL_OPTIONS = Array.from({ length: 19 }, (_, index) => (index + 2) ** 2);

const $ = (selector) => document.querySelector(selector);
const elements = {
  imageInput: $("#imageInput"),
  uploadBox: $("#uploadBox"),
  uploadTitle: $("#uploadTitle"),
  sourcePreview: $("#sourcePreview"),
  sourceCanvas: $("#sourceCanvas"),
  fileName: $("#fileName"),
  imageSize: $("#imageSize"),
  errorMessage: $("#errorMessage"),
  wavelength: $("#wavelength"),
  period: $("#period"),
  channels: $("#channels"),
  mStart: $("#mStart"),
  nStart: $("#nStart"),
  thetaRange: $("#thetaRange"),
  thetaNumber: $("#thetaNumber"),
  phiRange: $("#phiRange"),
  phiNumber: $("#phiNumber"),
  nIn: $("#nIn"),
  nOut: $("#nOut"),
  na: $("#na"),
  imageOpacity: $("#imageOpacity"),
  opacityValue: $("#opacityValue"),
  centerButton: $("#centerButton"),
  resetButton: $("#resetButton"),
  exportButton: $("#exportButton"),
  orderCanvas: $("#orderCanvas"),
  canvasWrap: $("#canvasWrap"),
  canvasPlaceholder: $("#canvasPlaceholder"),
  plotTooltip: $("#plotTooltip"),
  gridTitle: $("#gridTitle"),
  orderRange: $("#orderRange"),
  fovValue: $("#fovValue"),
  fovAxis: $("#fovAxis"),
  propagatingMetric: $("#propagatingMetric"),
  capturedMetric: $("#capturedMetric"),
  candidateMetric: $("#candidateMetric"),
  deltaMetric: $("#deltaMetric"),
  marginMetric: $("#marginMetric"),
  orderTableBody: $("#orderTableBody"),
  toast: $("#toast"),
};

const plotContext = elements.orderCanvas.getContext("2d");
const sourceContext = elements.sourceCanvas.getContext("2d");
let sourceImage = null;
let sourceObjectUrl = null;
let lastModel = null;
let hitRegions = [];
let activeOrderKey = null;
let toastTimer = null;
let recalculateTimer = null;

for (const channels of CHANNEL_OPTIONS) {
  const gridSize = Math.sqrt(channels);
  elements.channels.add(new Option(`${channels}（${gridSize} × ${gridSize}）`, String(channels)));
}
elements.channels.value = String(DEFAULTS.channels);

function parseNumber(input, label) {
  const value = Number(input.value);
  if (!Number.isFinite(value)) throw new Error(`${label}必须是有效数字。`);
  return value;
}

function readParameters() {
  const wavelength = parseNumber(elements.wavelength, "波长 λ");
  const period = parseNumber(elements.period, "周期 P");
  const channels = parseNumber(elements.channels, "Channel 数");
  const mStart = parseNumber(elements.mStart, "m 起始级次");
  const nStart = parseNumber(elements.nStart, "n 起始级次");
  const theta = parseNumber(elements.thetaNumber, "极角 θ");
  const phi = parseNumber(elements.phiNumber, "方位角 φ");
  const nIn = parseNumber(elements.nIn, "入射折射率");
  const nOut = parseNumber(elements.nOut, "输出折射率");
  const na = parseNumber(elements.na, "数值孔径 NA");
  const opacity = parseNumber(elements.imageOpacity, "图片透明度");

  if (wavelength <= 0 || period <= 0) throw new Error("波长 λ 和周期 P 必须大于 0 nm。");
  if (!CHANNEL_OPTIONS.includes(channels)) throw new Error("Channel 数必须从完全平方数列表中选择。");
  if (!Number.isInteger(mStart) || !Number.isInteger(nStart)) throw new Error("m、n 起始级次必须是整数。");
  if (theta < 0 || theta >= 90) throw new Error("极角 θ 必须满足 0° ≤ θ < 90°。");
  if (phi < -180 || phi > 180) throw new Error("方位角 φ 必须在 −180° 到 180° 之间。");
  if (nIn <= 0 || nOut <= 0) throw new Error("入射和输出折射率必须大于 0。");
  if (na <= 0 || na > nOut) throw new Error("NA 必须大于 0，且不能超过输出折射率 nout。");

  return { wavelength, period, channels, gridSize: Math.sqrt(channels), mStart, nStart, theta, phi, nIn, nOut, na, opacity };
}

function buildModel(parameters) {
  const radians = Math.PI / 180;
  const incidentX = parameters.nIn * Math.sin(parameters.theta * radians) * Math.cos(parameters.phi * radians);
  const incidentY = parameters.nIn * Math.sin(parameters.theta * radians) * Math.sin(parameters.phi * radians);
  const delta = parameters.wavelength / (parameters.period * parameters.nOut);
  const halfCell = delta / 2;
  const orders = [];

  const createOrder = (m, n, selection = null) => {
    const transverseX = incidentX + m * parameters.wavelength / parameters.period;
    const transverseY = incidentY + n * parameters.wavelength / parameters.period;
    const ux = transverseX / parameters.nOut;
    const uy = transverseY / parameters.nOut;
    const radius = Math.hypot(ux, uy);
    const nearestX = Math.max(Math.abs(ux) - halfCell, 0);
    const nearestY = Math.max(Math.abs(uy) - halfCell, 0);
    const farthestRadius = Math.hypot(Math.abs(ux) + halfCell, Math.abs(uy) + halfCell);
    const intersectsAir = Math.hypot(nearestX, nearestY) <= 1 + 1e-12;
    const fullyInsideAir = farthestRadius <= 1 + 1e-12;
    const propagating = radius <= 1 + 1e-12;
    const captured = propagating && Math.hypot(transverseX, transverseY) <= parameters.na + 1e-12;
    return {
      key: `${m},${n}`,
      m,
      n,
      ux,
      uy,
      radius,
      propagating,
      captured,
      intersectsAir,
      fullyInsideAir,
      airMargin: 1 - radius,
      thetaOut: propagating ? Math.asin(Math.min(1, radius)) / radians : null,
      selected: Boolean(selection),
      index: selection?.index ?? null,
      row: selection?.row ?? null,
      column: selection?.column ?? null,
    };
  };

  for (let row = 0; row < parameters.gridSize; row += 1) {
    for (let column = 0; column < parameters.gridSize; column += 1) {
      // Keep the source image visually intact on the physical plot:
      // image columns run left-to-right with m, while image rows run
      // top-to-bottom against the upward-positive n axis.
      const m = parameters.mStart + column;
      const n = parameters.nStart + (parameters.gridSize - 1 - row);
      orders.push(createOrder(m, n, { index: row * parameters.gridSize + column, row, column }));
    }
  }

  // Paper Figure S9 treats each diffraction order as a local square spectrum.
  // Build every integer order whose square touches the air propagation disk,
  // including cells whose center is outside but whose local spectrum overlaps it.
  const baseUx = incidentX / parameters.nOut;
  const baseUy = incidentY / parameters.nOut;
  const mMin = Math.ceil((-1 - halfCell - baseUx) / delta - 1e-12);
  const mMax = Math.floor((1 + halfCell - baseUx) / delta + 1e-12);
  const nMin = Math.ceil((-1 - halfCell - baseUy) / delta - 1e-12);
  const nMax = Math.floor((1 + halfCell - baseUy) / delta + 1e-12);
  const scanWidth = mMax - mMin + 1;
  const scanHeight = nMax - nMin + 1;
  const scanCount = scanWidth * scanHeight;
  if (!Number.isSafeInteger(scanCount) || scanCount > MAX_FULL_FIELD_SCAN) {
    const countText = Number.isFinite(scanCount) ? `约 ${scanCount.toLocaleString()} 个` : "过多";
    throw new Error(`当前 λ/P 需要扫描${countText}全视场级次，超过安全上限 ${MAX_FULL_FIELD_SCAN}。请增大波长 λ 或减小周期 P。`);
  }
  const selectedByKey = new Map(orders.map((order) => [order.key, order]));
  const candidateOrders = [];
  for (let n = nMax; n >= nMin; n -= 1) {
    for (let m = mMin; m <= mMax; m += 1) {
      const order = selectedByKey.get(`${m},${n}`) ?? createOrder(m, n);
      if (order.intersectsAir) candidateOrders.push(order);
    }
  }
  const candidateKeys = new Set(candidateOrders.map((order) => order.key));
  const displayOrders = [...candidateOrders, ...orders.filter((order) => !candidateKeys.has(order.key))]
    .sort((a, b) => b.n - a.n || a.m - b.m);

  const fullSpan = parameters.gridSize * delta;
  const fov = fullSpan <= 2 ? 2 * Math.asin(fullSpan / 2) / radians : null;
  return {
    parameters,
    orders,
    candidateOrders,
    displayOrders,
    delta,
    fovX: fov,
    fovY: fov,
    fovCircular: fov,
    propagatingCount: orders.filter((order) => order.propagating).length,
    capturedCount: orders.filter((order) => order.captured).length,
    minimumMargin: Math.min(...orders.map((order) => order.airMargin)),
  };
}

function formatSignedRange(start, size) {
  const end = start + size - 1;
  const sign = (value) => String(value).replace("-", "−");
  return `${sign(start)}…${sign(end)}`;
}

function formatFixed(value, digits = 4) {
  return Number.isFinite(value) ? value.toFixed(digits) : "—";
}

function statusFor(order) {
  if (order.captured) return { key: "captured", label: "NA 内" };
  if (order.propagating) return { key: "propagating", label: "仅传播" };
  return { key: "evanescent", label: "不可传播" };
}

function spectrumStatusFor(order) {
  if (order.fullyInsideAir) return { key: "full", label: "完整入圆" };
  if (order.intersectsAir) return { key: "partial", label: "部分入圆" };
  return { key: "outside", label: "圆外" };
}

function updateSummary(model) {
  const { parameters } = model;
  elements.gridTitle.textContent = `${parameters.gridSize} × ${parameters.gridSize} 连续级次`;
  elements.orderRange.textContent = `m = ${formatSignedRange(parameters.mStart, parameters.gridSize)}，n = ${formatSignedRange(parameters.nStart, parameters.gridSize)}`;
  if (model.fovCircular === null) {
    elements.fovValue.textContent = "不可完整覆盖";
    elements.fovAxis.textContent = "拼接宽度超过传播圆直径";
  } else {
    const fovText = `${model.fovCircular.toFixed(2)}°`;
    elements.fovValue.textContent = fovText;
    elements.fovAxis.textContent = `横向 ${model.fovX.toFixed(2)}° · 纵向 ${model.fovY.toFixed(2)}°`;
  }
  elements.propagatingMetric.textContent = `${model.propagatingCount} / ${parameters.channels}`;
  elements.capturedMetric.textContent = `${model.capturedCount} / ${parameters.channels}`;
  elements.candidateMetric.textContent = String(model.candidateOrders.length);
  elements.deltaMetric.textContent = model.delta.toFixed(4);
  elements.marginMetric.textContent = model.minimumMargin.toFixed(4);
  elements.opacityValue.textContent = `${parameters.opacity}%`;
  elements.canvasPlaceholder.querySelector("span").textContent = String(parameters.channels);
}

function updateTable(model) {
  const rows = document.createDocumentFragment();
  for (const order of model.displayOrders) {
    const status = statusFor(order);
    const spectrum = spectrumStatusFor(order);
    const row = document.createElement("tr");
    row.dataset.orderKey = order.key;
    if (order.key === activeOrderKey) row.classList.add("selected");
    const channel = order.selected ? String(order.index + 1) : "—";
    const selection = order.selected ? `${order.row + 1} 行 ${order.column + 1} 列` : "未选";
    row.innerHTML = `<td>${channel}</td><td>${selection}</td><td>(${order.m}, ${order.n})</td><td>${formatFixed(order.ux)}</td><td>${formatFixed(order.uy)}</td><td>${order.thetaOut === null ? "—" : `${order.thetaOut.toFixed(2)}°`}</td><td>${spectrum.label}</td><td><span class="status ${status.key}">${status.label}</span></td>`;
    row.addEventListener("click", () => selectOrder(order.key, true));
    rows.appendChild(row);
  }
  elements.orderTableBody.replaceChildren(rows);
}

function canvasGeometry(model) {
  const bounds = elements.orderCanvas.getBoundingClientRect();
  const padding = Math.max(48, Math.min(bounds.width, bounds.height) * 0.085);
  const halfCell = model.delta / 2;
  const extent = Math.max(
    1.12,
    ...model.displayOrders.flatMap((order) => [Math.abs(order.ux) + halfCell, Math.abs(order.uy) + halfCell]),
  );
  const scale = Math.min((bounds.width - 2 * padding) / (2 * extent), (bounds.height - 2 * padding) / (2 * extent));
  return { bounds, centerX: bounds.width / 2, centerY: bounds.height / 2, scale, extent };
}

function prepareCanvas() {
  const bounds = elements.orderCanvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  elements.orderCanvas.width = Math.max(1, Math.floor(bounds.width * ratio));
  elements.orderCanvas.height = Math.max(1, Math.floor(bounds.height * ratio));
  plotContext.setTransform(ratio, 0, 0, ratio, 0, 0);
  plotContext.clearRect(0, 0, bounds.width, bounds.height);
  plotContext.fillStyle = "#07121c";
  plotContext.fillRect(0, 0, bounds.width, bounds.height);
}

function drawPlot(model) {
  prepareCanvas();
  const geometry = canvasGeometry(model);
  const { bounds, centerX, centerY, scale } = geometry;
  const point = (x, y) => ({ x: centerX + x * scale, y: centerY - y * scale });
  hitRegions = [];

  plotContext.save();
  plotContext.strokeStyle = "rgba(148, 163, 184, 0.13)";
  plotContext.lineWidth = 1;
  for (let tick = -1; tick <= 1.001; tick += 0.25) {
    const vertical = point(tick, 0);
    const horizontal = point(0, tick);
    plotContext.beginPath();
    plotContext.moveTo(vertical.x, point(0, -1.06).y);
    plotContext.lineTo(vertical.x, point(0, 1.06).y);
    plotContext.stroke();
    plotContext.beginPath();
    plotContext.moveTo(point(-1.06, 0).x, horizontal.y);
    plotContext.lineTo(point(1.06, 0).x, horizontal.y);
    plotContext.stroke();
  }

  plotContext.strokeStyle = "rgba(148, 163, 184, 0.76)";
  plotContext.lineWidth = 1.5;
  plotContext.setLineDash([7, 7]);
  plotContext.beginPath();
  plotContext.arc(centerX, centerY, scale, 0, Math.PI * 2);
  plotContext.stroke();

  const naRadius = model.parameters.na / model.parameters.nOut;
  if (naRadius < 0.9999) {
    plotContext.strokeStyle = "rgba(251, 191, 36, 0.7)";
    plotContext.setLineDash([3, 5]);
    plotContext.beginPath();
    plotContext.arc(centerX, centerY, naRadius * scale, 0, Math.PI * 2);
    plotContext.stroke();
  }
  plotContext.setLineDash([]);

  const cellPixels = model.delta * scale;
  const drawOrder = (order) => {
    const center = point(order.ux, order.uy);
    const left = center.x - cellPixels / 2;
    const top = center.y - cellPixels / 2;
    const status = statusFor(order);
    const color = status.key === "captured" ? "#5eead4" : status.key === "propagating" ? "#fbbf24" : "#fb7185";
    const active = order.key === activeOrderKey;

    plotContext.fillStyle = order.selected
      ? (status.key === "evanescent" ? "rgba(251,113,133,.055)" : "rgba(94,234,212,.045)")
      : (order.fullyInsideAir ? "rgba(113,131,147,.035)" : "rgba(113,131,147,.018)");
    plotContext.fillRect(left, top, cellPixels, cellPixels);

    if (sourceImage && order.selected) {
      const sourceWidth = sourceImage.naturalWidth / model.parameters.gridSize;
      const sourceHeight = sourceImage.naturalHeight / model.parameters.gridSize;
      plotContext.save();
      plotContext.beginPath();
      plotContext.rect(left, top, cellPixels, cellPixels);
      plotContext.clip();
      plotContext.globalAlpha = model.parameters.opacity / 100;
      plotContext.drawImage(
        sourceImage,
        order.column * sourceWidth,
        order.row * sourceHeight,
        sourceWidth,
        sourceHeight,
        left,
        top,
        cellPixels,
        cellPixels,
      );
      plotContext.restore();
    }

    plotContext.strokeStyle = order.selected ? color : (active ? "#fbbf24" : "rgba(113,131,147,.62)");
    plotContext.globalAlpha = active ? 1 : (order.selected ? 0.82 : 0.72);
    plotContext.lineWidth = active ? 3 : (order.selected ? 1.2 : 0.9);
    if (!order.fullyInsideAir) plotContext.setLineDash([4, 4]);
    plotContext.strokeRect(left, top, cellPixels, cellPixels);
    plotContext.setLineDash([]);
    plotContext.globalAlpha = 1;

    plotContext.fillStyle = color;
    plotContext.beginPath();
    plotContext.arc(center.x, center.y, Math.max(2.3, Math.min(3.7, cellPixels * 0.035)), 0, Math.PI * 2);
    plotContext.fill();

    if (cellPixels >= 25) {
      plotContext.font = `${Math.max(9, Math.min(13, cellPixels * 0.1))}px ui-monospace, SFMono-Regular, Consolas, monospace`;
      plotContext.textAlign = "center";
      plotContext.textBaseline = "top";
      plotContext.fillStyle = order.selected ? "rgba(237, 246, 246, .94)" : "rgba(174, 196, 207, .72)";
      plotContext.shadowColor = "rgba(0,0,0,.9)";
      plotContext.shadowBlur = 4;
      plotContext.fillText(`(${order.m},${order.n})`, center.x, top + 6);
      plotContext.shadowBlur = 0;
    }
    hitRegions.push({ left, top, width: cellPixels, height: cellPixels, order });
  };

  for (const order of model.candidateOrders) if (!order.selected) drawOrder(order);
  for (const order of model.orders) drawOrder(order);

  plotContext.fillStyle = "rgba(174, 196, 207, .82)";
  plotContext.font = "12px Inter, Segoe UI, sans-serif";
  plotContext.textAlign = "right";
  plotContext.fillText("uₓ", bounds.width - 16, centerY - 8);
  plotContext.textAlign = "left";
  plotContext.fillText("uᵧ", centerX + 9, 18);
  plotContext.restore();
}

function drawSourcePreview(gridSize) {
  if (!sourceImage) return;
  const ratio = window.devicePixelRatio || 1;
  const cssSize = 76;
  elements.sourceCanvas.width = cssSize * ratio;
  elements.sourceCanvas.height = cssSize * ratio;
  sourceContext.setTransform(ratio, 0, 0, ratio, 0, 0);
  sourceContext.clearRect(0, 0, cssSize, cssSize);
  const imageRatio = sourceImage.naturalWidth / sourceImage.naturalHeight;
  const drawWidth = imageRatio >= 1 ? cssSize : cssSize * imageRatio;
  const drawHeight = imageRatio >= 1 ? cssSize / imageRatio : cssSize;
  const x = (cssSize - drawWidth) / 2;
  const y = (cssSize - drawHeight) / 2;
  sourceContext.drawImage(sourceImage, x, y, drawWidth, drawHeight);
  sourceContext.strokeStyle = "rgba(251,191,36,.9)";
  sourceContext.lineWidth = 0.8;
  for (let index = 1; index < gridSize; index += 1) {
    sourceContext.beginPath();
    sourceContext.moveTo(x + index * drawWidth / gridSize, y);
    sourceContext.lineTo(x + index * drawWidth / gridSize, y + drawHeight);
    sourceContext.stroke();
    sourceContext.beginPath();
    sourceContext.moveTo(x, y + index * drawHeight / gridSize);
    sourceContext.lineTo(x + drawWidth, y + index * drawHeight / gridSize);
    sourceContext.stroke();
  }
}

function showError(message, input = null) {
  elements.errorMessage.textContent = message;
  elements.channels.classList.remove("invalid");
  if (input) input.classList.add("invalid");
}

function clearError() {
  elements.errorMessage.textContent = "";
  document.querySelectorAll(".invalid").forEach((input) => input.classList.remove("invalid"));
}

function recalculate() {
  clearTimeout(recalculateTimer);
  recalculateTimer = null;
  try {
    const parameters = readParameters();
    clearError();
    lastModel = buildModel(parameters);
    if (lastModel.fovCircular === null) {
      elements.errorMessage.textContent = "当前 channel 数与 λ/P 使角谱拼接宽度超过传播圆直径，无法形成完整标称 FoV。";
    }
    if (activeOrderKey !== null && !lastModel.displayOrders.some((order) => order.key === activeOrderKey)) activeOrderKey = null;
    updateSummary(lastModel);
    updateTable(lastModel);
    drawPlot(lastModel);
    drawSourcePreview(parameters.gridSize);
    return lastModel;
  } catch (error) {
    const input = error.message.startsWith("Channel") ? elements.channels : null;
    showError(error.message, input);
    return null;
  }
}

function scheduleRecalculate() {
  clearTimeout(recalculateTimer);
  recalculateTimer = setTimeout(recalculate, PARAMETER_INPUT_DELAY_MS);
}

function selectOrder(orderKey, scrollTable = false) {
  activeOrderKey = orderKey;
  if (lastModel) {
    updateTable(lastModel);
    drawPlot(lastModel);
  }
  if (scrollTable) {
    const row = [...elements.orderTableBody.rows].find((candidate) => candidate.dataset.orderKey === orderKey);
    row?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }
}

function showToast(message) {
  clearTimeout(toastTimer);
  elements.toast.textContent = message;
  elements.toast.classList.add("visible");
  toastTimer = setTimeout(() => elements.toast.classList.remove("visible"), 2200);
}

function centerImageGrid({ showFeedback = true, throwOnError = false } = {}) {
  try {
    const parameters = readParameters();
    const meanM = parameters.mStart + (parameters.gridSize - 1) / 2;
    const meanN = parameters.nStart + (parameters.gridSize - 1) / 2;
    const requiredX = -meanM * parameters.wavelength / parameters.period;
    const requiredY = -meanN * parameters.wavelength / parameters.period;
    const requiredMagnitude = Math.hypot(requiredX, requiredY);
    const sineTheta = requiredMagnitude / parameters.nIn;

    if (sineTheta > 1 + 1e-12) {
      throw new Error(`当前级次需要的入射横向波矢为 ${requiredMagnitude.toFixed(4)}，超过 n_in=${parameters.nIn.toFixed(4)}，无法通过 θ、φ 居中。`);
    }

    const theta = Math.asin(Math.min(1, sineTheta)) * 180 / Math.PI;
    const phi = requiredMagnitude < 1e-12 ? 0 : Math.atan2(requiredY, requiredX) * 180 / Math.PI;
    const thetaText = theta.toFixed(4);
    const phiText = phi.toFixed(4);
    elements.thetaRange.value = thetaText;
    elements.thetaNumber.value = thetaText;
    elements.phiRange.value = phiText;
    elements.phiNumber.value = phiText;

    const model = recalculate();
    if (!model) throw new Error(elements.errorMessage.textContent || "自动居中失败。");
    if (showFeedback) showToast(`图像已居中：θ ${theta.toFixed(2)}°，φ ${phi.toFixed(2)}°`);
    return { theta, phi, centerUx: 0, centerUy: 0, model };
  } catch (error) {
    showError(error.message);
    if (showFeedback) showToast("当前参数无法自动居中");
    if (throwOnError) throw error;
    return null;
  }
}

function loadFile(file) {
  if (!file) return;
  if (!file.type.startsWith("image/") && !/\.(png|jpe?g|webp|gif|bmp|svg)$/i.test(file.name)) {
    showError("请选择浏览器可读取的图片文件。支持 PNG、JPG、WEBP、GIF、BMP 和 SVG。");
    return;
  }
  if (sourceObjectUrl) URL.revokeObjectURL(sourceObjectUrl);
  sourceObjectUrl = URL.createObjectURL(file);
  const image = new Image();
  image.onload = () => {
    sourceImage = image;
    elements.uploadTitle.textContent = "更换图片";
    elements.fileName.textContent = file.name;
    elements.imageSize.textContent = `${image.naturalWidth} × ${image.naturalHeight} px`;
    elements.sourcePreview.hidden = false;
    elements.canvasPlaceholder.hidden = true;
    clearError();
    recalculate();
    showToast("图片已完成本地分割");
  };
  image.onerror = () => showError("浏览器无法解码这张图片，请转换为 PNG、JPG 或 WEBP 后重试。");
  image.src = sourceObjectUrl;
}

function setDefaults() {
  elements.wavelength.value = DEFAULTS.wavelength;
  elements.period.value = DEFAULTS.period;
  elements.channels.value = DEFAULTS.channels;
  elements.mStart.value = DEFAULTS.mStart;
  elements.nStart.value = DEFAULTS.nStart;
  elements.thetaRange.value = DEFAULTS.theta;
  elements.thetaNumber.value = DEFAULTS.theta;
  elements.phiRange.value = DEFAULTS.phi;
  elements.phiNumber.value = DEFAULTS.phi;
  elements.nIn.value = DEFAULTS.nIn;
  elements.nOut.value = DEFAULTS.nOut;
  elements.na.value = DEFAULTS.na;
  elements.imageOpacity.value = DEFAULTS.opacity;
  activeOrderKey = null;
  recalculate();
  showToast("已恢复 3×3 默认参数");
}

function exportPlot() {
  if (!lastModel) return;
  const link = document.createElement("a");
  link.download = `diffraction-orders-${lastModel.parameters.gridSize}x${lastModel.parameters.gridSize}.png`;
  link.href = elements.orderCanvas.toDataURL("image/png");
  link.click();
  showToast("可视化图片已导出");
}

function syncAngle(source, target) {
  target.value = source.value;
  recalculate();
}

elements.imageInput.addEventListener("change", (event) => loadFile(event.target.files[0]));
for (const eventName of ["dragenter", "dragover"]) {
  elements.uploadBox.addEventListener(eventName, (event) => {
    event.preventDefault();
    elements.uploadBox.classList.add("dragging");
  });
}
for (const eventName of ["dragleave", "drop"]) {
  elements.uploadBox.addEventListener(eventName, (event) => {
    event.preventDefault();
    elements.uploadBox.classList.remove("dragging");
  });
}
elements.uploadBox.addEventListener("drop", (event) => loadFile(event.dataTransfer.files[0]));

elements.thetaRange.addEventListener("input", () => syncAngle(elements.thetaRange, elements.thetaNumber));
elements.thetaNumber.addEventListener("input", () => syncAngle(elements.thetaNumber, elements.thetaRange));
elements.phiRange.addEventListener("input", () => syncAngle(elements.phiRange, elements.phiNumber));
elements.phiNumber.addEventListener("input", () => syncAngle(elements.phiNumber, elements.phiRange));

for (const input of [elements.wavelength, elements.period, elements.mStart, elements.nStart, elements.nIn, elements.nOut, elements.na, elements.imageOpacity]) {
  input.addEventListener("input", scheduleRecalculate);
  input.addEventListener("change", recalculate);
}
elements.channels.addEventListener("change", recalculate);
elements.resetButton.addEventListener("click", setDefaults);
elements.centerButton.addEventListener("click", () => centerImageGrid());
elements.exportButton.addEventListener("click", exportPlot);

elements.orderCanvas.addEventListener("mousemove", (event) => {
  const bounds = elements.orderCanvas.getBoundingClientRect();
  const x = event.clientX - bounds.left;
  const y = event.clientY - bounds.top;
  const region = [...hitRegions].reverse().find((item) => x >= item.left && x <= item.left + item.width && y >= item.top && y <= item.top + item.height);
  if (!region) {
    elements.plotTooltip.hidden = true;
    return;
  }
  const status = statusFor(region.order);
  const spectrum = spectrumStatusFor(region.order);
  const selection = region.order.selected
    ? `Ch. ${region.order.index + 1} · 切片 ${region.order.row + 1} 行 ${region.order.column + 1} 列`
    : "未选择级次";
  elements.plotTooltip.innerHTML = `<strong>(${region.order.m}, ${region.order.n}) · ${selection}</strong><br>u = (${region.order.ux.toFixed(4)}, ${region.order.uy.toFixed(4)})<br>${spectrum.label} · 中心${status.label}`;
  elements.plotTooltip.hidden = false;
  elements.plotTooltip.style.left = `${Math.min(x + 14, bounds.width - 220)}px`;
  elements.plotTooltip.style.top = `${Math.min(y + 14, bounds.height - 110)}px`;
});
elements.orderCanvas.addEventListener("mouseleave", () => { elements.plotTooltip.hidden = true; });
elements.orderCanvas.addEventListener("click", (event) => {
  const bounds = elements.orderCanvas.getBoundingClientRect();
  const x = event.clientX - bounds.left;
  const y = event.clientY - bounds.top;
  const region = [...hitRegions].reverse().find((item) => x >= item.left && x <= item.left + item.width && y >= item.top && y <= item.top + item.height);
  if (region) selectOrder(region.order.key, true);
});

window.addEventListener("resize", () => { if (lastModel) drawPlot(lastModel); });

function registerWebMcpTools() {
  const context = document.modelContext;
  if (!context?.registerTool) return;
  const numericProperties = {
    wavelength: { type: "number", exclusiveMinimum: 0 },
    period: { type: "number", exclusiveMinimum: 0 },
    channels: { type: "integer", enum: CHANNEL_OPTIONS },
    mStart: { type: "integer" },
    nStart: { type: "integer" },
    theta: { type: "number", minimum: 0, exclusiveMaximum: 90 },
    phi: { type: "number", minimum: -180, maximum: 180 },
    nIn: { type: "number", exclusiveMinimum: 0 },
    nOut: { type: "number", exclusiveMinimum: 0 },
    na: { type: "number", exclusiveMinimum: 0 },
  };
  try {
    void Promise.resolve(context.registerTool({
      name: "configure_diffraction_view",
      title: "配置级次视图",
      description: "修改当前衍射级次可视化的物理参数并立即重新计算。Channel 数必须是完全平方数。",
      inputSchema: { type: "object", properties: numericProperties, additionalProperties: false },
      annotations: { readOnlyHint: false, untrustedContentHint: false },
      execute(input) {
        const mapping = { wavelength: elements.wavelength, period: elements.period, channels: elements.channels, mStart: elements.mStart, nStart: elements.nStart, nIn: elements.nIn, nOut: elements.nOut, na: elements.na };
        for (const [key, control] of Object.entries(mapping)) if (input[key] !== undefined) control.value = input[key];
        if (input.theta !== undefined) { elements.thetaNumber.value = input.theta; elements.thetaRange.value = input.theta; }
        if (input.phi !== undefined) { elements.phiNumber.value = input.phi; elements.phiRange.value = input.phi; }
        const model = recalculate();
        if (!model) throw new Error(elements.errorMessage.textContent || "参数无效");
        return { fovDegrees: model.fovCircular, propagating: model.propagatingCount, capturedByNa: model.capturedCount, channels: model.parameters.channels, fullFieldCandidates: model.candidateOrders.length };
      },
    }));
    void Promise.resolve(context.registerTool({
      name: "center_diffraction_view",
      title: "自动居中图像",
      description: "根据当前波长、周期、channel 数和连续级次范围反算共同入射角 θ、φ，使整组图片方格中心落在传播圆中心。",
      inputSchema: { type: "object", properties: {}, additionalProperties: false },
      annotations: { readOnlyHint: false, untrustedContentHint: false },
      execute() {
        const centered = centerImageGrid({ showFeedback: false, throwOnError: true });
        return { theta: centered.theta, phi: centered.phi, centerUx: centered.centerUx, centerUy: centered.centerUy };
      },
    }));
    void Promise.resolve(context.registerTool({
      name: "read_diffraction_summary",
      title: "读取级次结果",
      description: "读取当前级次视图的 FoV、传播数、NA 收集数、全视场候选级次数和级次间距。",
      inputSchema: { type: "object", properties: {}, additionalProperties: false },
      annotations: { readOnlyHint: true, untrustedContentHint: false },
      execute() {
        if (!lastModel) throw new Error("当前没有有效计算结果");
        const centerUx = lastModel.orders.reduce((sum, order) => sum + order.ux, 0) / lastModel.orders.length;
        const centerUy = lastModel.orders.reduce((sum, order) => sum + order.uy, 0) / lastModel.orders.length;
        return { fovDegrees: lastModel.fovCircular, propagating: lastModel.propagatingCount, capturedByNa: lastModel.capturedCount, fullFieldCandidates: lastModel.candidateOrders.length, channelCount: lastModel.parameters.channels, deltaU: lastModel.delta, minimumAirMargin: lastModel.minimumMargin, theta: lastModel.parameters.theta, phi: lastModel.parameters.phi, centerUx, centerUy };
      },
    }));
  } catch (error) {
    console.warn("WebMCP registration failed", error);
  }
}

recalculate();
registerWebMcpTools();
