"use strict";

// Dome panoramas use equal angular spacing; diffraction images retain direction cosines.
// Both are displayed on the same overhead hemisphere with unit RGB gain.
const $ = (id) => document.getElementById(id);
const canvas = $("sceneCanvas");
const sourceCanvas = $("sourceCanvas");
const sourceContext = sourceCanvas.getContext("2d");
const state = {
  mode: "inside", yaw: 0, pitch: 26, distance: 3.2, insideFov: 60,
  projection: "angular",
  coverage: 180, grid: false, autoRotate: false, speed: 8,
  image: null, name: "", centerX: 0, centerY: 0, radius: 1,
};
const radians = (degrees) => degrees * Math.PI / 180;
const clamp = (value, low, high) => Math.max(low, Math.min(high, value));
const wrapAngle = (value) => ((value + 180) % 360 + 360) % 360 - 180;
let gl, program, texture, uniforms;
let needsRender = true;
let sourceLayout = null;
let loadSequence = 0;
let lastFrame = null;
let drag = null;
const pointers = new Map();
let pinchDistance = 0;
let expanded = false;

function setExpanded(value) {
  expanded = value;
  $("scene").classList[value ? "add" : "remove"]("expanded");
  $("expandView").textContent = value ? "退出大屏" : "大屏观赏";
  $("expandView").setAttribute("aria-pressed", String(value));
  needsRender = true;
}

const vertexSource = `
attribute vec2 aPosition;
void main() { gl_Position = vec4(aPosition, 0.0, 1.0); }
`;

const fragmentSource = `
precision highp float;
uniform vec2 uResolution;
uniform vec3 uEye, uForward, uRight, uUp;
uniform float uTanFov, uHalfFov, uRimY, uDiskScale;
uniform vec2 uImageSize, uCenter;
uniform float uRadius;
uniform bool uHasImage, uGrid, uInside, uFisheye, uAngular;
uniform sampler2D uImage;
const float PI = 3.14159265359;

float gridAmount(vec3 p) {
  float latitude = asin(clamp(p.y, -1.0, 1.0));
  float longitude = atan(p.x, p.z);
  float latitudeLine = abs(sin(latitude * 12.0));
  float longitudeLine = abs(sin(longitude * 12.0)) * sqrt(max(0.0, 1.0-p.y*p.y));
  float line = 1.0-smoothstep(0.008, 0.026, min(latitudeLine, longitudeLine));
  return line;
}

void main() {
  vec2 screen = (gl_FragCoord.xy / uResolution - 0.5) * 2.0;
  screen.x *= uResolution.x / uResolution.y;
  vec3 ray = normalize(uForward + uRight * screen.x * uTanFov + uUp * screen.y * uTanFov);
  if (uFisheye) {
    // An equidistant 180-degree lens shows the entire dome from its base center.
    screen = (2.0*gl_FragCoord.xy-uResolution) / min(uResolution.x,uResolution.y) * 1.12;
    float radius = length(screen);
    if (radius > 1.0) { gl_FragColor = vec4(0.018,0.023,0.030,1.0); return; }
    vec3 lateral = radius > 0.00001 ? (uRight*screen.x+uUp*screen.y)/radius : uRight;
    ray = uForward*cos(radius*uHalfFov) + lateral*sin(radius*uHalfFov);
  }
  float glow = max(0.0, 1.0-length(screen*vec2(0.45, 0.5)));
  vec3 background = mix(vec3(0.034,0.060,0.089), vec3(0.069,0.116,0.147), glow);
  vec3 color = background;

  // A quiet reference floor helps judge orbiting geometry without shading RGB.
  if (!uInside && ray.y < -0.0001) {
    float planeT = (-0.04-uEye.y) / ray.y;
    if (planeT > 0.0) {
      vec3 floorPoint = uEye + ray * planeT;
      float attenuation = exp(-0.24*length(floorPoint.xz)) * 0.18;
      vec2 lines = abs(fract(floorPoint.xz*2.0+0.5)-0.5);
      float grid = 1.0-smoothstep(0.004,0.013,min(lines.x,lines.y));
      color += vec3(0.18,0.29,0.30)*grid*attenuation;
    }
  }
  float b = dot(uEye, ray);
  float discriminant = b*b - dot(uEye,uEye) + 1.0;
  if (uHasImage && discriminant >= 0.0) {
    float root = sqrt(discriminant);
    float tNear = -b-root;
    float tFar = -b+root;
    vec3 pointNear = uEye+ray*tNear;
    vec3 pointFar = uEye+ray*tFar;
    bool nearValid = tNear > 0.0 && pointNear.y >= uRimY-0.00001;
    bool farValid = tFar > 0.0 && pointFar.y >= uRimY-0.00001;
    if (nearValid || farValid) {
      vec3 p = nearValid ? pointNear : pointFar;
      // Angular textures preserve the finite vertical scale at the horizon.
      vec2 disk = vec2(p.x,p.z) / uDiskScale;
      if (uAngular) {
        float horizontal = length(p.xz);
        float polar = acos(clamp(p.y,-1.0,1.0));
        float halfCoverage = acos(clamp(uRimY,-1.0,1.0));
        disk = horizontal > 0.00001 ? vec2(p.x,p.z)/horizontal * (polar/halfCoverage) : vec2(0.0);
      }
      vec2 pixel = uCenter + disk * uRadius;
      vec2 uv = (pixel+0.5) / uImageSize;
      if (uv.x >= 0.0 && uv.x <= 1.0 && uv.y >= 0.0 && uv.y <= 1.0) {
        vec4 texel = texture2D(uImage,uv);
        color = mix(vec3(0.018,0.023,0.030), texel.rgb, texel.a);
      } else {
        // A source circle extending beyond the image has missing data.
        float hatch = step(0.5,fract((pixel.x+pixel.y)/18.0));
        color = mix(vec3(0.16,0.19,0.21),vec3(0.22,0.26,0.28),hatch);
      }
      if (uGrid) {
        color = mix(color,vec3(0.46,0.75,0.68),gridAmount(p)*0.28);
        float rim = 1.0-smoothstep(0.0,0.010,abs(p.y-uRimY));
        color = mix(color,vec3(0.54,0.82,0.73),rim*0.65);
      }
    } else if (!uInside && uGrid) {
      // Show the otherwise empty sphere as a faint wire reference.
      vec3 p = tNear > 0.0 ? pointNear : pointFar;
      color = mix(color,vec3(0.21,0.35,0.36),gridAmount(p)*0.25);
    }
  }
  gl_FragColor = vec4(color,1.0);
}
`;

function compileShader(type, code) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, code);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const detail = gl.getShaderInfoLog(shader);
    gl.deleteShader(shader);
    throw new Error(`球面渲染程序编译失败：${detail}`);
  }
  return shader;
}

function initializeRenderer() {
  gl = canvas.getContext("webgl", { alpha: false, preserveDrawingBuffer: true });
  if (!gl) throw new Error("浏览器未能启用 WebGL。请开启浏览器硬件加速后重新打开页面。");
  const vertexShader = compileShader(gl.VERTEX_SHADER, vertexSource);
  const fragmentShader = compileShader(gl.FRAGMENT_SHADER, fragmentSource);
  program = gl.createProgram();
  gl.attachShader(program, vertexShader);
  gl.attachShader(program, fragmentShader);
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    throw new Error(`球面渲染程序链接失败：${gl.getProgramInfoLog(program)}`);
  }
  gl.deleteShader(vertexShader);
  gl.deleteShader(fragmentShader);
  gl.useProgram(program);
  const quad = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, quad);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1, 1,-1, -1,1, -1,1, 1,-1, 1,1]), gl.STATIC_DRAW);
  const position = gl.getAttribLocation(program, "aPosition");
  gl.enableVertexAttribArray(position);
  gl.vertexAttribPointer(position, 2, gl.FLOAT, false, 0, 0);
  const names = ["Resolution", "Eye", "Forward", "Right", "Up", "TanFov", "HalfFov", "RimY", "DiskScale", "ImageSize", "Center", "Radius", "HasImage", "Grid", "Inside", "Fisheye", "Angular", "Image"];
  uniforms = Object.fromEntries(names.map((name) => [name, gl.getUniformLocation(program, `u${name}`)]));
  texture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.pixelStorei(gl.UNPACK_COLORSPACE_CONVERSION_WEBGL, gl.NONE);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([0,0,0,255]));
  gl.uniform1i(uniforms.Image, 0);
}

function updateReadouts() {
  $("coverageValue").textContent = `${state.coverage}°`;
  $("fieldReadout").textContent = `${state.coverage}°${state.coverage === 180 ? " · 完整穹顶" : " · 球冠"}`;
  $("viewReadout").textContent = `方位 ${Math.round(state.yaw)}° · 仰角 ${Math.round(state.pitch)}°`;
  $("zoomValue").textContent = `${(state.mode === "outside" ? 3.2 / state.distance : (state.mode === "overview" ? 180 : 60) / state.insideFov).toFixed(1)}×`;
  $("modeLabel").textContent = state.mode === "overview" ? "底面圆心 · 完整穹顶" : state.mode === "inside" ? "底面圆心 · 沉浸环顾" : "穹顶外观察";
  $("projectionLabel").textContent = `${state.projection === "angular" ? "穹顶全景" : "衍射圆图"} / ${state.coverage}°`;
  $("outsideButton").setAttribute("aria-pressed", String(state.mode === "outside"));
  $("insideButton").setAttribute("aria-pressed", String(state.mode === "inside"));
  $("overviewButton").setAttribute("aria-pressed", String(state.mode === "overview"));
  $("projection").value = state.projection;
  $("viewDescription").textContent = state.mode === "overview" ? "从圆心仰望完整穹顶，全景广角显示周围建筑与头顶校徽。"
    : state.mode === "inside" ? "站在底面圆心，用常规透视环顾建筑；拖动抬头可看校徽。"
    : "拖动旋转，观察穹顶的外部形状。";
  $("interactionHint").textContent = state.mode === "outside" ? "拖动旋转 · 滚轮缩放"
    : "拖动转头 · 滚轮放大 · 头顶查看校徽";
  for (const id of ["frontView", "obliqueView", "sideView"]) $(id).classList.remove("active");
  if (Math.abs(state.pitch-90) < 1 && Math.abs(state.yaw) < 1) $("frontView").classList.add("active");
  if (Math.abs(state.pitch-35) < 1 && Math.abs(state.yaw-40) < 1) $("obliqueView").classList.add("active");
  if (Math.abs(state.pitch) < 1 && Math.abs(state.yaw-90) < 1) $("sideView").classList.add("active");
  const lookY = Math.sin(radians(state.pitch));
  const emptyDirection = state.mode !== "outside" && state.image && lookY < Math.cos(radians(state.coverage/2))-1e-5;
  $("sceneNotice").hidden = !emptyDirection;
  $("sceneNotice").textContent = "当前视线朝向穹顶覆盖之外；底面以下没有图像。";
}

function renderScene() {
  if (!gl || gl.isContextLost()) return;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.round(canvas.clientWidth*dpr);
  const height = Math.round(canvas.clientHeight*dpr);
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  gl.viewport(0,0,width,height);
  const yaw = radians(state.yaw), pitch = radians(state.pitch);
  const direction = [Math.sin(yaw)*Math.cos(pitch), Math.sin(pitch), Math.cos(yaw)*Math.cos(pitch)];
  const inside = state.mode !== "outside";
  const halfField = radians(state.coverage/2);
  gl.uniform2f(uniforms.Resolution, width, height);
  gl.uniform3fv(uniforms.Eye, inside ? [0,0,0] : direction.map((x) => x*state.distance));
  gl.uniform3fv(uniforms.Forward, inside ? direction : direction.map((x) => -x));
  gl.uniform3f(uniforms.Right, Math.cos(yaw), 0, -Math.sin(yaw));
  gl.uniform3f(uniforms.Up, -Math.sin(yaw)*Math.sin(pitch), Math.cos(pitch), -Math.cos(yaw)*Math.sin(pitch));
  gl.uniform1f(uniforms.TanFov, Math.tan(radians((state.mode === "inside" ? state.insideFov : 42)/2)));
  gl.uniform1f(uniforms.HalfFov, radians(state.insideFov/2));
  gl.uniform1f(uniforms.RimY, Math.cos(halfField));
  gl.uniform1f(uniforms.DiskScale, Math.sin(halfField));
  gl.uniform2f(uniforms.ImageSize, state.image ? state.image.width : 1, state.image ? state.image.height : 1);
  gl.uniform2f(uniforms.Center, state.centerX, state.centerY);
  gl.uniform1f(uniforms.Radius, state.radius);
  gl.uniform1i(uniforms.HasImage, state.image !== null);
  gl.uniform1i(uniforms.Grid, state.grid);
  gl.uniform1i(uniforms.Inside, inside);
  gl.uniform1i(uniforms.Fisheye, state.mode === "overview");
  gl.uniform1i(uniforms.Angular, state.projection === "angular");
  gl.drawArrays(gl.TRIANGLES, 0, 6);
  updateReadouts();
  needsRender = false;
}

function drawSource() {
  const rect = sourceCanvas.getBoundingClientRect();
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  sourceCanvas.width = Math.max(1, Math.round(rect.width*dpr));
  sourceCanvas.height = Math.max(1, Math.round(rect.height*dpr));
  sourceContext.setTransform(dpr,0,0,dpr,0,0);
  sourceContext.clearRect(0,0,rect.width,rect.height);
  if (!state.image) return;
  const scale = Math.min((rect.width-20)/state.image.width,(rect.height-16)/state.image.height);
  const x = (rect.width-state.image.width*scale)/2;
  const y = (rect.height-state.image.height*scale)/2;
  sourceLayout = {x,y,scale};
  sourceContext.drawImage(state.image,x,y,state.image.width*scale,state.image.height*scale);
  sourceContext.strokeStyle = "#90d6b1";
  sourceContext.lineWidth = 1.2;
  sourceContext.setLineDash([5,4]);
  sourceContext.beginPath();
  sourceContext.arc(x+state.centerX*scale,y+state.centerY*scale,state.radius*scale,0,Math.PI*2);
  sourceContext.stroke();
  sourceContext.setLineDash([]);
  const cx = x+state.centerX*scale, cy = y+state.centerY*scale;
  sourceContext.beginPath();
  sourceContext.moveTo(cx-5,cy); sourceContext.lineTo(cx+5,cy);
  sourceContext.moveTo(cx,cy-5); sourceContext.lineTo(cx,cy+5);
  sourceContext.stroke();
}

function updateCircleFields() {
  $("centerX").value = state.centerX.toFixed(1);
  $("centerY").value = state.centerY.toFixed(1);
  $("sourceRadius").value = state.radius.toFixed(1);
  const missing = state.image && (state.centerX-state.radius < -0.5 || state.centerY-state.radius < -0.5
    || state.centerX+state.radius > state.image.width-0.5 || state.centerY+state.radius > state.image.height-0.5);
  $("sourceCaption").textContent = missing
    ? "所选圆超出原图；缺失部分在球面上以斜纹标出。"
    : "绿色虚线为映射范围；原图亮度保持 1×。点击可设置圆心。";
  drawSource();
  needsRender = true;
}

function fitCircle() {
  if (!state.image) return;
  state.centerX = (state.image.width-1)/2;
  state.centerY = (state.image.height-1)/2;
  state.radius = Math.min(state.image.width,state.image.height)/2;
  updateCircleFields();
}

function installImage(image, name, projection = state.projection) {
  const maxTextureSize = gl.getParameter(gl.MAX_TEXTURE_SIZE);
  if (image.width > maxTextureSize || image.height > maxTextureSize) {
    throw new Error(`图像 ${image.width}×${image.height} 超出浏览器纹理上限 ${maxTextureSize} 像素。请先缩小图片。`);
  }
  gl.bindTexture(gl.TEXTURE_2D,texture);
  gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,gl.RGBA,gl.UNSIGNED_BYTE,image);
  const code = gl.getError();
  if (code !== gl.NO_ERROR) throw new Error(`图像载入 GPU 失败（WebGL ${code}）。`);
  state.image = image;
  state.name = name;
  state.projection = projection;
  $("fileName").textContent = name;
  $("imageSize").textContent = `${image.width} × ${image.height}`;
  $("emptyState").hidden = true;
  $("sourcePlaceholder").hidden = true;
  $("errorMessage").textContent = "";
  $("exportButton").disabled = false;
  fitCircle();
  resetView();
}

async function loadFile(file) {
  if (!file || !gl || gl.isContextLost()) return;
  if (!/^image\/(png|jpeg|webp|bmp|x-ms-bmp)$/i.test(file.type) && !/\.(png|jpe?g|webp|bmp)$/i.test(file.name)) {
    $("errorMessage").textContent = "请选择 PNG、JPG、WEBP 或 BMP 图片。";
    return;
  }
  const url = URL.createObjectURL(file);
  try { await loadImageUrl(url,file.name); }
  finally { URL.revokeObjectURL(url); }
}

async function loadImageUrl(url,name,projection = state.projection) {
  if (!gl || gl.isContextLost()) return;
  const sequence = ++loadSequence;
  const image = new Image();
  try {
    image.src = url;
    await image.decode();
    if (sequence === loadSequence) installImage(image,name,projection);
  } catch (error) {
    if (sequence === loadSequence) $("errorMessage").textContent = `无法读取图片：${error.message}`;
  }
}

function loadDemo() {
  if (!gl || gl.isContextLost()) return;
  ++loadSequence;
  const demo = document.createElement("canvas");
  demo.width = demo.height = 1000;
  const context = demo.getContext("2d");
  context.fillStyle = "#080e14";
  context.fillRect(0,0,1000,1000);
  context.save();
  context.beginPath(); context.arc(499.5,499.5,500,0,2*Math.PI); context.clip();
  const gradient = context.createRadialGradient(350,330,10,500,500,650);
  gradient.addColorStop(0,"#6eac9b"); gradient.addColorStop(.5,"#245d61"); gradient.addColorStop(1,"#142b40");
  context.fillStyle = gradient;
  context.fillRect(0,0,1000,1000);
  context.strokeStyle = "#c5e1cb50"; context.lineWidth = 1.5;
  for (let p=0;p<=1000;p+=100) {
    context.beginPath(); context.moveTo(p,0); context.lineTo(p,1000); context.stroke();
    context.beginPath(); context.moveTo(0,p); context.lineTo(1000,p); context.stroke();
  }
  for (const radius of [160,320,480]) { context.beginPath(); context.arc(499.5,499.5,radius,0,Math.PI*2); context.stroke(); }
  context.textAlign = "center"; context.textBaseline = "middle";
  context.fillStyle = "#e8f4de";
  context.font = "500 72px 'Segoe UI', sans-serif";
  context.fillText("SPHERE",500,474);
  context.font = "18px 'Segoe UI', sans-serif";
  context.fillText("DIRECTION COSINE PROJECTION",500,536);
  context.font = "30px 'Microsoft YaHei', sans-serif";
  for (const [label,x,y] of [["上 / +Y",500,114],["下 / −Y",500,882],["左 / −X",124,500],["右 / +X",872,500]]) {
    context.fillText(label,x,y);
  }
  context.fillStyle = "#c2d994";
  for (const [x,y,size] of [[290,264,35],[720,294,20],[734,746,29],[275,738,18]]) {
    context.beginPath(); context.arc(x,y,size,0,Math.PI*2); context.fill();
  }
  context.restore();
  try { installImage(demo,"坐标示例 · 非实验数据","direction"); }
  catch (error) { $("errorMessage").textContent = error.message; }
}

function resetView() {
  state.yaw = 0;
  state.pitch = state.mode === "overview" ? 90 : state.mode === "inside" ? 26 : 35;
  state.distance = 3.2;
  state.insideFov = state.mode === "overview" ? 180 : 60;
  state.autoRotate = false;
  $("autoRotate").checked = false;
  needsRender = true;
}

function setMode(mode) {
  state.mode = mode;
  resetView();
}

function zoom(factor) {
  if (state.mode === "outside") state.distance = clamp(state.distance*factor,1.2,8);
  else state.insideFov = clamp(state.insideFov*factor,30,state.mode === "overview" ? 180 : 100);
  needsRender = true;
}

function turn(dx,dy) {
  state.yaw = wrapAngle(state.yaw+dx);
  state.pitch = clamp(state.pitch+dy,-85,90);
  needsRender = true;
}

function exportView() {
  if (!state.image || !gl || gl.isContextLost()) return;
  renderScene();
  const output = document.createElement("canvas");
  output.width = canvas.width;
  output.height = canvas.height+72;
  const context = output.getContext("2d");
  context.fillStyle = "#0e1b24"; context.fillRect(0,0,output.width,output.height);
  context.drawImage(canvas,0,0);
  context.fillStyle = "#a5c3bd";
  context.font = "15px 'Microsoft YaHei', sans-serif";
  context.fillText(`穹顶展示 · ${$("modeLabel").textContent} · 覆盖 ${state.coverage}° · 原图亮度 1×${state.grid ? " · 含参考线" : ""}`,20,canvas.height+28);
  context.fillStyle = "#708e96"; context.font = "12px 'Microsoft YaHei', sans-serif";
  context.fillText(`${$("projectionLabel").textContent} · ${state.mode === "overview" ? "等距鱼眼" : "透视"} · 方位 ${state.yaw.toFixed(1)}° / 仰角 ${state.pitch.toFixed(1)}° · 仅几何展示`,20,canvas.height+51);
  output.toBlob((blob) => {
    if (!blob) { $("errorMessage").textContent = "浏览器未能生成 PNG 文件。"; return; }
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    const stamp = new Date().toISOString().replace(/[:.]/g,"-");
    const stem = state.name.replace(/\.[^.]+$/,"").replace(/[<>:"/\\|?*]/g,"_");
    link.href = url;
    link.download = `${stem}_球面_${state.mode}_${Math.round(state.yaw)}deg_${stamp}.png`;
    link.click();
    setTimeout(() => URL.revokeObjectURL(url),1000);
  },"image/png");
}

$("imageInput").addEventListener("change",(event) => {
  loadFile(event.target.files[0]);
  event.target.value = "";
});
const uploadBox = $("uploadBox");
for (const name of ["dragenter","dragover"]) uploadBox.addEventListener(name,(event) => {
  event.preventDefault(); uploadBox.classList.add("drag-over");
});
for (const name of ["dragleave","drop"]) uploadBox.addEventListener(name,(event) => {
  event.preventDefault(); uploadBox.classList.remove("drag-over");
  if (name === "drop") loadFile(event.dataTransfer.files[0]);
});
// Dropping outside the input should not navigate away from the tool.
window.addEventListener("dragover",(event) => event.preventDefault());
window.addEventListener("drop",(event) => event.preventDefault());
$("demoButton").addEventListener("click",loadDemo);
$("campusButton").addEventListener("click",() => { setMode("inside"); loadImageUrl(campusImage.src,campusImage.name,"angular"); });
$("emptyDemo").addEventListener("click",loadDemo);
$("outsideButton").addEventListener("click",() => setMode("outside"));
$("insideButton").addEventListener("click",() => setMode("inside"));
$("overviewButton").addEventListener("click",() => setMode("overview"));
$("expandView").addEventListener("click",() => setExpanded(!expanded));
document.addEventListener("keydown",(event) => { if (event.key === "Escape") setExpanded(false); });
$("projection").addEventListener("change",(event) => { state.projection = event.target.value; needsRender = true; });
$("resetView").addEventListener("click",resetView);
$("fitCircle").addEventListener("click",fitCircle);
$("exportButton").addEventListener("click",exportView);
$("zoomIn").addEventListener("click",() => zoom(0.88));
$("zoomOut").addEventListener("click",() => zoom(1/0.88));
$("coverage").addEventListener("input",(event) => { state.coverage = Number(event.target.value); needsRender = true; });
$("showGrid").addEventListener("change",(event) => { state.grid = event.target.checked; needsRender = true; });
$("autoRotate").addEventListener("change",(event) => { state.autoRotate = event.target.checked; needsRender = true; });
$("rotationSpeed").addEventListener("input",(event) => {
  state.speed = Number(event.target.value);
  $("speedValue").textContent = `${state.speed}° / 秒`;
});
for (const id of ["centerX","centerY","sourceRadius"]) $(id).addEventListener("change",(event) => {
  if (!state.image) return;
  const value = Number(event.target.value);
  const valid = event.target.value !== "" && Number.isFinite(value) && (id !== "sourceRadius" || value > 0);
  if (!valid) { $("errorMessage").textContent = "圆心必须为有效数值，圆半径必须大于 0。"; updateCircleFields(); return; }
  state[id === "sourceRadius" ? "radius" : id] = value;
  $("errorMessage").textContent = "";
  updateCircleFields();
});
sourceCanvas.addEventListener("click",(event) => {
  if (!state.image || !sourceLayout) return;
  const rect = sourceCanvas.getBoundingClientRect();
  const {x,y,scale} = sourceLayout;
  const px = (event.clientX-rect.left-x)/scale;
  const py = (event.clientY-rect.top-y)/scale;
  if (px < 0 || py < 0 || px >= state.image.width || py >= state.image.height) return;
  state.centerX = px; state.centerY = py;
  $("sourceSettings").open = true;
  updateCircleFields();
});
for (const [id,yaw,pitch] of [["frontView",0,90],["obliqueView",40,35],["sideView",90,0]]) $(id).addEventListener("click",() => {
  resetView(); state.yaw = yaw; state.pitch = pitch; needsRender = true;
});

canvas.addEventListener("pointerdown",(event) => {
  if (event.button !== 0) return;
  canvas.setPointerCapture(event.pointerId);
  canvas.focus({preventScroll:true});
  pointers.set(event.pointerId,[event.clientX,event.clientY]);
  drag = [event.clientX,event.clientY];
  canvas.classList.add("dragging");
  state.autoRotate = false; $("autoRotate").checked = false;
  if (pointers.size === 2) {
    const [a,b] = [...pointers.values()]; pinchDistance = Math.hypot(a[0]-b[0],a[1]-b[1]);
  }
});
canvas.addEventListener("pointermove",(event) => {
  if (!pointers.has(event.pointerId)) return;
  pointers.set(event.pointerId,[event.clientX,event.clientY]);
  if (pointers.size === 2) {
    const [a,b] = [...pointers.values()];
    const distance = Math.hypot(a[0]-b[0],a[1]-b[1]);
    if (distance > 0 && pinchDistance > 0) zoom(pinchDistance/distance);
    pinchDistance = distance;
  } else if (pointers.size === 1 && drag) {
    const sign = state.mode === "outside" ? 1 : -1;
    turn((event.clientX-drag[0])*0.25*sign,(event.clientY-drag[1])*0.25);
  }
  drag = [event.clientX,event.clientY];
});
function endPointer(event) {
  pointers.delete(event.pointerId);
  drag = pointers.size === 1 ? [...pointers.values()][0] : null;
  pinchDistance = 0;
  if (!pointers.size) canvas.classList.remove("dragging");
}
for (const name of ["pointerup","pointercancel","lostpointercapture"]) canvas.addEventListener(name,endPointer);
canvas.addEventListener("wheel",(event) => { event.preventDefault(); zoom(Math.exp(event.deltaY*0.001)); },{passive:false});
canvas.addEventListener("keydown",(event) => {
  const amount = event.shiftKey ? 10 : 3;
  const moves = {ArrowLeft:[-amount,0],ArrowRight:[amount,0],ArrowUp:[0,amount],ArrowDown:[0,-amount]};
  if (moves[event.key]) { event.preventDefault(); turn(...moves[event.key]); }
  if (event.key === "Home") { event.preventDefault(); resetView(); }
});
canvas.addEventListener("webglcontextlost",(event) => {
  gl = null;
  state.autoRotate = false; $("autoRotate").checked = false;
  $("exportButton").disabled = true;
  $("sceneNotice").hidden = false;
  $("sceneNotice").textContent = "GPU 渲染连接已中断，请重新打开页面并载入图片。";
});
new ResizeObserver(() => { needsRender = true; drawSource(); }).observe($("scene"));
new ResizeObserver(drawSource).observe(sourceCanvas.parentElement);
document.addEventListener("visibilitychange",() => { lastFrame = null; needsRender = true; });

function frame(time) {
  if (state.autoRotate && state.image && !document.hidden && lastFrame !== null) {
    turn(state.speed*Math.min((time-lastFrame)/1000,0.1),0);
  }
  if (needsRender) renderScene();
  lastFrame = time;
  requestAnimationFrame(frame);
}

$("exportButton").disabled = true;
$("mappingDescription").textContent = "圆心 → 头顶 · 圆周 → 地平线";
$("mappingHelp").textContent = "穹顶全景贴图按角度排列：天空在头顶，建筑沿地平线围成一圈。沉浸环顾使用常规透视，看建筑比例；全景仰视一次显示整个穹顶。圆形衍射效果图请在素材类型中选择衍射圆图，以保留原有方向余弦映射。";
try {
  initializeRenderer();
  requestAnimationFrame(frame);
} catch (error) {
  gl = null;
  $("errorMessage").textContent = error.message;
  $("sceneNotice").hidden = false;
  $("sceneNotice").textContent = error.message;
}
