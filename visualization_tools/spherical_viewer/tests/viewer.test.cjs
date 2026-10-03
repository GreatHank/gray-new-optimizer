const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const root = path.join(__dirname, "..");
const html = fs.readFileSync(path.join(root,"index.html"),"utf8");
const code = fs.readFileSync(path.join(root,"app.js"),"utf8");

// DOM/WebGL substitutes exercise event and camera logic, not GPU compilation.
function harness() {
  const values = {}, listeners = new Map(), elements = new Map(), created = [];
  const context2d = new Proxy({}, { get: (_,key) => key === "createRadialGradient"
    ? () => ({addColorStop(){}}) : () => {} });
  const gl = new Proxy({
    MAX_TEXTURE_SIZE: "max", NO_ERROR: 0,
    getParameter: () => 4096, getError: () => 0, isContextLost: () => false,
    getShaderParameter: () => true, getProgramParameter: () => true,
    getUniformLocation: (_,key) => key, getAttribLocation: () => 0,
    createShader: () => ({}), createProgram: () => ({}), createTexture: () => ({}), createBuffer: () => ({}),
  }, { get(target,key) {
    if (key in target) return target[key];
    if (String(key).startsWith("uniform")) return (name,...args) => { values[name] = args.length === 1 ? args[0] : args; };
    return () => {};
  }});
  function element(id) {
    const handlers = new Map(), classes = new Set();
    const el = {
      id, value:"", textContent:"", disabled:false, checked:false, hidden:false, width:960, height:640,
      clientWidth:960, clientHeight:640, files:[],
      classList:{add:(name) => classes.add(name), remove:(name) => classes.delete(name)},
      addEventListener:(name,callback) => handlers.set(name,callback),
      setAttribute(key,value){this[key] = value;},
      getBoundingClientRect: () => ({left:0,top:0,width:960,height:640}),
      getContext: (kind) => kind === "2d" ? context2d : gl,
      setPointerCapture(){},focus(){},click(){this.clicked = true;},
      toBlob(callback){callback({type:"image/png"});},
    };
    el.parentElement = {};
    listeners.set(id,handlers);
    elements.set(id,el);
    return el;
  }
  for (const match of html.matchAll(/\bid="([^"]+)"/g)) element(match[1]);
  const document = {
    hidden:false, getElementById:(id) => {
      assert.ok(elements.has(id),`missing HTML element ${id}`);
      return elements.get(id);
    },
    createElement:(tag) => { const el = element(`${tag}-${created.length}`); created.push(el); return el; },
    addEventListener(){},
  };
  const sandbox = vm.createContext({
    document, window:{devicePixelRatio:1,addEventListener(){}},
    URL:{createObjectURL:() => "blob:test",revokeObjectURL(){}},
    ResizeObserver:class {observe(){}}, requestAnimationFrame(){},setTimeout(){},
    console,Image:class {},
  });
  vm.runInContext(code,sandbox,{filename:"app.js"});
  return {
    values,elements,created,
    run:(expression) => vm.runInContext(expression,sandbox),
    event:(id,name,event={}) => listeners.get(id).get(name)({preventDefault(){},...event}),
    install:() => vm.runInContext('installImage({width:1000,height:1000}, "test.png"); renderScene();',sandbox),
  };
}

const close = (actual,expected) => assert.ok(Math.abs(actual-expected) < 1e-6,`${actual} != ${expected}`);
const dot = (a,b) => a.reduce((sum,v,index) => sum+v*b[index],0);

// Independent ray/sphere reference, driven by the renderer's camera uniforms.
function referenceHit(uniforms,sx=0,sy=0) {
  let vector;
  if (uniforms.uFisheye) {
    const radius = Math.hypot(sx,sy);
    if (radius > 1) return null;
    const angle = radius*uniforms.uHalfFov;
    vector = uniforms.uForward.map((v,i) => v*Math.cos(angle) + (radius > 0
      ? (uniforms.uRight[i]*sx+uniforms.uUp[i]*sy)/radius*Math.sin(angle) : 0));
  } else {
    vector = uniforms.uForward.map((v,i) => v+uniforms.uRight[i]*sx*uniforms.uTanFov+uniforms.uUp[i]*sy*uniforms.uTanFov);
  }
  const length = Math.hypot(...vector), ray = vector.map((v) => v/length), eye = uniforms.uEye;
  const b = dot(eye,ray), disc = b*b-dot(eye,eye)+1;
  if (disc < 0) return null;
  for (const t of [-b-Math.sqrt(disc),-b+Math.sqrt(disc)]) {
    const p = eye.map((v,i) => v+t*ray[i]);
    if (t > 0 && p[1] >= uniforms.uRimY-1e-5) return p;
  }
  return null;
}

test("page initializes with all HTML controls and no selected image",() => {
  const h = harness(); h.run("renderScene()");
  assert.equal(h.elements.get("errorMessage").textContent,"");
  assert.equal(h.elements.get("exportButton").disabled,true);
  assert.equal(h.values.uHasImage,false);
  assert.equal(h.values.uInside,true);
  close(h.run("state.pitch"),26);
  close(h.run("state.insideFov"),60);
  assert.equal(h.values.uFisheye,false);
  assert.equal(h.values.uAngular,true);
  assert.equal(h.values.uGrid,false);
});

test("image loads at unit scale, centered circle, and enables export",() => {
  const h = harness(); h.install();
  close(h.values.uCenter[0],499.5); close(h.values.uCenter[1],499.5); close(h.values.uRadius,500);
  assert.equal(h.elements.get("fileName").textContent,"test.png");
  assert.equal(h.elements.get("exportButton").disabled,false);
  assert.equal(h.elements.get("emptyState").hidden,true);
  assert.ok(!/texel\.rgb\s*\*/.test(code),"texture must not apply lighting or gain");
});

test("outside top view preserves source right and up after rotating the hemisphere",() => {
  const h = harness(); h.install(); h.event("outsideButton","click");
  h.event("frontView","click"); h.run("renderScene()");
  const center = referenceHit(h.values);
  close(center[0],0); close(center[1],1); close(center[2],0);
  assert.ok(referenceHit(h.values,.15,0)[0] > 0);
  assert.ok(referenceHit(h.values,0,.15)[2] < 0);
  assert.match(code,/vec2\(p\.x,p\.z\)/,"image top must map to negative dome Z");
});

test("sphere-center camera stays at center at every viewing angle",() => {
  const h = harness(); h.install(); h.event("insideButton","click");
  for (const [yaw,pitch] of [[0,0],[40,20],[120,-45],[-170,85]]) {
    h.run(`state.yaw=${yaw}; state.pitch=${pitch}; renderScene();`);
    for (const value of h.values.uEye) close(value,0);
    for (const basis of [h.values.uForward,h.values.uRight,h.values.uUp]) close(Math.hypot(...basis),1);
    close(dot(h.values.uForward,h.values.uRight),0);
    close(dot(h.values.uForward,h.values.uUp),0);
    close(dot(h.values.uRight,h.values.uUp),0);
  }
});

test("overview from base center includes zenith and all four horizon directions",() => {
  const h = harness(); h.install(); h.event("overviewButton","click"); h.run("renderScene()");
  close(referenceHit(h.values)[1],1);
  assert.ok(referenceHit(h.values,.15,0)[0] > 0);
  assert.ok(referenceHit(h.values,0,.15)[2] < 0);
  for (const [sx,sy] of [[1,0],[-1,0],[0,1],[0,-1]]) {
    const p = referenceHit(h.values,sx,sy);
    close(p[1],0); close(Math.hypot(p[0],p[2]),1);
  }
  close(referenceHit(h.values,0.5,0)[1],Math.SQRT1_2);
  assert.equal(referenceHit(h.values,1.01,0),null);
  h.run("state.pitch=-45; renderScene()");
  assert.equal(referenceHit(h.values),null);
  assert.equal(h.elements.get("sceneNotice").hidden,false);
});

test("disk boundary reaches the specified sphere cap without moving image circle",() => {
  const h = harness(); h.install();
  for (const field of [20,90,180]) {
    h.event("coverage","input",{target:{value:String(field)}}); h.run("renderScene()");
    const halfAngle = field*Math.PI/360;
    close(h.values.uDiskScale,Math.sin(halfAngle)); close(h.values.uRimY,Math.cos(halfAngle));
    close(Math.hypot(h.values.uDiskScale,h.values.uRimY),1);
    close(h.values.uRadius,500);
  }
});

test("invalid circle edit leaves geometry intact and surfaces an error",() => {
  const h = harness(); h.install();
  const field = h.elements.get("sourceRadius"); field.value = "0";
  h.event("sourceRadius","change",{target:field}); h.run("renderScene()");
  close(h.values.uRadius,500);
  assert.match(h.elements.get("errorMessage").textContent,/大于 0/);
  field.value = "600"; h.event("sourceRadius","change",{target:field}); h.run("renderScene()");
  assert.match(h.elements.get("sourceCaption").textContent,/缺失部分/);
  close(h.values.uRadius,600);
});

test("drag rotates, zoom clamps, and view mode resets the camera",() => {
  const h = harness(); h.install(); h.event("outsideButton","click");
  h.event("sceneCanvas","pointerdown",{button:0,pointerId:1,clientX:20,clientY:20});
  h.event("sceneCanvas","pointermove",{pointerId:1,clientX:100,clientY:40});
  h.event("sceneCanvas","pointerup",{pointerId:1}); h.run("renderScene()");
  assert.match(h.elements.get("viewReadout").textContent,/20°.*40°/);
  h.run("zoom(0.001); renderScene()"); close(h.run("state.distance"),1.2);
  h.event("insideButton","click"); h.run("renderScene()");
  close(h.run("state.yaw"),0); close(h.run("state.pitch"),26);
  h.run("zoom(100); renderScene()"); close(h.run("state.insideFov"),100);
  assert.equal(h.elements.get("insideButton")["aria-pressed"],"true");
  h.event("overviewButton","click"); h.run("renderScene()");
  close(h.run("state.pitch"),90); close(h.run("state.insideFov"),180);
  assert.equal(h.values.uFisheye,true);
});

test("oversized image fails explicitly and keeps current selection",() => {
  const h = harness(); h.install();
  assert.throws(() => h.run('installImage({width:5000,height:1000},"oversized.png")'),/纹理上限/);
  assert.equal(h.elements.get("fileName").textContent,"test.png");
});

test("export includes annotations and a downloadable PNG filename",() => {
  const h = harness(); h.install(); h.event("exportButton","click");
  const output = h.created.find((el) => el.id.startsWith("canvas-"));
  const link = h.created.find((el) => el.id.startsWith("a-"));
  assert.equal(output.height,h.elements.get("sceneCanvas").height+72);
  assert.match(link.download,/test_球面_inside_0deg_.*\.png$/);
  assert.equal(link.clicked,true);
});

test("immersive view uses rectilinear perspective, not a fisheye lens",() => {
  const h = harness(); h.install();
  assert.equal(h.values.uFisheye,false);
  h.run("state.pitch=0; renderScene()");
  const p = referenceHit(h.values,0,0.4);
  const elevation = Math.atan2(p[1],p[2]);
  close(elevation,Math.atan(0.4*Math.tan(60*Math.PI/360)));
});

test("angular panoramas and physical diffraction circles have explicit separate mappings",() => {
  const h = harness(); h.install();
  assert.equal(h.values.uAngular,true);
  h.event("projection","change",{target:{value:"direction"}}); h.run("renderScene()");
  assert.equal(h.values.uAngular,false);
  assert.match(code,/polar\/halfCoverage/);
  assert.match(code,/vec2\(p\.x,p\.z\) \/ uDiskScale/);
});
