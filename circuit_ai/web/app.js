import {createWorkbenchStore, DRAFT_STORAGE_KEY, WORKBENCH_SCHEMA_VERSION} from "./store.js";

const $ = (selector) => document.querySelector(selector);
function reportFromError(error, fallbackPhase = "request", fallbackCode = "request_failed") {
  if (error?.report) return error.report;
  return {
    schema: "circuit_ai.error_report",
    schema_version: 1,
    phase: fallbackPhase,
    phase_label: fallbackPhase,
    code: fallbackCode,
    severity: "error",
    message: error?.message || "请求失败",
    detail: "",
    suggestion: "检查输入和服务状态后重试。",
    retryable: true,
    cause_type: error?.name || "",
    context: {}
  };
}

function reportText(value) {
  if (value === undefined || value === null) return "";
  if (typeof value === "string") return value;
  try { return JSON.stringify(value, null, 2); } catch (_) { return String(value); }
}

function markJsonEditorInvalid(invalid) {
  const editor = $("#json-editor");
  if (!editor) return;
  if (invalid) {
    editor.setAttribute("aria-invalid", "true");
    editor.setAttribute("aria-describedby", "error-message");
  } else {
    editor.removeAttribute("aria-invalid");
    editor.removeAttribute("aria-describedby");
  }
}

function clearError() {
  const panel = $("#error-panel");
  if (panel) panel.hidden = true;
  $(".workspace")?.classList.remove("has-error");
  const announcer = $("#error-announcer");
  if (announcer) announcer.textContent = "";
  markJsonEditorInvalid(false);
}

function showError(error, fallbackPhase = "request", fallbackCode = "request_failed") {
  const report = reportFromError(error, fallbackPhase, fallbackCode);
  markJsonEditorInvalid(fallbackPhase === "pbdl" || report.phase === "pbdl");
  const isExecutionError = fallbackPhase === "execution" || report.phase === "execution";
  if (isExecutionError) {
    workbenchState.execution = {status: "failed", startedAt: workbenchState.execution.startedAt, processed: workbenchState.execution.processed || 0};
    setWorkbenchState({execution: workbenchState.execution});
  } else {
    workbenchState.compile = {status: "blocked", data: workbenchState.compile.data || null};
    setWorkbenchState({compile: workbenchState.compile});
  }
  const panel = $("#error-panel");
  if (!panel) return report;
  panel.hidden = false;
  $("#error-phase").textContent = report.phase_label || report.phase || "错误";
  $("#error-title").textContent = report.message || "执行失败";
  $("#error-code").textContent = report.code || "-";
  $("#error-retry").textContent = report.retryable ? "可重试" : "需要处理输入或能力缺口";
  $("#error-message").textContent = reportText(report.detail);
  $("#error-detail").textContent = report.cause_type ? `内部类型：${report.cause_type}` : "";
  $("#error-detail").hidden = !report.cause_type;
  $("#error-suggestion-text").textContent = report.suggestion || "查看阶段日志和完整报告。";
  $("#run-status").textContent = `${report.phase_label || "失败"}：${report.code || "error"}`;
  $(".workspace")?.classList.add("has-error");
  const errorAnnouncer = $("#error-announcer");
  if (errorAnnouncer) errorAnnouncer.textContent = `${report.phase_label || "错误"}：${report.message || report.code || "执行失败"}`;
  announce(`${report.phase_label || "错误"}：${report.message || report.code || "执行失败"}`);
  return report;
}

function showInlineError(selector, error, fallbackPhase = "request") {
  const node = $(selector);
  if (!node) return;
  const report = reportFromError(error, fallbackPhase);
  node.classList.add("error-inline");
  node.textContent = `${report.phase_label || report.phase}：${report.code} · ${report.detail || report.message}`;
}

async function requestJson(url, options = {}, phase = "request") {
  let response;
  let data;
  try {
    response = await fetch(url, options);
    data = await response.json();
  } catch (error) {
    if (error.name === "AbortError") throw error;
    const wrapped = new Error("无法连接到本地设计服务。");
    wrapped.report = {
      schema: "circuit_ai.error_report",
      schema_version: 1,
      phase,
      phase_label: phase,
      code: "network_unavailable",
      severity: "error",
      message: "无法连接到本地设计服务",
      detail: error.message || "浏览器无法读取服务响应。",
      suggestion: "确认工作台服务仍在运行，然后重试。",
      retryable: true,
      cause_type: error.name || "",
      context: {url}
    };
    throw wrapped;
  }
  if (!response.ok) {
    const detail = data?.detail;
    const wrapped = new Error(typeof detail === "string" ? detail : detail?.message || "请求失败");
    wrapped.report = typeof detail === "object" && detail !== null ? detail : {
      schema: "circuit_ai.error_report",
      schema_version: 1,
      phase,
      phase_label: phase,
      code: `http_${response.status}`,
      severity: "error",
      message: wrapped.message,
      detail: "服务拒绝了本次请求。",
      suggestion: "检查输入内容和后端日志后重试。",
      retryable: response.status >= 500,
      cause_type: "HTTPException",
      context: {url, status: response.status}
    };
    throw wrapped;
  }
  return data;
}
const fields = ["name", "port-count", "analysis-kind", "behavior-kind", "frequency", "gain", "f-min", "f-max", "max-components", "top-k", "max-iterations", "waveform-kind", "signal-port", "signal-frequency", "signal-amplitude", "signal-offset", "signal-phase", "pulse-frequency", "pulse-duty", "pulse-low", "pulse-high", "pulse-rise", "pulse-fall", "sample-period", "sample-values"];
const workbenchStore = createWorkbenchStore({initialModel: {
  ports: [
    {id: "p1", name: "input", positive: "in", negative: "0", role: "input", domain: "ac", position: {edge: "left", offset: .5}, variables: [{id: "v", quantity: "voltage", role: "potential", unit: "V"}, {id: "i", quantity: "current", role: "flow", unit: "A"}, {id: "p", quantity: "power", role: "derived", unit: "W"}], constraints: [{id: "c1", variable: "v", analysis: "small_signal_ac", operator: "equal", value: 1}], excitation: null, verification: "pending"},
    {id: "p2", name: "output", positive: "out", negative: "0", role: "output", domain: "ac", position: {edge: "right", offset: .5}, variables: [{id: "v", quantity: "voltage", role: "potential", unit: "V"}, {id: "i", quantity: "current", role: "flow", unit: "A"}, {id: "p", quantity: "power", role: "derived", unit: "W"}], constraints: [{id: "c2", variable: "v", analysis: "small_signal_ac", operator: "equal", value: 1}], excitation: null, verification: "pending"}
  ],
  measurements: [{id: "m1", kind: "voltage_transfer", source_port: "p1", response_port: "p2"}],
  functions: [{function_id: "main_transfer", role: "target", domain: "frequency", source_port: "p1", response_port: "p2", body: {kind: "builtin", name: "lowpass", parameters: {cutoff: {value: 1000, unit: "Hz"}, gain: {value: 1, unit: "1"}}}, axis: {domain: "frequency", variable: "frequency", unit: "Hz", start: 10, stop: 100000, points: 96}, verification: "pending"}],
  realComponents: [],
  modelBindings: []
}});
const workbenchState = workbenchStore.state;
const draftModel = workbenchState.draft.model;
let draftSaveTimer = null;
let componentSearchTimer = null;
let componentSearchController = null;
let isBootstrapping = true;
let latest = null;
let compared = new Set([0]);
const portDefinitions = draftModel.ports;
const measurementDefinitions = draftModel.measurements;
const functionDefinitions = draftModel.functions;
let currentWizardStep = 0;
const selectedRealComponents = draftModel.realComponents;
const modelBindings = draftModel.modelBindings;
let modelDraft = null;
let canvasMode = "select";
let selectedObject = {kind: "circuit", id: null};
let functionLinkStart = null;
let draggingPortId = null;
let dragHistorySnapshot = null;
let dragMoved = false;
const canvasHistory = [];

function replaceDraftCollection(name, values) {
  return workbenchStore.replaceCollection(name, values);
}

function setDraftMeta(status, dirty) {
  workbenchState.draft = {...workbenchState.draft, status, dirty, model: draftModel};
  return workbenchState.draft;
}

function captureCanvasState() {
  return workbenchStore.snapshotModel(["ports", "measurements", "functions"]);
}

function updateCanvasUndoButton() {
  const button = $("#canvas-undo");
  if (button) button.disabled = canvasHistory.length === 0;
}

function pushCanvasHistory(snapshot = captureCanvasState()) {
  canvasHistory.push(snapshot);
  if (canvasHistory.length > 30) canvasHistory.shift();
  updateCanvasUndoButton();
}

function undoCanvasChange() {
  const snapshot = canvasHistory.pop();
  if (!snapshot) return;
  workbenchStore.restoreModel(snapshot, ["ports", "measurements", "functions"]);
  selectedObject = {kind: "circuit", id: null};
  functionLinkStart = null;
  renderPorts();
  renderMeasurements();
  renderCanvas();
  renderInspector();
  refreshEditor();
  updateCanvasUndoButton();
  announce("已撤销最近的画布编辑");
}

function announce(message) {
  const target = $("#status-announcer");
  if (target) target.textContent = message || "";
}

function setWorkbenchState(next, message = "") {
  Object.assign(workbenchState, next);
  const workspace = $(".workspace");
  if (workspace) {
    workspace.dataset.state = workbenchState.execution.status !== "idle" ? workbenchState.execution.status : workbenchState.compile.status;
    workspace.dataset.mode = workbenchState.mode;
    workspace.classList.toggle("results-mode", workbenchState.mode === "results");
  }
  if (message) {
    $("#run-status").textContent = message;
    announce(message);
  }
}

function markDraftDirty() {
  if (isBootstrapping) return;
  setDraftMeta("dirty", true);
  workbenchState.compile = {status: "idle", data: null};
  setWorkbenchState({draft: workbenchState.draft, compile: workbenchState.compile}, "草稿 · 需要重新检查");
  scheduleDraftSave();
}

function scheduleDraftSave() {
  window.clearTimeout(draftSaveTimer);
  draftSaveTimer = window.setTimeout(saveDraft, 350);
}

function saveDraft() {
  try {
    const payload = {schema_version: WORKBENCH_SCHEMA_VERSION, saved_at: new Date().toISOString(), step: currentWizardStep, spec: specFromForm()};
    localStorage.setItem(DRAFT_STORAGE_KEY, JSON.stringify(payload));
    $("#restore-draft")?.removeAttribute("hidden");
    setDraftMeta("saved", false);
    setWorkbenchState({draft: workbenchState.draft});
    $("#run-status").textContent = "草稿 · 已自动保存";
  } catch (_) { /* local storage can be disabled in private browsing */ }
}

function restoreDraft() {
  try {
    const raw = localStorage.getItem(DRAFT_STORAGE_KEY);
    if (!raw) return;
    const payload = JSON.parse(raw);
    if (payload.schema_version !== WORKBENCH_SCHEMA_VERSION || !payload.spec) return;
    applySpec(payload.spec);
    refreshEditor();
    showWizardStep(Number.isInteger(payload.step) ? payload.step : 0);
    setDraftMeta("draft", false);
    setWorkbenchState({draft: workbenchState.draft}, "草稿 · 已恢复");
  } catch (_) { /* ignore an invalid local draft */ }
}

function exportPbdl() {
  const blob = new Blob([JSON.stringify(specFromForm(), null, 2)], {type: "application/json"});
  const url = URL.createObjectURL(blob), link = document.createElement("a");
  link.href = url; link.download = `${value("name") || "circuit-spec"}.pbdl.json`; link.click();
  URL.revokeObjectURL(url);
  announce("PBDL 已导出");
}

function value(id) { return $("#" + id).value; }
function number(id) { return Number(value(id)); }

function waveformFromForm() {
  const kind = value("waveform-kind");
  if (kind === "sine") return {kind, frequency_hz: number("signal-frequency"), amplitude_v: number("signal-amplitude"), offset_v: number("signal-offset"), phase_deg: number("signal-phase")};
  if (kind === "pulse") return {kind, frequency_hz: number("pulse-frequency"), low_v: number("pulse-low"), high_v: number("pulse-high"), duty_cycle: number("pulse-duty") / 100, rise_time_s: number("pulse-rise"), fall_time_s: number("pulse-fall")};
  const samples = value("sample-values").split(",").map((item) => Number(item.trim())).filter(Number.isFinite);
  return {kind, period_s: number("sample-period"), value_v: samples, repeat: true};
}

function portSpec(port) {
  const variables = (port.variables && port.variables.length) ? port.variables : [
    {id: "v", name: "v", quantity: "voltage", role: "potential", unit: "V"},
    {id: "i", name: "i", quantity: "current", role: "flow", unit: "A"},
    {id: "p", name: "p", quantity: "power", role: "derived", unit: "W"}
  ];
  const result = {name: port.name, description: "", role: port.role || "bidirectional", domain: port.domain || "unspecified", terminals: [
    {name: port.positive, quantity: "voltage"},
    {name: port.negative, quantity: isReferenceNode(port.negative) ? "ground" : "voltage"}
  ], variables: variables.map((variable) => ({name: variable.id || variable.name, quantity: variable.quantity, role: variable.role, unit: variable.unit || unitForVariable(variable.id || variable.name), expression: variable.expression || (variable.id === "v" ? `V(${port.positive}) - V(${port.negative})` : variable.id === "i" ? `current entering ${port.positive}` : variable.id === "p" ? "v * i" : undefined)})), variable_constraints: port.constraints.map((constraint) => ({...constraint, unit: unitForVariable(constraint.variable)})), port_id: port.id};
  if (port.excitation) result.excitation = port.excitation;
  return result;
}

function unitForVariable(variable) { return ({v: "V", i: "A", p: "W"})[variable] || ""; }
function isReferenceNode(name) { const normalized = String(name || "").toLowerCase(); return normalized === "0" || normalized === "gnd" || normalized.endsWith("_0"); }
function dcConstraint(port, variable) { return (port.constraints || []).find((item) => item.variable === variable && item.analysis === "dc_operating_point"); }

function deriveDcIntent() {
  const voltagePorts = portDefinitions.filter((port) => dcConstraint(port, "v"));
  const input = voltagePorts[0], output = voltagePorts[1];
  if (!input || !output) return null;
  const outputCurrent = dcConstraint(output, "i");
  const outputCurrentMagnitude = outputCurrent ? Math.max(Math.abs(Number(outputCurrent.value || 0)), Math.abs(Number(outputCurrent.minimum || 0)), Math.abs(Number(outputCurrent.maximum || 0))) : 0;
  const isolated = measurementDefinitions.some((relation) => relation.kind === "galvanic_isolation" && [relation.source_port, relation.response_port].includes(input.id) && [relation.source_port, relation.response_port].includes(output.id));
  return {
    kind: isolated ? "isolated_dc_conversion" : "dc_conversion",
    input_port: input.name,
    input_voltage_v: Number(dcConstraint(input, "v").value),
    output_port: output.name,
    output_voltage_v: Number(dcConstraint(output, "v").value),
    output_current_a: outputCurrentMagnitude,
    voltage_ratio: Number(dcConstraint(input, "v").value) ? Number(dcConstraint(output, "v").value) / Number(dcConstraint(input, "v").value) : null,
    galvanic_isolation: isolated,
    ideal: true
  };
}

function ensureIsolationReferenceNodes() {
  for (const relation of measurementDefinitions.filter((item) => item.kind === "galvanic_isolation")) {
    const input = portDefinitions.find((port) => port.id === relation.source_port);
    const output = portDefinitions.find((port) => port.id === relation.response_port);
    if (!input || !output) continue;
    input.role = input.role === "bidirectional" || !input.role ? "input" : input.role;
    output.role = output.role === "bidirectional" || !output.role ? "output" : output.role;
    input.domain = "dc";
    output.domain = "dc";
    if (input.negative === output.negative) output.negative = `${output.name || "output"}_0`;
  }
}

function updateDomainVisibility() {
  const intent = deriveDcIntent();
  const mode = $("#design-mode");
  const route = $("#route-summary");
  if (mode) mode.textContent = intent ? "理想直流功率" : "线性频域";
  if (route) route.textContent = intent
    ? "PBDL -> Unified IR -> 功率候选 -> 理想平均模型 -> 约束验证 -> KiCad"
    : "PBDL -> Unified IR -> 线性候选 -> 参数优化 -> Linear MNA -> 约束验证";
  $("#frequency-target-fields").hidden = Boolean(intent);
  $("#dc-derived-behavior").hidden = !intent;
  $("#signal-definition-fields").hidden = Boolean(intent);
  if (!intent) { updateWaveform(); return; }
  const outputPower = intent.output_voltage_v * intent.output_current_a;
  const inputCurrent = intent.input_voltage_v ? outputPower / intent.input_voltage_v : 0;
  $("#dc-derived-behavior").innerHTML = `<b>${intent.galvanic_isolation ? "隔离" : "非隔离"}理想直流变换</b><span>${formatVoltage(intent.input_voltage_v)} → ${formatVoltage(intent.output_voltage_v)} / ${Number(intent.output_current_a).toPrecision(3)} A</span><span>变比 ${Number(intent.voltage_ratio).toPrecision(4)} · 输出 ${Number(outputPower).toPrecision(4)} W · 理想输入电流 ${Number(inputCurrent).toPrecision(4)} A</span>`;
  $("#port-capability").textContent = intent.galvanic_isolation
    ? "已识别为隔离直流变换：将调用理想 Flyback 专家、参数优化、约束验证和 KiCad 原理图输出。"
    : "已识别为非隔离直流变换：将调用理想 Boost 专家、参数优化、约束验证和 KiCad 原理图输出。";
}

function functionParamNumber(fn, name, fallback) {
  const value = fn?.body?.parameters?.[name];
  const numberValue = typeof value === "object" ? value.value : value;
  return Number.isFinite(Number(numberValue)) ? Number(numberValue) : fallback;
}

function primaryFunction() {
  return functionDefinitions.find((fn) => functionRelationKind(fn) !== "galvanic_isolation") || functionDefinitions[0];
}

function syncBehaviorControlsFromFunction(fn) {
  if (!fn || fn !== functionDefinitions[0]) return;
  const relationKind = functionRelationKind(fn);
  const bodyName = fn.body?.name || "lowpass";
  const parameterName = relationKind === "transimpedance" ? "transimpedance" : ["input_impedance", "output_impedance"].includes(relationKind) ? "ohms" : bodyName === "bandpass" ? "center" : "cutoff";
  const axis = fn.axis || {};
  $("#analysis-kind").value = relationKind;
  if (["lowpass", "highpass", "bandpass", "transimpedance", "impedance", "output_impedance"].includes(bodyName)) $("#behavior-kind").value = relationKind === "output_impedance" ? "output_impedance" : bodyName;
  $("#frequency").value = functionParamNumber(fn, parameterName, number("frequency"));
  $("#gain").value = functionParamNumber(fn, relationKind === "transimpedance" ? "transimpedance" : ["input_impedance", "output_impedance"].includes(relationKind) ? "ohms" : "gain", number("gain"));
  if (axis.start !== undefined) $("#f-min").value = axis.start;
  if (axis.stop !== undefined) $("#f-max").value = axis.stop;
}

function updatePrimaryFunctionFromControls() {
  const fn = primaryFunction();
  if (!fn) return;
  const selected = value("behavior-kind");
  const relationKind = selected === "transimpedance" ? "transimpedance" : selected === "impedance" ? "input_impedance" : selected === "output_impedance" ? "output_impedance" : "voltage_transfer";
  const bodyName = functionBodyName(relationKind, selected);
  const parameterName = relationKind === "transimpedance" ? "transimpedance" : ["input_impedance", "output_impedance"].includes(relationKind) ? "ohms" : bodyName === "bandpass" ? "center" : "cutoff";
  fn.relation_kind = relationKind;
  fn.body = {kind: "builtin", name: bodyName, parameters: {...(fn.body?.parameters || {}), [parameterName]: {value: number("frequency"), unit: relationKind === "transimpedance" || ["input_impedance", "output_impedance"].includes(relationKind) ? "ohm" : "Hz"}, gain: {value: number("gain"), unit: "1"}}};
  fn.domain = "frequency";
  fn.axis = {domain: "frequency", variable: "frequency", unit: "Hz", start: number("f-min"), stop: number("f-max"), points: 96};
  syncRelationsFromFunctions();
}

function specFromForm() {
  const primary = primaryFunction();
  const relationKind = primary ? functionRelationKind(primary) : "voltage_transfer";
  const bodyName = primary?.body?.name || value("behavior-kind");
  const parameterName = relationKind === "transimpedance" ? "transimpedance" : ["input_impedance", "output_impedance"].includes(relationKind) ? "ohms" : bodyName === "bandpass" ? "center" : "cutoff";
  const kind = relationKind === "voltage_transfer" ? bodyName : relationKind;
  const frequency = functionParamNumber(primary, parameterName, number("frequency"));
  const gain = functionParamNumber(primary, relationKind === "transimpedance" ? "transimpedance" : ["input_impedance", "output_impedance"].includes(relationKind) ? "ohms" : "gain", number("gain"));
  let behavior = { kind, frequency_range_hz: [number("f-min"), number("f-max")] };
  const preliminaryDcIntent = deriveDcIntent();
  if (preliminaryDcIntent) portDefinitions.forEach((port) => { port.excitation = null; });
  const activePort = portDefinitions.find((port) => port.id === value("signal-port"));
  if (activePort && !preliminaryDcIntent) activePort.excitation = {quantity: activePort.quantity, waveform: waveformFromForm()};
  const excitations = portDefinitions.filter((port) => port.excitation).map((port) => ({port: port.name, ...port.excitation}));
  if (excitations.length) behavior.excitation = excitations[0];
  behavior.excitations = excitations;
  if (kind === "bandpass") { behavior.center_hz = frequency; behavior.q = 3; behavior.gain = gain; }
  else if (kind === "transimpedance") { behavior.transimpedance_ohm = gain; behavior.cutoff_hz = frequency; }
  else if (kind === "impedance" || kind === "output_impedance") { behavior.ohms = gain; }
  else { behavior.cutoff_hz = frequency; behavior.gain = gain; behavior.order = 1; }
  const dcIntent = preliminaryDcIntent;
  if (dcIntent) behavior = dcIntent;
  const allowed = [...document.querySelectorAll("input[name=element]:checked")].map((node) => node.value);
  const required = [...document.querySelectorAll("input[name=required-element]:checked")].map((node) => node.value);
  syncRelationsFromFunctions();
  const relationSpecs = measurementDefinitions.map((measurement) => {
    const relation = {
      ...measurement,
      source_port: (portDefinitions.find((port) => port.id === measurement.source_port) || {}).name,
      response_port: (portDefinitions.find((port) => port.id === measurement.response_port) || {}).name
    };
    return relation;
  });
  const activeRelations = relationSpecs.filter((relation) => relation.kind !== "galvanic_isolation");
  const dcAnalysis = dcIntent ? [{kind: "dc_transfer", source_port: dcIntent.input_port, output_port: dcIntent.output_port}] : [];
  const analyses = dcIntent ? dcAnalysis : activeRelations.map((relation) => ({
    kind: ["input_impedance", "output_impedance"].includes(relation.kind) ? "impedance" : relation.kind,
    source_port: relation.source_port,
    output_port: relation.response_port,
    port: relation.kind === "output_impedance" ? "output" : relation.source_port,
    frequency_hz: relation.frequency_hz
  }));
  const target = dcIntent ? {
    target_kind: "dc",
    input_voltage_v: dcIntent.input_voltage_v,
    output_voltage_v: dcIntent.output_voltage_v,
    output_current_a: dcIntent.output_current_a
  } : kind === "bandpass" ? {
    target_kind: "filter", kind, cutoff_hz: frequency, q: 3, order: 1,
    gain_db: 20 * Math.log10(Math.max(gain, 1e-18))
  } : kind === "transimpedance" ? {
    target_kind: "amplifier", gain_db: 20 * Math.log10(Math.max(gain, 1e-18)),
    bandwidth_hz: [number("f-min"), frequency]
  } : ["impedance", "output_impedance"].includes(kind) ? {
    target_kind: "impedance", ohms: gain
  } : {
    target_kind: "filter", kind, cutoff_hz: frequency, order: 1,
    gain_db: 20 * Math.log10(Math.max(gain, 1e-18))
  };
  const targets = dcIntent ? [target] : activeRelations.map((relation) => {
    if (relation.kind === "transimpedance") return {
      target_kind: "amplifier",
      gain_db: 20 * Math.log10(Math.max(gain, 1e-18)),
      bandwidth_hz: [number("f-min"), frequency]
    };
    if (["input_impedance", "output_impedance"].includes(relation.kind)) return {
      target_kind: "impedance",
      ohms: gain
    };
    return target;
  });
  return {
    schema: "pbdl.circuit_spec",
    schema_version: 2,
    name: value("name"),
    description: "",
    ports: portDefinitions.map(portSpec),
    relations: relationSpecs,
    functions: functionDefinitions.filter((fn) => functionRelationKind(fn) !== "galvanic_isolation").map((fn) => ({
      function_id: fn.function_id,
      relation_id: fn.relation_id,
      role: fn.role || "target",
      domain: fn.domain || "frequency",
      inputs: [`ports.${fn.source_port}.variables.${fn.body?.name === "transimpedance" ? "i" : "v"}`],
      outputs: [`ports.${fn.response_port}.variables.v`],
      axis: fn.axis || undefined,
      body: fn.body
    })),
    analyses,
    targets,
    constraints: {
      element_types: allowed,
      required_elements: required,
      real_components: selectedRealComponents,
      model_bindings: modelBindings,
      parameter_ranges: { R: [100, 1000000], C: [1e-12, 1e-4], L: [1e-9, 1] },
      max_component_count: number("max-components")
    },
    operating_point: {temperature_c: 25},
    optimization: {
      ...(dcIntent ? {} : {points: 96}),
      max_components: number("max-components"),
      top_k: number("top-k"),
      max_iterations: number("max-iterations"),
      seed: 11
    }
  };
}

function refreshEditor() {
  $("#json-editor").value = JSON.stringify(specFromForm(), null, 2);
  renderDesignSummary();
  const resultSummary = $("#result-spec-summary");
  if (resultSummary) resultSummary.innerHTML = $("#design-summary")?.innerHTML || "";
  renderCanvas();
  if (!isBootstrapping) markDraftDirty();
}

function showWizardStep(step) {
  currentWizardStep = Math.max(0, Math.min(4, step));
  document.querySelectorAll("[data-wizard-step]").forEach((section) => { section.hidden = Number(section.dataset.wizardStep) !== currentWizardStep; });
  $("#wizard-progress").querySelectorAll("button").forEach((button) => {
    const index = Number(button.dataset.step);
    button.classList.toggle("active", index === currentWizardStep);
    button.classList.toggle("completed", index < currentWizardStep);
    button.setAttribute("aria-current", index === currentWizardStep ? "step" : "false");
  });
  $("#wizard-back").disabled = currentWizardStep === 0;
  $("#wizard-next").hidden = currentWizardStep === 4;
  $("#wizard-error").textContent = "";
  if (currentWizardStep === 4) {
    renderDesignSummary();
    if (["idle", "blocked"].includes(workbenchState.compile.status) && !isBootstrapping) window.setTimeout(() => compileFormSpec(), 0);
  }
}

function validateAllSteps() {
  clearFieldValidation();
  for (let step = 0; step < 4; step++) {
    const error = validateWizardStep(step);
    if (error) { showWizardStep(step); $("#wizard-error").textContent = error; focusValidationField(step); return false; }
  }
  return true;
}

function clearFieldValidation() {
  document.querySelectorAll("[aria-invalid='true']").forEach((node) => { node.removeAttribute("aria-invalid"); node.removeAttribute("aria-describedby"); });
}

function focusValidationField(step) {
  const selectors = {
    0: "#name",
    1: "#port-editor input[data-field='name'], #measurement-editor select",
    2: "#f-min, #frequency, #signal-port",
    3: "input[name=element], #max-components"
  };
  const node = $(selectors[step]);
  if (!node) return;
  node.setAttribute("aria-invalid", "true");
  node.setAttribute("aria-describedby", "wizard-error");
  window.setTimeout(() => node.focus({preventScroll: true}), 0);
}

function validateWizardStep(step) {
  if (step === 0) {
    if (!value("name").trim()) return "请填写任务名称。";
    if (portDefinitions.length < 1) return "至少需要一个端口。";
  }
  if (step === 1) {
    const names = portDefinitions.map((port) => port.name.trim());
    if (names.some((name) => !name)) return "每个端口都需要名称。";
    if (new Set(names).size !== names.length) return "端口名称不能重复。";
    if (portDefinitions.some((port) => !port.positive.trim() || !port.negative.trim())) return "每个端口都需要完整的正端子和参考端子。";
    if (!measurementDefinitions.length) return "至少添加一个端口关系。";
  }
  if (step === 2) {
    const intent = deriveDcIntent();
    if (intent) {
      if (!(intent.input_voltage_v > 0 && intent.output_voltage_v > 0)) return "直流输入和输出端口的端子间电压必须大于 0。";
      if (!(intent.output_current_a > 0)) return "直流输出端口需要填写输出电流能力。";
      const isolation = measurementDefinitions.find((item) => item.kind === "galvanic_isolation");
      if (isolation) {
        const input = portDefinitions.find((port) => port.id === isolation.source_port);
        const output = portDefinitions.find((port) => port.id === isolation.response_port);
        if (input && output && input.negative === output.negative) return "电气隔离要求输入与输出使用不同参考节点，例如输入 0、输出 out_0。";
      }
    } else {
      if (!(number("f-min") > 0 && number("f-max") > number("f-min"))) return "频率范围必须为正数且结束频率大于起始频率。";
      if (!(number("frequency") > 0)) return "截止或中心频率必须大于 0。";
      if (!portDefinitions.some((port) => port.excitation)) return "至少为一个输入、供电或双向端口定义激励。";
    }
  }
  if (step === 3) {
    const allowed = new Set([...document.querySelectorAll("input[name=element]:checked")].map((node) => node.value));
    if (!allowed.size) return "至少允许一种元件。";
    const required = [...document.querySelectorAll("input[name=required-element]:checked")].map((node) => node.value);
    if (required.some((item) => !allowed.has(item))) return "必须使用的元件也必须出现在允许列表中。";
  }
  return "";
}

// The editor renders a derived view, while the textarea remains canonical PBDL.
function pbdlView(spec) {
  const target = spec.targets[0] || {};
  const relation = (spec.relations || []).find((item) => item.kind !== "galvanic_isolation") || {};
  const analysis = spec.analyses[0] || {};
  let behavior;
  if (target.target_kind === "dc") {
    const isolated = (spec.relations || []).some((item) => item.kind === "galvanic_isolation");
    behavior = {kind: isolated ? "isolated_dc_conversion" : "dc_conversion", input_port: target.input_port || analysis.source_port, output_port: target.output_port || analysis.output_port, input_voltage_v: target.input_voltage_v, output_voltage_v: target.output_voltage_v, output_current_a: target.output_current_a, galvanic_isolation: isolated, ideal: true};
  } else if (target.target_kind === "filter") {
    behavior = {kind: target.kind || "lowpass", cutoff_hz: target.cutoff_hz, center_hz: target.cutoff_hz, gain: 10 ** ((target.gain_db || 0) / 20), frequency_range_hz: relation.frequency_hz || [10, 100000]};
  } else if (target.target_kind === "amplifier") {
    behavior = {kind: analysis.kind === "transimpedance" ? "transimpedance" : "lowpass", transimpedance_ohm: 10 ** ((target.gain_db || 0) / 20), gain: 10 ** ((target.gain_db || 0) / 20), cutoff_hz: (target.bandwidth_hz || [10, 100000])[1], frequency_range_hz: relation.frequency_hz || [10, 100000]};
  } else {
    behavior = {kind: analysis.kind === "impedance" && analysis.port === "output" ? "output_impedance" : "impedance", ohms: target.ohms, frequency_range_hz: relation.frequency_hz || [10, 100000]};
  }
  return {...spec, ports: spec.ports.length, interface: {ports: spec.ports, relations: spec.relations || [], measurements: spec.relations || []}, analysis: {kind: analysis.kind === "impedance" ? (analysis.port === "output" ? "output_impedance" : "input_impedance") : analysis.kind}, behavior, library: {allowed: spec.constraints.element_types || [], required: spec.constraints.required_elements || [], real_components: spec.constraints.real_components || [], model_bindings: spec.constraints.model_bindings || []}};
}

function renderDesignSummary() {
  const target = $("#design-summary");
  if (!target) return;
  const spec = pbdlView(specFromForm());
  const allowed = spec.library.allowed.join(", ") || "未选择";
  const required = spec.library.required.join(", ") || "无";
  const dc = ["dc_conversion", "isolated_dc_conversion"].includes(spec.behavior.kind);
  const behaviorText = dc ? `${spec.behavior.input_voltage_v} V → ${spec.behavior.output_voltage_v} V / ${spec.behavior.output_current_a} A${spec.behavior.galvanic_isolation ? " · 隔离" : ""}` : `${spec.behavior.kind} · ${format(spec.behavior.cutoff_hz || spec.behavior.center_hz || spec.behavior.ohms || spec.behavior.transimpedance_ohm || 0)}`;
  const boundary = dc ? "可提交有界功率拓扑搜索、仿真与统一约束验证" : spec.ports === 2 ? "可提交当前双端口频域引擎" : "规格可保存；需要多端口联合求解器或 PBDL 分阶段执行";
  target.innerHTML = `<dl><div><dt>任务</dt><dd>${escapeHtml(spec.name)}</dd></div><div><dt>接口</dt><dd>${spec.interface.ports.length} 个端口 · ${spec.interface.relations.length} 个端口关系</dd></div><div><dt>推导行为</dt><dd>${escapeHtml(behaviorText)}</dd></div><div><dt>元件</dt><dd>允许 ${escapeHtml(allowed)} · 必须 ${escapeHtml(required)}</dd></div><div><dt>执行边界</dt><dd>${boundary}</dd></div></dl>`;
}

function escapeHtml(text) { return String(text).replace(/[&<>"']/g, (char) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[char])); }
function applySpec(spec) {
  canvasHistory.length = 0;
  updateCanvasUndoButton();
  $("#name").value = spec.name || "browser_synthesis";
  if (Array.isArray(spec.ports) && Array.isArray(spec.analyses) && Array.isArray(spec.targets)) spec = pbdlView(spec);
  if (spec.interface && Array.isArray(spec.interface.ports)) {
    replaceDraftCollection("ports", spec.interface.ports.map((port, index) => {
      const terminals = port.terminals || [];
      const conditions = port.conditions || [];
      const voltage = conditions.find((condition) => condition.quantity === "voltage");
      const current = conditions.find((condition) => condition.quantity === "current");
      const terminalPair = Array.isArray(terminals) ? {positive: (terminals[0] || {}).name, negative: (terminals[1] || {}).name} : terminals;
      const constraints = Array.isArray(port.variable_constraints) ? port.variable_constraints.map((item, constraintIndex) => ({id: item.id || `c${index}_${constraintIndex}`, ...item})) : [
        ...(voltage ? [{id: `c${index}_v`, variable: "v", analysis: port.domain === "dc" ? "dc_operating_point" : "small_signal_ac", operator: "equal", value: voltage.value}] : []),
        ...(current ? [{id: `c${index}_i`, variable: "i", analysis: port.domain === "dc" ? "dc_operating_point" : "small_signal_ac", operator: current.operator === ">=" ? "minimum" : current.operator === "<=" ? "maximum" : "equal", value: current.value}] : [])
      ];
      const variables = (port.variables || []).map((item) => ({id: item.id || item.name, name: item.name || item.id, quantity: item.quantity, role: item.role, unit: item.unit}));
      return {id: port.port_id || port.id || `p${index + 1}`, name: port.name || `port_${index + 1}`, positive: terminalPair.positive || `p${index + 1}`, negative: terminalPair.negative || "0", role: port.role || "bidirectional", domain: port.domain || "unspecified", position: port.layout || port.position, variables, constraints, excitation: port.excitation || null, verification: "pending"};
    }));
    const savedRelations = spec.interface.relations || spec.interface.measurements;
    if (Array.isArray(savedRelations)) {
      replaceDraftCollection("measurements", savedRelations.map((measurement, index) => ({id: measurement.id || `m${index + 1}`, kind: measurement.kind || "voltage_transfer", source_port: (portDefinitions.find((port) => port.name === measurement.source_port) || portDefinitions[0] || {}).id, response_port: (portDefinitions.find((port) => port.name === measurement.response_port) || portDefinitions[1] || portDefinitions[0] || {}).id})));
    }
    if (Array.isArray(spec.functions) && spec.functions.length) {
      replaceDraftCollection("functions", spec.functions.map((fn) => ({
        ...fn,
        relation_kind: fn.relation_kind || relationKindForBodyName(fn.body?.name),
        source_port: parseFunctionPortId(fn.inputs?.[0], portDefinitions[0]?.id),
        response_port: parseFunctionPortId(fn.outputs?.[0], portDefinitions[1]?.id),
        relation_id: fn.relation_id || fn.function_id,
        verification: "pending"
      })));
      syncRelationsFromFunctions();
    } else syncFunctionDefinitionsFromRelations();
  } else {
    ensurePortCount(Number(spec.ports || 2));
  }
  $("#port-count").value = String(Math.min(Math.max(portDefinitions.length, 2), 4));
  renderPorts();
  $("#analysis-kind").value = (spec.analysis || {}).kind || "voltage_transfer";
  $("#behavior-kind").value = spec.behavior.kind || "lowpass";
  const b = spec.behavior;
  $("#frequency").value = b.cutoff_hz || b.center_hz || 1000;
  $("#gain").value = b.gain || b.transimpedance_ohm || b.ohms || b.resistance_ohm || 1;
  const range = b.frequency_range_hz || [10, 100000];
  $("#f-min").value = range[0]; $("#f-max").value = range[1];
  $("#max-components").value = (spec.optimization || {}).max_components || 5;
  $("#top-k").value = (spec.optimization || {}).top_k || 3;
  $("#max-iterations").value = (spec.optimization || {}).max_iterations || 45;
  const allowed = (spec.library || {}).allowed || [];
  document.querySelectorAll("input[name=element]").forEach((node) => { node.checked = allowed.includes(node.value); });
  const required = (spec.library || {}).required || [];
  document.querySelectorAll("input[name=required-element]").forEach((node) => { node.checked = required.includes(node.value); });
  replaceDraftCollection("realComponents", (spec.library || {}).real_components || (spec.library || {}).selected_components || []);
  replaceDraftCollection("modelBindings", (spec.library || {}).model_bindings || []);
  renderModelBindings();
  const savedExcitations = b.excitations || (b.excitation ? [b.excitation] : []);
  for (const excitation of savedExcitations) {
    const port = portDefinitions.find((item) => item.name === excitation.port);
    if (port) port.excitation = {quantity: excitation.quantity || port.quantity, waveform: excitation.waveform};
  }
  const selectedPort = portDefinitions.find((port) => port.excitation) || portDefinitions.find((port) => port.role === "input") || portDefinitions[0];
  if (selectedPort) { $("#signal-port").value = selectedPort.id; applyWaveform((selectedPort.excitation || {}).waveform, selectedPort.id); }
  $("#json-editor").value = JSON.stringify(spec, null, 2);
}

function applyWaveform(waveform, port) {
  if (!waveform) { updateWaveform(); return; }
  if (port) $("#signal-port").value = port;
  $("#waveform-kind").value = waveform.kind || "sine";
  if (waveform.kind === "sine") {
    $("#signal-frequency").value = waveform.frequency_hz ?? 1000000; $("#signal-amplitude").value = waveform.amplitude_v ?? 1; $("#signal-offset").value = waveform.offset_v ?? 0; $("#signal-phase").value = waveform.phase_deg ?? 0;
  } else if (waveform.kind === "pulse") {
    $("#pulse-frequency").value = waveform.frequency_hz ?? 1000000; $("#pulse-duty").value = 100 * (waveform.duty_cycle ?? .5); $("#pulse-low").value = waveform.low_v ?? 0; $("#pulse-high").value = waveform.high_v ?? 3.3; $("#pulse-rise").value = waveform.rise_time_s ?? 2e-9; $("#pulse-fall").value = waveform.fall_time_s ?? 2e-9;
  } else {
    $("#sample-period").value = waveform.period_s ?? 1e-6; $("#sample-values").value = (waveform.value_v || [0, 1, 0, -1, 0]).join(", ");
  }
  updateWaveform();
}

function updateWaveform() {
  const kind = value("waveform-kind");
  $("#sine-fields").hidden = kind !== "sine"; $("#pulse-fields").hidden = kind !== "pulse"; $("#sample-fields").hidden = kind !== "periodic_samples";
  const waveform = waveformFromForm(); const points = []; const count = 160;
  const activePort = portDefinitions.find((port) => port.id === value("signal-port"));
  if (activePort) activePort.excitation = {quantity: "voltage", waveform};
  let period = waveform.period_s || 1 / Math.max(waveform.frequency_hz || 1, 1e-30);
  for (let i = 0; i < count; i++) {
    const ratio = i / (count - 1); let y = 0;
    if (kind === "sine") y = waveform.offset_v + waveform.amplitude_v * Math.sin(2 * Math.PI * ratio + waveform.phase_deg * Math.PI / 180);
    else if (kind === "pulse") {
      const rise = Math.min(waveform.rise_time_s / period, .45), fall = Math.min(waveform.fall_time_s / period, .45), duty = waveform.duty_cycle;
      if (ratio < rise && rise > 0) y = waveform.low_v + (waveform.high_v - waveform.low_v) * ratio / rise;
      else if (ratio < duty) y = waveform.high_v;
      else if (ratio < duty + fall && fall > 0) y = waveform.high_v + (waveform.low_v - waveform.high_v) * (ratio - duty) / fall;
      else y = waveform.low_v;
    } else {
      const samples = waveform.value_v.length ? waveform.value_v : [0]; const index = ratio * (samples.length - 1); const lo = Math.floor(index), hi = Math.min(lo + 1, samples.length - 1); y = samples[lo] + (samples[hi] - samples[lo]) * (index - lo);
    }
    points.push({x: ratio, y});
  }
  const values = points.map((point) => point.y); let yMin = Math.min(...values), yMax = Math.max(...values); if (yMax === yMin) { yMax += 1; yMin -= 1; } const pad = (yMax - yMin) * .12;
  const sx = (x) => 48 + x * 454, sy = (y) => 14 + (yMax + pad - y) / (yMax - yMin + 2 * pad) * 132;
  const path = points.map((point, index) => `${index ? "L" : "M"}${sx(point.x).toFixed(2)},${sy(point.y).toFixed(2)}`).join(" ");
  $("#waveform-preview").innerHTML = `<title>输入信号波形</title><desc>显示所定义输入信号的一个周期</desc><g class="wave-grid"><line x1="48" y1="146" x2="502" y2="146"/><line x1="48" y1="14" x2="48" y2="146"/><line x1="48" y1="${sy(0)}" x2="502" y2="${sy(0)}"/><text x="48" y="166">0</text><text x="470" y="166">${formatTime(period)}</text><text x="5" y="20">${formatVoltage(yMax)}</text><text x="5" y="145">${formatVoltage(yMin)}</text></g><path class="wave-line" d="${path}"/>`;
  $("#waveform-summary").textContent = `${kind === "sine" ? "正弦" : kind === "pulse" ? "脉冲" : "周期采样"} · 周期 ${formatTime(period)} · ${formatVoltage(yMin)} 至 ${formatVoltage(yMax)}`;
}

function ensurePortCount(count) {
  count = Math.max(1, Math.min(count, 16));
  while (portDefinitions.length < count) {
    const index = portDefinitions.length + 1;
    portDefinitions.push({id: `p${Date.now()}_${index}`, name: `port_${index}`, positive: `p${index}`, negative: "0", role: "bidirectional", domain: "unspecified", position: {edge: index % 2 ? "left" : "right", offset: .25 + (index / Math.max(count, 2)) * .5}, variables: [defaultVariable("v"), defaultVariable("i"), defaultVariable("p")], constraints: [], excitation: null, verification: "pending"});
  }
  while (portDefinitions.length > count) portDefinitions.pop();
}

function primaryRelationKind() {
  return functionRelationKind(primaryFunction()) || "voltage_transfer";
}

function syncAnalysisKindControl() {
  const control = $("#analysis-kind");
  if (!control) return;
  control.value = primaryRelationKind();
  control.disabled = true;
  control.title = "分析类型由端口关系决定";
}

function renderMeasurements() {
  syncRelationsFromFunctions();
  const options = (selected) => portDefinitions.map((port) => `<option value="${port.id}" ${port.id === selected ? "selected" : ""}>${port.name}</option>`).join("");
  $("#measurement-editor").innerHTML = measurementDefinitions.map((measurement, index) => `<div class="measurement-row" data-measurement-id="${measurement.id}"><b>${index + 1}</b><select data-measurement-field="kind"><option value="voltage_transfer" ${measurement.kind === "voltage_transfer" ? "selected" : ""}>电压传输 V/V</option><option value="transimpedance" ${measurement.kind === "transimpedance" ? "selected" : ""}>跨阻 V/A</option><option value="input_impedance" ${measurement.kind === "input_impedance" ? "selected" : ""}>输入阻抗</option><option value="output_impedance" ${measurement.kind === "output_impedance" ? "selected" : ""}>输出阻抗</option><option value="s_parameter" ${measurement.kind === "s_parameter" ? "selected" : ""}>S 参数</option><option value="galvanic_isolation" ${measurement.kind === "galvanic_isolation" ? "selected" : ""}>电气隔离</option></select><select data-measurement-field="source_port" aria-label="端口 A">${options(measurement.source_port)}</select><span>↔</span><select data-measurement-field="response_port" aria-label="端口 B">${options(measurement.response_port)}</select><button type="button" data-remove-measurement="${measurement.id}" aria-label="删除端口关系">×</button></div>`).join("");
  $("#measurement-editor").querySelectorAll("[data-measurement-field]").forEach((node) => node.addEventListener("change", () => {
    const row = node.closest(".measurement-row"), relation = measurementDefinitions.find((item) => item.id === row.dataset.measurementId), fn = functionDefinitions.find((item) => item.relation_id === relation?.id);
    if (!relation || !fn) return;
    const field = node.dataset.measurementField, next = node.value;
    if (field === "kind") {
      fn.relation_kind = next;
      const nextName = next === "galvanic_isolation" ? "galvanic_isolation" : next === "voltage_transfer" && ["lowpass", "highpass", "bandpass"].includes(fn.body?.name) ? fn.body.name : functionBodyName(next);
      fn.body = {...fn.body, kind: "builtin", name: nextName, parameters: {...(fn.body?.parameters || {})}};
    } else if (field === "source_port" || field === "response_port") fn[field] = next;
    ensureIsolationReferenceNodes();
    syncRelationsFromFunctions();
    syncBehaviorControlsFromFunction(fn);
    renderPorts();
    renderMeasurements();
    renderCanvas();
    refreshEditor();
  }));
  $("#measurement-editor").querySelectorAll("[data-remove-measurement]").forEach((node) => node.addEventListener("click", () => {
    replaceDraftCollection("functions", functionDefinitions.filter((item) => item.relation_id !== node.dataset.removeMeasurement));
    syncRelationsFromFunctions();
    renderMeasurements();
    refreshEditor();
  }));
  syncAnalysisKindControl();
}

function renderConstraint(constraint) {
  const range = constraint.operator === "interval";
  const field = (name, label) => `<label>${label}<input data-constraint-field="${name}" type="number" step="any" value="${constraint[name] ?? ""}"></label>`;
  return `<div class="constraint-row" data-constraint-id="${constraint.id}"><label>物理量<select data-constraint-field="variable"><option value="v" ${constraint.variable === "v" ? "selected" : ""}>电压 v</option><option value="i" ${constraint.variable === "i" ? "selected" : ""}>电流 i</option><option value="p" ${constraint.variable === "p" ? "selected" : ""}>功率 p</option></select></label><label>分析上下文<select data-constraint-field="analysis"><option value="dc_operating_point" ${constraint.analysis === "dc_operating_point" ? "selected" : ""}>直流工作点</option><option value="small_signal_ac" ${constraint.analysis === "small_signal_ac" ? "selected" : ""}>小信号 AC</option><option value="transient" ${constraint.analysis === "transient" ? "selected" : ""}>瞬态</option><option value="periodic_steady_state" ${constraint.analysis === "periodic_steady_state" ? "selected" : ""}>周期稳态</option></select></label><label>约束<select data-constraint-field="operator"><option value="equal" ${constraint.operator === "equal" ? "selected" : ""}>等于</option><option value="minimum" ${constraint.operator === "minimum" ? "selected" : ""}>不小于</option><option value="maximum" ${constraint.operator === "maximum" ? "selected" : ""}>不大于</option><option value="interval" ${range ? "selected" : ""}>区间</option></select></label>${range ? field("minimum", "下限") + field("maximum", "上限") : field("value", "数值")}<span class="constraint-unit">${unitForVariable(constraint.variable)}</span><button type="button" data-remove-constraint="${constraint.id}" aria-label="删除约束">×</button></div>`;
}

function renderPorts() {
  const firstPort = portDefinitions[0], secondPort = portDefinitions[1] || firstPort;
  for (const relation of measurementDefinitions) { if (!portDefinitions.some((port) => port.id === relation.source_port)) relation.source_port = firstPort.id; if (!portDefinitions.some((port) => port.id === relation.response_port)) relation.response_port = secondPort.id; }
  const constraintSummary = (port) => {
    const constraints = port.constraints || [];
    if (!constraints.length) return "暂无约束";
    return constraints.map((constraint) => `${constraint.variable} ${constraint.operator === "equal" ? "=" : constraint.operator === "minimum" ? "≥" : constraint.operator === "maximum" ? "≤" : "区间"} ${constraint.value ?? ""}${unitForVariable(constraint.variable)}`).join(" · ");
  };
  $("#port-editor").innerHTML = portDefinitions.map((port, index) => `<div class="port-row" data-port-id="${port.id}">
    <div class="port-row-head"><b>${index + 1}. ${escapeHtml(port.name)}</b><button type="button" data-remove="${port.id}" aria-label="删除端口">×</button></div>
    <div class="port-summary"><span>${port.role === "input" ? "输入" : port.role === "output" ? "输出" : port.role === "power" ? "供电" : "双向"}</span><span>${port.domain === "ac" ? "小信号 AC" : port.domain === "dc" ? "直流" : port.domain === "transient" ? "瞬态" : port.domain === "periodic" ? "周期稳态" : "未指定"}</span><span>${escapeHtml(constraintSummary(port))}</span></div>
    <details class="port-interface-details"><summary>编辑端子与接口属性</summary><div class="port-fields"><label>端口名称<input data-field="name" value="${escapeHtml(port.name)}"></label><label>正端子 (+)<input data-field="positive" value="${escapeHtml(port.positive)}"></label><label>负端子 (-)<input data-field="negative" value="${escapeHtml(port.negative)}"></label></div></details>
    <div class="port-semantics"><code>v = V(+) - V(-)</code><span><code>i &gt; 0</code> 表示电流流入正端子</span><span><code>p = v × i</code>，正值表示端口吸收功率</span></div>
    <details class="port-constraints"><summary>编辑约束 <span class="constraint-summary">· ${escapeHtml(constraintSummary(port))}</span></summary><div class="constraint-list">${(port.constraints || []).map(renderConstraint).join("")}</div><button type="button" class="add-constraint" data-add-constraint="${port.id}">+ 添加变量约束</button></details>
  </div>`).join("");
  $("#port-editor").querySelectorAll(".port-row").forEach((row) => {
    const port = portDefinitions.find((item) => item.id === row.dataset.portId);
    row.querySelector(".port-fields").insertAdjacentHTML("beforeend", `<label>方向<select data-field="role"><option value="input" ${port.role === "input" ? "selected" : ""}>进入电路</option><option value="output" ${port.role === "output" ? "selected" : ""}>离开电路</option><option value="bidirectional" ${!port.role || port.role === "bidirectional" ? "selected" : ""}>双向</option><option value="power" ${port.role === "power" ? "selected" : ""}>供电</option><option value="reference" ${port.role === "reference" ? "selected" : ""}>参考</option></select></label><label>领域<select data-field="domain"><option value="dc" ${port.domain === "dc" ? "selected" : ""}>直流</option><option value="ac" ${port.domain === "ac" ? "selected" : ""}>小信号 AC</option><option value="transient" ${port.domain === "transient" ? "selected" : ""}>瞬态</option><option value="periodic" ${port.domain === "periodic" ? "selected" : ""}>周期稳态</option><option value="unspecified" ${!port.domain || port.domain === "unspecified" ? "selected" : ""}>未指定</option></select></label>`);
  });
  $("#port-editor").querySelectorAll("[data-field]").forEach((node) => node.addEventListener(node.tagName === "SELECT" ? "change" : "input", () => { const port = portDefinitions.find((item) => item.id === node.closest(".port-row").dataset.portId); port[node.dataset.field] = node.value; updateDomainVisibility(); refreshEditor(); }));
  $("#port-editor").querySelectorAll("[data-constraint-field]").forEach((node) => node.addEventListener(node.tagName === "SELECT" ? "change" : "input", () => { const port = portDefinitions.find((item) => item.id === node.closest(".port-row").dataset.portId); const constraint = port.constraints.find((item) => item.id === node.closest(".constraint-row").dataset.constraintId); constraint[node.dataset.constraintField] = node.type === "number" ? Number(node.value) : node.value; if (node.tagName === "SELECT") renderPorts(); else { updateDomainVisibility(); refreshEditor(); } }));
  $("#port-editor").querySelectorAll("[data-add-constraint]").forEach((node) => node.addEventListener("click", () => { const port = portDefinitions.find((item) => item.id === node.dataset.addConstraint); port.constraints.push({id: `c${Date.now()}`, variable: "v", analysis: "dc_operating_point", operator: "equal", value: 0}); renderPorts(); refreshEditor(); }));
  $("#port-editor").querySelectorAll("[data-remove-constraint]").forEach((node) => node.addEventListener("click", () => { const port = portDefinitions.find((item) => item.id === node.closest(".port-row").dataset.portId); port.constraints = port.constraints.filter((item) => item.id !== node.dataset.removeConstraint); renderPorts(); refreshEditor(); }));
  $("#port-editor").querySelectorAll("[data-remove]").forEach((node) => node.addEventListener("click", () => { if (portDefinitions.length > 1) { replaceDraftCollection("ports", portDefinitions.filter((port) => port.id !== node.dataset.remove)); renderPorts(); refreshEditor(); } }));
  const previous = $("#signal-port").value; $("#signal-port").innerHTML = portDefinitions.map((port) => `<option value="${port.id}">${escapeHtml(port.name)}</option>`).join(""); if (portDefinitions.some((port) => port.id === previous)) $("#signal-port").value = previous;
  $("#port-count").value = String(portDefinitions.length); renderMeasurements(); updateDomainVisibility(); $("#port-capability").textContent = "每个电气端口同时具有电压、电流和功率；分析上下文属于约束，不属于端口。";
}

function formatTime(seconds) { if (seconds < 1e-9) return `${(seconds * 1e12).toPrecision(3)} ps`; if (seconds < 1e-6) return `${(seconds * 1e9).toPrecision(3)} ns`; if (seconds < 1e-3) return `${(seconds * 1e6).toPrecision(3)} us`; return `${(seconds * 1e3).toPrecision(3)} ms`; }
function formatVoltage(volts) { return `${Number(volts).toPrecision(3)} V`; }

function format(value) { return Number(value).toPrecision(4); }

const CANVAS_BOUNDARY = {x: 180, y: 60, size: 400};
const CANVAS_VARIABLES = [
  {id: "v", label: "电压 v", quantity: "voltage", role: "potential", unit: "V"},
  {id: "i", label: "电流 i", quantity: "current", role: "flow", unit: "A"},
  {id: "p", label: "功率 p", quantity: "power", role: "derived", unit: "W"}
];

function clamp(value, min, max) { return Math.min(max, Math.max(min, value)); }
function functionRelationKind(fn) {
  if (fn.relation_kind) return fn.relation_kind;
  const name = fn.body?.name || "";
  if (name === "transimpedance") return "transimpedance";
  if (name === "impedance") return "input_impedance";
  return "voltage_transfer";
}
function functionBodyName(relationKind, selectedName = "lowpass") {
  if (relationKind === "transimpedance") return "transimpedance";
  if (relationKind === "input_impedance" || relationKind === "output_impedance") return "impedance";
  return selectedName || "lowpass";
}
function relationKindForBodyName(name, fallback = "voltage_transfer") {
  if (name === "transimpedance") return "transimpedance";
  if (name === "impedance") return fallback === "output_impedance" ? "output_impedance" : "input_impedance";
  if (name === "galvanic_isolation") return "galvanic_isolation";
  return "voltage_transfer";
}
function functionLabel(fn) { return fn.body?.name || fn.body?.kind || "function"; }
function parseFunctionPortId(ref, fallback) {
  const text = typeof ref === "string" ? ref : ref?.ref || "";
  const raw = text.split(".")[1] || text;
  return portDefinitions.find((port) => port.id === raw || port.name === raw)?.id || fallback;
}
function functionForRelation(relation) {
  return functionDefinitions.find((fn) => fn.relation_id === relation.id) || functionDefinitions.find((fn) => fn.source_port === relation.source_port && fn.response_port === relation.response_port);
}
function syncFunctionDefinitionsFromRelations() {
  // functionDefinitions is the canonical source. This function only exists as
  // a migration guard for an old draft that has relations but no functions.
  if (functionDefinitions.length || !measurementDefinitions.length) return functionDefinitions;
  replaceDraftCollection("functions", measurementDefinitions.map((relation, index) => ({
    function_id: index === 0 ? "main_transfer" : `function_${relation.id || Date.now()}`,
    relation_id: relation.id,
    relation_kind: relation.kind || "voltage_transfer",
    role: "target",
    domain: "frequency",
    source_port: relation.source_port,
    response_port: relation.response_port,
    body: {kind: "builtin", name: functionBodyName(relation.kind, relation.kind === "galvanic_isolation" ? "galvanic_isolation" : "lowpass"), parameters: {cutoff: {value: Number(value("frequency") || 1000), unit: "Hz"}, gain: {value: Number(value("gain") || 1), unit: "1"}}},
    axis: {domain: "frequency", variable: "frequency", unit: "Hz", start: Number(value("f-min") || 10), stop: Number(value("f-max") || 100000), points: 96},
    verification: "pending"
  })));
  return functionDefinitions;
}
function syncRelationsFromFunctions() {
  const relations = functionDefinitions.map((fn, index) => {
    const axis = fn.axis || {};
    const relation = {id: fn.relation_id || `m_${fn.function_id || index}`, kind: functionRelationKind(fn), source_port: fn.source_port, response_port: fn.response_port};
    if (fn.domain !== "dc") relation.frequency_hz = [Number(axis.start || value("f-min") || 10), Number(axis.stop || value("f-max") || 100000), Number(axis.points || 96)];
    return relation;
  });
  replaceDraftCollection("measurements", relations);
  return measurementDefinitions;
}
function variableById(port, id) { return (port.variables || []).find((item) => (item.id || item.name) === id); }
function defaultVariable(id) { return {...(CANVAS_VARIABLES.find((variable) => variable.id === id) || CANVAS_VARIABLES[0])}; }
function canvasPoint(event) {
  const svg = $("#circuit-canvas"), rect = svg.getBoundingClientRect();
  return {x: (event.clientX - rect.left) / rect.width * 760, y: (event.clientY - rect.top) / rect.height * 520};
}
function perimeterPosition(point) {
  const {x, y, size} = CANVAS_BOUNDARY;
  const distances = [{edge: "left", distance: Math.abs(point.x - x), offset: (point.y - y) / size}, {edge: "right", distance: Math.abs(point.x - (x + size)), offset: (point.y - y) / size}, {edge: "top", distance: Math.abs(point.y - y), offset: (point.x - x) / size}, {edge: "bottom", distance: Math.abs(point.y - (y + size)), offset: (point.x - x) / size}];
  const closest = distances.sort((a, b) => a.distance - b.distance)[0];
  return {edge: closest.edge, offset: clamp(closest.offset, .06, .94)};
}
function portAnchor(port, index = 0) {
  const position = port.position || {edge: index % 2 ? "right" : "left", offset: .25 + (index / Math.max(portDefinitions.length, 2)) * .5};
  const {x, y, size} = CANVAS_BOUNDARY, offset = clamp(Number(position.offset ?? .5), .04, .96);
  if (position.edge === "right") return {edge: "right", x: x + size, y: y + offset * size, outX: x + size + 42, outY: y + offset * size, normalX: 1, normalY: 0};
  if (position.edge === "top") return {edge: "top", x: x + offset * size, y, outX: x + offset * size, outY: y - 42, normalX: 0, normalY: -1};
  if (position.edge === "bottom") return {edge: "bottom", x: x + offset * size, y: y + size, outX: x + offset * size, outY: y + size + 42, normalX: 0, normalY: 1};
  return {edge: "left", x, y: y + offset * size, outX: x - 42, outY: y + offset * size, normalX: -1, normalY: 0};
}
function portStatusClass(port) { return `port-status-${port.verification || "pending"}`; }

function chart(svg, x, series, unit) {
  const width = 760, height = 210, left = 58, right = 16, top = 12, bottom = 32;
  const yValues = series.flatMap((entry) => entry.values);
  const yMin = Math.min(...yValues), yMax = Math.max(...yValues), pad = Math.max((yMax - yMin) * .12, 1);
  const minX = Math.log10(x[0]), maxX = Math.log10(x[x.length - 1]);
  const scaleX = (v) => left + (Math.log10(v) - minX) / (maxX - minX) * (width - left - right);
  const scaleY = (v) => top + (yMax + pad - v) / (yMax - yMin + 2 * pad) * (height - top - bottom);
  const path = (values) => values.map((v, i) => `${i ? "L" : "M"}${scaleX(x[i]).toFixed(2)},${scaleY(v).toFixed(2)}`).join(" ");
  const decades = Array.from({length: Math.floor(maxX) - Math.ceil(minX) + 1}, (_, i) => Math.ceil(minX) + i);
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.innerHTML = `<g class="grid">${decades.map(d => `<line x1="${scaleX(10 ** d)}" y1="${top}" x2="${scaleX(10 ** d)}" y2="${height-bottom}"/><text x="${scaleX(10 ** d)}" y="${height-9}" text-anchor="middle">10${d}</text>`).join("")}</g><line class="axis" x1="${left}" y1="${height-bottom}" x2="${width-right}" y2="${height-bottom}"/><text class="axis-label" x="8" y="${top + 10}">${unit}</text>${series.map((entry, index) => `<path class="line" style="stroke:${entry.color};stroke-dasharray:${entry.dashed ? "5 4" : entry.dash || "none"}" d="${path(entry.values)}"/><text class="legend" style="fill:${entry.color}" x="${70 + index * 64}" y="24">${entry.label}</text>`).join("")}`;
}

function redrawCharts() {
  const selected = [...compared].sort((a, b) => a - b).map((index, colorIndex) => {
    const candidate = latest.candidates[index];
    const colors = ["#58b7a5", "#c475d6", "#609ee8"];
    const dashes = ["", "8 5", "2 4", "12 4 2 4"];
    return { candidate, color: colors[colorIndex] || "#9eabb5", dash: dashes[colorIndex] || "4 4" };
  });
  const first = latest.candidates[0];
  const target = { values: first.plot.target_db, color: "#d9a441", dashed: true, label: "目标" };
  chart($("#magnitude-chart"), first.plot.frequency_hz, [target, ...selected.map(({candidate, color, dash}) => ({values: candidate.plot.response_db, color, dash, label: `#${candidate.rank}`}))], "dB");
  chart($("#phase-chart"), first.plot.frequency_hz, [{ values: first.plot.target_phase_deg, color: "#d9a441", dashed: true, label: "目标" }, ...selected.map(({candidate, color, dash}) => ({values: candidate.plot.response_phase_deg, color, dash, label: `#${candidate.rank}`}))], "deg");
}

function renderCanvas() {
  const portLayer = $("#port-layer"), functionLayer = $("#function-layer");
  if (!portLayer || !functionLayer) return;
  functionLayer.innerHTML = functionDefinitions.map((fn) => {
    const source = portDefinitions.find((port) => port.id === fn.source_port), target = portDefinitions.find((port) => port.id === fn.response_port);
    if (!source || !target) return "";
    const a = portAnchor(source, portDefinitions.indexOf(source)), b = portAnchor(target, portDefinitions.indexOf(target));
    const bend = Math.max(80, Math.abs(b.outX - a.outX) * .25);
    const c1x = a.outX + a.normalX * bend, c1y = a.outY + a.normalY * bend;
    const c2x = b.outX + b.normalX * bend, c2y = b.outY + b.normalY * bend;
    const selected = selectedObject.kind === "function" && selectedObject.id === fn.function_id ? " selected" : "";
    const curve = `M${a.outX},${a.outY} C${c1x},${c1y} ${c2x},${c2y} ${b.outX},${b.outY}`;
    return `<g class="canvas-function function-status-${fn.verification || "pending"}${selected}" data-function-id="${escapeHtml(fn.function_id)}" tabindex="0" role="button" aria-label="函数 ${escapeHtml(functionLabel(fn))}"><path class="canvas-function-hit" d="${curve}"></path><path class="canvas-function-line" d="${curve}" marker-end="url(#function-arrow)"></path><text x="${(a.outX + b.outX) / 2}" y="${(a.outY + b.outY) / 2 - 8}" text-anchor="middle">${escapeHtml(functionLabel(fn))}</text></g>`;
  }).join("");
  portLayer.innerHTML = portDefinitions.map((port, index) => {
    const anchor = portAnchor(port, index), selected = selectedObject.kind === "port" && selectedObject.id === port.id ? " selected" : "";
    const labelX = anchor.outX + anchor.normalX * 12, labelY = anchor.outY + anchor.normalY * 12 + (anchor.edge === "left" || anchor.edge === "right" ? 4 : 0);
    const horizontal = anchor.edge === "left" || anchor.edge === "right";
    return `<g class="canvas-port ${portStatusClass(port)}${selected}" data-port-id="${escapeHtml(port.id)}" tabindex="0" role="button" aria-label="端口 ${escapeHtml(port.name)}"><line x1="${anchor.x}" y1="${anchor.y}" x2="${anchor.outX}" y2="${anchor.outY}"></line><rect class="canvas-port-hit" x="${anchor.outX - (horizontal ? 20 : 16)}" y="${anchor.outY - (horizontal ? 16 : 20)}" width="${horizontal ? 40 : 32}" height="${horizontal ? 32 : 40}" rx="4"></rect><rect x="${anchor.outX - (horizontal ? 12 : 8)}" y="${anchor.outY - (horizontal ? 8 : 12)}" width="${horizontal ? 24 : 16}" height="${horizontal ? 16 : 24}" rx="2"></rect><text x="${labelX}" y="${labelY}" text-anchor="${anchor.edge === "left" ? "end" : anchor.edge === "right" ? "start" : "middle"}">${escapeHtml(port.name)}</text></g>`;
  }).join("");
  portLayer.querySelectorAll(".canvas-port").forEach((node) => {
    node.addEventListener("click", (event) => { event.stopPropagation(); handleCanvasPortClick(node.dataset.portId); });
    node.addEventListener("pointerdown", (event) => { event.stopPropagation(); event.preventDefault(); draggingPortId = node.dataset.portId; dragHistorySnapshot = captureCanvasState(); dragMoved = false; node.setPointerCapture?.(event.pointerId); });
    node.addEventListener("keydown", (event) => handleCanvasObjectKeydown(event, "port", node.dataset.portId));
  });
  functionLayer.querySelectorAll(".canvas-function").forEach((node) => {
    node.addEventListener("click", (event) => { event.stopPropagation(); selectCanvasObject("function", node.dataset.functionId); });
    node.addEventListener("keydown", (event) => handleCanvasObjectKeydown(event, "function", node.dataset.functionId));
  });
}
function setCanvasMode(mode) {
  canvasMode = mode;
  functionLinkStart = null;
  document.querySelectorAll(".canvas-tool").forEach((button) => button.classList.toggle("active", button.id === `canvas-${mode}-mode`));
  const status = $("#canvas-status");
  if (status) status.textContent = mode === "add" ? "点击黑盒边界放置端口" : mode === "function" ? "依次点击源端口和目标端口" : "点击端口、函数箭头或空白区域编辑";
}
function selectCanvasObject(kind, id = null) {
  selectedObject = {kind, id};
  renderCanvas();
  renderInspector();
}

function deleteSelectedCanvasObject() {
  if (selectedObject.kind === "port") {
    if (portDefinitions.length <= 1) return;
    pushCanvasHistory();
    const id = selectedObject.id;
    replaceDraftCollection("ports", portDefinitions.filter((port) => port.id !== id));
    replaceDraftCollection("measurements", measurementDefinitions.filter((relation) => relation.source_port !== id && relation.response_port !== id));
    replaceDraftCollection("functions", functionDefinitions.filter((fn) => fn.source_port !== id && fn.response_port !== id));
    renderPorts();
  } else if (selectedObject.kind === "function") {
    pushCanvasHistory();
    replaceDraftCollection("functions", functionDefinitions.filter((fn) => fn.function_id !== selectedObject.id));
    syncRelationsFromFunctions();
    renderMeasurements();
  } else return;
  selectCanvasObject("circuit");
  refreshEditor();
}

function moveCanvasPortByKey(id, key) {
  const port = portDefinitions.find((item) => item.id === id);
  if (!port) return;
  const position = port.position || {edge: "left", offset: .5};
  const along = (position.edge === "top" || position.edge === "bottom") ? ["ArrowLeft", "ArrowRight"].includes(key) : ["ArrowUp", "ArrowDown"].includes(key);
  if (!along) return;
  const direction = ["ArrowLeft", "ArrowUp"].includes(key) ? -.04 : .04;
  pushCanvasHistory();
  port.position = {...position, offset: clamp(Number(position.offset || .5) + direction, .06, .94)};
  renderCanvas();
  announce(`${port.name} 已移动到边界 ${port.position.edge}`);
}

function handleCanvasObjectKeydown(event, kind, id) {
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "z") {
    event.preventDefault();
    undoCanvasChange();
    return;
  }
  if (["Enter", " "].includes(event.key)) {
    event.preventDefault();
    if (kind === "port") handleCanvasPortClick(id); else selectCanvasObject(kind, id);
    return;
  }
  if (event.key === "Escape") {
    event.preventDefault();
    functionLinkStart = null;
    setCanvasMode("select");
    announce("已取消连接模式");
    return;
  }
  if (event.key === "Delete" || event.key === "Backspace") {
    event.preventDefault();
    deleteSelectedCanvasObject();
    return;
  }
  if (kind === "port" && event.key.startsWith("Arrow")) {
    event.preventDefault();
    moveCanvasPortByKey(id, event.key);
  }
}
function addCanvasPort(point) {
  pushCanvasHistory();
  const position = perimeterPosition(point), index = portDefinitions.length + 1, id = `p${Date.now()}`;
  const edgePrefix = {left: "in", right: "out", top: "vcc", bottom: "aux"}[position.edge];
  portDefinitions.push({id, name: `${edgePrefix}${index}`, positive: `${edgePrefix}${index}`, negative: "0", role: position.edge === "left" ? "input" : position.edge === "right" ? "output" : "bidirectional", domain: "unspecified", position, variables: [defaultVariable("v"), defaultVariable("i"), defaultVariable("p")], constraints: [], excitation: null, verification: "pending"});
  renderPorts();
  selectCanvasObject("port", id);
  setCanvasMode("select");
  refreshEditor();
}
function handleCanvasPortClick(id) {
  if (canvasMode === "add") return;
  if (canvasMode === "function" || (selectedObject.kind === "port" && selectedObject.id !== id)) {
    if (!functionLinkStart) { functionLinkStart = id; selectedObject = {kind: "port", id}; $("#canvas-status").textContent = "已选择源端口，请选择目标端口"; renderCanvas(); renderInspector(); return; }
    if (functionLinkStart !== id) createFunctionBetweenPorts(functionLinkStart, id);
    return;
  }
  selectCanvasObject("port", id);
}
function createFunctionBetweenPorts(sourceId, responseId) {
  const source = portDefinitions.find((port) => port.id === sourceId), response = portDefinitions.find((port) => port.id === responseId);
  if (!source || !response) return;
  const name = variableById(source, "i") && variableById(response, "v") ? "transimpedance" : "lowpass";
  const functionId = `function_${Date.now()}`;
  pushCanvasHistory();
  functionDefinitions.push({function_id: functionId, relation_id: `m_${functionId}`, role: "target", domain: "frequency", source_port: sourceId, response_port: responseId, body: {kind: "builtin", name, parameters: name === "transimpedance" ? {transimpedance: {value: 1000, unit: "ohm"}} : {cutoff: {value: Number(value("frequency") || 1000), unit: "Hz"}, gain: {value: Number(value("gain") || 1), unit: "1"}}}, axis: {domain: "frequency", variable: "frequency", unit: "Hz", start: Number(value("f-min") || 10), stop: Number(value("f-max") || 100000), points: 96}, verification: "pending"});
  syncRelationsFromFunctions();
  functionLinkStart = null;
  setCanvasMode("select");
  renderMeasurements();
  selectCanvasObject("function", functionId);
  refreshEditor();
}
function updateCanvasPortPosition(point) {
  const port = portDefinitions.find((item) => item.id === draggingPortId);
  if (!port) return;
  const nextPosition = perimeterPosition(point);
  dragMoved = dragMoved || JSON.stringify(port.position || {}) !== JSON.stringify(nextPosition);
  port.position = nextPosition;
  renderCanvas();
}

function finishCanvasDrag() {
  if (!draggingPortId) return;
  if (dragMoved && dragHistorySnapshot) pushCanvasHistory(dragHistorySnapshot);
  draggingPortId = null;
  dragHistorySnapshot = null;
  dragMoved = false;
  refreshEditor();
}

function initCanvas() {
  const canvas = $("#circuit-canvas");
  if (!canvas) return;
  $("#canvas-select-mode")?.addEventListener("click", () => setCanvasMode("select"));
  $("#canvas-add-port")?.addEventListener("click", () => setCanvasMode("add"));
  $("#canvas-function-mode")?.addEventListener("click", () => setCanvasMode("function"));
  $("#canvas-undo")?.addEventListener("click", undoCanvasChange);
  $("#canvas-fit")?.addEventListener("click", () => { pushCanvasHistory(); portDefinitions.forEach((port, index) => { if (!port.position) port.position = {edge: index % 2 ? "right" : "left", offset: .25 + (index / Math.max(portDefinitions.length, 2)) * .5}; }); renderCanvas(); refreshEditor(); });
  canvas.addEventListener("click", (event) => { if (event.target.id === "circuit-boundary" || event.target.classList?.contains("canvas-grid") || event.target === canvas) { if (canvasMode === "add") addCanvasPort(canvasPoint(event)); else selectCanvasObject("circuit"); } });
  canvas.addEventListener("keydown", (event) => { if (event.key === "Escape") { functionLinkStart = null; setCanvasMode("select"); announce("已取消连接模式"); } });
  canvas.addEventListener("pointermove", (event) => { if (draggingPortId) { event.preventDefault(); updateCanvasPortPosition(canvasPoint(event)); } });
  window.addEventListener("pointerup", finishCanvasDrag);
  window.addEventListener("pointercancel", finishCanvasDrag);
}

function paramValue(value, fallback) { return value && typeof value === "object" ? value.value : value ?? fallback; }
function renderInspector() {
  const kind = $("#inspector-kind"), title = $("#inspector-title"), help = $("#inspector-help"), content = $("#inspector-content");
  if (!content) return;
  if (selectedObject.kind === "port") {
    const port = portDefinitions.find((item) => item.id === selectedObject.id);
    if (!port) return selectCanvasObject("circuit");
    kind.textContent = "PORT"; title.textContent = port.name; help.textContent = "端口只声明可交换的变量；行为由函数连接。";
    content.innerHTML = `<label>端口名称<input data-port-field="name" value="${escapeHtml(port.name)}"></label><div class="inspector-two"><label>正端子<input data-port-field="positive" value="${escapeHtml(port.positive)}"></label><label>参考端子<input data-port-field="negative" value="${escapeHtml(port.negative)}"></label></div><div class="inspector-two"><label>方向<select data-port-field="role"><option value="input" ${port.role === "input" ? "selected" : ""}>输入</option><option value="output" ${port.role === "output" ? "selected" : ""}>输出</option><option value="bidirectional" ${!port.role || port.role === "bidirectional" ? "selected" : ""}>双向</option><option value="power" ${port.role === "power" ? "selected" : ""}>供电</option></select></label><label>领域<select data-port-field="domain"><option value="dc" ${port.domain === "dc" ? "selected" : ""}>直流</option><option value="ac" ${port.domain === "ac" ? "selected" : ""}>小信号 AC</option><option value="transient" ${port.domain === "transient" ? "selected" : ""}>瞬态</option><option value="periodic" ${port.domain === "periodic" ? "selected" : ""}>周期稳态</option><option value="unspecified" ${!port.domain || port.domain === "unspecified" ? "selected" : ""}>未指定</option></select></label></div><fieldset class="variable-picker"><legend>可交换变量</legend>${CANVAS_VARIABLES.map((variable) => `<label><input type="checkbox" data-port-variable="${variable.id}" ${variableById(port, variable.id) ? "checked" : ""}>${variable.label}<small>${variable.unit}</small></label>`).join("")}</fieldset><div class="inspector-constraints"><div class="inspector-section-title"><b>端口约束</b><button id="inspector-add-constraint" class="mini-action" type="button">＋</button></div>${(port.constraints || []).map((constraint) => `<div class="inspector-constraint" data-inspector-constraint-id="${constraint.id}"><select data-port-constraint="variable"><option value="v" ${constraint.variable === "v" ? "selected" : ""}>v</option><option value="i" ${constraint.variable === "i" ? "selected" : ""}>i</option><option value="p" ${constraint.variable === "p" ? "selected" : ""}>p</option></select><select data-port-constraint="operator"><option value="equal" ${constraint.operator === "equal" ? "selected" : ""}>等于</option><option value="minimum" ${constraint.operator === "minimum" ? "selected" : ""}>不小于</option><option value="maximum" ${constraint.operator === "maximum" ? "selected" : ""}>不大于</option></select><input type="number" step="any" data-port-constraint="value" value="${constraint.value ?? ""}"></div>`).join("") || `<p class="empty-inspector">尚未添加约束。</p>`}</div><button id="inspector-delete-port" class="danger-action" type="button">删除端口</button>`;
    bindInspectorPort(port);
    return;
  }
  if (selectedObject.kind === "function") {
    const fn = functionDefinitions.find((item) => item.function_id === selectedObject.id);
    if (!fn) return selectCanvasObject("circuit");
    const bodyName = fn.body?.name || "lowpass", params = fn.body?.parameters || {};
    kind.textContent = "FUNCTION"; title.textContent = functionLabel(fn); help.textContent = "函数连接端口变量，目标和约束由函数定义产生。";
    const parameterField = (name, label, unit, fallback) => `<label>${label}<span class="unit-input"><input type="number" step="any" data-function-param="${name}" value="${paramValue(params[name], fallback)}"><small>${unit}</small></span></label>`;
    content.innerHTML = `<label>函数类型<select data-function-field="body_name"><option value="lowpass" ${bodyName === "lowpass" ? "selected" : ""}>lowpass</option><option value="highpass" ${bodyName === "highpass" ? "selected" : ""}>highpass</option><option value="bandpass" ${bodyName === "bandpass" ? "selected" : ""}>bandpass</option><option value="gain" ${bodyName === "gain" ? "selected" : ""}>gain</option><option value="transimpedance" ${bodyName === "transimpedance" ? "selected" : ""}>transimpedance</option><option value="impedance" ${bodyName === "impedance" ? "selected" : ""}>impedance</option><option value="regulate" ${bodyName === "regulate" ? "selected" : ""}>regulate</option><option value="sampled" ${bodyName === "sampled" ? "selected" : ""}>sampled</option><option value="expression_ast" ${fn.body?.kind === "expression_ast" ? "selected" : ""}>expression_ast</option><option value="rational" ${fn.body?.kind === "rational" ? "selected" : ""}>rational</option></select></label><div class="inspector-two"><label>源端口<select data-function-field="source_port">${portDefinitions.map((port) => `<option value="${port.id}" ${port.id === fn.source_port ? "selected" : ""}>${escapeHtml(port.name)}</option>`).join("")}</select></label><label>目标端口<select data-function-field="response_port">${portDefinitions.map((port) => `<option value="${port.id}" ${port.id === fn.response_port ? "selected" : ""}>${escapeHtml(port.name)}</option>`).join("")}</select></label></div>${bodyName === "transimpedance" ? parameterField("transimpedance", "跨阻", "Ω", 1000) : bodyName === "impedance" ? parameterField("ohms", "阻抗", "Ω", 50) : bodyName === "regulate" ? `${parameterField("input_voltage", "输入电压", "V", 5)}${parameterField("output_voltage", "输出电压", "V", 10)}${parameterField("output_current", "输出电流", "A", 1)}` : `${parameterField(bodyName === "bandpass" ? "center" : bodyName === "gain" ? "gain" : "cutoff", bodyName === "gain" ? "增益" : bodyName === "bandpass" ? "中心频率" : "截止频率", bodyName === "gain" ? "倍" : "Hz", bodyName === "gain" ? 1 : 1000)}${bodyName !== "gain" ? parameterField("gain", "增益", "倍", 1) : ""}`}<div class="inspector-two"><label>轴起点<input type="number" data-function-axis="start" value="${fn.axis?.start ?? 10}"></label><label>轴终点<input type="number" data-function-axis="stop" value="${fn.axis?.stop ?? 100000}"></label></div><div class="function-contract"><code>ports.${fn.source_port}.variables.v → ports.${fn.response_port}.variables.v</code><span>状态：${fn.verification || "待验证"}</span></div><button id="inspector-delete-function" class="danger-action" type="button">删除函数</button>`;
    bindInspectorFunction(fn);
    return;
  }
  kind.textContent = "CIRCUIT"; title.textContent = "整块电路"; help.textContent = "正方形是待设计电路的黑盒边界，位置只属于界面布局。";
  content.innerHTML = `<label>任务名称<input data-global-field="name" value="${escapeHtml(value("name"))}"></label><label>设计说明<textarea data-global-field="description" rows="3" placeholder="描述整块电路的用途，不写求解代码"></textarea></label><div class="global-stats"><span>${portDefinitions.length} 个端口</span><span>${functionDefinitions.length} 个函数</span><span>${portDefinitions.reduce((total, port) => total + (port.constraints || []).length, 0)} 条端口约束</span></div><p class="inspector-note">点击空白区域查看全局设置；拖动端口只会改变边界上的位置，不会改变 PBDL 语义。</p>`;
  content.querySelectorAll("[data-global-field]").forEach((node) => node.addEventListener("input", () => { if (node.dataset.globalField === "name") $("#name").value = node.value; refreshEditor(); }));
}
function contentEvents(selector, handler) { $("#inspector-content")?.querySelectorAll(selector).forEach((node) => node.addEventListener(node.tagName === "SELECT" ? "change" : "input", () => handler(node))); }
function bindInspectorPort(port) {
  contentEvents("[data-port-field]", (node) => { port[node.dataset.portField] = node.value; renderCanvas(); refreshEditor(); });
  contentEvents("[data-port-variable]", (node) => { const id = node.dataset.portVariable; port.variables = node.checked ? [...(port.variables || []).filter((item) => (item.id || item.name) !== id), defaultVariable(id)] : (port.variables || []).filter((item) => (item.id || item.name) !== id); renderCanvas(); refreshEditor(); });
  contentEvents("[data-port-constraint]", (node) => { const row = node.closest("[data-inspector-constraint-id]"), constraint = port.constraints.find((item) => item.id === row.dataset.inspectorConstraintId); if (constraint) constraint[node.dataset.portConstraint] = node.type === "number" ? Number(node.value) : node.value; refreshEditor(); });
  $("#inspector-add-constraint")?.addEventListener("click", () => { pushCanvasHistory(); port.constraints.push({id: `c${Date.now()}`, variable: "v", analysis: "dc_operating_point", operator: "equal", value: 0}); renderInspector(); refreshEditor(); });
  $("#inspector-delete-port")?.addEventListener("click", () => { if (portDefinitions.length <= 1) return; pushCanvasHistory(); replaceDraftCollection("ports", portDefinitions.filter((item) => item.id !== port.id)); replaceDraftCollection("measurements", measurementDefinitions.filter((item) => item.source_port !== port.id && item.response_port !== port.id)); replaceDraftCollection("functions", functionDefinitions.filter((item) => item.source_port !== port.id && item.response_port !== port.id)); selectCanvasObject("circuit"); renderPorts(); refreshEditor(); });
}
function bindInspectorFunction(fn) {
  contentEvents("[data-function-field]", (node) => {
    if (node.dataset.functionField === "body_name") {
      fn.body = {...fn.body, kind: "builtin", name: node.value, parameters: {...fn.body.parameters}};
      fn.relation_kind = relationKindForBodyName(node.value, fn.relation_kind);
    } else fn[node.dataset.functionField] = node.value;
    syncRelationsFromFunctions();
    syncBehaviorControlsFromFunction(fn);
    renderMeasurements();
    renderCanvas();
    renderInspector();
    refreshEditor();
  });
  contentEvents("[data-function-param]", (node) => { fn.body.parameters = {...fn.body.parameters, [node.dataset.functionParam]: {value: Number(node.value), unit: node.dataset.functionParam === "gain" ? "1" : node.dataset.functionParam === "ohms" || node.dataset.functionParam === "transimpedance" ? "ohm" : node.dataset.functionParam.includes("voltage") ? "V" : node.dataset.functionParam.includes("current") ? "A" : "Hz"}}; syncBehaviorControlsFromFunction(fn); syncRelationsFromFunctions(); renderCanvas(); refreshEditor(); });
  contentEvents("[data-function-axis]", (node) => { fn.axis = {...fn.axis, [node.dataset.functionAxis]: Number(node.value)}; refreshEditor(); });
  $("#inspector-delete-function")?.addEventListener("click", () => { pushCanvasHistory(); replaceDraftCollection("functions", functionDefinitions.filter((item) => item.function_id !== fn.function_id)); syncRelationsFromFunctions(); selectCanvasObject("circuit"); renderMeasurements(); refreshEditor(); });
}

function renderCompareControls() {
  $("#compare-controls").innerHTML = `<span>曲线比较</span>${latest.candidates.map((candidate, index) => `<label><input type="checkbox" value="${index}" ${compared.has(index) ? "checked" : ""}> #${candidate.rank} ${candidate.template}</label>`).join("")}`;
  $("#compare-controls").querySelectorAll("input").forEach((node) => node.addEventListener("change", () => {
    const index = Number(node.value);
    if (node.checked) compared.add(index); else compared.delete(index);
    if (compared.size === 0) { compared.add(index); node.checked = true; }
    redrawCharts();
  }));
}

function setResultTab(tab, moveFocus = false) {
  const tabs = [...document.querySelectorAll("[data-result-tab]")];
  const panels = [...document.querySelectorAll("[data-result-panel]")];
  if (!tabs.length) return;
  const selected = tabs.find((node) => node.dataset.resultTab === tab) || tabs[0];
  tabs.forEach((node) => {
    const active = node === selected;
    node.setAttribute("aria-selected", active ? "true" : "false");
    node.tabIndex = active ? 0 : -1;
  });
  panels.forEach((panel) => { panel.hidden = panel.dataset.resultPanel !== selected.dataset.resultTab; });
  if (moveFocus) selected.focus();
}

function initResultTabs() {
  const tabs = [...document.querySelectorAll("[data-result-tab]")];
  tabs.forEach((tab, index) => {
    tab.addEventListener("click", () => setResultTab(tab.dataset.resultTab));
    tab.addEventListener("keydown", (event) => {
      if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
      event.preventDefault();
      const direction = event.key === "ArrowLeft" ? -1 : event.key === "ArrowRight" ? 1 : 0;
      const nextIndex = event.key === "Home" ? 0 : event.key === "End" ? tabs.length - 1 : (index + direction + tabs.length) % tabs.length;
      setResultTab(tabs[nextIndex].dataset.resultTab, true);
    });
  });
  setResultTab("overview");
}

function renderCandidate(index) {
  const candidate = latest.candidates[index];
  document.querySelectorAll(".candidate").forEach((node, i) => node.classList.toggle("active", i === index));
  $("#candidate-title").textContent = `#${candidate.rank} ${candidate.template}`;
  $("#candidate-description").textContent = candidate.description;
  const m = candidate.metrics;
  const report = candidate.verification?.constraint_report;
  $("#metric-grid").innerHTML = [["RMSE", `${format(m.rmse_db)} dB`], ["最大误差", `${format(m.max_abs_db)} dB`], ["元件", m.component_count], ["Pareto", m.pareto_rank]].map(([key, val]) => `<div><span>${key}</span><b>${val}</b></div>`).join("");
  $("#verification-summary").innerHTML = `<span class="verification-state ${report?.passed ? "pass" : report ? "fail" : "pending"}">${report?.passed ? "约束已通过" : report ? "约束未通过" : "等待验证"}</span><span>${candidate.verification?.fidelity_schedule?.selected_candidate_ids?.length || 0} 个候选进入真值复核</span><span>${candidate.verification?.optimization_run?.evaluations || 0} 次优化评估</span>`;
  $("#schematic-image").src = candidate.artifacts.schematic_svg;
  $("#kicad-link").href = candidate.artifacts.kicad_schematic;
  $("#spice-link").href = candidate.artifacts.spice_netlist || "#";
  $("#spice-link").classList.toggle("disabled", !candidate.artifacts.spice_netlist);
  $("#report-link").href = candidate.artifacts.report;
  $("#parameters").textContent = Object.entries(candidate.parameters).map(([key, val]) => `${key.padEnd(12)} ${format(val)}`).join("\n");
  const optimization = candidate.verification?.optimization_run || {};
  const simulationStatuses = candidate.verification?.simulation_statuses || [];
  const fidelity = candidate.verification?.fidelity_schedule || {};
  $("#candidate-evidence").innerHTML = [
    `<article><h3>优化</h3><p class="${m.optimizer_success ? "pass" : "fail"}">${m.optimizer_success ? "已收敛" : "未确认收敛"} · ${optimization.evaluations || 0} 次评估</p></article>`,
    `<article><h3>成本 / 面积</h3><p>${resultNumber(m.estimated_cost)} · ${resultNumber(m.estimated_area_mm2, "mm²")}</p></article>`,
    `<article><h3>仿真状态</h3><p>${escapeHtml(simulationStatuses.join(", ") || "未记录")}</p></article>`,
    `<article><h3>证据链</h3><p>${report?.passed ? "约束通过" : "缺少通过证据"} · 保真度 ${fidelity.selected_candidate_ids?.length || 0} 个候选</p></article>`
  ].join("");
  redrawCharts();
}

function renderResult(data) {
  latest = data; compared = new Set([0]);
  workbenchState.execution = {status: "verified", startedAt: workbenchState.execution.startedAt, processed: data.candidates?.length || 0};
  setWorkbenchState({mode: "results", execution: workbenchState.execution});
  $(".workspace")?.classList.remove("has-error");
  $("#execution-progress").hidden = false;
  $("#execution-progress").textContent = `${data.candidates?.length || 0} 个候选已完成搜索、仿真与约束验证`;
  $("#empty-state").hidden = true; $("#pbdl-results").hidden = true; $("#results").hidden = false;
  setResultTab("overview");
  $("#candidate-strip").innerHTML = data.candidates.map((candidate, index) => { const passed = candidate.verification?.constraint_report?.passed; return `<button class="candidate ${index === 0 ? "active" : ""}" data-index="${index}"><b>#${candidate.rank}</b><span>${escapeHtml(candidate.template)}</span><small>${passed ? "✓ 通过" : "× 未通过"} · RMSE ${format(candidate.metrics.rmse_db)} dB</small></button>`; }).join("");
  document.querySelectorAll(".candidate").forEach((node) => node.addEventListener("click", () => renderCandidate(Number(node.dataset.index))));
  renderCompareControls(); renderCandidate(0);
}

function resultNumber(value, unit = "") {
  const number = Number(value);
  return Number.isFinite(number) ? `${number.toPrecision(5)}${unit ? ` ${unit}` : ""}` : "-";
}

function renderStageError(report) {
  if (!report) return "";
  return `<div class="stage-error"><div class="stage-error-head"><b>${escapeHtml(report.phase_label || report.phase || "阶段错误")}</b><code>${escapeHtml(report.code || "error")}</code></div><strong>${escapeHtml(report.message || "执行失败")}</strong><p>${escapeHtml(report.detail || "")}</p><small>${escapeHtml(report.suggestion || "请检查该阶段输入后重试")}</small></div>`;
}

function renderPowerStage(stage) {
  const design = stage.design;
  if (!design) {
    return `<article class="stage ${stage.status}"><div><b>${stage.stage_index}. ${escapeHtml(stage.analysis_kind)}</b><span>${escapeHtml(stage.target_kind || "无目标")}</span></div><div><strong>${escapeHtml(stage.status)}</strong><small>${escapeHtml(stage.best_template || stage.message || "")}</small></div>${renderStageError(stage.error)}</article>`;
  }
  const topology = design.topology || {};
  const point = design.operating_point || {};
  const validation = design.validation || {};
  const search = design.topology_search || {};
  const bounds = search.bounds || {};
  const observations = (design.simulation_task_evaluations || []).flatMap((task) => task.observations || []);
  const checks = observations.length ? observations.map((item) => `<li class="${item.passed ? "pass" : "fail"}"><span class="check-mark">${item.passed ? "✓" : "×"}</span><span><b>${escapeHtml(item.name)}</b><small>${escapeHtml(item.message)}</small></span></li>`).join("") : `<li class="${validation.passed ? "pass" : "fail"}"><span class="check-mark">${validation.passed ? "✓" : "×"}</span><span><b>统一约束验证</b><small>${escapeHtml((validation.issues || []).join("；") || "全部通过")}</small></span></li>`;
  const candidates = (design.candidates || []).map((item, index) => `<tr class="${index === 0 ? "selected" : ""}"><td>${index + 1}</td><td><b>${escapeHtml(item.name || "-")}</b><small>${escapeHtml(item.family || "-")}</small></td><td>${resultNumber(item.objective)}</td><td class="candidate-state ${item.passed ? "pass" : "fail"}">${item.passed ? "通过" : "未通过"}</td></tr>`).join("");
  const registered = (bounds.registered_productions || []).map((name) => `<li>${escapeHtml(name)}</li>`).join("");
  const rejected = (search.rejected || []).map((item) => `<li><b>${escapeHtml(item.name)}</b><span>${escapeHtml(item.reason)}</span></li>`).join("");
  const sourceLabel = topology.origin === "functional_block_knowledge_base" ? "功能模块知识库" : (topology.origin || "内置专家");
  return `<article class="power-stage-result">
    <header class="power-result-head"><div><span class="result-kicker">阶段 ${stage.stage_index} · ${escapeHtml(stage.analysis_kind)}</span><h2>${escapeHtml(topology.name || stage.best_template || "电源电路")}</h2><p>${escapeHtml(topology.rationale || "")}</p></div><div class="result-state ${validation.passed ? "pass" : "fail"}"><span>${validation.passed ? "✓" : "×"}</span>${validation.passed ? "验证通过" : "验证失败"}</div></header>
    <div class="power-metrics"><div><span>输出电压</span><b>${resultNumber(point.output_voltage_v, "V")}</b></div><div><span>输出电流</span><b>${resultNumber(point.output_current_a, "A")}</b></div><div><span>效率</span><b>${Number.isFinite(Number(point.efficiency)) ? `${(Number(point.efficiency) * 100).toFixed(2)} %` : "-"}</b></div><div><span>预测纹波</span><b>${resultNumber(point.predicted_ripple_mv, "mV")}</b></div></div>
    <div class="power-result-layout"><section class="power-schematic"><h3>生成原理图</h3><img src="${stage.artifacts.schematic_svg}" alt="${escapeHtml(topology.name || "电源电路")} 原理图"></section><section class="power-inspection"><h3>端口与目标验收</h3><ul class="constraint-checks">${checks}</ul><h3>拓扑搜索</h3><dl class="search-summary"><div><dt>来源</dt><dd>${escapeHtml(sourceLabel)}</dd></div><div><dt>知识版本</dt><dd>${escapeHtml(topology.knowledge_id || "-")}</dd></div><div><dt>求解器</dt><dd>${escapeHtml(topology.solver || "-")}</dd></div><div><dt>搜索范围</dt><dd>${search.accepted_candidates || 0} / ${search.constructed_candidates || 0} 个候选通过结构约束</dd></div></dl></section></div>
    <section class="candidate-comparison"><h3>候选电路比较</h3><table><thead><tr><th>排名</th><th>拓扑</th><th>目标函数</th><th>状态</th></tr></thead><tbody>${candidates || `<tr><td colspan="4">没有候选明细</td></tr>`}</tbody></table></section>
    <details class="search-details"><summary>查看有界搜索内容</summary><div><section><h4>已注册生产式</h4><ul>${registered || "<li>无</li>"}</ul></section><section><h4>被拒绝候选</h4><ul>${rejected || "<li>无</li>"}</ul></section></div></details>
    ${renderStageError(stage.error)}<div class="stage-artifacts power-artifacts"><a href="${stage.artifacts.kicad_schematic}" download>KiCad 原理图</a><a href="${stage.artifacts.schematic_svg}" target="_blank">SVG 原理图</a><a href="${stage.artifacts.simulation_tasks}" target="_blank">验收任务</a><a href="${stage.artifacts.topology_search}" target="_blank">搜索证书</a><a href="${stage.artifacts.report}" target="_blank">完整报告</a></div>
  </article>`;
}

function renderPbdlResult(data) {
  workbenchState.execution = {status: "verified", startedAt: workbenchState.execution.startedAt, processed: data.executed_count || 0};
  setWorkbenchState({mode: "results", execution: workbenchState.execution});
  $(".workspace")?.classList.remove("has-error");
  $("#empty-state").hidden = true; $("#results").hidden = true; $("#pbdl-results").hidden = false;
  const counts = `${data.executed_count} 已执行 · ${data.unsupported_count} 待支持 · ${data.error_count} 错误`;
  $("#pbdl-results").innerHTML = `<section class="pbdl-report"><div class="pbdl-head"><div><h2>${escapeHtml(data.spec_name)}</h2><p>${counts}</p></div><a class="button-link" href="/artifacts/${data.run_id}/plan.json" target="_blank">计划 JSON</a></div><div class="stage-list">${data.stages.map(renderPowerStage).join("")}</div></section>`;
}

function renderCompileResult(data) {
  const preview = $("#compile-preview");
  if (!preview) return;
  const execution = data.execution || {};
  const diagnostics = data.diagnostics || {};
  const gaps = diagnostics.capability_gaps || [];
  const status = execution.status === "ready" ? "ready" : "blocked";
  workbenchState.compile = {status: execution.status === "ready" ? "ready" : "blocked", data};
  setDraftMeta("ready", false);
  setWorkbenchState({compile: workbenchState.compile, draft: workbenchState.draft}, execution.status === "ready" ? "可执行 · 检查通过" : "存在能力缺口 · 暂不可执行");
  preview.hidden = false;
  $("#compile-title").textContent = execution.status === "ready" ? "规范已编译，可进入执行" : "规范已编译，但执行被阻止";
  $("#compile-status").className = `compile-status ${status}`;
  $("#compile-status").textContent = execution.status === "ready" ? "READY" : "BLOCKED";
  $("#compile-summary").innerHTML = [
    `<div><b>端口：</b>${execution.port_count || 0} / 执行适配器要求 ${execution.required_port_count || 2}</div>`,
    `<div><b>执行模式：</b>${escapeHtml(execution.mode || "-")} · <b>阶段：</b>${(data.plan?.stages || []).length}</div>`,
    `<div><b>能力缺口：</b>${gaps.length ? gaps.map((item) => escapeHtml(`${item.context?.stage_name || "stage"}: ${item.detail || item.message}`)).join("；") : "无"}</div>`,
    `<div><b>未识别顶层字段：</b>${(diagnostics.ignored_fields || []).length ? escapeHtml(diagnostics.ignored_fields.join(", ")) : "无"}</div>`
  ].join("");
  $("#compile-ir-preview").textContent = JSON.stringify(data.ir || {}, null, 2);
}

async function compileSpecPayload(spec) {
  clearResultView();
  workbenchState.compile = {status: "checking", data: null};
  setWorkbenchState({mode: "editing", compile: workbenchState.compile}, "检查中 · 正在编译 PBDL");
  const data = await requestJson("/api/spec/compile", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({spec})
  }, "pbdl");
  renderCompileResult(data);
  return data;
}

function parseEditorSpec(editorId) {
  try {
    return JSON.parse($(editorId).value);
  } catch (error) {
    const wrapped = new Error("PBDL JSON 格式错误");
    wrapped.report = {schema: "circuit_ai.error_report", schema_version: 1, phase: "pbdl", phase_label: "PBDL 解析", code: "invalid_pbdl", severity: "error", message: "PBDL JSON 格式错误", detail: error.message, suggestion: "修正 JSON 语法后再提交。", retryable: false, cause_type: error.name || "", context: {editor: editorId}};
    throw wrapped;
  }
}

function firstCompileGap(data) {
  const report = data?.errors?.[0];
  return report ? {report} : null;
}

function clearResultView() {
  latest = null;
  workbenchState.mode = "editing";
  workbenchState.execution = {status: "idle", startedAt: null, processed: 0};
  setWorkbenchState({mode: "editing", execution: workbenchState.execution});
  $(".workspace")?.classList.remove("results-mode");
  $("#execution-progress")?.setAttribute("hidden", "");
  $("#empty-state").hidden = false;
  $("#results").hidden = true;
  $("#pbdl-results").hidden = true;
}

async function run() {
  clearError();
  if (!validateAllSteps()) return;
  let spec;
  try { spec = parseEditorSpec("#json-editor"); } catch (error) { showError(error, "pbdl", "invalid_pbdl"); return; }
  workbenchState.execution = {status: "running", startedAt: Date.now(), processed: 0};
  setWorkbenchState({mode: "editing", execution: workbenchState.execution}, "执行中 · 正在搜索候选");
  $("#run-button").disabled = true;
  try {
    const compile = await compileSpecPayload(spec);
    if (compile.execution?.status !== "ready") {
      const gap = firstCompileGap(compile);
      if (gap) showError(gap, "candidate_search", "capability_gap");
      return;
    }
    const isPowerDesign = (spec.targets || []).some((target) => target.target_kind === "dc");
    const isMultiStage = (spec.analyses || []).length > 1;
    const usePbdlRunner = isPowerDesign || isMultiStage;
    const topK = Math.max(1, Number(spec.optimization?.top_k || number("top-k") || 1));
    const data = await requestJson(usePbdlRunner ? "/api/pbdl/synthesize" : "/api/synthesize", { method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(usePbdlRunner ? {spec, execute: true, top_k: topK} : {spec}) }, "execution");
    clearError();
    if (usePbdlRunner) {
      $("#run-status").textContent = `执行中 · 已处理 ${data.stages?.length || 0} 个阶段`;
      renderPbdlResult(data);
      const firstError = data.stages.find((stage) => stage.error)?.error;
      if (firstError) showError({report: firstError}, "execution");
      else $("#run-status").textContent = `验证完成 · ${data.executed_count} 个阶段已完成`;
    } else {
      renderResult(data);
      $("#run-status").textContent = `验证完成 · ${data.candidates.length} 个候选已完成`;
    }
  } catch (error) { showError(error, "execution"); }
  finally { $("#run-button").disabled = false; }
}

async function compileFormSpec() {
  clearError();
  if (!validateAllSteps()) return null;
  try {
    const data = await compileSpecPayload(parseEditorSpec("#json-editor"));
    const gap = firstCompileGap(data);
    if (gap) showError(gap, "candidate_search", "capability_gap");
    return data;
  } catch (error) {
    showError(error, "pbdl");
    return null;
  }
}

async function suggestTopologies() {
  clearError();
  const target = $("#topology-suggestions");
  target.textContent = "正在检索已有拓扑与本地训练记录...";
  try {
    const data = await requestJson("/api/topology-suggestions", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({spec: specFromForm()})}, "candidate_search");
    clearError();
    const evidence = new Map(data.local_evidence.map((item) => [item.template, item.observations]));
    target.innerHTML = data.catalog.length ? data.catalog.map((item) => `<div><b>${item.name}</b><span>${item.family} · ${item.tags.join(", ")}${evidence.has(item.name) ? ` · 本地记录 ${evidence.get(item.name)}` : ""}</span></div>`).join("") : "没有满足当前元件库、必选元件和行为约束的已有拓扑。";
  } catch (error) { showError(error, "candidate_search"); showInlineError("#topology-suggestions", error, "candidate_search"); }
}

async function loadComponentLibraryStatus() {
  try {
    const status = await requestJson("/api/components/status", {}, "components");
    $("#component-library-status").classList.remove("error-inline");
    $("#component-library-status").textContent = status.ready ? `${status.symbol_count} 个符号 · ${status.footprint_count} 个封装 · ${status.model_count || 0} 个 SPICE 模型 · ${status.roots.length} 个库根目录` : "尚未建立真实元件索引";
  } catch (error) {
    showError(error, "components");
    showInlineError("#component-library-status", error, "components");
  }
}

async function loadExecutionPlan() {
  const target = $("#execution-plan");
  const limits = $("#execution-limits");
  if (!target || !limits) return;
  try {
    const data = await requestJson("/api/workbench/architecture", {}, "request");
    target.innerHTML = data.pipeline.map((stage, index) => `<li><span>${String(index + 1).padStart(2, "0")}</span><div><b>${escapeHtml(stage.label)}</b><small>${escapeHtml(stage.detail)}</small></div></li>`).join("");
    limits.innerHTML = data.limits.map((limit) => `<li>${escapeHtml(limit)}</li>`).join("");
  } catch (error) {
    target.innerHTML = `<li class="plan-error">执行能力不可用：${escapeHtml(error.message)}</li>`;
  }
}

async function reindexComponents() {
  const root = value("component-library-root").trim();
  $("#component-library-status").textContent = "正在扫描 KiCad 库…";
  try {
    const status = await requestJson("/api/components/reindex", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({roots: root ? [root] : []})}, "components");
    $("#component-library-status").classList.remove("error-inline");
    $("#component-library-status").textContent = `${status.symbol_count} 个符号 · ${status.footprint_count} 个封装 · ${status.model_count || 0} 个 SPICE 模型 · ${status.roots.length} 个库根目录`;
    await searchComponents();
  } catch (error) {
    showError(error, "components");
    showInlineError("#component-library-status", error, "components");
  }
}

async function searchComponents() {
  const query = encodeURIComponent(value("component-search").trim()), kind = encodeURIComponent(value("component-kind"));
  componentSearchController?.abort();
  componentSearchController = new AbortController();
  let data;
  try { data = await requestJson(`/api/components/search?q=${query}&kind=${kind}&limit=50`, {signal: componentSearchController.signal}, "components"); }
  catch (error) { if (error.name === "AbortError") return; showInlineError("#component-search-results", error, "components"); return; }
  $("#component-search-results").classList.remove("error-inline");
  $("#component-search-results").innerHTML = data.items.length ? data.items.map((item) => { const identifier = item.library ? `${item.library}:${item.name}` : item.name; const detail = item.kind === "model" ? `${item.kind} · ${item.source_path}` : `${item.library} · ${item.description || item.source_path}`; return item.kind === "model" ? `<button type="button" class="component-search-item" data-real-model="${escapeHtml(JSON.stringify(item))}"><b>${escapeHtml(item.name)}</b><span>${escapeHtml(detail)}</span></button>` : `<button type="button" data-real-component="${escapeHtml(identifier)}"><b>${escapeHtml(item.name)}</b><span>${escapeHtml(detail)}</span></button>`; }).join("") : "<span>没有匹配的本机 KiCad 元件</span>";
}

function scheduleComponentSearch() {
  window.clearTimeout(componentSearchTimer);
  componentSearchTimer = window.setTimeout(() => searchComponents(), 280);
}

function renderModelBindingEditor() {
  const status = $("#model-selection-status"), mapping = $("#model-pin-mapping"), add = $("#add-model-binding");
  if (!modelDraft) { status.textContent = "先搜索并选择一个 SPICE 模型"; mapping.innerHTML = ""; add.disabled = true; renderModelBindings(); return; }
  status.textContent = `${modelDraft.name} · ${modelDraft.library}`; add.disabled = false;
  mapping.innerHTML = modelDraft.mapping.map((pin, index) => `<label>符号引脚 ${index + 1}<input data-model-symbol-pin="${index}" value="${escapeHtml(pin.symbol)}"><span class="model-pin-arrow">→</span><input data-model-node="${index}" value="${escapeHtml(pin.model)}"></label>`).join("");
  mapping.querySelectorAll("input").forEach((node) => node.addEventListener("input", () => { const index = Number(node.dataset.modelSymbolPin ?? node.dataset.modelNode); if (node.dataset.modelSymbolPin !== undefined) modelDraft.mapping[index].symbol = node.value; else modelDraft.mapping[index].model = node.value; }));
  renderModelBindings();
}

function renderModelBindings() {
  const target = $("#model-bindings-list"); if (!target) return;
  target.innerHTML = modelBindings.map((binding, index) => `<div class="model-binding-row"><span><b>${escapeHtml(binding.component)}</b> ← ${escapeHtml(binding.name)}</span><code>${escapeHtml(binding.pins)}</code><button type="button" data-remove-model-binding="${index}" aria-label="删除模型绑定">×</button></div>`).join("");
  target.querySelectorAll("[data-remove-model-binding]").forEach((node) => node.addEventListener("click", () => { modelBindings.splice(Number(node.dataset.removeModelBinding), 1); renderModelBindings(); refreshEditor(); }));
}

async function runPbdl() {
  clearError();
  let spec;
  try { spec = parseEditorSpec("#pbdl-editor"); } catch (error) { showError(error, "pbdl", "invalid_pbdl"); return; }
  $("#run-pbdl-button").disabled = true; $("#run-status").textContent = "正在规划多阶段电路...";
  try {
    const compile = await compileSpecPayload(spec);
    if (compile.execution?.status !== "ready") {
      const gap = firstCompileGap(compile);
      if (gap) showError(gap, "candidate_search", "capability_gap");
      return;
    }
    const topK = Math.max(1, Number(spec.optimization?.top_k || number("top-k") || 1));
    const data = await requestJson("/api/pbdl/synthesize", { method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({spec, execute: $("#pbdl-execute").checked, top_k: topK}) }, "execution");
    renderPbdlResult(data);
    const firstError = data.stages.find((stage) => stage.error)?.error;
    if (firstError) showError({report: firstError}, "execution");
    else $("#run-status").textContent = `${data.executed_count} 个阶段已执行`;
  } catch (error) { showError(error, "execution"); }
  finally { $("#run-pbdl-button").disabled = false; }
}

fields.forEach((id) => $("#" + id).addEventListener("input", () => {
  if (id === "port-count") { ensurePortCount(number("port-count")); renderPorts(); }
  if (["behavior-kind", "frequency", "gain", "f-min", "f-max"].includes(id)) updatePrimaryFunctionFromControls();
  if (id === "signal-port") { const port = portDefinitions.find((item) => item.id === value("signal-port")); if (port && port.excitation) applyWaveform(port.excitation.waveform, port.id); else updateWaveform(); }
  else updateWaveform();
  refreshEditor();
}));
$("#add-port").addEventListener("click", () => { pushCanvasHistory(); ensurePortCount(portDefinitions.length + 1); renderPorts(); refreshEditor(); });
$("#add-measurement").addEventListener("click", () => { pushCanvasHistory(); const first = portDefinitions[0], second = portDefinitions[1] || first; measurementDefinitions.push({id: `m${Date.now()}`, kind: "voltage_transfer", source_port: first.id, response_port: second.id}); renderMeasurements(); refreshEditor(); });
$("#wizard-back").addEventListener("click", () => showWizardStep(currentWizardStep - 1));
$("#wizard-next").addEventListener("click", () => { clearFieldValidation(); const error = validateWizardStep(currentWizardStep); if (error) { $("#wizard-error").textContent = error; focusValidationField(currentWizardStep); return; } showWizardStep(currentWizardStep + 1); });
$("#wizard-progress").querySelectorAll("button").forEach((button) => button.addEventListener("click", () => { const target = Number(button.dataset.step); if (target <= currentWizardStep) showWizardStep(target); }));
document.querySelectorAll("input[name=element], input[name=required-element]").forEach((node) => node.addEventListener("change", () => {
  if (node.name === "required-element" && node.checked) document.querySelector(`input[name=element][value="${node.value}"]`).checked = true;
  refreshEditor();
}));
$("#run-button").addEventListener("click", run);
$("#compile-button").addEventListener("click", compileFormSpec);
$("#edit-spec-button").addEventListener("click", () => { clearError(); workbenchState.mode = "editing"; setWorkbenchState({mode: "editing"}); showWizardStep(4); $("#control-panel")?.scrollIntoView({behavior: "smooth", block: "start"}); });
$("#result-edit-button").addEventListener("click", () => { $("#edit-spec-button")?.click(); });
$("#compile-result-button").addEventListener("click", compileFormSpec);
$("#suggest-topologies").addEventListener("click", suggestTopologies);
$("#reindex-components").addEventListener("click", reindexComponents);
$("#component-search").addEventListener("input", scheduleComponentSearch);
$("#component-kind").addEventListener("change", searchComponents);
$("#component-search-results").addEventListener("click", (event) => {
  const node = event.target.closest("[data-real-component], [data-real-model]");
  if (!node) return;
  if (node.dataset.realComponent) {
    const identifier = node.dataset.realComponent;
    if (!selectedRealComponents.includes(identifier)) selectedRealComponents.push(identifier);
    node.classList.add("selected");
    $("#component-library-status").textContent = `已选择 ${selectedRealComponents.length} 个真实元件`;
    refreshEditor();
    return;
  }
  const item = JSON.parse(node.dataset.realModel);
  modelDraft = {component: value("model-component").trim() || "R1", library: item.source_path, name: item.name, modelPins: item.pins || [], mapping: (item.pins || []).map((pin, index) => ({symbol: String(index + 1), model: pin})), device: item.model_type || "X"};
  renderModelBindingEditor();
});
$("#model-component").addEventListener("input", () => { if (modelDraft) { modelDraft.component = value("model-component").trim(); renderModelBindingEditor(); } });
$("#add-model-binding").addEventListener("click", () => { if (!modelDraft) return; const binding = {...modelDraft, pins: modelDraft.mapping.map((pin) => `${pin.symbol}=${pin.model}`).join(" ")}; binding.model_pins = [...modelDraft.modelPins]; delete binding.modelPins; delete binding.mapping; replaceDraftCollection("modelBindings", [...modelBindings.filter((item) => item.component !== binding.component), binding]); renderModelBindings(); refreshEditor(); });
$("#run-pbdl-button").addEventListener("click", runPbdl);
$("#json-editor").addEventListener("change", () => {
  const editor = $("#json-editor");
  try {
    applySpec(JSON.parse(editor.value));
    markJsonEditorInvalid(false);
    clearError();
    refreshEditor();
  } catch (error) {
    markJsonEditorInvalid(true);
    showError(error, "pbdl", "invalid_pbdl");
  }
});
$("#save-draft").addEventListener("click", () => { saveDraft(); setWorkbenchState({draft: setDraftMeta("saved", false)}, "草稿 · 已保存"); });
$("#restore-draft").addEventListener("click", restoreDraft);
$("#export-pbdl").addEventListener("click", exportPbdl);
window.addEventListener("beforeunload", (event) => { if (workbenchState.draft.dirty) { event.preventDefault(); event.returnValue = ""; } });
initCanvas();
initResultTabs();
renderPorts();
updateWaveform();
refreshEditor();
selectCanvasObject("circuit");
renderModelBindingEditor();
loadComponentLibraryStatus();
loadExecutionPlan();
showWizardStep(0);
try { if (localStorage.getItem(DRAFT_STORAGE_KEY)) $("#restore-draft")?.removeAttribute("hidden"); } catch (_) {}
isBootstrapping = false;
