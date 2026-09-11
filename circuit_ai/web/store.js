export const WORKBENCH_SCHEMA_VERSION = 3;
export const DRAFT_STORAGE_KEY = "circuit-ai-workbench-draft";

function clone(value) {
  return JSON.parse(JSON.stringify(value));
}

export function createWorkbenchStore({initialModel = {}} = {}) {
  const model = {
    ports: clone(initialModel.ports || []),
    measurements: clone(initialModel.measurements || []),
    functions: clone(initialModel.functions || []),
    realComponents: clone(initialModel.realComponents || []),
    modelBindings: clone(initialModel.modelBindings || [])
  };
  const state = {
    schemaVersion: WORKBENCH_SCHEMA_VERSION,
    mode: "editing",
    draft: {status: "draft", dirty: true, model},
    compile: {status: "idle", data: null},
    execution: {status: "idle", startedAt: null, processed: 0}
  };

  return {
    state,
    replaceCollection(name, values) {
      if (!Array.isArray(state.draft.model[name])) throw new Error(`Unknown draft collection: ${name}`);
      state.draft.model[name].splice(0, state.draft.model[name].length, ...clone(values));
      return state.draft.model[name];
    },
    snapshotModel(names = Object.keys(state.draft.model)) {
      return Object.fromEntries(names.map((name) => [name, clone(state.draft.model[name])]));
    },
    restoreModel(snapshot, names = Object.keys(snapshot)) {
      names.forEach((name) => this.replaceCollection(name, snapshot[name] || []));
    }
  };
}
