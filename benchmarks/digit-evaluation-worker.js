import {scoreDigitOutput} from './digit-evaluation.js';

const digest = async bytes => Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)))
  .map(value => value.toString(16).padStart(2, '0')).join('');
const requireTrue = (condition, message) => { if (!condition) throw new Error(message); };
const acknowledgments = new Map();

export async function runDigitEvaluation(spec, api, persist) {
  const {EngineHost, VxInferenceServiceClient, pb} = api;
  const started = performance.now();
  for (const [file, expected] of Object.entries(spec.model.sha256)) {
    const response = await fetch(spec.model.base + '/' + file);
    requireTrue(response.ok && await digest(await response.arrayBuffer()) === expected, 'Model hash mismatch: ' + file);
  }
  const host = new EngineHost({wasmUrl: spec.wasmUrl});
  try {
    const client = new VxInferenceServiceClient(host);
    const runtime = await client.createRuntime(new pb.CreateRuntimeRequest({cpuThreads: 1}));
    const model = await client.loadModel(new pb.LoadModelRequest({runtimeId: runtime.runtimeId,
      graphPath: spec.model.base + '/graph.json', weightPaths: [spec.model.base + '/model.safetensors']}));
    const arithmetic = spec.model.packed_arithmetic === 'dynamic-i8' ? pb.PackedArithmetic.PACKED_ARITHMETIC_DYNAMIC_I8
      : pb.PackedArithmetic.PACKED_ARITHMETIC_F32;
    const compiled = await client.compileModel(new pb.CompileModelRequest({modelId: model.modelId,
      packedArithmetic: arithmetic, policy: new pb.BackendPolicy({mode: pb.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE,
        backends: ['wasm'], operatorFallback: pb.OperatorFallback.OPERATOR_FALLBACK_FORBID})}));
    requireTrue(compiled.report?.backend === 'wasm' && compiled.packedArithmetic === arithmetic, 'Required WASM arithmetic unavailable');
    const context = await client.createExecutionContext(new pb.CreateExecutionContextRequest({
      compiledModelId: compiled.compiledModelId, graph: new pb.GraphExecution()}));
    const coldMs = performance.now() - started;
    const input = async item => {
      const response = await fetch(item.url);
      requireTrue(response.ok, 'Evaluation input fetch failed');
      const bytes = new Uint8Array(await response.arrayBuffer());
      requireTrue(bytes.byteLength === 320 * 672 * 4 && await digest(bytes) === item.sha256, 'Input hash/shape mismatch');
      return bytes;
    };
    const request = bytes => new pb.ExecuteRequest({contextId: context.contextId,
      scope: new pb.ExecutionScope({entrypoint: 'evaluate', full: new pb.Empty()}),
      inputs: new pb.InputBindings({values: [new pb.Tensor({name: 'input0', dtype: pb.DataType.DATA_TYPE_F32,
        shape: [1n, 1n, 320n, 672n], inline: bytes})]}),
      outputs: new pb.OutputOptions({selection: new pb.TensorOutputSelection({names: ['slot_logits']}),
        storage: pb.OutputStorage.OUTPUT_STORAGE_INLINE})});
    const execute = async prepared => {
      const before = performance.now();
      const result = await client.execute(prepared);
      requireTrue(result.report?.backend === 'wasm', 'Execution left WASM backend');
      const tensor = result.outputs.find(output => output.name === 'slot_logits');
      requireTrue(tensor?.dtype === pb.DataType.DATA_TYPE_F32 && tensor.inline?.byteLength === 704
        && JSON.stringify(tensor.shape.map(Number)) === '[1,16,11]', 'Unexpected Digit output tensor');
      const bytes = tensor.inline.slice();
      return {bytes, inference_ms: performance.now() - before};
    };
    const warmupRequest = request(await input(spec.cases[0]));
    const cold = await execute(warmupRequest), signature = await digest(cold.bytes);
    for (let repeat = 0; repeat < spec.warmups; repeat++)
      requireTrue(await digest((await execute(warmupRequest)).bytes) === signature, 'Warmup changed Digit output');
    for (let index = spec.startIndex; index < spec.cases.length; index++) {
      const item = spec.cases[index], reference = spec.references[index];
      const prepared = request(await input(item));
      const execution = await execute(prepared);
      const bytes = execution.bytes;
      const questions = item.questions ?? (typeof item.question === 'string' ? [{question: item.question, answer: item.answer}] : []);
      const scored = scoreDigitOutput(new Float32Array(bytes.buffer, bytes.byteOffset, 176), item.record, reference, {...spec, questions});
      const outputHash = await digest(bytes);
      await persist({index, model: spec.model.name, id: item.id, input_sha256: item.sha256,
        output_sha256: outputHash, inference_ms: execution.inference_ms, ...scored,
        native: {...scored.native, bytes_equal: outputHash === reference.sha256}});
    }
    return {cold_verification_load_compile_context_ms: coldMs, first_execute_ms: cold.inference_ms,
      warmups: spec.warmups, processed_this_run: spec.cases.length - spec.startIndex};
  } finally { await host.close(); }
}

if (typeof self !== 'undefined') self.onmessage = async ({data}) => {
  if (data.type === 'record-ack') {
    const pending = acknowledgments.get(data.index); acknowledgments.delete(data.index);
    if (data.error) pending?.reject(new Error(data.error)); else pending?.resolve();
    return;
  }
  try {
    const api = await import(data.jsUrl);
    const result = await runDigitEvaluation(data, api, async record => {
      const acknowledgment = new Promise((resolve, reject) => acknowledgments.set(record.index, {resolve, reject}));
      self.postMessage({type: 'record', record});
      await acknowledgment;
    });
    self.postMessage({type: 'result', ok: true, result});
  } catch (error) { self.postMessage({type: 'result', ok: false, error: {type: error.name, message: error.message}}); }
};
