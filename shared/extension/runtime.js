// One engine/heap owns every graph. No model-specific preprocessing or decoding.
import { EngineHost, VxInferenceServiceClient, VxProfilingServiceClient, pb } from '../../vendor/volvoxai.lite.min.js';
export { pb };

// Retain existing intermediate tensors by exposing them in a derived document.
// Operators, inputs and weights stay identical to the packaged graph.
export function exposeGraphOutputs(document, outputs, entrypoint = 'evaluate') {
  const graph = structuredClone(document);
  const entry = graph.execution?.entrypoints?.find((value) => value.name === entrypoint);
  if (!Array.isArray(graph.outputs) || !Array.isArray(entry?.outputs)) {
    throw new Error('모델의 출력 실행 계약을 확인하지 못했습니다.');
  }
  for (const expected of outputs) {
    const found = graph.nodes.flatMap((node) => Object.values(node.outputs || {})
      .filter((value) => value.tensor === expected.name).map((value) => ({ node, value })));
    if (found.length !== 1 || found[0].node.opType !== expected.opType
        || found[0].value.dtype !== expected.dtype
        || JSON.stringify(found[0].value.shape) !== JSON.stringify(expected.shape)
        || (expected.axis !== undefined && found[0].node.params?.axis !== expected.axis)) {
      throw new Error(`어텐션 출력 형식이 올바르지 않습니다: ${expected.name}`);
    }
    if (!graph.outputs.includes(expected.name)) graph.outputs.push(expected.name);
    if (!entry.outputs.includes(expected.name)) entry.outputs.push(expected.name);
  }
  return graph;
}

export async function openSessions(models) {
  const host = new EngineHost({ wasmUrl: new URL('../../vendor/volvoxai.lite.wasm', import.meta.url) });
  const client = new VxInferenceServiceClient(host);
  const sessions = {};
  try {
    const runtime = await client.createRuntime(new pb.CreateRuntimeRequest());
    const profiler = new VxProfilingServiceClient(host);
    for (const [name, { graph, weights, entrypoint = 'evaluate', packedArithmetic = 'f32', exposeOutputs = [] }] of Object.entries(models)) {
      const arithmetic = {
        f32: pb.PackedArithmetic.PACKED_ARITHMETIC_F32,
        'dynamic-i8': pb.PackedArithmetic.PACKED_ARITHMETIC_DYNAMIC_I8,
      }[packedArithmetic];
      if (arithmetic === undefined) throw new Error(`지원하지 않는 모델 연산 형식입니다: ${packedArithmetic}`);
      let source = { graphPath: new URL(graph, import.meta.url).href };
      if (exposeOutputs.length) {
        const response = await fetch(source.graphPath);
        if (!response.ok) throw new Error('어텐션 모델 그래프를 읽지 못했습니다.');
        source = { graphDocument: new TextEncoder().encode(JSON.stringify(
          exposeGraphOutputs(await response.json(), exposeOutputs, entrypoint))) };
      }
      const model = await client.loadModel(new pb.LoadModelRequest({
        runtimeId: runtime.runtimeId, ...source,
        weightPaths: weights.map((path) => new URL(path, import.meta.url).href),
      }));
      const compiled = await client.compileModel(new pb.CompileModelRequest({
        modelId: model.modelId, packedArithmetic: arithmetic,
        policy: new pb.BackendPolicy({ mode: pb.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE,
          backends: ['wasm'], operatorFallback: pb.OperatorFallback.OPERATOR_FALLBACK_FORBID }),
      }));
      if (compiled.report?.backend !== 'wasm') throw new Error('WASM 모델 실행 경로를 확인하지 못했습니다.');
      if (compiled.packedArithmetic !== arithmetic) throw new Error('모델의 양자화 연산 형식을 확인하지 못했습니다.');
      const context = await client.createExecutionContext(new pb.CreateExecutionContextRequest({
        compiledModelId: compiled.compiledModelId, graph: new pb.GraphExecution(),
      }));
      sessions[name] = { contextId: context.contextId, entrypoint };
    }
    return {
      async profile(operation) {
        let trace;
        try {
          trace = await profiler.startTrace(new pb.StartTraceRequest({ runtimeId: runtime.runtimeId,
            options: new pb.TraceOptions({ detail: pb.TraceDetail.TRACE_DETAIL_NODES,
              capacityBytes: 1_048_576n, memory: true }) }));
        } catch {
          // Diagnostic collection must not prevent a normal receipt read.
          return { value: await operation(), profile: { error: 'profiling-unavailable' } };
        }
        const ref = new pb.TraceRef({ traceId: trace.traceId });
        try {
          const value = await operation();
          try {
            await profiler.stopTrace(ref);
            const summary = await profiler.getTraceSummary(new pb.GetTraceSummaryRequest({
              traceId: trace.traceId, groupBy: pb.TraceSummaryGrouping.TRACE_SUMMARY_GROUPING_OPERATOR, maxRows: 32,
            }));
            // Summary includes operator metadata and timings, never tensor values.
            return { value, profile: summary.toJson() };
          } catch { return { value, profile: { error: 'profiling-unavailable' } }; }
        } finally {
          await profiler.stopTrace(ref).catch(() => {});
          await profiler.releaseTrace(ref).catch(() => {});
        }
      },
      async run(name, inputs, outputNames, { reuse = [], feedback = [] } = {}) {
        if (!Object.hasOwn(sessions, name)) throw new Error(`Unknown model: ${name}`);
        const session = sessions[name];
        const result = await client.execute(new pb.ExecuteRequest({
          contextId: session.contextId,
          scope: new pb.ExecutionScope({ entrypoint: session.entrypoint, full: new pb.Empty() }),
          inputs: new pb.InputBindings({ reuse,
            feedback: feedback.map((binding) => new pb.TensorFeedback(binding)),
            values: inputs.map((input) => new pb.Tensor({
            ...input, shape: input.shape.map(BigInt),
          })) }),
          outputs: new pb.OutputOptions({
            selection: new pb.TensorOutputSelection({ names: outputNames }),
            storage: pb.OutputStorage.OUTPUT_STORAGE_INLINE,
          }),
        }));
        // Only selected outputs cross the transport. Other graph outputs remain
        // in the context for the next call's checked feedback bindings.
        const outputs = {};
        for (const output of outputNames) {
          const tensor = result.outputs.find((value) => value.name === output);
          if (!tensor?.inline) throw new Error(`모델 출력을 읽지 못했습니다: ${output}`);
          outputs[output] = { dtype: tensor.dtype, shape: [...tensor.shape], inline: tensor.inline.slice() };
        }
        return outputs;
      },
      close: () => host.close(),
    };
  } catch (error) {
    await host.close();
    throw error;
  }
}

// Single-graph adapters can keep the same API as the digit-reader extension.
export async function openSession({ graph = '../../model/graph.json', weights = ['../../model/model.safetensors'],
  packedArithmetic = 'f32', exposeOutputs = [] } = {}) {
  const sessions = await openSessions({ model: { graph, weights, packedArithmetic, exposeOutputs } });
  return { run: (inputs, outputs) => sessions.run('model', inputs, outputs), close: sessions.close, profile: sessions.profile };
}
