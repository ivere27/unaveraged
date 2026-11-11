import {generate, cleanText, CROSS, PAST, PRESENT} from '/vqa/adapter.js';
import {TinyReceiptByteFallbackBPEVocab} from '/vqa/tokenizer.js';

const FAMILIES = ['phone', 'address', 'store', 'item_row', 'item_math', 'item_lookup', 'math', 'other'];
const recordAcks = new Map();
const digest = async bytes => Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)))
  .map(value => value.toString(16).padStart(2, '0')).join('');
const assert = (condition, message) => { if (!condition) throw new Error(message); };
const median = values => {
  const ordered = [...values].sort((a, b) => a - b), middle = Math.floor(ordered.length / 2);
  return ordered.length % 2 ? ordered[middle] : (ordered[middle - 1] + ordered[middle]) / 2;
};
const tensor = (pb, shape, values) => ({shape,
  dtype: values instanceof Float32Array ? pb.DataType.DATA_TYPE_F32 : pb.DataType.DATA_TYPE_I32,
  inline: new Uint8Array(values.buffer, values.byteOffset, values.byteLength)});
const copyStats = state => ({encoder_ms: state.stats.encoder_ms, decoder_ms: state.stats.decoder_ms,
  encoder_rpc_ms: state.stats.encoder_rpc_ms, encoder_output_copy_ms: state.stats.encoder_output_copy_ms,
  encoder_output_bytes: state.stats.encoder_output_bytes,
  decoder_steps: state.stats.decoder_step_ms.length, decoder_step_ms: [...state.stats.decoder_step_ms]});
const encoderSignature = async state => state.captureEncoder ? Object.fromEntries(await Promise.all(
  Object.entries(state.encoderOutputs).map(async ([name, value]) => [name, {
    dtype: value.dtype, shape: value.shape.map(Number), bytes: value.inline.byteLength,
    sha256: await digest(value.inline),
  }]))) : undefined;
const sameEncoder = (actual, expected) => JSON.stringify(actual.encoder_outputs) === JSON.stringify(expected.encoder_outputs);

async function openState(model, wasmUrl, captureEncoder = false) {
  const started = performance.now();
  const api = await import('/runtime/volvoxai.lite.min.js');
  const {EngineHost, VxInferenceServiceClient, pb} = api;
  for (const [file, hash] of Object.entries(model.sha256)) {
    const response = await fetch(model.base + '/' + file);
    assert(response.ok && await digest(await response.arrayBuffer()) === hash, `Model hash mismatch: ${file}`);
  }
  const manifest = model.manifest;
  const vocab = await (await fetch(model.base + '/vocab.json')).json();
  const tokenizer = await TinyReceiptByteFallbackBPEVocab.fromJSON(vocab, {pad: 0, bos: 1, eos: 2, unk: 3}, {
    vocabSize: manifest.tokenizer.vocab_size, normalization: manifest.tokenizer.normalization,
    tokenizerHash: manifest.tokenizer.tokenizer_hash,
  });
  const host = new EngineHost({wasmUrl});
  const client = new VxInferenceServiceClient(host);
  const arithmeticName = manifest.quantization?.packed_arithmetic ?? 'f32';
  assert(['f32', 'dynamic-i8'].includes(arithmeticName), 'Unsupported packed arithmetic');
  const arithmetic = arithmeticName === 'dynamic-i8' ? pb.PackedArithmetic.PACKED_ARITHMETIC_DYNAMIC_I8
    : pb.PackedArithmetic.PACKED_ARITHMETIC_F32;
  const state = {api, host, client, manifest, tokenizer, contexts: {}, arithmeticName, captureEncoder};
  try {
    const runtime = await client.createRuntime(new pb.CreateRuntimeRequest({cpuThreads: 1}));
    state.runtimeId = runtime.runtimeId;
    for (const role of ['encoder', 'decoder']) {
      const loaded = await client.loadModel(new pb.LoadModelRequest({runtimeId: runtime.runtimeId,
        graphPath: model.base + '/' + role + '/graph.json',
        weightPaths: (manifest.graphs[role].weights ?? [role + '/model.safetensors']).map(file => model.base + '/' + file)}));
      const compiled = await client.compileModel(new pb.CompileModelRequest({modelId: loaded.modelId,
        packedArithmetic: arithmetic, policy: new pb.BackendPolicy({
          mode: pb.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE, backends: ['wasm'],
          operatorFallback: pb.OperatorFallback.OPERATOR_FALLBACK_FORBID})}));
      assert(compiled.report?.backend === 'wasm' && compiled.packedArithmetic === arithmetic,
        'Compiled backend/arithmetic differs from requested WASM execution');
      const context = await client.createExecutionContext(new pb.CreateExecutionContextRequest({
        compiledModelId: compiled.compiledModelId, graph: new pb.GraphExecution()}));
      state.contexts[role] = context.contextId;
    }
    state.reset = () => {
      state.stats = {encoder_ms: 0, decoder_ms: 0, decoder_step_ms: [],
        encoder_rpc_ms: 0, encoder_output_copy_ms: 0, encoder_output_bytes: 0};
      state.encoderOutputs = null;
      state.tokens = [tokenizer.bos];
      state.logitChunks = [];
      state.familyIndex = null;
    };
    state.run = async (role, feed, outputNames, bindings = {}) => {
      const start = performance.now();
      const mapping = manifest.graphs[role].inputs;
      const result = await client.execute(new pb.ExecuteRequest({contextId: state.contexts[role],
        scope: new pb.ExecutionScope({entrypoint: 'evaluate', full: new pb.Empty()}),
        inputs: new pb.InputBindings({
          values: Object.entries(feed).map(([name, value]) => new pb.Tensor({
            ...value, name: mapping[name], shape: value.shape.map(BigInt)})),
          reuse: (bindings.reuse ?? []).map(name => mapping[name]),
          feedback: (bindings.feedback ?? []).map(({inputName, outputName}) => new pb.TensorFeedback({
            inputName: mapping[inputName], outputName})),
        }), outputs: new pb.OutputOptions({selection: new pb.TensorOutputSelection({names: outputNames}),
          storage: pb.OutputStorage.OUTPUT_STORAGE_INLINE})}));
      const copyStart = role === 'encoder' ? performance.now() : 0;
      assert(result.report?.backend === 'wasm', 'Execution left the WASM backend');
      const outputs = {};
      for (const name of outputNames) {
        const value = result.outputs.find(item => item.name === name);
        assert(value?.inline, 'Missing requested output: ' + name);
        outputs[name] = {dtype: value.dtype, shape: [...value.shape], inline: value.inline.slice()};
      }
      const finished = performance.now(), duration = finished - start;
      state.stats[role + '_ms'] += duration;
      if (role === 'encoder') {
        state.stats.encoder_rpc_ms += copyStart - start;
        state.stats.encoder_output_copy_ms += finished - copyStart;
        state.stats.encoder_output_bytes += Object.values(outputs).reduce((sum, value) => sum + value.inline.byteLength, 0);
        if (captureEncoder) state.encoderOutputs = outputs;
      }
      if (role === 'encoder' && outputs.selected_family_ids)
        state.familyIndex = new Int32Array(outputs.selected_family_ids.inline.slice().buffer)[0];
      if (role === 'decoder') {
        state.stats.decoder_step_ms.push(duration);
        const raw = outputs.logits.inline;
        assert(raw.byteLength === 1536 * 4, 'Unexpected batch-one decoder output shape');
        const logits = new Float32Array(raw.buffer, raw.byteOffset, 1536);
        assert(logits.every(Number.isFinite), 'Nonfinite decoder output');
        let best = 0;
        for (let index = 1; index < logits.length; index++) if (logits[index] > logits[best]) best = index;
        state.tokens.push(best);
        state.logitChunks.push(raw);
      }
      return outputs;
    };
    state.reset();
    state.cold_ms = performance.now() - started;
    return state;
  } catch (error) {
    await host.close();
    throw error;
  }
}

async function greedy(state, item, image) {
  state.reset();
  let parsed = null, outputError = null;
  const started = performance.now();
  try {
    parsed = await generate({run: state.run, pb: state.api.pb, tokenizer: state.tokenizer, image, question: item.question});
  } catch (error) {
    // A malformed/nonterminated quantized answer is an accuracy result. Runtime
    // errors and nonfinite tensors must still fail the benchmark.
    const ended = state.tokens.at(-1) === state.tokenizer.eos;
    const exhausted = state.tokens.length === 192 && error.message.includes('최대 길이');
    if (!ended && !exhausted) throw error;
    outputError = error.message;
  }
  const generationMs = performance.now() - started;
  const tokens = [...state.tokens];
  assert(tokens.length > 1, 'Generation failed before producing any token');
  let generatedText = null;
  try { generatedText = state.tokenizer.decode(tokens, {errors: 'strict'}); } catch {}
  return {...copyStats(state), generation_ms: generationMs,
    ...(state.captureEncoder ? {encoder_outputs: await encoderSignature(state)} : {}),
    tokens, terminated: tokens.at(-1) === state.tokenizer.eos, valid_output: parsed !== null,
    parsed, generated_text: generatedText, selected_family: FAMILIES[state.familyIndex] ?? null,
    output_error: outputError};
}

async function fixedPrefix(state, item, image, steps) {
  const {pb} = state.api;
  state.reset();
  const ids = state.tokenizer.encode(item.question, {addEos: true});
  assert(ids.length <= 192, 'Question exceeds fixed encoder capacity');
  const padded = new Int32Array(192); padded.set(ids);
  const encoded = await state.run('encoder', {
    image: tensor(pb, [1, 1, 320, 672], image), question_ids: tensor(pb, [1, 192], padded),
    family_ids: tensor(pb, [1], new Int32Array([-1])),
  }, ['memory_padding_mask', 'selected_family_ids', ...CROSS]);
  const initial = {memory_padding_mask: encoded.memory_padding_mask,
    family_ids: encoded.selected_family_ids, past_padding_mask: tensor(pb, [1, 1], new Int32Array([1]))};
  for (const name of CROSS) initial[name] = encoded[name];
  for (const name of PAST) initial[name] = tensor(pb, [1, 8, 1, 40], new Float32Array(320));
  const bindings = {reuse: ['memory_padding_mask', 'family_ids', ...CROSS], feedback: [
    {inputName: 'past_padding_mask', outputName: 'present_padding_mask'},
    ...PAST.map((inputName, index) => ({inputName, outputName: PRESENT[index]})),
  ]};
  const prefix = item.decoder_prefix;
  assert(Array.isArray(prefix) && prefix[0] === state.tokenizer.bos, 'Fixed timing requires a BOS-prefixed reference');
  const consumed = [];
  for (let position = 0; position < steps; position++) {
    const token = position < prefix.length ? prefix[position] : state.tokenizer.eos;
    assert(Number.isInteger(token) && token >= 0 && token < 1536, 'Invalid fixed-prefix token');
    consumed.push(token);
    await state.run('decoder', {...(position === 0 ? initial : {}),
      decoder_input_ids: tensor(pb, [1, 1], new Int32Array([token])),
      position_ids: tensor(pb, [1], new Int32Array([position])),
    }, ['logits'], position === 0 ? {} : bindings);
  }
  const stats = copyStats(state);
  const bytes = new Uint8Array(state.logitChunks.reduce((sum, chunk) => sum + chunk.byteLength, 0));
  let offset = 0;
  for (const chunk of state.logitChunks) { bytes.set(chunk, offset); offset += chunk.byteLength; }
  return {...stats, ms_per_step: stats.decoder_ms / steps,
    ...(state.captureEncoder ? {encoder_outputs: await encoderSignature(state)} : {}),
    input_tokens: consumed, argmax_tokens: [...state.tokens], logits_sha256: await digest(bytes)};
}

async function profileEncoder(state, cases, inputs, references) {
  const api = state.api, {pb} = api;
  const profiler = new api.VxProfilingServiceClient(state.host);
  const trace = await profiler.startTrace(new pb.StartTraceRequest({runtimeId: state.runtimeId,
    options: new pb.TraceOptions({detail: pb.TraceDetail.TRACE_DETAIL_NODES, capacityBytes: 33554432n})}));
  const executions = cases.length * 3;
  try {
    for (let repeat = 0; repeat < 3; repeat++) for (let index = 0; index < cases.length; index++) {
      state.reset();
      const ids = state.tokenizer.encode(cleanText(cases[index].question), {addEos: true});
      assert(ids.length <= 192, 'Question exceeds fixed encoder capacity');
      const padded = new Int32Array(192); padded.set(ids);
      await state.run('encoder', {
        image: tensor(pb, [1, 1, 320, 672], inputs[index]), question_ids: tensor(pb, [1, 192], padded),
        family_ids: tensor(pb, [1], new Int32Array([-1])),
      }, ['memory_padding_mask', 'selected_family_ids', ...CROSS]);
      assert(sameEncoder({encoder_outputs: await encoderSignature(state)}, references[index].output),
        'Tracing changed encoder outputs');
    }
    await profiler.stopTrace(new pb.TraceRef({traceId: trace.traceId}));
    const summaries = {};
    for (const [key, groupBy] of Object.entries({operators: pb.TraceSummaryGrouping.TRACE_SUMMARY_GROUPING_OPERATOR,
      nodes: pb.TraceSummaryGrouping.TRACE_SUMMARY_GROUPING_NODE,
      activities: pb.TraceSummaryGrouping.TRACE_SUMMARY_GROUPING_ACTIVITY})) {
      const summary = await profiler.getTraceSummary(new pb.GetTraceSummaryRequest({traceId: trace.traceId,
        groupBy, maxRows: 4096}));
      assert(!summary.truncated && Number(summary.events.dropped) === 0 && Number(summary.excludedEvents) === 0,
        'Incomplete encoder trace: ' + key);
      summaries[key] = summary.rows.map(row => ({name: row.name, backend: row.backend, domain: row.domain,
        phase: row.phase, activity: row.activity, output_name: row.outputName,
        source_node_ids: [...row.sourceNodeIds], schedule_index: row.scheduleIndex,
        count: Number(row.count), total_ms: Number(row.totalNs) / 1e6,
        mean_per_execute_ms: Number(row.totalNs) / 1e6 / executions,
        median_ms: Number(row.medianNs) / 1e6, copy_bytes: Number(row.copyBytes)}));
    }
    return {executions, scope: 'Separate encoder-only diagnostic trace after all warm timings; host operator spans, not end-to-end latency',
      dropped_events: 0, excluded_events: 0, ...summaries};
  } finally {
    await profiler.releaseTrace(new pb.TraceRef({traceId: trace.traceId}));
  }
}

export async function runVqaBenchmark(spec) {
  const inputs = [];
  for (const item of spec.cases) {
    const response = await fetch(item.url);
    assert(response.ok, 'Input fetch failed');
    const raw = await response.arrayBuffer();
    assert(raw.byteLength === 320 * 672 * 4 && await digest(raw) === item.sha256, 'Input hash/shape mismatch');
    inputs.push(new Float32Array(raw));
  }
  const names = Object.keys(spec.models), states = {}, results = {};
  const progress = value => self.postMessage({type: 'progress', ...value});
  const compare = (name, item, output) => {
    const reference = item.references?.[name]?.tokens;
    return reference ? JSON.stringify(reference) === JSON.stringify(output.tokens) : null;
  };
  try {
    for (const name of names) {
      progress({stage: 'load', model: name});
      const state = states[name] = await openState(spec.models[name], spec.wasmUrl, true);
      results[name] = {sha256: spec.models[name].sha256, package_bytes: spec.models[name].package_bytes,
        packed_arithmetic: state.arithmeticName, cold_verification_load_compile_ms: state.cold_ms,
        cases: []};
      for (let index = 0; index < spec.cases.length; index++) {
        const item = spec.cases[index];
        const first = await greedy(state, item, inputs[index]);
        const fixed = spec.fixedSteps ? await fixedPrefix(state, item, inputs[index], spec.fixedSteps) : null;
        const result = {id: item.id, input_sha256: item.sha256, question: item.question,
          output: first, native_tokens_equal: compare(name, item, first), fixed_prefix: fixed,
          samples: [], fixed_samples: []};
        results[name].cases.push(result);
        for (let repeat = 0; repeat < spec.warmups; repeat++) {
          const warm = await greedy(state, item, inputs[index]);
          assert(JSON.stringify(warm.tokens) === JSON.stringify(first.tokens), 'Warmup changed greedy tokens');
          assert(sameEncoder(warm, first), 'Warmup changed encoder outputs');
          if (fixed) {
            const warmFixed = await fixedPrefix(state, item, inputs[index], spec.fixedSteps);
            assert(warmFixed.logits_sha256 === fixed.logits_sha256, 'Warmup changed fixed-prefix logits');
            assert(sameEncoder(warmFixed, fixed), 'Warmup changed fixed-prefix encoder outputs');
          }
        }
        progress({stage: 'warmup', model: name, case: item.id});
      }
    }
    for (let cycle = 0; cycle < spec.cycles; cycle++) {
      for (let index = 0; index < spec.cases.length; index++) {
        const item = spec.cases[index];
        for (const name of cycle % 2 ? [...names].reverse() : names) {
          const result = results[name].cases[index];
          const output = await greedy(states[name], item, inputs[index]);
          assert(JSON.stringify(output.tokens) === JSON.stringify(result.output.tokens), 'Measured greedy tokens changed');
          assert(sameEncoder(output, result.output), 'Measured encoder outputs changed');
          result.samples.push({generation_ms: output.generation_ms, encoder_ms: output.encoder_ms,
            encoder_rpc_ms: output.encoder_rpc_ms, encoder_output_copy_ms: output.encoder_output_copy_ms,
            encoder_output_bytes: output.encoder_output_bytes,
            decoder_ms: output.decoder_ms, decoder_steps: output.decoder_steps,
            decoder_step_ms: output.decoder_step_ms});
          if (spec.fixedSteps) {
            const fixed = await fixedPrefix(states[name], item, inputs[index], spec.fixedSteps);
            assert(fixed.logits_sha256 === result.fixed_prefix.logits_sha256, 'Measured fixed-prefix logits changed');
            assert(sameEncoder(fixed, result.fixed_prefix), 'Measured fixed-prefix encoder outputs changed');
            result.fixed_samples.push({encoder_ms: fixed.encoder_ms, decoder_ms: fixed.decoder_ms,
              encoder_rpc_ms: fixed.encoder_rpc_ms, encoder_output_copy_ms: fixed.encoder_output_copy_ms,
              encoder_output_bytes: fixed.encoder_output_bytes,
              decoder_steps: fixed.decoder_steps, decoder_step_ms: fixed.decoder_step_ms, ms_per_step: fixed.ms_per_step});
          }
        }
      }
      progress({stage: 'cycle', cycle: cycle + 1, cycles: spec.cycles});
    }
    if (spec.profileOperators) for (const name of names) {
      progress({stage: 'profile-encoder', model: name});
      results[name].encoder_profile = await profileEncoder(states[name], spec.cases, inputs, results[name].cases);
    }
    for (const model of Object.values(results)) {
      for (const result of model.cases) {
        result.median = Object.fromEntries(['generation_ms', 'encoder_ms', 'decoder_ms'].map(key =>
          [key, median(result.samples.map(sample => sample[key]))]));
        if (spec.fixedSteps) result.fixed_median = Object.fromEntries(['decoder_ms', 'ms_per_step'].map(key =>
          [key, median(result.fixed_samples.map(sample => sample[key]))]));
      }
      model.pooled_median_generation_ms = median(model.cases.flatMap(item => item.samples.map(row => row.generation_ms)));
      if (spec.fixedSteps) model.pooled_median_fixed_decoder_ms_per_step = median(model.cases.flatMap(item =>
        item.fixed_samples.map(row => row.ms_per_step)));
    }
    const comparisons = Object.values(results).flatMap(model => model.cases.map(item => item.native_tokens_equal))
      .filter(item => item !== null);
    return {status: comparisons.some(equal => !equal) ? 'completed-with-native-differences' : 'passed',
      backend: 'wasm', cpu_threads: 1, batch_size: 1, cycles: spec.cycles, warmups_per_input: spec.warmups,
      fixed_decoder_steps: spec.fixedSteps, models: results,
      native_token_comparisons: comparisons.length,
      native_token_parity_passed: comparisons.length ? comparisons.every(Boolean) : null,
      all_outputs_valid: Object.values(results).every(model => model.cases.every(item => item.output.valid_output)),
      repeated_encoder_outputs_bit_exact: true,
      output_validation_scope: 'SHA-256 of every returned encoder tensor (dtype and shape retained), greedy tokens, and fixed-prefix logits; digests computed outside timers',
      encoder_transport_scope: 'encoder_rpc_ms includes request construction, transport and execution; encoder_output_copy_ms measures JS output selection and inline.slice only',
      order: 'Alternate forward/reverse model order each measured cycle', host_idle_claimed: false,
      execution_context: 'Real Chrome dedicated module worker',
      greedy_timing_scope: 'Full autoregressive generation; output lengths can differ between quantized models',
      fixed_timing_scope: spec.fixedSteps ? 'Identical reference token prefix and cache lengths; EOS pads short prefixes; timing only, not task accuracy' : null,
      timer_includes: ['execute dispatch', 'input/output transport', 'copied selected outputs',
        'tokenization/argmax/parsing in total generation time'],
      timer_excludes: ['image preprocessing', 'model verification/load/compile', 'warmups', 'UI', 'extension queue'],
    };
  } finally {
    for (const state of Object.values(states)) await state.host.close();
  }
}

export async function runVqaEvaluation(spec) {
  const state = await openState(spec.model, spec.wasmUrl);
  const cold = state.cold_ms;
  let processed = 0;
  try {
    for (let index = spec.startIndex; index < spec.cases.length; index++) {
      const item = spec.cases[index];
      const response = await fetch(item.url);
      assert(response.ok, 'Evaluation input fetch failed');
      const raw = await response.arrayBuffer();
      assert(raw.byteLength === 320 * 672 * 4 && await digest(raw) === item.sha256,
        'Evaluation input hash/shape mismatch');
      const output = await greedy(state, item, new Float32Array(raw));
      const valid = output.terminated && output.valid_output;
      const parsed = output.parsed;
      const record = {index, id: item.id, input_sha256: item.sha256, model: spec.model.name,
        ...output, expected_answer: item.answer,
        scores: {
          structured_valid: valid,
          answer_correct: valid && cleanText(parsed.answer) === cleanText(item.answer),
          field_correct: item.field == null ? null : valid && parsed.field === item.field,
          op_correct: item.op == null ? null : valid && parsed.op === item.op,
          target_correct: item.target == null ? null : valid && cleanText(output.generated_text) === cleanText(item.target),
          family_correct: item.family == null ? null : output.selected_family === item.family,
        }};
      // Parent acknowledges only after the private JSONL row is durably appended.
      const persisted = new Promise((resolve, reject) => recordAcks.set(index, {resolve, reject}));
      self.postMessage({type: 'record', record});
      await persisted;
      processed++;
      if ((index + 1) % 25 === 0 || index + 1 === spec.cases.length)
        self.postMessage({type: 'progress', stage: 'evaluation', completed: index + 1, total: spec.cases.length});
    }
    return {status: 'passed', backend: 'wasm', cpu_threads: 1, batch_size: 1,
      processed_this_run: processed, cold_verification_load_compile_ms: cold,
      execution_context: 'Real Chrome dedicated module worker',
      timing_scope: 'One full greedy generation per evaluation record; image file reading/preprocessing and model load excluded',
      host_idle_claimed: false};
  } finally {
    await state.host.close();
  }
}

self.onmessage = async ({data}) => {
  if (data.type === 'record-ack') {
    const pending = recordAcks.get(data.index);
    recordAcks.delete(data.index);
    if (data.error) pending?.reject(new Error(data.error)); else pending?.resolve();
    return;
  }
  try { self.postMessage({type: 'result', ok: true,
    result: await (data.mode === 'evaluation' ? runVqaEvaluation(data) : runVqaBenchmark(data))}); }
  catch (error) { self.postMessage({type: 'result', ok: false, error: {type: error.name, message: error.message}}); }
};
