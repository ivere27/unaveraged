// Shared numerical/timing loop for Node, Chrome and Firefox WASM engines.
export async function runBenchmark(spec, api, runtimeApis = {}) {
  const batchSize=spec.batchSize??1;
  if(!Number.isInteger(batchSize)||batchSize<1||batchSize>8)throw new Error('Unsupported benchmark batch size');
  const digest=async bytes=>Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)))
    .map(x=>x.toString(16).padStart(2,'0')).join('');
  const requireTrue=(value,message)=>{if(!value)throw new Error(message);};
  const median=a=>{const v=[...a].sort((x,y)=>x-y),i=Math.floor(v.length/2);return v.length%2?v[i]:(v[i-1]+v[i])/2;};
  const inputs=[];
  for(const item of spec.cases){
    const response=await fetch(item.url);requireTrue(response.ok,'Input fetch failed');
    const bytes=new Uint8Array(await response.arrayBuffer());
    requireTrue(await digest(bytes)===item.sha256,'Input hash mismatch');
    inputs.push(bytes);
  }
  const states={};const samples={};const parity=[];const cold={};
  const names=Object.keys(spec.models);
  const execute=async(name,i)=>{
    const result=await states[name].client.execute(states[name].requests[i]);
    requireTrue(result.report.backend==='wasm','WASM backend required');
    const output=result.outputs.find(x=>x.name==='slot_logits');
    requireTrue(output?.inline?.byteLength===704*batchSize,'Unexpected slot output');
    return output.inline.slice();
  };
  const validate=async(name,i,raw)=>{
    const reference=spec.cases[i].references[name];
    const values=new Float32Array(raw.buffer,raw.byteOffset,176*batchSize);let maxAbs=0,failedLogits=0,maxToleranceRatio=0;
    requireTrue(values.every(Number.isFinite),'Nonfinite output');
    const slots=[];
    for(let row=0;row<16*batchSize;row++){
      let best=0;for(let c=1;c<11;c++)if(values[row*11+c]>values[row*11+best])best=c;
      slots.push(best);
    }
    const slotsEqual=JSON.stringify(slots)===JSON.stringify(reference.slots);
    for(let j=0;j<176*batchSize;j++){
      const delta=Math.abs(values[j]-reference.values[j]);maxAbs=Math.max(maxAbs,delta);
      const tolerance=spec.atol+spec.rtol*Math.abs(reference.values[j]);
      if(delta>tolerance)failedLogits++;
      maxToleranceRatio=Math.max(maxToleranceRatio,delta/tolerance);
    }
    return {profile:name,input_index:i,slots_equal:slotsEqual,max_abs:maxAbs,
      logits_within_tolerance:failedLogits===0,outside_tolerance:failedLogits,max_tolerance_ratio:maxToleranceRatio,
      native_bytes_equal:await digest(raw)===reference.sha256};
  };
  try{
    for(const name of names){
      const model=spec.models[name];const started=performance.now();
      const selectedApi=runtimeApis[model.runtime]??api;
      const {EngineHost,VxInferenceServiceClient,pb}=selectedApi;
      requireTrue(['f32','dynamic-i8'].includes(model.packed_arithmetic),'Explicit supported arithmetic required');
      const arithmetic=model.packed_arithmetic==='dynamic-i8'
        ?pb.PackedArithmetic.PACKED_ARITHMETIC_DYNAMIC_I8:pb.PackedArithmetic.PACKED_ARITHMETIC_F32;
      for(const file of ['graph.json','model.safetensors']){
        const response=await fetch(model.base+'/'+file);requireTrue(response.ok,'Model fetch failed');
        requireTrue(await digest(await response.arrayBuffer())===model.sha256[file],'Model hash mismatch');
      }
      const host=new EngineHost({wasmUrl:model.wasmUrl??spec.wasmUrl});states[name]={host,api:selectedApi};
      const client=new VxInferenceServiceClient(host);
      const runtime=await client.createRuntime(new pb.CreateRuntimeRequest({cpuThreads:1}));
      states[name].runtimeId=runtime.runtimeId;
      const loaded=await client.loadModel(new pb.LoadModelRequest({runtimeId:runtime.runtimeId,
        graphPath:model.base+'/graph.json',weightPaths:[model.base+'/model.safetensors']}));
      const compiled=await client.compileModel(new pb.CompileModelRequest({modelId:loaded.modelId,
        packedArithmetic:arithmetic,
        policy:new pb.BackendPolicy({mode:pb.BackendPolicyMode.BACKEND_POLICY_MODE_REQUIRE,
          backends:['wasm'],operatorFallback:pb.OperatorFallback.OPERATOR_FALLBACK_FORBID})}));
      requireTrue(compiled.packedArithmetic===arithmetic,'Compiled arithmetic differs from model profile');
      const context=await client.createExecutionContext(new pb.CreateExecutionContextRequest({
        compiledModelId:compiled.compiledModelId,graph:new pb.GraphExecution()}));
      states[name].client=client;
      states[name].signatures=[];
      if(spec.comparisonPairs)states[name].referenceOutputs=[];
      states[name].requests=inputs.map(bytes=>new pb.ExecuteRequest({contextId:context.contextId,
        scope:new pb.ExecutionScope({entrypoint:'evaluate',full:new pb.Empty()}),
        inputs:new pb.InputBindings({values:[new pb.Tensor({name:'input0',dtype:pb.DataType.DATA_TYPE_F32,
          shape:[BigInt(batchSize),1n,320n,672n],inline:bytes})]}),
        outputs:new pb.OutputOptions({selection:new pb.TensorOutputSelection({names:['slot_logits']}),
          storage:pb.OutputStorage.OUTPUT_STORAGE_INLINE})}));
      cold[name]={verification_load_compile_context_ms:performance.now()-started};
      samples[name]=inputs.map(()=>[]);
      for(let i=0;i<inputs.length;i++){
        const firstStart=performance.now();const first=await execute(name,i);
        if(i===0)cold[name].first_execute_ms=performance.now()-firstStart;
        parity.push(await validate(name,i,first));
        const signature=await digest(first);
        states[name].signatures[i]=signature;
        if(spec.comparisonPairs)states[name].referenceOutputs[i]=first;
        for(let repeat=0;repeat<spec.warmups;repeat++){
          requireTrue(await digest(await execute(name,i))===signature,'Warmup changed output');
        }
      }
    }
    const runtimeComparisons=[];
    for(const [baseline,candidate] of spec.comparisonPairs??[]){
      requireTrue(states[baseline]&&states[candidate],'Unknown runtime comparison model');
      for(let i=0;i<inputs.length;i++){
        const a=states[baseline].referenceOutputs[i],b=states[candidate].referenceOutputs[i];
        runtimeComparisons.push({baseline,candidate,input_index:i,
          bytes_equal:a.length===b.length&&a.every((value,j)=>value===b[j]),
          baseline_sha256:states[baseline].signatures[i],candidate_sha256:states[candidate].signatures[i]});
      }
    }
    for(let cycle=0;cycle<spec.cycles;cycle++){
      for(let i=0;i<inputs.length;i++)for(const name of cycle%2?[...names].reverse():names){
        const started=performance.now();const raw=await execute(name,i);
        samples[name][i].push(performance.now()-started);
        // Full numerical validation is outside the measured interval.
        const comparison=await validate(name,i,raw);
        requireTrue(await digest(raw)===states[name].signatures[i],'Measured repeated WASM output changed');
        // Record cross-backend drift, and still require deterministic repeated WASM outputs.
        requireTrue(comparison.slots_equal===parity.find(p=>p.profile===name&&p.input_index===i).slots_equal,
          'Measured WASM slot parity changed');
      }
    }
    const operatorProfiles={};
    if(spec.profileOperators){
      for(const name of names){
        const {pb,VxProfilingServiceClient}=states[name].api;
        const profiler=new VxProfilingServiceClient(states[name].host);
        const trace=await profiler.startTrace(new pb.StartTraceRequest({runtimeId:states[name].runtimeId,
          options:new pb.TraceOptions({detail:pb.TraceDetail.TRACE_DETAIL_NODES,capacityBytes:4194304n})}));
        try{
          for(let repeat=0;repeat<5;repeat++)for(let i=0;i<inputs.length;i++){
            requireTrue(await digest(await execute(name,i))===states[name].signatures[i],'Tracing changed output');
          }
          await profiler.stopTrace(new pb.TraceRef({traceId:trace.traceId}));
          const summary=await profiler.getTraceSummary(new pb.GetTraceSummaryRequest({traceId:trace.traceId,
            groupBy:pb.TraceSummaryGrouping.TRACE_SUMMARY_GROUPING_OPERATOR,maxRows:256}));
          requireTrue(!summary.truncated&&Number(summary.events.dropped)===0,'Incomplete trace');
          operatorProfiles[name]={executions:5*inputs.length,scope:'Traced host operator times; diagnostic, separate from warm benchmark',
            rows:summary.rows.map(row=>({name:row.name,backend:row.backend,domain:row.domain,
              count:Number(row.count),total_ms:Number(row.totalNs)/1e6,
              mean_per_execute_ms:Number(row.totalNs)/1e6/(5*inputs.length),
              median_call_ms:Number(row.medianNs)/1e6,share:row.share}))};
        }finally{await profiler.releaseTrace(new pb.TraceRef({traceId:trace.traceId}));}
      }
    }
    const numericalPassed=parity.every(p=>p.slots_equal&&p.logits_within_tolerance);
    return {status:numericalPassed?'passed':'completed-with-numerical-differences',
      numerical_parity_passed:numericalPassed,slot_parity_passed:parity.every(p=>p.slots_equal),
      tolerance:{rtol:spec.rtol,atol:spec.atol},backend:'wasm',cpu_threads:1,cycles:spec.cycles,warmups_per_input:spec.warmups,
      batch_size:batchSize,input_count:inputs.length,input_source:spec.inputSource??'Two public HF receipts and a deterministic synthetic gradient',
      order:'Alternate forward/reverse model order each cycle',host_idle_claimed:false,
      timer_includes:['execute dispatch, input/output transport and copied inline output'],
      timer_excludes:['preprocessing','model verification','load','compile','context creation','numerical validation','UI','extension queue'],
      cold_scope:'Separate load/verification/compile/context and first execution observations; one observation per profile, not a cold-start distribution.',
      ...(spec.profileOperators?{operator_profiles:operatorProfiles}:{}),
      ...(spec.comparisonPairs?{runtime_comparisons:runtimeComparisons}:{}),
      parity,models:Object.fromEntries(names.map(name=>[name,{sha256:spec.models[name].sha256,
        packed_arithmetic:spec.models[name].packed_arithmetic,
        numerical_parity_passed:parity.filter(p=>p.profile===name).every(p=>p.slots_equal&&p.logits_within_tolerance),
        slot_parity_passed:parity.filter(p=>p.profile===name).every(p=>p.slots_equal),
        cold:cold[name],samples_ms_per_input:samples[name],median_ms_per_input:samples[name].map(median),
        pooled_median_ms:median(samples[name].flat())}]))};
  }finally{for(const state of Object.values(states))await state.host.close();}
}
