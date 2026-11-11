#!/usr/bin/env node
import {createHash} from 'node:crypto';
import {readFile,writeFile,mkdir} from 'node:fs/promises';
import http from 'node:http';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import puppeteer from 'puppeteer';
import {getInstalledBrowsers,Browser} from '@puppeteer/browsers';
import {runBenchmark} from './runtime-benchmark.js';

const root=fileURLToPath(new URL('../',import.meta.url));
const runtimeVersion = JSON.parse(await readFile(path.join(root, 'package.json'), 'utf8')).config.volvoxai_version;
const argument=(name,fallback)=>process.argv.includes(name)?process.argv[process.argv.indexOf(name)+1]:fallback;
const installedVersion = JSON.parse(await readFile(path.join(root, 'node_modules/volvoxai-lite/package.json'), 'utf8')).version;
if (installedVersion !== runtimeVersion) throw new Error('Installed volvoxai-lite differs from the project pin; run npm ci');
const requested=argument('--browser','all');
if(!['all','node','chrome','firefox'].includes(requested))throw new Error('Use --browser all|node|chrome|firefox');
const out=path.resolve(argument('--out',path.join(root,'.cache/benchmarks/retest')));
const fixture=JSON.parse(await readFile(path.join(out,'fixtures.json'),'utf8'));
const runtime=path.join(root,'node_modules/volvoxai-lite/dist',runtimeVersion);
const files=new Map();
for(const name of ['volvoxai.lite.min.js','volvoxai.lite.wasm'])files.set('/runtime/'+name,path.join(runtime,name));
files.set('/runner.js',fileURLToPath(new URL('./runtime-benchmark.js',import.meta.url)));
files.set('/worker.js',fileURLToPath(new URL('./worker.js',import.meta.url)));
for(const [name,model]of Object.entries(fixture.models))for(const file of ['graph.json','model.safetensors'])
  files.set('/models/'+name+'/'+file,path.join(model.directory,file));
for(const item of fixture.cases)files.set('/inputs/'+item.file,path.join(out,item.file));
const server=http.createServer(async(request,response)=>{
  try{
    if(request.url==='/'){response.writeHead(200,{'Content-Type':'text/html'});response.end('<!doctype html><title>unaveraged WASM benchmark</title>');return;}
    const file=files.get(request.url);if(!file){response.writeHead(404);response.end();return;}
    const type=file.endsWith('.js')?'text/javascript':file.endsWith('.wasm')?'application/wasm':file.endsWith('.json')?'application/json':'application/octet-stream';
    response.writeHead(200,{'Content-Type':type});response.end(await readFile(file));
  }catch(error){response.writeHead(500);response.end('Fixture error');}
});
await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
const origin='http://127.0.0.1:'+server.address().port;
const profiling=process.argv.includes('--profile-operators');
const spec={...fixture,profileOperators:profiling,wasmUrl:origin+'/runtime/volvoxai.lite.wasm',
  models:Object.fromEntries(Object.entries(fixture.models).map(([name,item])=>[name,{
    sha256:item.sha256,packed_arithmetic:item.packed_arithmetic,base:origin+'/models/'+name}])),
  cases:fixture.cases.map(item=>({...item,url:origin+'/inputs/'+item.file}))};
const hash=bytes=>createHash('sha256').update(bytes).digest('hex');
const runtimeHashes=Object.fromEntries(await Promise.all(['volvoxai.lite.min.js','volvoxai.lite.wasm'].map(async name=>[name,hash(await readFile(path.join(runtime,name)))])));
const reports=[];
async function firefoxPath(){
  if(process.env.RECEIPT_FIREFOX)return process.env.RECEIPT_FIREFOX;
  const installed=(await getInstalledBrowsers({cacheDir:await puppeteer.defaultDownloadPath()}))
    .filter(item=>item.browser===Browser.FIREFOX)
    .sort((a,b)=>b.buildId.localeCompare(a.buildId,undefined,{numeric:true}));
  if(!installed.length)throw new Error('Install Firefox with: npm exec -- puppeteer browsers install firefox@stable');
  return installed[0].executablePath;
}
try{
  for(const name of requested==='all'?['node','chrome','firefox']:[requested]){
    let browser;
    const report={format:'unaveraged-wasm-benchmark/v1',engine:name,status:'running',
      created_at:new Date().toISOString(),runtime:{package:'volvoxai-lite',version:runtimeVersion,sha256:runtimeHashes}};
    try{
      if(name==='node'){
        const api=await import('volvoxai-lite/min');
        Object.assign(report,await runBenchmark(spec,api),{version:process.version,execution_context:'Node main thread'});
      }else{
        browser=await puppeteer.launch({browser:name==='firefox'?'firefox':'chrome',headless:true,
          executablePath:name==='firefox'?await firefoxPath():(process.env.RECEIPT_CHROME||undefined),
          args:name==='firefox'?[]:['--no-sandbox','--disable-dev-shm-usage'],protocolTimeout:300000});
        const page=await browser.newPage();await page.goto(origin);await page.bringToFront();
        const errors=[];page.on('pageerror',e=>errors.push(String(e)));
        const result=await page.evaluate(async({spec,origin})=>{
          return await new Promise((resolve,reject)=>{
            const worker=new Worker(origin+'/worker.js',{type:'module'});
            worker.onmessage=({data})=>{worker.terminate();data.ok?resolve(data.result):reject(new Error(data.error.message));};
            worker.onerror=error=>{worker.terminate();reject(new Error(error.message));};
            worker.postMessage(spec);
          });
        },{spec,origin});
        if(errors.length)throw new Error(errors.join('\n'));
        Object.assign(report,result,{version:await browser.version(),execution_context:'Dedicated module worker'});
      }
      console.log('WASM',report.status,name,Object.fromEntries(Object.entries(report.models).map(([n,m])=>[n,m.pooled_median_ms])));
    }catch(error){report.status='failed';report.error={type:error.name,message:error.message};console.error('WASM FAIL',name,error.message);}
    finally{await browser?.close();await mkdir(out,{recursive:true});await writeFile(path.join(out,`${profiling?'profile':'wasm'}-${name}.json`),JSON.stringify(report,null,2)+'\n');}
    reports.push(report);
  }
}finally{await new Promise(resolve=>server.close(resolve));}
if(reports.some(r=>r.status!=='passed'))process.exitCode=1;
