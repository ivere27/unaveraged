import {runBenchmark} from './runner.js';
import * as api from './runtime/volvoxai.lite.min.js';

self.onmessage=async({data})=>{
  try{self.postMessage({ok:true,result:await runBenchmark(data,api)});}
  catch(error){self.postMessage({ok:false,error:{type:error.name,message:error.message}});}
};
