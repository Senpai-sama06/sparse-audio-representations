// Open the standalone report as a local file in headless Edge with networking off.
import {spawn} from 'node:child_process';
import {readFile,writeFile,mkdir} from 'node:fs/promises';
import {resolve} from 'node:path';
import {pathToFileURL} from 'node:url';

const root=resolve('runs/sparse_ae_features_20261110/browser_qa');
await mkdir(root,{recursive:true});
const report=resolve('approaches/sparse-ae/reports/sparse_ae_feature_inspection.html');
const verification=resolve('approaches/sparse-ae/reports/sparse_ae_feature_inspection_verification.json');
const proc=spawn('C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',[
  '--headless=new','--disable-gpu','--no-first-run','--no-default-browser-check',
  '--disable-background-networking','--autoplay-policy=no-user-gesture-required',
  '--remote-debugging-port=9337',`--user-data-dir=${root}/profile`,'about:blank'
],{windowsHide:true,stdio:'ignore'});
let socket;
try {
  let targets;
  for(let i=0;i<80;i++) {
    try {targets=await (await fetch('http://127.0.0.1:9337/json/list')).json();if(targets.some(t=>t.type==='page'))break;}catch{}
    await new Promise(r=>setTimeout(r,250));
  }
  const target=targets.find(t=>t.type==='page');
  socket=new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((ok,fail)=>{socket.onopen=ok;socket.onerror=fail;});
  let id=0;const waiting=new Map();
  socket.onmessage=e=>{const m=JSON.parse(e.data);if(m.id&&waiting.has(m.id)){const {ok,fail}=waiting.get(m.id);waiting.delete(m.id);m.error?fail(m.error):ok(m.result);}};
  const call=(method,params={})=>new Promise((ok,fail)=>{const n=++id;waiting.set(n,{ok,fail});socket.send(JSON.stringify({id:n,method,params}));});
  const evaluate=async expression=>{
    const r=await call('Runtime.evaluate',{expression,awaitPromise:true,returnByValue:true});
    if(r.exceptionDetails)throw new Error(JSON.stringify(r.exceptionDetails));
    return r.result.value;
  };
  await call('Page.enable');await call('Network.enable');
  await call('Network.emulateNetworkConditions',{offline:true,latency:0,downloadThroughput:0,uploadThroughput:0});
  await call('Emulation.setDeviceMetricsOverride',{width:1440,height:1100,deviceScaleFactor:1,mobile:false});
  await call('Page.navigate',{url:pathToFileURL(report).href});
  let state;
  for(let i=0;i<120;i++) {
    state=await evaluate(`({ready:document.readyState,tables:document.querySelectorAll('table.case').length,images:[...document.images].filter(i=>i.complete&&i.naturalWidth>0).length})`);
    if(state.ready==='complete'&&state.tables===88&&state.images===314)break;
    await new Promise(r=>setTimeout(r,500));
  }
  if(state.tables!==88||state.images!==314)throw new Error(JSON.stringify(state));
  const screenshots=[];
  for(const [name,expression] of [
    ['overview','window.scrollTo(0,0)'],
    ['baseline',`document.querySelector('table.case').previousElementSibling.scrollIntoView()`],
    ['intervention',`document.querySelector('table.case[id^="l1_ch"]').previousElementSibling.scrollIntoView()`]
  ]) {
    await evaluate(expression);await new Promise(r=>setTimeout(r,300));
    const shot=await call('Page.captureScreenshot',{format:'png'});
    const path=`${root}/${name}.png`;await writeFile(path,Buffer.from(shot.data,'base64'));screenshots.push(path);
  }
  const audio=await evaluate(`(async()=>{let result=[];const all=[...document.querySelectorAll('audio')];for(let index=0;index<all.length;index++){let a=all[index];await a.play();await new Promise(r=>setTimeout(r,200));result.push({index,duration:a.duration,time:a.currentTime,error:a.error?.code??null});a.pause();}return result})()`);
  if(audio.some(a=>a.error||a.time<=0))throw new Error(JSON.stringify(audio));
  const records=await evaluate(`(()=>{const r=JSON.parse(document.getElementById('records').textContent);return {cases:r.baseline_cases.length,interventions:r.interventions.length}})()`);
  const result={network:'Offline via CDP before local file navigation',state,audio,records,screenshots};
  await writeFile(`${root}/browser_checks.json`,JSON.stringify(result,null,2));
  const checks=JSON.parse(await readFile(verification,'utf8'));
  checks.offline_browser_visual_check='Rendered in headless Edge offline; all 314 images loaded; overview and case screenshots saved. Human/model screenshot review recorded separately.';
  checks.audio_playback_browser_check={tested:audio.length,passed:true,details:audio};
  checks.browser_qa_screenshots=screenshots;
  await writeFile(verification,JSON.stringify(checks,null,2));
  console.log(JSON.stringify({network:result.network,state,playback_checks:audio.length,all_playback_passed:true,records,screenshots},null,2));
  await call('Browser.close');
} finally {socket?.close();proc.kill();}
