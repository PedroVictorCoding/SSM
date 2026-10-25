'use strict';
const $ = id => document.getElementById(id);
let selected = new URLSearchParams(location.search).get('run'), snapshot = null, paused = false, layer = 0, sampleStep = null, busy = false;
const fmt = (x, digits=3) => Number.isFinite(x) ? x.toLocaleString(undefined,{maximumFractionDigits:digits,minimumFractionDigits:digits}) : '—';
const int = x => Number.isFinite(x) ? Math.round(x).toLocaleString() : '—';
const duration = s => Number.isFinite(s) ? (s >= 3600 ? `${Math.floor(s/3600)}h ${Math.floor(s%3600/60)}m` : `${Math.floor(s/60)}m ${Math.floor(s%60)}s`) : '—';
const last = (rows,key) => rows.filter(r=>Number.isFinite(r[key])).at(-1)?.[key];
function text(id,value){$(id).textContent=value;}
async function get(url){const r=await fetch(url);if(!r.ok)throw new Error(`API ${r.status}`);return r.json();}
function pair(container,label,value){const dt=document.createElement('dt'),dd=document.createElement('dd');dt.textContent=label;dd.textContent=value;container.append(dt,dd);}
function render(data){
  snapshot=data;const c=data.train_config,m=data.model_config,r=data.metrics,t=data.telemetry,s=data.status;
  text('run-title',data.id.replaceAll('_',' '));text('crumb',data.id);
  const step=Math.max(s.step||0,...r.map(x=>x.step)),total=c.steps||0,elapsed=Math.max(s.elapsed_secs||0,last(r,'elapsed_secs')||0);
  text('step-label',`STEP ${int(step)} / ${int(total)} · ${fmt(total?step/total*100:0,1)}%`);
  $('progress-bar').style.width=`${Math.min(100,total?step/total*100:0)}%`;
  text('phase',({training:'Training',evaluating:'Evaluating',finished:'Finished',failed:'Failed',untracked:'State unknown'})[s.phase]||'Unknown');
  text('model-label',`${m.n_layers||0} layers · d_model ${m.d_model||'—'} · ${t.backend||'backend unrecorded'}`);
  text('train-loss',fmt(last(r,'train_loss'),4));text('val-loss',fmt(last(r,'val_loss'),4));
  const losses=r.filter(x=>x.train_loss!==null), first=losses[0]?.train_loss,latest=last(r,'train_loss');
  text('loss-change',Number.isFinite(first)&&Number.isFinite(latest)?`${fmt((first-latest)/first*100,1)}% decrease from first log`:'Waiting for metrics');
  const vals=r.filter(x=>Number.isFinite(x.val_loss));text('val-best',vals.length?`Best ${fmt(Math.min(...vals.map(x=>x.val_loss)),4)} · ${fmt(last(r,'val_bpc'),3)} bits/char`:'At the next evaluation');
  const speed=last(r,'tokens_per_sec');text('throughput',int(speed));text('elapsed',duration(elapsed));
  text('eta',s.phase==='finished'?'Run finished':speed?`≈ ${duration(Math.max(0,total-step)*(c.batch_size||0)*(c.seq_len||0)/speed)} remaining · excludes evaluation`:'ETA unavailable');
  const age=s.report_age_secs;
  $('phase').setAttribute('data-phase',s.phase);
  $('notice').hidden=s.phase!=='failed'&&s.phase!=='untracked'&&!(s.phase!=='finished'&&age>120);
  text('notice',s.phase==='failed'?(data.failure?.message||'Training stopped after a numerical failure. Check the trainer output.'):s.phase==='untracked'?'This run has no lifecycle telemetry. CSV metrics update live; whether the trainer is running is unknown.':`Last trainer report ${duration(age)} ago. It may be computing a slow step, stopped, or disconnected.`);
  text('lr-value',Number.isFinite(last(r,'lr'))?last(r,'lr').toExponential(2):'—');
  $('schedule-details').replaceChildren();for(const [a,b] of [['Peak',c.lr],['Minimum',c.min_lr],['Warmup',c.warmup_steps]]){const div=document.createElement('div'),bEl=document.createElement('b');div.textContent=a;bEl.textContent=a==='Warmup'?`${int(b)} steps`:b?.toExponential(1)||'—';div.append(bEl);$('schedule-details').append(div);}
  text('params',t.total_params?`${int(t.total_params)} params · ${int(t.active_params_per_token)} active/token`:'Parameter counts available on new runs');
  renderNetwork(m);renderSamples(data.samples);$('config').replaceChildren();
  for(const [a,b] of [['Batch size',c.batch_size],['Sequence length',c.seq_len],['Model width',m.d_model],['SSM state',m.d_state],['Experts / top-k',`${m.moe_experts} / ${m.moe_top_k}`],['Vocabulary',m.vocab_size],['Optimizer','AdamW'],['Weight decay',c.weight_decay],['Gradient clip',c.grad_clip],['Random seed',c.seed]])pair($('config'),a,b??'—');
  drawCharts();text('updated',`Snapshot ${new Date().toLocaleTimeString()} · refresh every 2s`);
  if (typeof updateNeural === 'function') updateNeural(data);
}
function renderNetwork(m){
  const network=$('network');network.replaceChildren();
  const nodes=[{name:'Embedding',sub:`${m.vocab_size} → ${m.d_model}`,type:'edge',index:-1},...Array.from({length:m.n_layers||0},(_,i)=>({name:(m.attn_layers||[]).includes(i)?'Attention':'Mamba',sub:m.mlp_ratio>0?(m.moe_experts>0?`${m.moe_experts} experts · top ${m.moe_top_k}`:'SwiGLU MLP'):`state ${m.d_state}`,type:(m.attn_layers||[]).includes(i)?'attention':'',index:i})),{name:'LM head',sub:`${m.d_model} → ${m.vocab_size}`,type:'edge',index:m.n_layers}];
  nodes.forEach((n,i)=>{if(i){const w=document.createElement('div');w.className='wire';network.append(w);}const b=document.createElement('button');b.type='button';b.className=`node ${n.type} ${n.index===layer?'chosen':''}`;b.setAttribute('aria-pressed',String(n.index===layer));const label=document.createElement('span');label.className='node-number';label.textContent=n.index>=0&&n.index<m.n_layers?`BLOCK ${String(n.index).padStart(2,'0')}`:'';const strong=document.createElement('strong');strong.textContent=n.name;const small=document.createElement('small');small.textContent=n.sub;b.append(label,strong,small);b.onclick=()=>{layer=n.index;renderNetwork(m);};network.append(b);});
  const attention=(m.attn_layers||[]).includes(layer);
  text('network-detail',layer===-1?`Token embedding: ${m.vocab_size} character IDs mapped into ${m.d_model}-dimensional vectors.`:layer===m.n_layers?`Final RMSNorm → linear projection into ${m.vocab_size} next-character logits.`:`Block ${layer} · RMSNorm → ${attention?`causal attention (${m.n_heads} heads, ${m.attn_window||'full'} window)`:`Mamba (inner width ${m.d_model*m.expand}, state ${m.d_state}, conv ${m.d_conv}, Δ rank ${m.dt_rank||Math.ceil(m.d_model/16)})`} → residual${m.mlp_ratio>0?` → RMSNorm → ${m.moe_experts>0?`${m.moe_experts} SwiGLU experts (${m.moe_threshold>0?'dynamic threshold '+m.moe_threshold:'top-'+m.moe_top_k})`:'SwiGLU'} → residual`:''}.`);
}
function renderSamples(samples){const select=$('sample-select');select.replaceChildren();if(!samples.length){const o=new Option('No samples yet','');select.add(o);text('sample-text','Samples appear after validation on runs started with telemetry enabled.');return;}if(!samples.some(x=>x.step===sampleStep))sampleStep=samples.at(-1).step;for(const s of samples)select.add(new Option(`Step ${int(s.step)}`,s.step,false,s.step===sampleStep));text('sample-text',samples.find(s=>s.step===sampleStep)?.text||'');}
$('sample-select').onchange=()=>{sampleStep=Number($('sample-select').value);renderSamples(snapshot.samples);};
function chart(canvas,series){
  const dpr=window.devicePixelRatio||1,w=canvas.clientWidth,h=canvas.clientHeight;if(!w)return;
  canvas.width=w*dpr;canvas.height=h*dpr;const ctx=canvas.getContext('2d');ctx.scale(dpr,dpr);
  const all=series.flatMap(s=>s.points).filter(p=>Number.isFinite(p.y));const p={l:48,r:18,t:18,b:28},pw=w-p.l-p.r,ph=h-p.t-p.b;
  if(!all.length){ctx.fillStyle='#89929f';ctx.font='11px system-ui';ctx.textAlign='center';ctx.fillText('Waiting for recorded values',w/2,h/2);return;}
  let lo=Math.min(...all.map(x=>x.y)),hi=Math.max(...all.map(x=>x.y));let margin=(hi-lo)*.14||Math.max(Math.abs(hi)*.1,.001);lo=Math.max(0,lo-margin);hi+=margin;
  const maxX=Math.max(...all.map(x=>x.x),1),x=v=>p.l+v/maxX*pw,y=v=>p.t+(1-(v-lo)/(hi-lo))*ph;
  ctx.font='9px ui-monospace,monospace';ctx.lineWidth=1;
  for(let i=0;i<=4;i++){const yy=p.t+ph*i/4;ctx.strokeStyle='#232a34';ctx.beginPath();ctx.moveTo(p.l,yy);ctx.lineTo(w-p.r,yy);ctx.stroke();ctx.fillStyle='#7f8b9c';ctx.textAlign='right';const value=hi-(hi-lo)*i/4;ctx.fillText(hi<.01?value.toExponential(1):value.toFixed(2),p.l-9,yy+3);}
  for(let i=0;i<=4;i++){ctx.textAlign='center';ctx.fillText(int(maxX*i/4),x(maxX*i/4),h-8);}
  for(const s of series){const pts=s.points;if(!pts.length)continue;ctx.strokeStyle=s.color;ctx.lineWidth=2;ctx.beginPath();pts.forEach((v,i)=>i?ctx.lineTo(x(v.x),y(v.y)):ctx.moveTo(x(v.x),y(v.y)));ctx.stroke();if(s.fill&&pts.length>1){ctx.lineTo(x(pts.at(-1).x),h-p.b);ctx.lineTo(x(pts[0].x),h-p.b);ctx.closePath();const gradient=ctx.createLinearGradient(0,p.t,0,h-p.b);gradient.addColorStop(0,s.color+'25');gradient.addColorStop(1,s.color+'00');ctx.fillStyle=gradient;ctx.fill();}if(pts.length<30||!s.fill){ctx.fillStyle=s.color;for(const v of pts){ctx.beginPath();ctx.arc(x(v.x),y(v.y),2.7,0,Math.PI*2);ctx.fill();}}}
  canvas._chart={maxX,p,w};
}
function drawCharts(){if(!snapshot)return;const rows=snapshot.metrics,points=key=>rows.filter(r=>Number.isFinite(r[key])).map(r=>({x:r.step,y:r[key]}));let train=points('train_loss');if($('smooth').checked){let avg=train[0]?.y;train=train.map(p=>({...p,y:avg=.25*p.y+.75*avg}));}chart($('loss-chart'),[{points:train,color:'#77e4bc',fill:true},{points:points('val_loss'),color:'#b39bfa'}]);chart($('lr-chart'),[{points:points('lr'),color:'#b39bfa',fill:true}]);}
$('loss-chart').onmousemove=e=>{const c=$('loss-chart')._chart;if(!c||!snapshot.metrics.length)return;const x=(e.offsetX-c.p.l)/(c.w-c.p.l-c.p.r)*c.maxX;const row=snapshot.metrics.reduce((a,b)=>Math.abs(b.step-x)<Math.abs(a.step-x)?b:a);text('chart-tip',`Step ${int(row.step)}\nTraining ${fmt(row.train_loss,4)}\nValidation ${fmt(row.val_loss,4)}`);$('chart-tip').hidden=false;};
$('loss-chart').onmouseleave=()=>{$('chart-tip').hidden=true;};
$('smooth').onchange=drawCharts;window.addEventListener('resize',drawCharts);
$('pause').onclick=()=>{paused=!paused;text('pause',paused?'▶ Resume updates':'Ⅱ Pause updates');text('connection',paused?'Updates paused':'Connecting…');if(!paused)poll();};
$('download').onclick=()=>{if(!snapshot)return;const url=URL.createObjectURL(new Blob([JSON.stringify(snapshot,null,2)],{type:'application/json'})),a=document.createElement('a');a.href=url;a.download=`${snapshot.id}-snapshot.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
async function poll(){if(paused||busy)return;busy=true;try{const data=await get('/api/runs');text('run-count',data.runs.length);if(!data.runs.some(r=>r.id===selected))selected=data.runs[0]?.id;const runs=$('runs');runs.replaceChildren();for(const r of data.runs){const b=document.createElement('button');b.type='button';b.className=`run-button ${r.id===selected?'selected':''}`;b.textContent=r.id;b.onclick=()=>{selected=r.id;layer=0;sampleStep=null;const u=new URL(location);u.searchParams.set('run',selected);history.replaceState(null,'',u);poll();};runs.append(b);}if(selected){const id=selected,result=await get(`/api/runs/${encodeURIComponent(id)}`);if(selected===id)render(result);}else{text('run-title','Waiting for an experiment');text('notice','Start a training run, or point the server at your runs directory with --runs.');$('notice').hidden=false;}text('connection','● API connected');}catch(e){text('connection','● Connection lost · retrying');text('notice',`Could not refresh the dashboard (${e.message}). Reconnecting automatically.`);$('notice').hidden=false;}finally{busy=false;}}
poll();setInterval(poll,2000);
let neural=null, previousNeural=null, neuralVersion=null, neuralRun=null, tensorName=null, neuralPage=0, neuralFetching=false;
async function updateNeural(data){
  if(neuralRun!==data.id){neuralRun=data.id;neural=null;previousNeural=null;neuralVersion=null;tensorName=null;neuralPage=0;window.liveGraph.update(null,null);renderTensorOptions();renderHeatmap();}
  if(!data.neural_version){text('neural-status','No neural telemetry for this run. Launch the updated trainer with --viz-every 1 for capture every step.');return;}
  if(neuralVersion===data.neural_version||neuralFetching)return;
  const run=data.id,version=data.neural_version;neuralFetching=true;
  try{const next=await get(`/api/runs/${encodeURIComponent(run)}/neural`);if(selected!==run)return;previousNeural=neural;neural=next;neuralVersion=version;
    text('neural-status',`${int(next.tensors.length)} parameter tensors · ${int(next.tensors.reduce((n,t)=>n+t.count,0))} trainable scalars · ${next.max_values===0?'all weights captured':'sampled weights'}`);
    text('neural-step',`GRADIENT STEP ${int(next.step)} · WEIGHTS AFTER STEP ${int(next.weights_step)}`);renderTensorOptions();renderHeatmap();window.liveGraph.update(neural,previousNeural);
  }catch(e){text('neural-status',`Neural capture unavailable (${e.message}); retrying.`);}finally{neuralFetching=false;}
}
function neuralItems(){return $('neural-mode').value==='activations'?(neural?.activations||[]):(neural?.tensors||[]);}
function renderTensorOptions(){const list=neuralItems(),q=$('tensor-search').value.toLowerCase(),filtered=list.filter(t=>t.name.toLowerCase().includes(q));if(!filtered.some(t=>t.name===tensorName)){tensorName=filtered[0]?.name;neuralPage=0;}$('tensor-select').replaceChildren();for(const t of filtered)$('tensor-select').add(new Option(`${t.name}${t.shape?'  ['+t.shape.join(' × ')+']':''}`,t.name,false,t.name===tensorName));}
function tensorValues(){const t=neuralItems().find(t=>t.name===tensorName);if(!t)return {tensor:null,values:[]};if($('neural-mode').value==='changes'){const prev=previousNeural?.tensors.find(x=>x.name===t.name);if(!prev||prev.stride!==t.stride||prev.values.length!==t.values.length)return {tensor:t,values:[]};return {tensor:t,values:t.values.map((v,i)=>v===null||prev.values[i]===null?null:v-prev.values[i])};}return {tensor:t,values:t.values};}
function renderHeatmap(){
  const canvas=$('heatmap'),w=canvas.clientWidth,h=canvas.clientHeight,dpr=devicePixelRatio||1;canvas.width=w*dpr;canvas.height=h*dpr;const ctx=canvas.getContext('2d');ctx.scale(dpr,dpr);ctx.clearRect(0,0,w,h);
  const {tensor:t,values}=tensorValues(),pageSize=1024,mode=$('neural-mode').value;neuralPage=Math.min(neuralPage,Math.max(0,Math.ceil(values.length/pageSize)-1));
  $('prev-page').disabled=neuralPage===0;$('next-page').disabled=(neuralPage+1)*pageSize>=values.length;$('tensor-stats').replaceChildren();
  if(!t||!values.length){ctx.fillStyle='#89929f';ctx.font='12px system-ui';ctx.textAlign='center';ctx.fillText(mode==='changes'?'Waiting for two captures of this tensor':'Waiting for neural values',w/2,h/2);text('tensor-name',t?.name||'No tensor selected');text('page-info','—');canvas._cells=null;return;}
  text('tensor-name',t.name);const offset=neuralPage*pageSize,visible=values.slice(offset,offset+pageSize),finite=values.filter(Number.isFinite),scale=Math.max(...(finite.length<50000?finite.map(Math.abs):[finite.reduce((m,v)=>Math.max(m,Math.abs(v)),0)]),1e-12);
  const cols=Math.min(32,Math.max(1,t.shape?.at(-1)||Math.ceil(Math.sqrt(Math.min(pageSize,values.length))))),rows=Math.ceil(visible.length/cols),cw=w/cols,ch=h/rows;
  visible.forEach((v,i)=>{const a=Number.isFinite(v)?Math.min(1,Math.abs(v)/scale):0,base=[22,29,39],target=v>=0?[119,228,188]:[179,155,250];ctx.fillStyle=v===null?'#e48f77':`rgb(${base.map((b,j)=>Math.round(b+(target[j]-b)*Math.sqrt(a))).join(',')})`;ctx.fillRect((i%cols)*cw,Math.floor(i/cols)*ch,Math.max(1,cw-1),Math.max(1,ch-1));});
  canvas._cells={tensor:t,values,cols,cw,ch,offset,visibleCount:visible.length};
  text('page-info',`${int(offset+1)}–${int(Math.min(values.length,offset+pageSize))} / ${int(values.length)} values`);
  if(t.shape){pair($('tensor-stats'),'Shape',t.shape.join(' × '));pair($('tensor-stats'),'Parameter count',int(t.count));pair($('tensor-stats'),'Capture stride',t.stride===1?'Every scalar':`Every ${t.stride}th scalar`);pair($('tensor-stats'),'Weight mean',fmt(t.stats.mean,6));pair($('tensor-stats'),'Weight RMS',fmt(t.stats.rms,6));pair($('tensor-stats'),'Weight range',`${fmt(t.stats.min,4)} … ${fmt(t.stats.max,4)}`);pair($('tensor-stats'),'Gradient RMS',t.gradient?Number(t.gradient.rms).toExponential(3):'Not captured');pair($('tensor-stats'),'Non-finite weights',int(t.stats.non_finite));}
  else{pair($('tensor-stats'),'Channels',int(t.values.length));pair($('tensor-stats'),'Sequence in batch',t.sequence_index);pair($('tensor-stats'),'Token position',t.token_position);}
  text('capture-note',mode==='activations'?'Actual output channel values for the last token of the first sequence in the current random training batch. Changes reflect both inputs and learned weights. Internal attention maps and SSM hidden trajectories are not captured.':mode==='changes'?`Signed weight difference between captures at steps ${previousNeural?.step} and ${neural.step}. All weight pages are available; the color scale is normalized to this tensor.`:'Weights before the optimizer update, with current backward-pass gradient RMS before clipping. Browse pages or hover cells for exact scalar values. Colors are normalized per tensor.');
}
$('heatmap').onmousemove=e=>{const cells=$('heatmap')._cells;if(!cells)return;const local=Math.floor(e.offsetY/cells.ch)*cells.cols+Math.floor(e.offsetX/cells.cw);if(local>=cells.visibleCount)return;const i=cells.offset+local,flat=i*(cells.tensor.stride||1),shape=cells.tensor.shape||[cells.values.length];let rem=flat;const indices=Array(shape.length);for(let j=shape.length-1;j>=0;j--){indices[j]=rem%shape[j];rem=Math.floor(rem/shape[j]);}text('cell-info',`[${indices.join(', ')}] · flat ${int(flat)} · ${cells.values[i]===null?'non-finite':Number(cells.values[i]).toPrecision(8)}`);};
$('tensor-select').onchange=()=>{tensorName=$('tensor-select').value;neuralPage=0;renderHeatmap();};$('tensor-search').oninput=()=>{renderTensorOptions();renderHeatmap();};$('neural-mode').onchange=()=>{tensorName=null;neuralPage=0;renderTensorOptions();renderHeatmap();};$('prev-page').onclick=()=>{neuralPage--;renderHeatmap();};$('next-page').onclick=()=>{neuralPage++;renderHeatmap();};window.addEventListener('resize',renderHeatmap);
