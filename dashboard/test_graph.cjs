const {test} = require('node:test');
const assert = require('node:assert/strict');
const {ConnectionData: data, ConnectionGraph} = require('./static/graph.js');

function tensor(shape, stride=1) {
  const count=shape.reduce((a,b)=>a*b,1);
  return {name:'blocks.0.mamba.in_proj.weight',shape,count,stride,
    values:Array.from({length:Math.ceil(count/stride)},(_,i)=>i*stride/100-1)};
}

test('every weight has an exact coordinate and appears once across all pages',()=>{
  for(const shape of [[19,35],[3,7,19],[37]]){
    const t=tensor(shape),layout=data.layout(t),seen=new Set();
    for(let row=0;row<layout.rows;row+=16)for(let col=0;col<layout.columns;col+=16){
      for(const cell of data.window(t,row,col).cells){
        assert(!seen.has(cell.flat));seen.add(cell.flat);
        assert.equal(cell.value,t.values[cell.flat]);
        const indices=data.coordinates(t,cell.flat);
        assert.equal(indices.reduce((flat,index,axis)=>flat*shape[axis]+index,0),cell.flat);
      }
    }
    assert.equal(seen.size,t.count);
  }
});

test('sampling, missing gradients, and nonfinite values are never invented',()=>{
  const t=tensor([5,7],3);
  assert.equal(data.value(t,1),undefined);
  assert.equal(data.value(t,3),t.values[1]);
  assert.equal(data.value(t,-1),undefined);
  assert.equal(data.value(t,t.count),undefined);
  assert.equal(data.delta(t,undefined,3),undefined);
  const old={...t,values:t.values.map(v=>v-.25)};
  assert(Math.abs(data.delta(t,old,3)-.25)<1e-12);
  t.values[1]=null;
  assert.equal(data.value(t,3),null);
  assert.equal(data.delta(t,old,3),undefined);
  assert.equal(data.window(t,-20,9999).startRow,0);
  assert.equal(data.window(t,-20,9999).startColumn,6);
});

test('graph renders real edges, pages channels, retains selection, and resets runs',()=>{
  const elements=new Map();
  const ctx=new Proxy({}, {get:(target,key)=>target[key]||(()=>{})});
  class Element {
    constructor(id){this.value=id==='graph-mode'?'weights':id==='graph-search'?'':'0';this.children=[];this.clientWidth=800;this.clientHeight=520;}
    replaceChildren(){this.children=[];}
    add(value){this.children.push(value);}
    getContext(){return ctx;}
  }
  global.document={getElementById:id=>{if(!elements.has(id))elements.set(id,new Element(id));return elements.get(id);}};
  global.window={devicePixelRatio:1,addEventListener(){},matchMedia:()=>({matches:true})};
  global.Option=function(text,value,_,selected){return {text,value,selected};};
  const graph=new ConnectionGraph(),t=tensor([19,35]);
  const previous={step:1,weights_step:0,tensors:[t]};
  graph.update(previous,null);
  assert.equal(graph.hitTargets.filter(t=>t.kind==='edge').length,256);
  graph.pinned={kind:'edge',flat:2};
  const next={step:2,weights_step:1,tensors:[{...t,values:t.values.map(v=>v+.125)}]};
  graph.update(next,previous);
  assert.match(elements.get('graph-info').textContent,/Δ 1.25000e-1/);
  assert.match(elements.get('graph-info').textContent,/selected/);
  graph.page(1);
  assert.equal(elements.get('graph-column').value,'16');
  assert.equal(graph.pinned,null);
  graph.page(1);graph.page(1);
  assert.equal(elements.get('graph-row').value,'16');
  assert.equal(elements.get('graph-column').value,'0');
  elements.get('graph-mode').value='changes';graph.draw();
  assert(graph.hitTargets.some(t=>t.kind==='edge'));
  graph.update({step:0,weights_step:0,tensors:[tensor([37])]},null);
  assert.equal(graph.hitTargets.filter(t=>t.kind==='edge').length,0);
  assert.equal(graph.hitTargets.filter(t=>t.kind==='scalar').length,16);
  graph.update(null,null);
  assert.equal(graph.hitTargets.length,0);
  assert.match(elements.get('graph-status').textContent,/No neural capture/);
});
