'use strict';

// Parameter axes are displayed directly. A matrix isn't assumed to be a dense
// neuron layer: embeddings, convolution taps and SSM state parameters differ.
const ConnectionData = {
  layout(tensor) {
    const shape = tensor.shape || [tensor.count];
    return shape.length === 1
      ? {rows: shape[0], columns: 1, vector: true}
      : {rows: shape.slice(0, -1).reduce((a, b) => a * b, 1), columns: shape.at(-1), vector: false};
  },
  value(tensor, flat) {
    if (!tensor || flat < 0 || flat >= tensor.count) return undefined;
    const stride = tensor.stride || 1;
    return flat % stride === 0 ? tensor.values[flat / stride] : undefined;
  },
  coordinates(tensor, flat) {
    const indices = [];
    for (const size of [...tensor.shape].reverse()) {
      indices.unshift(flat % size);
      flat = Math.floor(flat / size);
    }
    return indices;
  },
  window(tensor, row, column, size = 16) {
    const layout = this.layout(tensor);
    const startRow = Math.min(Math.max(0, Math.floor(row) || 0), Math.max(0, layout.rows - 1));
    const startColumn = layout.vector ? 0 : Math.min(Math.max(0, Math.floor(column) || 0), Math.max(0, layout.columns - 1));
    const cells = [];
    for (let r = startRow; r < Math.min(startRow + size, layout.rows); r++) {
      for (let c = startColumn; c < Math.min(startColumn + size, layout.columns); c++) {
        const flat = r * layout.columns + c;
        cells.push({row: r, column: c, flat, value: this.value(tensor, flat)});
      }
    }
    return {...layout, startRow, startColumn, cells};
  },
  delta(current, previous, flat) {
    const a = this.value(current, flat), b = this.value(previous, flat);
    return Number.isFinite(a) && Number.isFinite(b) ? a - b : undefined;
  },
};

class ConnectionGraph {
  constructor() {
    this.current = null;
    this.previous = null;
    this.name = null;
    this.pinned = null;
    this.hitTargets = [];
    this.frame = null;
    const el = id => document.getElementById(id);
    this.el = el;
    this.canvas = el('connection-canvas');
    el('graph-search').oninput = () => { this.options(); this.draw(); };
    el('graph-tensor').onchange = () => {
      this.name = el('graph-tensor').value; this.resetRange(); this.draw();
    };
    el('graph-mode').onchange = () => this.draw();
    for (const id of ['graph-row', 'graph-column']) el(id).onchange = () => {this.pinned = null; this.draw();};
    el('graph-prev').onclick = () => this.page(-1);
    el('graph-next').onclick = () => this.page(1);
    el('graph-clear').onclick = () => { this.pinned = null; this.draw(); };
    this.canvas.onmousemove = event => {if (!this.pinned) this.inspect(this.hit(event));};
    this.canvas.onmouseleave = () => { if (!this.pinned) this.inspect(null); };
    this.canvas.onclick = event => {this.pinned = this.hit(event); this.draw();};
    this.canvas.onkeydown = event => {if (event.key === 'Escape') {this.pinned = null; this.draw();}};
    window.addEventListener('resize', () => this.draw());
  }

  update(current, previous) {
    if (this.frame !== null) cancelAnimationFrame(this.frame);
    this.frame = null;
    this.current = current; this.previous = previous;
    if (!current) {this.name = null; this.resetRange();}
    this.options();
    this.changedAt = performance.now();
    this.draw();
    // A short fade only on measured updates, never synthetic firing activity.
    if (current && previous && !window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      const fade = () => {
        this.draw();
        if (performance.now() - this.changedAt < 800) this.frame = requestAnimationFrame(fade);
        else this.frame = null;
      };
      this.frame = requestAnimationFrame(fade);
    }
  }

  resetRange() {
    this.el('graph-row').value = '0'; this.el('graph-column').value = '0'; this.pinned = null;
  }

  options() {
    const search = this.el('graph-search').value.toLowerCase();
    const tensors = (this.current?.tensors || []).filter(t => t.name.toLowerCase().includes(search));
    if (!tensors.some(t => t.name === this.name)) { this.name = tensors[0]?.name; this.resetRange(); }
    const select = this.el('graph-tensor'); select.replaceChildren();
    for (const t of tensors) select.add(new Option(`${t.name} [${t.shape.join(' × ')}]`, t.name, false, t.name === this.name));
  }

  tensor() { return this.current?.tensors.find(t => t.name === this.name); }
  oldTensor() { return this.previous?.tensors.find(t => t.name === this.name && t.shape.join(',') === this.tensor()?.shape.join(',')); }

  page(direction) {
    const t = this.tensor(); if (!t) return;
    const view = ConnectionData.window(t, Number(this.el('graph-row').value), Number(this.el('graph-column').value));
    const columnPages = Math.ceil(view.columns / 16), rowPages = Math.ceil(view.rows / 16);
    let page = Math.floor(view.startRow / 16) * columnPages + Math.floor(view.startColumn / 16);
    page = Math.max(0, Math.min(rowPages * columnPages - 1, page + direction));
    this.el('graph-row').value = String(Math.floor(page / columnPages) * 16);
    this.el('graph-column').value = String(page % columnPages * 16);
    this.pinned = null; this.draw();
  }

  draw() {
    const canvas = this.canvas, w = canvas.clientWidth, h = canvas.clientHeight;
    if (!w || !h) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = w * dpr; canvas.height = h * dpr;
    const ctx = canvas.getContext('2d'); ctx.scale(dpr, dpr);
    this.hitTargets = [];
    const t = this.tensor();
    if (!t) {
      ctx.fillStyle = '#89929f'; ctx.font = '12px system-ui'; ctx.textAlign = 'center';
      ctx.fillText(this.current ? 'No tensors match your search' : 'Start a training run with --viz-every 1 to see live connections', w/2, h/2);
      this.el('graph-status').textContent = this.current ? 'No matching tensors.' : 'No neural capture for this run.';
      this.el('graph-step').textContent = '—'; this.el('graph-range').textContent = 'No weights available';
      this.el('graph-prev').disabled = true; this.el('graph-next').disabled = true;
      this.el('graph-column').disabled = true; this.inspect(null); return;
    }
    const view = ConnectionData.window(t, Number(this.el('graph-row').value), Number(this.el('graph-column').value));
    this.el('graph-row').value = String(view.startRow); this.el('graph-column').value = String(view.startColumn);
    this.el('graph-row').max = view.rows - 1; this.el('graph-column').max = view.columns - 1;
    this.el('graph-column').disabled = view.vector;
    const columnPages = Math.ceil(view.columns / 16), page = Math.floor(view.startRow/16)*columnPages+Math.floor(view.startColumn/16);
    this.el('graph-prev').disabled = page === 0;
    this.el('graph-next').disabled = page >= Math.ceil(view.rows/16)*columnPages-1;
    const sampled = (t.stride || 1) > 1;
    this.el('graph-status').textContent = `${t.name} · ${t.shape.join(' × ')} · ${sampled ? `sampled: every ${t.stride}th scalar; missing edges omitted` : 'every weight available across ranges'}`;
    this.el('graph-step').textContent = `WEIGHTS AFTER STEP ${this.current.weights_step} · CAPTURE ${this.current.step}`;
    const captured = view.cells.filter(c => Number.isFinite(c.value)).length;
    this.el('graph-range').textContent = `Rows ${view.startRow}–${Math.min(view.startRow+15,view.rows-1)} / ${view.rows}${view.vector?'':` · columns ${view.startColumn}–${Math.min(view.startColumn+15,view.columns-1)} / ${view.columns}`} · ${captured}/${view.cells.length} visible weights`;
    const previous = this.oldTensor(), changeMode = this.el('graph-mode').value === 'changes';
    const metric = c => changeMode ? ConnectionData.delta(t, previous, c.flat) : c.value;
    const scale = Math.max(1e-12, ...view.cells.map(c => Math.abs(metric(c) || 0)));
    const color = value => value === null ? '#e48f77' : value >= 0 ? '#77e4bc' : '#b39bfa';
    const y = (index, count) => 60 + (index + .5) * (h - 110) / count;
    const rows = Math.min(16, view.rows-view.startRow), columns = Math.min(16, view.columns-view.startColumn);
    const x1 = Math.min(110, w*.23), x2 = w-x1;
    ctx.font = '10px ui-monospace,monospace'; ctx.fillStyle = '#89929f';
    ctx.textAlign = 'center'; ctx.fillText(view.vector ? 'PARAMETER ELEMENTS' : 'ROW CHANNELS', view.vector?w/2:x1, 28);
    if (!view.vector) ctx.fillText('COLUMN CHANNELS', x2, 28);
    if (changeMode && !previous) ctx.fillText('Waiting for a second capture to measure changes', w/2, h-12);
    for (const c of view.cells) {
      const value = metric(c);
      const ay = y(c.row-view.startRow, rows), by = y(c.column-view.startColumn, columns);
      if (!view.vector && value !== undefined) {
        const focused = !this.pinned || (this.pinned.kind==='edge' && this.pinned.flat===c.flat) || (this.pinned.kind==='row' && this.pinned.index===c.row) || (this.pinned.kind==='column' && this.pinned.index===c.column);
        ctx.globalAlpha = focused ? .15 + .75*Math.sqrt(Math.min(1,Math.abs(value || 0)/scale)) : .035;
        ctx.strokeStyle = color(value); ctx.lineWidth = focused ? .5 + 2*Math.abs(value || 0)/scale : .5;
        ctx.beginPath(); ctx.moveTo(x1,ay); ctx.lineTo(x2,by); ctx.stroke();
        const delta = ConnectionData.delta(t,previous,c.flat), flash = Math.max(0,1-(performance.now()-this.changedAt)/800);
        if (focused && Number.isFinite(delta) && delta !== 0 && flash>0) {
          ctx.globalAlpha = flash*.5; ctx.strokeStyle='#ecfff6'; ctx.lineWidth=1;
          ctx.beginPath();ctx.moveTo(x1,ay);ctx.lineTo(x2,by);ctx.stroke();
        }
        this.hitTargets.push({kind:'edge',flat:c.flat,x1,y1:ay,x2,y2:by});
      }
    }
    ctx.globalAlpha = 1;
    const node = (kind,index,x,yy,value) => {
      ctx.fillStyle = value === undefined ? '#15242a' : color(value);
      ctx.strokeStyle = '#8499a8';ctx.lineWidth=1;ctx.beginPath();ctx.arc(x,yy,5,0,Math.PI*2);ctx.fill();ctx.stroke();
      ctx.fillStyle='#acb7c3';ctx.textAlign=kind==='column'?'left':'right';ctx.fillText(String(index),x+(kind==='column'?14:-14),yy+3);
      this.hitTargets.push({kind,index,x,y:yy});
    };
    for(let r=0;r<rows;r++) node(view.vector?'scalar':'row',view.startRow+r,view.vector?w/2:x1,y(r,rows),view.vector?metric(view.cells[r]):undefined);
    if(!view.vector)for(let c=0;c<columns;c++)node('column',view.startColumn+c,x2,y(c,columns));
    this.inspect(this.pinned);
  }

  hit(event) {
    const px=event.offsetX,py=event.offsetY;
    for(const target of [...this.hitTargets].reverse()){
      if(target.kind!=='edge'){if(Math.hypot(px-target.x,py-target.y)<12)return target;}
      else{const dx=target.x2-target.x1,dy=target.y2-target.y1,u=Math.max(0,Math.min(1,((px-target.x1)*dx+(py-target.y1)*dy)/(dx*dx+dy*dy)));
        if(Math.hypot(px-(target.x1+u*dx),py-(target.y1+u*dy))<3)return target;}
    }
    return null;
  }

  inspect(target) {
    const t=this.tensor(), output=this.el('graph-info');
    if(!target||!t){output.textContent='Hover a node or connection. Click to keep it selected as training updates. Escape clears selection.';return;}
    if(target.kind==='row'||target.kind==='column'){
      output.textContent=`${t.name} · ${target.kind} channel ${target.index} · click to isolate its visible connections`;return;
    }
    const flat=target.kind==='edge'?target.flat:target.index,value=ConnectionData.value(t,flat),delta=ConnectionData.delta(t,this.oldTensor(),flat);
    const valueText=value===undefined?'not captured':value===null?'non-finite':value.toPrecision(8);
    output.textContent=`${t.name}[${ConnectionData.coordinates(t,flat).join(', ')}] · flat ${flat} · weight ${valueText} · Δ ${Number.isFinite(delta)?delta.toExponential(5):'unavailable'}${this.pinned?' · selected':''}`;
  }
}

if (typeof module !== 'undefined' && module.exports) module.exports = {ConnectionData, ConnectionGraph};
if (typeof window !== 'undefined') window.liveGraph = new ConnectionGraph();
