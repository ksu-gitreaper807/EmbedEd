'use strict';
const $=id=>document.getElementById(id);
const fields=['p1a','p1b','p2a','p2b'].map($);
const colours=['#64748b','#0e7490','#b45309','#6d28d9','#15803d','#b91c1c','#c026d3'];
let state=null, busy=false;
function node(tag,text,cls){const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(cls)n.className=cls;return n;}
function notice(text,error=false){$('notice').textContent=text;$('notice').className=error?'error':'success';}
function paint(rows=null){
  $('results').replaceChildren();
  (state?.catalog||[]).forEach((entry,i)=>{
    const row=node('tr'), label=node('td'), model=node('div',undefined,'model'), badge=node('span',entry.condition,'badge');badge.style.color=colours[i];
    const meta=node('div');meta.append(node('div',entry.name,'model-name'),node('div',`Seed 13 · τ ${entry.threshold===undefined?'unavailable':String(entry.threshold)}`,'model-meta'));
    model.append(badge,meta);label.append(model);row.append(label);
    for(let p=0;p<2;p++){
      const cell=node('td'), result=rows?.[i];
      if(!result){cell.append(node('span','—   Awaiting inference','pending'));}
      else{
        const line=node('div',undefined,'score-line'),score=node('span',result.scores[p].toFixed(9),'score');score.title=String(result.scores[p]);
        line.append(score,node('span',result.decisions[p]?'Clone':'Not clone',`decision ${result.decisions[p]?'clone':'non-clone'}`));
        const track=node('div',undefined,'track'),fill=node('span',undefined,'fill'),mark=node('span',undefined,'mark');
        fill.style.width=`${Math.max(0,Math.min(1,result.scores[p]))*100}%`;fill.style.background=colours[i];
        mark.style.left=`${Math.max(0,Math.min(1,result.threshold))*100}%`;mark.title=`Frozen threshold: ${result.threshold}`;track.append(fill,mark);cell.append(line,track);
        if(result.saturated)cell.append(node('div','Near-one score/threshold: inspect precision','warning'));
      }
      row.append(cell);
    }
    $('results').append(row);
  });
}
function reset(){if(!state)return;state.examples.forEach((p,i)=>{fields[2*i].value=p.a;fields[2*i+1].value=p.b;});paint();$('performance').textContent='Awaiting a comparison';notice('Examples restored. They are illustrative and have no verified ground-truth labels.');}
fields.forEach(f=>f.addEventListener('input',()=>{paint();$('performance').textContent='Inputs changed · compare again';notice('');}));
$('reset').addEventListener('click',reset);
$('compare').addEventListener('click',async()=>{
  if(busy)return;
  if(fields.some(f=>!f.value.trim()))return notice('Fill all four snippets before comparing.',true);
  busy=true;$('compare').disabled=true;$('reset').disabled=true;fields.forEach(f=>f.disabled=true);$('compare').textContent='Comparing C0–C6…';paint();$('performance').textContent='Inference in progress';notice('Comparing sequentially. No Java code is being executed.');
  const controller=new AbortController(),timer=setTimeout(()=>controller.abort(),180000);
  try{
    const response=await fetch('api/compare',{method:'POST',headers:{'Content-Type':'application/json','X-EmbedEd-Request':'1'},body:JSON.stringify({pairs:[{a:fields[0].value,b:fields[1].value},{a:fields[2].value,b:fields[3].value}]}),signal:controller.signal});
    const data=await response.json();if(!response.ok)throw new Error(data.error||'Runtime request failed.');
    if(!Array.isArray(data.results)||data.results.length!==7||data.results.some((r,i)=>r.condition!==state.catalog[i].condition||!Array.isArray(r.scores)||r.scores.length!==2||r.scores.some(x=>!Number.isFinite(x))||r.decisions.length!==2))throw new Error('Incomplete runtime response. No predictions shown.');
    paint(data.results);
    $('performance').textContent=`${(data.latency_ms/1000).toFixed(1)} s · ${data.device}${data.peak_cuda_mb===null?'':` · peak GPU ${data.peak_cuda_mb} MB`}`;
    const truncated=data.truncated.map((x,i)=>x?`pair ${Math.floor(i/2)+1} snippet ${i%2?'B':'A'}`:null).filter(Boolean);
    notice(`14 predictions complete · ${data.precision}. ${truncated.length?`Truncated to 512 tokens: ${truncated.join(', ')}.`:'No snippet was truncated.'} Results are illustrative, not benchmark evidence.`);
    $('provenance').textContent=JSON.stringify({revision:data.revision,base_revision:data.base_revision,seed:13,max_tokens:data.max_tokens,precision:data.precision,token_counts:data.token_counts,models:data.results.map(({condition,threshold,threshold_sha256,checkpoint_sha256,fingerprint})=>({condition,threshold,threshold_sha256,checkpoint_sha256,fingerprint}))},null,2);
  }catch(e){paint();$('performance').textContent='No predictions generated';notice(e.name==='AbortError'?'The request timed out. The runtime may still be working; check Colab before retrying.':e.message,true);}
  finally{clearTimeout(timer);busy=false;$('compare').disabled=false;$('reset').disabled=false;fields.forEach(f=>f.disabled=false);$('compare').textContent='Compare all seven ↗';}
});
(async()=>{try{const response=await fetch('api/status');if(!response.ok)throw new Error('Runtime status unavailable.');state=await response.json();reset();$('runtime').textContent=state.ready?'7 verified weight sets · sequential':'Preview · weights unavailable';$('runtime').className=`runtime ${state.ready?'ready':'offline'}`;if(!state.ready)notice('This is the local UI preview. Run notebooks/Phase6_Colab.ipynb for real model predictions.',true);$('provenance').textContent=JSON.stringify({revision:state.revision,base_revision:state.base_revision,seed:13,models:state.catalog},null,2);}catch(e){$('runtime').textContent='Runtime unavailable';notice('Could not connect to the inference runtime. Start or reconnect the Phase 6 Colab notebook.',true);}})();
