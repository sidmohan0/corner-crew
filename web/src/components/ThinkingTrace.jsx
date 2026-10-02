// Adapted from Beautiful UI ThinkingState (MIT). See vendor/BEAUTIFUL-UI-LICENSE.
// Replaces its timed demo sequence with real backend lifecycle events.
import { useState } from 'react';
import { ChevronDown, Check, Sparkles, CircleAlert } from 'lucide-react';
export default function ThinkingTrace({events, running, failed}) {
  const [expanded,setExpanded]=useState(true);
  return <div className="thinking-trace"><button className="trace-toggle" onClick={()=>setExpanded(!expanded)} aria-expanded={expanded}>
    <Sparkles size={15}/><span className={running?'shimmer':''}>{running?'Working through your request':failed?'Run stopped':`Processed in ${((events.at(-1)?.elapsed_ms||0)/1000).toFixed(1)}s`}</span><ChevronDown size={14} style={{transform:expanded?'rotate(180deg)':undefined}}/>
  </button><div className="trace-collapse" style={{gridTemplateRows:expanded?'1fr':'0fr'}}><div><div className="trace-list">
  {events.filter(e=>['classify','privacy_policy','model_selection','execute','evaluate','verify_independent','tool_check','escalate','consensus','budget_exceeded'].includes(e.stage)).map((e,i,rows)=><div className="trace-mini" key={i}>
    {e.stage==='budget_exceeded'||e.passed===false||(failed&&i===rows.length-1)?<CircleAlert size={13}/>:running&&['classify','execute','evaluate','verify_independent','escalate'].includes(e.stage)&&!events.some(done=>done.stage===e.stage+'_complete'&&done.elapsed_ms>=e.elapsed_ms)?<span className="spinner"/>:<Check size={13}/>}
    <span>{({classify:'Classifying the request',privacy_policy:'Applying privacy policy',model_selection:'Choosing a model',execute:'Generating an answer',evaluate:'Evaluating the answer',verify_independent:'Independent review',tool_check:'Checking with tools',escalate:'Trying a stronger model',consensus:'Comparing independent answers',budget_exceeded:'Budget limit reached'})[e.stage]}</span>
    <small>{e.model||e.effective_privacy||''}</small>
  </div>)}</div></div></div></div>
}
