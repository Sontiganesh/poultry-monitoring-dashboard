const byId=(id)=>document.getElementById(id);
const video=byId("video"), select=byId("videoSelect"), loading=byId("videoLoading");
let videos=[], current=null, summary=null, frames=[], fps=30, loop=true, lastSnapshotTime=-1;
const fmt=(n)=>Number(n||0).toLocaleString();
const time=(seconds)=>{if(!Number.isFinite(seconds))return"00:00";const s=Math.max(0,Math.floor(seconds));return String(Math.floor(s/60)).padStart(2,"0")+":"+String(s%60).padStart(2,"0")};
function escapeHtml(s){return String(s).replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]))}
function getFrame(){
  if(!frames.length||!video.duration)return null;
  const index=Math.max(0,Math.min(frames.length-1,Math.floor(video.currentTime*fps)));
  return {index,...frames[index]};
}
function setMetric(id,value){byId(id).textContent=value===null||value===undefined?"—":fmt(value)}
function renderFrame(){
  const frame=getFrame();if(!frame)return;
  byId("frameLabel").textContent=fmt(frame.index);
  setMetric("frameVehicles",frame.count);setMetric("movingCount",frame.moving);
  setMetric("stationaryCount",frame.stationary);setMetric("trackedCount",frame.tracked);
  const classes=frame.classes||{},entries=Object.entries(classes).sort((a,b)=>b[1]-a[1]),max=Math.max(1,...entries.map(([,n])=>Number(n)||0));
  byId("classMix").innerHTML=entries.length?entries.map(([name,n])=>'<div class="class-row"><span class="class-name">'+escapeHtml(name.replaceAll("_"," "))+'</span><div class="class-track"><div class="class-fill" style="width:'+Math.max(3,100*n/max)+'%"></div></div><span class="class-value">'+fmt(n)+'</span></div>').join(""):'<div class="empty">No detections in this frame.</div>';
  const dirs=Object.entries(frame.directions||{});
  byId("directions").innerHTML=dirs.length?dirs.map(([dir,n])=>'<span class="direction-pill">'+escapeHtml(dir)+' · '+fmt(n)+'</span>').join(""):'<span class="muted">No confident directions in this frame</span>';
  byId("playbackTime").textContent=time(video.currentTime);
}
function renderSegments(data){
  const states=data.final_segment_states||{},values=Object.entries(states).map(([key,item])=>[key.split(":")[0],item]).sort((a,b)=>a[0].localeCompare(b[0]));
  byId("segments").innerHTML=values.length?values.map(([name,s])=>{
    const level=String(s.level||s.congestion_level||"unknown").toLowerCase(),cls=["green","yellow","red"].includes(level)?level:"unknown";
    const count=s.vehicle_count??s.current_vehicle_count??0,occ=s.occupancy_pct,speed=s.average_speed_kmh??s.speed_kmh;
    const detail=[fmt(count)+" vehicles",occ==null?null:Number(occ).toFixed(1)+"% occupancy",speed==null?null:Number(speed).toFixed(1)+" km/h"].filter(Boolean).join(" · ");
    return '<div class="segment"><i class="segment-dot '+cls+'"></i><div><div class="segment-name">'+escapeHtml(name.replaceAll("_"," "))+'</div><div class="segment-meta">'+(detail||"No segment reading")+'</div></div><span class="segment-state '+cls+'">'+escapeHtml(level)+'</span></div>';
  }).join(""):'<div class="empty">No saved segment analytics for this video.</div>';
}
function renderSummary(data,meta){
  const final=data.final_frame_analytics||{},full=data.full_inference_summary||{},tracked=data.tracking_quality||{};
  byId("framesProcessed").textContent=fmt(data.frames_processed??final.frames_processed??full.frames_processed);
  byId("observationCount").textContent=fmt(final.observation_count??full.detection_observations_total);
  byId("trackQuality").textContent=tracked.track_ids==null?"Not measured":fmt(tracked.track_ids)+" IDs";
  byId("trackQualityNote").textContent=tracked.median_track_seconds==null?"Tracking quality not included":"Median track "+Number(tracked.median_track_seconds).toFixed(1)+" sec";
  byId("speedBasis").textContent=data.speed_basis||"Not measured";
  byId("videoMeta").textContent=time(meta.duration_sec||video.duration)+" · "+fmt(meta.frames_processed??full.frames_processed)+" frames analyzed";
  renderSegments(data);
}
async function selectVideo(id){
  current=videos.find(v=>v.id===id);if(!current)return;
  summary=null;frames=[];fps=30;lastSnapshotTime=-1;loading.classList.remove("hidden");
  byId("videoTitle").textContent=current.title;byId("videoMeta").textContent="Loading saved analytics…";
  byId("classMix").innerHTML='<div class="empty">Loading frame analytics…</div>';
  video.pause();video.src=current.video_url;video.loop=loop;video.load();
  const response=await fetch("/api/videos/"+encodeURIComponent(id));
  if(!response.ok){byId("videoMeta").textContent="Could not load saved analysis.";return}
  summary=await response.json();const details=summary.full_inference_summary||{};
  fps=Number(details.output_video_fps||details.source_fps||summary.source_fps)||30;
  frames=Array.isArray(summary.playback_frames)?summary.playback_frames:[];
  fps=Number(summary.playback_fps)||fps;
  renderSummary(summary,current);
  byId("cameraLabel").textContent=new URLSearchParams(location.search).get("camera")||"CAM_02";
  if(video.readyState>=1){byId("durationTime").textContent=time(video.duration);renderFrame();summaryEvent();snapshotEvent(true)}
}
function webhookBase(){const endpoint=new URLSearchParams(location.search).get("webhook");try{const u=new URL(endpoint);return u.protocol==="https:"?u.href:""}catch{return""}}
function sendWebhook(payload){const endpoint=webhookBase();if(!endpoint)return;fetch(endpoint,{method:"POST",mode:"no-cors",headers:{"Content-Type":"text/plain;charset=UTF-8"},body:JSON.stringify(payload),keepalive:true}).catch(()=>{})}
function summaryEvent(){
  if(!summary||!current)return;const full=summary.full_inference_summary||{};
  const {playback_frames,playback_fps,...savedSummary}=summary;
  sendWebhook({...savedSummary,event:"drone_traffic_summary",camera_id:new URLSearchParams(location.search).get("camera")||"CAM_02",timestamp:new Date().toISOString(),timestamp_sec:Number((video.duration||full.duration_sec||0).toFixed(2)),source_video:full.video||summary.source||current.title});
}
function snapshotEvent(force=false){
  if(!summary||!current||video.readyState<1)return;const now=Date.now();if(!force&&now-lastSnapshotTime<30000)return;lastSnapshotTime=now;
  const frame=getFrame()||{};
  sendWebhook({event:"drone_traffic_playback_snapshot",camera_id:new URLSearchParams(location.search).get("camera")||"CAM_02",timestamp:new Date().toISOString(),source_video:(summary.full_inference_summary||{}).video||summary.source||current.title,playback_time_sec:Number(video.currentTime.toFixed(2)),frame_index:frame.index??null,total_count:frame.count??0,vehicle_count:frame.count??0,class_counts:frame.classes||{},tracked_count:frame.tracked??null,moving_count:frame.moving??null,stationary_count:frame.stationary??null,uncertain_count:frame.uncertain??null,direction_counts:frame.directions||{}});
}
video.addEventListener("loadedmetadata",()=>{loading.classList.add("hidden");byId("durationTime").textContent=time(video.duration);if(summary&&current)renderSummary(summary,current);renderFrame();summaryEvent();snapshotEvent(true);video.play().catch(()=>{})});
video.addEventListener("waiting",()=>loading.classList.remove("hidden"));video.addEventListener("playing",()=>loading.classList.add("hidden"));
video.addEventListener("timeupdate",()=>{renderFrame();snapshotEvent()});video.addEventListener("seeked",renderFrame);
video.addEventListener("error",()=>{loading.classList.remove("hidden");loading.innerHTML="<span>Could not load this video.</span>"});
select.addEventListener("change",()=>selectVideo(select.value));
byId("loopButton").addEventListener("click",()=>{loop=!loop;video.loop=loop;byId("loopButton").classList.toggle("active",loop)});
async function init(){
  try{
    const response=await fetch("/api/videos");if(!response.ok)throw new Error("Could not load videos");
    videos=(await response.json()).videos||[];
    if(!videos.length){select.innerHTML="<option>No analyzed videos available</option>";byId("videoTitle").textContent="No analyzed videos";return}
    const requested=new URLSearchParams(location.search).get("video"),preferred=requested&&videos.some(v=>v.id===requested)?requested:(videos.some(v=>v.id==="himachal_kullu_bypass")?"himachal_kullu_bypass":videos.some(v=>v.id==="test1")?"test1":videos[0].id);
    select.innerHTML=videos.map(v=>'<option value="'+escapeHtml(v.id)+'">'+escapeHtml(v.title)+'</option>').join("");select.value=preferred;await selectVideo(preferred);
  }catch(error){byId("videoTitle").textContent="Dashboard unavailable";byId("videoMeta").textContent=error.message;loading.innerHTML="<span>Could not load saved videos.</span>"}
}
init();
