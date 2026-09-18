let currentData=null,currentSurveyId=null,currentFile=null,intelMap=null,resultMap=null,charts={},scanAnimTimer=null,currentHazard='ALL';
const $=id=>document.getElementById(id);
const esc=s=>String(s??'').replace(/[&<>"']/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[m]));
function toast(msg){$('toast').textContent=msg;$('toast').style.display='block';setTimeout(()=>$('toast').style.display='none',2800)}
function toggleSidebar(){$('sidebar').classList.toggle('open')}
function go(id){document.querySelectorAll('.page').forEach(p=>p.classList.toggle('active',p.id===id));document.querySelectorAll('.navbtn').forEach(b=>b.classList.toggle('active',b.dataset.page===id));$('sidebar').classList.remove('open');if(id==='overview')loadDashboard();if(id==='history'){loadHistory();loadInvestigations()}if(id==='investigation')loadInvestigations();if(id==='map')setTimeout(()=>{loadMap();loadHotspots()},100);if(id==='analytics')loadAnalytics();if(id==='reports')renderReport();window.scrollTo({top:0,behavior:'smooth'})}
document.querySelectorAll('.navbtn').forEach(b=>b.onclick=()=>go(b.dataset.page));
function badge(x){return `<span class="badge ${String(x).toLowerCase()}">${esc(x)}</span>`}
function formatDate(x){return x?new Date(x).toLocaleString():'—'}

async function loadDashboard(){try{const [s,r]=await Promise.all([fetch('/api/dashboard/stats'),fetch('/api/dashboard/recent?limit=8')]);const stats=await s.json(),recent=await r.json();$('dSurveys').textContent=stats.total_surveys;$('dAnomalies').textContent=stats.total_anomalies;$('dConfidence').textContent=stats.avg_confidence+'%';$('dCritical').textContent=stats.critical;$('recentFeed').innerHTML=(recent.anomalies||[]).map(a=>`<div class="feed"><div>${badge(a.hazard_level)}</div><div><b class="text-xs">${esc(a.anomaly_id)} · ${esc(a.classification)}</b><div class="muted text-[10px]">${a.confidence}% · priority ${a.priority_score}</div></div></div>`).join('')||'<div class="muted text-sm">No anomalies yet.</div>';$('notifCount').textContent=stats.critical||0;drawTrend(stats.daily_trend||[])}catch(e){toast(e.message)}}
function drawTrend(rows){if(charts.trend)charts.trend.destroy();charts.trend=new Chart($('trendChart'),{type:'line',data:{labels:rows.map(x=>x.date),datasets:[{label:'Detections',data:rows.map(x=>x.detections),tension:.35}]},options:{plugins:{legend:{display:false}},scales:{x:{ticks:{color:'#60798a'}},y:{ticks:{color:'#60798a'},beginAtZero:true}}}})}

function fileSetup(){
const drop=$('drop');
const startAutoScan=file=>{if(!file)return;currentFile=file;$('fileName').textContent=file.name;setTimeout(()=>analyze(),180)};
$('fileInput').onchange=e=>startAutoScan(e.target.files[0]);
['dragenter','dragover'].forEach(x=>drop.addEventListener(x,e=>{e.preventDefault();drop.classList.add('drag')}));
['dragleave','drop'].forEach(x=>drop.addEventListener(x,e=>{e.preventDefault();drop.classList.remove('drag')}));
drop.addEventListener('drop',e=>startAutoScan(e.dataTransfer.files[0]));
}
async function analyze(){
if(!currentFile)return toast('Select a sonar image first.');
$('scanWorkspace').classList.add('scan-running');$('scanResult').classList.add('hidden');runScanAnimation();
try{
const fd=new FormData();fd.append('file',currentFile);fd.append('survey_name',$('surveyName').value);fd.append('confidence',Number($('confidence').value)/100);fd.append('range',$('range').value);fd.append('latitude',$('lat').value);fd.append('longitude',$('lon').value);
const health=await (await fetch('/api/health')).json();if(!health.model_loaded)throw Error('Model not loaded: '+(health.error||'unknown'));
const res=await fetch('/api/detect',{method:'POST',body:fd});const data=await res.json();if(!data.success)throw Error(data.error||'Detection failed');
currentData=data;currentSurveyId=data.survey_id;finishScanAnimation();renderResult(data);toast('Analysis completed and survey saved.');loadDashboard();loadHistory();
}catch(e){stopScanAnimation();$('scanWorkspace').classList.remove('scan-running');toast(e.message)}
}
function runScanAnimation() {
    const stages = [
        ['step1', 'Acquiring sonar frame', 'Reading side-scan imagery...'],
        ['step2', 'Enhancing acoustic scene', 'Applying sonar noise reduction and contrast enhancement...'],
        ['step3', 'Running YOLO detection', 'Searching trained marine-debris classes...'],
        ['step4', 'Calculating anomaly position', 'Converting detections into geospatial coordinates...'],
        ['step5', 'Building intelligence report', 'Preparing priority, review and map metadata...']
    ];

    let index = 0;

    function applyStage() {
        stages.forEach(function(stage, stageIndex) {
            const step = $(stage[0]);
            if (!step) return;
            step.classList.toggle('active', stageIndex === index);
            step.classList.toggle('done', stageIndex < index);
        });

        const title = $('scanStage');
        const message = $('scanMessage');
        const progress = $('scanProgress');

        if (title) title.textContent = stages[index][1];
        if (message) message.textContent = stages[index][2];
        if (progress) progress.style.width = ((index + 1) * 20) + '%';

        index += 1;
        if (index >= stages.length) index = 0;
    }

    clearInterval(scanAnimTimer);
    applyStage();
    scanAnimTimer = setInterval(applyStage, 700);
}

function finishScanAnimation() {
    clearInterval(scanAnimTimer);
    scanAnimTimer = null;

    ['step1', 'step2', 'step3', 'step4', 'step5'].forEach(function(id) {
        const step = $(id);
        if (step) {
            step.classList.remove('active');
            step.classList.add('done');
        }
    });

    if ($('scanProgress')) $('scanProgress').style.width = '100%';
    if ($('scanStage')) $('scanStage').textContent = 'Analysis complete';
    if ($('scanMessage')) $('scanMessage').textContent = 'Anomaly candidates, priority scores and geospatial metadata generated.';

    setTimeout(function() {
        $('scanWorkspace').classList.remove('scan-running');
        $('scanFormPanel').style.display = 'none';
        $('scanResult').classList.remove('hidden');
    }, 650);
}

function stopScanAnimation() {
    clearInterval(scanAnimTimer);
    scanAnimTimer = null;
}

function newScan() {
    stopScanAnimation();

    currentData = null;
    currentSurveyId = null;
    currentFile = null;

    const fileInput = $('fileInput');
    const fileName = $('fileName');
    const surveyName = $('surveyName');
    const scanWorkspace = $('scanWorkspace');
    const scanFormPanel = $('scanFormPanel');
    const scanResult = $('scanResult');

    if (fileInput) fileInput.value = '';
    if (fileName) fileName.textContent = '';
    if (surveyName) surveyName.value = '';

    if (scanWorkspace) scanWorkspace.classList.remove('scan-running');
    if (scanFormPanel) scanFormPanel.style.display = 'block';
    if (scanResult) scanResult.classList.add('hidden');

    if ($('scanProgress')) $('scanProgress').style.width = '20%';

    document.querySelectorAll('.scan-step').forEach(function(step, index) {
        step.classList.toggle('active', index === 0);
        step.classList.remove('done');
    });

    if ($('scanStage')) $('scanStage').textContent = 'Acquiring sonar frame';
    if ($('scanMessage')) $('scanMessage').textContent = 'Reading side-scan imagery...';
}

function renderResult(d){
$('scanResult').classList.remove('hidden');$('rTotal').textContent=d.summary.total;$('rCritical').textContent=d.summary.critical;$('rConf').textContent=d.summary.avg_confidence+'%';$('rArea').textContent=d.summary.total_area+' m²';$('rawImg').src=d.raw_image||'';$('resultImg').src=d.result_image;$('resultCount').textContent=(d.detections||[]).length+' object'+((d.detections||[]).length===1?'':'s')+' detected';
$('resultList').innerHTML=(d.detections||[]).sort((a,b)=>b.priority_score-a.priority_score).map(a=>`<div class="feed"><div>${badge(a.hazard_level)}</div><div class="flex-1"><b class="text-xs">${esc(a.anomaly_id)} · ${esc(a.classification)}</b><div class="muted text-[10px]">${a.confidence}% · ${a.width_m}×${a.length_m}m · ${a.latitude}, ${a.longitude}</div><div class="mt-2 text-[10px]">Priority <b>${a.priority_score}/100</b> · Shadow evidence <b>${a.shadow_evidence}%</b><div class="mini-meter mt-1"><span style="width:${a.priority_score}%"></span></div></div></div><button class="btn" onclick="showExplanation(${a.id||0},${JSON.stringify(a).replaceAll('"','&quot;')})"><i class="fa-solid fa-microscope"></i></button></div>`).join('')||'<div class="muted text-sm">No objects detected.</div>';
if(resultMap)resultMap.remove();let center=[Number($('lat').value),Number($('lon').value)];if(d.detections.length)center=[d.detections[0].latitude,d.detections[0].longitude];resultMap=L.map('resultMap').setView(center,15);L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',{attribution:'© OpenStreetMap'}).addTo(resultMap);d.detections.forEach(a=>L.circleMarker([a.latitude,a.longitude],{radius:8,weight:2,fillOpacity:.8}).addTo(resultMap).bindPopup(`<b>${esc(a.anomaly_id)}</b><br>${esc(a.classification)}<br>${a.hazard_level} · ${a.confidence}%<br>Priority: ${a.priority_score}`));renderReport();loadHotspots('hotspotSummary')
}
async function showExplanation(id,encoded){let a=encoded;try{a=JSON.parse(String(encoded).replaceAll('&quot;','"'))}catch{}if(id){try{const r=await fetch('/api/anomalies/'+id+'/context');const c=await r.json();if(c.success)a={...a,history_count:c.history_count,history:c.history}}catch{}}$('modalTitle').textContent=(a.anomaly_id||'Anomaly')+' · AI Explanation';$('modalBody').innerHTML=`<div class="grid2"><div class="explain-card"><div class="sectionlabel">Detection reasoning</div>${(a.explanation||[]).map(x=>`<div class="explain-row"><span>✓ ${esc(x)}</span></div>`).join('')}<div class="explain-row"><b>Confidence</b><b>${a.confidence}%</b></div><div class="explain-row"><b>Shadow evidence</b><b>${a.shadow_evidence||'—'}%</b></div><div class="explain-row"><b>Cleanup priority</b><b>${a.priority_score||'—'}/100</b></div></div><div class="explain-card"><div class="sectionlabel">Spatial context</div><div class="explain-row"><b>Latitude</b><span>${a.latitude}</span></div><div class="explain-row"><b>Longitude</b><span>${a.longitude}</span></div><div class="explain-row"><b>Dimensions</b><span>${a.width_m} × ${a.length_m} m</span></div><div class="explain-row"><b>Area</b><span>${a.area_sq_m} m²</span></div><div class="explain-row"><b>Nearby same-class records</b><span>${a.history_count??'—'}</span></div></div></div>`;$('modal').style.display='flex'}
async function reviewQuick(id,status){if(!id)return toast('This historical record has no editable anomaly ID.');const r=await fetch('/api/anomalies/'+id+'/review',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({decision:status,reviewed_by:'Operator'})});const d=await r.json();if(d.success){toast('Review updated');loadInvestigations();loadDashboard()}else toast(d.error||'Review failed')}
async function loadHistory(){const q=encodeURIComponent($('historySearch')?.value||'');const d=await(await fetch('/api/surveys?per_page=100&q='+q)).json();if(!d.success)return;$('historyBody').innerHTML=d.surveys.map(s=>`<tr><td>#${s.id}</td><td><b>${esc(s.survey_name)}</b><div class="muted">${esc(s.original_filename)}</div></td><td>${formatDate(s.created_at)}</td><td>${s.summary.total}</td><td>${s.summary.critical}</td><td>${s.summary.avg_confidence}%</td><td>${badge(s.status)}</td><td><button class="btn" onclick="viewSurvey(${s.id})">View</button> <button class="btn danger" onclick="deleteSurvey(${s.id})">Delete</button></td></tr>`).join('')||'<tr><td colspan="8" class="muted">No surveys found.</td></tr>';const opts=d.surveys.map(s=>`<option value="${s.id}">#${s.id} · ${esc(s.survey_name)}</option>`).join('');$('compareA').innerHTML=opts;$('compareB').innerHTML=opts}
async function viewSurvey(id){const d=await(await fetch('/api/surveys/'+id)).json();if(!d.success)return;const s=d.survey;currentData={survey_id:s.id,raw_image:s.raw_image,result_image:s.result_image,detections:s.detections,summary:s.summary};currentSurveyId=s.id;$('modalTitle').textContent=s.survey_name;$('modalBody').innerHTML=`<div class="grid2"><img class="resultimg" src="${s.result_image}"><div><div class="cards" style="grid-template-columns:1fr 1fr"><div class="metric"><div class="small">Objects</div><div class="num">${s.summary.total}</div></div><div class="metric"><div class="small">Confidence</div><div class="num">${s.summary.avg_confidence}%</div></div></div><div class="mt-4">${s.detections.map(a=>`<div class="feed"><div>${badge(a.hazard_level)}</div><div><b>${a.anomaly_id} · ${a.classification}</b><div class="muted text-xs">${a.confidence}% · priority ${a.priority_score} · ${a.review_status}</div></div></div>`).join('')}</div></div></div>`;$('modal').style.display='flex';renderReport()}
async function deleteSurvey(id){if(!confirm('Delete this survey and its anomalies?'))return;const r=await fetch('/api/surveys/'+id,{method:'DELETE'});const d=await r.json();if(d.success){toast('Survey deleted');loadHistory();loadDashboard();loadInvestigations()}else toast(d.error||'Delete failed')}
function closeModal(){$('modal').style.display='none'}
async function loadInvestigations(){const d=await(await fetch('/api/dashboard/recent?limit=100')).json();$('investigationBody').innerHTML=(d.anomalies||[]).map(a=>`<tr><td><b>${a.anomaly_id}</b><div class="muted">Survey #${a.survey_id}</div></td><td>${esc(a.classification)}</td><td>${a.confidence}%</td><td class="priority">${a.priority_score}</td><td>${badge(a.hazard_level)}</td><td>${Number(a.latitude).toFixed(5)}, ${Number(a.longitude).toFixed(5)}</td><td>${badge(a.review_status)}</td><td><button class="btn" onclick="showExplanation(${a.id},${JSON.stringify(a).replaceAll('"','&quot;')})"><i class="fa-solid fa-eye"></i></button> <button class="btn" onclick="reviewQuick(${a.id},'approved')">Approve</button> <button class="btn danger" onclick="reviewQuick(${a.id},'rejected')">Reject</button></td></tr>`).join('')||'<tr><td colspan="8" class="muted">No anomalies.</td></tr>'}
async function loadMap(hazard='ALL'){currentHazard=hazard;const params=new URLSearchParams({hazard_level:hazard,classification:$('mapClass').value||'ALL',review_status:$('mapReview').value||'ALL',min_confidence:$('mapConf').value||0});const d=await(await fetch('/api/map/anomalies?'+params)).json();if(intelMap)intelMap.remove();intelMap=L.map('intelMap').setView([18.9438,72.8360],12);L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',{attribution:'© OpenStreetMap'}).addTo(intelMap);const points=[];(d.anomalies||[]).forEach(a=>{L.circleMarker([a.latitude,a.longitude],{radius:Math.max(6,Math.min(12,a.priority_score/10)),weight:2,fillOpacity:.8}).addTo(intelMap).bindPopup(`<b>${a.anomaly_id}</b><br>${esc(a.classification)}<br>${a.hazard_level} · ${a.confidence}%<br>Priority ${a.priority_score}/100`);points.push([a.latitude,a.longitude])});if(points.length)intelMap.fitBounds(points,{padding:[25,25]});populateMapClasses(d.anomalies||[])}
function populateMapClasses(rows){const vals=[...new Set(rows.map(x=>x.classification))];const old=$('mapClass').value;$('mapClass').innerHTML='<option>ALL</option>'+vals.map(x=>`<option>${esc(x)}</option>`).join('');$('mapClass').value=vals.includes(old)?old:'ALL'}
function mapMode(btn,h){document.querySelectorAll('.switch .btn').forEach(x=>x.classList.remove('active'));btn.classList.add('active');loadMap(h)}
async function loadHotspots(targetId='hotspots'){const d=await(await fetch('/api/hotspots')).json();const rows=d.hotspots||[];$(targetId).innerHTML=rows.slice(0,6).map((x,i)=>`<div class="metric"><div class="small">Hotspot #${i+1}</div><div class="num">${x.priority}</div><div class="muted text-xs mt-1">${x.count} anomalies · ${x.critical} critical · ${x.area.toFixed(1)}m²</div><div class="muted text-[10px] mt-1">${x.latitude}, ${x.longitude}</div></div>`).join('')||'<div class="muted text-sm">No hotspot data yet.</div>'}
async function loadAnalytics(){const d=await(await fetch('/api/analytics')).json();$('aTotal').textContent=d.total;$('aConf').textContent=d.avg_confidence+'%';$('aArea').textContent=d.total_area+' m²';$('aPending').textContent=d.review.pending||0;drawPie('hazardChart',d.hazard);drawPie('reviewChart',d.review);drawBar('classChart',d.classes)}
function drawPie(id,obj){if(charts[id])charts[id].destroy();charts[id]=new Chart($(id),{type:'doughnut',data:{labels:Object.keys(obj),datasets:[{data:Object.values(obj)}]},options:{plugins:{legend:{labels:{color:'#8aa0af'}}}}})}
function drawBar(id,obj){if(charts[id])charts[id].destroy();charts[id]=new Chart($(id),{type:'bar',data:{labels:Object.keys(obj),datasets:[{data:Object.values(obj)}]},options:{plugins:{legend:{display:false}},scales:{x:{ticks:{color:'#8aa0af'}},y:{beginAtZero:true,ticks:{color:'#8aa0af'}}}}})}
async function compareSurveys(){const a=$('compareA').value,b=$('compareB').value;if(!a||!b)return;const A=await(await fetch('/api/surveys/'+a)).json(),B=await(await fetch('/api/surveys/'+b)).json();if(!A.success||!B.success)return;$('compareResult').innerHTML=`<div class="cards"><div class="metric"><div class="small">Object delta</div><div class="num">${B.survey.summary.total-A.survey.summary.total>=0?'+':''}${B.survey.summary.total-A.survey.summary.total}</div></div><div class="metric"><div class="small">Critical delta</div><div class="num">${B.survey.summary.critical-A.survey.summary.critical>=0?'+':''}${B.survey.summary.critical-A.survey.summary.critical}</div></div><div class="metric"><div class="small">High delta</div><div class="num">${B.survey.summary.high-A.survey.summary.high>=0?'+':''}${B.survey.summary.high-A.survey.summary.high}</div></div><div class="metric"><div class="small">Area delta</div><div class="num">${(B.survey.summary.total_area-A.survey.summary.total_area)>=0?'+':''}${(B.survey.summary.total_area-A.survey.summary.total_area).toFixed(2)}m²</div></div></div>`}
async function exportCurrent(type){if(!currentData)return toast('Run a scan or open a survey first.');const r=await fetch('/api/download/'+type,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(currentData)});if(type==='json'){const text=await r.text();const blob=new Blob([text],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='varun_netra_report.json';a.click();URL.revokeObjectURL(a.href)}else{const blob=await r.blob();const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='varun_netra_report.csv';a.click();URL.revokeObjectURL(a.href)}}
function renderReport(){if(!currentData)return;$('reportContent').innerHTML=`<div class="grid3"><div><b>Survey</b><br>#${currentSurveyId||'—'}</div><div><b>Detections</b><br>${currentData.summary?.total||0}</div><div><b>Confidence</b><br>${currentData.summary?.avg_confidence||0}%</div></div><div class="mt-4"><table class="table"><thead><tr><th>ID</th><th>Class</th><th>Hazard</th><th>Priority</th><th>Confidence</th><th>Coordinates</th></tr></thead><tbody>${(currentData.detections||[]).map(a=>`<tr><td>${a.anomaly_id}</td><td>${a.classification}</td><td>${a.hazard_level}</td><td>${a.priority_score}</td><td>${a.confidence}%</td><td>${a.latitude}, ${a.longitude}</td></tr>`).join('')}</tbody></table></div>`}
function toggleNotifications(){go('overview');toast('Critical alerts are shown in the dashboard feed.')}
fileSetup();loadDashboard();
