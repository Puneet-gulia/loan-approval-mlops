const definitions = [
  ['no_of_dependents','Dependents','profile',0,5,1,'0 to 5'],
  ['education','Education','profile',['Graduate','Not Graduate']],
  ['self_employed','Self employed','profile',['No','Yes']],
  ['cibil_score','CIBIL score','profile',300,900,1,'300 to 900'],
  ['income_annum','Annual income (INR)','loan',100000,9900000,1,'100,000 to 9,900,000'],
  ['loan_amount','Requested amount (INR)','loan',300000,39500000,1,'300,000 to 39,500,000'],
  ['loan_term','Loan term (years)','loan',2,20,2,'2 to 20 years, in steps of 2'],
  ['residential_assets_value','Residential assets (INR)','assets',0,29100000,1,'0 to 29,100,000'],
  ['commercial_assets_value','Commercial assets (INR)','assets',0,19400000,1,'0 to 19,400,000'],
  ['luxury_assets_value','Luxury assets (INR)','assets',0,39200000,1,'0 to 39,200,000'],
  ['bank_asset_value','Bank assets (INR)','assets',0,14700000,1,'0 to 14,700,000']
];
const sample = {no_of_dependents:2,education:'Graduate',self_employed:'No',cibil_score:780,income_annum:5000000,loan_amount:15000000,loan_term:10,residential_assets_value:8000000,commercial_assets_value:3000000,luxury_assets_value:10000000,bank_asset_value:4000000};
for (const [name,title,section,min,max,step,help] of definitions) {
  const label=document.createElement('label'); label.htmlFor=name; label.textContent=title;
  const choices=Array.isArray(min); const input=document.createElement(choices?'select':'input');
  input.id=name; input.name=name; input.required=true;
  if (choices) {const empty=new Option('Choose an option',''); empty.disabled=true; empty.selected=true; input.add(empty); for(const choice of min) input.add(new Option(choice,choice));}
  else {input.type='number'; input.min=min; input.max=max; input.step=step; input.inputMode='numeric';}
  label.append(input);
  if(help){const hint=document.createElement('span');hint.id=`${name}-hint`;hint.className='range';hint.textContent=help;input.setAttribute('aria-describedby',hint.id);label.append(hint);}
  document.getElementById(section).append(label);
}
document.getElementById('example').addEventListener('click',()=>{
  for(const [name,value] of Object.entries(sample)) document.getElementById(name).value=value;
  document.getElementById('error').hidden=true;
});
document.getElementById('application').addEventListener('submit',async event=>{
  event.preventDefault(); const error=document.getElementById('error');error.hidden=true;
  const button=document.getElementById('submit');button.disabled=true;button.textContent='Estimating…';
  const values={};for(const [name,,,min] of definitions)values[name]=Array.isArray(min)?document.getElementById(name).value:Number(document.getElementById(name).value);
  try {
    const response=await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(values)});
    const data=await response.json();
    if(!response.ok){throw new Error(Array.isArray(data.detail)?data.detail.map(d=>`${d.loc.slice(1).join('.')}: ${d.msg}`).join('; '):(data.detail||'The server could not complete this estimate.'));}
    document.getElementById('decision').textContent=data.prediction;
    document.getElementById('probability').textContent=`${(data.approval_probability*100).toFixed(1)}%`;
    document.getElementById('score-fill').style.width=`${data.approval_probability*100}%`;
    const tips=document.getElementById('tips');tips.replaceChildren();for(const tip of data.tips){const li=document.createElement('li');li.textContent=tip;tips.append(li);}
    document.getElementById('version').textContent=`Model ${data.model_version} / Request ${data.request_id.slice(0,8)}`;
    document.getElementById('empty').hidden=true;document.getElementById('outcome').hidden=false;
    if(window.innerWidth<850)document.querySelector('.result').scrollIntoView({behavior:'smooth',block:'start'});
  } catch(e) {error.textContent=e.message==='Failed to fetch'?'Connection unavailable. Check that the server is running and retry.':e.message;error.hidden=false;document.getElementById('outcome').hidden=true;document.getElementById('empty').hidden=false;}
  finally {button.disabled=false;button.innerHTML='Estimate loan outcome <span>↗</span>';}
});
