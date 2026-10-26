'use strict';

// Success rates (%) from the paper's HumanoidArena main-results table.
const modelResults = [
  {name: 'ACT †', values: [20,30,60,58.30,21.70,58.30,28.30,36.67,41.65,39.51]},
  {name: 'DP †', values: [53.30,35,0,83.30,83.30,21.70,50,29.43,59.58,46.66]},
  {name: 'FM †', values: [21.70,20,0,76.70,48.30,18.30,38.30,13.90,45.40,31.90]},
  {name: 'π₀.₅ †', values: [10,30,75,85,70,71.70,35,38.33,65.43,53.81]},
  {name: 'GR00T (scratch)', values: [0,0,75,11.67,13.33,68.33,36.67,25,32.50,29.29]},
  {name: 'GR00T (N1.7)', values: [13.33,0,76.67,8.33,75,11.67,26.67,30,30.42,30.24]},
  {name: 'HuGe-OFT', ours: true, values: [48.33,0,53.33,75,55,73.33,73.33,33.89,69.17,54.05]},
  {name: 'HuGe-GR00T', ours: true, values: [71.67,15,81.67,95,78.33,68.33,56.67,56.11,74.58,66.67]},
  {name: 'HuGe-Pi', ours: true, values: [63.33,11.67,76.67,88.33,83.33,68.33,63.33,50.56,75.83,65]}
];
const metrics = {
  football: [0, 'Football · human-object interaction'],
  doubledesk: [1, 'DoubleDesk · human-object interaction'],
  ppbox: [2, 'P&PBox · human-object interaction'],
  opendoor: [3, 'OpenDoor · human-scene interaction'],
  sitsofa: [4, 'SitSofa · human-scene interaction'],
  boxing: [5, 'Boxing · human-scene interaction'],
  visnavi: [6, 'VisNavi · human-scene interaction'],
  hoi: [7, 'Unweighted average across 3 HOI tasks'],
  hsi: [8, 'Unweighted average across 4 HSI tasks'],
  overall: [9, 'Unweighted average across all 7 tasks']
};
const metricSelect = document.querySelector('#result-metric');
const chart = document.querySelector('#results-chart');
function renderChart(announce = false) {
  const [index, description] = metrics[metricSelect.value];
  const sorted = [...modelResults].sort((a,b) => b.values[index] - a.values[index]);
  const fragment = document.createDocumentFragment();
  sorted.forEach(model => {
    const row = document.createElement('div');
    row.className = 'chart-row';
    const label = document.createElement('span');
    label.className = `chart-label${model.ours ? ' ours' : ''}`;
    label.textContent = model.name;
    const track = document.createElement('div');
    track.className = 'chart-track';
    track.setAttribute('aria-hidden', 'true');
    const bar = document.createElement('span');
    bar.className = `chart-bar${model.ours ? ' ours' : ''}`;
    bar.style.setProperty('--value', `${model.values[index]}%`);
    track.append(bar);
    const value = document.createElement('span');
    value.className = 'chart-value';
    value.textContent = model.values[index].toFixed(2);
    row.append(label, track, value);
    fragment.append(row);
  });
  chart.replaceChildren(fragment);
  chart.setAttribute('aria-label', `${description}. Success rates in percent.`);
  document.querySelector('#metric-description').textContent = description;
  if (announce) document.querySelector('#chart-announcement').textContent = `${description}. Highest success rate: ${sorted[0].name}, ${sorted[0].values[index].toFixed(2)} percent.`;
}
metricSelect.addEventListener('change', () => renderChart(true));
renderChart();

// The native SVG remains readable without JavaScript; the legend adds focus.
const scalingKeys = document.querySelectorAll('[data-scaling-model]');
const scalingCurves = document.querySelectorAll('[data-scaling-series]');
scalingKeys.forEach(button => {
  button.addEventListener('click', () => {
    const selected = button.getAttribute('aria-pressed') === 'true' ? null : button.dataset.scalingModel;
    scalingKeys.forEach(key => key.setAttribute('aria-pressed', String(key.dataset.scalingModel === selected)));
    scalingCurves.forEach(curve => curve.classList.toggle('is-muted', selected !== null && curve.dataset.scalingSeries !== selected));
  });
});

// The source montage contains 15 aligned observations in each of two rows.
const slider = document.querySelector('#frame-slider');
slider.addEventListener('input', () => {
  const frame = Number(slider.value);
  document.querySelector('#frame-number').textContent = `${String(frame).padStart(2,'0')} / 15`;
  for (const element of document.querySelectorAll('.adaptation-frame')) {
    element.style.backgroundPositionX = `${(frame - 1) / 14 * 100}%`;
    const type = element.classList.contains('human-frame') ? 'Original human observation' : 'Visually adapted robot observation';
    element.setAttribute('aria-label', `${type}, frame ${frame} of 15`);
  }
});

const menuToggle = document.querySelector('.menu-toggle');
const navigation = document.querySelector('#navigation');
function closeMenu() {
  navigation.classList.remove('is-open');
  menuToggle.setAttribute('aria-expanded', 'false');
}
menuToggle.addEventListener('click', () => {
  const opened = menuToggle.getAttribute('aria-expanded') === 'true';
  navigation.classList.toggle('is-open', !opened);
  menuToggle.setAttribute('aria-expanded', String(!opened));
});
navigation.addEventListener('click', event => {
  if (event.target.closest('a')) closeMenu();
});
document.addEventListener('keydown', event => {
  if (event.key === 'Escape' && menuToggle.getAttribute('aria-expanded') === 'true') {
    closeMenu();
    menuToggle.focus();
  }
});
window.matchMedia('(min-width: 601px)').addEventListener('change', closeMenu);

const copyButton = document.querySelector('#copy-citation');
// Keep attention on the selected demonstration when switching between clips.
const demoVideos = document.querySelectorAll('.demo-card video');
demoVideos.forEach(video => {
  video.addEventListener('play', () => {
    demoVideos.forEach(other => {
      if (other !== video) other.pause();
    });
  });
});

copyButton.addEventListener('click', async () => {
  const status = document.querySelector('#copy-status');
  const citation = document.querySelector('#bibtex');
  try {
    await navigator.clipboard.writeText(citation.textContent);
    status.textContent = 'Citation copied to clipboard.';
    copyButton.textContent = 'Copied ✓';
    setTimeout(() => { copyButton.textContent = 'Copy citation ⧉'; }, 2500);
  } catch {
    const range = document.createRange();
    range.selectNodeContents(citation);
    const selection = window.getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
    status.textContent = 'Text selected. Press ⌘C / Ctrl+C to copy.';
  }
});
