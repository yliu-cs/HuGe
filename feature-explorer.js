(() => {
  'use strict';

  const root = document.getElementById('feature-explorer');
  const data = window.HUGE_FEATURE_DATA;
  if (!root) return;
  if (!data?.encoders?.length || !data?.groups?.length) {
    const note = root.querySelector('#feature-data-note');
    if (note) {
      note.hidden = false;
      note.textContent = 'Feature projections could not be loaded.';
    }
    return;
  }

  const SVG_NS = 'http://www.w3.org/2000/svg';
  const WIDTH = 560;
  const HEIGHT = 352;
  const BOX = { left: 58, right: 540, top: 22, bottom: 302 };
  const metrics = {
    wanvae: { raw: 2.0311, adapted: 1.3499, reduction: 33.54 },
    videomae: { raw: 7.5697, adapted: 5.5204, reduction: 27.07 },
    vjepa2: { raw: 305.5143, adapted: 263.2697, reduction: 13.83 }
  };
  const groupInfo = new Map(data.groups.map(group => [group.id, group]));
  const encoders = new Map(data.encoders.map(encoder => [encoder.id, encoder]));
  const encoderButtons = [...root.querySelectorAll('[data-feature-encoder]')];
  const viewButtons = [...root.querySelectorAll('[data-feature-view]')];
  const groupButtons = [...root.querySelectorAll('[data-feature-group]')];
  const densityButton = root.querySelector('#feature-density');
  const panels = [...root.querySelectorAll('[data-feature-panel]')].map(element => ({
    element,
    key: element.dataset.featurePanel,
    svg: element.querySelector('.feature-scatter'),
    canvas: element.querySelector('.feature-density-canvas'),
    tooltip: element.querySelector('.feature-tooltip'),
    empty: element.querySelector('.feature-panel-empty'),
    points: []
  })).filter(panel => panel.svg && panel.canvas);
  if (!panels.length) return;

  let encoder = encoders.get('wanvae') || data.encoders[0];
  let view = 'compare';
  let showDensity = true;
  let visible = new Set(encoder.groups.map(group => group.id));
  let selected = null;
  let hovered = null;
  const densityCache = new Map();
  const contourCache = new Map();

  function text(selector, value) {
    const element = root.querySelector(selector);
    if (element) element.textContent = value;
  }

  function announce(value) {
    text('#feature-live', value);
  }

  function svgElement(tag, attributes = {}, content) {
    const element = document.createElementNS(SVG_NS, tag);
    for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, String(value));
    if (content !== undefined) element.textContent = content;
    return element;
  }

  function project(point) {
    const [xmin, xmax, ymin, ymax] = encoder.bounds;
    return [
      BOX.left + (point[0] - xmin) / (xmax - xmin || 1) * (BOX.right - BOX.left),
      BOX.bottom - (point[1] - ymin) / (ymax - ymin || 1) * (BOX.bottom - BOX.top)
    ];
  }

  function number(value, precision = 3) {
    return Math.abs(value) < 0.5 * 10 ** -precision ? '0' : value.toFixed(precision);
  }

  function marker(info, x, y, radius, className) {
    const common = { class: className, fill: info.color };
    if (info.marker === 'square') {
      return svgElement('rect', { ...common, x: x - radius, y: y - radius, width: radius * 2, height: radius * 2, rx: 0.4 });
    }
    if (info.marker === 'triangle') {
      return svgElement('path', { ...common, d: `M${x - radius * 1.2},${y - radius * 0.8} L${x + radius * 1.2},${y - radius * 0.8} L${x},${y + radius * 1.3} Z` });
    }
    return svgElement('circle', { ...common, cx: x, cy: y, r: radius });
  }

  function panelGroups(panel) {
    const ids = view === 'overlay' ? data.groups.map(group => group.id)
      : panel.key === 'before' ? ['human_original', 'real_demo'] : ['processed_final', 'real_demo'];
    return encoder.groups.filter(group => ids.includes(group.id) && visible.has(group.id));
  }

  function frame(panel) {
    const svg = panel.svg;
    svg.replaceChildren();
    const title = view === 'overlay' ? 'All feature distributions' : panel.key === 'before' ? 'Before visual adaptation' : 'After visual adaptation';
    const count = panelGroups(panel).reduce((total, group) => total + group.points.length, 0);
    svg.setAttribute('aria-label', `${encoder.label}: ${title}. ${count} samples. PC1 and PC2 projection. Use Left and Right arrows to inspect samples, Home and End to jump, and Escape to clear selection.`);
    svg.append(svgElement('title', {}, `${encoder.label} — ${title}`));
    svg.append(svgElement('desc', {}, 'Actual feature samples projected by PCA. Each array row is local to its group; rows across groups are not paired.'));
    const clipId = `feature-clip-${panel.key}`;
    const defs = svgElement('defs');
    const clip = svgElement('clipPath', { id: clipId });
    clip.append(svgElement('rect', { x: BOX.left, y: BOX.top, width: BOX.right - BOX.left, height: BOX.bottom - BOX.top }));
    defs.append(clip);
    svg.append(defs);

    const axes = svgElement('g');
    const [xmin, xmax, ymin, ymax] = encoder.bounds;
    for (let index = 0; index <= 4; index += 1) {
      const fraction = index / 4;
      const x = BOX.left + fraction * (BOX.right - BOX.left);
      const y = BOX.bottom - fraction * (BOX.bottom - BOX.top);
      axes.append(svgElement('line', { class: 'fe-grid-line', x1: x, x2: x, y1: BOX.top, y2: BOX.bottom }));
      axes.append(svgElement('line', { class: 'fe-grid-line', x1: BOX.left, x2: BOX.right, y1: y, y2: y }));
      axes.append(svgElement('text', { class: 'fe-tick-label', x, y: BOX.bottom + 18, 'text-anchor': 'middle' }, number(xmin + fraction * (xmax - xmin), 2)));
      axes.append(svgElement('text', { class: 'fe-tick-label', x: BOX.left - 10, y: y + 3, 'text-anchor': 'end' }, number(ymin + fraction * (ymax - ymin), 2)));
    }
    axes.append(svgElement('path', { d: `M${BOX.left},${BOX.top}V${BOX.bottom}H${BOX.right}`, fill: 'none', stroke: '#7f8b83', 'stroke-width': 1 }));
    axes.append(svgElement('text', { class: 'fe-axis-label', x: (BOX.left + BOX.right) / 2, y: HEIGHT - 7, 'text-anchor': 'middle' }, `PC1 · ${(encoder.explainedVariance[0] * 100).toFixed(1)}%`));
    axes.append(svgElement('text', { class: 'fe-axis-label', transform: `translate(15 ${(BOX.top + BOX.bottom) / 2}) rotate(-90)`, 'text-anchor': 'middle' }, `PC2 · ${(encoder.explainedVariance[1] * 100).toFixed(1)}%`));
    svg.append(axes);
    const layers = svgElement('g', { 'clip-path': `url(#${clipId})` });
    svg.append(layers);
    return layers;
  }

  function densityImage(group) {
    const key = `${encoder.id}/${group.id}`;
    if (densityCache.has(key)) return densityCache.get(key);
    const { size, values } = group.density;
    const canvas = document.createElement('canvas');
    canvas.width = canvas.height = size;
    const context = canvas.getContext('2d');
    if (!context) return null;
    const image = context.createImageData(size, size);
    const color = groupInfo.get(group.id).color.replace('#', '');
    const rgb = [0, 2, 4].map(start => parseInt(color.slice(start, start + 2), 16));
    for (let y = 0; y < size; y += 1) {
      for (let x = 0; x < size; x += 1) {
        const offset = (y * size + x) * 4;
        image.data[offset] = rgb[0];
        image.data[offset + 1] = rgb[1];
        image.data[offset + 2] = rgb[2];
        image.data[offset + 3] = Math.round(42 * Math.pow(values[(size - 1 - y) * size + x], 0.7));
      }
    }
    context.putImageData(image, 0, 0);
    densityCache.set(key, canvas);
    return canvas;
  }

  function paintDensity(panel, groups) {
    panel.canvas.hidden = !showDensity;
    if (!showDensity) return;
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    panel.canvas.width = WIDTH * ratio;
    panel.canvas.height = HEIGHT * ratio;
    const context = panel.canvas.getContext('2d');
    if (!context) return;
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.clearRect(0, 0, WIDTH, HEIGHT);
    for (const group of groups) {
      if (!group.density?.values?.length) continue;
      const image = densityImage(group);
      if (image) context.drawImage(image, BOX.left, BOX.top, BOX.right - BOX.left, BOX.bottom - BOX.top);
    }
  }

  // Marching squares on each group's normalized KDE. The grid's first row is ymin.
  function contours(group) {
    const key = `${encoder.id}/${group.id}`;
    if (contourCache.has(key)) return contourCache.get(key);
    const density = group.density;
    if (!density?.values?.length) return [];
    const { size, values } = density;
    const width = (BOX.right - BOX.left) / (size - 1);
    const height = (BOX.bottom - BOX.top) / (size - 1);
    const result = density.levels.filter(level => level > 0 && level < 1).map(level => {
      const segments = [];
      for (let y = 0; y < size - 1; y += 1) {
        for (let x = 0; x < size - 1; x += 1) {
          const v = [values[y * size + x], values[y * size + x + 1], values[(y + 1) * size + x + 1], values[(y + 1) * size + x]];
          const corners = [[x, y], [x + 1, y], [x + 1, y + 1], [x, y + 1]];
          const intersections = [];
          for (let edge = 0; edge < 4; edge += 1) {
            const next = (edge + 1) % 4;
            if ((v[edge] >= level) === (v[next] >= level)) continue;
            const fraction = (level - v[edge]) / (v[next] - v[edge]);
            intersections.push([
              BOX.left + (corners[edge][0] + fraction * (corners[next][0] - corners[edge][0])) * width,
              BOX.bottom - (corners[edge][1] + fraction * (corners[next][1] - corners[edge][1])) * height
            ]);
          }
          if (intersections.length === 4 && (v.reduce((sum, value) => sum + value, 0) / 4 >= level) !== (v[0] >= level)) {
            intersections.push(intersections.shift());
          }
          for (let index = 0; index + 1 < intersections.length; index += 2) {
            const a = intersections[index];
            const b = intersections[index + 1];
            segments.push(`M${a[0].toFixed(2)},${a[1].toFixed(2)}L${b[0].toFixed(2)},${b[1].toFixed(2)}`);
          }
        }
      }
      return segments.join('');
    });
    contourCache.set(key, result);
    return result;
  }

  function renderPanel(panel) {
    panel.element.hidden = view === 'overlay' && panel.key === 'after';
    panel.points = [];
    if (panel.element.hidden) return;
    const groups = panelGroups(panel);
    const layers = frame(panel);
    paintDensity(panel, groups);
    const title = panel.element.querySelector('[data-panel-title]');
    if (title) title.textContent = view === 'overlay' ? 'All feature distributions' : panel.key === 'before' ? 'Before visual adaptation' : 'After visual adaptation';
    const count = panel.element.querySelector('[data-panel-count]');
    if (count) count.textContent = `${groups.reduce((sum, group) => sum + group.points.length, 0).toLocaleString()} samples`;
    if (showDensity) {
      for (const group of groups) {
        for (const path of contours(group)) {
          layers.append(svgElement('path', { class: 'fe-contour', d: path, stroke: groupInfo.get(group.id).color, fill: 'none', opacity: 0.36, 'stroke-width': 0.75, 'vector-effect': 'non-scaling-stroke' }));
        }
      }
    }
    for (const group of groups) {
      const info = groupInfo.get(group.id);
      const points = svgElement('g', { opacity: 0.65 });
      group.points.forEach((coordinates, row) => {
        const [x, y] = project(coordinates);
        points.append(marker(info, x, y, 2.15, 'fe-point'));
        panel.points.push({ group: group.id, row, coordinates, x, y });
      });
      layers.append(points);
    }
    layers.append(svgElement('g', { 'data-feature-selection': '', 'pointer-events': 'none' }));
    if (panel.empty) {
      const missingRaw = view === 'compare' && panel.key === 'before' && encoder.missingGroups.includes('human_original');
      panel.empty.hidden = groups.length > 0 && !missingRaw;
      panel.empty.textContent = missingRaw ? 'Raw ego features were not supplied for this encoder. Only the available reference samples are shown.' : 'No visible groups in this panel. Use the legend to show a distribution.';
    }
  }

  function description(point) {
    return `${groupInfo.get(point.group).label} · sample ${point.row + 1}`;
  }

  function updateSelection() {
    const active = hovered || selected;
    for (const panel of panels) {
      panel.svg.querySelector('[data-feature-selection]')?.replaceChildren();
      if (panel.tooltip) panel.tooltip.hidden = true;
      if (!active || panel.element.hidden) continue;
      const point = panel.points.find(item => item.group === active.group && item.row === active.row);
      if (!point) continue;
      const layer = panel.svg.querySelector('[data-feature-selection]');
      const halo = svgElement('circle', { class: 'fe-selection', cx: point.x, cy: point.y, r: 7, fill: '#fffdf6', 'fill-opacity': 0.95, stroke: '#243c38', 'stroke-width': 1.4 });
      layer?.append(halo, marker(groupInfo.get(point.group), point.x, point.y, 3.4, 'fe-point'));
      if (panel.tooltip) {
        panel.tooltip.textContent = `${description(point)}\nPC1 ${number(point.coordinates[0], 4)} · PC2 ${number(point.coordinates[1], 4)}`;
        panel.tooltip.hidden = false;
        panel.tooltip.style.transform = 'none';
        // Measure at the origin so an old right-edge position cannot shrink the tooltip.
        panel.tooltip.style.left = '0px';
        panel.tooltip.style.top = '0px';
        const plot = panel.svg.closest('.feature-plot');
        const matrix = panel.svg.getScreenCTM();
        if (plot && matrix) {
          const bounds = plot.getBoundingClientRect();
          const screen = new DOMPoint(point.x, point.y).matrixTransform(matrix);
          const scaleX = bounds.width / plot.offsetWidth || 1;
          const scaleY = bounds.height / plot.offsetHeight || 1;
          const x = (screen.x - bounds.left) / scaleX - plot.clientLeft;
          const y = (screen.y - bounds.top) / scaleY - plot.clientTop;
          const width = panel.tooltip.offsetWidth;
          const height = panel.tooltip.offsetHeight;
          const margin = 8;
          const gap = 12;
          const clamp = (value, maximum) => Math.max(margin, Math.min(maximum - margin, value));
          let left = clamp(x - width / 2, plot.clientWidth - width);
          let top = y - height - gap;
          if (top < margin) {
            top = y + gap;
            if (top + height > plot.clientHeight - margin) {
              top = clamp(y - height / 2, plot.clientHeight - height);
              if (x + gap + width <= plot.clientWidth - margin) left = x + gap;
              else if (x - width - gap >= margin) left = x - width - gap;
              else top = clamp(y > plot.clientHeight / 2 ? y - height - gap : y + gap, plot.clientHeight - height);
            }
          }
          panel.tooltip.style.left = `${left}px`;
          panel.tooltip.style.top = `${top}px`;
        }
      }
    }
    text('#feature-point-readout', active ? `${selected && !hovered ? 'Selected · ' : ''}${description(active)}` : 'Hover or select a point · ← / → to browse · Esc to clear');
  }

  function updateSummary() {
    const missing = encoder.missingGroups.map(id => groupInfo.get(id)?.label || id);
    const note = root.querySelector('#feature-data-note');
    if (note) {
      note.hidden = missing.length === 0;
      note.textContent = `${missing.join(', ')} features are not available for ${encoder.label}. This PCA is fitted to the supplied groups only; no missing samples have been reconstructed.`;
    }
    const variance = encoder.explainedVariance.reduce((sum, value) => sum + value, 0) * 100;
    text('#feature-projection-info', `${encoder.count.toLocaleString()} samples · ${encoder.dimensions}-D features · PC1 + PC2: ${variance.toFixed(1)}% variance`);
    const metric = metrics[encoder.id];
    if (metric) {
      text('#feature-distance-raw', metric.raw.toFixed(4));
      text('#feature-distance-adapted', metric.adapted.toFixed(4));
      text('#feature-distance-reduction', `${metric.reduction.toFixed(2)}%`);
      text('#feature-distance-model', `${encoder.label} · original feature space`);
      const raw = root.querySelector('#feature-bar-raw');
      const adapted = root.querySelector('#feature-bar-adapted');
      if (raw) raw.style.width = '100%';
      if (adapted) adapted.style.width = `${metric.adapted / metric.raw * 100}%`;
    }
  }

  function render() {
    const available = new Set(encoder.groups.map(group => group.id));
    for (const button of encoderButtons) button.setAttribute('aria-pressed', String(button.dataset.featureEncoder === encoder.id));
    for (const button of viewButtons) button.setAttribute('aria-pressed', String(button.dataset.featureView === view));
    for (const button of groupButtons) {
      const id = button.dataset.featureGroup;
      const group = encoder.groups.find(item => item.id === id);
      const count = button.querySelector('[data-feature-count]');
      if (count) count.textContent = group ? `n=${group.count.toLocaleString()}` : 'unavailable';
      button.setAttribute('aria-pressed', String(visible.has(id)));
      button.disabled = !available.has(id) || (visible.has(id) && visible.size === 1);
      button.title = available.has(id) ? `Show or hide ${groupInfo.get(id)?.label || id}` : 'Not available in the supplied feature files';
    }
    densityButton?.setAttribute('aria-pressed', String(showDensity));
    root.querySelector('#feature-panels')?.classList.toggle('is-overlay', view === 'overlay');
    panels.forEach(renderPanel);
    updateSummary();
    updateSelection();
  }

  function nearest(panel, event) {
    const matrix = panel.svg.getScreenCTM();
    if (!matrix) return null;
    const position = new DOMPoint(event.clientX, event.clientY).matrixTransform(matrix.inverse());
    if (position.x < BOX.left || position.x > BOX.right || position.y < BOX.top || position.y > BOX.bottom) return null;
    let result = null;
    let distance = 14 ** 2;
    for (const point of panel.points) {
      const candidate = (point.x - position.x) ** 2 + (point.y - position.y) ** 2;
      if (candidate < distance) {
        result = point;
        distance = candidate;
      }
    }
    return result;
  }

  for (const panel of panels) {
    panel.svg.addEventListener('pointermove', event => {
      const point = nearest(panel, event);
      if (point?.group === hovered?.group && point?.row === hovered?.row) return;
      hovered = point;
      updateSelection();
    });
    panel.svg.addEventListener('pointerleave', () => {
      hovered = null;
      updateSelection();
    });
    panel.svg.addEventListener('click', event => {
      selected = nearest(panel, event);
      hovered = null;
      updateSelection();
      announce(selected ? `${description(selected)} selected.` : 'Point selection cleared.');
    });
    panel.svg.addEventListener('keydown', event => {
      if (!['ArrowLeft', 'ArrowRight', 'Home', 'End', 'Escape'].includes(event.key)) return;
      event.preventDefault();
      hovered = null;
      if (event.key === 'Escape') selected = null;
      else if (panel.points.length) {
        const current = panel.points.findIndex(point => point.group === selected?.group && point.row === selected?.row);
        let index = current < 0 ? 0 : current;
        if (event.key === 'Home') index = 0;
        else if (event.key === 'End') index = panel.points.length - 1;
        else if (current >= 0) index = (current + (event.key === 'ArrowRight' ? 1 : -1) + panel.points.length) % panel.points.length;
        selected = panel.points[index];
      }
      updateSelection();
      announce(selected ? `${description(selected)} selected.` : 'Point selection cleared.');
    });
  }

  for (const button of encoderButtons) button.addEventListener('click', () => {
    const next = encoders.get(button.dataset.featureEncoder);
    if (!next) return;
    encoder = next;
    visible = new Set(encoder.groups.map(group => group.id));
    selected = hovered = null;
    render();
    announce(`${encoder.label} feature distributions. ${encoder.count} supplied samples.`);
  });
  for (const button of viewButtons) button.addEventListener('click', () => {
    view = button.dataset.featureView === 'overlay' ? 'overlay' : 'compare';
    hovered = null;
    render();
    announce(view === 'overlay' ? 'All available distributions overlaid in one plot.' : 'Before and after visual adaptation, shown side by side.');
  });
  for (const button of groupButtons) button.addEventListener('click', () => {
    const id = button.dataset.featureGroup;
    if (!encoder.groups.some(group => group.id === id)) return;
    if (visible.has(id)) {
      if (visible.size === 1) return;
      visible.delete(id);
    } else visible.add(id);
    if (selected?.group === id && !visible.has(id)) selected = null;
    hovered = null;
    render();
    announce(`${groupInfo.get(id).label} ${visible.has(id) ? 'shown' : 'hidden'}.`);
  });
  densityButton?.addEventListener('click', () => {
    showDensity = !showDensity;
    render();
    announce(`Density shading and contours ${showDensity ? 'shown' : 'hidden'}.`);
  });
  window.addEventListener('resize', updateSelection);
  render();
})();
