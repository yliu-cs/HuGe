/* HuGe's framework stays readable as HTML; SVG adds the connections. */
(() => {
  'use strict';

  const figure = document.querySelector('#huge-framework');
  if (!figure) return;
  const viewport = figure.querySelector('.hf-viewport');
  const board = figure.querySelector('.hf-board');
  const wires = figure.querySelector('.hf-wires');
  if (!viewport || !board || !wires) return;
  const panButtons = [...figure.querySelectorAll('[data-hf-pan]')];
  const handRouteLabel = figure.querySelector('[data-hf-route="hands"]');

  const svgNS = 'http://www.w3.org/2000/svg';
  const nodes = new Map([...board.querySelectorAll('[data-hf-node]')]
    .map(node => [node.dataset.hfNode, node]));
  const connections = [
    ['robot', 'robot-encoder'],
    ['hybrid', 'hybrid-encoder'],
    ['human', 'human-encoder'],
    ['robot-encoder', 'token', 'curve'],
    ['hybrid-encoder', 'token', 'curve'],
    ['human-encoder', 'token', 'curve'],
    ['token', 'decoder'],
    ['decoder', 'execution'],
    ['human', 'simulation', 'playback', 'state'],
    ['simulation', 'state', 'horizontal', 'state'],
    ['state', 'vlm', 'horizontal', 'state'],
    ['vlm', 'action', 'horizontal'],
    ['action', 'token', 'motion'],
    ['action', 'execution', 'hands', 'hand'],
    ['stage1', 'stage2'],
    ['stage2', 'stage3']
  ].filter(([from, to]) => nodes.has(from) && nodes.has(to));

  function svgElement(tag, attributes = {}) {
    const element = document.createElementNS(svgNS, tag);
    Object.entries(attributes).forEach(([key, value]) => element.setAttribute(key, value));
    return element;
  }

  const definitions = svgElement('defs');
  const colors = { flow: '#317f77', state: '#6486a5', hand: '#ab8152' };
  Object.entries(colors).forEach(([kind, color]) => {
    const marker = svgElement('marker', {
      id: `hf-arrow-${kind}`, viewBox: '0 0 8 8', refX: '7', refY: '4',
      markerWidth: '6', markerHeight: '6', orient: 'auto', markerUnits: 'strokeWidth'
    });
    marker.append(svgElement('path', { d: 'M 1 1 L 7 4 L 1 7', fill: 'none', stroke: color,
      'stroke-width': '1.4', 'stroke-linecap': 'round', 'stroke-linejoin': 'round' }));
    definitions.append(marker);
  });
  wires.replaceChildren(definitions);
  wires.setAttribute('aria-hidden', 'true');
  wires.setAttribute('focusable', 'false');

  const edges = connections.map(([from, to, route, kind = 'flow'], index) => {
    const group = svgElement('g', {
      class: `hf-edge${kind !== 'flow' ? ` hf-edge-${kind}` : ''}`,
      'data-from': from, 'data-to': to
    });
    const line = svgElement('path', {
      class: 'hf-edge-line', fill: 'none', 'marker-end': `url(#hf-arrow-${kind})`
    });
    const pulse = svgElement('path', { class: 'hf-edge-pulse', fill: 'none', pathLength: '1' });
    pulse.style.animationDelay = `${-index * 0.31}s`;
    group.append(line, pulse);
    wires.append(group);
    return { from, to, route, group, line, pulse };
  });

  // Coordinates are in the unzoomed board, even when the viewport is scrolled.
  function bounds(node, origin, scale) {
    const rect = node.getBoundingClientRect();
    return {
      left: (rect.left - origin.left) / scale,
      right: (rect.right - origin.left) / scale,
      top: (rect.top - origin.top) / scale,
      bottom: (rect.bottom - origin.top) / scale,
      x: (rect.left + rect.width / 2 - origin.left) / scale,
      y: (rect.top + rect.height / 2 - origin.top) / scale
    };
  }

  function orthogonalPath(points) {
    let path = `M ${points[0][0]} ${points[0][1]}`;
    for (let i = 1; i < points.length; i += 1) {
      const [x, y] = points[i];
      if (x !== points[i - 1][0]) path += ` H ${x}`;
      if (y !== points[i - 1][1]) path += ` V ${y}`;
    }
    return path;
  }

  function connectionPath(from, to, route, handLaneX) {
    if (route === 'playback') {
      return orthogonalPath([[from.left, from.y], [14, from.y], [14, to.y], [to.left, to.y]]);
    }
    if (route === 'motion') {
      return orthogonalPath([[from.x, from.top], [from.x, 392], [to.x, 392], [to.x, to.bottom]]);
    }
    if (route === 'hands') {
      return orthogonalPath([[from.right, from.y], [handLaneX, from.y], [handLaneX, to.y], [to.right, to.y]]);
    }
    // Policy cards share one centerline; ignore subpixel differences from zoom.
    if (route === 'horizontal' || Math.abs(from.y - to.y) < 1) {
      return `M ${from.right} ${(from.y + to.y) / 2} H ${to.left}`;
    }
    const start = [from.right, from.y];
    const end = [to.left, to.y];
    const middle = (start[0] + end[0]) / 2;
    if (route === 'curve') {
      return `M ${start.join(' ')} C ${middle} ${start[1]}, ${middle} ${end[1]}, ${end.join(' ')}`;
    }
    return orthogonalPath([start, [middle, start[1]], [middle, end[1]], end]);
  }

  let drawRequest = 0;
  function draw() {
    drawRequest = 0;
    const width = board.offsetWidth;
    if (!width || !viewport.clientWidth) return;
    board.style.zoom = String(Math.min(1, Math.max(0.82, viewport.clientWidth / width)));
    const origin = board.getBoundingClientRect();
    const scale = origin.width / width;
    wires.setAttribute('viewBox', `0 0 ${width} ${board.offsetHeight}`);
    const measured = new Map([...nodes].map(([key, node]) => [key, bounds(node, origin, scale)]));
    // Anchor the hand-control route to its label in the reserved right gutter.
    const handLaneX = handRouteLabel ? bounds(handRouteLabel, origin, scale).x : width - 68;
    edges.forEach(edge => {
      const path = connectionPath(measured.get(edge.from), measured.get(edge.to), edge.route, handLaneX);
      edge.line.setAttribute('d', path);
      edge.pulse.setAttribute('d', path);
    });
    updateScroll();
  }
  function scheduleDraw() {
    if (!drawRequest) drawRequest = requestAnimationFrame(draw);
  }
  function updateScroll() {
    const overflowing = viewport.scrollWidth > viewport.clientWidth + 2;
    viewport.classList.toggle('hf-scrollable', overflowing);
    figure.classList.toggle('hf-scrollable', overflowing);
    viewport.classList.toggle('hf-at-end', !overflowing ||
      viewport.scrollLeft + viewport.clientWidth >= viewport.scrollWidth - 4);
    panButtons.forEach(button => {
      button.disabled = !overflowing || (Number(button.dataset.hfPan) < 0
        ? viewport.scrollLeft <= 2
        : viewport.scrollLeft + viewport.clientWidth >= viewport.scrollWidth - 4);
    });
  }
  if ('ResizeObserver' in window) {
    const resizeObserver = new ResizeObserver(scheduleDraw);
    resizeObserver.observe(viewport);
    nodes.forEach(node => resizeObserver.observe(node));
  } else {
    window.addEventListener('resize', scheduleDraw, { passive: true });
  }
  viewport.addEventListener('scroll', updateScroll, { passive: true });
  board.querySelectorAll('img').forEach(img => img.addEventListener('load', scheduleDraw, { once: true }));
  if (document.fonts) document.fonts.ready.then(scheduleDraw);
  scheduleDraw();

  let hovered = null;
  let focused = null;
  let pinned = null;
  function trace() {
    const selected = hovered || focused || pinned;
    const related = new Set(selected ? [selected] : []);
    edges.forEach(edge => {
      const connected = selected !== null && (edge.from === selected || edge.to === selected);
      edge.group.classList.toggle('is-related', connected);
      edge.group.classList.toggle('is-active', connected);
      if (connected) { related.add(edge.from); related.add(edge.to); }
    });
    nodes.forEach((node, key) => {
      node.classList.toggle('is-active', key === selected);
      node.classList.toggle('is-related', related.has(key));
    });
    figure.classList.toggle('hf-tracing', selected !== null);
    handRouteLabel?.classList.toggle('is-related', selected === 'action' || selected === 'execution');
  }
  nodes.forEach((node, key) => {
    node.addEventListener('pointerenter', event => {
      if (event.pointerType !== 'touch') { hovered = key; trace(); }
    });
    node.addEventListener('pointerleave', () => { hovered = null; trace(); });
    node.addEventListener('focusin', () => { focused = key; trace(); });
    node.addEventListener('focusout', event => {
      if (!node.contains(event.relatedTarget)) { focused = null; trace(); }
    });
    node.addEventListener('click', event => {
      if (event.target.closest('button, a, input, select')) return;
      const clearing = pinned === key;
      pinned = clearing ? null : key;
      if (clearing) { hovered = null; focused = null; }
      trace();
    });
    node.addEventListener('keydown', event => {
      if (event.target !== node || !['Enter', ' '].includes(event.key)) return;
      event.preventDefault();
      pinned = pinned === key ? null : key;
      trace();
    });
  });
  figure.addEventListener('keydown', event => {
    if (event.key === 'Escape') {
      hovered = null; focused = null; pinned = null; trace();
    }
  });
  figure.addEventListener('click', event => {
    if (!event.target.closest('[data-hf-node]')) {
      hovered = null; focused = null; pinned = null; trace();
    }
  });

  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  function pan(direction) {
    viewport.scrollBy({ left: direction * viewport.clientWidth * 0.75,
      behavior: reducedMotion.matches ? 'instant' : 'smooth' });
  }
  panButtons.forEach(button => button.addEventListener('click', () => pan(Number(button.dataset.hfPan))));
  viewport.addEventListener('keydown', event => {
    if (event.target !== viewport || !['ArrowLeft', 'ArrowRight'].includes(event.key)) return;
    event.preventDefault();
    pan(event.key === 'ArrowLeft' ? -1 : 1);
  });
  let inView = true;
  function updateMotion() {
    figure.classList.toggle('hf-paused', reducedMotion.matches);
    figure.classList.toggle('hf-offscreen', !inView || document.hidden);
  }
  reducedMotion.addEventListener('change', updateMotion);
  document.addEventListener('visibilitychange', updateMotion);
  if ('IntersectionObserver' in window) {
    const visibilityObserver = new IntersectionObserver(entries => {
      inView = entries[0].isIntersecting;
      updateMotion();
    }, { threshold: 0 });
    visibilityObserver.observe(figure);
  }
  updateMotion();

  const modelDetails = {
    oft: { title: 'HuGe-OFT', conditioning: 'Action-token features', head: 'MLP',
      description: 'Parallel action regression' },
    groot: { title: 'HuGe-GR00T', conditioning: 'Backbone output features', head: 'DiT',
      description: 'Flow matching · cross-attention' },
    pi: { title: 'HuGe-Pi', conditioning: 'Layer-wise backbone features', head: 'DiT',
      description: 'Flow matching · layer-wise conditioning' }
  };
  const modelButtons = [...figure.querySelectorAll('[data-hf-model]')];
  function selectModel(key, announce = false) {
    const model = modelDetails[key];
    if (!model) return;
    modelButtons.forEach(button => button.setAttribute('aria-pressed', String(button.dataset.hfModel === key)));
    Object.entries(model).forEach(([part, content]) => {
      const element = figure.querySelector(`#hf-model-${part}`);
      if (element) element.textContent = content;
    });
    const noise = figure.querySelector('#hf-model-noise');
    if (noise) noise.hidden = key === 'oft';
    const vlm = nodes.get('vlm');
    if (vlm) vlm.dataset.model = key;
    const status = figure.querySelector('#hf-model-status');
    if (announce && status) status.textContent = `${model.title}: ${model.description}. ${model.conditioning} feed the ${model.head} action head.${key === 'oft' ? ' No noise input.' : ' Conditioned on state and noise.'}`;
    scheduleDraw();
  }
  modelButtons.forEach(button => button.addEventListener('click', () => selectModel(button.dataset.hfModel, true)));
  selectModel('pi');
})();
