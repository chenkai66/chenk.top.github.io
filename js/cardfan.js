// Card-fan carousel — adapted from React Bits (card-fan-carousel) to vanilla + GSAP.
import gsap from 'https://cdn.jsdelivr.net/npm/gsap@3.12.5/+esm';

const MAX_VISIBLE = 7, HALF = 3;
const FAN_POSITIONS = [
  { rot: -21, scale: 0.7756, x: -30, y: 7.3, zIndex: 1 },
  { rot: -14, scale: 0.8498, x: -22, y: 4.0, zIndex: 2 },
  { rot: -7,  scale: 0.9346, x: -11, y: 1.3, zIndex: 3 },
  { rot: 0,   scale: 1.0,    x: 0,   y: 0.0, zIndex: 10 },
  { rot: 7,   scale: 0.9346, x: 11,  y: 1.3, zIndex: 3 },
  { rot: 14,  scale: 0.8498, x: 22,  y: 4.0, zIndex: 2 },
  { rot: 21,  scale: 0.7756, x: 30,  y: 7.3, zIndex: 1 },
];
function getResponsiveMultiplier(w) {
  if (w < 480) return 0.28;
  if (w < 640) return 0.38;
  if (w < 768) return 0.5;
  if (w < 1024) return 0.75;
  return 1.0;
}
function getHeightMultiplier(w) {
  let ideal;
  if (w < 480) ideal = 22 * 16;
  else if (w < 640) ideal = 26 * 16;
  else if (w < 768) ideal = 28 * 16;
  else if (w < 1024) ideal = 34 * 16;
  else ideal = 38 * 16;
  const avail = window.innerHeight * 0.7;
  return avail >= ideal ? 1 : avail / ideal;
}
function getSlotConfig(total, slot) {
  if (total >= MAX_VISIBLE) return FAN_POSITIONS[slot];
  const center = total >> 1;
  const distance = total > 1 ? (slot - center) / center : 0;
  const ad = Math.abs(distance);
  return { rot: distance * 21, scale: 1.0 - 0.2244 * ad * ad, x: distance * 30, y: ad * ad * 7.3, zIndex: 10 - Math.abs(slot - center) };
}

function initCardFan(root) {
  const container = root.querySelector('.fan-layout');
  if (!container) return;
  const cardEls = Array.from(container.querySelectorAll('.fan-card'));
  const totalCards = cardEls.length;
  if (!totalCards) return;

  const needsPagination = totalCards > MAX_VISIBLE;
  let centerIndex = needsPagination ? HALF : totalCards >> 1;
  let isAnimating = false, hasEntered = false, direction = null;
  let prevVisible = new Set();
  let cleanup = null;
  let dots = [];

  function getVisibleMap(center) {
    const map = new Map();
    if (!needsPagination) { cardEls.forEach((_, i) => map.set(i, i)); return map; }
    for (let slot = 0; slot < MAX_VISIBLE; slot++) {
      map.set(((center + slot - HALF) % totalCards + totalCards) % totalCards, slot);
    }
    return map;
  }

  function applyLayout() {
    if (cleanup) { cleanup(); cleanup = null; }
    const visibleMap = getVisibleMap(centerIndex);
    const previouslyVisible = prevVisible;
    const dir = direction;
    const isFirstMount = !hasEntered;
    const multiplier = getResponsiveMultiplier(window.innerWidth);
    const hMult = getHeightMultiplier(window.innerWidth);
    const slotCount = needsPagination ? MAX_VISIBLE : totalCards;
    const config = (slot) => getSlotConfig(slotCount, slot);

    if (isFirstMount) isAnimating = true;
    let completed = 0;
    const visibleCount = visibleMap.size;
    const onDone = () => { if (++completed >= visibleCount) { isAnimating = false; if (isFirstMount) hasEntered = true; } };

    cardEls.forEach((card, ci) => {
      const slot = visibleMap.get(ci);
      const wasVisible = previouslyVisible.has(ci);
      if (slot !== undefined) {
        const { x, y, rot, scale, zIndex } = config(slot);
        const target = { x: `${x * multiplier}rem`, y: `${y * hMult}rem`, rotation: rot, scale, opacity: 1, zIndex };
        if (isFirstMount) {
          gsap.set(card, { x: 0, y: `${12 * hMult}rem`, rotation: 0, scale: 0.5, opacity: 0 });
          gsap.to(card, { ...target, duration: 1.2, ease: 'elastic.out(1.05,.78)', delay: 0.2 + slot * 0.06, onComplete: onDone });
        } else if (!wasVisible) {
          const enterX = dir === 'right' ? 40 : -40;
          gsap.set(card, { x: `${enterX}rem`, y: `${y * hMult}rem`, rotation: dir === 'right' ? 30 : -30, scale: 0.5, opacity: 0 });
          gsap.to(card, { ...target, duration: 0.6, ease: 'power2.out', onComplete: onDone });
        } else {
          gsap.to(card, { ...target, duration: 0.5, ease: 'power2.out', onComplete: onDone });
        }
      } else if (wasVisible) {
        const exitX = dir === 'right' ? -40 : 40;
        gsap.to(card, { x: `${exitX}rem`, opacity: 0, scale: 0.5, rotation: dir === 'right' ? -30 : 30, duration: 0.4, ease: 'power2.in', zIndex: 0 });
      } else if (isFirstMount) {
        gsap.set(card, { opacity: 0, scale: 0.3, x: 0, y: 0, zIndex: 0 });
      }
    });
    prevVisible = new Set(visibleMap.keys());

    // Hover interactions
    const visibleEntries = [];
    cardEls.forEach((el, i) => { const slot = visibleMap.get(i); if (slot !== undefined) visibleEntries.push({ el, slot }); });
    visibleEntries.sort((a, b) => a.slot - b.slot);

    let activeSlot = null, leaveTimer = null;
    const centerSlot = visibleEntries.length >> 1;

    const updateHover = (hoveredSlot) => {
      const mult = getResponsiveMultiplier(window.innerWidth);
      const hM = getHeightMultiplier(window.innerWidth);
      visibleEntries.forEach(({ el, slot }) => {
        const base = config(slot);
        let targetX = base.x * mult, targetY = base.y * hM, targetRot = base.rot, targetScale = base.scale, delay = 0;
        if (hoveredSlot !== null) {
          const distance = Math.abs(slot - hoveredSlot);
          delay = distance * 0.02;
          if (slot === hoveredSlot) {
            targetY -= 2.5 * hM; targetScale *= 1.08;
          } else {
            const normalized = centerSlot > 0 ? (slot - centerSlot) / centerSlot : 0;
            const pushStrength = 8 * (1 - Math.abs(normalized)) * (1 + 0.2 * Math.max(0, 3 - distance));
            if (slot < hoveredSlot) { targetX -= pushStrength * mult; targetRot -= 3 / (distance + 1); }
            else { targetX += pushStrength * mult; targetRot += 3 / (distance + 1); }
            if (slot === visibleEntries.length - 1 && hoveredSlot < centerSlot) targetY -= 1 * hM;
            if (slot === 0 && hoveredSlot > centerSlot) targetY -= 1 * hM;
          }
        } else {
          delay = Math.abs(slot - centerSlot) * 0.02;
        }
        gsap.to(el, { x: `${targetX}rem`, y: `${targetY}rem`, rotation: targetRot, scale: targetScale, duration: 0.5, delay, ease: 'elastic.out(1,.75)', overwrite: 'auto' });
        gsap.set(el, { zIndex: base.zIndex });
      });
    };

    const enterHandlers = visibleEntries.map(({ el, slot }) => {
      const handler = () => {
        if (isAnimating) return;
        if (leaveTimer) { clearTimeout(leaveTimer); leaveTimer = null; }
        if (activeSlot !== slot) { activeSlot = slot; updateHover(slot); }
      };
      el.addEventListener('mouseenter', handler);
      return { el, handler };
    });
    const onLeave = () => {
      if (isAnimating) return;
      if (leaveTimer) clearTimeout(leaveTimer);
      leaveTimer = setTimeout(() => { activeSlot = null; updateHover(null); }, 50);
    };
    container.addEventListener('mouseleave', onLeave);
    const onResize = () => { if (!isAnimating) updateHover(activeSlot); };
    window.addEventListener('resize', onResize);

    cleanup = () => {
      enterHandlers.forEach(({ el, handler }) => el.removeEventListener('mouseenter', handler));
      container.removeEventListener('mouseleave', onLeave);
      window.removeEventListener('resize', onResize);
      if (leaveTimer) clearTimeout(leaveTimer);
    };
  }

  function updateDots() { dots.forEach((d, i) => d.classList.toggle('is-active', i === centerIndex)); }

  function cycle(dir) {
    if (isAnimating || !needsPagination) return;
    isAnimating = true; direction = dir;
    centerIndex = dir === 'right' ? (centerIndex + 1) % totalCards : (centerIndex - 1 + totalCards) % totalCards;
    updateDots();
    applyLayout();
  }

  if (needsPagination) {
    const arrow = (dir, label, pts) => {
      const b = document.createElement('button');
      b.className = 'fan-arrow'; b.type = 'button'; b.setAttribute('aria-label', label);
      b.innerHTML = `<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="${pts}"/></svg>`;
      b.addEventListener('click', () => cycle(dir));
      return b;
    };
    const nav = document.createElement('div');
    nav.className = 'fan-nav';
    const prev = arrow('left', 'Previous', '15 18 9 12 15 6');
    const next = arrow('right', 'Next', '9 18 15 12 9 6');
    const dotsWrap = document.createElement('div');
    dotsWrap.className = 'fan-dots';
    for (let i = 0; i < totalCards; i++) { const d = document.createElement('span'); d.className = 'fan-dot'; dots.push(d); dotsWrap.appendChild(d); }
    nav.append(prev, dotsWrap, next);
    root.appendChild(nav);
    updateDots();
  }

  applyLayout();
}

function boot() {
  const reduce = window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches;
  document.querySelectorAll('.card-fan').forEach(root => {
    if (reduce) { root.classList.add('fan-static'); return; }
    try { initCardFan(root); } catch (e) { console.error('CardFan init failed:', e); root.classList.add('fan-static'); }
  });
}
if (document.readyState !== 'loading') boot();
else document.addEventListener('DOMContentLoaded', boot);
