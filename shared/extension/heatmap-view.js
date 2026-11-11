// Render the model's real spatial attention once, at its native grid resolution.
// Resizing/scrolling changes CSS geometry only; there is no animation/render loop.
(() => {
  const stops = [
    [0, 46, 151, 185], [.3, 62, 191, 169], [.55, 248, 218, 94],
    [.8, 247, 139, 63], [1, 225, 68, 79],
  ];

  function validateMap(map) {
    return Boolean(map && map.kind === 'attention'
      && Number.isInteger(map.width) && map.width > 0 && map.width <= 256
      && Number.isInteger(map.height) && map.height > 0 && map.height <= 256
      && (Array.isArray(map.values) || ArrayBuffer.isView(map.values))
      && map.values.length === map.width * map.height
      && Array.from(map.values).every((value) => Number.isFinite(value) && value >= 0));
  }

  function colorize(map) {
    if (!validateMap(map)) throw new TypeError('A finite, nonnegative spatial attention grid is required.');
    let minimum = Infinity, maximum = 0;
    for (const value of map.values) { minimum = Math.min(minimum, value); maximum = Math.max(maximum, value); }
    const pixels = new Uint8ClampedArray(map.values.length * 4);
    const range = maximum - minimum;
    for (let index = 0; index < map.values.length; index++) {
      // Relative intensity within this map. Uniform/no-focus maps stay clear.
      const strength = range > 0 ? (map.values[index] - minimum) / range : 0;
      const t = Math.pow(strength, .75);
      let upper = 1;
      while (upper < stops.length - 1 && t > stops[upper][0]) upper++;
      const lower = stops[upper - 1], higher = stops[upper];
      const mix = (t - lower[0]) / (higher[0] - lower[0]);
      for (let channel = 0; channel < 3; channel++) pixels[index * 4 + channel] = lower[channel + 1] * (1 - mix) + higher[channel + 1] * mix;
      pixels[index * 4 + 3] = Math.round(158 * t);
    }
    return { width: map.width, height: map.height, pixels, minimum, maximum };
  }

  function position(value, available) {
    if (value === 'left' || value === 'top') return 0;
    if (value === 'right' || value === 'bottom') return available;
    if (value === 'center') return available / 2;
    if (value?.endsWith('%')) return available * parseFloat(value) / 100;
    return Number.parseFloat(value) || 0;
  }

  function createOverlay(image, { visible = true, parent = image.ownerDocument.body, zIndex = 2147483000 } = {}) {
    const doc = image.ownerDocument;
    const win = doc.defaultView;
    const layer = doc.createElement('div');
    const canvas = doc.createElement('canvas');
    layer.dataset.receiptHeatmap = '';
    layer.setAttribute('aria-hidden', 'true');
    Object.assign(layer.style, {
      position: 'fixed', pointerEvents: 'none', overflow: 'hidden', display: 'none',
      zIndex: String(zIndex), margin: '0', padding: '0', border: '0', contain: 'layout paint style',
    });
    Object.assign(canvas.style, {
      position: 'absolute', display: 'block', pointerEvents: 'none', imageRendering: 'auto',
      margin: '0', padding: '0', border: '0', maxWidth: 'none', maxHeight: 'none',
    });
    layer.append(canvas);
    parent.append(layer);
    let hasMap = false, disposed = false, scheduled = 0;

    function refresh() {
      if (disposed) return;
      if (!visible || !hasMap || !image.isConnected || !image.naturalWidth || !image.getClientRects().length) {
        layer.style.display = 'none';
        return;
      }
      const rect = image.getBoundingClientRect();
      const style = win.getComputedStyle(image);
      if (!rect.width || !rect.height || style.visibility === 'hidden' || Number(style.opacity) === 0) {
        layer.style.display = 'none';
        return;
      }
      const scaleX = image.offsetWidth ? rect.width / image.offsetWidth : 1;
      const scaleY = image.offsetHeight ? rect.height / image.offsetHeight : 1;
      const px = (name) => Number.parseFloat(style[name]) || 0;
      const leftInset = (px('borderLeftWidth') + px('paddingLeft')) * scaleX;
      const topInset = (px('borderTopWidth') + px('paddingTop')) * scaleY;
      const width = rect.width - leftInset - (px('borderRightWidth') + px('paddingRight')) * scaleX;
      const height = rect.height - topInset - (px('borderBottomWidth') + px('paddingBottom')) * scaleY;
      let drawnWidth = width, drawnHeight = height;
      const fit = style.objectFit;
      if (fit && fit !== 'fill') {
        let scale = fit === 'cover' ? Math.max(width / image.naturalWidth, height / image.naturalHeight)
          : Math.min(width / image.naturalWidth, height / image.naturalHeight);
        if (fit === 'none') scale = 1;
        if (fit === 'scale-down') scale = Math.min(1, scale);
        drawnWidth = image.naturalWidth * scale;
        drawnHeight = image.naturalHeight * scale;
      }
      const parts = (style.objectPosition || '50% 50%').split(/\s+/);
      const x = position(parts[0], width - drawnWidth);
      const y = position(parts[1] ?? '50%', height - drawnHeight);
      const imageLeft = rect.left + leftInset, imageTop = rect.top + topInset;
      let left = imageLeft, top = imageTop, right = left + width, bottom = top + height;
      // Fixed overlays must respect scrolling/clipping containers around images.
      for (let node = image.parentElement; node && node !== doc.body; node = node.parentElement) {
        const ancestorStyle = win.getComputedStyle(node);
        if (Number(ancestorStyle.opacity) === 0) { right = left; break; }
        const clipX = /hidden|clip|scroll|auto/.test(ancestorStyle.overflowX);
        const clipY = /hidden|clip|scroll|auto/.test(ancestorStyle.overflowY);
        if (!clipX && !clipY) continue;
        const bounds = node.getBoundingClientRect();
        if (clipX) { left = Math.max(left, bounds.left); right = Math.min(right, bounds.right); }
        if (clipY) { top = Math.max(top, bounds.top); bottom = Math.min(bottom, bounds.bottom); }
      }
      Object.assign(layer.style, {
        display: 'block', left: `${left}px`, top: `${top}px`,
        width: `${Math.max(0, right - left)}px`, height: `${Math.max(0, bottom - top)}px`, borderRadius: style.borderRadius,
      });
      Object.assign(canvas.style, { left: `${x + imageLeft - left}px`, top: `${y + imageTop - top}px`, width: `${drawnWidth}px`, height: `${drawnHeight}px` });
    }

    function schedule() {
      if (!disposed && !scheduled && visible && hasMap) scheduled = win.requestAnimationFrame(() => {
        scheduled = 0;
        refresh();
      });
    }
    const observer = new win.ResizeObserver(schedule);
    observer.observe(image);
    image.addEventListener('load', schedule);
    win.addEventListener('resize', schedule, { passive: true });
    win.addEventListener('scroll', schedule, { passive: true, capture: true });

    return {
      element: layer,
      setMap(map) {
        if (disposed) return false;
        hasMap = validateMap(map);
        if (hasMap) {
          const raster = colorize(map);
          canvas.width = raster.width;
          canvas.height = raster.height;
          const context = canvas.getContext('2d');
          // Allocate in the canvas realm. Firefox Xrays reject a foreign backing
          // buffer and TypedArray.set across this boundary; copy scalar pixels.
          const pixels = context.createImageData(raster.width, raster.height);
          for (let index = 0; index < raster.pixels.length; index++) pixels.data[index] = raster.pixels[index];
          context.putImageData(pixels, 0, 0);
          layer.dataset.heatmapKind = map.kind;
        }
        refresh();
        return hasMap;
      },
      setVisible(value) { visible = Boolean(value); refresh(); },
      refresh,
      dispose() {
        if (disposed) return;
        disposed = true;
        observer.disconnect();
        if (scheduled) win.cancelAnimationFrame(scheduled);
        image.removeEventListener('load', schedule);
        win.removeEventListener('resize', schedule);
        win.removeEventListener('scroll', schedule, true);
        layer.remove();
        canvas.width = canvas.height = 0;
      },
    };
  }

  globalThis.RECEIPT_HEATMAP = Object.freeze({ validateMap, colorize, createOverlay });
})();
