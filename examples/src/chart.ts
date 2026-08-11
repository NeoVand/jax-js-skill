// A minimal loss chart on canvas: train in --accent, validation in --warm,
// log-scaled y so early progress and late progress are both visible.
//
// Canvas conventions worth copying: cap devicePixelRatio at 2, size from
// clientWidth, reset the transform every frame, and read colours from CSS
// custom properties at draw time so the chart follows light/dark.

const token = (el: HTMLElement, name: string, fallback: string) =>
	getComputedStyle(el).getPropertyValue(name).trim() || fallback;

export function drawLossChart(
	canvas: HTMLCanvasElement,
	train: Array<[number, number]>,
	val: Array<[number, number]>,
	height = 180
) {
	const dpr = Math.min(2, window.devicePixelRatio || 1);
	const W = canvas.clientWidth || 600;
	if (canvas.width !== W * dpr || canvas.height !== height * dpr) {
		canvas.width = W * dpr;
		canvas.height = height * dpr;
		canvas.style.height = `${height}px`;
	}
	const ctx = canvas.getContext('2d')!;
	ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
	ctx.clearRect(0, 0, W, height);

	const all = [...train, ...val];
	if (all.length < 2) return;

	const pad = { l: 34, r: 6, t: 8, b: 16 };
	const xs = all.map((p) => p[0]);
	const ys = all.map((p) => p[1]).filter((v) => Number.isFinite(v) && v > 0);
	if (ys.length < 2) return;
	const x0 = Math.min(...xs);
	const x1 = Math.max(...xs) || 1;
	const lo = Math.log(Math.min(...ys) * 0.9);
	const hi = Math.log(Math.max(...ys) * 1.1);

	const px = (v: number) => pad.l + ((v - x0) / Math.max(1, x1 - x0)) * (W - pad.l - pad.r);
	const py = (v: number) =>
		pad.t + (1 - (Math.log(Math.max(v, 1e-9)) - lo) / Math.max(1e-9, hi - lo)) * (height - pad.t - pad.b);

	// gridlines
	ctx.strokeStyle = token(canvas, '--line', '#e5e2d8');
	ctx.fillStyle = token(canvas, '--ink-3', '#a3a094');
	ctx.font = '10px ui-monospace, monospace';
	ctx.lineWidth = 1;
	for (let i = 0; i <= 3; i++) {
		const v = Math.exp(lo + ((hi - lo) * i) / 3);
		const y = Math.round(py(v)) + 0.5;
		ctx.beginPath();
		ctx.moveTo(pad.l, y);
		ctx.lineTo(W - pad.r, y);
		ctx.stroke();
		ctx.fillText(v.toFixed(2), 2, y + 3);
	}

	const line = (pts: Array<[number, number]>, colour: string, width = 1.5) => {
		if (pts.length < 2) return;
		ctx.strokeStyle = colour;
		ctx.lineWidth = width;
		ctx.beginPath();
		pts.forEach(([s, v], i) => {
			const [X, Y] = [px(s), py(v)];
			i === 0 ? ctx.moveTo(X, Y) : ctx.lineTo(X, Y);
		});
		ctx.stroke();
	};

	line(train, token(canvas, '--accent', '#2b45d8'), 1.2);
	line(val, token(canvas, '--warm', '#d3541f'), 1.8);
	ctx.lineWidth = 1;
}
