// The same training plate in React. Two React-specific hazards it defends
// against:
//
//   1. StrictMode double-mounts effects in development. Without the `cancelled`
//      guard you create two engines, they compete for one GPUDevice, and the
//      boot never finishes. This is THE React + WebGPU bug.
//   2. A metrics callback fires every step. setState on each one re-renders
//      hundreds of times a second and becomes the bottleneck — buffer into a
//      ref and flush once per animation frame.

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Engine, detectWebGPU, type ModelConfig, type TrainMetrics } from './engine';

type Phase = 'idle' | 'loading' | 'ready' | 'training' | 'error' | 'no-webgpu';

export interface TrainingPlateProps {
	config: ModelConfig;
	tokenData: Uint16Array;
	/** Steps per burst; an eval follows each one. */
	chunk?: number;
}

export function TrainingPlate({ config, tokenData, chunk = 40 }: TrainingPlateProps) {
	const [phase, setPhase] = useState<Phase>('idle');
	const [loadNote, setLoadNote] = useState('');
	const [errorMsg, setErrorMsg] = useState('');
	const [metrics, setMetrics] = useState({ step: 0, loss: NaN, stepMs: 0 });
	const [, forceCurve] = useState(0);

	// Refs, not state: a Worker handle must never be proxied or re-created.
	const engineRef = useRef<Engine | null>(null);
	const playingRef = useRef(false);
	const trainCurve = useRef<Array<[number, number]>>([]);
	const valCurve = useRef<Array<[number, number]>>([]);
	const pendingRef = useRef<TrainMetrics | null>(null);
	const rafRef = useRef(0);

	/** Coalesce per-step metrics into one render per frame. */
	const pushMetrics = useCallback((m: TrainMetrics) => {
		trainCurve.current.push([m.step, m.loss]);
		pendingRef.current = m;
		if (rafRef.current) return;
		rafRef.current = requestAnimationFrame(() => {
			rafRef.current = 0;
			const p = pendingRef.current;
			if (p) setMetrics({ step: p.step, loss: p.loss, stepMs: p.stepMs });
			forceCurve((n) => n + 1);
		});
	}, []);

	useEffect(() => {
		let cancelled = false; // StrictMode guard — see the note above
		(async () => {
			setPhase('loading');
			setLoadNote('checking for a GPU…');
			try {
				if (!(await detectWebGPU())) {
					if (!cancelled) setPhase('no-webgpu');
					return;
				}
				setLoadNote('building the model on your GPU…');
				const e = new Engine({ tokenData, seed: 42 });
				await e.init(config);
				if (cancelled) {
					void e.dispose(); // hand the device back
					return;
				}
				engineRef.current = e;
				valCurve.current = [[0, await e.valLoss()]];
				setPhase('ready');
			} catch (err) {
				if (cancelled) return;
				setErrorMsg(err instanceof Error ? err.message : String(err));
				setPhase('error');
			}
		})();

		return () => {
			cancelled = true;
			playingRef.current = false;
			cancelAnimationFrame(rafRef.current);
			const e = engineRef.current;
			engineRef.current = null;
			if (e) void e.dispose();
		};
	}, [config, tokenData]);

	const toggle = useCallback(async () => {
		const e = engineRef.current;
		if (!e) return;
		if (playingRef.current) {
			playingRef.current = false;
			await e.stop();
			setPhase('ready');
			return;
		}
		playingRef.current = true;
		setPhase('training');
		while (playingRef.current && engineRef.current) {
			try {
				await e.train(chunk, pushMetrics);
				if (!playingRef.current || !engineRef.current) break;
				valCurve.current.push([trainCurve.current.at(-1)?.[0] ?? 0, await e.valLoss()]);
			} catch (err) {
				playingRef.current = false;
				setErrorMsg(err instanceof Error ? err.message : String(err));
				setPhase('error');
				return;
			}
		}
	}, [chunk, pushMetrics]);

	const uniform = useMemo(() => Math.log(config.vocab), [config.vocab]);
	const canPlay = phase === 'ready' || phase === 'training';

	const path = (points: Array<[number, number]>, w = 600, h = 160) => {
		const all = [...trainCurve.current, ...valCurve.current].map((p) => p[1]).filter((v) => v > 0);
		if (points.length < 2 || all.length < 2) return '';
		const xMax = Math.max(1, ...[...trainCurve.current, ...valCurve.current].map((p) => p[0]));
		const lo = Math.log(Math.min(...all) * 0.9);
		const hi = Math.log(Math.max(...all) * 1.1);
		return points
			.map(([s, v], i) => {
				const x = (s / xMax) * w;
				const y = h - ((Math.log(Math.max(v, 1e-9)) - lo) / (hi - lo)) * h;
				return `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`;
			})
			.join('');
	};

	return (
		<figure className="plate">
			<header>
				<span className="eyebrow">Plate 1 · training</span>
				{canPlay && (
					<button className="primary" onClick={toggle}>
						{phase === 'training' ? 'Pause' : 'Train'}
					</button>
				)}
				<span className="status num">
					{phase === 'loading' && loadNote}
					{phase === 'no-webgpu' && 'needs WebGPU — try Chrome or Edge on desktop'}
					{phase === 'error' && errorMsg}
					{canPlay &&
						(metrics.step > 0
							? `step ${metrics.step} · loss ${metrics.loss.toFixed(3)} nats · ${metrics.stepMs.toFixed(0)} ms/step`
							: 'ready')}
				</span>
			</header>

			<div className="stage">
				{phase === 'no-webgpu' ? (
					<p className="fallback">
						This model trains on your own GPU, and this browser has no WebGPU. Everything below
						describes what it does; the numbers are from a real run.
					</p>
				) : (
					<>
						<svg viewBox="0 0 600 160" preserveAspectRatio="none" role="img"
							aria-label="training loss over steps">
							<path d={path(trainCurve.current)} className="train" />
							<path d={path(valCurve.current)} className="val" />
						</svg>
						<p className="caption">
							Training loss in accent, held-out loss in warm. Knowing nothing would cost{' '}
							{uniform.toFixed(2)} nats per token.
						</p>
					</>
				)}
			</div>
		</figure>
	);
}
