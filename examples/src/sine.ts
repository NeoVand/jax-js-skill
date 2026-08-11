// Example 1 — the standalone-lab template, wired to a page.
import { runSineLab, type LabHandle } from '../../skills/jax-js/templates/standalone-lab';

const canvas = document.getElementById('stage') as HTMLCanvasElement;
const out = document.getElementById('out') as HTMLPreElement;
const go = document.getElementById('go') as HTMLButtonElement;
const status = document.getElementById('status') as HTMLSpanElement;

const lines: string[] = [];
const log = (s: string) => {
	lines.push(s);
	out.textContent = lines.slice(-12).join('\n');
	const m = s.match(/step\s+(\d+)\s+loss\s+([\d.]+)/);
	if (m) status.textContent = `step ${m[1]} · loss ${m[2]}`;
};

let lab: LabHandle | null = null;

go.addEventListener('click', () => {
	if (lab) {
		lab.stop();
		lab = null;
		go.textContent = 'Train';
		return;
	}
	lines.length = 0;
	go.textContent = 'Stop';
	lab = runSineLab({ canvas, log, steps: 4000 });
	void lab.done.then(() => {
		lab = null;
		go.textContent = 'Train';
		// e2e marker
		document.body.dataset.labDone = '1';
	});
});

// Auto-start so the page proves itself without a click (and so e2e can watch).
go.click();
