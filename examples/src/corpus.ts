// A procedurally generated corpus — no downloads, no copyright, and small
// enough that a 2-layer transformer masters it in a few hundred steps on a
// laptop GPU. The grammar is regular enough that you can *read* the model
// learning: first letter frequencies, then words, then word order.

const DET = ['the', 'a', 'that', 'this'];
const ADJ = ['small', 'quiet', 'bright', 'heavy', 'clever', 'restless', 'pale'];
const NOUN = ['machine', 'river', 'gradient', 'window', 'signal', 'garden', 'lantern', 'circuit'];
const VERB = ['learns', 'remembers', 'follows', 'forgets', 'measures', 'answers', 'turns'];
const PREP = ['under', 'beside', 'through', 'against', 'without'];

function mulberry32(seed: number) {
	return function () {
		seed |= 0;
		seed = (seed + 0x6d2b79f5) | 0;
		let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
		t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
		return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
	};
}

export interface Corpus {
	text: string;
	tokens: Uint16Array;
	chars: string[];
	encode: (s: string) => number[];
	decode: (ids: number[]) => string;
}

export function buildCorpus(chars = 120_000, seed = 7): Corpus {
	const rand = mulberry32(seed);
	const pick = <T>(xs: T[]) => xs[Math.floor(rand() * xs.length)];
	const parts: string[] = [];
	let len = 0;
	while (len < chars) {
		const s =
			`${pick(DET)} ${pick(ADJ)} ${pick(NOUN)} ${pick(VERB)} ` +
			(rand() < 0.55 ? `${pick(PREP)} ${pick(DET)} ${pick(NOUN)}. ` : '. ');
		parts.push(s);
		len += s.length;
	}
	const text = parts.join('');

	const set = new Set<string>();
	for (const ch of text) set.add(ch);
	const table = [...set].sort();
	const index = new Map(table.map((c, i) => [c, i]));

	const tokens = new Uint16Array(text.length);
	for (let i = 0; i < text.length; i++) tokens[i] = index.get(text[i]) ?? 0;

	return {
		text,
		tokens,
		chars: table,
		encode: (s) => [...s].map((c) => index.get(c) ?? 0),
		decode: (ids) => ids.map((i) => table[i] ?? '').join('')
	};
}
