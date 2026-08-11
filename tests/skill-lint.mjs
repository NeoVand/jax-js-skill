// Structural checks on the skill itself: frontmatter the `skills` CLI needs,
// every internal link resolving, every referenced template existing, and no
// file so long that an agent will skim it.
//
//   node tests/skill-lint.mjs

import { readFileSync, readdirSync, statSync, existsSync } from 'node:fs';
import { join, dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const skillDir = join(root, 'skills', 'jax-js');

const problems = [];
const fail = (m) => problems.push(m);
const ok = [];

// ── frontmatter ─────────────────────────────────────────────────────────────
const skillPath = join(skillDir, 'SKILL.md');
const skill = readFileSync(skillPath, 'utf8');

const fm = skill.match(/^---\n([\s\S]*?)\n---\n/);
if (!fm) fail('SKILL.md has no YAML frontmatter');
else {
	const body = fm[1];
	const name = body.match(/^name:\s*(.+)$/m)?.[1]?.trim();
	const desc = body.match(/^description:\s*([\s\S]*?)(?=\n\w+:|$)/m)?.[1]?.trim();

	if (!name) fail('frontmatter: `name` is required');
	else if (!/^[a-z0-9][a-z0-9-]*$/.test(name))
		fail(`frontmatter: name "${name}" must be lowercase letters, digits and hyphens`);
	else ok.push(`name: ${name}`);

	if (!desc) fail('frontmatter: `description` is required');
	else {
		const len = desc.length;
		if (len < 60) fail(`description is only ${len} chars — too vague to trigger reliably`);
		else if (len > 1024) fail(`description is ${len} chars — over the 1024 limit`);
		else ok.push(`description: ${len} chars`);
		// The description is the ONLY thing an agent sees when deciding to load
		// the skill. It must contain the literal package names.
		for (const term of ['jax-js', '@jax-js/jax', 'WebGPU', 'browser']) {
			if (!desc.includes(term)) fail(`description should mention "${term}" — it is a trigger term`);
		}
	}
}

// ── size ────────────────────────────────────────────────────────────────────
const lines = (p) => readFileSync(p, 'utf8').split('\n').length;
const skillLines = lines(skillPath);
if (skillLines > 500) fail(`SKILL.md is ${skillLines} lines — keep it under 500; move detail to references/`);
else ok.push(`SKILL.md: ${skillLines} lines`);

for (const f of readdirSync(join(skillDir, 'references'))) {
	const n = lines(join(skillDir, 'references', f));
	if (n > 600) fail(`references/${f} is ${n} lines — split it`);
}

// ── every internal link resolves ────────────────────────────────────────────
const mdFiles = [
	skillPath,
	...readdirSync(join(skillDir, 'references')).map((f) => join(skillDir, 'references', f))
];

for (const file of mdFiles) {
	const text = readFileSync(file, 'utf8');
	for (const m of text.matchAll(/\]\((?!https?:|#|mailto:)([^)#]+)(#[^)]*)?\)/g)) {
		const target = resolve(dirname(file), m[1]);
		if (!existsSync(target)) fail(`${file.replace(root + '/', '')}: broken link → ${m[1]}`);
	}
}

// ── every template and script named in SKILL.md exists ──────────────────────
for (const m of skill.matchAll(/`(templates\/[\w.-]+|scripts\/[\w.-]+)`/g)) {
	const p = join(skillDir, m[1]);
	if (!existsSync(p)) fail(`SKILL.md names ${m[1]} but it does not exist`);
}

// ── templates are non-trivial and carry a why-comment ───────────────────────
for (const f of readdirSync(join(skillDir, 'templates'))) {
	const p = join(skillDir, 'templates', f);
	const text = readFileSync(p, 'utf8');
	if (statSync(p).size < 400) fail(`templates/${f} is suspiciously small`);
	if (!/^(\/\/|<!--)/.test(text.trimStart()) && !text.trimStart().startsWith('<script'))
		fail(`templates/${f} should open with a comment explaining what it is`);
}
ok.push(`${readdirSync(join(skillDir, 'templates')).length} templates`);
ok.push(`${readdirSync(join(skillDir, 'references')).length} references`);

// ── the laws are actually numbered and referenced ───────────────────────────
for (let i = 1; i <= 5; i++) {
	if (!new RegExp(`### ${i}\\.`).test(skill)) fail(`SKILL.md is missing law ${i}`);
}

// ── report ──────────────────────────────────────────────────────────────────
for (const line of ok) console.log(`  ✓ ${line}`);
if (problems.length === 0) {
	console.log('\nskill-lint: clean');
	process.exit(0);
}
console.log('');
for (const p of problems) console.log(`  ✗ ${p}`);
console.log(`\nskill-lint: ${problems.length} problem(s)`);
process.exit(1);
