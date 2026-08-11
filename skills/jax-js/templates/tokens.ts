// The token boundary.
//
// Nothing but validated integer token IDs crosses into a training worker. Free
// text is encoded in the application layer, checked here, and only then handed
// over — so there is no path from an arbitrary string into model execution, and
// a malformed or out-of-range ID cannot reach nn.oneHot() and corrupt a batch.
//
// Both sides call this: the engine before postMessage, and the worker again on
// receipt. The worker does not trust the main thread — that is the actual trust
// boundary, and re-checking there costs a few microseconds.

/** Thrown when a caller hands over something that is not a token sequence. */
export class InvalidTokensError extends Error {
	constructor(message: string) {
		super(message);
		this.name = 'InvalidTokensError';
	}
}

export interface TokenBounds {
	/** Vocabulary size; valid IDs are integers in [0, vocab). */
	vocab: number;
	/** Hard cap on how many IDs may be accepted. */
	maxLen: number;
}

/**
 * Validate a token sequence, returning a defensive copy.
 *
 * Rejects (rather than silently repairing) anything that is not a finite
 * integer in range: a bad ID means the caller's encoder and the model's
 * vocabulary disagree, and quietly clamping would hide that behind gibberish
 * samples. Over-long input is truncated to the most recent `maxLen` IDs, which
 * is what a context window does anyway.
 */
export function toPromptTokens(ids: unknown, { vocab, maxLen }: TokenBounds): number[] {
	if (!Array.isArray(ids)) {
		throw new InvalidTokensError(`expected an array of token ids, got ${typeof ids}`);
	}
	if (!Number.isInteger(vocab) || vocab <= 0) {
		throw new InvalidTokensError(`bad vocab: ${vocab}`);
	}
	const trimmed = ids.length > maxLen ? ids.slice(-maxLen) : ids;
	const out = new Array<number>(trimmed.length);
	for (let i = 0; i < trimmed.length; i++) {
		const id = trimmed[i];
		if (typeof id !== 'number' || !Number.isInteger(id) || id < 0 || id >= vocab) {
			throw new InvalidTokensError(
				`token ${i} is ${JSON.stringify(id)}; expected an integer in [0, ${vocab})`
			);
		}
		out[i] = id;
	}
	return out;
}

/**
 * Encode free text to token IDs and validate the result in one step.
 *
 * This is the ONLY place a string should become model input. Keeping it in the
 * application layer — never inside the worker or the engine — means the worker's
 * message contract is integers-only, and a static reading of the code finds no
 * route from arbitrary text to model execution.
 */
export function encodePrompt(
	text: string,
	encode: (s: string) => number[],
	bounds: TokenBounds
): number[] {
	if (typeof text !== 'string') {
		throw new InvalidTokensError(`expected a string to encode, got ${typeof text}`);
	}
	return toPromptTokens(encode(text), bounds);
}
