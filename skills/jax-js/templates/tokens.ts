// The token boundary.
//
// These templates encode in the application layer and send integer IDs to the
// worker. Both sides validate vocabulary bounds and context length to catch
// malformed tensors and tokenizer/model mismatches. Other apps may tokenize in
// the worker. Tokenization preserves text meaning; it is not a trust boundary
// or a prompt-injection defense.

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
 * Application-layer convenience used by these templates. A worker may own its
 * tokenizer instead; the integer bounds still need validation before one-hot.
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
