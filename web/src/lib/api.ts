export type TopKMatch = {
	species: string;
	percent: number;
};

export type PredictionResponse = {
	model_id: string;
	model_name: string;
	prediction: string;
	confidence: number;
	top_5: TopKMatch[];
};

export type ModelInfo = {
	id: string;
	name: string;
	description: string;
	default?: boolean;
	best_val_acc?: number;
};

export type ModelsResponse = {
	models: ModelInfo[];
};

/** Must match BIRDBRAIN_MAX_UPLOAD_MB / nginx client_max_body_size. */
export const MAX_UPLOAD_MB = 10;
export const MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024;

const ALLOWED_IMAGE_TYPES = new Set(['image/jpeg', 'image/png', 'image/webp']);

const API_BASE = import.meta.env.VITE_API_URL ?? '';

export function validateImageFile(file: File): string | null {
	if (!ALLOWED_IMAGE_TYPES.has(file.type)) {
		return 'Please upload a JPEG, PNG, or WebP image.';
	}
	if (file.size > MAX_UPLOAD_BYTES) {
		return `File too large (max ${MAX_UPLOAD_MB} MB)`;
	}
	return null;
}

async function readApiError(response: Response): Promise<string> {
	const text = await response.text();
	const trimmed = text.trim();

	if (trimmed.startsWith('{')) {
		try {
			const body = JSON.parse(trimmed) as { detail?: string | Array<{ msg: string }> };
			if (typeof body.detail === 'string') {
				return body.detail;
			}
			if (Array.isArray(body.detail)) {
				return body.detail.map((item) => item.msg).join('; ');
			}
		} catch {
			// fall through to generic handling
		}
	}

	if (trimmed.startsWith('<')) {
		switch (response.status) {
			case 413:
				return `File too large (max ${MAX_UPLOAD_MB} MB)`;
			case 502:
			case 503:
				return 'The inference service is unavailable. Try again in a moment.';
			default:
				return `Request failed (${response.status})`;
		}
	}

	if (trimmed) {
		return trimmed;
	}

	return `Request failed (${response.status})`;
}

export async function listModels(): Promise<ModelInfo[]> {
	const response = await fetch(`${API_BASE}/api/models`);

	if (!response.ok) {
		throw new Error(await readApiError(response));
	}

	const data: ModelsResponse = await response.json();
	return data.models;
}

export async function predict(image: File, modelId: string): Promise<PredictionResponse> {
	const validationError = validateImageFile(image);
	if (validationError) {
		throw new Error(validationError);
	}

	const form = new FormData();
	form.append('file', image);
	form.append('model_id', modelId);

	const response = await fetch(`${API_BASE}/api/predict`, {
		method: 'POST',
		body: form
	});

	if (!response.ok) {
		throw new Error(await readApiError(response));
	}

	return response.json();
}
