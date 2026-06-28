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

const API_BASE = import.meta.env.VITE_API_URL ?? '';

export async function listModels(): Promise<ModelInfo[]> {
	const response = await fetch(`${API_BASE}/api/models`);

	if (!response.ok) {
		const detail = await response.text();
		throw new Error(detail || `Request failed (${response.status})`);
	}

	const data: ModelsResponse = await response.json();
	return data.models;
}

export async function predict(image: File, modelId: string): Promise<PredictionResponse> {
	const form = new FormData();
	form.append('file', image);
	form.append('model_id', modelId);

	const response = await fetch(`${API_BASE}/api/predict`, {
		method: 'POST',
		body: form
	});

	if (!response.ok) {
		const detail = await response.text();
		throw new Error(detail || `Request failed (${response.status})`);
	}

	return response.json();
}
