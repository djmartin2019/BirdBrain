export type PredictionResponse = {
	prediction: string;
	confidence: number;
	top_5: [string, number][];
};

const API_BASE = import.meta.env.VITE_API_URL ?? '';

export async function predict(image: File): Promise<PredictionResponse> {
	const form = new FormData();
	form.append('file', image);

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
