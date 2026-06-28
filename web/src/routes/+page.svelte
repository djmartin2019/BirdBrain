<script lang="ts">
	import { onMount } from 'svelte';
	import { listModels, predict, type ModelInfo, type PredictionResponse } from '$lib/api';

	let file = $state<File | null>(null);
	let previewUrl = $state<string | null>(null);
	let loading = $state(false);
	let modelsLoading = $state(true);
	let error = $state<string | null>(null);
	let modelsError = $state<string | null>(null);
	let models = $state<ModelInfo[]>([]);
	let selectedModelId = $state<string | null>(null);
	let result = $state<PredictionResponse | null>(null);

	onMount(async () => {
		try {
			models = await listModels();
			const defaultModel = models.find((m) => m.default) ?? models[0];
			selectedModelId = defaultModel?.id ?? null;
		} catch (err) {
			modelsError = err instanceof Error ? err.message : 'Failed to load models';
		} finally {
			modelsLoading = false;
		}
	});

	function onFileChange(event: Event) {
		const input = event.target as HTMLInputElement;
		const selected = input.files?.[0] ?? null;

		if (previewUrl) {
			URL.revokeObjectURL(previewUrl);
		}

		file = selected;
		result = null;
		error = null;
		previewUrl = selected ? URL.createObjectURL(selected) : null;
	}

	async function onSubmit(event: Event) {
		event.preventDefault();
		if (!file || !selectedModelId) return;

		loading = true;
		error = null;
		result = null;

		try {
			result = await predict(file, selectedModelId);
		} catch (err) {
			error = err instanceof Error ? err.message : 'Prediction failed';
		} finally {
			loading = false;
		}
	}
</script>

<h1 class="page-title">Identify a bird</h1>
<p class="lead">
	Upload a photo to classify it among 200 CUB-200 species. Choose a model, then submit your
	image for top-5 predictions.
</p>

<form class="card" onsubmit={onSubmit}>
	<h2>Model</h2>
	{#if modelsLoading}
		<p>Loading models…</p>
	{:else if modelsError}
		<p class="error">{modelsError}</p>
	{:else if models.length === 0}
		<p class="error">No models available. Start the API with checkpoints in place.</p>
	{:else}
		<fieldset class="model-picker">
			<legend class="sr-only">Choose a model</legend>
			{#each models as model (model.id)}
				<label class="model-option">
					<input
						type="radio"
						name="model"
						value={model.id}
						checked={selectedModelId === model.id}
						onchange={() => (selectedModelId = model.id)}
					/>
					<span>
						<strong>{model.name}</strong>
						{#if model.description}
							<small>{model.description}</small>
						{/if}
					</span>
				</label>
			{/each}
		</fieldset>
	{/if}

	<h2>Upload</h2>
	<input type="file" accept="image/*" onchange={onFileChange} />
	{#if previewUrl}
		<img src={previewUrl} alt="Uploaded bird preview" style="max-width: 100%; border-radius: 0.5rem;" />
	{/if}
	<button type="submit" disabled={!file || !selectedModelId || loading || models.length === 0}>
		{loading ? 'Identifying…' : 'Identify species'}
	</button>
	{#if error}
		<p class="error">{error}</p>
	{/if}
</form>

{#if result}
	<section class="card results" aria-live="polite">
		<h2>Results</h2>
		<p class="model-used">Model: {result.model_name}</p>
		<dl>
			<dt>Top prediction</dt>
			<dd>{result.prediction} ({result.confidence.toFixed(1)}%)</dd>
		</dl>
		<h3>Top 5</h3>
		<ol>
			{#each result.top_5 as match}
				<li>{match.species} — {match.percent.toFixed(1)}%</li>
			{/each}
		</ol>
	</section>
{/if}

<style>
	.model-picker {
		border: none;
		padding: 0;
		margin: 0 0 1rem;
		display: flex;
		flex-direction: column;
		gap: 0.75rem;
	}

	.model-option {
		display: flex;
		align-items: flex-start;
		gap: 0.5rem;
		cursor: pointer;
	}

	.model-option small {
		display: block;
		color: var(--muted);
		margin-top: 0.15rem;
	}

	.model-used {
		color: var(--muted);
		margin: 0 0 1rem;
		font-size: 0.9rem;
	}

	.sr-only {
		position: absolute;
		width: 1px;
		height: 1px;
		padding: 0;
		margin: -1px;
		overflow: hidden;
		clip: rect(0, 0, 0, 0);
		white-space: nowrap;
		border: 0;
	}
</style>
