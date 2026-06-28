<script lang="ts">
	import { predict, type PredictionResponse } from '$lib/api';

	let file = $state<File | null>(null);
	let previewUrl = $state<string | null>(null);
	let loading = $state(false);
	let error = $state<string | null>(null);
	let result = $state<PredictionResponse | null>(null);

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
		if (!file) return;

		loading = true;
		error = null;
		result = null;

		try {
			result = await predict(file);
		} catch (err) {
			error = err instanceof Error ? err.message : 'Prediction failed';
		} finally {
			loading = false;
		}
	}
</script>

<h1 class="page-title">Identify a bird</h1>
<p class="lead">
	Upload a photo to classify it among 200 CUB-200 species. The inference API is not wired up
	yet — this page is ready for when <code>api/</code> is deployed.
</p>

<form class="card" onsubmit={onSubmit}>
	<h2>Upload</h2>
	<input type="file" accept="image/*" onchange={onFileChange} />
	{#if previewUrl}
		<img src={previewUrl} alt="Uploaded bird preview" style="max-width: 100%; border-radius: 0.5rem;" />
	{/if}
	<button type="submit" disabled={!file || loading}>
		{loading ? 'Identifying…' : 'Identify species'}
	</button>
	{#if error}
		<p class="error">{error}</p>
	{/if}
</form>

{#if result}
	<section class="card results" aria-live="polite">
		<h2>Results</h2>
		<dl>
			<dt>Top prediction</dt>
			<dd>{result.prediction} ({(result.confidence * 100).toFixed(1)}%)</dd>
		</dl>
		<h3>Top 5</h3>
		<ol>
			{#each result.top_5 as [species, score]}
				<li>{species} — {(score * 100).toFixed(1)}%</li>
			{/each}
		</ol>
	</section>
{/if}
