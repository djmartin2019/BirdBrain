<script lang="ts">
	import { SPECIES_COUNT, speciesAlphabetical } from '$lib/species';

	let query = $state('');

	const filtered = $derived(
		query.trim()
			? speciesAlphabetical.filter((name) =>
					name.toLowerCase().includes(query.trim().toLowerCase())
				)
			: speciesAlphabetical
	);
</script>

<h1 class="page-title">Supported species</h1>
<p class="lead">
	Birdbrain classifies photos among {SPECIES_COUNT} bird species from the
	<a href="https://www.vision.caltech.edu/datasets/cub_200_2011/">Caltech-UCSD Birds-200-2011</a>
	dataset. Both production models (EfficientNet-B0 and ResNet-50) were trained on this same label
	set.
</p>

<section class="card">
	<label class="search-label" for="species-search">Search species</label>
	<input
		id="species-search"
		type="search"
		placeholder="e.g. Cardinal, Albatross…"
		bind:value={query}
		autocomplete="off"
	/>
	<p class="count" aria-live="polite">
		Showing {filtered.length} of {SPECIES_COUNT}
	</p>

	<ol class="species-list">
		{#each filtered as name (name)}
			<li>{name}</li>
		{/each}
	</ol>

	{#if filtered.length === 0}
		<p class="muted">No species match your search.</p>
	{/if}
</section>

<p class="muted footer-note">
	Names follow the CUB-200 display convention (folder names with underscores replaced by spaces).
	See the <a href="/citation">citation page</a> to credit the dataset.
</p>

<style>
	.search-label {
		display: block;
		font-weight: 600;
		margin-bottom: 0.5rem;
	}

	input[type='search'] {
		width: 100%;
		padding: 0.6rem 0.75rem;
		border: 1px solid var(--border);
		border-radius: var(--radius);
		background: var(--bg);
		color: var(--text);
		font-size: 1rem;
		margin-bottom: 0.75rem;
	}

	input[type='search']::placeholder {
		color: var(--text-muted);
	}

	.count {
		margin: 0 0 1rem;
		color: var(--text-muted);
		font-size: 0.9rem;
	}

	.species-list {
		margin: 0;
		padding-left: 1.25rem;
		column-count: 2;
		column-gap: 2rem;
	}

	.species-list li {
		break-inside: avoid;
		padding: 0.15rem 0;
	}

	.footer-note {
		margin-top: 0.5rem;
	}

	@media (max-width: 640px) {
		.species-list {
			column-count: 1;
		}
	}
</style>
