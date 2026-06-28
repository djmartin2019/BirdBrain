<script lang="ts">
	import '../app.css';
	import favicon from '$lib/assets/favicon.svg';
	import { navLinks } from '$lib/nav';
	import { page } from '$app/state';

	let { children } = $props();
</script>

<svelte:head>
	<link rel="icon" href={favicon} />
	<title>Birdbrain</title>
	<meta
		name="description"
		content="Identify North American bird species from photos using a CUB-200 trained classifier."
	/>
</svelte:head>

<header class="site-header">
	<div class="site-header-inner">
		<a class="brand" href="/">Birdbrain</a>
		<nav aria-label="Main">
			{#each navLinks as link}
				<a href={link.href} aria-current={page.url.pathname === link.href ? 'page' : undefined}>
					{link.label}
				</a>
			{/each}
		</nav>
	</div>
</header>

<main>{@render children()}</main>

<footer class="site-footer">
	<p>Birdbrain — CUB-200-2011 classifier · Non-commercial research use</p>
</footer>
