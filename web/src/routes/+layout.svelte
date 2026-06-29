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
		content="Identify bird species from photos. BirdBrain Voyager covers 1,535 combined species; CUB models cover 200 North American species."
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
	<p>Birdbrain — bird species classifier · Non-commercial research use</p>
</footer>
