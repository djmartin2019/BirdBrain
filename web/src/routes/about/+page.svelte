<h1 class="page-title">About</h1>
<p class="lead">
	Birdbrain is a fine-grained bird species classifier with models trained on CUB-200-2011 and
	iNaturalist 2021 (birds only).
</p>

<section class="card">
	<h2>What it does</h2>
	<p>
		Given a bird photo, a model predicts a species and returns a confidence score plus the top
		five candidates. See the full
		<a href="/species">supported species list</a>. Training uses transfer learning with
		ImageNet-pretrained EfficientNet-B0 backbones and staged fine-tuning.
	</p>
</section>

<section class="card">
	<h2>Models</h2>
	<ul>
		<li>
			<strong>BirdBrain Voyager</strong> — global bird specialist trained on the combined CUB + iNat
			species set; default on the Identify page
		</li>
		<li>
			<strong>EfficientNet-B0 (CUB)</strong> — 200 North American species
		</li>
		<li>
			<strong>ResNet-50 (CUB)</strong> — same 200-species CUB label set with a deeper backbone
		</li>
	</ul>
</section>

<section class="card">
	<h2>Project structure</h2>
	<ul>
		<li><code>training/</code> — YAML-driven training, evaluation, and analysis</li>
		<li><code>web/</code> — this SvelteKit frontend</li>
		<li><code>api/</code> — FastAPI inference service</li>
		<li><code>docs/</code> — technical documentation for the ML pipeline</li>
	</ul>
</section>

<section class="card">
	<h2>Deployment</h2>
	<p>
		Live at <a href="https://birdbrain.djm-apps.com">birdbrain.djm-apps.com</a>. The site calls
		<code>POST /api/predict</code>; training and serving stay separate processes in one monorepo.
	</p>
</section>
