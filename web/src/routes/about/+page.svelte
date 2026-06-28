<h1 class="page-title">About</h1>
<p class="lead">
	Birdbrain is a fine-grained bird species classifier trained on the Caltech-UCSD Birds-200-2011
	(CUB-200) dataset.
</p>

<section class="card">
	<h2>What it does</h2>
	<p>
		Given a bird photo, the model predicts one of 200 North American species and returns a
		confidence score plus the top five candidates. See the full
		<a href="/species">supported species list</a>. Training uses transfer learning with
		ImageNet-pretrained backbones (EfficientNet-B0 and ResNet50) and a five-stage fine-tuning
		pipeline.
	</p>
</section>

<section class="card">
	<h2>Models</h2>
	<ul>
		<li><strong>EfficientNet-B0</strong> — staged checkpoints under <code>birdbrain_v1*.pt</code></li>
		<li><strong>ResNet50</strong> — staged checkpoints under <code>birdbrain_resnet50_v1*.pt</code></li>
	</ul>
	<p class="muted">
		Production deployment will serve a single best checkpoint with matching preprocessing
		(image size, normalization, optional bounding-box crop).
	</p>
</section>

<section class="card">
	<h2>Project structure</h2>
	<ul>
		<li><code>training/</code> — YAML-driven training, evaluation, and analysis</li>
		<li><code>web/</code> — this SvelteKit frontend</li>
		<li><code>api/</code> — planned FastAPI inference service</li>
		<li><code>docs/</code> — technical documentation for the ML pipeline</li>
	</ul>
</section>

<section class="card">
	<h2>Deployment</h2>
	<p>
		Planned home: <a href="https://birdbrain.djm-apps.com">birdbrain.djm-apps.com</a>. The site
		will call a backend <code>POST /api/predict</code> endpoint; training and serving stay
		separate processes in one monorepo.
	</p>
</section>
