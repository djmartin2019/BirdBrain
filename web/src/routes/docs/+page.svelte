<h1 class="page-title">Documentation</h1>
<p class="lead">
	User guide and pointers to the technical docs in the repository.
</p>

<section class="card">
	<h2>Using this site</h2>
	<ol>
		<li>Go to <a href="/">Identify</a> and upload a bird photo (JPEG or PNG).</li>
		<li>The API returns the top species name, confidence, and top-five alternatives.</li>
		<li>Use confidence as a hint, not a guarantee — fine-grained species are easy to confuse.</li>
	</ol>
	<p class="muted">
		The inference API is under construction. Until it is deployed, uploads will show a connection
		error.
	</p>
</section>

<section class="card">
	<h2>Technical documentation</h2>
	<p>
		The ML pipeline, data splits, training stages, evaluation, and config reference live in the
		repo under <code>docs/</code>:
	</p>
	<ul>
		<li><strong>Overview</strong> — architecture and workflow</li>
		<li><strong>Data &amp; splits</strong> — CUB dataset, train/val/test, augmentations</li>
		<li><strong>Training pipeline</strong> — staged fine-tuning, MLflow</li>
		<li><strong>Models</strong> — architectures, checkpoints, freeze policy</li>
		<li><strong>Evaluation &amp; analysis</strong> — <code>evaluate.py</code>, confusion matrix reports</li>
		<li><strong>Config reference</strong> — YAML fields per stage</li>
		<li><strong>Citation &amp; use</strong> — dataset citations, research/educational terms</li>
	</ul>
	<p class="muted">
		Clone the repository and open <code>docs/README.md</code> for the full index.
	</p>
</section>

<section class="card">
	<h2>Training (local)</h2>
	<pre><code>cd training
python train.py --config ../configs/resnet50_stage1_head.yaml
# … stages 2–5 …
python evaluate.py --config ../configs/resnet50_stage5_bbox.yaml \\
  --checkpoint ../models/birdbrain_resnet50_v1-4.pt --split test</code></pre>
</section>

<section class="card">
	<h2>Frontend development</h2>
	<pre><code>cd web
npm install
npm run dev</code></pre>
	<p class="muted">
		Dev server proxies <code>/api</code> to <code>http://localhost:8000</code> when the FastAPI
		service is running.
	</p>
</section>
