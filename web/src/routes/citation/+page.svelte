<script lang="ts">
	const bibtex = `@techreport{WelinderEtal2010,
  Author = {C. Wah and S. Branson and P. Welinder and P. Perona and S. Belongie},
  Title = {Caltech-UCSD Birds-200-2011},
  Institution = {California Institute of Technology},
  Year = {2011},
  Number = {CNS-TR-2011-001}
}`;
</script>

<h1 class="page-title">Citation</h1>
<p class="lead">
	Birdbrain is trained on the Caltech-UCSD Birds-200-2011 dataset. If you use the dataset or
	this project in published work, please cite the sources below.
</p>

<section class="card">
	<h2>CUB-200-2011 dataset</h2>
	<blockquote>
		<p>
			<strong>The Caltech-UCSD Birds-200-2011 Dataset</strong><br />
			Wah, C.; Branson, S.; Welinder, P.; Perona, P.; Belongie, S.<br />
			California Institute of Technology, 2011.<br />
			Technical Report CNS-TR-2011-001
		</p>
	</blockquote>
	<p>
		Dataset home:
		<a href="https://www.vision.caltech.edu/datasets/cub_200_2011/">
			https://www.vision.caltech.edu/datasets/cub_200_2011/
		</a>
	</p>
	<p class="muted">
		CUB images are restricted to non-commercial research and educational use. See the official
		dataset website for terms.
	</p>
</section>

<section class="card">
	<h2>BibTeX (dataset)</h2>
	<pre><code>{bibtex}</code></pre>
</section>

<section class="card">
	<h2>Model weights</h2>
	<p class="muted">
		Checkpoints use ImageNet-pretrained backbones from torchvision (EfficientNet-B0, ResNet50).
		Cite torchvision / the original backbone papers if you publish results that depend on those
		weights.
	</p>
</section>
