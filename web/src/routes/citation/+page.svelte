<script lang="ts">
	const cubBibtex = `@techreport{WelinderEtal2010,
  Author = {C. Wah and S. Branson and P. Welinder and P. Perona and S. Belongie},
  Title = {Caltech-UCSD Birds-200-2011},
  Institution = {California Institute of Technology},
  Year = {2011},
  Number = {CNS-TR-2011-001}
}`;

	const inatBibtex = `@misc{inaturalist2021,
  author = {Grant Van Horn and macaodha},
  title = {iNat Challenge 2021 - {FGVC8}},
  howpublished = {\\url{https://kaggle.com/competitions/inaturalist-2021}},
  year = {2021},
  publisher = {Kaggle}
}`;
</script>

<h1 class="page-title">Citation</h1>
<p class="lead">
	Birdbrain is for non-commercial research and educational use only. If you use the datasets or
	this project in published work, please cite the sources below.
</p>

<section class="card">
	<h2>CUB-200-2011 dataset</h2>
	<p class="muted">
		EfficientNet-B0 and ResNet-50 CUB models use CUB-200-2011. BirdBrain Voyager uses the
		iNaturalist 2021 birds subset.
	</p>
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
	<h2>iNaturalist 2021 dataset</h2>
	<p class="muted">
		iNat training experiments use the FGVC8 challenge data; Birdbrain filters to birds (Aves) only.
	</p>
	<blockquote>
		<p>
			Grant Van Horn and macaodha. <strong>iNat Challenge 2021 - FGVC8</strong>.<br />
			<a href="https://kaggle.com/competitions/inaturalist-2021">
				https://kaggle.com/competitions/inaturalist-2021
			</a>, 2021. Kaggle.
		</p>
	</blockquote>
	<p class="muted">
		Follow the Kaggle competition and iNaturalist dataset terms. This project uses the data for
		research and educational fine-tuning only—not for commercial products or redistribution of
		the raw images.
	</p>
</section>

<section class="card">
	<h2>BibTeX (CUB)</h2>
	<pre><code>{cubBibtex}</code></pre>
</section>

<section class="card">
	<h2>BibTeX (iNat 2021)</h2>
	<pre><code>{inatBibtex}</code></pre>
</section>

<section class="card">
	<h2>Model weights</h2>
	<p class="muted">
		Checkpoints use ImageNet-pretrained backbones from torchvision (EfficientNet-B0, ResNet50).
		Cite torchvision / the original backbone papers if you publish results that depend on those
		weights.
	</p>
</section>

<p class="muted">
	Full citations and use notes are also in the repository at <code>docs/citation.md</code>.
</p>
