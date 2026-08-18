# GRAMLANG

Do DNA foundation models learn regulatory grammar, or only regulatory vocabulary?

The working hypothesis here is the billboard model: a model can encode which
transcription factor binding sites are present without encoding how they are
arranged — order, orientation, spacing. If that is what the big sequence models
have learned, then shuffling motifs around inside an enhancer should barely move
their predictions, while changing the motif set should move them a lot.

Testing that turns out to be harder than it sounds, because the obvious
shuffle operator also rewrites the spacer DNA between motifs, and the spacers
carry most of the signal. Most of the code here is about separating those two
things.

## What is in here

`src/` has the library: model wrappers and expression probes (`models/`), the
grammar sensitivity index and its spacer-factored version (`grammar/`), the
sequence perturbation operators (`perturbation/`), and the biophysics,
sequence-design and cross-species analyses (`decomposition/`, `design/`,
`transfer/`).

`scripts/` has the pipelines. `scripts/rebuttal/` holds the corrected second
round of experiments — a spacer-exact arrangement operator, a simulation study
that validates SF-GSI as an estimator, an ablation over readout heads, and an
end-to-end run on PARM which has its own expression head and therefore needs no
probe at all.

`results/` holds the outputs and figures. `manuscript/` holds the paper source.

Models tested: DNABERT-2, Nucleotide Transformer v2, HyenaDNA, Enformer, PARM.
Datasets: Agarwal (K562), Klein (HepG2), Inoue (neural induction), Vaishnav
(yeast), Jores (plant).

## Install

```bash
conda env create -f environment.yml
conda activate gramlang
```

Python 3.10+, PyTorch 2.1+, and a CUDA GPU. FIMO from the MEME suite is
expected under `tools/meme/bin`.

The MPRA datasets are not in the repo. They come from the supplementary
material of Agarwal et al. 2025 (Nature 639:411), Klein et al. 2020 (Nat
Methods 17:1147), Inoue et al. 2019 (Cell Stem Cell 25:713), Vaishnav et al.
2022 (Nature 603:455) and Jores et al. 2021 (Nat Plants 7:842). Once they are
downloaded, `scripts/preprocess_mpra.py` writes the parquet files that
everything else reads out of `data/processed/`.

## Run

```bash
python scripts/train_probes.py --models dnabert2 nt hyenadna \
    --datasets agarwal klein inoue vaishnav jores
python scripts/run_sf_gsi.py --dataset agarwal
python scripts/generate_final_figures.py
```

The corrected census is a separate entry point, since it caches token
embeddings once per (model, dataset) and then scores five readout heads off the
same cache:

```bash
bash scripts/rebuttal/run_all.sh        # train the readout ladder
bash scripts/rebuttal/run_census_all.sh # 15 pairs x 300 enhancers x 5 heads
```

## Results

With a genuinely spacer-exact arrangement operator, arrangement accounts for a
median 15–29% of perturbation variance and spacer composition for 67–76%. Only
a small fraction of enhancers show significant arrangement sensitivity (4–7%,
i.e. about the nominal alpha), and a Storey pi0 on the original 7,650-enhancer
census puts the billboard fraction at 0.906 (bootstrap 95% CI 0.883–0.927,
though it is sensitive to the lambda choice).

Order-aware readout heads predict expression better than the mean-pooled linear
probe — cnn1d gains +0.064 Pearson r on average and wins on 14 of 15 pairs — but
they are not more arrangement-sensitive. Paired Wilcoxon against mean_linear
gives p = 0.31 to 0.71. So the insensitivity is not an artifact of mean pooling,
which is the main thing the first version of this work could not rule out. PARM,
run end to end with no probe and no pooling, agrees: 9.9% arrangement share,
66.4% spacer share.

Two results went against the original story and are worth stating plainly.
Matched-vocabulary MPRA pairs are not more different in expression than random
pairs from the same library, so they do not establish that arrangement matters
biologically. And under a single cross-validation protocol, sequence-derived
grammar features give a negative delta R-squared in all five datasets while
composition features help by +0.018 to +0.250 — grammar features do not merely
fail to add signal, they hurt.

A simulation with a planted-arrangement oracle recovers the true arrangement SD
at r = 0.9995 and fires on 0/180 null cases, so SF-GSI itself is calibrated.
The submitted z-score statistic, by contrast, has a flag rate of 0.03–0.10
regardless of the planted effect — it was answering a different question.

## License

MIT. See LICENSE.
