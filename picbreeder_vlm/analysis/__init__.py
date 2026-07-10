"""Metrics: coverage, embeddings, captioning, phylogeny, ratings.

THE THREE EVAL METRICS
======================

The paper reports three metrics over an archive. Each has one paper name and one
code name, and they do not always match: the on-disk artifacts predate the paper's
vocabulary, and the completed runs behind the paper are stored under the old names
both locally and on the Hub. Renaming them would orphan those results, so the
artifact names are frozen and only the Python-side names track the paper.

Use this table to go from a paper name to the code that produces it.

1. SEMANTIC RECALL -- "does the archive contain an image for this word?"
   Mean, over the nouns in a noun list, of each noun's cosine similarity to its
   best-matching archive image (SigLIP2 text-image space). Higher is better.
       produced by : compute_semantic_recall.py  (human: _human_archive.py)
       sweep flag  : eval_semantic_recall=true
       artifact    : noun_similarity_metrics.json
                     noun_similarity_over_time_<nounlist>_<model>.json
       json key    : mean_max_similarity            [FROZEN -- old name]
   The reverse direction (each image's best noun, averaged over images) is also
   recorded as `mean_max_per_image_similarity`. That is a different metric.

2. VISUAL COVERAGE -- "how much of image space does the archive span?"
   Greedy k-center over the IMAGE embeddings; the metric is the k-covering RADIUS,
   i.e. the distance from the worst-covered image to its nearest of k
   representatives. HIGHER is better: a larger radius means k representatives no
   longer suffice to cover the archive tightly, so the archive is more spread out.
   (See figures/metric_figs/visual_coverage.tex: "larger radius => more spread-out
   (visually diverse) archive".) The paper reports k=100.
       produced by : compute_visual_coverage.py    (human: _human_archive.py)
       sweep flag  : eval_visual_coverage=true
       artifact    : embedding_mean_pairwise_distance_over_time_<model>.json
                                                    [FROZEN -- named for the
                                                     secondary stat, not the metric]
       json key    : k_covering_radii  (dict keyed by k, as a string)

3. SEMANTIC COVERAGE -- "how much of meaning space does the archive span?"
   The same greedy k-center construction as Visual Coverage, run over the CAPTION
   embeddings instead of the image embeddings. Higher is better; k=100.
       produced by : caption_and_embed_archive.py  (human: _human.py)
                     -- there is no compute_semantic_coverage.py; the metric falls
                        out of the captioning pass
       sweep flag  : eval_captions=true            [not eval_semantic_coverage]
       artifact    : archive/metrics_<caption_model>_<embed_model>.json
                     caption_metrics_res<size>_<model>.json
       json key    : k_covering_radii

BOTH coverage modules also record `mean_pairwise_distance`. That is a diversity
statistic, NOT Coverage, and the two can move independently -- an archive can grow
its mean pairwise distance while leaving a whole region uncovered. Do not report
one for the other; several call sites historically did.

Internal metric-selector strings (`load_human_baseline(metric=...)` in
sweep_analysis_utils.py) still use the older vocabulary, because they double as
artifact lookups. They map onto the table above as:
    noun_similarity    -> Semantic Recall
    visual_k_covering  -> Visual Coverage
    caption_k_covering -> Semantic Coverage
    caption_diversity  -> caption mean pairwise distance (not a paper metric)
    novelty            -> image mean pairwise distance (not a paper metric)
"""
