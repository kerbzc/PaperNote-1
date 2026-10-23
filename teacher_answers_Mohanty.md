# Teacher's six-question answers — Mohanty, Hughes & Salathé (2016) — model answer

| Question | Answer, with location |
|---|---|
| Problem | Classify 38 crop–disease classes from single-leaf images collected under controlled conditions (Abstract; Materials & Methods, dataset). Users: growers with smartphones (Introduction). |
| Assumption | That the visual differences between classes in PlantVillage are the disease — not the site, day, camera, or background of each class's collection. Not stated; implied by using within-dataset splits as the test. |
| Evidence | Results: 99.35 % accuracy / F1 0.9934 for GoogLeNet · transfer · colour · 80–20 — supports "high accuracy on held-out PlantVillage images". Discussion: 31.40 % top-1 / 52.89 % top-5 on 121 external images (31.69 % on a second set of 119) — supports "does not transfer as-is". |
| Weakness | All 60 experiments use within-process splits; the only cross-process evidence is 121 + 119 images (about ±8 points). Segmented (background-removed) images still score high, which suggests the leaf carries signal but does not rule out lighting/camera/scale cues that survive segmentation. |
| Falsification | (a) If a background-swap or lighting-swap intervention on PlantVillage test images left accuracy unchanged, "the model relies on the setting" would be weakened. (b) If a PlantVillage-trained model scored well on PlantDoc, the transfer concern would be weakened. |
| Next step | Split PlantVillage by collection site/date (metadata) with a group split and report accuracy — existing data, no new model. Then one intervention at a time on the test images (background, lighting, blur) with a sham control. |

Careful sentence: The reported 99.35 % supports performance under the tested conditions. It does not, by itself, establish performance under a new acquisition process — the paper's own external test (31.4 %) shows that.

What a vague answer looks like (do not write these): "the dataset is not realistic", "the model overfits", "they need more data", "deep learning is a black box".
