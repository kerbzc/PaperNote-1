# Week 5 reading guide — read this first (10 minutes), then the papers

## Why these two papers
Mohanty et al. (2016) showed that a CNN can reach 99.35 % on 54,306 lab images of plant leaves — and, in the same paper,
that the same model reaches about 31 % on 121 images from other sources. PlantDoc (Singh et al., 2020) is a small dataset
of *field* images built to make that gap measurable. Together they are the cleanest published example of
"same protocol vs new protocol", the idea behind every robustness question in this module.

## Paper 1 — Mohanty, Hughes & Salathé (2016). *Using Deep Learning for Image-Based Plant Disease Detection.* Frontiers in Plant Science 7:1419. Open access.
Read (about 40 min):
1. Abstract — find the two numbers: 99.35 % and "just above 31 %".
2. Materials and Methods → Dataset description — underline "collected under controlled conditions". Note: 38 classes, 14 crops.
3. The results table (the 60 experiments: 2 architectures × 2 training modes × 3 image types × 5 splits) — locate the best cell.
4. Discussion, last two paragraphs — the external test (121 and 119 images; top-1 and top-5).
Words you need: held-out / transfer learning / segmented (background removed) / top-5 accuracy / F1.

## Paper 2 — Singh et al. (2020). *PlantDoc: A Dataset for Visual Plant Disease Detection.* CoDS-COMAD. arXiv:1911.10317.
Read (about 45 min):
1. Abstract — 2,598 images, 13 species, 27 classes, ~300 human hours, "up to 31 %".
2. Section on the dataset — how images were collected (internet), curated, and annotated (bounding boxes).
3. The classification results table — which rows train on PlantVillage, which fine-tune on PlantDoc; same architecture?
4. Limitations / discussion — what the authors themselves say is missing.
Words you need: fine-tuning / mAP (detection) / class imbalance / curation.

## How to read (the five steps from class)
1. Rewrite your doubt as a checkable question. 2. Decide where the evidence lives (data → Methods; effect → results table).
3. Find the sentence/table and say what it shows and does not show. 4. Write one alternative explanation. 5. Best answer + smallest next check.
If you use an AI tool to explain a term, go back to the paper and check the sentence it points to.
