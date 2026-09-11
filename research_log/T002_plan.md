# T002 pre-experiment plan

2026-09-12. User explicitly authorized immediate execution of assigned tasks,
including future queue tasks on the 15-minute heartbeat. T002 is IN_PROGRESS.
No agent delegation is requested; implementation and experiment run in this task.

## Reuse and bounded increments

- Keep T001 ISP, functional adapt(), regularization, tests and diagnostics.
- Add frozen Hugging Face Transformers CLIP (OpenAI ViT-B/32) adapter with
  tensor resize/center crop/normalize, generic prompt-bank projection direction.
  Reuse cached revision 3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268.
- Add torchvision Faster R-CNN ResNet50 FPN COCO_V1 eval adapter; use cached
  fasterrcnn_resnet50_fpn_coco-258fb6c6.pth. Analysis-only task loss uses native
  detector loss API with fixed RNG and only loss-branch flags enabled; backbone
  and all weights remain frozen. No annotation imports in deployment code.
- Add deterministic COCO subset, three corruption families/two severities,
  paired gradient/loss diagnostics, and official pycocotools subset bbox AP.
- Run focused tests per increment, tiny real smoke, then full fixed 200-image
  study on A6000. Save every sample and predictions, aggregate all families.

## Protocol fixed before observing scientific results

Seed 20260912; choose 200 IDs uniformly without replacement from sorted COCO
val2017 image IDs, sort chosen IDs for traversal. No category or result filtering.
Clean baseline plus gamma darkening exponents 1.5/2.0, contrast factors 0.6/0.3
about 0.5, red/blue casts gains [1.2,1.0,0.8]/[1.4,1.0,0.6], clipped [0,1].
No optional haze in this first study. Original image resolution retained;
detector uses its native 800/1333 resize, CLIP uses 224 center crop.

Zero phi initialization for every image. Generic text direction: normalized
positive prompt-bank mean minus normalized negative prompt-bank mean; normalize
difference. Same direction for every image, with no corruption metadata input.
Semantic negative displacement projection uses unit-normalized image features.
CLIP-only adaptation: raw SGD lr=0.1, K=1/3, all other loss weights=0.
Oracle one-step diagnostic: raw SGD lr=0.01, fixed native supervised loss,
not a deployable method or a guaranteed mathematical performance upper bound.
The 1-step semantic output is used for the task-loss decrease statistic.
Report relative gradient norms as well as cosines; no zero-vector cosine is
misreported as positive. Keep original T001 hard-clamp and measure saturation.
No prompt/lr/severity tuning based on this validation subset; negative outcomes
are retained. No learned prompts/predictor/source/meta-training.

## Primary sources and provenance

- Model/preprocessing: https://huggingface.co/openai/clip-vit-base-patch32
  and its preprocessor_config.json (224, bicubic, specified RGB mean/std).
- Detector and losses: torchvision 0.19 documentation and v0.19.0 source;
  https://docs.pytorch.org/vision/0.19/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html
- Use installed package APIs, not copied third-party modules.
- Existing remote torch 2.4.0+cu121. Pin torchvision 0.19.0+cu121,
  transformers 4.44.2, pycocotools 2.0.10 in the TAISP venv.

Remote inventory found cached CLIP and detector weights; initial depth-limited
search did not find COCO val annotations. Data will be obtained from official
COCO endpoints if no reusable complete copy is available.
