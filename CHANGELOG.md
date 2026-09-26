# [0.6.0](https://github.com/ai-ecoverse/kev.js/compare/v0.5.1...v0.6.0) (2026-09-26)


### Bug Fixes

* **test:** floor q8 max |dp| bound for cross-CPU variance ([b7b36c9](https://github.com/ai-ecoverse/kev.js/commit/b7b36c90a6e0329c7834e9ddc6396a94eafe5c69))


### Features

* **export:** ship 64k-ready graphs (SERVE_MAX_BRANCH rope) ([b37154b](https://github.com/ai-ecoverse/kev.js/commit/b37154b2509986bbac56d7b1ba1c4ee293c09a7c))

## [0.5.1](https://github.com/ai-ecoverse/kev.js/compare/v0.5.0...v0.5.1) (2026-09-26)


### Bug Fixes

* **api:** Score confidence uses TypeSafe's uniform-MAD normaliser ([3c0725e](https://github.com/ai-ecoverse/kev.js/commit/3c0725e86b3ade668a86c9857f84a7eda665e7db)), closes [jaredpalmer/kev#95](https://github.com/jaredpalmer/kev/issues/95) [#139](https://github.com/ai-ecoverse/kev.js/issues/139)

# [0.5.0](https://github.com/ai-ecoverse/kev.js/compare/v0.4.0...v0.5.0) (2026-09-24)


### Bug Fixes

* cache bundles by content, reject requests past the rotary tables ([8d36d8e](https://github.com/ai-ecoverse/kev.js/commit/8d36d8ee22e58f750db2324b97ec0d2a4e460660))
* **export:** leave temperature out of kev.json when head.pt has none ([bd01806](https://github.com/ai-ecoverse/kev.js/commit/bd01806624c022e33655be4b3b64df5f16dca71a))
* **export:** upload the vision tower's files with -vision bundles ([a75ea84](https://github.com/ai-ecoverse/kev.js/commit/a75ea8481f541652145c71cabf7efe809a2ca67e))
* **vision:** resize from the RGBA bytes; keep image-token counts on cache hits ([7fb3d49](https://github.com/ai-ecoverse/kev.js/commit/7fb3d49e0ba019f6a9e6d174f4e65e00fac2b5ad))


### Features

* **vision:** image input on WebGPU through Qwen3.5's stock vision tower ([920f061](https://github.com/ai-ecoverse/kev.js/commit/920f061ccfd217492d849aeaabd016cac9b51739))
* **vision:** Kev behind Qwen3.5's stock vision tower, PyTorch feasibility ([70c6c5b](https://github.com/ai-ecoverse/kev.js/commit/70c6c5bf0d315a4d3044d7d337de16fca9b429d0))

# [0.4.0](https://github.com/ai-ecoverse/kev.js/compare/v0.3.0...v0.4.0) (2026-09-23)


### Features

* **load:** load a bundle from a directory handle or a read function ([831fe2b](https://github.com/ai-ecoverse/kev.js/commit/831fe2b9a7f65d72fb978df335ba7d8ef126964e)), closes [ai-ecoverse/skills#423](https://github.com/ai-ecoverse/skills/issues/423)

# [0.3.0](https://github.com/ai-ecoverse/kev.js/compare/v0.2.0...v0.3.0) (2026-09-23)


### Features

* **api:** follow kev 557598f: 4-decimal probabilities, optional instructions, one-level scores ([bd3da87](https://github.com/ai-ecoverse/kev.js/commit/bd3da87e56dc4fa73901fce0b3b97d67d73e0608)), closes [jaredpalmer/kev#45](https://github.com/jaredpalmer/kev/issues/45) [#50](https://github.com/ai-ecoverse/kev.js/issues/50)

# Changelog

## [0.2.0](https://github.com/ai-ecoverse/kev.js/releases/tag/v0.2.0) (2026-09-21)

Match `kev.serve` on the Qwen3.5 checkpoints.

### Features

* apply each checkpoint's fitted pointer-head temperature (0.8B 2.406, 4B 2.144, 9B 2.297); pass `{ temperature: 1 }` for raw logits
* `systemOne(req, { dateFacts: true })` appends day counts between absolute dates (`KEV_DATE_FACTS=1`)
* export writes `temperature` into `manifest.json`; night-2 bundle SHAs get the fitted T without a weight re-export

### Bug Fixes

* parse years 0001–0099 in `dateFacts` the way Python `strptime` does (`Date.UTC` remaps 0–99 to 1900–1999)
