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
