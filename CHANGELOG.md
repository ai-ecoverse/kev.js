# Changelog

## [0.2.0](https://github.com/ai-ecoverse/kev.js/releases/tag/v0.2.0) (2026-09-21)

Match `kev.serve` on the Qwen3.5 checkpoints.

### Features

* apply each checkpoint's fitted pointer-head temperature (0.8B 2.406, 4B 2.144, 9B 2.297); pass `{ temperature: 1 }` for raw logits
* `systemOne(req, { dateFacts: true })` appends day counts between absolute dates (`KEV_DATE_FACTS=1`)
* export writes `temperature` into `manifest.json`; night-2 bundle SHAs get the fitted T without a weight re-export

### Bug Fixes

* parse years 0001–0099 in `dateFacts` the way Python `strptime` does (`Date.UTC` remaps 0–99 to 1900–1999)
