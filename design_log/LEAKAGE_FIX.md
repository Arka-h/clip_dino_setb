# Data-leakage episode + fix (honesty record — handover §8/§10)

**Found & fixed** (commit `ee7cafb`, "CASF data: leak-free seeded subset draw + versioned cache").
Recorded here because the project's whole value is reproducibility, and §10 flags holdout leakage as the
prime suspect when OW≈CW — so any leak we hit must be documented, not buried.

## The leak
The original `build_filtered_json` drew the data-fraction subset with an inline unseeded
`np.random.choice(..., replace=True)` and **subset the held-out (unseen / Set-B) set along with the seen
set**. Consequence: at `data_frac < 1`, the unseen-exclusion used a *shrunk* unseen set, so fewer Set-B
images were excluded → **Set-B images leaked into training**. At `data_frac == 1.0` the subset == full set,
so the two are identical — which is exactly why the headline (100%) run was unaffected and the bug stayed
hidden. (This is the §10 trap: a leak that only appears at <100% data.)

## The fix (`casf/subset_select.py`, 28/28 tests, 0 Set-B leak on real COCO)
1. **Seeded + persisted** draw (`np.random.default_rng(subset_seed)`), id-list written to disk.
2. **Without replacement** → exact fraction (no `set()` dedup collapse).
3. **Nested** (0.25 ⊂ 0.50 ⊂ 1.00): permute the full per-class list once, slice a prefix.
4. **Subset SEEN only; exclude the FULL unseen set at every fraction** (`finalize_train_ids`,
   `seen -= unseen` with the complete unseen set). This is the actual leak fix.
5. **Versioned cache** (`SUBSET_LOGIC_VERSION` sidecar): a cache computed by different logic is treated as
   stale and regenerated — a logic change can never be silently served from an old (leaky) cache. The old
   leaky `annotations_casf/*.json` caches were quarantined and regenerate clean. These caches are now
   gitignored (regenerable + historically leaky).

## In-pipeline re-validation (this port, after the fix)
`casf/smoke_data.py` at `data_frac=0.01` (the leaky case): **"no Set-B category selected in train" PASS**
(882 imgs, clean cache `casf_train_open_frac0.01_seed14.json`). seed-14 val picks reproducible; open_qd
⊆ Set B; one train step finite. So our integrated data path is leak-free at <100% too.

## Gate-2 implication
When Gate 2 runs, **OW ≈ CW would still signal a leak** — but the partition itself is now verified clean,
so a Gate-2 OW≈CW would point elsewhere (e.g. encoder sketch-freeness D6, or eval GT/pred coupling), not
the subset draw.
