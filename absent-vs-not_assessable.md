# `absent` vs `not_assessable` in Stage 3 sharded schemas

*Audit date: 2026-07-14. Verified against the live `pxgpt shard-schema` generator
and the on-disk shard sets.*

Two Stage 3 values look similar but mean opposite things:

- **`absent`** (or an ordinal level whose codebook definition denotes absence,
  e.g. `level 0 (none)`) — a **supported, scored value**: the structure is
  genuinely **not present** and *would be visible if it were there*.
- **`not_assessable`** — **"cannot be scored from these images"**: the image set
  does not show the trait (occluded, out of frame, wrong angle), *or* a
  conditional trait's parent feature is absent. **NOT biological absence.**

This distinction comes from the **system prompt** (`system_2_schema.txt` →
`/nas/nas6/PlantGPT/003_BigPlant_All_20260518/prompt_schema/system_2_schema.txt`).
This document verifies that the sharded schemas the model is actually constrained
by are consistent with that distinction, and shows, per trait, where `absent` is
expressible vs. where only `not_assessable` is.

---

## How `pxgpt shard-schema` handles the two values

- **`not_assessable` is appended to every nominal/ordinal enum**, automatically,
  in the generated `shard_NN.schema.json`
  (`pxgpt/core/shard_builder.py` → `value_schema`):
  - nominal → `{"enum": [...master categories..., "not_assessable"]}`
  - ordinal → `{"enum": [...level ids..., "not_assessable"]}`
  - quantitative → `{"type": "string"}` (a number-as-string, or the literal
    `not_assessable`; parsed downstream).
- **`absent` is NOT injected by the generator.** It exists in a shard enum **only
  if the master schema author declared it** for that trait (a nominal `absent`
  category, or an ordinal level defined as absence). The generator preserves
  every master category verbatim and adds nothing but `not_assessable`.

**Consequence:** the absent-vs-`not_assessable` distinction is *expressible* for a
trait **iff its master-schema entry carries an absence token**. Traits without one
can only fall back to `not_assessable`. The lever to change this is the **master
schema**, never `shards_system.md` or the generator.

### Note on `shards_system.md`

`pxgpt shard-schema` also emits `shards_system.md` (the shared preamble / "Block
A"). All audited pipelines run with a `--system-prompt` override
(`system_2_schema.txt`), and `sharding.load_system_prompt` returns the override and
**never reads `shards_system.md`** in that case. So `shards_system.md` is inert for these runs —
its (different, absence-conflating) wording does not reach the API and editing it
changes nothing. See `dispatch_batch_vs_sequential.md` and the Stage 3 section of
`user_manual.md`.

### Audit method

For each pipeline, the master schema is parsed with the generator's own
`normalize_master` / `nominal_categories`, then cross-checked against every
`shard_*.schema.json`: trait coverage (1:1, no dup/missing/extra), `not_assessable`
present in each nominal/ordinal enum, and master categories preserved verbatim.
Absence tokens are detected as a nominal `absent`/`none` category or an ordinal
lowest level whose label/definition matches the whole words
`absent|absence|none|not present|lacking`. The WITH/WITHOUT categorization below is
interpretive (by trait semantics); the integrity checks are mechanical and verified.

**Heuristic caveat.** The detector is deliberately strict: it flags an ordinal only
when its lowest level *reads* as absence (label/definition contains one of those
whole words). It therefore lists some magnitude/intensity ordinals under WITHOUT
even though their lowest level connotes a near-absent floor — e.g. `stem_elongation`
(l1 `compressed`), `leaf_surface_texture` (l1 `smooth`), `leaf_surface_glaucousness`
(l0 `none_glossy`, missed only because the underscore breaks the `none` word
boundary). The model can still *return* those lowest levels; they are just not
counted here as explicit "absent" tokens. (An earlier draft used a looser regex that
also matched a bare `no` inside level definitions, which over-counted these three as
WITH — the numbers below use the strict script in the appendix.)

---

## Audit 1 — `master_schema_v2.json` (mature; `02_MaturePlant/VER_2`)

Shard set: `02_MaturePlant/VER_2/stage_3/shard_master_schema` (10 shards).

### Integrity — all clean ✓

| Check | Result |
|---|---|
| Trait coverage | 49 master → 49 shard, **0 missing, 0 duplicated, 0 extra** |
| `not_assessable` in every nominal/ordinal enum | **all present** |
| Master categories preserved verbatim | **none dropped/changed** |
| Scale-type breakdown | 45 nominal/ordinal, 4 quantitative |

### Absence-token coverage: 20 of 45 nominal/ordinal traits

**WITH an absence token (20)** — `absent`/`none`/level-0 is a scored value:

`plant_axillary_bud_development` (absent), `plant_head_formation` (absent),
`stem_base_anthocyanin` (absent), `stem_leaf_scars` (absent),
`leaf_blade_anthocyanin_coverage` (ord. l0 none), `leaf_heterophylly_presence`
(absent), `leaf_margin_anthocyanin` (absent), `leaf_abaxial_anthocyanin` (absent),
`leaf_vein_anthocyanin` (absent), `petiole_anthocyanin` (absent),
`inflorescence_stage` (ord. l0 absent), `inflorescence_curd_formation` (absent),
`fruit_silique_presence` (absent), `cotyledon_persistence` (absent),
`foliar_senescence` (ord. l0 none), `leaf_interveinal_chlorosis` (absent),
`leaf_necrotic_lesions` (absent), `leaf_variegation` (absent), `leaf_damage_type`
(none), `leaf_damage_extent` (ord. l0 none).

**WITHOUT an absence token (25)** — only `not_assessable` available. Grouped by why:

- *Structural descriptors of an always-present organ (correct to omit — "can't
  see it" = `not_assessable` is the right fallback):* `plant_growth_habit`,
  `plant_branching_habit`, `leaf_phyllotaxy`, `leaf_blade_shape`,
  `leaf_blade_apex_shape`, `leaf_blade_base_shape`, `leaf_blade_curvature`,
  `leaf_margin_type`, `leaf_venation_pattern`, `petiole_cross_section_shape`,
  `stem_surface_texture`, `leaf_surface_pubescence`.
- *Magnitude/intensity ordinals whose lowest level connotes minimal/none but is
  not a literal absence token (see the heuristic caveat under "Audit method"):*
  `stem_elongation` (l1 compressed), `leaf_surface_texture` (l1 smooth),
  `leaf_surface_glaucousness` (l0 none_glossy), `stem_thickness`,
  `leaf_blade_green_intensity`, `petiole_thickness`, `petiole_relative_length`,
  `plant_developmental_stage`.
- *Visibility, not absence (roots hidden in cube ⇒ `not_assessable`):*
  `root_density`, `root_color`, `root_hair_visibility`, `root_colonization_extent`.
- *Conditional / flowering (judgment call):* `flower_petal_color_hue` — no petals
  in vegetative plants ⇒ currently `not_assessable`, which **matches** the system
  prompt's "conditional parent absent ⇒ `not_assessable`" rule. Add an `absent`
  token only if you want "no flowers" recorded as scored absence for petal colour.

---

## Audit 2 — `master_schema_opus4-8_v2.json` (`01_analysis`)

Shard set: `10_MaturePlant_20260518/01_analysis/stage_3_v2/shard_master_schema`
(9 shards).

### Integrity — all clean ✓

| Check | Result |
|---|---|
| Trait coverage | 50 master → 50 shard, **0 missing, 0 duplicated, 0 extra** |
| `not_assessable` in every nominal/ordinal enum | **all present** |
| Master categories preserved verbatim | **none dropped/changed** |
| Scale-type breakdown | 43 nominal/ordinal, 7 quantitative |

### Absence-token coverage: 15 of 43 nominal/ordinal traits

**WITH an absence token (15):** `storage_organ` (absent),
`stem_anthocyanin_extent` (ord. l0 none), `leaf_anthocyanin_extent` (ord. l0
absent), `leaf_variegation` (absent), `reproductive_or_head_structure` (absent),
`root_extrusion` (ord. l0 none), `substrate_algae` (absent),
`substrate_fungal_growth` (absent), `senescence_extent` (ord. l0 none),
`marginal_tipburn_necrosis` (absent), `interveinal_chlorosis` (absent),
`pest_or_disease_signs` (absent), `leaf_disc_sampling_marks` (absent),
`wilting_turgor_loss` (absent), `mechanical_damage` (absent).

**WITHOUT an absence token (28)** — only `not_assessable` available. Grouped:

- *Structural descriptors of an always-present organ:* `plant_growth_habit`,
  `leaf_posture`, `leaf_blade_shape`, `heterophylly`, `leaf_margin_dentition`,
  `leaf_apex_shape`, `leaf_base_shape`, `petiole_form`, `petiole_cross_section`,
  `growing_medium`.
- *Magnitude/stage ordinals:* `internode_elongation`, `leaf_margin_undulation`,
  `midrib_prominence`, `vein_lamina_contrast`, `leaf_glaucousness`,
  `leaf_bullation`, `leaf_pubescence`, `leaf_green_intensity`,
  `developmental_stage`, `bolting_status`.
- *Colour-hue descriptors (a hue exists whenever the organ is visible; "no
  pigment" is carried by the paired `*_extent` trait):* `petiole_color_hue`,
  `vein_color_hue`, `leaf_adaxial_hue`.
- *Conditional on a parent that DOES carry absence (⇒ `not_assessable` when the
  parent is absent, per the system-prompt rule):* `storage_organ_surface_color`
  (parent `storage_organ`), `leaf_anthocyanin_hue` and
  `leaf_anthocyanin_distribution` (parent `leaf_anthocyanin_extent`).
- *Visibility, not absence:* `root_visibility`, `root_color_hue`.

---

## Audit 3 — `master_schema.json` (seedling; `01_Claude_Input_Image/VER_2`)

Shard set: `01_Claude_Input_Image/VER_2/stage_3/shard_master_schema` (7 shards).
This is a **seedling-stage** schema (cotyledon / hypocotyl traits); its run used
`--dispatch sequential` with its own local `--system-prompt system_2_schema.txt`.

### Integrity — all clean ✓

| Check | Result |
|---|---|
| Trait coverage | 42 master → 42 shard, **0 missing, 0 duplicated, 0 extra** |
| `not_assessable` in every nominal/ordinal enum | **all present** |
| Master categories preserved verbatim | **none dropped/changed** |
| Scale-type breakdown | 38 nominal/ordinal, 4 quantitative |

### Absence-token coverage: 9 of 38 nominal/ordinal traits

**WITH an absence token (9):** `hypocotyl_stem_anthocyanin_intensity` (ord. l1
absent), `stem_base_swelling` (absent), `cotyledon_anthocyanin_presence` (absent),
`leaf_blade_anthocyanin_coverage` (ord. l1 absent), `leaf_surface_glaucousness`
(ord. l1 absent), `midrib_anthocyanin_intensity` (ord. l1 absent),
`petiole_anthocyanin_intensity` (ord. l1 absent), `foliar_lesion_presence`
(absent), and `developmental_stage` (ord. l1 `cotyledon_dominant`, defined as
"true leaves absent or just emerging" — a stage floor rather than trait absence;
flagged by the heuristic on its definition text).

**WITHOUT an absence token (29)** — only `not_assessable` available. Grouped:

- *Structural descriptors of an always-present organ:* `plant_growth_habit`,
  `cotyledon_shape`, `leaf_blade_shape`, `leaf_margin_type`, `leaf_apex_shape`,
  `leaf_base_shape`, `leaf_surface_glossiness`, `secondary_vein_pattern`,
  `petiole_form`.
- *Magnitude/stage ordinals:* `leaf_posture`, `hypocotyl_stem_elongation`,
  `cotyledon_persistence_senescence`, `leaf_blade_lobing`,
  `leaf_blade_green_intensity`, `leaf_surface_rugosity`, `leaf_trichome_density`,
  `leaf_trichome_distribution`, `midrib_prominence`, `secondary_vein_prominence`,
  `petiole_relative_length`, `leaf_turgor`, `plant_senescence_degree`.
- *Conditional on a paired/parent trait that carries absence (⇒ `not_assessable`
  when the parent is absent):* `leaf_blade_anthocyanin_intensity` and
  `leaf_blade_anthocyanin_surface` (parent `leaf_blade_anthocyanin_coverage`, whose
  l1 = absent); `flower_color_hue` (parent `reproductive_status` = `vegetative`).
- *Absence encoded via a domain category, not a literal `absent` token:*
  `reproductive_status` (`vegetative` = no reproductive structures present).
- *Visibility, not absence (roots hidden in cube):* `root_density`,
  `root_color_hue`, `root_hair_density`.

---

## Verdict

- **`pxgpt shard-schema` is correct** in all three pipelines: full 1:1 trait
  coverage, `not_assessable` appended to every nominal/ordinal enum, every master
  category preserved verbatim, quantitatives as strings. No defect in the generator
  or the shard folders.

  | Pipeline | master | shard | miss/dup/extra | `not_assessable` | dropped | WITH absence |
  |---|---|---|---|---|---|---|
  | `v2` (mature) | 49 | 49 | 0/0/0 | all | 0 | 20/45 |
  | `opus4-8_v2` (01_analysis) | 50 | 50 | 0/0/0 | all | 0 | 15/43 |
  | `master_schema` (seedling) | 42 | 42 | 0/0/0 | all | 0 | 9/38 |

- **The absent-vs-`not_assessable` distinction is honoured** wherever the master
  schema gives a trait an absence token (20/45 `v2`, 15/43 `opus4-8_v2`, 9/38
  seedling; see the heuristic caveat above regarding a few borderline ordinal
  floors).
  The traits lacking one are almost all cases where absence is not a meaningful
  state (descriptors of always-present organs, magnitude/stage scales, visibility
  issues) or conditional traits whose parent already carries absence — for which
  `not_assessable` is the intended value per `system_2_schema.txt`.
- **`shards_system.md` is irrelevant** to these runs (overridden by
  `--system-prompt`); do not edit it to change behaviour.

### The only lever

To make a trait record absence as a *scored* value, edit the **master schema** —
add a nominal `absent` category (or an ordinal `level 0` defined as "none/absent")
to that trait — then regenerate:

```bash
pxgpt shard-schema --master <master_schema.json> \
                   --shard-dir shard_master_schema --shard-budget 40
```

Re-running this audit after a change confirms coverage:
compare each master trait's `values` against the matching
`shard_*.schema.json` enum (must equal the master categories **+** `not_assessable`).

---

## Appendix — audit script & raw output (lab notebook)

Everything above is reproduced by the single script below. It imports the live
generator (`pxgpt.core.shard_builder`) so it parses each master schema exactly as
`pxgpt shard-schema` does. Recorded here verbatim so the numbers are reproducible.

### A. Environment

```bash
module load miniconda3/3.12.4
source activate pxgpt
cd <repo>/10_MaturePlant_20260518/00_scripts/pxgpt   # so `import pxgpt` resolves
```

### B. Script (`audit_absence_coverage.py`)

```python
#!/usr/bin/env python
"""Audit a Stage 3 master schema against its generated shard schemas.

Checks, per pipeline:
  * trait coverage (1:1 master<->shard, no dup/missing/extra),
  * `not_assessable` present in every nominal/ordinal shard enum,
  * master categories preserved verbatim in the shard enum,
  * absence-token coverage (which traits can express `absent` vs only
    `not_assessable`).

Absence token = a nominal `absent`/`none` category, or an ordinal lowest level
whose label/definition matches absent|absence|none|not present|lacking.

Usage:
  python audit_absence_coverage.py <master_schema.json> <shard_dir> <label>
"""
import sys, json, glob, os, re
from pxgpt.core import shard_builder as sb

ABSENCE = re.compile(r'\b(absent|absence|none|not present|lacking)\b', re.I)
NA = "not_assessable"


def audit(master_path, shard_dir, label):
    master = sb.normalize_master(json.load(open(master_path)))

    mt = {}                                        # trait -> {group, scale, values, abs_tok}
    for g, gobj in master["trait_groups"].items():
        for tr in gobj["traits"]:
            st = tr["scale_type"]
            vals = [v for v, _ in sb.nominal_categories(tr)] if st == "nominal" else \
                   ([l["level"] for l in tr.get("values", [])] if st == "ordinal" else [])
            abs_tok = None
            if st == "nominal":
                for v, _ in sb.nominal_categories(tr):
                    if str(v).strip().lower() in ("absent", "none", "absent/none"):
                        abs_tok = v
                        break
            elif st == "ordinal" and tr.get("values"):
                lo = tr["values"][0]
                if ABSENCE.search(f"{lo.get('label','')} {lo.get('definition','')}"):
                    abs_tok = f"level {lo.get('level')} ({lo.get('label')})"
            mt[tr["trait_name"]] = dict(group=g, scale=st, values=vals, abs_tok=abs_tok)

    se, dup = {}, []                               # trait -> enum list / "STRING"
    shard_files = sorted(glob.glob(os.path.join(shard_dir, "shard_*.schema.json")))
    for sf in shard_files:
        sch = json.load(open(sf))
        for g, gobj in sch.get("properties", {}).items():
            for tname, tobj in gobj.get("properties", {}).items():
                vsv = tobj["properties"]["value"]
                enum = vsv.get("enum", "STRING" if vsv.get("type") == "string" else None)
                if tname in se:
                    dup.append(tname)
                se[tname] = enum

    no_na, dropped = [], []
    for t, m in mt.items():
        e = se.get(t)
        if e is None or e == "STRING":
            continue
        if m["scale"] in ("nominal", "ordinal"):
            if NA not in e:
                no_na.append(t)
            mv = [str(x) for x in m["values"]]
            ev = [str(x) for x in e if x != NA]
            if mv != ev:
                dropped.append((t, mv, ev))

    no = [t for t in mt if mt[t]["scale"] in ("nominal", "ordinal")]
    with_abs = [t for t in mt if mt[t]["abs_tok"]]
    without = [t for t in no if not mt[t]["abs_tok"]]

    print(f"########## {label} ##########")
    print(f"master={master_path}")
    print(f"shards={shard_dir}  ({len(shard_files)} shard files)")
    print(f"coverage: master={len(mt)} shard={len(se)} | missing="
          f"{[t for t in mt if t not in se] or 'none'} extra="
          f"{[t for t in se if t not in mt] or 'none'} dup={dup or 'none'}")
    print(f"not_assessable missing from: {no_na or 'NONE (all enums OK)'}")
    print(f"categories dropped/changed:  {dropped or 'NONE (verbatim)'}")
    print(f"scale mix: nominal/ordinal={len(no)} quantitative="
          f"{len([t for t in mt if mt[t]['scale']=='quantitative'])}")
    print(f"absence tokens: WITH={len(with_abs)} WITHOUT={len(without)} (of {len(no)})")
    print("\nWITH absence token:")
    for t in with_abs:
        print(f"  + {t} [{mt[t]['scale']}] -> {mt[t]['abs_tok']}  ({mt[t]['group']})")
    print("\nWITHOUT absence token (only not_assessable):")
    for t in without:
        print(f"  - {t} [{mt[t]['scale']}]  ({mt[t]['group']})")
    print()


if __name__ == "__main__":
    audit(sys.argv[1], sys.argv[2], sys.argv[3])
```

### C. Invocations

```bash
P=/nfs/project_ssd/project3/pxzhe/PlantGPT
python audit_absence_coverage.py \
  "$P/02_MaturePlant/VER_2/master_schema/master_schema_v2.json" \
  "$P/02_MaturePlant/VER_2/stage_3/shard_master_schema" "Audit 1 — v2 (mature)"
python audit_absence_coverage.py \
  "$P/10_MaturePlant_20260518/01_analysis/master_schema_generation/master_schema_opus4-8_v2.json" \
  "$P/10_MaturePlant_20260518/01_analysis/stage_3_v2/shard_master_schema" "Audit 2 — opus4-8_v2 (01_analysis)"
python audit_absence_coverage.py \
  "$P/01_Claude_Input_Image/VER_2/master_schema/master_schema.json" \
  "$P/01_Claude_Input_Image/VER_2/stage_3/shard_master_schema" "Audit 3 — master_schema (seedling)"
```

### D. Raw output (2026-07-14)

```text
########## Audit 1 — v2 (mature) ##########
master=.../02_MaturePlant/VER_2/master_schema/master_schema_v2.json
shards=.../02_MaturePlant/VER_2/stage_3/shard_master_schema  (10 shard files)
coverage: master=49 shard=49 | missing=none extra=none dup=none
not_assessable missing from: NONE (all enums OK)
categories dropped/changed:  NONE (verbatim)
scale mix: nominal/ordinal=45 quantitative=4
absence tokens: WITH=20 WITHOUT=25 (of 45)

WITH absence token:
  + plant_axillary_bud_development [nominal] -> absent  (whole_plant_architecture)
  + plant_head_formation [nominal] -> absent  (whole_plant_architecture)
  + stem_base_anthocyanin [nominal] -> absent  (stem)
  + stem_leaf_scars [nominal] -> absent  (stem)
  + leaf_blade_anthocyanin_coverage [ordinal] -> level 0 (none)  (leaf_blade)
  + leaf_heterophylly_presence [nominal] -> absent  (leaf_blade)
  + leaf_margin_anthocyanin [nominal] -> absent  (leaf_margin)
  + leaf_abaxial_anthocyanin [nominal] -> absent  (leaf_surface)
  + leaf_vein_anthocyanin [nominal] -> absent  (venation)
  + petiole_anthocyanin [nominal] -> absent  (petiole)
  + inflorescence_stage [ordinal] -> level 0 (absent)  (inflorescence)
  + inflorescence_curd_formation [nominal] -> absent  (inflorescence)
  + fruit_silique_presence [nominal] -> absent  (inflorescence)
  + cotyledon_persistence [nominal] -> absent  (phenology)
  + foliar_senescence [ordinal] -> level 0 (none)  (phenology)
  + leaf_interveinal_chlorosis [nominal] -> absent  (foliar_condition)
  + leaf_necrotic_lesions [nominal] -> absent  (foliar_condition)
  + leaf_variegation [nominal] -> absent  (foliar_condition)
  + leaf_damage_type [nominal] -> none  (leaf_damage)
  + leaf_damage_extent [ordinal] -> level 0 (none)  (leaf_damage)

WITHOUT absence token (only not_assessable):
  - plant_growth_habit [nominal]  (whole_plant_architecture)
  - plant_branching_habit [nominal]  (whole_plant_architecture)
  - leaf_phyllotaxy [nominal]  (whole_plant_architecture)
  - stem_elongation [ordinal]  (stem)
  - stem_surface_texture [nominal]  (stem)
  - stem_thickness [ordinal]  (stem)
  - leaf_blade_shape [nominal]  (leaf_blade)
  - leaf_blade_apex_shape [nominal]  (leaf_blade)
  - leaf_blade_base_shape [nominal]  (leaf_blade)
  - leaf_blade_curvature [nominal]  (leaf_blade)
  - leaf_blade_green_intensity [ordinal]  (leaf_blade)
  - leaf_margin_type [nominal]  (leaf_margin)
  - leaf_surface_texture [ordinal]  (leaf_surface)
  - leaf_surface_glaucousness [ordinal]  (leaf_surface)
  - leaf_surface_pubescence [nominal]  (leaf_surface)
  - leaf_venation_pattern [nominal]  (venation)
  - petiole_thickness [ordinal]  (petiole)
  - petiole_relative_length [ordinal]  (petiole)
  - petiole_cross_section_shape [nominal]  (petiole)
  - flower_petal_color_hue [nominal]  (inflorescence)
  - root_density [ordinal]  (root_system)
  - root_color [nominal]  (root_system)
  - root_hair_visibility [nominal]  (root_system)
  - root_colonization_extent [ordinal]  (root_system)
  - plant_developmental_stage [ordinal]  (phenology)

########## Audit 2 — opus4-8_v2 (01_analysis) ##########
master=.../01_analysis/master_schema_generation/master_schema_opus4-8_v2.json
shards=.../01_analysis/stage_3_v2/shard_master_schema  (9 shard files)
coverage: master=50 shard=50 | missing=none extra=none dup=none
not_assessable missing from: NONE (all enums OK)
categories dropped/changed:  NONE (verbatim)
scale mix: nominal/ordinal=43 quantitative=7
absence tokens: WITH=15 WITHOUT=28 (of 43)

WITH absence token:
  + storage_organ [nominal] -> absent  (whole_plant_architecture)
  + stem_anthocyanin_extent [ordinal] -> level 0 (none)  (stem)
  + leaf_anthocyanin_extent [ordinal] -> level 0 (absent)  (leaf_coloration)
  + leaf_variegation [nominal] -> absent  (leaf_coloration)
  + reproductive_or_head_structure [nominal] -> absent  (phenology)
  + root_extrusion [ordinal] -> level 0 (none)  (root_system)
  + substrate_algae [nominal] -> absent  (substrate)
  + substrate_fungal_growth [nominal] -> absent  (substrate)
  + senescence_extent [ordinal] -> level 0 (none)  (physiological_status)
  + marginal_tipburn_necrosis [nominal] -> absent  (physiological_status)
  + interveinal_chlorosis [nominal] -> absent  (physiological_status)
  + pest_or_disease_signs [nominal] -> absent  (physiological_status)
  + leaf_disc_sampling_marks [nominal] -> absent  (physiological_status)
  + wilting_turgor_loss [nominal] -> absent  (physiological_status)
  + mechanical_damage [nominal] -> absent  (physiological_status)

WITHOUT absence token (only not_assessable):
  - plant_growth_habit [nominal]  (whole_plant_architecture)
  - leaf_posture [nominal]  (whole_plant_architecture)
  - storage_organ_surface_color [nominal]  (whole_plant_architecture)
  - internode_elongation [ordinal]  (stem)
  - leaf_blade_shape [nominal]  (leaf_blade)
  - heterophylly [nominal]  (leaf_blade)
  - leaf_margin_dentition [nominal]  (leaf_margin)
  - leaf_margin_undulation [ordinal]  (leaf_margin)
  - leaf_apex_shape [nominal]  (leaf_apex_base)
  - leaf_base_shape [nominal]  (leaf_apex_base)
  - petiole_form [nominal]  (petiole)
  - petiole_cross_section [nominal]  (petiole)
  - petiole_color_hue [nominal]  (petiole)
  - midrib_prominence [ordinal]  (venation)
  - vein_color_hue [nominal]  (venation)
  - vein_lamina_contrast [ordinal]  (venation)
  - leaf_glaucousness [ordinal]  (leaf_surface)
  - leaf_bullation [ordinal]  (leaf_surface)
  - leaf_pubescence [ordinal]  (leaf_surface)
  - leaf_adaxial_hue [nominal]  (leaf_coloration)
  - leaf_green_intensity [ordinal]  (leaf_coloration)
  - leaf_anthocyanin_hue [nominal]  (leaf_coloration)
  - leaf_anthocyanin_distribution [nominal]  (leaf_coloration)
  - developmental_stage [ordinal]  (phenology)
  - bolting_status [ordinal]  (phenology)
  - root_visibility [nominal]  (root_system)
  - root_color_hue [nominal]  (root_system)
  - growing_medium [nominal]  (substrate)

########## Audit 3 — master_schema (seedling) ##########
master=.../01_Claude_Input_Image/VER_2/master_schema/master_schema.json
shards=.../01_Claude_Input_Image/VER_2/stage_3/shard_master_schema  (7 shard files)
coverage: master=42 shard=42 | missing=none extra=none dup=none
not_assessable missing from: NONE (all enums OK)
categories dropped/changed:  NONE (verbatim)
scale mix: nominal/ordinal=38 quantitative=4
absence tokens: WITH=9 WITHOUT=29 (of 38)

WITH absence token:
  + hypocotyl_stem_anthocyanin_intensity [ordinal] -> level 1 (absent)  (hypocotyl_stem)
  + stem_base_swelling [nominal] -> absent  (hypocotyl_stem)
  + cotyledon_anthocyanin_presence [nominal] -> absent  (cotyledon)
  + leaf_blade_anthocyanin_coverage [ordinal] -> level 1 (absent)  (leaf_blade)
  + leaf_surface_glaucousness [ordinal] -> level 1 (absent)  (leaf_surface)
  + midrib_anthocyanin_intensity [ordinal] -> level 1 (absent)  (venation)
  + petiole_anthocyanin_intensity [ordinal] -> level 1 (absent)  (petiole)
  + developmental_stage [ordinal] -> level 1 (cotyledon_dominant)  (phenology)
  + foliar_lesion_presence [nominal] -> absent  (plant_health)

WITHOUT absence token (only not_assessable):
  - plant_growth_habit [nominal]  (whole_plant_architecture)
  - leaf_posture [ordinal]  (whole_plant_architecture)
  - root_density [ordinal]  (root_system)
  - root_color_hue [nominal]  (root_system)
  - root_hair_density [ordinal]  (root_system)
  - hypocotyl_stem_elongation [ordinal]  (hypocotyl_stem)
  - cotyledon_persistence_senescence [ordinal]  (cotyledon)
  - cotyledon_shape [nominal]  (cotyledon)
  - leaf_blade_shape [nominal]  (leaf_blade)
  - leaf_blade_lobing [ordinal]  (leaf_blade)
  - leaf_blade_green_intensity [ordinal]  (leaf_blade)
  - leaf_blade_anthocyanin_intensity [ordinal]  (leaf_blade)
  - leaf_blade_anthocyanin_surface [nominal]  (leaf_blade)
  - leaf_margin_type [nominal]  (leaf_margin)
  - leaf_apex_shape [nominal]  (leaf_apex_base)
  - leaf_base_shape [nominal]  (leaf_apex_base)
  - leaf_surface_glossiness [nominal]  (leaf_surface)
  - leaf_surface_rugosity [ordinal]  (leaf_surface)
  - leaf_trichome_density [ordinal]  (leaf_surface)
  - leaf_trichome_distribution [ordinal]  (leaf_surface)
  - midrib_prominence [ordinal]  (venation)
  - secondary_vein_pattern [nominal]  (venation)
  - secondary_vein_prominence [ordinal]  (venation)
  - petiole_relative_length [ordinal]  (petiole)
  - petiole_form [nominal]  (petiole)
  - reproductive_status [nominal]  (phenology)
  - flower_color_hue [nominal]  (phenology)
  - leaf_turgor [ordinal]  (plant_health)
  - plant_senescence_degree [ordinal]  (plant_health)
```
