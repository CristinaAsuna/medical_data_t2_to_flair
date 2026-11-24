# medical_data_t2_to_flair

Lesion-aware bridge flow matching for brain MRI T2 → FLAIR synthesis on BraTS 2021, with a lesion-level hallucination evaluation (missed vs. fabricated lesions, measured by an independent segmenter).

Course project for *Medical Big Data and AI* (ShanghaiTech BME), 4 members, ~2 weeks.

## Where to read

| What | File |
| --- | --- |
| Rules for humans and coding agents | `AGENTS.md` |
| Current trunk, the single open question, queue | `STATE.md` |
| All experiments, one line each | `EXPERIMENTS.md` |
| Goal, background, literature, method | `docs/PROJECT.md` |
| Data source, contract, subject-level splits | `docs/DATA.md` |
| Commands: data → train → infer → evaluate | `docs/PIPELINE.md` |
| Decisions and reasons | `docs/DECISIONS.md` |

## Layout

```
src/t2flair/      trunk code (the only package)
configs/          trunk.yaml, exp/<id>.yaml (differences only), splits/
experiments/<id>/ CARD.md, result.json, ≤4 figures
third_party/      verbatim external code, each with UPSTREAM.txt
  research-paper-writing/   paper-writing skill (MIT), basis for the course Skill
tools/            project.py (check/new/audit), env.sh
runs/, scratch/   large outputs and smoke runs, git-ignored
```

## Machines

Planning and coding on the Mac; data processing and training on the BME Slurm cluster. Datasets live in `$LAB_DATA_DIR/brats2021/` on the cluster, never in this repo. Runs are registered with `lab run start`; see `AGENTS.md`.

## Status

Repository scaffolded; no experiment code yet. The trunk card `experiments/2026-11-24-trunk-bridge-fm/CARD.md` awaits approval.
