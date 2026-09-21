# GitHub Actions policy for GeoCebada

GeoCebada separates lightweight automatic CI from expensive derived-output builds.

## Automatic CI

`.github/workflows/ci.yml` remains automatic on pushes to `main` and on pull requests.

It performs:
- official-tabular schema audit;
- Ruff;
- pytest;
- generated function-index verification;
- Sphinx documentation validation.

It does not build the technical PDF or regenerate the heavy research figures.

## Manual derived-output workflows

The following workflows use only `workflow_dispatch`:

- `.github/workflows/build_agronomic_features.yml`
- `.github/workflows/build_empirical_features.yml`
- `.github/workflows/build_research_outputs.yml`

A normal commit or push does not trigger them.

### Build research outputs

The manual workflow has a `target` choice:

- `figures`
- `paper`
- `all`

The figure path reuses the official project runners:

    python tools/run_checkpoint_04a.py
    python tools/run_checkpoint_04b.py
    python tools/run_checkpoint_04d1.py

These regenerate the repository-backed figures used by the technical report.

Checkpoint 04E.1 is intentionally not rerun in GitHub Actions because its detailed SIAP raw
input lives under the ignored external-data area rather than in the repository checkout. Its
already-versioned aggregate figures/report are included in the figure artifact. Checkpoint
04C.1 is also not rerun by this workflow because the canonical runner is GPU-first and
GitHub-hosted `ubuntu-latest` runners do not provide the project's required GPU execution
environment.

The technical report is compiled with the project's documented command:

    cd reporte/tecnico
    latexmk -pdf -interaction=nonstopmode main.tex

## Artifacts

Artifacts use `actions/upload-artifact@v4`, are retained for 30 days, and include the short
commit SHA in their names.

The figure artifact contains only explicit figure paths plus aggregate reports/summaries. It
does not upload raw source directories, `.env`, credentials, or the full repository.

The report artifact contains:

- `reporte/tecnico/main.pdf`
- `reporte/tecnico/main.log`
- `reporte/tecnico/main.blg`

Feature-build workflows upload only their explicitly named derived feature artifacts and build
metadata. No workflow commits generated outputs back to the repository.

## Exact repository state

Manual workflows check out `${{ github.sha }}`, so outputs correspond to the exact commit
selected when the workflow is launched.

The repository currently has no `.gitattributes` file and therefore no configured Git LFS
tracking. If Git LFS is introduced later for required inputs, checkout must be updated with
`lfs: true`.

## How to run

1. Open the repository on GitHub.
2. Open **Actions**.
3. Select the desired manual workflow.
4. Click **Run workflow**.
5. For **Build research outputs**, choose `figures`, `paper`, or `all`.
6. Choose the branch/ref and run it.
7. When the job finishes, download the named artifact from the workflow run.

No workflow should be run merely to validate that its YAML exists; configuration changes are
reviewed statically first.
