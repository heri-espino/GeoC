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

The research-output workflow does not rerun scientific checkpoints. Figures and aggregate
summaries already versioned under reports/ are treated as frozen inputs.

The report build entry point is:

    python reporte/tecnico/build.py

The script first validates all LaTeX inputs, frozen figure references, and bibliography keys.
It then invokes the repository's documented LaTeX command:

    latexmk -pdf -interaction=nonstopmode main.tex

For static validation only:

    python reporte/tecnico/build.py --check

The figures target merely packages the existing versioned figures. Regenerating a figure
requires an explicit scientific runner invocation outside this document-build workflow.

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
