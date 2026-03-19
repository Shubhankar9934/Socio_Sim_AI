# JADU API docs (source for MkDocs)

The **canonical source** for this API reference lives under **`docs/jadu-api/`** so it is included when you run **MkDocs** and when **GitHub Actions** deploys to **GitHub Pages**.

- **Edit pages here:** [docs/jadu-api/](../docs/jadu-api/) (same content as before: population, agents, survey, simulation, analytics, evaluation, discovery, calibration).
- **Local preview:** `pip install -r requirements-docs.txt` then `mkdocs serve`
- **After push to `main` / `master`:** workflow [.github/workflows/mkdocs.yml](../.github/workflows/mkdocs.yml) builds and deploys the site (repo **Settings → Pages → GitHub Actions**).

This `docs2/` folder only holds this pointer so older links to `docs2` still explain where the files went.
