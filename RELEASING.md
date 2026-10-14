# Releasing

The workspace version in `Cargo.toml` is always the next release. Releasing
publishes that version and then bumps to the one after it. Versioning rules are
in [STABILITY.md](STABILITY.md).

1. Land everything on `master` with CI green, and check out a clean, up to date
   `master`.
2. Run the release checks: fmt, clippy, the workspace and `linux-full` suites,
   a current `docs/elements.html`, the API-break list, and a publish dry run.

   ```sh
   tools/release.sh check
   ```

3. Log in to crates.io (the token is never stored in the repo):

   ```sh
   cargo login <token>
   ```

4. Publish. This moves the `## Unreleased` CHANGELOG entries under a heading
   for the version, commits `release X.Y.Z`, tags `vX.Y.Z`, publishes every
   crate in dependency order, then pushes `master` and the tag.

   ```sh
   tools/release.sh publish
   ```

5. Bump to the next version. Before 1.0 a breaking change needs the next minor
   (`0.9.0` to `0.10.0`), see STABILITY.md.

   ```sh
   tools/release.sh bump 0.10.0
   git push origin master
   ```

## Notes

- `publish` can be rerun after a failure. It skips the tag step once the tag
  exists and skips every crate already on crates.io at that version.
- crates.io limits how fast crates are published. On a failed upload the
  script waits 10 minutes and retries the same crate.
- `g2g-pyapi`, `xtask` and the benchmark and web crates are `publish = false`.
  The Python wheel is built with maturin, not published by this script.
