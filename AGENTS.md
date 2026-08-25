# Public Blog Contract

This repository is a public Jekyll publishing target, not the primary technical-note workspace.

- Treat `../engineering-literacy/studies/notes/` as the canonical source for managed study posts.
- Publish managed posts with `../engineering-literacy/scripts/publish-blog`.
- Do not publish a note unless its private source explicitly contains `publish: true`.
- Preserve public URLs, the Jekyll build, local changes, and untracked files.
- Publishing, committing, and pushing are separate actions; never infer commit or push authorization.
