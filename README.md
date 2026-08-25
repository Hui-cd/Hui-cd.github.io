# Hui@Blog

This public repository is the Jekyll publishing target for <https://hui-cd.github.io/>.

New technical studies are authored in the private sibling repository `engineering-literacy/studies/notes/`. Files with `publish: true` are rendered into `_posts/` by:

```bash
cd ../engineering-literacy
./scripts/publish-blog
```

The publishing command changes files only. It does not commit or push.

Historical MkDocs sources, mirrors, and generated output were removed from this repository after their unique material was preserved in the private notebook. Git history remains the recovery path for deleted public-repository files.
