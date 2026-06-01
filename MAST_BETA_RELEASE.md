# MAST Beta Release Developer Notes

The `mast-beta-release` branch provides access to MAST features and bugfixes before they are merged into the main Astroquery repository and included in an official release. This branch is maintained on this fork of Astroquery and contains MAST enhancements that may still be awaiting maintainer approval or merge into the upstream repository. Users who need access to the latest MAST functionality (i.e., for Roman purposes) can install Astroquery directly from this branch.

## Branch Maintenance

The `mast-beta-release` branch should be synchronized with both:

- The upstream Astroquery repository.
- MAST-specific feature branches that are ready for user testing or deployment.

We are essentially treating `mast-beta-release` as an integration branch for multiple features in review. We need to keep this branch up-to-date with all of the feature branches and the upstream repository. As we add things to this branch, we will need to carefully test and verify that the module still works and features fit together.

### Updating from Upstream

Periodically update `mast-beta-release` with the latest changes from the Astroquery main branch:

```
$ git checkout mast-beta-release
$ git fetch upstream/main
$ git rebase upstream/main
$ git push --force
```

If you encounter merge conflicts, contact Sam Bianco before continuing with the rebase. This means that the branch fell out of sync with one of the feature branches. When making additions to this branch, we also need to be careful to test and verify that all the functionality continues to work together.

### Updating from Feature Branches

When a MAST pull request contains functionality that should be made available before it is merged upstream:

1. Ensure the feature has comprehensive tests.
2. Rebase the mast-beta-release  branch onto the feature branch.
3. Resolve any merge conflicts (ask Sam for help if needed).
4. Run the test suites to make sure that nothing was broken while rebasing.
5. Push the updated branch

```
$ git checkout mast-beta-release
$ git fetch feature-branch
$ git rebase feature-branch
$ git push --force
```

If `mast-beta-release` already has the feature, and new commits are added to the feature branch, it will be easier to just cherry pick commits:

```
$ git checkout mast-beta-release
$ git cherry-pick <commit-hash>
$ git push --force
```
