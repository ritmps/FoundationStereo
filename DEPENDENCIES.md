# Dependencies Note

This fork uses `environment.yml` for all dependency management.

The dinov2 directory is vendored code from the original FoundationStereo implementation.
The dinov2 package is NOT installed separately - its needed components are loaded via torch.hub.

All dependencies are managed through the root `environment.yml` file with updated, secure versions.
