# Ueye patches on egui 0.36.2

This branch (`ueye/0.36.2`) is egui's `0.36.2` tag plus the patches Ueye needs.
Ueye and the applications built on it use it through `[patch.crates-io]`,
pinned to a commit:

```toml
[patch.crates-io]
egui      = { git = "https://github.com/tracel-ai/egui", rev = "<commit>" }
eframe    = { git = "https://github.com/tracel-ai/egui", rev = "<commit>" }
egui-wgpu = { git = "https://github.com/tracel-ai/egui", rev = "<commit>" }
egui_glow = { git = "https://github.com/tracel-ai/egui", rev = "<commit>" }
```

The crates keep their names and nothing is published to crates.io.

## Patches

| Commit | Crate | Patch | Upstream |
| --- | --- | --- | --- |
| `c1d087a36` | egui | `Context::set_repaint_observer`: a second callback that sees every repaint request | [emilk/egui#8644](https://github.com/emilk/egui/pull/8644) |
| `253c466a5` | egui | `Areas::set_state` public: a layer shown without an `Area` is hit-testable | [emilk/egui#8645](https://github.com/emilk/egui/pull/8645) |
| `8a3e12863` | egui-wgpu | Retained replay (`paint_with_cached_meshes`, `update_callbacks_only`), cached static prefix, `Renderer::texture_bind_group_layout` | not proposed yet |
| `b6fa37f7c` | egui_glow | Retained replay (`paint_primitives_retained`), the safe `custom` GL interface (Ueye-specific) | not proposed yet (`custom`: not intended upstream) |
| `747d045ad` | eframe | Retained paint-only frames (native and web), pointer-move filter, `one_pass_per_input`, no `requestAnimationFrame` at rest, lost repaint request fix, `App::on_native_pinch` | not proposed yet |
| `ce04c7002` | egui | Tests adapted to 0.36.2 (`Id::new`; delays minus one predicted frame, removed upstream by #8595) | branch only |

## Rebasing onto a new egui release

```sh
git fetch upstream --tags
git switch -c ueye/<version> ueye/0.36.2
git rebase --onto <version> 0.36.2
```

Drop the commits upstream has merged (and the branch-only test fix), run
`cargo test -p egui -p egui-wgpu -p egui_glow -p eframe`, push, update this
file, then bump the `rev` in Ueye and its applications.
