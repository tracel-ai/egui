//! Note that this file contains code very similar to [`super::glow_integration`].
//! When making changes to one you often also want to apply it to the other.
//!
//! This is also very complex code, and not very pretty.
//! There is a bunch of improvements we could do,
//! like removing a bunch of `unwraps`.

use core::{cell::RefCell, num::NonZeroU32};
use std::{
    rc::Rc,
    sync::Arc,
    time::{Duration, Instant},
};

use egui_winit::ActionRequested;
use parking_lot::Mutex;
use raw_window_handle::{HasDisplayHandle as _, HasWindowHandle as _};
use winit::{
    event_loop::{ActiveEventLoop, EventLoop, EventLoopProxy},
    window::{Window, WindowId},
};

use ahash::HashMap;
use egui::{
    DeferredViewportUiCallback, FullOutput, ImmediateViewport, OrderedViewportIdMap, TexturesDelta,
    ViewportBuilder, ViewportClass, ViewportId, ViewportIdPair, ViewportIdSet, ViewportInfo,
    ViewportOutput,
};
#[cfg(feature = "accesskit")]
use egui_winit::accesskit_winit;
use log::warn;
use winit_integration::UserEvent;

use crate::{
    App, AppCreator, CreationContext, NativeOptions, Result, Storage,
    native::{
        epi_integration::EpiIntegration,
        winit_integration::{EventResult, sleep_if_invisible_or_minimized},
    },
};

use super::{epi_integration, event_loop_context, winit_integration, winit_integration::WinitApp};

// ----------------------------------------------------------------------------
// Types:

pub struct WgpuWinitApp<'app> {
    repaint_proxy: Arc<Mutex<EventLoopProxy<UserEvent>>>,
    app_name: String,
    native_options: NativeOptions,

    /// Set at initialization, then taken and set to `None` in `init_run_state`.
    app_creator: Option<AppCreator<'app>>,

    /// Set when we are actually up and running.
    running: Option<WgpuWinitRunning<'app>>,

    /// An optional pre-existing egui context. If `Some`, it is used instead of
    /// creating a new one via [`winit_integration::create_egui_context`]. Taken during initialization.
    egui_ctx: Option<egui::Context>,
}

/// State that is initialized when the application is first starts running via
/// a Resumed event. On Android this ensures that any graphics state is only
/// initialized once the application has an associated `SurfaceView`.
struct WgpuWinitRunning<'app> {
    integration: EpiIntegration,
    repaint_proxy: Arc<Mutex<EventLoopProxy<UserEvent>>>,

    /// The users application.
    app: Box<dyn 'app + App>,

    /// Wrapped in an `Rc<RefCell<…>>` so it can be re-entrantly shared via a weak-pointer.
    shared: Rc<RefCell<SharedState>>,

    pending_deltas: TexturesDelta,

    retained_repaint_after:
        Option<fn(&egui::Context, Option<u32>, bool, Instant) -> Option<core::time::Duration>>,
    retained_root: Option<RetainedFrame>,
    monitor_refresh: MonitorRefreshCache,
}

/// Root viewport paint jobs and the mesh buffers uploaded for them.
/// A paint-only frame can replay them until a normal UI pass changes state.
struct RetainedFrame {
    primitives: Vec<egui::ClippedPrimitive>,
    pixels_per_point: f32,
    clear_color: [f32; 4],
    window_size_px: winit::dpi::PhysicalSize<u32>,
    surface_config: egui_wgpu::SurfaceConfig,
    repaint_after: core::time::Duration,
}

fn retained_paint_deadline(frame_started: Instant, repaint_after: core::time::Duration) -> Instant {
    // Use the start of the frame. Present can wait for vsync; starting the
    // interval after it would under-run the display refresh rate.
    frame_started + repaint_after
}

fn multipass_repaint_request(
    output: &FullOutput,
    viewport_id: ViewportId,
    completed_pass_nr: u64,
    frame_started: Instant,
) -> Option<UserEvent> {
    if output.platform_output.num_completed_passes < 2 {
        return None;
    }
    let delay = output.viewport_output.get(&viewport_id)?.repaint_delay;
    if delay == Duration::MAX {
        return None;
    }
    // egui keeps the earliest repaint delay across discarded passes, but its
    // callback records the pass that requested it. Once a second pass ends,
    // the event loop can reject that callback as stale. Reissue the combined
    // frame's deadline using its final pass, preserving delayed UI updates
    // while retained GPU paints continue between them.
    Some(UserEvent::RequestRepaint {
        viewport_id,
        when: frame_started + delay,
        cumulative_pass_nr: completed_pass_nr.saturating_sub(1),
    })
}

/// Refresh-rate lookup can be relatively expensive on some hosts (notably
/// macOS), so retained frames share the last answer for up to two seconds.
#[derive(Default)]
struct MonitorRefreshCache {
    refresh_millihertz: Option<u32>,
    sampled_at: Option<Instant>,
}

impl MonitorRefreshCache {
    const MAX_AGE: Duration = Duration::from_secs(2);

    fn get_or_sample(&mut self, now: Instant, sample: impl FnOnce() -> Option<u32>) -> Option<u32> {
        if self
            .sampled_at
            .is_none_or(|last| now.saturating_duration_since(last) >= Self::MAX_AGE)
        {
            self.refresh_millihertz = sample().filter(|rate| *rate > 0);
            self.sampled_at = Some(now);
        }
        self.refresh_millihertz
    }

    fn for_window(&mut self, window: &Window) -> Option<u32> {
        self.get_or_sample(Instant::now(), || {
            window
                .current_monitor()
                .and_then(|monitor| monitor.refresh_rate_millihertz())
        })
    }
}

impl Drop for WgpuWinitRunning<'_> {
    fn drop(&mut self) {
        // Avoid debug panic when dropping unapplied deltas on teardown
        self.pending_deltas.clear();
    }
}

/// Everything needed by the immediate viewport renderer.\
///
/// This is shared by all viewports.
///
/// Wrapped in an `Rc<RefCell<…>>` so it can be re-entrantly shared via a weak-pointer.
pub struct SharedState {
    egui_ctx: egui::Context,
    viewports: Viewports,
    painter: egui_wgpu::winit::Painter,
    viewport_from_window: HashMap<WindowId, ViewportId>,
    focused_viewport: Option<ViewportId>,
    resized_viewport: Option<ViewportId>,
}

pub type Viewports = egui::OrderedViewportIdMap<Viewport>;

pub struct Viewport {
    ids: ViewportIdPair,
    class: ViewportClass,
    builder: ViewportBuilder,
    deferred_commands: Vec<egui::viewport::ViewportCommand>,
    info: ViewportInfo,
    actions_requested: Vec<ActionRequested>,

    /// Any not yet applied deltas for this viewport.
    pending_delta: TexturesDelta,

    /// `None` for sync viewports.
    viewport_ui_cb: Option<Arc<DeferredViewportUiCallback>>,

    /// Window surface state that's initialized when the app starts running via a Resumed event
    /// and on Android will also be destroyed if the application is paused.
    window: Option<Arc<Window>>,

    /// `window` and `egui_winit` are initialized together.
    egui_winit: Option<egui_winit::State>,
}

impl Drop for Viewport {
    fn drop(&mut self) {
        // Avoid debug panic when dropping unapplied deltas on teardown
        self.pending_delta.clear();
    }
}

// ----------------------------------------------------------------------------

impl<'app> WgpuWinitApp<'app> {
    pub fn new(
        event_loop: &EventLoop<UserEvent>,
        app_name: &str,
        native_options: NativeOptions,
        egui_ctx: Option<egui::Context>,
        app_creator: AppCreator<'app>,
    ) -> Self {
        profiling::function_scope!();

        #[cfg(feature = "__screenshot")]
        assert!(
            std::env::var("EFRAME_SCREENSHOT_TO").is_err(),
            "EFRAME_SCREENSHOT_TO not yet implemented for wgpu backend"
        );

        Self {
            repaint_proxy: Arc::new(Mutex::new(event_loop.create_proxy())),
            app_name: app_name.to_owned(),
            native_options,
            running: None,
            app_creator: Some(app_creator),
            egui_ctx,
        }
    }

    /// Create a window for all viewports lacking one.
    fn initialized_all_windows(&mut self, event_loop: &ActiveEventLoop) {
        let Some(running) = &mut self.running else {
            return;
        };
        let mut shared = running.shared.borrow_mut();
        let SharedState {
            viewports,
            painter,
            viewport_from_window,
            ..
        } = &mut *shared;

        for viewport in viewports.values_mut() {
            viewport.initialize_window(
                event_loop,
                &running.integration.egui_ctx,
                viewport_from_window,
                painter,
            );
        }
    }

    #[cfg(target_os = "android")]
    fn recreate_window(&self, event_loop: &ActiveEventLoop, running: &WgpuWinitRunning<'app>) {
        let SharedState {
            egui_ctx,
            viewports,
            viewport_from_window,
            painter,
            ..
        } = &mut *running.shared.borrow_mut();

        initialize_or_update_viewport(
            viewports,
            ViewportIdPair::ROOT,
            ViewportClass::Root,
            self.native_options.viewport.clone(),
            None,
            painter,
        )
        .initialize_window(event_loop, egui_ctx, viewport_from_window, painter);
    }

    #[cfg(target_os = "android")]
    fn drop_window(&mut self) -> Result<(), egui_wgpu::WgpuError> {
        if let Some(running) = &mut self.running {
            let mut shared = running.shared.borrow_mut();
            shared.viewports.remove(&ViewportId::ROOT);
            pollster::block_on(shared.painter.set_window(ViewportId::ROOT, None))?;
        }
        Ok(())
    }

    fn init_run_state(
        &mut self,
        egui_ctx: egui::Context,
        event_loop: &ActiveEventLoop,
        storage: Option<Box<dyn Storage>>,
        window: Window,
        builder: ViewportBuilder,
    ) -> crate::Result<&mut WgpuWinitRunning<'app>> {
        profiling::function_scope!();
        // Inject the display handle into the wgpu setup so that wgpu can create
        // surfaces on platforms that require it (e.g. GLES on Wayland).
        let mut wgpu_options = self.native_options.wgpu_options.clone();
        if let egui_wgpu::WgpuSetup::CreateNew(ref mut create_new) = wgpu_options.wgpu_setup
            && create_new.display_handle.is_none()
        {
            create_new.display_handle = Some(Box::new(event_loop.owned_display_handle()));
        }
        let mut painter = pollster::block_on(egui_wgpu::winit::Painter::new(
            egui_ctx.clone(),
            wgpu_options,
            self.native_options.viewport.transparent.unwrap_or(false),
            egui_wgpu::RendererOptions {
                msaa_samples: self.native_options.multisampling as _,
                depth_stencil_format: egui_wgpu::depth_format_from_bits(
                    self.native_options.depth_buffer,
                    self.native_options.stencil_buffer,
                ),
                dithering: self.native_options.dithering,
                ..Default::default()
            },
        ));

        let mut viewport_info = ViewportInfo::default();
        egui_winit::update_viewport_info(&mut viewport_info, &egui_ctx, &window, true);

        {
            // Tell egui right away about native_pixels_per_point etc,
            // so that the app knows about it during app creation:
            let pixels_per_point = egui_winit::pixels_per_point(&egui_ctx, &window);

            egui_ctx.input_mut(|i| {
                i.raw
                    .viewports
                    .insert(ViewportId::ROOT, viewport_info.clone());
                i.pixels_per_point = pixels_per_point;
            });
        }

        let window = Arc::new(window);

        {
            profiling::scope!("set_window");
            pollster::block_on(painter.set_window(ViewportId::ROOT, Some(Arc::clone(&window))))?;
        }

        let wgpu_render_state = painter.render_state();

        let mut integration = EpiIntegration::new(
            egui_ctx.clone(),
            &window,
            &self.app_name,
            &self.native_options,
            storage,
            #[cfg(feature = "glow")]
            None,
            #[cfg(feature = "glow")]
            None,
            wgpu_render_state.clone(),
        );

        {
            let event_loop_proxy = Arc::clone(&self.repaint_proxy);
            // Ueye patch: shared with glow, see `RepaintWatch`.
            integration.install_repaint_callbacks(move |event| {
                event_loop_proxy.lock().send_event(event).ok();
            });
        }

        #[allow(clippy::allow_attributes, unused_mut)] // used for accesskit
        let mut egui_winit = egui_winit::State::new(
            egui_ctx.clone(),
            ViewportId::ROOT,
            event_loop,
            Some(window.scale_factor() as f32),
            event_loop.system_theme(),
            painter.max_texture_side(),
        );

        #[cfg(feature = "accesskit")]
        {
            let event_loop_proxy = self.repaint_proxy.lock().clone();
            egui_winit.init_accesskit(event_loop, &window, event_loop_proxy);
        }

        let app_creator = core::mem::take(&mut self.app_creator)
            .expect("Single-use AppCreator has unexpectedly already been taken");

        crate::maybe_attach_inspection_plugin(&egui_ctx, Some(self.app_name.clone()));

        let cc = CreationContext {
            egui_ctx: egui_ctx.clone(),
            integration_info: integration.frame.info().clone(),
            storage: integration.frame.storage(),
            #[cfg(feature = "glow")]
            gl: None,
            #[cfg(feature = "glow")]
            get_proc_address: None,
            wgpu_render_state,
            window: Some(Arc::clone(&window)),
            raw_display_handle: window.display_handle().map(|h| h.as_raw()),
            raw_window_handle: window.window_handle().map(|h| h.as_raw()),
        };
        let app = {
            profiling::scope!("user_app_creator");
            app_creator(&cc).map_err(crate::Error::AppCreation)?
        };

        let mut viewport_from_window = HashMap::default();
        viewport_from_window.insert(window.id(), ViewportId::ROOT);

        let mut viewports = Viewports::default();
        viewports.insert(
            ViewportId::ROOT,
            Viewport {
                ids: ViewportIdPair::ROOT,
                class: ViewportClass::Root,
                builder,
                deferred_commands: vec![],
                info: viewport_info,
                actions_requested: Default::default(),
                viewport_ui_cb: None,
                window: Some(window),
                egui_winit: Some(egui_winit),
                pending_delta: Default::default(),
            },
        );

        let shared = Rc::new(RefCell::new(SharedState {
            egui_ctx,
            viewport_from_window,
            viewports,
            painter,
            focused_viewport: Some(ViewportId::ROOT),
            resized_viewport: None,
        }));

        {
            // Create a weak pointer so that we don't keep state alive for too long.
            let shared = Rc::downgrade(&shared);
            let beginning = integration.beginning;

            egui::Context::set_immediate_viewport_renderer(move |_egui_ctx, immediate_viewport| {
                if let Some(shared) = shared.upgrade() {
                    render_immediate_viewport(beginning, &shared, immediate_viewport);
                } else {
                    log::warn!("render_sync_callback called after window closed");
                }
            });
        }

        Ok(self.running.insert(WgpuWinitRunning {
            integration,
            repaint_proxy: Arc::clone(&self.repaint_proxy),
            app,
            shared,
            pending_deltas: Default::default(),
            retained_repaint_after: self.native_options.retained_repaint_after,
            retained_root: None,
            monitor_refresh: MonitorRefreshCache::default(),
        }))
    }
}

impl WinitApp for WgpuWinitApp<'_> {
    fn egui_ctx(&self) -> Option<&egui::Context> {
        self.running.as_ref().map(|r| &r.integration.egui_ctx)
    }

    fn window(&self, window_id: WindowId) -> Option<Arc<Window>> {
        self.running
            .as_ref()
            .and_then(|r| {
                let shared = r.shared.borrow();
                let id = shared.viewport_from_window.get(&window_id)?;
                shared.viewports.get(id).map(|v| v.window.clone())
            })
            .flatten()
    }

    fn window_id_from_viewport_id(&self, id: ViewportId) -> Option<WindowId> {
        Some(
            self.running
                .as_ref()?
                .shared
                .borrow()
                .viewports
                .get(&id)?
                .window
                .as_ref()?
                .id(),
        )
    }

    fn save(&mut self) {
        log::debug!("WinitApp::save called");
        if let Some(running) = self.running.as_mut() {
            running.save();
        }
    }

    fn save_and_destroy(&mut self) {
        if let Some(mut running) = self.running.take() {
            running.save_and_destroy();
        }
    }

    fn run_ui_and_paint(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
    ) -> Result<EventResult> {
        self.initialized_all_windows(event_loop);

        if let Some(running) = &mut self.running {
            running.run_ui_and_paint(window_id, event_loop)
        } else {
            Ok(EventResult::Wait)
        }
    }

    fn run_paint_only(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
    ) -> crate::Result<EventResult> {
        self.initialized_all_windows(event_loop);

        if let Some(running) = &mut self.running {
            running.run_paint_only(window_id, event_loop)
        } else {
            Ok(EventResult::Wait)
        }
    }

    fn resumed(&mut self, event_loop: &ActiveEventLoop) -> crate::Result<EventResult> {
        log::debug!("Event::Resumed");

        let running = if let Some(running) = &self.running {
            #[cfg(target_os = "android")]
            self.recreate_window(event_loop, running);
            running
        } else {
            let storage = if let Some(file) = &self.native_options.persistence_path {
                epi_integration::create_storage_with_file(file)
            } else {
                epi_integration::create_storage(
                    self.native_options
                        .viewport
                        .app_id
                        .as_ref()
                        .unwrap_or(&self.app_name),
                )
            };
            let egui_ctx = self
                .egui_ctx
                .take()
                .unwrap_or_else(|| winit_integration::create_egui_context(storage.as_deref()));
            let (window, builder) = create_window(
                &egui_ctx,
                event_loop,
                storage.as_deref(),
                &mut self.native_options,
            )?;
            self.init_run_state(egui_ctx, event_loop, storage, window, builder)?
        };

        let viewport = &running.shared.borrow().viewports[&ViewportId::ROOT];
        if let Some(window) = &viewport.window {
            Ok(EventResult::RepaintNow(window.id()))
        } else {
            Ok(EventResult::Wait)
        }
    }

    fn suspended(&mut self, _: &ActiveEventLoop) -> crate::Result<EventResult> {
        #[cfg(target_os = "android")]
        self.drop_window()?;
        Ok(EventResult::Save)
    }

    fn device_event(
        &mut self,
        _: &ActiveEventLoop,
        _: winit::event::DeviceId,
        event: winit::event::DeviceEvent,
    ) -> crate::Result<EventResult> {
        if let winit::event::DeviceEvent::MouseMotion { delta } = event
            && let Some(running) = &mut self.running
        {
            let mut shared = running.shared.borrow_mut();
            if let Some(viewport) = shared
                .focused_viewport
                .and_then(|viewport| shared.viewports.get_mut(&viewport))
                && let Some(window) = viewport.window.as_ref()
            {
                if !window.has_focus()
                    && !viewport
                        .egui_winit
                        .as_ref()
                        .map(|state| state.is_any_pointer_button_down())
                        .unwrap_or(false)
                {
                    return Ok(EventResult::Wait);
                }

                if let Some(egui_winit) = viewport.egui_winit.as_mut()
                    && egui_winit.on_mouse_motion(delta)
                {
                    // Ueye patch (DESIGN.md 9.4): with pointer filtering, raw
                    // motion runs no pass by itself unless a button is held;
                    // the window's own pointer moves decide.
                    if self.native_options.pointer_move_needs_ui_pass.is_some()
                        && !egui_winit.is_any_pointer_button_down()
                    {
                        crate::pointer_filter::merge_mouse_motion(egui_winit.egui_input_mut());
                        return Ok(EventResult::Wait);
                    }
                    return Ok(EventResult::RepaintNext(window.id()));
                }
            }
        }

        Ok(EventResult::Wait)
    }

    fn pointer_move_filtered(
        &mut self,
        window_id: WindowId,
        event: &winit::event::WindowEvent,
    ) -> bool {
        let (Some(needs_ui_pass), Some(running)) = (
            self.native_options.pointer_move_needs_ui_pass,
            &mut self.running,
        ) else {
            return false;
        };
        let egui_ctx = &running.integration.egui_ctx;
        let mut shared = running.shared.borrow_mut();
        let SharedState {
            viewport_from_window,
            viewports,
            ..
        } = &mut *shared;
        if viewport_from_window.get(&window_id) != Some(&ViewportId::ROOT) {
            return false;
        }
        let Some(viewport) = viewports.get_mut(&ViewportId::ROOT) else {
            return false;
        };
        let (Some(window), Some(egui_winit)) = (&viewport.window, &mut viewport.egui_winit) else {
            return false;
        };
        let pixels_per_point = egui_winit::pixels_per_point(egui_ctx, window);
        let Some(pointer) = winit_integration::pointer_move(event, pixels_per_point) else {
            return false;
        };
        if needs_ui_pass(egui_ctx, pointer) {
            return false;
        }
        crate::pointer_filter::drop_superseded_pointer_move(egui_winit.egui_input_mut());
        true
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: winit::event::WindowEvent,
    ) -> crate::Result<EventResult> {
        self.initialized_all_windows(event_loop);

        if let Some(running) = &mut self.running {
            Ok(running.on_window_event(window_id, &event))
        } else {
            // running is removed to get ready for exiting
            Ok(EventResult::Exit)
        }
    }

    #[cfg(feature = "accesskit")]
    fn on_accesskit_event(&mut self, event: accesskit_winit::Event) -> crate::Result<EventResult> {
        if let Some(running) = &mut self.running {
            let mut shared_lock = running.shared.borrow_mut();
            let SharedState {
                viewport_from_window,
                viewports,
                ..
            } = &mut *shared_lock;
            if let Some(viewport) = viewport_from_window
                .get(&event.window_id)
                .and_then(|id| viewports.get_mut(id))
                && let Some(egui_winit) = &mut viewport.egui_winit
            {
                return Ok(winit_integration::on_accesskit_window_event(
                    egui_winit,
                    event.window_id,
                    &event.window_event,
                ));
            }
        }

        Ok(EventResult::Wait)
    }
}

impl WgpuWinitRunning<'_> {
    /// Saves the application state
    fn save(&mut self) {
        let shared = self.shared.borrow();
        // This is done because of the "save on suspend" logic on Android. Once the application is suspended, there is no window associated to it.
        let window = if let Some(Viewport { window, .. }) = shared.viewports.get(&ViewportId::ROOT)
        {
            window.as_deref()
        } else {
            None
        };
        self.integration.save(self.app.as_mut(), window);
    }

    fn save_and_destroy(&mut self) {
        profiling::function_scope!();

        self.save();

        #[cfg(feature = "glow")]
        self.app.on_exit(None);

        #[cfg(not(feature = "glow"))]
        self.app.on_exit();

        let mut shared = self.shared.borrow_mut();
        shared.painter.destroy();
    }

    fn run_paint_only(
        &mut self,
        window_id: WindowId,
        event_loop: &ActiveEventLoop,
    ) -> Result<EventResult> {
        let can_replay = {
            let shared = self.shared.borrow();
            let Some(viewport_id) = shared.viewport_from_window.get(&window_id) else {
                return Ok(EventResult::Wait);
            };
            let Some(viewport) = shared.viewports.get(viewport_id) else {
                return Ok(EventResult::Wait);
            };
            let Some(window) = viewport.window.as_ref() else {
                return Ok(EventResult::Wait);
            };
            *viewport_id == ViewportId::ROOT
                && self.retained_root.as_ref().is_some_and(|frame| {
                    window.inner_size() == frame.window_size_px
                        && self.integration.egui_ctx.pixels_per_point() == frame.pixels_per_point
                        && shared.painter.can_replay_cached_meshes(
                            ViewportId::ROOT,
                            [frame.window_size_px.width, frame.window_size_px.height],
                            frame.surface_config,
                        )
                })
                && viewport.info.visible().unwrap_or(true)
                && viewport.info.events.is_empty()
                && viewport.egui_winit.as_ref().is_some_and(|state| {
                    let input = state.egui_input();
                    crate::pointer_filter::only_pointer_moves(&input.events)
                        && input.hovered_files.is_empty()
                        && input.dropped_files.is_empty()
                })
                && viewport.pending_delta.set.is_empty()
                && viewport.pending_delta.free.is_empty()
                && viewport.actions_requested.is_empty()
                && self.pending_deltas.set.is_empty()
                && self.pending_deltas.free.is_empty()
                && !self.integration.retained_paint_needs_full_ui()
        };
        if !can_replay {
            self.retained_root = None;
            return self.run_ui_and_paint(window_id, event_loop);
        }

        profiling::scope!("retained_paint_only");
        let frame_timer = Instant::now();
        let mut shared = self.shared.borrow_mut();
        let SharedState {
            viewports, painter, ..
        } = &mut *shared;
        let Some(window) = viewports
            .get(&ViewportId::ROOT)
            .and_then(|viewport| viewport.window.as_ref())
        else {
            return Ok(EventResult::Wait);
        };
        let frame = self.retained_root.as_ref().expect("checked above");
        let mut no_texture_changes = TexturesDelta::default();
        let vsync_secs = painter.paint_with_cached_meshes(
            ViewportId::ROOT,
            frame.pixels_per_point,
            frame.clear_color,
            &frame.primitives,
            &mut no_texture_changes,
            vec![],
            window,
        );
        self.integration.post_rendering(window);
        self.integration
            .report_frame_time(frame_timer.elapsed().as_secs_f32() - vsync_secs);
        self.integration
            .maybe_autosave(self.app.as_mut(), Some(window.as_ref()));
        let repaint_after = self.retained_repaint_after.and_then(|cadence| {
            cadence(
                &self.integration.egui_ctx,
                self.monitor_refresh.for_window(window),
                true,
                frame_timer,
            )
        });
        if let Some(repaint_after) = repaint_after {
            Ok(EventResult::PaintOnlyAt(
                window_id,
                retained_paint_deadline(frame_timer, repaint_after),
            ))
        } else {
            self.retained_root = None;
            Ok(EventResult::Wait)
        }
    }

    /// This is called both for the root viewport, and all deferred viewports
    fn run_ui_and_paint(
        &mut self,
        window_id: WindowId,
        event_loop: &ActiveEventLoop,
    ) -> Result<EventResult> {
        profiling::function_scope!();
        let frame_started = Instant::now();

        let Some(viewport_id) = self
            .shared
            .borrow()
            .viewport_from_window
            .get(&window_id)
            .copied()
        else {
            return Ok(EventResult::Wait);
        };

        profiling::finish_frame!();

        let Self {
            app,
            integration,
            repaint_proxy,
            shared,
            pending_deltas,
            retained_repaint_after,
            retained_root,
            monitor_refresh,
        } = self;

        if viewport_id != ViewportId::ROOT {
            // egui-wgpu shares one mesh buffer across viewports. Rendering any
            // other viewport replaces the root's retained mesh slices.
            *retained_root = None;
        }

        let mut frame_timer = crate::stopwatch::Stopwatch::new();
        frame_timer.start();

        let (viewport_ui_cb, raw_input, is_visible, show_ui) = {
            profiling::scope!("Prepare");
            let mut shared_lock = shared.borrow_mut();

            let SharedState {
                viewports, painter, ..
            } = &mut *shared_lock;

            if viewport_id != ViewportId::ROOT {
                let Some(viewport) = viewports.get(&viewport_id) else {
                    return Ok(EventResult::Wait);
                };

                if viewport.viewport_ui_cb.is_none() {
                    // This will only happen if this is an immediate viewport.
                    // That means that the viewport cannot be rendered by itself and needs his parent to be rendered.
                    if let Some(viewport) = viewports.get(&viewport.ids.parent)
                        && let Some(window) = viewport.window.as_ref()
                    {
                        return Ok(EventResult::RepaintNext(window.id()));
                    }
                    return Ok(EventResult::Wait);
                }
            }

            let Some(viewport) = viewports.get_mut(&viewport_id) else {
                return Ok(EventResult::Wait);
            };

            let Viewport {
                viewport_ui_cb,
                window,
                egui_winit,
                info,
                ..
            } = viewport;

            let viewport_ui_cb = viewport_ui_cb.clone();

            let Some(window) = window else {
                return Ok(EventResult::Wait);
            };
            egui_winit::update_viewport_info(info, &integration.egui_ctx, window, false);

            let is_visible = viewport.info.visible().unwrap_or(true);

            {
                profiling::scope!("set_window");
                pollster::block_on(painter.set_window(viewport_id, Some(Arc::clone(window))))?;
            }

            let Some(egui_winit) = egui_winit.as_mut() else {
                return Ok(EventResult::Wait);
            };
            let mut raw_input = egui_winit.take_egui_input(window);

            let show_ui = is_visible || is_viewport_or_descendant_visible(viewports, viewport_id);

            integration.pre_update();

            raw_input.time = Some(integration.beginning.elapsed().as_secs_f64());
            raw_input.viewports = viewports
                .iter()
                .map(|(id, viewport)| (*id, viewport.info.clone()))
                .collect();

            painter.handle_screenshots(&mut raw_input.events);

            (viewport_ui_cb, raw_input, is_visible, show_ui)
        };

        if !show_ui {
            *retained_root = None;
            // Nothing will be shown, so we run no egui pass at all.
            // That way all ui state is left untouched, and is still there
            // when this viewport becomes visible again.
            let is_root_viewport = viewport_ui_cb.is_none();
            if is_root_viewport {
                // The app logic keeps ticking, so it can e.g. ask to be shown again:
                let egui::LogicOutput {
                    platform_output,
                    viewport_commands,
                } = integration.update_logic_only(app.as_mut(), raw_input);

                let mut shared_mut = shared.borrow_mut();
                let SharedState { viewports, .. } = &mut *shared_mut;

                if let Some(viewport) = viewports.get_mut(&viewport_id) {
                    viewport.info.events.clear(); // they should have been processed
                    if let Viewport {
                        window: Some(window),
                        egui_winit: Some(egui_winit),
                        ..
                    } = viewport
                    {
                        egui_winit.handle_platform_output_with_event_loop(
                            window,
                            event_loop,
                            platform_output,
                        );
                    }
                }

                for (id, commands) in viewport_commands {
                    if let Some(viewport) = viewports.get_mut(&id) {
                        viewport.process_commands(&integration.egui_ctx, commands);
                    }
                }
            }

            sleep_if_invisible_or_minimized(
                shared
                    .borrow()
                    .viewports
                    .get(&viewport_id)
                    .and_then(|viewport| viewport.window.as_deref()),
            );

            return Ok(if integration.should_close() {
                EventResult::CloseRequested
            } else {
                EventResult::Wait
            });
        }

        // ------------------------------------------------------------

        // Runs the update, which could call immediate viewports,
        // so make sure we hold no locks here!
        let full_output = integration.update(app.as_mut(), viewport_ui_cb.as_deref(), raw_input);
        // A root pass under the `RepaintWatch` posted the repaint of its
        // last pass already.
        if (integration.repaint_watch.is_none() || viewport_id != ViewportId::ROOT)
            && let Some(request) = multipass_repaint_request(
                &full_output,
                viewport_id,
                integration.egui_ctx.cumulative_pass_nr_for(viewport_id),
                frame_started,
            )
        {
            repaint_proxy.lock().send_event(request).ok();
        }

        // ------------------------------------------------------------

        let mut shared_mut = shared.borrow_mut();

        let SharedState {
            egui_ctx,
            viewports,
            painter,
            viewport_from_window,
            ..
        } = &mut *shared_mut;

        let FullOutput {
            platform_output,
            textures_delta,
            shapes,
            pixels_per_point,
            viewport_output,
        } = full_output;

        pending_deltas.append(textures_delta);

        remove_viewports_not_in(viewports, painter, viewport_from_window, &viewport_output);

        let Some(viewport) = viewports.get_mut(&viewport_id) else {
            return Ok(EventResult::Wait);
        };

        viewport.info.events.clear(); // they should have been processed

        let Viewport {
            window: Some(window),
            egui_winit: Some(egui_winit),
            ..
        } = viewport
        else {
            return Ok(EventResult::Wait);
        };

        egui_winit.handle_platform_output_with_event_loop(window, event_loop, platform_output);

        let vsync_secs = if is_visible {
            let clipped_primitives = egui_ctx.tessellate(shapes, pixels_per_point);

            let mut screenshot_commands = vec![];
            viewport.actions_requested.retain(|cmd| {
                if let ActionRequested::Screenshot(info) = cmd {
                    screenshot_commands.push(info.clone());
                    false
                } else {
                    true
                }
            });
            let clear_color = app.clear_color(&egui_ctx.global_style().visuals);
            let surface_config = painter.surface_config();
            let can_retain = viewport_id == ViewportId::ROOT
                && window.inner_size().width > 0
                && window.inner_size().height > 0
                && pending_deltas.free.is_empty()
                && screenshot_commands.is_empty()
                && surface_config.is_some();
            let vsync_secs = painter.paint_and_update_textures(
                viewport_id,
                pixels_per_point,
                clear_color,
                &clipped_primitives,
                pending_deltas,
                screenshot_commands,
                window,
            );

            *retained_root = if can_retain {
                retained_repaint_after
                    .and_then(|cadence| {
                        cadence(
                            egui_ctx,
                            monitor_refresh.for_window(window),
                            false,
                            frame_started,
                        )
                    })
                    .map(|repaint_after| RetainedFrame {
                        primitives: clipped_primitives,
                        pixels_per_point,
                        clear_color,
                        window_size_px: window.inner_size(),
                        surface_config: surface_config.expect("checked above"),
                        repaint_after,
                    })
            } else {
                None
            };

            for action in viewport.actions_requested.drain(..) {
                match action {
                    ActionRequested::Screenshot { .. } => {
                        // already handled above
                    }
                    ActionRequested::Cut => {
                        egui_winit.egui_input_mut().events.push(egui::Event::Cut);
                    }
                    ActionRequested::Copy => {
                        egui_winit.egui_input_mut().events.push(egui::Event::Copy);
                    }
                    ActionRequested::Paste => {
                        if let Some(contents) = egui_winit.clipboard_text() {
                            let contents = contents.replace("\r\n", "\n");
                            if !contents.is_empty() {
                                egui_winit
                                    .egui_input_mut()
                                    .events
                                    .push(egui::Event::Paste(contents));
                            }
                        }
                    }
                }
            }

            integration.post_rendering(window);

            vsync_secs
        } else {
            *retained_root = None;
            0.0
        };

        let active_viewports_ids: ViewportIdSet = viewport_output.keys().copied().collect();

        handle_viewport_output(
            &integration.egui_ctx,
            &viewport_output,
            viewports,
            painter,
            viewport_from_window,
        );

        // Prune dead viewports:
        viewports.retain(|id, _| active_viewports_ids.contains(id));
        viewport_from_window.retain(|_, id| active_viewports_ids.contains(id));
        painter.gc_viewports(&active_viewports_ids);

        if viewports.len() > 1 {
            *retained_root = None;
        }

        let window = viewport_from_window
            .get(&window_id)
            .and_then(|id| viewports.get(id))
            .and_then(|vp| vp.window.as_ref());

        integration.report_frame_time(frame_timer.total_time_sec() - vsync_secs); // don't count auto-save time as part of regular frame time

        integration.maybe_autosave(app.as_mut(), window.map(|w| w.as_ref()));

        sleep_if_invisible_or_minimized(window.map(|window| window.as_ref()));

        if integration.should_close() {
            Ok(EventResult::CloseRequested)
        } else if viewport_id == ViewportId::ROOT
            && let Some(frame) = retained_root.as_ref()
        {
            Ok(EventResult::PaintOnlyAt(
                window_id,
                retained_paint_deadline(frame_started, frame.repaint_after),
            ))
        } else if viewport_id == ViewportId::ROOT
            && let Some(cadence) = retained_repaint_after.and_then(|cadence| {
                cadence(
                    egui_ctx,
                    window.and_then(|window| monitor_refresh.for_window(window)),
                    false,
                    frame_started,
                )
            })
        {
            // If a transient texture, screenshot, or extra viewport prevents
            // retention, keep the animation alive with a regular egui frame.
            Ok(EventResult::RepaintAt(
                window_id,
                retained_paint_deadline(frame_started, cadence),
            ))
        } else {
            Ok(EventResult::Wait)
        }
    }

    fn on_window_event(
        &mut self,
        window_id: WindowId,
        event: &winit::event::WindowEvent,
    ) -> EventResult {
        let Self {
            integration,
            app,
            shared,
            ..
        } = self;
        let mut shared = shared.borrow_mut();

        let viewport_id = shared.viewport_from_window.get(&window_id).copied();

        if viewport_id == Some(ViewportId::ROOT)
            && let winit::event::WindowEvent::PinchGesture { delta, phase, .. } = event
        {
            app.on_native_pinch(
                *delta,
                match phase {
                    winit::event::TouchPhase::Started => egui::TouchPhase::Start,
                    winit::event::TouchPhase::Moved => egui::TouchPhase::Move,
                    winit::event::TouchPhase::Ended => egui::TouchPhase::End,
                    winit::event::TouchPhase::Cancelled => egui::TouchPhase::Cancel,
                },
            );
        }

        // On Windows, if a window is resized by the user, it should repaint synchronously, inside the
        // event handler. If this is not done, the compositor will assume that the window does not want
        // to redraw and continue ahead.
        //
        // In eframe's case, that causes the window to rapidly flicker, as it struggles to deliver
        // new frames to the compositor in time. The flickering is technically glutin or glow's fault, but we should be responding properly
        // to resizes anyway, as doing so avoids dropping frames.
        //
        // See: https://github.com/emilk/egui/issues/903
        let mut repaint_asap = false;

        // On MacOS the asap repaint is not enough. The drawn frames must be synchronized with
        // the CoreAnimation transactions driving the window resize process.
        //
        // Thus, Painter, responsible for wgpu surfaces and their resize, has to be notified of the
        // resize lifecycle, yet winit does not provide any events for that. To work around,
        // the last resized viewport is tracked until a later event outside the live resize stream
        // is received.
        //
        // AppKit can emit `Moved` events during top/left live resize because the window origin
        // changes along with the content size. Treat those as part of live resize on macOS.
        //
        // See: https://github.com/emilk/egui/issues/903
        let event_keeps_resize_active = matches!(event, winit::event::WindowEvent::Resized(_))
            || (cfg!(target_os = "macos") && matches!(event, winit::event::WindowEvent::Moved(_)));

        if !event_keeps_resize_active
            && let Some(id) = viewport_id
            && shared.resized_viewport == viewport_id
        {
            shared.painter.on_window_resize_state_change(id, false);
            shared.resized_viewport = None;
        }

        match event {
            winit::event::WindowEvent::Focused(focused) => {
                let focused = if cfg!(target_os = "macos")
                    && let Some(viewport_id) = viewport_id
                    && let Some(viewport) = shared.viewports.get(&viewport_id)
                    && let Some(window) = &viewport.window
                {
                    // TODO(emilk): remove this work-around once we update winit
                    // https://github.com/rust-windowing/winit/issues/4371
                    // https://github.com/emilk/egui/issues/7588
                    window.has_focus()
                } else {
                    *focused
                };

                shared.focused_viewport = focused.then_some(viewport_id).flatten();
            }

            winit::event::WindowEvent::Resized(physical_size) => {
                // Resize with 0 width and height is used by winit to signal a minimize event on Windows.
                // See: https://github.com/rust-windowing/winit/issues/208
                // This solves an issue where the app would panic when minimizing on Windows.
                if let Some(id) = viewport_id
                    && let (Some(width), Some(height)) = (
                        NonZeroU32::new(physical_size.width),
                        NonZeroU32::new(physical_size.height),
                    )
                {
                    if shared.resized_viewport != viewport_id {
                        shared.resized_viewport = viewport_id;
                        shared.painter.on_window_resize_state_change(id, true);
                    }
                    shared.painter.on_window_resized(id, width, height);
                    repaint_asap = true;
                }
            }

            winit::event::WindowEvent::Occluded(is_occluded) => {
                if let Some(viewport_id) = viewport_id
                    && let Some(viewport) = shared.viewports.get_mut(&viewport_id)
                {
                    viewport.info.occluded = Some(*is_occluded);
                }
            }

            winit::event::WindowEvent::CloseRequested => {
                if viewport_id == Some(ViewportId::ROOT) && integration.should_close() {
                    log::debug!(
                        "Received WindowEvent::CloseRequested for main viewport - shutting down."
                    );
                    return EventResult::CloseRequested;
                }

                log::debug!("Received WindowEvent::CloseRequested for viewport {viewport_id:?}");

                if let Some(viewport_id) = viewport_id
                    && let Some(viewport) = shared.viewports.get_mut(&viewport_id)
                {
                    // Tell viewport it should close:
                    viewport.info.events.push(egui::ViewportEvent::Close);

                    // We may need to repaint both us and our parent to close the window,
                    // and perhaps twice (once to notice the close-event, once again to enforce it).
                    // `request_repaint_of` does a double-repaint though:
                    integration.egui_ctx.request_repaint_of(viewport_id);
                    integration.egui_ctx.request_repaint_of(viewport.ids.parent);
                }
            }

            _ => {}
        }

        let event_response = viewport_id
            .and_then(|viewport_id| {
                let viewport = shared.viewports.get_mut(&viewport_id)?;
                Some(integration.on_window_event(
                    viewport.window.as_deref()?,
                    viewport.egui_winit.as_mut()?,
                    event,
                ))
            })
            .unwrap_or_default();

        if integration.should_close() {
            EventResult::CloseRequested
        } else if event_response.repaint {
            if repaint_asap {
                EventResult::RepaintNow(window_id)
            } else {
                EventResult::RepaintNext(window_id)
            }
        } else {
            EventResult::Wait
        }
    }
}

impl Viewport {
    /// Apply the commands, or defer them until we have a window.
    fn process_commands(
        &mut self,
        egui_ctx: &egui::Context,
        mut commands: Vec<egui::ViewportCommand>,
    ) {
        self.deferred_commands.append(&mut commands);

        if let Some(window) = self.window.as_ref() {
            egui_winit::process_viewport_commands(
                egui_ctx,
                &mut self.info,
                core::mem::take(&mut self.deferred_commands),
                window,
                &mut self.actions_requested,
            );
        }
    }

    /// Create winit window, if needed.
    fn initialize_window(
        &mut self,
        event_loop: &ActiveEventLoop,
        egui_ctx: &egui::Context,
        windows_id: &mut HashMap<WindowId, ViewportId>,
        painter: &mut egui_wgpu::winit::Painter,
    ) {
        if self.window.is_some() {
            return; // we already have one
        }

        profiling::function_scope!();

        let viewport_id = self.ids.this;

        match egui_winit::create_window(egui_ctx, event_loop, &self.builder) {
            Ok(window) => {
                windows_id.insert(window.id(), viewport_id);

                let window = Arc::new(window);

                if let Err(err) =
                    pollster::block_on(painter.set_window(viewport_id, Some(Arc::clone(&window))))
                {
                    log::error!("on set_window: viewport_id {viewport_id:?} {err}");
                }

                self.egui_winit = Some(egui_winit::State::new(
                    egui_ctx.clone(),
                    viewport_id,
                    event_loop,
                    Some(window.scale_factor() as f32),
                    event_loop.system_theme(),
                    painter.max_texture_side(),
                ));

                egui_winit::update_viewport_info(&mut self.info, egui_ctx, &window, true);
                self.window = Some(window);
            }
            Err(err) => {
                log::error!("Failed to create window: {err}");
            }
        }
    }
}

fn create_window(
    egui_ctx: &egui::Context,
    event_loop: &ActiveEventLoop,
    storage: Option<&dyn Storage>,
    native_options: &mut NativeOptions,
) -> Result<(Window, ViewportBuilder), winit::error::OsError> {
    profiling::function_scope!();

    let window_settings = epi_integration::load_window_settings(storage);
    let viewport_builder = epi_integration::viewport_builder(
        egui_ctx.zoom_factor(),
        event_loop,
        native_options,
        window_settings,
    )
    .with_visible(false); // Start hidden until we render the first frame to fix white flash on startup (https://github.com/emilk/egui/pull/3631)

    let window = egui_winit::create_window(egui_ctx, event_loop, &viewport_builder)?;
    epi_integration::apply_window_settings(&window, window_settings);
    Ok((window, viewport_builder))
}

/// Is this viewport, or any of its (transitive) descendant viewports, visible?
///
/// Immediate viewports are rendered inline while their parent's UI runs, so even
/// if this viewport's window is occluded or minimized we must still run its UI to
/// give any visible descendant a chance to be painted.
fn is_viewport_or_descendant_visible(viewports: &Viewports, viewport_id: ViewportId) -> bool {
    let Some(viewport) = viewports.get(&viewport_id) else {
        return false;
    };
    if viewport.info.visible().unwrap_or(true) {
        return true;
    }
    viewports.values().any(|child| {
        child.ids.parent == viewport_id
            && child.ids.this != viewport_id
            && is_viewport_or_descendant_visible(viewports, child.ids.this)
    })
}

fn render_immediate_viewport(
    beginning: Instant,
    shared: &RefCell<SharedState>,
    immediate_viewport: ImmediateViewport<'_>,
) {
    profiling::function_scope!();

    let ImmediateViewport {
        ids,
        builder,
        mut viewport_ui_cb,
    } = immediate_viewport;

    let input = {
        let SharedState {
            egui_ctx,
            viewports,
            painter,
            viewport_from_window,
            ..
        } = &mut *shared.borrow_mut();

        let viewport = initialize_or_update_viewport(
            viewports,
            ids,
            ViewportClass::Immediate,
            builder,
            None,
            painter,
        );
        if viewport.window.is_none() {
            event_loop_context::with_current_event_loop(|event_loop| {
                viewport.initialize_window(event_loop, egui_ctx, viewport_from_window, painter);
            });
        }

        let (Some(window), Some(egui_winit)) = (&viewport.window, &mut viewport.egui_winit) else {
            return;
        };
        egui_winit::update_viewport_info(&mut viewport.info, egui_ctx, window, false);

        let mut input = egui_winit.take_egui_input(window);
        input.viewports = viewports
            .iter()
            .map(|(id, viewport)| (*id, viewport.info.clone()))
            .collect();
        input.time = Some(beginning.elapsed().as_secs_f64());
        input
    };

    let egui_ctx = shared.borrow().egui_ctx.clone();

    // ------------------------------------------

    // Run the user code, which could re-entrantly call this function again (!).
    // Make sure no locks are held during this call.
    let egui::FullOutput {
        platform_output,
        textures_delta,
        shapes,
        pixels_per_point,
        viewport_output,
    } = egui_ctx.run_ui(input, |ui| {
        viewport_ui_cb(ui);
    });

    // ------------------------------------------

    let mut shared_mut = shared.borrow_mut();
    let SharedState {
        viewports,
        painter,
        viewport_from_window,
        ..
    } = &mut *shared_mut;

    let Some(viewport) = viewports.get_mut(&ids.this) else {
        warn!("Viewport disappeared unexpectedly!");
        return;
    };
    viewport.pending_delta.append(textures_delta);

    viewport.info.events.clear(); // they should have been processed
    let (Some(egui_winit), Some(window)) = (&mut viewport.egui_winit, &viewport.window) else {
        return;
    };

    {
        profiling::scope!("set_window");
        if let Err(err) = pollster::block_on(painter.set_window(ids.this, Some(Arc::clone(window))))
        {
            log::error!(
                "when rendering viewport_id={:?}, set_window Error {err}",
                ids.this
            );
        }
    }

    let clipped_primitives = egui_ctx.tessellate(shapes, pixels_per_point);
    painter.paint_and_update_textures(
        ids.this,
        pixels_per_point,
        [0.0, 0.0, 0.0, 0.0],
        &clipped_primitives,
        &mut viewport.pending_delta,
        vec![],
        window,
    );

    egui_winit.handle_platform_output(window, platform_output);

    handle_viewport_output(
        &egui_ctx,
        &viewport_output,
        viewports,
        painter,
        viewport_from_window,
    );
}

#[cfg(test)]
mod retained_paint_tests {
    use super::{
        MonitorRefreshCache, UserEvent, multipass_repaint_request, retained_paint_deadline,
    };
    use crate::native::epi_integration::retained_paint_needs_full_ui;
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, Instant};

    #[test]
    fn delayed_repaint_survives_a_discarded_egui_pass() {
        let ctx = egui::Context::default();
        for _ in 0..4 {
            let mut output = ctx.run_ui(egui::RawInput::default(), |_| {});
            output.textures_delta.clear();
        }
        let requests = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&requests);
        ctx.set_request_repaint_callback(move |request| {
            captured.lock().unwrap().push(request);
        });

        let mut pass = 0;
        let frame_started = Instant::now();
        let mut output = ctx.run_ui(egui::RawInput::default(), |ui| {
            if pass == 0 {
                ui.ctx().request_repaint_after(Duration::from_millis(40));
                ui.ctx().request_discard("verify multipass repaint deadline");
            }
            pass += 1;
        });
        output.textures_delta.clear();
        assert_eq!(pass, 2);
        let completed_pass_nr = ctx.cumulative_pass_nr_for(egui::ViewportId::ROOT);
        let original_request = requests
            .lock()
            .unwrap()
            .iter()
            .copied()
            .find(|request| request.delay > Duration::ZERO)
            .expect("first pass requested a delayed repaint");
        assert_eq!(
            completed_pass_nr,
            original_request.current_cumulative_pass_nr + 2,
            "the event loop would reject the first-pass callback as stale"
        );
        let delay = output.viewport_output[&egui::ViewportId::ROOT].repaint_delay;
        let recovered = multipass_repaint_request(
            &output,
            egui::ViewportId::ROOT,
            completed_pass_nr,
            frame_started,
        )
        .expect("the combined output retains the first pass's deadline");
        match recovered {
            UserEvent::RequestRepaint {
                when,
                cumulative_pass_nr,
                viewport_id,
            } => {
                assert_eq!(when, frame_started + delay);
                assert_eq!(cumulative_pass_nr + 1, completed_pass_nr);
                assert_eq!(viewport_id, egui::ViewportId::ROOT);
            }
            #[cfg(feature = "accesskit")]
            UserEvent::AccessKitActionRequest(_) => panic!("expected a repaint request"),
        }
    }

    #[test]
    fn present_wait_does_not_add_a_second_frame_interval() {
        let frame_started = Instant::now();
        let cadence = Duration::from_secs_f64(1.0 / 120.0);
        let frame_finished = frame_started + Duration::from_secs_f64(1.0 / 60.0);
        assert_eq!(
            retained_paint_deadline(frame_started, cadence),
            frame_started + cadence
        );
        assert!(retained_paint_deadline(frame_started, cadence) < frame_finished);
    }

    #[test]
    fn delayed_ui_timer_does_not_prevent_cached_gpu_paint() {
        let ctx = egui::Context::default();
        for _ in 0..4 {
            let mut output = ctx.run_ui(egui::RawInput::default(), |_| {});
            output.textures_delta.clear();
        }

        let mut output = ctx.run_ui(egui::RawInput::default(), |ui| {
            ui.ctx().request_repaint_after(Duration::from_secs(10));
        });
        output.textures_delta.clear();
        assert!(ctx.has_requested_repaint_for(&egui::ViewportId::ROOT));
        assert!(!retained_paint_needs_full_ui(&ctx, None));

        let mut output = ctx.run_ui(egui::RawInput::default(), |ui| {
            ui.ctx().request_repaint();
        });
        output.textures_delta.clear();
        let mut output = ctx.run_ui(egui::RawInput::default(), |_| {});
        output.textures_delta.clear();
        assert!(retained_paint_needs_full_ui(&ctx, None));
    }

    #[test]
    fn monitor_refresh_is_queried_at_most_once_per_two_seconds() {
        let mut cache = MonitorRefreshCache::default();
        let start = Instant::now();
        assert_eq!(cache.get_or_sample(start, || Some(120_000)), Some(120_000));
        assert_eq!(
            cache.get_or_sample(start + Duration::from_secs(1), || {
                panic!("refresh rate queried before cache expired")
            }),
            Some(120_000)
        );
        assert_eq!(
            cache.get_or_sample(start + Duration::from_secs(2), || Some(60_000)),
            Some(60_000)
        );
    }

    #[test]
    fn unavailable_monitor_refresh_is_cached_and_zero_is_ignored() {
        let mut cache = MonitorRefreshCache::default();
        let start = Instant::now();
        assert_eq!(cache.get_or_sample(start, || None), None);
        assert_eq!(
            cache.get_or_sample(start + Duration::from_secs(1), || {
                panic!("unavailable result queried before cache expired")
            }),
            None
        );
        assert_eq!(
            cache.get_or_sample(start + Duration::from_secs(2), || Some(0)),
            None
        );
    }
}

pub(crate) fn remove_viewports_not_in(
    viewports: &mut Viewports,
    painter: &mut egui_wgpu::winit::Painter,
    viewport_from_window: &mut HashMap<WindowId, ViewportId>,
    viewport_output: &OrderedViewportIdMap<ViewportOutput>,
) {
    let active_viewports_ids: ViewportIdSet = viewport_output.keys().copied().collect();

    // Prune dead viewports:
    viewports.retain(|id, _| active_viewports_ids.contains(id));
    viewport_from_window.retain(|_, id| active_viewports_ids.contains(id));
    painter.gc_viewports(&active_viewports_ids);
}

/// Add new viewports, and update existing ones:
fn handle_viewport_output(
    egui_ctx: &egui::Context,
    viewport_output: &OrderedViewportIdMap<ViewportOutput>,
    viewports: &mut Viewports,
    painter: &mut egui_wgpu::winit::Painter,
    viewport_from_window: &mut HashMap<WindowId, ViewportId>,
) {
    for (
        viewport_id,
        ViewportOutput {
            parent,
            class,
            builder,
            viewport_ui_cb,
            commands,
            repaint_delay: _, // ignored - we listened to the repaint callback instead
        },
    ) in viewport_output.clone()
    {
        let ids = ViewportIdPair::from_self_and_parent(viewport_id, parent);

        let viewport =
            initialize_or_update_viewport(viewports, ids, class, builder, viewport_ui_cb, painter);

        let old_inner_size = viewport.window.as_ref().map(|window| window.inner_size());

        viewport.process_commands(egui_ctx, commands);

        // For Wayland : https://github.com/emilk/egui/issues/4196
        if cfg!(target_os = "linux")
            && let Some(window) = viewport.window.as_ref()
            && let Some(old_inner_size) = old_inner_size
        {
            let new_inner_size = window.inner_size();
            if new_inner_size != old_inner_size
                && let (Some(width), Some(height)) = (
                    NonZeroU32::new(new_inner_size.width),
                    NonZeroU32::new(new_inner_size.height),
                )
            {
                painter.on_window_resized(viewport_id, width, height);
            }
        }
    }

    remove_viewports_not_in(viewports, painter, viewport_from_window, viewport_output);
}

fn initialize_or_update_viewport<'a>(
    viewports: &'a mut Viewports,
    ids: ViewportIdPair,
    class: ViewportClass,
    mut builder: ViewportBuilder,
    viewport_ui_cb: Option<Arc<dyn Fn(&mut egui::Ui) + Send + Sync>>,
    painter: &mut egui_wgpu::winit::Painter,
) -> &'a mut Viewport {
    use std::collections::btree_map::Entry;

    profiling::function_scope!();

    if builder.icon.is_none() {
        // Inherit icon from parent
        builder.icon = viewports
            .get_mut(&ids.parent)
            .and_then(|vp| vp.builder.icon.clone());
    }

    match viewports.entry(ids.this) {
        Entry::Vacant(entry) => {
            // New viewport:
            log::debug!("Creating new viewport {:?} ({:?})", ids.this, builder.title);
            entry.insert(Viewport {
                ids,
                class,
                builder,
                deferred_commands: vec![],
                info: Default::default(),
                actions_requested: Vec::new(),
                viewport_ui_cb,
                window: None,
                egui_winit: None,
                pending_delta: Default::default(),
            })
        }

        Entry::Occupied(mut entry) => {
            // Patch an existing viewport:
            let viewport = entry.get_mut();

            viewport.class = class;
            viewport.ids.parent = ids.parent;
            viewport.viewport_ui_cb = viewport_ui_cb;

            let (mut delta_commands, recreate) = viewport.builder.patch(builder);

            if recreate {
                log::debug!(
                    "Recreating window for viewport {:?} ({:?})",
                    ids.this,
                    viewport.builder.title
                );
                viewport.window = None;
                viewport.egui_winit = None;
                if let Err(err) = pollster::block_on(painter.set_window(viewport.ids.this, None)) {
                    log::error!(
                        "when rendering viewport_id={:?}, set_window Error {err}",
                        viewport.ids.this
                    );
                }
            }

            viewport.deferred_commands.append(&mut delta_commands);

            entry.into_mut()
        }
    }
}
