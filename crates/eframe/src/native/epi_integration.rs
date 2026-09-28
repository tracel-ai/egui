//! Common tools used by [`super::glow_integration`] and [`super::wgpu_integration`].

use web_time::Instant;

use core::time::Duration;
use std::{path::PathBuf, sync::Arc};
use winit::event_loop::ActiveEventLoop;

use raw_window_handle::{HasDisplayHandle as _, HasWindowHandle as _};

use egui::{DeferredViewportUiCallback, ViewportBuilder, ViewportId};
use egui_winit::{EventResponse, WindowSettings};

use super::winit_integration::UserEvent;
use crate::epi;

#[cfg_attr(target_os = "ios", allow(dead_code, unused_variables, unused_mut))]
pub fn viewport_builder(
    egui_zoom_factor: f32,
    event_loop: &ActiveEventLoop,
    native_options: &mut epi::NativeOptions,
    window_settings: Option<WindowSettings>,
) -> ViewportBuilder {
    profiling::function_scope!();

    let mut viewport_builder = native_options.viewport.clone();

    // On some Linux systems, a window size larger than the monitor causes crashes,
    // and on Windows the window does not appear at all.
    let clamp_size_to_monitor_size = viewport_builder.clamp_size_to_monitor_size.unwrap_or(true);

    // Always use the default window size / position on iOS. Trying to restore the previous position
    // causes the window to be shown too small.
    #[cfg(not(target_os = "ios"))]
    let inner_size_points = if let Some(mut window_settings) = window_settings {
        // Restore pos/size from previous session

        if clamp_size_to_monitor_size {
            window_settings.clamp_size_to_sane_values(largest_monitor_point_size(
                egui_zoom_factor,
                event_loop,
            ));
        }
        window_settings.clamp_position_to_monitors(egui_zoom_factor, event_loop);

        viewport_builder = window_settings.initialize_viewport_builder(
            egui_zoom_factor,
            event_loop,
            viewport_builder,
        );
        window_settings.inner_size_points()
    } else {
        if let Some(pos) = viewport_builder.position {
            viewport_builder = viewport_builder.with_position(pos);
        }

        if clamp_size_to_monitor_size && let Some(initial_window_size) = viewport_builder.inner_size
        {
            let initial_window_size = egui::NumExt::at_most(
                initial_window_size,
                largest_monitor_point_size(egui_zoom_factor, event_loop),
            );
            viewport_builder = viewport_builder.with_inner_size(initial_window_size);
        }

        viewport_builder.inner_size
    };

    #[cfg(not(target_os = "ios"))]
    if native_options.centered {
        profiling::scope!("center");
        if let Some(monitor) = event_loop
            .primary_monitor()
            .or_else(|| event_loop.available_monitors().next())
        {
            let monitor_size = monitor
                .size()
                .to_logical::<f32>(egui_zoom_factor as f64 * monitor.scale_factor());
            let inner_size = inner_size_points.unwrap_or(egui::Vec2 { x: 800.0, y: 600.0 });
            if 0.0 < monitor_size.width && 0.0 < monitor_size.height {
                let x = (monitor_size.width - inner_size.x) / 2.0;
                let y = (monitor_size.height - inner_size.y) / 2.0;
                viewport_builder = viewport_builder.with_position([x, y]);
            }
        }
    }

    match core::mem::take(&mut native_options.window_builder) {
        Some(hook) => hook(viewport_builder),
        None => viewport_builder,
    }
}

pub fn apply_window_settings(
    window: &winit::window::Window,
    window_settings: Option<WindowSettings>,
) {
    profiling::function_scope!();
    if let Some(window_settings) = window_settings {
        window_settings.initialize_window(window);
    }
}

#[cfg(not(target_os = "ios"))]
fn largest_monitor_point_size(egui_zoom_factor: f32, event_loop: &ActiveEventLoop) -> egui::Vec2 {
    profiling::function_scope!();
    let mut max_size = egui::Vec2::ZERO;

    let available_monitors = {
        profiling::scope!("available_monitors");
        event_loop.available_monitors()
    };

    for monitor in available_monitors {
        let size = monitor
            .size()
            .to_logical::<f32>(egui_zoom_factor as f64 * monitor.scale_factor());
        let size = egui::vec2(size.width, size.height);
        max_size = max_size.max(size);
    }

    if max_size == egui::Vec2::ZERO {
        egui::Vec2::splat(16000.0)
    } else {
        max_size
    }
}

// ----------------------------------------------------------------------------

/// For loading/saving app state and/or egui memory to disk.
pub fn create_storage(_app_name: &str) -> Option<Box<dyn epi::Storage>> {
    #[cfg(feature = "persistence")]
    if let Some(storage) = super::file_storage::FileStorage::from_app_id(_app_name) {
        return Some(Box::new(storage));
    }
    None
}

#[allow(clippy::allow_attributes, clippy::unnecessary_wraps)]
pub fn create_storage_with_file(_file: impl Into<PathBuf>) -> Option<Box<dyn epi::Storage>> {
    #[cfg(feature = "persistence")]
    return Some(Box::new(
        super::file_storage::FileStorage::from_ron_filepath(_file),
    ));
    #[cfg(not(feature = "persistence"))]
    None
}

// ----------------------------------------------------------------------------

/// Ueye patch (DESIGN.md 9.8): schedules the root viewport's passes under
/// `NativeOptions::one_pass_per_input`.
///
/// egui asks for an immediate repaint whenever a pass had input, and every
/// immediate request costs two more passes. Under the watch, a root pass
/// settles in that pass: nothing it asks for is posted to the event loop
/// while it runs; after it, one repaint is posted for what the app and its
/// plugins asked for while it ran, keeping egui's own follow-up only while
/// scrolling, touching, dragging or hovering files. Other threads, and any
/// request between passes, still wake the event loop right away (from egui's
/// repaint observer: egui's request callback only sees the requests sooner
/// than the delay it last scheduled, which the watch may have dropped).
#[derive(Debug)]
pub struct RepaintWatch {
    state: parking_lot::Mutex<WatchState>,
}

#[derive(Debug)]
struct WatchState {
    /// The thread running the root pass, while it runs.
    pass_thread: Option<std::thread::ThreadId>,

    /// When the pass started.
    started: Instant,

    /// The app's closure has started (egui's own `begin_pass` requests come before).
    logic: bool,

    /// Soonest delay the pass asked for since the app's closure started, or
    /// egui's own non-zero requests.
    soonest: Duration,

    /// The soonest repaint posted since the last root pass began.
    posted: Option<Instant>,
}

impl Default for RepaintWatch {
    fn default() -> Self {
        Self {
            state: parking_lot::Mutex::new(WatchState {
                pass_thread: None,
                started: Instant::now(),
                logic: false,
                soonest: Duration::MAX,
                posted: None,
            }),
        }
    }
}

impl WatchState {
    /// The event that wakes the event loop at `when`, unless a sooner one
    /// was already posted.
    fn post(&mut self, when: Instant, cumulative_pass_nr: u64) -> Option<UserEvent> {
        if self.posted.is_some_and(|posted| posted <= when) {
            return None;
        }
        self.posted = Some(when);
        Some(UserEvent::RequestRepaint {
            viewport_id: ViewportId::ROOT,
            when,
            cumulative_pass_nr,
        })
    }
}

impl RepaintWatch {
    /// Handles a request egui's repaint observer sees: records the ones the
    /// running root pass makes, and returns the event to post for the others.
    pub fn observe(&self, info: egui::RequestRepaintInfo) -> Option<UserEvent> {
        if info.viewport_id != ViewportId::ROOT {
            return None; // egui's request callback posts it
        }
        let mut state = self.state.lock();
        if state.pass_thread == Some(std::thread::current().id()) {
            if state.logic || !info.delay.is_zero() {
                state.soonest = state.soonest.min(info.delay);
            }
            return None;
        }
        let when = Instant::now().checked_add(info.delay)?;
        state.post(when, info.current_cumulative_pass_nr)
    }

    /// Whether egui's request callback posts `info`: the watch posts the
    /// root's requests.
    pub fn callback_posts(watch: Option<&Self>, info: &egui::RequestRepaintInfo) -> bool {
        watch.is_none() || info.viewport_id != ViewportId::ROOT
    }

    /// Runs a root pass under the watch, as [`EpiIntegration::update`] does:
    /// returns its output, whose root `repaint_delay` is the one scheduled,
    /// and the event to post for it.
    pub fn run_ui(
        &self,
        egui_ctx: &egui::Context,
        raw_input: egui::RawInput,
        mut run_ui: impl FnMut(&mut egui::Ui),
    ) -> (egui::FullOutput, Option<UserEvent>) {
        {
            let mut state = self.state.lock();
            state.pass_thread = Some(std::thread::current().id());
            state.started = Instant::now();
            state.logic = false;
            state.soonest = Duration::MAX;
            // egui forgets the delay it scheduled when a pass begins, too.
            state.posted = None;
        }
        let mut full_output = egui_ctx.run_ui(raw_input, |ui| {
            self.state.lock().logic = true;
            run_ui(ui);
        });
        let needs_follow_up = egui_ctx.input(input_needs_follow_up);
        // The pass that ended is the one before this number, which the event
        // loop accepts as current.
        let pass_nr = egui_ctx
            .cumulative_pass_nr_for(ViewportId::ROOT)
            .saturating_sub(1);

        let mut state = self.state.lock();
        state.pass_thread = None;
        let root = full_output.viewport_output.get_mut(&ViewportId::ROOT);
        let egui_delay = root
            .as_ref()
            .map_or(Duration::MAX, |root| root.repaint_delay);
        let delay = if egui_delay.is_zero() && !needs_follow_up {
            state.soonest
        } else {
            egui_delay
        };
        if let Some(root) = root {
            root.repaint_delay = delay;
        }
        let request = state
            .started
            .checked_add(delay)
            .and_then(|when| state.post(when, pass_nr));
        (full_output, request)
    }

    /// Before a logic-only run of the root (the window is hidden): egui
    /// forgets the delay it scheduled, and so does the watch.
    fn begin_logic_only(&self) {
        self.state.lock().posted = None;
    }

    /// Whether a repaint posted since the last root pass began is due.
    fn repaint_due(&self, now: Instant) -> bool {
        self.state.lock().posted.is_some_and(|when| when <= now)
    }
}

/// Ueye patch: installs egui's request callback, and the repaint observer of
/// `watch`; both hand the events to post to `post`, which is returned for
/// the requests the watch posts after a pass.
pub fn install_repaint_callbacks(
    egui_ctx: &egui::Context,
    watch: Option<Arc<RepaintWatch>>,
    post: impl Fn(UserEvent) + Send + Sync + 'static,
) -> Arc<dyn Fn(UserEvent) + Send + Sync> {
    let post: Arc<dyn Fn(UserEvent) + Send + Sync> = Arc::new(post);
    if let Some(watch) = watch.clone() {
        let post = Arc::clone(&post);
        egui_ctx.set_repaint_observer(move |info| {
            if let Some(event) = watch.observe(info) {
                post(event);
            }
        });
    }
    let callback_post = Arc::clone(&post);
    egui_ctx.set_request_repaint_callback(move |info| {
        log::trace!("request_repaint_callback: {info:?}");
        if RepaintWatch::callback_posts(watch.as_deref(), &info) {
            callback_post(UserEvent::RequestRepaint {
                when: Instant::now() + info.delay,
                cumulative_pass_nr: info.current_cumulative_pass_nr,
                viewport_id: info.viewport_id,
            });
        }
    });
    post
}

/// Ueye patch: whether a retained paint-only frame must give way to a full
/// UI pass because a repaint was asked for.
pub fn retained_paint_needs_full_ui(
    egui_ctx: &egui::Context,
    watch: Option<&RepaintWatch>,
) -> bool {
    match watch {
        // egui's own flag describes the pass before the last one, and stays
        // set after the input passes whose follow-up the watch dropped.
        Some(watch) => watch.repaint_due(Instant::now()),
        // A future `request_repaint_after` also makes `has_requested_repaint_for`
        // true. The event loop already tracks that deadline and prioritizes its
        // full repaint when due. Forcing a UI pass on every paint-only animation
        // tick until then would defeat mesh retention.
        None => egui_ctx.requested_repaint_last_pass_for(&ViewportId::ROOT),
    }
}

/// Ueye patch: whether egui's own follow-up of an input pass is still
/// needed: something keeps moving without new input.
fn input_needs_follow_up(input: &egui::InputState) -> bool {
    input.is_scrolling()
        || input.any_touches()
        || input.pointer.any_down()
        || !input.raw.hovered_files.is_empty()
}

/// Everything needed to make a winit-based integration for [`epi`].
///
/// Only one instance per app (not one per viewport).
pub struct EpiIntegration {
    pub frame: epi::Frame,
    last_auto_save: Instant,
    pub beginning: Instant,
    is_first_frame: bool,
    pub egui_ctx: egui::Context,

    /// Input that we have received, but not yet given to egui,
    /// because we haven't run any pass since (see [`Self::update_logic_only`]).
    pending_raw_input: egui::RawInput,

    pending_full_output: egui::FullOutput,

    /// When set, it is time to close the native window.
    close: bool,

    can_drag_window: bool,
    #[cfg(feature = "persistence")]
    persist_window: bool,
    app_icon_setter: super::app_icon::AppTitleIconSetter,

    /// Ueye patch: see [`RepaintWatch`]; `None` keeps egui's scheduling.
    pub repaint_watch: Option<Arc<RepaintWatch>>,

    /// Ueye patch: posts repaint requests to the event loop (see
    /// [`Self::install_repaint_callbacks`]).
    repaint_post: Option<Arc<dyn Fn(UserEvent) + Send + Sync>>,
}

impl EpiIntegration {
    #[allow(clippy::allow_attributes, clippy::too_many_arguments)]
    pub fn new(
        egui_ctx: egui::Context,
        window: &Arc<winit::window::Window>,
        app_name: &str,
        native_options: &crate::NativeOptions,
        storage: Option<Box<dyn epi::Storage>>,
        #[cfg(feature = "glow")] gl: Option<std::sync::Arc<glow::Context>>,
        #[cfg(feature = "glow")] glow_register_native_texture: Option<
            Box<dyn FnMut(glow::Texture) -> egui::TextureId>,
        >,
        #[cfg(feature = "wgpu_no_default_features")] wgpu_render_state: Option<
            egui_wgpu::RenderState,
        >,
    ) -> Self {
        let frame = epi::Frame {
            info: epi::IntegrationInfo { cpu_usage: None },
            storage,
            #[cfg(feature = "glow")]
            gl,
            #[cfg(feature = "glow")]
            glow_register_native_texture,
            #[cfg(feature = "wgpu_no_default_features")]
            wgpu_render_state,
            window: Some(Arc::clone(window)),
            raw_display_handle: window.display_handle().map(|h| h.as_raw()),
            raw_window_handle: window.window_handle().map(|h| h.as_raw()),
        };

        let icon = native_options
            .viewport
            .icon
            .clone()
            .unwrap_or_else(|| std::sync::Arc::new(load_default_egui_icon()));

        let app_icon_setter = super::app_icon::AppTitleIconSetter::new(
            native_options
                .viewport
                .title
                .clone()
                .unwrap_or_else(|| app_name.to_owned()),
            Some(icon),
        );

        Self {
            frame,
            last_auto_save: Instant::now(),
            pending_raw_input: Default::default(),
            pending_full_output: Default::default(),
            close: false,
            can_drag_window: false,
            #[cfg(feature = "persistence")]
            persist_window: native_options.persist_window,
            app_icon_setter,
            repaint_watch: native_options
                .one_pass_per_input
                .then(|| Arc::new(RepaintWatch::default())),
            repaint_post: None,
            beginning: Instant::now()
                .checked_sub(web_time::Duration::from_secs_f64(egui_ctx.time()))
                .unwrap_or_else(Instant::now),
            is_first_frame: true,
            egui_ctx,
        }
    }

    /// Ueye patch: installs egui's repaint callbacks, which wake the event
    /// loop through `post` (see [`install_repaint_callbacks`]).
    pub fn install_repaint_callbacks(&mut self, post: impl Fn(UserEvent) + Send + Sync + 'static) {
        self.repaint_post = Some(install_repaint_callbacks(
            &self.egui_ctx,
            self.repaint_watch.clone(),
            post,
        ));
    }

    /// Ueye patch: see [`retained_paint_needs_full_ui`].
    pub fn retained_paint_needs_full_ui(&self) -> bool {
        retained_paint_needs_full_ui(&self.egui_ctx, self.repaint_watch.as_deref())
    }

    /// If `true`, it is time to close the native window.
    pub fn should_close(&self) -> bool {
        self.close
    }

    pub fn on_window_event(
        &mut self,
        window: &winit::window::Window,
        egui_winit: &mut egui_winit::State,
        event: &winit::event::WindowEvent,
    ) -> EventResponse {
        profiling::function_scope!(egui_winit::short_window_event_description(event));

        use winit::event::{ElementState, MouseButton, WindowEvent};

        if let WindowEvent::MouseInput {
            button: MouseButton::Left,
            state: ElementState::Pressed,
            ..
        } = event
        {
            self.can_drag_window = true;
        }

        egui_winit.on_window_event(window, event)
    }

    pub fn pre_update(&mut self) {
        self.app_icon_setter.update();
    }

    /// Run user code - this can create immediate viewports, so hold no locks over this!
    ///
    /// If `viewport_ui_cb` is None, we are in the root viewport and will call
    /// [`crate::App::logic`] and [`crate::App::ui`].
    ///
    /// Only call this when the ui will actually be shown;
    /// use [`Self::update_logic_only`] otherwise.
    pub fn update(
        &mut self,
        app: &mut dyn epi::App,
        viewport_ui_cb: Option<&DeferredViewportUiCallback>,
        raw_input: egui::RawInput,
    ) -> egui::FullOutput {
        let raw_input = self.prepare_raw_input(app, raw_input);

        let close_requested = raw_input.viewport().close_requested();

        let is_root_viewport = viewport_ui_cb.is_none();

        let frame = &mut self.frame;
        let run_ui = |ui: &mut egui::Ui| {
            if let Some(viewport_ui_cb) = viewport_ui_cb {
                // Child viewport
                profiling::scope!("viewport_callback");
                viewport_ui_cb(ui);
            } else {
                {
                    profiling::scope!("App::logic");
                    app.logic(ui.ctx(), frame);
                }
                {
                    profiling::scope!("App::ui");
                    app.ui(ui, frame);
                }
            }
        };
        // Ueye patch: a root pass runs under the watch, which schedules the
        // next one (see [`RepaintWatch`]).
        let full_output = match self.repaint_watch.as_ref().filter(|_| is_root_viewport) {
            Some(watch) => {
                let (full_output, request) = watch.run_ui(&self.egui_ctx, raw_input, run_ui);
                if let (Some(request), Some(post)) = (request, &self.repaint_post) {
                    post(request);
                }
                full_output
            }
            None => self.egui_ctx.run_ui(raw_input, run_ui),
        };

        if is_root_viewport && close_requested {
            let canceled = full_output.viewport_output[&ViewportId::ROOT]
                .commands
                .contains(&egui::ViewportCommand::CancelClose);
            self.handle_close_request(canceled);
        }

        self.pending_full_output.append(full_output);
        core::mem::take(&mut self.pending_full_output)
    }

    /// Let the app tick its logic without showing any ui,
    /// because the window is minimized or occluded.
    ///
    /// No egui pass is run, so all ui state is left untouched:
    /// the app will find everything where it left it once the window is visible again.
    ///
    /// Only call this for the root viewport: only it has [`crate::App::logic`].
    pub fn update_logic_only(
        &mut self,
        app: &mut dyn epi::App,
        raw_input: egui::RawInput,
    ) -> egui::LogicOutput {
        let raw_input = self.prepare_raw_input(app, raw_input);

        let close_requested = raw_input.viewport().close_requested();

        if let Some(watch) = &self.repaint_watch {
            watch.begin_logic_only();
        }
        let logic_output = self.egui_ctx.run_logic(&raw_input, |ctx| {
            profiling::scope!("App::logic");
            app.logic(ctx, &mut self.frame);
        });

        // No pass consumed the input, so save it for the next one:
        self.pending_raw_input = raw_input;

        if close_requested {
            let canceled = logic_output
                .viewport_commands
                .get(&ViewportId::ROOT)
                .is_some_and(|commands| commands.contains(&egui::ViewportCommand::CancelClose));
            self.handle_close_request(canceled);
        }

        logic_output
    }

    /// Prepend any input we couldn't give to egui earlier, set the time, and run the app hook.
    fn prepare_raw_input(
        &mut self,
        app: &mut dyn epi::App,
        new_input: egui::RawInput,
    ) -> egui::RawInput {
        let mut raw_input = core::mem::take(&mut self.pending_raw_input);
        raw_input.append(new_input); // The new input wins where they overlap

        raw_input.time = Some(self.beginning.elapsed().as_secs_f64());

        app.raw_input_hook(&self.egui_ctx, &mut raw_input);

        raw_input
    }

    fn handle_close_request(&mut self, canceled: bool) {
        if canceled {
            log::debug!("Closing of root viewport canceled with ViewportCommand::CancelClose");
        } else {
            log::debug!("Closing root viewport (ViewportCommand::CancelClose was not sent)");
            self.close = true;
        }
    }

    pub fn report_frame_time(&mut self, seconds: f32) {
        self.frame.info.cpu_usage = Some(seconds);
    }

    pub fn post_rendering(&mut self, window: &winit::window::Window) {
        profiling::function_scope!();
        if core::mem::take(&mut self.is_first_frame) {
            // We keep hidden until we've painted something. See https://github.com/emilk/egui/pull/2279
            window.set_visible(true);
        }
    }

    // ------------------------------------------------------------------------
    // Persistence stuff:

    pub fn maybe_autosave(
        &mut self,
        app: &mut dyn epi::App,
        window: Option<&winit::window::Window>,
    ) {
        let now = Instant::now();
        if now - self.last_auto_save > app.auto_save_interval() {
            self.save(app, window);
            self.last_auto_save = now;
        }
    }

    pub fn save(&mut self, app: &mut dyn epi::App, window: Option<&winit::window::Window>) {
        #[cfg(not(feature = "persistence"))]
        let _ = (self, app, window);

        #[cfg(feature = "persistence")]
        if let Some(storage) = self.frame.storage_mut() {
            profiling::function_scope!();

            if let Some(window) = window
                && self.persist_window
            {
                profiling::scope!("native_window");
                epi::set_value(
                    storage,
                    STORAGE_WINDOW_KEY,
                    &WindowSettings::from_window(self.egui_ctx.zoom_factor(), window),
                );
            }
            if app.persist_egui_memory() {
                profiling::scope!("egui_memory");
                self.egui_ctx
                    .memory(|mem| epi::set_value(storage, STORAGE_EGUI_MEMORY_KEY, mem));
            }
            {
                profiling::scope!("App::save");
                app.save(storage);
            }

            profiling::scope!("Storage::flush");
            storage.flush();
        }
    }
}

fn load_default_egui_icon() -> egui::IconData {
    profiling::function_scope!();
    #[expect(clippy::unwrap_used)]
    crate::icon_data::from_png_bytes(&include_bytes!("../../data/icon.png")[..]).unwrap()
}

#[cfg(feature = "persistence")]
const STORAGE_EGUI_MEMORY_KEY: &str = "egui";

#[cfg(feature = "persistence")]
const STORAGE_WINDOW_KEY: &str = "window";

pub fn load_window_settings(_storage: Option<&dyn epi::Storage>) -> Option<WindowSettings> {
    profiling::function_scope!();
    #[cfg(feature = "persistence")]
    {
        epi::get_value(_storage?, STORAGE_WINDOW_KEY)
    }
    #[cfg(not(feature = "persistence"))]
    None
}

pub fn load_egui_memory(_storage: Option<&dyn epi::Storage>) -> Option<egui::Memory> {
    profiling::function_scope!();
    #[cfg(feature = "persistence")]
    {
        epi::get_value(_storage?, STORAGE_EGUI_MEMORY_KEY)
    }
    #[cfg(not(feature = "persistence"))]
    None
}

#[cfg(test)]
mod one_pass_per_input_tests {
    //! Ueye patch: with `one_pass_per_input`, the native integrations wake
    //! the event loop once after a root pass, for what the app asked for
    //! while it ran (DESIGN.md 9.8).
    #![expect(
        clippy::disallowed_methods,
        reason = "the tests ask for repaints as an app does"
    )]

    use super::{RepaintWatch, install_repaint_callbacks, retained_paint_needs_full_ui};
    use crate::native::{run::repaint_request_is_current, winit_integration::UserEvent};
    use core::time::Duration;
    use egui::{Event, Modifiers, MouseWheelUnit, RawInput, TouchPhase, ViewportId, pos2, vec2};
    use std::sync::{Arc, Mutex};
    use std::time::Instant;

    /// A root viewport run as the native integrations run it, keeping the
    /// repaint requests they post to the event loop.
    struct Root {
        ctx: egui::Context,
        watch: Arc<RepaintWatch>,
        posted: Arc<Mutex<Vec<UserEvent>>>,
    }

    impl Root {
        fn new() -> Self {
            let ctx = egui::Context::default();
            let watch = Arc::new(RepaintWatch::default());
            let posted = Arc::new(Mutex::new(Vec::new()));
            let sink = Arc::clone(&posted);
            install_repaint_callbacks(&ctx, Some(Arc::clone(&watch)), move |event| {
                sink.lock().expect("posted lock").push(event);
            });
            let root = Self { ctx, watch, posted };
            // Past egui's start-up passes.
            for _ in 0..4 {
                root.pass(RawInput::default(), |_| {});
            }
            root
        }

        /// Runs a root pass as `EpiIntegration::update` does; returns the
        /// repaints posted during and after it that the event loop accepts.
        fn pass(&self, input: RawInput, mut app: impl FnMut(&mut egui::Ui)) -> Vec<Duration> {
            let started = Instant::now();
            let (mut output, request) = self.watch.run_ui(&self.ctx, input, |ui| app(ui));
            output.textures_delta.clear();
            if let Some(request) = request {
                self.posted.lock().expect("posted lock").push(request);
            }
            self.take(started)
        }

        /// The repaints posted since the last call that the event loop
        /// (`run.rs`) accepts now, as delays from `since`.
        fn take(&self, since: Instant) -> Vec<Duration> {
            let current = self.ctx.cumulative_pass_nr_for(ViewportId::ROOT);
            core::mem::take(&mut *self.posted.lock().expect("posted lock"))
                .into_iter()
                .filter_map(|event| match event {
                    UserEvent::RequestRepaint {
                        viewport_id,
                        when,
                        cumulative_pass_nr,
                    } => (viewport_id == ViewportId::ROOT
                        && repaint_request_is_current(current, cumulative_pass_nr))
                    .then(|| when.saturating_duration_since(since)),
                    #[cfg(feature = "accesskit")]
                    UserEvent::AccessKitActionRequest(_) => None,
                })
                .collect()
        }

        fn needs_full_ui(&self) -> bool {
            retained_paint_needs_full_ui(&self.ctx, Some(&self.watch))
        }
    }

    fn pointer_moved() -> RawInput {
        RawInput {
            events: vec![Event::PointerMoved(pos2(10.0, 10.0))],
            ..Default::default()
        }
    }

    fn scrolled() -> RawInput {
        RawInput {
            events: vec![Event::MouseWheel {
                unit: MouseWheelUnit::Point,
                delta: vec2(0.0, -40.0),
                phase: TouchPhase::Move,
                modifiers: Modifiers::NONE,
            }],
            ..Default::default()
        }
    }

    /// egui brings a delayed repaint forward by its predicted frame time.
    fn predicted_frame() -> Duration {
        Duration::from_secs_f32(RawInput::default().predicted_dt)
    }

    fn is_immediate(delay: &Duration) -> bool {
        *delay < Duration::from_millis(100)
    }

    fn is_about(delay: &Duration, expected: Duration) -> bool {
        let early = expected
            .saturating_sub(predicted_frame())
            .saturating_sub(Duration::from_millis(1));
        early <= *delay && *delay < expected + Duration::from_millis(100)
    }

    #[test]
    fn an_input_pass_the_app_ignores_wakes_nothing() {
        let root = Root::new();
        assert_eq!(root.pass(pointer_moved(), |_| {}), []);
        assert_eq!(root.pass(pointer_moved(), |_| {}), []);
    }

    #[test]
    fn a_delayed_request_wakes_once_after_its_delay() {
        let root = Root::new();
        let posted = root.pass(pointer_moved(), |ui| {
            ui.ctx().request_repaint_after(Duration::from_secs(1));
        });
        assert!(
            matches!(posted.as_slice(), [delay] if is_about(delay, Duration::from_secs(1))),
            "{posted:?}"
        );
        // The pass it wakes asks for nothing more.
        assert_eq!(root.pass(RawInput::default(), |_| {}), []);
    }

    #[test]
    fn an_immediate_request_wakes_one_pass() {
        let root = Root::new();
        let posted = root.pass(pointer_moved(), |ui| ui.ctx().request_repaint());
        assert!(
            matches!(posted.as_slice(), [delay] if is_immediate(delay)),
            "{posted:?}"
        );
        // egui's second pass for an immediate request is not run.
        assert_eq!(root.pass(RawInput::default(), |_| {}), []);
    }

    #[test]
    fn scrolling_keeps_eguis_follow_up() {
        let root = Root::new();
        let posted = root.pass(scrolled(), |_| {});
        assert!(
            matches!(posted.as_slice(), [delay] if is_immediate(delay)),
            "{posted:?}"
        );
    }

    #[test]
    fn another_thread_wakes_a_pass_while_one_runs() {
        struct WakeFromAnotherThread;
        impl egui::plugin::Plugin for WakeFromAnotherThread {
            fn debug_name(&self) -> &'static str {
                "wake from another thread"
            }

            fn on_begin_pass(&mut self, ui: &mut egui::Ui) {
                let ctx = ui.ctx().clone();
                std::thread::spawn(move || ctx.request_repaint_of(ViewportId::ROOT))
                    .join()
                    .expect("the other thread asks for a repaint");
            }
        }

        let root = Root::new();
        // While the app runs.
        let posted = root.pass(pointer_moved(), |ui| {
            let ctx = ui.ctx().clone();
            std::thread::spawn(move || ctx.request_repaint_of(ViewportId::ROOT))
                .join()
                .expect("the other thread asks for a repaint");
        });
        assert!(
            matches!(posted.as_slice(), [delay] if is_immediate(delay)),
            "{posted:?}"
        );
        // Before the app runs, when egui's own requests are not recorded.
        root.ctx.add_plugin(WakeFromAnotherThread);
        let posted = root.pass(pointer_moved(), |_| {});
        assert!(
            matches!(posted.as_slice(), [delay] if is_immediate(delay)),
            "{posted:?}"
        );
    }

    #[test]
    fn a_request_between_passes_wakes_a_pass() {
        let root = Root::new();
        // egui's own delay is left at zero by this pass.
        assert_eq!(root.pass(pointer_moved(), |_| {}), []);

        let since = Instant::now();
        let ctx = root.ctx.clone();
        std::thread::spawn(move || {
            ctx.request_repaint_after_for(Duration::from_millis(500), ViewportId::ROOT);
        })
        .join()
        .expect("the other thread asks for a repaint");
        let posted = root.take(since);
        assert!(
            matches!(posted.as_slice(), [delay] if is_about(delay, Duration::from_millis(500))),
            "{posted:?}"
        );

        let since = Instant::now();
        root.ctx.request_repaint();
        let posted = root.take(since);
        assert!(
            matches!(posted.as_slice(), [delay] if is_immediate(delay)),
            "{posted:?}"
        );
    }

    #[test]
    fn paint_only_frames_replay_after_input_passes() {
        let root = Root::new();
        root.pass(pointer_moved(), |_| {});
        root.pass(pointer_moved(), |_| {});
        assert!(!root.needs_full_ui());

        root.pass(pointer_moved(), |ui| {
            ui.ctx().request_repaint_after(Duration::from_secs(1));
        });
        assert!(!root.needs_full_ui());

        root.pass(pointer_moved(), |ui| ui.ctx().request_repaint());
        assert!(root.needs_full_ui());
    }
}
