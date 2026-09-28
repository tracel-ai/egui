use core::{num::NonZeroU64, ops::Range};
use std::borrow::Cow;

use ahash::HashMap;
use bytemuck::Zeroable as _;
use epaint::{PaintCallbackInfo, Primitive, Vertex, emath::NumExt as _};

use wgpu::util::DeviceExt as _;

// Only implements Send + Sync on wasm32 in order to allow storing wgpu resources on the type map.
#[cfg(not(all(
    target_arch = "wasm32",
    not(feature = "fragile-send-sync-non-atomic-wasm"),
)))]
/// You can use this for storage when implementing [`CallbackTrait`].
pub type CallbackResources = type_map::concurrent::TypeMap;
#[cfg(all(
    target_arch = "wasm32",
    not(feature = "fragile-send-sync-non-atomic-wasm"),
))]
/// You can use this for storage when implementing [`CallbackTrait`].
pub type CallbackResources = type_map::TypeMap;

/// You can use this to do custom [`wgpu`] rendering in an egui app.
///
/// Implement [`CallbackTrait`] and call [`Callback::new_paint_callback`].
///
/// This can be turned into a [`epaint::PaintCallback`] and [`epaint::Shape`].
pub struct Callback(Box<dyn CallbackTrait>);

impl Callback {
    /// Creates a new [`epaint::PaintCallback`] from a callback trait instance.
    pub fn new_paint_callback(
        rect: epaint::emath::Rect,
        callback: impl CallbackTrait + 'static,
    ) -> epaint::PaintCallback {
        epaint::PaintCallback {
            rect,
            callback: std::sync::Arc::new(Self(Box::new(callback))),
        }
    }
}

/// A callback trait that can be used to compose an [`epaint::PaintCallback`] via [`Callback`]
/// for custom WGPU rendering.
///
/// Callbacks in [`Renderer`] are done in three steps:
/// * [`CallbackTrait::prepare`]: called for all registered callbacks before the main egui render pass.
/// * [`CallbackTrait::finish_prepare`]: called for all registered callbacks after all callbacks finished calling prepare.
/// * [`CallbackTrait::paint`]: called for all registered callbacks during the main egui render pass.
///
/// Each callback has access to an instance of [`CallbackResources`] that is stored in the [`Renderer`].
/// This can be used to store wgpu resources that need to be accessed during the [`CallbackTrait::paint`] step.
///
/// The callbacks implementing [`CallbackTrait`] itself must always be Send + Sync, but resources stored in
/// [`Renderer::callback_resources`] are not required to implement Send + Sync when building for wasm.
/// (this is because wgpu stores references to the JS heap in most of its resources which can not be shared with other threads).
///
///
/// # Command submission
///
/// ## Command Encoder
///
/// The passed-in [`wgpu::CommandEncoder`] is egui's and can be used directly to register
/// wgpu commands for simple use cases.
/// This allows reusing the same [`wgpu::CommandEncoder`] for all callbacks and egui
/// rendering itself.
///
/// ## Command Buffers
///
/// For more complicated use cases, one can also return a list of arbitrary
/// [`wgpu::CommandBuffer`]s and have complete control over how they get created and fed.
/// In particular, this gives an opportunity to parallelize command registration and
/// prevents a faulty callback from poisoning the main wgpu pipeline.
///
/// When using eframe, the main egui command buffer, as well as all user-defined
/// command buffers returned by this function, are guaranteed to all be submitted
/// at once in a single call.
///
/// Command Buffers returned by [`CallbackTrait::finish_prepare`] will always be issued *after*
/// those returned by [`CallbackTrait::prepare`].
/// Order within command buffers returned by [`CallbackTrait::prepare`] is dependent
/// on the order the respective [`epaint::Shape::Callback`]s were submitted in.
///
/// # Example
///
/// See the [`custom3d_wgpu`](https://github.com/emilk/egui/blob/main/crates/egui_demo_app/src/apps/custom3d_wgpu.rs) demo source for a detailed usage example.
pub trait CallbackTrait: Send + Sync {
    fn prepare(
        &self,
        _device: &wgpu::Device,
        _queue: &wgpu::Queue,
        _screen_descriptor: &ScreenDescriptor,
        _egui_encoder: &mut wgpu::CommandEncoder,
        _callback_resources: &mut CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        Vec::new()
    }

    /// Prepare a replay of the same paint jobs and screen descriptor as the
    /// preceding full frame. Callbacks with persistent GPU resources may only
    /// update their animation inputs here. The default preserves the behavior
    /// of existing callbacks.
    fn prepare_retained(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        screen_descriptor: &ScreenDescriptor,
        egui_encoder: &mut wgpu::CommandEncoder,
        callback_resources: &mut CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        self.prepare(
            device,
            queue,
            screen_descriptor,
            egui_encoder,
            callback_resources,
        )
    }

    /// Called after all [`CallbackTrait::prepare`] calls are done.
    fn finish_prepare(
        &self,
        _device: &wgpu::Device,
        _queue: &wgpu::Queue,
        _egui_encoder: &mut wgpu::CommandEncoder,
        _callback_resources: &mut CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        Vec::new()
    }

    /// Complete a retained replay. By default, use the regular completion path
    /// so callbacks outside this renderer need no changes.
    fn finish_prepare_retained(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        egui_encoder: &mut wgpu::CommandEncoder,
        callback_resources: &mut CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        self.finish_prepare(device, queue, egui_encoder, callback_resources)
    }

    /// Called after all [`CallbackTrait::finish_prepare`] calls are done.
    ///
    /// It is given access to the [`wgpu::RenderPass`] so that it can issue draw commands
    /// into the same [`wgpu::RenderPass`] that is used for all other egui elements.
    fn paint(
        &self,
        info: PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        callback_resources: &CallbackResources,
    );
}

/// Information about the screen used for rendering.
pub struct ScreenDescriptor {
    /// Size of the window in physical pixels.
    pub size_in_pixels: [u32; 2],

    /// High-DPI scale factor (pixels per point).
    pub pixels_per_point: f32,
}

impl ScreenDescriptor {
    /// size in "logical" points
    fn screen_size_in_points(&self) -> [f32; 2] {
        [
            self.size_in_pixels[0] as f32 / self.pixels_per_point,
            self.size_in_pixels[1] as f32 / self.pixels_per_point,
        ]
    }
}

/// Uniform buffer used when rendering.
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct UniformBuffer {
    screen_size_in_points: [f32; 2],
    dithering: u32,

    /// 1 to do manual filtering for more predictable kittest snapshot images.
    ///
    /// See also <https://github.com/emilk/egui/issues/5295>.
    predictable_texture_filtering: u32,
}

struct SlicedBuffer {
    buffer: wgpu::Buffer,
    slices: Vec<Range<usize>>,
    capacity: wgpu::BufferAddress,
}

struct CachedPrefixPipeline {
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::RenderPipeline,
}

struct CachedPrefix {
    // Keep the texture alive while its view is bound by the composite draw.
    _texture: wgpu::Texture,
    bind_group: wgpu::BindGroup,
    primitive_count: usize,
    mesh_count: usize,
    size_in_pixels: [u32; 2],
    pixels_per_point: f32,
    clear_color: [f32; 4],
}

pub struct Texture {
    /// The texture may be None if the `TextureId` is just a handle to a user-provided bind-group.
    pub texture: Option<wgpu::Texture>,

    /// Bindgroup for the texture + sampler.
    pub bind_group: wgpu::BindGroup,

    /// Options describing the sampler used in the bind group. This may be None if the `TextureId`
    /// is just a handle to a user-provided bind-group.
    pub options: Option<epaint::textures::TextureOptions>,
}

/// Ways to configure [`Renderer`] during creation.
#[derive(Clone, Copy, Debug)]
pub struct RendererOptions {
    /// Set the level of the multisampling anti-aliasing (MSAA).
    ///
    /// Must be a power-of-two. Higher = more smooth 3D.
    ///
    /// A value of `0` or `1` turns it off (default).
    ///
    /// `egui` already performs anti-aliasing via "feathering"
    /// (controlled by [`egui::epaint::TessellationOptions`]),
    /// but if you are embedding 3D in egui you may want to turn on multisampling.
    pub msaa_samples: u32,

    /// What format to use for the depth and stencil buffers,
    /// e.g. [`wgpu::TextureFormat::Depth32FloatStencil8`].
    ///
    /// egui doesn't need depth/stencil, so the default value is `None` (no depth or stancil buffers).
    pub depth_stencil_format: Option<wgpu::TextureFormat>,

    /// Controls whether to apply dithering to minimize banding artifacts.
    ///
    /// Dithering assumes an sRGB output and thus will apply noise to any input value that lies between
    /// two 8bit values after applying the sRGB OETF function, i.e. if it's not a whole 8bit value in "gamma space".
    /// This means that only inputs from texture interpolation and vertex colors should be affected in practice.
    ///
    /// Defaults to true.
    pub dithering: bool,

    /// Perform texture filtering in software?
    ///
    /// This is useful when you want predictable rendering across
    /// different hardware, e.g. for kittest snapshots.
    ///
    /// Default is `false`.
    ///
    /// See also <https://github.com/emilk/egui/issues/5295>.
    pub predictable_texture_filtering: bool,
}

impl RendererOptions {
    /// Set options that produce the most predicatable output.
    ///
    /// Useful for image snapshot tests.
    pub const PREDICTABLE: Self = Self {
        msaa_samples: 1,
        depth_stencil_format: None,
        dithering: false,
        predictable_texture_filtering: true,
    };
}

impl Default for RendererOptions {
    fn default() -> Self {
        Self {
            msaa_samples: 0,
            depth_stencil_format: None,
            dithering: true,
            predictable_texture_filtering: false,
        }
    }
}

/// Renderer for a egui based GUI.
pub struct Renderer {
    pipeline: wgpu::RenderPipeline,

    output_color_format: wgpu::TextureFormat,
    cached_prefix_pipeline: Option<CachedPrefixPipeline>,
    cached_prefix: Option<CachedPrefix>,

    index_buffer: SlicedBuffer,
    vertex_buffer: SlicedBuffer,

    uniform_buffer: wgpu::Buffer,
    previous_uniform_buffer_content: UniformBuffer,
    uniform_bind_group: wgpu::BindGroup,
    texture_bind_group_layout: wgpu::BindGroupLayout,

    /// Map of egui texture IDs to textures and their associated bindgroups (texture view +
    /// sampler). The texture may be None if the `TextureId` is just a handle to a user-provided
    /// sampler.
    textures: HashMap<epaint::TextureId, Texture>,
    next_user_texture_id: u64,
    samplers: HashMap<epaint::textures::TextureOptions, wgpu::Sampler>,

    options: RendererOptions,

    /// Storage for resources shared with all invocations of [`CallbackTrait`]'s methods.
    ///
    /// See also [`CallbackTrait`].
    pub callback_resources: CallbackResources,
}

impl Renderer {
    /// Creates a renderer for a egui UI.
    ///
    /// `output_color_format` should preferably be [`wgpu::TextureFormat::Rgba8Unorm`] or
    /// [`wgpu::TextureFormat::Bgra8Unorm`], i.e. in gamma-space.
    pub fn new(
        device: &wgpu::Device,
        output_color_format: wgpu::TextureFormat,
        options: RendererOptions,
    ) -> Self {
        profiling::function_scope!();

        let shader = wgpu::ShaderModuleDescriptor {
            label: Some("egui"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("egui.wgsl"))),
        };
        let module = {
            profiling::scope!("create_shader_module");
            device.create_shader_module(shader)
        };

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("egui_uniform_buffer"),
            contents: bytemuck::cast_slice(&[UniformBuffer {
                screen_size_in_points: [0.0, 0.0],
                dithering: u32::from(options.dithering),
                predictable_texture_filtering: u32::from(options.predictable_texture_filtering),
            }]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let uniform_bind_group_layout = {
            profiling::scope!("create_bind_group_layout");
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("egui_uniform_bind_group_layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(
                            core::mem::size_of::<UniformBuffer>() as _
                        ),
                        ty: wgpu::BufferBindingType::Uniform,
                    },
                    count: None,
                }],
            })
        };

        let uniform_bind_group = {
            profiling::scope!("create_bind_group");
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("egui_uniform_bind_group"),
                layout: &uniform_bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &uniform_buffer,
                        offset: 0,
                        size: None,
                    }),
                }],
            })
        };

        let texture_bind_group_layout = {
            profiling::scope!("create_bind_group_layout");
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("egui_texture_bind_group_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            })
        };

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("egui_pipeline_layout"),
            bind_group_layouts: &[
                Some(&uniform_bind_group_layout),
                Some(&texture_bind_group_layout),
            ],
            immediate_size: 0,
        });

        let depth_stencil = options
            .depth_stencil_format
            .map(|format| wgpu::DepthStencilState {
                format,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::Always),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            });

        let pipeline = {
            profiling::scope!("create_render_pipeline");
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("egui_pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    entry_point: Some("vs_main"),
                    module: &module,
                    buffers: &[Some(wgpu::VertexBufferLayout {
                        array_stride: 5 * 4,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        // 0: vec2 position
                        // 1: vec2 texture coordinates
                        // 2: uint color
                        attributes: &wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x2, 2 => Uint32],
                    })],
                    compilation_options: wgpu::PipelineCompilationOptions::default()
                },
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    unclipped_depth: false,
                    conservative: false,
                    cull_mode: None,
                    front_face: wgpu::FrontFace::default(),
                    polygon_mode: wgpu::PolygonMode::default(),
                    strip_index_format: None,
                },
                depth_stencil,
                multisample: wgpu::MultisampleState {
                    alpha_to_coverage_enabled: false,
                    count: options.msaa_samples.max(1),
                    mask: !0,
                },

                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some(if output_color_format.is_srgb() {
                        log::warn!("Detected a linear (sRGBA aware) framebuffer {output_color_format:?}. egui prefers Rgba8Unorm or Bgra8Unorm");
                        "fs_main_linear_framebuffer"
                    } else {
                        "fs_main_gamma_framebuffer" // this is what we prefer
                    }),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: output_color_format,
                        blend: Some(wgpu::BlendState {
                            color: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::One,
                                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                                operation: wgpu::BlendOperation::Add,
                            },
                            alpha: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::OneMinusDstAlpha,
                                dst_factor: wgpu::BlendFactor::One,
                                operation: wgpu::BlendOperation::Add,
                            },
                        }),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: wgpu::PipelineCompilationOptions::default()
                }),
                multiview_mask: None,
                cache: None,
            }
        )
        };

        const VERTEX_BUFFER_START_CAPACITY: wgpu::BufferAddress =
            (core::mem::size_of::<Vertex>() * 1024) as _;
        const INDEX_BUFFER_START_CAPACITY: wgpu::BufferAddress =
            (core::mem::size_of::<u32>() * 1024 * 3) as _;

        Self {
            pipeline,
            output_color_format,
            cached_prefix_pipeline: None,
            cached_prefix: None,
            vertex_buffer: SlicedBuffer {
                buffer: create_vertex_buffer(device, VERTEX_BUFFER_START_CAPACITY),
                slices: Vec::with_capacity(64),
                capacity: VERTEX_BUFFER_START_CAPACITY,
            },
            index_buffer: SlicedBuffer {
                buffer: create_index_buffer(device, INDEX_BUFFER_START_CAPACITY),
                slices: Vec::with_capacity(64),
                capacity: INDEX_BUFFER_START_CAPACITY,
            },
            uniform_buffer,
            // Buffers on wgpu are zero initialized, so this is indeed its current state!
            previous_uniform_buffer_content: UniformBuffer::zeroed(),
            uniform_bind_group,
            texture_bind_group_layout,
            textures: HashMap::default(),
            next_user_texture_id: 0,
            samplers: HashMap::default(),
            options,
            callback_resources: CallbackResources::default(),
        }
    }

    /// Executes the egui renderer onto an existing wgpu renderpass.
    ///
    /// Note that the lifetime of `render_pass` is `'static` which requires a call to [`wgpu::RenderPass::forget_lifetime`].
    /// This allows users to pass resources that live outside of the callback resources to the render pass.
    /// The render pass internally keeps all referenced resources alive as long as necessary.
    /// The only consequence of `forget_lifetime` is that any operation on the parent encoder will cause a runtime error
    /// instead of a compile time error.
    ///
    /// # Panic
    /// Always ensure that [`Renderer::update_buffers`] has been called otherwise calling [`Renderer::render`] will panic!
    pub fn render(
        &self,
        render_pass: &mut wgpu::RenderPass<'static>,
        paint_jobs: &[epaint::ClippedPrimitive],
        screen_descriptor: &ScreenDescriptor,
    ) {
        self.render_impl(render_pass, paint_jobs, screen_descriptor, false);
    }

    /// Replay a retained frame with its previously rendered static prefix.
    /// Falls back to regular rendering if the prefix is not cacheable.
    pub(crate) fn render_with_cached_prefix(
        &self,
        render_pass: &mut wgpu::RenderPass<'static>,
        paint_jobs: &[epaint::ClippedPrimitive],
        screen_descriptor: &ScreenDescriptor,
    ) {
        self.render_impl(render_pass, paint_jobs, screen_descriptor, true);
    }

    fn render_impl(
        &self,
        render_pass: &mut wgpu::RenderPass<'static>,
        paint_jobs: &[epaint::ClippedPrimitive],
        screen_descriptor: &ScreenDescriptor,
        use_cached_prefix: bool,
    ) {
        profiling::function_scope!();

        let pixels_per_point = screen_descriptor.pixels_per_point;
        let size_in_pixels = screen_descriptor.size_in_pixels;

        // Whether or not we need to reset the render pass because a paint callback has just
        // run.
        let mut needs_reset = true;

        let mut index_buffer_slices = self.index_buffer.slices.iter();
        let mut vertex_buffer_slices = self.vertex_buffer.slices.iter();

        let paint_jobs = if use_cached_prefix {
            if let (Some(cache), Some(blit)) = (
                self.cached_prefix.as_ref(),
                self.cached_prefix_pipeline.as_ref(),
            ) {
                if cache.size_in_pixels == size_in_pixels
                    && cache.pixels_per_point == pixels_per_point
                    && cache.primitive_count <= paint_jobs.len()
                {
                    render_pass.set_viewport(
                        0.0,
                        0.0,
                        size_in_pixels[0] as f32,
                        size_in_pixels[1] as f32,
                        0.0,
                        1.0,
                    );
                    render_pass.set_scissor_rect(0, 0, size_in_pixels[0], size_in_pixels[1]);
                    render_pass.set_pipeline(&blit.pipeline);
                    render_pass.set_bind_group(0, &cache.bind_group, &[]);
                    render_pass.draw(0..3, 0..1);
                    for _ in 0..cache.mesh_count {
                        index_buffer_slices.next();
                        vertex_buffer_slices.next();
                    }
                    &paint_jobs[cache.primitive_count..]
                } else {
                    paint_jobs
                }
            } else {
                paint_jobs
            }
        } else {
            paint_jobs
        };

        for epaint::ClippedPrimitive {
            clip_rect,
            primitive,
        } in paint_jobs
        {
            if needs_reset {
                render_pass.set_viewport(
                    0.0,
                    0.0,
                    size_in_pixels[0] as f32,
                    size_in_pixels[1] as f32,
                    0.0,
                    1.0,
                );
                render_pass.set_pipeline(&self.pipeline);
                render_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
                // update_buffers packs every mesh into these two buffers. Keep
                // them bound for the whole mesh run, then use draw offsets for
                // each mesh. Callbacks may change the bindings, so rebind after
                // each callback before drawing the next mesh.
                if !self.index_buffer.slices.is_empty() {
                    render_pass.set_index_buffer(
                        self.index_buffer.buffer.slice(..),
                        wgpu::IndexFormat::Uint32,
                    );
                    render_pass.set_vertex_buffer(0, self.vertex_buffer.buffer.slice(..));
                }
                needs_reset = false;
            }

            {
                let rect = ScissorRect::new(clip_rect, pixels_per_point, size_in_pixels);

                if rect.width == 0 || rect.height == 0 {
                    // Skip rendering zero-sized clip areas.
                    if let Primitive::Mesh(_) = primitive {
                        // If this is a mesh, we need to advance the index and vertex buffer iterators:
                        index_buffer_slices
                            .next()
                            .expect("You must call .update_buffers() before .render()");
                        vertex_buffer_slices
                            .next()
                            .expect("You must call .update_buffers() before .render()");
                    }
                    continue;
                }

                render_pass.set_scissor_rect(rect.x, rect.y, rect.width, rect.height);
            }

            match primitive {
                Primitive::Mesh(mesh) => {
                    let index_buffer_slice = index_buffer_slices
                        .next()
                        .expect("You must call .update_buffers() before .render()");
                    let vertex_buffer_slice = vertex_buffer_slices
                        .next()
                        .expect("You must call .update_buffers() before .render()");

                    if let Some(Texture { bind_group, .. }) = self.textures.get(&mesh.texture_id) {
                        render_pass.set_bind_group(1, bind_group, &[]);
                        let first_index =
                            (index_buffer_slice.start / core::mem::size_of::<u32>()) as u32;
                        let base_vertex =
                            (vertex_buffer_slice.start / core::mem::size_of::<Vertex>()) as i32;
                        render_pass.draw_indexed(
                            first_index..first_index + mesh.indices.len() as u32,
                            base_vertex,
                            0..1,
                        );
                    } else {
                        log::warn!("Missing texture: {:?}", mesh.texture_id);
                    }
                }
                Primitive::Callback(callback) => {
                    let Some(cbfn) = callback.callback.downcast_ref::<Callback>() else {
                        // We already warned in the `prepare` callback
                        continue;
                    };

                    let info = PaintCallbackInfo {
                        viewport: callback.rect,
                        clip_rect: *clip_rect,
                        pixels_per_point,
                        screen_size_px: size_in_pixels,
                    };

                    let viewport_px = info.viewport_in_pixels();
                    if viewport_px.width_px > 0 && viewport_px.height_px > 0 {
                        profiling::scope!("callback");

                        needs_reset = true;

                        // We're setting a default viewport for the render pass as a
                        // courtesy for the user, so that they don't have to think about
                        // it in the simple case where they just want to fill the whole
                        // paint area.
                        //
                        // The user still has the possibility of setting their own custom
                        // viewport during the paint callback, effectively overriding this
                        // one.
                        render_pass.set_viewport(
                            viewport_px.left_px as f32,
                            viewport_px.top_px as f32,
                            viewport_px.width_px as f32,
                            viewport_px.height_px as f32,
                            0.0,
                            1.0,
                        );

                        cbfn.0.paint(info, render_pass, &self.callback_resources);
                    }
                }
            }
        }

        render_pass.set_scissor_rect(0, 0, size_in_pixels[0], size_in_pixels[1]);
    }

    /// Flatten the leading static meshes into the same pixel state they would
    /// leave in the window before its first callback. This is used only for
    /// retained paints, when no egui meshes or managed textures have changed.
    pub fn prepare_cached_prefix(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        paint_jobs: &[epaint::ClippedPrimitive],
        screen_descriptor: &ScreenDescriptor,
        clear_color: [f32; 4],
    ) {
        if self.cached_prefix.as_ref().is_some_and(|cache| {
            cache.size_in_pixels == screen_descriptor.size_in_pixels
                && cache.pixels_per_point == screen_descriptor.pixels_per_point
                && cache.clear_color == clear_color
        }) {
            return;
        }
        self.cached_prefix = None;

        // A resolved MSAA image cannot preserve the sample state seen by later
        // callbacks. Native/user textures may change without an egui repaint.
        if self.options.msaa_samples > 1
            || self.options.depth_stencil_format.is_some()
            || !matches!(
                self.output_color_format,
                wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Bgra8Unorm
            )
        {
            return;
        }

        const MIN_PREFIX_MESHES: usize = 12;
        const MAX_CACHE_BYTES: u64 = 32 * 1024 * 1024;
        let primitive_count = paint_jobs
            .iter()
            .take_while(|job| matches!(job.primitive, Primitive::Mesh(_)))
            .count();
        if primitive_count < MIN_PREFIX_MESHES
            || paint_jobs[..primitive_count].iter().any(|job| {
                let Primitive::Mesh(mesh) = &job.primitive else {
                    unreachable!("prefix contains only meshes")
                };
                !matches!(mesh.texture_id, epaint::TextureId::Managed(_))
                    || !self.textures.contains_key(&mesh.texture_id)
            })
        {
            return;
        }

        let [width, height] = screen_descriptor.size_in_pixels;
        if width == 0 || height == 0 || u64::from(width) * u64::from(height) * 4 > MAX_CACHE_BYTES {
            return;
        }

        if self.cached_prefix_pipeline.is_none() {
            self.cached_prefix_pipeline = Some(self.create_cached_prefix_pipeline(device));
        }
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("egui_retained_prefix"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.output_color_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        {
            let render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("egui_retained_prefix_paint"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: clear_color[0] as f64,
                            g: clear_color[1] as f64,
                            b: clear_color[2] as f64,
                            a: clear_color[3] as f64,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            self.render(
                &mut render_pass.forget_lifetime(),
                &paint_jobs[..primitive_count],
                screen_descriptor,
            );
        }

        let blit = self
            .cached_prefix_pipeline
            .as_ref()
            .expect("created before the prefix pass");
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("egui_retained_prefix_bind_group"),
            layout: &blit.bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&view),
            }],
        });
        self.cached_prefix = Some(CachedPrefix {
            _texture: texture,
            bind_group,
            primitive_count,
            mesh_count: primitive_count,
            size_in_pixels: screen_descriptor.size_in_pixels,
            pixels_per_point: screen_descriptor.pixels_per_point,
            clear_color,
        });
    }

    fn create_cached_prefix_pipeline(&self, device: &wgpu::Device) -> CachedPrefixPipeline {
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("egui_retained_prefix_layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    multisampled: false,
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            }],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("egui_retained_prefix_pipeline_layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("egui_retained_prefix_shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("cached_prefix.wgsl"))),
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("egui_retained_prefix_pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: self.output_color_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            multiview_mask: None,
            cache: None,
        });
        CachedPrefixPipeline {
            bind_group_layout,
            pipeline,
        }
    }

    /// Should be called before [`Self::render`].
    pub fn update_texture(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        id: epaint::TextureId,
        image_delta: &epaint::ImageDelta,
    ) {
        profiling::function_scope!();
        self.cached_prefix = None;

        let width = image_delta.image.width() as u32;
        let height = image_delta.image.height() as u32;

        let size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };

        let data_color32 = match &image_delta.image {
            epaint::ImageData::Color(image) => {
                assert_eq!(
                    width as usize * height as usize,
                    image.pixels.len(),
                    "Mismatch between texture size and texel count"
                );
                Cow::Borrowed(&image.pixels)
            }
        };
        let data_bytes: &[u8] = bytemuck::cast_slice(data_color32.as_slice());

        let queue_write_data_to_texture = |texture, origin| {
            profiling::scope!("write_texture");
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture,
                    mip_level: 0,
                    origin,
                    aspect: wgpu::TextureAspect::All,
                },
                data_bytes,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(4 * width),
                    rows_per_image: Some(height),
                },
                size,
            );
        };

        // Use same label for all resources associated with this texture id (no point in retyping the type)
        let label_str = format!("egui_texid_{id:?}");
        let label = Some(label_str.as_str());

        let (texture, origin, bind_group) = if let Some(pos) = image_delta.pos {
            // update the existing texture
            let Texture {
                texture,
                bind_group,
                options,
            } = self
                .textures
                .remove(&id)
                .expect("Tried to update a texture that has not been allocated yet.");
            let texture = texture.expect("Tried to update user texture.");
            let options = options.expect("Tried to update user texture.");
            let origin = wgpu::Origin3d {
                x: pos[0] as u32,
                y: pos[1] as u32,
                z: 0,
            };

            (
                texture,
                origin,
                // If the TextureOptions are the same as the previous ones, we can reuse the bind group. Otherwise we
                // have to recreate it.
                if image_delta.options == options {
                    Some(bind_group)
                } else {
                    None
                },
            )
        } else {
            // allocate a new texture
            let texture = {
                profiling::scope!("create_texture");
                device.create_texture(&wgpu::TextureDescriptor {
                    label,
                    size,
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[wgpu::TextureFormat::Rgba8Unorm],
                })
            };
            let origin = wgpu::Origin3d::ZERO;
            (texture, origin, None)
        };

        let bind_group = bind_group.unwrap_or_else(|| {
            let sampler = self
                .samplers
                .entry(image_delta.options)
                .or_insert_with(|| create_sampler(image_delta.options, device));
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label,
                layout: &self.texture_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(
                            &texture.create_view(&wgpu::TextureViewDescriptor::default()),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(sampler),
                    },
                ],
            })
        });

        queue_write_data_to_texture(&texture, origin);

        // A full update must (re)create the texture at exactly the delta's size,
        // or glyph UVs (normalized by the CPU atlas size) will sample the wrong rows.
        debug_assert!(
            image_delta.pos.is_some() || [texture.width(), texture.height()] == [width, height],
            "egui texture {id:?}: GPU texture is {}x{} but full delta is {width}x{height}",
            texture.width(),
            texture.height(),
        );

        self.textures.insert(
            id,
            Texture {
                texture: Some(texture),
                bind_group,
                options: Some(image_delta.options),
            },
        );
    }

    pub fn free_texture(&mut self, id: &epaint::TextureId) {
        self.cached_prefix = None;
        if let Some(texture) = self.textures.remove(id).and_then(|t| t.texture) {
            texture.destroy();
        }
    }

    /// The layout of the texture bind groups [`Self::texture`] returns, so a
    /// paint callback can draw egui meshes with its own pipeline (ueye patch:
    /// retained motion layers).
    pub fn texture_bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.texture_bind_group_layout
    }

    /// Get the WGPU texture and bind group associated to a texture that has been allocated by egui.
    ///
    /// This could be used by custom paint hooks to render images that have been added through
    /// [`epaint::Context::load_texture`](https://docs.rs/egui/latest/egui/struct.Context.html#method.load_texture).
    pub fn texture(&self, id: &epaint::TextureId) -> Option<&Texture> {
        self.textures.get(id)
    }

    /// Registers a [`wgpu::Texture`] with a [`epaint::TextureId`].
    ///
    /// This enables the application to reference the texture inside an image ui element.
    /// This effectively enables off-screen rendering inside the egui UI. Texture must have
    /// the texture format [`wgpu::TextureFormat::Rgba8Unorm`].
    pub fn register_native_texture(
        &mut self,
        device: &wgpu::Device,
        texture: &wgpu::TextureView,
        texture_filter: wgpu::FilterMode,
    ) -> epaint::TextureId {
        self.register_native_texture_with_sampler_options(
            device,
            texture,
            wgpu::SamplerDescriptor {
                label: Some(format!("egui_user_image_{}", self.next_user_texture_id).as_str()),
                mag_filter: texture_filter,
                min_filter: texture_filter,
                ..Default::default()
            },
        )
    }

    /// Registers a [`wgpu::Texture`] with an existing [`epaint::TextureId`].
    ///
    /// This enables applications to reuse [`epaint::TextureId`]s.
    pub fn update_egui_texture_from_wgpu_texture(
        &mut self,
        device: &wgpu::Device,
        texture: &wgpu::TextureView,
        texture_filter: wgpu::FilterMode,
        id: epaint::TextureId,
    ) {
        self.update_egui_texture_from_wgpu_texture_with_sampler_options(
            device,
            texture,
            wgpu::SamplerDescriptor {
                label: Some(format!("egui_user_image_{}", self.next_user_texture_id).as_str()),
                mag_filter: texture_filter,
                min_filter: texture_filter,
                ..Default::default()
            },
            id,
        );
    }

    /// Registers a [`wgpu::Texture`] with a [`epaint::TextureId`] while also accepting custom
    /// [`wgpu::SamplerDescriptor`] options.
    ///
    /// This allows applications to specify individual minification/magnification filters as well as
    /// custom mipmap and tiling options.
    ///
    /// The texture must have the format [`wgpu::TextureFormat::Rgba8Unorm`].
    /// Any compare function supplied in the [`wgpu::SamplerDescriptor`] will be ignored.
    #[expect(clippy::needless_pass_by_value)] // false positive
    pub fn register_native_texture_with_sampler_options(
        &mut self,
        device: &wgpu::Device,
        texture: &wgpu::TextureView,
        sampler_descriptor: wgpu::SamplerDescriptor<'_>,
    ) -> epaint::TextureId {
        profiling::function_scope!();

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            compare: None,
            ..sampler_descriptor
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(format!("egui_user_image_{}", self.next_user_texture_id).as_str()),
            layout: &self.texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(texture),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        let id = epaint::TextureId::User(self.next_user_texture_id);
        self.textures.insert(
            id,
            Texture {
                texture: None,
                bind_group,
                options: None,
            },
        );
        self.next_user_texture_id += 1;

        id
    }

    /// Registers a [`wgpu::Texture`] with an existing [`epaint::TextureId`] while also accepting custom
    /// [`wgpu::SamplerDescriptor`] options.
    ///
    /// This allows applications to reuse [`epaint::TextureId`]s created with custom sampler options.
    #[expect(clippy::needless_pass_by_value)] // false positive
    pub fn update_egui_texture_from_wgpu_texture_with_sampler_options(
        &mut self,
        device: &wgpu::Device,
        texture: &wgpu::TextureView,
        sampler_descriptor: wgpu::SamplerDescriptor<'_>,
        id: epaint::TextureId,
    ) {
        profiling::function_scope!();
        self.cached_prefix = None;

        let Texture {
            bind_group: user_texture_binding,
            ..
        } = self
            .textures
            .get_mut(&id)
            .expect("Tried to update a texture that has not been allocated yet.");

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            compare: None,
            ..sampler_descriptor
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(format!("egui_user_image_{}", self.next_user_texture_id).as_str()),
            layout: &self.texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(texture),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        *user_texture_binding = bind_group;
    }

    /// Uploads the uniform, vertex and index data used by the renderer.
    /// Should be called before [`Self::render`].
    ///
    /// Returns all user-defined command buffers gathered from [`CallbackTrait::prepare`] & [`CallbackTrait::finish_prepare`] callbacks.
    pub fn update_buffers(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        paint_jobs: &[epaint::ClippedPrimitive],
        screen_descriptor: &ScreenDescriptor,
    ) -> Vec<wgpu::CommandBuffer> {
        profiling::function_scope!();
        self.cached_prefix = None;

        let screen_size_in_points = screen_descriptor.screen_size_in_points();

        let uniform_buffer_content = UniformBuffer {
            screen_size_in_points,
            dithering: u32::from(self.options.dithering),
            predictable_texture_filtering: u32::from(self.options.predictable_texture_filtering),
        };
        if uniform_buffer_content != self.previous_uniform_buffer_content {
            profiling::scope!("update uniforms");
            queue.write_buffer(
                &self.uniform_buffer,
                0,
                bytemuck::cast_slice(&[uniform_buffer_content]),
            );
            self.previous_uniform_buffer_content = uniform_buffer_content;
        }

        // Determine how many vertices & indices need to be rendered.
        let (vertex_count, index_count) = {
            profiling::scope!("count_vertices_indices");
            paint_jobs.iter().fold((0, 0), |acc, clipped_primitive| {
                match &clipped_primitive.primitive {
                    Primitive::Mesh(mesh) => {
                        (acc.0 + mesh.vertices.len(), acc.1 + mesh.indices.len())
                    }
                    Primitive::Callback(_) => acc,
                }
            })
        };

        if index_count > 0 {
            profiling::scope!("indices", index_count.to_string().as_str());

            self.index_buffer.slices.clear();

            let required_index_buffer_size = (core::mem::size_of::<u32>() * index_count) as u64;
            if self.index_buffer.capacity < required_index_buffer_size {
                // Resize index buffer if needed.
                self.index_buffer.capacity =
                    (self.index_buffer.capacity * 2).at_least(required_index_buffer_size);
                self.index_buffer.buffer = create_index_buffer(device, self.index_buffer.capacity);
            }

            let index_buffer_staging = queue.write_buffer_with(
                &self.index_buffer.buffer,
                0,
                #[expect(clippy::unwrap_used)] // Checked above
                NonZeroU64::new(required_index_buffer_size).unwrap(),
            );

            let Some(mut index_buffer_staging) = index_buffer_staging else {
                panic!(
                    "Failed to create staging buffer for index data. Index count: {index_count}. Required index buffer size: {required_index_buffer_size}. Actual size {} and capacity: {} (bytes)",
                    self.index_buffer.buffer.size(),
                    self.index_buffer.capacity
                );
            };

            let mut index_offset = 0;
            for epaint::ClippedPrimitive { primitive, .. } in paint_jobs {
                match primitive {
                    Primitive::Mesh(mesh) => {
                        let size = mesh.indices.len() * core::mem::size_of::<u32>();
                        let slice = index_offset..(size + index_offset);
                        index_buffer_staging
                            .slice(slice.clone())
                            .copy_from_slice(bytemuck::cast_slice(&mesh.indices));
                        self.index_buffer.slices.push(slice);
                        index_offset += size;
                    }
                    Primitive::Callback(_) => {}
                }
            }
        }
        if vertex_count > 0 {
            profiling::scope!("vertices", vertex_count.to_string().as_str());

            self.vertex_buffer.slices.clear();

            let required_vertex_buffer_size =
                (core::mem::size_of::<Vertex>() * vertex_count) as u64;
            if self.vertex_buffer.capacity < required_vertex_buffer_size {
                // Resize vertex buffer if needed.
                self.vertex_buffer.capacity =
                    (self.vertex_buffer.capacity * 2).at_least(required_vertex_buffer_size);
                self.vertex_buffer.buffer =
                    create_vertex_buffer(device, self.vertex_buffer.capacity);
            }

            let vertex_buffer_staging = queue.write_buffer_with(
                &self.vertex_buffer.buffer,
                0,
                #[expect(clippy::unwrap_used)] // Checked above
                NonZeroU64::new(required_vertex_buffer_size).unwrap(),
            );

            let Some(mut vertex_buffer_staging) = vertex_buffer_staging else {
                panic!(
                    "Failed to create staging buffer for vertex data. Vertex count: {vertex_count}. Required vertex buffer size: {required_vertex_buffer_size}. Actual size {} and capacity: {} (bytes)",
                    self.vertex_buffer.buffer.size(),
                    self.vertex_buffer.capacity
                );
            };

            let mut vertex_offset = 0;
            for epaint::ClippedPrimitive { primitive, .. } in paint_jobs {
                match primitive {
                    Primitive::Mesh(mesh) => {
                        let size = mesh.vertices.len() * core::mem::size_of::<Vertex>();
                        let slice = vertex_offset..(size + vertex_offset);
                        vertex_buffer_staging
                            .slice(slice.clone())
                            .copy_from_slice(bytemuck::cast_slice(&mesh.vertices));
                        self.vertex_buffer.slices.push(slice);
                        vertex_offset += size;
                    }
                    Primitive::Callback(_) => {}
                }
            }
        }

        self.prepare_callbacks(device, queue, encoder, paint_jobs, screen_descriptor, false)
    }

    /// Prepare paint callbacks while retaining the mesh buffers from the previous paint.
    ///
    /// A host may use this only when it replays the exact same paint jobs and
    /// screen descriptor as the last `update_buffers` call. Input, resized
    /// surfaces, texture changes, or another viewport's mesh update require a
    /// normal update first.
    pub fn update_callbacks_only(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        paint_jobs: &[epaint::ClippedPrimitive],
        screen_descriptor: &ScreenDescriptor,
    ) -> Vec<wgpu::CommandBuffer> {
        self.prepare_callbacks(device, queue, encoder, paint_jobs, screen_descriptor, true)
    }

    fn prepare_callbacks(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        paint_jobs: &[epaint::ClippedPrimitive],
        screen_descriptor: &ScreenDescriptor,
        retained: bool,
    ) -> Vec<wgpu::CommandBuffer> {
        let mut callbacks = Vec::new();
        for clipped_primitive in paint_jobs {
            if let Primitive::Callback(callback) = &clipped_primitive.primitive {
                if let Some(callback) = callback.callback.downcast_ref::<Callback>() {
                    callbacks.push(callback.0.as_ref());
                } else {
                    log::warn!("Unknown paint callback: expected `egui_wgpu::Callback`");
                }
            }
        }
        let mut user_cmd_bufs = Vec::new();
        {
            profiling::scope!("prepare callbacks");
            for callback in &callbacks {
                user_cmd_bufs.extend(if retained {
                    callback.prepare_retained(
                        device,
                        queue,
                        screen_descriptor,
                        encoder,
                        &mut self.callback_resources,
                    )
                } else {
                    callback.prepare(
                        device,
                        queue,
                        screen_descriptor,
                        encoder,
                        &mut self.callback_resources,
                    )
                });
            }
        }
        {
            profiling::scope!("finish prepare callbacks");
            for callback in &callbacks {
                user_cmd_bufs.extend(if retained {
                    callback.finish_prepare_retained(
                        device,
                        queue,
                        encoder,
                        &mut self.callback_resources,
                    )
                } else {
                    callback.finish_prepare(device, queue, encoder, &mut self.callback_resources)
                });
            }
        }

        user_cmd_bufs
    }
}

fn create_sampler(
    options: epaint::textures::TextureOptions,
    device: &wgpu::Device,
) -> wgpu::Sampler {
    let mag_filter = match options.magnification {
        epaint::textures::TextureFilter::Nearest => wgpu::FilterMode::Nearest,
        epaint::textures::TextureFilter::Linear => wgpu::FilterMode::Linear,
    };
    let min_filter = match options.minification {
        epaint::textures::TextureFilter::Nearest => wgpu::FilterMode::Nearest,
        epaint::textures::TextureFilter::Linear => wgpu::FilterMode::Linear,
    };
    let address_mode = match options.wrap_mode {
        epaint::textures::TextureWrapMode::ClampToEdge => wgpu::AddressMode::ClampToEdge,
        epaint::textures::TextureWrapMode::Repeat => wgpu::AddressMode::Repeat,
        epaint::textures::TextureWrapMode::MirroredRepeat => wgpu::AddressMode::MirrorRepeat,
    };
    device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some(&format!(
            "egui sampler (mag: {mag_filter:?}, min {min_filter:?})"
        )),
        mag_filter,
        min_filter,
        address_mode_u: address_mode,
        address_mode_v: address_mode,
        ..Default::default()
    })
}

fn create_vertex_buffer(device: &wgpu::Device, size: u64) -> wgpu::Buffer {
    profiling::function_scope!();
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("egui_vertex_buffer"),
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        size,
        mapped_at_creation: false,
    })
}

fn create_index_buffer(device: &wgpu::Device, size: u64) -> wgpu::Buffer {
    profiling::function_scope!();
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("egui_index_buffer"),
        usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
        size,
        mapped_at_creation: false,
    })
}

/// A Rect in physical pixel space, used for setting clipping rectangles.
struct ScissorRect {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
}

impl ScissorRect {
    fn new(clip_rect: &epaint::Rect, pixels_per_point: f32, target_size: [u32; 2]) -> Self {
        // Transform clip rect to physical pixels:
        let clip_min_x = pixels_per_point * clip_rect.min.x;
        let clip_min_y = pixels_per_point * clip_rect.min.y;
        let clip_max_x = pixels_per_point * clip_rect.max.x;
        let clip_max_y = pixels_per_point * clip_rect.max.y;

        // Round to integer:
        let clip_min_x = clip_min_x.round() as u32;
        let clip_min_y = clip_min_y.round() as u32;
        let clip_max_x = clip_max_x.round() as u32;
        let clip_max_y = clip_max_y.round() as u32;

        // Clamp:
        let clip_min_x = clip_min_x.clamp(0, target_size[0]);
        let clip_min_y = clip_min_y.clamp(0, target_size[1]);
        let clip_max_x = clip_max_x.clamp(clip_min_x, target_size[0]);
        let clip_max_y = clip_max_y.clamp(clip_min_y, target_size[1]);

        Self {
            x: clip_min_x,
            y: clip_min_y,
            width: clip_max_x - clip_min_x,
            height: clip_max_y - clip_min_y,
        }
    }
}

// Look at the feature flag for an explanation.
#[cfg(not(all(
    target_arch = "wasm32",
    not(feature = "fragile-send-sync-non-atomic-wasm"),
)))]
#[test]
fn renderer_impl_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Renderer>();
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod retained_prefix_tests {
    use super::*;
    use std::{sync::mpsc, time::Duration};

    const SIZE: [u32; 2] = [64, 64];

    struct ColorCallback {
        pipeline: wgpu::RenderPipeline,
    }

    impl CallbackTrait for ColorCallback {
        fn paint(
            &self,
            _info: PaintCallbackInfo,
            pass: &mut wgpu::RenderPass<'static>,
            _resources: &CallbackResources,
        ) {
            pass.set_pipeline(&self.pipeline);
            pass.draw(0..3, 0..1);
        }
    }

    fn color_callback(device: &wgpu::Device) -> epaint::PaintCallback {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("prefix test callback"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(
                r#"
@vertex fn vs_main(@builtin(vertex_index) id: u32) -> @builtin(position) vec4<f32> {
    let corners = array<vec2<f32>, 3>(vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
    return vec4<f32>(corners[id], 0.0, 1.0);
}
@fragment fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(0.12, 0.32, 0.04, 0.55);
}
"#,
            )),
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("prefix test callback pipeline"),
            layout: None,
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::OneMinusDstAlpha,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            multiview_mask: None,
            cache: None,
        });
        Callback::new_paint_callback(
            epaint::emath::Rect::from_min_max(
                epaint::emath::pos2(15.0, 9.0),
                epaint::emath::pos2(52.0, 48.0),
            ),
            ColorCallback { pipeline },
        )
    }

    fn mesh_job(
        rect: epaint::emath::Rect,
        clip: epaint::emath::Rect,
        color: epaint::Color32,
    ) -> epaint::ClippedPrimitive {
        let mut mesh = epaint::Mesh::default();
        mesh.add_colored_rect(rect, color);
        epaint::ClippedPrimitive {
            clip_rect: clip,
            primitive: Primitive::Mesh(mesh),
        }
    }

    fn read_frame(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        renderer: &Renderer,
        jobs: &[epaint::ClippedPrimitive],
        screen: &ScreenDescriptor,
        cached: bool,
        mut encoder: wgpu::CommandEncoder,
    ) -> Vec<u8> {
        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("prefix comparison target"),
            size: wgpu::Extent3d {
                width: SIZE[0],
                height: SIZE[1],
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        {
            let view = target.create_view(&wgpu::TextureViewDescriptor::default());
            let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("prefix comparison pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                ..Default::default()
            });
            let mut pass = pass.forget_lifetime();
            if cached {
                renderer.render_with_cached_prefix(&mut pass, jobs, screen);
            } else {
                renderer.render(&mut pass, jobs, screen);
            }
        }
        let output = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("prefix comparison readback"),
            size: u64::from(SIZE[0] * SIZE[1] * 4),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            target.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &output,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(SIZE[0] * 4),
                    rows_per_image: Some(SIZE[1]),
                },
            },
            target.size(),
        );
        queue.submit([encoder.finish()]);
        let (send, receive) = mpsc::channel();
        output
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                send.send(result).expect("receiver is alive");
            });
        device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(Duration::from_secs(30)),
            })
            .expect("GPU completes");
        receive.recv().unwrap().expect("map readback");
        let pixels = output.slice(..).get_mapped_range().unwrap().to_vec();
        output.unmap();
        pixels
    }

    #[test]
    fn retained_prefix_matches_full_paint_with_clips_alpha_and_callback() {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = match pollster::block_on(
            instance.request_adapter(&wgpu::RequestAdapterOptions::default()),
        ) {
            Ok(adapter) => adapter,
            Err(_) => {
                eprintln!("skipped retained prefix GPU readback: no adapter");
                return;
            }
        };
        eprintln!(
            "retained prefix GPU readback: {:?}",
            adapter.get_info().backend
        );
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).unwrap();
        let mut renderer = Renderer::new(
            &device,
            wgpu::TextureFormat::Rgba8Unorm,
            RendererOptions {
                dithering: false,
                ..Default::default()
            },
        );
        renderer.update_texture(
            &device,
            &queue,
            epaint::TextureId::Managed(0),
            &epaint::ImageDelta::full(
                epaint::ColorImage::filled([1, 1], epaint::Color32::WHITE),
                epaint::textures::TextureOptions::NEAREST,
            ),
        );
        let full = epaint::emath::Rect::from_min_max(
            epaint::emath::pos2(0.0, 0.0),
            epaint::emath::pos2(64.0, 64.0),
        );
        let mut jobs = Vec::new();
        for i in 0..14 {
            let inset = i as f32;
            jobs.push(mesh_job(
                epaint::emath::Rect::from_min_max(
                    epaint::emath::pos2(inset, inset * 0.5),
                    epaint::emath::pos2(64.0 - inset, 64.0 - inset * 0.5),
                ),
                epaint::emath::Rect::from_min_max(
                    epaint::emath::pos2(inset + 2.0, 4.0),
                    epaint::emath::pos2(60.0, 58.0 - inset),
                ),
                epaint::Color32::from_rgba_unmultiplied(20 + i * 8, 120, 220, 100),
            ));
        }
        jobs.push(epaint::ClippedPrimitive {
            clip_rect: full,
            primitive: Primitive::Callback(color_callback(&device)),
        });
        jobs.push(mesh_job(
            full,
            epaint::emath::Rect::from_min_max(
                epaint::emath::pos2(8.0, 20.0),
                epaint::emath::pos2(56.0, 52.0),
            ),
            epaint::Color32::from_rgba_unmultiplied(230, 45, 20, 90),
        ));
        let screen = ScreenDescriptor {
            size_in_pixels: SIZE,
            pixels_per_point: 1.0,
        };
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        renderer.update_buffers(&device, &queue, &mut encoder, &jobs, &screen);
        let full_pixels = read_frame(&device, &queue, &renderer, &jobs, &screen, false, encoder);

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        renderer.prepare_cached_prefix(&device, &mut encoder, &jobs, &screen, [0.0; 4]);
        assert_eq!(renderer.cached_prefix.as_ref().unwrap().mesh_count, 14);
        let cached_pixels = read_frame(&device, &queue, &renderer, &jobs, &screen, true, encoder);
        assert_eq!(cached_pixels, full_pixels);
    }
}
