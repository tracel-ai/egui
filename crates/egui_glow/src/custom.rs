//! Ueye patch: draw egui meshes with an application's own GLSL program.
//!
//! A crate that forbids unsafe code can compile a program, upload meshes
//! once, render into a texture and draw with a uniform block through this
//! safe interface. Ueye's motion layers use it for glow parity with their
//! wgpu pipelines (DESIGN.md 9.6). Every object keeps the `glow::Context` it
//! was made with and deletes its GL objects when dropped, so drop them on
//! the thread that owns the context.
#![expect(unsafe_code)]

use std::sync::Arc;

use egui::epaint::Vertex;
use glow::HasContext as _;

/// A linked program whose uniform blocks all read binding point 0 and
/// whose sampler, if any, reads texture unit 0.
pub struct CustomProgram {
    gl: Arc<glow::Context>,
    program: glow::Program,
    sampler: Option<glow::UniformLocation>,
    uniforms: glow::Buffer,
    empty: glow::VertexArray,
}

fn compile(gl: &glow::Context, kind: u32, source: &str) -> Result<glow::Shader, String> {
    unsafe {
        let shader = gl.create_shader(kind)?;
        gl.shader_source(shader, source);
        gl.compile_shader(shader);
        if gl.get_shader_compile_status(shader) {
            Ok(shader)
        } else {
            let log = gl.get_shader_info_log(shader);
            gl.delete_shader(shader);
            Err(log)
        }
    }
}

impl CustomProgram {
    /// Compiles and links `vertex` and `fragment`, binding each uniform
    /// block in `blocks` to binding point 0 and `sampler` to texture unit 0.
    pub fn new(
        gl: &Arc<glow::Context>,
        vertex: &str,
        fragment: &str,
        blocks: &[&str],
        sampler: Option<&str>,
    ) -> Result<Self, String> {
        let vertex = compile(gl, glow::VERTEX_SHADER, vertex)?;
        let fragment = match compile(gl, glow::FRAGMENT_SHADER, fragment) {
            Ok(fragment) => fragment,
            Err(error) => {
                unsafe { gl.delete_shader(vertex) };
                return Err(error);
            }
        };
        unsafe {
            let program = gl.create_program()?;
            gl.attach_shader(program, vertex);
            gl.attach_shader(program, fragment);
            gl.link_program(program);
            gl.detach_shader(program, vertex);
            gl.detach_shader(program, fragment);
            gl.delete_shader(vertex);
            gl.delete_shader(fragment);
            if !gl.get_program_link_status(program) {
                let log = gl.get_program_info_log(program);
                gl.delete_program(program);
                return Err(log);
            }
            for block in blocks {
                if let Some(index) = gl.get_uniform_block_index(program, block) {
                    gl.uniform_block_binding(program, index, 0);
                }
            }
            let sampler = sampler.and_then(|name| gl.get_uniform_location(program, name));
            let uniforms = gl.create_buffer()?;
            let empty = gl.create_vertex_array()?;
            Ok(Self {
                gl: gl.clone(),
                program,
                sampler,
                uniforms,
                empty,
            })
        }
    }
}

impl Drop for CustomProgram {
    fn drop(&mut self) {
        unsafe {
            self.gl.delete_program(self.program);
            self.gl.delete_buffer(self.uniforms);
            self.gl.delete_vertex_array(self.empty);
        }
    }
}

/// An egui mesh uploaded once: position (location 0), texture coordinates
/// (location 1) and colour as a packed `uint` (location 2).
pub struct CustomMesh {
    gl: Arc<glow::Context>,
    vao: glow::VertexArray,
    vbo: glow::Buffer,
    ebo: glow::Buffer,
    count: i32,
}

impl CustomMesh {
    /// Uploads `vertices` and `indices`.
    pub fn new(
        gl: &Arc<glow::Context>,
        vertices: &[Vertex],
        indices: &[u32],
    ) -> Result<Self, String> {
        unsafe {
            let vao = gl.create_vertex_array()?;
            let vbo = gl.create_buffer()?;
            let ebo = gl.create_buffer()?;
            gl.bind_vertex_array(Some(vao));
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo));
            gl.buffer_data_u8_slice(
                glow::ARRAY_BUFFER,
                bytemuck::cast_slice(vertices),
                glow::STATIC_DRAW,
            );
            gl.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, Some(ebo));
            gl.buffer_data_u8_slice(
                glow::ELEMENT_ARRAY_BUFFER,
                bytemuck::cast_slice(indices),
                glow::STATIC_DRAW,
            );
            let stride = core::mem::size_of::<Vertex>() as i32;
            gl.enable_vertex_attrib_array(0);
            gl.vertex_attrib_pointer_f32(0, 2, glow::FLOAT, false, stride, 0);
            gl.enable_vertex_attrib_array(1);
            gl.vertex_attrib_pointer_f32(1, 2, glow::FLOAT, false, stride, 8);
            gl.enable_vertex_attrib_array(2);
            gl.vertex_attrib_pointer_i32(2, 1, glow::UNSIGNED_INT, stride, 16);
            gl.bind_vertex_array(None);
            gl.bind_buffer(glow::ARRAY_BUFFER, None);
            Ok(Self {
                gl: gl.clone(),
                vao,
                vbo,
                ebo,
                count: indices.len() as i32,
            })
        }
    }
}

impl Drop for CustomMesh {
    fn drop(&mut self) {
        unsafe {
            self.gl.delete_vertex_array(self.vao);
            self.gl.delete_buffer(self.vbo);
            self.gl.delete_buffer(self.ebo);
        }
    }
}

/// A texture a program can render into.
pub struct CustomTarget {
    gl: Arc<glow::Context>,
    framebuffer: glow::Framebuffer,
    texture: glow::Texture,
    size: [i32; 2],
}

impl CustomTarget {
    /// A transparent RGBA texture of `width` × `height` pixels.
    pub fn new(gl: &Arc<glow::Context>, width: u32, height: u32) -> Result<Self, String> {
        let size = [width.max(1) as i32, height.max(1) as i32];
        unsafe {
            let texture = gl.create_texture()?;
            gl.bind_texture(glow::TEXTURE_2D, Some(texture));
            gl.tex_image_2d(
                glow::TEXTURE_2D,
                0,
                glow::RGBA8 as i32,
                size[0],
                size[1],
                0,
                glow::RGBA,
                glow::UNSIGNED_BYTE,
                glow::PixelUnpackData::Slice(None),
            );
            for (parameter, value) in [
                (glow::TEXTURE_MIN_FILTER, glow::LINEAR),
                (glow::TEXTURE_MAG_FILTER, glow::LINEAR),
                (glow::TEXTURE_WRAP_S, glow::CLAMP_TO_EDGE),
                (glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE),
            ] {
                gl.tex_parameter_i32(glow::TEXTURE_2D, parameter, value as i32);
            }
            let framebuffer = gl.create_framebuffer()?;
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(framebuffer));
            gl.framebuffer_texture_2d(
                glow::FRAMEBUFFER,
                glow::COLOR_ATTACHMENT0,
                glow::TEXTURE_2D,
                Some(texture),
                0,
            );
            gl.clear_color(0.0, 0.0, 0.0, 0.0);
            gl.clear(glow::COLOR_BUFFER_BIT);
            gl.bind_framebuffer(glow::FRAMEBUFFER, None);
            gl.bind_texture(glow::TEXTURE_2D, None);
            Ok(Self {
                gl: gl.clone(),
                framebuffer,
                texture,
                size,
            })
        }
    }

    /// The texture holding what was drawn into the target.
    pub fn texture(&self) -> glow::Texture {
        self.texture
    }

    /// Width and height in pixels.
    pub fn size(&self) -> [i32; 2] {
        self.size
    }
}

impl Drop for CustomTarget {
    fn drop(&mut self) {
        unsafe {
            self.gl.delete_framebuffer(self.framebuffer);
            self.gl.delete_texture(self.texture);
        }
    }
}

/// One draw with a [`CustomProgram`].
pub struct CustomDraw<'a> {
    /// The program.
    pub program: &'a CustomProgram,
    /// Bytes of the uniform block, std140.
    pub uniforms: &'a [u8],
    /// The mesh, or `None` to draw `vertices` generated vertices.
    pub mesh: Option<&'a CustomMesh>,
    /// Vertices to draw without a mesh.
    pub vertices: i32,
    /// Texture on unit 0.
    pub texture: Option<glow::Texture>,
    /// Viewport in pixels: x, y from the bottom, width, height.
    pub viewport: [i32; 4],
    /// Scissor in pixels: x, y from the bottom, width, height.
    pub scissor: Option<[i32; 4]>,
    /// Render into this target instead of the current framebuffer.
    pub target: Option<&'a CustomTarget>,
}

/// Draws `draw` with premultiplied alpha blending. The framebuffer, program,
/// vertex array and uniform buffer bindings are left for egui to restore.
pub fn draw(gl: &glow::Context, draw: &CustomDraw<'_>) {
    unsafe {
        if let Some(target) = draw.target {
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(target.framebuffer));
        }
        let [x, y, width, height] = draw.viewport;
        gl.viewport(x, y, width, height);
        match draw.scissor {
            Some([x, y, width, height]) => {
                gl.enable(glow::SCISSOR_TEST);
                gl.scissor(x, y, width, height);
            }
            None => gl.disable(glow::SCISSOR_TEST),
        }
        gl.enable(glow::BLEND);
        gl.blend_equation_separate(glow::FUNC_ADD, glow::FUNC_ADD);
        gl.blend_func_separate(
            glow::ONE,
            glow::ONE_MINUS_SRC_ALPHA,
            glow::ONE_MINUS_DST_ALPHA,
            glow::ONE,
        );
        gl.use_program(Some(draw.program.program));
        gl.bind_buffer(glow::UNIFORM_BUFFER, Some(draw.program.uniforms));
        gl.buffer_data_u8_slice(glow::UNIFORM_BUFFER, draw.uniforms, glow::DYNAMIC_DRAW);
        gl.bind_buffer_base(glow::UNIFORM_BUFFER, 0, Some(draw.program.uniforms));
        if let Some(texture) = draw.texture {
            gl.active_texture(glow::TEXTURE0);
            gl.bind_texture(glow::TEXTURE_2D, Some(texture));
            if let Some(sampler) = &draw.program.sampler {
                gl.uniform_1_i32(Some(sampler), 0);
            }
        }
        match draw.mesh {
            Some(mesh) => {
                gl.bind_vertex_array(Some(mesh.vao));
                gl.draw_elements(glow::TRIANGLES, mesh.count, glow::UNSIGNED_INT, 0);
            }
            None => {
                gl.bind_vertex_array(Some(draw.program.empty));
                gl.draw_arrays(glow::TRIANGLES, 0, draw.vertices);
            }
        }
        gl.bind_vertex_array(None);
        gl.bind_buffer(glow::UNIFORM_BUFFER, None);
        if draw.target.is_some() {
            gl.bind_framebuffer(glow::FRAMEBUFFER, None);
        }
    }
}
