use anyhow::{anyhow, Context, Result};
use log::info;
use std::fs::File;
use std::io::Read;
use std::path::Path;

pub struct CompileParams {
    pub chunk_size: (usize, usize, usize),
}

impl CompileParams {
    fn set_options(&self, options: &mut shaderc::CompileOptions) {
        let mut define_chunk_size = |dim, val| {
            let name = format!("CHUNK_SIZE_{}", dim);
            let val_str = format!("{}", val);
            options.add_macro_definition(name.as_str(), Some(val_str.as_str()));
        };

        let (x, y, z) = self.chunk_size;
        define_chunk_size("X", x);
        define_chunk_size("Y", y);
        define_chunk_size("Z", z);
    }
}

fn compile_shader(
    params: &CompileParams,
    path: &Path,
    compiler: &mut shaderc::Compiler,
    kind: shaderc::ShaderKind,
) -> Result<Vec<u8>> {
    // let mut compiler = shaderc::Compiler::new().unwrap();
    // options.add_macro_definition("EP", Some("main"));

    let fname = path
        .file_name()
        .ok_or_else(|| anyhow!("Expected file path, got directory path"))?;

    let mut source = String::new();
    File::open(path)
        .context("Opening shader file")?
        .read_to_string(&mut source)
        .context("Reading shader file")?;

    let mut options = shaderc::CompileOptions::new()
        .ok_or_else(|| anyhow!("Creating shaderc compile options"))?;
    params.set_options(&mut options);

    let compiled = compiler
        .compile_into_spirv(
            &source,
            kind,
            fname.to_string_lossy().as_ref(),
            "main",
            Some(&options),
        )
        .context("Compiling shader")?;

    Ok(compiled.as_binary_u8().into())
}

pub fn compile(
    params: &CompileParams,
    vertex_path: &Path,
    fragment_path: &Path,
) -> Result<(Vec<u8>, Vec<u8>)> {
    let mut compiler = shaderc::Compiler::new().context("Instantiating shaderc compiler")?;

    info!("Compiling vertex shader");
    let vertex_bytes = compile_shader(
        params,
        vertex_path,
        &mut compiler,
        shaderc::ShaderKind::Vertex,
    )?;
    info!("Compiling fragment shader");
    let fragment_bytes = compile_shader(
        params,
        fragment_path,
        &mut compiler,
        shaderc::ShaderKind::Fragment,
    )?;
    Ok((vertex_bytes, fragment_bytes))
}
