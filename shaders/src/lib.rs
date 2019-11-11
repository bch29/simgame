use anyhow::{anyhow, Context, Result};
use std::fs::File;
use std::io::Read;
use std::path::Path;

fn compile_shader(
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

    let options = shaderc::CompileOptions::new()
        .ok_or_else(|| anyhow!("Creating shaderc compile options"))?;
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

pub fn compile(vertex_path: &Path, fragment_path: &Path) -> Result<(Vec<u8>, Vec<u8>)> {
    let mut compiler = shaderc::Compiler::new().context("Instantiating shaderc compiler")?;
    let vertex_bytes = compile_shader(vertex_path, &mut compiler, shaderc::ShaderKind::Vertex)?;
    let fragment_bytes =
        compile_shader(fragment_path, &mut compiler, shaderc::ShaderKind::Fragment)?;
    Ok((vertex_bytes, fragment_bytes))
}
