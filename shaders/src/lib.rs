use anyhow::{anyhow, Context, Result};
use log::info;
use std::fs::File;
use std::io::Read;
use std::path::Path;

pub struct CompileParams {
    pub chunk_size: (usize, usize, usize),
}

pub struct Compiler {
    params: CompileParams,
    compiler: shaderc::Compiler,
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

impl Compiler {
    pub fn new(params: CompileParams) -> Result<Self> {
        Ok(Self {
            params,
            compiler: shaderc::Compiler::new().context("Instantiating shaderc compiler")?,
        })
    }

    pub fn compile_vert(&mut self, path: &Path) -> Result<Vec<u32>> {
        info!("Compiling vertex shader {:?}", path);
        self.compile_impl(path, shaderc::ShaderKind::Vertex)
    }

    pub fn compile_frag(&mut self, path: &Path) -> Result<Vec<u32>> {
        info!("Compiling fragment shader {:?}", path);
        self.compile_impl(path, shaderc::ShaderKind::Fragment)
    }

    pub fn compile_compute(&mut self, path: &Path) -> Result<Vec<u32>> {
        info!("Compiling compute shader {:?}", path);
        self.compile_impl(path, shaderc::ShaderKind::Compute)
    }

    fn compile_impl(&mut self, path: &Path, kind: shaderc::ShaderKind) -> Result<Vec<u32>> {
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
        self.params.set_options(&mut options);

        let compiled = self
            .compiler
            .compile_into_spirv(
                &source,
                kind,
                fname.to_string_lossy().as_ref(),
                "main",
                Some(&options),
            )
            .context("Compiling shader")?;

        Ok(compiled.as_binary().into())
    }
}
