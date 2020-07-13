use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ShaderKind {
    Fragment,
    Vertex,
    Compute,
}

#[derive(Debug, Clone, Copy)]
pub struct CompileParams {
    pub chunk_size: (i64, i64, i64),
}

/// Real implementation when "shader-compiler" feature is enabled.
#[cfg(feature = "shader-compiler")]
mod shaderc_impl {
    use super::*;

    use std::cell::RefCell;
    use std::fs::File;
    use std::io::Read;
    use std::path::Path;

    use anyhow::{anyhow, Context, Result};

    pub struct ShaderCompiler {
        params: CompileParams,
        compiler: RefCell<shaderc::Compiler>,
    }

    impl ShaderCompiler {
        pub fn new(params: CompileParams) -> Result<Self> {
            Ok(Self {
                params,
                compiler: RefCell::new(
                    shaderc::Compiler::new().context("Instantiating shaderc compiler")?,
                ),
            })
        }

        pub fn compile(&self, path: &Path, kind: ShaderKind) -> Result<Vec<u32>> {
            let fname = path
                .file_name()
                .ok_or_else(|| anyhow!("Expected file path, got directory path"))?;

            let mut source = String::new();

            File::open(path)
                .with_context(|| format!("Could not open shader file {:?}", fname))?
                .read_to_string(&mut source)
                .with_context(|| format!("Could not read shader file {:?}", fname))?;

            let mut options = shaderc::CompileOptions::new()
                .ok_or_else(|| anyhow!("Creating shaderc compile options"))?;
            options_from_params(&self.params, &mut options);

            let mut compiler = self.compiler.borrow_mut();

            let compiled = compiler
                .compile_into_spirv(
                    &source,
                    kind_to_shaderc(kind),
                    fname.to_string_lossy().as_ref(),
                    "main",
                    Some(&options),
                )
                .with_context(|| format!("Could not compile shader {:?}", fname))?;

            Ok(compiled.as_binary().into())
        }
    }

    fn kind_to_shaderc(kind: ShaderKind) -> shaderc::ShaderKind {
        match kind {
            ShaderKind::Fragment => shaderc::ShaderKind::Fragment,
            ShaderKind::Vertex => shaderc::ShaderKind::Vertex,
            ShaderKind::Compute => shaderc::ShaderKind::Compute,
        }
    }

    fn options_from_params(params: &CompileParams, options: &mut shaderc::CompileOptions) {
        let mut define_chunk_size = |dim, val| {
            let name = format!("CHUNK_SIZE_{}", dim);
            let val_str = format!("{}", val);
            options.add_macro_definition(name.as_str(), Some(val_str.as_str()));
        };

        let (x, y, z) = params.chunk_size;
        define_chunk_size("X", x);
        define_chunk_size("Y", y);
        define_chunk_size("Z", z);
    }
}

/// Dummy implementation when "shader-compiler" feature is not enabled.
#[allow(dead_code)]
mod dummy_impl {
    use super::*;

    use anyhow::{anyhow, Result};
    use std::path::Path;

    pub struct ShaderCompiler {
        params: CompileParams,
    }

    impl ShaderCompiler {
        pub fn new(params: CompileParams) -> Result<Self> {
            Ok(Self { params })
        }

        pub fn compile(&self, _path: &Path, _kind: ShaderKind) -> Result<Vec<u32>> {
            Err(anyhow!(
                "Cannot compile shaders: simgame_render not built with \"shader-compiler\" feature"
            ))
        }
    }
}

#[cfg(not(feature = "shader-compiler"))]
pub use dummy_impl::ShaderCompiler;

#[cfg(feature = "shader-compiler")]
pub use shaderc_impl::ShaderCompiler;
