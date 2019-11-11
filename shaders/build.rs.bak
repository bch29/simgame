use std::env;
use std::fs::File;
use std::io::Read;
use std::io::Write;
use std::path::Path;

fn main() {
    let out_dir = env::var("OUT_DIR").unwrap();

    let vert_dest = Path::new(&out_dir).join("shader.vert.spv");
    let frag_dest = Path::new(&out_dir).join("shader.frag.spv");

    compile_shaders(&vert_dest, &frag_dest);
}

fn compile_shaders(vert_dest: &Path, frag_dest: &Path) {
    println!("cargo:rerun-if-changed=shader.vert");
    println!("cargo:rerun-if-changed=shader.frag");

    let mut compiler = shaderc::Compiler::new().unwrap();
    // options.add_macro_definition("EP", Some("main"));
    {
        println!("Compiling vertex shader");
        let mut source = String::new();
        File::open("shader.vert")
            .unwrap()
            .read_to_string(&mut source)
            .unwrap();

        let options = shaderc::CompileOptions::new().unwrap();
        let compiled = match compiler.compile_into_spirv(
            &source,
            shaderc::ShaderKind::Vertex,
            "shader.vert",
            "main",
            Some(&options),
        ) {
            Ok(x) => x,
            Err(e) => {
                println!("{}", e);
                std::process::exit(1);
            }
        };

        File::create(vert_dest)
            .unwrap()
            .write_all(compiled.as_binary_u8())
            .unwrap();
    };

    {
        println!("Compiling fragment shader");
        let mut source = String::new();
        File::open("shader.frag")
            .unwrap()
            .read_to_string(&mut source)
            .unwrap();

        let options = shaderc::CompileOptions::new().unwrap();
        let compiled = match compiler.compile_into_spirv(
            &source,
            shaderc::ShaderKind::Fragment,
            "shader.frag",
            "main",
            Some(&options),
        ) {
            Ok(x) => x,
            Err(e) => {
                println!("{}", e);
                std::process::exit(1)
            }
        };

        File::create(frag_dest)
            .unwrap()
            .write_all(compiled.as_binary_u8())
            .unwrap();
    };
}
