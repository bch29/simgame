use std::env;
use std::io::Read;
use std::io::Write;
use std::fs::File;
use std::path::Path;

fn main() {
    let out_dir = env::var("OUT_DIR").unwrap();

    let vert_dest = Path::new(&out_dir).join("shader.vert.spv");
    let frag_dest = Path::new(&out_dir).join("shader.frag.spv");

    compile_shaders(&vert_dest, &frag_dest);
}

fn compile_shaders(vert_dest: &Path, frag_dest: &Path) {
    println!("cargo:rerun-if-changed=shader/shader.vert");
    println!("cargo:rerun-if-changed=shader/shader.frag");

    let mut compiler = shaderc::Compiler::new().unwrap();
    // options.add_macro_definition("EP", Some("main"));
    {
        println!("Compiling vertex shader");
        let mut source = String::new();
        File::open("shader/shader.vert").unwrap().read_to_string(&mut source).unwrap();

        let options = shaderc::CompileOptions::new().unwrap();
        let compiled = compiler.compile_into_spirv(
            &source,
            shaderc::ShaderKind::Vertex,
            "shader.vert",
            "main",
            Some(&options),
        ).unwrap();

        File::create(vert_dest).unwrap().write_all(compiled.as_binary_u8()).unwrap();
    };

    {
        println!("Compiling fragment shader");
        let mut source = String::new();
        File::open("shader/shader.frag").unwrap().read_to_string(&mut source).unwrap();

        let options = shaderc::CompileOptions::new().unwrap();
        let compiled = compiler.compile_into_spirv(
            &source,
            shaderc::ShaderKind::Fragment,
            "shader.frag",
            "main",
            Some(&options),
        ).unwrap();

        File::create(frag_dest).unwrap().write_all(compiled.as_binary_u8()).unwrap();
    };
}
