use std::env;
use std::fs;
use std::io;
use std::path::PathBuf;

fn main() {
    // Force build.rs to run every time to ensure DLL copy happens
    println!("cargo:rerun-if-changed=build_always_trigger");

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();
    if target_os != "windows" {
        println!("cargo:rustc-link-lib=openblas");
        return;
    }

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let openblas_root = out_dir.join("openblas");

    // Download and extract OpenBLAS if not exists
    if !openblas_root.exists() {
        let url = "https://github.com/OpenMathLib/OpenBLAS/releases/download/v0.3.30/OpenBLAS-0.3.30-x64.zip";
        println!("cargo:warning=Downloading OpenBLAS from {}", url);

        let client = reqwest::blocking::Client::builder()
            .build()
            .expect("Failed to create client");

        let resp = client.get(url).send().expect("Failed to download OpenBLAS");

        let content = io::Cursor::new(resp.bytes().expect("Failed to get bytes"));
        let mut archive = zip::ZipArchive::new(content).expect("Failed to open zip");

        archive
            .extract(&openblas_root)
            .expect("Failed to extract OpenBLAS");
    }

    let lib_dir = openblas_root.join("lib");
    let bin_dir = openblas_root.join("bin");

    // Link configuration
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=libopenblas");

    // Also add bin dir to search path for DLLs (might help maturin)
    println!("cargo:rustc-link-search=native={}", bin_dir.display());

    // Copy DLL to target directory to ensure it's available
    // Find target directory (e.g. target/debug)
    let target_dir = out_dir
        .ancestors()
        .nth(3)
        .expect("failed to find target directory");

    let dll_name = "libopenblas.dll";
    let dll_source = bin_dir.join(dll_name);

    if dll_source.exists() {
        let dest = target_dir.join(dll_name);
        let _ = fs::copy(&dll_source, &dest);

        let deps_dir = target_dir.join("deps");
        if deps_dir.exists() {
            let dest_deps = deps_dir.join(dll_name);
            let _ = fs::copy(&dll_source, &dest_deps);
        }
    }

    // Copy DLL to project root for maturin to include it (for wheel build)
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let dest_root = manifest_dir.join(dll_name);
    if dll_source.exists() {
        match fs::copy(&dll_source, &dest_root) {
            Ok(_) => println!("cargo:warning=Copied {} to project root", dll_name),
            Err(e) => println!("cargo:warning=Failed to copy to project root: {}", e),
        }
    }

    // Copy to site-packages for maturin develop (for local development)
    // Try to determine site-packages from python command or VIRTUAL_ENV
    let site_packages = if let Ok(output) = std::process::Command::new("python")
        .args([
            "-c",
            "import sysconfig; print(sysconfig.get_path('purelib'))",
        ])
        .output()
    {
        if output.status.success() {
            Some(PathBuf::from(
                String::from_utf8_lossy(&output.stdout).trim(),
            ))
        } else {
            None
        }
    } else {
        None
    };

    let site_packages = site_packages.or_else(|| {
        env::var("VIRTUAL_ENV").ok().map(|venv| {
            PathBuf::from(venv).join("Lib").join("site-packages")
        })
    });

    if let Some(site_packages) = site_packages {
        let package_dir = site_packages.join("fir_decimator");
        // Ensure directory exists (maturin might have created it, or not yet)
        if !package_dir.exists() {
            let _ = fs::create_dir_all(&package_dir);
        }

        if package_dir.exists() {
            let dest = package_dir.join(dll_name);
            match fs::copy(&dll_source, &dest) {
                Ok(_) => println!(
                    "cargo:warning=Copied {} to site-packages/fir_decimator",
                    dll_name
                ),
                Err(e) => println!("cargo:warning=Failed to copy to site-packages: {}", e),
            }
        }
    }
}
