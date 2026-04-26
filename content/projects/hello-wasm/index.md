+++
title = "Hello WASM"
date = 2026-04-26
description = "A minimal WebAssembly demo: a Rust function called from JavaScript."

[taxonomies]
tags = ["rust", "wasm"]

[extra]
featured = true
wasm_demo = "/demos/hello-wasm/"
github = "https://github.com/Cielbird/Cielbird.github.io/tree/main/demos/hello-wasm"
+++

A minimal Rust + WebAssembly example. A single `add` function is compiled to `.wasm` and called from JavaScript.

{{ wasm_demo(src="/demos/hello-wasm/", height=300, title="Hello WASM", fullscreen=true) }}

## Rust source

```rust
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn add(a: i32, b: i32) -> i32 {
    a + b
}
```

## Build & deploy

From the repo root:

```sh
bash demos/hello-wasm/build.sh
```
