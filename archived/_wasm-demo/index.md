+++
title = "Particle Sim"
date = 2025-04-20
description = "An n-body particle simulation written in Rust, compiled to WebAssembly."

[taxonomies]
tags = ["rust", "wasm", "graphics"]

[extra]
featured = true
wasm_demo = "/demos/particle-sim/"
github = "https://github.com/cielbird/particle-sim"
+++

A real-time particle simulation running entirely in the browser via WebAssembly. The physics loop runs at native speed inside a Rust-compiled `.wasm` module; only the canvas drawing calls cross the JS boundary.

{{ wasm_demo(src="/demos/particle-sim/", height=520, title="Particle Sim", fullscreen=true) }}

## How it works

The core simulation step:

```rust
#[wasm_bindgen]
pub fn step(dt: f32) {
    let mut particles = PARTICLES.lock().unwrap();
    for i in 0..particles.len() {
        for j in (i + 1)..particles.len() {
            let dx = particles[j].x - particles[i].x;
            let dy = particles[j].y - particles[i].y;
            let dist = (dx * dx + dy * dy).sqrt().max(1.0);
            let force = G / (dist * dist);
            particles[i].vx += force * dx / dist;
            particles[i].vy += force * dy / dist;
        }
    }
    for p in particles.iter_mut() {
        p.x += p.vx * dt;
        p.y += p.vy * dt;
    }
}
```

## Deployment

Build with `wasm-pack build --target web`, then copy `pkg/` and `index.html` into `static/demos/particle-sim/`. The shortcode above embeds it via iframe.
