+++
title = "Geþeode - Phonetics Engine"
description = "A phonetics engine for creative linguistics projects, written in Rust"
date = 2025-01-01

[extra]
image = "rules.png"
github = "https://github.com/Cielbird/getheode"

[taxonomies]
tags = ["rust", "wasm", "linguistics"]
+++

Here is all the information about my linguistics project: Geþeode

## Blogs
- [Modeling phonology](@/blog/modeling_phonology.md)

## Example demo

This demo showcases the `getheode apply` tool.

{{ wasm_demo(src="/demos/getheode-demo/", height=120, title="Geþeode Apply Demo", fullscreen=true) }}

### Syntax

Rules follow the notation `X → Y / A _ B`: change X to Y when preceded by A and followed by B.

| Example | Meaning |
|---|---|
| `X -> Y` | change X to Y unconditionally |
| `X -> Y / A _ B` | change X to Y when preceded by A, followed by B |
| `_` | position of the target in the environment |
| `#` | word boundary |
| `$` | syllable boundary |
| `V`, `C` | natural classes — vowel, consonant |
| `[+voi]` | feature set (any combination of `±` features) |
| `(X)` | X is optional |
| `{a,b,c}` | one of a, b, or c |
| `∅` or `Ø` | empty segment (deletion or insertion) |
| `A B → X Y` | multiple simultaneous rules |

### Examples

| Rule | Input | Output | Description |
|---|---|---|---|
| `t → d / V_V` | `ata` | `ada` | intervocalic voicing |
| `ə → ∅ / _#` | `taɪmə` | `taɪm` | word-final schwa deletion |
| `θ ð → t d` | `ði.θo` | `di.to` | multiple rules applied simultaneously |
| `{n,q,h} → Ø / _(s)` | `oqs.in.ihso` | `os.i.iso` | deletion before optional s |
| `n → l / #_(V){s,ʃ,h}V{m,b}#` | `niham` | `liham` | word-initial context with alternatives |
| `j → ∅ / Ck_$` | `eskj.mo` | `esk.mo` | deletion at syllable boundary |
