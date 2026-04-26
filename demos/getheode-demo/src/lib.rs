use wasm_bindgen::prelude::*;

use getheode::phonology::{rule::{PhonoRuleParseOpts, PhonoRuleSet}, string::PhonoString};

#[wasm_bindgen]
pub fn apply_rule(rule: &str, input: &str) -> Result<String, JsValue> {
    let opts = PhonoRuleParseOpts::default();

    let rule_set = PhonoRuleSet::parse(rule, opts)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let (rem, string) = PhonoString::parse(input)
        .map_err(|e| JsValue::from_str(&format!("{e:?}")))?;

    if !rem.is_empty() {
        return Err(JsValue::from_str(&format!(
            "Couldn't parse input, remainder: \"{rem}\""
        )));
    }

    Ok(rule_set.apply(string).to_string())
}
