+++
title = "Modeling a conlang's phonology"
date = "2026-04-25"

[taxonomies]
tags=["rust", "games"]

[extra]
comment = true
+++

# Context

I've worked on making a couple fictional languanges in the past. They're known as 
[conlangs](en.wikipedia.org/wiki/Constructed_language), and there's a whole online community 
around the art of constructing fictional languages.

Something people like to do is to "evolve" their languages to create dialects or descendent 
languages. You apply phonological and grammatical changes to the language to create a new one, 
much like real languages change over time.

# Geþeode: the project

After tediously applying changes to words in my fictional languages, I wanted to make software to
automate the process. I called this project "geþeode", after an Old English word meaning language.

I started with phonological changes, or sound changes, as they are pretty predictable. They can 
also be pretty simple.
The same sound changes often occur across different languages and in different time periods. 
As an example, here's a change that happened from classical latin (~200 BCE) to vulgar 
latin (~500 BCE) :

> kʷ → ɡʷ / V_V

This is a pretty standard notation. Here it means a "kw" sound became voiced when it occurred 
in contexts between two vowels (V). Words like "aqua" [ˈaɡ.wa] became pronounced more like 
"agua" [ˈaɡ.wa]. Modern Spanish "agua" [ˈaɣ.wa] inherited from this change.

If you're unfamiliar with this ([ˈaɣ.wa]) notation, it's the 
[International Phonetic Alphabet](https://en.wikipedia.org/wiki/International_Phonetic_Alphabet). 
Parsing IPA notation was a large part of this project.

An important part of this was to stay close to real linguistic notation. I didn't want to invent my 
own notation, which is something most other tools do. For reference, I used
[this amazing compilation](https://chridd.nfshost.com/diachronica/) of sound 
changes.

## Development

Being a performance obsessed low-level software guy, I decided to code everything in Rust.

I used `nom` for parsing, which has been such a great experience. I will never write my own parser 
again in rust !

## Universal phonological model

Linguists can't agree on how language sounds exist and work in our brains, so there's no "universal"
phonological model. However, there's concensus on enough stuff for me to pick a middle ground. 
Linguists agree that all phonological sounds (called segments) can be defined as sets of
[distinctive features](https://en.wikipedia.org/wiki/Distinctive_feature). Either the feature is
present (+) or not (-), so they're binary.

Most other sound change tools like [SCA2](https://www.zompist.com/sca2.html) require you to 
essentially define the phonological model you use. They don't define a universal set of 
phonological features, and they're basically fancy text editing engines. For many cases, it's quite 
enough.

A pretty defining trait of my tool is that I *assume* a certain phonological model, and try to 
accommodate for all use cases.

## Trees, not strings

Most features define individual segments. However, some features describe syllables, 
and some describe groups of syllables (called feet). These are 
[suprasegmental features](https://en.wikipedia.org/wiki/Prosody_(linguistics)#Phonology), and are 
usually things like stress, or tone. They're actually pretty few in number compared to regular 
segmental features, so it's easy to put them aside. All the sound change software tools I've found
ignore these, or deal with them in hacky ways.
But if you account for them, the model of a phonological sequence completely changes. 
It goes from a simple string to **a tree**. Here's "bailar" (Spanish, to dance) decomposed 
for illustration :

```
        [ _ ]              < word or foot level features
       /      \
 [-stress]     [+stress]    < syllable level features
 |   |   |     |    |   |
[b] [a] [ɪ]   [l]  [a]  [ɾ]   < segment level features
```

The model is a tree, where each node has a set of features relative to it's depth, and where 
each leaf is at the maximum depth.

Some questions are hard to answer :
- How many levels in the tree ?
  - Can sound changes consider [prosodic foot](http://glottopedia.org/index.php/Foot) ?
- Which features should exist ?
  - What binary features should be used to describe different rising/falling tones ?

## Rules

Now since the phonological sequence is a tree, sound changes must be interpreted as tree operations.

Getheode actually compiles sound changes into a set of search and replace operations that not 
only match segment features, but also surrounding and internal tree structure. This model allows 
for much easier application of sound changes.

For example, the following rule only applies in certain word-boundary contexts :

> t -> d / V#_V

Here, a [t] sound becomes voiced, becoming [d], when it is word-initial (# indicates word boundary),
and where the surrounding sounds are vowels.

## Branching

Getheode also supports optionals or branches in the rule. For example :

> {n,q,h} -> Ø / _(s)a

Here, either a [n], a [q], or a [h] are removed, before a [a] or a [sa]. This rule is imaginary 
and very unlikely. But it works in Getheode.

## Tagging

The tool supports manually tagging features. For example :

> V_0ʔV_0 -> Vː_0

Here, we want two vowels (V) that are the same (both have _0) that surround a glottal stop 
(ʔ) merge into one long (ː) vowel , removing the glottal stop.

(The glottal stop is the sound you make when you go "a, a, a" = [ʔaʔaʔa])

Under this sound change, [aʔaʔila] becomes [aːʔila].

## The largest issue

Many sound change rules ignore syllable and word boundaries. However, a rule's output cannot be 
ambiguous, and it cannot be easily inferred. So, this means that some rules are interpreted more 
strictly than they should, which I'll talk about in a different post. 

## Conclusion

The project's page is [here](@/projects/getheode/index.md). Most rules are applied correctly.

I will make a WASM web demo soon.

Next, I will tackle [phonotactics](https://en.wikipedia.org/wiki/Phonotactics), 
which are the rules that govern how phonemes can be arranged in 
words in different languages. I'd like to apply a set of sound changes on not only a set of words, 
but a generalized phonotactic rule set.
