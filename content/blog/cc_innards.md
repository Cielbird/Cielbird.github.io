+++
title = "Claude Code's innards"
date = "2026-09-16"  

[taxonomies]
tags=["rant"]
+++

I don't like JavaScript. And I'm not alone, I've found this is a common sentiment among many of my 
peers.

Web frameworks like Tailwind and React are bandages on top of a broken archaic
system. The HTML/CSS/JS holy trinity was never meant to serve the current demands
of the web. Modern web apps work by wrangling fancy electronic snail-mail into interactive 
environments. For every crack in the ship that is modern web development, we slap on more hunks 
of software, making the whole ecosystem more frankensteined than it ever needed to be. 
JavaScript is a language that we (as developers) will probably never shake off.

And the JavaScript fever has spread. The most hyped-up CLI tool today, Claude Code, is written 
entirely in JS (Well, TS transpiled to JS). Lets think about that for two seconds. 
A *terminal programming tool*, written in *web scripting language*.
A program with one of the most basic interface requirements (the command line) doing nothing more 
than text and file manipulation and API calls, built with one of the most complex and 
badly designed language environments ever. When you install Claude Code, the binary you're 
installing is literally a JS interpreter bundled with a fat-ass string containing JavaScript code.
Today, with languages like Go and Rust, safe and fast CLI tools are more accessible than ever. 
From an engineering POV, it's ridiculous.

Something has clearly gone wrong in today's software world. Imagine if everyone making a CLI tool 
used this strategy. Every single program on your machine (`git`, `grep`, `curl`) would launch its 
own personal language interpreter and runtime, which would in turn run the program, stored as a 
string in the executable. It would be chaotic. And yet, they took this choice. Why ?

We have a direct answer 
[from the team at Anthropic](https://newsletter.pragmaticengineer.com/p/how-claude-code-is-built): 
it was a tech stack that the LLM was already good at. So since Claude (the model) is best with TS 
and React, lets make it build our CLI tool with TS and React.

First of all, Claude code is perfectly capable of writing high quality Rust and CPP code.
I know because I've used Claude to write useful and fast CLI tools, with fun interactive TUIs.
To be fair, it has been 1.5 years since the first release of Claude Code, but I still believe Claude
would have been capable of writing the tool in a compiled language. 
They have a tool capable of writing code very well in any language. Claude may be trained more on 
JS, but it is perfectly capable of writing spotless C++, C, or Rust. Claude isn't bad enough in 
those systems languages to deserve being restricted to JS for a CLI tool. 
Plus, these systems languages would offer much better performance, so they'd be the obviously better 
engineering design choice. 

> Not to mention it's a huge anti-advertisement for Anthropic's product. 
> Your LLM is only capable of coding correctly in JS ?!

It's absurd reasoning, and because of that, I think there's a deeper reason for using JS. 

I think it's a cultural problem in the industry. 
There was a huge push in the 2010s (for very valid reasons) to vulgarize computer science with 
"simpler" languages like Python and JS. This created waves of developers that are simply 
unfamiliar with systems programming languages. Sure, they took a C++ course in college, sure they 
know what the stack and heap are, but they don't have actual field experience with these languages.
Sure, compiled languages are a *little* more complicated than JS, but anyone with practical 
experience can tell you there are *very good* reasons to use them. But that requires practical 
experience.

I think the industry, especially in the hyped-up AI world, has prioritized the wrong things.
They would test "practical knowledge" with toy leetcode problems where 99% of users will use 
Python or JS. I'm sorry, that's not practical at all. They reward short term results over 
reliability and safety. 

Unfortunately, another important point is that *it's not a big enough issue to cause a problem now*.
For now, the CPU, RAM and disk usage of Claude Code CLI aren't shocking (because it's a stupid 
simple application, duh), so nobody bats an eye.

I think the developers who created Claude Code had so little experience with actually making a CLI 
tool, so they reached for the only thing they actually knew: JS. And of course, they didn't 
go through the hassle of actually developing it, because their LLM took the blunt work. As we know, 
LLMs don't complain much, so the underlying design failure went unseen. 

To be fair, this is all my personal opinion and speculation.
