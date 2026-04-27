+++
title = "Designing a 12 bit processor"
date = "2026-04-26"

[taxonomies]
tags=["vlsi"]

[extra]
comment = true
+++

I wanted to post this beauty here. 

{{ image(path="blog/r12_layout/layout.png", alt="R12 CPU Layout")}}

This is a 12 bit processor, designed from the ground up, with the help of a friend. 
It's my first time designing an integrated circuit this large and complicated.

What you're seeing is the layout for fabrication on silicon on a 180nm TSMC process. 
Each most colors here are different metal interconnect layers. 
Below them in blue are TSMC's logic cells. 
The inputs and outputs go off the screen. Those are the data and address bus lines.

I'm pretty satisfied with the performance. With my limited setup I was able to design it for a
whopping 28.6MHz clock speed ! That takes into account all sorts of delays due to logic cell 
capacitances and interconnect parasitics.

And yes, it works. Here's a simple program running :

{{ image(path="blog/r12_layout/modelsim_sim.png", alt="R12 CPU Sim")}}
