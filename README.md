# pASCNN
pASCNN: an experimental research neural network architecture that uses sheaf-theoretic restrictions and a p-adic ultrametric to structure logical inference in complex space

disclaimer: pASCNN is the experimental variant of unusual neural network architecture, I maintain that the tests that were conducted were correct, but they may perhaps be somewhat less rigorous, reader can by himself check all code and make a correctness verdict ; This document prioritizes clarity over strict formality

My work is not SOTA, this arch is extremely new, but it produces results consistently higher than those presented to it by its opponents, the basic ViT-like transformer

Most of the code was written using LLM (which can make the code difficult to read) under my complete control. The code was reviewed, but I could inevitably miss something. So, when talking about the correctness of the work done, I will refer to the correct test results and partially to the code, but it is obvious that the code cannot be without bugs and gaps

The work was completed by one person

I also don't deny that similar work may have been done, but I personally haven't seen architectures with the makings of p-adic structures and bundles, along with a ternary readout system in the form of a codebook (which can currently compete with others in only one of three tests conducted)

Here the description about technology

## Part 0 | Definitions & Scope:

---

About Prerequisites

This text assumes that the reader is familiar with the basics of higher algebra. We use standard notation and terminology (sets, mappings, compositions), so the repository does not provide detailed explanations of basic mathematical constructs

---

Let's introduce some specifics for further understanding, so we can distinguish between reality and implementation.

In the tests, you can see two different pASCNN types:
codebook and linear_state.
What are these? These are different readout variants from the core, running on the same pASCNNCell (more on this below)

The mathematics isn't the most rigorous, I tried to convey this in the title

Regarding p-adic, if we take the [academic definition of p-adic numbers](https://en.wikipedia.org/wiki/P-adic_number), then compared to pASCNN, we get the following verdict:

no Q_p, no infinite expansions, no true norm

Why? Doing this on a regular PyTorch, on a regular GPU -> a road to nowhere

Instead, an interesting system is used: "a computational surrogate for the p-adic ultrametric"

It's a rather complex name, but it makes sense due to the prefixes used in the architecture and hierarchical paths

These ideas are similar to the basic idea of ​​p-adic data structures and are suitable for constructing trees. Therefore, in what follows, I will argue that this is a minimal p-adic representation in the context of neural network architectures. When I speak of p-adic, I will mean exactly this

Regarding sheaf, if we take the [academic definition of sheaf theory](https://stacks.math.columbia.edu/tag/00VL), then in comparison with pASCNN, we get the following verdict:

No sheaf objects, no gluing axioms, no functors

If we refer to papers on sheaf neural networks, then

In papers on [Neural Sheaf Diffusion](https://arxiv.org/abs/2202.04579) and related SNN papers, cellular sheaf on a graph is defined as follows: each node and edge is associated with a vector space, and each The incident node-edge pair is a linear mapping F_v◃e​:F(v)→F(e)), called a restriction map. This is precisely the level of definition used in ML works.

What does pASCNN actually contain from sheaf theory, and why does the architecture qualify for the name sheaf-coherence, but with a caveat (see below):

- Vertices with complex states vertex_state on a fixed 5-vertex graph;

- Typed linear transport maps source_transport_diagonal and target_transport_diagonal;

- Transported edge-aligned states source_transport, target_transport

This is clearly visible in PASCNNCell, TransportMaps, and complex_restrict_defect

The most correct formulation (clause) would be:
pASCNN contains an explicit cellular-sheaf-style transport/consistency mechanism on a fixed graph cell

From now on, when I talk about sheaf, I will mean exactly this definition, and you should understand it

At the moment these are all the basic definitions, more may be added in the future

---

## Part 1 | Core Structure

I mentioned above that the architecture has two distinct readouts that are quite different from each other.

These are **codebook** and **linear_state**.

If we were to characterize it, it's most likely a hybrid with a basic ViT-like frontend; the non-standard part is specifically in PASCNNCell (or, for convenience, I'll call it Core).

First, I want to describe how the most interesting part of Core works. I'll do this so that we can further analyze linear_state and codebook and understand what they work with.

What's non-standard about the model?

The non-standard part begins after the encoder.

Instead of a regular linear classifier on top of the encoder embedding, pASCNN has a separate PASCNNCell. This core:

- builds complex states on 5 fixed vertices

- builds finite-depth branch-code logits and branch probabilities

- calculates prefix matching between branch codes on edges

- calculates typed transport defects between vertex states
- calculates coherence amplitude and phase

- builds complex edge messages

- scatters them back to vertices

- updates vertex states

- then returns either Born/codebook readout or logical states to the linear head

All this can be seen and examined in core/cell.py & types.py & ops/reference.py

How the core is structured

If we remove all the noise, the core's mental model is as follows:

- there are 5 internal roles: L, R, -1, 0, +1

- between them are fixed typed edges: logic / wave / cross

- each vertex has a complex state

- each vertex has a finite-depth branch code

- branch codes are compared via soft prefix similarity

- complex states are compared via a typed transport defect

- the prefix and defect together control the coherence gate

- the coherence gate sets the complex edge message

- edge messages update vertex states

- the decision is then read from the logical part of this state

In other words, the core is a small typed complex graph machine, not attention-over-patches.

## Part 3 | Two main readout methods

As you read above, there are 2 different readouts in pASCNN (or 2 different readout branches), one of them is standard and the other is not quite

Let's start with linear_state (or pASCNN linear_state).

In all currently available tests, the pASCNN linear_state architecture can be described as follows:

image -> ViT-like patch encoder -> PASCNNCell -> decoder

### What is a linear_state branch?

How a ViT-like patch encoder works

The usual part is the image encoder.

It does exactly what you'd expect from a small ViT-like encoder:

- cuts the image into patches

- projects the patches into the embedding space

- adds a class token

- adds positional embedding

- runs everything through a TransformerEncoder

- takes an embedding class token

There's not much new to say here; it's a well-known and classic method

What's interesting is that, due to its simplicity and pragmatism, it wins in most tests, although there have been cases where **codebook** has outperformed it

We won't dwell on this for long and will move on to the most interesting example

### What is a codebook branch?

Codebook is a more "internal-architectural" option
and, in my opinion, one of the most appropriate for this particular Core. Everything generally depends on testing; I'll explain this more specifically in the summary.

The scheme is as follows:

image -> encoder -> PASCNNCell -> ternary Born-style readout -> fixed codebook decoder

What does this mean?

- the core first generates logical amplitudes and logical probabilities
- these logical probabilities have a ternary structure
- then the class is built using a fixed codebook on top of these ternary probabilities

How should this be understood?

- this branch tries to force the solution to follow the logical ternary structure of the core itself
- this is a more rigid and "ideological" decoder
- if the logical structure within the core is well-formed, this is a strong option
- if it is not, this decoder becomes a bottleneck

## Part 3.1 | Summary

In that summary i want to say about "Everything generally depends on testing", what this mean?

In short:

- a codebook is better where the solution has already been well compressed into discrete ternary logical probabilities

- a linear_state is better where useful information still exists in a fully complex logical state and has not been reduced to a pure ternary bottleneck

Let's add more specifics

You may have already noticed the difference when I explained what these branches actually are. For a complete understanding, I'll elaborate on this.

In code, the difference is exactly this:

- Codebook takes readout.logic_probabilities, which is a Born-style compressed representation, and then reads the class through
a fixed codebook.

- Linear_state takes logical_vertex_state.real/imag and throws them into the linear head.

This is a significant difference. I'll now give you my guesses on situations where one might be better or worse than the other.

These guesses will be supported by the results of tests conducted in this repository.

### When Codebook is likely better:

- When classes are naturally expressed through small, discrete code.

- When strong compression and regularization are beneficial, rather than the widest possible readout.

- When Core has already learned to produce clean and stable logical probabilities.

- When extra real/imag geometry is more noise than help.

This is similar to what we saw on MNIST:

- codebook: 98.81%

- linear_state: 98.35%

The situation here is twofold: codebook is higher than linear_state, but only by 0.5%, but we'll still take this into account
Conclusion: That is, for "pure" discrete classes, codebook may even be better

### Where linear_state is likely better?

- When the useful signal hasn't yet folded into neat ternary probabilities

- When the class depends on finer state geometry, not just on "which logical bin is most probable"

- When a wider readout channel is needed

- When the core is alive, but the logical bottleneck isn't yet perfectly formed

This is very clearly visible in the frozen-cell probe:

- source codebook: 26.04%

- linear_state head on the same states: 44.16%

Practical conclusion: This is perhaps the strongest signal: the states already contain useful information, but the codebook can't always extract it

### What about CIFAR-10?
On CIFAR-10, the picture so far favors linear_state.

- linear_state: best 52.75%
- codebook: best 48.95%
Artifact: .test_artifacts/
step39_cifar10_20epoch_compare_2026-04-19/cifar10_benchmark.json

And on the low-data sweep, linear_state is also systematically higher than codebook at all budgets, although not always higher than transformer:

- 50/class: 22.6% vs. 16.3%
- 500/class: 37.7% vs. 31.15%
- 1000/class: 45.65% vs. 40.3%
Artifact: .test_artifacts/
step41_cifar10_lowdata_3seed_selected_2026-04-19/lowdata_3seed_summary.json

This suggests that for natural images, linear_state is now generally safer and stronger

### What about relation OOD?
This is where things get more interesting

On one successful seed of 8, codebook was slightly higher than linear_state:

- 75.4% vs. 75.2%
But on seed 3, the average is already in favor of linear_state:
- linear_state OOD mean: 72.83%
- codebook OOD mean: 64.2%
And most importantly: codebook has a very wide spread of seeds. Artifact: .test_artifacts/
step47_left_right_relation_expanded_ood_pool_3seed_2026-04-20/left_right_relation_3seed_summary.json

That is, relational problems may be codebook-friendly, but linear_state seems more stable for now

You might have some questions when I talked about where exactly a particular readout branch might be better

### What might quality depend on?

**Problem complexity?***

- Indirectly, yes, but that's a bad axis
- It's better to think not about "complexity/simpleness," but rather "how compressible is the solution into small, discrete logical code"

**Dimensionality?**

- Yes, but also indirectly
- Linear_state directly reads 3 * hidden_dim * 2 features from logical states
- Codebook doesn't benefit as much from increasing hidden_dim; It only wins if a larger core makes the logical
probabilities cleaner
- Therefore, increasing hidden_dim often helps linear_state more, although this isn't an ironclad law

**Due to the test itself?**

- Yes. That's essentially the main factor
- MNIST and some relational modes are more like problems where discrete bottlenecks can work
- CIFAR and frozen-state probe are currently stronger in favor of linear_state

## Resources:

The applications of sheaf theory in deep learning, data science, and computer science in general: https://arxiv.org/abs/2502.15476

Academic Sheaf Theory definition: https://stacks.math.columbia.edu/tag/00VL

Academic P-adic numbers definition: https://mathworld.wolfram.com/p-adicNumber.html or if needed more info https://en.wikipedia.org/wiki/P-adic_number

Neural Sheaf Diffusion: https://arxiv.org/abs/2202.04579
