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

- builds complex states on 5 fixed vertices — let h = encoder embedding; D = hidden dim;

$$s_{b,v,d}=\frac{W_{re}(h)_{b,v,d}+i\,W_{im}(h)_{b,v,d}}{\sqrt{D}}$$

  confirmed by core/cell.py:103-119

- builds finite-depth branch-code logits and branch probabilities

  Code: core/cell.py:121-127 -> return self.branch_logits_init(h).reshape(batch_size, self.config.num_vertices, self.config.branch_depth, self.config.branch_base)

  and core/cell.py:200 -> digit_probabilities = torch.softmax(digit_logits, dim=-1)

- calculates prefix matching between branch codes on edges — let q^(s), q^(t) = source/target digit probabilities;

$$a_{b,e,k}=\sum_p q^{(s)}_{b,e,k,p}q^{(t)}_{b,e,k,p},\;\;c_{b,e,k}=\prod_{j\le k}a_{b,e,j},\;\;m_{b,e}=\sum_k c_{b,e,k}$$

  confirmed by ops/reference.py:266-268

- calculates typed transport defects between vertex states

  Code: ops/reference.py:317-320 -> source_transport = selected_source_diagonal * source ; target_transport = selected_target_diagonal * target ; defect = source_transport - target_transport ; defect_norm_sq = defect.abs().square().sum(dim=-1)

- calculates coherence amplitude and phase — let m = prefix depth; δ = defect norm; t = edge type;

$$\phi_{b,e}=b_t+\gamma_t\,m_{b,e},\;\;A_{b,e}=\sigma\!\big(\alpha_t(m_{b,e}-\tau_t)\big)\exp(-|\beta_t|\,\delta_{b,e})$$

  confirmed by ops/reference.py:396-399

- builds complex edge messages

  Code: ops/reference.py:400-403 -> edge_coefficient = torch.polar(edge_amplitude, edge_phase) ; edge_message = edge_mean * edge_coefficient.unsqueeze(-1)

- scatters them back to vertices

  Code: core/cell.py:237-243 -> vertex_message_sum = complex_scatter_add(edge_messages=coherence.edge_message, incidence_index=self.incidence_index, num_vertices=self.config.num_vertices, backend=op_backend)

- updates vertex states

  Code: core/cell.py:245-246 -> self_update = self.self_diagonal.unsqueeze(0) * vertex_state ; vertex_state = self_update + vertex_message_sum

- then returns either Born/codebook readout or logical states to the linear head

  Code: core/cell.py:254 -> readout = self.readout(vertex_state)

  training/cifar10_benchmark.py:397-402 -> class_log_scores = torch.gather(...).squeeze(-1).sum(dim=-1)

  training/cifar10_benchmark.py:476-484 -> logical_vertex_state = cell_outputs.vertex_state[:, LOGIC_VERTEX_INDICES, :] ; class_log_scores = self.classifier(state_features)

All this can be seen and examined in core/cell.py & types.py & ops/reference.py

How the core is structured

If we remove all the noise, the core's mental model is as follows:

- there are 5 internal roles: L, R, -1, 0, +1

  Code: types.py:15 -> VERTEX_LABELS = ("L", "R", "-1", "0", "+1")

### <div align="center">Visualizing</div>

---

<img width="1280" height="720" alt="lsosmfjdhfdh" src="https://github.com/user-attachments/assets/d8b9434f-3854-42ce-8569-141507881432" />

---

This figure shows the fixed internal topology of the pASCNN core

L and R form the wave edge, -1, 0, and +1 form the logic subgraph, and the remaining red edges are cross-connections between the wave and logical parts

Code: types.py:38-54 -> EDGE_ENDPOINTS = (...) ; EDGE_TYPES = ("logic", "logic", "logic", "wave", "cross", ...)

The readout is taken from the updated logical state and is not itself a graph vertex

Code: types.py:29-32 -> LOGIC_VERTEX_INDICES = (VERTEX_INDEX["-1"], VERTEX_INDEX["0"], VERTEX_INDEX["+1"])

and core/readout.py:40-49 -> logic_amplitudes ... logic_probabilities = logic_energy / normalization

---

- between them are fixed typed edges: logic / wave / cross

  Code: types.py:50-64 -> EDGE_TYPES = (...) ; CANONICAL_EDGE_TYPE_INDEX = tuple(EDGE_TYPE_INDEX[edge_type] for edge_type in EDGE_TYPES)

- each vertex has a complex state — let h = encoder embedding; D = hidden dim;

$$
s_{b,v,d}=\frac{W_{re}(h)_{b,v,d}+i\,W_{im}(h)_{b,v,d}}{\sqrt{D}}
$$

  confirmed by core/cell.py:103-119

- each vertex has a finite-depth branch code

  Code: core/cell.py:121-127 -> branch_logits_init(h).reshape(batch_size, self.config.num_vertices, self.config.branch_depth, self.config.branch_base)

  and core/cell.py:200 -> digit_probabilities = torch.softmax(digit_logits, dim=-1)

- branch codes are compared via soft prefix similarity — let q^(s), q^(t) = source/target digit probabilities;

$$a_{b,e,k}=\sum_p q^{(s)}_{b,e,k,p}q^{(t)}_{b,e,k,p},\;\;c_{b,e,k}=\prod_{j\le k}a_{b,e,j},\;\;m_{b,e}=\sum_k c_{b,e,k}$$

  confirmed by ops/reference.py:266-268

- complex states are compared via a typed transport defect

  Code: ops/reference.py:317-320 -> source_transport = selected_source_diagonal * source ; target_transport = selected_target_diagonal * target ; defect = source_transport - target_transport ; defect_norm_sq = defect.abs().square().sum(dim=-1)

- the prefix and defect together control the coherence gate

  Test: .test_artifacts/step39_cifar10_20epoch_compare_2026-04-19/cifar10_benchmark.json logs edge_prefix_depth_to_uniform_ratio_mean, edge_defect_norm_sq_normalized_mean, and edge_amplitude_mean in the same run

- the coherence gate sets the complex edge message

  Code: ops/reference.py:396-403 -> edge_phase = ... ; edge_amplitude = ... ; edge_coefficient = torch.polar(edge_amplitude, edge_phase) ; edge_message = edge_mean * edge_coefficient.unsqueeze(-1)

- edge messages update vertex states

  Code: core/cell.py:238-246 -> vertex_message_sum = complex_scatter_add(...) ; self_update = self.self_diagonal.unsqueeze(0) * vertex_state ; vertex_state = self_update + vertex_message_sum

- the decision is then read from the logical part of this state

  Test: .test_artifacts/step25_frozen_cell_readout_sweep_2026-04-18/frozen_cell_readout_sweep_summary.json compares a0_source_pascnn_codebook vs a3_linear_state_head, i.e. the two readout paths on the same core-state family

In other words, the core is a small typed complex graph machine, not attention-over-patches.

## Part 2 | Two main readout methods

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

## Part 2.1 | Summary

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

Funny note:

Why are codebook and linear_state called that way?
There's one real reason for this.

Explanation

- linear_state says: "Give me the raw state, I'll figure it out myself."

- codebook says: "Each class already has ready-made code, I'll check which code the output most closely resembles."

We can literally see this in code

---


## Resources:

The applications of sheaf theory in deep learning, data science, and computer science in general: https://arxiv.org/abs/2502.15476

Academic Sheaf Theory definition: https://stacks.math.columbia.edu/tag/00VL

Academic P-adic numbers definition: https://mathworld.wolfram.com/p-adicNumber.html or if needed more info https://en.wikipedia.org/wiki/P-adic_number

Neural Sheaf Diffusion: https://arxiv.org/abs/2202.04579
