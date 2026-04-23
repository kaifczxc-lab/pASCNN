# pASCNN
pASCNN: an experimental research neural network architecture that uses sheaf-theoretic restrictions and a p-adic ultrametric to structure logical inference in complex space

disclaimer: pASCNN is the experimental variant of unusual neural network architecture, I maintain that the tests that were conducted were correct, but they may perhaps be somewhat less rigorous, reader can by himself check all code and make a correctness verdict ; This document prioritizes clarity over strict formality

My work is not SOTA, this arch is extremely new, but it produces results consistently higher than those presented to it by its opponents, the basic ViT-like transformer

Most of the code was written using LLM (which can make the code difficult to read) under my complete control. The code was reviewed, but I could inevitably miss something. So, when talking about the correctness of the work done, I will refer to the correct test results and partially to the code, but it is obvious that the code cannot be without bugs and gaps (I would like to point out that LLM did not invent this architecture, did not invent the formula, but only implemented the architecture in code)

The work was completed by one person

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

Regarding p-adic, if we take the [academic definition of p-adic numbers](https://mathworld.wolfram.com/p-adicNumber.html), then compared to pASCNN, we get the following verdict:

no Q_p, no infinite expansions, no true norm

Why? Doing this on a regular PyTorch, on a regular GPU -> a road to nowhere

full Q_p arithmetic is not implemented in the current PyTorch/GPU realization

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

  confirmed by [core/cell.py:103-119](https://github.com/kaifczxc-lab/pASCNN/blob/SiritoriProjects/core/cell.py#L103-L119)

- builds finite-depth branch-code logits and branch probabilities

  Code: [core/cell.py:121-127](https://github.com/kaifczxc-lab/pASCNN/blob/SiritoriProjects/core/cell.py#L121-L127) -> return self.branch_logits_init(h).reshape(batch_size, self.config.num_vertices, self.config.branch_depth, self.config.branch_base)

  and [core/cell.py:200](https://github.com/kaifczxc-lab/pASCNN/blob/SiritoriProjects/core/cell.py#L200) -> digit_probabilities = torch.softmax(digit_logits, dim=-1)

- calculates prefix matching between branch codes on edges — let q^(s), q^(t) = source/target digit probabilities;

$$
a_{b,e,k}=\sum_p q^{(s)}_{b,e,k,p}q^{(t)}_{b,e,k,p}
$$
$$
c_{b,e,k}=\prod_{j\le k}a_{b,e,j}
$$
$$
m_{b,e}=\sum_k c_{b,e,k}
$$

  confirmed by [ops/reference.py:266-268](https://github.com/kaifczxc-lab/pASCNN/blob/SiritoriProjects/ops/reference.py#L266-L268)

- calculates typed transport defects between vertex states

  Code: [ops/reference.py:317-320](https://github.com/kaifczxc-lab/pASCNN/blob/SiritoriProjects/ops/reference.py#L317-L320) -> source_transport = selected_source_diagonal * source ; target_transport = selected_target_diagonal * target ; defect = source_transport - target_transport ; defect_norm_sq = defect.abs().square().sum(dim=-1)

- calculates coherence amplitude and phase — let m = prefix depth; δ = defect norm; t = edge type;

$$\phi_{b,e}=b_t+\gamma_t\,m_{b,e},\;\;A_{b,e}=\sigma\!\big(\alpha_t(m_{b,e}-\tau_t)\big)\exp(-|\beta_t|\,\delta_{b,e})$$

  confirmed by [ops/reference.py:396-399](https://github.com/kaifczxc-lab/pASCNN/blob/SiritoriProjects/ops/reference.py#L396-L399)

- builds complex edge messages

  Code: [ops/reference.py:400-403](https://github.com/kaifczxc-lab/pASCNN/blob/SiritoriProjects/ops/reference.py#L400-L403) -> edge_coefficient = torch.polar(edge_amplitude, edge_phase) ; edge_message = edge_mean * edge_coefficient.unsqueeze(-1)

- scatters them back to vertices

  Code: [core/cell.py:237-243](https://github.com/kaifczxc-lab/pASCNN/blob/SiritoriProjects/core/cell.py#L237-L243) -> vertex_message_sum = complex_scatter_add(edge_messages=coherence.edge_message, incidence_index=self.incidence_index, num_vertices=self.config.num_vertices, backend=op_backend)

- updates vertex states

  Code: [core/cell.py:245-246](https://github.com/kaifczxc-lab/pASCNN/blob/SiritoriProjects/core/cell.py#L245-L246) -> self_update = self.self_diagonal.unsqueeze(0) * vertex_state ; vertex_state = self_update + vertex_message_sum

- then returns either Born/codebook readout or logical states to the linear head

  Code: [core/cell.py:254](https://github.com/kaifczxc-lab/pASCNN/blob/SiritoriProjects/core/cell.py#L254) -> readout = self.readout(vertex_state)

  [training/cifar10_benchmark.py:397-402](https://github.com/kaifczxc-lab/pASCNN/blob/SiritoriProjects/training/cifar10_benchmark.py#L397-L402) -> class_log_scores = torch.gather(...).squeeze(-1).sum(dim=-1)

  [training/cifar10_benchmark.py:476-484](https://github.com/kaifczxc-lab/pASCNN/blob/SiritoriProjects/training/cifar10_benchmark.py#L476-L484) -> logical_vertex_state = cell_outputs.vertex_state[:, LOGIC_VERTEX_INDICES, :] ; class_log_scores = self.classifier(state_features)

All this can be seen and examined in [core/cell.py](https://github.com/kaifczxc-lab/pASCNN/blob/SiritoriProjects/core/cell.py) & [types.py](https://github.com/kaifczxc-lab/pASCNN/blob/SiritoriProjects/types.py) & [ops/reference.py](https://github.com/kaifczxc-lab/pASCNN/blob/SiritoriProjects/ops/reference.py)

How the core is structured

If we remove all the noise, the core's mental model is as follows:

- there are 5 internal roles: L, R, -1, 0, +1

  Code: [types.py:15](https://github.com/kaifczxc-lab/pASCNN/blob/SiritoriProjects/types.py#L15) -> VERTEX_LABELS = ("L", "R", "-1", "0", "+1")

### <div align="center">Visualizing</div>

---

<img width="1280" height="720" alt="lsosmfjdhfdh" src="https://github.com/user-attachments/assets/d8b9434f-3854-42ce-8569-141507881432" />

---

This figure shows the fixed internal topology of the pASCNN core

L and R form the wave edge, -1, 0, and +1 form the logic subgraph, and the remaining red edges are cross-connections between the wave and logical parts

Code: [types.py:38-54](https://github.com/kaifczxc-lab/pASCNN/blob/SiritoriProjects/types.py#L38-L54) -> EDGE_ENDPOINTS = (...) ; EDGE_TYPES = ("logic", "logic", "logic", "wave", "cross", ...)

The readout is taken from the updated logical state and is not itself a graph vertex

Code: [types.py:29-32](https://github.com/kaifczxc-lab/pASCNN/blob/SiritoriProjects/types.py#L29-L32) -> LOGIC_VERTEX_INDICES = (VERTEX_INDEX["-1"], VERTEX_INDEX["0"], VERTEX_INDEX["+1"])

and [core/readout.py:40-49](https://github.com/kaifczxc-lab/pASCNN/blob/SiritoriProjects/core/readout.py#L40-L49) -> logic_amplitudes ... logic_probabilities = logic_energy / normalization

---

- between them are fixed typed edges: logic / wave / cross

  Code: [types.py:50-64](https://github.com/kaifczxc-lab/pASCNN/blob/SiritoriProjects/types.py#L50-L64) -> EDGE_TYPES = (...) ; CANONICAL_EDGE_TYPE_INDEX = tuple(EDGE_TYPE_INDEX[edge_type] for edge_type in EDGE_TYPES)

- each vertex has a complex state — let h = encoder embedding; D = hidden dim;

$$
s_{b,v,d}=\frac{W_{re}(h)_{b,v,d}+i\,W_{im}(h)_{b,v,d}}{\sqrt{D}}
$$

  confirmed by [core/cell.py:103-119](https://github.com/kaifczxc-lab/pASCNN/blob/SiritoriProjects/core/cell.py#L103-L119)

- each vertex has a finite-depth branch code

  Code: [core/cell.py:121-127](https://github.com/kaifczxc-lab/pASCNN/blob/SiritoriProjects/core/cell.py#L121-L127) -> branch_logits_init(h).reshape(batch_size, self.config.num_vertices, self.config.branch_depth, self.config.branch_base)

  and [core/cell.py:200](https://github.com/kaifczxc-lab/pASCNN/blob/SiritoriProjects/core/cell.py#L200) -> digit_probabilities = torch.softmax(digit_logits, dim=-1)

- branch codes are compared via soft prefix similarity — let q^(s), q^(t) = source/target digit probabilities;

$$a_{b,e,k}=\sum_p q^{(s)}_{b,e,k,p}q^{(t)}_{b,e,k,p},$$

$$c_{b,e,k}=\prod_{j\le k}a_{b,e,j},$$

$$m_{b,e}=\sum_k c_{b,e,k}$$

  confirmed by [ops/reference.py:266-268](https://github.com/kaifczxc-lab/pASCNN/blob/SiritoriProjects/ops/reference.py#L266-L268)

- complex states are compared via a typed transport defect

  Code: [ops/reference.py:317-320](https://github.com/kaifczxc-lab/pASCNN/blob/SiritoriProjects/ops/reference.py#L317-L320) -> source_transport = selected_source_diagonal * source ; target_transport = selected_target_diagonal * target ; defect = source_transport - target_transport ; defect_norm_sq = defect.abs().square().sum(dim=-1)

- the prefix and defect together control the coherence gate

  Test: [test_artifacts/cifar10_20epoch_compare/cifar10_benchmark.json](https://github.com/kaifczxc-lab/pASCNN/blob/SiritoriProjects/test_artifacts/cifar10_20epoch_compare/cifar10_benchmark.json) logs edge_prefix_depth_to_uniform_ratio_mean, edge_defect_norm_sq_normalized_mean, and edge_amplitude_mean in the same run

- the coherence gate sets the complex edge message

  Code: [ops/reference.py:396-403](https://github.com/kaifczxc-lab/pASCNN/blob/SiritoriProjects/ops/reference.py#L396-L403) -> edge_phase = ... ; edge_amplitude = ... ; edge_coefficient = torch.polar(edge_amplitude, edge_phase) ; edge_message = edge_mean * edge_coefficient.unsqueeze(-1)

- edge messages update vertex states

  Code: [core/cell.py:238-246](https://github.com/kaifczxc-lab/pASCNN/blob/SiritoriProjects/core/cell.py#L238-L246) -> vertex_message_sum = complex_scatter_add(...) ; self_update = self.self_diagonal.unsqueeze(0) * vertex_state ; vertex_state = self_update + vertex_message_sum

- the decision is then read from the logical part of this state

  Test: [test_artifacts/frozen_cell_readout_sweep/frozen_cell_readout_sweep_summary.json](https://github.com/kaifczxc-lab/pASCNN/blob/SiritoriProjects/test_artifacts/frozen_cell_readout_sweep/frozen_cell_readout_sweep_summary.json) compares a0_source_pascnn_codebook vs a3_linear_state_head, i.e. the two readout paths on the same core-state family

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
Artifact: [test_artifacts/cifar10_20epoch_compare/cifar10_benchmark.json](https://github.com/kaifczxc-lab/pASCNN/blob/SiritoriProjects/test_artifacts/cifar10_20epoch_compare/cifar10_benchmark.json)

And on the low-data sweep, linear_state is also systematically higher than codebook at all budgets, although not always higher than transformer:

- 50/class: 22.6% vs. 16.3%
- 500/class: 37.7% vs. 31.15%
- 1000/class: 45.65% vs. 40.3%
Artifact: [test_artifacts/cifar10_lowdata_3seed_selected/lowdata_3seed_summary.json](https://github.com/kaifczxc-lab/pASCNN/blob/SiritoriProjects/test_artifacts/cifar10_lowdata_3seed_selected/lowdata_3seed_summary.json)

This suggests that for natural images, linear_state is now generally safer and stronger

### What about relation OOD?
This is where things get more interesting

On one successful seed of 8, codebook was slightly higher than linear_state:

- 75.4% vs. 75.2%
But on seed 3, the average is already in favor of linear_state:
- linear_state OOD mean: 72.83%
- codebook OOD mean: 64.2%
And most importantly: codebook has a very wide spread of seeds. Artifact: [test_artifacts/left_right_relation_3seed/left_right_relation_3seed_summary.json](https://github.com/kaifczxc-lab/pASCNN/blob/SiritoriProjects/test_artifacts/left_right_relation_3seed/left_right_relation_3seed_summary.json)

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


---


## Part 3 | Audit / Validity Checks / Tests and results

This section is not intended to demonstrate the perfection of the architecture (that would be unserious), but only to strengthen its basic capabilities and indicate the correctness of the passed tests, while also discarding questions about data leakage.

The strongest audit was done on the synthetic relation OOD benchmark, because that is the place where one could most reasonably suspect accidental leakage, shortcuting, or bad split construction.

The relevant artifact is:

[test_artifacts/left_right_relation_leakage_audit_seed8/left_right_relation_leakage_audit.json](https://github.com/kaifczxc-lab/pASCNN/blob/SiritoriProjects/test_artifacts/left_right_relation_leakage_audit_seed8/left_right_relation_leakage_audit.json)

The corresponding benchmark report is:

[test_artifacts/left_right_relation_leakage_audit_seed8/benchmark/left_right_relation_benchmark.json](https://github.com/kaifczxc-lab/pASCNN/blob/SiritoriProjects/test_artifacts/left_right_relation_leakage_audit_seed8/benchmark/left_right_relation_benchmark.json)

What was checked there?

First, exact duplicate hashes across splits were checked.

The result was:
```
- train_val = 0
- train_val_ood = 0
- train_test_iid = 0
- train_test_ood = 0
- val_ood_test_ood = 0
- test_iid_test_ood = 0
```

This does not prove that the distributions are “good” in any deep sense. It proves something much simpler and more important: there are no exact repeated samples crossing the major split boundaries.

Second, the reported accuracies were recomputed in three different ways:

- from the JSON benchmark report
- from exported CSV prediction tables
- from the saved selected checkpoint itself

For all three models in that audit run, these numbers matched exactly.

The selected accuracies were:
```
- transformer_relation_classifier: iid 0.779, ood 0.745
- pascnn_relation_codebook_classifier: iid 0.764, ood 0.754
- pascnn_relation_linear_state_classifier: iid 0.771, ood 0.752
```

and the same values were recovered from **json/csv/checkpoint selected**

This matters because it removes a very stupid but very real failure mode: “pretty numbers in summary, different numbers in actual predictions”.

Third, single-side leakage was checked.

The relation benchmark is supposed to depend on the relation between left and right halves. So if one side alone already carries the answer, the task is partly broken.

The audit therefore evaluated:

- left_only_test_ood_accuracy
- right_only_test_ood_accuracy

Chance level in that benchmark is 0.25.

The results were:
```
- transformer: left-only 0.242, right-only 0.230
- codebook: left-only 0.260, right-only 0.226
- linear_state: left-only 0.240, right-only 0.221
```

This is close to chance for all three models. So there is no sign that the label is leaking in any strong way through only one half of the image.

Fourth, a broken-pair control was checked.

In that control, the left half is taken from one sample and the right half from another, while the original label is kept. This destroys the intended relation.

If accuracy remains high there, then the model is probably not solving the relation task honestly.

The results were:
```
- transformer: 0.247
- codebook: 0.255
- linear_state: 0.247
```

Again, this is approximately chance. So once the left-right relation is destroyed, the models stop working. This is the expected behavior.

That is one of the strongest checks in the repository.

There is also a more general validity point about checkpoint selection.

In the relation benchmark, the selected checkpoint is chosen by val_ood, not by test_ood. This is visible in the benchmark code and is reflected in the saved report. So the test split is not directly used to pick the final model.

That still does not make the benchmark sacred. It simply removes the most obvious form of test-time model selection.

Another validity check is repetition across seeds.

A single good run is not enough, especially on a synthetic benchmark. That is why the expanded relation OOD benchmark was rerun for three seeds:

[test_artifacts/left_right_relation_3seed/left_right_relation_3seed_summary.json](https://github.com/kaifczxc-lab/pASCNN/blob/SiritoriProjects/test_artifacts/left_right_relation_3seed/left_right_relation_3seed_summary.json)

The mean selected accuracies there were:
```
- transformer: iid 0.7600, ood 0.7187
- codebook: iid 0.6630, ood 0.6420
- linear_state: iid 0.7667, ood 0.7283
```

The important point here is not that the gap is huge. It is not huge.

The important point is that the single-run result did not completely disappear under repetition. linear_state remained approximately at parity or slightly above the matched transformer on this benchmark, while codebook showed noticeably larger instability.

A similar point applies to CIFAR low-data. That result was also not left as a single-run anecdote.

Artifact:

[test_artifacts/cifar10_lowdata_3seed_selected/lowdata_3seed_summary.json](https://github.com/kaifczxc-lab/pASCNN/blob/SiritoriProjects/test_artifacts/cifar10_lowdata_3seed_selected/lowdata_3seed_summary.json)

Mean best test accuracies there were:
```
For 50 examples per class:

- transformer: 19.55%
- linear_state: 22.60%
- codebook: 16.33%

For 500 examples per class:

- transformer: 38.28%
- linear_state: 37.70%
- codebook: 31.15%

For 1000 examples per class:

- transformer: 46.07%
- linear_state: 45.65%
- codebook: 40.30%
```

This is not an “always wins” story. It is more modest than that.

It says that under very low data, linear_state can outperform the matched ViT-like baseline, but the advantage weakens as the budget grows.

One more validity point is internal diagnostics.

On CIFAR-10, the early problem was not bad top-1 alone, but a dead core: edge_amplitude was collapsing near zero.
After the scale fixes, the benchmark showed that the core was no longer numerically dormant.

Artifact:

[test_artifacts/cifar10_20epoch_compare/cifar10_benchmark.json](https://github.com/kaifczxc-lab/pASCNN/blob/SiritoriProjects/test_artifacts/cifar10_20epoch_compare/cifar10_benchmark.json)

At 20 epochs:
```
- pASCNN + linear_state: best test accuracy 52.75%, final edge_amplitude_mean about 0.913
- pASCNN + codebook: best/final test accuracy 48.95%, final edge_amplitude_mean about 0.559
- transformer: best test accuracy 51.10%
```

This does not prove a theorem. It shows that the improvement did not come from a dead decorative core. The diagnostics moved together with the training behavior.

Finally, there is the frozen-cell readout probe.

Artifact:

[test_artifacts/frozen_cell_readout_sweep/frozen_cell_readout_sweep_summary.json](https://github.com/kaifczxc-lab/pASCNN/blob/SiritoriProjects/test_artifacts/frozen_cell_readout_sweep/frozen_cell_readout_sweep_summary.json)

Summary values:
```
- source pASCNN codebook: 26.04% ± 2.09
- linear_state head on cached state: 44.16% ± 1.99
- linear_logp head: 31.13% ± 2.36
- soft trainable codebook head: 29.40% ± 2.13
```

This is not a leakage audit in the strict sense. It is a structural sanity check.

It shows that the core state can contain useful information even when the default codebook readout is weak. That distinction matters later when discussing why codebook and linear_state behave differently.

### Additional tree-side audit should be read carefully.

The first tree experiment was an autoregressive tree-generation setup. It produced `valid_tree_rate = 0.0` for both pASCNN and the matched transformer even after longer training, so it was treated as a failed representation setup rather than as positive evidence. For that reason, tree evaluation was moved away from string generation and into sample-level structural classification.

The tree-side classification code and generators used here are:
- [training/tree_relation_benchmark.py](https://github.com/kaifczxc-lab/pASCNN/blob/SiritoriProjects/training/tree_relation_benchmark.py)
- [training/tree_balance_relation_benchmark.py](https://github.com/kaifczxc-lab/pASCNN/blob/SiritoriProjects/training/tree_balance_relation_benchmark.py)
- [training/tree_dataset.py](https://github.com/kaifczxc-lab/pASCNN/blob/SiritoriProjects/training/tree_dataset.py)
- [training/tree_synth.py](https://github.com/kaifczxc-lab/pASCNN/blob/SiritoriProjects/training/tree_synth.py)

The corresponding tests are:
- [tests/test_tree_relation_benchmark.py](https://github.com/kaifczxc-lab/pASCNN/blob/SiritoriProjects/tests/test_tree_relation_benchmark.py)
- [tests/test_tree_balance_relation_benchmark.py](https://github.com/kaifczxc-lab/pASCNN/blob/SiritoriProjects/tests/test_tree_balance_relation_benchmark.py)
- [tests/test_tree_dataset.py](https://github.com/kaifczxc-lab/pASCNN/blob/SiritoriProjects/tests/test_tree_dataset.py)
- [tests/test_tree_synth.py](https://github.com/kaifczxc-lab/pASCNN/blob/SiritoriProjects/tests/test_tree_synth.py)

Two tree classification benchmarks were then used.

The first benchmark was a tree depth-relation task:
train on depths `2, 3`, test OOD on depths `4, 5`, with ternary labels `left_deeper / equal_depth / right_deeper`.

The 3-seed aggregate summary is:
- [test_artifacts/tree_benchmark/tree_relation_3seed_summary.json](https://github.com/kaifczxc-lab/pASCNN/blob/SiritoriProjects/test_artifacts/tree_benchmark/tree_relation_3seed_summary.json)

Per-seed reports and split manifests are:
- [seed_7_basic_tree_bench/tree_relation_benchmark.json](https://github.com/kaifczxc-lab/pASCNN/blob/SiritoriProjects/test_artifacts/tree_benchmark/seed_7_basic_tree_bench/tree_relation_benchmark.json)
- [seed_7_basic_tree_bench/tree_relation_splits.json](https://github.com/kaifczxc-lab/pASCNN/blob/SiritoriProjects/test_artifacts/tree_benchmark/seed_7_basic_tree_bench/tree_relation_splits.json)
- [seed_8_basic_tree_bench/tree_relation_benchmark.json](https://github.com/kaifczxc-lab/pASCNN/blob/SiritoriProjects/test_artifacts/tree_benchmark/seed_8_basic_tree_bench/tree_relation_benchmark.json)
- [seed_8_basic_tree_bench/tree_relation_splits.json](https://github.com/kaifczxc-lab/pASCNN/blob/SiritoriProjects/test_artifacts/tree_benchmark/seed_8_basic_tree_bench/tree_relation_splits.json)
- [seed_9_basic_tree_bench/tree_relation_benchmark.json](https://github.com/kaifczxc-lab/pASCNN/blob/SiritoriProjects/test_artifacts/tree_benchmark/seed_9_basic_tree_bench/tree_relation_benchmark.json)
- [seed_9_basic_tree_bench/tree_relation_splits.json](https://github.com/kaifczxc-lab/pASCNN/blob/SiritoriProjects/test_artifacts/tree_benchmark/seed_9_basic_tree_bench/tree_relation_splits.json)

Across 3 seeds, the final mean test accuracies were:
```

- transformer: IID `57.99%`, OOD `36.81%`
- pASCNN + linear_state: IID `53.47%`, OOD `35.24%`
- pASCNN + codebook: IID `36.46%`, OOD `33.51%`

```

This benchmark therefore does not support a tree-side advantage for pASCNN. The matched transformer remained stronger here.

The second benchmark was a harder tree balance-relation task.
In that setup, both trees in each pair have the same total depth, and the label depends on comparing local root-balance states `left_heavy / balanced / right_heavy`. This removes the most obvious shortcut through global tree depth alone.

The 3-seed aggregate summary is:
- [test_artifacts/tree_benchmark/tree_balance_relation_3seed_summary.json](https://github.com/kaifczxc-lab/pASCNN/blob/SiritoriProjects/test_artifacts/tree_benchmark/tree_balance_relation_3seed_summary.json)

Per-seed reports and split manifests are:
- [seed_7_harder_tree_bench/tree_balance_relation_benchmark.json](https://github.com/kaifczxc-lab/pASCNN/blob/SiritoriProjects/test_artifacts/tree_benchmark/seed_7_harder_tree_bench/tree_balance_relation_benchmark.json)
- [seed_7_harder_tree_bench/tree_balance_relation_splits.json](https://github.com/kaifczxc-lab/pASCNN/blob/SiritoriProjects/test_artifacts/tree_benchmark/seed_7_harder_tree_bench/tree_balance_relation_splits.json)
- [seed_8_harder_tree_bench/tree_balance_relation_benchmark.json](https://github.com/kaifczxc-lab/pASCNN/blob/SiritoriProjects/test_artifacts/tree_benchmark/seed_8_harder_tree_bench/tree_balance_relation_benchmark.json)
- [seed_8_harder_tree_bench/tree_balance_relation_splits.json](https://github.com/kaifczxc-lab/pASCNN/blob/SiritoriProjects/test_artifacts/tree_benchmark/seed_8_harder_tree_bench/tree_balance_relation_splits.json)
- [seed_9_harder_tree_bench/tree_balance_relation_benchmark.json](https://github.com/kaifczxc-lab/pASCNN/blob/SiritoriProjects/test_artifacts/tree_benchmark/seed_9_harder_tree_bench/tree_balance_relation_benchmark.json)
- [seed_9_harder_tree_bench/tree_balance_relation_splits.json](https://github.com/kaifczxc-lab/pASCNN/blob/SiritoriProjects/test_artifacts/tree_benchmark/seed_9_harder_tree_bench/tree_balance_relation_splits.json)

Across 3 seeds, the final mean test accuracies were:

```
- transformer: IID `47.40%`, OOD `35.94%`
- pASCNN + linear_state: IID `46.70%`, OOD `38.37%`
- pASCNN + codebook: IID `33.33%`, OOD `32.99%`
```

What can honestly be concluded from all of this?

The repository does not prove the absence of all possible bugs.

It does remove the most obvious and dangerous failure modes:

- direct duplicate leakage across splits
- one-sided label leakage in the relation benchmark
- fake relation performance under broken left-right pairing
- reporting mismatch between summary metrics and actual saved predictions
- single-seed storytelling without any repeated runs

What remains possible?

- ordinary implementation bugs
- imperfect synthetic benchmark design
- benchmark-specific bias
- instability that only appears under broader sweeps
- claims that are still too strong if one generalizes beyond the tested tasks

That is fine. A research document does not need to pretend otherwise.

The point of this part is simply that the results here were not accepted blindly. The obvious failure modes were checked, and where checks were available, they were passed.


  ## Part 4 | Re-running the attached tests

A very quick description, if you really want to check the work, unfortunately there is no easy way yet

  The repository is presented in a document-oriented layout, not as a ready-made Python package.
  So before running the attached tests, the reader should perform one small manual packaging step.

  After cloning the repository, create a folder named `pascnn` in the repository root.

  Then move the following into `pascnn/`:

  - `adapters/`
  - `core/`
  - `models/`
  - `native/`
  - `ops/`
  - `training/`
  - `diagnostics.py`
  - `types.py`

  The folders `tests/` and `.test_artifacts/` should remain in the repository root.

  After that, the package layout matches the import paths used by the tests, and the attached test subset can be run with:

```
  python -m unittest \
    tests.test_pascnn_cell \
    tests.test_pascnn_diagnostics \
    tests.test_cifar10_benchmark \
    tests.test_left_right_relation_benchmark \
    tests.test_left_right_relation_leakage_audit
```
  These tests are the ones referred to in this document.

  What they check:

  - tests.test_pascnn_cell verifies core forward shapes, diagnostics, and gradient flow.
  - tests.test_pascnn_diagnostics verifies that the diagnostic quantities are computed consistently.
  - tests.test_cifar10_benchmark runs a small one-epoch benchmark smoke test.
  - tests.test_left_right_relation_benchmark runs a small one-epoch relation benchmark smoke test.
  - tests.test_left_right_relation_leakage_audit reruns the leakage audit on the relation setup.

  The attached tests do not require rebuilding the repository into a full installable package.
  They only require the module layout to match the pascnn.* import paths used in the code.



---

Finally, I'd like to address one detail. I don't think it's worth making a separate section explaining the specifics.
This has already been done above. For convenience, perhaps a table like this will be included, but that's beside the point.

All the main processes occur in a five-vertex core. You may already have seen the approximate topological drawing above, but what about the readout?

The final decision is not read from all five vertices equally.

In the current implementation, only the logical vertices -1, 0, and +1 participate directly in the output path. The vertices L and R remain part of the internal wave-side dynamics, but they are not themselves class vertices.

This matters because the output is not taken from a flat hidden vector in the usual way. It is taken from the logical part of the updated core state after message passing has already occurred.

At this point the architecture splits into two readout branches.

The first is codebook.

In this branch, the logical complex state is converted into ternary Born-style logical probabilities, and the class is then decoded through a fixed class codebook. In other words, the model is forced to pass through a discrete logical bottleneck before producing class scores.

The second is linear_state.

In this branch, the model does not decode through the ternary codebook. Instead, it takes the updated logical complex state itself, separates real and imaginary parts, and feeds them to a linear classifier. This is a wider and less restrictive readout path.

So the difference is not in the Core itself. The difference is in how the final information is read from it.

Why am I talking about this?

Earlier, while creating [my previous work](https://github.com/kaifczxc-lab/PyQITNN) (which I consider rather mediocre), I encountered exactly this problem: when projecting, my three states were reset to two states and one ignore state (due to the fact that my topological triangle in the core was converted into a 1D scalar in the readout. I think I don't need an explanation why this is bad, at least for me). Perhaps you'll say, "But that's normal for ternary architectures."

And in some cases, this answer is indeed correct. The whole problem was that I wanted state 0 to be as useful as state -1 and state 1, which is why I had to rework the readout method (funny note: this didn't particularly help the logical component of the model, and it seems to have even overloaded it, although this is just my guess).

Thanks for reading this document. This repository will most often only be updated with additional architecture tests, if at all. At the moment, this is just an experimental architecture, another new presentation. and so on, I don't claim to be the best, I'm just showing what I've been working on


## Resources:

The applications of sheaf theory in deep learning, data science, and computer science in general: https://arxiv.org/abs/2502.15476

Academic Sheaf Theory definition: https://stacks.math.columbia.edu/tag/00VL

Academic P-adic numbers definition: https://mathworld.wolfram.com/p-adicNumber.html or if needed more info https://en.wikipedia.org/wiki/P-adic_number

Neural Sheaf Diffusion: https://arxiv.org/abs/2202.04579
