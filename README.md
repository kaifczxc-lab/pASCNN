# pASCNN
pASCNN: the experimental research neural network architecture that uses sheaf-theoretic restrictions and a p-adic ultrametric to structure logical inference in complex space

disclaimer: pASCNN is the experimental variant of unusual neural network architecture, but I do not confirm the correctness of tests that are stored in this repository, reader can by himself check all code and make a correctness verdict

My work is not SOTA, this arch is extremely new, but it produces results consistently higher than those presented to it by its opponents, the basic ViT-like transformer

Most of the code was written using LLM. The code was reviewed, but I inevitably missed something. So, when talking about the correctness of the work done, I will refer to the correct test results and partially to the code, but it is obvious that the code cannot be without bugs and gaps

The work was completed by one person

I also don't deny that similar work may have been done, but I personally haven't seen architectures with the makings of p-adic structures and bundles, along with a ternary readout system in the form of a codebook (which can currently compete with others in only one of three tests conducted)

Here the description about technology

### Part 0 | Classification:

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





## Resources:

The applications of sheaf theory in deep learning, data science, and computer science in general: https://arxiv.org/abs/2502.15476

Academic Sheaf Theory definition: https://stacks.math.columbia.edu/tag/00VL

Academic P-adic numbers definition: https://mathworld.wolfram.com/p-adicNumber.html or if needed more info https://en.wikipedia.org/wiki/P-adic_number
