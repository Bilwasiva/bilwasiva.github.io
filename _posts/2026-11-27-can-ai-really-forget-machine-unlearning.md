---
layout: post
title: "Can AI Really Forget? Machine Unlearning, Data Lineage, and the Engineering of Verifiable Deletion"
subtitle: "What it actually takes to remove data from intelligent systems—from training influence to embeddings, deployed models, caches, backups, and evidence"
description: "A comprehensive technical guide to machine unlearning: exact and approximate removal, SISA, influence functions, LLM forgetting, RAG deletion, differential privacy, governance, evaluation, mathematical derivations, and a tested Python example."
meta-title: "Can AI Really Forget? | Machine Unlearning, Privacy & Verifiable Deletion"
meta-description: "Deleting a record is not the same as deleting its influence. Explore exact unlearning, LLM limitations, RAG deletion, data lineage, verification, and a runnable example."
date: 2026-11-27
published: true
use_math: true
image: "/assets/img/blog/machine-unlearning-cover.jpg"
thumbnail-img: "/assets/img/blog/machine-unlearning-cover.jpg"
share-img: "https://bilwasiva.github.io/assets/img/blog/machine-unlearning-cover.jpg"
ext-js:
  - "/assets/js/unlearning-math-config.js"
  - "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"
tags:
  - Artificial Intelligence
  - Machine Learning
  - Machine Unlearning
  - Privacy
  - AI Safety
  - LLMs
  - MLOps
  - AI Governance
  - Data Engineering
  - Responsible AI
---

![Conceptual editorial illustration: a neural network, a deletion request, linked data systems, and a verification shield]({{ '/assets/img/blog/machine-unlearning-cover.jpg' | relative_url }})

*Cover artwork is a conceptual illustration—not evidence that a particular AI model has been successfully unlearned or audited.*

> **The central question:** If a person, creator, or organization asks an AI system to remove information, what would it take to demonstrate that the system no longer retains the *contribution* of that information—not merely that a chatbot has been instructed to stop mentioning it?

Imagine an organization with an AI-powered customer support assistant. A user requests the removal of an old conversation containing sensitive personal information. The engineering team deletes the row from its application database. The support dashboard shows no record. The deletion ticket is marked *complete*.

But the conversation also helped train a classifier, appeared in an analytics warehouse, was embedded into a retrieval index, populated a response cache, entered an evaluation dataset, and may have influenced a fine-tuned language model. A stale container may still serve the previous model. A backup may hold a historical copy under a legitimate retention process. The chatbot might have learned an association that no longer exists as a retrievable sentence.

**Was the information deleted? Which copy? Which influence? Under what guarantee?**

This is the problem of **machine unlearning**—one of the most interesting intersections of optimization, privacy, distributed systems, model evaluation, and AI governance. It forces us to rethink a familiar idea: in conventional software, data often resides in explicitly addressable records. In machine learning, training data can influence shared parameters distributed throughout a model. Removing a record from storage does not simply undo its contribution to the learned function.

This article develops a technical and operational answer. We will derive a case where deletion *can* be exact, examine why modern neural networks are harder, compare the main algorithmic families, trace the consequences for RAG and LLM products, design a deletion workflow, and test an executable implementation against its clean-retraining reference. We will also be explicit about what is not yet solved.

**Readers:** ML scientists, infrastructure engineers, AI architects, security and privacy practitioners, researchers, technical leaders, and anyone responsible for a system that must eventually forget something.

---

## At a glance

| The question | The short answer |
|---|---|
| Does deleting a source record erase its influence on trained weights? | **No.** Not unless the model is updated under a relevant removal procedure. |
| Does deleting a vector database entry erase its source from the LLM? | It may remove **retrieval access**, but it does not necessarily change the model's parameters. |
| Can unlearning be exact? | **Yes, under precisely defined training procedures and methods**; a clean retrain is the standard counterfactual reference. |
| Is making a model refuse an answer equivalent to unlearning? | **No.** Refusal changes observable behavior but does not establish parameter-level removal. |
| Do today's LLM benchmarks prove complete forgetting? | **No.** Benchmarks provide scoped evidence, and their limitations are an active research topic. |
| Is unlearning a model-only task? | **No.** It also involves identity, authorization, lineage, storage, serving, backups, monitoring, and audit. |

### Contents

1. [The erasure paradox](#1-the-erasure-paradox)
2. [Define what it means to forget](#2-define-what-it-means-to-forget)
3. [Formal foundations and guarantees](#3-formal-foundations-and-guarantees)
4. [Why neural networks complicate deletion](#4-why-neural-networks-complicate-deletion)
5. [The algorithmic landscape](#5-the-algorithmic-landscape)
6. [A mathematical case of exact unlearning](#6-a-mathematical-case-of-exact-unlearning)
7. [Influence functions and approximate updates](#7-influence-functions-and-approximate-updates)
8. [Why LLM unlearning is different](#8-why-llm-unlearning-is-different)
9. [RAG, embeddings, and the rest of the stack](#9-rag-embeddings-and-the-rest-of-the-stack)
10. [An architecture for system-wide deletion](#10-an-architecture-for-system-wide-deletion)
11. [What counts as evidence of forgetting?](#11-what-counts-as-evidence-of-forgetting)
12. [Security, governance, and compliance](#12-security-governance-and-compliance)
13. [A runnable exact-deletion experiment](#13-a-runnable-exact-deletion-experiment)
14. [Production architecture and operations](#14-production-architecture-and-operations)
15. [Open problems and a practical roadmap](#15-open-problems-and-a-practical-roadmap)
16. [Conclusion and further reading](#16-conclusion-and-further-reading)

---

## 1. The erasure paradox

### Information does not live in just one place

Consider an AI customer-support pipeline:

1. A user submits text through a web application.
2. The message is stored in a primary database.
3. A streaming job copies it to an analytical warehouse.
4. A preprocessing job cleans and deduplicates the message.
5. An embedding model converts text chunks into numerical vectors.
6. A vector index serves those representations to a retrieval system.
7. A labeled version of the conversation contributes to model training.
8. The model is checkpointed, evaluated, packaged, and deployed.
9. Request/response traces enter logs and observability systems.
10. Some artifacts are backed up or shared with authorized downstream processors.

The user thinks of this as *one conversation*. The system sees a graph of primary objects and derivatives.

A conventional `DELETE FROM messages WHERE id = ...` changes one node of that graph. It does not necessarily invalidate the others, and it cannot undo an update already incorporated into model parameters. Even well-implemented storage deletion can be defeated operationally if a nightly ETL job reintroduces an old copy.

The fundamental principle is:

> **Erasure must follow data lineage, not just the current application schema.**

### Three distinct operations

It is useful to separate three frequently conflated ideas:

- **Data deletion:** remove or render inaccessible stored data and applicable derived copies, subject to the governing retention rules.
- **Model unlearning:** update a trained artifact to remove, or approximately remove, the effect of designated training data under a specified technical definition.
- **Behavioral suppression:** modify the model's responses through instructions, moderation, post-processing, or retrieval rules.

All three may be needed in a real application. None automatically implies the other two.

A support assistant can refuse to answer a question because a content filter blocks it while the model's internal parameters remain unchanged. Conversely, a model can be trained without a particular record and still reproduce a fact that appears independently in other lawful, retained sources. The latter is not automatically evidence that unlearning failed.

### First define the object of deletion

An erasure request can target an exact row, a group of rows, all contributions by a subject, a document collection, a fine-tuning dataset, a model checkpoint, or an entire source system. These are materially different operations.

The deletion unit should be explicit:

| Target | Example | Why it is difficult |
|---|---|---|
| Individual record | One support message | Duplicates and derivative representations |
| Person or account | All messages by one user | Identity resolution across systems |
| Document | A copyrighted PDF | Chunks, excerpts, copies, OCR, summaries |
| Concept or fact | A confidential product codename | May also be present in other retained records |
| Training contribution | A dataset shard | Shared optimization history and checkpoints |
| External disclosure | A previously delivered output | Independent recipients may retain copies |

The first error in an unlearning project is often not mathematical. It is failing to specify which object should disappear and what **continued use** must stop.

---

## 2. Define what it means to forget

An engineering claim must be testable. “The AI forgot” is not a specification.

### A hierarchy of claims

![Four different levels of evidence and guarantees for machine unlearning]({{ '/assets/img/blog/unlearning-guarantees.svg' | relative_url }})

*Figure 1. Conceptual taxonomy. These categories describe different claims, not a universal ranking applicable to every legal or operational context.*

**Level A — Output suppression.** A policy, system prompt, or filter causes a system not to disclose a specified answer. This may be valuable for risk mitigation but is not a removal guarantee.

**Level B — Empirical forgetting.** A model's behavior on a designated forget set changes in ways measured by a benchmark, test suite, or attack. The strength of the evidence depends on sampling, baseline selection, and adversarial coverage.

**Level C — Certified approximate removal.** A method gives a formal bound on how distinguishable the resulting model is from a prescribed retraining reference, under stated assumptions.

**Level D — Exact counterfactual unlearning.** The unlearning procedure yields the same model—or the same *distribution of models*, when training is random—as the specified training procedure run without the removed data.

These categories matter because a claim such as “we achieved 99% forgetting accuracy” says little without a definition of the test, the retraining baseline, the attacker, and the residual model utility.

### What about the user-facing product?

A product could require two success predicates:

$$
\operatorname{Success}_{\mathrm{system}}
= \operatorname{StorageResolved}\ \land\ \operatorname{ModelResolved}
\ \land\ \operatorname{ServingUpdated}\ \land\ \operatorname{Verified}.
$$

This is a **conceptual specification**, not an established industry formula. Each term must be defined operationally. For example, `StorageResolved` may mean erasure of eligible primary and derivative data plus documented handling of legally retained backups. It should not silently mean every copy ever created has ceased to exist.

A robust design distinguishes **completed**, **partially completed**, **not applicable**, **blocked**, and **exception-approved** states. Hiding those distinctions behind a single “deleted” flag creates misleading assurances.

---

## 3. Formal foundations and guarantees

### Training as an algorithm

Let $\mathcal{D}$ be a training dataset, and let $\mathcal{A}$ be the training algorithm:

$$
\theta = \mathcal{A}(\mathcal{D}).
$$

The algorithm may depend on random initialization, batch order, data augmentation, or other stochastic choices. Let $\mathcal{F}\subseteq\mathcal{D}$ denote the *forget set*, and define the remaining dataset as

$$
\mathcal{R}=\mathcal{D}\setminus\mathcal{F}.
$$

An unlearning algorithm $\mathcal{U}$ takes the existing model and relevant information, then produces a replacement:

$$
\theta^{-}=\mathcal{U}(\theta,\mathcal{F},\text{metadata}).
$$

### The clean-retraining counterfactual

The natural reference is a model trained using the same specified procedure on the retained data:

$$
\theta^{\star}_{\mathcal{R}}=\mathcal{A}(\mathcal{R}).
$$

In a deterministic setting, an exact remover produces the same model (allowing defined numerical tolerance where applicable). In a stochastic setting, a common formal target is equality of output distributions:

$$
\mathcal{L}\!\left(\mathcal{U}(\mathcal{A}(\mathcal{D}),\mathcal{F})\right)
=
\mathcal{L}\!\left(\mathcal{A}(\mathcal{R})\right).
$$

Here $\mathcal{L}$ denotes the probability law of the random training or unlearning outcome. This equality is **definition-dependent**: assumptions about seeds, randomness, preprocessing, and training pipelines must be specified, and not every paper uses precisely the same definition.

### Certified approximate removal

Some work formalizes a bounded distributional deviation from the retraining reference. One illustrative form is:

$$
\Pr[\theta^{-}\in S]
\leq e^{\varepsilon}\Pr[\theta^{\star}_{\mathcal{R}}\in S]+\delta
$$

for measurable sets $S$, accompanied by an appropriate reverse-direction condition when required by the selected definition. Smaller $\varepsilon$ and $\delta$ represent tighter bounds in this form, but the interpretation depends on the algorithm, assumptions, and data-removal specification. Certified removal research includes work by Guo and colleagues [2].

A proof under a restricted convex model family is not automatically a guarantee about a 70-billion-parameter autoregressive language model.

### Why the counterfactual matters

Suppose a model trained without a particular author still knows a widely reported fact about that author. That fact may have come from other retained sources, or from pretraining that was outside the unlearning scope. Exact sample-level deletion does **not** require the model to become universally ignorant about the person or topic.

The correct question is whether the target's contribution persists relative to the specified counterfactual. This is much harder to determine from casual prompting than from a formal training setup.

---

## 4. Why neural networks complicate deletion

Classical databases maintain addressable records. A neural network learns a *shared representation*.

### Optimization distributes influence

A typical training objective is

$$
\theta^*=\arg\min_{\theta}
\left[\frac{1}{N}\sum_{i=1}^{N}\ell(f_{\theta}(x_i),y_i)
+\lambda\Omega(\theta)\right].
$$

Each training example can affect gradients, and each update can affect the parameters that influence many predictions. In deep networks, effects propagate through nonlinear layers, learned features, optimizer states, and subsequent training steps.

Delete one sample from the original dataset and you have changed the **training history**. The original trajectory might never have occurred without that sample.

For stochastic gradient descent,

$$
\theta_{t+1}=\theta_t-\eta_t\,\nabla_{\theta}\ell(\theta_t;z_{i_t}).
$$

Removing $z_j$ can change the updates at its occurrences and also the later gradients, because those gradients are computed at different parameter values. With nonlinearities and optimizer state, a simple “negative gradient” update is generally not the same as rolling back the entire trajectory.

### Memorization versus influence

A training example may be memorized verbatim, contribute to a more general abstraction, influence a classification boundary, or have negligible measurable effect. These are distinct phenomena.

A model can have a sample's influence without reproducing its text. A model can produce the same answer after unlearning because another valid data source teaches the same fact.

This creates two separate requirements:

1. **Contribution removal:** Does the learned artifact resemble the specified clean-retraining reference?
2. **Information exposure:** Can an attacker or ordinary user still elicit sensitive information from the deployed system, regardless of provenance?

An effective privacy program needs to consider both.

### Checkpoints and derivatives matter

Updating the currently served weights does not retroactively update past checkpoints, distillation teachers, derivative fine-tunes, adapters, speculative draft models, or replicas. An old checkpoint can reintroduce the contribution during rollback. A nightly retraining job can do the same if its data manifest still references an excluded dataset.

That is why model unlearning belongs inside model registries, lineage graphs, deployment controls, and retention policy—not exclusively inside the training loop.

---

## 5. The algorithmic landscape

Different methods trade off retraining cost, deployment complexity, model utility, and the strength of the forgetting claim.

| Family | Core idea | Best suited to | Primary limitations |
|---|---|---|---|
| Clean retraining | Train from scratch without the forget set | Strong reference, smaller models, high-assurance cases | Expensive for very large training runs |
| Partitioned / SISA | Limit how much training must be repeated | Pipelines designed for deletion in advance | Requires architecture/data partitioning; possible utility cost |
| Sufficient-statistics update | Subtract exact per-record contributions | Some linear/convex statistical models | Does not generalize to arbitrary deep networks |
| Influence / Newton-style correction | Approximate the leave-one-out parameter effect | Locally well-behaved differentiable models | Hessian approximation and non-convexity |
| Gradient ascent / negative training | Push away from selected examples | Some exploratory deep-model/LLM workflows | Can damage utility or produce superficial forgetting |
| Distillation / teacher–student | Transfer retained knowledge to a new model | Selective model replacement approaches | Teacher leakage and incomplete forgetting |
| Adapter or component replacement | Replace or disable localized components | Modular systems designed with isolation | Non-target components may still encode influence |
| Behavioral filtering | Refuse or redirect targeted outputs | Immediate mitigation and policy enforcement | Does not establish parameter-level removal |

### 5.1 Full retraining: the dependable reference

If the objective is reproducibly specified, rerunning training on $\mathcal{R}$ is the simplest conceptual counterfactual. It is often the reference against which faster methods are judged.

But “retraining” is only well specified when preprocessing, tokenizer, deduplication, sampling, hyperparameters, initialization policy, optimizer versions, and external data sources are appropriately defined. If a user’s data also affected feature normalization, a vocabulary, or a synthetic dataset, merely omitting their final training row may not remove every contribution.

### 5.2 SISA: design for deletion before training

SISA stands for **Sharded, Isolated, Sliced, and Aggregated** training. Bourtoule and colleagues introduced it as a framework for reducing unlearning costs by structuring the training pipeline [1].

At a high level:

1. Partition records into isolated shards.
2. Train component models using specified slices and checkpoints.
3. Associate each record with its assigned shard/slice.
4. When a record must be removed, retrain only the affected portion from an appropriate earlier checkpoint.
5. Reassemble predictions using the designated aggregation procedure.

The exact guarantees depend on how isolation, replay, randomness, and aggregation are configured. SISA can reduce the work affected by a deletion, but it may impose accuracy, storage, or engineering trade-offs.

### 5.3 Updating sufficient statistics

Some models are special: the entire solution can be expressed using additive dataset statistics. That lets us subtract a record's contribution and solve again without replaying every example. The ridge-regression example in Section 6 demonstrates this precisely.

### 5.4 Approximate parameter editing

An approximation may seek to update weights directly using gradients, second-order information, or distillation. The main challenge is that plausible behavioral forgetting is easier to obtain than a rigorous guarantee of influence removal.

A successful method must also preserve the legitimate capabilities learned from the retained data. Catastrophic degradation is not an acceptable way to “forget.”

---

## 6. A mathematical case of exact unlearning

Not all unlearning is speculative. For some models we can derive an exact reversible contribution update.

### Ridge regression with fixed features

Consider a design matrix $X\in\mathbb{R}^{N\times d}$, target vector $y\in\mathbb{R}^N$, and a fixed regularization strength $\lambda>0$:

$$
J(w)=\|Xw-y\|_2^2+\lambda\|w\|_2^2.
$$

The gradient is

$$
\nabla_w J=2X^T(Xw-y)+2\lambda w.
$$

Setting the gradient to zero gives

$$
(X^TX+\lambda I)w=X^Ty,
$$

and therefore

$$
\boxed{w^*=(X^TX+\lambda I)^{-1}X^Ty.}
$$

Define two sufficient statistics:

$$
A=\sum_{i=1}^{N}x_ix_i^T,\qquad
b=\sum_{i=1}^{N}x_iy_i.
$$

Then

$$
w^*=(A+\lambda I)^{-1}b.
$$

If a record $j$ must be removed, update those statistics:

$$
A^{-}=A-x_jx_j^T,\qquad
b^{-}=b-x_jy_j.
$$

Now solve:

$$
\boxed{w^{-}=(A^{-}+\lambda I)^{-1}b^{-}.}
$$

**This equals the solution of training the same ridge-regression objective on the retained records**, up to floating-point effects. We did not need to re-read every retained row in order to compute the new optimum.

The key is not that we magically erased a neural network's memory. The key is that the model objective has a known, additive mathematical structure.

### Important assumptions

The example assumes a **fixed feature transformation**, a **fixed penalty**, and a **deterministic optimizer/solver**. It does not dynamically fit a scaler, vocabulary, imputer, or target transformation using the full dataset. If any such derived preprocessing is trained on the removed row, its state must also be updated. The row's contribution is still stored somewhere to permit subtraction, and that storage requires its own governance.

Also, equality of the final ridge solution does not prove that every copy of the deleted record was removed from storage, logs, or backups. It establishes one specific model-level equivalence.

![Illustrative exact deletion for ridge regression: fit with a target record versus fit after deletion and clean retraining]({{ '/assets/img/blog/unlearning-ridge-example.svg' | relative_url }})

*Figure 2. Original illustrative experiment. The orange fit includes a target outlier. The teal fit after algebraically removing that row coincides numerically with a clean retrain. This is a deliberately small deterministic model, not an LLM result.*

### Complexity intuition

For $d$ features, accumulating sufficient statistics costs roughly $O(Nd^2)$ in the generic dense case; solving the resulting $d\times d$ system typically costs $O(d^3)$ with standard dense methods. Removing one row's additive contribution costs $O(d^2)$, followed by the solve or an appropriate rank-one update.

This is attractive when $d$ is manageable and repeated removal requests are anticipated. It does not apply unchanged to a transformer with billions of nonlinear parameters.

---

## 7. Influence functions and approximate updates

Influence functions provide a powerful way to study how model parameters or predictions respond to small changes in the training distribution. Koh and Liang demonstrated their usefulness for tracing model predictions back to influential examples [3].

### Local approximation

Let the empirical objective be

$$
L(\theta)=\frac{1}{N}\sum_{i=1}^N \ell(z_i,\theta),
$$

with optimum $\hat{\theta}$ and Hessian $H=\nabla_{\theta}^2L(\hat{\theta})$. Around a sufficiently regular local optimum, a first-order approximation to removing example $j$ has the form

$$
\Delta\theta_{-j}
\approx \frac{1}{N}H^{-1}\nabla_{\theta}\ell(z_j,\hat{\theta}),
$$

where factors can differ slightly depending on the leave-one-out objective normalization. The sign is positive here because removing an example reverses its first-order contribution to the optimum.

### When the approximation can fail

The expression relies on a locally meaningful inverse Hessian, a sufficiently small perturbation, and stable optimization behavior. Large neural networks complicate every assumption:

- The Hessian is enormous and may be ill-conditioned or singular.
- Training is non-convex; a local linearization may be inadequate.
- A forget set might represent a large or highly correlated data region.
- Adaptive optimizers and long training trajectories create dependencies not represented by a single final Hessian.
- Repeated approximate removals can accumulate error.

Efficient curvature approximations can be useful, but **a computed correction is not by itself a certified deletion guarantee**. Compare it against a meaningful reference and evaluate its stability over multiple requests.

### Why gradient ascent is not a substitute for proof

One heuristic is to maximize the loss on the forget set:

$$
\theta^{-}=\theta+\eta\nabla_{\theta}L_{\mathcal{F}}(\theta).
$$

This can reduce the likelihood of targeted outputs, but it may harm adjacent knowledge, cause instability, and encourage a model to output different errors rather than remove the original influence. The choice of learning rate, retained-data updates, evaluation, and stopping conditions matters substantially.

The difference between *not producing an answer* and *having been trained as though the data were never present* remains central.

---

## 8. Why LLM unlearning is different

A modern language model is not a table of facts indexed by name. It models dependencies among tokens after training on large, overlapping, and frequently redundant corpora. Its internal representations often blend information from many sources.

### Example: a fictional author

Suppose a small language model is fine-tuned on a collection of fictional author biographies. A subset of those biographies is marked for removal.

A naive test might ask the model a question from the forget set and measure whether it still returns the remembered answer. But several outcomes are ambiguous:

- It may refuse to answer because it has learned a refusal template.
- It may replace a memorized detail with another plausible fabrication.
- It may still reveal the data under a paraphrased prompt.
- It may score poorly on the exact question while exposing the detail through a different completion.
- It may be less capable overall, giving the appearance of successful forgetting.

The [TOFU benchmark](https://locuslab.github.io/tofu/) uses fictitious authors to make targeted evaluation more tractable [5]. The [MUSE benchmark](https://proceedings.iclr.cc/paper_files/paper/2025/hash/4556f5398bd2c61bd7500e306b4e560a-Abstract-Conference.html) expands the evaluation to multiple properties including verbatim and knowledge memorization, privacy leakage, retained utility, scalability, and repeated requests [6]. These are important contributions, not complete guarantees.

### Benchmarks should be treated as evidence, not verdicts

Research has cautioned that common LLM unlearning benchmarks can be misleading and that apparent gains may not generalize beyond the test distribution [7]. The lesson is not to abandon benchmarks; it is to construct an evaluation that matches the actual removal objective and adversarial threat model.

A stronger LLM protocol should use:

1. A precisely defined training/fine-tuning scope.
2. A clean retrain reference when feasible.
3. A forget set and a truly independent retained set.
4. Direct and paraphrased queries, plus multiple attack styles.
5. Memorization, extraction, and membership-inference assessments.
6. General-task capability checks.
7. Tests for sequential removal and subsequent fine-tuning.
8. A clear statement of what has **not** been established.

### Three questions that must remain separate

**Did the exact text disappear?** This is a *verbatim extraction* question.

**Did the model's behavior move toward the clean-retraining reference?** This is a *counterfactual influence* question.

**Can an attacker still infer sensitive information?** This is a *privacy exposure* question.

No single benchmark captures all three in every setting.

### Why refusal may make evaluation worse

A refusal-trained model can score as though it has forgotten a target question even when carefully framed alternate prompts reveal much of the original information. Refusal can be appropriate as a separate protection layer, but it should not contaminate the interpretation of model-unlearning results.

---

## 9. RAG, embeddings, and the rest of the stack

Many AI products use retrieval-augmented generation (RAG), and their deletion story differs from parameter unlearning.

### 9.1 RAG deletion is primarily a lineage and serving problem

In a standard RAG pipeline, the external knowledge store may consist of raw files, parsed text, chunks, embeddings, indexes, caches, and document metadata. If a source document is removed, the system should determine which of those artifacts derive from it.

A reasonable process might include:

1. Identify the authoritative source object and all content hashes/IDs associated with it.
2. Stop new ingestion of the source and put a tombstone or exclusion rule in place.
3. Remove eligible original and derivative storage objects.
4. Delete or invalidate vector-index entries and keyword-search records.
5. Invalidate relevant retrieval, prompt, and answer caches.
6. Prevent background jobs from rebuilding deleted chunks from stale source copies.
7. Confirm search no longer returns the deleted source through supported query paths.
8. Record the relevant exclusions, exceptions, and verification evidence.

This removes **retrievability** from the external system when executed correctly. It is not evidence that a separate pretrained LLM has erased a fact from its weights.

### 9.2 Embeddings are derivative artifacts

An embedding is a transformation of source content. It may not be human-readable, but that does not automatically make it harmless or outside applicable governance. Depending on the system and threat model, embeddings can encode attributes that warrant protection.

A source-to-chunk-to-vector mapping should therefore be explicit:

```text
source_document_id
  -> parser_version + document_version
  -> chunk_id(s)
  -> embedding_model_version + embedding_id(s)
  -> vector_index_id(s) + replica(s)
  -> retrieval_cache_key(s)
```

If you cannot trace a source ID to its derived vectors, it will be difficult to demonstrate that removal has propagated to every relevant index.

### 9.3 What about fine-tuning?

If retrieved documents were also used in supervised fine-tuning, preference optimization, distillation, or model adaptation, removing them from a vector database is insufficient. The model lineage now includes training artifacts. You may need to retrain, unlearn, replace an adapter, or retire a model version depending on the applicable scope and assurance requirement.

### 9.4 Other model families

**Recommendation systems** may distribute user interactions across item embeddings, user embeddings, co-occurrence matrices, and ranking models. Deleting a profile row does not automatically update every learned representation.

**Computer vision** systems may have learned from images, crops, synthetic augmentations, or embeddings. Face removal and biometric-related questions also raise identity and security concerns.

**Graph neural networks** present additional dependencies because removing a node or edge may affect neighbors and message passing.

**Multimodal foundation models** combine images, text, audio, and video. One source's influence may enter multiple modality encoders, shared projections, or distilled products.

The engineering principle is consistent: identify **all training and serving derivatives**, not merely the visible artifact.

---

## 10. An architecture for system-wide deletion

The strongest operational design treats a request as a tracked workflow with idempotent steps, lineage expansion, policy gates, validation, and deployment controls.

![A six-stage system-wide deletion workflow with request validation, lineage, removal, model update, serving replacement, and verification]({{ '/assets/img/blog/unlearning-deletion-lifecycle.svg' | relative_url }})

*Figure 3. A conceptual enterprise architecture. The workflow is illustrative and must be adapted to the organization's actual data architecture, permissions, and legal obligations.*

### 10.1 A practical state machine

```text
REQUESTED
  -> IDENTITY_AND_AUTHORIZATION_VERIFIED
  -> SCOPE_AND_APPLICABLE_POLICY_RESOLVED
  -> INGESTION_BLOCKED
  -> ARTIFACT_LINEAGE_COMPUTED
  -> ELIGIBLE_COPIES_RETIRED
  -> MODEL_UPDATE_REQUIRED?  -- no --> SERVING_STATE_VERIFIED
             |
            yes
             v
      MODEL_UPDATED_AND_TESTED
             |
             v
      CANDIDATE_DEPLOYED
             |
             v
      RETRIEVAL_AND_SERVING_VERIFIED
             |
             v
      AUDIT_COMPLETED

Any stage -> BLOCKED / EXCEPTION_REVIEW / RETRY_PENDING
```

Stages need **durable state** and well-defined retry semantics. In distributed systems, the failure may occur after a storage mutation succeeds but before the orchestrator records the acknowledgment. A blind retry may be harmless for an idempotent deletion by stable ID, but harmful for an operation that acts on mutable index positions or reuses credentials improperly.

### 10.2 Example request manifest

The following is a **conceptual schema**, not a universal regulatory format:

```yaml
request_id: erase-2026-001827
subject_handle: pseudonymous-subject-417
scope:
  source_systems: [support-messages, analytics, retrieval]
  record_ids: [msg-1482, msg-1510]
  model_lineages: [support-classifier-v12]
policy:
  eligibility_review: approved
  retention_exceptions: []
execution:
  ingest_tombstone: installed
  storage: completed
  vector_index: completed
  model_update: pending
  serving_cutover: not_started
verification:
  artifact_inventory: pending
  retrain_reference: pending
  reviewer_signoff: pending
```

Do not put the actual sensitive content or a raw identifying credential into the audit log. Separate the minimum necessary evidence from restricted data, and secure any linkage material.

### 10.3 Deploy the new model, not just new training files

A model that has been successfully updated in an experiment but is not serving real traffic has not resolved the production issue. The rollout must consider:

- Model registry status and immutable version identifiers.
- Replica inventories and stale containers.
- Traffic cutover and safe rollback strategy.
- Downstream adapters, distilled models, and student models.
- Cache invalidation and response traces.
- Backup lifecycle and possible legally required retention.
- Reintroduction prevention during retraining.
- Explicit exceptions where an action cannot be performed or is not required.

The rollback policy is particularly important: restoring an older model can reinstate information that was specifically requested to be removed. A registry should make such restorations detectably invalid unless they meet the applicable deletion scope.

### 10.4 Evidence should accompany state changes

For each artifact class, store a suitable event record: identity of the artifact, method, timestamp, version before/after, validation result, and any accepted exception. The audit record is evidence of the **workflow performed**, not necessarily a mathematical proof that a deep model contains zero residual influence.

---

## 11. What counts as evidence of forgetting?

Evaluation should answer the exact claim and should not collapse model utility, privacy exposure, and removal cost into one opaque number.

### 11.1 Compare to a meaningful reference

Where feasible, define a clean-retrained model $\theta_{\mathcal R}^*$ and compare the candidate $\theta^-$. Depending on the task, relevant distances may include parameter distance, prediction-distribution difference, application loss, and privacy attack performance.

A simple predictive discrepancy is:

$$
\Delta_{\mathrm{pred}}=
\mathbb{E}_{x\sim P_{\mathrm{eval}}}
\left[d\big(p_{\theta^-}(\cdot\mid x),
                p_{\theta^*_{\mathcal R}}(\cdot\mid x)\big)\right],
$$

where $d$ is a chosen divergence or distance. The expectation must specify the evaluation distribution. A small distance on one test set is not a proof of exact unlearning.

### 11.2 Keep a multi-axis scorecard

| Axis | Example measures | Main caveat |
|---|---|---|
| Retained utility | Accuracy, calibration, task completion, domain quality | Can hide worst-case subgroup failure |
| Target memorization | Verbatim extraction and targeted generation | Low extraction does not prove zero influence |
| Privacy exposure | Membership inference or reconstruction assessments | Threat-model- and attack-dependent |
| Reference similarity | Predictive or distributional distance to clean retrain | Reference may be costly or underspecified |
| System removal | Absent index IDs, no stale serving, blocked reingestion | Hard to observe unmanaged downstream copies |
| Operational cost | Time to complete, GPU-hours, deployment effort | Depends on service and compliance requirements |
| Sequential stability | Performance after repeated removal cycles | Approximation errors can accumulate |

### 11.3 Membership inference is not an oracle

Membership inference seeks to infer whether a candidate record was present in a model's training set [4]. It is useful for understanding some privacy risks, but an attack's failure is not proof that influence vanished. A weak attack, mismatched distribution, or limited query budget may not find an existing signal.

### 11.4 The evaluation threat model

Define the adversary's capability before interpreting results. Can the attacker see logits, only generated text, internal embeddings, or the model weights? Can they query repeatedly? Do they know the target record or just an attribute? Do they control prompts? Can they access old endpoints or checkpoints?

For a deployed assistant, testing should also cover prompt paraphrases, extraction attempts, retrieval bypasses, alternate endpoints, and cache states.

### 11.5 The distinction between a proof and an attestation

An audit report can attest that a documented process ran, that artifacts were removed, and that tests passed. A formal certification can establish a mathematical property under a model. Both can be useful, but one should not be described as the other.

**If the result is approximate, say approximate. If a guarantee is scoped to one artifact, do not describe it as a universal system guarantee.**

---

## 12. Security, governance, and compliance

Machine unlearning is connected to data-protection law, but a mathematical unlearning paper and a legal right to erasure are not synonymous.

### 12.1 Legal obligations are conditional and jurisdiction-specific

The EU General Data Protection Regulation includes a right to erasure in Article 17, subject to specified conditions and exceptions. Article 19 addresses notification to certain recipients of rectification, erasure, or restriction, with exceptions [8]. Similar or different rights and obligations can apply in other jurisdictions.

**An important limitation:** A law does not necessarily prescribe “run gradient ascent” or any other machine-unlearning algorithm. A compliance assessment must consider the actual legal obligation, personal-data status, processing purpose, recipients, retention duties, technical feasibility, and evidence. The regulatory and case-law interpretation of model-parameter deletion is context-dependent. The discussion here is technical background, **not legal advice**.

The European Commission's guidance on responding to individual requests highlights these practical duties and exceptions [9].

### 12.2 Security can defeat an otherwise correct deletion workflow

Common failure modes include:

- An attacker submits unauthorized deletion requests for another person's records.
- Old deleted content is replayed through a stale ingestion job.
- Logs, error traces, or caches preserve restricted text.
- A compromised operator restores an outdated checkpoint.
- An observability pipeline contains personal data outside the primary inventory.
- A vendor retains an export that the organization forgot to track.
- The deletion receipt itself exposes identifying information.

Controls should include identity verification, authorization, least privilege, tamper-evident logging where appropriate, retention governance, change control, and independent review of high-impact operations.

### 12.3 Relationship to differential privacy

Differential privacy (DP) bounds how much a randomized mechanism's output distribution can change when the input dataset changes by a defined neighboring relation. Informally, for neighboring datasets $D$ and $D'$:

$$
\Pr[\mathcal{M}(D)\in S]
\leq e^{\varepsilon}\Pr[\mathcal{M}(D')\in S]+\delta.
$$

DP is a **training/privacy stability property**, not a workflow that locates and deletes every stored record. Differentially private training can limit the information attributable to a single example and inform some certified-removal approaches, but it does not itself execute a request or erase backups and retrieval indexes. See Abadi and colleagues on private deep learning [10].

### 12.4 Data retention does not disappear from the discussion

Some artifacts may need to be retained for reasons recognized by applicable law, such as legal obligations or claims. Where deletion does not apply, the system should accurately document the exception, restrict inappropriate uses, and prevent those artifacts from silently feeding future AI training where the relevant processing is no longer authorized.

Retention handling and model unlearning should be coordinated, not conflated.

---

## 13. A runnable exact-deletion experiment

To connect the theory with working code, the accompanying repository includes a **standard-library-only Python implementation** of exact sample removal for a two-parameter ridge-regression model.

The example maintains the sufficient statistics:

$$
A=\sum_i\begin{bmatrix}1\\x_i\end{bmatrix}
\begin{bmatrix}1 & x_i\end{bmatrix},
\qquad
b=\sum_i\begin{bmatrix}1\\x_i\end{bmatrix}y_i.
$$

Each row contributes one outer product and one target-weighted feature vector. On an authorized toy deletion request, the code subtracts the row's contribution, records an in-memory tombstone, and recomputes the exact ridge solution.

### What the experiment checks

- A trained model produces one set of coefficients with the target record included.
- Removing that contribution produces updated coefficients.
- Fitting an independent model from scratch on the retained records produces matching coefficients to numerical tolerance.
- A repeated request does not subtract the same contribution twice.
- A deletion request ID cannot be reused to target a different record.
- A tombstoned record is not reintroduced within the same running process.

You can inspect the complete implementation and tests:

- [Exact ridge sample-removal implementation](https://github.com/Bilwasiva/bilwasiva.github.io/blob/master/examples/unlearning/exact_ridge_deletion.py)
- [Six unit tests](https://github.com/Bilwasiva/bilwasiva.github.io/blob/master/examples/unlearning/test_exact_ridge_deletion.py)
- [Execution instructions and limitations](https://github.com/Bilwasiva/bilwasiva.github.io/blob/master/examples/unlearning/README.md)

### Run it locally

```bash
python examples/unlearning/exact_ridge_deletion.py
python -m unittest discover -s examples/unlearning -v
```

The illustrative run yields:

```text
deletion applied: True
before deletion: intercept=2.121002; slope=2.078548
after deletion:  intercept=0.921411; slope=2.011531
clean retrain:    intercept=0.921411; slope=2.011531
max coefficient discrepancy: 1.78e-15
A correct sample-removal update matched clean retraining numerically.
```

The tiny floating-point difference is numerical roundoff. The *mathematical* solutions are identical under the fixed objective and feature map. The experiment has no access to private data and makes no claim about removing learned knowledge from an LLM.

### The production caveat that matters most

The example stores per-record contributions to enable subtraction. In an operational product, those retained contributions may themselves be derived data and must be managed under the same authorization, retention, and deletion design. The tombstones and request journal in the demonstration are **in memory**, not transactional or durable. They teach idempotency, not enterprise-grade compliance.

This small example makes a larger point: **strong guarantees become possible when we design the learning process with deletion in mind.**

---

## 14. Production architecture and operations

A practical system should make erasure a first-class software capability rather than an emergency manual procedure.

### 14.1 Build a provenance graph

A useful abstraction is a directed graph $G=(V,E)$ in which nodes represent artifacts (records, chunks, embeddings, model snapshots, adapters, outputs) and edges represent derivation or dependency relationships.

For a target set $T$, the affected set is conceptually the reachable closure:

$$
\operatorname{Affected}(T)
=\{v\in V:\exists\ \text{a derivation path from some }t\in T\text{ to }v\}.
$$

This is an engineering **over-approximation** in many real deployments, because a dependency edge does not prove material influence. Nevertheless, it is useful for identifying artifacts that must be examined.

Record at least source ID, transform code version, dataset manifest, model training run, deployed model version, embedding/index IDs, and relevant retention category. A lineage graph that stops at the data lake while ignoring served model versions is incomplete.

### 14.2 Policy gates before deletion

Do not start destructively processing a request before confirming identity, authorization, scope, applicable exceptions, and dependencies. Provide careful handling for overlapping requests and ambiguous subject matches.

### 14.3 Idempotency and retries

Treat a deletion request as a versioned operation with a stable request identifier. Replaying the same request should not double-subtract statistics or corrupt indexes. After partial failure, resume from recorded checkpoints rather than redoing actions blindly.

### 14.4 Batch and serving consistency

An index update may be complete while a running server still holds stale data. Use versioned manifests, deploy gates, and post-cutover tests. Where possible, direct new traffic to verified replacement artifacts and retire older versions in a controlled manner.

### 14.5 Verifiable cutover

Before closing the request, verify the relevant production endpoints, not just an offline model file. Queries should target the same service versions and routing conditions customers use. Sample-only testing should be described as such and accompanied by observability of incomplete replicas or caches.

### 14.6 Costs and service-level objectives

Deletion latency matters. A request may be simple at the primary database but expensive if it triggers a full model retrain. A useful operational cost model is:

$$
C_{\mathrm{erase}}=C_{\mathrm{discovery}}+C_{\mathrm{storage}}
+C_{\mathrm{model}}+C_{\mathrm{deployment}}+C_{\mathrm{verification}}.
$$

This is an illustrative decomposition, not a universal accounting standard. Model retraining is often the largest variable term, but human review, validation, and lineage repair can also dominate.

A complete service-level objective should specify **which completion state** and **which artifacts** its clock covers. “Deleted within 24 hours” is ambiguous without a scope and exception model.

### 14.7 A reference test plan

| Layer | Test | Expected outcome |
|---|---|---|
| Intake | Unauthorized actor submits request | Rejected without exposing records |
| Lineage | Target appears in two datasets | Both derived lineages are discovered |
| Storage | Deletion is retried after lost acknowledgment | Idempotent completion, no corruption |
| Retrieval | Deleted source is queried | No eligible stale source returned |
| Training | Next retrain runs | Excluded records are not re-ingested |
| Model | Selected unlearning method is evaluated | Claims supported relative to reference |
| Serving | Old and new versions coexist | Traffic cutover and retirement verified |
| Backup | Restored snapshot contains old artifact | Restore procedure reapplies exclusions |
| Audit | Reviewer inspects completion | Evidence, limits, and exceptions visible |

### 14.8 A worked enterprise scenario

Suppose an enterprise assistant is fine-tuned on internal support tickets and retrieves historical knowledge-base pages. A legitimate request targets an individual's sensitive information in a group of tickets.

The team resolves the individual to stable pseudonymous IDs, identifies the eligible tickets, blocks re-ingestion, and maps them to parsed chunks, embeddings, response caches, and model training manifests. The source tickets and approved derivatives are removed. The team determines that the deployed fine-tune used the target records and selects either clean retraining or a suitably justified removal procedure, based on its risk and assurance requirements. It evaluates target exposure, retained performance, and reference similarity, then routes traffic to a verified model version.

Only after the serving layer and audit evidence are reviewed should the request reach its completed state. If a legally retained backup remains, the completion record explains the retention and the controls that prevent inappropriate reprocessing.

This scenario illustrates coordination. It is not a claim that any one algorithm can guarantee deletion across every enterprise component.

---

## 15. Open problems and a practical roadmap

### Research questions still worth solving

**1. Better counterfactual evaluation.** How can we compare unlearned foundation models against genuine retraining behavior without requiring prohibitively expensive full retraining for every request?

**2. Robustness to alternate prompts and attacks.** A model that appears to forget under one test may still leak information under others. Evaluation needs to be adversarial, multi-view, and matched to real access capabilities.

**3. Sequences of requests.** Errors can accumulate as models undergo many approximate removals. Sustainable evaluation must track long-running deployments and later fine-tunes.

**4. Shared and redundant data.** Source overlap means a fact or representation may remain legitimately learnable after one particular source is excluded.

**5. Model composition.** Adapters, teachers, students, quantized copies, retrieval indexes, and serving caches complicate model lineage.

**6. Systems and formal methods.** Stronger tools are needed to connect mathematical removal guarantees to concrete distributed-system actions and evidence.

**7. Privacy–utility–efficiency trade-offs.** Faster removal may carry weaker guarantees or larger utility losses. Choosing the method depends on risk, requirements, and reproducibility.

### A responsible roadmap for organizations

**Stage 1 — Inventory.** List all sources, derivatives, model training runs, and public/third-party dependencies. Track stable IDs and transformations.

**Stage 2 — Prevent reintroduction.** Build exclusion manifests or tombstones, validate retraining inputs, and define retention handling.

**Stage 3 — Define assurance tiers.** Separate storage erasure, retrieval erasure, behavioral safeguards, empirical model unlearning, and formally supported removal.

**Stage 4 — Establish baselines.** For tractable models, use clean retraining. For larger systems, explicitly document approximate references and their limitations.

**Stage 5 — Implement durable orchestration.** Handle identity, authorization, retries, idempotency, deployment, model versioning, and audit evidence.

**Stage 6 — Test adversarially.** Measure retained utility, target exposure, reference similarity, prompt robustness, and stale-serving risk.

**Stage 7 — Review continuously.** Revisit the deletion method as models, training pipelines, retention rules, and available evidence evolve.

### Common misconceptions

| Misconception | Better interpretation |
|---|---|
| “We deleted the row, so the AI forgot.” | Deletion from storage does not undo model training. |
| “The chatbot refuses to answer, so the weights are clean.” | Behavioral suppression is not weight-level evidence. |
| “The answer still appears, so unlearning failed.” | Another retained source may independently support the answer. |
| “No successful membership attack means privacy is guaranteed.” | Attack power and coverage are limited. |
| “A certified removal paper proves all LLMs can forget.” | Guarantees have model, data, and algorithm assumptions. |
| “A vector database is just an index; it can be ignored.” | Embeddings and caches may be relevant derivative artifacts. |
| “The newer model is deployed; deletion is complete.” | Retired versions, replicas, restores, and downstream systems matter. |

### Glossary

- **Forget set:** designated training examples or data contributions to remove.
- **Retain set:** training data remaining after the forget set is excluded.
- **Clean retrain:** running the specified training algorithm on the retain set.
- **Exact unlearning:** matching the defined clean-retrain result or distribution.
- **Approximate unlearning:** pursuing a bounded or empirical approximation to that result.
- **Certified removal:** a mathematically supported removal claim under explicit assumptions.
- **Membership inference:** attempting to infer whether an example participated in training.
- **SISA:** sharded, isolated, sliced, and aggregated training.
- **Sufficient statistics:** aggregate quantities that determine a model's fit under a defined objective.
- **Data lineage:** mappings among source records and derived artifacts.
- **Tombstone:** a marker preventing an artifact from being treated as an active eligible source.
- **Serving cutover:** moving production requests to a new verified artifact version.
- **Deletion attestation:** a record of performed actions and their observed evidence, not automatically a mathematical proof.

---

## 16. Conclusion and further reading

The most important lesson of machine unlearning is not that artificial intelligence is incapable of forgetting. For certain model families, removal can be exact and elegantly expressible through algebra. For others, carefully engineered retraining or approximation can be useful, but the guarantee is narrower and the evaluation harder.

The larger lesson is that **forgetting is a property of an entire lifecycle**.

A data subject sees one conversation. An AI platform sees sources, ETL transformations, training examples, parameters, embeddings, replicas, adapters, caches, backups, and downstream outputs. Responsible deletion requires these systems to coordinate—and requires engineers to be precise about the difference between **erasure, suppression, inference-time access, model influence, and evidence**.

The interesting technical frontier is therefore twofold: design learning procedures that support meaningful, measurable removal, and build software architectures that can carry a deletion request all the way from authorization to serving verification without losing track of what was affected.

**A system that can learn from data should be engineered to account for what happens when it must stop using that data.**

### A question for the AI engineering community

If you had to design unlearning for a production foundation-model product today, where would you invest first: **train-time modularity and provenance**, **better removal algorithms**, **adversarial verification**, or **end-to-end deletion infrastructure**? Which failure mode worries you most?

### Research and primary references

1. **Bourtoule, L., et al. (2021).** *Machine Unlearning*. IEEE Symposium on Security and Privacy. Introduces SISA training. https://ieeexplore.ieee.org/abstract/document/9519428/
2. **Guo, C., Goldstein, T., Hannun, A., & van der Maaten, L. (2020).** *Certified Data Removal from Machine Learning Models*. ICML / PMLR 119. https://proceedings.mlr.press/v119/guo20c.html
3. **Koh, P. W., & Liang, P. (2017).** *Understanding Black-box Predictions via Influence Functions*. ICML / PMLR 70. https://proceedings.mlr.press/v70/koh17a.html
4. **Shokri, R., Stronati, M., Song, C., & Shmatikov, V. (2017).** *Membership Inference Attacks Against Machine Learning Models*. IEEE Symposium on Security and Privacy. https://arxiv.org/abs/1610.05820
5. **Maini, P., et al. (2024).** *TOFU: A Task of Fictitious Unlearning for LLMs*. ICLR 2024 Workshop. https://mlanthology.org/iclrw/2024/maini2024iclrw-tofu/
6. **Shi, W., et al. (2025).** *MUSE: Machine Unlearning Six-Way Evaluation for Language Models*. ICLR 2025. https://proceedings.iclr.cc/paper_files/paper/2025/hash/4556f5398bd2c61bd7500e306b4e560a-Abstract-Conference.html
7. **CMU ML Blog (2025).** *LLM Unlearning Benchmarks are Weak Measures of Progress*. Explains benchmark reliability pitfalls. https://blog.ml.cmu.edu/2025/04/18/llm-unlearning-benchmarks-are-weak-measures-of-progress/
8. **European Union.** *General Data Protection Regulation, Article 17 and Article 19*. Official consolidated legislation. https://eur-lex.europa.eu/eli/reg/2016/679/oj/eng/
9. **European Commission.** *Dealing with requests from individuals*. Practical guidance on erasure requests. https://commission.europa.eu/law/law-topic/data-protection/information-business-and-organisations/dealing-requests-individuals_en
10. **Abadi, M., et al. (2016).** *Deep Learning with Differential Privacy*. ACM CCS 2016. https://arxiv.org/abs/1607.00133
11. **Google Research (2023).** *Announcing the First Machine Unlearning Challenge*. https://research.google/blog/announcing-the-first-machine-unlearning-challenge/
12. **NIST (2024).** *AI Risk Management Framework: Generative Artificial Intelligence Profile (AI 600-1)*. https://www.nist.gov/publications/artificial-intelligence-risk-management-framework-generative-artificial-intelligence
13. **Stanford (2024).** *Machine Unlearning in 2024*. Conceptual discussion of benchmarks, privacy, and definitions. https://ai.stanford.edu/~kzliu/blog/unlearning/
14. **OpenUnlearning research (2025).** *Accelerating LLM Unlearning via Unified Benchmarking of Methods and Metrics*. https://arxiv.org/abs/2506.12618

**Method note:** The figures and code in this article are original educational examples. Numerical demonstration results are from the included toy regression code, not measurements of a deployed language model. Formal claims refer to the cited research and their stated assumptions. Legal discussion is informational rather than legal advice.

*Last reviewed: November 27, 2026.*