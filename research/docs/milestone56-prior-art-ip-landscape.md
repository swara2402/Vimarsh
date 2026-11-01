# Milestone 56: Prior-Art and IP Landscape Review

**Status:** Preliminary desk research, 2 November 2026  
**Repository:** VIMARSH  
**Purpose:** Identify crowded technical territory, map the current implementation against relevant patent publications and research, and define a narrower, testable invention hypothesis.

> **Important:** This is a technical prior-art scan, not a legal opinion, patentability opinion, freedom-to-operate (FTO) analysis, or claim-clearance search. Google Patents' legal-status fields are expressly caveated by the service. A patent professional should review the original claims, prosecution histories, family members, jurisdictions, and priority dates before any filing or commercialization decision.

## 1. Executive finding

The broad idea of **estimating prediction reliability and using it to decide whether to accept, abstain, defer, verify, or switch models under resource constraints is already crowded**.

The current VIMARSH concept should not be presented as a first invention in any of these broad areas:

- multi-model disagreement as a signal of prediction reliability;
- learned prediction-error estimation from uncertainty features;
- selective prediction, abstention, or learning to defer;
- cost-aware model selection and resource-constrained routing;
- reliability thresholds based on verification data;
- budgeted evidence acquisition and iterative verification.

The most relevant prior-art clusters are:

1. **Reliability from agreement between multiple models:** Ummon Health's JP2024546340A describes executing a primary and other prediction models, computing similarity between outputs, and using a reliability model to assess the primary prediction.
2. **Model reliability assessment using verification data and thresholds:** Nokia's WO2024199858A1 describes configurable evaluation windows, verification data, reliability metrics, acceptance thresholds, qualification, and authorization feedback.
3. **Selective prediction and abstention:** US20230267333A1 claims a trainable selective network with a selection node that chooses whether to predict, using a differentiable reparameterization.
4. **Cost-, risk-, and constraint-aware model selection:** WO2020247943A1 describes multi-criteria model desirability incorporating accuracy, deployment cost, prediction-error cost, data-acquisition cost, risks, and hard/soft constraints.
5. **System-sensitive model routing and resource budgets:** US20250322216A1 describes dynamic model routing using user/profile and current system state; its description includes estimated model resource use and cost-budget checks before switching to another model.
6. **Budgeted evidence-driven verification loops:** CN122347229B describes action selection intended to maximize uncertainty reduction per budget, execution, evidence-pool updates, evidence gating, and iterative verification.
7. **Learning to defer to experts:** AISTATS 2023 and AAAI 2023 work already studies learned rejectors/defer decisions and human-AI allocation, including limited expert labels.

This means the current VIMARSH implementation is best understood as a research integration and evaluation platform, not yet a clearly distinguished patentable invention.

## 2. Selected patent landscape

The following are high-relevance documents identified in this preliminary search. The list is not exhaustive and is not ranked by legal strength.

| Publication | Priority / publication | Main disclosed subject | Relevance to VIMARSH |
|---|---|---|---|
| [JP2024546340A](https://patents.google.com/patent/JP2024546340A/en) | Priority 30 Dec 2021; published 19 Dec 2024 | Reliability of a model output based on similarity between a primary model and one or more other model outputs; disclosed similarity methods include cosine distance, KL information, top-class equivalence, Dirichlet variance and information radius. | High overlap with VIMARSH's multi-model disagreement and learned error-probability features. A broad claim to "use model disagreement to estimate reliability" would be weakly differentiated. |
| [WO2024199858A1](https://patents.google.com/patent/WO2024199858A1/en) | Priority 31 Mar 2023; published 3 Oct 2024 | Model reliability assessment using configured evaluation windows, verification data, evaluation functions, reliability thresholds, qualification and authorization feedback. | High overlap with verification-based reliability assessment, thresholds and feedback loops. The disclosed setting is wireless/network model qualification. |
| [US20230267333A1](https://patents.google.com/patent/US20230267333A1/en) | Priority 23 Feb 2022; published 24 Aug 2023 | End-to-end training of selective networks with a selection node that learns whether to produce a prediction, using a differentiable reparameterization. | High overlap with selective prediction/abstention as a general mechanism. VIMARSH currently uses a post-hoc policy and is not the claimed selective-network training method. |
| [WO2020247943A1](https://patents.google.com/patent/WO2020247943A1/en) | Priority 6 Jun 2019; published 10 Dec 2020 | Multi-criteria predictive-model desirability, including model accuracy, quality, deployment cost, scoring cost, false-positive/false-negative cost, data-acquisition cost, risks and hard/soft constraints. | High overlap with broad cost-, risk- and constraint-aware model selection. |
| [US20250322216A1](https://patents.google.com/patent/US20250322216A1/en) | Priority 11 Apr 2024; published 16 Oct 2025 | System-sensitive model routing for output generation, including dynamic system-resource monitoring, routing models, model selection, and cost/resource budget checks before alternate-model execution. | Relevant to runtime routing, alternate-model selection and resource-aware switching. Its claims are directed to an LLM/output-generation environment and particular routing architecture. |
| [CN122347229B](https://patents.google.com/patent/CN122347229B/en) | Priority 8 Jun 2026; published/granted 31 Jul 2026 | Evidence-driven closed-loop reasoning control; selects actions to maximize uncertainty reduction under a unified budget, executes actions, updates an evidence pool and arbitrates evidence, with further verification loops. | Very high conceptual overlap with broad budgeted verification/action selection. Its stated context is generative-model reasoning and evidence/claim arbitration, rather than VIMARSH's current tabular prediction-error estimation pipeline. |
| [US12688479B2](https://patents.justia.com/patent/12688479) | Published as US20260311100A1 in 2026; patent granted 21 Jul 2026 | AI-based electronic-data verification routing: estimates verification-success likelihood, selects a verification service provider using request/provider attributes, impact/cost and latency, and learns from prior verification outcomes. | Very relevant to action-conditioned success prediction and routing among verification providers. It is framed around document/data verification providers and routing, but makes generic claims to selecting verification providers using predicted success, cost and latency difficult to distinguish. |

### Important claim-level distinctions

- **JP2024546340A:** The independent method claim is specifically directed to obtaining outputs from a first and other predictive models, computing a similarity score between outputs, and determining reliability from that score. VIMARSH's existing disagreement features therefore cannot alone support a novelty claim.
- **WO2024199858A1:** The independent claim centers on a client/network node workflow with configuration, an evaluation window, verification data, reliability metrics, thresholds, qualification and authorization. VIMARSH does not currently implement this network-node authorization protocol.
- **US20230267333A1:** The independent claim focuses on training a selective network using a differentiably reparameterized selection node. VIMARSH's current rule-based recommendation endpoint is materially different in implementation, but abstention itself is not new.
- **WO2020247943A1:** The independent claim covers a desirability function and automated model generation/selection based on accuracy, model quality and deployment cost. A generic weighted score of reliability, cost and risk would be difficult to distinguish.
- **US20250322216A1:** The independent claim includes user identifiers, user profiles, system-resource measurements, routing models and LLM output generation. It is not a blanket claim over every system that switches models based on cost.
- **CN122347229B:** The disclosed action router, uncertainty-reduction-per-budget objective, action execution, evidence-pool update and evidence-gating loop is a close conceptual warning for broad claims around budgeted verification. Its grant and jurisdiction-specific scope require professional review; do not infer global enforceability from a Google Patents status label.

## 3. India-specific computer-related invention considerations

The official Indian Patent Office's **Guidelines for Examination of Computer Related Inventions (CRIs), 2025** are particularly relevant because VIMARSH is being developed in India:

- Section 3(k) lists mathematical or business methods, computer programmes per se, and algorithms among excluded subject matter. The guidelines discuss the need to assess the claimed invention as a whole and whether it produces a technical effect or technical contribution. [Official IP India CRI Guidelines 2025 (PDF)](https://ipindia.gov.in/frontend/pdf/patents/guidelines/GUIDELINES%20FOR%20EXAMINATION%20OF%20COMPUTER%20RELATED%20INVENTIONS%20%28CRIs%29%20-%202025.pdf)
- The AI/ML section says the key question is whether an abstract theoretical construct is transformed into an implementable technical solution, with sufficient detail for a skilled person to reproduce it. It also emphasizes disclosure of the model, training data, training methodology, and validation/real-world results where relevant.
- The guidelines describe technical effect as more than a user-facing presentation or administrative convenience. They discuss measurable effects such as improved computation, memory use, throughput, latency, system robustness, or technical operation.
- The guidelines distinguish a computer-programme exclusion from a business-method exclusion. A business-method claim may face an absolute exclusion under section 3(k), so framing a commercial verification-cost or workflow-allocation problem as a business method is a significant risk.

**Implication for VIMARSH:** a mathematical action-scoring rule or a dashboard that recommends actions is not enough. Any Indian CRI strategy would need to identify a concrete technical problem and demonstrate a credible technical effect from a specific implementation, such as measured reductions in inference latency/resource consumption or improved reliability of a technical computing/network process, without relying on generic business savings or user-interface presentation. The current API and dashboard do not establish that technical effect. This is a research planning observation, not a legal conclusion.

## 4. Relevant research literature

Patent documents are not the only relevant prior art. Published academic methods can affect novelty and inventive-step analysis even when they are not patents.

- **Mozannar & Sontag (2020), “Consistent Estimators for Learning to Defer to an Expert.”** Develops a theoretical learning-to-defer formulation using a classifier and rejector. [PMLR paper](https://proceedings.mlr.press/v119/mozannar20b/mozannar20b.pdf)
- **Mozannar et al. (2023), “Who Should Predict? Exact Algorithms For Learning to Defer to Humans.”** Studies jointly trained classifiers/rejectors, gives a MILP formulation for the linear setting, and proposes a surrogate loss. [PMLR paper](https://proceedings.mlr.press/v206/mozannar23a.html)
- **Hemmer et al. (2023), “Learning to Defer with Limited Expert Predictions.”** Reduces the amount of expert-labelled data needed to train learning-to-defer algorithms. [AAAI paper](https://ojs.aaai.org/index.php/AAAI/article/view/25742)
- **“A benchmarking framework and dataset for learning to defer in human-AI decision-making” (Scientific Data, 2025).** Evaluates deferral with heterogeneous expert performance and capacity constraints; highlights the need to measure the human side rather than assume it is perfect. [Scientific Data paper](https://www.nature.com/articles/s41597-025-04664-y)
- **“Learning Selective Deferral Policies for Reliable Medical Text Classification” (2026).** Describes a learned deferral framework combining uncertainty features, calibration, a deferral policy and fixed review budgets. Its discussion explicitly distinguishes simulated review from actual expert outcomes. [Article](https://www.mdpi.com/2227-7080/14/6/359)
- **“Selective prediction-set models with coverage rate guarantees” (Biometrics, 2023).** Studies abstention with coverage-rate guarantees to balance reliability and expert workload. [PubMed record](https://pubmed.ncbi.nlm.nih.gov/34854476/)

## 5. Mapping against the current VIMARSH implementation

| VIMARSH capability | Current status | Prior-art pressure | What remains unproven |
|---|---|---|---|
| Multi-model uncertainty/disagreement and learned error-probability estimation | Implemented in research pipeline | High: JP2024546340A and extensive failure-prediction literature | Whether the combination yields a reproducible advantage over confidence-only and other baselines |
| Calibration, uncertainty and safety diagnostics | Research benchmark implemented | High: calibration and selective-prediction literature | Whether any novel calibration-conditioned action policy improves out-of-domain outcomes |
| Risk/cost/coverage-aware action evaluation | Legacy research engine and simulations | High: WO2020247943A1, learning-to-defer and budgeted-routing work | Real measured action costs and action-specific outcome probabilities |
| Budget/availability constraints | Implemented in research engine and simple API baseline | High: broad constrained optimization and budgeted verification prior art | A distinct optimization objective or algorithm with measurable technical benefit |
| Human review | Simulated only; not operational | High: learning-to-defer research | Real reviewer labels, reviewer-specific accuracy, latency, capacity and disagreement |
| Additional evidence | Not operational in the API; legacy simulation only | High: evidence acquisition and evidence-driven reasoning systems | A real evidence source, provenance, cost, quality and measured incremental value |
| Alternate model | Recommendation-only API; offline measured substitution in research audit | High: model routing and model selection patents | Actual inference-time runner, measured latency/cost and paired action outcomes |
| Paired outcome audit | Implemented for offline alternate-model substitution | Potentially useful differentiator, but not automatically novel | Replication, external datasets, confidence intervals, causal limitations and operational outcomes |
| React dashboard | Implemented; research preview | Low patent relevance by itself | No independent technical contribution established by presentation/UI |

## 6. Preliminary novelty assessment

### What is not a defensible novelty claim

Avoid claims such as:

- "A new system that estimates ML prediction reliability using confidence and disagreement."
- "A novel AI system that decides when to trust, verify or reject a prediction."
- "A budget-aware AI verification framework."
- "An AI system that selects between models based on reliability, cost and risk."
- "A human-in-the-loop system that sends uncertain predictions for review."

Each is broad and substantially overlaps with existing patent or academic work.

### A narrower research hypothesis worth testing

A more specific direction to investigate is:

**Outcome-grounded, action-conditioned verification allocation under hard resource constraints.**

The technical question is not simply "which prediction looks least reliable?" It is:

> Given a calibrated estimate of prediction error, a harm estimate, a set of actually available verification actions, action-specific empirical outcome distributions, and a hard remaining budget, which action (if any) has the greatest conservative expected reduction in residual decision risk per unit cost, while respecting safety and capacity constraints?

A possible technical architecture would require all of the following, not just a weighted score:

1. **Action-conditioned outcome models:** Estimate the probability distribution of each action's result for a given prediction/context, using paired, observed pre-action and post-action outcomes rather than fixed assumed effectiveness factors.
2. **Conservative benefit estimates:** Account for uncertainty in the estimated action benefit, for example with lower confidence bounds or distributionally robust estimates. Abstain if evidence is insufficient.
3. **Constrained allocation:** Select actions under a shared per-request and/or batch budget, action availability, capacity, latency and policy constraints. Compare against confidence-only, risk-only, random matched-budget, greedy value-per-cost, and established learning-to-defer baselines.
4. **Outcome provenance:** Record action request, action provider/version, cost, elapsed time, result, whether the original prediction changed, and eventual ground truth. Keep retrospective labels out of inference-time selection.
5. **Auditable policy updates:** Update action-benefit estimates only from completed, quality-controlled outcome records, with drift and calibration monitoring.

This is only a **hypothesis**, not a finding of novelty. Budgeted uncertainty reduction, evidence acquisition, model routing and human deferral already have close prior art. The proposed contribution would need a precise mathematical formulation and an experimental advantage over those methods.

## 7. Critical experiments before considering a patent draft

1. **Real action outcome dataset:** Collect paired outcomes for at least one genuinely operational action. For human review, record actual reviewer decisions and reviewer identity/qualification at an appropriate privacy level. For alternate models, record actual predictions, latency, resource cost and subsequent ground truth.
2. **Action benefit estimation:** Compare fixed-effectiveness assumptions, simple empirical action success rates, and a context-conditioned action-outcome estimator. Evaluate calibration of both error probability and action benefit.
3. **Matched-budget policy comparison:** Compare VIMARSH with confidence-only, entropy, disagreement, random selection, risk-per-cost greedy selection, and suitable learning-to-defer baselines at the same budget and coverage.
4. **Uncertainty-aware decision rule:** Test whether conservative benefit estimates improve worst-case or tail risk without collapsing coverage. Pre-register the primary metric and thresholds.
5. **External validation:** Use multiple independent datasets and at least one real operational domain with actual action outcomes. Do not tune on the final test set.
6. **Ablation:** Remove action-outcome modelling, benefit uncertainty, budget coupling, calibration, and each reliability feature group separately.
7. **Statistical evidence:** Report paired confidence intervals, bootstrap uncertainty, per-seed results, and subgroup/shift analyses. Include negative results.
8. **Systems evaluation:** Measure real inference latency, action latency, memory, throughput, and cost. The current caller-supplied abstract cost units are not operational cost evidence.

## 8. IP hygiene and next actions

- Preserve dated, version-controlled design notes, experiment configurations, results, and authorship/contribution records.
- Do not publicly disclose a potentially novel implementation before discussing filing strategy with the institution's IP cell or a patent professional.
- Review ownership, institutional IP policies, sponsor/funding conditions, open-source licenses, and third-party dataset/model licenses.
- Ask a patent professional to conduct a claim-focused search in relevant jurisdictions, including India, WIPO/PCT, US, EP, CN, JP and KR, and inspect family members, continuations, prosecution history, citations and legal status.
- Have counsel evaluate novelty, inventive step/non-obviousness, enablement, claim scope and freedom to operate separately. A patent being granted or pending is not the same question as whether VIMARSH infringes it.
- Keep the proposed novelty hypothesis private until it has been technically specified and reviewed.

## 9. Preliminary conclusion

VIMARSH has a credible research platform and a growing empirical record, but the current broad concept is not sufficiently differentiated to support a confident patentability claim. The strongest next step is **not drafting broad patent claims**. It is implementing and validating a specific action-conditioned outcome model with real paired verification data, conservative benefit estimation and a constrained allocator, then comparing it against established selective prediction and learning-to-defer baselines.

This review is a starting map, not an exhaustive search. It does not determine whether any particular claim would be patentable or infringing.
