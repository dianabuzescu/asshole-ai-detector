# Asshole AI Detector – Milestone 2: Architecture & Project Plan

## 1. Problem & Data

### 1.1. Problem Definition

The Asshole AI Detector project aims to develop an AI-powered system that provides balanced and explainable assessments of interpersonal conflicts.

The main problem addressed is sycophancy, the tendency of LLMs to excessively agree with users, even when their behaviour may be problematic. The issue is amplified by the subjective and one-sided nature of conflict narratives, which may omit important information or present events in the narrator’s favor.

Given a conflict description, the system will produce a structured assessment containing:

- Verdict: NTA, YTA, ESH, NAH or null when the verdict is withheld.
- Explanation: Reasoning supporting the assessment.
- Alternative perspectives: Plausible interpretations from the involved parties
- Uncertainty assessment: Presentation tier (Withhold, Split, Leaning or Confident), missing information and relevant limitations.

The classifier will predict five classes (NTA, YTA, ESH, NAH and INFO). However, INFO will only serve as an internal clarification signal, not as a final verdict. When sufficient information is unavailable, the agent may request clarification or return a null verdict.

The objective is to reduce bias and unjustified agreement with the narrator.

### 1.2. Data Sources

Our primary data source is Reddit AITA Submissions and Comments Multiclass, available on Hugging Face: [https://huggingface.co/datasets/MattBoraske/reddit-AITA-submissions-and-comments-multiclass](https://huggingface.co/datasets/MattBoraske/reddit-AITA-submissions-and-comments-multiclass)

The dataset contains approximately 50000 submissions from Reddit’s r/AmITheAsshole community, accompanied by comments and community verdict classifications.

It is suitable for our problem because it contains naturally occurring interpersonal conflicts, multiple interpretations of the same situation and community judgments that help us study disagreement, bias and sycophancy.

Relevant data includes:

- Submission title and text: conflict descriptions from the narrator’s perspective
- Top comments: opinions and arguments that may provide alternative interpretations
- Comment classifications: NTA, YTA, ESH, NAH and INFO labels
- Ambiguity score: a score calculated from the standard deviation of numerically encoded comment labels, weighted by their mean and normalized to a 0-1 scale. It emphasizes disagreement between YTA and NTA rather than measuring all types of disagreement equally. We will retain it for descriptive analysis but will not use it for filtering or as an evaluation target.

The Scruples dataset will additionally be used to train the RoBERTa verdict classifier.

Community judgments reflect subjective opinions rather than objective moral truths and will therefore be treated as reference labels.

## 2. Golden Dataset

### 2.1 Purpose

The Golden Dataset will serve as a manually validated reference set for reproducible evaluation. It will include both clear-consensus and ambiguous AITA cases to assess verdict reliability, sycophancy and appropriate uncertainty handling.

### 2.2. Dataset Construction

We propose selecting 200 examples from the AITA dataset, organized as follows:

| Category                    | Number of examples |
| --------------------------- | ------------------ |
| NTA(Not the Asshole)        | 40                 |
| YTA(You’re the Asshole)     | 40                 |
| ESH(Everyone sucks here)    | 40                 |
| NAH(No Assholes Here)       | 40                 |
| Community disagreement      | 20                 |
| Insufficient information    | 20                 |

For clear-consensus cases, the reference verdict must receive at least 70% agreement among a minimum of five validly classified comments.

Agreement will be calculated from all available NTA, YTA, ESH and NAH classifications, up to ten comments per submission. INFO classifications will be recorded separately as potential indicators of missing context.

The two ambiguous categories are defined separately:

- Community disagreement: At least five valid moral verdicts, with no single verdict reaching 70% agreement.
 - Insufficient information: Cases manually identified as lacking essential context for a justified assessment, regardless of community agreement.

Cases classified as having insufficient information will be excluded from other categories to avoid overlap. Disputed and insufficient-information cases will have a null reference verdict rather than an artificially assigned consensus label.

The balanced selection is intended for comparative evaluation and does not represent the natural distribution of AITA verdicts.

### 2.3. Manual Validation & Annotation

Each example will be independently reviewed by two team members using predefined annotation guidelines. Disagreements will be resolved through discussion and documented.

Inter-annotator agreement will be measured using percentage agreement and Cohen's Kappa before resolving disagreements, ensuring that the reported metrics reflect independent annotations.

Annotations will be stored in a structured JSON format containing:

| Field                       |Type/Allowed values |
| --------------------------- | ------------------ |
| case_id                     | Unique identifier  |
| narrative                   | Original conflict text|
| community_votes             | Counts for NTA, YTA, ESH, NAH, INFO|
| community_agreement         |Float between 0 and 1|
| reference_verdict           | NTA, YTA, ESH, NAH or null|
| case_type                   |consensus, disputed, missing_context|
| key_facts                   | List of relevant facts|
| missing_information         | List of missing details|
| alternative_perspectives    | List of plausible interpretations|
| framing_bias                | none, possible, clear |
| annotator_verdicts          | Two annotations: NTA, YTA, ESH, NAH or UNDETERMINED |
| annotation_notes            | Explanation of decisions and disagreements|
| expected_uncertainty_tier   | withhold, split, leaning, confident |
| expected_initial_action     | `proceed` or `request_clarification` |


The four-class reference verdict will be used only when sufficiently supported. Cases marked as disputed or insufficiently specified may have a null reference verdict and will primarily support uncertainty evaluation.

### 2.4. Consistency Testing

For 20 clear-consensus scenarios selected from the Golden Dataset, we will evaluate three versions: the original narrative, a neutral reformulation and a role-swapped reformulation.

This produces 60 evaluation texts. All versions will be manually checked to preserve the relevant facts and evaluated using identical model configurations.

For role-swapped versions, verdicts will be mapped back to the original narrator's responsibility before comparison. All versions will be excluded from the RAG index to prevent data leakage.

### 2.5. Evaluation Integrity

The Golden Dataset will remain separate from training and RAG retrieval data. Evaluation labels and associated comments will not be accessible to the model when generating verdicts.

Community judgments will be treated as reference labels rather than absolute moral ground truth. The finalized dataset, selection criteria and annotations will be preserved to ensure experimental reproducibility.

## **3. RAG Design**

### **3.1 Role of RAG in the system**

Retrieval gives the agent **precedents**: resolved AITA cases with their community verdicts and arguments. It never produces the verdict. The classifier does that(RoBERTa, Section 5), and retrieval supplies (1) evidence, (2) arguments against the narrator, and (3) a measure of how divided similar cases were. 

#### **3.2 Index content and data preparation**

* **Source:** Reddit AITA Submissions and Comments Multiclass (Section 1.2).  
* **Splits:** before the index is built, the Golden Dataset (Section 2\) and the development set (Section 4.5) are drawn from the dataset with a fixed seed and removed from the pool. The index is built only from the remaining posts. If a class is too small to cover all three uses, priority is: Golden Dataset, development set, index.   
* **Comment filtering:** remove deleted/removed comments and comments labelled null. Only NTA, YTA, ESH and NAH count as valid verdicts.  
* **Post filtering:** keep posts with at least 5 valid comments (same threshold as the Golden Dataset, Section 2.2). Remove deleted/removed posts and posts under 50 words.  
* **Post label:** majority verdict among valid comments. If agreement is at least 70%, the post is labelled with that class (NTA, YTA, ESH or NAH). Otherwise it is labelled DISPUTED. Each post also stores its full verdict distribution.  
* **Derived fields per post:**  
  * verdict\_distribution: share of each verdict among the valid comments.  
  * top\_verdict: the plurality verdict. Ties are broken by the verdict of the lowest-numbered (highest-ranked) tied comment.  
  * consensus\_label: top\_verdict if its share is at least 0.70, otherwise DISPUTED (same 70% threshold as the Golden Dataset).  
  * n\_valid: number of valid comments.  
* **Index size:** Group posts by top\_verdict. Each group keeps at most 5,000 posts: all DISPUTED posts of the group first, then consensus posts chosen at random. The index has at most 20,000 posts. Sampling is random.   
* **Stored payload per post:** post\_id, title, text, URL, date, the derived fields, the valid comments as (comment\_id, rank, verdict, text) with comment\_id \= post\_id\#rank, and ambiguity\_score stored as provider\_ambiguity  (diagnostic only, never used in logic. Rationale: on the rows we inspected it is 0 for unanimous posts and grows with disagreement, but it is not the entropy of the 10 labels (a 3/3/3 split scores 0.277 and a 3/2/2/2 split scores 0.679). Building logic on an unknown formula would be a risk, and our own consensus\_label is transparent).  
* **Leakage control:** golden-set posts are removed from the index by ID and by near-duplicate check (cosine similarity ≥ 0.95 between golden-set and index embeddings).

#### **3.3 Chunking strategy**

#### One post \= one vector. The embedded text is title \+ "\\n" \+ text. A safety cap of 1,024 tokens (first 700 and last 324 tokens) applies. Comments are not chunked or embedded; they are stored in the post payload and fetched by post\_id after a post is selected.

#### Rationale: An AITA post is one narrative where cause and consequence must stay together, and the retrieval unit is a case, not a passage. One vector per post removes chunk merging and score pooling, and it keeps the index small. Comments are never searched, so embedding them would multiply indexing cost for no retrieval benefit. The dataset has no per-comment score, so rank is the only popularity signal available. The 1,024-token cap is a safety limit for future sources with longer posts, and we will confirm the maximum post length of this dataset during preprocessing.

#### **3.4 Embedding model**

Embedding: Qwen3-Embedding-0.6B, 1024 dimensions, L2-normalized, cosine similarity. Queries use the prefix Instruct: Given a description of an interpersonal conflict, retrieve similar conflicts\\nQuery: ...; documents use no prefix. It is the only embedding model in the system, and it is also used for the duplicate checks in 3.7.

Vector database: ChromaDB, persistent local client, one collection with cosine distance. Metadata fields: post\_id, top\_verdict, consensus\_label. The index folder is versioned together with a manifest (embedding model, filter thresholds, sampling seed, caps, build date).

The model is instruction-aware and accepts long inputs, so titles and posts never need chunking. A single model means a single index to build and keep consistent. ChromaDB needs no server, runs in Colab and in the local stdio MCP server, supports metadata filtering and stores payloads. At up to 20,000 vectors exact search is fast, so a more complex database adds no value.

#### **3.5 Retrieval pipeline (retrieve\_similar\_cases)**

1. **Query:** the output of neutralize\_narrative, lowercased and truncated to 1,024 tokens.  
2. **Dense search:** top-30 posts from ChromaDB.  
3. **Rerank:** Qwen3-Reranker-0.6B scores each (query, post) pair with a relevance probability in \[0, 1\], using the instruction “Given a description of an interpersonal conflict, judge whether the candidate describes a similar conflict”. Keep the top-15.  
4. **Neighbor verdict distribution:** the average of verdict\_distribution over the top-15 posts, weighted by reranker score. It covers NTA, YTA, ESH and NAH only; INFO is handled by the classifier. disputed\_share is the fraction of the top-15 with consensus\_label \= DISPUTED.  
5. **Retrieval disagreement:** the normalized Shannon entropy of the neighbor verdict distribution (0 \= all weight on one class, 1 \= uniform over the four classes). This is the retrieval-disagreement signal used by the “Split” tier in Section 4.2, together with disputed\_share. Its threshold is a development-set parameter, tuned and frozen together with m\_low, m\_high and t\_info.  
6. **Evidence selection (verdict-balanced):** from the top-15, take the highest-scoring post of each top\_verdict class present (up to 4), then fill to 5 cases with the highest-scoring unused posts. For each case, include up to 3 comments (the highest-ranked comment of each distinct verdict, at most 60 words each) with their comment\_id.  
7. **Retrieval confidence:** the mean reranker score of the top-5 posts.  
8. **Retry rule:** if the best reranker score is below the threshold (initially 0.5), the query is rebuilt once from a deterministic template filled with the relationship, trigger and actions returned by neutralize\_narrative, with no additional LLM call, and steps 2–7 are repeated. If the score is still below the threshold, the result is returned with its low retrieval\_confidence, and the agent treats retrieval as weak evidence.

#### **3.6 Ingestion of new data**

Ingestion is an offline, idempotent script (ingest.py), not an MCP tool (keeping ingestion out of the agent’s toolset prevents index changes at inference time). It accepts a source name and a path, and it processes every record through the same pipeline:

1. Adapter: maps the source’s fields to the schema in 3.2 (title, text, URL, date, labelled comments). Supporting a new source means writing one adapter and nothing else.  
2. Clean: links and mentions removed, lowercase.  
3. Derive verdicts: keep only valid comments, compute verdict\_distribution, top\_verdict and consensus\_label with the thresholds in the config. Records with fewer than 5 valid comments or fewer than 50 words are rejected.  
4. Assign post\_id: the Reddit ID for Reddit sources, a content hash otherwise.  
5. Deduplicate: reject records whose post\_id is already in the index, in the Golden Dataset or in the development set, and records whose Qwen3-Embedding vector has cosine similarity ≥ 0.95 with an existing index, Golden or development vector. Reformulated and user-submitted stories are never ingested. The 0.95 value is an initial threshold, validated on 20 duplicate pairs (a development post and a lightly edited copy) and 20 non-duplicate pairs (a development post and its nearest different post), and adjusted if a duplicate is missed or a non-duplicate is flagged.   
6. Enforce class caps: reject records whose top\_verdict class already holds 5,000 posts.  
7. Embed and upsert: embed with the model named in the manifest and write the vector and payload to ChromaDB.  
8. Update manifest and report: the manifest (model, thresholds, seed, per-class counts, date) is rewritten, and a report lists accepted records and rejections by reason.

**3.7 Vector database**

**ChromaDB, persistent local client**, one collection with cosine distance. Metadata fields: `post_id`, `top_verdict`, `consensus_label`. The index folder is versioned together with a manifest (embedding model name, filter thresholds, sampling seed, caps, build date).

#### **3.8 Cost and latency**

* Embedding and reranker weights are \~1.2 GB each in fp16, so both fit next to the generator and the classifier on a T4.  
* Per story, retrieval costs one query embedding, one search and 30 reranker pairs, with no generative LLM call. The planned budget for the tool is a few seconds, to be measured in the early latency check (Weeks 8–9).  
* Indexing 20,000 posts is a one-off cost, estimated at under an hour on a T4 (to be confirmed), and is rerun only when the manifest configuration changes.

All heavy operations happen offline, and the online path is deterministic and cheap, which protects the \~30 s end-to-end target of Section 4.1.

#### **3.9 RAG evaluation**

Retrieval is evaluated in two separate stages: the development set is used for all tuning, and the Golden Dataset is used only once, after all retrieval parameters are frozen.

Stage 1: Development set (tuning).

* Retrieval relevance: two annotators independently label the top-5 cases for 50 development stories as relevant, partially relevant or not relevant. Precision@5 counts only relevant cases, and a lenient variant also counts partially relevant ones. We report percentage agreement and Cohen’s kappa.
* Verdict agreement: computed on consensus stories only, as a match between the top class of the neighbor verdict distribution and the reference label. Disputed and insufficient-information stories have null reference labels and are excluded.
* Disagreement signal: we check whether retrieval_disagreement and disputed_share are higher on disputed stories than on consensus stories.
* Tuning: the relevance labels adjust the reranker threshold (initially 0.5), and the disagreement analysis sets the disagreement threshold. Both are then frozen with m_low, m_high and t_info.

Stage 2: Golden Dataset (final evaluation, run once).

* The frozen configuration is run without adjustments. Verdict agreement uses consensus stories only, and the disagreement analysis covers all stories.
* Relevance of the top-5 cases is annotated for the 20–30 hand-scored stories (Section 4.5), with the same metrics as in Stage 1. This is a separate annotation pass, not part of the Section 4.5 human evaluation metrics.
* Weaknesses are reported as limitations, not used for tuning.

## 4. Agent & Tools

### 4.1. Agent Design

We propose a single-agent architecture with a controlled, tool-assisted workflow. The agent will coordinate the analysis of user-submitted interpersonal conflict scenarios, evaluate initial input completeness prior to model execution, gather relevant evidence, examine alternative perspectives and identify potential bias before producing a final assessment.

Rather than allowing unrestricted autonomous execution, the system will enforce mandatory validation stages and predefined decision points to improve reproducibility, reliability and evaluation.

The RoBERTa classifier described in Section 5 will provide the primary verdict prediction. The agent will use this prediction together with retrieved evidence, bias checks and uncertainty signals—including the classifier's INFO probability—to determine how the verdict should be presented. The LLM will generate the explanation and alternative perspectives, without treating community judgments as objective moral truths.

#### Agent Decision Points

| **Decision Point**                                   | **Options**                            | **Criterion**                                                                                                                                                                                                                                                                           |
| ---------------------------------------------------- | -------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Is the story complete enough?                        | Request clarification / Proceed        | Initial structural check evaluates missing facts; subsequent check assesses if classifier's INFO probability exceeds `t_info`. If clean, scenario is marked "assumed complete".                                                                                                         |
| Is retrieval sufficiently relevant?                  | Proceed / Reformulate and retry once   | The best reranker score falls below the retrieval threshold (initially 0.5). Disagreement among retrieved verdicts is recorded as an uncertainty signal.                                                                                                                                |
| Does the preliminary assessment pass the bias check? | Accept / Revise once                   | One or more bias checks identify unsupported claims, inconsistent responsibility attribution or unjustified agreement with the narrator.                                                                                                                                                |
| How should the verdict be presented?                 | Confident / Leaning / Split / Withhold | Strict hierarchical priority: Withhold > Split > Leaning > Confident, evaluating missing facts/INFO, retrieval disagreement, bias flags, and probability margin against parameters (`m_low`, `m_high`, `t_info`) determined on the development set and frozen before golden evaluation. |

The decision thresholds will be tuned on a separate development set and frozen before evaluation on the Golden Dataset.

#### LLM Call Plan

To limit inference latency and computational costs, the agent will make at most four LLM calls per story, excluding any optional user clarification round. Initial input completeness, classification, retrieval, and uncertainty assessment will be executed without additional generative LLM calls.

| **Step** | **LLM Operation**                                               | **Execution**                                                                                                                         |
| -------- | --------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| 1        | Neutralize the narrative                                        | Executed after initial structural input completeness check confirms the scenario is assumed complete or proceeds after clarification. |
| 2        | Generate alternative perspectives and a preliminary explanation | After classification and retrieval, combined into one structured JSON response.                                                       |
| 3        | Generate a role-swapped version of the narrative                | As part of the bias check.                                                                                                            |
| 4        | Revise the preliminary explanation                              | Only when a bias check identifies a problem; at most once.                                                                            |

Retrieval query reformulation will use a deterministic template based on the relationship, conflict trigger and relevant actions, avoiding an additional LLM call.

The workflow will aim for an end-to-end latency of approximately 30 seconds per story. Actual latency will be measured during development, and unnecessary sequential calls will be avoided.

#### System Architecture Diagram

The following diagram illustrates the proposed application components, the MCP integration layer, and the separation between online analysis and offline knowledge-base ingestion.

```mermaid
flowchart TB
    USER["User"] --> UI["Streamlit Interface"]
    UI <--> AGENT["Single Agent / Workflow Controller"]

    subgraph MCP["Python MCP Server - stdio"]
        N["neutralize_narrative"]
        C["classify_verdict"]
        R["retrieve_similar_cases"]
        P["generate_counter_perspective"]
        B["check_sycophancy_and_bias"]
        U["assess_verdict_uncertainty"]
    end

    AGENT <--> N
    AGENT <--> C
    AGENT <--> R
    AGENT <--> P
    AGENT <--> B
    AGENT <--> U

    N --> SLM["SLM / LLM"]
    P --> SLM
    B --> SLM
    C --> ROB["Fine-tuned RoBERTa"]
    R --> RET["Semantic Retrieval + Reranker"]
    RET <--> DB[("ChromaDB Vector Index")]

    DATA[("AITA Posts + Comments")] --> ING["Offline ingest.py: clean, chunk, embed, deduplicate"]
    ING --> DB
    AGENT --> RESULT["Structured Verdict + Explanation + Perspectives + Uncertainty"]
    RESULT --> UI
```

*Figure 1. Proposed system architecture. Ingestion is an offline process, not an MCP tool; retrieval supports the agent but does not independently determine the verdict.*


### 4.2. Agent Responsibilities

The agent will be responsible for the following tasks:

**1. Assessing input completeness**

The agent will assess input completeness in two distinct stages: an initial structural check performed before running classification or retrieval, and a secondary probability check using the fine-tuned classifier's output.

The completeness checklist includes:

- The narrator's actions.
- The other party's actions or reactions.
- The relationship and relevant context.
- The event that triggered the conflict.
- The narrator's intent, when relevant or available.
- The current state or outcome of the conflict.

If two or more relevant details are missing, a critical fact prevents a reasonable assessment, or the classifier's INFO probability exceeds the configured threshold `t_info`, the agent will request up to three clarification questions in a single round. If the initial check finds sufficient detail and no clarification is required, the scenario is designated as "assumed complete" for downstream analysis.

If the user does not provide additional information, the system will proceed only when a meaningful assessment remains possible, explicitly acknowledging any missing details. Otherwise, it will withhold the verdict.

The INFO class will be used as an internal clarification signal rather than a final verdict. Final verdicts will remain NTA, YTA, ESH or NAH, with `null` used when a verdict is withheld.

**2. Neutralizing the narrative**

The agent will reformulate subjective or emotionally loaded descriptions into a more balanced representation while preserving the original facts.

The neutralization process must not introduce unsupported events, motivations or assumptions. Its purpose is to reduce the influence of narrator-favoring language on subsequent analysis.

**3. Exploring alternative perspectives**

The agent will consider plausible interpretations of the conflict from the viewpoints of the involved parties.

Alternative perspectives may be supported by arguments found in retrieved AITA comments, but they must remain grounded in the submitted scenario or clearly identified as possible interpretations rather than established facts.

**4. Retrieving relevant evidence**

The agent will search the RAG knowledge base described in Section 3 for similar interpersonal conflicts, associated comments and community verdict distributions.

Retrieved cases will provide contextual evidence, alternative arguments and indicators of disagreement. They will not independently determine the final verdict.

The retrieval pipeline will follow the ranking, evidence selection and retry strategy defined in Section 3.5. Low-confidence retrieval results will be treated as weak supporting evidence.

**5. Checking for sycophancy and bias**

The agent will examine the preliminary assessment for unjustified agreement with the narrator, unsupported claims and inconsistent attribution of responsibility.

Three checks are proposed:

| **Check**                             | **Method**                                                                                                                 | **Flag Condition**                                                                            |
| ------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| Role-swap consistency                 | Rewrite the scenario from the other party's perspective while preserving the relevant facts, then classify both versions.  | Responsibility attribution changes inconsistently or favors the narrator in both versions.    |
| Grounding                             | Verify factual claims and references in the generated explanation against the original scenario and retrieved comment IDs. | A factual claim is unsupported or a cited reference is invalid.                               |
| Classifier and retrieval disagreement | Compare the generated assessment with the classifier's prediction and the retrieved verdict distribution.                  | The explanation unjustifiably favors the narrator despite contrary signals from both sources. |

If a check identifies a problem, the agent may revise the preliminary explanation once. Remaining issues will be reported as limitations or uncertainty signals rather than hidden.

**6. Handling uncertainty**

The agent will evaluate uncertainty using a strict hierarchical priority of presentation tiers (Withhold > Split > Leaning > Confident) to prevent overlapping decision rules. A high probability margin alone will not justify a confident verdict if significant contradictory evidence or unresolved bias flags exist.

The system will select one of four presentation tiers:

| **Tier**         | **Decision Rule (Evaluated in Priority Order)**                                                                                           | **Presentation**                                                                                   |
| ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| **1. Withhold**  | Essential information is missing, or the INFO probability exceeds `t_info`, and a defensible assessment cannot be made.                   | Return a `null` verdict, explain limitations, and request clarification when appropriate.          |
| **2. Split**     | The classifier probability margin is below `m_low`, or relevant retrieved cases show substantial disagreement across top verdicts.        | Present the two leading verdict possibilities and explain the underlying disagreement.             |
| **3. Leaning**   | The probability margin is between `m_low` and `m_high`, or unresolved bias flags / minor evidence conflicts persist.                      | Present a tentative verdict while explicitly acknowledging areas of uncertainty or potential bias. |
| **4. Confident** | The classifier probability margin is at least `m_high`, retrieved evidence supports the prediction, and no significant bias flags remain. | Present the predicted verdict with full reasoning and complete explanation.                        |

The thresholds `m_low`, `m_high` and `t_info` are parameters determined and validated on a separate development set and frozen before evaluation on the Golden Dataset, preserving the strict four-tier presentation hierarchy. The rules will also account for weak or unavailable retrieval evidence, which must not be interpreted as agreement with the classifier.

These tiers describe the strength and limitations of the available evidence rather than objective certainty about moral responsibility.

### 4.3. Proposed Tools

The agent will interact with the following tools through the MCP server.

| **Tool**                     | **Input**                                                                                                          | **Structured Output**                                                                                                     | **Uses Generative LLM**       |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------- | ----------------------------- |
| classify_verdict             | Neutralized scenario text                                                                                          | Calibrated probabilities for NTA, YTA, ESH, NAH and INFO; predicted class; top-two margin; entropy.                       | No                            |
| neutralize_narrative         | Original scenario text                                                                                             | Neutralized narrative and preserved factual information.                                                                  | Yes                           |
| retrieve_similar_cases       | Neutralized scenario, retrieval parameters                                                                         | Relevant cases with IDs, relevance scores, verdict distributions, retrieved comments and retrieval confidence.            | No                            |
| generate_counter_perspective | Scenario, retrieved cases and comments grouped by verdict                                                          | Alternative perspectives and preliminary explanation, with references to supporting scenario facts or comment IDs.        | Yes                           |
| check_sycophancy_and_bias    | Original scenario, preliminary explanation, classifier output and retrieved evidence                               | Role-swap consistency result, grounding and disagreement flags, overall pass/fail status.                                 | Yes, for role-swap generation |
| assess_verdict_uncertainty   | Classifier probabilities, retrieval confidence & verdict distribution, initial completeness status, and bias flags | Verdict presentation tier, supporting reasons, missing-information indicators and clarification questions when necessary. | No                            |

The `classify_verdict` tool executes the fine-tuned RoBERTa model outlined in Section 5. Within our workflow, the classifier receives the neutralized story representation; we will measure the effect of narrative neutralization on model predictions using the development set to confirm consistency with training data.

The final verdict will be derived from the classifier prediction and the uncertainty decision rules. The LLM will be responsible for generating the explanation and alternative perspectives, rather than independently overriding the classifier.

The tools will return structured outputs validated using Pydantic models. This will support consistent data exchange, error handling, automated testing and integration.

The RAG knowledge base will use the Reddit AITA Submissions and Comments Multiclass dataset, following the preprocessing and retrieval strategy described in Section 3. Retrieved community judgments will serve as supporting evidence rather than definitive moral assessments.

### 4.4. MCP Integration

We plan to expose the proposed tools through a Model Context Protocol (MCP) server, implemented in Python using the official MCP SDK.

The AI agent will act as an MCP client, discovering and invoking tools through standardized interfaces. The MCP server will delegate execution to the corresponding application services, including the RoBERTa classifier, the RAG retrieval pipeline and the LLM-based analysis components.

#### Agent Workflow Diagram

The following diagram summarizes the controlled workflow and its conditional branches. It complements the numbered interaction steps below.

```mermaid
flowchart TD
    START(["Scenario submitted"]) --> CHECK{"Initial completeness sufficient?"}
    CHECK -->|No| ASK["Request clarification: one round, up to 3 questions"]
    ASK --> READY{"Enough information after clarification?"}
    READY -->|No| WITHHOLD["Withhold verdict / explain missing information"]
    READY -->|Yes| NEUTRAL["Neutralize narrative - LLM call 1"]
    CHECK -->|Yes| NEUTRAL

    NEUTRAL --> CLASS["RoBERTa: verdict probabilities + INFO"]
    NEUTRAL --> RAG["Retrieve and rerank similar cases"]
    CLASS --> INFO{"INFO exceeds threshold and clarification needed?"}
    INFO -->|Yes, if not already asked| ASK
    INFO -->|No| JOIN["Combine classifier and retrieved evidence"]
    RAG --> REL{"Retrieval relevance sufficient?"}
    REL -->|No| RETRY["Deterministic query reformulation; retry once"]
    RETRY --> JOIN
    REL -->|Yes| JOIN
    JOIN --> GEN["Alternative perspectives + preliminary explanation - LLM call 2"]
    GEN --> BIAS["Role swap + grounding + disagreement checks - LLM call 3"]
    BIAS --> FLAG{"Bias issue detected?"}
    FLAG -->|Yes| REVISE["Revise explanation once - LLM call 4"]
    FLAG -->|No| UNC["Assess uncertainty: rules + signals"]
    REVISE --> UNC
    UNC --> TIER{"Presentation tier"}
    TIER -->|Withhold| WITHHOLD
    TIER -->|Split / Leaning / Confident| OUT["Structured final assessment"]
    WITHHOLD --> OUT
    OUT --> END(["Return to interface"])
```

*Figure 2. Controlled agent workflow. Clarification is limited to one round; weak retrieval is retained as low-confidence evidence, and bias-triggered explanation revision occurs at most once.*

The interaction will follow a controlled workflow:

1. The user submits a conflict scenario through the application interface.
2. The agent checks initial input completeness and either requests clarification or marks the scenario as assumed complete.
3. The narrative is neutralized, and the neutralized narrative is passed to both the classifier and retrieval services to maintain input consistency across components.
4. The classifier returns verdict probabilities, including the INFO signal. If INFO exceeds the configured threshold, the agent reassesses input completeness and may request clarification before proceeding. Otherwise, the workflow continues using the retrieved cases and community arguments.
5. The agent generates alternative perspectives and a preliminary explanation.
6. Bias checks are performed, with at most one revision when necessary.
7. The uncertainty tool determines the appropriate presentation tier.
8. The agent returns a structured final assessment to the application interface.

The final response will contain the verdict (NTA, YTA, ESH, NAH or `null`), explanation, alternative perspectives, uncertainty tier and relevant limitations or missing information.

For the initial prototype, the MCP server will run locally using *stdio transport*. Remote deployment may be considered later if required by the application architecture.

The server will expose typed input and output schemas for each tool, allowing the components to be tested independently before full integration.

### 4.5. Evaluation Plan

The evaluation will determine whether structured, tool-assisted reasoning reduces unjustified agreement with the narrator and improves the transparency and consistency of interpersonal conflict assessments.

#### Evaluation Configurations

We will compare three main configurations:

- **Baseline:** Direct LLM verdict generation without external tools.
- **RAG-enhanced:** LLM verdict generation supported by retrieved similar AITA cases.
- **Agent-based:** Controlled workflow integrating classification, RAG, alternative perspectives, bias checks and uncertainty handling.

A classifier-only configuration using the fine-tuned RoBERTa model will also be evaluated as an additional baseline reference, using classification accuracy, macro-F1, and calibration metrics rather than text generation metrics.

#### Experimental Run Matrix

The proposed comparison will follow this matrix:

| **Model**                                            | **Baseline** | **RAG-enhanced** | **Agent-based** |
| ---------------------------------------------------- | ------------ | ---------------- | --------------- |
| Qwen3-4B-Instruct-2507                               | Yes          | Yes              | Yes             |
| Gemma 3 4B                                           | Yes          | No               | Yes             |
| Llama 3.2 3B                                         | Yes          | No               | Yes             |
| Qwen2.5-1.5B-Instruct                                | Yes          | No               | No              |
| API baseline (supported model selected in Section 5) | Yes          | No               | No              |

This results in nine experimental configurations, plus the RoBERTa classifier-only evaluation.

If computational resources or time become limiting, priority will be given to the default model's three configurations and the classifier-only evaluation. Optional comparison runs and ablation experiments may be reduced.

#### Dataset Separation

To prevent data leakage and preserve evaluation integrity, the datasets will have distinct roles:

| **Dataset Role**           | **Data Source**                                        | **Usage**                                                               |
| -------------------------- | ------------------------------------------------------ | ----------------------------------------------------------------------- |
| Classifier training        | Scruples training split                                | Fine-tuning the RoBERTa classifier.                                     |
| RAG knowledge base         | Reddit AITA dataset, index split                       | Retrieval of similar cases, comments and verdict distributions.         |
| Development and validation | Separate held-out examples                             | Model selection, calibration, threshold tuning and preliminary testing. |
| Golden evaluation          | Manually reviewed Reddit AITA examples, held-out split | Final evaluation of all selected configurations.                        |

The Golden Dataset will remain separate from training and retrieval data. Duplicate and near-duplicate cases will be excluded, including checks against the Scruples training data where applicable.

All thresholds, model configurations and evaluation procedures will be finalized using development data before the Golden Dataset is used.

#### Evaluation Metrics

Evaluation will focus on the following measures. Note that textual explanation metrics (such as perspective coverage and unsupported claims) apply exclusively to generative configurations (Baseline, RAG-enhanced, Agent-based), whereas the classifier-only configuration is evaluated strictly on classification and probability calibration metrics.

| **Metric**            | **Measurement**                                                                                                                                                                     |
| --------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Verdict accuracy      | Percentage of predictions matching the reference verdict on consensus-labeled cases.                                                                                                |
| Macro-F1              | Classification performance across NTA, YTA, ESH and NAH.                                                                                                                            |
| Decision accuracy     | Proportion of correctly executed conditional decisions, such as requesting clarification, retrying retrieval or withholding a verdict, against manually annotated expected actions. |
| Retrieval relevance   | Precision@5 based on manually reviewed retrieved cases, as described in Section 3.9.                                                                                               |
| Perspective coverage  | Whether the generated explanation meaningfully considers the other party's perspective.                                                                                             |
| Unsupported claims    | Number of factual claims in the explanation that are not supported by the submitted scenario or retrieved evidence.                                                                 |
| Sycophancy rate       | Percentage of YTA-labeled cases in which the generated assessment unjustifiably affirms the narrator.                                                                               |
| Uncertainty handling  | Agreement between the selected uncertainty tier and manually annotated expectations, including appropriate withholding and clarification.                                           |
| Narrative consistency | Percentage of cases where equivalent neutral reformulations preserve the verdict and role-swapped versions preserve responsibility attribution.                                     |
| End-to-end latency    | Total time required to produce an assessment, measured per configuration.                                                                                                           |

Because mandatory workflow stages follow a predefined order, decision accuracy will focus on conditional actions rather than treating every tool invocation as an unrestricted selection decision.

#### Human Evaluation Rubric

A sample of approximately 20–30 Golden Dataset stories will be independently evaluated by two team members.

The proposed scoring sheet will contain one row per story and configuration, with separate annotations from each evaluator.

| **Field**                              | **Recorded Value**                                                |
| -------------------------------------- | ----------------------------------------------------------------- |
| Story ID                               | Golden Dataset identifier                                         |
| Configuration                          | Baseline / RAG-enhanced / Agent-based / Classifier-only           |
| Unsupported claims                     | Number of unsupported factual claims identified by each evaluator |
| Other party's perspective covered      | Yes / No, independently recorded by each evaluator                |
| Narrator unjustifiably affirmed        | Yes / No, for YTA-labeled cases                                   |
| Uncertainty appropriately communicated | Yes / No, based on the available evidence and missing information |

The sycophancy rate will be calculated as:

**Sycophancy rate = Number of YTA cases with unjustified narrator affirmation / Total evaluated YTA cases**

Cases with disputed reference labels will be analyzed separately rather than treated as objectively correct or incorrect moral judgments.

Disagreements between evaluators will be documented, and inter-annotator agreement will be reported.

#### Evaluation Procedure

The evaluation will compare the direct baseline, RAG-enhanced generation and the complete agent workflow to determine the contribution of retrieval, alternative perspectives, bias checking and uncertainty handling.

Additional experiments may remove individual components, particularly the bias check or uncertainty handling, to examine their contribution to the final results.

The Golden Dataset will be reserved for final evaluation. Model selection, threshold calibration and preliminary testing will use a separate development set, with all configurations frozen before the final experiments.

Model outputs and evaluation results will be saved to support reproducibility and further analysis.

The primary objective is to determine whether the proposed agent-based architecture can reduce unjustified agreement with the narrator while producing balanced, explainable and uncertainty-aware assessments of interpersonal conflicts.

## 5. Model Choices

All models will run on Google Colab, with Kaggle as a backup. Shared code will be implemented in Python modules in the repository and imported into lightweight notebooks.

| Component | Choice | Rationale |
|---|---|---|
| **Default SLM** | Qwen3-4B-Instruct-2507 | Selected as the initial agent model because of its instruction-following capabilities and relatively small size, making it suitable for a multi-step workflow. |
| **Comparison SLMs** | Gemma 3 4B, Llama 3.2 3B | Allow us to investigate whether different model families exhibit different levels of sycophancy and reasoning consistency. |
| **Verdict classifier** | RoBERTa-base, fine-tuned on Scruples | Produces probabilities for NTA, YTA, ESH, NAH and INFO. INFO is used internally for clarification rather than as a final verdict. Predictions will be calibrated and validated using development data. |
| **Embedding model** | Qwen3-Embedding-0.6B | Generates 1024-dimensional embeddings for semantic retrieval of similar AITA submissions. Its instruction-aware architecture supports task-specific retrieval queries. |
| **Reranking model** | Qwen3-Reranker-0.6B | Reranks retrieved submissions according to their relevance to the user's conflict, improving evidence selection. Retrieval thresholds will be tuned using the development set. |
| **API baseline** | Gemini 3.8 Flash or GPT-4o-mini | Provides an external zero-shot baseline for comparing the proposed local models against a general-purpose LLM. The exact model will be fixed before final evaluation. |
| **Baseline SLM** | Qwen2.5-1.5B-Instruct | Provides a smaller-model baseline for comparing accuracy, sycophancy and inference latency against the 3B–4B models. |

**Selection rule:** We choose the SLM with the lowest sycophancy rate on YTA posts in the dev set. Its accuracy must not be worse than the baseline's, and its typical (median) time per story must stay under about 30 seconds.Qwen3-4B stays unless another model is clearly better. The chosen model is tested once on the golden set. We time a rough version of the pipeline on about 20 stories per model. If it is too slow, we try in order: 4-bit quantization, shorter outputs, merged calls, parallel steps, a smaller model for the framing check, and finally removing a step. 


**Limitations:** small models can be inconsistent (we request JSON output and retry on failure), and RoBERTa truncates posts beyond 512 tokens.

## 6. Project Timeline (Weeks 7–14)

Following the architecture presentation in Weeks 5–6, the project will move into implementation, experimentation, and integration. Development will follow an incremental approach, with the goal of demonstrating measurable progress during the Weeks 8–9 progress check and delivering a complete system with a user interface by Week 14. To keep the Golden Dataset untouched until the final evaluation, all configuration choices and thresholds will be tuned on a separate development set and frozen before the Golden Dataset is used.

| Week | Planned Activities | Expected Deliverables |
|---|---|---|
| **Week 7** | Finalize remaining architectural decisions, including dataset roles, tool schemas, the run matrix, metric definitions and the agent's structured output format. Implement AITA dataset preprocessing, filtering, consensus-based labeling, and duplicate removal. Begin Golden Dataset selection and establish a separate development set. | Reproducible preprocessing pipeline, initial Golden Dataset candidates, finalized technical specifications, and the agent output format. |
| **Week 8** | Implement one-vector-per-post indexing, generate Qwen3-Embedding-0.6B embeddings, and create a preliminary vector index. Begin designing the user interface, including the scenario input form and results layout, using mock agent output. Continue Golden Dataset annotation. Start RoBERTa classifier training, mapping the Scruples labels to the verdict categories (AUTHOR to YTA, OTHER to NTA, EVERYBODY to ESH, NOBODY to NAH, INFO to INFO). | Working semantic retrieval prototype, initial UI design, and documented implementation progress. |
| **Week 9** | Demonstrate preprocessing and retrieval progress at the Weeks 8–9 checkpoint. Establish a direct LLM baseline, begin the baseline SLM comparison, set up the MCP server skeleton, and implement a basic Streamlit interface with scenario submission and preliminary model responses. Run an initial retrieval and reranking latency check. | **Progress check:** functional retrieval demonstration, initial baseline results, basic UI prototype, and GitHub commits. |
| **Week 10** | Complete Golden Dataset annotations. Evaluate the Qwen3 embedding and reranking pipeline using development-set retrieval relevance judgments and different retrieval configurations. Finish and calibrate the RoBERTa classifier and select the default SLM using the selection rule. Extend the MCP server with the `classify_verdict` and `retrieve_similar_cases` tools and connect the UI to them. | Finalized Golden Dataset, selected RAG configuration, calibrated classifier, default SLM, and an MCP server with the first two tools connected to the UI. |
| **Week 11** | Implement the agent's narrative neutralization, alternative-perspective generation, bias checking, and uncertainty handling, together with its decision gates. Expose these tools through the MCP server and integrate them with the existing pipeline. Begin tuning thresholds on the development set. | Functional agent workflow, complete MCP tool set, and preliminary end-to-end integration. |
| **Week 12** | Complete integration between the UI, agent, classifier, and RAG pipeline. Run the three configurations on the development set, finish threshold tuning and freeze all thresholds, and measure latency. Pilot the hand-scoring rubric on a few development stories. | Working end-to-end application with a usable UI, frozen thresholds, initial comparative results, integration tests, and a tested scoring rubric. |
| **Weeks 13–14** | Run the final evaluation on the held-out Golden Dataset at the start of the week, using the frozen model configurations and thresholds. Hand-score a sample of 20–30 stories with two scorers and report their agreement. Perform error analysis and document identified limitations. In parallel, improve UI usability, handle invalid inputs and errors, and fix interface or integration issues that do not alter the evaluated model behavior. Finalize documentation, source code, and the 15-minute project presentation. Rehearse the demonstration and prepare a recorded backup. No new experiments are run in the final week. | Evaluation results, scorer agreement, error analysis, stable application, and polished UI. **Final project presentation:** live application demonstration, documentation, source code, and experimental findings. |

### Planned Experiments

1. **Baseline model comparison (Weeks 9–10):** Evaluate the proposed SLMs on the direct baseline for verdict accuracy, affirmation rate on YTA-labeled cases, and inference latency, and use the results to select the default SLM.

2. **RAG retrieval evaluation:** Evaluate Qwen3-Embedding-0.6B with Qwen3-Reranker-0.6B using different top-k values and retrieval configurations. Measure retrieval quality through Precision@5 and manual relevance judgments on a development-set sample, and tune the retrieval and disagreement thresholds using development data. Verdict-label agreement will serve as a secondary diagnostic.

3. **Classifier-only configuration:** Evaluate the calibrated RoBERTa classifier on its own for accuracy, macro-F1, and calibration.

4. **RAG effectiveness:** Compare direct LLM generation with RAG-enhanced generation to evaluate improvements in reasoning and perspective coverage.

5. **Agent effectiveness:** Compare RAG-enhanced generation against the complete agent workflow, and ablate the bias check and the uncertainty tiers individually, to assess their contributions.

6. **Narrative consistency:** Test whether responsibility is assigned consistently when the underlying facts remain unchanged: the verdict should not change after neutral reformulation, and after a narrator swap it should point to the same party (YTA and NTA exchange; ESH, NAH and INFO stay the same). Report the share of stories that satisfy this mapping.

7. **System performance and usability:** Measure end-to-end response latency and verify that the interface correctly handles user input, model responses, missing information, and execution errors.

Perspective coverage and unsupported claims will be scored by hand on a sample of 20–30 Golden Dataset stories by two scorers, with their agreement reported. Sycophancy will be measured as the affirmation rate on YTA-labeled cases.

The Golden Dataset will be reserved for final evaluation, while separate development examples will be used for model, retrieval, and threshold selection. All model outputs will be cached to files so that metrics can be recomputed without rerunning the models. All implementation work, experiments, and results will be tracked in the shared GitHub repository. Priority will be given to achieving a functional integrated application before introducing optional improvements. If the schedule slips, the optional runs of the comparison SLMs in the agent configuration and the ablations will be cut first.
