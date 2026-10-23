# Am I The Asshole? — Request for Comments (RFC)

## 1. Problem Definition

* **Context:** Interpersonal conflicts in everyday life (family, friends, partners,
  roommates, coworkers), where people want an outside opinion on whether they
  acted wrongly. Today they get one in two main ways: posting the story publicly on
  communities like Reddit's r/AmItheAsshole and waiting for the crowd to vote,
  or asking a general-purpose AI chatbot, which has become a popular shortcut
  (a growing number of "AI judge" apps are built on this idea) [8][9][10].

* **The Problem:** Both options are flawed. Public forums are slow, expose personal
  details, and give inconsistent answers depending on who replies first. Chatbots are
  instant and private, but they are sycophantic: on AITA posts where the crowd
  judged the poster to be in the wrong, leading LLMs still affirmed the user's
  behavior in about 42% of cases on average [1], and a later
  Science study found AI systems affirmed users in 51% of such cases, versus 0%
  for humans [3]. This is hard to fix with traditional software because the input is
  a subjective, one-sided narrative with no rules to check against, and verdicts
  are often genuinely divisive: even the crowd disagrees on many cases [4],
  so a single confident YES/NO is itself misleading.

* **Impact & Audience:** Anyone seeking advice on a personal conflict, especially
  people who treat the AI's answer as a neutral referee. Sycophantic advice has
  measurable costs: in controlled experiments it made users more convinced they
  were right and less willing to repair the relationship, while increasing their
  trust in and reliance on the AI [2]. Users also tend to prefer
  the flattering answers [2][11], so existing products have little incentive to fix this.
  The result is worse decisions in real relationships, delivered with false
  certainty.

## 2. State of the Art

* **Existing Solutions:** (1) General LLM chatbots (ChatGPT/Claude) prompted ad hoc;
  (2) Consumer "AI judge" apps (Moral Compass, Am I Wrong?, The Docket) that wrap an LLM
  with a persona and a verdict card [8][9][10]; (3) Research classifiers fine-tuned on r/AITA
  (RoBERTa/Longformer [5]; Scruples [4]).
* **Pros:**
  * Fluent, instant, zero-friction input (just paste your story).
  * Fine-tuned classifiers reach ~87% accuracy at predicting the crowd verdict [5].
  * Persona/shareable-verdict formats drive engagement [9][10].
* **Cons:**
  * Sycophancy: LLMs affirm behavior the crowd judged inappropriate in ~42% of AITA
    cases [1], and affirm users 49% more than humans [3].
    Users prefer sycophantic answers and become less willing to repair relationships [2][11].
  * Single-narrator bias: the only input is the asker's own framing.
  * False certainty: crowd judgments are often divisive [4], yet apps output one verdict.
  * Black-box outputs with no explanation or comparable cases (a limitation acknowledged
    by the Delphi authors) [6][7].
  * Privacy/monetization: sensitive stories sent to third-party free-tier APIs;
    pay-to-win verdict features undermine trust [10].

## 3. Proposal

* **Our Solution:** An AITA-judging app that reframes the user's story neutrally,
  generates the other party's strongest case, and returns a probability distribution
  over verdicts (NTA/YTA/ESH/NAH) grounded in retrieved, similar resolved AITA cases.
* **What is Different:** Anti-sycophancy is a measured design goal, not an afterthought:
  neutral third-person rewrite, forced counter-perspective, and a calibration check
  against a fine-tuned classifier, evaluated on YTA-labeled AITA posts [1][2].
* **What Makes it Better:** Lower false-negative rate than prompting a general LLM
  (target to be measured with the ELEPHANT benchmark [1]), calibrated uncertainty instead of
  fake certainty, explanations via similar cases, and a no-retention privacy mode.

---

## References

### Cited above

* [1] M. Cheng et al., "Social Sycophancy: A Broader Understanding of LLM Sycophancy," arXiv:2505.13995, 2025. https://arxiv.org/abs/2505.13995v1 (ELEPHANT code & data: https://github.com/myracheng/elephant)
* [2] M. Cheng et al., "Sycophantic AI Decreases Prosocial Intentions and Promotes Dependence," arXiv:2510.01395, 2025 (published in Science, 2026). https://arxiv.org/pdf/2510.01395
* [3] Neuroscience News, summary of the Science study (source of the 49% / 51% vs 0% figures). https://neurosciencenews.com/ai-sycophancy-moral-judgment-30397/
* [4] N. Lourie et al., "Scruples: A Corpus of Community Ethical Judgments on 32,000 Real-Life Anecdotes," AAAI 2021, arXiv:2008.09094. https://arxiv.org/pdf/2008.09094 (DOI: 10.1609/aaai.v35i15.17589; data/code: https://github.com/allenai/scruples)
* [5] A. Alhassan, J. Zhang, V. Schlegel, "'Am I the Bad One'? Predicting the Moral Judgement of the Crowd Using Pre-trained Language Models," LREC 2022. https://aclanthology.org/2022.lrec-1.28/
* [6] L. Jiang et al., Delphi / Commonsense Norm Bank, arXiv:2110.07574, 2021. https://arxiv.org/pdf/2110.07574v1.pdf
* [7] IEEE Spectrum, coverage of Delphi (including the authors' note that its reasoning is hard to explain). https://spectrum.ieee.org/ai-ethics-machines-learn-good
* [8] "Moral Compass - AITA Community," Apple App Store listing. https://apps.apple.com/us/app/-/id6748293841
* [9] "Am I Wrong? AI Judge," AppGoblin listing. https://appgoblin.info/apps/6768784454
* [10] "The Docket," Product Hunt. https://www.producthunt.com/p/the-docket
* [11] Let's Data Science, coverage of the Stanford/Science sycophancy study (reporting AP and NPR). https://letsdatascience.com/news/study-finds-ai-chatbots-favor-flattery-over-facts-a2ef7288
