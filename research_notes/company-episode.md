# Research notes: earlier Latent Space episode (neural operators) + Accelerated Understanding

Compiled 2026-09-05. All claims cited by URL. Items I could not verify are marked explicitly.

---

## TASK A — Earlier Latent Space episode with Anima Anandkumar

**Identified episode:** "We have foundation models for language, not for physics" — Anima Anandkumar, Bren Professor of Computing.
- YouTube: https://www.youtube.com/watch?v=79mIutht1f4
- Show notes / Substack: https://www.latent.space/p/anima
- Published August 26, 2026 — one week before the September 4, 2026 episode with Benedikt Jenik (KS_IpnX7n9I), matching the "about one week before" clue given in the task. [www.latent.space/p/anima](https://www.latent.space/p/anima)
- Hosts: Brandon (works on RNA therapeutics/AI, Atomic AI) and R.J. Honicky (spatial transcriptomics, CTO/founder of Mirror Omics). [www.latent.space/p/anima](https://www.latent.space/p/anima)
- Note: I could not confirm host R.J.'s last name spelling ("Honiki" as auto-captioned vs. "Honicky") independently — flagging as auto-caption noise, not a verified fact.

**Source used:** yt-dlp pulled English auto-captions successfully (no sign-in wall encountered). Cleaned, word-timestamped transcript (~1h23m runtime) saved at:
`research_notes/earlier_episode_transcript.txt`

Timestamps below are `[HH:MM:SS]` and refer to line-starts in that transcript file, which itself mirrors the source video's timeline.

### How Anima explains neural operators (core mental model)

- Neural operators are presented as a generalization of neural networks: standard nets have fixed-size inputs/outputs (fixed vocabulary for language, fixed resolution for images/video); neural operators instead learn mappings **between function spaces**, so they can take input and produce output "at any resolution." [00:16:12]–[00:17:51]
- Direct quote (paraphrased from captions): "neural operators are... a generalization of neural networks... with standard neural networks the inputs and outputs are a fixed size... whereas with a lot of this physical data the idea is our world is inherently multiscale... [neural operators] enable because they model inputs and outputs as continuous functions that can be infinitely resolved." [00:16:12]–[00:17:51]
- This gives "zero-shot super-resolution" — predicting at resolutions not seen in training — regularized/guided by adding physical constraints (PDE constraints, conservation laws) as extra information at the higher resolution, rather than as hard constraints. [00:18:04]–[00:19:07]

### Physics-informed losses vs. PINNs — why neural operators exist

- She contrasts neural operators against Physics-Informed Neural Networks (PINNs): PINNs try to solve one instance of a PDE from scratch each time via optimization, and that optimization landscape is often intractable, especially for time-dependent/turbulent problems (e.g., fluid dynamics becoming chaotic). "This is not an optimization landscape... we can have any handle on... pins don't work everywhere." [00:11:58]–[00:14:04]
- Neural operators instead train on lots of data (a supervised training phase across many equation instances), and can *still* add physics constraints as guidance at test time — "datadriven and physics informed together." [00:14:55]–[00:15:45]
- Physical constraints (conservation laws, PDE constraints) are added as **loss-function terms**, not hard constraints, because hard constraints are computationally intractable; balancing the physics loss against the data loss is an active design question ("what to impose"). [00:48:18]–[00:49:34]

### Fourier Neural Operator (FNO) — why the Fourier domain

- FNO strikes a tradeoff between efficiency and expressivity: the Fourier domain captures **non-local phenomena** efficiently (differentiation is local, but the inverse/integration is non-local; fluid dynamics, material deformation, quantum chemistry are all non-local). [00:19:19]–[00:21:02]
- Architecturally: think of a transformer but instead of the attention map you have the Fourier transform; nonlinearities, residual connections, and channel-lifting to higher dimensions are retained around the Fourier layers to preserve expressivity. It is not a purely linear Fourier-basis representation — nonlinearity is added between Fourier layers so the model can find "the right latent space," not just the raw Fourier basis. [00:23:35]–[00:25:53]
- Complexity comparison: full attention/transformers at very high resolution would be computationally "untenable" due to quadratic complexity; Fourier transforms give quasi-linear complexity while still modeling global (non-local) connections — "a nice middle ground." [00:21:41]–[00:22:20]
- She connects this to her own history: her undergraduate thesis (20+ years ago) was on the fractional Fourier transform, so she has long-standing interest in classical basis/orthogonal-function techniques even while embracing deep feature learning. [00:27:33]–[00:28:12]

### Scale argument: why not just use a transformer/video model on physical data

- Key quantitative claim: physical-world data is 3D + time (4D), and "industrial scale" starts around ~1,000 grid points per dimension. That implies context lengths of "hundreds of billions to even a trillion" — she says flatly "forget ever having a transformer for anything of this scale — all of the world's compute will not be enough." [01:09:14]–[01:09:54] (quote continues into the "colloccated" compute remark)
- When pushed by a host that video/vision-language models already learn function-like mappings, she draws the resolution distinction explicitly: video/vision models operate at far lower resolution than physical simulation requires, and video generation is autoregressive/next-frame, optimized for "looking good" rather than precise, high-fidelity simulation. Physical problems (fluid dynamics, plasma, material deformation) require retaining fine detail that autoregressive vision/video "codebook" compression would throw away. [01:09:54]–[01:11:47]

### Resolution invariance (the throughline concept)

- Explicit framing: "the real world happens at... infinite resolution and that's what neural operators enable." [00:17:02]–[00:17:39]
- Zero-shot super-resolution and its limits: because the underlying function space is under-constrained, purely resolution-extending predictions risk overfitting/guessing; adding physics constraints at the higher resolution is what makes the higher-resolution answers trustworthy. [00:18:04]–[00:19:07]
- The concept is described as scale-agnostic across domains — she shows a visual (not reproducible in transcript) of phenomena "at different scales from atomic to protein to even planetary scales." [00:46:14]–[00:46:38]

### FourCastNet / spherical FNO / weather

- Origin story: around 2021, her team took on weather forecasting because ERA5 reanalysis weather data (from ECMWF, the European weather agency) was open, despite domain experts telling them AI could not beat decades of physics-based numerical weather prediction. [00:00:00]–[00:00:24], [00:32:38]–[00:33:16]
- Result: the neural-operator-based model was "not only accurate... almost as close to what the traditional weather models can do accurately but also tens of thousands of times faster," runnable on a single consumer-grade GPU rather than a supercomputer. [00:00:49]–[00:01:15], [00:33:41]–[00:34:19]
- They were first to open-source the weather model, FourCastNet, "permissively," which she says let other labs (she names DeepMind, Huawei) build/release their own AI weather models about a year later. This let smaller/global-south weather agencies access forecasting fidelity previously limited to major agencies ("democratizing weather modeling"). [00:34:19]–[00:35:23]
- Architecture evolution: earlier FourCastNet versions used FNOs on a flat/rectangular (e.g., Mercator-like) projection of the globe, which is geometrically wrong and destabilizes long rollouts. **FourCastNet 3** incorporates spherical geometry (spherical harmonics as the natural basis on a sphere — she affirms this explicitly at [00:58:21]–[00:58:47]), which stabilizes long-horizon rollouts and lets a single model span both short-term weather and long-term climate. [00:35:36]–[00:38:30], [00:54:22]–[00:56:38]
- Direct claim: "none of the other architectures work for climate because climate requires us to assume the world is a globe" — a flat-Earth-assumption model "blows up very quickly" on long rollouts; the spherical model degrades much more gracefully (though she concedes it still shows some instability/singularity near the poles). [01:03:11]–[01:03:59]
- The Allen Institute for AI has reportedly built climate models on this neural-operator architecture, which she says is "the only one that works as an AI emulator" for climate. [00:37:40]–[00:37:53]
- Training/data specifics she gives directly:
  - ~50,000 training samples of "fairly high resolution... global weather maps" — she calls this "nothing like what we see with language." [00:28:25]–[00:28:38]
  - Data resolution: ~0.25° (quarter-degree) global grid, which she says comes out to roughly "700 by a few thousand" grid resolution. [01:01:18]–[01:02:21] (she is vague on exact spherical-harmonic mode count — "I forget the details" [01:01:43]–[01:01:56])
  - Training/rollout scheme: the model is trained to predict only 6 hours ahead, with "a little bit of multi-step fine-tuning" for autoregressive rollout — she flags this as "very surprising," i.e., a 6-hour training horizon nonetheless generalizes to stable multi-month rollouts. [01:00:03]–[01:00:40]
  - Ensembles: for probabilistic/extreme-event and climate-scale prediction, they run "a few tens" of ensemble rollouts (each independently enforcing physical constraints, not an averaged/coarse-grained probability). [00:55:50]–[00:58:21]
  - Underlying data type: "reanalysis data" — historical satellite observations assimilated with short-horizon classical physics solvers by weather agencies (she names ECMWF/ERA5). [00:50:24]–[00:51:39]
- Deployment/validation: FourCastNet has been made available via ECMWF (she says launched "fall 2023," "more than 2 years ago" relative to the podcast). She cites Hurricane Lee as a case where FourCastNet predicted landfall timing days earlier than standard forecasting models. [00:51:39]–[00:52:56]
- Extreme events: she says even their first FourCastNet attempt visualized hurricanes/storms well despite skepticism that rare events would be hard for AI — her explanation is that extreme events like hurricanes have "a very specific physical signature," so the physical world may be more sample-efficient than expected because "nature... has a lot of latent space structure." [00:42:41]–[00:44:33]
- Open research question she flags directly: enforcing physical constraints through very long rollouts without either "washing out fine details" (inaccuracy) or keeping unphysical details — called "still an open problem." [00:41:14]–[00:42:03]

### Fusion / plasma disruption

- Digital-twin modeling of tokamak plasma evolution, run "a million times faster" than traditional (presumably MHD-based) simulation, using only "a few thousand samples." [00:43:44]–[00:44:08], [01:04:08]–[01:04:26]
- She confirms (in response to a host's direct question) that the underlying physics is magnetohydrodynamics (MHD) equations. [01:04:39]–[01:04:51]
- "Disruption" is explained as the failure mode where plasma collapses into a thin beam and strikes the containment vessel, potentially damaging the reactor and forcing shutdown — a major bottleneck for sustained fusion. [01:04:51]–[01:05:17]
- Next research step described: coupling simulation with control — adjusting magnetic fields in real time to contain/stabilize a forming disruption — plus joint design+control work. [01:05:30]–[01:05:56]
- Named collaborators: UK Atomic Energy Authority (tokamak work); also working with additional (unnamed) US-based groups and with stellarator designs, explicitly to stay "agnostic" across fusion approaches rather than picking one. [01:05:56]–[01:06:59]

### Other named applications (design / inverse problems)

- CO2 sequestration: modeling underground CO2 plume expansion/pressure buildup and multi-decade migration, faster than traditional simulation. [01:10:36]–[01:11:22] (No catheter-design example appears anywhere in this transcript — see note below.)
- Aerodynamics / arbitrary geometry: modeling car and aircraft aerodynamics by mapping arbitrary shapes into a shared latent geometric space (she uses the "donut" topology joke, explicitly declining the classic donut/coffee-cup pun) so one model generalizes across geometries. [01:11:36]–[01:12:14]
- Inverse design examples: inverse lithography mask design, and gate design for quantum dots (nonlinear photonics) — framed as cases where a forward physics simulator is embedded in an optimization loop to produce novel designs humans struggled to find manually, with the simulator-in-the-loop providing confidence the designs "actually work." [01:16:00]–[01:17:15]
- Multi-physics transfer: she is explicit that a model cannot transfer to physics regimes with literally zero related data/examples, but that coupled-physics fine-tuning (e.g., heat propagation + material stretching → coupled thermal-stretching) needs far fewer samples than training from scratch, because of curriculum-style composition of previously learned sub-phenomena. [01:14:20]–[01:15:48]
- Explicit "foundation model for physics" framing: "we have foundation models for language, maybe vision, but not for physics" — the goal is one broad, multi-physics (coupled-physics) model that can both simulate and do inverse design, replacing the older workflow of human-designed candidates validated by simulation/wind-tunnel testing. [01:12:52]–[01:14:20]

### Explicitly noted difference from transformers/video models (consolidated)

- Quadratic attention cost makes transformers computationally infeasible at the resolution required for physical simulation (hundreds of billions to ~1 trillion effective context at industrial-scale grids). [01:09:14]–[01:09:54]
- Video/vision-language models run at much lower resolution than industrial physical simulation requires and are optimized for visual plausibility/autoregressive next-frame prediction, not high-fidelity precision needed for engineering-grade simulation. [01:09:54]–[01:11:47]

### Other topics covered (not explicitly requested but germane)

- TorchLean: a framework for writing/formalizing neural networks in the Lean proof assistant, aimed at giving formal verification (e.g., certified-robustness bounds like "CROWN," sensitivity analysis, finite-precision-error bounds) for neural nets used in control loops (drones, "a nuclear reactor" is used as the example) — she notes Lean is currently CPU-based with real scalability limitations for very large networks/GPU use. [00:06:30]–[00:10:43]
- UN Scientific Advisory Board: she recently joined it; stated goal is bringing unbiased scientific evidence into AI policy discussions, cautioning that most AI regulatory framing conflates "AI" with language models and misses AI-for-science benefits/considerations. [01:18:06]–[01:20:47]
- Closing "bottleneck" question: her answer is simply "more compute," specifically for research/experimentation capacity (she credits Nvidia in a joking aside). [01:21:00]–[01:21:38]
- Call to action: she points listeners to the open-source **neural operator library**, which she says is already part of the PyTorch ecosystem, with documentation, architectures, examples/recipes, used by researchers and companies. [01:17:41]–[01:18:06], [01:22:16]–[01:22:29]

### Note on "catheter design"
The task brief mentioned catheter design as an expected topic. I searched the full transcript (grep for "cathet") and found **no mention of catheters** anywhere in this episode. It's possible this was conflated with the semiconductor/"chip design" theme that the *later* (KS_IpnX7n9I) episode covers, or with a different Anandkumar talk/paper not part of this podcast. Flagging this as unverified/likely-not-present rather than guessing.

---

## TASK B — Accelerated Understanding (company)

### Official site
- https://acceleratedunderstanding.com/ — tagline: "AI that can simulate and understand physics to invent and discover."
- Company-stated mission: build AI with universal physical understanding to replace physical experiments as the bottleneck in R&D.
- Company-stated technical claims (self-reported, not independently verified):
  - **4D architecture**: models operate directly in 3D space + time simultaneously ("not flattening dimensions").
  - **Scale**: up to 1 trillion parameters at training; 5+ trillion token/context scale claimed at inference.
  - **Resolution invariance**: models can operate at "any detail level" across training and inference.
  - **Multi-physics**: single model trained across diverse physics domains, with claimed "cross-physics uplift" (i.e., a shared multi-physics model outperforming same-size single-domain specialist models).
  - **Self-improvement loop**: uses known physical laws to validate model outputs and drive iterative improvement, providing "directional feedback" (not just pass/fail outcomes) — described as reducing reliance on labeled data alone.
- Site lists Anandkumar's TIME100 Impact Award and a TED talk ("AI that connects the digital and physical worlds") as credentials.
- No funding/investor disclosure found on the official site.
[acceleratedunderstanding.com](https://acceleratedunderstanding.com/)

### Launch / press coverage (2026)
- Launch date: **August 25, 2026**. [KuCoin](https://www.kucoin.com/news/flash/anima-anandkumar-and-benedikt-jenik-launch-ai-startup-with-5-trillion-data-point-model), [vktr.com](https://www.vktr.com/ai-news/the-ai-researchers-who-walked-away-from-bezos-just-launched-a-physics-model-that-ditches-transformer/)
- Co-founder Anima Anandkumar publicly confirmed the launch and framed the model as understanding "the world directly in 4D (3D + time) and across physical phenomena," per her own tweet cited by search results (not independently re-fetched from X due to platform access limits): [x.com/AnimaAnandkumar/status/2092236528898675014](https://x.com/AnimaAnandkumar/status/2092236528898675014)
- Reuters reportedly covered the launch (per Anandkumar's own tweet reference); I did not independently locate/fetch the Reuters article itself, so treat that specific claim as company-sourced, not independently confirmed.
- Central reported technical claim across outlets: the model processed **5 trillion data points in a single prompt** during testing — reported by multiple outlets but **not independently verified/benchmarked** by any of them. [KuCoin](https://www.kucoin.com/news/flash/anima-anandkumar-and-benedikt-jenik-launch-ai-startup-with-5-trillion-data-point-model), [vktr.com](https://www.vktr.com/ai-news/the-ai-researchers-who-walked-away-from-bezos-just-launched-a-physics-model-that-ditches-transformer/)
- The architecture is explicitly reported as **non-Transformer**, built on neural operators (the framework Anandkumar helped pioneer academically). [KuCoin](https://www.kucoin.com/news/flash/anima-anandkumar-and-benedikt-jenik-launch-ai-startup-with-5-trillion-data-point-model)
- A Tech Times headline explicitly frames this as "**No Benchmark Proof Yet**" (I could not fetch the full Tech Times article body — blocked with HTTP 403 — but the headline itself is a direct, citable skepticism signal). [techtimes.com](https://www.techtimes.com/articles/325647/20260826/caltech-startup-unveils-physics-ai-that-skips-transformers-no-benchmark-proof-yet.htm)
- vktr.com explicitly notes: no independent verification of the 5-trillion-data-point claim, no third-party benchmarks, and frames the real open question as commercial ("whether enterprises will pay for physics simulation AI at this scale, when the market is overwhelmingly focused on language and code"). [vktr.com](https://www.vktr.com/ai-news/the-ai-researchers-who-walked-away-from-bezos-just-launched-a-physics-model-that-ditches-transformer/)

### Target markets (as stated by company/press)
Multiple outlets converge on four application areas:
1. **Chip design / semiconductors** — using physics modeling for chip performance under heat/materials constraints. [analyticsinsight.net](https://www.analyticsinsight.net/news/ai-researchers-walk-away-from-bezos-backed-prometheus-to-build-physical-world-ai), [vktr.com](https://www.vktr.com/ai-news/the-ai-researchers-who-walked-away-from-bezos-just-launched-a-physics-model-that-ditches-transformer/)
2. **Energy** — split into (a) designing physical objects with specific performance capabilities (nuclear and other generation tech) and (b) interpreting observational data (geothermal resource location, critical-mineral discovery for electronics supply chains). [WebSearch synthesis citing vktr.com/analyticsinsight.net reporting]
3. **Robotics** — spatial/physical object-interaction understanding. [analyticsinsight.net](https://www.analyticsinsight.net/news/ai-researchers-walk-away-from-bezos-backed-prometheus-to-build-physical-world-ai)
4. **Weather / extreme-event prediction**. [analyticsinsight.net](https://www.analyticsinsight.net/news/ai-researchers-walk-away-from-bezos-backed-prometheus-to-build-physical-world-ai)

Geothermal is specifically named as an energy sub-application (resource location), not a standalone market. [vktr.com](https://www.vktr.com/ai-news/the-ai-researchers-who-walked-away-from-bezos-just-launched-a-physics-model-that-ditches-transformer/)

### Founding, funding, investors
- **Founding date:** August 25, 2026 (launch date; could not independently confirm this equals legal incorporation date — treating "founded"/"launched" as reported together). [KuCoin](https://www.kucoin.com/news/flash/anima-anandkumar-and-benedikt-jenik-launch-ai-startup-with-5-trillion-data-point-model)
- **Funding/investors:** **Not disclosed** in any source checked (official site, KuCoin, vktr.com, analyticsinsight.net). vktr.com explicitly notes the company's valuation is undisclosed. No named investors found anywhere in this research. This should be treated as an open/unverified item, not "no funding."
- **The Prometheus decision:** Anandkumar and Jenik were offered a combined **35% equity stake** in Project Prometheus (the Jeff Bezos-backed AI venture that raised **$12 billion** in June 2026, reportedly valuing Prometheus around $41 billion per one source), a combined salary package reportedly rising to **$2 million**, and **more than $2 billion in committed financing** (one source specifies "through Series B"). They declined this and built Accelerated Understanding independently instead. [KuCoin](https://www.kucoin.com/news/flash/anima-anandkumar-and-benedikt-jenik-launch-ai-startup-with-5-trillion-data-point-model), [analyticsinsight.net](https://www.analyticsinsight.net/news/ai-researchers-walk-away-from-bezos-backed-prometheus-to-build-physical-world-ai), [vktr.com](https://www.vktr.com/ai-news/the-ai-researchers-who-walked-away-from-bezos-just-launched-a-physics-model-that-ditches-transformer/)
- I attempted to fetch Rappler's founder profile piece (which appeared to have the most depth on this decision) but it returned HTTP 403 Forbidden and could not be read directly — the KuCoin/analyticsinsight/vktr figures above are corroborating but derivative secondary coverage, not the original Rappler reporting.

### Team
- **Anima Anandkumar**: Bren Professor of Computing (Caltech), formerly Senior Director of AI Research at NVIDIA (also described elsewhere as having "led AI research" at NVIDIA — see her own words in the earlier podcast episode, [00:02:43]) and, before that, helped found the cloud AI team at Amazon Web Services. TIME100 Impact Award recipient; gave a TED talk on AI connecting digital and physical worlds; recently joined the UN Scientific Advisory Board (per her own statement in the earlier episode, [01:18:06]). [acceleratedunderstanding.com](https://acceleratedunderstanding.com/), episode self-report
- **Benedikt Jenik**: co-founder; multiple secondary sources describe him only vaguely as "an AI infrastructure engineer" with a background in "large-scale ML systems and self-driving cars," and note he is Anandkumar's spouse. [vktr.com](https://www.vktr.com/ai-news/the-ai-researchers-who-walked-away-from-bezos-just-launched-a-physics-model-that-ditches-transformer/), [acceleratedunderstanding.com](https://acceleratedunderstanding.com/)
  - Independent search turned up a **Benedikt Jenik** with an MIT affiliation: teaching assistant for MIT's "Deep Learning for Self-Driving Cars" course, and research tied to the MIT Advanced Vehicle Technology (MIT-AVT) study on large-scale naturalistic driving data and deep-learning perception systems (Google Scholar profile cited by search, 527 citations reported). **I could not confirm this MIT-affiliated Benedikt Jenik is the same person as the Accelerated Understanding co-founder** — the name match is suggestive (fits the "self-driving cars" background mentioned by press) but I found no source directly linking the MIT researcher's identity to the company bio. Treat as a plausible but unverified match. [Google Scholar](https://scholar.google.com/citations?user=UaeunUcAAAAJ&hl=en), [DeepAI profile](https://deepai.org/profile/benedikt-jenik) (also unverified as same individual)

### The KS_IpnX7n9I episode (company-focused Latent Space episode)
- Title: **"Trillion Token Context. No, Really" / "Faster Chips That Don't Melt"** — Anima Anandkumar & Benedikt Jenik, Accelerated Understanding. (Two slightly different titles surfaced across sources — YouTube page title vs. BigGo Finance's rendering — both refer to the same episode.) [YouTube](https://www.youtube.com/watch?v=KS_IpnX7n9I), [BigGo Finance](https://finance.biggo.com/podcast/111670bc22657d84)
- Published: **September 4, 2026**; duration 26m56s; hosts R.J. and Brandon (same hosts as the earlier episode). [BigGo Finance](https://finance.biggo.com/podcast/111670bc22657d84)
- I could not retrieve a full YouTube description/show-notes text for this video directly — YouTube's page returned only boilerplate footer content via fetch, not the actual description or chapters. The BigGo Finance podcast page (a secondary aggregator, not Latent Space's own site) is the best secondary summary I found; I could not locate a dedicated latent.space Substack post for this specific episode (only the earlier one at /p/anima was found).
- Reported technical content from that episode (relayed here for reference, since it's germane to why Task A was requested, but sourced from secondary coverage, not primary transcript, since Task A's brief was to focus on the earlier episode):
  - Emphasis on **resolution invariance** as enabling flexible context lengths at inference, contrasted with video models needing fixed resolution.
  - Trillion-parameter training, up to 1 trillion training tokens/context, ~5 trillion inference context, across 4D (3D + time).
  - Core claimed finding: "the shared model wins" — a multi-physics model beats equal-parameter single-domain specialist models.
  - Named example domains for "diverse regimes sharing underlying structure": catheter fluid dynamics, rocket nozzles, fusion reactors — though the source notes actual proprietary training domains were not disclosed in detail.
  - Anandkumar quoted (via secondary source) describing the self-improvement mechanism as providing "dense feedback because the physics laws" enable training signal beyond supervised learning alone.
  - **This is where the "catheter design" reference in the task brief actually originates** — it is from this later (KS_IpnX7n9I) episode's reported content, not from the earlier (79mIutht1f4) episode's transcript, which contains no catheter mention at all. [BigGo Finance](https://finance.biggo.com/podcast/111670bc22657d84)

---

## Source list (all URLs cited above)
- https://www.latent.space/p/anima (earlier episode show notes)
- https://www.youtube.com/watch?v=79mIutht1f4 (earlier episode video)
- https://www.youtube.com/watch?v=KS_IpnX7n9I (company-focused episode video)
- https://finance.biggo.com/podcast/111670bc22657d84 (secondary summary of KS_IpnX7n9I episode)
- https://acceleratedunderstanding.com/ (official company site)
- https://www.kucoin.com/news/flash/anima-anandkumar-and-benedikt-jenik-launch-ai-startup-with-5-trillion-data-point-model
- https://www.analyticsinsight.net/news/ai-researchers-walk-away-from-bezos-backed-prometheus-to-build-physical-world-ai
- https://www.vktr.com/ai-news/the-ai-researchers-who-walked-away-from-bezos-just-launched-a-physics-model-that-ditches-transformer/
- https://www.techtimes.com/articles/325647/20260826/caltech-startup-unveils-physics-ai-that-skips-transformers-no-benchmark-proof-yet.htm (headline only — body blocked, HTTP 403)
- https://www.rappler.com/technology/accelerated-understanding-anima-anandkumar-benedikt-jenik-profile/ (blocked, HTTP 403 — not read)
- https://x.com/AnimaAnandkumar/status/2092236528898675014 (referenced via search snippet only, not independently fetched)
- https://x.com/AndrewCurran_/status/2092244031002771643 (referenced via search snippet only, not independently fetched)
- https://scholar.google.com/citations?user=UaeunUcAAAAJ&hl=en (possible/unverified match for Benedikt Jenik)
- https://deepai.org/profile/benedikt-jenik (possible/unverified match for Benedikt Jenik)

## Not verified / could not access
- Rappler's full founder-profile article (403 Forbidden)
- Tech Times' full article body (403 Forbidden)
- Any funding amount, investor names, or formal incorporation date for Accelerated Understanding
- Whether the MIT-affiliated "Benedikt Jenik" (self-driving car perception research) is the same person as the co-founder
- Exact spherical-harmonic mode count used in FourCastNet 3 (Anandkumar herself says "I forget the details" on the podcast, [01:01:43])
- Full YouTube description/chapters for the KS_IpnX7n9I episode (fetch returned only page boilerplate, not the actual description)
