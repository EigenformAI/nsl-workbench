---------------------------- MODULE LearningLab ----------------------------
(***************************************************************************)
(* Two concurrent episode slots on each generation floor. Each episode runs *)
(* Explore -> Hypothesise -> Code -> Verify -> Rewrite in order, then makes *)
(* one durable commit. A signal-passing rewrite adds training rows to this  *)
(* floor's tray. An archive-admitted one also adds a persistent shared fact *)
(* whose parents are the facts the agent read when its episode started.     *)
(* The floor publishes only when both slots have settled. Training reloads  *)
(* the fixed configured base on the published rows, and only a validated    *)
(* candidate opens the next floor. The final floor publishes and stops.     *)
(*                                                                         *)
(* Bounds: two slots, three floors, one episode per slot per floor. Fixed   *)
(* scenarios choose every outcome in advance; the free scenarios let each   *)
(* outcome vary. No time, crashes, deadlines, recovery or model quality.    *)
(***************************************************************************)
EXTENDS Integers, FiniteSets

CONSTANT Scenarios

Agents == {"A", "B"}
Generations == 3
LastFloor == Generations - 1
Floors == 0..LastFloor
Episodes == 0..1
EpisodeOf(a) == IF a = "A" THEN 0 ELSE 1
Stages == {"idle", "explore", "hypothesise", "code", "verify", "rewrite"}
NextStage(s) == IF s = "explore" THEN "hypothesise"
                ELSE IF s = "hypothesise" THEN "code"
                ELSE "verify"
Phases == {"collecting", "published", "training", "candidate", "validated",
           "complete", "trainer-failed", "artifact-rejected"}
Terminal == {"complete", "trainer-failed", "artifact-rejected"}
Committed == {"rows", "rows+fact"}

FixedScenarios == {"materials", "no-archive", "trainer-failure", "invalid-artifact"}
FreeScenarios == {"free-outcomes", "free-outcomes-no-archive"}
Graph(s) == s \notin {"no-archive", "free-outcomes-no-archive"}

\* Fixed episode outcomes, by floor g and episode k (k = 0 is slot A).
Signal(g, k) == ~(g = 1 /\ k = 0)
Rows(g, k) == IF g = 1 /\ k = 0 THEN 0
              ELSE IF (g = 0 /\ k = 0) \/ (g = 2 /\ k = 1) THEN 2
              ELSE 1
Admit(g, k) == (g = 0 /\ k = 0) \/ (g = 1 /\ k = 1) \/ (g = 2 /\ k = 0)
Training(s, g) == IF s = "trainer-failure" /\ g = 0 THEN "trainer-error"
                  ELSE IF s = "invalid-artifact" /\ g = 1 THEN "invalid"
                  ELSE "valid"

VerifyChoices(s, g, k) == IF s \in FreeScenarios THEN BOOLEAN ELSE {Signal(g, k)}
CommitChoices(s, g, k) ==
  IF s \in FreeScenarios
  THEN [rows: {1, 2}, admit: IF Graph(s) THEN BOOLEAN ELSE {FALSE}]
  ELSE {[rows |-> Rows(g, k), admit |-> Admit(g, k) /\ Graph(s)]}
TrainerChoices(s, g) ==
  IF s \in FreeScenarios THEN {"candidate", "trainer-error"}
  ELSE IF Training(s, g) = "trainer-error" THEN {"trainer-error"} ELSE {"candidate"}
ValidationChoices(s, g) ==
  IF s \in FreeScenarios THEN {"valid", "invalid"}
  ELSE IF Training(s, g) = "invalid" THEN {"invalid"} ELSE {"valid"}

VARIABLES
  scn,        \* scenario, fixed for the whole run
  gen,        \* current floor (generation)
  phase,      \* floor phase
  stage,      \* stage of each slot's episode
  ep,         \* episode index each slot runs, -1 when idle
  ctx,        \* fact ids a running agent read at episode start
  facts,      \* persistent shared knowledge: [id |-> <<floor, episode>>, parents |-> ids]
  episodes,   \* per floor, per episode: [status, rows, verdict]
  rowCount,   \* per floor: rows in the tray
  published,  \* per floor
  serving,    \* per floor: -2 not open, -1 configured base, n adapter trained on floor n
  training,   \* per floor: none, training, candidate, valid, invalid, trainer-error
  trainBase,  \* per floor: -2 no candidate, -1 configured base, n adapter of floor n
  trainRows   \* per floor: episodes whose rows the candidate used

vars == <<scn, gen, phase, stage, ep, ctx, facts, episodes, rowCount, published,
          serving, training, trainBase, trainRows>>
historyVars == <<facts, episodes, rowCount, published, serving, training, trainBase, trainRows>>

FactIds == {f.id : f \in facts}
Idle == [a \in Agents |-> "idle"]

Init ==
  /\ scn \in Scenarios
  /\ gen = 0
  /\ phase = "collecting"
  /\ stage = Idle
  /\ ep = [a \in Agents |-> -1]
  /\ ctx = [a \in Agents |-> {}]
  /\ facts = {}
  /\ episodes = [g \in Floors |-> [k \in Episodes |-> [status |-> "unissued", rows |-> 0, verdict |-> "pending"]]]
  /\ rowCount = [g \in Floors |-> 0]
  /\ published = [g \in Floors |-> FALSE]
  /\ serving = [g \in Floors |-> IF g = 0 THEN -1 ELSE -2]
  /\ training = [g \in Floors |-> "none"]
  /\ trainBase = [g \in Floors |-> -2]
  /\ trainRows = [g \in Floors |-> {}]

\* An idle slot starts its one episode for this floor and snapshots the
\* committed facts. It does not see facts committed later.
StartEpisode(a) ==
  /\ phase = "collecting"
  /\ stage[a] = "idle"
  /\ episodes[gen][EpisodeOf(a)].status = "unissued"
  /\ stage' = [stage EXCEPT ![a] = "explore"]
  /\ ep' = [ep EXCEPT ![a] = EpisodeOf(a)]
  /\ ctx' = [ctx EXCEPT ![a] = FactIds]
  /\ episodes' = [episodes EXCEPT ![gen][EpisodeOf(a)].status = "running"]
  /\ UNCHANGED <<scn, gen, phase, facts, rowCount, published, serving, training, trainBase, trainRows>>

Advance(a) ==
  /\ phase = "collecting"
  /\ stage[a] \in {"explore", "hypothesise", "code"}
  /\ stage' = [stage EXCEPT ![a] = NextStage(stage[a])]
  /\ UNCHANGED <<scn, gen, phase, ep, ctx, historyVars>>

VerifyPasses(a) ==
  /\ phase = "collecting"
  /\ stage[a] = "verify"
  /\ TRUE \in VerifyChoices(scn, gen, ep[a])
  /\ stage' = [stage EXCEPT ![a] = "rewrite"]
  /\ episodes' = [episodes EXCEPT ![gen][ep[a]].verdict = "pass"]
  /\ UNCHANGED <<scn, gen, phase, ep, ctx, facts, rowCount, published, serving, training, trainBase, trainRows>>

VerifyFails(a) ==
  /\ phase = "collecting"
  /\ stage[a] = "verify"
  /\ FALSE \in VerifyChoices(scn, gen, ep[a])
  /\ stage' = [stage EXCEPT ![a] = "idle"]
  /\ ep' = [ep EXCEPT ![a] = -1]
  /\ ctx' = [ctx EXCEPT ![a] = {}]
  /\ episodes' = [episodes EXCEPT ![gen][ep[a]] = [status |-> "no-rows", rows |-> 0, verdict |-> "fail"]]
  /\ UNCHANGED <<scn, gen, phase, facts, rowCount, published, serving, training, trainBase, trainRows>>

\* Durable commit: rows enter the tray; an admitted fact enters shared knowledge.
CommitRewrite(a) ==
  /\ phase = "collecting"
  /\ stage[a] = "rewrite"
  /\ \E c \in CommitChoices(scn, gen, ep[a]) :
       /\ episodes' = [episodes EXCEPT ![gen][ep[a]] =
                         [status |-> IF c.admit THEN "rows+fact" ELSE "rows", rows |-> c.rows, verdict |-> "pass"]]
       /\ rowCount' = [rowCount EXCEPT ![gen] = @ + c.rows]
       /\ facts' = IF c.admit THEN facts \cup {[id |-> <<gen, ep[a]>>, parents |-> ctx[a]]} ELSE facts
  /\ stage' = [stage EXCEPT ![a] = "idle"]
  /\ ep' = [ep EXCEPT ![a] = -1]
  /\ ctx' = [ctx EXCEPT ![a] = {}]
  /\ UNCHANGED <<scn, gen, phase, published, serving, training, trainBase, trainRows>>

\* Generation barrier: every slot has run its episode and settled.
Publish ==
  /\ phase = "collecting"
  /\ stage = Idle
  /\ \A k \in Episodes : episodes[gen][k].status \notin {"unissued", "running"}
  /\ published' = [published EXCEPT ![gen] = TRUE]
  /\ phase' = IF gen = LastFloor THEN "complete" ELSE "published"
  /\ UNCHANGED <<scn, gen, stage, ep, ctx, facts, episodes, rowCount, serving, training, trainBase, trainRows>>

StartTraining ==
  /\ phase = "published"
  /\ gen < LastFloor
  /\ phase' = "training"
  /\ training' = [training EXCEPT ![gen] = "training"]
  /\ UNCHANGED <<scn, gen, stage, ep, ctx, facts, episodes, rowCount, published, serving, trainBase, trainRows>>

\* The trainer loads the fixed configured base, never the serving adapter.
TrainerProduces ==
  /\ phase = "training"
  /\ "candidate" \in TrainerChoices(scn, gen)
  /\ phase' = "candidate"
  /\ training' = [training EXCEPT ![gen] = "candidate"]
  /\ trainBase' = [trainBase EXCEPT ![gen] = -1]
  /\ trainRows' = [trainRows EXCEPT ![gen] = {k \in Episodes : episodes[gen][k].rows > 0}]
  /\ UNCHANGED <<scn, gen, stage, ep, ctx, facts, episodes, rowCount, published, serving>>

TrainerFails ==
  /\ phase = "training"
  /\ "trainer-error" \in TrainerChoices(scn, gen)
  /\ phase' = "trainer-failed"
  /\ training' = [training EXCEPT ![gen] = "trainer-error"]
  /\ UNCHANGED <<scn, gen, stage, ep, ctx, facts, episodes, rowCount, published, serving, trainBase, trainRows>>

\* Validation checks artifact format, identity and configuration, not quality.
ArtifactValid ==
  /\ phase = "candidate"
  /\ "valid" \in ValidationChoices(scn, gen)
  /\ phase' = "validated"
  /\ training' = [training EXCEPT ![gen] = "valid"]
  /\ UNCHANGED <<scn, gen, stage, ep, ctx, facts, episodes, rowCount, published, serving, trainBase, trainRows>>

ArtifactInvalid ==
  /\ phase = "candidate"
  /\ "invalid" \in ValidationChoices(scn, gen)
  /\ phase' = "artifact-rejected"
  /\ training' = [training EXCEPT ![gen] = "invalid"]
  /\ UNCHANGED <<scn, gen, stage, ep, ctx, facts, episodes, rowCount, published, serving, trainBase, trainRows>>

ServeNextFloor ==
  /\ phase = "validated"
  /\ gen' = gen + 1
  /\ phase' = "collecting"
  /\ serving' = [serving EXCEPT ![gen + 1] = gen]
  /\ UNCHANGED <<scn, stage, ep, ctx, facts, episodes, rowCount, published, training, trainBase, trainRows>>

\* End states change nothing.
Finished == phase \in Terminal /\ UNCHANGED vars

Next ==
  \/ \E a \in Agents : StartEpisode(a) \/ Advance(a) \/ VerifyPasses(a) \/ VerifyFails(a) \/ CommitRewrite(a)
  \/ Publish \/ StartTraining \/ TrainerProduces \/ TrainerFails
  \/ ArtifactValid \/ ArtifactInvalid \/ ServeNextFloor \/ Finished

Spec == Init /\ [][Next]_vars

-----------------------------------------------------------------------------
\* Rules

EpisodeRecord == [status: {"unissued", "running", "no-rows", "rows", "rows+fact"},
                  rows: 0..2, verdict: {"pending", "pass", "fail"}]
FactId == Floors \X Episodes

TypeOK ==
  /\ scn \in FixedScenarios \cup FreeScenarios
  /\ gen \in Floors
  /\ phase \in Phases
  /\ stage \in [Agents -> Stages]
  /\ ep \in [Agents -> {-1} \cup Episodes]
  /\ ctx \in [Agents -> SUBSET FactId]
  /\ facts \subseteq [id: FactId, parents: SUBSET FactId]
  /\ episodes \in [Floors -> [Episodes -> EpisodeRecord]]
  /\ rowCount \in [Floors -> 0..4]
  /\ published \in [Floors -> BOOLEAN]
  /\ serving \in [Floors -> -2..(LastFloor - 1)]
  /\ training \in [Floors -> {"none", "training", "candidate", "valid", "invalid", "trainer-error"}]
  /\ trainBase \in [Floors -> -2..(LastFloor - 1)]
  /\ trainRows \in [Floors -> SUBSET Episodes]

\* Rows come only from committed, signal-passing rewrites.
RowsOnlyFromEligibleRewrite ==
  \A g \in Floors, k \in Episodes :
    /\ episodes[g][k].rows > 0 => (episodes[g][k].status \in Committed /\ episodes[g][k].verdict = "pass")
    /\ (scn \in FixedScenarios /\ episodes[g][k].status \in Committed) => Signal(g, k)

\* Every settled eligible rewrite keeps its rows, admitted or not.
EligibleRowsKept ==
  \A g \in Floors, k \in Episodes :
    /\ (episodes[g][k].verdict = "pass" /\ episodes[g][k].status # "running") => episodes[g][k].status \in Committed
    /\ episodes[g][k].status \in Committed => episodes[g][k].rows >= 1
    /\ (scn \in FixedScenarios /\ episodes[g][k].status \in Committed) => episodes[g][k].rows = Rows(g, k)

\* A shared fact exists exactly for committed, admitted episodes.
FactsOnlyAfterCommit ==
  /\ \A f \in facts : episodes[f.id[1]][f.id[2]].status = "rows+fact"
  /\ \A g \in Floors, k \in Episodes : episodes[g][k].status = "rows+fact" => <<g, k>> \in FactIds
  /\ \A a \in Agents : stage[a] # "idle" => <<gen, ep[a]>> \notin FactIds
  /\ \A f \in facts : scn \in FixedScenarios => Admit(f.id[1], f.id[2])

\* No episode contributes twice: tray counts match per-episode rows, and each
\* fact id has one record.
NoDoubleContribution ==
  /\ \A g \in Floors : rowCount[g] = episodes[g][0].rows + episodes[g][1].rows
  /\ Cardinality(facts) = Cardinality(FactIds)
  /\ \A a \in Agents : stage[a] # "idle" => (ep[a] = EpisodeOf(a) /\ episodes[gen][ep[a]].status = "running")

NoGraphWithoutArchive == ~Graph(scn) => facts = {}

\* Agents read only committed facts, and a fact's parents were committed earlier.
ContextFromCommittedFacts ==
  /\ \A a \in Agents : ctx[a] \subseteq FactIds
  /\ \A a \in Agents : stage[a] = "idle" => (ctx[a] = {} /\ ep[a] = -1)
  /\ \A f \in facts : f.parents \subseteq FactIds \ {f.id}

\* The generation barrier.
NoPublishWhileActive ==
  \A g \in Floors : published[g] =>
    /\ \A k \in Episodes : episodes[g][k].status \notin {"unissued", "running"}
    /\ (g = gen => stage = Idle)

NoTrainingBeforePublish == \A g \in Floors : training[g] # "none" => published[g]

FixedTrainingBase ==
  \A g \in Floors :
    /\ trainBase[g] \in {-2, -1}
    /\ training[g] \in {"candidate", "valid", "invalid"} => trainBase[g] = -1

TrainsOnPublishedRows ==
  \A g \in Floors : trainBase[g] # -2 =>
    /\ published[g]
    /\ trainRows[g] = {k \in Episodes : episodes[g][k].rows > 0}

\* Floors open one at a time, each above the first serving a validated candidate.
ServeOnlyValidated ==
  /\ serving[0] = -1
  /\ \A g \in Floors : (serving[g] = -2) <=> (g > gen)
  /\ \A g \in 1..LastFloor : serving[g] # -2 => (serving[g] = g - 1 /\ training[g - 1] = "valid")

NoFinalTraining == training[LastFloor] = "none" /\ trainBase[LastFloor] = -2
=============================================================================
