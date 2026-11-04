# LearningLab

TLC reached 477 states and 783 steps. The chart draws 20 states and 21 arrows.
The view hides 4 steps, so the chart does not draw them.
The step starts an episode, reads committed facts changes only one of Agent A and Agent B at a time.
The step finishes exploring changes only one of Agent A and Agent B at a time.
The step writes a hypothesis changes only one of Agent A and Agent B at a time.
The step submits descriptor code changes only one of Agent A and Agent B at a time.
The step signal check passes changes only one of Agent A and Agent B at a time.
The step commits rows durably, plus a fact if admitted changes only one of Agent A and Agent B at a time.

## Rules that hold
TypeOK, RowsOnlyFromEligibleRewrite, EligibleRowsKept, FactsOnlyAfterCommit, NoDoubleContribution, NoGraphWithoutArchive, ContextFromCommittedFacts, NoPublishWhileActive, NoTrainingBeforePublish, FixedTrainingBase, TrainsOnPublishedRows, ServeOnlyValidated, NoFinalTraining, NoDeadlock
