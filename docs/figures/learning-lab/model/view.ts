// Chart labels for LearningLab.tla. The region labels come from the process
// module, so the chart and the JavaScript snapshot share one vocabulary.
import { chartRegionsFromModel } from "../lab-process.js";

export const title = "Learning lab: two episode agents per floor, then a checked handoff";

export const state = (v) => chartRegionsFromModel(v);

const handoff = ["published", "training", "candidate", "validated"];
export const group = (v) =>
  v.phase === "collecting"
    ? "lab"
    : handoff.includes(v.phase)
      ? "handoff"
      : v.phase === "complete"
        ? "done"
        : "stopped";
export const colors = { lab: "teal", handoff: "yellow", done: "blue", stopped: "pink" };

const finished = { explore: "finishes exploring", hypothesise: "writes a hypothesis", code: "submits descriptor code" };
const changedAgent = (from, to) => (from.stage.A !== to.stage.A ? "A" : "B");

export const step = (action, from, to) => {
  switch (action) {
    case "StartEpisode":
      return "starts an episode, reads committed facts";
    case "Advance":
      return finished[from.stage[changedAgent(from, to)]];
    case "VerifyPasses":
      return "signal check passes";
    case "VerifyFails":
      return "signal check fails, no rows";
    case "CommitRewrite":
      // One label whether or not the archive admits a fact: the chart tracks
      // stages, and the fact count lives in the snapshot.
      return "commits rows durably, plus a fact if admitted";
    case "Publish":
      return to.phase === "complete" ? "publishes the final floor" : "publishes the floor";
    case "StartTraining":
      return "reloads the configured base on published rows";
    case "TrainerProduces":
      return "trainer writes a candidate";
    case "TrainerFails":
      return "trainer errors";
    case "ArtifactValid":
      return "format and identity checks pass";
    case "ArtifactInvalid":
      return "format or identity checks fail";
    case "ServeNextFloor":
      return "next floor opens, serving the candidate";
    case "Finished":
      return null;
    default:
      return action;
  }
};
