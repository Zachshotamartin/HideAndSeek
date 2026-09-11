export const PHYSICAL_POLICY_FILE = "models/physical-policy-trained-2c67b56afe3f.json";
export const PHYSICAL_POLICY_DETAILS = {
  "training": "Evaluated reference adapted to the new jump and tool rules. It retains previous learned movement; the jump output is untrained. Continued training uses reduced tool exploration. New candidates replace this reference only after a better evaluation.",
  "evaluation": "Evaluated against the same migrated reference on 120 layouts. Useful tool strategies and reliable jumping are not yet established.",
  "modes": "Both sight overlays are shown. Playback continues until paused or reset; training uses finite episodes for evaluation."
};
