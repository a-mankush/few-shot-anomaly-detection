# Glossary

**Part cycle**
A period during which exactly one part type is inspected. A new part cycle starts when the part type changes, and the setup stays physically the same.

**Reference set**
The N (5–15) images of a good part supplied at the start of a Part cycle. It defines what "normal" means for that cycle. Parts may shift slightly in position; surface appearance is assumed not to vary.

**Enrollment**
Operator-started loading of a new Reference set so the system treats it as normal. Runs on the device with no retraining, and is fast enough to demonstrate live.

**Anomaly**
Anything that deviates from the Reference set: either a defect on an otherwise correct part, or the wrong part or no part in the frame.

**Verdict**
The outcome for one image: pass unless an Anomaly is found, reported with a heatmap of where it deviates. Tuned to prefer false alarms over missed anomalies. The operator has no tuning controls.

**Leave-one-out threshold**
The pass/fail cut-off, derived automatically at Enrollment by scoring each Reference set image against the rest, so no anomalous images are needed.

**Bad example**
An image of a known-anomalous part. Deliberately not used by the system: bad examples can't cover every anomaly, so relying on them would be unreliable.
