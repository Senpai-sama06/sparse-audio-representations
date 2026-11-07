# Report instructions

Use these instructions for future runs in this repository.

## Quick guide

| Report type | Use | Output |
| --- | --- | --- |
| Documentation | Explain a method, procedure, change, or problem. | A plain Markdown file. Use plain HTML if needed. |
| Results compilation | Compare measured results and inspect each test case. | One HTML file with all audio, images, and required data inside it. |

For documentation, explain the purpose, method, evidence, and limits.
Keep the text simple. Add only tables and figures that help the reader.

For results, show every case in the declared test group.
Use one table for each case.
Include before and after audio, spectrograms, scores, and rates.
Use the same format as the PitchFlower report.

## Language rules

Use ASD-STE100 as the writing target.
Use technical names when a simpler word would change the meaning.
Do not claim certified compliance without a formal language check.

- Use short, direct sentences.
- Give one instruction in each sentence.
- Use the same word for the same item.
- Define each abbreviation at its first use.
- State the actor and the action.
- Give units with each measured value.
- Separate observations from possible causes.
- State the evidence for each conclusion.
- Do not use promotional claims or decorative text.
- Do not describe an expected result as a measured result.

Example observation: "The loss increased after step 1,000."

Example inference: "The increase can indicate an unstable update. The cause is not confirmed."

## Documentation reports

### Required content

1. Give the report a clear title.
2. State the purpose in one short paragraph.
3. State the scope and the items that are excluded.
4. Describe the method or change.
5. Give the steps in the order of use.
6. Show the evidence or result.
7. State the limits and unresolved problems.
8. Give the next action, if an action is needed.

Record the run ID, code revision, and source files when they affect the result.
Link to the evidence. Do not copy large logs into the report.
Keep a short error message if it explains a failure.

### Format

Use plain headings, paragraphs, and lists.
Use tables for direct comparisons.
Use code blocks for commands.
Use a figure only when it helps explain the method or evidence.
Do not add a cover page, cards, banners, themes, or custom fonts.

Keep the main explanation short.
Put long procedures and reference details at the end.

### Check before release

- Check each command and file link.
- Check that the description matches the current code.
- Check that each conclusion has supporting evidence.
- Remove claims that the available evidence does not support.

## Results compilation reports

### 1. Fix the test scope

Define the test group before inference.
Record the source split, case IDs, speaker IDs, and selection rule.
Record each crop start, crop length, and sample rate.
State whether the cases are full recordings or short crops.

State the number of selected, completed, failed, and excluded cases.
Give the reason for each exclusion or failure.
Include every selected case. Do not select only the best outputs.
For an incomplete case, show its status and the missing items.

Keep training, checkpoint selection, calibration, and final testing separate.
Do not call a validation result a final test result.
Do not describe a selected group as the complete dataset.

### 2. Record the test method

Record these items:

- Run ID and code revision.
- Checkpoint path, step, and SHA-256.
- Checkpoint selection rule.
- Input data and preprocessing.
- Random seed and inference settings.
- Decoder and vocoder versions.
- Metric names, settings, and package versions.
- Alignment, gain changes, resampling, and length handling.
- Audio sample format and spectrogram settings.

Check that inference loads the stated checkpoint.
Do not use final weights when the report names the best checkpoint.

### 3. Make one table for each test case

Use this column order:

| Item | Before | After | Control, if available |
| --- | --- | --- | --- |
| Audio | Embedded source audio | Embedded model output | Embedded control output |
| Spectrogram | Source image | Output image | Control image |
| PESQ-WB | Reference | Score or failure reason | Score or failure reason |
| STOI | Reference | Score or failure reason | Score or failure reason |
| Playback audio data rate | Value and unit | Value and unit | Value and unit |
| Encoded data rate | Not applicable | Measured value or clear estimate | Value, if applicable |

Put the case ID and speaker ID above the table.
Give source and output durations.
Put a short description and inference below the table.

State what the scores show.
Do not infer intelligibility or listening quality from a score alone.
Do not state an unconfirmed cause as a fact.
Use "Not available" for missing measurements. Do not replace them with zero.

Include a control only when it helps explain the result.
For PitchFlower, the vocoder-only output is a useful control.
Explain the input and function of each control.

### 4. Check audio, images, and scores

Use the exact embedded audio to calculate the displayed scores.
If audio conversion changes the samples, calculate the scores again.
State the playback sample rate, channel count, and sample format.

Use the same spectrogram settings for all compared signals.
Use the same frequency range, time scale, and colour scale.
Label time, frequency, and magnitude with their units.
State the window, hop size, and magnitude reference.

Define PESQ and STOI at first use.
State the PESQ mode and metric sample rate.
Record each metric failure and the number of valid scores.
State the averaging rule and the number of cases in each mean.
Keep the same case group for model comparisons.

### 5. Label rates correctly

Keep these rates separate:

- Playback PCM data rate.
- Measured encoded bitstream rate.
- Estimated content-code rate.
- Pitch and other required side-data rates.

Use bits per second or kilobits per second.
State that 1 kbit/s means 1,000 bit/s.

PCM data rate = sample rate x bits per sample x channel count.

Measured encoded rate = encoded payload bits / source duration.

State whether headers and other metadata are included.
List excluded data, model weights, and codebook storage.
Do not call a playback WAV rate a compressed codec rate.
Do not call a latent-code estimate a measured bitstream rate.
Show "Not available" if a complete bitstream has not been measured.

For an RVQ estimate, state the codebook count and entry count.
Use the actual index count for the tested crop.
State the assumed number of bits for each index and pitch value.

### 6. Build one portable HTML file

Use plain HTML.
Use headings, paragraphs, and simple tables.
Use native audio controls.
Do not add a theme, dashboard, cards, banners, or custom fonts.

Embed every audio clip and image in the HTML file.
Use data URLs or another fully embedded format.
Keep all required playback code and data inside the file.
Do not require a server, network connection, or local asset folder.

Add a short scope statement and a mean-score table near the start.
State the test limits before the case tables.
Include the record of the test method.
Keep descriptions and inferences in the same language style as documentation.

### 7. Verify the file

1. Count the case tables and compare them with the declared test group.
2. Check that every audio clip decodes and plays.
3. Check that every image decodes and is visible.
4. Check sample rates, durations, and channel counts.
5. Check that displayed scores match the saved metric records.
6. Check the bitrate formulas and exclusions.
7. Check that no required external resources remain.
8. Open the HTML file without a network connection.
9. Check the tables, audio controls, and spectrograms.
10. Record failed checks. Do not claim that they passed.

Share the HTML file itself.
Keep metric records and verification results beside it.
Give the file size and test scope when you deliver it.

## Repository reference

- Reference report: [PitchFlower test cases](reports/pitchflower_best_testcases.html).
- Metric records: [PitchFlower metrics](reports/pitchflower_best_metrics.json).
- Report builder: [build_pitchflower_report.py](../tools/build_pitchflower_report.py).

The reference report covers 40 short dev-clean crops.
It does not cover all dev-clean files.
The builder has fixed paths and settings for that run.
Check and change them before a new run.
Use a new output path. Keep prior reports available.

The portable-report checker in `tools/check_portable_report.py` checks one embedded-audio format.
It does not replace all verification steps above.
Check that its format matches the report before you use it.
