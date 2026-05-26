# Semantic Zone Progress Summary For Meeting

## 1. Why Semantic Zones Were Added

- The previous four equal image-space zones were useful for technical validation but were not biologically meaningful.
- This experiment replaces them with manually defined semantic zones: drinking, feeding, and general area.

## 2. Data Used

- Two days of Room 1 Aug 16-17 video windows
- Embedded MP4 audio only
- Daily Room 1 environment context from `Combined Room 1.xlsx`
- Manual caretaker-entry event label

## 3. What The Semantic Zones Mean

- `drinking_zone`: red-box annotated drinking area
- `feeding_zone`: green-box annotated feeding area
- `general_zone`: full-frame remainder outside the drinking and feeding areas

## 4. Outputs Produced

- Semantic-zone feature rows: 34590
- Semantic biomarker windows: 11530
- Embedded-audio windows available: 11530
- Event-labelled semantic windows: 192

## 5. Main Findings

- The semantic-zone experiment showed that caretaker entry increased overall activity and mobility, with the response becoming more interpretable through functional-area fractions and a dominant `feeder_concentrated_activity` latent state.
- The semantic video-only HMM retained stronger caretaker-event contrast than the semantic multimodal HMM.

## 6. How This Helps The Next Stage

- Semantic-zone structure makes the activity response easier to interpret in terms of functional areas.
- This prepares the analytics backbone for later individual-behaviour or occupancy modelling.

## 7. Limitations

- Semantic zones are manually defined from one reference view.
- Activity is not the same as true occupancy.
- Audio is whole-room embedded MP4 audio rather than isolated chicken vocalization.
- Environment is daily Room 1 context.
- Risk score remains a prototype heuristic and not a validated welfare diagnosis.
