# Image World Model Approximation

<table>
  <tr>
    <td align="center" valign="top" width="33%">
      <p><strong>01 · Accept baseline</strong></p>
      <img src="../images/world-model/01-baseline.png" width="256" height="256" alt="Original Mint-maid sprite">
    </td>
    <td align="center" valign="top" width="33%">
      <p><strong>02 · Propose boxes</strong></p>
      <img src="../images/world-model/02-candidate-boxes.png" width="256" height="256" alt="Candidate regions with red outline-only boxes">
    </td>
    <td align="center" valign="top" width="33%">
      <p><strong>03 · Classify regions</strong></p>
      <img src="../images/world-model/03-part-labels.png" width="256" height="256" alt="Illustrative hair, eye and mouth region outlines">
    </td>
  </tr>
  <tr>
    <td align="center" valign="top" width="33%">
      <p><strong>04 · Plan safe edit</strong></p>
      <img src="../images/world-model/04-edit-mask.png" width="256" height="256" alt="Actual eye-and-mouth edit mask; white pixels are editable">
    </td>
    <td align="center" valign="top" width="33%">
      <p><strong>05 · Regenerate part</strong></p>
      <img src="../images/world-model/05-regenerated.png" width="256" height="256" alt="Saved raw happy-expression proposal">
    </td>
    <td align="center" valign="top" width="33%">
      <p><strong>06 · Overlay and review</strong></p>
      <img src="../images/world-model/06-overlay-result.png" width="256" height="256" alt="Happy sprite with original pixels restored outside the edit mask">
    </td>
  </tr>
</table>

Region labels are illustrative, not automatic detections. White mask pixels are editable; the final result retains original pixels everywhere else.

[Standalone overview](image-world-model.html)

## Flowchart

```mermaid
flowchart TD
    A[Generate and Accept Baseline Image] --> B[Freeze Original Image]
    B --> C[Brute-Force Candidate Selection Boxes]
    C --> D[Decisions API: Shared Image and Batched Region Questions]
    D --> E[Refine and Validate Semantic Pixel Masks]
    E --> F[Cache Source-Bound Part Map and Landmarks]
    F --> G[Choose Expression or Motion]
    G --> H[Plan Target Geometry and Safe Edit Mask]
    H --> I[Reprompt Image Model with Original Context and Mask]
    I --> J[Keep Generated Pixels Inside Mask]
    J --> K[Overlay Original Pixels Outside Mask]
    K --> L[Check Pixel Preservation and Independent Image Verdict]
    L --> M{Accept, Reject or Uncertain?}
    M -->|Accept| N[Save Variant and Updated Part Geometry]
    M -->|Reject or Uncertain| O[Retain Baseline and Record Failure]
```
