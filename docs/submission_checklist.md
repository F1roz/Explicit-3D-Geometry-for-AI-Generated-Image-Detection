# Venue options and submission checklist

## Venue options (checked 11 Oct 2026; recheck before submitting)

Quartiles are from SCImago Journal Rank 2025 (scimagojr.com); conference rank from the CORE portal. Quartiles vary by subject category and year, and acceptance is never guaranteed. Fees and review times were not verified; check each journal's guide for authors.

| Venue | Type | Verified facts | Fit with this work | Pros | Cons |
| --- | --- | --- | --- | --- | --- |
| **Forensic Science International: Digital Investigation** (Elsevier) | Journal | SJR 2025 0.857; Q1 in Information Systems, Law, Pathology and Forensic Medicine; Q2 in Computer Science Applications; h index 72. Scope includes multimedia provenance, integrity and authenticity | **Best fit**: forensic question, emphasis on false positives and validity, negative results acceptable if rigorous | Q1, modest impact, scope matches exactly | Readers less ML focused; explain the statistics clearly |
| Journal of Information Security and Applications (Elsevier) | Journal | SJR 2025 0.905; Q1 in Computer Networks and Communications, Safety Risk Reliability and Quality, Software; h index 80. Scope includes digital forensics | Good fit | Q1, broader security readership | Less image specific |
| Image and Vision Computing (Elsevier) | Journal | SJR 2025 0.881; Q1 in Computer Vision and Pattern Recognition, Electrical and Electronic Engineering, Signal Processing; h index 154 | Fits if results are positive and the geometry analysis is strong | Strong vision readership | More competitive; expects methodological novelty |
| Journal of Visual Communication and Image Representation | Journal | SJR 2025 0.573; best quartile Q1 (Media Technology), Q2 in Computer Vision and Pattern Recognition | Acceptable fallback | Image processing readership | Q2 in the most relevant category |
| IEEE ICIP | Conference | CORE rank B in CORE2023 (and B in every edition since 2013) | Short version (check the current page limit in the call for papers) | Fast, B ranked, image processing audience | Very short format; full analysis must go to a journal |

Recommendation: target *Forensic Science International: Digital Investigation* with the full paper; consider an ICIP version only if the venue's dual submission rules allow it.

## Checklist before submission

### Science

- [ ] Pilot decision recorded (GO/REVISE/STOP) and followed
- [ ] All hypothesis tests reported, including failed ones
- [ ] Leakage report shows no overlaps
- [ ] Number of unseen generators per test set stated; intervals labelled indicative when fewer than 5
- [ ] Discussion paragraph matches the actual outcome (others deleted)
- [ ] Limitations section updated with real numbers (retrieved images, generators)

### Numbers and figures

- [ ] `python scripts/check_paper.py` reports 0 `[xyz]` in main.tex, supplementary.tex and research_paper.docx
- [ ] 0 figure placeholders
- [ ] Abstract, tables, figures and conclusion agree
- [ ] Every figure referenced in the text, readable at print size

### References

- [ ] Every entry in references.bib opened and checked
- [ ] arXiv entries replaced by published versions where available
- [ ] Author lists complete where the venue requires it

### Formatting (Elsevier example)

- [ ] Body moved into `elsarticle` template; reference style as required
- [ ] Highlights (3 to 5 bullet points, check character limit)
- [ ] Graphical abstract if required (the architecture figure is a candidate)
- [ ] Keywords
- [ ] Data and code availability statement with a working link
- [ ] Declaration of generative AI use (required by Elsevier)
- [ ] CRediT author contributions, funding, competing interests
- [ ] Cover letter stating the contribution and that the work is not under review elsewhere

### Reproducibility

- [ ] Code repository public (or anonymised for double blind review)
- [ ] Tagged version matching the paper
- [ ] Feature tables and split manifests released (respecting dataset licences: release ids, not images)
- [ ] Reproducibility audit on a fresh runtime done (SOP step 17)
