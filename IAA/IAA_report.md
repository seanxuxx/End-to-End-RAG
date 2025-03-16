# Inter-Annotator Agreement (IAA) quality check
## 1. Sample Questions
`IAA_annotation.json`: 50 random questions extracted from the test set

## 2. Agreement Criteria
- **Strict Agreement**: Exact string match  
`e.g., "Charlotte, Pittsburgh, San Jose, Tallahassee, Tempe, Seattle" vs. "Charlotte, Pittsburgh, San Jose, Tallahassee, Tempe, Seattle area" = agree`
- **Lenient Agreement**: Semantic equivalence  
`e.g., “22 Market Square in Pittsburgh” vs. “22 Market Square, Pittsburgh, PA” = agree`

## 3. Evaluation Metric
Percentage Agreement in strict and lenient agreement

## 4. Report Findings
### Agreement Metrics
calculation see `IAA_metric_calculation.ipynb`
- Strict Agreement: 13/50 (26%)
- Lenient Agreement: 45/50 (90%)

### Key Disagreement Patterns.
- Typos and Terminology: "UMPC" vs. "UPMC" (Q19), "zoning.board" typo (Q34).
- Organization Name Variation: Q9
- Full name vs. partial: Q24
- Inclusion/exclusion of details: Q38