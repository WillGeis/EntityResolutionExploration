# EntityResolutionExploration

## Table of Contents
- [Section One](#blocking-methodologies)
- [Section Two](#notebook-methods)
  - [Subsection 2.1](#__precleaning)
  - [Subsection 2.2](#__preprocessRecipients)
  - [Subsection 2.3](#__add_entity_ids)
  - [Subsection 2.4](#__display_waterfall_chart)

## Blocking Methodologies
Entity deduplication is handled by [Splink](https://github.com/moj-analytical-services/splink), which is a probabilistic record linkage library. The configuration below controls how candidate pairs are generated and scored/blocked.

| Field | Method | Rationale |
|---|---|---|
| `entity_name` | Exact match | Rewards records that share an identical name |
| `entity_name` | Levenshtein distance | Captures minor typos and abbreviation differences (e.g. "Corp" vs "Corp.") |
| `country` | Exact match | Rewards records that share an identical country |
| `state` | Exact match | Further rewards records that share an identical state (sub to US because that is where interest is) |
| `address_and_city` | Exact match | Strengthens confidence when location details align, this is hardly ever tripped |

Predictions are generated at a match probability `>= 0.7` and clusters are formed at `>= 0.95`, reflecting a conservative approach: casting a moderately wide net during inference while only merging records with very high confidence; likely this would be tuned better if training is done on m and u values.

## Notebook Methods
This is a small notebook I worked on while exploring Entity Resolution (ER). Some of the more noteable methods include:

```python
def __precleaning(entry):
    entry = entry.upper()
    entry = entry.replace(",", " ")
    entry = re.sub(r' +', ' ', entry)
...
```
#### __precleaning
Standardizes a raw entity string before any structured parsing.

 - Uppercases and strips punctuation/special characters
 - Collapses extra whitespace
 - Detects and appends a US country suffix if the entity appears to be American
 - Replaces company monikers, state names, and country names with their canonical abbreviations/ISO codes — first via phrase-level substitution, then token-by-token
 - Removes adjacent duplicate tokens and normalizes "USA" --> "US"

```python
def __preprocessRecipients(entity_str):
    entity_str = entity_str.strip()
    country_guess = entity_str[-2:].upper()
    country_obj = pycountry.countries.get(alpha_2=country_guess)
...
```

#### __preprocessRecipients
Parses a raw entity string into structured components.

 - Country: checks the last two characters as an ISO alpha-2 code; falls back to scanning the full string for a country name or code
 - Company moniker: searches for known legal suffixes (LLC, Inc., etc.) to bound the entity name
 - State: looks for a US state ISO code in the remaining string
 - ZIP code: extracts a 5-digit US ZIP using a regex
 - Returns a dict with keys: `entity_name`, `address_and_city`, `state`, `zip`, `country`

```python
def __add_entity_ids(entities):
    df = pd.DataFrame(entities)
    df["unique_id"] = df.index.astype(str)

    comparisons = [
        cl.ExactMatch("entity_name"),
        cl.LevenshteinAtThresholds("entity_name"),
        cl.ExactMatch("country"),
        cl.ExactMatch("state"),
        cl.ExactMatch("address_and_city"),
    ]

    blocking_rules = [
        "l.entity_name = r.entity_name",
        "(l.country = r.country) and (l.state = r.state)",
        "(l.entity_name = r.entity_name and l.country = r.country)",
        "levenshtein(l.entity_name, r.entity_name) < 2",
    ]

    settings = {
        "link_type": "dedupe_only",
        "unique_id_column_name": "unique_id",
        "blocking_rules_to_generate_predictions": blocking_rules,
        "comparisons": comparisons
    }
...
```
#### __add_entity_ids
Groups duplicate or near-duplicate entities and assigns shared cluster IDs.

 - Builds a DataFrame from the entity list and runs record linkage via Splink (Linker)
 - Comparison features: exact name match, Levenshtein distance on name, exact country/state/city match
 - Blocking rules limit candidate pairs to plausible matches (same name, same region, edit distance < 2)
 - Predicts match probability and clusters at a 0.95 threshold
 - Writes a `entity_id` (cluster ID) back onto each entity dict

```python
def __display_waterfall_chart(entities):
    df = pd.DataFrame(entities)
    df["unique_id"] = df.index.astype(str)

    comparisons = [
        cl.ExactMatch("entity_name"),
        cl.LevenshteinAtThresholds("entity_name"),
        cl.ExactMatch("country"),
        cl.ExactMatch("state"),
        cl.ExactMatch("address_and_city"),
    ]

    blocking_rules = [
        "l.entity_name = r.entity_name",
        "(l.country = r.country) and (l.state = r.state)",
        "(l.entity_name = r.entity_name and l.country = r.country)",
        "levenshtein(l.entity_name, r.entity_name) < 2",
    ]

    settings = {
        "link_type": "dedupe_only",
        "unique_id_column_name": "unique_id",
        "blocking_rules_to_generate_predictions": blocking_rules,
        "comparisons": comparisons,
        "retain_intermediate_calculation_columns": True,
        "retain_matching_columns": True
    }
...
```
#### __display_waterfall_chart

Same deduplication logic as above, but oriented toward inspection rather than production use.

 - Runs Splink with intermediate calculation columns retained for explainability
 - Returns both a waterfall chart (visual breakdown of how each feature contributed to a match score) and the cluster ID dictionary
 - Useful for tuning thresholds and debugging why two records were or weren't linked
