# SocioFM Dataset Card

## Summary

The processed corpus contains 138,066,694 retained GDELT 2.0 event records from
2022–2025 and approximately 50 GB of processed files. It includes 317,934
observed country-day aggregates. Calendarization produces 374,842 leakage-safe
next-day examples across 263 action-location country codes.

## Source

Source records are public GDELT 2.0 event exports. The pipeline retains event
metadata and normalized source hosts. It does not redistribute full article
text.

## Inclusion and exclusion

Eligible rows require GlobalEventID, event date, event code, action-location
country and source URL. Rows outside the requested date window, weak records
and stable-tuple duplicates are excluded. The final build disables per-host
daily caps and evaluates source concentration downstream.

## Missing source intervals

Ninety-two archive intervals were unavailable or failed retrieval. They remain
listed in the acquisition manifests and are not imputed.

## Temporal partitions

- Training target dates: before 1 January 2025.
- Validation target dates: 1 January–30 June 2025.
- Forward-test target dates: 1 July–31 December 2025.

## Intended use

The corpus supports research on structured event representation, temporal
generalization, event forecasting, calibration, media-coverage sensitivity and
compute-efficient domain pretraining.

## Measurement scope

GDELT records machine-coded, news-reported events. Event counts represent the
observed media-derived event stream rather than a census of latent societal
incidence.
## Archived record

The versioned processed dataset is available at
https://doi.org/10.5281/zenodo.21577601. The associated software is available at
https://doi.org/10.5281/zenodo.21576730, and the five reported model checkpoints are
available at https://doi.org/10.5281/zenodo.21577245.
