# SocioFM Model Card

## Model family

SocioFM uses GPT-2-style causal decoders with the public GPT-2 tokenizer and a
256-token context window.

| Release name | Layers | Hidden size | Heads | Measured parameters |
|---|---:|---:|---:|---:|
| SocioFM-26M | 4 | 384 | 6 | 26,495,616 |
| SocioFM-57M | 10 | 512 | 8 | 57,387,520 |
| SocioFM-101M | 14 | 640 | 10 | 101,258,880 |

## Training

The common capacity comparison uses 100,000 optimization steps and an effective
batch of 32 sequences. The release also includes a 26M checkpoint at 50,000
steps and a 101M continuation at 200,000 steps.

## Evaluation

The reported evaluation includes chronological language modelling, next-day
event-volume forecasting, dominant event-root classification, calibration,
country-held-out transfer, field masking, media-density analysis and an
explicit Global North/South coverage sensitivity audit.

## Intended use

The models are research artifacts for representation learning and benchmarking
over structured societal event streams.

## Scope

The models inherit the machine-coding errors, source selection and geographic
coverage patterns of the underlying event database. Output probabilities and
generated sequences are not measurements of unobserved societal incidence.
## Archived records

The five reported checkpoints are available at
https://doi.org/10.5281/zenodo.21577245. The associated software is available at
https://doi.org/10.5281/zenodo.21576730, and the processed training and benchmark data
are available at https://doi.org/10.5281/zenodo.21577601.
