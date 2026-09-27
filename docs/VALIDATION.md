# Validation record

Review date: 27 September 2026. The 3 code cells in the included notebook were executed in order
in a fresh IPython process launched from `notebooks/`, on Linux with Python 3.12 and CPU execution.
The saved notebook contains the resulting text and figure outputs, with no saved execution errors.

Three regression tests passed: reference log-probability agreement (including punctuation, Unicode,
unknown words and empty prediction documents); rejected invalid training inputs; prediction before fitting.
The notebook also matched scikit-learn log probabilities to an absolute tolerance of 1e-12 on the toy split.

Core package versions match the pins in `requirements.txt`. The notebook web interface and installation
on Windows/macOS were not separately exercised. Stochastic results can vary across platforms and
library builds. This validation covers the supplied example and focused regression cases, not every
possible input or production deployment.

Re-run regression checks from the repository root:

```bash
python -m unittest discover -s tests -v
```
