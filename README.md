## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```


## Running the model

Every default comes from `calibration/primary_model.json`, which is the single
source of the configuration. The entry point names the workflows instead of
reimplementing them, so each command takes the options of the runner behind it.

```bash
python -m main                 # the commands, with one line on each
python -m main config          # the calibrated configuration and its provenance
python -m main run             # one simulation of a scenario, with its plots
python -m main accept          # the acceptance panel against the frozen targets
python -m main arms            # the resource matched comparison of the arms
python -m main welfare --arm reserve
python -m main selection       # the markout of the flow each venue fills
python -m main figures         # the figures of the article, from the current artifacts
python -m main calibrate       # the coordinate search over the declared parameters
```

### The comparison

Adding a facility to a dealer market moves three things at once: the capital it
commits, the obligation to keep quoting and the rule by which it prices. A
comparison against the same market without a facility pools all three, so it is
not the comparison this model makes. `arms` holds the first two fixed across
arms and varies only the third, which identifies

* what standing availability is worth, as the obliged quoter against the control
* what the pricing schedule adds, as the reserve priced pool against that quoter

Both arms carry the same total economic capital, the same two sided capacity and
a price matched to the pool's own cost, and the outcome is the best executable
round trip cost across both venues.

### Where the numbers land

`arms` writes `output/facility_arms.json` and draws `output/facility_arms.png`.
`welfare` writes one report per arm under `output/resilience/`. `accept` writes
the panel and its provenance to the path given by `--output`.
