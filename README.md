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
python -m main identify        # what moves the calm quoted spread, parameter by parameter
python -m main participation   # the crisis share a provider covers, by outside option
python -m main figures         # the figures of the article, from the current artifacts
python -m main calibrate       # the coordinate search over the declared parameters
```

### This branch: EUR/CHF

The model on this branch is calibrated to EUR/CHF and its episode is the removal
of the Swiss franc floor on 15 January 2015. Three quantities differ from the
EUR/USD fit and no equation does: the quoting increment, the annual volatility
of the latent price and the overnight cost of committed capital, which is zero
here because both legs of the pair carried negative policy rates. EBS is the
primary venue for this pair, which is what makes the same sources usable.

The episode imposes one number, a repricing of −14.40 per cent, taken from the
ECB daily reference rate: 1.2010 from 9 to 14 January and 1.0280 on the 15th.
What the dislocation then looks like is the model's to produce, and it is
checked against a crisis gate drawn from the effective spreads Breedon, Chen,
Ranaldo and Vause measure on EBS around the event. That gate is the first on
this model that is not a calm-state statistic; without it the size of the shock
is unconstrained.

Every value and its source is recorded in `calibration/primary_model.json` under
`calibration_notes`, and the rotation of the frozen protocol is recorded in
`calibration/final_protocol.json` under `revision_history`.

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
