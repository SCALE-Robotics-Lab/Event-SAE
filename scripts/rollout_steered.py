"""Run LIBERO episodes with an SAE intervention or a matched native baseline."""

from event_sae.rollout import main

if __name__ == "__main__":
    main(steered=True)
