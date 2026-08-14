"""Multigrid-hierarchy loading helper shared by the reproduce_*.py scripts."""

from phoamg.multigrid.multigrid import MultiGrid


def load_mg(directory):
    """Load and return the MultiGrid hierarchy saved under `directory`."""
    mg = MultiGrid()
    mg.load(directory)
    return mg
