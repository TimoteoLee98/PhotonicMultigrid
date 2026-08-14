"""Right-hand side for the 2-D Poisson "parallel capacitors" problem.

Two oppositely charged plates on an Nx by Ny grid: a horizontal strip of charge
+2 above the midline and one of -2 the same distance below it. Shared by the PC
reproduction scripts so both solve exactly the same system.
"""

import numpy as np


def set_rhs_value(x, y, Ny, b, value):
    b[x + y * Ny] = value
    return b


def build_capacitor_rhs(Nx=64, Ny=64):
    b = np.zeros(Nx * Ny)

    start_x, end_x = Nx // 4, (Nx // 4) * 3
    start_y, step_y = Ny // 2, 1
    end_y = start_y + step_y
    steps_away_from_center_y, charge = 8, 2

    for xi in range(start_x, end_x):
        for yi in range(start_y, end_y):
            b = set_rhs_value(xi, yi - step_y * steps_away_from_center_y, Ny, b, charge)
    for xi in range(start_x, end_x):
        for yi in range(start_y, end_y):
            b = set_rhs_value(xi, yi + step_y * steps_away_from_center_y, Ny, b, -charge)

    return b
