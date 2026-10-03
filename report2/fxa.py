# Copyright 2026 MarcosHCK
#
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from fx import f, interv, xlim, ylim
from typing import TypeVar

T = TypeVar ('T', float, np.ndarray)

def f_n (x: T, n: int = 10) -> np.ndarray:

  f_ = 20.0 / np.pi
  x_ = np.asarray (x, dtype = float) # type: ignore
  y_ = np.full_like (x_, -2.0)

  for k_ in range (0, n):

    p_ = 2*k_ + 1
    y_ -= (f_ * np.sin (x_ * p_)) / p_

  return y_

if __name__ == '__main__':

  matplotlib.use ('pgf')

  matplotlib.rcParams.update (
    {
      'pgf.preamble': R'\usepackage{amsmath}\usepackage{siunitx}',
      'pgf.rcfonts': False,
      'pgf.texsystem': 'lualatex',
      'text.usetex': True,
    })

  x = np.linspace (*xlim, interv * 1000)

  plt.figure (figsize = (12, 4))

  plt.plot (x, f (x), 'k--', label = 'f(x)', linewidth = 2)

  for n in [ 1, 3, 10, 100 ]:

    plt.plot (x, f_n (x, n = n), label = f'f(x, n = {n})', linewidth = 2)

  plt.grid (True, alpha = 0.3)
  plt.axhline (y = 0, color = 'k', linewidth = 0.5)
  plt.axvline (x = 0, color = 'k', linewidth = 0.5)

  plt.xlabel ('x')
  plt.ylabel ('y')
  plt.xlim (*xlim)
  plt.ylim (*ylim)

  ticks = tuple[float, float] ( int (n / np.pi) for n in xlim )
  ticks = np.linspace (*ticks, interv)

  # Set x-axis labels in terms of π
  plt.xticks (list ( n * np.pi for n in ticks ), list ( fR'${n}\pi$' for n in ticks ))

  plt.tight_layout ()
  plt.legend (loc = 'lower left')

  plt.savefig ('fxa.png')
