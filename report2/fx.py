# Copyright 2026 MarcosHCK
#
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from typing import TypeVar

T = TypeVar ('T', float, np.ndarray)

def f (x: T) -> np.ndarray:

  x_ = np.asarray (x, dtype = float)
  x_ = ((x_ + np.pi) % (2 * np.pi)) - np.pi

  return np.piecewise (x_, [ x_ < 0, x_ >= 0 ], [ 3, -7 ])

xlim = (-3 * np.pi, 3 * np.pi)
ylim = (-9, 5)
interv = 7

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
  y = f (x)

  plt.figure (figsize = (12, 4))
  plt.plot (x, y, 'b-', linewidth = 2)
  plt.grid (True, alpha = 0.3)
  plt.axhline (y = 0, color = 'k', linewidth = 0.5)
  plt.axvline (x = 0, color = 'k', linewidth = 0.5)

  plt.xlabel ('x')
  plt.ylabel ('f(x)')
  plt.xlim (*xlim)
  plt.ylim (*ylim)

  ticks = tuple[float, float] ( int (n / np.pi) for n in xlim )
  ticks = np.linspace (*ticks, interv)

  plt.xticks (list ( n * np.pi for n in ticks ), list ( fR'${n}\pi$' for n in ticks ))

  plt.tight_layout ()
  plt.savefig ('fx.pgf')
