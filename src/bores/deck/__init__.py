"""
Eclipse / GRDECL keyword deck parsing engine.

This package provides a composable framework for reading Eclipse-style keyword
decks (GRDECL include files, full `.DATA` decks, etc.) without each keyword
having to re-implement comment stripping, tokenisation, `N*value` repeat
expansion, or `BOX` / `EQUALS` / `ADD` / `MULTIPLY` / `COPY`
operator resolution.

**Usage Example**:

```python
from bores.deck import DeckFile

df = DeckFile("model.DATA")
poro = df.get("PORO")  # ndarray (n_cells,) or None
coord = df.get("COORD")  # ndarray (ny+1, nx+1, 6) or None
```
"""

from bores.deck.core import *  # noqa
from bores.deck.file import DeckFile  # noqa
from bores.deck.keywords import *  # noqa
from bores.deck.operators import *  # noqa
