#!/usr/bin/env python3
"""Backward-compatible entry point for the Figure 2 print-scale audit.

The old geometry comparator enforced the superseded 183 x 245 mm page and therefore
cannot be a valid release test.  Keep this filename for callers in older automation,
but delegate to the submission-scale QA that checks the current physical placement.
"""

from verify_printscale import main


if __name__ == "__main__":
    raise SystemExit(main())
