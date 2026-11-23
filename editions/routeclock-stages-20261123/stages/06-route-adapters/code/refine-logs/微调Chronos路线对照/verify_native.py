from replay_route import *
from native_map import native_forecast
import verify_fast_cell as verify
from types import SimpleNamespace
for arm in ('RC-special','CR-transfer'):
    from unittest.mock import patch
    with patch.object(verify,'vectorized_forecast',native_forecast):
        # Save a separate verification result, preserving the earlier vectorized run.
        def save_check(path,obj):
            write(OUT/('native_check_'+arm+'.json'),obj)
        with patch.object(verify,'write',save_check):verify.check(arm,0,'T')
