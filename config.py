# experiment variables

ASSET_COUNTS = [25, 50, 100, 150]
LOOKBACK_WINDOWS = [126, 252, 504, 756]
REBALANCE_FREQUENCY = 21

# 11 years of data from 2015 to 2025 so that we have many rebalancing periods for each lookback window
START_DATE = "2015-01-01"
END_DATE = "2025-12-31"

RANDOM_SEED = 42