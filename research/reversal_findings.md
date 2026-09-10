# Reversal / falling-knife findings (RESEARCH)

- source: reversal_features.csv   rows(after weak_rr drop): 146,666   R metric: tr_2r_g1 (live trailing exit)
- train = date < 2025-01-01 (2021-2024) | test = 2025-2026 | knife5 = fixed stop hit <=5 bars
- MAE from the fixed-stop path; avgR/win/PF/maxConsecLoss from the trailing exit
- rows are ~daily samples of the same setups (autocorrelated); episode-deduped headline below

## 0. Baselines (train / test / 2022) — wide net vs current gates

**wide net**
  train  n= 96001  win= 36.8%  avgR=+0.302  medR=-1.00  PF= 1.49  MAE=-1.53  knife5= 41.2%  maxConsecLoss=131
  test   n= 50665  win= 38.3%  avgR=+0.292  medR=-1.00  PF= 1.49  MAE=-1.51  knife5= 41.0%  maxConsecLoss=93
  2022   n= 26777  win= 30.6%  avgR=+0.046  medR=-1.00  PF= 1.07  MAE=-1.69  knife5= 47.8%  maxConsecLoss=189
  train (episode-deduped)  n= 17357  win= 37.0%  avgR=+0.372  medR=-1.00  PF=  1.6  MAE=-1.57  knife5= 41.9%  maxConsecLoss=56
  test  (episode-deduped)  n=  9167  win= 38.0%  avgR=+0.331  medR=-1.00  PF= 1.55  MAE=-1.57  knife5= 42.8%  maxConsecLoss=35

**current gates**
  train  n= 19094  win= 36.3%  avgR=+0.412  medR=-1.00  PF= 1.65  MAE=-1.79  knife5= 50.8%  maxConsecLoss=108
  test   n= 11190  win= 38.4%  avgR=+0.485  medR=-1.00  PF=  1.8  MAE=-1.75  knife5= 48.5%  maxConsecLoss=133
  2022   n=  6459  win= 31.7%  avgR=+0.200  medR=-1.00  PF= 1.29  MAE=-1.82  knife5= 56.6%  maxConsecLoss=104
  train (episode-deduped)  n=  4441  win= 36.9%  avgR=+0.611  medR=-1.00  PF= 1.98  MAE=-1.95  knife5= 52.4%  maxConsecLoss=30
  test  (episode-deduped)  n=  2515  win= 38.4%  avgR=+0.758  medR=-1.00  PF= 2.25  MAE=-1.86  knife5= 50.7%  maxConsecLoss=32


## 1. Falling-knife discriminators — univariate, current-gate subset

Spearman corr vs realised R and vs knife5, TRAIN only (current-gate rows):

  feature                            rho(R)  rho(knife5)
  rev_macd_hist                      -0.038       -0.063
  rev_ema50_slope_20d_pct            +0.033       +0.007
  rev_down_days_10                   +0.032       +0.087
  rev_last_10d_return_pct            -0.027       -0.088
  rev_close_vs_ema10_pct             -0.026       -0.130
  rev_days_since_low_20              -0.026       -0.056
  rev_close_up_days_5                -0.025       -0.104
  rev_days_since_low_10              -0.025       -0.105
  rev_made_new_low_3d                +0.025       +0.084
  rev_higher_low_pct                 -0.024       -0.079
  rev_decline_in_atr                 +0.023       +0.038
  rev_close_vs_ema20_pct             -0.022       -0.108
  rev_dist_to_swing_low_20_pct       -0.022       -0.088
  rev_pct_from_20d_high              -0.021       -0.076
  rev_ema20_slope_5d_pct             -0.019       -0.045
  rev_last_5d_return_pct             -0.019       -0.116
  rev_ema50_slope_10d_pct            +0.018       +0.005
  rev_dist_to_swing_low_10_pct       -0.017       -0.117
  rev_vol_at_low_rel                 +0.015       -0.023
  rev_rsi_bull_div                   -0.013       +0.021
  rev_rsi14                          -0.012       -0.089
  rev_max_3d_drop_pct                -0.011       -0.002
  rev_rsi_up_3d                      -0.008       -0.114
  rev_rsi_turn_up                    +0.008       -0.002
  rev_ema10_gt_ema20                 +0.007       -0.016
  rev_vol_trend_stab                 +0.006       +0.043
  rev_trend_linearity_120            -0.006       -0.016
  rev_waterfall_ratio                -0.006       +0.030
  rev_macd_hist_up                   -0.005       -0.098
  rev_up_vol_expansion_5             +0.003       +0.018
  rev_reclaimed_ema20                -0.003       -0.049
  rev_rsi_min_10                     +0.002       +0.054
  rev_max_1d_drop_pct                -0.002       -0.025
  rev_days_below_ema20               -0.001       +0.030
  rev_worst_gap_pct                  +0.001       -0.018
  rev_pct_bars_above_ema50_126       -0.001       -0.023
  rev_down_up_vol_ratio_12           -0.001       -0.023


## 2. Bin tables (train + test) — most-informative features

### rev_days_since_low_20

_train_
     (-0.101, 0.5]  n=  2552  win= 37.1%  avgR=+1.041  medR=-1.00  PF= 2.66  MAE=-3.11  knife5= 61.8%  maxConsecLoss=34
        (0.5, 1.5]  n=  1945  win= 36.8%  avgR=+0.388  medR=-1.00  PF= 1.62  MAE=-1.69  knife5= 55.2%  maxConsecLoss=26
        (1.5, 3.5]  n=  3054  win= 38.3%  avgR=+0.398  medR=-1.00  PF= 1.65  MAE=-1.62  knife5= 49.6%  maxConsecLoss=23
        (3.5, 6.5]  n=  3366  win= 36.5%  avgR=+0.276  medR=-1.00  PF= 1.44  MAE=-1.49  knife5= 46.0%  maxConsecLoss=37
       (6.5, 10.5]  n=  3317  win= 35.8%  avgR=+0.228  medR=-1.00  PF= 1.36  MAE=-1.48  knife5= 46.0%  maxConsecLoss=34
      (10.5, 20.1]  n=  4860  win= 34.7%  avgR=+0.319  medR=-1.00  PF=  1.5  MAE=-1.69  knife5= 50.8%  maxConsecLoss=76

_test_
     (-0.101, 0.5]  n=  1304  win= 34.6%  avgR=+1.622  medR=-1.00  PF= 3.49  MAE=-3.13  knife5= 64.6%  maxConsecLoss=32
        (0.5, 1.5]  n=  1032  win= 40.6%  avgR=+0.465  medR=-1.00  PF= 1.79  MAE=-1.70  knife5= 53.2%  maxConsecLoss=20
        (1.5, 3.5]  n=  1696  win= 39.2%  avgR=+0.387  medR=-1.00  PF= 1.64  MAE=-1.69  knife5= 47.9%  maxConsecLoss=31
        (3.5, 6.5]  n=  1896  win= 38.9%  avgR=+0.335  medR=-1.00  PF= 1.56  MAE=-1.48  knife5= 41.5%  maxConsecLoss=26
       (6.5, 10.5]  n=  2035  win= 38.8%  avgR=+0.302  medR=-1.00  PF= 1.51  MAE=-1.44  knife5= 43.0%  maxConsecLoss=52
      (10.5, 20.1]  n=  3227  win= 38.3%  avgR=+0.288  medR=-1.00  PF= 1.48  MAE=-1.58  knife5= 48.5%  maxConsecLoss=60

### rev_higher_low_pct

_train_
      (-0.01, 1.0]  n=  8142  win= 37.0%  avgR=+0.590  medR=-1.00  PF= 1.94  MAE=-2.11  knife5= 55.9%  maxConsecLoss=47
        (1.0, 3.0]  n=  3673  win= 36.3%  avgR=+0.292  medR=-1.00  PF= 1.46  MAE=-1.57  knife5= 47.0%  maxConsecLoss=40
        (3.0, 6.0]  n=  3735  win= 36.5%  avgR=+0.234  medR=-1.00  PF= 1.37  MAE=-1.54  knife5= 46.2%  maxConsecLoss=31
       (6.0, 12.0]  n=  2744  win= 35.0%  avgR=+0.319  medR=-1.00  PF=  1.5  MAE=-1.53  knife5= 47.2%  maxConsecLoss=58
      (12.0, 60.0]  n=   795  win= 33.6%  avgR=+0.302  medR=-1.00  PF= 1.46  MAE=-1.67  knife5= 50.9%  maxConsecLoss=21

_test_
      (-0.01, 1.0]  n=  4309  win= 37.9%  avgR=+0.768  medR=-1.00  PF= 2.24  MAE=-2.15  knife5= 55.0%  maxConsecLoss=101
        (1.0, 3.0]  n=  2079  win= 36.4%  avgR=+0.275  medR=-1.00  PF= 1.44  MAE=-1.55  knife5= 46.6%  maxConsecLoss=43
        (3.0, 6.0]  n=  2161  win= 37.3%  avgR=+0.226  medR=-1.00  PF= 1.37  MAE=-1.56  knife5= 43.5%  maxConsecLoss=46
       (6.0, 12.0]  n=  1814  win= 39.5%  avgR=+0.330  medR=-1.00  PF= 1.57  MAE=-1.41  knife5= 43.2%  maxConsecLoss=69
      (12.0, 60.0]  n=   827  win= 46.6%  avgR=+0.555  medR=-1.00  PF= 2.09  MAE=-1.34  knife5= 44.4%  maxConsecLoss=16

### rev_close_vs_ema20_pct

_train_
  (-40.001, -12.0]  n=    17  win= 23.5%  avgR=+0.887  medR=-1.00  PF= 2.24  MAE=-1.94  knife5= 58.8%  maxConsecLoss=8
     (-12.0, -8.0]  n=   601  win= 38.1%  avgR=+0.932  medR=-1.00  PF= 2.51  MAE=-2.52  knife5= 61.4%  maxConsecLoss=32
      (-8.0, -4.0]  n=  4072  win= 37.7%  avgR=+0.733  medR=-1.00  PF= 2.18  MAE=-2.27  knife5= 57.7%  maxConsecLoss=28
      (-4.0, -1.0]  n=  7581  win= 35.5%  avgR=+0.345  medR=-1.00  PF= 1.54  MAE=-1.76  knife5= 51.4%  maxConsecLoss=58
       (-1.0, 1.0]  n=  4313  win= 36.3%  avgR=+0.235  medR=-1.00  PF= 1.37  MAE=-1.46  knife5= 45.0%  maxConsecLoss=44
        (1.0, 4.0]  n=  2034  win= 35.9%  avgR=+0.258  medR=-1.00  PF= 1.41  MAE=-1.50  knife5= 45.3%  maxConsecLoss=34
       (4.0, 20.0]  n=   476  win= 37.4%  avgR=+0.318  medR=-1.00  PF= 1.52  MAE=-1.54  knife5= 46.2%  maxConsecLoss=22

_test_
  (-40.001, -12.0]  n=    30  win= 23.3%  avgR=+5.386  medR=-1.00  PF= 8.02  MAE=-5.45  knife5= 76.7%  maxConsecLoss=10
     (-12.0, -8.0]  n=   402  win= 36.8%  avgR=+0.750  medR=-1.00  PF=  2.2  MAE=-2.86  knife5= 63.9%  maxConsecLoss=30
      (-8.0, -4.0]  n=  2239  win= 35.6%  avgR=+0.684  medR=-1.00  PF= 2.07  MAE=-2.26  knife5= 57.7%  maxConsecLoss=30
      (-4.0, -1.0]  n=  4118  win= 37.9%  avgR=+0.504  medR=-1.00  PF= 1.82  MAE=-1.70  knife5= 49.4%  maxConsecLoss=88
       (-1.0, 1.0]  n=  2580  win= 38.8%  avgR=+0.262  medR=-1.00  PF= 1.44  MAE=-1.42  knife5= 42.8%  maxConsecLoss=51
        (1.0, 4.0]  n=  1419  win= 43.3%  avgR=+0.367  medR=-1.00  PF= 1.69  MAE=-1.38  knife5= 39.7%  maxConsecLoss=40
       (4.0, 20.0]  n=   402  win= 42.3%  avgR=+0.399  medR=-1.00  PF= 1.73  MAE=-1.35  knife5= 39.1%  maxConsecLoss=15

### rev_days_below_ema20

_train_
     (-0.101, 0.5]  n=  4319  win= 37.0%  avgR=+0.279  medR=-1.00  PF= 1.45  MAE=-1.49  knife5= 44.3%  maxConsecLoss=52
        (0.5, 3.5]  n=  3226  win= 35.2%  avgR=+0.346  medR=-1.00  PF= 1.54  MAE=-1.85  knife5= 54.1%  maxConsecLoss=54
        (3.5, 8.5]  n=  3256  win= 37.1%  avgR=+0.591  medR=-1.00  PF= 1.94  MAE=-2.17  knife5= 56.6%  maxConsecLoss=47
       (8.5, 15.5]  n=  3068  win= 37.3%  avgR=+0.508  medR=-1.00  PF= 1.82  MAE=-1.89  knife5= 51.5%  maxConsecLoss=52
      (15.5, 40.0]  n=  4901  win= 35.2%  avgR=+0.386  medR=-1.00  PF=  1.6  MAE=-1.70  knife5= 50.4%  maxConsecLoss=42
     (40.0, 300.0]  n=   324  win= 38.9%  avgR=+0.514  medR=-1.00  PF= 1.85  MAE=-2.05  knife5= 48.5%  maxConsecLoss=25

_test_
     (-0.101, 0.5]  n=  2877  win= 41.1%  avgR=+0.323  medR=-1.00  PF= 1.57  MAE=-1.36  knife5= 40.5%  maxConsecLoss=54
        (0.5, 3.5]  n=  1862  win= 37.5%  avgR=+0.683  medR=-1.00  PF= 2.11  MAE=-1.75  knife5= 52.1%  maxConsecLoss=45
        (3.5, 8.5]  n=  1675  win= 38.3%  avgR=+0.604  medR=-1.00  PF= 1.99  MAE=-2.16  knife5= 55.7%  maxConsecLoss=55
       (8.5, 15.5]  n=  1545  win= 36.6%  avgR=+0.289  medR=-1.00  PF= 1.46  MAE=-1.83  knife5= 49.2%  maxConsecLoss=48
      (15.5, 40.0]  n=  3063  win= 37.1%  avgR=+0.534  medR=-1.00  PF= 1.86  MAE=-1.82  knife5= 49.5%  maxConsecLoss=57
     (40.0, 300.0]  n=   168  win= 44.6%  avgR=+0.790  medR=-1.00  PF= 2.43  MAE=-2.08  knife5= 51.2%  maxConsecLoss=12

### rev_ema20_slope_5d_pct

_train_
   (-30.001, -4.0]  n=   610  win= 37.2%  avgR=+0.497  medR=-1.00  PF=  1.8  MAE=-1.82  knife5= 53.4%  maxConsecLoss=34
      (-4.0, -2.0]  n=  5032  win= 37.6%  avgR=+0.458  medR=-1.00  PF= 1.74  MAE=-1.84  knife5= 53.2%  maxConsecLoss=33
      (-2.0, -0.5]  n=  7790  win= 35.6%  avgR=+0.460  medR=-1.00  PF= 1.72  MAE=-1.93  knife5= 51.5%  maxConsecLoss=50
       (-0.5, 0.5]  n=  3713  win= 35.5%  avgR=+0.240  medR=-1.00  PF= 1.38  MAE=-1.57  knife5= 47.9%  maxConsecLoss=46
        (0.5, 2.0]  n=  1610  win= 37.8%  avgR=+0.432  medR=-1.00  PF=  1.7  MAE=-1.54  knife5= 46.9%  maxConsecLoss=35
       (2.0, 20.0]  n=   339  win= 33.0%  avgR=+0.234  medR=-1.00  PF= 1.35  MAE=-1.57  knife5= 47.8%  maxConsecLoss=21

_test_
   (-30.001, -4.0]  n=   402  win= 35.1%  avgR=+0.686  medR=-1.00  PF= 2.07  MAE=-2.06  knife5= 56.0%  maxConsecLoss=26
      (-4.0, -2.0]  n=  2944  win= 37.0%  avgR=+0.503  medR=-1.00  PF=  1.8  MAE=-2.01  knife5= 51.4%  maxConsecLoss=50
      (-2.0, -0.5]  n=  4236  win= 38.1%  avgR=+0.473  medR=-1.00  PF= 1.77  MAE=-1.77  knife5= 50.1%  maxConsecLoss=74
       (-0.5, 0.5]  n=  2197  win= 38.0%  avgR=+0.489  medR=-1.00  PF= 1.82  MAE=-1.54  knife5= 45.3%  maxConsecLoss=56
        (0.5, 2.0]  n=  1123  win= 45.1%  avgR=+0.459  medR=-1.00  PF= 1.87  MAE=-1.35  knife5= 39.7%  maxConsecLoss=40
       (2.0, 20.0]  n=   288  win= 39.6%  avgR=+0.280  medR=-1.00  PF=  1.5  MAE=-1.40  knife5= 45.1%  maxConsecLoss=12

### rev_reclaimed_ema20

_train_
     (-0.101, 0.5]  n= 16353  win= 36.2%  avgR=+0.439  medR=-1.00  PF= 1.69  MAE=-1.84  knife5= 51.8%  maxConsecLoss=103
        (0.5, 1.1]  n=  2741  win= 36.8%  avgR=+0.250  medR=-1.00  PF=  1.4  MAE=-1.50  knife5= 44.9%  maxConsecLoss=30

_test_
     (-0.101, 0.5]  n=  9437  win= 38.5%  avgR=+0.534  medR=-1.00  PF= 1.88  MAE=-1.81  knife5= 49.6%  maxConsecLoss=104
        (0.5, 1.1]  n=  1753  win= 37.8%  avgR=+0.221  medR=-1.00  PF= 1.37  MAE=-1.39  knife5= 42.6%  maxConsecLoss=60

### rev_max_1d_drop_pct

_train_
  (-60.001, -15.0]  n=    50  win= 34.0%  avgR=+0.143  medR=-1.00  PF= 1.23  MAE=-1.47  knife5= 40.0%  maxConsecLoss=7
    (-15.0, -10.0]  n=   427  win= 35.4%  avgR=+0.472  medR=-1.00  PF= 1.74  MAE=-1.73  knife5= 46.4%  maxConsecLoss=24
     (-10.0, -7.0]  n=  1839  win= 38.1%  avgR=+0.562  medR=-1.00  PF= 1.91  MAE=-1.73  knife5= 50.8%  maxConsecLoss=33
      (-7.0, -5.0]  n=  3610  win= 37.2%  avgR=+0.379  medR=-1.00  PF= 1.61  MAE=-1.72  knife5= 51.8%  maxConsecLoss=47
      (-5.0, -3.0]  n=  7751  win= 34.9%  avgR=+0.369  medR=-1.00  PF= 1.57  MAE=-1.79  knife5= 52.8%  maxConsecLoss=67
       (-3.0, 0.0]  n=  5417  win= 37.3%  avgR=+0.442  medR=-1.00  PF= 1.71  MAE=-1.88  knife5= 47.9%  maxConsecLoss=59

_test_
  (-60.001, -15.0]  n=    66  win= 33.3%  avgR=+0.009  medR=-1.00  PF= 1.01  MAE=-1.81  knife5= 42.4%  maxConsecLoss=13
    (-15.0, -10.0]  n=   507  win= 37.5%  avgR=+0.357  medR=-1.00  PF= 1.57  MAE=-1.98  knife5= 51.1%  maxConsecLoss=35
     (-10.0, -7.0]  n=  1373  win= 35.8%  avgR=+0.329  medR=-1.00  PF= 1.52  MAE=-1.85  knife5= 53.6%  maxConsecLoss=33
      (-7.0, -5.0]  n=  2477  win= 37.5%  avgR=+0.485  medR=-1.00  PF= 1.78  MAE=-1.73  knife5= 50.9%  maxConsecLoss=60
      (-5.0, -3.0]  n=  3778  win= 38.5%  avgR=+0.512  medR=-1.00  PF= 1.85  MAE=-1.73  knife5= 49.3%  maxConsecLoss=47
       (-3.0, 0.0]  n=  2989  win= 40.5%  avgR=+0.554  medR=-1.00  PF= 1.97  MAE=-1.70  knife5= 43.0%  maxConsecLoss=49

### rev_max_3d_drop_pct

_train_
  (-70.001, -20.0]  n=    74  win= 40.5%  avgR=+0.789  medR=-1.00  PF= 2.46  MAE=-1.46  knife5= 41.9%  maxConsecLoss=7
    (-20.0, -14.0]  n=   488  win= 37.7%  avgR=+0.538  medR=-1.00  PF= 1.87  MAE=-1.72  knife5= 44.7%  maxConsecLoss=24
    (-14.0, -10.0]  n=  2439  win= 36.9%  avgR=+0.415  medR=-1.00  PF= 1.67  MAE=-1.70  knife5= 49.2%  maxConsecLoss=28
     (-10.0, -7.0]  n=  4751  win= 38.0%  avgR=+0.471  medR=-1.00  PF= 1.77  MAE=-1.71  knife5= 51.4%  maxConsecLoss=38
       (-7.0, 0.0]  n= 11341  win= 35.4%  avgR=+0.378  medR=-1.00  PF= 1.59  MAE=-1.86  knife5= 51.3%  maxConsecLoss=74

_test_
  (-70.001, -20.0]  n=    54  win= 24.1%  avgR=-0.204  medR=-1.00  PF= 0.73  MAE=-2.00  knife5= 70.4%  maxConsecLoss=14
    (-20.0, -14.0]  n=   574  win= 42.7%  avgR=+0.503  medR=-1.00  PF= 1.91  MAE=-1.47  knife5= 47.7%  maxConsecLoss=37
    (-14.0, -10.0]  n=  1777  win= 36.3%  avgR=+0.260  medR=-1.00  PF= 1.41  MAE=-1.82  knife5= 53.9%  maxConsecLoss=36
     (-10.0, -7.0]  n=  2595  win= 37.8%  avgR=+0.520  medR=-1.00  PF= 1.84  MAE=-1.86  knife5= 51.0%  maxConsecLoss=49
       (-7.0, 0.0]  n=  6185  win= 38.9%  avgR=+0.539  medR=-1.00  PF= 1.91  MAE=-1.70  knife5= 45.9%  maxConsecLoss=67

### rev_worst_gap_pct

_train_
  (-60.001, -12.0]  n=    84  win= 36.9%  avgR=+0.601  medR=-1.00  PF= 1.96  MAE=-1.64  knife5= 42.9%  maxConsecLoss=8
     (-12.0, -7.0]  n=   530  win= 39.6%  avgR=+0.521  medR=-1.00  PF= 1.87  MAE=-1.50  knife5= 46.6%  maxConsecLoss=26
      (-7.0, -4.0]  n=  2000  win= 37.4%  avgR=+0.403  medR=-1.00  PF= 1.65  MAE=-1.67  knife5= 47.9%  maxConsecLoss=35
      (-4.0, -2.0]  n=  6877  win= 36.6%  avgR=+0.423  medR=-1.00  PF= 1.67  MAE=-1.77  knife5= 53.2%  maxConsecLoss=52
       (-2.0, 0.0]  n=  9603  win= 35.7%  avgR=+0.398  medR=-1.00  PF= 1.62  MAE=-1.85  knife5= 50.1%  maxConsecLoss=80

_test_
  (-60.001, -12.0]  n=    59  win= 35.6%  avgR=+0.096  medR=-1.00  PF= 1.15  MAE=-1.44  knife5= 50.8%  maxConsecLoss=12
     (-12.0, -7.0]  n=   395  win= 31.9%  avgR=+0.135  medR=-1.00  PF=  1.2  MAE=-1.82  knife5= 49.9%  maxConsecLoss=24
      (-7.0, -4.0]  n=  1791  win= 40.4%  avgR=+0.400  medR=-1.00  PF= 1.68  MAE=-1.64  knife5= 50.7%  maxConsecLoss=59
      (-4.0, -2.0]  n=  3864  win= 39.1%  avgR=+0.567  medR=-1.00  PF= 1.95  MAE=-1.77  knife5= 49.3%  maxConsecLoss=43
       (-2.0, 0.0]  n=  5081  win= 37.7%  avgR=+0.485  medR=-1.00  PF=  1.8  MAE=-1.77  knife5= 47.0%  maxConsecLoss=66

### rev_decline_in_atr

_train_
     (-0.001, 2.0]  n=  4882  win= 36.0%  avgR=+0.277  medR=-1.00  PF= 1.44  MAE=-1.49  knife5= 46.5%  maxConsecLoss=75
        (2.0, 3.5]  n=  6017  win= 35.2%  avgR=+0.307  medR=-1.00  PF= 1.48  MAE=-1.73  knife5= 53.0%  maxConsecLoss=62
        (3.5, 5.0]  n=  5078  win= 37.8%  avgR=+0.568  medR=-1.00  PF= 1.92  MAE=-1.96  knife5= 51.6%  maxConsecLoss=41
        (5.0, 7.0]  n=  2621  win= 36.3%  avgR=+0.523  medR=-1.00  PF= 1.83  MAE=-2.08  knife5= 52.7%  maxConsecLoss=28
       (7.0, 10.0]  n=   467  win= 37.5%  avgR=+0.787  medR=-1.00  PF= 2.26  MAE=-2.38  knife5= 52.0%  maxConsecLoss=11
      (10.0, 40.0]  n=    29  win= 55.2%  avgR=+1.237  medR=+1.12  PF= 3.89  MAE=-1.43  knife5= 13.8%  maxConsecLoss=5

_test_
     (-0.001, 2.0]  n=  3218  win= 41.4%  avgR=+0.376  medR=-1.00  PF= 1.67  MAE=-1.48  knife5= 41.1%  maxConsecLoss=50
        (2.0, 3.5]  n=  3706  win= 38.9%  avgR=+0.360  medR=-1.00  PF=  1.6  MAE=-1.62  knife5= 49.7%  maxConsecLoss=57
        (3.5, 5.0]  n=  2700  win= 36.1%  avgR=+0.801  medR=-1.00  PF= 2.26  MAE=-1.92  knife5= 52.2%  maxConsecLoss=31
        (5.0, 7.0]  n=  1334  win= 34.4%  avgR=+0.362  medR=-1.00  PF= 1.56  MAE=-2.14  knife5= 54.9%  maxConsecLoss=33
       (7.0, 10.0]  n=   231  win= 38.1%  avgR=+1.034  medR=-1.00  PF= 2.69  MAE=-3.05  knife5= 54.1%  maxConsecLoss=13
      (10.0, 40.0]  n=     1  win=  0.0%  avgR=-1.000  medR=-1.00  PF=  0.0  MAE=-1.28  knife5=100.0%  maxConsecLoss=1

### rev_waterfall_ratio

_train_
     (-0.001, 0.3]  n=   592  win= 36.7%  avgR=+0.314  medR=-1.00  PF= 1.51  MAE=-1.41  knife5= 44.4%  maxConsecLoss=14
       (0.3, 0.45]  n=  2345  win= 34.4%  avgR=+0.280  medR=-1.00  PF= 1.43  MAE=-1.68  knife5= 49.1%  maxConsecLoss=32
       (0.45, 0.6]  n=  5161  win= 37.1%  avgR=+0.440  medR=-1.00  PF= 1.71  MAE=-1.79  knife5= 50.8%  maxConsecLoss=36
        (0.6, 0.8]  n=  6945  win= 37.2%  avgR=+0.453  medR=-1.00  PF= 1.73  MAE=-1.81  knife5= 50.6%  maxConsecLoss=45
        (0.8, 3.0]  n=  4051  win= 35.0%  avgR=+0.396  medR=-1.00  PF= 1.61  MAE=-1.90  knife5= 53.2%  maxConsecLoss=59

_test_
     (-0.001, 0.3]  n=   419  win= 46.1%  avgR=+0.438  medR=-0.33  PF= 1.88  MAE=-1.24  knife5= 32.7%  maxConsecLoss=24
       (0.3, 0.45]  n=  1426  win= 38.7%  avgR=+0.333  medR=-1.00  PF= 1.56  MAE=-1.50  knife5= 47.7%  maxConsecLoss=22
       (0.45, 0.6]  n=  2801  win= 36.2%  avgR=+0.392  medR=-1.00  PF= 1.63  MAE=-1.90  knife5= 48.4%  maxConsecLoss=54
        (0.6, 0.8]  n=  3984  win= 38.4%  avgR=+0.491  medR=-1.00  PF= 1.81  MAE=-1.74  knife5= 49.8%  maxConsecLoss=30
        (0.8, 3.0]  n=  2560  win= 39.5%  avgR=+0.669  medR=-1.00  PF= 2.12  MAE=-1.81  knife5= 49.8%  maxConsecLoss=61

### rev_down_days_10

_train_
     (-0.101, 3.5]  n=  1929  win= 34.5%  avgR=+0.149  medR=-1.00  PF= 1.23  MAE=-1.51  knife5= 42.5%  maxConsecLoss=36
        (3.5, 5.5]  n=  8891  win= 35.9%  avgR=+0.320  medR=-1.00  PF=  1.5  MAE=-1.59  knife5= 48.8%  maxConsecLoss=78
        (5.5, 6.5]  n=  4641  win= 36.7%  avgR=+0.475  medR=-1.00  PF= 1.76  MAE=-1.90  knife5= 53.8%  maxConsecLoss=38
        (6.5, 7.5]  n=  2629  win= 38.2%  avgR=+0.620  medR=-1.00  PF= 2.01  MAE=-2.21  knife5= 55.4%  maxConsecLoss=31
       (7.5, 10.1]  n=  1004  win= 37.4%  avgR=+0.898  medR=-1.00  PF= 2.44  MAE=-2.56  knife5= 59.4%  maxConsecLoss=24

_test_
     (-0.101, 3.5]  n=  1459  win= 40.8%  avgR=+0.291  medR=-1.00  PF= 1.52  MAE=-1.32  knife5= 43.2%  maxConsecLoss=33
        (3.5, 5.5]  n=  5414  win= 37.9%  avgR=+0.375  medR=-1.00  PF= 1.62  MAE=-1.65  knife5= 46.6%  maxConsecLoss=69
        (5.5, 6.5]  n=  2595  win= 39.2%  avgR=+0.516  medR=-1.00  PF= 1.86  MAE=-1.83  knife5= 50.3%  maxConsecLoss=66
        (6.5, 7.5]  n=  1302  win= 36.2%  avgR=+0.634  medR=-1.00  PF=  2.0  MAE=-2.20  knife5= 54.5%  maxConsecLoss=53
       (7.5, 10.1]  n=   420  win= 37.9%  avgR=+1.923  medR=-1.00  PF= 4.11  MAE=-2.61  knife5= 62.4%  maxConsecLoss=10

### rev_last_5d_return_pct

_train_
   (-40.001, -8.0]  n=   915  win= 38.9%  avgR=+0.784  medR=-1.00  PF= 2.29  MAE=-2.31  knife5= 59.0%  maxConsecLoss=18
      (-8.0, -4.0]  n=  2773  win= 35.9%  avgR=+0.789  medR=-1.00  PF= 2.24  MAE=-2.58  knife5= 59.9%  maxConsecLoss=28
      (-4.0, -1.0]  n=  4670  win= 37.2%  avgR=+0.464  medR=-1.00  PF= 1.74  MAE=-1.90  knife5= 55.2%  maxConsecLoss=43
       (-1.0, 1.0]  n=  3961  win= 35.7%  avgR=+0.287  medR=-1.00  PF= 1.45  MAE=-1.54  knife5= 48.0%  maxConsecLoss=37
        (1.0, 4.0]  n=  4320  win= 35.2%  avgR=+0.210  medR=-1.00  PF= 1.33  MAE=-1.46  knife5= 44.9%  maxConsecLoss=37
       (4.0, 40.0]  n=  2455  win= 37.3%  avgR=+0.304  medR=-1.00  PF= 1.49  MAE=-1.51  knife5= 44.2%  maxConsecLoss=26

_test_
   (-40.001, -8.0]  n=   613  win= 32.1%  avgR=+0.616  medR=-1.00  PF= 1.91  MAE=-3.01  knife5= 65.9%  maxConsecLoss=40
      (-8.0, -4.0]  n=  1469  win= 35.5%  avgR=+1.151  medR=-1.00  PF= 2.79  MAE=-2.32  knife5= 61.5%  maxConsecLoss=25
      (-4.0, -1.0]  n=  2467  win= 37.3%  avgR=+0.524  medR=-1.00  PF= 1.84  MAE=-1.94  knife5= 54.0%  maxConsecLoss=53
       (-1.0, 1.0]  n=  2350  win= 41.1%  avgR=+0.342  medR=-1.00  PF= 1.59  MAE=-1.52  knife5= 43.7%  maxConsecLoss=34
        (1.0, 4.0]  n=  2560  win= 39.3%  avgR=+0.293  medR=-1.00  PF=  1.5  MAE=-1.35  knife5= 40.5%  maxConsecLoss=48
       (4.0, 40.0]  n=  1731  win= 39.6%  avgR=+0.297  medR=-1.00  PF= 1.51  MAE=-1.44  knife5= 42.2%  maxConsecLoss=41

### rev_down_up_vol_ratio_12

_train_
     (-0.001, 0.7]  n=  1705  win= 34.5%  avgR=+0.389  medR=-1.00  PF=  1.6  MAE=-1.78  knife5= 54.8%  maxConsecLoss=24
        (0.7, 1.0]  n=  6786  win= 36.6%  avgR=+0.386  medR=-1.00  PF= 1.61  MAE=-1.77  knife5= 51.7%  maxConsecLoss=57
        (1.0, 1.3]  n=  6288  win= 36.7%  avgR=+0.457  medR=-1.00  PF= 1.73  MAE=-1.80  knife5= 48.9%  maxConsecLoss=46
        (1.3, 1.8]  n=  3473  win= 36.4%  avgR=+0.382  medR=-1.00  PF= 1.61  MAE=-1.80  knife5= 50.0%  maxConsecLoss=31
       (1.8, 10.0]  n=   795  win= 34.8%  avgR=+0.460  medR=-1.00  PF= 1.72  MAE=-1.83  knife5= 53.6%  maxConsecLoss=20

_test_
     (-0.001, 0.7]  n=   687  win= 40.8%  avgR=+0.660  medR=-1.00  PF= 2.15  MAE=-1.73  knife5= 45.0%  maxConsecLoss=50
        (0.7, 1.0]  n=  4137  win= 37.8%  avgR=+0.529  medR=-1.00  PF= 1.87  MAE=-1.75  knife5= 50.0%  maxConsecLoss=61
        (1.0, 1.3]  n=  4158  win= 38.5%  avgR=+0.434  medR=-1.00  PF= 1.72  MAE=-1.72  knife5= 48.5%  maxConsecLoss=41
        (1.3, 1.8]  n=  1923  win= 39.3%  avgR=+0.384  medR=-1.00  PF= 1.64  MAE=-1.78  knife5= 46.3%  maxConsecLoss=79
       (1.8, 10.0]  n=   271  win= 32.5%  avgR=+0.870  medR=-1.00  PF= 2.32  MAE=-1.82  knife5= 51.7%  maxConsecLoss=10

### rev_vol_at_low_rel

_train_
     (-0.001, 0.8]  n=  4324  win= 34.7%  avgR=+0.368  medR=-1.00  PF= 1.57  MAE=-1.80  knife5= 52.2%  maxConsecLoss=36
        (0.8, 1.2]  n=  5894  win= 36.2%  avgR=+0.347  medR=-1.00  PF= 1.55  MAE=-1.88  knife5= 52.0%  maxConsecLoss=54
        (1.2, 1.8]  n=  5094  win= 36.8%  avgR=+0.457  medR=-1.00  PF= 1.73  MAE=-1.78  knife5= 49.3%  maxConsecLoss=39
        (1.8, 2.5]  n=  2186  win= 38.8%  avgR=+0.599  medR=-1.00  PF= 1.99  MAE=-1.67  knife5= 48.5%  maxConsecLoss=37
       (2.5, 20.0]  n=  1596  win= 36.5%  avgR=+0.366  medR=-1.00  PF= 1.59  MAE=-1.68  knife5= 51.3%  maxConsecLoss=32

_test_
     (-0.001, 0.8]  n=  2001  win= 36.5%  avgR=+0.544  medR=-1.00  PF= 1.87  MAE=-1.84  knife5= 49.4%  maxConsecLoss=37
        (0.8, 1.2]  n=  3303  win= 38.6%  avgR=+0.455  medR=-1.00  PF= 1.75  MAE=-1.84  knife5= 48.9%  maxConsecLoss=48
        (1.2, 1.8]  n=  3560  win= 38.4%  avgR=+0.551  medR=-1.00  PF= 1.91  MAE=-1.73  knife5= 48.0%  maxConsecLoss=112
        (1.8, 2.5]  n=  1373  win= 39.5%  avgR=+0.420  medR=-1.00  PF=  1.7  MAE=-1.55  knife5= 49.4%  maxConsecLoss=46
       (2.5, 20.0]  n=   953  win= 40.1%  avgR=+0.311  medR=-1.00  PF= 1.53  MAE=-1.54  knife5= 46.1%  maxConsecLoss=24

### rev_up_vol_expansion_5

_train_
     (-0.001, 0.6]  n=  2192  win= 37.7%  avgR=+0.437  medR=-1.00  PF= 1.71  MAE=-1.71  knife5= 48.4%  maxConsecLoss=27
        (0.6, 0.9]  n=  6537  win= 35.6%  avgR=+0.308  medR=-1.00  PF= 1.48  MAE=-1.73  knife5= 50.5%  maxConsecLoss=81
        (0.9, 1.2]  n=  5522  win= 36.3%  avgR=+0.375  medR=-1.00  PF= 1.59  MAE=-1.77  knife5= 49.7%  maxConsecLoss=42
        (1.2, 1.6]  n=  3081  win= 35.6%  avgR=+0.469  medR=-1.00  PF= 1.73  MAE=-1.79  knife5= 52.7%  maxConsecLoss=32
       (1.6, 10.0]  n=  1268  win= 38.2%  avgR=+0.590  medR=-1.00  PF= 1.96  MAE=-1.69  knife5= 52.8%  maxConsecLoss=21

_test_
     (-0.001, 0.6]  n=   921  win= 36.6%  avgR=+0.325  medR=-1.00  PF= 1.52  MAE=-1.77  knife5= 52.2%  maxConsecLoss=40
        (0.6, 0.9]  n=  4131  win= 36.6%  avgR=+0.353  medR=-1.00  PF= 1.57  MAE=-1.71  knife5= 49.5%  maxConsecLoss=114
        (0.9, 1.2]  n=  3655  win= 39.7%  avgR=+0.418  medR=-1.00  PF= 1.71  MAE=-1.68  knife5= 46.8%  maxConsecLoss=55
        (1.2, 1.6]  n=  1705  win= 41.2%  avgR=+0.711  medR=-1.00  PF= 2.23  MAE=-1.81  knife5= 44.5%  maxConsecLoss=22
       (1.6, 10.0]  n=   566  win= 39.6%  avgR=+0.558  medR=-1.00  PF= 1.94  MAE=-1.88  knife5= 52.1%  maxConsecLoss=36

### rev_rsi14

_train_
    (-0.001, 25.0]  n=   172  win= 40.1%  avgR=+1.077  medR=-1.00  PF=  2.8  MAE=-3.44  knife5= 63.4%  maxConsecLoss=12
      (25.0, 32.0]  n=  1229  win= 37.4%  avgR=+1.009  medR=-1.00  PF= 2.63  MAE=-2.73  knife5= 58.2%  maxConsecLoss=30
      (32.0, 38.0]  n=  3260  win= 36.8%  avgR=+0.597  medR=-1.00  PF= 1.95  MAE=-2.25  knife5= 56.8%  maxConsecLoss=28
      (38.0, 45.0]  n=  6850  win= 36.0%  avgR=+0.349  medR=-1.00  PF= 1.55  MAE=-1.67  knife5= 50.9%  maxConsecLoss=41
      (45.0, 55.0]  n=  6699  win= 35.9%  avgR=+0.276  medR=-1.00  PF= 1.44  MAE=-1.53  knife5= 47.2%  maxConsecLoss=64
     (55.0, 100.0]  n=   884  win= 38.5%  avgR=+0.287  medR=-1.00  PF= 1.47  MAE=-1.40  knife5= 43.4%  maxConsecLoss=22

_test_
    (-0.001, 25.0]  n=    94  win= 43.6%  avgR=+1.373  medR=-1.00  PF= 3.44  MAE=-2.23  knife5= 58.5%  maxConsecLoss=6
      (25.0, 32.0]  n=   613  win= 35.2%  avgR=+1.050  medR=-1.00  PF= 2.63  MAE=-3.26  knife5= 62.2%  maxConsecLoss=22
      (32.0, 38.0]  n=  1753  win= 36.0%  avgR=+1.025  medR=-1.00  PF= 2.61  MAE=-2.23  knife5= 56.9%  maxConsecLoss=46
      (38.0, 45.0]  n=  3921  win= 37.7%  avgR=+0.340  medR=-1.00  PF= 1.55  MAE=-1.68  knife5= 49.0%  maxConsecLoss=60
      (45.0, 55.0]  n=  4123  win= 39.6%  avgR=+0.313  medR=-1.00  PF= 1.53  MAE=-1.45  knife5= 44.3%  maxConsecLoss=47
     (55.0, 100.0]  n=   686  win= 43.6%  avgR=+0.343  medR=-0.81  PF= 1.66  MAE=-1.31  knife5= 36.3%  maxConsecLoss=22

### rev_rsi_min_10

_train_
    (-0.001, 20.0]  n=   327  win= 39.1%  avgR=+0.365  medR=-1.00  PF= 1.61  MAE=-1.46  knife5= 41.3%  maxConsecLoss=20
      (20.0, 27.0]  n=  2259  win= 36.5%  avgR=+0.353  medR=-1.00  PF= 1.57  MAE=-1.69  knife5= 44.7%  maxConsecLoss=22
      (27.0, 33.0]  n=  5767  win= 35.5%  avgR=+0.393  medR=-1.00  PF= 1.62  MAE=-1.76  knife5= 50.7%  maxConsecLoss=56
      (33.0, 40.0]  n=  7081  win= 37.2%  avgR=+0.465  medR=-1.00  PF= 1.75  MAE=-1.90  knife5= 51.5%  maxConsecLoss=57
     (40.0, 100.0]  n=  3660  win= 35.6%  avgR=+0.377  medR=-1.00  PF= 1.59  MAE=-1.73  knife5= 54.5%  maxConsecLoss=58

_test_
    (-0.001, 20.0]  n=   142  win= 38.0%  avgR=+0.122  medR=-1.00  PF=  1.2  MAE=-1.54  knife5= 35.2%  maxConsecLoss=11
      (20.0, 27.0]  n=  1590  win= 40.6%  avgR=+0.410  medR=-1.00  PF=  1.7  MAE=-1.83  knife5= 42.0%  maxConsecLoss=65
      (27.0, 33.0]  n=  2986  win= 38.6%  avgR=+0.555  medR=-1.00  PF= 1.92  MAE=-1.78  knife5= 49.1%  maxConsecLoss=43
      (33.0, 40.0]  n=  4175  win= 37.7%  avgR=+0.579  medR=-1.00  PF= 1.94  MAE=-1.78  knife5= 50.8%  maxConsecLoss=81
     (40.0, 100.0]  n=  2297  win= 37.9%  avgR=+0.298  medR=-1.00  PF=  1.5  MAE=-1.60  knife5= 49.1%  maxConsecLoss=46

### rev_rsi_turn_up

_train_
     (-0.101, 0.5]  n= 15567  win= 36.2%  avgR=+0.420  medR=-1.00  PF= 1.66  MAE=-1.84  knife5= 50.9%  maxConsecLoss=118
        (0.5, 1.1]  n=  3527  win= 36.9%  avgR=+0.373  medR=-1.00  PF=  1.6  MAE=-1.59  knife5= 50.7%  maxConsecLoss=42

_test_
     (-0.101, 0.5]  n=  9184  win= 38.7%  avgR=+0.528  medR=-1.00  PF= 1.88  MAE=-1.76  knife5= 48.1%  maxConsecLoss=89
        (0.5, 1.1]  n=  2006  win= 36.8%  avgR=+0.291  medR=-1.00  PF= 1.46  MAE=-1.69  knife5= 50.4%  maxConsecLoss=33

### rev_rsi_bull_div

_train_
     (-0.101, 0.5]  n= 17123  win= 36.5%  avgR=+0.430  medR=-1.00  PF= 1.68  MAE=-1.82  knife5= 50.5%  maxConsecLoss=103
        (0.5, 1.1]  n=  1971  win= 34.8%  avgR=+0.254  medR=-1.00  PF= 1.39  MAE=-1.57  knife5= 54.0%  maxConsecLoss=38

_test_
     (-0.101, 0.5]  n= 10124  win= 38.0%  avgR=+0.480  medR=-1.00  PF= 1.79  MAE=-1.76  knife5= 48.2%  maxConsecLoss=103
        (0.5, 1.1]  n=  1066  win= 41.9%  avgR=+0.535  medR=-1.00  PF= 1.92  MAE=-1.65  knife5= 52.1%  maxConsecLoss=20

### rev_macd_hist_up

_train_
     (-0.101, 0.5]  n=  7605  win= 35.9%  avgR=+0.592  medR=-1.00  PF= 1.93  MAE=-2.18  knife5= 56.9%  maxConsecLoss=97
        (0.5, 1.1]  n= 11489  win= 36.6%  avgR=+0.293  medR=-1.00  PF= 1.47  MAE=-1.54  knife5= 46.9%  maxConsecLoss=44

_test_
     (-0.101, 0.5]  n=  4267  win= 37.8%  avgR=+0.782  medR=-1.00  PF= 2.28  MAE=-2.11  knife5= 54.7%  maxConsecLoss=72
        (0.5, 1.1]  n=  6923  win= 38.8%  avgR=+0.302  medR=-1.00  PF=  1.5  MAE=-1.52  knife5= 44.7%  maxConsecLoss=85

### rev_dist_to_swing_low_20_pct

_train_
     (-0.101, 1.0]  n=  1220  win= 36.1%  avgR=+1.793  medR=-1.00  PF= 3.82  MAE=-4.76  knife5= 67.0%  maxConsecLoss=24
        (1.0, 3.0]  n=  4144  win= 37.5%  avgR=+0.412  medR=-1.00  PF= 1.66  MAE=-1.75  knife5= 57.5%  maxConsecLoss=49
        (3.0, 6.0]  n=  6158  win= 36.2%  avgR=+0.259  medR=-1.00  PF= 1.41  MAE=-1.55  knife5= 47.2%  maxConsecLoss=59
       (6.0, 10.0]  n=  4488  win= 36.8%  avgR=+0.344  medR=-1.00  PF= 1.55  MAE=-1.50  knife5= 46.2%  maxConsecLoss=41
      (10.0, 60.0]  n=  3079  win= 34.4%  avgR=+0.274  medR=-1.00  PF= 1.42  MAE=-1.61  knife5= 49.6%  maxConsecLoss=57

_test_
     (-0.101, 1.0]  n=   529  win= 39.1%  avgR=+3.633  medR=-1.00  PF= 6.99  MAE=-5.02  knife5= 65.4%  maxConsecLoss=23
        (1.0, 3.0]  n=  2217  win= 38.5%  avgR=+0.507  medR=-1.00  PF= 1.82  MAE=-1.80  knife5= 57.1%  maxConsecLoss=48
        (3.0, 6.0]  n=  3336  win= 37.2%  avgR=+0.250  medR=-1.00  PF=  1.4  MAE=-1.55  knife5= 45.7%  maxConsecLoss=48
       (6.0, 10.0]  n=  2609  win= 36.6%  avgR=+0.200  medR=-1.00  PF= 1.32  MAE=-1.56  knife5= 44.4%  maxConsecLoss=113
      (10.0, 60.0]  n=  2499  win= 41.6%  avgR=+0.415  medR=-1.00  PF= 1.74  MAE=-1.47  knife5= 45.4%  maxConsecLoss=35

### rev_ema50_slope_20d_pct

_train_
   (-30.001, -1.0]  n= 15719  win= 36.4%  avgR=+0.386  medR=-1.00  PF= 1.61  MAE=-1.72  knife5= 49.9%  maxConsecLoss=115
       (-1.0, 0.0]  n=  1560  win= 36.2%  avgR=+0.584  medR=-1.00  PF= 1.92  MAE=-2.05  knife5= 53.5%  maxConsecLoss=26
        (0.0, 1.0]  n=   958  win= 34.6%  avgR=+0.513  medR=-1.00  PF= 1.79  MAE=-2.34  knife5= 56.3%  maxConsecLoss=22
        (1.0, 3.0]  n=   695  win= 38.0%  avgR=+0.448  medR=-1.00  PF= 1.73  MAE=-2.06  knife5= 56.5%  maxConsecLoss=18
        (3.0, 6.0]  n=   155  win= 38.1%  avgR=+0.568  medR=-1.00  PF= 1.92  MAE=-2.63  knife5= 57.4%  maxConsecLoss=12
       (6.0, 40.0]  n=     7  win= 14.3%  avgR=-0.630  medR=-1.00  PF= 0.27  MAE=-2.43  knife5=100.0%  maxConsecLoss=4

_test_
   (-30.001, -1.0]  n=  9653  win= 38.3%  avgR=+0.410  medR=-1.00  PF= 1.68  MAE=-1.71  knife5= 47.8%  maxConsecLoss=95
       (-1.0, 0.0]  n=   767  win= 40.3%  avgR=+0.728  medR=-1.00  PF= 2.25  MAE=-1.94  knife5= 50.7%  maxConsecLoss=25
        (0.0, 1.0]  n=   380  win= 37.4%  avgR=+2.086  medR=-1.00  PF= 4.38  MAE=-1.96  knife5= 53.9%  maxConsecLoss=10
        (1.0, 3.0]  n=   319  win= 37.9%  avgR=+0.317  medR=-1.00  PF= 1.51  MAE=-1.83  knife5= 57.7%  maxConsecLoss=19
        (3.0, 6.0]  n=    62  win= 35.5%  avgR=+0.172  medR=-1.00  PF= 1.27  MAE=-2.75  knife5= 53.2%  maxConsecLoss=9
       (6.0, 40.0]  n=     9  win= 44.4%  avgR=+1.055  medR=-1.00  PF=  2.9  MAE=-2.24  knife5= 66.7%  maxConsecLoss=2

### rev_pct_bars_above_ema50_126

_train_
    (-0.001, 40.0]  n=  2136  win= 36.9%  avgR=+0.445  medR=-1.00  PF= 1.71  MAE=-1.79  knife5= 51.3%  maxConsecLoss=28
      (40.0, 55.0]  n=  6337  win= 36.5%  avgR=+0.461  medR=-1.00  PF= 1.73  MAE=-1.76  knife5= 51.6%  maxConsecLoss=60
      (55.0, 70.0]  n=  7537  win= 35.1%  avgR=+0.332  medR=-1.00  PF= 1.52  MAE=-1.79  knife5= 51.8%  maxConsecLoss=52
      (70.0, 85.0]  n=  2892  win= 38.7%  avgR=+0.453  medR=-1.00  PF= 1.75  MAE=-1.86  knife5= 46.7%  maxConsecLoss=28
     (85.0, 100.0]  n=   192  win= 38.0%  avgR=+0.919  medR=-1.00  PF= 2.49  MAE=-1.85  knife5= 46.9%  maxConsecLoss=20

_test_
    (-0.001, 40.0]  n=  1016  win= 40.3%  avgR=+0.471  medR=-1.00  PF=  1.8  MAE=-1.66  knife5= 48.9%  maxConsecLoss=23
      (40.0, 55.0]  n=  3679  win= 36.6%  avgR=+0.332  medR=-1.00  PF= 1.53  MAE=-1.76  knife5= 50.9%  maxConsecLoss=64
      (55.0, 70.0]  n=  4690  win= 39.4%  avgR=+0.543  medR=-1.00  PF= 1.92  MAE=-1.70  knife5= 46.3%  maxConsecLoss=70
      (70.0, 85.0]  n=  1671  win= 38.8%  avgR=+0.654  medR=-1.00  PF= 2.08  MAE=-1.85  knife5= 48.9%  maxConsecLoss=36
     (85.0, 100.0]  n=   134  win= 33.6%  avgR=+0.647  medR=-1.00  PF= 1.99  MAE=-2.39  knife5= 54.5%  maxConsecLoss=16

### rev_trend_linearity_120

_train_
     (-1.011, 0.0]  n=  7111  win= 37.1%  avgR=+0.469  medR=-1.00  PF= 1.75  MAE=-1.76  knife5= 51.4%  maxConsecLoss=68
        (0.0, 0.4]  n=  4177  win= 34.5%  avgR=+0.369  medR=-1.00  PF= 1.57  MAE=-1.78  knife5= 51.4%  maxConsecLoss=50
        (0.4, 0.7]  n=  4296  win= 35.7%  avgR=+0.310  medR=-1.00  PF= 1.49  MAE=-1.82  knife5= 51.2%  maxConsecLoss=39
       (0.7, 0.85]  n=  2416  win= 37.4%  avgR=+0.470  medR=-1.00  PF= 1.76  MAE=-1.70  knife5= 50.3%  maxConsecLoss=36
      (0.85, 1.01]  n=  1094  win= 38.3%  avgR=+0.470  medR=-1.00  PF= 1.77  MAE=-2.16  knife5= 45.2%  maxConsecLoss=23

_test_
     (-1.011, 0.0]  n=  4122  win= 39.0%  avgR=+0.454  medR=-1.00  PF= 1.76  MAE=-1.69  knife5= 48.0%  maxConsecLoss=69
        (0.0, 0.4]  n=  2380  win= 36.7%  avgR=+0.314  medR=-1.00  PF= 1.51  MAE=-1.78  knife5= 51.1%  maxConsecLoss=39
        (0.4, 0.7]  n=  2508  win= 38.4%  avgR=+0.482  medR=-1.00  PF= 1.79  MAE=-1.75  knife5= 47.7%  maxConsecLoss=102
       (0.7, 0.85]  n=  1518  win= 39.1%  avgR=+0.621  medR=-1.00  PF= 2.03  MAE=-1.65  knife5= 47.8%  maxConsecLoss=26
      (0.85, 1.01]  n=   662  win= 39.4%  avgR=+0.995  medR=-1.00  PF= 2.68  MAE=-2.18  knife5= 47.1%  maxConsecLoss=27

### price_vs_ema200_pct

_train_
    (-20.0, -15.0]  n=   592  win= 38.9%  avgR=+0.675  medR=-1.00  PF= 2.11  MAE=-2.03  knife5= 56.2%  maxConsecLoss=20
    (-15.0, -10.0]  n=  1548  win= 33.9%  avgR=+0.365  medR=-1.00  PF= 1.55  MAE=-1.88  knife5= 57.0%  maxConsecLoss=28
     (-10.0, -5.0]  n=  3606  win= 35.0%  avgR=+0.423  medR=-1.00  PF= 1.66  MAE=-1.97  knife5= 52.9%  maxConsecLoss=40
       (-5.0, 0.0]  n=  7415  win= 36.7%  avgR=+0.439  medR=-1.00  PF=  1.7  MAE=-1.69  knife5= 49.1%  maxConsecLoss=39
        (0.0, 3.0]  n=  5933  win= 37.1%  avgR=+0.357  medR=-1.00  PF= 1.57  MAE=-1.77  knife5= 49.6%  maxConsecLoss=40

_test_
    (-25.0, -20.0]  n=     1  win=  0.0%  avgR=-1.000  medR=-1.00  PF=  0.0  MAE=-1.28  knife5=100.0%  maxConsecLoss=1
    (-20.0, -15.0]  n=   379  win= 34.0%  avgR=+0.782  medR=-1.00  PF= 2.19  MAE=-2.39  knife5= 59.1%  maxConsecLoss=34
    (-15.0, -10.0]  n=   960  win= 41.7%  avgR=+0.616  medR=-1.00  PF= 2.07  MAE=-1.93  knife5= 50.7%  maxConsecLoss=45
     (-10.0, -5.0]  n=  2332  win= 37.9%  avgR=+0.366  medR=-1.00  PF=  1.6  MAE=-1.80  knife5= 50.7%  maxConsecLoss=34
       (-5.0, 0.0]  n=  4332  win= 36.9%  avgR=+0.488  medR=-1.00  PF= 1.78  MAE=-1.68  knife5= 48.6%  maxConsecLoss=79
        (0.0, 3.0]  n=  3186  win= 40.4%  avgR=+0.495  medR=-1.00  PF= 1.86  MAE=-1.66  knife5= 44.9%  maxConsecLoss=57

### dist_52w_high_pct

_train_
  (-70.001, -40.0]  n=   960  win= 35.4%  avgR=+0.623  medR=-1.00  PF= 1.97  MAE=-1.87  knife5= 57.7%  maxConsecLoss=35
    (-40.0, -30.0]  n=  2286  win= 37.6%  avgR=+0.570  medR=-1.00  PF= 1.92  MAE=-1.99  knife5= 55.6%  maxConsecLoss=26
    (-30.0, -20.0]  n=  5993  win= 34.5%  avgR=+0.323  medR=-1.00  PF=  1.5  MAE=-1.82  knife5= 54.1%  maxConsecLoss=50
    (-20.0, -12.0]  n=  6894  win= 36.3%  avgR=+0.393  medR=-1.00  PF= 1.62  MAE=-1.71  knife5= 48.3%  maxConsecLoss=54
     (-12.0, -7.0]  n=  2772  win= 39.2%  avgR=+0.457  medR=-1.00  PF= 1.76  MAE=-1.79  knife5= 44.3%  maxConsecLoss=32
      (-7.0, -4.0]  n=   189  win= 41.8%  avgR=+0.251  medR=-1.00  PF= 1.44  MAE=-1.62  knife5= 45.5%  maxConsecLoss=12

_test_
  (-70.001, -40.0]  n=   515  win= 32.8%  avgR=+0.672  medR=-1.00  PF=  2.0  MAE=-2.04  knife5= 58.3%  maxConsecLoss=24
    (-40.0, -30.0]  n=  1633  win= 37.9%  avgR=+0.577  medR=-1.00  PF= 1.94  MAE=-1.93  knife5= 55.3%  maxConsecLoss=36
    (-30.0, -20.0]  n=  3886  win= 38.1%  avgR=+0.379  medR=-1.00  PF= 1.62  MAE=-1.75  knife5= 48.8%  maxConsecLoss=106
    (-20.0, -12.0]  n=  3882  win= 38.2%  avgR=+0.491  medR=-1.00  PF= 1.81  MAE=-1.72  knife5= 47.8%  maxConsecLoss=46
     (-12.0, -7.0]  n=  1163  win= 42.8%  avgR=+0.614  medR=-1.00  PF= 2.14  MAE=-1.51  knife5= 36.4%  maxConsecLoss=18
      (-7.0, -4.0]  n=   109  win= 45.9%  avgR=+0.433  medR=-1.00  PF= 1.82  MAE=-1.20  knife5= 47.7%  maxConsecLoss=11


## 3. Confirmation trade-off — cost of waiting for stabilisation

Compare current-gate rows split by whether stabilisation is ALREADY visible at the signal.
`stabilising` = days_since_low_20>=3 AND higher_low_pct>0 AND close_vs_ema20_pct>-6 AND last_5d_return_pct>-4
`still_falling` = the complement. Entry-price-given-up = how far price already sits above the 20d low.

_train_
  stabilising     n= 11609  win= 35.9%  avgR=+0.273  medR=-1.00  PF= 1.43  MAE=-1.53  knife5= 46.6%  maxConsecLoss=98   dist_above_20d_low(med)=6.4%
  still_falling    n=  7485  win= 37.1%  avgR=+0.627  medR=-1.00  PF=  2.0  MAE=-2.20  knife5= 57.4%  maxConsecLoss=62   dist_above_20d_low(med)=2.9%

_test_
  stabilising     n=  7113  win= 39.2%  avgR=+0.305  medR=-1.00  PF= 1.52  MAE=-1.48  knife5= 43.3%  maxConsecLoss=68   dist_above_20d_low(med)=7.1%
  still_falling    n=  4077  win= 37.0%  avgR=+0.800  medR=-1.00  PF= 2.27  MAE=-2.22  knife5= 57.7%  maxConsecLoss=81   dist_above_20d_low(med)=3.2%

Within `stabilising`: near the low (<=6% above 20d low) vs already bounced (>6%):
_train_
  stabilising & near low   n=  5262  win= 35.8%  avgR=+0.245  medR=-1.00  PF= 1.38  MAE=-1.56  knife5= 47.5%  maxConsecLoss=57
  stabilising & bounced    n=  6347  win= 35.9%  avgR=+0.296  medR=-1.00  PF= 1.47  MAE=-1.51  knife5= 45.9%  maxConsecLoss=85

_test_
  stabilising & near low   n=  2885  win= 38.0%  avgR=+0.297  medR=-1.00  PF= 1.49  MAE=-1.52  knife5= 44.6%  maxConsecLoss=34
  stabilising & bounced    n=  4228  win= 40.0%  avgR=+0.310  medR=-1.00  PF= 1.54  MAE=-1.45  knife5= 42.3%  maxConsecLoss=70


## 4. Gate ablation — independent value of each current gate

Applied to the wide net. Each line = that gate set, train then test.

**wide net (no gates)**
  train n= 96001  win= 36.8%  avgR=+0.302  medR=-1.00  PF= 1.49  MAE=-1.53  knife5= 41.2%  maxConsecLoss=131
  test  n= 50665  win= 38.3%  avgR=+0.292  medR=-1.00  PF= 1.49  MAE=-1.51  knife5= 41.0%  maxConsecLoss=93

**only G1_ema200_up>=5**
  train n= 60446  win= 36.2%  avgR=+0.299  medR=-1.00  PF= 1.48  MAE=-1.56  knife5= 42.7%  maxConsecLoss=100
  test  n= 33581  win= 38.8%  avgR=+0.326  medR=-1.00  PF= 1.55  MAE=-1.53  knife5= 41.9%  maxConsecLoss=122

**only G2_price_vs_ema200**
  train n= 41707  win= 37.3%  avgR=+0.442  medR=-1.00  PF= 1.71  MAE=-1.74  knife5= 47.8%  maxConsecLoss=267
  test  n= 22925  win= 39.0%  avgR=+0.429  medR=-1.00  PF= 1.72  MAE=-1.67  knife5= 45.9%  maxConsecLoss=171

**only G3_range<=20**
  train n= 92972  win= 36.9%  avgR=+0.289  medR=-1.00  PF= 1.47  MAE=-1.52  knife5= 40.8%  maxConsecLoss=129
  test  n= 48547  win= 38.2%  avgR=+0.281  medR=-1.00  PF= 1.47  MAE=-1.50  knife5= 40.7%  maxConsecLoss=133

**only G4_vah<=-4**
  train n= 53513  win= 36.8%  avgR=+0.444  medR=-1.00  PF= 1.71  MAE=-1.80  knife5= 48.9%  maxConsecLoss=153
  test  n= 29389  win= 38.2%  avgR=+0.425  medR=-1.00  PF=  1.7  MAE=-1.74  knife5= 47.9%  maxConsecLoss=178

**ALL current gates**
  train n= 19094  win= 36.3%  avgR=+0.412  medR=-1.00  PF= 1.65  MAE=-1.79  knife5= 50.8%  maxConsecLoss=108
  test  n= 11190  win= 38.4%  avgR=+0.485  medR=-1.00  PF=  1.8  MAE=-1.75  knife5= 48.5%  maxConsecLoss=133

**ALL except G1_ema200_up>=5**
  train n= 33384  win= 37.2%  avgR=+0.446  medR=-1.00  PF= 1.72  MAE=-1.81  knife5= 49.0%  maxConsecLoss=211
  test  n= 17958  win= 38.9%  avgR=+0.471  medR=-1.00  PF= 1.79  MAE=-1.73  knife5= 47.4%  maxConsecLoss=133

**ALL except G2_price_vs_ema200**
  train n= 33789  win= 36.1%  avgR=+0.416  medR=-1.00  PF= 1.66  MAE=-1.78  knife5= 49.7%  maxConsecLoss=105
  test  n= 19500  win= 37.8%  avgR=+0.416  medR=-1.00  PF= 1.68  MAE=-1.75  knife5= 48.5%  maxConsecLoss=129

**ALL except G3_range<=20**
  train n= 20592  win= 36.4%  avgR=+0.420  medR=-1.00  PF= 1.67  MAE=-1.80  knife5= 51.1%  maxConsecLoss=135
  test  n= 12252  win= 38.5%  avgR=+0.493  medR=-1.00  PF= 1.82  MAE=-1.75  knife5= 48.6%  maxConsecLoss=144

**ALL except G4_vah<=-4**
  train n= 20742  win= 36.0%  avgR=+0.392  medR=-1.00  PF= 1.62  MAE=-1.77  knife5= 50.4%  maxConsecLoss=109
  test  n= 12249  win= 38.5%  avgR=+0.463  medR=-1.00  PF= 1.77  MAE=-1.71  knife5= 47.7%  maxConsecLoss=94

**G2 lower-bound variants (with G1,G3,G4 held):**
  [ -30, 3] train n= 19217  win= 36.3%  avgR=+0.414  medR=-1.00  PF= 1.66  MAE=-1.80  knife5= 50.9%  maxConsecLoss=119
  [ -30, 3] test  n= 11290  win= 38.3%  avgR=+0.484  medR=-1.00  PF=  1.8  MAE=-1.75  knife5= 48.7%  maxConsecLoss=135

  [ -30, 8] train n= 27845  win= 36.1%  avgR=+0.401  medR=-1.00  PF= 1.63  MAE=-1.79  knife5= 50.1%  maxConsecLoss=151
  [ -30, 8] test  n= 16136  win= 38.1%  avgR=+0.439  medR=-1.00  PF= 1.72  MAE=-1.74  knife5= 48.4%  maxConsecLoss=154

  [ -25, 3] train n= 19217  win= 36.3%  avgR=+0.414  medR=-1.00  PF= 1.66  MAE=-1.80  knife5= 50.9%  maxConsecLoss=119
  [ -25, 3] test  n= 11290  win= 38.3%  avgR=+0.484  medR=-1.00  PF=  1.8  MAE=-1.75  knife5= 48.7%  maxConsecLoss=135

  [ -25, 8] train n= 27845  win= 36.1%  avgR=+0.401  medR=-1.00  PF= 1.63  MAE=-1.79  knife5= 50.1%  maxConsecLoss=151
  [ -25, 8] test  n= 16136  win= 38.1%  avgR=+0.439  medR=-1.00  PF= 1.72  MAE=-1.74  knife5= 48.4%  maxConsecLoss=154

  [ -20, 3] train n= 19094  win= 36.3%  avgR=+0.412  medR=-1.00  PF= 1.65  MAE=-1.79  knife5= 50.8%  maxConsecLoss=108
  [ -20, 3] test  n= 11190  win= 38.4%  avgR=+0.485  medR=-1.00  PF=  1.8  MAE=-1.75  knife5= 48.5%  maxConsecLoss=133

  [ -20, 8] train n= 27722  win= 36.1%  avgR=+0.400  medR=-1.00  PF= 1.63  MAE=-1.79  knife5= 50.1%  maxConsecLoss=138
  [ -20, 8] test  n= 16036  win= 38.1%  avgR=+0.440  medR=-1.00  PF= 1.72  MAE=-1.74  knife5= 48.3%  maxConsecLoss=149

  [ -15, 3] train n= 18503  win= 36.3%  avgR=+0.403  medR=-1.00  PF= 1.64  MAE=-1.79  knife5= 50.7%  maxConsecLoss=112
  [ -15, 3] test  n= 10810  win= 38.6%  avgR=+0.475  medR=-1.00  PF= 1.79  MAE=-1.72  knife5= 48.2%  maxConsecLoss=104

  [ -15, 8] train n= 27131  win= 36.0%  avgR=+0.394  medR=-1.00  PF= 1.62  MAE=-1.78  knife5= 50.0%  maxConsecLoss=137
  [ -15, 8] test  n= 15656  win= 38.2%  avgR=+0.431  medR=-1.00  PF= 1.71  MAE=-1.72  knife5= 48.0%  maxConsecLoss=116

**G1 threshold variants (with G2[-20,3],G3,G4 held):**
  >= 0 train n= 33384  win= 37.2%  avgR=+0.446  medR=-1.00  PF= 1.72  MAE=-1.81  knife5= 49.0%  maxConsecLoss=211
  >= 0 test  n= 17958  win= 38.9%  avgR=+0.471  medR=-1.00  PF= 1.79  MAE=-1.73  knife5= 47.4%  maxConsecLoss=133

  >= 3 train n= 24445  win= 37.1%  avgR=+0.444  medR=-1.00  PF= 1.71  MAE=-1.78  knife5= 49.6%  maxConsecLoss=146
  >= 3 test  n= 13851  win= 38.7%  avgR=+0.474  medR=-1.00  PF= 1.79  MAE=-1.74  knife5= 48.5%  maxConsecLoss=95

  >= 5 train n= 19094  win= 36.3%  avgR=+0.412  medR=-1.00  PF= 1.65  MAE=-1.79  knife5= 50.8%  maxConsecLoss=108
  >= 5 test  n= 11190  win= 38.4%  avgR=+0.485  medR=-1.00  PF=  1.8  MAE=-1.75  knife5= 48.5%  maxConsecLoss=133

  >= 8 train n= 13068  win= 36.3%  avgR=+0.408  medR=-1.00  PF= 1.65  MAE=-1.81  knife5= 51.4%  maxConsecLoss=78
  >= 8 test  n=  7908  win= 37.3%  avgR=+0.479  medR=-1.00  PF= 1.78  MAE=-1.80  knife5= 49.8%  maxConsecLoss=86

  >=12 train n=  7485  win= 36.4%  avgR=+0.448  medR=-1.00  PF= 1.71  MAE=-1.84  knife5= 53.4%  maxConsecLoss=53
  >=12 test  n=  4881  win= 37.2%  avgR=+0.568  medR=-1.00  PF= 1.91  MAE=-1.92  knife5= 52.0%  maxConsecLoss=47
