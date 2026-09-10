# Reversal audit — supplementary (RESEARCH)

## A. Decline character vs falling knife (current-gate set, train/test)

### rev_max_1d_drop_pct
_train_
  (-60.001, -12.0] n=   208 win=  31% avgR=+0.03 PF=1.04 MAE=-1.62 knife5=  46%
     (-12.0, -8.0] n=  1189 win=  37% avgR=+0.61 PF=1.97 MAE=-1.72 knife5=  51%
      (-8.0, -6.0] n=  2444 win=  37% avgR=+0.37 PF=1.60 MAE=-1.72 knife5=  51%
      (-6.0, -4.0] n=  5367 win=  36% avgR=+0.41 PF=1.65 MAE=-1.83 knife5=  53%
       (-4.0, 0.0] n=  9886 win=  36% avgR=+0.41 PF=1.64 MAE=-1.81 knife5=  50%
_test_
  (-60.001, -12.0] n=   248 win=  38% avgR=+0.31 PF=1.51 MAE=-2.02 knife5=  50%
     (-12.0, -8.0] n=   879 win=  37% avgR=+0.31 PF=1.50 MAE=-1.99 knife5=  53%
      (-8.0, -6.0] n=  1808 win=  38% avgR=+0.44 PF=1.71 MAE=-1.68 knife5=  50%
      (-6.0, -4.0] n=  3060 win=  39% avgR=+0.49 PF=1.81 MAE=-1.77 knife5=  51%
       (-4.0, 0.0] n=  5195 win=  39% avgR=+0.53 PF=1.90 MAE=-1.70 knife5=  46%

### rev_worst_gap_pct
_train_
  (-60.001, -10.0] n=   182 win=  30% avgR=+0.45 PF=1.64 MAE=-1.78 knife5=  46%
     (-10.0, -6.0] n=   736 win=  42% avgR=+0.65 PF=2.14 MAE=-1.51 knife5=  45%
      (-6.0, -3.0] n=  4023 win=  35% avgR=+0.39 PF=1.60 MAE=-1.78 knife5=  52%
      (-3.0, -1.0] n= 11398 win=  36% avgR=+0.37 PF=1.59 MAE=-1.80 knife5=  52%
      (-1.0, 0.01] n=  2755 win=  38% avgR=+0.54 PF=1.89 MAE=-1.87 knife5=  47%
_test_
  (-60.001, -10.0] n=   126 win=  27% avgR=-0.19 PF=0.74 MAE=-1.81 knife5=  63%
     (-10.0, -6.0] n=   581 win=  41% avgR=+0.43 PF=1.74 MAE=-1.59 knife5=  44%
      (-6.0, -3.0] n=  3031 win=  39% avgR=+0.45 PF=1.75 MAE=-1.71 knife5=  51%
      (-3.0, -1.0] n=  6041 win=  37% avgR=+0.50 PF=1.81 MAE=-1.78 knife5=  49%
      (-1.0, 0.01] n=  1411 win=  43% avgR=+0.58 PF=2.06 MAE=-1.72 knife5=  43%

### rev_waterfall_ratio
_train_
     (-0.001, 0.4] n=  1776 win=  35% avgR=+0.29 PF=1.45 MAE=-1.57 knife5=  47%
       (0.4, 0.55] n=  4379 win=  36% avgR=+0.43 PF=1.68 MAE=-1.74 knife5=  50%
       (0.55, 0.7] n=  5891 win=  37% avgR=+0.42 PF=1.68 MAE=-1.77 knife5=  51%
       (0.7, 0.85] n=  4160 win=  36% avgR=+0.42 PF=1.67 MAE=-1.88 knife5=  52%
       (0.85, 3.0] n=  2888 win=  34% avgR=+0.43 PF=1.66 MAE=-1.93 knife5=  54%
_test_
     (-0.001, 0.4] n=  1231 win=  43% avgR=+0.38 PF=1.70 MAE=-1.35 knife5=  41%
       (0.4, 0.55] n=  2395 win=  36% avgR=+0.35 PF=1.55 MAE=-1.80 knife5=  50%
       (0.55, 0.7] n=  3066 win=  37% avgR=+0.57 PF=1.92 MAE=-1.81 knife5=  49%
       (0.7, 0.85] n=  2658 win=  38% avgR=+0.45 PF=1.74 MAE=-1.72 knife5=  50%
       (0.85, 3.0] n=  1840 win=  41% avgR=+0.63 PF=2.10 MAE=-1.85 knife5=  49%

### rev_decline_in_atr
_train_
     (-0.001, 2.5] n=  6894 win=  35% avgR=+0.28 PF=1.43 MAE=-1.54 knife5=  48%
        (2.5, 4.0] n=  5975 win=  37% avgR=+0.36 PF=1.58 MAE=-1.80 knife5=  53%
        (4.0, 5.5] n=  4140 win=  37% avgR=+0.62 PF=1.99 MAE=-1.99 knife5=  51%
        (5.5, 8.0] n=  1918 win=  37% avgR=+0.57 PF=1.91 MAE=-2.26 knife5=  54%
       (8.0, 40.0] n=   167 win=  41% avgR=+0.83 PF=2.43 MAE=-2.01 knife5=  42%
_test_
     (-0.001, 2.5] n=  4445 win=  41% avgR=+0.37 PF=1.65 MAE=-1.51 knife5=  43%
        (2.5, 4.0] n=  3509 win=  38% avgR=+0.40 PF=1.66 MAE=-1.67 knife5=  50%
        (4.0, 5.5] n=  2256 win=  35% avgR=+0.85 PF=2.33 MAE=-2.01 knife5=  54%
        (5.5, 8.0] n=   922 win=  34% avgR=+0.44 PF=1.67 MAE=-2.51 knife5=  54%
       (8.0, 40.0] n=    58 win=  36% avgR=+0.95 PF=2.50 MAE=-2.07 knife5=  60%

## B. 'Abnormal single-day collapse' — worst 1-day drop <= -12% anywhere in the pullback

_train_  collapse:   n=   208 win=  31% avgR=+0.03 PF=1.04 MAE=-1.62 knife5=  46%
_train_  no-collapse:n= 18886 win=  36% avgR=+0.42 PF=1.66 MAE=-1.80 knife5=  51%
_test_  collapse:   n=   248 win=  38% avgR=+0.31 PF=1.51 MAE=-2.02 knife5=  50%
_test_  no-collapse:n= 10942 win=  38% avgR=+0.49 PF=1.81 MAE=-1.74 knife5=  48%

## C. Stabilisation filter — candidate-volume impact

- current-gate signals/month: median 503  (p25 288, p75 652)
- stabilising signals/month: median 256  (p25 168, p75 429)
- stabilising share: mean 61%
- weeks with 0 stabilising signals: 0% of months

- trimmed-gate (no G1/G3) signals/month: median all 868, stabilising 512

## D. Depth x days-since-low (current-gate, test set)

avgR / knife5 by (price_vs_ema200 bin) x (days_since_low bin)

avgR:
lb                (-0.101, 0.5]  (0.5, 2.5]  (2.5, 5.5]  (5.5, 20.1]
db                                                                  
(-25.001, -10.0]           1.28        0.49        0.81         0.46
(-10.0, -5.0]              0.73        0.33        0.45         0.26
(-5.0, 0.0]                2.10        0.53        0.31         0.21
(0.0, 3.0]                 1.98        0.32        0.24         0.36

knife5:
lb                (-0.101, 0.5]  (0.5, 2.5]  (2.5, 5.5]  (5.5, 20.1]
db                                                                  
(-25.001, -10.0]           68.0        59.0        45.0         49.0
(-10.0, -5.0]              67.0        51.0        44.0         49.0
(-5.0, 0.0]                64.0        52.0        41.0         47.0
(0.0, 3.0]                 61.0        49.0        44.0         41.0

## E. Small model — can knife5 be predicted? (logistic, train-fit, test-eval)

- knife5 base rate: train 50.8%  test 48.5%
- logistic AUC: train 0.586  test 0.590
    rev_close_vs_ema20_pct          -0.347
    rev_dist_to_swing_low_20_pct    +0.274
    rev_ema20_slope_5d_pct          +0.258
    rev_higher_low_pct              -0.256
    rev_rsi14                       -0.229
    rev_decline_in_atr              -0.153
    rev_down_up_vol_ratio_12        -0.059
    rev_days_since_low_20           +0.029
    price_vs_ema200_pct             -0.023
    rev_last_5d_return_pct          -0.011

  test rows by predicted-knife-risk quintile (0=safest):
    Q0: n=  2235 win=  40% avgR=+0.21 PF=1.37 MAE=-1.32 knife5=  39%
    Q1: n=  2234 win=  38% avgR=+0.24 PF=1.39 MAE=-1.47 knife5=  44%
    Q2: n=  2235 win=  41% avgR=+0.37 PF=1.63 MAE=-1.53 knife5=  46%
    Q3: n=  2234 win=  39% avgR=+0.71 PF=2.18 MAE=-2.01 knife5=  53%
    Q4: n=  2235 win=  35% avgR=+0.89 PF=2.36 MAE=-2.40 knife5=  62%

  test rows by predicted-R quintile (4=best):
    Q0: n=  2235 win=  42% avgR=+0.36 PF=1.65 MAE=-1.42 knife5=  40%
    Q1: n=  2234 win=  38% avgR=+0.24 PF=1.40 MAE=-1.51 knife5=  43%
    Q2: n=  2235 win=  38% avgR=+0.28 PF=1.47 MAE=-1.48 knife5=  46%
    Q3: n=  2234 win=  38% avgR=+0.43 PF=1.70 MAE=-1.75 knife5=  51%
    Q4: n=  2235 win=  35% avgR=+1.11 PF=2.71 MAE=-2.57 knife5=  62%
