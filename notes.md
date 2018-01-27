Case 1: Without Backwashing

## Basic cleanup

### 1. Used pandas to ffill nan values in data

### 2. Rename columns
#### Here are the columns that we have:
```
Index(['date & time', 'time', 'perm_mass', 'perm_time', 'perm_cond', 'perm_totalflow', 'perm_flowrate', 'reject_vol', 'backwash water', 'backwash '
, 'reject_vol_ml', 'recovery', 'reject_tds', 'membr_reject_tds', 'membr_reject_flowrate_ml', 'membr_reject_flowrate_l', 'tank_tds', 'membr_feed_tds
', 'membr_feed_flow', 'membr_feed_pressure', 'inlet_flow', 'inlet_tds', 'ro/bw'], dtype='object')
```

#### Delete the invalid rows
After rename: 
            date & time time perm_mass perm_time perm_cond perm_totalflow perm_flowrate total rej backwash water backwash   ...  reject_tds reject_flowrate_ml reject_flowrate_l tank_tds membr_tds membr_flow membr_pressure inlet_flow inlet_tds ro/bw
1   03.04.2017 02:39 PM    0    143.06     30.28        12              0         16.96      0.00            NaN         0  ...         292               1035              62.1      264       264      79.06             75      16.96       119    RO
2   03.04.2017 02:49 PM   10    137.31     30.59        11            2.5         16.11      0.00            NaN         0  ...         471               1116             66.96      375       375      83.07             75      16.11        86    RO

#### Delete unused columns
`'date & time', 'backwash water', 'backwash'`


## Basic analytics
### 1. Prelim graphing
#### Against time: