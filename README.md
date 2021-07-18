# paa
Python implementation of Keller and Keunig's Protective Asset Allocation (PAA).
Original paper: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2759734

How to use: 
python3 paa.py \
    --amount 100000 \
    --current_fn ib.csv \
    --risky VTI QQQ IWM VGK EWJ VWO VNQ DBC GLD HYG LQD TLT \
    --safe BIL SHV SHY IEI IEF TLT AGG IEF STIP TIP \
    --protection_range 1
