import json,sys,numpy as np
from scipy import stats
def mcn(a,b):
    a=np.array(a)>.5;b=np.array(b)>.5;n01,n10=int((~a&b).sum()),int((a&~b).sum())
    return (stats.binomtest(n01,n01+n10).pvalue if n01+n10 else 1.0),n01,n10
def wil(a,b):
    d=np.array(b)-np.array(a)
    return 1.0 if np.allclose(d,0) else float(stats.wilcoxon(a,b).pvalue)
for p in sys.argv[1:]:
    R=json.load(open(p)); PK={'exact':'exact','bleu':'bleu','rouge':'rougeL'}
    b=R['base_test']; pk='exact' if 'exact' in b['scores'] else ('bleu' if 'bleu' in b['scores'] else 'rougeL')
    ik={'exact':'exact','bleu':'sbleu','rougeL':'rougeL'}[pk]
    print('==',p,'NCM',round(b['scores'][pk],2))
    for v in ('loop','loop_sel','proj','proj_sel','sel_halt'):
        S=R['seeds']; x=[S[s][v]['scores'][pk] for s in S]
        tp=stats.ttest_1samp(x,b['scores'][pk]).pvalue if np.std(x)>0 else 1.0
        if pk=='exact': ps=[mcn(b['items'][ik],S[s][v]['items'][ik]) for s in S]; pstr=[f"{q[0]:.2f}({q[1]}/{q[2]})" for q in ps]
        else: pstr=[f"{wil(b['items'][ik],S[s][v]['items'][ik]):.2g}" for s in S]
        print(f" {v:9s} {np.mean(x):.2f}±{np.std(x,ddof=1):.2f} {[round(y,2) for y in x]} t={tp:.3g} paired={pstr} lev={np.mean([S[s][v]['levels'] for s in S]):.2f}")
    print(' s_loop',[S[s]['s_loop'] for s in S],'s_proj',[S[s]['s_proj'] for s in S])
