import json,glob,sys,numpy as np
from scipy import stats
d=sys.argv[1]; root=sys.argv[2]
out={}
for arm in ('tf','dagger','dagger_c'):
    fs=sorted(glob.glob(f'{root}/results/sudoku-{d}-{arm}-s*/results.json'))
    if not fs: continue
    rows=[]
    for f in fs:
        R=json.load(open(f)); b=R['base_test']; S=R['arms'][arm]
        for s,v in S.items():
            a_=np.array(b['items']['exact'])>.5; c=np.array(v['test']['items']['exact'])>.5
            n01,n10=int((~a_&c).sum()),int((a_&~c).sum())
            rows.append((s,v['test']['scores']['exact'],stats.binomtest(n01,n01+n10).pvalue if n01+n10 else 1.0,v['best'],n01,n10))
    ex=[r[1] for r in rows]; base=b['scores']['exact']
    tp=stats.ttest_1samp(ex,base).pvalue if np.std(ex)>0 else 1.0
    out[arm]={'n':len(rows),'mean':float(np.mean(ex)),'sd':float(np.std(ex,ddof=1)) if len(ex)>1 else 0.0,'t_p':float(tp),'base':base,
              'per_seed':[{'seed':r[0],'exact':r[1],'mcnemar_p':r[2],'best':r[3],'gain':r[4],'loss':r[5]} for r in rows]}
    print(d,arm,len(rows),'%.2f±%.2f'%(out[arm]['mean'],out[arm]['sd']),'t=%.3g'%tp,[(r[0],round(r[1],2),round(r[2],3),r[3]) for r in rows],'NCM',round(base,2))
json.dump(out,open(f'{root}/dagsum.json','w'),indent=1)
