import json,numpy as np,sys
from scipy import stats
for d in sys.argv[1:]:
    R=json.load(open(f'kout/dagger_{d}/results/sudoku-{d}/results.json'))
    b=R['base_test']; print('==',d,'NCM',{k:round(v,2) for k,v in b['scores'].items()},'done',R.get('done'))
    for arm,S in R['arms'].items():
        ex=[S[s]['test']['scores']['exact'] for s in S]; ce=[S[s]['test']['scores']['cell'] for s in S]
        ps=[]
        for s in S:
            a_=np.array(b['items']['exact'])>.5; c=np.array(S[s]['test']['items']['exact'])>.5
            n01,n10=int((~a_&c).sum()),int((a_&~c).sum()); ps.append(stats.binomtest(n01,n01+n10).pvalue if n01+n10 else 1.0)
        tp=stats.ttest_1samp(ex,b['scores']['exact']).pvalue if len(ex)>1 and np.std(ex)>0 else 1
        print(arm,len(S),'exact %.2f±%.2f'%(np.mean(ex),np.std(ex,ddof=1) if len(ex)>1 else 0),[round(x,2) for x in ex],'cell %.2f'%np.mean(ce),'mcn',[round(p,4) for p in ps],'t %.3g'%tp,'best',[S[s]['best'] for s in S],'train_s',round(np.mean([S[s]['train_s'] for s in S])))
