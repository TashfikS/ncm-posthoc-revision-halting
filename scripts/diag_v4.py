import sys,json,torch; sys.path.insert(0,'.'); import apc_lib as A, apc_v4 as V
out={}
T=A.Task('sudoku-hard','ck/sudoku-hard','data/sudoku-hard')
ii=list(range(500,532)); pf=torch.tensor([T.prefix(T.items[i]) for i in ii]); refs=[T.ref_str(T.refs[i]) for i in ii]
R=json.load(open('kout/sudoku_v1/results/sudoku-hard/results.json')); h=R['hidden_1.2M']
r=V.generate_v4(T,pf); out['proj']={'n':32,'ncm':A.corpus_scores(T,A.decode_latent(T,r['z']),refs)[0]}
for v in ['v1','v2']:
    C=A.LatentCorrector(T.d,len(T.levels)-1,hidden=h); C.load_state_dict(torch.load(f'kout/sudoku_{v}/results/sudoku-hard/corrector_seed42.pt',map_location='cpu')); C.eval()
    for p in (False,True):
        r=V.generate_v4(T,pf,corrector=C,project=p); out['proj'][f'{v}_proj{int(p)}']=A.corpus_scores(T,A.decode_latent(T,r['z']),refs)[0]
out['margin']={}
for t in ['sudoku-easy','sudoku-medium','sudoku-hard']:
    T=A.Task(t,f'ck/{t}',f'data/{t}'); pf=torch.tensor([T.prefix(T.items[i]) for i in range(64)])
    S=A.apc_generate(T,pf,record=True); fin=T.tk.vqvae.decoder(S['zs'][-1]).argmax(-1); prev=None; rows=[]
    for k,z in enumerate(S['zs']):
        mu,stab,arg=V.margin_stats(T,z,prev)
        rows.append({'level':k+1,'eq_final_pct':100*(arg==fin).all(1).float().mean().item(),'mu_median':mu.median().item(),
                     'flips_vs_prev':(None if prev is None else (arg!=prev).float().sum(1).mean().item())}); prev=arg
    out['margin'][t]=rows
json.dump(out,open('diag_v4.json','w'),indent=1); print(json.dumps(out['proj']))
for t,rw in out['margin'].items(): print(t,max(r['eq_final_pct'] for r in rw[:-1]),rw[-2]['mu_median'],rw[-1]['flips_vs_prev'])
