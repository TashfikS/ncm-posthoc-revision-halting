import sys,json,torch,numpy as np; sys.path.insert(0,'.'); import apc_lib as A
import torch.nn.functional as F
out={}
for t in ['sudoku-hard','wmt14']:
    T=A.Task(t,f'ck/{t}',f'data/{t}')
    refs=[T.refs[i] for i in range(64 if t.startswith("sudoku") else 12)]
    with torch.no_grad():
        y=torch.tensor([T.target_ids(r) for r in refs]); z=T.tk.vqvae.encode_z(y)
        idx,zh=T.vq.z_to_idx(z,return_zhats=True)
        d=[ (zh[k]-zh[k-1]).flatten(1).norm(dim=1) for k in range(1,len(zh))]
        rel=[ (dd/(zh[k].flatten(1).norm(dim=1))).mean().item() for k,dd in enumerate(d)]
        rho=[(d[k+1]/d[k]).median().item() for k in range(len(d)-1)]
        # empirical decoder Lipschitz ratio along trajectory: KL(p_K||p_k)/ (0.5||z_K-z_k||^2)
        lpK=F.log_softmax(T.tk.vqvae.decoder(zh[-1]),-1); r=[]
        for k in range(len(zh)-1):
            lpk=F.log_softmax(T.tk.vqvae.decoder(zh[k]),-1)
            kl=(lpK.exp()*(lpK-lpk)).sum(-1).sum(-1)
            r.append((kl/(0.5*(zh[-1]-zh[k]).flatten(1).norm(dim=1)**2)).max().item())
    out[t]={'rel_flux':rel,'rho_median':rho,'LD2_hat_max':r}
    print(t,json.dumps(out[t]))
json.dump(out,open('flux_gt.json','w'))
