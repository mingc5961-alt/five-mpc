from grudata import *
from neural import Expert,FUT,angular
import torch,json
from functools import lru_cache
from copy import deepcopy
from paid_full import model
torch.set_num_threads(1)
def forward(net,x,mask):
    inp=torch.cat((x*mask[...,None],mask[...,None].float()),-1).reshape(-1,4,16)
    valid=mask.reshape(-1,4,2).any(2)
    idx=torch.argsort((~valid).long()*4+torch.arange(4),dim=1)
    tokens=net.embed(inp).gather(1,idx[...,None].expand(-1,-1,24))
    seq,_=net.core(tokens)
    return net.head(seq[torch.arange(len(seq)),valid.sum(1)-1]).reshape(-1,20,3)*(3*FUT[:,None])
def data(split):
    ds=[dict(np.load(D/f"{split}{k}.npz")) for k in range(SPLITS[split])]
    return {s:torch.as_tensor(np.concatenate([d[s] for d in ds])).to(torch.float32) for s in ds[0]}|{"world":torch.from_numpy(np.concatenate([np.full(len(d["n"]),i) for i,d in enumerate(ds)]))}
def errs(x,y):
    d=angular(x)-angular(y)
    return torch.rad2deg(torch.linalg.vector_norm(torch.atan2(d.sin(),d.cos()),dim=-1))
def historybin(n):
    return np.digitize(n,[1,3,8],right=True)
def train():
    torch.manual_seed(418)
    tr,val=data("train"),data("val")
    real=tr["x"][tr["mask"].bool()]
    mu=real.mean(0);sd=real.std(0).clamp_min(.001)
    for a in (tr,val):a["x"]=((a["x"]-mu)/sd).clamp(-6,6);a["mask"]=a["mask"].bool()
    net=Expert("gru",8); opt=torch.optim.AdamW(net.parameters(),lr=.002)
    best=float("inf")
    hist=[]
    for ep in range(35):
        for ix in torch.randperm(len(tr["n"])).split(160):
            pred=tr["imm"][ix]+forward(net,tr["x"][ix],tr["mask"][ix])
            e=(errs(pred,tr["y"][ix])/(.5+FUT)).mean(1)
            loss=(e*torch.where(tr["n"][ix]<=3,3.,1.)).mean()
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
        with torch.no_grad():
            e=errs(val["imm"]+forward(net,val["x"],val["mask"]),val["y"])
            score=float((e/(.5+FUT)*torch.where(val["n"][:,None]<=3,3.,1.)).mean())
        hist.append(score)
        if score<best:best=score;bestep=ep;state=deepcopy(net.state_dict())
        elif ep-bestep>=6:break
    net.load_state_dict(state)
    with torch.no_grad():diff=(errs(val["imm"]+forward(net,val["x"],val["mask"]),val["y"])-errs(val["imm"],val["y"])).numpy()
    bins=historybin(val["n"].numpy())
    # Six contiguous forecast bands: decision requires a gain in 3 of 4 validation worlds.
    hb=np.digitize(FUT.numpy(),[1,2,3,5,8],right=True)
    gates=np.zeros((4,20))
    diagnostic=[]
    for b in range(4):
        for j in range(6):
            scores=np.array([diff[(bins==b)&(val["world"].numpy()==w)][:,hb==j].mean() for w in range(4)])
            enable=scores.mean()<-.01 and np.sum(scores<0)>=3
            gates[b,hb==j]=enable
            diagnostic.append([b,j,scores.tolist(),bool(enable)])
    torch.save(dict(state=state,mu=mu,sd=sd),D/"cold_gru.pt")
    (D/"residual.json").write_text(json.dumps(dict(gates=gates.tolist(),validation=diagnostic,curve=hist,epoch=bestep+1,train=len(tr["n"]),val=len(val["n"]))))
    print("training",len(tr["n"]),len(val["n"]),bestep+1,gates[:,[0,2,4,6,10,16]],flush=True)
class Cold:
    def __init__(self,mode):
        ref=model(); self.profile,self.kappa,self.prior=ref.profile,ref.kappa,ref.prior
        ck=torch.load(D/"cold_gru.pt",weights_only=True)
        self.net=Expert("gru",8).eval();self.net.load_state_dict(ck["state"])
        self.mu,self.sd=ck["mu"],ck["sd"];self.gates=np.array(json.loads((D/"residual.json").read_text())["gates"])
        self.n=0;self.mode=mode
    @torch.inference_mode()
    def residual(self,h):
        x,mask=encode(h)
        y=forward(self.net,((torch.from_numpy(x)[None]-self.mu)/self.sd).clamp(-6,6),torch.from_numpy(mask)[None])[0].numpy()
        return y if self.mode=="Always" else y*self.gates[int(historybin(self.n)),:,None]
if __name__=="__main__":
    import time
    while not (D/"val3.npz").exists() or len(list(D.glob("val*.npz")))<4:time.sleep(2)
    train()

