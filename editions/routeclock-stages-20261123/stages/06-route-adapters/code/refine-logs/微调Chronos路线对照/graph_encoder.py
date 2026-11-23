"""Replay the frozen encoder with unchanged FP32 kernels for single-user refreshes."""
import torch
def enable(net):
    enc=net.encoder
    if getattr(enc,'_paired_graph',False):return
    enc._paired_graph=True
    forward=enc.forward
    tables={}
    @torch.inference_mode()
    def paired(inputs_embeds,*,group_ids,attention_mask=None,position_ids=None,output_attentions=False):
        args=dict(inputs_embeds=inputs_embeds,group_ids=group_ids,attention_mask=attention_mask,position_ids=position_ids,output_attentions=output_attentions)
        if inputs_embeds.shape[0] not in (3,10) or position_ids is not None or output_attentions:return forward(**args)
        key=(tuple(inputs_embeds.shape),inputs_embeds.dtype)
        if key not in tables:
            stream=torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                fixed={k:v.clone() if isinstance(v,torch.Tensor) else v for k,v in args.items()}
                for _ in range(2):forward(**fixed)
            torch.cuda.current_stream().wait_stream(stream)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream):
                result=forward(**fixed)
            tables[key]=graph,fixed,result
        graph,fixed,result=tables[key]
        for k in ('inputs_embeds','attention_mask','group_ids'):fixed[k].copy_(args[k])
        graph.replay()
        return result
    enc.forward=paired

if __name__=='__main__':
    from replay_route import *
    setup();route=Routed();w=get_world(1);ts=runtime.warm_trackers(w)
    hs=[tr.history for tr in ts[:12]];ps=[tr.snapshots[-1] for tr in ts[:12]]
    pairs=[]
    def score():
        for h,p in zip(hs,ps,strict=True):
            r=route.batch([h],[p])[0]
            yield route.engine.raw([h])[0][:2],r
    a=list(score());enable(route.engine.net)
    b=list(score());largest=0.
    for arow,brow in zip(a,b,strict=True):
        for ar,br in zip(arow,brow,strict=True):
            for x,y in zip(ar,br,strict=True):
                np.testing.assert_array_equal(x,y);largest=max(largest,float(abs(x-y).max()))
    write(OUT/'graph_validation.json',dict(n=len(hs),arms=['C','CR'],max_abs_difference=largest))
