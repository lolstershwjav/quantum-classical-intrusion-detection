"""Compare matched classical and simulated quantum intrusion classifiers."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.svm import SVC
import torch
from torch import nn

from qids.circuit import Circuit, QuantumNet, fidelity, measure_shots, noisy_overlap
from qids.data import load, process, subsets


def score(y, val_y, val, test):
    cutoff = np.quantile(val[val_y == 0], .95, method="higher")
    decision = test > cutoff
    return dict(ap=float(average_precision_score(y, test)),
                auc=float(roc_auc_score(y,test)),
                balanced_accuracy=float(balanced_accuracy_score(y, decision)),
                fpr=float(decision[y == 0].mean()), recall=float(decision[y == 1].mean()))


def fit_svm(train, y, val, vy, kernel):
    best = None
    train_time = perf_counter()
    for c in (.3, 1., 3.):
        model = SVC(kernel=kernel, C=c).fit(train,y)
        ranking = average_precision_score(vy,model.decision_function(val))
        if best is None or ranking > best[0]:
            best = ranking, model
    return best[1], perf_counter()-train_time


def project_gram(gram, *cross):
    # Clip negative eigenvalues caused by finite-shot sampling.
    values, vectors = np.linalg.eigh(gram)
    supported = vectors[:,values > 1e-7]
    corrected = (supported * values[values > 1e-7]) @ supported.T
    return corrected, *[(g @ supported) @ supported.T for g in cross], supported, int((values < -1e-7).sum())


def kernel_runs(encoded, labels, qubits, encoding, conditions, seed, encode_time, test_encode_time):
    y, vy, ty = labels
    measures = []
    for i,c in enumerate(conditions):
        start=perf_counter()
        rng=np.random.default_rng([seed,i,qubits])
        kw=dict(qubits=qubits,mix=c["mix"],shots=c["shots"],rng=rng)
        train,validation,test=encoded
        gram = noisy_overlap(fidelity(train,train),symmetric=True,**kw)
        validation_gram = noisy_overlap(fidelity(validation,train),**kw)
        fit_gram, fit_val, supported, negative = project_gram(gram,validation_gram)
        classifier,_ = fit_svm(fit_gram,y,fit_val,vy,"precomputed")
        # Time only this reference-state calculation and the classical training.
        fit_time=perf_counter()-start+encode_time
        start=perf_counter()
        test_gram=noisy_overlap(fidelity(test,train),**kw)
        # Use the training-only support already present in the repaired Gram matrix.
        test_features=(test_gram@supported)@supported.T
        ts=classifier.decision_function(test_features)
        infer=perf_counter()-start+test_encode_time
        vals=classifier.decision_function(fit_val)
        measures.append(dict(model="qkernel_"+encoding,condition=c["name"],
                             fit_seconds=fit_time,infer_seconds=infer,
                             parameters=int(classifier.n_support_.sum()),
                             negative_gram_eigenvalues=negative,**score(ty,vy,vals,ts)))
    return measures


class SmallMLP(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.layer=nn.Sequential(nn.Linear(d,6), nn.Tanh(), nn.Linear(6,1))

    def forward(self,x):
        return self.layer(x).ravel()


def train_network(net,x,y,val,vy,epochs,seed):
    rng=torch.Generator().manual_seed(seed)
    opt=torch.optim.AdamW(net.parameters(),lr=.018,weight_decay=.001)
    target=torch.tensor(y,dtype=torch.float32)
    peak=-1
    state=None
    for _ in range(epochs):
        for idx in torch.randperm(len(x),generator=rng).split(96):
            loss=nn.functional.binary_cross_entropy_with_logits(net(x[idx]),target[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.inference_mode():
            metric=average_precision_score(vy,net(val).numpy())
        if metric > peak:
            peak=metric
            state={k:v.detach().clone() for k,v in net.state_dict().items()}
    net.load_state_dict(state)


def quantum_rows(name, xs, labels, qubits, conditions, epochs, seed):
    x,val,test=xs
    y,vy,ty=labels
    torch.manual_seed(seed)
    model = SmallMLP(qubits) if name == "mlp" else QuantumNet(qubits,hybrid=name=="hybrid")
    start=perf_counter()
    train_network(model,x,y,val,vy,epochs,seed)
    fitted=perf_counter()-start
    params=sum(p.numel() for p in model.parameters())
    results=[]
    with torch.inference_mode():
        if name=="mlp":
            before=perf_counter()
            prediction=model(test).numpy()
            elapsed=perf_counter()-before
            return [dict(model=name,condition="ideal",fit_seconds=fitted,infer_seconds=elapsed,
                         parameters=params,**score(ty,vy,model(val).numpy(),prediction))]
        measurement_start=perf_counter()
        states=model.states(test).numpy()
        state_time=perf_counter()-measurement_start
        reference=model.states(val).numpy()
        sign=model.circuit.z.numpy()
        for i,c in enumerate(conditions):
            rng=np.random.default_rng([seed,i,qubits])
            val_p=measure_shots(reference,sign,c["shots"],c["mix"],rng)
            before=perf_counter()
            test_p=measure_shots(states,sign,c["shots"],c["mix"],rng)
            predicted=model.logits(test,torch.tensor(test_p,dtype=torch.float32)).numpy()
            elapsed=perf_counter()-before+state_time
            val_s=model.logits(val,torch.tensor(val_p,dtype=torch.float32)).numpy()
            results.append(dict(model=name,condition=c["name"],fit_seconds=fitted,
                                infer_seconds=elapsed,parameters=params,**score(ty,vy,val_s,predicted)))
    return results


def summarize(rows):
    groups=defaultdict(list)
    rng=np.random.default_rng(672)
    for r in rows:
        groups[r["qubits"],r["train_size"],r["model"],r["condition"]].append(r)
    combinations=[]
    for (q,n,m,c), selected in sorted(groups.items()):
        out={}
        fields=("ap","auc","balanced_accuracy","fpr","recall","fit_seconds","infer_seconds")
        baseline={r["seed"]:r for r in groups[q,n,"rbf_svm","ideal"]}
        for field in fields:
            vals=[r[field] for r in selected]
            lo,hi=np.quantile(rng.choice(vals,(2500,len(vals))).mean(1),[.025,.975])
            out[field]=dict(mean=float(np.mean(vals)),low=float(lo),high=float(hi))
            if field=="ap":
                delta=[r[field]-baseline[r["seed"]][field] for r in selected]
                lower,upper=np.quantile(rng.choice(delta,(2500,len(delta))).mean(1),[.025,.975])
                out["ap_delta_vs_rbf"]=dict(mean=float(np.mean(delta)),low=float(lower),high=float(upper))
        combinations.append(dict(qubits=q,train_size=n,model=m,condition=c,metrics=out))
    return combinations


def run(directory,cfg):
    torch.set_num_threads(1)
    dataset=load(directory)
    rows=[]
    for seed in cfg["seeds"]:
        for n in cfg["train_sizes"]:
            x,y,val,vy,test,ty=subsets(dataset,seed,n,cfg["validation_size"],cfg["test_size"])
            for d in cfg["qubits"]:
                (xc,vc,tc), variance=process(x,val,test,d)
                vals=[]
                for kernel in ("linear","rbf"):
                    model,fitted=fit_svm(xc,y,vc,vy,kernel)
                    before=perf_counter()
                    pred=model.decision_function(tc)
                    elapsed=perf_counter()-before
                    vals.append(dict(model=kernel+"_svm",condition="ideal",fit_seconds=fitted,
                                     infer_seconds=elapsed,parameters=int(model.n_support_.sum()),
                                     **score(ty,vy,model.decision_function(vc),pred)))
                tensors=tuple(torch.tensor(a,dtype=torch.float32) for a in (xc,vc,tc))
                circuit=Circuit(d)
                for encoding in ("angle","phase"):
                    start=perf_counter()
                    encoded_fit=tuple(circuit.encode(a,encoding).numpy() for a in tensors[:2])
                    elapsed=perf_counter()-start
                    start=perf_counter()
                    encoded_test=circuit.encode(tensors[2],encoding).numpy()
                    elapsed_test=perf_counter()-start
                    vals.extend(kernel_runs((*encoded_fit,encoded_test),(y,vy,ty),d,encoding,
                                            cfg["conditions"],seed,elapsed,elapsed_test))
                for learner in ("mlp","vqc","hybrid"):
                    vals.extend(quantum_rows(learner,tensors,(y,vy,ty),d,cfg["conditions"],cfg["epochs"],seed))
                rows.extend(dict(seed=seed,qubits=d,train_size=n,pca_variance=variance,**r) for r in vals)
        print(f"Finished seed {seed}",flush=True)
    return dict(config=cfg,source_counts=[len(v) for v in dataset],summary=summarize(rows),runs=rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",type=Path,default=Path("configs/study.json"))
    p.add_argument("--data",type=Path,default=Path("data"))
    p.add_argument("--output",type=Path,default=Path("results/study.json"))
    args=p.parse_args()
    cfg=json.loads(args.config.read_text())
    if any(v<=0 or v%2 for v in [*cfg["train_sizes"],cfg["validation_size"],cfg["test_size"]]):
        raise ValueError("All row counts must be positive and even")
    result=run(args.data,cfg)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    temp=args.output.with_suffix(".tmp")
    temp.write_text(json.dumps(result,separators=(",",":"),allow_nan=False))
    temp.replace(args.output)


if __name__=="__main__":
    main()
