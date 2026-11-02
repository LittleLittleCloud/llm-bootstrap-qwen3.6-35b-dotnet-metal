#!/usr/bin/env python3
"""MLX half/affine-8 microbenchmarks matching compare-mlx-kernels fixtures.

Run in an isolated environment with mlx==0.32.4 and numpy. Timed calls build a
fresh lazy result, evaluate it and synchronize. Inputs, quantization, weight
conversion, CPU checks and compilation warmups are outside measured intervals.
All operation boundaries return float32 like the .NET implementation; MLX
matmul/SDPA still round their intermediate output to FP16, noted in the report.
"""
import argparse
import gc
import importlib.metadata
import json
import platform
import time

import mlx.core as mx
import numpy as np


def emit(**record):
    print(json.dumps(record), flush=True)


def bench(name, fn, check, runs, **case):
    mx.synchronize()
    for _ in range(3):
        y = fn()
        mx.eval(y)
        mx.synchronize()
    check(y)
    del y
    # CPU reference checks can leave the GPU idle. Warm again by time, not just
    # call count, so short kernels are not primarily measuring clock ramp-up.
    warm_until = time.perf_counter() + .25
    while time.perf_counter() < warm_until:
        y = fn(); mx.eval(y); mx.synchronize()
    del y
    for run in range(runs):
        start = time.perf_counter_ns()
        y = fn()
        mx.eval(y)
        mx.synchronize()
        elapsed = (time.perf_counter_ns() - start) / 1e6
        emit(type="mlx_benchmark", name=name, run=run, wall_ms=elapsed,
             boundary_dtype="float32", matmul_output_dtype="float16", **case)
        del y


def inputs(m, k):
    i = np.arange(m * k, dtype=np.int64)
    return (((i * 7 + i // 113) % 67 - 33) / 64).astype(np.float32).reshape(m, k)


def weights(n, k, experts=1):
    # Same E4M3 bytes and BF16 block scale 0x3b80 (1/256) as the C# fixture.
    b = np.arange(256, dtype=np.int32)
    e, mant = (b >> 3) & 15, b & 7
    lut = ((np.where(b < 128, 1, -1) * np.where(e == 0, mant / 512., (1 + mant / 8.) * 2.**(e - 7))) / 256).astype(np.float16)
    i = np.arange(n * k, dtype=np.int32)
    result = np.empty((experts, n, k), dtype=np.float16)
    for expert in range(experts):
        raw = ((i * 13 + expert * 3) % 112) | np.where(i % 3 == 0, 128, 0)
        result[expert] = lut[raw].reshape(n, k)
    return result


def dense(runs):
    for m, n, k in [(512,512,2048), (2048,512,2048), (2048,2048,2048), (2048,8192,2048), (2048,2048,4096)]:
        xn, wn = inputs(m, k), weights(n, k)[0]
        x, w = mx.array(xn.astype(np.float16)), mx.array(wn)
        x32 = mx.array(xn)
        qw, scales, biases = mx.quantize(w, group_size=64, bits=8)
        restored = mx.dequantize(qw, scales, biases, group_size=64, bits=8)
        mx.eval(x, w, x32, qw, scales, biases, restored)
        wqn = np.array(restored)
        def checker(reference, name):
            def check(y):
                actual = np.array(y)
                errors = []
                for row in [0, m//2, m-1]:
                    for col in [0, n//2, n-1]:
                        terms = xn[row].astype(np.float64) * reference[col].astype(np.float64)
                        expected = terms.sum()
                        error = abs(float(actual[row,col]) - expected)
                        assert np.isfinite(actual[row,col]) and error <= .001 * abs(expected) + 3e-6 * np.abs(terms).sum() + 1e-5, (name,row,col,error,expected)
                        errors.append(error)
                emit(type="mlx_check", family="dense", name=name, m=m,n=n,k=k,max_abs_error=max(errors))
            return check
        variants = [
            ("fp16_core", lambda: (x @ w.T).astype(mx.float32), wn),
            ("fp16_input_cast", lambda: (x32.astype(mx.float16) @ w.T).astype(mx.float32), wn),
            ("affine8_input_cast", lambda: mx.quantized_matmul(x32.astype(mx.float16), qw, scales, biases, group_size=64, bits=8).astype(mx.float32), wqn),
        ]
        emit(type="quantization_error", family="dense",m=m,n=n,k=k,max_weight_abs_error=float(np.max(np.abs(wqn.astype(np.float32)-wn.astype(np.float32)))))
        for name, fn, ref in variants:
            bench(name, fn, checker(ref,name), runs, family="dense",m=m,n=n,k=k)
        del variants, x,w,x32,qw,scales,biases,restored,xn,wn,wqn
        gc.collect();mx.clear_cache()


def experts(runs):
    m, expert_count = 16384, 256
    for k,n in [(2048,512),(512,2048)]:
        xn, wn = inputs(m,k), weights(n,k,expert_count)
        x32, w = mx.array(xn), mx.array(wn)
        qw, scales, biases = mx.quantize(w,group_size=64,bits=8)
        restored = mx.dequantize(qw,scales,biases,group_size=64,bits=8)
        tokens_np = ((np.arange(m,dtype=np.uint32)*31+7)%2048)
        tokens = mx.array(tokens_np)
        mx.eval(x32,w,qw,scales,biases,restored,tokens)
        # Only these expert rows are required by the CPU spot reference.
        for skew in [False,True]:
            counts = np.tile([1,127],128) if skew else np.full(256,64)
            indices_np = np.repeat(np.arange(256,dtype=np.uint32),counts)
            indices = mx.array(indices_np)
            mx.eval(indices)
            def packed_input():
                return (x32 if k==512 else x32[tokens]).astype(mx.float16)[:,None,:]
            xhalf = packed_input();mx.eval(xhalf)
            def checker(quantized,name):
                def check(y):
                    actual = np.array(y).reshape(m,n)
                    rows = [0,1,63,64,127,m//2,m-1]
                    cols = [0,n//2,n-1]
                    errors=[]
                    for row in rows:
                        expert=int(indices_np[row]);inp=row if k==512 else int(tokens_np[row])
                        for col in cols:
                            weight = np.array(restored[expert,col]) if quantized else wn[expert,col]
                            terms=xn[inp].astype(np.float64)*weight.astype(np.float64)
                            expected=terms.sum();error=abs(float(actual[row,col])-expected)
                            assert np.isfinite(actual[row,col]) and error <= .001*abs(expected)+3e-6*np.abs(terms).sum()+1e-5, (name,skew,row,col,error,expected)
                            errors.append(error)
                    emit(type="mlx_check",family="experts",name=name,k=k,n=n,skew=skew,max_abs_error=max(errors))
                return check
            variants=[
                ("fp16_predecoded",lambda: mx.gather_mm(packed_input(),w.swapaxes(-1,-2),rhs_indices=indices,sorted_indices=True).astype(mx.float32),False),
                ("affine8",lambda: mx.gather_qmm(packed_input(),qw,scales,biases,rhs_indices=indices,sorted_indices=True,group_size=64,bits=8).astype(mx.float32),True),
                ("affine8_core",lambda: mx.gather_qmm(xhalf,qw,scales,biases,rhs_indices=indices,sorted_indices=True,group_size=64,bits=8).astype(mx.float32),True),
            ]
            for name,fn,quantized in variants:
                bench(name,fn,checker(quantized,name),runs,family="experts",m=m,n=n,k=k,skew=skew,sorted_indices=True)
            del variants,xhalf,indices
        del x32,w,qw,scales,biases,restored,tokens,xn,wn
        gc.collect();mx.clear_cache()


def attention_values(count, seed, scale):
    h=np.arange(count,dtype=np.uint32)+np.uint32((seed*0x9e3779b9)&0xffffffff)
    h ^= h >> 16; h *= np.uint32(0x7feb352d); h ^= h >> 15
    return (((h&65535).astype(np.float32)-np.float32(32768))/np.float32(49157)*np.float32(scale))


def attention(runs):
    for history,rows in [(0,512),(0,2048),(8192,2048),(10520,190),(32768,63),(30000,2048)]:
        length=history+rows
        qn=attention_values(rows*4096,17,3).reshape(rows,16,256)
        kn=attention_values(length*512,31,3).reshape(length,2,256).astype(np.float16)
        vn=attention_values(length*512,23,1).reshape(length,2,256).astype(np.float16)
        gn=attention_values(rows*4096,11,1).reshape(rows,16,256)
        q,k,v,g=mx.array(qn),mx.array(kn),mx.array(vn),mx.array(gn)
        mx.eval(q,k,v,g)
        @mx.compile
        def operation(q,k,v,g):
            result=mx.fast.scaled_dot_product_attention(q.astype(mx.float16).transpose(1,0,2)[None],k.transpose(1,0,2)[None],v.transpose(1,0,2)[None],scale=1/16,mask="causal")
            return result[0].transpose(1,0,2).astype(mx.float32)*mx.sigmoid(g)
        def check(y):
            actual=np.array(y);worst=0;worst_relative=0
            # High precision reference with the exact shared FP16 Q/K/V inputs.
            # Unlike ours, fused SDPA need not round normalized probabilities.
            for row in [0,rows//2,rows-1]:
                valid=history+row+1
                for head in [0,7,8,15]:
                    logits=kn[:valid,head//8].astype(np.float64) @ qn[row,head].astype(np.float16).astype(np.float64)/16
                    probs=np.exp(logits-logits.max());probs/=probs.sum()
                    for d in [0,127,255]:
                        expected=(probs @ vn[:valid,head//8,d].astype(np.float64))/(1+np.exp(-float(gn[row,head,d])))
                        error=abs(float(actual[row,head,d])-expected);worst=max(worst,error)
                        assert np.isfinite(actual[row,head,d]) and error < 5e-4, (history,rows,row,head,d,error,expected)
            emit(type="mlx_check",family="attention",history=history,rows=rows,max_abs_error=worst,reference="FP64, FP16 Q/K/V, full precision softmax")
        bench("sdpa",lambda:operation(q,k,v,g),check,runs,family="attention",history=history,rows=rows,length=length,heads=16,kv_heads=2,head_dim=256,kv_dtype="float16",causal=True,includes_q_cast_and_gate=True)
        del q,k,v,g,qn,kn,vn,gn
        gc.collect();mx.clear_cache()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--family',choices=['all','dense','experts','attention'],default='all');parser.add_argument('--runs',type=int,default=9)
    args=parser.parse_args()
    if args.runs<1: parser.error('--runs must be positive')
    mx.set_default_device(mx.gpu)
    emit(type="environment",mlx=importlib.metadata.version('mlx'),numpy=np.__version__,python=platform.python_version(),macos=platform.mac_ver()[0],device=mx.device_info(),runs=args.runs,timing="synchronized wall; 3 validation warmups + 250ms warmup after CPU checks; fresh lazy output per sample",precision_note="FP16 matmul/SDPA output is cast to FP32 at boundary; C# accumulates and outputs FP32")
    for name,fn in [('dense',dense),('experts',experts),('attention',attention)]:
        if args.family in ('all',name):fn(args.runs)

if __name__=='__main__':main()
