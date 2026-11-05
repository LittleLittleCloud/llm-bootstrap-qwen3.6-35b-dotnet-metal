#!/usr/bin/env python3
"""Validate captures, export Perfetto/Chrome traces and publication-size timelines."""
import argparse
from collections import Counter, defaultdict
import csv
import importlib.util
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('summary',Path(__file__).with_name('summarize-current-gap.py'))
grouping=importlib.util.module_from_spec(spec);spec.loader.exec_module(grouping)
LABELS={'ours-fp16':'Ours · FP16 KV','splash-int8':'Splash · INT8 KV','splash-bf16':'Splash · BF16 KV'}
COLORS={'MoE incl shared / routing / combine':'#d68037','Dense + norm / residual':'#507ac4',
        'DeltaNet + preparation':'#2e9b80','Attention + preparation':'#a668bd',
        'Draft context':'#cf5e7d','Embedding / head':'#8b8e95'}
SHORT={'MoE incl shared / routing / combine':'MoE + routing/combine','Dense + norm / residual':'Dense + norm/residual',
       'DeltaNet + preparation':'DeltaNet','Attention + preparation':'Attention','Draft context':'Draft','Embedding / head':'Embed / head'}
LANES=['CPU dispatch','CPU encode','CPU commit','CPU wait','GPU command','GPU kernel']

def read(p):
    return [json.loads(line) for line in p.read_text().splitlines() if line.startswith('{')]

def union(events):
    end=-float('inf');total=0
    for e in sorted(events,key=lambda e:e['start_ns']):
        a,b=e['start_ns'],e['start_ns']+e['duration_ns']
        total+=max(0,b-max(a,end));end=max(end,b)
    return total/1e6

def load(out,name):
    path=out/name
    stdout=read(path/'stdout.jsonl');runs=[r for r in stdout if r['type']=='prefill']
    assert [r['run'] for r in runs]==list(range(6))
    if name.startswith('ours'):
        assert all(r['first_token']==22 for r in runs)
        raw=read(path/'timeline.jsonl')
        assert len(raw)==2 and [r['run'] for r in raw]==[4,5]
        modes={('sampled' if r['sampled'] else 'cpu'):r['events'] for r in raw}
        group=grouping.ours_group
    else:
        checks=[r for r in stdout if r['type']=='check']
        assert len(checks)==6 and all(r['first_token']==22 and r['repeat_equal'] for r in checks)
        modes={mode:[e for row in read(path/f'intervals.jsonl.{suffix}.jsonl') for e in row['events']] for mode,suffix in [('cpu','cpu'),('sampled','gpu')]}
        for row in read(path/'intervals.jsonl'):
            modes['sampled'] += [dict(name=e['kernel'],lane='GPU kernel',start_ns=e['start_ns'],duration_ns=e['duration_ns'],command=row['sequence']) for e in row['events']]
        group=grouping.splash_group
    warm=[r['wall_ms'] for r in runs if 1<=r['run']<=3]
    summary={'warm_wall_ms':warm,'median_wall_ms':statistics.median(warm),
             'median_tok_s':10520/(statistics.median(warm)/1000),'first_token':22,'modes':{}}
    for mode,events in modes.items():
        start=min(e['start_ns'] for e in events if e['lane']=='CPU encode')
        commands=[e for e in events if e['lane']=='GPU command']
        dispatches=[e for e in events if e['lane']=='CPU dispatch']
        kernels=[e for e in events if e['lane']=='GPU kernel']
        assert all(e['duration_ns']>=0 and e['start_ns']>=start-1e6 for e in events)
        cmd_by_id=defaultdict(list)
        for e in commands: cmd_by_id[e['command']].append(e)
        for e in kernels:
            # Calibrated encoder timestamps must fall within their command's GPU range.
            assert any(e['start_ns']>=c['start_ns']-100000 and e['start_ns']+e['duration_ns']<=c['start_ns']+c['duration_ns']+100000 for c in cmd_by_id[e['command']]), (name,e)
            e['group']=group(e['name'])
        assert not kernels or len(kernels)==len(dispatches), (name,len(kernels),len(dispatches))
        if name.startswith('ours'):
            r=runs[4 if mode=='cpu' else 5]
            assert len(commands)==r['timing']['command_buffers'] and len(dispatches)==r['timing']['dispatches']
        end=max(e['start_ns']+e['duration_ns'] for e in commands)
        for e in events:
            e['relative_ms']=(e['start_ns']-start)/1e6
        cpu_ms=sum(e['duration_ns'] for e in dispatches)/1e6
        command_rows=[]
        sorted_cmds=sorted(commands,key=lambda e:e['start_ns'])
        for i,c in enumerate(sorted_cmds):
            ds=[e for e in dispatches if e['command']==c['command']]
            es=[e for e in events if e['command']==c['command'] and e['lane']=='CPU encode']
            command_rows.append({'index':i,'id':c['command'],'gpu_start_ms':c['relative_ms'],'gpu_ms':c['duration_ns']/1e6,
                 'dispatches':len(ds),'cpu_dispatch_ms':sum(e['duration_ns'] for e in ds)/1e6,
                 'cpu_encode_ms':sum(e['duration_ns'] for e in es)/1e6,
                 'gap_from_previous_gpu_end_ms':None if not i else (c['start_ns']-sorted_cmds[i-1]['start_ns']-sorted_cmds[i-1]['duration_ns'])/1e6})
        summary['modes'][mode]={'wall_ms':runs[4 if mode=='cpu' else 5]['wall_ms'],
            'overhead_percent':(runs[4 if mode=='cpu' else 5]['wall_ms']/summary['median_wall_ms']-1)*100,
            'dispatches':len(dispatches),'commands':len(commands),'cpu_dispatch_ms':cpu_ms,
            'cpu_dispatch_median_us':statistics.median(e['duration_ns'] for e in dispatches)/1000,
            'cpu_encode_ms':sum(e['duration_ns'] for e in events if e['lane']=='CPU encode')/1e6,
            'cpu_commit_ms':sum(e['duration_ns'] for e in events if e['lane']=='CPU commit')/1e6,
            'gpu_command_union_ms':union(commands),'gpu_kernel_union_ms':union(kernels),
            'gpu_kernel_sum_ms':sum(e['duration_ns'] for e in kernels)/1e6,
            'timeline_span_ms':(end-start)/1e6,'command_detail':command_rows}
        # Independent process clocks are normalized to each first CPU encode.
        chrome=[]
        for lane in LANES:
            chrome.append({'ph':'M','name':'thread_name','pid':1,'tid':LANES.index(lane)+1,'args':{'name':lane}})
        for e in events:
            chrome.append({'ph':'X','pid':1,'tid':LANES.index(e['lane'])+1,'name':e['name'],
                           'cat':e.get('group',e['lane']),'ts':e['relative_ms']*1000,'dur':e['duration_ns']/1000,
                           'args':{'command':e['command'],'absolute_start_ns':e['start_ns']}})
        (out/f'{name}-{mode}.trace.json').write_text(json.dumps({'displayTimeUnit':'ms','traceEvents':chrome},separators=(',',':'))+'\n')
    assert summary['modes']['cpu']['dispatches']==summary['modes']['sampled']['dispatches']
    return modes,summary

def bars(ax,events,lane,y,color=None,origin=0,window=None):
    verts=[];colors=[]
    for e in events:
        if e['lane']!=lane: continue
        a=e['relative_ms']-origin;b=a+e['duration_ns']/1e6
        if window and (b<window[0] or a>window[1]):continue
        verts.append([(a,y-.30),(b,y-.30),(b,y+.30),(a,y+.30)])
        colors.append(color or COLORS[e['group']])
    ax.add_collection(PolyCollection(verts,facecolors=colors,edgecolors='none'))

def style(ax):
    ax.spines[['top','right','left']].set_visible(False)
    ax.tick_params(axis='y',length=0,pad=10)
    ax.grid(axis='x',color='#dfe4ea',linewidth=.65);ax.set_axisbelow(True)
    ax.set_facecolor('#fafbfc')

def save(fig,out,name):
    for ext in ['png','svg']:
        fig.savefig(out/f'{name}.{ext}',dpi=170,bbox_inches='tight',facecolor='white')
    plt.close(fig)

def render(out,data,summaries):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11})
    fig,axes=plt.subplots(3,1,figsize=(15,8.7),sharex=True)
    maximum=max(s['modes']['cpu']['timeline_span_ms'] for s in summaries.values())
    for ax,(name,modes) in zip(axes,data.items()):
        events=modes['cpu'];s=summaries[name];m=s['modes']['cpu']
        bars(ax,events,'CPU dispatch',2.5,'#e1a12a');bars(ax,events,'CPU encode',1.5,'#366ac3')
        bars(ax,events,'CPU commit',1.5,'#df503a');bars(ax,events,'GPU command',.5,'#248a7e')
        # At a seconds-wide scale, true CPU durations are subpixel. Mark each
        # burst's onset, without stretching its measured duration.
        for lane,y,color in [('CPU encode',1.5,'#366ac3'),('CPU dispatch',2.5,'#e1a12a')]:
            seen=set()
            for e in events:
                if e['lane']==lane and e['command'] not in seen:
                    seen.add(e['command']);ax.plot(e['relative_ms'],y,marker='|',markersize=15,color=color,markeredgewidth=1.3)
        ax.set_yticks([.5,1.5,2.5],['GPU commands','CPU encode / commit','CPU dispatch bursts'])
        ax.set_ylim(-.15,3.1);ax.set_xlim(0,maximum*1.015);style(ax)
        ax.set_title(f"{LABELS[name]}   |   warm median {s['median_wall_ms']:.1f} ms   |   {m['dispatches']:,} dispatches / {m['commands']} commands",loc='left',pad=12,fontweight='bold')
        for c in m['command_detail']:
            if c['gpu_ms']>120:
                ax.text(c['gpu_start_ms']+c['gpu_ms']/2,.5,f"{c['gpu_ms']:.0f} ms\n{c['dispatches']} calls",ha='center',va='center',color='white',fontsize=8)
        if name=='ours-fp16':
            c=m['command_detail'][-1]
            ax.annotate('8-token tail: 59 ms',xy=(c['gpu_start_ms']+c['gpu_ms']/2,.80),xytext=(c['gpu_start_ms']-290,1.05),
                        fontsize=8,color='#263c5c',arrowprops={'arrowstyle':'->','color':'#263c5c','lw':.8})
    axes[-1].set_xlabel('Elapsed from first CPU encode (ms); each process starts at 0')
    fig.suptitle('10,520-token prefill · normal command boundaries',x=.01,ha='left',fontsize=19,fontweight='bold')
    fig.text(.01,.925,'CPU trace with normal encoders. Ticks mark CPU burst starts (subpixel durations); GPU bars use command-buffer timestamps.',fontsize=10,color='#526173')
    fig.text(.01,.015,'Warm latency = 3 untraced repeats. Shown CPU-trace runs have measured overhead; native weights and numeric formats differ.',fontsize=9,color='#526173')
    fig.subplots_adjust(top=.87,bottom=.095,hspace=.50,left=.15,right=.99)
    save(fig,out,'prefill-10k-overview')

    fig,axes=plt.subplots(3,2,figsize=(16,10),gridspec_kw={'width_ratios':[1,1.45]})
    for row,(name,modes) in enumerate(data.items()):
        normal=modes['cpu'];sampled=modes['sampled'];s=summaries[name]
        # Show the first substantial GPU command, skipping zeroing-only commands.
        command=next(c for c in s['modes']['cpu']['command_detail'] if c['dispatches']>100)
        did=command['id'];es=[e for e in normal if e['command']==did]
        enc=min(e['relative_ms'] for e in es if e['lane']=='CPU encode')
        ax=axes[row,0]
        bars(ax,es,'CPU dispatch',1.5,'#e1a12a',enc)
        bars(ax,es,'CPU encode',.5,'#366ac3',enc)
        bars(ax,es,'CPU commit',.5,'#df503a',enc)
        ax.set_yticks([.5,1.5],['Encode / commit','Dispatch calls']);ax.set_ylim(-.1,2.1)
        ax.set_xlim(0,max(command['cpu_encode_ms']*1.05,1));style(ax)
        ax.set_title(f"{LABELS[name]}\nFirst large command: {command['dispatches']} calls, encode {command['cpu_encode_ms']:.2f} ms",loc='left',fontweight='bold')
        ax.set_xlabel('CPU time from command encoding start (ms)')
        ax=axes[row,1]
        kernels=[e for e in sampled if e['lane']=='GPU kernel']
        first=min(e['relative_ms'] for e in kernels)
        bars(ax,sampled,'GPU kernel',.5,None,first,(0,45))
        ax.set_xlim(0,45);ax.set_ylim(-.15,1.15);ax.set_yticks([.5],['GPU encoders']);style(ax)
        ax.set_title(f"First 45 ms of GPU kernels · separate counter capture\n{len(kernels):,} sampled dispatches; wall overhead {s['modes']['sampled']['overhead_percent']:+.1f}%",loc='left',fontweight='bold')
        ax.set_xlabel('GPU time from first sampled kernel (ms)')
    fig.suptitle('Dispatch encoding and GPU kernel sequence',x=.01,ha='left',fontsize=19,fontweight='bold')
    fig.legend(handles=[Patch(color=c,label=SHORT[g]) for g,c in COLORS.items()],loc='lower center',ncol=6,frameon=False,bbox_to_anchor=(.52,.015),fontsize=9)
    fig.text(.01,.065,'Left/right use separate captures and independent origins. Encoder intervals may overlap; colors are attribution, not additive wall costs.',fontsize=9,color='#526173')
    fig.subplots_adjust(top=.89,bottom=.14,hspace=.8,wspace=.29,left=.09,right=.99)
    save(fig,out,'dispatch-and-kernels-zoom')

    fig,axes=plt.subplots(3,1,figsize=(15,6.5),sharex=True)
    maximum=max(s['modes']['sampled']['timeline_span_ms'] for s in summaries.values())
    for ax,(name,modes) in zip(axes,data.items()):
        events=modes['sampled'];s=summaries[name]['modes']['sampled']
        bars(ax,events,'GPU kernel',.5)
        bars(ax,events,'CPU encode',1.5,'#263c5c')
        ax.set_yticks([.5,1.5],['GPU kernels','CPU encode']);ax.set_ylim(-.1,2.1);ax.set_xlim(0,maximum*1.01);style(ax)
        ax.set_title(f"{LABELS[name]}   |   sampled wall {s['wall_ms']:.1f} ms ({s['overhead_percent']:+.1f}% vs warm median)",loc='left',fontweight='bold')
    axes[-1].set_xlabel('Elapsed from first CPU encode (ms)')
    fig.suptitle('10K prefill · sampled GPU kernel timeline',x=.01,ha='left',fontsize=19,fontweight='bold')
    fig.legend(handles=[Patch(color=c,label=SHORT[g]) for g,c in COLORS.items()],loc='lower center',ncol=6,frameon=False,bbox_to_anchor=(.52,.0),fontsize=9)
    fig.text(.01,.905,'One counter encoder per dispatch; original command-buffer boundaries retained. This instrumentation changes timing.',fontsize=10,color='#526173')
    fig.subplots_adjust(top=.82,bottom=.13,hspace=.65,left=.12,right=.99)
    save(fig,out,'prefill-10k-kernels')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('out',type=Path);args=ap.parse_args();out=args.out.resolve()
    data={};summaries={}
    for name in LABELS:data[name],summaries[name]=load(out,name)
    (out/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
    with (out/'commands.csv').open('w') as f:
        rows=[dict(engine=name,mode=mode,**c) for name,s in summaries.items() for mode,m in s['modes'].items() for c in m['command_detail']]
        writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    render(out,data,summaries)
    print(json.dumps({name:{k:v for k,v in s.items() if k!='modes'} | {'cpu':{k:v for k,v in s['modes']['cpu'].items() if k!='command_detail'},'sampled':{k:v for k,v in s['modes']['sampled'].items() if k!='command_detail'}} for name,s in summaries.items()},indent=2))

if __name__=='__main__':main()
