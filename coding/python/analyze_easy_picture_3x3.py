import argparse, csv, json, math
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TRANSFORMS = {
    'identity': lambda a: a,
    'rotate_90': lambda a: np.rot90(a, 1),
    'rotate_180': lambda a: np.rot90(a, 2),
    'rotate_270': lambda a: np.rot90(a, 3),
    'mirror_lr': np.fliplr,
    'mirror_ud': np.flipud,
    'mirror_diag': lambda a: a.T,
    'mirror_anti_diag': lambda a: np.fliplr(np.flipud(a)).T,
}

def complexity(image, size=3):
    x = image.astype(float) / 255.0
    ye = np.linspace(0, x.shape[0], size + 1, dtype=int); xe = np.linspace(0, x.shape[1], size + 1, dtype=int)
    density = np.zeros((size,size)); tv = np.zeros((size,size))
    for r in range(size):
        for c in range(size):
            t=x[ye[r]:ye[r+1],xe[c]:xe[c+1]]
            density[r,c]=np.mean(t)
            tv[r,c]=(np.abs(np.diff(t,axis=0)).sum()+np.abs(np.diff(t,axis=1)).sum())/t.size
    dn=density/(density.max() or 1); tn=tv/(tv.max() or 1)
    return density,tv,dn,tn,0.5*(dn+tn)

def orders(ms,ns): return np.asarray([[(ms+r,ns+c) for c in range(3)] for r in range(3)],dtype=int)
def primitive(m,n):
    g=math.gcd(abs(int(m)),abs(int(n)))
    return (0,0) if g==0 else (int(m)//g,int(n)//g)
def groups(o):
    d=defaultdict(list)
    for r,c in np.ndindex(3,3): d[primitive(*o[r,c])].append((r,c,int(o[r,c,0]),int(o[r,c,1])))
    sizes=np.zeros((3,3),int)
    for v in d.values():
        for r,c,*_ in v: sizes[r,c]=len(v)
    return d,sizes
def conjugates(o):
    s={tuple(v) for v in o.reshape(-1,2)}
    return sum((-m,-n) in s for m,n in s)//2
def transforms(size=3):
    b=np.arange(size*size).reshape(size,size)
    return {'identity':b,'rotate_90':np.rot90(b,1),'rotate_180':np.rot90(b,2),'rotate_270':np.rot90(b,3),'mirror_lr':np.fliplr(b),'mirror_ud':np.flipud(b),'mirror_diag':b.T,'mirror_anti_diag':np.fliplr(np.flipud(b)).T}
def physics(o, lam, period):
    step=lam/period; pts=o.reshape(-1,2)*step; center=pts.mean(0); shift=-center
    if np.linalg.norm(shift)>1: return None
    p=pts+shift; radii=np.linalg.norm(p,axis=1); corners=np.array([[x,y] for x in (pts[:,0].min()-step/2,pts[:,0].max()+step/2) for y in (pts[:,1].min()-step/2,pts[:,1].max()+step/2)])+shift
    theta=math.degrees(math.asin(min(1,np.linalg.norm(shift)))); phi=math.degrees(math.atan2(shift[1],shift[0]))
    return dict(step=step,shift=shift,points=p,radii=radii,prop=int(np.sum(radii<=1+1e-12)),captured=int(np.sum(radii<=1+1e-12)),margin=float(1-radii.max()),cell_margin=float(1-np.linalg.norm(corners,axis=1).max()),theta=theta,phi=phi,corner_radii=np.linalg.norm(corners,axis=1))
def best_map(comp,o):
    rank=np.argsort(comp.ravel())[::-1]; top=set(rank[:3]); d,sz=groups(o); axis=(o[:,:,0]==0)|(o[:,:,1]==0); out=[]
    for name,im in transforms().items():
        a=comp.ravel()[im]; at=np.array([i in top for i in im.ravel()]).reshape(3,3); partners=sz-1
        burden=float(np.sum(a*partners)); axisbur=float(np.sum(a[axis]*partners[axis])); topno=int(np.sum(at&(sz==1)))
        score=burden+3*axisbur-0.5*topno
        out.append((score,name,im,burden,axisbur,topno,at))
    return min(out,key=lambda x:(x[0],-x[5]))
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input-file',type=Path,required=True); ap.add_argument('--output-dir',type=Path,required=True); ap.add_argument('--lam',type=float,default=480); ap.add_argument('--period',type=float,default=2000); ap.add_argument('--start-min',type=int,default=-12); ap.add_argument('--start-max',type=int,default=12); args=ap.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=False)
    image=np.asarray(Image.open(args.input_file).convert('L')); den,tv,dn,tn,comp=complexity(image)
    ranks=np.empty(9,int); ranks[np.argsort(comp.ravel())[::-1]]=np.arange(1,10); ranks=ranks.reshape(3,3)
    rows=[]
    for ms in range(args.start_min,args.start_max+1):
      for ns in range(args.start_min,args.start_max+1):
        o=orders(ms,ns)
        if np.any(np.all(o==0,axis=2)): continue
        ph=physics(o,args.lam,args.period)
        if ph is None or ph['prop']<9: continue
        d,sz=groups(o); bm=best_map(comp,o); score,name,im,burden,axisbur,topno,at=bm
        assigned=comp.ravel()[im]; high3=np.argsort(comp.ravel())[::-1][:3]; high_groups=sz.ravel()[im.ravel()][high3]
        axis_count=int(np.sum((o[:,:,0]==0)|(o[:,:,1]==0)))
        rows.append(dict(m_start=ms,m_end=ms+2,n_start=ns,n_end=ns+2,transform=name,orders=';'.join(f'({m},{n})' for m,n in o.reshape(-1,2)),zero_order_count=0,conjugate_pairs=conjugates(o),axis_members=axis_count,group_count=len(d),coupled_groups=sum(len(v)>1 for v in d.values()),largest_group=max(map(len,d.values())),no_multiple=int(np.sum(sz==1)),top3_no_multiple=int(np.sum(high_groups==1)),top3_group_sizes=';'.join(map(str,high_groups)),weighted_multiple_burden=burden,complex_axis_penalty=axisbur,score=score,theta_i=ph['theta'],phi_i=ph['phi'],u_ix=float(ph['shift'][0]),u_iy=float(ph['shift'][1]),propagating=ph['prop'],collected=ph['captured'],worst_center_margin=ph['margin'],full_local_square_margin=ph['cell_margin'],fov_deg=2*math.degrees(math.asin(3*(args.lam/args.period)/2)),center_order=str(tuple(o[1,1])),center_group_size=int(sz[1,1])))
    rows.sort(key=lambda x:(-x['propagating'],x['conjugate_pairs'], -x['top3_no_multiple'],x['weighted_multiple_burden'], -x['full_local_square_margin']))
    with (args.output_dir/'candidate_ranking.csv').open('w',newline='',encoding='utf-8-sig') as f: w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with (args.output_dir/'tile_complexity.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f); w.writerow(['tile','row','column','brightness_density','edge_total_variation','complexity','rank'])
        for r,c in np.ndindex(3,3): w.writerow([f'{r+1},{c+1}',r+1,c+1,den[r,c],tv[r,c],comp[r,c],ranks[r,c]])
    fig,ax=plt.subplots(figsize=(7,7)); ax.imshow(image,cmap='gray'); h,w=image.shape
    for i in range(1,3): ax.axhline(i*h/3,color='red'); ax.axvline(i*w/3,color='red')
    for r,c in np.ndindex(3,3): ax.text((c+.5)*w/3,(r+.5)*h/3,f'{r+1},{c+1}\nC={comp[r,c]:.3f}\nR{ranks[r,c]}',ha='center',va='center',color='yellow',bbox={'facecolor':'black','alpha':.55,'pad':2})
    ax.axis('off'); fig.tight_layout(); fig.savefig(args.output_dir/'tile_complexity_map.png',dpi=180); plt.close(fig)
    top=rows[:5]; rec=top[0]; chosen=TRANSFORMS[rec['transform']](image); ye=np.linspace(0,chosen.shape[0],4,dtype=int); xe=np.linspace(0,chosen.shape[1],4,dtype=int); tiles=[]; pos=[]
    for r,c in np.ndindex(3,3): tiles.append(np.asarray(Image.fromarray(chosen[ye[r]:ye[r+1],xe[c]:xe[c+1]]).resize((500,500),Image.Resampling.NEAREST),dtype=float)/255); pos.append((r,c))
    targets=np.stack(tiles).astype(np.float32)
    levels=np.asarray([0,85,170,255],dtype=np.float32)
    targets=levels[np.argmin(np.abs(targets[...,None]*255-levels),axis=-1)]/255.0
    pair_mat=np.asarray([tuple(map(int,v)) for v in orders(rec['m_start'],rec['n_start']).reshape(-1,2)],dtype=np.int16); sio.savemat(args.output_dir/'easy_picture_3x3_target.mat',{'bw_all':targets,'grid_positions':np.asarray(pos,dtype=np.int16),'pairMat':pair_mat,'mapping_transform':rec['transform']},do_compression=True)
    with (args.output_dir/'summary.json').open('w',encoding='utf-8') as f: json.dump({'input':str(args.input_file),'parameters':{'lambda_nm':args.lam,'period_nm':args.period,'delta_u':args.lam/args.period,'na':1,'n_in':1,'n_out':1},'tile_ranking':sorted([{'tile':f'{r+1},{c+1}','complexity':float(comp[r,c]),'rank':int(ranks[r,c])} for r,c in np.ndindex(3,3)],key=lambda x:x['rank']),'top5':top,'recommendation':rec,'notes':'Centers are propagated; full local square margin is reported separately.'},f,ensure_ascii=False,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x)
    print(json.dumps({'top5':top,'target':str(args.output_dir/'easy_picture_3x3_target.mat')},ensure_ascii=False,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x))
if __name__=='__main__': main()
