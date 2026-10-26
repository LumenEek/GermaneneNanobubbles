"""Validate existing full partial-OBC results and export Section 6 comparisons."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/honeycomb-mpl')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
from pathlib import Path
import ast
import hashlib
import json
import shutil
import sys
import numpy as np
from scipy.special import ndtr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm
from figure_export import figure_destination

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'results/ribbons_partial_obc/comparison'
CASES = [('07_gaussian_partial_obc', 'edge_gaussian_h1p5_R2p5.npz', 'Gaussian'),
         ('08_spherical_cap_partial_obc', 'edge_cap_h1p5_R2p5.npz', 'Spherical cap')]


def read(path):
    with np.load(path) as data:
        return {k:data[k].copy() for k in data.files}


def notebook_definitions(path):
    notebook = json.loads(path.read_text())
    ns = {'__name__':'ribbon_validation'}
    code = [c for c in notebook['cells'] if c['cell_type']=='code']
    os.chdir(ROOT)
    exec(''.join(code[0]['source']), ns)
    for cell in code[1:]:
        tree = ast.parse(''.join(cell['source']))
        functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
        if functions:
            exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), ns)
    return ns


def validate(data, metadata):
    assert metadata['quick'] is False and metadata['nk']==81
    assert metadata['pbc']==[True,False,False]
    assert metadata['parameters']['height']==1.5
    assert metadata['parameters']['radius']==2.5
    assert metadata['parameters']['lambda_r2_mev']==0
    assert data['ldos'].shape==(1500,1201)
    assert data['A_unfolded'].shape==(1201,2025)
    for key in ('A','Aedge','A_unfolded','Aedge_unfolded','ldos','maps','gap_weight'):
        assert np.isfinite(data[key]).all() and data[key].min()>=0, key
    np.testing.assert_allclose(data['ldos'].mean(axis=0)/2,data['A'].mean(axis=1),rtol=1e-9,atol=1e-12)
    np.testing.assert_allclose(data['ldos'][data['edge_mask']].mean(axis=0)/2,
                               data['Aedge'].mean(axis=1),rtol=1e-9,atol=1e-12)
    for folded,unfolded in [('A','A_unfolded'),('Aedge','Aedge_unfolded')]:
        np.testing.assert_allclose(data[folded].mean(axis=1),data[unfolded].mean(axis=1),rtol=1e-9,atol=1e-12)
    e=data['eigenvalues']; sigma=metadata['gaussian_sigma_ev']; lo,hi=data['gap_window']
    integrated=ndtr((hi-e)/sigma)-ndtr((lo-e)/sigma)
    np.testing.assert_allclose(data['gap_weight'].sum(),integrated.sum()/81,rtol=1e-9)
    np.testing.assert_allclose(data['gap_weight'][data['edge_mask']].sum(),
                              (integrated*data['edge_weights']).sum()/81,rtol=1e-9)
    return {'total_gap_weight_states':float(data['gap_weight'].sum()),
            'lower_strip_gap_weight_states':float(data['gap_weight'][data['edge_mask']].sum())}


def label(ax, letter, title):
    ax.set_title(f'({letter})',loc='left',fontweight='bold')
    ax.set_title(title)


def save(fig,name,manifest):
    path=OUT/f'POBC_{name}.png'
    fig.savefig(path,dpi=300)
    plt.close(fig)
    dest=figure_destination(path.name)
    shutil.copy2(path,dest)
    manifest['figures'].append({'path':dest.name,
                                'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    report={'cases':{},'figures':[]}
    flat=None; bubbles=[]; metadata=None
    for stem,filename,name in CASES:
        directory=ROOT/'results/ribbons_partial_obc'/stem/'full'
        meta=json.loads((directory/'parameters.json').read_text())
        reference=read(directory/'flat.npz'); data=read(directory/filename)
        if flat is None:
            flat=reference;metadata=meta
            report['cases']['Flat']=validate(flat,meta)
        else:
            for key in flat:
                np.testing.assert_array_equal(flat[key],reference[key])
            for key in ('dimensions','bubble_center_nm','nk','gaussian_sigma_ev','edge_depth_nm'):
                assert metadata[key]==meta[key]
        for key in ('xy','edge_mask','energy','reduced_k','primitive_reduced_k','map_energies','gap_window'):
            np.testing.assert_array_equal(flat[key],data[key])
        metrics=validate(data,meta)
        ns=notebook_definitions(ROOT/'notebooks/ribbons_partial_obc'/f'{stem}.ipynb')
        ribbon=ns['build_ribbon'](meta['parameters']['height'])
        np.testing.assert_allclose(ribbon['positions'],data['positions'],atol=1e-12)
        np.testing.assert_allclose(ribbon['height'],data['height'],atol=1e-12)
        np.testing.assert_array_equal(ribbon['edge'],data['edge_mask'])
        # An independent re-evaluation at one stored k anchors saved eigenvalues
        # to the current geometry, hopping law and notebook settings.
        ik=40
        ham=ns['bloch_hamiltonian'](ribbon,data['reduced_k'][ik])
        vals,_=ns['window_eigensystem'](ham[::2,::2].tocsr())
        selected=(data['state_k']==ik)&(data['state_spin']==0)
        saved=np.sort(data['eigenvalues'][selected])
        np.testing.assert_allclose(vals,saved,atol=1e-8,rtol=1e-7)
        metrics.update(geometry_matches=True,checked_k_index=ik,
                       eigenvalue_max_difference_eV=float(abs(vals-saved).max()),
                       source_file=str((directory/filename).relative_to(ROOT)),
                       source_sha256=hashlib.sha256((directory/filename).read_bytes()).hexdigest())
        report['cases'][name]=metrics;bubbles.append(data)
    cases=[flat,*bubbles]; names=['Flat','Gaussian','Spherical cap']
    plt.rcParams.update({'font.family':'serif','font.serif':['Computer Modern Roman'],
                         'text.usetex':True,'font.size':12,'axes.labelsize':12,
                         'legend.fontsize':10,'xtick.labelsize':10,'ytick.labelsize':10})
    xy=flat['xy']; center=metadata['bubble_center_nm']; radius=2.5

    fig,axes=plt.subplots(1,3,figsize=(12,4.2),layout='constrained',sharex=True,sharey=True)
    for i,(ax,data,name) in enumerate(zip(axes,cases,names)):
        im=ax.scatter(*xy.T,c=data['height'],s=9,cmap='viridis',vmin=0,vmax=1.5)
        if i:ax.add_patch(plt.Circle(center,radius,fill=False,color='darkorange'))
        ax.axhline(1,color='tomato',ls=':',lw=1)
        ax.set(xlabel='x (nm)',ylabel='y (nm)',aspect='equal',ylim=(-.3,10.5))
        label(ax,chr(97+i),name)
    fig.colorbar(im,ax=list(axes),label='height (nm)',shrink=.85)
    save(fig,'geometry',report)

    fig,axes=plt.subplots(3,2,figsize=(11.5,10),layout='constrained',sharex=True,sharey=True)
    positive=np.mod(flat['primitive_reduced_k'],1);order=np.argsort(positive);positive=positive[order]
    k=np.r_[positive[-1]-1,positive,positive[0]+1]
    for col,(key,title) in enumerate([('A_unfolded','Full ribbon'),('Aedge_unfolded','Lower edge')]):
        vmax=max(d[key].max() for d in cases)
        for row,(data,name) in enumerate(zip(cases,names)):
            ax=axes[row,col];z=data[key][:,order];z=np.column_stack([z[:,-1],z,z[:,0]])
            im=ax.pcolormesh(k,flat['energy'],z,shading='auto',cmap='magma',vmin=0,vmax=vmax,rasterized=True)
            for valley in (1/3,2/3):ax.axvline(valley,color='white',ls='--',lw=.7,alpha=.7)
            for bound in (-.08,.08):ax.axhline(bound,color='cyan',ls=':',lw=.6)
            ax.set(xlim=(0,1),ylim=(-.3,.3),xlabel=r'$ka/(2\pi)$',ylabel='energy (eV)')
            ax.set_xticks([0,1/3,.5,2/3,1],[r'$\Gamma$',r'$K$',r'$1/2$',r"$K'$",r'$\Gamma$'])
            label(ax,chr(97+2*row+col),f'{name}: {title}')
        fig.colorbar(im,ax=list(axes[:,col]),label=r'density per spin orbital (eV$^{-1}$)',shrink=.85)
    save(fig,'spectral_functions',report)

    fig,axes=plt.subplots(3,3,figsize=(12,10),layout='constrained',sharex=True,sharey=True)
    for col,energy in enumerate(flat['map_energies']):
        peak=max(d['maps'][:,col].max() for d in cases);norm=LogNorm(peak*1e-4,peak,clip=True)
        for row,(data,name) in enumerate(zip(cases,names)):
            ax=axes[row,col]
            im=ax.scatter(*xy.T,c=data['maps'][:,col],s=8,cmap='magma',norm=norm)
            ax.axhline(1,color='cyan',ls=':',lw=.6)
            ax.set(xlabel='x (nm)',ylabel='y (nm)',aspect='equal')
            label(ax,chr(97+3*row+col),f'{name}\n{energy*1000:g} meV')
        fig.colorbar(im,ax=list(axes[:,col]),label=r'LDOS (eV$^{-1}$ per atom)',shrink=.75)
    save(fig,'gap_ldos',report)

    fig=plt.figure(figsize=(12,8.5),layout='constrained');grid=fig.add_gridspec(2,6)
    top=[fig.add_subplot(grid[0,2*i:2*i+2]) for i in range(3)]
    bottom=[fig.add_subplot(grid[1,:3]),fig.add_subplot(grid[1,3:])]
    peak=max(d['gap_weight'].max() for d in cases);norm=LogNorm(peak*1e-4,peak,clip=True)
    for i,(ax,data,name) in enumerate(zip(top,cases,names)):
        im=ax.scatter(*xy.T,c=data['gap_weight'],s=8,cmap='magma',norm=norm)
        ax.set(xlabel='x (nm)',ylabel='y (nm)',aspect='equal');label(ax,chr(97+i),name)
    fig.colorbar(im,ax=top,label='states per atom',shrink=.8)
    diffs=[d['gap_weight']-flat['gap_weight'] for d in bubbles]
    limit=max(abs(d).max() for d in diffs);norm=SymLogNorm(limit*1e-3,vmin=-limit,vmax=limit)
    for i,(ax,diff,name) in enumerate(zip(bottom,diffs,names[1:])):
        im=ax.scatter(*xy.T,c=diff,s=8,cmap='RdBu_r',norm=norm)
        ax.set(xlabel='x (nm)',ylabel='y (nm)',aspect='equal');label(ax,chr(100+i),name+' minus flat')
    fig.colorbar(im,ax=bottom,label='change in states per atom',shrink=.8,
                 ticks=[-.1,-.01,0,.01,.1])
    save(fig,'integrated_gap_ldos',report)

    fig,axes=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
    for i,(ax,key,title) in enumerate(zip(axes,['A','Aedge'],['Full-ribbon mean DOS','Lower-edge mean LDOS'])):
        for data,name,color,ls in zip(cases,names,['black','#377eb8','#d95f02'],['--','-','-']):
            ax.plot(flat['energy'],data[key].mean(axis=1),label=name,color=color,ls=ls,lw=1.4)
        ax.axvspan(-.08,.08,color='grey',alpha=.12)
        ax.set(xlabel='energy (eV)',ylabel=r'density per spin orbital (eV$^{-1}$)',xlim=(-.3,.3))
        label(ax,chr(97+i),title);ax.legend()
    save(fig,'mean_spectra',report)
    report['flat_references_identical']=True
    (OUT/'validation_and_figures.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
