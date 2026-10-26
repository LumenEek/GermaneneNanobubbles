"""Additional partial-OBC calculations with explicit Gaussian/cap size matching."""
from pathlib import Path
from dataclasses import asdict, replace
import hashlib
import json
import csv
import shutil
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import ndtr
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from ribbon_manuscript import ROOT, notebook_definitions, read, label
from figure_export import figure_destination

SOURCE = ROOT / 'notebooks/ribbons_partial_obc/07_gaussian_partial_obc.ipynb'
RESULTS = ROOT / 'results/ribbons_partial_obc'
OUT = RESULTS / '09_gaussian_size_matching/full'


def matching_parameters():
    """Match nominal half-height radius, or volume on the actual periodic ribbon."""
    meta = json.loads((RESULTS/'08_spherical_cap_partial_obc/full/parameters.json').read_text())
    h, radius = meta['parameters']['height'], meta['parameters']['radius']
    length, width = meta['actual_length_nm'], meta['actual_atomic_width_nm']
    y0 = meta['bubble_center_nm'][1]
    sphere = (radius**2+h**2)/(2*h)
    half_radius = np.sqrt(radius**2/2+h**2/4)
    sigma_width = half_radius/np.sqrt(2*np.log(2))

    def cap_cross_section(dy):
        a = np.sqrt(max(radius**2-dy**2,0))
        b = np.sqrt(sphere**2-dy**2)
        return b*b*np.arcsin(a/b)-a*(sphere-h)

    full_volume = np.pi*h*(3*radius**2+h**2)/6
    np.testing.assert_allclose(quad(cap_cross_section,-radius,radius,epsabs=1e-10)[0],full_volume,rtol=1e-10)
    cap_volume = quad(cap_cross_section,max(-radius,-y0),min(radius,width-y0),epsabs=1e-10)[0]

    def gaussian_volume(sigma):
        images = int(np.ceil(8*sigma/length))+1
        norm = sum(np.exp(-(m*length)**2/(2*sigma*sigma)) for m in range(-images,images+1))
        # Integral of the periodized Gaussian over one full period and the open width.
        return 2*np.pi*h*sigma*sigma/norm*(ndtr((width-y0)/sigma)-ndtr(-y0/sigma))

    sigma_volume = brentq(lambda s:gaussian_volume(s)-cap_volume,radius/3,2*radius,xtol=1e-13)
    np.testing.assert_allclose(gaussian_volume(sigma_volume),cap_volume,rtol=1e-11)
    np.testing.assert_allclose(h*np.exp(-half_radius**2/(2*sigma_width**2)),h/2,rtol=1e-13)
    np.testing.assert_allclose(np.sqrt(sphere*sphere-half_radius**2)-(sphere-h),h/2,rtol=1e-13)
    rows=[]
    for key,title,sigma in [('original','Original Gaussian',radius/3),
                            ('volume_matched','Volume-matched Gaussian',sigma_volume),
                            ('width_matched','Width-matched Gaussian',sigma_width)]:
        mid = sum(np.exp(-((.5-m)*length)**2/(2*sigma*sigma)) for m in range(-10,11))
        norm = sum(np.exp(-(m*length)**2/(2*sigma*sigma)) for m in range(-10,11))
        rows.append(dict(key=key,title=title,profile='gaussian',radius_parameter_nm=3*sigma,
                         sigma_nm=sigma,half_height_radius_nm=sigma*np.sqrt(2*np.log(2)),
                         ribbon_volume_nm3=gaussian_volume(sigma),periodic_midpoint_height_fraction=mid/norm))
    rows.append(dict(key='cap',title='Spherical cap',profile='spherical_cap',radius_parameter_nm=radius,
                     sigma_nm=None,half_height_radius_nm=half_radius,ribbon_volume_nm3=cap_volume,
                     periodic_midpoint_height_fraction=0.))
    return dict(height_nm=h,cap_radius_nm=radius,sphere_radius_nm=sphere,
                actual_length_nm=length,actual_width_nm=width,center_nm=meta['bubble_center_nm'],
                cap_full_plane_volume_nm3=full_volume,cases=rows)


def validate(data, flat):
    for key in ('xy','edge_mask','energy','reduced_k','primitive_reduced_k','gap_window','map_energies'):
        np.testing.assert_array_equal(data[key],flat[key])
    for key in ('A','Aedge','A_unfolded','Aedge_unfolded','ldos','maps','gap_weight'):
        assert np.isfinite(data[key]).all() and data[key].min()>=0
    for key, unfolded, local in [('A','A_unfolded',data['ldos']),('Aedge','Aedge_unfolded',data['ldos'][data['edge_mask']])]:
        np.testing.assert_allclose(data[key].mean(axis=1),data[unfolded].mean(axis=1),rtol=1e-9,atol=1e-12)
        np.testing.assert_allclose(local.mean(axis=0)/2,data[key].mean(axis=1),rtol=1e-9,atol=1e-12)
    lo,hi=data['gap_window'];e=data['eigenvalues']
    weight=ndtr((hi-e)/.005)-ndtr((lo-e)/.005)
    np.testing.assert_allclose(data['gap_weight'].sum(),weight.sum()/81,rtol=1e-9)
    np.testing.assert_allclose(data['gap_weight'][data['edge_mask']].sum(),(weight*data['edge_weights']).sum()/81,rtol=1e-9)


def run_study():
    OUT.mkdir(parents=True,exist_ok=True)
    settings=matching_parameters()
    print('Size matching:',json.dumps(settings,indent=2),flush=True)
    flat=read(RESULTS/'07_gaussian_partial_obc/full/flat.npz')
    data={'original':read(RESULTS/'07_gaussian_partial_obc/full/edge_gaussian_h1p5_R2p5.npz'),
          'cap':read(RESULTS/'08_spherical_cap_partial_obc/full/edge_cap_h1p5_R2p5.npz')}
    report={'settings':settings,'cases':{},'new_calculations_full':True}
    ns=notebook_definitions(SOURCE)
    assert ns['NK']==81 and len(ns['ENERGY'])==1201 and not ns['QUICK']
    for case in settings['cases'][1:3]:
        key=case['key'];radius=case['radius_parameter_nm']
        ns['RADIUS_NM']=radius
        ns['p']=replace(ns['p'],radius=radius,profile='gaussian')
        directory=OUT/key;directory.mkdir(exist_ok=True);ns['out']=directory
        metadata=dict(case=case,parameters=asdict(ns['p']),pbc=[True,False,False],
                      dimensions=[ns['nx'],ns['ny'],1],nk=81,energy_points=1201,
                      gaussian_sigma_ev=.005,center_nm=settings['center_nm'],quick=False,
                      model_sha256=hashlib.sha256((ROOT/'model.py').read_bytes()).hexdigest(),
                      notebook_source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest())
        path=directory/f'{key}.npz';config=directory/'parameters.json'
        if path.exists() and config.exists() and json.loads(config.read_text())==metadata:
            print('Reuse verified matching cache:',key,flush=True)
        else:
            ribbon=ns['build_ribbon'](settings['height_nm'])
            np.testing.assert_allclose(ribbon['xy'],flat['xy'],atol=1e-12)
            np.testing.assert_array_equal(ribbon['edge'],flat['edge_mask'])
            print(f'Calculating {key}: R_G={radius:.9f} nm',flush=True)
            ns['calculate'](ribbon,key)
            config.write_text(json.dumps(metadata,indent=2)+'\n')
        data[key]=read(path)
        validate(data[key],flat)
    for case in settings['cases']:
        key=case['key'];d=data[key];validate(d,flat)
        report['cases'][key]=dict(total_gap_weight=float(d['gap_weight'].sum()),
                                 lower_edge_gap_weight=float(d['gap_weight'][d['edge_mask']].sum()),
                                 gap_weight_difference=float((d['gap_weight']-flat['gap_weight']).sum()))
    report['flat']=dict(total_gap_weight=float(flat['gap_weight'].sum()),
                        lower_edge_gap_weight=float(flat['gap_weight'][flat['edge_mask']].sum()))
    (OUT/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    with (OUT/'size_matching.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(settings['cases'][0]));writer.writeheader();writer.writerows(settings['cases'])
    print('Validated gap-weight comparison:',json.dumps(report['cases'],indent=2),flush=True)
    return settings,flat,data


def plot_study(settings, flat, data):
    plt.rcParams.update({'font.family':'serif','font.serif':['Computer Modern Roman'],
                         'text.usetex':True,'font.size':12,'axes.labelsize':12,
                         'legend.fontsize':9,'xtick.labelsize':10,'ytick.labelsize':10})
    rows=settings['cases'];keys=[r['key'] for r in rows]
    colors=['#777777','#377eb8','#009e73','#d95f02'];styles=[':', '-.', '-', '-']
    short=['Original Gaussian','Volume match','Width match','Spherical cap']
    manifest=[]

    def save(fig,name):
        p=OUT/f'POBC_size_{name}.png';fig.savefig(p,dpi=300)
        target=figure_destination(p.name);shutil.copy2(p,target)
        manifest.append(dict(filename=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
        plt.show()

    fig,axes=plt.subplots(1,2,figsize=(11.5,4.3),layout='constrained')
    h=settings['height_nm'];R=settings['cap_radius_nm'];S=settings['sphere_radius_nm'];L=settings['actual_length_nm']
    x=np.linspace(-L/2,L/2,1001)
    for row,color,style,name in zip(rows,colors,styles,short):
        if row['key']=='cap':
            z=np.zeros_like(x);inside=abs(x)<R
            z[inside]=np.sqrt(S*S-x[inside]**2)-(S-h)
        else:
            sigma=row['sigma_nm'];m=np.arange(-10,11)
            z=h*np.exp(-(x[:,None]-m[None,:]*L)**2/(2*sigma*sigma)).sum(axis=1)/np.exp(-(m*L)**2/(2*sigma*sigma)).sum()
        axes[0].plot(x,z,color=color,ls=style,label=name,lw=1.6)
        y0=settings['center_nm'][1];y=np.linspace(0,5.5,701);dy=y-y0
        if row['key']=='cap':
            z=np.zeros_like(y);inside=abs(dy)<R;z[inside]=np.sqrt(S*S-dy[inside]**2)-(S-h)
        else:z=h*np.exp(-dy*dy/(2*row['sigma_nm']**2))
        axes[1].plot(y,z,color=color,ls=style,lw=1.6)
    axes[0].axhline(h/2,color='black',lw=.7,ls='--')
    axes[0].set(xlabel=r'$x-x_0$ (nm)',ylabel='height (nm)');axes[0].legend()
    axes[1].axvline(y0,color='black',lw=.7,ls='--');axes[1].set(xlabel='y (nm)',ylabel='height (nm)')
    label(axes[0],'a',r'Along the ribbon at $y=y_0$');label(axes[1],'b',r'Across the ribbon at $x=x_0$')
    save(fig,'profiles')

    fig,axes=plt.subplots(1,2,figsize=(11.5,4.3),layout='constrained')
    for i,(ax,key,title) in enumerate(zip(axes,['A','Aedge'],['Full-ribbon mean DOS','Lower-edge mean LDOS'])):
        ax.plot(flat['energy'],flat[key].mean(axis=1),'k--',label='Flat',lw=1)
        for k,color,style,name in zip(keys,colors,styles,short):
            ax.plot(flat['energy'],data[k][key].mean(axis=1),color=color,ls=style,label=name,lw=1.5)
        ax.axvspan(-.08,.08,color='grey',alpha=.1)
        ax.set(xlabel='energy (eV)',ylabel=r'density per spin orbital (eV$^{-1}$)',xlim=(-.3,.3))
        label(ax,chr(97+i),title)
    axes[0].legend(fontsize=8)
    save(fig,'mean_spectra')

    fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained',sharex=True,sharey=True)
    positive=np.mod(flat['primitive_reduced_k'],1);order=np.argsort(positive);positive=positive[order]
    kplot=np.r_[positive[-1]-1,positive,positive[0]+1]
    vmax=max(data[k]['Aedge_unfolded'].max() for k in keys)
    for i,(ax,k,name) in enumerate(zip(axes.flat,keys,short)):
        z=data[k]['Aedge_unfolded'][:,order];z=np.column_stack([z[:,-1],z,z[:,0]])
        im=ax.pcolormesh(kplot,flat['energy'],z,shading='auto',cmap='magma',vmin=0,vmax=vmax,rasterized=True)
        for bound in (-.08,.08):ax.axhline(bound,color='cyan',ls=':',lw=.7)
        ax.set(xlim=(0,1),ylim=(-.3,.3),xlabel=r'$ka/(2\pi)$',ylabel='energy (eV)')
        ax.set_xticks([0,1/3,.5,2/3,1],[r'$\Gamma$',r'$K$',r'$1/2$',r"$K'$",r'$\Gamma$'])
        label(ax,chr(97+i),name)
    fig.colorbar(im,ax=list(axes.flat),label=r'lower-edge density per spin orbital (eV$^{-1}$)',shrink=.85)
    save(fig,'edge_spectra')

    fig,axes=plt.subplots(3,3,figsize=(12,10),layout='constrained',sharex=True,sharey=True)
    xy=flat['xy']
    map_keys=['volume_matched','width_matched','cap']
    map_names=['Volume-matched Gaussian','Width-matched Gaussian','Spherical cap']
    for col,energy in enumerate(flat['map_energies']):
        peak=max(data[k]['maps'][:,col].max() for k in keys)
        norm=LogNorm(peak*1e-4,peak,clip=True)
        for row,(k,name) in enumerate(zip(map_keys,map_names)):
            ax=axes[row,col]
            im=ax.scatter(*xy.T,c=data[k]['maps'][:,col],s=8,cmap='magma',norm=norm)
            ax.axhline(1,color='cyan',lw=.6,ls=':')
            ax.set(xlabel='x (nm)',ylabel='y (nm)',aspect='equal')
            label(ax,chr(97+3*row+col),f'{name}\n{1000*energy:g} meV')
        fig.colorbar(im,ax=list(axes[:,col]),label=r'LDOS (eV$^{-1}$ per atom)',shrink=.75)
    save(fig,'ldos_maps')
    (OUT/'figure_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':
    plot_study(*run_study())
