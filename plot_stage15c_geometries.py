"""Show frozen input material geometries; no PDE solve or filtering."""
from pathlib import Path
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
from stage15c_cases import from_spec


def run(raw,output):
    raw,output=Path(raw),Path(output)
    if output.exists():raise FileExistsError('Choose a new figure path')
    frozen=json.loads((raw/'freeze.json').read_text())
    materials=[from_spec(c['metadata']) for c in frozen['cases']]
    x=(np.arange(128)+.5)/128;xx,yy=np.meshgrid(x,x)
    fig,axes=plt.subplots(4,6,figsize=(14.4,10.3),constrained_layout=True)
    for ax,case in zip(axes.ravel(),materials):
        im=ax.imshow(case.coefficient(xx,yy),origin='lower',extent=(0,1,0,1),
                     cmap='viridis',norm=LogNorm(1e-3,1),interpolation='nearest')
        ax.add_patch(Rectangle((.25-.18,.5-.18),.36,.36,fill=False,color='white',lw=1.3))
        ax.add_patch(Rectangle((.75-.18,.5-.18),.36,.36,fill=False,color='#ff6b6b',lw=1.3))
        ax.set_title(f"{case.name} / contrast {case.metadata['contrast']}",fontsize=9)
        ax.set_xticks([0,.5,1]);ax.set_yticks([0,.5,1]);ax.tick_params(labelsize=7)
    fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.75,label='permeability k (common log scale)')
    fig.suptitle('Stage15C: 24 prespecified material geometries\nWhite: source support; red: observation support',fontsize=13)
    fig.savefig(output,dpi=160)
    fig.savefig(output.with_suffix('.pdf'))
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--input',default='results/stage15c_classical')
    parser.add_argument('--output',default='results/stage15c_geometries.png')
    args=parser.parse_args();run(args.input,args.output)
