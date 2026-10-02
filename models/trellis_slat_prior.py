"""Frozen endpoint-conditioned TRELLIS SLat local projection prior."""
import json, math
import torch
from torch import nn
from torch.nn import functional as F
from modules import sparse as sp
TRELLIS_PRIOR_REPO="microsoft/TRELLIS-image-large"
TRELLIS_PRIOR_CHECKPOINT="ckpts/slat_flow_img_dit_L_64l8p2_fp16"
TRELLIS_PRIOR_METRIC_NAMES=(
 "trellis_prior_loss","trellis_prior_projection_loss","trellis_prior_projection_delta_rms",
 "trellis_prior_projection_tangent_delta_rms","trellis_prior_projection_radial_fraction",
 "trellis_prior_projection_raw_cosine_z","trellis_prior_projection_delta_clipped_rms",
 "trellis_prior_projection_clip_fraction","trellis_prior_projection_x0_rms",
 "trellis_prior_velocity_rms","trellis_prior_t_mean","trellis_prior_scale_loss",
 "trellis_prior_scale_active_fraction","trellis_prior_scale_reference_rms",
 "trellis_prior_scale_ratio","trellis_prior_guard_loss","trellis_prior_guard_fraction",
 "trellis_prior_guard_low","trellis_prior_guard_high","trellis_prior_endpoint_rms",
 "trellis_prior_sample_endpoint_rms_ratio","trellis_prior_delta_endpoint_rms_ratio",
 "trellis_prior_src1_fraction","trellis_prior_slat_stat_loss",
 "trellis_prior_slat_stat_active_fraction","trellis_prior_slat_mean_deviation",
 "trellis_prior_slat_std_ratio")

def _build_original_slat_flow(config_args):
    from models.structured_latent_flow import SLatFlowModel
    if "separate_cond" in config_args or "separate_cond_gate" in config_args:
        raise ValueError("original TRELLIS SLat config expected")
    class Original(SLatFlowModel):
        def __init__(self,**kw):
            super().__init__(**kw,separate_cond=False); del self.alpha_embedder
        def forward(self,x,t,cond):
            h=self.input_layer(x).type(self.dtype); te=self.t_embedder(t)
            if self.share_mod: te=self.adaLN_modulation(te)
            te=te.type(self.dtype); cond=cond.type(self.dtype); skips=[]
            for block in self.input_blocks:
                h=block(h,te); skips.append(h.feats)
            if self.pe_mode=="ape": h=h+self.pos_embedder(h.coords[:,1:]).type(self.dtype)
            for block in self.blocks: h=block(h,te,cond)
            for block,skip in zip(self.out_blocks,reversed(skips)):
                h=block(h.replace(torch.cat([h.feats,skip],dim=1)),te) if self.use_skip_connection else block(h,te)
            h=h.replace(F.layer_norm(h.feats,h.feats.shape[-1:])); return self.out_layer(h.type(x.dtype))
    args=dict(config_args); args["use_checkpoint"]=False; return Original(**args)

class TrellisSLatPrior(nn.Module):
    def __init__(self,flow,image_encoder,*,sigma_min=1e-5,t_min=.05,t_max=.20,projection_clip_ratio=.08,
                 tangent_projection=False,stat_anchor_weight=1.,stat_std_low_ratio=.65,
                 stat_std_high_ratio=1.5,stat_mean_tolerance=.5,rms_guard_weight=1.,
                 rms_guard_low_ratio=.4,rms_guard_high_ratio=2.5):
        super().__init__(); self.flow=flow.requires_grad_(False).eval(); self.image_encoder=image_encoder.requires_grad_(False).eval()
        self.sigma_min=float(sigma_min); self.t_min=float(t_min); self.t_max=float(t_max); self.projection_clip_ratio=float(projection_clip_ratio)
        self.tangent_projection=bool(tangent_projection); self.stat_anchor_weight=float(stat_anchor_weight)
        self.stat_std_low_ratio=float(stat_std_low_ratio); self.stat_std_high_ratio=float(stat_std_high_ratio); self.stat_mean_tolerance=float(stat_mean_tolerance)
        self.rms_guard_weight=float(rms_guard_weight); self.rms_guard_low_ratio=float(rms_guard_low_ratio); self.rms_guard_high_ratio=float(rms_guard_high_ratio)
        if not (0<self.t_min<self.t_max<1): raise ValueError("invalid prior t range")
        if not (0<self.stat_std_low_ratio<1<self.stat_std_high_ratio): raise ValueError("invalid SLat std dead-zone")
        if self.rms_guard_low_ratio>=self.rms_guard_high_ratio: raise ValueError("invalid guard ratios")
        self.register_buffer("image_mean",torch.tensor([.485,.456,.406]).view(1,3,1,1),persistent=False)
        self.register_buffer("image_std",torch.tensor([.229,.224,.225]).view(1,3,1,1),persistent=False); self.eval()
    @classmethod
    def from_pretrained(cls,*,device=None,dino_model="dinov2_vitl14_reg",**kw):
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        cp=hf_hub_download(repo_id=TRELLIS_PRIOR_REPO,filename=TRELLIS_PRIOR_CHECKPOINT+".json")
        wp=hf_hub_download(repo_id=TRELLIS_PRIOR_REPO,filename=TRELLIS_PRIOR_CHECKPOINT+".safetensors")
        with open(cp,encoding="utf-8") as f: cfg=json.load(f)
        if cfg.get("name")!="SLatFlowModel": raise ValueError("expected SLatFlowModel checkpoint")
        with torch.random.fork_rng(devices=[]): flow=_build_original_slat_flow(cfg["args"])
        flow.load_state_dict(load_file(wp),strict=True)
        with torch.random.fork_rng(devices=[]): enc=torch.hub.load("facebookresearch/dinov2",dino_model,pretrained=True)
        obj=cls(flow,enc,**kw); return obj.to(device=device) if device is not None else obj
    def train(self,mode=True):
        super().train(False); self.flow.eval(); self.image_encoder.eval(); return self
    @torch.no_grad()
    def encode_image(self,image):
        x=(image.float()-self.image_mean.to(image.device))/self.image_std.to(image.device)
        f=self.image_encoder(x,is_training=True)["x_prenorm"]; return F.layer_norm(f,f.shape[-1:])
    @staticmethod
    def _ids(z): return z.coords[:,0].long()
    @classmethod
    def _stats(cls,z):
        ids=cls._ids(z); feats=z.feats.float(); B=int(z.shape[0]); ms=[]; ss=[]; rs=[]
        for i in range(B):
            f=feats[ids==i]
            if f.numel()==0: raise ValueError("empty SLat item")
            ms.append(f.mean(0)); ss.append(f.std(0,unbiased=False).clamp_min(1e-4)); rs.append(f.square().mean().clamp_min(1e-12).sqrt())
        return torch.stack(ms),torch.stack(ss),torch.stack(rs)
    @classmethod
    def _rms(cls,feats,z):
        ids=cls._ids(z); return torch.stack([feats[ids==i].square().mean().clamp_min(1e-12).sqrt() for i in range(int(z.shape[0]))])
    @classmethod
    def _tangent(cls,d,z):
        ids=cls._ids(z); x=z.feats.detach().float(); out=torch.empty_like(d); rf=[]; co=[]
        for i in range(int(z.shape[0])):
            m=ids==i; dv=d[m].reshape(-1); xv=x[m].reshape(-1); dot=torch.dot(dv,xv); xs=xv.square().sum().clamp_min(1e-12)
            rad=x[m]*(dot/xs); out[m]=d[m]-rad
            rf.append(rad.reshape(-1).square().mean().sqrt()/dv.square().mean().clamp_min(1e-12).sqrt())
            co.append(dot/(dv.square().sum().sqrt()*xs.sqrt()).clamp_min(1e-12))
        return out,torch.stack(rf),torch.stack(co)
    def forward(self,z,*,src1_image,src2_image,alpha,src1_slat,src2_slat,tau=None,noise=None,return_loss_terms=False):
        if not isinstance(z,sp.SparseTensor): raise TypeError("SLat prior expects SparseTensor")
        B=int(z.shape[0]); alpha=alpha[:B].detach().float().reshape(-1).clamp(0,1); choose=torch.rand(B,device=z.device)<alpha
        image=torch.where(choose.view(B,1,1,1),src1_image[:B],src2_image[:B])
        m1,s1,r1=self._stats(src1_slat); m2,s2,r2=self._stats(src2_slat); m1,s1,r1=m1[:B],s1[:B],r1[:B]; m2,s2,r2=m2[:B],s2[:B],r2[:B]
        endpoint_rms=torch.where(choose,r1,r2).detach()
        with torch.no_grad(),torch.autocast(device_type=z.device.type,enabled=False):
            dz=z.replace(z.feats.detach().float()); ids=self._ids(dz)
            if tau is None: tau=torch.rand(B,device=z.device)*(self.t_max-self.t_min)+self.t_min
            else:
                tau=torch.as_tensor(tau,device=z.device,dtype=torch.float32).detach(); tau=tau.expand(B) if tau.numel()==1 else tau
            nf=torch.randn_like(dz.feats) if noise is None else noise.detach().to(z.device,dtype=torch.float32)
            cond=self.encode_image(image).float(); tf=tau[ids,None]; sf=self.sigma_min+(1-self.sigma_min)*tf
            xt=dz.replace((1-tf)*dz.feats+sf*nf); vel=self.flow(xt,tau*1000.,cond).feats.float()
            px=(1-self.sigma_min)*xt.feats.float()-sf*vel; raw=px-dz.feats; rawr=self._rms(raw,dz)
            tan,radfrac,cos=self._tangent(raw,dz); tanr=self._rms(tan,dz); pd=tan if self.tangent_projection else raw; pdr=tanr if self.tangent_projection else rawr
            if self.projection_clip_ratio>0:
                sc=(self.projection_clip_ratio*endpoint_rms/pdr.clamp_min(1e-12)).clamp(max=1.); delta=pd*sc[ids,None]; clip=(sc<.999999).float().mean()
            else: delta=pd; clip=raw.new_zeros(())
            target=dz.feats+delta; clipr=self._rms(delta,dz)
            rm=alpha[:,None]*m1+(1-alpha[:,None])*m2
            rs=torch.exp(alpha[:,None]*torch.log(s1.clamp_min(1e-4))+(1-alpha[:,None])*torch.log(s2.clamp_min(1e-4)))
            rr=torch.exp(alpha*torch.log(r1.clamp_min(1e-8))+(1-alpha)*torch.log(r2.clamp_min(1e-8)))
            gl=self.rms_guard_low_ratio*torch.minimum(r1,r2); gh=self.rms_guard_high_ratio*torch.maximum(r1,r2)
        pl=.5*F.mse_loss(z.feats.float(),target); zm,zs,zr=self._stats(z)
        md=(zm-rm).abs()/rs.clamp_min(1e-4); mv=F.relu(md-self.stat_mean_tolerance); sr=zs/rs.clamp_min(1e-4); sl=torch.log(sr.clamp_min(1e-6))
        lo=F.relu(sl.new_tensor(math.log(self.stat_std_low_ratio))-sl); hi=F.relu(sl-sl.new_tensor(math.log(self.stat_std_high_ratio)))
        stat=(mv.square()+lo.square()+hi.square()).mean(); active=((mv>0)|(lo>0)|(hi>0)).any(1).float().mean()
        lg=F.relu(gl-zr); hg=F.relu(zr-gh); guard=(lg.square()+hg.square()).mean(); gf=((lg>0)|(hg>0)).float().mean(); ratio=zr/rr.clamp_min(1e-8)
        loss=pl+self.stat_anchor_weight*stat+self.rms_guard_weight*guard
        metrics={
          "trellis_prior_loss":loss.detach(),"trellis_prior_projection_loss":pl.detach(),"trellis_prior_projection_delta_rms":rawr.mean().detach(),
          "trellis_prior_projection_tangent_delta_rms":tanr.mean().detach(),"trellis_prior_projection_radial_fraction":radfrac.mean().detach(),
          "trellis_prior_projection_raw_cosine_z":cos.mean().detach(),"trellis_prior_projection_delta_clipped_rms":clipr.mean().detach(),
          "trellis_prior_projection_clip_fraction":clip.detach(),"trellis_prior_projection_x0_rms":px.square().mean().sqrt().detach(),
          "trellis_prior_velocity_rms":vel.square().mean().sqrt().detach(),"trellis_prior_t_mean":tau.mean().detach(),"trellis_prior_scale_loss":stat.detach(),
          "trellis_prior_scale_active_fraction":active.detach(),"trellis_prior_scale_reference_rms":rr.mean().detach(),"trellis_prior_scale_ratio":ratio.mean().detach(),
          "trellis_prior_guard_loss":guard.detach(),"trellis_prior_guard_fraction":gf.detach(),"trellis_prior_guard_low":gl.mean().detach(),
          "trellis_prior_guard_high":gh.mean().detach(),"trellis_prior_endpoint_rms":endpoint_rms.mean().detach(),
          "trellis_prior_sample_endpoint_rms_ratio":(zr/endpoint_rms.clamp_min(1e-8)).mean().detach(),"trellis_prior_delta_endpoint_rms_ratio":(clipr/endpoint_rms.clamp_min(1e-8)).mean().detach(),
          "trellis_prior_src1_fraction":choose.float().mean().detach(),"trellis_prior_slat_stat_loss":stat.detach(),"trellis_prior_slat_stat_active_fraction":active.detach(),
          "trellis_prior_slat_mean_deviation":md.mean().detach(),"trellis_prior_slat_std_ratio":sr.mean().detach()}
        if return_loss_terms: return loss,metrics,{"projection":pl,"scale":self.stat_anchor_weight*stat,"guard":self.rms_guard_weight*guard}
        return loss,metrics
