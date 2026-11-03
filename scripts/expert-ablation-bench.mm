// Development ablations, using the exact GPU timer and fixture of the matched
// Splash comparison. No official evaluation code or reports are used here.
#define main matched_comparison_main
#include "splash-kernel-bench.mm"
#undef main

static void ablations(GPU&g,int runs,bool reverse,bool broad) {
 for(bool fused:{true,false}) {
  int k=fused?2048:512,n=fused?512:2048,choices=fused?2:1;
  Weights w(g,k,n,256,choices);
  Buffer packed=g.alloc(w.raw.length),scratch=g.alloc(w.raw.length);
  Buffer decoded=g.alloc(w.raw.length*2),dd=g.alloc(w.desc.length);
  size_t metaSlot=size_t(n)*k+size_t(n/32)*(k/128)*2;
  Buffer meta=g.alloc(metaSlot*256*choices),md=g.alloc(w.desc.length);
  memcpy(dd.contents,w.desc.contents,w.desc.length);memcpy(md.contents,w.desc.contents,w.desc.length);
  for(int e=0;e<256;e++)for(int h=0;h<choices;h++) {
   size_t off=(size_t(e)*choices+h)*w.slot,mo=(size_t(e)*choices+h)*metaSlot;
   auto bytes=(uint8_t*)w.raw.contents+off;auto scales=(uint16_t*)(bytes+size_t(n)*k);
   if(broad){
    for(int c=0;c<n;c++)for(int j=0;j<k;j++){uint32_t v=uint32_t(e*104729+c*8191+j*131+h*67);v^=v>>7;uint8_t b=v%256;bytes[size_t(c)*k+j]=(b&127)==127?126:b;}
    for(int c=0;c<n;c+=128)for(int j=0;j<k;j+=128)scales[c/128*(k/128)+j/128]=bf(std::ldexp(1.137f+0.031f*((c+j+e+h)%11),-11+(e+h+c/128)%3));
   }
   memcpy((uint8_t*)packed.contents+off+size_t(n)*k,scales,w.slot-size_t(n)*k);
   for(int d=0;d<3;d++)if(choices==1||d==h){((uint64_t*)dd.contents)[e*6+2*d]=off*2;((uint64_t*)md.contents)[e*6+2*d]=mo;((uint64_t*)md.contents)[e*6+2*d+1]=mo+size_t(n)*k;}
   for(int c=0;c<n;c++)for(int j=0;j<k;j++){
    size_t at=(size_t(c)/32)*k*32+(j/32)*1024+((j%32)/4)*128+(c%32)*4+j%4;
    ((uint8_t*)packed.contents)[off+at]=bytes[size_t(c)*k+j];((uint8_t*)meta.contents)[mo+at]=bytes[size_t(c)*k+j];
    uint8_t v=bytes[size_t(c)*k+j];int ex=(v>>3)&15,m=v&7;
    float f=ex?std::ldexp(1.f+m/8.f,ex-7):std::ldexp(float(m),-9);if(v&128)f=-f;
    ((_Float16*)((uint8_t*)decoded.contents+off*2))[size_t(c)*k+j]=f*unbf(scales[c/128*(k/128)+j/128]);
   }
   for(int c=0;c<n;c+=32)for(int j=0;j<k;j+=128)((uint16_t*)((uint8_t*)meta.contents+mo+size_t(n)*k))[c/32*(k/128)+j/128]=scales[c/128*(k/128)+j/128];
  }
  for(int m:{32,512,2048})for(bool skew:{false,true}) {
   int rows=m*8,nt=0,pos=0;std::vector<int> first,live,expert;
   for(int e=0;e<256;e++){int count=skew?(e%2?2*rows/256-1:1):rows/256;
    // At M=32 cover zero experts and 1/2-row tails while preserving the sum.
    if(m==32&&skew)count=e%4==0?0:e%4==3?2:1;
    for(int r=0;r<count;r+=32){first.push_back(pos+r);live.push_back(std::min(32,count-r));expert.push_back(e);nt++;}pos+=count;}
   if(pos!=rows)throw std::runtime_error("route sum");
   Buffer x=g.alloc(size_t(rows+32)*k*2),tiles=g.alloc(nt*16),tc=g.alloc(16),args=g.alloc(16),y=g.alloc(size_t(rows)*n*4+256);
   ((uint32_t*)tc.contents)[0]=nt;((uint32_t*)tc.contents)[1]=n/(fused?32:64);((uint32_t*)tc.contents)[2]=nt;((uint32_t*)tc.contents)[3]=1;
   for(int t=0;t<nt;t++){auto a=(uint32_t*)tiles.contents+t*4;a[0]=expert[t];a[1]=first[t];a[2]=live[t];}
   for(int r=0;r<rows;r++)for(int j=0;j<k;j++)((_Float16*)x.contents)[size_t(r)*k+j]=inputValue(fused?r%m:r,j);
   Args p;p.N=n;p.K=k;p.Kind=fused?0:2;p.A=256;Args dp;dp.N=n/(fused?32:64);
   int limit=std::min(rows,(rows+31)/32+256);
   struct Spec {std::string name;Buffer raw,desc;bool half,indirect,prepare,repack;};
   std::vector<Spec> specs={{"staged",w.raw,w.desc,false,false,false,false},{"predecoded",decoded,dd,false,false,false,false},
    {"half",w.raw,w.desc,true,false,false,false},{"scale",w.raw,w.desc,false,false,false,false},
    {"packed",packed,w.desc,false,false,false,false},{"metadata",meta,md,false,false,false,false},
    {"tiny",w.raw,w.desc,false,false,false,false},{"packed_tiny",packed,w.desc,false,false,false,false},
    {"combined",packed,w.desc,true,false,false,false},{"original_combined",w.raw,w.desc,true,false,false,false},
    {"staged_indirect",w.raw,w.desc,false,true,false,false},{"staged_prepare_indirect",w.raw,w.desc,false,true,true,false},
    {"combined_indirect",packed,w.desc,true,true,true,false},{"packed_with_repack",scratch,w.desc,false,true,true,true}};
   std::vector<Variant> variants;std::vector<float> baseline;
   for(auto spec:specs){std::string kernel="ablation_"+std::string(fused?"gate_up_":"matrix_")+spec.name;
    if(spec.name=="predecoded"&&!fused)kernel="ablation_matrix_predecoded_tile";
    if(spec.name=="staged"||spec.name.find("staged_")==0)kernel="expert_"+std::string(fused?"gate_up_":"matrix_")+"staged";
    if(spec.name=="combined_indirect")kernel="ablation_"+std::string(fused?"gate_up_":"matrix_")+"combined";
    if(spec.repack)kernel="ablation_"+std::string(fused?"gate_up_":"matrix_")+"packed";
    auto encode=[&,spec,kernel](id<MTLComputeCommandEncoder>enc){
     if(spec.repack)for(int h=0;h<choices;h++){Args rp=p;rp.Kind=fused?h:2;g.dispatch(enc,"ours","expert_prepare_packed",{w.raw,w.desc,scratch},&rp,sizeof(rp),15,(256*n*k+127)/128,1,128);}
     if(spec.prepare)g.dispatch(enc,"ours","expert_dispatch_args",{tc,args},&dp,sizeof(dp),15,1,1,1);
     g.dispatch(enc,"ours",kernel,{x,spec.raw,spec.desc,tiles,tc,y},&p,sizeof(p),15,n/(fused?32:64),limit,64,spec.indirect?(spec.prepare?args:tc):nil);
    };
    bool half=fused||spec.half;size_t used=size_t(rows)*n*(half?2:4);initOutput(y,used);g.run(encode);guard(y,used);
    double maxerr=0,err2=0,base2=0;for(size_t i=0;i<size_t(rows)*n;i++){float v=half?float(((_Float16*)y.contents)[i]):((float*)y.contents)[i];if(!std::isfinite(v))throw std::runtime_error("nonfinite "+spec.name);if(spec.name=="staged")baseline.push_back(v);double err=v-baseline[i];maxerr=std::max(maxerr,std::abs(err));err2+=err*err;base2+=double(baseline[i])*baseline[i];}
    double l2=std::sqrt(err2/std::max(base2,1e-30));if(l2>0.0015)throw std::runtime_error("L2 mismatch "+spec.name+" "+std::to_string(l2));
    if(spec.repack&&memcmp(packed.contents,scratch.contents,w.raw.length))throw std::runtime_error("packed bytes/scales mismatch");
    int checks=0;double cpuMax=0;
    for(int t=0;t<nt;t++)if(expert[t]==0||expert[t]==1||expert[t]==127||expert[t]==128||expert[t]==255)for(int r:{0,live[t]-1})for(int c:{0,127,128,n-1}){
     double sum[2]={};for(int h=0;h<choices;h++){size_t off=(size_t(expert[t])*choices+h)*w.slot*2;auto wf=(_Float16*)((uint8_t*)decoded.contents+off);for(int j=0;j<k;j++)sum[h]+=double(((_Float16*)x.contents)[size_t(first[t]+r)*k+j])*float(wf[size_t(c)*k+j]);}
     if(fused&&spec.half)sum[0]=float((_Float16)sum[0]);double expected=fused?double((_Float16)(sum[0]/(1+std::exp(-sum[0]))*sum[1])):half?double((_Float16)sum[0]):sum[0];
     size_t at=size_t(first[t]+r)*n+c;float actual=half?float(((_Float16*)y.contents)[at]):((float*)y.contents)[at];cpuMax=std::max(cpuMax,check(actual,expected,0.0002+0.003*std::abs(expected),spec.name));checks++;
    }
    emit(@{@"type":@"check",@"family":fused?@"gate_up":@"down",@"variant":@(spec.name.c_str()),@"m":@(m),@"routing":skew?@"skewed":@"uniform",@"broad_fp8":@(broad),@"max_abs_vs_baseline":@(maxerr),@"relative_l2":@(l2),@"cpu_reference_points":@(checks),@"cpu_max_abs":@(cpuMax),@"guards":@YES,@"all_live_outputs_finite":@YES});
    variants.push_back({spec.name,encode});
   }
   measure(g,variants,runs,reverse,fused?@"gate_up":@"down",m,n,k,skew?@"skewed":@"uniform",rows,nt*32);
  }
 }
}
int main(int argc,char**argv){@autoreleasepool{try{if(argc!=7)throw std::runtime_error("expected ours moe dense runs fixture order");GPU g(argv);ablations(g,std::stoi(argv[4]),std::string(argv[6])=="reverse",std::string(argv[5])=="broad");return 0;}catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}}}
