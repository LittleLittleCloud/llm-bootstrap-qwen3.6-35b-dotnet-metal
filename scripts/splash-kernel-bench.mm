// Isolated kernel comparison. Original third-party shaders are provided at run time.
// Native formats/layouts/epilogues are retained. Inputs and decoded weights are
// identical dyadic values, exactly representable in both FP16 and BF16.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "metal/abi/Gguf.h"
#include "metal/abi/MoE.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <functional>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

using Buffer = id<MTLBuffer>;
using Clock = std::chrono::steady_clock;
struct Args { uint32_t N=0,M=0,K=0,Kind=0,Position=0,Flags=0,A=0,B=0; float Epsilon=0,Theta=0,Scale=0,Value=0; uint32_t C=0,D=0,E=0,F=0; };
static_assert(sizeof(Args)==64);
static uint16_t bf(float x) { uint32_t u; memcpy(&u,&x,4); u += 0x7fff + ((u>>16)&1); return u>>16; }
static float unbf(uint16_t x) { uint32_t u=uint32_t(x)<<16; float f; memcpy(&f,&u,4); return f; }
static void emit(NSDictionary *obj) { auto data=[NSJSONSerialization dataWithJSONObject:obj options:0 error:nil]; std::cout.write((const char*)data.bytes,data.length); std::cout<<std::endl; }
static float inputValue(size_t row,int j) { return float(int((row*131+j*17+j/31)%67)-33)/64; }
static int code(int e,int c,int j,int choice) { uint32_t x=uint32_t(e)*15485863u+uint32_t(c)*32452843u+uint32_t(j)*49979687u+uint32_t(choice)*67867967u; x^=x>>13; x*=0x85ebca6bu; x^=x>>16; return int(x&31)-16; }
static float scale(int e,int c,int j,int choice) { return std::ldexp(1.f,-8+((e+c/128+j/128+choice)%3)); }
static float weight(int e,int c,int j,int choice) { return code(e,c,j,choice)*scale(e,c,j,choice); }
static uint8_t fp8integer(int v) { if(!v)return 0; int a=std::abs(v),ex=0; while((1<<(ex+1))<=a)++ex; return (v<0?128:0)|((ex+7)<<3)|((a-(1<<ex))*8/(1<<ex)); }
struct GPU {
 id<MTLDevice> device=MTLCreateSystemDefaultDevice(); id<MTLCommandQueue> queue=[device newCommandQueue];
 std::map<std::string,id<MTLLibrary>> libraries;
 std::map<std::string,id<MTLComputePipelineState>> pipelines;
 GPU(char **argv) {
  for(int i=1;i<=3;i++) { NSError *err=nil; auto source=[NSString stringWithContentsOfFile:@(argv[i]) encoding:NSUTF8StringEncoding error:&err];
   if(!source)throw std::runtime_error(err.description.UTF8String);
   auto opts=[MTLCompileOptions new]; opts.languageVersion=MTLLanguageVersion4_0; opts.mathMode=i==1?MTLMathModeSafe:MTLMathModeFast;
   auto lib=[device newLibraryWithSource:source options:opts error:&err]; if(!lib)throw std::runtime_error(err.description.UTF8String);
   libraries[i==1?"ours":i==2?"moe":"dense"]=lib;
  }
 }
 Buffer alloc(size_t n) { auto b=[device newBufferWithLength:std::max(n,size_t(256)) options:MTLResourceStorageModeShared]; if(!b)throw std::runtime_error("buffer allocation failed"); memset(b.contents,0,b.length); return b; }
 id<MTLComputePipelineState> pipe(std::string lib,std::string name) { auto key=lib+name; if(!pipelines.count(key)) {
  NSError *err=nil; auto fn=[libraries.at(lib) newFunctionWithName:@(name.c_str())]; if(!fn)throw std::runtime_error("missing kernel "+name);
  auto p=[device newComputePipelineStateWithFunction:fn error:&err]; if(!p)throw std::runtime_error(err.description.UTF8String); pipelines[key]=p; }return pipelines.at(key); }
 void dispatch(id<MTLComputeCommandEncoder> enc,std::string lib,std::string name,std::vector<Buffer> buffers,const void *p,size_t bytes,int at,int gx,int gy,int threads,Buffer indirect=nil) {
  [enc setComputePipelineState:pipe(lib,name)]; for(size_t i=0;i<buffers.size();i++)[enc setBuffer:buffers[i] offset:0 atIndex:i];
  [enc setBytes:p length:bytes atIndex:at];
  if(indirect)[enc dispatchThreadgroupsWithIndirectBuffer:indirect indirectBufferOffset:4 threadsPerThreadgroup:MTLSizeMake(threads,1,1)];
  else [enc dispatchThreadgroups:MTLSizeMake(gx,gy,1) threadsPerThreadgroup:MTLSizeMake(threads,1,1)];
 }
 double run(const std::function<void(id<MTLComputeCommandEncoder>)>& encode) { @autoreleasepool {
  auto cb=[queue commandBuffer]; auto enc=[cb computeCommandEncoder]; encode(enc);[enc endEncoding];[cb commit];[cb waitUntilCompleted];
  if(cb.status!=MTLCommandBufferStatusCompleted)throw std::runtime_error(cb.error.description.UTF8String);
  double ms=(cb.GPUEndTime-cb.GPUStartTime)*1000; if(!(ms>0&&std::isfinite(ms)))throw std::runtime_error("invalid GPU timestamps"); return ms;
 } }
};
struct Weights { Buffer raw,desc,plane,meta,decoded; int k,n,experts,choices; size_t slot;
 Weights(GPU&g,int k_,int n_,int e_,int choices_,bool dense=false):k(k_),n(n_),experts(e_),choices(choices_) {
  size_t nk=size_t(n)*k,sc=size_t(n/128)*(k/128);slot=nk+sc*2;
  raw=g.alloc(size_t(experts)*choices*slot);desc=g.alloc(size_t(experts)*48);plane=g.alloc(size_t(experts)*choices*nk);meta=g.alloc(size_t(experts)*choices*n*(k/64)*4);
  if(dense)decoded=g.alloc(nk*2);
  for(int e=0;e<experts;e++)for(int h=0;h<choices;h++) {
   size_t off=(size_t(e)*choices+h)*slot; auto descriptors=(uint64_t*)desc.contents;
   for(int d=0;d<3;d++)if(choices==1||d==h){descriptors[e*6+d*2]=off;descriptors[e*6+d*2+1]=off+nk;}
   auto fp=(uint8_t*)raw.contents+off; auto scales=(uint16_t*)(fp+nk);
   for(int c=0;c<n;c++)for(int j=0;j<k;j++) {
    fp[size_t(c)*k+j]=fp8integer(code(e,c,j,h));
    // Splash stores each projection separately, then expert and tiled column.
    size_t pc=(size_t(h)*experts+e)*nk+quant_tile_index(c,j/32,k/32)*32+quant_slot(j%32);
    ((uint8_t*)plane.contents)[pc]=uint8_t((code(e,c,j,h)+16)*8);
    if(dense)((_Float16*)decoded.contents)[size_t(c)*k+j]=weight(e,c,j,h);
   }
   for(int c=0;c<n;c+=128)for(int j=0;j<k;j+=128)scales[c/128*(k/128)+j/128]=bf(scale(e,c,j,h));
   for(int c=0;c<n;c++)for(int j=0;j<k;j+=64){size_t at=((size_t(h)*experts+e)*n*(k/64)+quant_tile_index(c,j/64,k/64))*2;
    ((uint16_t*)meta.contents)[at]=bf(scale(e,c,j,h)/8);((uint16_t*)meta.contents)[at+1]=bf(-16*scale(e,c,j,h));}
  }
 }
};
struct Variant { std::string name; std::function<void(id<MTLComputeCommandEncoder>)> encode; };
static void measure(GPU&gpu,std::vector<Variant> variants,int runs,bool reverse,NSString *family,int m,int n,int k,NSString *routing,int live,int padded) {
 for(auto &v:variants)gpu.run(v.encode); // compile every pipeline before any samples
 for(int run=0;run<runs;run++) {
  bool rev=(run%2!=0)^reverse;
  for(size_t j=0;j<variants.size();j++) {auto &v=variants[rev?variants.size()-1-j:j];
   auto warm=Clock::now(); do {gpu.run(v.encode);}while(std::chrono::duration<double>(Clock::now()-warm).count()<0.25);
   auto start=Clock::now();double ms=gpu.run(v.encode);double wall=std::chrono::duration<double,std::milli>(Clock::now()-start).count();
   emit(@{@"type":@"timing",@"family":family,@"variant":@(v.name.c_str()),@"m":@(m),@"n":@(n),@"k":@(k),@"routing":routing,@"routed_rows":@(live),@"splash_storage_rows":@(padded),@"run":@(run),@"gpu_ms":@(ms),@"wall_ms":@(wall)});
  }
 }
}
static double check(float actual,double expected,double tolerance,const std::string&where) { double err=std::abs(actual-expected);if(!std::isfinite(actual)||err>tolerance)throw std::runtime_error(where+": "+std::to_string(actual)+" expected "+std::to_string(expected));return err; }
static void guard(Buffer b,size_t used) {for(size_t i=used;i<used+256;i++)if(((uint8_t*)b.contents)[i]!=0xa5)throw std::runtime_error("output guard overwritten");}
static void initOutput(Buffer b,size_t used) {memset(b.contents,0xff,used);memset((uint8_t*)b.contents+used,0xa5,256);}

static void experts(GPU&g,int runs,bool reverse) {
 for(bool gate:{true,false}) {int k=gate?2048:512,n=gate?512:2048;Weights w(g,k,n,256,gate?2:1);
  // Extract Splash's up projection without modifying its byte layout; outside timing.
  Buffer upPlane=g.alloc(size_t(256)*n*k),upMeta=g.alloc(size_t(256)*n*(k/64)*4);
  if(gate){memcpy(upPlane.contents,(uint8_t*)w.plane.contents+size_t(256)*n*k,upPlane.length);memcpy(upMeta.contents,(uint8_t*)w.meta.contents+size_t(256)*n*(k/64)*4,upMeta.length);}
  for(int m:{512,2048})for(bool skew:{false,true}) {int rows=m*8;std::vector<int> counts(256),first(256),tileFirst,tileExpert,tileRows;
   int nt=0,pos=0;for(int e=0;e<256;e++){counts[e]=skew?(e%2?2*rows/256-1:1):rows/256;first[e]=pos;
    for(int r=0;r<counts[e];r+=32){tileFirst.push_back(pos+r);tileExpert.push_back(e);tileRows.push_back(std::min(32,counts[e]-r));nt++;}pos+=counts[e];}
   if(pos!=rows)throw std::runtime_error("bad routing total");int padded=nt*32;
   Buffer ox=g.alloc(size_t(rows+32)*k*2),sx=g.alloc(size_t(padded)*k*2),ot=g.alloc(size_t(nt)*16),st=g.alloc(size_t(nt)*8),tc=g.alloc(4),slots=g.alloc(256*4);
   Buffer oy=g.alloc(size_t(rows)*n*(gate?2:4)+256),sy=g.alloc(size_t(padded)*n*2+256);((uint32_t*)tc.contents)[0]=nt;
   ((uint32_t*)tc.contents)[1]=n/(gate?32:64);((uint32_t*)tc.contents)[2]=nt;((uint32_t*)tc.contents)[3]=1;
   for(int e=0;e<256;e++)((uint32_t*)slots.contents)[e]=e;
   for(int t=0;t<nt;t++){auto o=(uint32_t*)ot.contents+t*4;o[0]=tileExpert[t];o[1]=tileFirst[t];o[2]=tileRows[t];auto s=(MoeTileDescriptor*)st.contents+t;s->expert=tileExpert[t];s->rows=tileRows[t];
    for(int r=0;r<tileRows[t];r++)for(int j=0;j<k;j++){int row=tileFirst[t]+r;float value=inputValue(gate?row%m:row,j);((_Float16*)ox.contents)[size_t(row)*k+j]=value;((uint16_t*)sx.contents)[size_t(t*32+r)*k+j]=bf(value);}}
   Args op;op.N=n;op.K=k;op.Kind=gate?0:2;MoeGgufExpertParams sp{uint32_t(k),uint32_t(n),256,GGUF_FMT_AF8G64,GGUF_FMT_AF8G64};MoeGgufGateUpParams gp{uint32_t(k),uint32_t(n),256,GGUF_FMT_AF8G64,GGUF_FMT_AF8G64,GGUF_FMT_AF8G64,GGUF_FMT_AF8G64};
   auto ours=[&](id<MTLComputeCommandEncoder> enc){g.dispatch(enc,"ours",gate?"expert_gate_up_staged":"expert_matrix_staged",{ox,w.raw,w.desc,ot,tc,oy},&op,sizeof(op),15,n/(gate?32:64),std::min(rows,(rows+31)/32+256),64);};
   auto splash=[&](id<MTLComputeCommandEncoder> enc){if(gate)g.dispatch(enc,"moe","moe_expert_gguf_m32_gate_up",{sx,st,tc,w.plane,w.meta,w.meta,w.plane,w.meta,w.meta,upPlane,upMeta,upMeta,upPlane,upMeta,upMeta,sy,slots},&gp,sizeof(gp),17,n/32,nt,64,tc);
    else g.dispatch(enc,"moe","moe_expert_gguf_m32_a",{sx,st,tc,w.plane,w.meta,w.meta,w.plane,w.meta,w.meta,sy,sy,slots},&sp,sizeof(sp),12,n/64,nt,64,tc);};
   initOutput(oy,size_t(rows)*n*(gate?2:4));initOutput(sy,size_t(padded)*n*2);g.run(ours);g.run(splash);
   double worstO=0,worstS=0;int checks=0;
   for(int t=0;t<nt;t++){int e=tileExpert[t];for(int r=0;r<tileRows[t];r++)for(int c=0;c<n;c++){size_t oi=size_t(tileFirst[t]+r)*n+c,si=size_t(t*32+r)*n+c;float a=gate?float(((_Float16*)oy.contents)[oi]):((float*)oy.contents)[oi],b=unbf(((uint16_t*)sy.contents)[si]);if(!std::isfinite(a)||!std::isfinite(b))throw std::runtime_error("nonfinite expert output");}
    if(e==0||e==1||e==127||e==128||e==255)for(int r:{0,tileRows[t]-1})for(int c:{0,127,128,n-1}){double sums[2]={};for(int h=0;h<(gate?2:1);h++)for(int j=0;j<k;j++)sums[h]+=double(inputValue(gate?(tileFirst[t]+r)%m:tileFirst[t]+r,j))*weight(e,c,j,h);
     double eo=gate?double((_Float16)(sums[0]/(1+std::exp(-sums[0]))*sums[1])):sums[0];float gs=unbf(bf(sums[0])),us=unbf(bf(sums[1]));double es=unbf(bf(gate?us*gs/(1+std::exp(-gs)):sums[0]));
     float a=gate?float(((_Float16*)oy.contents)[size_t(tileFirst[t]+r)*n+c]):((float*)oy.contents)[size_t(tileFirst[t]+r)*n+c],b=unbf(((uint16_t*)sy.contents)[size_t(t*32+r)*n+c]);
     worstO=std::max(worstO,check(a,eo,0.0001+0.003*std::abs(eo),"ours expert"));worstS=std::max(worstS,check(b,es,0.0001+0.012*std::abs(es),"splash expert"));checks++;}}
   guard(oy,size_t(rows)*n*(gate?2:4));guard(sy,size_t(padded)*n*2);
   emit(@{@"type":@"check",@"family":gate?@"gate_up":@"down",@"m":@(m),@"routing":skew?@"skewed":@"uniform",@"cpu_reference_points":@(checks),@"ours_max_abs":@(worstO),@"splash_max_abs":@(worstS),@"all_live_outputs_finite":@YES,@"guards":@YES,@"live_tiles":@(nt),@"ours_grid_tiles":@(std::min(rows,(rows+31)/32+256)),@"experts":@256,@"top_k":@8});
   measure(g,{{"ours",ours},{"splash",splash}},runs,reverse,gate?@"gate_up":@"down",m,n,k,skew?@"skewed":@"uniform",rows,padded);
  }
 }
}

static void dense(GPU&g,int runs,bool reverse) {
 for(auto shape:std::vector<std::vector<int>>{{512,512,2048},{2048,512,2048},{2048,2048,2048},{2048,8192,2048},{2048,2048,4096},{2048,2048,512}}){int m=shape[0],n=shape[1],k=shape[2];Weights w(g,k,n,1,1,true);
  Buffer ox=g.alloc(size_t(m)*k*2),sx=g.alloc(size_t(m)*k*2),decoded=g.alloc(size_t(n)*k*2),scales=g.alloc(size_t(n/128)*(k/128)*2),oy=g.alloc(size_t(m)*n*4+256),sy=g.alloc(size_t(m)*n*2+256);
  memcpy(scales.contents,(uint8_t*)w.raw.contents+size_t(n)*k,scales.length);
  for(int r=0;r<m;r++)for(int j=0;j<k;j++){float x=inputValue(r,j);((_Float16*)ox.contents)[size_t(r)*k+j]=x;((uint16_t*)sx.contents)[size_t(r)*k+j]=bf(x);}
  Args op;op.N=n;op.M=m;op.K=k;GgufPrefillSegmentsParams sp{uint32_t(k),uint32_t(m),uint32_t(n),{uint32_t(n),0,0},{0,0,0}};
  auto oursCore=[&](id<MTLComputeCommandEncoder> enc){g.dispatch(enc,"ours","matrix_product_predecoded",{ox,w.decoded,oy},&op,sizeof(op),15,(n+63)/64,(m+63)/64,128);};
  auto ours=[&](id<MTLComputeCommandEncoder> enc){g.dispatch(enc,"ours","projection_unpack_half",{w.raw,scales,decoded},&op,sizeof(op),15,(n*k/4+127)/128,1,128);g.dispatch(enc,"ours","matrix_product_predecoded",{ox,decoded,oy},&op,sizeof(op),15,(n+63)/64,(m+63)/64,128);};
  auto splash=[&](id<MTLComputeCommandEncoder> enc){g.dispatch(enc,"dense","gguf_prefill_af8g64_a",{sx,w.plane,w.meta,w.meta,w.plane,w.meta,w.meta,w.plane,w.meta,w.meta,sy},&sp,sizeof(sp),11,(m+127)/128,n/64,128);};
  initOutput(oy,size_t(m)*n*4);initOutput(sy,size_t(m)*n*2);g.run(oursCore);std::vector<float> core((float*)oy.contents,(float*)oy.contents+size_t(m)*n);g.run(ours);g.run(splash);
  for(size_t i=0;i<size_t(m)*n;i++){if(core[i]!=((float*)oy.contents)[i])throw std::runtime_error("dense decode/core mismatch");if(!std::isfinite(core[i])||!std::isfinite(unbf(((uint16_t*)sy.contents)[i])))throw std::runtime_error("nonfinite dense output");}
  // Check all unpacked weights, not only output samples.
  if(memcmp(decoded.contents,w.decoded.contents,size_t(n)*k*2))throw std::runtime_error("dense decoded weights mismatch");
  double wo=0,ws=0;for(int r:{0,m/2,m-1})for(int c:{0,127,128,n/2,n-1}){double sum=0;for(int j=0;j<k;j++)sum+=double(inputValue(r,j))*weight(0,c,j,0);wo=std::max(wo,check(core[size_t(r)*n+c],sum,0.0001+0.0001*std::abs(sum),"ours dense"));ws=std::max(ws,check(unbf(((uint16_t*)sy.contents)[size_t(r)*n+c]),unbf(bf(sum)),0.0001+0.008*std::abs(sum),"splash dense"));}
  guard(oy,size_t(m)*n*4);guard(sy,size_t(m)*n*2);emit(@{@"type":@"check",@"family":@"dense",@"m":@(m),@"n":@(n),@"k":@(k),@"cpu_reference_points":@15,@"ours_max_abs":@(wo),@"splash_max_abs":@(ws),@"all_live_outputs_finite":@YES,@"guards":@YES});
  measure(g,{{"ours_core",oursCore},{"ours_with_weight_decode",ours},{"splash",splash}},runs,reverse,@"dense",m,n,k,@"none",m,m);
 }
}
int main(int argc,char**argv) { @autoreleasepool { try {if(argc!=7)throw std::runtime_error("expected ours.metal moe.metal dense.metal runs family order");GPU gpu(argv);int runs=std::stoi(argv[4]);std::string family=argv[5];bool reverse=std::string(argv[6])=="reverse";
 emit(@{@"type":@"environment",@"device":gpu.device.name,@"memory":@(gpu.device.recommendedMaxWorkingSetSize),@"runs":@(runs),@"order":@(argv[6])});
 if(family=="all"||family=="experts")experts(gpu,runs,reverse);if(family=="all"||family=="dense")dense(gpu,runs,reverse);return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;} } }
