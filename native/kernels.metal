#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;
using namespace metal;
// Expert pipeline experiments keep the checkpoint's FP8 bytes and BF16 scales.
struct P { uint n,m,k,kind,pos,flags,a,b; float eps,theta,scale,value; uint c,d,e,f; };
struct Expert { ulong gate,gate_scale,up,up_scale,down,down_scale; };
inline float bf(ushort x) { return as_type<float>(uint(x)<<16); }
inline float f8(uchar x) {
    uint e=(x>>3)&15,m=x&7;
    if(e==0)return (x&128?-1.0f:1.0f)*float(m)*(1.0f/512.0f);
    if(e==15&&m==7)return NAN;
    return as_type<float>((uint(x&128)<<24)|((e+120)<<23)|(m<<20));
}
// Bit 8 selects FP16 cache storage; compute and softmax remain FP32.
inline float kv_load(device const float*cache,uint index,uint flags) {
    return flags&256 ? float(((device const half*)cache)[index]) : cache[index];
}
inline void kv_store(device float*cache,uint index,float value,uint flags) {
    if(flags&256)((device half*)cache)[index]=half(value);else cache[index]=value;
}
inline float sigmoid(float x) {return 1.0f/(1.0f+exp(-x));}
inline float silu(float x) {return x*sigmoid(x);}
inline float softplus(float x) {return max(x,0.0f)+log(1.0f+exp(-abs(x)));}
inline float4 f8x4(uchar4 bytes) {
    uint4 v=uint4(bytes),e=(v>>3)&15,m=v&7;
    float4 normal=as_type<float4>(((v&128)<<24)|((e+120)<<23)|(m<<20));
    float4 subnormal=select(float4(1),float4(-1),(v&128)!=0)*float4(m)*(1.0f/512.0f);
    return select(select(normal,subnormal,e==0),float4(NAN),(e==15)&(m==7));
}
inline float4 weight4(device const uchar*w,device const ushort*s,uint row,uint col,uint cols,uint kind) {
    if(kind==3){ushort2 sb=((device const ushort2*)s)[row*(cols/64)+col/64];return fma(float4(*((device const uchar4*)(w+row*cols+col))),float4(bf(sb.x)),float4(bf(sb.y)));}
    if(kind==0)return f8x4(*((device const uchar4*)(w+row*cols+col)))*bf(s[(row/128)*((cols+127)/128)+col/128]);
    if(kind==1){ushort4 bits=*((device const ushort4*)((device const ushort*)w+row*cols+col));return as_type<float4>(uint4(bits)<<16);}
    device const float* p=(device const float*)w+row*cols+col;return float4(p[0],p[1],p[2],p[3]);
}
inline float weight(device const uchar *w,device const ushort *s,uint row,uint col,uint cols,uint kind) {
    if(kind==3){ushort2 sb=((device const ushort2*)s)[row*(cols/64)+col/64];return fma(float(w[row*cols+col]),bf(sb.x),bf(sb.y));}
    if(kind==0)return f8(w[row*cols+col])*bf(s[(row/128)*((cols+127)/128)+col/128]);
    if(kind==1)return bf(((device const ushort*)w)[row*cols+col]);
    return ((device const float*)w)[row*cols+col];
}
kernel void fp8_values(device const uchar *x [[buffer(0)]],device float *y [[buffer(1)]],constant P&p [[buffer(15)]],uint i [[thread_position_in_grid]]) {if(i<p.n)y[i]=f8(x[i]);}
kernel void embedding(device const ushort*w [[buffer(0)]],device float*y [[buffer(1)]],constant P&p [[buffer(15)]],uint i [[thread_position_in_grid]]) {if(i<p.n)y[i]=bf(w[p.pos*p.n+i]);}
kernel void embedding_affine(device const uchar*w [[buffer(0)]],device float*y [[buffer(1)]],device const ushort*s [[buffer(2)]],constant P&p [[buffer(15)]],uint i [[thread_position_in_grid]]) {if(i<p.n)y[i]=weight(w,s,p.pos,i,p.n,3);}
kernel void gemv(device const float*x [[buffer(0)]],device const uchar*w [[buffer(1)]],device const ushort*s [[buffer(2)]],device float*y [[buffer(3)]],constant P&p [[buffer(15)]],uint g [[thread_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
    uint row=g/32;if(row>=p.n)return;float4 accum=0;
    for(uint j=lane*4;j<p.k;j+=128)accum+=*((device const float4*)(x+j))*weight4(w,s,row,j,p.k,p.kind);
    float sum=simd_sum(accum.x+accum.y+accum.z+accum.w);if(lane==0)y[row]=sum;
}
kernel void rmsnorm(device const float*x [[buffer(0)]],device const ushort*w [[buffer(1)]],device float*y [[buffer(2)]],device const float*gate [[buffer(3)]],constant P&p [[buffer(15)]],uint tid [[thread_index_in_threadgroup]],uint group [[threadgroup_position_in_grid]]) {
    threadgroup float sums[8];threadgroup float inv;
    float total=0;for(uint j=tid;j<p.n;j+=256){float v=x[group*p.n+j];total+=v*v;}
    total=simd_sum(total);if((tid&31)==0)sums[tid/32]=total;threadgroup_barrier(mem_flags::mem_threadgroup);
    total=tid<8?sums[tid]:0;total=simd_sum(total);if(tid==0)inv=rsqrt(total/float(p.n)+p.eps);threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint j=tid;j<p.n;j+=256){float v=x[group*p.n+j]*inv*(bf(w[j])+((p.flags&1)?1.0f:0.0f));if(p.flags&2)v*=silu(gate[group*p.n+j]);y[group*p.n+j]=v;}
}
kernel void add(device const float*x [[buffer(0)]],device const float*z [[buffer(1)]],device float*y [[buffer(2)]],constant P&p [[buffer(15)]],uint i [[thread_position_in_grid]]) {if(i<p.n)y[i]=x[i]+z[i];}
kernel void silu_product(device const float*x [[buffer(0)]],device const float*z [[buffer(1)]],device float*y [[buffer(2)]],constant P&p [[buffer(15)]],uint i [[thread_position_in_grid]]) {if(i<p.n)y[i]=silu(x[i])*z[i];}
inline void route_simd(device const float*x,device int*ids,device float*prob,uint lane) {
    float values[8],chosen[8];
    for(uint j=0;j<8;++j)values[j]=x[j*32+lane];
    float peak=0,denominator=0;
    for(uint choice=0;choice<8;++choice){
        float local=-INFINITY;uint index=0xffffffff;
        for(uint j=0;j<8;++j){uint at=j*32+lane;float value=values[j];if(value>local||(value==local&&at<index)){local=value;index=at;}}
        float best=simd_max(local);uint selected=simd_min(local==best?index:0xffffffff);
        for(uint j=0;j<8;++j)if(j*32+lane==selected)values[j]=-INFINITY;
        if(choice==0)peak=best;chosen[choice]=exp(best-peak);denominator+=chosen[choice];
        if(lane==0)ids[choice]=int(selected);
    }
    if(lane==0)for(uint choice=0;choice<8;++choice)prob[choice]=chosen[choice]/denominator;
}
kernel void router(device const float*x [[buffer(0)]],device int*ids [[buffer(1)]],device float*prob [[buffer(2)]],constant P&p [[buffer(15)]],uint lane [[thread_index_in_simdgroup]]) {
    route_simd(x,ids,prob,lane);
}
// Bit 10 selects the byte-preserving expert layout used by --packed-experts.
inline float4 expert_weight4(device const uchar*w,device const ushort*s,uint row,uint col,uint k,uint flags){
    if(!(flags&1024))return weight4(w,s,row,col,k,0);
    uint at=(row/32)*k*32+(col/32)*1024+((col%32)/4)*128+(row%32)*4;
    if(flags&2048){ushort2 sb=((device const ushort2*)s)[(row/32)*(k/64)*32+(col/64)*32+row%32];return fma(float4(*((device const uchar4*)(w+at))),float4(bf(sb.x)),float4(bf(sb.y)));}
    return f8x4(*((device const uchar4*)(w+at)))*bf(s[(row/128)*(k/128)+col/128]);
}
kernel void expert_gate_up(device const float*x [[buffer(0)]],device const uchar*raw [[buffer(1)]],device const Expert*experts [[buffer(2)]],device const int*ids [[buffer(3)]],device float*y [[buffer(4)]],constant P&p [[buffer(15)]],uint g [[thread_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
    uint row=g/32,e=row/p.n,r=row%p.n;if(e>=p.m)return;Expert d=experts[ids[e]];float4 ga=0,ua=0;
    for(uint j=lane*4;j<p.k;j+=128){float4 v=*((device const float4*)(x+j));ga+=v*expert_weight4(raw+d.gate,(device const ushort*)(raw+d.gate_scale),r,j,p.k,p.flags);ua+=v*expert_weight4(raw+d.up,(device const ushort*)(raw+d.up_scale),r,j,p.k,p.flags);}
    float gate=simd_sum(ga.x+ga.y+ga.z+ga.w),up=simd_sum(ua.x+ua.y+ua.z+ua.w);if(lane==0)y[e*p.n+r]=silu(gate)*up;
}
kernel void expert_down(device const float*x [[buffer(0)]],device const uchar*raw [[buffer(1)]],device const Expert*experts [[buffer(2)]],device const int*ids [[buffer(3)]],device float*y [[buffer(4)]],constant P&p [[buffer(15)]],uint g [[thread_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
    uint row=g/32,e=row/p.n,r=row%p.n;if(e>=p.m)return;Expert d=experts[ids[e]];float4 accum=0;
    for(uint j=lane*4;j<p.k;j+=128)accum+=*((device const float4*)(x+e*p.k+j))*expert_weight4(raw+d.down,(device const ushort*)(raw+d.down_scale),r,j,p.k,p.flags);
    float total=simd_sum(accum.x+accum.y+accum.z+accum.w);if(lane==0)y[e*p.n+r]=total;
}
kernel void mix(device const float*residual [[buffer(0)]],device const float*experts [[buffer(1)]],device const float*prob [[buffer(2)]],device const float*shared [[buffer(3)]],device const float*gate [[buffer(4)]],device float*y [[buffer(5)]],constant P&p [[buffer(15)]],uint i [[thread_position_in_grid]]) {
    if(i>=p.n)return;float value=residual[i]+shared[i]*sigmoid(gate[0]);for(uint e=0;e<p.m;++e)value+=prob[e]*experts[e*p.n+i];y[i]=value;
}
kernel void conv(device const float*x [[buffer(0)]],device const ushort*w [[buffer(1)]],device float*state [[buffer(2)]],device float*y [[buffer(3)]],constant P&p [[buffer(15)]],uint i [[thread_position_in_grid]]) {
    if(i>=p.n)return;float v=state[i]*bf(w[i*4])+state[p.n+i]*bf(w[i*4+1])+state[2*p.n+i]*bf(w[i*4+2])+x[i]*bf(w[i*4+3]);state[i]=state[p.n+i];state[p.n+i]=state[2*p.n+i];state[2*p.n+i]=x[i];y[i]=silu(v);
}
kernel void l2_qk(device const float*x [[buffer(0)]],device float*q [[buffer(1)]],device float*k [[buffer(2)]],constant P&p [[buffer(15)]],uint tid [[thread_index_in_threadgroup]],uint head [[threadgroup_position_in_grid]]) {
    threadgroup float qs[4],ks[4];threadgroup float qi,ki;uint j=head*128+tid;float a=x[j],b=x[2048+j];float qa=simd_sum(a*a),ka=simd_sum(b*b);
    if((tid&31)==0){qs[tid/32]=qa;ks[tid/32]=ka;}threadgroup_barrier(mem_flags::mem_threadgroup);
    qa=simd_sum(tid<4?qs[tid]:0);ka=simd_sum(tid<4?ks[tid]:0);if(tid==0){qi=rsqrt(qa+1e-6f)*rsqrt(128.0f);ki=rsqrt(ka+1e-6f);}threadgroup_barrier(mem_flags::mem_threadgroup);q[j]=a*qi;k[j]=b*ki;
}
kernel void delta(device const float*q [[buffer(0)]],device const float*k [[buffer(1)]],device const float*v [[buffer(2)]],device const float*a [[buffer(3)]],device const float*b [[buffer(4)]],device const ushort*alog [[buffer(5)]],device const ushort*dt [[buffer(6)]],device float*state [[buffer(7)]],device float*y [[buffer(8)]],constant P&p [[buffer(15)]],uint g [[thread_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
    uint row=g/32,h=row/128,d=row%128;if(h>=32)return;uint kh=h/2;float decay=exp(-exp(bf(alog[h]))*softplus(a[h]+bf(dt[h]))),beta=sigmoid(b[h]);float prediction=0;
    for(uint j=lane;j<128;j+=32)prediction+=state[(h*128+d)*128+j]*decay*k[kh*128+j];prediction=simd_sum(prediction);float correction=beta*(v[h*128+d]-prediction),out=0;
    for(uint j=lane;j<128;j+=32){uint at=(h*128+d)*128+j;float updated=state[at]*decay+correction*k[kh*128+j];state[at]=updated;out+=updated*q[kh*128+j];}out=simd_sum(out);if(lane==0)y[h*128+d]=out;
}
kernel void split_q(device const float*x [[buffer(0)]],device float*q [[buffer(1)]],device float*gate [[buffer(2)]],constant P&p [[buffer(15)]],uint i [[thread_position_in_grid]]) {if(i<4096){uint h=i/256,d=i%256;q[i]=x[h*512+d];gate[i]=x[h*512+256+d];}}
inline float rotated(device const float*x,uint head,uint d,uint position,float theta) {
    if(d>=64)return x[head*256+d];uint j=d%32;float angle=float(position)*pow(theta,-float(2*j)/64.0f),a=x[head*256+j],b=x[head*256+32+j];return d<32?a*cos(angle)-b*sin(angle):b*cos(angle)+a*sin(angle);
}
kernel void rope_cache(device const float*q [[buffer(0)]],device const float*k [[buffer(1)]],device const float*v [[buffer(2)]],device float*qr [[buffer(3)]],device float*kc [[buffer(4)]],device float*vc [[buffer(5)]],constant P&p [[buffer(15)]],uint i [[thread_position_in_grid]]) {
    if(i<4096)qr[i]=rotated(q,i/256,i%256,p.pos,p.theta);if(i<512){kv_store(kc,p.pos*512+i,rotated(k,i/256,i%256,p.pos,p.theta),p.flags);kv_store(vc,p.pos*512+i,v[i],p.flags);}
}
kernel void attention_scores(device const float*q [[buffer(0)]],device const float*k [[buffer(1)]],device float*scores [[buffer(2)]],constant P&p [[buffer(15)]],uint g [[thread_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
    uint row=g/32,h=row/(p.pos+1),t=row%(p.pos+1);if(h>=16)return;float sum=0;for(uint d=lane;d<256;d+=32)sum+=q[h*256+d]*kv_load(k,(t*2+h/8)*256+d,p.flags);sum=simd_sum(sum);if(lane==0)scores[h*p.k+t]=sum*(1.0f/16.0f);
}
kernel void attention_softmax(device float*x [[buffer(0)]],constant P&p [[buffer(15)]],uint tid [[thread_index_in_threadgroup]],uint head [[threadgroup_position_in_grid]]) {
    threadgroup float partial[8];threadgroup float maximum,denominator;float top=-INFINITY;
    for(uint i=tid;i<=p.pos;i+=256)top=max(top,x[head*p.k+i]);top=simd_max(top);if((tid&31)==0)partial[tid/32]=top;threadgroup_barrier(mem_flags::mem_threadgroup);top=simd_max(tid<8?partial[tid]:-INFINITY);if(tid==0)maximum=top;threadgroup_barrier(mem_flags::mem_threadgroup);
    float sum=0;for(uint i=tid;i<=p.pos;i+=256){float value=exp(x[head*p.k+i]-maximum);x[head*p.k+i]=value;sum+=value;}sum=simd_sum(sum);if((tid&31)==0)partial[tid/32]=sum;threadgroup_barrier(mem_flags::mem_threadgroup);sum=simd_sum(tid<8?partial[tid]:0);if(tid==0)denominator=sum;threadgroup_barrier(mem_flags::mem_threadgroup);for(uint i=tid;i<=p.pos;i+=256)x[head*p.k+i]/=denominator;
}
kernel void attention_values(device const float*scores [[buffer(0)]],device const float*v [[buffer(1)]],device const float*gate [[buffer(2)]],device float*y [[buffer(3)]],constant P&p [[buffer(15)]],uint i [[thread_position_in_grid]]) {
    if(i>=4096)return;uint h=i/256,d=i%256;float sum=0;for(uint t=0;t<=p.pos;++t)sum+=scores[h*p.k+t]*kv_load(v,(t*2+h/8)*256+d,p.flags);y[i]=sum*sigmoid(gate[i]);
}
kernel void argmax(device const float*x [[buffer(0)]],device int*out [[buffer(1)]],constant P&p [[buffer(15)]],uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float values[8];threadgroup uint indices[8];float best=-INFINITY;uint index=0xffffffff;
    for(uint i=tid;i<p.n;i+=256){if((p.flags&1)&&(i==p.a||i==p.b))continue;float value=x[i];if(value>best||(value==best&&i<index)){best=value;index=i;}}
    float top=simd_max(best);uint at=simd_min(best==top?index:0xffffffff);if((tid&31)==0){values[tid/32]=top;indices[tid/32]=at;}threadgroup_barrier(mem_flags::mem_threadgroup);best=tid<8?values[tid]:-INFINITY;index=tid<8?indices[tid]:0xffffffff;top=simd_max(best);at=simd_min(best==top?index:0xffffffff);if(tid==0)out[0]=int(at);
}

// Layer-wise prefill kernels. FP8 remains packed; all tile operands and
// accumulators are float32. The original single-token path remains for decode.
struct ExpertTile { uint expert,first,count,pad; };
kernel void batch_embedding(device const ushort*w [[buffer(0)]],device const int*tokens [[buffer(1)]],device float*y [[buffer(2)]],constant P&p [[buffer(15)]],uint i [[thread_position_in_grid]]) {if(i<p.m*2048)y[i]=bf(w[uint(tokens[i/2048])*2048+i%2048]);}
kernel void batch_embedding_affine(device const uchar*w [[buffer(0)]],device const int*tokens [[buffer(1)]],device float*y [[buffer(2)]],device const ushort*s [[buffer(3)]],constant P&p [[buffer(15)]],uint i [[thread_position_in_grid]]) {if(i<p.m*2048)y[i]=weight(w,s,uint(tokens[i/2048]),i%2048,2048,3);}

kernel void matrix_product_small(device const float*x [[buffer(0)]],device const uchar*w [[buffer(1)]],device const ushort*s [[buffer(2)]],device float*y [[buffer(3)]],constant P&p [[buffer(15)]],uint tid [[thread_index_in_threadgroup]],uint sid [[simdgroup_index_in_threadgroup]],uint3 group [[threadgroup_position_in_grid]]) {
    threadgroup float xt[16*33],wt[32*33],result[16*33];
    uint firstRow=group.y*16,firstColumn=group.x*32;
    for(uint i=tid;i<16*33;i+=128)result[i]=0;threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 c0,c1;simdgroup_load(c0,result+sid*8,33);simdgroup_load(c1,result+8*33+sid*8,33);
    for(uint block=0;block<p.k;block+=32){
        for(uint i=tid;i<16*32;i+=128){uint row=i/32,col=i%32;xt[row*33+col]=firstRow+row<p.m?x[(firstRow+row)*p.k+block+col]:0;}
        for(uint i=tid;i<32*32;i+=128){uint row=i/32,col=i%32;wt[col*33+row]=firstColumn+row<p.n?weight(w,s,firstColumn+row,block+col,p.k,p.kind):0;}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint kk=0;kk<32;kk+=8){simdgroup_float8x8 a0,a1,b;simdgroup_load(a0,xt+kk,33);simdgroup_load(a1,xt+8*33+kk,33);simdgroup_load(b,wt+kk*33+sid*8,33);simdgroup_multiply_accumulate(c0,a0,b,c0);simdgroup_multiply_accumulate(c1,a1,b,c1);}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_store(c0,result+sid*8,33);simdgroup_store(c1,result+8*33+sid*8,33);threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint i=tid;i<16*32;i+=128){uint row=i/32,col=i%32;if(firstRow+row<p.m&&firstColumn+col<p.n)y[(firstRow+row)*p.n+firstColumn+col]=result[row*33+col];}
}

// Wider FP32 tiles reuse each decoded weight across 32 prompt tokens. Staging
// buffers are reused for the output so the threadgroup footprint stays <32 KiB.
kernel void matrix_product(device const float*x [[buffer(0)]],device const uchar*w [[buffer(1)]],device const ushort*s [[buffer(2)]],device float*y [[buffer(3)]],constant P&p [[buffer(15)]],uint tid [[thread_index_in_threadgroup]],uint sid [[simdgroup_index_in_threadgroup]],uint3 group [[threadgroup_position_in_grid]]) {
    threadgroup float xt[32*65],wt[64*65];
    uint firstRow=group.y*32,firstColumn=group.x*64,rr=(sid/4)*16,cc=(sid%4)*16;
    for(uint i=tid;i<32*65;i+=256)xt[i]=0;threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 c00,c01,c10,c11;
    simdgroup_load(c00,xt+rr*65+cc,65);simdgroup_load(c01,xt+rr*65+cc+8,65);
    simdgroup_load(c10,xt+(rr+8)*65+cc,65);simdgroup_load(c11,xt+(rr+8)*65+cc+8,65);threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint block=0;block<p.k;block+=64){
        for(uint i=tid;i<32*16;i+=256){uint row=i/16,col=(i%16)*4;float4 value=firstRow+row<p.m?*((device const float4*)(x+(firstRow+row)*p.k+block+col)):float4(0);for(uint d=0;d<4;++d)xt[row*65+col+d]=value[d];}
        for(uint i=tid;i<64*16;i+=256){uint row=i/16,col=(i%16)*4;float4 value=firstColumn+row<p.n?weight4(w,s,firstColumn+row,block+col,p.k,p.kind):float4(0);for(uint d=0;d<4;++d)wt[(col+d)*65+row]=value[d];}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint kk=0;kk<64;kk+=8){
            simdgroup_float8x8 a0,a1,b0,b1;simdgroup_load(a0,xt+rr*65+kk,65);simdgroup_load(a1,xt+(rr+8)*65+kk,65);simdgroup_load(b0,wt+kk*65+cc,65);simdgroup_load(b1,wt+kk*65+cc+8,65);
            simdgroup_multiply_accumulate(c00,a0,b0,c00);simdgroup_multiply_accumulate(c01,a0,b1,c01);simdgroup_multiply_accumulate(c10,a1,b0,c10);simdgroup_multiply_accumulate(c11,a1,b1,c11);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_store(c00,xt+rr*65+cc,65);simdgroup_store(c01,xt+rr*65+cc+8,65);simdgroup_store(c10,xt+(rr+8)*65+cc,65);simdgroup_store(c11,xt+(rr+8)*65+cc+8,65);threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint i=tid;i<32*64;i+=256){uint row=i/64,col=i%64;if(firstRow+row<p.m&&firstColumn+col<p.n)y[(firstRow+row)*p.n+firstColumn+col]=xt[row*65+col];}
}

kernel void batch_router(device const float*x [[buffer(0)]],device int*ids [[buffer(1)]],device float*prob [[buffer(2)]],constant P&p [[buffer(15)]],uint g [[thread_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
    uint token=g/32;if(token>=p.k)return;route_simd(x+token*256,ids+token*8,prob+token*8,lane);
}
kernel void expert_counts(device const int*ids [[buffer(0)]],device uint*counts [[buffer(1)]],constant P&p [[buffer(15)]],uint e [[thread_position_in_grid]]) {if(e<256){uint count=0;for(uint i=0;i<p.m*8;++i)count+=ids[i]==int(e);counts[e]=count;}}
kernel void expert_tiles(device const uint*counts [[buffer(0)]],device uint*offsets [[buffer(1)]],device ExpertTile*tiles [[buffer(2)]],device uint*tileCount [[buffer(3)]],constant P&p [[buffer(15)]],uint i [[thread_position_in_grid]]) {
    if(i)return;uint rows=p.a?p.a:16;uint offset=0,tile=0;for(uint e=0;e<256;++e){offsets[e]=offset;for(uint j=0;j<counts[e];j+=rows)tiles[tile++]={e,offset+j,min(rows,counts[e]-j),0};offset+=counts[e];}offsets[256]=offset;tileCount[0]=tile;
}
kernel void expert_group(device const int*ids [[buffer(0)]],device const uint*offsets [[buffer(1)]],device uint*tokens [[buffer(2)]],device uint*inverse [[buffer(3)]],constant P&p [[buffer(15)]],uint e [[thread_position_in_grid]]) {
    if(e>=256)return;uint at=offsets[e];for(uint i=0;i<p.m*8;++i)if(ids[i]==int(e)){tokens[at]=i/8;inverse[i]=at++;}
}
kernel void expert_matrix(device const float*x [[buffer(0)]],device const uchar*raw [[buffer(1)]],device const Expert*experts [[buffer(2)]],device const ExpertTile*tiles [[buffer(3)]],device const uint*tileCount [[buffer(4)]],device const uint*tokens [[buffer(5)]],device float*y [[buffer(6)]],constant P&p [[buffer(15)]],uint tid [[thread_index_in_threadgroup]],uint sid [[simdgroup_index_in_threadgroup]],uint3 group [[threadgroup_position_in_grid]]) {
    if(group.y>=tileCount[0])return;ExpertTile tile=tiles[group.y];Expert descriptor=experts[tile.expert];uint firstColumn=group.x*64;
    ulong woff=p.kind==0?descriptor.gate:p.kind==1?descriptor.up:descriptor.down,soff=p.kind==0?descriptor.gate_scale:p.kind==1?descriptor.up_scale:descriptor.down_scale;
    device const uchar*w=raw+woff;device const ushort*s=(device const ushort*)(raw+soff);
    threadgroup float xt[16*65],wt[64*65];
    for(uint i=tid;i<16*65;i+=256)xt[i]=0;threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 c0,c1;simdgroup_load(c0,xt+sid*8,65);simdgroup_load(c1,xt+8*65+sid*8,65);threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint block=0;block<p.k;block+=64){
        for(uint i=tid;i<16*16;i+=256){uint row=i/16,col=(i%16)*4;uint inputRow=row<tile.count?(p.kind==2?tile.first+row:tokens[tile.first+row]):0;float4 value=row<tile.count?*((device const float4*)(x+inputRow*p.k+block+col)):float4(0);for(uint d=0;d<4;++d)xt[row*65+col+d]=value[d];}
        for(uint i=tid;i<64*16;i+=256){uint row=i/16,col=(i%16)*4;float4 value=firstColumn+row<p.n?weight4(w,s,firstColumn+row,block+col,p.k,0):float4(0);for(uint d=0;d<4;++d)wt[(col+d)*65+row]=value[d];}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint kk=0;kk<64;kk+=8){simdgroup_float8x8 a0,a1,b;simdgroup_load(a0,xt+kk,65);simdgroup_load(a1,xt+8*65+kk,65);simdgroup_load(b,wt+kk*65+sid*8,65);simdgroup_multiply_accumulate(c0,a0,b,c0);simdgroup_multiply_accumulate(c1,a1,b,c1);}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_store(c0,xt+sid*8,65);simdgroup_store(c1,xt+8*65+sid*8,65);threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint i=tid;i<16*64;i+=256){uint row=i/64,col=i%64;if(row<tile.count&&firstColumn+col<p.n)y[(tile.first+row)*p.n+firstColumn+col]=xt[row*65+col];}
}

kernel void batch_mix(device const float*residual [[buffer(0)]],device const float*expert [[buffer(1)]],device const uint*inverse [[buffer(2)]],device const float*prob [[buffer(3)]],device const float*shared [[buffer(4)]],device const float*gate [[buffer(5)]],device float*y [[buffer(6)]],constant P&p [[buffer(15)]],uint i [[thread_position_in_grid]]) {
    if(i>=p.m*2048)return;uint token=i/2048,d=i%2048;float value=residual[i]+shared[i]*sigmoid(gate[token]);for(uint e=0;e<8;++e)value+=prob[token*8+e]*expert[inverse[token*8+e]*2048+d];y[i]=value;
}
// Four adjacent dimensions share routing metadata and the scalar shared gate.
// Keep FP32 storage and the original left-to-right expert accumulation order.
kernel void batch_mix_vec4(device const float4*residual [[buffer(0)]],device const float4*expert [[buffer(1)]],device const uint*inverse [[buffer(2)]],device const float*prob [[buffer(3)]],device const float4*shared [[buffer(4)]],device const float*gate [[buffer(5)]],device float4*y [[buffer(6)]],constant P&p [[buffer(15)]],uint i [[thread_position_in_grid]]) {
    if(i>=p.m*512)return;uint token=i/512,d=i%512;
    float4 value=residual[i]+shared[i]*sigmoid(gate[token]);
    for(uint e=0;e<8;++e)value+=prob[token*8+e]*expert[inverse[token*8+e]*512+d];
    y[i]=value;
}
kernel void batch_conv(device const float*x [[buffer(0)]],device const ushort*w [[buffer(1)]],device float*state [[buffer(2)]],device float*y [[buffer(3)]],device float*tape [[buffer(4)]],constant P&p [[buffer(15)]],uint i [[thread_position_in_grid]]) {
    if(i>=8192)return;float s0=state[i],s1=state[8192+i],s2=state[16384+i],w0=bf(w[i*4]),w1=bf(w[i*4+1]),w2=bf(w[i*4+2]),w3=bf(w[i*4+3]);
    for(uint t=0;t<p.m;++t){float value=x[t*8192+i];y[t*8192+i]=silu(s0*w0+s1*w1+s2*w2+value*w3);s0=s1;s1=s2;s2=value;if(p.flags&1){tape[t*24576+i]=s0;tape[t*24576+8192+i]=s1;tape[t*24576+16384+i]=s2;}}state[i]=s0;state[8192+i]=s1;state[16384+i]=s2;
}
// Read the original carry throughout the dispatch. Updating it is a separate
// dispatch: a last-token thread cannot overwrite history still read by t=0..2.
inline float prefill_conv_value(device const float*x,device const ushort*w,device const float*state,uint t,uint c) {
    float s0=t>=3?x[(t-3)*8192+c]:state[t*8192+c];
    float s1=t>=2?x[(t-2)*8192+c]:state[(t+1)*8192+c];
    float s2=t>=1?x[(t-1)*8192+c]:state[16384+c];
    return silu(s0*bf(w[c*4])+s1*bf(w[c*4+1])+s2*bf(w[c*4+2])+x[t*8192+c]*bf(w[c*4+3]));
}
kernel void batch_conv_parallel(device const float*x [[buffer(0)]],device const ushort*w [[buffer(1)]],device const float*state [[buffer(2)]],device float*y [[buffer(3)]],constant P&p [[buffer(15)]],uint i [[thread_position_in_grid]]) {
    if(i<p.m*8192)y[i]=prefill_conv_value(x,w,state,i/8192,i%8192);
}
kernel void batch_conv_carry(device const float*x [[buffer(0)]],device float*state [[buffer(1)]],constant P&p [[buffer(15)]],uint c [[thread_position_in_grid]]) {
    if(c>=8192)return;
    float s0=p.m>=3?x[(p.m-3)*8192+c]:state[p.m*8192+c];
    float s1=p.m>=2?x[(p.m-2)*8192+c]:state[(p.m+1)*8192+c];
    float s2=x[(p.m-1)*8192+c];
    state[c]=s0;state[8192+c]=s1;state[16384+c]=s2;
}
// FP32 convolution, q/k normalization and gates, with the baseline reduction
// order and recurrence inputs. Only V occupies the Conv buffer on this path.
kernel void delta_prepare_fused(device const float*x [[buffer(0)]],device const ushort*w [[buffer(1)]],device const float*state [[buffer(2)]],device float*v [[buffer(3)]],device float*q [[buffer(4)]],device float*k [[buffer(5)]],device const float*a [[buffer(6)]],device const float*b [[buffer(7)]],device const ushort*alog [[buffer(8)]],device const ushort*dt [[buffer(9)]],device float*gates [[buffer(10)]],constant P&p [[buffer(15)]],uint tid [[thread_index_in_threadgroup]],uint group [[threadgroup_position_in_grid]]) {
    uint token=group/16,head=group%16,j=head*128+tid;
    float query=prefill_conv_value(x,w,state,token,j),key=prefill_conv_value(x,w,state,token,2048+j);
    threadgroup float qs[4],ks[4];threadgroup float qi,ki;
    float qa=simd_sum(query*query),ka=simd_sum(key*key);
    if((tid&31)==0){qs[tid/32]=qa;ks[tid/32]=ka;}threadgroup_barrier(mem_flags::mem_threadgroup);
    qa=simd_sum(tid<4?qs[tid]:0);ka=simd_sum(tid<4?ks[tid]:0);
    if(tid==0){qi=rsqrt(qa+1e-6f)*rsqrt(128.0f);ki=rsqrt(ka+1e-6f);}threadgroup_barrier(mem_flags::mem_threadgroup);
    q[token*2048+j]=query*qi;k[token*2048+j]=key*ki;
    for(uint h=0;h<2;h++){uint c=4096+(head*2+h)*128+tid;v[token*8192+c]=prefill_conv_value(x,w,state,token,c);}
    if(tid<2){uint h=head*2+tid,i=token*32+h;gates[token*64+h]=exp(-exp(bf(alog[h]))*softplus(a[i]+bf(dt[h])));gates[token*64+32+h]=sigmoid(b[i]);}
}
kernel void batch_l2(device const float*x [[buffer(0)]],device float*q [[buffer(1)]],device float*k [[buffer(2)]],constant P&p [[buffer(15)]],uint tid [[thread_index_in_threadgroup]],uint group [[threadgroup_position_in_grid]]) {
    uint token=group/16,head=group%16,j=head*128+tid;threadgroup float qs[4],ks[4];threadgroup float qi,ki;float a=x[token*8192+j],b=x[token*8192+2048+j];float qa=simd_sum(a*a),ka=simd_sum(b*b);
    if((tid&31)==0){qs[tid/32]=qa;ks[tid/32]=ka;}threadgroup_barrier(mem_flags::mem_threadgroup);qa=simd_sum(tid<4?qs[tid]:0);ka=simd_sum(tid<4?ks[tid]:0);if(tid==0){qi=rsqrt(qa+1e-6f)*rsqrt(128.0f);ki=rsqrt(ka+1e-6f);}threadgroup_barrier(mem_flags::mem_threadgroup);q[token*2048+j]=a*qi;k[token*2048+j]=b*ki;
}
kernel void batch_delta(device const float*q [[buffer(0)]],device const float*k [[buffer(1)]],device const float*v [[buffer(2)]],device const float*a [[buffer(3)]],device const float*b [[buffer(4)]],device const ushort*alog [[buffer(5)]],device const ushort*dt [[buffer(6)]],device float*state [[buffer(7)]],device float*y [[buffer(8)]],device float*tape [[buffer(9)]],constant P&p [[buffer(15)]],uint g [[thread_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
    uint row=g/32,h=row/128,d=row%128;if(h>=32)return;uint base=(h*128+d)*128+lane,kh=h/2;float4 cell=float4(state[base],state[base+32],state[base+64],state[base+96]);float gain=exp(bf(alog[h])),bias=bf(dt[h]);
    for(uint t=0;t<p.m;++t){uint key=t*2048+kh*128+lane;float4 kv=float4(k[key],k[key+32],k[key+64],k[key+96]),qv=float4(q[key],q[key+32],q[key+64],q[key+96]);cell*=exp(-gain*softplus(a[t*32+h]+bias));float prediction=simd_sum(dot(cell,kv)),correction=sigmoid(b[t*32+h])*(v[t*8192+4096+h*128+d]-prediction);cell+=correction*kv;float value=simd_sum(dot(cell,qv));if(lane==0)y[t*4096+h*128+d]=value;if(p.flags&1){uint at=t*524288+base;tape[at]=cell.x;tape[at+32]=cell.y;tape[at+64]=cell.z;tape[at+96]=cell.w;}}
    state[base]=cell.x;state[base+32]=cell.y;state[base+64]=cell.z;state[base+96]=cell.w;
}
kernel void batch_split_q(device const float*x [[buffer(0)]],device float*q [[buffer(1)]],device float*gate [[buffer(2)]],constant P&p [[buffer(15)]],uint i [[thread_position_in_grid]]) {if(i<p.m*4096){uint token=i/4096,h=(i%4096)/256,d=i%256;q[i]=x[token*8192+h*512+d];gate[i]=x[token*8192+h*512+256+d];}}
kernel void batch_rope(device const float*q [[buffer(0)]],device const float*k [[buffer(1)]],device const float*v [[buffer(2)]],device float*qr [[buffer(3)]],device float*kc [[buffer(4)]],device float*vc [[buffer(5)]],constant P&p [[buffer(15)]],uint i [[thread_position_in_grid]]) {
    if(i>=p.m*4096)return;uint token=i/4096,j=i%4096;qr[i]=rotated(q+token*4096,j/256,j%256,p.pos+token,p.theta);if(j<512){kv_store(kc,(p.pos+token)*512+j,rotated(k+token*512,j/256,j%256,p.pos+token,p.theta),p.flags);kv_store(vc,(p.pos+token)*512+j,v[token*512+j],p.flags);}
}
kernel void batch_attention_scores(device const float*q [[buffer(0)]],device const float*k [[buffer(1)]],device float*scores [[buffer(2)]],constant P&p [[buffer(15)]],uint3 g [[thread_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
    uint token=g.y/16,h=g.y%16,t=g.x/32;if(token>=p.m)return;float sum=0;if(t<=p.pos+token)for(uint d=lane;d<256;d+=32)sum+=q[token*4096+h*256+d]*kv_load(k,(t*2+h/8)*256+d,p.flags);sum=simd_sum(sum);if(lane==0)scores[(token*16+h)*p.k+t]=t<=p.pos+token?sum*(1.0f/16.0f):-INFINITY;
}
kernel void batch_attention_values(device const float*scores [[buffer(0)]],device const float*v [[buffer(1)]],device const float*gate [[buffer(2)]],device float*y [[buffer(3)]],constant P&p [[buffer(15)]],uint i [[thread_position_in_grid]]) {
    if(i>=p.m*4096)return;uint token=i/4096,h=(i%4096)/256,d=i%256;float sum=0;for(uint t=0;t<=p.pos+token;++t)sum+=scores[(token*16+h)*p.k+t]*kv_load(v,(t*2+h/8)*256+d,p.flags);y[i]=sum*sigmoid(gate[i]);
}

// Copies also support Binding byte offsets for accepted-row state commits.
kernel void copy_floats(device const float*x [[buffer(0)]],device float*y [[buffer(1)]],constant P&p [[buffer(15)]],uint i [[thread_position_in_grid]]) { if(i<p.n)y[i]=x[i]; }

// DFlash2: concatenate post-layer target features, then retain projected context.
kernel void draft_capture(device const float*x [[buffer(0)]],device float*y [[buffer(1)]],constant P&p [[buffer(15)]],uint i [[thread_position_in_grid]]) {
    if(i<p.m*2048)y[(i/2048)*16384+p.a*2048+i%2048]=x[i];
}
kernel void draft_context_store(device const float*x [[buffer(0)]],device float*y [[buffer(1)]],constant P&p [[buffer(15)]],uint i [[thread_position_in_grid]]) {
    if(i<p.m*2048)y[((p.pos+i/2048)%2048)*2048+i%2048]=x[i];
}
kernel void draft_context_load(device const float*x [[buffer(0)]],device float*y [[buffer(1)]],constant P&p [[buffer(15)]],uint i [[thread_position_in_grid]]) {
    if(i<p.m*2048)y[i]=x[((p.pos+i/2048)%2048)*2048+i%2048];
}
kernel void draft_dynamic_conv(device const float*x [[buffer(0)]],device const float*dynamic [[buffer(1)]],device const ushort*base [[buffer(2)]],device float*y [[buffer(3)]],constant P&p [[buffer(15)]],uint i [[thread_position_in_grid]]) {
    if(i>=p.m*2048)return;uint row=i/2048,d=i%2048,group=d/16;float sum=0;
    for(uint offset=0;offset<2;++offset)if(row>=offset){float value=x[(row-offset)*2048+d];sum+=bf(base[(p.a*2+offset)*2048+d])*value;sum+=dynamic[row*512+p.a*256+offset*128+group]*value;}
    y[i]=sum;
}
inline float draft_rotated(device const float*x,uint d,uint pos,float theta) {
    uint j=d%128,base=d-j;float angle=float(pos)*pow(theta,-float(j%64)/64.0f),c=cos(angle),s=sin(angle);
    return x[d]*c+(j<64?-x[base+j+64]:x[base+j-64])*s;
}
kernel void draft_context_kv(device const float*k [[buffer(0)]],device const float*v [[buffer(1)]],device float*kc [[buffer(2)]],device float*vc [[buffer(3)]],constant P&p [[buffer(15)]],uint i [[thread_position_in_grid]]) {
    if(i>=p.m*1024)return;uint row=i/1024,d=i%1024,at=((p.pos+row)%2048)*1024+d;
    kv_store(kc,at,draft_rotated(k+row*1024,d,p.pos+row,p.theta),p.flags);kv_store(vc,at,v[i],p.flags);
}
// Each thread owns a complete rotary pair, making this in-place update race-free.
kernel void draft_rotate(device float*q [[buffer(0)]],device float*k [[buffer(1)]],constant P&p [[buffer(15)]],uint i [[thread_position_in_grid]]) {
    if(i>=p.m*2048)return;uint row=i/2048,pair=i%2048,head=pair/64,d=pair%64,at=row*4096+head*128+d;
    float angle=float(p.pos+row)*pow(p.theta,-float(d)/64.0f),c=cos(angle),s=sin(angle),a=q[at],b=q[at+64];q[at]=a*c-b*s;q[at+64]=b*c+a*s;
    if(pair<512){at=row*1024+head*128+d;a=k[at];b=k[at+64];k[at]=a*c-b*s;k[at+64]=b*c+a*s;}
}
kernel void draft_attention_scores(device const float*q [[buffer(0)]],device const float*k [[buffer(1)]],device const float*kc [[buffer(2)]],device float*scores [[buffer(3)]],constant P&p [[buffer(15)]],uint i [[thread_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
    uint index=i/32,slot=index%p.k,h=(index/p.k)%32,row=index/(p.k*32);if(row>=p.m)return;
    uint start=p.pos>2048?p.pos-2048:0,t=start+slot;bool context=slot<p.a,valid=context?(t<p.pos && p.pos+row-t<2048):(slot-p.a<p.m);
    float sum=0;if(valid){uint at=context?((t%2048)*8+h/4)*128:((slot-p.a)*8+h/4)*128;for(uint d=lane;d<128;d+=32)sum+=q[(row*32+h)*128+d]*(context?kv_load(kc,at+d,p.flags):k[at+d]);}
    sum=simd_sum(sum);if(lane==0)scores[index]=valid?sum*rsqrt(128.0f):-INFINITY;
}
kernel void draft_attention_values(device const float*scores [[buffer(0)]],device const float*v [[buffer(1)]],device const float*vc [[buffer(2)]],device float*y [[buffer(3)]],constant P&p [[buffer(15)]],uint i [[thread_position_in_grid]]) {
    if(i>=p.m*4096)return;uint row=i/4096,h=(i%4096)/128,d=i%128,start=p.pos>2048?p.pos-2048:0;float sum=0;
    for(uint t=start;t<p.pos;++t)sum+=scores[(row*32+h)*p.k+t-start]*kv_load(vc,((t%2048)*8+h/4)*128+d,p.flags);
    for(uint t=0;t<p.m;++t)sum+=scores[(row*32+h)*p.k+p.a+t]*v[(t*8+h/4)*128+d];y[i]=sum;
}
// Greedy top-16 unary shortlist followed by the learned predecessor/successor
// compatibility score. Anchor + selected predecessors form the candidate path.
kernel void draft_select(device const float*logits [[buffer(0)]],device const float*hidden [[buffer(1)]],device const ushort*pre [[buffer(2)]],device const ushort*post [[buffer(3)]],device int*out [[buffer(4)]],constant P&p [[buffer(15)]],uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float partial[8],unary[16],compat[16];threadgroup uint indices[8],candidates[16],predecessor;
    if(tid==0)predecessor=p.pos;threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint row=1;row<p.m;++row){
        float best[16];uint ids[16];for(uint j=0;j<16;++j){best[j]=-INFINITY;ids[j]=0xffffffff;}
        for(uint token=tid;token<p.n;token+=256){float value=logits[row*p.n+token];if(value<best[15]||(value==best[15]&&token>=ids[15]))continue;uint at=15;while(at>0&&(value>best[at-1]||(value==best[at-1]&&token<ids[at-1]))){best[at]=best[at-1];ids[at]=ids[at-1];--at;}best[at]=value;ids[at]=token;}
        for(uint choice=0;choice<16;++choice){
            float top=simd_max(best[0]);uint index=simd_min(best[0]==top?ids[0]:0xffffffff);
            if((tid&31)==0){partial[tid/32]=top;indices[tid/32]=index;}threadgroup_barrier(mem_flags::mem_threadgroup);
            float value=tid<8?partial[tid]:-INFINITY;uint id=tid<8?indices[tid]:0xffffffff;top=simd_max(value);index=simd_min(value==top?id:0xffffffff);
            if(tid==0){unary[choice]=top;candidates[choice]=index;}threadgroup_barrier(mem_flags::mem_threadgroup);
            if(ids[0]==candidates[choice]){for(uint j=0;j<15;++j){best[j]=best[j+1];ids[j]=ids[j+1];}best[15]=-INFINITY;ids[15]=0xffffffff;}
        }
        // One SIMD group per candidate, two waves for 16 candidates.
        for(uint wave=0;wave<2;++wave){uint choice=wave*8+tid/32;float sum=0;for(uint d=tid%32;d<256;d+=32)sum+=bf(pre[predecessor*256+d])*hidden[row*256+d]*bf(post[candidates[choice]*256+d]);sum=simd_sum(sum);if((tid&31)==0)compat[choice]=unary[choice]+sum;}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(tid==0){uint winner=0;for(uint j=1;j<16;++j)if(compat[j]>compat[winner]||(compat[j]==compat[winner]&&candidates[j]<candidates[winner]))winner=j;predecessor=candidates[winner];out[row-1]=int(predecessor);}threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// Small-block verification/drafting reuses each decoded weight for up to eight
// input rows, without filling the 16/32-row tiles used by long prefill. Each
// row keeps the serial GEMV float4 accumulation order and FP32 precision.
kernel void matrix_product_tiny(device const float*x [[buffer(0)]],device const uchar*w [[buffer(1)]],device const ushort*s [[buffer(2)]],device float*y [[buffer(3)]],constant P&p [[buffer(15)]],uint g [[thread_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
    uint column=g/32;if(column>=p.n)return;float4 accum[8];
    for(uint r=0;r<8;++r)accum[r]=0;
    for(uint j=lane*4;j<p.k;j+=128){float4 weight=weight4(w,s,column,j,p.k,p.kind);
        #pragma unroll
        for(uint r=0;r<8;++r)if(r<p.m)accum[r]+=*((device const float4*)(x+r*p.k+j))*weight;
    }
    #pragma unroll
    for(uint r=0;r<8;++r)if(r<p.m){float4 a=accum[r];float sum=simd_sum(a.x+a.y+a.z+a.w);if(lane==0)y[r*p.n+column]=sum;}
}
kernel void expert_matrix_tiny(device const float*x [[buffer(0)]],device const uchar*raw [[buffer(1)]],device const Expert*experts [[buffer(2)]],device const ExpertTile*tiles [[buffer(3)]],device const uint*tileCount [[buffer(4)]],device const uint*tokens [[buffer(5)]],device float*y [[buffer(6)]],constant P&p [[buffer(15)]],uint3 g [[thread_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
    uint column=g.x/32;if(g.y>=tileCount[0]||column>=p.n)return;
    ExpertTile tile=tiles[g.y];Expert descriptor=experts[tile.expert];
    ulong woff=p.kind==0?descriptor.gate:p.kind==1?descriptor.up:descriptor.down,soff=p.kind==0?descriptor.gate_scale:p.kind==1?descriptor.up_scale:descriptor.down_scale;
    float4 accum[8];uint inputRows[8];
    for(uint r=0;r<8;++r){accum[r]=0;inputRows[r]=r<tile.count?(p.kind==2?tile.first+r:tokens[tile.first+r]):0;}
    for(uint j=lane*4;j<p.k;j+=128){float4 weight=expert_weight4(raw+woff,(device const ushort*)(raw+soff),column,j,p.k,p.flags);
        #pragma unroll
        for(uint r=0;r<8;++r)if(r<tile.count)accum[r]+=*((device const float4*)(x+inputRows[r]*p.k+j))*weight;
    }
    #pragma unroll
    for(uint r=0;r<8;++r)if(r<tile.count){float4 a=accum[r];float sum=simd_sum(a.x+a.y+a.z+a.w);if(lane==0)y[(tile.first+r)*p.n+column]=sum;}
}


kernel void matrix_product_small_mpp(device const float*x [[buffer(0)]],device const uchar*w [[buffer(1)]],device const ushort*s [[buffer(2)]],device float*y [[buffer(3)]],constant P&p [[buffer(15)]],uint tid [[thread_index_in_threadgroup]],uint sid [[simdgroup_index_in_threadgroup]],uint3 group [[threadgroup_position_in_grid]]) {
    threadgroup float xt[16*33],wt[32*33],result[16*33];
    uint firstRow=group.y*16,firstColumn=group.x*32;
    // Strict FP32 operands and accumulation: relaxed_precision is explicitly false.
    constexpr auto desc=matmul2d_descriptor(16,32,32,false,false,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc,execution_simdgroups<4>> op;
    auto left=tensor(xt,dextents<int,2>{32,16},array<int,2>{1,33});
    auto right=tensor(wt,dextents<int,2>{32,32},array<int,2>{1,33});
    auto acc=op.get_destination_cooperative_tensor<decltype(left),decltype(right),float>();
    #pragma unroll
    for(uint i=0;i<acc.get_capacity();++i) acc[i]=0.0f;
    for(uint block=0;block<p.k;block+=32){
        for(uint i=tid;i<16*32;i+=128){uint row=i/32,col=i%32;xt[row*33+col]=firstRow+row<p.m?x[(firstRow+row)*p.k+block+col]:0;}
        for(uint i=tid;i<32*32;i+=128){uint row=i/32,col=i%32;wt[col*33+row]=firstColumn+row<p.n?weight(w,s,firstColumn+row,block+col,p.k,p.kind):0;}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        op.run(left,right,acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    auto destination=tensor(result,dextents<int,2>{32,16},array<int,2>{1,33});
    acc.store(destination);threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint i=tid;i<16*32;i+=128){uint row=i/32,col=i%32;if(firstRow+row<p.m&&firstColumn+col<p.n)y[(firstRow+row)*p.n+firstColumn+col]=result[row*33+col];}
}


kernel void matrix_product_mpp(device const float*x [[buffer(0)]],device const uchar*w [[buffer(1)]],device const ushort*s [[buffer(2)]],device float*y [[buffer(3)]],constant P&p [[buffer(15)]],uint tid [[thread_index_in_threadgroup]],uint sid [[simdgroup_index_in_threadgroup]],uint3 group [[threadgroup_position_in_grid]]) {
    threadgroup float xt[32*65],wt[64*65];
    uint firstRow=group.y*32,firstColumn=group.x*64,rr=(sid/4)*16,cc=(sid%4)*16;
    // Strict FP32 operands and accumulation: relaxed_precision is explicitly false.
    constexpr auto desc=matmul2d_descriptor(32,64,64,false,false,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc,execution_simdgroups<8>> op;
    auto left=tensor(xt,dextents<int,2>{64,32},array<int,2>{1,65});
    auto right=tensor(wt,dextents<int,2>{64,64},array<int,2>{1,65});
    auto acc=op.get_destination_cooperative_tensor<decltype(left),decltype(right),float>();
    #pragma unroll
    for(uint i=0;i<acc.get_capacity();++i) acc[i]=0.0f;
    for(uint block=0;block<p.k;block+=64){
        for(uint i=tid;i<32*16;i+=256){uint row=i/16,col=(i%16)*4;float4 value=firstRow+row<p.m?*((device const float4*)(x+(firstRow+row)*p.k+block+col)):float4(0);for(uint d=0;d<4;++d)xt[row*65+col+d]=value[d];}
        for(uint i=tid;i<64*16;i+=256){uint row=i/16,col=(i%16)*4;float4 value=firstColumn+row<p.n?weight4(w,s,firstColumn+row,block+col,p.k,p.kind):float4(0);for(uint d=0;d<4;++d)wt[(col+d)*65+row]=value[d];}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        op.run(left,right,acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    auto destination=tensor(xt,dextents<int,2>{64,32},array<int,2>{1,65});
    acc.store(destination);threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint i=tid;i<32*64;i+=256){uint row=i/64,col=i%64;if(firstRow+row<p.m&&firstColumn+col<p.n)y[(firstRow+row)*p.n+firstColumn+col]=xt[row*65+col];}
}


kernel void expert_matrix_mpp(device const float*x [[buffer(0)]],device const uchar*raw [[buffer(1)]],device const Expert*experts [[buffer(2)]],device const ExpertTile*tiles [[buffer(3)]],device const uint*tileCount [[buffer(4)]],device const uint*tokens [[buffer(5)]],device float*y [[buffer(6)]],constant P&p [[buffer(15)]],uint tid [[thread_index_in_threadgroup]],uint sid [[simdgroup_index_in_threadgroup]],uint3 group [[threadgroup_position_in_grid]]) {
    if(group.y>=tileCount[0])return;ExpertTile tile=tiles[group.y];Expert descriptor=experts[tile.expert];uint firstColumn=group.x*64;
    ulong woff=p.kind==0?descriptor.gate:p.kind==1?descriptor.up:descriptor.down,soff=p.kind==0?descriptor.gate_scale:p.kind==1?descriptor.up_scale:descriptor.down_scale;
    device const uchar*w=raw+woff;device const ushort*s=(device const ushort*)(raw+soff);
    threadgroup float xt[16*65],wt[64*65];
    // Strict FP32 operands and accumulation: relaxed_precision is explicitly false.
    constexpr auto desc=matmul2d_descriptor(16,64,64,false,false,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc,execution_simdgroups<8>> op;
    auto left=tensor(xt,dextents<int,2>{64,16},array<int,2>{1,65});
    auto right=tensor(wt,dextents<int,2>{64,64},array<int,2>{1,65});
    auto acc=op.get_destination_cooperative_tensor<decltype(left),decltype(right),float>();
    #pragma unroll
    for(uint i=0;i<acc.get_capacity();++i) acc[i]=0.0f;
    for(uint block=0;block<p.k;block+=64){
        for(uint i=tid;i<16*16;i+=256){uint row=i/16,col=(i%16)*4;uint inputRow=row<tile.count?(p.kind==2?tile.first+row:tokens[tile.first+row]):0;float4 value=row<tile.count?*((device const float4*)(x+inputRow*p.k+block+col)):float4(0);for(uint d=0;d<4;++d)xt[row*65+col+d]=value[d];}
        for(uint i=tid;i<64*16;i+=256){uint row=i/16,col=(i%16)*4;float4 value=firstColumn+row<p.n?weight4(w,s,firstColumn+row,block+col,p.k,0):float4(0);for(uint d=0;d<4;++d)wt[(col+d)*65+row]=value[d];}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        op.run(left,right,acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    auto destination=tensor(xt,dextents<int,2>{64,16},array<int,2>{1,65});
    acc.store(destination);threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint i=tid;i<16*64;i+=256){uint row=i/64,col=i%64;if(row<tile.count&&firstColumn+col<p.n)y[(tile.first+row)*p.n+firstColumn+col]=xt[row*65+col];}
}


kernel void matrix_product_small_fp16(device const float*x [[buffer(0)]],device const uchar*w [[buffer(1)]],device const ushort*s [[buffer(2)]],device float*y [[buffer(3)]],constant P&p [[buffer(15)]],uint tid [[thread_index_in_threadgroup]],uint sid [[simdgroup_index_in_threadgroup]],uint3 group [[threadgroup_position_in_grid]]) {
    threadgroup half xt[16*33],wt[32*33];
    threadgroup float result[16*33];
    uint firstRow=group.y*16,firstColumn=group.x*32;
    // FP16 staged operands, FP32 accumulation and output; no relaxed precision.
    constexpr auto desc=matmul2d_descriptor(16,32,32,false,false,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc,execution_simdgroups<4>> op;
    auto left=tensor(xt,dextents<int,2>{32,16},array<int,2>{1,33});
    auto right=tensor(wt,dextents<int,2>{32,32},array<int,2>{1,33});
    auto acc=op.get_destination_cooperative_tensor<decltype(left),decltype(right),float>();
    #pragma unroll
    for(uint i=0;i<acc.get_capacity();++i) acc[i]=0.0f;
    for(uint block=0;block<p.k;block+=32){
        for(uint i=tid;i<16*32;i+=128){uint row=i/32,col=i%32;xt[row*33+col]=firstRow+row<p.m?x[(firstRow+row)*p.k+block+col]:0;}
        for(uint i=tid;i<32*32;i+=128){uint row=i/32,col=i%32;wt[col*33+row]=firstColumn+row<p.n?weight(w,s,firstColumn+row,block+col,p.k,p.kind):0;}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        op.run(left,right,acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    auto destination=tensor(result,dextents<int,2>{32,16},array<int,2>{1,33});
    acc.store(destination);threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint i=tid;i<16*32;i+=128){uint row=i/32,col=i%32;if(firstRow+row<p.m&&firstColumn+col<p.n)y[(firstRow+row)*p.n+firstColumn+col]=result[row*33+col];}
}


kernel void matrix_product_fp16(device const float*x [[buffer(0)]],device const uchar*w [[buffer(1)]],device const ushort*s [[buffer(2)]],device float*y [[buffer(3)]],constant P&p [[buffer(15)]],uint tid [[thread_index_in_threadgroup]],uint sid [[simdgroup_index_in_threadgroup]],uint3 group [[threadgroup_position_in_grid]]) {
    threadgroup half xt[32*65],wt[64*65];
    threadgroup float result[32*65];
    uint firstRow=group.y*32,firstColumn=group.x*64,rr=(sid/4)*16,cc=(sid%4)*16;
    // FP16 staged operands, FP32 accumulation and output; no relaxed precision.
    constexpr auto desc=matmul2d_descriptor(32,64,64,false,false,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc,execution_simdgroups<8>> op;
    auto left=tensor(xt,dextents<int,2>{64,32},array<int,2>{1,65});
    auto right=tensor(wt,dextents<int,2>{64,64},array<int,2>{1,65});
    auto acc=op.get_destination_cooperative_tensor<decltype(left),decltype(right),float>();
    #pragma unroll
    for(uint i=0;i<acc.get_capacity();++i) acc[i]=0.0f;
    for(uint block=0;block<p.k;block+=64){
        for(uint i=tid;i<32*16;i+=256){uint row=i/16,col=(i%16)*4;float4 value=firstRow+row<p.m?*((device const float4*)(x+(firstRow+row)*p.k+block+col)):float4(0);for(uint d=0;d<4;++d)xt[row*65+col+d]=value[d];}
        for(uint i=tid;i<64*16;i+=256){uint row=i/16,col=(i%16)*4;float4 value=firstColumn+row<p.n?weight4(w,s,firstColumn+row,block+col,p.k,p.kind):float4(0);for(uint d=0;d<4;++d)wt[(col+d)*65+row]=value[d];}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        op.run(left,right,acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    auto destination=tensor(result,dextents<int,2>{64,32},array<int,2>{1,65});
    acc.store(destination);threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint i=tid;i<32*64;i+=256){uint row=i/64,col=i%64;if(firstRow+row<p.m&&firstColumn+col<p.n)y[(firstRow+row)*p.n+firstColumn+col]=result[row*65+col];}
}


kernel void expert_matrix_fp16(device const float*x [[buffer(0)]],device const uchar*raw [[buffer(1)]],device const Expert*experts [[buffer(2)]],device const ExpertTile*tiles [[buffer(3)]],device const uint*tileCount [[buffer(4)]],device const uint*tokens [[buffer(5)]],device float*y [[buffer(6)]],constant P&p [[buffer(15)]],uint tid [[thread_index_in_threadgroup]],uint sid [[simdgroup_index_in_threadgroup]],uint3 group [[threadgroup_position_in_grid]]) {
    if(group.y>=tileCount[0])return;ExpertTile tile=tiles[group.y];Expert descriptor=experts[tile.expert];uint firstColumn=group.x*64;
    ulong woff=p.kind==0?descriptor.gate:p.kind==1?descriptor.up:descriptor.down,soff=p.kind==0?descriptor.gate_scale:p.kind==1?descriptor.up_scale:descriptor.down_scale;
    device const uchar*w=raw+woff;device const ushort*s=(device const ushort*)(raw+soff);
    threadgroup half xt[16*65],wt[64*65];
    threadgroup float result[16*65];
    // FP16 staged operands, FP32 accumulation and output; no relaxed precision.
    constexpr auto desc=matmul2d_descriptor(16,64,64,false,false,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc,execution_simdgroups<8>> op;
    auto left=tensor(xt,dextents<int,2>{64,16},array<int,2>{1,65});
    auto right=tensor(wt,dextents<int,2>{64,64},array<int,2>{1,65});
    auto acc=op.get_destination_cooperative_tensor<decltype(left),decltype(right),float>();
    #pragma unroll
    for(uint i=0;i<acc.get_capacity();++i) acc[i]=0.0f;
    for(uint block=0;block<p.k;block+=64){
        for(uint i=tid;i<16*16;i+=256){uint row=i/16,col=(i%16)*4;uint inputRow=row<tile.count?(p.kind==2?tile.first+row:tokens[tile.first+row]):0;float4 value=row<tile.count?*((device const float4*)(x+inputRow*p.k+block+col)):float4(0);for(uint d=0;d<4;++d)xt[row*65+col+d]=value[d];}
        for(uint i=tid;i<64*16;i+=256){uint row=i/16,col=(i%16)*4;float4 value=firstColumn+row<p.n?weight4(w,s,firstColumn+row,block+col,p.k,0):float4(0);for(uint d=0;d<4;++d)wt[(col+d)*65+row]=value[d];}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        op.run(left,right,acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    auto destination=tensor(result,dextents<int,2>{64,16},array<int,2>{1,65});
    acc.store(destination);threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint i=tid;i<16*64;i+=256){uint row=i/64,col=i%64;if(row<tile.count&&firstColumn+col<p.n)y[(tile.first+row)*p.n+firstColumn+col]=result[row*65+col];}
}


kernel void matrix_product_small_bf16(device const float*x [[buffer(0)]],device const uchar*w [[buffer(1)]],device const ushort*s [[buffer(2)]],device float*y [[buffer(3)]],constant P&p [[buffer(15)]],uint tid [[thread_index_in_threadgroup]],uint sid [[simdgroup_index_in_threadgroup]],uint3 group [[threadgroup_position_in_grid]]) {
    threadgroup bfloat xt[16*33],wt[32*33];
    threadgroup float result[16*33];
    uint firstRow=group.y*16,firstColumn=group.x*32;
    // BF16 staged operands, FP32 accumulation and output; no relaxed precision.
    constexpr auto desc=matmul2d_descriptor(16,32,32,false,false,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc,execution_simdgroups<4>> op;
    auto left=tensor(xt,dextents<int,2>{32,16},array<int,2>{1,33});
    auto right=tensor(wt,dextents<int,2>{32,32},array<int,2>{1,33});
    auto acc=op.get_destination_cooperative_tensor<decltype(left),decltype(right),float>();
    #pragma unroll
    for(uint i=0;i<acc.get_capacity();++i) acc[i]=0.0f;
    for(uint block=0;block<p.k;block+=32){
        for(uint i=tid;i<16*32;i+=128){uint row=i/32,col=i%32;xt[row*33+col]=bfloat(firstRow+row<p.m?x[(firstRow+row)*p.k+block+col]:0);}
        for(uint i=tid;i<32*32;i+=128){uint row=i/32,col=i%32;wt[col*33+row]=bfloat(firstColumn+row<p.n?weight(w,s,firstColumn+row,block+col,p.k,p.kind):0);}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        op.run(left,right,acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    auto destination=tensor(result,dextents<int,2>{32,16},array<int,2>{1,33});
    acc.store(destination);threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint i=tid;i<16*32;i+=128){uint row=i/32,col=i%32;if(firstRow+row<p.m&&firstColumn+col<p.n)y[(firstRow+row)*p.n+firstColumn+col]=result[row*33+col];}
}


kernel void matrix_product_bf16(device const float*x [[buffer(0)]],device const uchar*w [[buffer(1)]],device const ushort*s [[buffer(2)]],device float*y [[buffer(3)]],constant P&p [[buffer(15)]],uint tid [[thread_index_in_threadgroup]],uint sid [[simdgroup_index_in_threadgroup]],uint3 group [[threadgroup_position_in_grid]]) {
    threadgroup bfloat xt[32*65],wt[64*65];
    threadgroup float result[32*65];
    uint firstRow=group.y*32,firstColumn=group.x*64,rr=(sid/4)*16,cc=(sid%4)*16;
    // BF16 staged operands, FP32 accumulation and output; no relaxed precision.
    constexpr auto desc=matmul2d_descriptor(32,64,64,false,false,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc,execution_simdgroups<8>> op;
    auto left=tensor(xt,dextents<int,2>{64,32},array<int,2>{1,65});
    auto right=tensor(wt,dextents<int,2>{64,64},array<int,2>{1,65});
    auto acc=op.get_destination_cooperative_tensor<decltype(left),decltype(right),float>();
    #pragma unroll
    for(uint i=0;i<acc.get_capacity();++i) acc[i]=0.0f;
    for(uint block=0;block<p.k;block+=64){
        for(uint i=tid;i<32*16;i+=256){uint row=i/16,col=(i%16)*4;float4 value=firstRow+row<p.m?*((device const float4*)(x+(firstRow+row)*p.k+block+col)):float4(0);for(uint d=0;d<4;++d)xt[row*65+col+d]=bfloat(value[d]);}
        for(uint i=tid;i<64*16;i+=256){uint row=i/16,col=(i%16)*4;float4 value=firstColumn+row<p.n?weight4(w,s,firstColumn+row,block+col,p.k,p.kind):float4(0);for(uint d=0;d<4;++d)wt[(col+d)*65+row]=bfloat(value[d]);}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        op.run(left,right,acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    auto destination=tensor(result,dextents<int,2>{64,32},array<int,2>{1,65});
    acc.store(destination);threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint i=tid;i<32*64;i+=256){uint row=i/64,col=i%64;if(firstRow+row<p.m&&firstColumn+col<p.n)y[(firstRow+row)*p.n+firstColumn+col]=result[row*65+col];}
}


kernel void expert_matrix_bf16(device const float*x [[buffer(0)]],device const uchar*raw [[buffer(1)]],device const Expert*experts [[buffer(2)]],device const ExpertTile*tiles [[buffer(3)]],device const uint*tileCount [[buffer(4)]],device const uint*tokens [[buffer(5)]],device float*y [[buffer(6)]],constant P&p [[buffer(15)]],uint tid [[thread_index_in_threadgroup]],uint sid [[simdgroup_index_in_threadgroup]],uint3 group [[threadgroup_position_in_grid]]) {
    if(group.y>=tileCount[0])return;ExpertTile tile=tiles[group.y];Expert descriptor=experts[tile.expert];uint firstColumn=group.x*64;
    ulong woff=p.kind==0?descriptor.gate:p.kind==1?descriptor.up:descriptor.down,soff=p.kind==0?descriptor.gate_scale:p.kind==1?descriptor.up_scale:descriptor.down_scale;
    device const uchar*w=raw+woff;device const ushort*s=(device const ushort*)(raw+soff);
    threadgroup bfloat xt[16*65],wt[64*65];
    threadgroup float result[16*65];
    // BF16 staged operands, FP32 accumulation and output; no relaxed precision.
    constexpr auto desc=matmul2d_descriptor(16,64,64,false,false,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc,execution_simdgroups<8>> op;
    auto left=tensor(xt,dextents<int,2>{64,16},array<int,2>{1,65});
    auto right=tensor(wt,dextents<int,2>{64,64},array<int,2>{1,65});
    auto acc=op.get_destination_cooperative_tensor<decltype(left),decltype(right),float>();
    #pragma unroll
    for(uint i=0;i<acc.get_capacity();++i) acc[i]=0.0f;
    for(uint block=0;block<p.k;block+=64){
        for(uint i=tid;i<16*16;i+=256){uint row=i/16,col=(i%16)*4;uint inputRow=row<tile.count?(p.kind==2?tile.first+row:tokens[tile.first+row]):0;float4 value=row<tile.count?*((device const float4*)(x+inputRow*p.k+block+col)):float4(0);for(uint d=0;d<4;++d)xt[row*65+col+d]=bfloat(value[d]);}
        for(uint i=tid;i<64*16;i+=256){uint row=i/16,col=(i%16)*4;float4 value=firstColumn+row<p.n?weight4(w,s,firstColumn+row,block+col,p.k,0):float4(0);for(uint d=0;d<4;++d)wt[(col+d)*65+row]=bfloat(value[d]);}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        op.run(left,right,acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    auto destination=tensor(result,dextents<int,2>{64,16},array<int,2>{1,65});
    acc.store(destination);threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint i=tid;i<16*64;i+=256){uint row=i/64,col=i%64;if(row<tile.count&&firstColumn+col<p.n)y[(tile.first+row)*p.n+firstColumn+col]=result[row*65+col];}
}

// FP32 attention tiles reuse each Q/K/V block across multiple queries/channels.
// Softmax stays in FP32; scores retain the existing [query, head, key] layout.
kernel void batch_attention_scores_tiled(device const float*q [[buffer(0)]],device const float*k [[buffer(1)]],device float*scores [[buffer(2)]],constant P&p [[buffer(15)]],uint tid [[thread_index_in_threadgroup]],uint3 group [[threadgroup_position_in_grid]]) {
    uint head=group.y%16,firstRow=(group.y/16)*16,firstKey=group.x*32;
    threadgroup float qt[16*33],kt[32*33],result[16*33];
    constexpr auto desc=matmul2d_descriptor(16,32,32,false,false,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc,execution_simdgroups<4>> op;
    auto left=tensor(qt,dextents<int,2>{32,16},array<int,2>{1,33});
    auto right=tensor(kt,dextents<int,2>{32,32},array<int,2>{1,33});
    auto acc=op.get_destination_cooperative_tensor<decltype(left),decltype(right),float>();
    #pragma unroll
    for(uint i=0;i<acc.get_capacity();++i)acc[i]=0;
    for(uint block=0;block<256;block+=32){
        for(uint i=tid;i<16*32;i+=128){uint r=i/32,d=i%32;qt[r*33+d]=firstRow+r<p.m?q[((firstRow+r)*16+head)*256+block+d]:0;}
        for(uint i=tid;i<32*32;i+=128){uint t=i/32,d=i%32;kt[d*33+t]=firstKey+t<p.k?kv_load(k,((firstKey+t)*2+head/8)*256+block+d,p.flags):0;}
        threadgroup_barrier(mem_flags::mem_threadgroup);op.run(left,right,acc);threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    auto destination=tensor(result,dextents<int,2>{32,16},array<int,2>{1,33});acc.store(destination);threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint i=tid;i<16*32;i+=128){uint r=firstRow+i/32,t=firstKey+i%32;if(r<p.m&&t<p.k)scores[(r*16+head)*p.k+t]=t<=p.pos+r?result[(i/32)*33+i%32]*(1.0f/16.0f):-INFINITY;}
}

kernel void batch_attention_values_tiled(device const float*scores [[buffer(0)]],device const float*v [[buffer(1)]],device const float*gate [[buffer(2)]],device float*y [[buffer(3)]],constant P&p [[buffer(15)]],uint tid [[thread_index_in_threadgroup]],uint3 group [[threadgroup_position_in_grid]]) {
    uint head=group.y%16,firstRow=(group.y/16)*16,firstDim=group.x*32;
    threadgroup float pt[16*65],vt[64*33],result[16*33];
    constexpr auto desc=matmul2d_descriptor(16,32,64,false,false,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc,execution_simdgroups<4>> op;
    auto left=tensor(pt,dextents<int,2>{64,16},array<int,2>{1,65});
    auto right=tensor(vt,dextents<int,2>{32,64},array<int,2>{1,33});
    auto acc=op.get_destination_cooperative_tensor<decltype(left),decltype(right),float>();
    #pragma unroll
    for(uint i=0;i<acc.get_capacity();++i)acc[i]=0;
    uint end=min(p.k,p.pos+firstRow+16);
    for(uint block=0;block<end;block+=64){
        for(uint i=tid;i<16*64;i+=128){uint r=firstRow+i/64,t=block+i%64;pt[(i/64)*65+i%64]=r<p.m&&t<p.k&&t<=p.pos+r?scores[(r*16+head)*p.k+t]:0;}
        for(uint i=tid;i<64*32;i+=128){uint t=block+i/32,d=firstDim+i%32;vt[(i/32)*33+i%32]=t<p.k?kv_load(v,(t*2+head/8)*256+d,p.flags):0;}
        threadgroup_barrier(mem_flags::mem_threadgroup);op.run(left,right,acc);threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    auto destination=tensor(result,dextents<int,2>{32,16},array<int,2>{1,33});acc.store(destination);threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint i=tid;i<16*32;i+=128){uint r=firstRow+i/32,d=firstDim+i%32;if(r<p.m){uint at=(r*16+head)*256+d;y[at]=result[(i/32)*33+i%32]*sigmoid(gate[at]);}}
}

kernel void batch_attention_scores_tiled_fp16(device const float*q [[buffer(0)]],device const float*k [[buffer(1)]],device float*scores [[buffer(2)]],constant P&p [[buffer(15)]],uint tid [[thread_index_in_threadgroup]],uint3 group [[threadgroup_position_in_grid]]) {
    uint head=group.y%16,firstRow=(group.y/16)*16,firstKey=group.x*32;
    threadgroup half qt[16*33],kt[32*33];
    threadgroup float result[16*33];
    constexpr auto desc=matmul2d_descriptor(16,32,32,false,false,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc,execution_simdgroups<4>> op;
    auto left=tensor(qt,dextents<int,2>{32,16},array<int,2>{1,33});
    auto right=tensor(kt,dextents<int,2>{32,32},array<int,2>{1,33});
    auto acc=op.get_destination_cooperative_tensor<decltype(left),decltype(right),float>();
    #pragma unroll
    for(uint i=0;i<acc.get_capacity();++i)acc[i]=0;
    for(uint block=0;block<256;block+=32){
        for(uint i=tid;i<16*32;i+=128){uint r=i/32,d=i%32;qt[r*33+d]=firstRow+r<p.m?q[((firstRow+r)*16+head)*256+block+d]:0;}
        for(uint i=tid;i<32*32;i+=128){uint t=i/32,d=i%32;kt[d*33+t]=firstKey+t<p.k?kv_load(k,((firstKey+t)*2+head/8)*256+block+d,p.flags):0;}
        threadgroup_barrier(mem_flags::mem_threadgroup);op.run(left,right,acc);threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    auto destination=tensor(result,dextents<int,2>{32,16},array<int,2>{1,33});acc.store(destination);threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint i=tid;i<16*32;i+=128){uint r=firstRow+i/32,t=firstKey+i%32;if(r<p.m&&t<p.k)scores[(r*16+head)*p.k+t]=t<=p.pos+r?result[(i/32)*33+i%32]*(1.0f/16.0f):-INFINITY;}
}

kernel void batch_attention_values_tiled_fp16(device const float*scores [[buffer(0)]],device const float*v [[buffer(1)]],device const float*gate [[buffer(2)]],device float*y [[buffer(3)]],constant P&p [[buffer(15)]],uint tid [[thread_index_in_threadgroup]],uint3 group [[threadgroup_position_in_grid]]) {
    uint head=group.y%16,firstRow=(group.y/16)*16,firstDim=group.x*32;
    threadgroup half pt[16*65],vt[64*33];
    threadgroup float result[16*33];
    constexpr auto desc=matmul2d_descriptor(16,32,64,false,false,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc,execution_simdgroups<4>> op;
    auto left=tensor(pt,dextents<int,2>{64,16},array<int,2>{1,65});
    auto right=tensor(vt,dextents<int,2>{32,64},array<int,2>{1,33});
    auto acc=op.get_destination_cooperative_tensor<decltype(left),decltype(right),float>();
    #pragma unroll
    for(uint i=0;i<acc.get_capacity();++i)acc[i]=0;
    uint end=min(p.k,p.pos+firstRow+16);
    for(uint block=0;block<end;block+=64){
        for(uint i=tid;i<16*64;i+=128){uint r=firstRow+i/64,t=block+i%64;pt[(i/64)*65+i%64]=r<p.m&&t<p.k&&t<=p.pos+r?scores[(r*16+head)*p.k+t]:0;}
        for(uint i=tid;i<64*32;i+=128){uint t=block+i/32,d=firstDim+i%32;vt[(i/32)*33+i%32]=t<p.k?kv_load(v,(t*2+head/8)*256+d,p.flags):0;}
        threadgroup_barrier(mem_flags::mem_threadgroup);op.run(left,right,acc);threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    auto destination=tensor(result,dextents<int,2>{32,16},array<int,2>{1,33});acc.store(destination);threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint i=tid;i<16*32;i+=128){uint r=firstRow+i/32,d=firstDim+i%32;if(r<p.m){uint at=(r*16+head)*256+d;y[at]=result[(i/32)*33+i%32]*sigmoid(gate[at]);}}
}

// Online softmax: keep only a 16-query x 64-key score tile on-chip.
// Q/K/V/probability operands are FP16; maxima, sums, output accumulation are FP32.
// Keep four output tiles in FP32 cooperative registers (14,528 bytes shared).
kernel void batch_attention_fused_fp16(device const float*q [[buffer(0)]],device const float*k [[buffer(1)]],device const float*v [[buffer(2)]],device const float*gate [[buffer(3)]],device float*y [[buffer(4)]],constant P&p [[buffer(15)]],uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint sid [[simdgroup_index_in_threadgroup]],uint3 group [[threadgroup_position_in_grid]]) {
    uint firstRow=group.x*16,head=group.y;
    threadgroup half leftData[16*64],rightData[64*64];
    threadgroup float score[16*64],maxima[16],denoms[16],alpha[16];
    if(tid<16){maxima[tid]=-INFINITY;denoms[tid]=0;}
    constexpr auto desc=matmul2d_descriptor(16,64,64,false,false,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc,execution_simdgroups<4>> op;
    auto left=tensor(leftData,dextents<int,2>{64,16},array<int,2>{1,64});
    auto right=tensor(rightData,dextents<int,2>{64,64},array<int,2>{1,64});
    auto acc=op.get_destination_cooperative_tensor<decltype(left),decltype(right),float>();
    auto out0=op.get_destination_cooperative_tensor<decltype(left),decltype(right),float>();
    auto out1=op.get_destination_cooperative_tensor<decltype(left),decltype(right),float>();
    auto out2=op.get_destination_cooperative_tensor<decltype(left),decltype(right),float>();
    auto out3=op.get_destination_cooperative_tensor<decltype(left),decltype(right),float>();
    #pragma unroll
    for(uint i=0;i<out0.get_capacity();++i){out0[i]=0;out1[i]=0;out2[i]=0;out3[i]=0;}
    uint end=min(p.k,p.pos+min(p.m,firstRow+16));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint firstKey=0;firstKey<end;firstKey+=64){
        #pragma unroll
        for(uint i=0;i<acc.get_capacity();++i)acc[i]=0;
        for(uint block=0;block<256;block+=64){
            for(uint i=tid;i<16*64;i+=128){uint r=firstRow+i/64,d=block+i%64;leftData[i]=r<p.m?half(q[(r*16+head)*256+d]):half(0);}
            for(uint i=tid;i<64*64;i+=128){uint t=firstKey+i/64,d=block+i%64;rightData[(i%64)*64+i/64]=t<p.k?half(kv_load(k,(t*2+head/8)*256+d,p.flags)):half(0);}
            threadgroup_barrier(mem_flags::mem_threadgroup);op.run(left,right,acc);threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        auto scores=tensor(score,dextents<int,2>{64,16},array<int,2>{1,64});acc.store(scores);threadgroup_barrier(mem_flags::mem_threadgroup);
        // One SIMD group normalizes each query; four groups cover sixteen rows.
        for(uint r=sid;r<16;r+=4){
            uint row=firstRow+r;
            float a=(row<p.m&&firstKey+lane<=p.pos+row&&firstKey+lane<p.k)?score[r*64+lane]*(1.0f/16.0f):-INFINITY;
            float b=(row<p.m&&firstKey+lane+32<=p.pos+row&&firstKey+lane+32<p.k)?score[r*64+lane+32]*(1.0f/16.0f):-INFINITY;
            float nextMax=max(maxima[r],simd_max(max(a,b)));
            float decay=row<p.m?exp(maxima[r]-nextMax):1.0f;
            float pa=row<p.m?exp(a-nextMax):0.0f,pb=row<p.m?exp(b-nextMax):0.0f;
            float total=simd_sum(pa+pb);
            leftData[r*64+lane]=half(pa);leftData[r*64+lane+32]=half(pb);
            if(lane==0){alpha[r]=decay;denoms[r]=denoms[r]*decay+total;maxima[r]=nextMax;}
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma unroll
        for(uint i=0;i<out0.get_capacity();++i){auto ij=out0.get_multidimensional_index(i);if(ij[0]<64&&ij[1]<16){float scale=alpha[ij[1]];out0[i]*=scale;out1[i]*=scale;out2[i]*=scale;out3[i]*=scale;}}
        #pragma unroll
        for(uint firstDim=0;firstDim<256;firstDim+=64){
            for(uint i=tid;i<64*64;i+=128){uint t=firstKey+i/64,d=firstDim+i%64;rightData[i]=t<p.k?half(kv_load(v,(t*2+head/8)*256+d,p.flags)):half(0);}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if(firstDim==0)op.run(left,right,out0);
            else if(firstDim==64)op.run(left,right,out1);
            else if(firstDim==128)op.run(left,right,out2);
            else op.run(left,right,out3);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    #pragma unroll
    for(uint i=0;i<out0.get_capacity();++i){auto ij=out0.get_multidimensional_index(i);uint row=firstRow+ij[1],d=ij[0];if(ij[1]<16&&d<64&&row<p.m){uint at=(row*16+head)*256+d;float inv=1.0f/denoms[ij[1]];y[at]=out0[i]*inv*sigmoid(gate[at]);y[at+64]=out1[i]*inv*sigmoid(gate[at+64]);y[at+128]=out2[i]*inv*sigmoid(gate[at+128]);y[at+192]=out3[i]*inv*sigmoid(gate[at+192]);}}
}

// 128-key variant: fewer online-softmax steps and barriers, 28,864 shared bytes.
kernel void batch_attention_fused_fp16_bk128(device const float*q [[buffer(0)]],device const float*k [[buffer(1)]],device const float*v [[buffer(2)]],device const float*gate [[buffer(3)]],device float*y [[buffer(4)]],constant P&p [[buffer(15)]],uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint sid [[simdgroup_index_in_threadgroup]],uint3 group [[threadgroup_position_in_grid]]) {
    uint firstRow=group.x*16,head=group.y;
    threadgroup half leftData[16*128],rightData[128*64];
    threadgroup float score[16*128],maxima[16],denoms[16],alpha[16];
    if(tid<16){maxima[tid]=-INFINITY;denoms[tid]=0;}
    constexpr auto qdesc=matmul2d_descriptor(16,128,64,false,false,false,matmul2d_descriptor::mode::multiply_accumulate);
    constexpr auto vdesc=matmul2d_descriptor(16,64,128,false,false,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<qdesc,execution_simdgroups<4>> qop;
    matmul2d<vdesc,execution_simdgroups<4>> vop;
    auto qleft=tensor(leftData,dextents<int,2>{64,16},array<int,2>{1,64});
    auto qright=tensor(rightData,dextents<int,2>{128,64},array<int,2>{1,128});
    auto pleft=tensor(leftData,dextents<int,2>{128,16},array<int,2>{1,128});
    auto vright=tensor(rightData,dextents<int,2>{64,128},array<int,2>{1,64});
    auto acc=qop.get_destination_cooperative_tensor<decltype(qleft),decltype(qright),float>();
    auto out0=vop.get_destination_cooperative_tensor<decltype(pleft),decltype(vright),float>();
    auto out1=vop.get_destination_cooperative_tensor<decltype(pleft),decltype(vright),float>();
    auto out2=vop.get_destination_cooperative_tensor<decltype(pleft),decltype(vright),float>();
    auto out3=vop.get_destination_cooperative_tensor<decltype(pleft),decltype(vright),float>();
    #pragma unroll
    for(uint i=0;i<out0.get_capacity();++i){out0[i]=0;out1[i]=0;out2[i]=0;out3[i]=0;}
    uint end=min(p.k,p.pos+min(p.m,firstRow+16));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint firstKey=0;firstKey<end;firstKey+=128){
        #pragma unroll
        for(uint i=0;i<acc.get_capacity();++i)acc[i]=0;
        for(uint block=0;block<256;block+=64){
            for(uint i=tid;i<16*64;i+=128){uint r=firstRow+i/64,d=block+i%64;leftData[i]=r<p.m?half(q[(r*16+head)*256+d]):half(0);}
            for(uint i=tid;i<128*64;i+=128){uint t=firstKey+i/64,d=block+i%64;rightData[(i%64)*128+i/64]=t<p.k?half(kv_load(k,(t*2+head/8)*256+d,p.flags)):half(0);}
            threadgroup_barrier(mem_flags::mem_threadgroup);qop.run(qleft,qright,acc);threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        auto scores=tensor(score,dextents<int,2>{128,16},array<int,2>{1,128});acc.store(scores);threadgroup_barrier(mem_flags::mem_threadgroup);
        // One SIMD group normalizes each query; four groups cover sixteen rows.
        for(uint r=sid;r<16;r+=4){
            uint row=firstRow+r;
            float a=(row<p.m&&firstKey+lane<=p.pos+row&&firstKey+lane<p.k)?score[r*128+lane]*(1.0f/16.0f):-INFINITY;
            float b=(row<p.m&&firstKey+lane+32<=p.pos+row&&firstKey+lane+32<p.k)?score[r*128+lane+32]*(1.0f/16.0f):-INFINITY;
            float c=(row<p.m&&firstKey+lane+64<=p.pos+row&&firstKey+lane+64<p.k)?score[r*128+lane+64]*(1.0f/16.0f):-INFINITY;
            float d=(row<p.m&&firstKey+lane+96<=p.pos+row&&firstKey+lane+96<p.k)?score[r*128+lane+96]*(1.0f/16.0f):-INFINITY;
            float nextMax=max(maxima[r],simd_max(max(max(a,b),max(c,d))));
            float decay=row<p.m?exp(maxima[r]-nextMax):1.0f;
            float pa=row<p.m?exp(a-nextMax):0.0f,pb=row<p.m?exp(b-nextMax):0.0f;
            float pc=row<p.m?exp(c-nextMax):0.0f,pd=row<p.m?exp(d-nextMax):0.0f;
            float total=simd_sum(pa+pb+pc+pd);
            leftData[r*128+lane]=half(pa);leftData[r*128+lane+32]=half(pb);
            leftData[r*128+lane+64]=half(pc);leftData[r*128+lane+96]=half(pd);
            if(lane==0){alpha[r]=decay;denoms[r]=denoms[r]*decay+total;maxima[r]=nextMax;}
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma unroll
        for(uint i=0;i<out0.get_capacity();++i){auto ij=out0.get_multidimensional_index(i);if(ij[0]<64&&ij[1]<16){float scale=alpha[ij[1]];out0[i]*=scale;out1[i]*=scale;out2[i]*=scale;out3[i]*=scale;}}
        #pragma unroll
        for(uint firstDim=0;firstDim<256;firstDim+=64){
            for(uint i=tid;i<128*64;i+=128){uint t=firstKey+i/64,d=firstDim+i%64;rightData[i]=t<p.k?half(kv_load(v,(t*2+head/8)*256+d,p.flags)):half(0);}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if(firstDim==0)vop.run(pleft,vright,out0);
            else if(firstDim==64)vop.run(pleft,vright,out1);
            else if(firstDim==128)vop.run(pleft,vright,out2);
            else vop.run(pleft,vright,out3);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    #pragma unroll
    for(uint i=0;i<out0.get_capacity();++i){auto ij=out0.get_multidimensional_index(i);uint row=firstRow+ij[1],d=ij[0];if(ij[1]<16&&d<64&&row<p.m){uint at=(row*16+head)*256+d;float inv=1.0f/denoms[ij[1]];y[at]=out0[i]*inv*sigmoid(gate[at]);y[at+64]=out1[i]*inv*sigmoid(gate[at+64]);y[at+128]=out2[i]*inv*sigmoid(gate[at+128]);y[at+192]=out3[i]*inv*sigmoid(gate[at+192]);}}
}

kernel void matrix_product_fp16_tile(device const float*x [[buffer(0)]],device const uchar*w [[buffer(1)]],device const ushort*s [[buffer(2)]],device float*y [[buffer(3)]],constant P&p [[buffer(15)]],uint tid [[thread_index_in_threadgroup]],uint sid [[simdgroup_index_in_threadgroup]],uint3 group [[threadgroup_position_in_grid]]) {
    threadgroup half xt[32*65],wt[64*65];

    uint firstRow=group.y*32,firstColumn=group.x*64,rr=(sid/4)*16,cc=(sid%4)*16;
    // FP16 staged operands, FP32 accumulation and output; no relaxed precision.
    constexpr auto desc=matmul2d_descriptor(32,64,64,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc,execution_simdgroups<4>> op;
    auto left=tensor(xt,extents<int,64,32>{},array<int,2>{1,65});
    auto right=tensor(wt,extents<int,64,64>{},array<int,2>{1,65});
    auto acc=op.get_destination_cooperative_tensor<decltype(left),decltype(right),float>();
    #pragma unroll
    for(uint i=0;i<acc.get_capacity();++i) acc[i]=0.0f;
    for(uint block=0;block<p.k;block+=64){
        for(uint i=tid;i<32*16;i+=128){uint row=i/16,col=(i%16)*4;float4 value=firstRow+row<p.m?*((device const float4*)(x+(firstRow+row)*p.k+block+col)):float4(0);for(uint d=0;d<4;++d)xt[row*65+col+d]=value[d];}
        for(uint i=tid;i<64*16;i+=128){uint row=i/16,col=(i%16)*4;float4 value=firstColumn+row<p.n?weight4(w,s,firstColumn+row,block+col,p.k,p.kind):float4(0);for(uint d=0;d<4;++d)wt[row*65+col+d]=value[d];}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        op.run(left,right,acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    #pragma unroll
    for(uint i=0;i<acc.get_capacity();++i){auto coord=acc.get_multidimensional_index(i);uint col=coord[0],row=coord[1];
        if(row<32&&col<64&&firstRow+row<p.m&&firstColumn+col<p.n)y[(firstRow+row)*p.n+firstColumn+col]=acc[i];
    }
}

kernel void expert_matrix_fp16_tile_up(device const float*x [[buffer(0)]],device const uchar*raw [[buffer(1)]],device const Expert*experts [[buffer(2)]],device const ExpertTile*tiles [[buffer(3)]],device const uint*tileCount [[buffer(4)]],device const uint*tokens [[buffer(5)]],device float*y [[buffer(6)]],constant P&p [[buffer(15)]],uint tid [[thread_index_in_threadgroup]],uint sid [[simdgroup_index_in_threadgroup]],uint3 group [[threadgroup_position_in_grid]]) {
    if(group.y>=tileCount[0])return;ExpertTile tile=tiles[group.y];Expert descriptor=experts[tile.expert];uint firstColumn=group.x*64;
    ulong woff=p.kind==0?descriptor.gate:p.kind==1?descriptor.up:descriptor.down,soff=p.kind==0?descriptor.gate_scale:p.kind==1?descriptor.up_scale:descriptor.down_scale;
    device const uchar*w=raw+woff;device const ushort*s=(device const ushort*)(raw+soff);
    threadgroup half xt[16*65],wt[64*65];

    // FP16 staged operands, FP32 accumulation and output; no relaxed precision.
    constexpr auto desc=matmul2d_descriptor(16,64,64,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc,execution_simdgroups<4>> op;
    auto left=tensor(xt,dextents<int,2>{64,16},array<int,2>{1,65});
    auto right=tensor(wt,dextents<int,2>{64,64},array<int,2>{1,65});
    auto acc=op.get_destination_cooperative_tensor<decltype(left),decltype(right),float>();
    #pragma unroll
    for(uint i=0;i<acc.get_capacity();++i) acc[i]=0.0f;
    for(uint block=0;block<p.k;block+=64){
        for(uint i=tid;i<16*16;i+=128){uint row=i/16,col=(i%16)*4;uint inputRow=row<tile.count?(p.kind==2?tile.first+row:tokens[tile.first+row]):0;float4 value=row<tile.count?*((device const float4*)(x+inputRow*p.k+block+col)):float4(0);for(uint d=0;d<4;++d)xt[row*65+col+d]=value[d];}
        for(uint i=tid;i<64*16;i+=128){uint row=i/16,col=(i%16)*4;float4 value=firstColumn+row<p.n?weight4(w,s,firstColumn+row,block+col,p.k,0):float4(0);for(uint d=0;d<4;++d)wt[row*65+col+d]=value[d];}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        op.run(left,right,acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    #pragma unroll
    for(uint i=0;i<acc.get_capacity();++i){auto coord=acc.get_multidimensional_index(i);uint col=coord[0],row=coord[1];
        if(row<tile.count&&col<64&&firstColumn+col<p.n)y[(tile.first+row)*p.n+firstColumn+col]=acc[i];
    }
}

kernel void expert_matrix_fp16_tile_down(device const float*x [[buffer(0)]],device const uchar*raw [[buffer(1)]],device const Expert*experts [[buffer(2)]],device const ExpertTile*tiles [[buffer(3)]],device const uint*tileCount [[buffer(4)]],device const uint*tokens [[buffer(5)]],device float*y [[buffer(6)]],constant P&p [[buffer(15)]],uint tid [[thread_index_in_threadgroup]],uint sid [[simdgroup_index_in_threadgroup]],uint3 group [[threadgroup_position_in_grid]]) {
    if(group.y>=tileCount[0])return;ExpertTile tile=tiles[group.y];Expert descriptor=experts[tile.expert];uint firstColumn=group.x*64;
    ulong woff=p.kind==0?descriptor.gate:p.kind==1?descriptor.up:descriptor.down,soff=p.kind==0?descriptor.gate_scale:p.kind==1?descriptor.up_scale:descriptor.down_scale;
    device const uchar*w=raw+woff;device const ushort*s=(device const ushort*)(raw+soff);
    threadgroup half xt[16*65],wt[64*65];

    // FP16 staged operands, FP32 accumulation and output; no relaxed precision.
    constexpr auto desc=matmul2d_descriptor(16,64,64,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc,execution_simdgroups<8>> op;
    auto left=tensor(xt,dextents<int,2>{64,16},array<int,2>{1,65});
    auto right=tensor(wt,dextents<int,2>{64,64},array<int,2>{1,65});
    auto acc=op.get_destination_cooperative_tensor<decltype(left),decltype(right),float>();
    #pragma unroll
    for(uint i=0;i<acc.get_capacity();++i) acc[i]=0.0f;
    for(uint block=0;block<p.k;block+=64){
        for(uint i=tid;i<16*16;i+=256){uint row=i/16,col=(i%16)*4;uint inputRow=row<tile.count?(p.kind==2?tile.first+row:tokens[tile.first+row]):0;float4 value=row<tile.count?*((device const float4*)(x+inputRow*p.k+block+col)):float4(0);for(uint d=0;d<4;++d)xt[row*65+col+d]=value[d];}
        for(uint i=tid;i<64*16;i+=256){uint row=i/16,col=(i%16)*4;float4 value=firstColumn+row<p.n?weight4(w,s,firstColumn+row,block+col,p.k,0):float4(0);for(uint d=0;d<4;++d)wt[row*65+col+d]=value[d];}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        op.run(left,right,acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    #pragma unroll
    for(uint i=0;i<acc.get_capacity();++i){auto coord=acc.get_multidimensional_index(i);uint col=coord[0],row=coord[1];
        if(row<tile.count&&col<64&&firstColumn+col<p.n)y[(tile.first+row)*p.n+firstColumn+col]=acc[i];
    }
}

kernel void expert_matrix_fp16_tile32_up(device const float*x [[buffer(0)]],device const uchar*raw [[buffer(1)]],device const Expert*experts [[buffer(2)]],device const ExpertTile*tiles [[buffer(3)]],device const uint*tileCount [[buffer(4)]],device const uint*tokens [[buffer(5)]],device float*y [[buffer(6)]],constant P&p [[buffer(15)]],uint tid [[thread_index_in_threadgroup]],uint sid [[simdgroup_index_in_threadgroup]],uint3 group [[threadgroup_position_in_grid]]) {
    if(group.y>=tileCount[0])return;ExpertTile tile=tiles[group.y];Expert descriptor=experts[tile.expert];uint firstColumn=group.x*64;
    ulong woff=p.kind==0?descriptor.gate:p.kind==1?descriptor.up:descriptor.down,soff=p.kind==0?descriptor.gate_scale:p.kind==1?descriptor.up_scale:descriptor.down_scale;
    device const uchar*w=raw+woff;device const ushort*s=(device const ushort*)(raw+soff);
    threadgroup half xt[32*65],wt[64*65];

    // FP16 staged operands, FP32 accumulation and output; no relaxed precision.
    constexpr auto desc=matmul2d_descriptor(32,64,64,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc,execution_simdgroups<4>> op;
    auto left=tensor(xt,extents<int,64,32>{},array<int,2>{1,65});
    auto right=tensor(wt,dextents<int,2>{64,64},array<int,2>{1,65});
    auto acc=op.get_destination_cooperative_tensor<decltype(left),decltype(right),float>();
    #pragma unroll
    for(uint i=0;i<acc.get_capacity();++i) acc[i]=0.0f;
    for(uint block=0;block<p.k;block+=64){
        for(uint i=tid;i<32*16;i+=128){uint row=i/16,col=(i%16)*4;uint inputRow=row<tile.count?(p.kind==2?tile.first+row:tokens[tile.first+row]):0;float4 value=row<tile.count?*((device const float4*)(x+inputRow*p.k+block+col)):float4(0);for(uint d=0;d<4;++d)xt[row*65+col+d]=value[d];}
        for(uint i=tid;i<64*16;i+=128){uint row=i/16,col=(i%16)*4;float4 value=firstColumn+row<p.n?weight4(w,s,firstColumn+row,block+col,p.k,0):float4(0);for(uint d=0;d<4;++d)wt[row*65+col+d]=value[d];}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        op.run(left,right,acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    #pragma unroll
    for(uint i=0;i<acc.get_capacity();++i){auto coord=acc.get_multidimensional_index(i);uint col=coord[0],row=coord[1];
        if(row<tile.count&&col<64&&firstColumn+col<p.n)y[(tile.first+row)*p.n+firstColumn+col]=acc[i];
    }
}

kernel void expert_matrix_fp16_tile32_down(device const float*x [[buffer(0)]],device const uchar*raw [[buffer(1)]],device const Expert*experts [[buffer(2)]],device const ExpertTile*tiles [[buffer(3)]],device const uint*tileCount [[buffer(4)]],device const uint*tokens [[buffer(5)]],device float*y [[buffer(6)]],constant P&p [[buffer(15)]],uint tid [[thread_index_in_threadgroup]],uint sid [[simdgroup_index_in_threadgroup]],uint3 group [[threadgroup_position_in_grid]]) {
    if(group.y>=tileCount[0])return;ExpertTile tile=tiles[group.y];Expert descriptor=experts[tile.expert];uint firstColumn=group.x*64;
    ulong woff=p.kind==0?descriptor.gate:p.kind==1?descriptor.up:descriptor.down,soff=p.kind==0?descriptor.gate_scale:p.kind==1?descriptor.up_scale:descriptor.down_scale;
    device const uchar*w=raw+woff;device const ushort*s=(device const ushort*)(raw+soff);
    threadgroup half xt[32*65],wt[64*65];

    // FP16 staged operands, FP32 accumulation and output; no relaxed precision.
    constexpr auto desc=matmul2d_descriptor(32,64,64,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc,execution_simdgroups<8>> op;
    auto left=tensor(xt,extents<int,64,32>{},array<int,2>{1,65});
    auto right=tensor(wt,dextents<int,2>{64,64},array<int,2>{1,65});
    auto acc=op.get_destination_cooperative_tensor<decltype(left),decltype(right),float>();
    #pragma unroll
    for(uint i=0;i<acc.get_capacity();++i) acc[i]=0.0f;
    for(uint block=0;block<p.k;block+=64){
        for(uint i=tid;i<32*16;i+=256){uint row=i/16,col=(i%16)*4;uint inputRow=row<tile.count?(p.kind==2?tile.first+row:tokens[tile.first+row]):0;float4 value=row<tile.count?*((device const float4*)(x+inputRow*p.k+block+col)):float4(0);for(uint d=0;d<4;++d)xt[row*65+col+d]=value[d];}
        for(uint i=tid;i<64*16;i+=256){uint row=i/16,col=(i%16)*4;float4 value=firstColumn+row<p.n?weight4(w,s,firstColumn+row,block+col,p.k,0):float4(0);for(uint d=0;d<4;++d)wt[row*65+col+d]=value[d];}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        op.run(left,right,acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    #pragma unroll
    for(uint i=0;i<acc.get_capacity();++i){auto coord=acc.get_multidimensional_index(i);uint col=coord[0],row=coord[1];
        if(row<tile.count&&col<64&&firstColumn+col<p.n)y[(tile.first+row)*p.n+firstColumn+col]=acc[i];
    }
}

kernel void expert_matrix_fp16_tile64_up(device const float*x [[buffer(0)]],device const uchar*raw [[buffer(1)]],device const Expert*experts [[buffer(2)]],device const ExpertTile*tiles [[buffer(3)]],device const uint*tileCount [[buffer(4)]],device const uint*tokens [[buffer(5)]],device float*y [[buffer(6)]],constant P&p [[buffer(15)]],uint tid [[thread_index_in_threadgroup]],uint sid [[simdgroup_index_in_threadgroup]],uint3 group [[threadgroup_position_in_grid]]) {
    if(group.y>=tileCount[0])return;ExpertTile tile=tiles[group.y];Expert descriptor=experts[tile.expert];uint firstColumn=group.x*64;
    ulong woff=p.kind==0?descriptor.gate:p.kind==1?descriptor.up:descriptor.down,soff=p.kind==0?descriptor.gate_scale:p.kind==1?descriptor.up_scale:descriptor.down_scale;
    device const uchar*w=raw+woff;device const ushort*s=(device const ushort*)(raw+soff);
    threadgroup half xt[64*65],wt[64*65];

    // FP16 staged operands, FP32 accumulation and output; no relaxed precision.
    constexpr auto desc=matmul2d_descriptor(64,64,64,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc,execution_simdgroups<8>> op;
    auto left=tensor(xt,extents<int,64,64>{},array<int,2>{1,65});
    auto right=tensor(wt,dextents<int,2>{64,64},array<int,2>{1,65});
    auto acc=op.get_destination_cooperative_tensor<decltype(left),decltype(right),float>();
    #pragma unroll
    for(uint i=0;i<acc.get_capacity();++i) acc[i]=0.0f;
    for(uint block=0;block<p.k;block+=64){
        for(uint i=tid;i<64*16;i+=256){uint row=i/16,col=(i%16)*4;uint inputRow=row<tile.count?(p.kind==2?tile.first+row:tokens[tile.first+row]):0;float4 value=row<tile.count?*((device const float4*)(x+inputRow*p.k+block+col)):float4(0);for(uint d=0;d<4;++d)xt[row*65+col+d]=value[d];}
        for(uint i=tid;i<64*16;i+=256){uint row=i/16,col=(i%16)*4;float4 value=firstColumn+row<p.n?weight4(w,s,firstColumn+row,block+col,p.k,0):float4(0);for(uint d=0;d<4;++d)wt[row*65+col+d]=value[d];}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        op.run(left,right,acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    #pragma unroll
    for(uint i=0;i<acc.get_capacity();++i){auto coord=acc.get_multidimensional_index(i);uint col=coord[0],row=coord[1];
        if(row<tile.count&&col<64&&firstColumn+col<p.n)y[(tile.first+row)*p.n+firstColumn+col]=acc[i];
    }
}

kernel void expert_matrix_fp16_tile64_down(device const float*x [[buffer(0)]],device const uchar*raw [[buffer(1)]],device const Expert*experts [[buffer(2)]],device const ExpertTile*tiles [[buffer(3)]],device const uint*tileCount [[buffer(4)]],device const uint*tokens [[buffer(5)]],device float*y [[buffer(6)]],constant P&p [[buffer(15)]],uint tid [[thread_index_in_threadgroup]],uint sid [[simdgroup_index_in_threadgroup]],uint3 group [[threadgroup_position_in_grid]]) {
    if(group.y>=tileCount[0])return;ExpertTile tile=tiles[group.y];Expert descriptor=experts[tile.expert];uint firstColumn=group.x*64;
    ulong woff=p.kind==0?descriptor.gate:p.kind==1?descriptor.up:descriptor.down,soff=p.kind==0?descriptor.gate_scale:p.kind==1?descriptor.up_scale:descriptor.down_scale;
    device const uchar*w=raw+woff;device const ushort*s=(device const ushort*)(raw+soff);
    threadgroup half xt[64*65],wt[64*65];

    // FP16 staged operands, FP32 accumulation and output; no relaxed precision.
    constexpr auto desc=matmul2d_descriptor(64,64,64,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc,execution_simdgroups<8>> op;
    auto left=tensor(xt,extents<int,64,64>{},array<int,2>{1,65});
    auto right=tensor(wt,dextents<int,2>{64,64},array<int,2>{1,65});
    auto acc=op.get_destination_cooperative_tensor<decltype(left),decltype(right),float>();
    #pragma unroll
    for(uint i=0;i<acc.get_capacity();++i) acc[i]=0.0f;
    for(uint block=0;block<p.k;block+=64){
        for(uint i=tid;i<64*16;i+=256){uint row=i/16,col=(i%16)*4;uint inputRow=row<tile.count?(p.kind==2?tile.first+row:tokens[tile.first+row]):0;float4 value=row<tile.count?*((device const float4*)(x+inputRow*p.k+block+col)):float4(0);for(uint d=0;d<4;++d)xt[row*65+col+d]=value[d];}
        for(uint i=tid;i<64*16;i+=256){uint row=i/16,col=(i%16)*4;float4 value=firstColumn+row<p.n?weight4(w,s,firstColumn+row,block+col,p.k,0):float4(0);for(uint d=0;d<4;++d)wt[row*65+col+d]=value[d];}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        op.run(left,right,acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    #pragma unroll
    for(uint i=0;i<acc.get_capacity();++i){auto coord=acc.get_multidimensional_index(i);uint col=coord[0],row=coord[1];
        if(row<tile.count&&col<64&&firstColumn+col<p.n)y[(tile.first+row)*p.n+firstColumn+col]=acc[i];
    }
}

kernel void batch_attention_scores_tiled_fp16_opt64(device const float*q [[buffer(0)]],device const float*k [[buffer(1)]],device float*scores [[buffer(2)]],constant P&p [[buffer(15)]],uint tid [[thread_index_in_threadgroup]],uint3 group [[threadgroup_position_in_grid]]) {
    uint head=group.y%16,firstRow=(group.y/16)*16,firstKey=group.x*32;
    threadgroup half qt[16*65],kt[32*65];

    constexpr auto desc=matmul2d_descriptor(16,32,64,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc,execution_simdgroups<4>> op;
    auto left=tensor(qt,extents<int,64,16>{},array<int,2>{1,65});
    auto right=tensor(kt,extents<int,64,32>{},array<int,2>{1,65});
    auto acc=op.get_destination_cooperative_tensor<decltype(left),decltype(right),float>();
    #pragma unroll
    for(uint i=0;i<acc.get_capacity();++i)acc[i]=0;
    for(uint block=0;block<256;block+=64){
        for(uint i=tid;i<16*64;i+=128){uint r=i/64,d=i%64;qt[r*65+d]=firstRow+r<p.m?q[((firstRow+r)*16+head)*256+block+d]:0;}
        for(uint i=tid;i<32*64;i+=128){uint t=i/64,d=i%64;kt[t*65+d]=firstKey+t<p.k?kv_load(k,((firstKey+t)*2+head/8)*256+block+d,p.flags):0;}
        threadgroup_barrier(mem_flags::mem_threadgroup);op.run(left,right,acc);threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    #pragma unroll
    for(uint i=0;i<acc.get_capacity();++i){auto c=acc.get_multidimensional_index(i);uint r=firstRow+c[1],t=firstKey+c[0];if(c[1]<16&&c[0]<32&&r<p.m&&t<p.k)scores[(r*16+head)*p.k+t]=t<=p.pos+r?acc[i]*(1.0f/16.0f):-INFINITY;}
}

kernel void batch_attention_values_tiled_fp16_opt64(device const float*scores [[buffer(0)]],device const float*v [[buffer(1)]],device const float*gate [[buffer(2)]],device float*y [[buffer(3)]],constant P&p [[buffer(15)]],uint tid [[thread_index_in_threadgroup]],uint3 group [[threadgroup_position_in_grid]]) {
    uint head=group.y%16,firstRow=(group.y/16)*16,firstDim=group.x*32;
    threadgroup half pt[16*65],vt[64*33];

    constexpr auto desc=matmul2d_descriptor(16,32,64,false,false,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc,execution_simdgroups<4>> op;
    auto left=tensor(pt,extents<int,64,16>{},array<int,2>{1,65});
    auto right=tensor(vt,extents<int,32,64>{},array<int,2>{1,33});
    auto acc=op.get_destination_cooperative_tensor<decltype(left),decltype(right),float>();
    #pragma unroll
    for(uint i=0;i<acc.get_capacity();++i)acc[i]=0;
    uint end=min(p.k,p.pos+firstRow+16);
    for(uint block=0;block<end;block+=64){
        for(uint i=tid;i<16*64;i+=128){uint r=firstRow+i/64,t=block+i%64;pt[(i/64)*65+i%64]=r<p.m&&t<p.k&&t<=p.pos+r?scores[(r*16+head)*p.k+t]:0;}
        for(uint i=tid;i<64*32;i+=128){uint t=block+i/32,d=firstDim+i%32;vt[(i/32)*33+i%32]=t<p.k?kv_load(v,(t*2+head/8)*256+d,p.flags):0;}
        threadgroup_barrier(mem_flags::mem_threadgroup);op.run(left,right,acc);threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    #pragma unroll
    for(uint i=0;i<acc.get_capacity();++i){auto c=acc.get_multidimensional_index(i);uint r=firstRow+c[1],d=firstDim+c[0];if(c[1]<16&&c[0]<32&&r<p.m){uint at=(r*16+head)*256+d;y[at]=acc[i]*sigmoid(gate[at]);}}
}

// Decode each projection once, rather than independently inside every output tile.
kernel void projection_unpack_half(device const uchar*w [[buffer(0)]],device const ushort*s [[buffer(1)]],device half*y [[buffer(2)]],constant P&p [[buffer(15)]],uint i [[thread_position_in_grid]]) {
    uint at=i*4;if(at>=p.n*p.k)return;
    *((device half4*)(y+at))=half4(weight4(w,s,at/p.k,at%p.k,p.k,p.kind));
}
kernel void projection_input_half(device const float*x [[buffer(0)]],device half*y [[buffer(1)]],constant P&p [[buffer(15)]],uint i [[thread_position_in_grid]]) {
    uint at=i*4;if(at>=p.m*p.k)return;
    *((device half4*)(y+at))=half4(*((device const float4*)(x+at)));
}
kernel void matrix_product_predecoded(device half*x [[buffer(0)]],device half*w [[buffer(1)]],device float*y [[buffer(2)]],constant P&p [[buffer(15)]],uint3 group [[threadgroup_position_in_grid]]) {
    constexpr auto desc=matmul2d_descriptor(64,64,dynamic_extent,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc,execution_simdgroups<4>> op;
    auto left=tensor(x,dextents<int,2>{int(p.k),int(p.m)},array<int,2>{1,int(p.k)});
    auto right=tensor(w,dextents<int,2>{int(p.k),int(p.n)},array<int,2>{1,int(p.k)});
    auto output=tensor(y,dextents<int,2>{int(p.n),int(p.m)},array<int,2>{1,int(p.n)});
    auto a=left.slice(0,int(group.y)*64);auto b=right.slice(0,int(group.x)*64);auto c=output.slice(int(group.x)*64,int(group.y)*64);
    auto acc=op.get_destination_cooperative_tensor<decltype(a),decltype(b),float>();
    #pragma unroll
    for(uint i=0;i<acc.get_capacity();++i)acc[i]=0;
    op.run(a,b,acc);acc.store(c);
}

kernel void expert_unpack_half(device const uchar*raw [[buffer(0)]],device const Expert*experts [[buffer(1)]],device half*y [[buffer(2)]],constant P&p [[buffer(15)]],uint i [[thread_position_in_grid]]) {
    uint at=i*4,weightSize=p.n*p.k;if(at>=p.a*weightSize)return;
    Expert d=experts[at/weightSize];uint local=at%weightSize;
    ulong woff=p.kind==0?d.gate:p.kind==1?d.up:d.down,soff=p.kind==0?d.gate_scale:p.kind==1?d.up_scale:d.down_scale;
    *((device half4*)(y+at))=half4(weight4(raw+woff,(device const ushort*)(raw+soff),local/p.k,local%p.k,p.k,0));
}
kernel void expert_input_half(device const float*x [[buffer(0)]],device const uint*tokens [[buffer(1)]],device half*y [[buffer(2)]],constant P&p [[buffer(15)]],uint i [[thread_position_in_grid]]) {
    uint at=i*4;if(at>=p.m*p.k)return;uint row=at/p.k,col=at%p.k,inputRow=p.kind==2?row:tokens[row];
    *((device half4*)(y+at))=half4(*((device const float4*)(x+inputRow*p.k+col)));
}
kernel void expert_matrix_predecoded(device half*x [[buffer(0)]],device half*w [[buffer(1)]],device const ExpertTile*tiles [[buffer(2)]],device const uint*tileCount [[buffer(3)]],device float*y [[buffer(4)]],constant P&p [[buffer(15)]],uint3 group [[threadgroup_position_in_grid]]) {
    if(group.y>=tileCount[0])return;ExpertTile tile=tiles[group.y];
    constexpr auto desc=matmul2d_descriptor(64,64,dynamic_extent,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc,execution_simdgroups<4>> op;
    auto left=tensor(x+tile.first*p.k,dextents<int,2>{int(p.k),int(tile.count)},array<int,2>{1,int(p.k)});
    auto right=tensor(w+tile.expert*p.n*p.k,dextents<int,2>{int(p.k),int(p.n)},array<int,2>{1,int(p.k)});
    auto output=tensor(y+tile.first*p.n,dextents<int,2>{int(p.n),int(tile.count)},array<int,2>{1,int(p.n)});
    auto b=right.slice(0,int(group.x)*64);auto c=output.slice(int(group.x)*64,0);
    auto acc=op.get_destination_cooperative_tensor<decltype(left),decltype(b),float>();
    #pragma unroll
    for(uint i=0;i<acc.get_capacity();++i)acc[i]=0;
    op.run(left,b,acc);acc.store(c);
}

kernel void attention_input_half(device const float*x [[buffer(0)]],device half*y [[buffer(1)]],constant P&p [[buffer(15)]],uint i [[thread_position_in_grid]]) {
    if(i*4<p.n)*((device half4*)(y+i*4))=half4(*((device const float4*)(x+i*4)));
}
kernel void attention_scores_device(device half*q [[buffer(0)]],device half*k [[buffer(1)]],device float*scores [[buffer(2)]],constant P&p [[buffer(15)]],uint3 group [[threadgroup_position_in_grid]]) {
    uint head=group.y%16,firstRow=(group.y/16)*16,firstKey=group.x*64;
    constexpr auto desc=matmul2d_descriptor(16,64,256,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc,execution_simdgroups<4>> op;
    auto qfull=tensor(q+head*256,dextents<int,2>{256,int(p.m)},array<int,2>{1,4096});
    auto kfull=tensor(k+(head/8)*256,dextents<int,2>{256,int(p.k)},array<int,2>{1,512});
    auto left=qfull.slice(0,int(firstRow));auto right=kfull.slice(0,int(firstKey));
    auto acc=op.get_destination_cooperative_tensor<decltype(left),decltype(right),float>();
    #pragma unroll
    for(uint i=0;i<acc.get_capacity();++i)acc[i]=0;
    op.run(left,right,acc);
    #pragma unroll
    for(uint i=0;i<acc.get_capacity();++i){auto c=acc.get_multidimensional_index(i);uint r=firstRow+c[1],t=firstKey+c[0];if(c[1]<16&&c[0]<64&&r<p.m&&t<p.k)scores[(r*16+head)*p.k+t]=t<=p.pos+r?acc[i]*(1.0f/16.0f):-INFINITY;}
}
kernel void attention_values_device(device half*scores [[buffer(0)]],device half*v [[buffer(1)]],device const float*gate [[buffer(2)]],device float*y [[buffer(3)]],constant P&p [[buffer(15)]],uint3 group [[threadgroup_position_in_grid]]) {
    uint head=group.y%16,firstRow=(group.y/16)*16,firstDim=group.x*64;
    constexpr auto desc=matmul2d_descriptor(16,64,dynamic_extent,false,false,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc,execution_simdgroups<4>> op;
    auto pfull=tensor(scores+head*p.k,dextents<int,2>{int(p.k),int(p.m)},array<int,2>{1,int(16*p.k)});
    auto vfull=tensor(v+(head/8)*256,dextents<int,2>{256,int(p.k)},array<int,2>{1,512});
    auto left=pfull.slice(0,int(firstRow));auto right=vfull.slice(int(firstDim),0);
    auto acc=op.get_destination_cooperative_tensor<decltype(left),decltype(right),float>();
    #pragma unroll
    for(uint i=0;i<acc.get_capacity();++i)acc[i]=0;
    op.run(left,right,acc);
    #pragma unroll
    for(uint i=0;i<acc.get_capacity();++i){auto c=acc.get_multidimensional_index(i);uint r=firstRow+c[1],d=firstDim+c[0];if(c[1]<16&&c[0]<64&&r<p.m){uint at=(r*16+head)*256+d;y[at]=acc[i]*sigmoid(gate[at]);}}
}

// Gathered FP16 input is shared by gate/up and read directly by TensorOps.
// Weights remain packed FP8 and only one 64x64 block is unpacked per group.
template<int Rows>
inline void expert_gathered_product(device half*x,device const uchar*raw,device const Expert*experts,device const ExpertTile*tiles,device const uint*tileCount,device float*y,constant P&p,uint tid,uint3 group,threadgroup half*wt) {
    if(group.y>=tileCount[0])return;ExpertTile tile=tiles[group.y];Expert descriptor=experts[tile.expert];uint firstColumn=group.x*64;
    ulong woff=p.kind==0?descriptor.gate:p.kind==1?descriptor.up:descriptor.down,soff=p.kind==0?descriptor.gate_scale:p.kind==1?descriptor.up_scale:descriptor.down_scale;
    device const uchar*w=raw+woff;device const ushort*s=(device const ushort*)(raw+soff);
    constexpr auto desc=matmul2d_descriptor(Rows,64,64,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc,execution_simdgroups<4>> op;
    auto full=tensor(x+tile.first*p.k,dextents<int,2>{int(p.k),int(tile.count)},array<int,2>{1,int(p.k)});
    auto left=full.slice<64,dynamic_extent>(0,0);
    auto right=tensor(wt,extents<int,64,64>{},array<int,2>{1,65});
    auto acc=op.template get_destination_cooperative_tensor<decltype(left),decltype(right),float>();
    for(uint i=0;i<acc.get_capacity();++i)acc[i]=0;
    for(uint block=0;block<p.k;block+=64){
        for(uint i=tid;i<64*16;i+=128){uint row=i/16,col=(i%16)*4;float4 value=firstColumn+row<p.n?weight4(w,s,firstColumn+row,block+col,p.k,0):float4(0);for(uint d=0;d<4;++d)wt[row*65+col+d]=value[d];}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto left=full.slice<64,dynamic_extent>(int(block),0);op.run(left,right,acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for(uint i=0;i<acc.get_capacity();++i){auto c=acc.get_multidimensional_index(i);if(c[1]<tile.count&&c[0]<64&&firstColumn+c[0]<p.n)y[(tile.first+c[1])*p.n+firstColumn+c[0]]=acc[i];}
}
kernel void expert_matrix_gathered(device half*x [[buffer(0)]],device const uchar*raw [[buffer(1)]],device const Expert*experts [[buffer(2)]],device const ExpertTile*tiles [[buffer(3)]],device const uint*tileCount [[buffer(4)]],device float*y [[buffer(5)]],constant P&p [[buffer(15)]],uint tid [[thread_index_in_threadgroup]],uint3 group [[threadgroup_position_in_grid]]) {
    threadgroup half wt[64*65];
    if(group.y>=tileCount[0])return;uint rows=tiles[group.y].count;
    if((p.flags&1)&&rows<=16)expert_gathered_product<16>(x,raw,experts,tiles,tileCount,y,p,tid,group,wt);
    else if((p.flags&1)&&rows<=32)expert_gathered_product<32>(x,raw,experts,tiles,tileCount,y,p,tid,group,wt);
    else expert_gathered_product<64>(x,raw,experts,tiles,tileCount,y,p,tid,group,wt);
}

kernel void attention_scores_device_causal(device half*q [[buffer(0)]],device half*k [[buffer(1)]],device float*scores [[buffer(2)]],constant P&p [[buffer(15)]],uint3 group [[threadgroup_position_in_grid]]) {
    uint head=group.y%16,firstRow=(group.y/16)*16,firstKey=group.x*64;
    if(firstKey>=p.pos+min(p.m,firstRow+16))return;
    constexpr auto desc=matmul2d_descriptor(16,64,256,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc,execution_simdgroups<4>> op;
    auto qfull=tensor(q+head*256,dextents<int,2>{256,int(p.m)},array<int,2>{1,4096});
    auto kfull=tensor(k+(head/8)*256,dextents<int,2>{256,int(p.k)},array<int,2>{1,512});
    auto left=qfull.slice(0,int(firstRow));auto right=kfull.slice(0,int(firstKey));
    auto acc=op.get_destination_cooperative_tensor<decltype(left),decltype(right),float>();
    #pragma unroll
    for(uint i=0;i<acc.get_capacity();++i)acc[i]=0;
    op.run(left,right,acc);
    #pragma unroll
    for(uint i=0;i<acc.get_capacity();++i){auto c=acc.get_multidimensional_index(i);uint r=firstRow+c[1],t=firstKey+c[0];if(c[1]<16&&c[0]<64&&r<p.m&&t<p.k)scores[(r*16+head)*p.k+t]=t<=p.pos+r?acc[i]*(1.0f/16.0f):-INFINITY;}
}

kernel void attention_values_device_causal(device half*scores [[buffer(0)]],device half*v [[buffer(1)]],device const float*gate [[buffer(2)]],device float*y [[buffer(3)]],constant P&p [[buffer(15)]],uint3 group [[threadgroup_position_in_grid]]) {
    uint head=group.y%16,firstRow=(group.y/16)*16,firstDim=group.x*64;
    uint used=min(p.k,p.pos+min(p.m,firstRow+16));
    constexpr auto desc=matmul2d_descriptor(16,64,dynamic_extent,false,false,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc,execution_simdgroups<4>> op;
    auto pfull=tensor(scores+head*p.k,dextents<int,2>{int(used),int(p.m)},array<int,2>{1,int(16*p.k)});
    auto vfull=tensor(v+(head/8)*256,dextents<int,2>{256,int(used)},array<int,2>{1,512});
    auto left=pfull.slice(0,int(firstRow));auto right=vfull.slice(int(firstDim),0);
    auto acc=op.get_destination_cooperative_tensor<decltype(left),decltype(right),float>();
    #pragma unroll
    for(uint i=0;i<acc.get_capacity();++i)acc[i]=0;
    op.run(left,right,acc);
    #pragma unroll
    for(uint i=0;i<acc.get_capacity();++i){auto c=acc.get_multidimensional_index(i);uint r=firstRow+c[1],d=firstDim+c[0];if(c[1]<16&&c[0]<64&&r<p.m){uint at=(r*16+head)*256+d;y[at]=acc[i]*sigmoid(gate[at]);}}
}


// Causal row limits also bound the AV input. Write half probabilities directly,
// with zero padding only through the final row of the current 16-query tile.
kernel void attention_softmax_half_causal(device float*x [[buffer(0)]],device half*y [[buffer(1)]],constant P&p [[buffer(15)]],uint tid [[thread_index_in_threadgroup]],uint head [[threadgroup_position_in_grid]]) {
    uint row=head/16,valid=min(p.k,p.pos+row+1),end=min(p.k,p.pos+min(p.m,(row/16+1)*16));
    threadgroup float partial[8];threadgroup float maximum,denominator;float top=-INFINITY;
    for(uint i=tid;i<valid;i+=256)top=max(top,x[head*p.k+i]);
    top=simd_max(top);if((tid&31)==0)partial[tid/32]=top;threadgroup_barrier(mem_flags::mem_threadgroup);
    top=simd_max(tid<8?partial[tid]:-INFINITY);if(tid==0)maximum=top;threadgroup_barrier(mem_flags::mem_threadgroup);
    float sum=0;for(uint i=tid;i<valid;i+=256){float value=exp(x[head*p.k+i]-maximum);x[head*p.k+i]=value;sum+=value;}
    sum=simd_sum(sum);if((tid&31)==0)partial[tid/32]=sum;threadgroup_barrier(mem_flags::mem_threadgroup);
    sum=simd_sum(tid<8?partial[tid]:0);if(tid==0)denominator=sum;threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint i=tid;i<end;i+=256)y[head*p.k+i]=i<valid?half(x[head*p.k+i]/denominator):half(0);
}

// Online attention: grouped queries are staged once, while QK, softmax and AV
// remain inside a threadgroup. G adjacent query heads share one KV head.
kernel void attention_query_grouped_half(device const float*x [[buffer(0)]],device half*y [[buffer(1)]],constant P&p [[buffer(15)]],uint i [[thread_position_in_grid]]) {
    uint at=i*4;if(at>=p.m*4096)return;
    uint d=at%256,logical=at/256,block=logical/(p.m*p.a),local=logical%(p.m*p.a);
    uint row=local/p.a,head=block*p.a+local%p.a;
    *((device half4*)(y+at))=half4(*((device const float4*)(x+(row*16+head)*256+d)));
}

// Reuse each FP32 score row for FP16 probabilities after its reduction.
// The unused second half holds rescaling and denominator values.
template<int BQ,int BK,int SG,int G,bool Interior=false,bool ReuseScale=false>
inline void attention_online_reuse(device half*q,device half*k,device half*v,device const float*gate,device float*y,constant P&p,uint tid,uint lane,uint sid,uint3 group,threadgroup float*score) {
    threadgroup half*prob=(threadgroup half*)score;
    float maxima[BQ/SG],denoms[BQ/SG];
    uint first=group.x*BQ,headGroup=group.y,kvHead=headGroup*G/8;
    if(first>=p.m*G)return;
    constexpr auto qdesc=matmul2d_descriptor(BQ,BK,256,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
    constexpr auto vdesc=matmul2d_descriptor(BQ,256,BK,false,false,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<qdesc,execution_simdgroups<SG>> qop;
    matmul2d<vdesc,execution_simdgroups<SG>> vop;
    auto queries=tensor(q+headGroup*p.m*G*256,dextents<int,2>{256,int(p.m*G)},array<int,2>{1,256});
    auto keys=tensor(k+kvHead*256,dextents<int,2>{256,int(p.k)},array<int,2>{1,512});
    auto values=tensor(v+kvHead*256,dextents<int,2>{256,int(p.k)},array<int,2>{1,512});
    auto qleft=queries.slice(0,int(first));auto kright=keys.slice(0,0);
    auto probabilities=tensor(prob,extents<int,BK,BQ>{},array<int,2>{1,BK*2});
    auto vright=values.slice(0,0);
    auto out=vop.template get_destination_cooperative_tensor<decltype(probabilities),decltype(vright),float>();
    for(uint i=0;i<out.get_capacity();++i)out[i]=0;
    for(uint i=0;i<BQ/SG;++i){maxima[i]=-INFINITY;denoms[i]=0;}
    uint end=min(p.k,p.pos+(min(p.m*G,first+BQ)+G-1)/G);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint key=0;key<end;key+=BK){
        auto right=keys.slice(0,int(key));
        auto scores=qop.template get_destination_cooperative_tensor<decltype(qleft),decltype(right),float>();
        for(uint i=0;i<scores.get_capacity();++i)scores[i]=0;
        if constexpr(Interior){
            if(first+BQ<=p.m*G&&key+BK<=p.k){auto a=queries.template slice<256,BQ>(0,int(first));auto b=keys.template slice<256,BK>(0,int(key));qop.run(a,b,scores);}
            else qop.run(qleft,right,scores);
        }else qop.run(qleft,right,scores);
        auto shared=tensor(score,extents<int,BK,BQ>{},array<int,2>{1,BK});scores.store(shared);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint ir=0;ir<BQ/SG;++ir){
            uint r=sid+ir*SG;
            uint logical=first+r,row=logical/G;float vals[BK/32],top=-INFINITY;
            // A full historical block is visible to every query in this tile.
            // Keep the original per-row mask at the causal edge and all tails.
            if(ReuseScale&&first+BQ<=p.m*G&&key+BK<=min(p.k,p.pos+first/G+1)){
                for(uint j=0;j<BK/32;++j){float a=score[r*BK+lane+j*32]*(1.0f/16.0f);vals[j]=a;top=max(top,a);}
            }else{
                for(uint j=0;j<BK/32;++j){uint t=key+lane+j*32;float a=(row<p.m&&t<p.k&&t<=p.pos+row)?score[r*BK+lane+j*32]*(1.0f/16.0f):-INFINITY;vals[j]=a;top=max(top,a);}
            }
            float next=max(maxima[ir],simd_max(top));
            float decay=row<p.m?exp(maxima[ir]-next):1.0f,total=0;
            for(uint j=0;j<BK/32;++j){float value=row<p.m?exp(vals[j]-next):0.0f;prob[r*(BK*2)+lane+j*32]=half(value);total+=value;}
            total=simd_sum(total);
            denoms[ir]=denoms[ir]*decay+total;maxima[ir]=next;
            if(lane==0){score[r*BK+BK/2]=decay;score[r*BK+BK/2+1]=denoms[ir];}
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Every SIMD group checks every row, so the branch is uniform across
        // the threadgroup. No extra shared storage beyond the 32 KiB tile.
        bool changed=!ReuseScale;
        if constexpr(ReuseScale)for(uint r=lane;r<BQ;r+=32)changed|=score[r*BK+BK/2]!=1.0f;
        if(simd_any(changed))for(uint i=0;i<out.get_capacity();++i){auto c=out.get_multidimensional_index(i);if(c[1]<BQ&&c[0]<256)out[i]*=score[c[1]*BK+BK/2];}
        auto rightValues=values.slice(0,int(key));
        if constexpr(Interior){if(key+BK<=p.k){auto b=values.template slice<256,BK>(0,int(key));vop.run(probabilities,b,out);}else vop.run(probabilities,rightValues,out);}
        else vop.run(probabilities,rightValues,out);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for(uint i=0;i<out.get_capacity();++i){auto c=out.get_multidimensional_index(i);uint logical=first+c[1],row=logical/G,head=headGroup*G+logical%G;
        if(c[1]<BQ&&c[0]<256&&row<p.m){uint at=(row*16+head)*256+c[0];y[at]=out[i]/score[c[1]*BK+BK/2+1]*sigmoid(gate[at]);}}
}

#define ONLINE_REUSE(BQ,BK,SG,G) \
kernel void attention_online_q##BQ##_k##BK##_s##SG##_g##G##_r1(device half*q [[buffer(0)]],device half*k [[buffer(1)]],device half*v [[buffer(2)]],device const float*gate [[buffer(3)]],device float*y [[buffer(4)]],constant P&p [[buffer(15)]],uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint sid [[simdgroup_index_in_threadgroup]],uint3 group [[threadgroup_position_in_grid]]) { \
    threadgroup float score[BQ*BK];attention_online_reuse<BQ,BK,SG,G,true>(q,k,v,gate,y,p,tid,lane,sid,group,score); }
ONLINE_REUSE(32,256,8,1)
ONLINE_REUSE(32,256,8,8)
#undef ONLINE_REUSE

kernel void attention_online_q32_k256_s8_g8_r2(device half*q [[buffer(0)]],device half*k [[buffer(1)]],device half*v [[buffer(2)]],device const float*gate [[buffer(3)]],device float*y [[buffer(4)]],constant P&p [[buffer(15)]],uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint sid [[simdgroup_index_in_threadgroup]],uint3 group [[threadgroup_position_in_grid]]) {
    threadgroup float score[32*256];attention_online_reuse<32,256,8,8,true,true>(q,k,v,gate,y,p,tid,lane,sid,group,score);
}


// One SIMD group owns 32 output columns. Prefetch packed FP8 bytes into
// registers while alternating two FP16 weight stages; accumulate in FP32.
// Layout 1: [N/32, K/32, lane, byte]. Layout 2 interleaves
// four-byte loads across lanes: [N/32, K/32, byte/4, lane, byte%4].
// Both layouts preserve every FP8 byte and use the original BF16 scales.
template<int Layout>
inline uint expert_byte_offset(uint row,uint col,uint k) {
    if constexpr(Layout==1)return (row/32)*k*32+(col/32)*1024+(row%32)*32+col%32;
    if constexpr(Layout==2)return (row/32)*k*32+(col/32)*1024+((col%32)/4)*128+(row%32)*4+col%4;
    return row*k+col;
}
kernel void expert_repack_bytes(device const uchar*raw [[buffer(0)]],device const Expert*experts [[buffer(1)]],device uchar*packed [[buffer(2)]],constant P&p [[buffer(15)]],uint i [[thread_position_in_grid]]) {
    uint size=p.n*p.k,e=i/size;if(e>=p.a)return;
    uint row=(i%size)/p.k,col=i%p.k;Expert d=experts[e];ulong off=p.kind==0?d.gate:p.kind==1?d.up:d.down;
    uint dest=p.flags==1?expert_byte_offset<1>(row,col,p.k):expert_byte_offset<2>(row,col,p.k);
    packed[off+dest]=raw[off+row*p.k+col];
}
template<int Rows, int KS, bool Prefetch, int Layout, class Acc>
inline void expert_staged_accumulate(device half*x,device const uchar*w,device const ushort*s,
    uint k,uint n,uint firstColumn,uint live,uint lane,threadgroup half*stage,thread Acc &acc) {
    constexpr auto desc=matmul2d_descriptor(Rows,32,KS,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc,execution_simdgroups<1>> op;
    auto full=tensor(x,dextents<int,2>{int(k),int(live)},array<int,2>{1,int(k)});
    auto left=full.slice<KS,Rows>(0,0);
    auto b0=tensor(stage,extents<int,KS,32>{},array<int,2>{1,KS});
    auto b1=tensor(stage+32*KS,extents<int,KS,32>{},array<int,2>{1,KS});

    const uint row=firstColumn+lane;
    uchar4 bytes[KS/4];float scale=0;
    if(Prefetch&&row<n){for(uint j=0;j<KS/4;j++)bytes[j]=*((device const uchar4*)(w+expert_byte_offset<Layout>(row,j*4,k)));scale=bf(s[(row/128)*(k/128)]);}
    for(uint block=0;block<k;block+=KS){
        uint buffer=Prefetch?((block/KS)&1):0;
        if(!Prefetch&&row<n){for(uint j=0;j<KS/4;j++)bytes[j]=*((device const uchar4*)(w+expert_byte_offset<Layout>(row,block+j*4,k)));scale=bf(s[(row/128)*(k/128)+block/128]);}
        for(uint j=0;j<KS/4;j++)*((threadgroup half4*)(stage+buffer*32*KS+lane*KS+j*4))=row<n?half4(f8x4(bytes[j])*scale):half4(0);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if(Prefetch&&block+KS<k&&row<n){
            for(uint j=0;j<KS/4;j++)bytes[j]=*((device const uchar4*)(w+expert_byte_offset<Layout>(row,block+KS+j*4,k)));
            if((block+KS)%128==0)scale=bf(s[(row/128)*(k/128)+(block+KS)/128]);
        }
        auto a=full.slice<KS,Rows>(int(block),0);
        if(buffer)op.run(a,b1,acc);else op.run(a,b0,acc);
        if(!Prefetch)simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
}

template<int Rows, int KS, bool Prefetch, bool Fused, int Layout>
inline void expert_staged_run(device half*x,device const uchar*raw,device const Expert*experts,
    device const ExpertTile*tiles,device float*y,constant P&p,uint3 group,uint lane,uint sid,threadgroup half*stage){
    ExpertTile tile=tiles[group.y];Expert d=experts[tile.expert];
    uint choice=Fused?sid:p.kind;
    ulong wo=choice==0?d.gate:choice==1?d.up:d.down,so=choice==0?d.gate_scale:choice==1?d.up_scale:d.down_scale;
    uint column=Fused?group.x*32:group.x*64+sid*32;
    constexpr auto desc=matmul2d_descriptor(Rows,32,KS,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc,execution_simdgroups<1>> op;
    auto full=tensor(x+tile.first*p.k,dextents<int,2>{int(p.k),int(tile.count)},array<int,2>{1,int(p.k)});
    auto a=full.slice<KS,Rows>(0,0);
    auto b=tensor(stage,extents<int,KS,32>{},array<int,2>{1,KS});
    auto acc=op.template get_destination_cooperative_tensor<decltype(a),decltype(b),float>();
    for(uint i=0;i<acc.get_capacity();++i)acc[i]=0;
    expert_staged_accumulate<Rows,KS,Prefetch,Layout>(x+tile.first*p.k,raw+wo,(device const ushort*)(raw+so),p.k,p.n,column,tile.count,lane,stage+sid*64*KS,acc);
    if constexpr(Fused){
        threadgroup float*gate=(threadgroup float*)stage;
        if(sid==0)for(uint i=0;i<acc.get_capacity();++i){if(!acc.is_valid_element(i))continue;auto c=acc.get_multidimensional_index(i);gate[c[1]*32+c[0]]=acc[i];}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(sid==1)for(uint i=0;i<acc.get_capacity();++i){if(!acc.is_valid_element(i))continue;auto c=acc.get_multidimensional_index(i);if(c[1]<tile.count&&column+c[0]<p.n)((device half*)y)[(tile.first+c[1])*p.n+column+c[0]]=half(silu(gate[c[1]*32+c[0]])*acc[i]);}
    }else{
        for(uint i=0;i<acc.get_capacity();++i){if(!acc.is_valid_element(i))continue;auto c=acc.get_multidimensional_index(i);if(c[1]<tile.count&&column+c[0]<p.n)y[(tile.first+c[1])*p.n+column+c[0]]=acc[i];}
    }
}
template<bool Prefetch,bool Fused,int KS=32,int Layout=0>
kernel void expert_staged(device half*x [[buffer(0)]],device const uchar*raw [[buffer(1)]],device const Expert*experts [[buffer(2)]],device const ExpertTile*tiles [[buffer(3)]],device const uint*tileCount [[buffer(4)]],device float*y [[buffer(5)]],constant P&p [[buffer(15)]],uint3 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]],uint sid [[simdgroup_index_in_threadgroup]]){
    if(group.y>=tileCount[0])return;
    threadgroup half stage[128*KS];
    uint rows=tiles[group.y].count;
    if(rows<=8)expert_staged_run<8,KS,Prefetch,Fused,Layout>(x,raw,experts,tiles,y,p,group,lane,sid,stage);
    else if(rows<=16)expert_staged_run<16,KS,Prefetch,Fused,Layout>(x,raw,experts,tiles,y,p,group,lane,sid,stage);
    else expert_staged_run<32,KS,Prefetch,Fused,Layout>(x,raw,experts,tiles,y,p,group,lane,sid,stage);
}
template [[host_name("expert_matrix_staged")]] kernel decltype(expert_staged<false,false>) expert_staged<false,false>;
template [[host_name("expert_matrix_pipeline")]] kernel decltype(expert_staged<true,false>) expert_staged<true,false>;
template [[host_name("expert_gate_up_pipeline")]] kernel decltype(expert_staged<true,true>) expert_staged<true,true>;

// All threads cooperatively scan routes twice, instead of once per expert.
// Ordering inside an expert is unspecified; inverse restores token/top-k order.
kernel void expert_group_parallel(device const int*ids [[buffer(0)]],device uint*countsOut [[buffer(1)]],device uint*offsets [[buffer(2)]],device ExpertTile*tiles [[buffer(3)]],device uint*tileCount [[buffer(4)]],device uint*tokens [[buffer(5)]],device uint*inverse [[buffer(6)]],constant P&p [[buffer(15)]],uint tid [[thread_index_in_threadgroup]]) {
    threadgroup atomic_uint counts[256],cursors[256];
    threadgroup uint rowsBase[256],rowSums[8],tileSums[8];
    atomic_store_explicit(counts+tid,0,memory_order_relaxed);
    atomic_store_explicit(cursors+tid,0,memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint r=tid;r<p.m*8;r+=256)atomic_fetch_add_explicit(counts+ids[r],1,memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint count=atomic_load_explicit(counts+tid,memory_order_relaxed),numTiles=(count+p.a-1)/p.a;
    uint rowPrefix=simd_prefix_exclusive_sum(count),tilePrefix=simd_prefix_exclusive_sum(numTiles);
    if((tid&31)==31){rowSums[tid/32]=rowPrefix+count;tileSums[tid/32]=tilePrefix+numTiles;}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint sg=0;sg<tid/32;sg++){rowPrefix+=rowSums[sg];tilePrefix+=tileSums[sg];}
    rowsBase[tid]=rowPrefix;
    countsOut[tid]=count;offsets[tid]=rowPrefix;
    for(uint t=0;t<numTiles;t++)tiles[tilePrefix+t]={tid,rowPrefix+t*p.a,min(p.a,count-t*p.a),0};
    if(tid==255){offsets[256]=rowPrefix+count;tileCount[0]=tilePrefix+numTiles;}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint r=tid;r<p.m*8;r+=256){uint e=ids[r],out=rowsBase[e]+atomic_fetch_add_explicit(cursors+e,1,memory_order_relaxed);tokens[out]=r/8;inverse[r]=out;}
}
template [[host_name("expert_gate_up_staged")]] kernel decltype(expert_staged<false,true>) expert_staged<false,true>;

template [[host_name("expert_matrix_staged64")]] kernel decltype(expert_staged<false,false,64>) expert_staged<false,false,64>;
template [[host_name("expert_matrix_staged128")]] kernel decltype(expert_staged<false,false,128>) expert_staged<false,false,128>;
template [[host_name("expert_gate_up_staged64")]] kernel decltype(expert_staged<false,true,64>) expert_staged<false,true,64>;
template [[host_name("expert_gate_up_staged128")]] kernel decltype(expert_staged<false,true,128>) expert_staged<false,true,128>;

template [[host_name("expert_matrix_layout1")]] kernel decltype(expert_staged<false,false,32,1>) expert_staged<false,false,32,1>;
template [[host_name("expert_gate_up_layout1")]] kernel decltype(expert_staged<false,true,32,1>) expert_staged<false,true,32,1>;
template [[host_name("expert_matrix_layout1_pipeline")]] kernel decltype(expert_staged<true,false,32,1>) expert_staged<true,false,32,1>;
template [[host_name("expert_gate_up_layout1_pipeline")]] kernel decltype(expert_staged<true,true,32,1>) expert_staged<true,true,32,1>;
template [[host_name("expert_matrix_layout2")]] kernel decltype(expert_staged<false,false,32,2>) expert_staged<false,false,32,2>;
template [[host_name("expert_gate_up_layout2")]] kernel decltype(expert_staged<false,true,32,2>) expert_staged<false,true,32,2>;
template [[host_name("expert_matrix_layout2_pipeline")]] kernel decltype(expert_staged<true,false,32,2>) expert_staged<true,false,32,2>;
template [[host_name("expert_gate_up_layout2_pipeline")]] kernel decltype(expert_staged<true,true,32,2>) expert_staged<true,true,32,2>;

kernel void batch_delta_gates(device const float*q [[buffer(0)]],device const float*k [[buffer(1)]],device const float*v [[buffer(2)]],device const float*a [[buffer(3)]],device const float*b [[buffer(4)]],device const ushort*alog [[buffer(5)]],device const ushort*dt [[buffer(6)]],device float*state [[buffer(7)]],device float*y [[buffer(8)]],device float*tape [[buffer(9)]],device const float*gates [[buffer(10)]],constant P&p [[buffer(15)]],uint g [[thread_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
    uint row=g/32,h=row/128,d=row%128;if(h>=32)return;uint base=(h*128+d)*128+lane,kh=h/2;float4 cell=float4(state[base],state[base+32],state[base+64],state[base+96]);float gain=exp(bf(alog[h])),bias=bf(dt[h]);
    for(uint t=0;t<p.m;++t){uint key=t*2048+kh*128+lane;float4 kv=float4(k[key],k[key+32],k[key+64],k[key+96]),qv=float4(q[key],q[key+32],q[key+64],q[key+96]);cell*=gates[t*64+h];float prediction=simd_sum(dot(cell,kv)),correction=gates[t*64+32+h]*(v[t*8192+4096+h*128+d]-prediction);cell+=correction*kv;float value=simd_sum(dot(cell,qv));if(lane==0)y[t*4096+h*128+d]=value;if(p.flags&1){uint at=t*524288+base;tape[at]=cell.x;tape[at+32]=cell.y;tape[at+64]=cell.z;tape[at+96]=cell.w;}}
    state[base]=cell.x;state[base+32]=cell.y;state[base+64]=cell.z;state[base+96]=cell.w;
}

// Development variants retain FP32 Q/K/V, outputs, and recurrent state.
// Precompute token/head gates once instead of in each value-row recurrence.
kernel void delta_prepare_gates(device const float*a [[buffer(0)]],device const float*b [[buffer(1)]],device const ushort*alog [[buffer(2)]],device const ushort*dt [[buffer(3)]],device float*gates [[buffer(4)]],constant P&p [[buffer(15)]],uint i [[thread_position_in_grid]]) {
    if(i>=p.m*32)return;uint t=i/32,h=i%32;
    gates[t*64+h]=exp(-exp(bf(alog[h]))*softplus(a[i]+bf(dt[h])));
    gates[t*64+32+h]=sigmoid(b[i]);
}

// Eight lanes share a value row. A group carries sixteen rows in registers.
// Blocked variants share Q/K across those rows, without changing input precision.
template<int Block,bool Staged,bool Prefetch=false>
kernel void delta_scan(device const float*q [[buffer(0)]],device const float*k [[buffer(1)]],device const float*v [[buffer(2)]],device const float*gates [[buffer(3)]],device float*state [[buffer(4)]],device float*y [[buffer(5)]],constant P&p [[buffer(15)]],uint group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    static_assert(!Prefetch || (Staged && Block%4==0), "prefetch requires whole FP32 vectors and a shared stage");
    uint h=group/8,first=(group%8)*16,row=tid/8,col=(tid%8)*16,kh=h/2;
    uint base=(h*128+first+row)*128+col;
    float4 cell[4];
    for(uint j=0;j<4;j++)cell[j]=*((device const float4*)(state+base+j*4));
    threadgroup float keys[Staged?Block*128:1],queries[Staged?Block*128:1];
    threadgroup float values[Staged?Block*16:1],decay[Staged?Block:1],beta[Staged?Block:1];
    // Fetch the next block while the dependent recurrence uses shared memory.
    // FP32 inputs, state and the order of every recurrence operation are kept.
    float4 keyFetch[Prefetch?Block/4:1]={},queryFetch[Prefetch?Block/4:1]={},valueFetch=0;
    float2 gateFetch=0;
    auto fetch=[&](uint start){
        if constexpr(Prefetch){
            for(uint j=0;j<Block/4;j++){
                uint at=(j*128+tid)*4,t=start+at/128,c=at%128;
                if(t<p.m){keyFetch[j]=*((device const float4*)(k+t*2048+kh*128+c));queryFetch[j]=*((device const float4*)(q+t*2048+kh*128+c));}
            }
            if(tid<Block*4){uint at=tid*4,t=start+at/16;if(t<p.m)valueFetch=*((device const float4*)(v+t*8192+4096+h*128+first+at%16));}
            if(tid<Block&&start+tid<p.m)gateFetch=float2(gates[(start+tid)*64+h],gates[(start+tid)*64+32+h]);
        }
    };
    if constexpr(Prefetch)fetch(0);
    for(uint start=0;start<p.m;start+=Block){
        uint count=min(uint(Block),p.m-start);
        if constexpr(Staged){
            if constexpr(Prefetch){
                for(uint j=0;j<Block/4;j++){uint at=(j*128+tid)*4;*((threadgroup float4*)(keys+at))=keyFetch[j];*((threadgroup float4*)(queries+at))=queryFetch[j];}
                if(tid<Block*4)*((threadgroup float4*)(values+tid*4))=valueFetch;
                if(tid<Block){decay[tid]=gateFetch.x;beta[tid]=gateFetch.y;}
            }else{
                for(uint i=tid;i<count*128;i+=128){uint t=start+i/128,c=i%128;keys[i]=k[t*2048+kh*128+c];queries[i]=q[t*2048+kh*128+c];}
                for(uint i=tid;i<count*16;i+=128)values[i]=v[(start+i/16)*8192+4096+h*128+first+i%16];
                if(tid<count){decay[tid]=gates[(start+tid)*64+h];beta[tid]=gates[(start+tid)*64+32+h];}
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if constexpr(Prefetch)if(start+Block<p.m)fetch(start+Block);
        for(uint t=0;t<count;t++){
            float d=Staged?decay[t]:gates[(start+t)*64+h];
            float b=Staged?beta[t]:gates[(start+t)*64+32+h];
            float value=Staged?values[t*16+row]:v[(start+t)*8192+4096+h*128+first+row];
            float4 kv[4],qv[4],partial=0;
            for(uint j=0;j<4;j++){
                if constexpr(Staged){kv[j]=*((threadgroup float4*)(keys+t*128+col+j*4));qv[j]=*((threadgroup float4*)(queries+t*128+col+j*4));}
                else {kv[j]=*((device const float4*)(k+(start+t)*2048+kh*128+col+j*4));qv[j]=*((device const float4*)(q+(start+t)*2048+kh*128+col+j*4));}
                cell[j]*=d;partial=fma(cell[j],kv[j],partial);
            }
            float prediction=(partial.x+partial.y)+(partial.z+partial.w);
            for(uint step=1;step<8;step*=2)prediction+=simd_shuffle_xor(prediction,step);
            float correction=b*(value-prediction);partial=0;
            for(uint j=0;j<4;j++){cell[j]=fma(correction,kv[j],cell[j]);partial=fma(cell[j],qv[j],partial);}
            float out=(partial.x+partial.y)+(partial.z+partial.w);
            for(uint step=1;step<8;step*=2)out+=simd_shuffle_xor(out,step);
            if(tid%8==0)y[(start+t)*4096+h*128+first+row]=out;
        }
        if constexpr(Staged)threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for(uint j=0;j<4;j++)*((device float4*)(state+base+j*4))=cell[j];
}
template [[host_name("delta_direct8")]] kernel decltype(delta_scan<16,false>) delta_scan<16,false>;
template [[host_name("delta_staged8_reference")]] kernel decltype(delta_scan<8,true>) delta_scan<8,true>;
template [[host_name("delta_staged8")]] kernel decltype(delta_scan<8,true,true>) delta_scan<8,true,true>;
template [[host_name("delta_staged16")]] kernel decltype(delta_scan<16,true>) delta_scan<16,true>;

// Development sweep: direct device FP16 operands and FP32 accumulation/output.
template<int Rows,int Cols,int Groups>
kernel void dense_tile(device half*x [[buffer(0)]],device half*w [[buffer(1)]],device float*y [[buffer(2)]],constant P&p [[buffer(15)]],uint3 group [[threadgroup_position_in_grid]]) {
    constexpr auto desc=matmul2d_descriptor(Rows,Cols,dynamic_extent,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc,execution_simdgroups<Groups>> op;
    auto left=tensor(x,dextents<int,2>{int(p.k),int(p.m)},array<int,2>{1,int(p.k)});
    auto right=tensor(w,dextents<int,2>{int(p.k),int(p.n)},array<int,2>{1,int(p.k)});
    auto output=tensor(y,dextents<int,2>{int(p.n),int(p.m)},array<int,2>{1,int(p.n)});
    auto a=left.slice(0,int(group.y)*Rows);auto b=right.slice(0,int(group.x)*Cols);auto c=output.slice(int(group.x)*Cols,int(group.y)*Rows);
    auto acc=op.template get_destination_cooperative_tensor<decltype(a),decltype(b),float>();
    for(uint i=0;i<acc.get_capacity();++i)acc[i]=0;
    op.run(a,b,acc);acc.store(c);
}
template [[host_name("dense_m32_n64_s1")]] kernel decltype(dense_tile<32,64,1>) dense_tile<32,64,1>;
template [[host_name("dense_m64_n64_s2")]] kernel decltype(dense_tile<64,64,2>) dense_tile<64,64,2>;
template [[host_name("dense_m64_n64_s4")]] kernel decltype(dense_tile<64,64,4>) dense_tile<64,64,4>;
template [[host_name("dense_m128_n64_s4")]] kernel decltype(dense_tile<128,64,4>) dense_tile<128,64,4>;
template [[host_name("dense_m64_n128_s4")]] kernel decltype(dense_tile<64,128,4>) dense_tile<64,128,4>;
template [[host_name("dense_m128_n128_s4")]] kernel decltype(dense_tile<128,128,4>) dense_tile<128,128,4>;

// Research variants: isolate decoding, metadata, epilogue and sparse dispatch costs.
// Every matrix path retains the baseline SIMD ownership and FP32 accumulation.
template<bool Fused,int Layout,int Options>
__attribute__((noinline)) void expert_ablation_tiny(device half*x,device const uchar*raw,device const Expert*experts,
    device const ExpertTile*tiles,device float*y,constant P&p,uint3 group,uint lane,uint sid,threadgroup half*stage){
    ExpertTile t=tiles[group.y];Expert d=experts[t.expert];uint choice=Fused?sid:p.kind;
    ulong wo=choice==0?d.gate:choice==1?d.up:d.down,so=choice==0?d.gate_scale:choice==1?d.up_scale:d.down_scale;
    uint column=(Fused?group.x*32:group.x*64+sid*32)+lane;
    device const uchar*w=raw+wo;device const ushort*s=(device const ushort*)(raw+so);
    float4 sums[2]={float4(0),float4(0)};float scale=0;
    for(uint j=0;j<p.k;j+=4){
        if(j%128==0)scale=bf(s[(column/(Layout==3?32:128))*(p.k/128)+j/128]);
        half4 weights=half4(f8x4(*((device const uchar4*)(w+expert_byte_offset<(Layout==3?2:Layout)>(column,j,p.k))))*scale);
        for(uint r=0;r<t.count;r++)sums[r]=fma(float4(*((device half4*)(x+(t.first+r)*p.k+j))),float4(weights),sums[r]);
    }
    float out[2];for(uint r=0;r<t.count;r++)out[r]=(sums[r].x+sums[r].y)+(sums[r].z+sums[r].w);
    if constexpr(Fused){
        using Gate=typename conditional<(Options&1)!=0,half,float>::type;
        threadgroup Gate*gate=(threadgroup Gate*)stage;
        if(sid==0)for(uint r=0;r<t.count;r++)gate[r*32+lane]=out[r];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(sid==1)for(uint r=0;r<t.count;r++)((device half*)y)[(t.first+r)*p.n+column]=half(silu(float(gate[r*32+lane]))*out[r]);
    }else for(uint r=0;r<t.count;r++){uint at=(t.first+r)*p.n+column;if constexpr(Options&2)((device half*)y)[at]=half(out[r]);else y[at]=out[r];}
}
// Affine-8 uses the same TensorOps geometry as packed FP8. Metadata and byte
// loads are contiguous across lanes; one scale/bias pair covers two K32 steps.
template<int Rows, class Acc>
inline void expert_affine_accumulate(device half*x,device const uchar*w,device const ushort*s,
    uint k,uint n,uint firstColumn,uint live,uint lane,threadgroup half*stage,thread Acc &acc) {
    constexpr auto desc=matmul2d_descriptor(Rows,32,32,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc,execution_simdgroups<1>> op;
    auto full=tensor(x,dextents<int,2>{int(k),int(live)},array<int,2>{1,int(k)});
    auto b=tensor(stage,extents<int,32,32>{},array<int,2>{1,32});
    uint row=firstColumn+lane;float scale=0,bias=0;uchar4 bytes[8];
    for(uint block=0;block<k;block+=32){
        // Load the K32 payload before decoding/storing it, exposing independent
        // reads without changing operand rounding or the single-stage barriers.
        if(row<n){
            for(uint j=0;j<8;j++)bytes[j]=*((device const uchar4*)(w+expert_byte_offset<2>(row,block+j*4,k)));
            if(block%64==0){ushort2 sb=((device const ushort2*)s)[(row/32)*(k/64)*32+(block/64)*32+row%32];scale=bf(sb.x);bias=bf(sb.y);}
        }
        for(uint j=0;j<8;j++)*((threadgroup half4*)(stage+lane*32+j*4))=
            row<n?half4(fma(float4(bytes[j]),float4(scale),float4(bias))):half4(0);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        auto a=full.slice<32,Rows>(int(block),0);op.run(a,b,acc);
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
}

template<int Rows, int KS, bool Prefetch, int Layout, int Options, class Acc>
inline void expert_ablation_accumulate(device half*x,device const uchar*w,device const ushort*s,
    uint k,uint n,uint firstColumn,uint live,uint lane,threadgroup half*stage,thread Acc &acc) {
    constexpr auto desc=matmul2d_descriptor(Rows,32,KS,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc,execution_simdgroups<1>> op;
    auto full=tensor(x,dextents<int,2>{int(k),int(live)},array<int,2>{1,int(k)});
    auto left=full.slice<KS,Rows>(0,0);
    auto b0=tensor(stage,extents<int,KS,32>{},array<int,2>{1,KS});
    auto b1=tensor(stage+32*KS,extents<int,KS,32>{},array<int,2>{1,KS});

    const uint row=firstColumn+lane;
    uchar4 bytes[KS/4];float scale=0;
    if(Prefetch&&row<n){for(uint j=0;j<KS/4;j++)bytes[j]=*((device const uchar4*)(w+expert_byte_offset<(Layout==3?2:Layout)>(row,j*4,k)));scale=bf(s[(row/128)*(k/128)]);}
    for(uint block=0;block<k;block+=KS){
        uint buffer=Prefetch?((block/KS)&1):0;
        if(!Prefetch&&Layout!=-1&&row<n){for(uint j=0;j<KS/4;j++)bytes[j]=*((device const uchar4*)(w+expert_byte_offset<(Layout==3?2:Layout)>(row,block+j*4,k)));if(!(Options&8)||block%128==0)scale=bf(s[(row/(Layout==3?32:128))*(k/128)+block/128]);}
        for(uint j=0;j<KS/4;j++)*((threadgroup half4*)(stage+buffer*32*KS+lane*KS+j*4))=row<n?(Layout==-1?*((device const half4*)((device const half*)w+row*k+block+j*4)):half4(f8x4(bytes[j])*scale)):half4(0);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if(Prefetch&&block+KS<k&&row<n){
            for(uint j=0;j<KS/4;j++)bytes[j]=*((device const uchar4*)(w+expert_byte_offset<(Layout==3?2:Layout)>(row,block+KS+j*4,k)));
            if((block+KS)%128==0)scale=bf(s[(row/128)*(k/128)+(block+KS)/128]);
        }
        auto a=full.slice<KS,Rows>(int(block),0);
        if(buffer)op.run(a,b1,acc);else op.run(a,b0,acc);
        if(!Prefetch)simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
}

template<int Rows, int KS, bool Prefetch, bool Fused, int Layout, int Options>
inline void expert_ablation_run(device half*x,device const uchar*raw,device const Expert*experts,
    device const ExpertTile*tiles,device float*y,constant P&p,uint3 group,uint lane,uint sid,threadgroup half*stage){
    ExpertTile tile=tiles[group.y];Expert d=experts[tile.expert];
    uint choice=Fused?sid:p.kind;
    ulong wo=choice==0?d.gate:choice==1?d.up:d.down,so=choice==0?d.gate_scale:choice==1?d.up_scale:d.down_scale;
    uint column=Fused?group.x*32:group.x*64+sid*32;
    constexpr auto desc=matmul2d_descriptor(Rows,32,KS,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc,execution_simdgroups<1>> op;
    auto full=tensor(x+tile.first*p.k,dextents<int,2>{int(p.k),int(tile.count)},array<int,2>{1,int(p.k)});
    auto a=full.slice<KS,Rows>(0,0);
    auto b=tensor(stage,extents<int,KS,32>{},array<int,2>{1,KS});
    auto acc=op.template get_destination_cooperative_tensor<decltype(a),decltype(b),float>();
    for(uint i=0;i<acc.get_capacity();++i)acc[i]=0;
    if constexpr(Layout==4)expert_affine_accumulate<Rows>(x+tile.first*p.k,raw+wo,(device const ushort*)(raw+so),p.k,p.n,column,tile.count,lane,stage+sid*64*KS,acc);
    else expert_ablation_accumulate<Rows,KS,Prefetch,Layout,Options>(x+tile.first*p.k,raw+wo,(device const ushort*)(raw+so),p.k,p.n,column,tile.count,lane,stage+sid*64*KS,acc);
    if constexpr(Fused){
        using Gate=typename conditional<(Options&1)!=0,half,float>::type;
        threadgroup Gate*gate=(threadgroup Gate*)stage;
        if(sid==0)for(uint i=0;i<acc.get_capacity();++i){if(!acc.is_valid_element(i))continue;auto c=acc.get_multidimensional_index(i);gate[c[1]*32+c[0]]=acc[i];}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(sid==1)for(uint i=0;i<acc.get_capacity();++i){if(!acc.is_valid_element(i))continue;auto c=acc.get_multidimensional_index(i);if(c[1]<tile.count&&column+c[0]<p.n)((device half*)y)[(tile.first+c[1])*p.n+column+c[0]]=half(silu(float(gate[c[1]*32+c[0]]))*acc[i]);}
    }else{
        if constexpr(Layout==4 && !(Options&2)){
            auto result=tensor(y+tile.first*p.n,dextents<int,2>{int(p.n),int(tile.count)},array<int,2>{1,int(p.n)});
            auto dest=result.slice(int(column),0);acc.store(dest);
        }else for(uint i=0;i<acc.get_capacity();++i){if(!acc.is_valid_element(i))continue;auto c=acc.get_multidimensional_index(i);if(c[1]<tile.count&&column+c[0]<p.n){uint at=(tile.first+c[1])*p.n+column+c[0];if constexpr(Options&2)((device half*)y)[at]=half(acc[i]);else y[at]=acc[i];}}
    }
}
template<bool Prefetch,bool Fused,int KS=32,int Layout=0,int Options=0>
kernel void expert_ablation(device half*x [[buffer(0)]],device const uchar*raw [[buffer(1)]],device const Expert*experts [[buffer(2)]],device const ExpertTile*tiles [[buffer(3)]],device const uint*tileCount [[buffer(4)]],device float*y [[buffer(5)]],constant P&p [[buffer(15)]],uint3 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]],uint sid [[simdgroup_index_in_threadgroup]]){
    if(group.y>=tileCount[0])return;
    threadgroup half stage[128*KS];
    uint rows=tiles[group.y].count;
    if constexpr(Options&4)if(rows<=2){
        expert_ablation_tiny<Fused,Layout,Options>(x,raw,experts,tiles,y,p,group,lane,sid,stage);return;
    }
    if(rows<=8)expert_ablation_run<8,KS,Prefetch,Fused,Layout,Options>(x,raw,experts,tiles,y,p,group,lane,sid,stage);
    else if(rows<=16)expert_ablation_run<16,KS,Prefetch,Fused,Layout,Options>(x,raw,experts,tiles,y,p,group,lane,sid,stage);
    else expert_ablation_run<32,KS,Prefetch,Fused,Layout,Options>(x,raw,experts,tiles,y,p,group,lane,sid,stage);
}
template [[host_name("ablation_gate_up_predecoded")]] kernel decltype(expert_ablation<false,true,32,-1,0>) expert_ablation<false,true,32,-1,0>;
template [[host_name("ablation_matrix_predecoded_tile")]] kernel decltype(expert_ablation<false,false,32,-1,0>) expert_ablation<false,false,32,-1,0>;
template [[host_name("ablation_gate_up_half")]] kernel decltype(expert_ablation<false,true,32,0,3>) expert_ablation<false,true,32,0,3>;
template [[host_name("ablation_matrix_half")]] kernel decltype(expert_ablation<false,false,32,0,3>) expert_ablation<false,false,32,0,3>;
template [[host_name("ablation_gate_up_scale")]] kernel decltype(expert_ablation<false,true,32,0,8>) expert_ablation<false,true,32,0,8>;
template [[host_name("ablation_matrix_scale")]] kernel decltype(expert_ablation<false,false,32,0,8>) expert_ablation<false,false,32,0,8>;
template [[host_name("ablation_gate_up_packed")]] kernel decltype(expert_ablation<false,true,32,2,8>) expert_ablation<false,true,32,2,8>;
template [[host_name("expert_gate_up_affine")]] kernel decltype(expert_ablation<false,true,32,4,0>) expert_ablation<false,true,32,4,0>;
template [[host_name("expert_matrix_affine")]] kernel decltype(expert_ablation<false,false,32,4,0>) expert_ablation<false,false,32,4,0>;
template [[host_name("ablation_matrix_packed")]] kernel decltype(expert_ablation<false,false,32,2,8>) expert_ablation<false,false,32,2,8>;
template [[host_name("ablation_gate_up_metadata")]] kernel decltype(expert_ablation<false,true,32,3,8>) expert_ablation<false,true,32,3,8>;
template [[host_name("ablation_matrix_metadata")]] kernel decltype(expert_ablation<false,false,32,3,8>) expert_ablation<false,false,32,3,8>;
template [[host_name("ablation_gate_up_tiny")]] kernel decltype(expert_ablation<false,true,32,0,4>) expert_ablation<false,true,32,0,4>;
template [[host_name("ablation_matrix_tiny")]] kernel decltype(expert_ablation<false,false,32,0,4>) expert_ablation<false,false,32,0,4>;
template [[host_name("ablation_gate_up_packed_tiny")]] kernel decltype(expert_ablation<false,true,32,2,12>) expert_ablation<false,true,32,2,12>;
template [[host_name("ablation_matrix_packed_tiny")]] kernel decltype(expert_ablation<false,false,32,2,12>) expert_ablation<false,false,32,2,12>;
template [[host_name("ablation_gate_up_combined")]] kernel decltype(expert_ablation<false,true,32,2,15>) expert_ablation<false,true,32,2,15>;
template [[host_name("ablation_matrix_combined")]] kernel decltype(expert_ablation<false,false,32,2,15>) expert_ablation<false,false,32,2,15>;
template [[host_name("ablation_gate_up_original_combined")]] kernel decltype(expert_ablation<false,true,32,0,15>) expert_ablation<false,true,32,0,15>;
template [[host_name("ablation_matrix_original_combined")]] kernel decltype(expert_ablation<false,false,32,0,15>) expert_ablation<false,false,32,0,15>;

kernel void expert_prepare_packed(device const uchar*raw [[buffer(0)]],device const Expert*experts [[buffer(1)]],device uchar*packed [[buffer(2)]],constant P&p [[buffer(15)]],uint i [[thread_position_in_grid]]){
    uint size=p.n*p.k,e=i/size;if(e>=p.a)return;uint row=(i%size)/p.k,col=i%p.k;
    Expert d=experts[e];ulong off=p.kind==0?d.gate:p.kind==1?d.up:d.down;
    packed[off+expert_byte_offset<2>(row,col,p.k)]=raw[off+row*p.k+col];
    if(row%128==0&&col%128==0){ulong so=p.kind==0?d.gate_scale:p.kind==1?d.up_scale:d.down_scale;uint at=(row/128)*(p.k/128)+col/128;((device ushort*)(packed+so))[at]=((device const ushort*)(raw+so))[at];}
}
kernel void expert_dispatch_args(device const uint*count [[buffer(0)]],device uint*args [[buffer(1)]],constant P&p [[buffer(15)]],uint i [[thread_position_in_grid]]){
    if(i==0){args[0]=count[0];args[1]=p.n;args[2]=count[0];args[3]=1;}
}
