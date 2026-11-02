#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;
using namespace metal;
struct P { uint n,m,k,kind,pos,flags,a,b; float eps,theta,scale,value; uint c,d,e,f; };
struct Expert { ulong gate,gate_scale,up,up_scale,down,down_scale; };
inline float bf(ushort x) { return as_type<float>(uint(x)<<16); }
inline float f8(uchar x) {
    uint e=(x>>3)&15,m=x&7;
    if(e==0)return (x&128?-1.0f:1.0f)*float(m)*(1.0f/512.0f);
    if(e==15&&m==7)return NAN;
    return as_type<float>((uint(x&128)<<24)|((e+120)<<23)|(m<<20));
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
    if(kind==0)return f8x4(*((device const uchar4*)(w+row*cols+col)))*bf(s[(row/128)*((cols+127)/128)+col/128]);
    if(kind==1){ushort4 bits=*((device const ushort4*)((device const ushort*)w+row*cols+col));return as_type<float4>(uint4(bits)<<16);}
    device const float* p=(device const float*)w+row*cols+col;return float4(p[0],p[1],p[2],p[3]);
}
inline float weight(device const uchar *w,device const ushort *s,uint row,uint col,uint cols,uint kind) {
    if(kind==0)return f8(w[row*cols+col])*bf(s[(row/128)*((cols+127)/128)+col/128]);
    if(kind==1)return bf(((device const ushort*)w)[row*cols+col]);
    return ((device const float*)w)[row*cols+col];
}
kernel void fp8_values(device const uchar *x [[buffer(0)]],device float *y [[buffer(1)]],constant P&p [[buffer(15)]],uint i [[thread_position_in_grid]]) {if(i<p.n)y[i]=f8(x[i]);}
kernel void embedding(device const ushort*w [[buffer(0)]],device float*y [[buffer(1)]],constant P&p [[buffer(15)]],uint i [[thread_position_in_grid]]) {if(i<p.n)y[i]=bf(w[p.pos*p.n+i]);}
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
kernel void expert_gate_up(device const float*x [[buffer(0)]],device const uchar*raw [[buffer(1)]],device const Expert*experts [[buffer(2)]],device const int*ids [[buffer(3)]],device float*y [[buffer(4)]],constant P&p [[buffer(15)]],uint g [[thread_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
    uint row=g/32,e=row/p.n,r=row%p.n;if(e>=p.m)return;Expert d=experts[ids[e]];float4 ga=0,ua=0;
    for(uint j=lane*4;j<p.k;j+=128){float4 v=*((device const float4*)(x+j));ga+=v*weight4(raw+d.gate,(device const ushort*)(raw+d.gate_scale),r,j,p.k,0);ua+=v*weight4(raw+d.up,(device const ushort*)(raw+d.up_scale),r,j,p.k,0);}
    float gate=simd_sum(ga.x+ga.y+ga.z+ga.w),up=simd_sum(ua.x+ua.y+ua.z+ua.w);if(lane==0)y[e*p.n+r]=silu(gate)*up;
}
kernel void expert_down(device const float*x [[buffer(0)]],device const uchar*raw [[buffer(1)]],device const Expert*experts [[buffer(2)]],device const int*ids [[buffer(3)]],device float*y [[buffer(4)]],constant P&p [[buffer(15)]],uint g [[thread_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
    uint row=g/32,e=row/p.n,r=row%p.n;if(e>=p.m)return;Expert d=experts[ids[e]];float4 accum=0;
    for(uint j=lane*4;j<p.k;j+=128)accum+=*((device const float4*)(x+e*p.k+j))*weight4(raw+d.down,(device const ushort*)(raw+d.down_scale),r,j,p.k,0);
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
    if(i<4096)qr[i]=rotated(q,i/256,i%256,p.pos,p.theta);if(i<512){kc[p.pos*512+i]=rotated(k,i/256,i%256,p.pos,p.theta);vc[p.pos*512+i]=v[i];}
}
kernel void attention_scores(device const float*q [[buffer(0)]],device const float*k [[buffer(1)]],device float*scores [[buffer(2)]],constant P&p [[buffer(15)]],uint g [[thread_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
    uint row=g/32,h=row/(p.pos+1),t=row%(p.pos+1);if(h>=16)return;float sum=0;for(uint d=lane;d<256;d+=32)sum+=q[h*256+d]*k[(t*2+h/8)*256+d];sum=simd_sum(sum);if(lane==0)scores[h*p.k+t]=sum*(1.0f/16.0f);
}
kernel void attention_softmax(device float*x [[buffer(0)]],constant P&p [[buffer(15)]],uint tid [[thread_index_in_threadgroup]],uint head [[threadgroup_position_in_grid]]) {
    threadgroup float partial[8];threadgroup float maximum,denominator;float top=-INFINITY;
    for(uint i=tid;i<=p.pos;i+=256)top=max(top,x[head*p.k+i]);top=simd_max(top);if((tid&31)==0)partial[tid/32]=top;threadgroup_barrier(mem_flags::mem_threadgroup);top=simd_max(tid<8?partial[tid]:-INFINITY);if(tid==0)maximum=top;threadgroup_barrier(mem_flags::mem_threadgroup);
    float sum=0;for(uint i=tid;i<=p.pos;i+=256){float value=exp(x[head*p.k+i]-maximum);x[head*p.k+i]=value;sum+=value;}sum=simd_sum(sum);if((tid&31)==0)partial[tid/32]=sum;threadgroup_barrier(mem_flags::mem_threadgroup);sum=simd_sum(tid<8?partial[tid]:0);if(tid==0)denominator=sum;threadgroup_barrier(mem_flags::mem_threadgroup);for(uint i=tid;i<=p.pos;i+=256)x[head*p.k+i]/=denominator;
}
kernel void attention_values(device const float*scores [[buffer(0)]],device const float*v [[buffer(1)]],device const float*gate [[buffer(2)]],device float*y [[buffer(3)]],constant P&p [[buffer(15)]],uint i [[thread_position_in_grid]]) {
    if(i>=4096)return;uint h=i/256,d=i%256;float sum=0;for(uint t=0;t<=p.pos;++t)sum+=scores[h*p.k+t]*v[(t*2+h/8)*256+d];y[i]=sum*sigmoid(gate[i]);
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
kernel void batch_conv(device const float*x [[buffer(0)]],device const ushort*w [[buffer(1)]],device float*state [[buffer(2)]],device float*y [[buffer(3)]],device float*tape [[buffer(4)]],constant P&p [[buffer(15)]],uint i [[thread_position_in_grid]]) {
    if(i>=8192)return;float s0=state[i],s1=state[8192+i],s2=state[16384+i],w0=bf(w[i*4]),w1=bf(w[i*4+1]),w2=bf(w[i*4+2]),w3=bf(w[i*4+3]);
    for(uint t=0;t<p.m;++t){float value=x[t*8192+i];y[t*8192+i]=silu(s0*w0+s1*w1+s2*w2+value*w3);s0=s1;s1=s2;s2=value;if(p.flags&1){tape[t*24576+i]=s0;tape[t*24576+8192+i]=s1;tape[t*24576+16384+i]=s2;}}state[i]=s0;state[8192+i]=s1;state[16384+i]=s2;
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
    if(i>=p.m*4096)return;uint token=i/4096,j=i%4096;qr[i]=rotated(q+token*4096,j/256,j%256,p.pos+token,p.theta);if(j<512){kc[(p.pos+token)*512+j]=rotated(k+token*512,j/256,j%256,p.pos+token,p.theta);vc[(p.pos+token)*512+j]=v[token*512+j];}
}
kernel void batch_attention_scores(device const float*q [[buffer(0)]],device const float*k [[buffer(1)]],device float*scores [[buffer(2)]],constant P&p [[buffer(15)]],uint3 g [[thread_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
    uint token=g.y/16,h=g.y%16,t=g.x/32;if(token>=p.m)return;float sum=0;if(t<=p.pos+token)for(uint d=lane;d<256;d+=32)sum+=q[token*4096+h*256+d]*k[(t*2+h/8)*256+d];sum=simd_sum(sum);if(lane==0)scores[(token*16+h)*p.k+t]=t<=p.pos+token?sum*(1.0f/16.0f):-INFINITY;
}
kernel void batch_attention_values(device const float*scores [[buffer(0)]],device const float*v [[buffer(1)]],device const float*gate [[buffer(2)]],device float*y [[buffer(3)]],constant P&p [[buffer(15)]],uint i [[thread_position_in_grid]]) {
    if(i>=p.m*4096)return;uint token=i/4096,h=(i%4096)/256,d=i%256;float sum=0;for(uint t=0;t<=p.pos+token;++t)sum+=scores[(token*16+h)*p.k+t]*v[(t*2+h/8)*256+d];y[i]=sum*sigmoid(gate[i]);
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
    kc[at]=draft_rotated(k+row*1024,d,p.pos+row,p.theta);vc[at]=v[i];
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
    float sum=0;if(valid){uint at=context?((t%2048)*8+h/4)*128:((slot-p.a)*8+h/4)*128;device const float*key=context?kc:k;for(uint d=lane;d<128;d+=32)sum+=q[(row*32+h)*128+d]*key[at+d];}
    sum=simd_sum(sum);if(lane==0)scores[index]=valid?sum*rsqrt(128.0f):-INFINITY;
}
kernel void draft_attention_values(device const float*scores [[buffer(0)]],device const float*v [[buffer(1)]],device const float*vc [[buffer(2)]],device float*y [[buffer(3)]],constant P&p [[buffer(15)]],uint i [[thread_position_in_grid]]) {
    if(i>=p.m*4096)return;uint row=i/4096,h=(i%4096)/128,d=i%128,start=p.pos>2048?p.pos-2048:0;float sum=0;
    for(uint t=start;t<p.pos;++t)sum+=scores[(row*32+h)*p.k+t-start]*vc[((t%2048)*8+h/4)*128+d];
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
    for(uint j=lane*4;j<p.k;j+=128){float4 weight=weight4(raw+woff,(device const ushort*)(raw+soff),column,j,p.k,0);
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
        for(uint i=tid;i<32*32;i+=128){uint t=i/32,d=i%32;kt[d*33+t]=firstKey+t<p.k?k[((firstKey+t)*2+head/8)*256+block+d]:0;}
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
        for(uint i=tid;i<64*32;i+=128){uint t=block+i/32,d=firstDim+i%32;vt[(i/32)*33+i%32]=t<p.k?v[(t*2+head/8)*256+d]:0;}
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
        for(uint i=tid;i<32*32;i+=128){uint t=i/32,d=i%32;kt[d*33+t]=firstKey+t<p.k?k[((firstKey+t)*2+head/8)*256+block+d]:0;}
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
        for(uint i=tid;i<64*32;i+=128){uint t=block+i/32,d=firstDim+i%32;vt[(i/32)*33+i%32]=t<p.k?v[(t*2+head/8)*256+d]:0;}
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
            for(uint i=tid;i<64*64;i+=128){uint t=firstKey+i/64,d=block+i%64;rightData[(i%64)*64+i/64]=t<p.k?half(k[(t*2+head/8)*256+d]):half(0);}
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
            for(uint i=tid;i<64*64;i+=128){uint t=firstKey+i/64,d=firstDim+i%64;rightData[i]=t<p.k?half(v[(t*2+head/8)*256+d]):half(0);}
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
            for(uint i=tid;i<128*64;i+=128){uint t=firstKey+i/64,d=block+i%64;rightData[(i%64)*128+i/64]=t<p.k?half(k[(t*2+head/8)*256+d]):half(0);}
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
            for(uint i=tid;i<128*64;i+=128){uint t=firstKey+i/64,d=firstDim+i%64;rightData[i]=t<p.k?half(v[(t*2+head/8)*256+d]):half(0);}
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
        for(uint i=tid;i<32*64;i+=128){uint t=i/64,d=i%64;kt[t*65+d]=firstKey+t<p.k?k[((firstKey+t)*2+head/8)*256+block+d]:0;}
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
        for(uint i=tid;i<64*32;i+=128){uint t=block+i/32,d=firstDim+i%32;vt[(i/32)*33+i%32]=t<p.k?v[(t*2+head/8)*256+d]:0;}
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
