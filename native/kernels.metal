#include <metal_stdlib>
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
kernel void router(device const float*x [[buffer(0)]],device int*ids [[buffer(1)]],device float*prob [[buffer(2)]],constant P&p [[buffer(15)]],uint i [[thread_position_in_grid]]) {
    if(i)return;float peak=-INFINITY,den=0;
    for(uint k=0;k<p.m;++k){float best=-INFINITY;int index=-1;for(uint j=0;j<p.n;++j){bool used=false;for(uint t=0;t<k;++t)used|=ids[t]==int(j);if(!used&&(x[j]>best||(x[j]==best&&(index<0||j<uint(index))))){best=x[j];index=int(j);}}ids[k]=index;prob[k]=best;if(k==0)peak=best;}
    for(uint k=0;k<p.m;++k){prob[k]=exp(prob[k]-peak);den+=prob[k];}for(uint k=0;k<p.m;++k)prob[k]/=den;
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

kernel void matrix_product(device const float*x [[buffer(0)]],device const uchar*w [[buffer(1)]],device const ushort*s [[buffer(2)]],device float*y [[buffer(3)]],constant P&p [[buffer(15)]],uint tid [[thread_index_in_threadgroup]],uint sid [[simdgroup_index_in_threadgroup]],uint3 group [[threadgroup_position_in_grid]]) {
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

kernel void batch_router(device const float*x [[buffer(0)]],device int*ids [[buffer(1)]],device float*prob [[buffer(2)]],constant P&p [[buffer(15)]],uint token [[thread_position_in_grid]]) {
    if(token>=p.k)return;float peak=-INFINITY,den=0;
    for(uint choice=0;choice<8;++choice){float best=-INFINITY;int index=-1;for(uint e=0;e<256;++e){bool used=false;for(uint t=0;t<choice;++t)used|=ids[token*8+t]==int(e);float score=x[token*256+e];if(!used&&(score>best||(score==best&&(index<0||e<uint(index))))){best=score;index=int(e);}}ids[token*8+choice]=index;prob[token*8+choice]=best;if(choice==0)peak=best;}
    for(uint choice=0;choice<8;++choice){prob[token*8+choice]=exp(prob[token*8+choice]-peak);den+=prob[token*8+choice];}for(uint choice=0;choice<8;++choice)prob[token*8+choice]/=den;
}
kernel void expert_counts(device const int*ids [[buffer(0)]],device uint*counts [[buffer(1)]],constant P&p [[buffer(15)]],uint e [[thread_position_in_grid]]) {if(e<256){uint count=0;for(uint i=0;i<p.m*8;++i)count+=ids[i]==int(e);counts[e]=count;}}
kernel void expert_tiles(device const uint*counts [[buffer(0)]],device uint*offsets [[buffer(1)]],device ExpertTile*tiles [[buffer(2)]],device uint*tileCount [[buffer(3)]],constant P&p [[buffer(15)]],uint i [[thread_position_in_grid]]) {
    if(i)return;uint offset=0,tile=0;for(uint e=0;e<256;++e){offsets[e]=offset;for(uint j=0;j<counts[e];j+=16)tiles[tile++]={e,offset+j,min(16u,counts[e]-j),0};offset+=counts[e];}offsets[256]=offset;tileCount[0]=tile;
}
kernel void expert_group(device const int*ids [[buffer(0)]],device const uint*offsets [[buffer(1)]],device uint*tokens [[buffer(2)]],device uint*inverse [[buffer(3)]],constant P&p [[buffer(15)]],uint e [[thread_position_in_grid]]) {
    if(e>=256)return;uint at=offsets[e];for(uint i=0;i<p.m*8;++i)if(ids[i]==int(e)){tokens[at]=i/8;inverse[i]=at++;}
}
kernel void expert_matrix(device const float*x [[buffer(0)]],device const uchar*raw [[buffer(1)]],device const Expert*experts [[buffer(2)]],device const ExpertTile*tiles [[buffer(3)]],device const uint*tileCount [[buffer(4)]],device const uint*tokens [[buffer(5)]],device float*y [[buffer(6)]],constant P&p [[buffer(15)]],uint tid [[thread_index_in_threadgroup]],uint sid [[simdgroup_index_in_threadgroup]],uint3 group [[threadgroup_position_in_grid]]) {
    if(group.y>=tileCount[0])return;ExpertTile tile=tiles[group.y];Expert descriptor=experts[tile.expert];uint firstColumn=group.x*32;
    ulong woff=p.kind==0?descriptor.gate:p.kind==1?descriptor.up:descriptor.down,soff=p.kind==0?descriptor.gate_scale:p.kind==1?descriptor.up_scale:descriptor.down_scale;
    device const uchar*w=raw+woff;device const ushort*s=(device const ushort*)(raw+soff);
    threadgroup float xt[16*33],wt[32*33],result[16*33];
    for(uint i=tid;i<16*33;i+=128)result[i]=0;threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 c0,c1;simdgroup_load(c0,result+sid*8,33);simdgroup_load(c1,result+8*33+sid*8,33);
    for(uint block=0;block<p.k;block+=32){
        for(uint i=tid;i<16*32;i+=128){uint row=i/32,col=i%32;uint inputRow=row<tile.count?(p.kind==2?tile.first+row:tokens[tile.first+row]):0;xt[row*33+col]=row<tile.count?x[inputRow*p.k+block+col]:0;}
        for(uint i=tid;i<32*32;i+=128){uint row=i/32,col=i%32;wt[col*33+row]=firstColumn+row<p.n?weight(w,s,firstColumn+row,block+col,p.k,0):0;}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint kk=0;kk<32;kk+=8){simdgroup_float8x8 a0,a1,b;simdgroup_load(a0,xt+kk,33);simdgroup_load(a1,xt+8*33+kk,33);simdgroup_load(b,wt+kk*33+sid*8,33);simdgroup_multiply_accumulate(c0,a0,b,c0);simdgroup_multiply_accumulate(c1,a1,b,c1);}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_store(c0,result+sid*8,33);simdgroup_store(c1,result+8*33+sid*8,33);threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint i=tid;i<16*32;i+=128){uint row=i/32,col=i%32;if(row<tile.count&&firstColumn+col<p.n)y[(tile.first+row)*p.n+firstColumn+col]=result[row*33+col];}
}
kernel void batch_mix(device const float*residual [[buffer(0)]],device const float*expert [[buffer(1)]],device const uint*inverse [[buffer(2)]],device const float*prob [[buffer(3)]],device const float*shared [[buffer(4)]],device const float*gate [[buffer(5)]],device float*y [[buffer(6)]],constant P&p [[buffer(15)]],uint i [[thread_position_in_grid]]) {
    if(i>=p.m*2048)return;uint token=i/2048,d=i%2048;float value=residual[i]+shared[i]*sigmoid(gate[token]);for(uint e=0;e<8;++e)value+=prob[token*8+e]*expert[inverse[token*8+e]*2048+d];y[i]=value;
}
kernel void batch_conv(device const float*x [[buffer(0)]],device const ushort*w [[buffer(1)]],device float*state [[buffer(2)]],device float*y [[buffer(3)]],constant P&p [[buffer(15)]],uint i [[thread_position_in_grid]]) {
    if(i>=8192)return;float s0=state[i],s1=state[8192+i],s2=state[16384+i],w0=bf(w[i*4]),w1=bf(w[i*4+1]),w2=bf(w[i*4+2]),w3=bf(w[i*4+3]);
    for(uint t=0;t<p.m;++t){float value=x[t*8192+i];y[t*8192+i]=silu(s0*w0+s1*w1+s2*w2+value*w3);s0=s1;s1=s2;s2=value;}state[i]=s0;state[8192+i]=s1;state[16384+i]=s2;
}
kernel void batch_l2(device const float*x [[buffer(0)]],device float*q [[buffer(1)]],device float*k [[buffer(2)]],constant P&p [[buffer(15)]],uint tid [[thread_index_in_threadgroup]],uint group [[threadgroup_position_in_grid]]) {
    uint token=group/16,head=group%16,j=head*128+tid;threadgroup float qs[4],ks[4];threadgroup float qi,ki;float a=x[token*8192+j],b=x[token*8192+2048+j];float qa=simd_sum(a*a),ka=simd_sum(b*b);
    if((tid&31)==0){qs[tid/32]=qa;ks[tid/32]=ka;}threadgroup_barrier(mem_flags::mem_threadgroup);qa=simd_sum(tid<4?qs[tid]:0);ka=simd_sum(tid<4?ks[tid]:0);if(tid==0){qi=rsqrt(qa+1e-6f)*rsqrt(128.0f);ki=rsqrt(ka+1e-6f);}threadgroup_barrier(mem_flags::mem_threadgroup);q[token*2048+j]=a*qi;k[token*2048+j]=b*ki;
}
kernel void batch_delta(device const float*q [[buffer(0)]],device const float*k [[buffer(1)]],device const float*v [[buffer(2)]],device const float*a [[buffer(3)]],device const float*b [[buffer(4)]],device const ushort*alog [[buffer(5)]],device const ushort*dt [[buffer(6)]],device float*state [[buffer(7)]],device float*y [[buffer(8)]],constant P&p [[buffer(15)]],uint g [[thread_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
    uint row=g/32,h=row/128,d=row%128;if(h>=32)return;uint base=(h*128+d)*128+lane,kh=h/2;float4 cell=float4(state[base],state[base+32],state[base+64],state[base+96]);float gain=exp(bf(alog[h])),bias=bf(dt[h]);
    for(uint t=0;t<p.m;++t){uint key=t*2048+kh*128+lane;float4 kv=float4(k[key],k[key+32],k[key+64],k[key+96]),qv=float4(q[key],q[key+32],q[key+64],q[key+96]);cell*=exp(-gain*softplus(a[t*32+h]+bias));float prediction=simd_sum(dot(cell,kv)),correction=sigmoid(b[t*32+h])*(v[t*8192+4096+h*128+d]-prediction);cell+=correction*kv;float value=simd_sum(dot(cell,qv));if(lane==0)y[t*4096+h*128+d]=value;}
    state[base]=cell.x;state[base+32]=cell.y;state[base+64]=cell.z;state[base+96]=cell.w;
}
kernel void batch_split_q(device const float*x [[buffer(0)]],device float*q [[buffer(1)]],device float*gate [[buffer(2)]],constant P&p [[buffer(15)]],uint i [[thread_position_in_grid]]) {if(i<p.m*4096){uint token=i/4096,h=(i%4096)/256,d=i%256;q[i]=x[token*8192+h*512+d];gate[i]=x[token*8192+h*512+256+d];}}
kernel void batch_rope(device const float*q [[buffer(0)]],device const float*k [[buffer(1)]],device const float*v [[buffer(2)]],device float*qr [[buffer(3)]],device float*kc [[buffer(4)]],device float*vc [[buffer(5)]],constant P&p [[buffer(15)]],uint i [[thread_position_in_grid]]) {
    if(i>=p.m*4096)return;uint token=i/4096,j=i%4096;qr[i]=rotated(q+token*4096,j/256,j%256,p.pos+token,p.theta);if(j<512){kc[(p.pos+token)*512+j]=rotated(k+token*512,j/256,j%256,p.pos+token,p.theta);vc[(p.pos+token)*512+j]=v[token*512+j];}
}
kernel void batch_attention_scores(device const float*q [[buffer(0)]],device const float*k [[buffer(1)]],device float*scores [[buffer(2)]],constant P&p [[buffer(15)]],uint g [[thread_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
    uint row=g/32,span=p.pos+p.m,token=row/(16*span),h=(row/span)%16,t=row%span;if(token>=p.m)return;float sum=0;if(t<=p.pos+token)for(uint d=lane;d<256;d+=32)sum+=q[token*4096+h*256+d]*k[(t*2+h/8)*256+d];sum=simd_sum(sum);if(lane==0)scores[(token*16+h)*p.k+t]=t<=p.pos+token?sum*(1.0f/16.0f):-INFINITY;
}
kernel void batch_attention_values(device const float*scores [[buffer(0)]],device const float*v [[buffer(1)]],device const float*gate [[buffer(2)]],device float*y [[buffer(3)]],constant P&p [[buffer(15)]],uint i [[thread_position_in_grid]]) {
    if(i>=p.m*4096)return;uint token=i/4096,h=(i%4096)/256,d=i%256;float sum=0;for(uint t=0;t<=p.pos+token;++t)sum+=scores[(token*16+h)*p.k+t]*v[(t*2+h/8)*256+d];y[i]=sum*sigmoid(gate[i]);
}
