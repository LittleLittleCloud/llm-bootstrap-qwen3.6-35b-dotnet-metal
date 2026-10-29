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
inline float weight(device const uchar *w,device const ushort *s,uint row,uint col,uint cols,uint kind) {
    if(kind==0)return f8(w[row*cols+col])*bf(s[(row/128)*((cols+127)/128)+col/128]);
    if(kind==1)return bf(((device const ushort*)w)[row*cols+col]);
    return ((device const float*)w)[row*cols+col];
}
kernel void fp8_values(device const uchar *x [[buffer(0)]],device float *y [[buffer(1)]],constant P&p [[buffer(15)]],uint i [[thread_position_in_grid]]) {if(i<p.n)y[i]=f8(x[i]);}
kernel void embedding(device const ushort*w [[buffer(0)]],device float*y [[buffer(1)]],constant P&p [[buffer(15)]],uint i [[thread_position_in_grid]]) {if(i<p.n)y[i]=bf(w[p.pos*p.n+i]);}
kernel void gemv(device const float*x [[buffer(0)]],device const uchar*w [[buffer(1)]],device const ushort*s [[buffer(2)]],device float*y [[buffer(3)]],constant P&p [[buffer(15)]],uint g [[thread_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
    uint row=g/32;if(row>=p.n)return;float sum=0;
    for(uint j=lane;j<p.k;j+=32)sum+=x[j]*weight(w,s,row,j,p.k,p.kind);
    sum=simd_sum(sum);if(lane==0)y[row]=sum;
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
    uint row=g/32,e=row/p.n,r=row%p.n;if(e>=p.m)return;Expert d=experts[ids[e]];float gate=0,up=0;
    for(uint j=lane;j<p.k;j+=32){float v=x[j];gate+=v*weight(raw+d.gate,(device const ushort*)(raw+d.gate_scale),r,j,p.k,0);up+=v*weight(raw+d.up,(device const ushort*)(raw+d.up_scale),r,j,p.k,0);}
    gate=simd_sum(gate);up=simd_sum(up);if(lane==0)y[e*p.n+r]=silu(gate)*up;
}
kernel void expert_down(device const float*x [[buffer(0)]],device const uchar*raw [[buffer(1)]],device const Expert*experts [[buffer(2)]],device const int*ids [[buffer(3)]],device float*y [[buffer(4)]],constant P&p [[buffer(15)]],uint g [[thread_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
    uint row=g/32,e=row/p.n,r=row%p.n;if(e>=p.m)return;Expert d=experts[ids[e]];float total=0;
    for(uint j=lane;j<p.k;j+=32)total+=x[e*p.k+j]*weight(raw+d.down,(device const ushort*)(raw+d.down_scale),r,j,p.k,0);
    total=simd_sum(total);if(lane==0)y[e*p.n+r]=total;
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
