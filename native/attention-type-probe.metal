// Diagnostic only: appended to a clone of the production attention template.
#define PROBE_ONLINE(NAME,Q,PROB,SCALE) \
kernel void probe_online_##NAME(device Q*q [[buffer(0)]],device int8_t*k [[buffer(1)]],device int8_t*v [[buffer(2)]],device const float*gate [[buffer(3)]],device float*y [[buffer(4)]],device const float*ks [[buffer(5)]],device const float*vs [[buffer(6)]],constant P&p [[buffer(15)]],uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint sid [[simdgroup_index_in_threadgroup]],uint3 group [[threadgroup_position_in_grid]]) { \
    threadgroup float score[32*128];attention_online_type_probe<32,128,8,8,true,true,true,int8_t,PROB,Q,SCALE>(q,k,v,gate,y,p,tid,lane,sid,group,score,ks,vs); }
PROBE_ONLINE(hh,half,half,true)
PROBE_ONLINE(hb,half,bfloat,true)
PROBE_ONLINE(bh,bfloat,half,true)
PROBE_ONLINE(bb,bfloat,bfloat,true)
PROBE_ONLINE(hb_unit,half,bfloat,true)
PROBE_ONLINE(hb_noscale,half,bfloat,false)
#undef PROBE_ONLINE

template<typename L,typename R,bool PV>
kernel void probe_matmul(device L*a [[buffer(0)]],device R*b [[buffer(1)]],device float*y [[buffer(2)]],uint tid [[thread_index_in_threadgroup]],uint group [[threadgroup_position_in_grid]]) {
    constexpr int M=32,N=PV?256:128,K=PV?128:256;
    constexpr auto desc=matmul2d_descriptor(M,N,K,false,true,PV,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc,execution_simdgroups<8>> op;
    auto right=tensor(b+group*N*K,extents<int,K,N>{},array<int,2>{1,K});
    auto dest=tensor(y+group*M*N,extents<int,N,M>{},array<int,2>{1,N});
    if constexpr(PV){
        threadgroup L stage[M*K];
        for(uint i=tid;i<M*K;i+=256)stage[i]=a[group*M*K+i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto left=tensor(stage,extents<int,K,M>{},array<int,2>{1,K});
        auto out=op.template get_destination_cooperative_tensor<decltype(left),decltype(right),float>();
        for(uint i=0;i<out.get_capacity();i++)out[i]=0;
        op.run(left,right,out);out.store(dest);
    }else{
        auto left=tensor(a+group*M*K,extents<int,K,M>{},array<int,2>{1,K});
        auto out=op.template get_destination_cooperative_tensor<decltype(left),decltype(right),float>();
        for(uint i=0;i<out.get_capacity();i++)out[i]=0;
        op.run(left,right,out);out.store(dest);
    }
}
#define MAT(NAME,L,R,PV) template [[host_name("probe_" #NAME)]] kernel decltype(probe_matmul<L,R,PV>) probe_matmul<L,R,PV>;
MAT(qk_hh,half,half,false)
MAT(qk_hi,half,int8_t,false)
MAT(qk_bb,bfloat,bfloat,false)
MAT(qk_bi,bfloat,int8_t,false)
MAT(pv_hh,half,half,true)
MAT(pv_hi,half,int8_t,true)
MAT(pv_bb,bfloat,bfloat,true)
MAT(pv_bi,bfloat,int8_t,true)
#undef MAT
