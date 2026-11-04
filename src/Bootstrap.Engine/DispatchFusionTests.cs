using System.Diagnostics;
using System.Text.Json;
namespace Bootstrap.Engine;

internal static class DispatchFusionTests
{
    static void Check(float[] expected, Span<float> actual, string name, double tolerance = 0)
    {
        double max = 0, sq = 0, norm = 0;
        for (int i = 0; i < expected.Length; i++) {
            if (!float.IsFinite(actual[i])) throw new Exception(name + " nonfinite");
            double e = actual[i] - expected[i]; max = Math.Max(max, Math.Abs(e)); sq += e * e; norm += (double)expected[i] * expected[i];
            if (Math.Abs(e) > tolerance * (1 + Math.Abs(expected[i]))) throw new Exception($"{name} mismatch at {i}: {actual[i]} vs {expected[i]}");
        }
        if (!float.IsNaN(actual[expected.Length])) throw new Exception(name + " guard overwritten");
        Console.WriteLine(JsonSerializer.Serialize(new { type = "check", name, elements = expected.Length, max_abs_error = max, relative_l2 = Math.Sqrt(sq / Math.Max(norm, 1e-30)), guard = true }));
    }
    static void Bench(Metal gpu, string name, int runs, Action<bool> encode)
    {
        for (int run = 0; run <= runs; run++) foreach (bool candidate in run % 2 == 0 ? new[] { false, true } : new[] { true, false }) {
            var warm = Stopwatch.StartNew(); while (warm.ElapsedMilliseconds < 50) encode(candidate);
            gpu.ResetWallTiming(); var watch = Stopwatch.StartNew(); encode(candidate);
            Console.WriteLine(JsonSerializer.Serialize(new { type = "bench", name, run, candidate, wall_ms = watch.Elapsed.TotalMilliseconds, timing = gpu.ReadWallTiming() }));
        }
    }
    public static void Run(string source, int runs, string denseKernel = "dense_affine_pipeline")
    {
        if (runs < 0) throw new ArgumentOutOfRangeException(nameof(runs));
        if (denseKernel is not ("dense_affine_staged" or "dense_affine_pipeline" or "dense_affine_pipeline64")) throw new ArgumentException("Invalid dense diagnostic");
        int tileRows = denseKernel == "dense_affine_staged" ? 64 : 128;
        using var gpu = new Metal(source);
        foreach (var (m,n,k) in new[] { (9,32,128), (17,64,256), (65,128,512), (2048,256,2048), (2048,512,2048), (2048,2048,512), (2048,2048,4096), (2048,8192,2048) }) {
            using var x=gpu.Allocate((ulong)m*(ulong)k*2);using var w=gpu.Allocate((ulong)n*(ulong)k);using var s=gpu.Allocate((ulong)n*(ulong)(k/64)*4);
            using var wh=gpu.Allocate((ulong)n*(ulong)k*2);using var y=gpu.Allocate((ulong)(m*n+1)*4);
            var rng=new Random(723+m+n+k);for(int i=0;i<m*k;i++)x.Values<Half>(m*k)[i]=(Half)((rng.NextDouble()*2-1)*0.5);
            rng.NextBytes(w.Values<byte>(n*k));for(int i=0;i<n*k/64;i++){s.Values<ushort>(n*k/32)[i*2]=(ushort)(BitConverter.SingleToUInt32Bits(0.00013f*(1+i%7))>>16);s.Values<ushort>(n*k/32)[i*2+1]=(ushort)(BitConverter.SingleToUInt32Bits(-0.017f*(1+i%7))>>16);}
            var p=new KernelArgs{M=(uint)m,N=(uint)n,K=(uint)k,Kind=3};
            void Encode(bool fused){gpu.Begin();if(fused)gpu.Dispatch2D(denseKernel,p,(uint)((n+63)/64)*128,(uint)((m+tileRows-1)/tileRows),128,x,w,s,y);else{gpu.Dispatch("projection_unpack_half",p,(uint)(n*k/4),128,w,s,wh);gpu.Dispatch2D("matrix_product_predecoded",p,(uint)((n+63)/64)*128,(uint)((m+63)/64),128,x,wh,y);}gpu.End();}
            Encode(false);var reference=y.Values<float>(m*n).ToArray();
            // Independent sampled CPU dot with the baseline's FP16 operands.
            foreach(int row in new[]{0,m-1})foreach(int col in new[]{0,n/2,n-1}){double sum=0;for(int j=0;j<k;j++){int gi=col*(k/64)+j/64;float scale=BitConverter.UInt32BitsToSingle((uint)s.Values<ushort>(n*k/32)[gi*2]<<16),bias=BitConverter.UInt32BitsToSingle((uint)s.Values<ushort>(n*k/32)[gi*2+1]<<16);float weight=(float)(Half)MathF.FusedMultiplyAdd(w.Values<byte>(n*k)[col*k+j],scale,bias);sum+=(double)(float)x.Values<Half>(m*k)[row*k+j]*weight;}if(Math.Abs(reference[row*n+col]-sum)>0.0002*(1+Math.Abs(sum)))throw new Exception("Independent dense reference failed");}
            y.Values<float>(m*n+1).Fill(float.NaN);Encode(true);Check(reference,y.Values<float>(m*n+1),$"staged-{m}-{n}-{k}",0.0002);
            if(m==2048&&runs>0)Bench(gpu,$"staged-{m}-{n}-{k}",runs,Encode);
        }
        foreach(int m in new[]{9,17,65,2048}) Multi(gpu,m,runs);
        foreach(int m in new[]{1,9,32,65,2048}) foreach(int position in new[]{0,8192}) Attention(gpu,m,position,runs);
    }
    static void Multi(Metal gpu,int m,int runs)
    {
        const int k=2048,n=512;using var x=gpu.Allocate((ulong)m*k*2);using var w0=gpu.Allocate(n*k*2);using var w1=gpu.Allocate(n*k*2);using var w2=gpu.Allocate(32*k*2);
        using var packed=gpu.Allocate((n*2+32)*k*2);using var y0=gpu.Allocate((ulong)(m*n+1)*4);using var y1=gpu.Allocate((ulong)(m*n+1)*4);using var y2=gpu.Allocate((ulong)(m*32+1)*4);
        using var activation=gpu.Allocate((ulong)(m*n+1)*4);using var half=gpu.Allocate((ulong)(m*n+2)*2);using var dummy=gpu.Allocate(4);
        var rng=new Random(231+m);for(int i=0;i<m*k;i++)x.Values<Half>(m*k)[i]=(Half)(rng.NextDouble()*2-1);
        foreach(var w in new[]{w0,w1,w2})for(int i=0;i<(int)w.Length/2;i++)w.Values<ushort>((int)w.Length/2)[i]=(ushort)(BitConverter.SingleToUInt32Bits((float)((rng.NextDouble()*2-1)/Math.Sqrt(k)))>>16);
        var p=new KernelArgs{M=(uint)m,K=k,A=n,B=n,C=32,D=1,E=1,F=1};
        Binding[] ws=[w0,w1,w2];Binding[] ys=[y0,y1,y2];int[] ns=[n,n,32];
        void Encode(bool many){gpu.Begin();if(many){gpu.Dispatch("projection_unpack_many",p,(n*2+32)*k/4,128,w0,dummy,w1,dummy,w2,dummy,packed);gpu.Dispatch2D("matrix_product_many",p,17*128,(uint)((m+63)/64),128,x,packed,new Binding(packed,n*k*2),new Binding(packed,n*k*4),y0,y1,y2);}else for(int j=0;j<3;j++){var q=new KernelArgs{M=(uint)m,N=(uint)ns[j],K=k,Kind=1};gpu.Dispatch("projection_unpack_half",q,(uint)(ns[j]*k/4),128,ws[j],dummy,new Binding(packed,(ulong)(j==0?0:j==1?n*k*2:n*k*4)));gpu.Dispatch2D("matrix_product_predecoded",q,(uint)((ns[j]+63)/64)*128,(uint)((m+63)/64),128,x,new Binding(packed,(ulong)(j==0?0:j==1?n*k*2:n*k*4)),ys[j]);}gpu.End();}
        Encode(false);var r0=y0.Values<float>(m*n).ToArray();var r1=y1.Values<float>(m*n).ToArray();var r2=y2.Values<float>(m*32).ToArray();
        y0.Values<float>(m*n+1).Fill(float.NaN);y1.Values<float>(m*n+1).Fill(float.NaN);y2.Values<float>(m*32+1).Fill(float.NaN);Encode(true);
        Check(r0,y0.Values<float>(m*n+1),$"many0-{m}");Check(r1,y1.Values<float>(m*n+1),$"many1-{m}");Check(r2,y2.Values<float>(m*32+1),$"many2-{m}");
        if(m==2048&&runs>0)Bench(gpu,"many",runs,Encode);
        var sp=new KernelArgs{M=(uint)m,N=n,K=k,Flags=1};
        void Shared(bool fused){gpu.Begin();if(fused)gpu.Dispatch2D("shared_gate_up_pack",sp,8*128,(uint)((m+63)/64),128,x,packed,new Binding(packed,n*k*2),activation,half);else{for(int j=0;j<2;j++)gpu.Dispatch2D("matrix_product_predecoded",sp,8*128,(uint)((m+63)/64),128,x,new Binding(packed,(ulong)(j*n*k*2)),ys[j]);gpu.Dispatch("silu_product",new(){N=(uint)(m*n)},(uint)(m*n),128,y0,y1,activation);gpu.Dispatch("projection_input_half",new(){M=(uint)m,K=n},(uint)(m*n/4),128,activation,half);}gpu.End();}
        Shared(false);var ar=activation.Values<float>(m*n).ToArray();var hr=half.Values<ushort>(m*n).ToArray();activation.Values<float>(m*n+1).Fill(float.NaN);half.Values<ushort>(m*n+2).Fill(0x7e01);Shared(true);
        Check(ar,activation.Values<float>(m*n+1),$"shared-{m}");if(!hr.AsSpan().SequenceEqual(half.Values<ushort>(m*n))||half.Values<ushort>(m*n+2)[m*n]!=0x7e01)throw new Exception("Shared packed output mismatch");
        // Production consumes only the FP16 operands: do not materialize the
        // otherwise dead FP32 activation. Check both diagnostic and production stores.
        sp.Flags=0;activation.Values<float>(m*n+1).Fill(float.NaN);half.Values<ushort>(m*n+2).Fill(0x7e01);Shared(true);
        if(!hr.AsSpan().SequenceEqual(half.Values<ushort>(m*n))||half.Values<ushort>(m*n+2)[m*n]!=0x7e01||activation.Values<float>(m*n+1).ToArray().Any(float.IsFinite))throw new Exception("Shared packed-only output mismatch");
        if(m==2048&&runs>0)Bench(gpu,"shared",runs,Shared);
    }
    static void Attention(Metal gpu,int m,int position,int runs)
    {
        using var q=gpu.Allocate((ulong)m*8192*4);using var k=gpu.Allocate((ulong)m*512*4);using var v=gpu.Allocate((ulong)m*512*4);using var qw=gpu.Allocate(512);using var kw=gpu.Allocate(512);
        using var split=gpu.Allocate((ulong)m*4096*4);using var kn=gpu.Allocate((ulong)m*512*4);using var rot=gpu.Allocate((ulong)m*4096*4);using var gate=gpu.Allocate((ulong)(m*4096+1)*4);using var packed=gpu.Allocate((ulong)(m*4096+2)*2);
        int cache=(position+m)*512;using var kc=gpu.Allocate((ulong)(cache+2)*2);using var vc=gpu.Allocate((ulong)(cache+2)*2);using var dummy=gpu.Allocate(4);
        var rng=new Random(319+m+position);foreach(var b in new[]{q,k,v})for(int i=0;i<(int)b.Length/4;i++)b.Values<float>((int)b.Length/4)[i]=(float)(rng.NextDouble()*2-1);
        foreach(var w in new[]{qw,kw})for(int i=0;i<256;i++)w.Values<ushort>(256)[i]=(ushort)(BitConverter.SingleToUInt32Bits((float)(rng.NextDouble()*0.5))>>16);
        var p=new KernelArgs{M=(uint)m,K=(uint)(position+m),Position=(uint)position,Theta=10000000,Epsilon=1e-6f,Flags=256,A=8};
        void Encode(bool fused){gpu.Begin();if(fused)gpu.Dispatch("attention_prepare_fused",p,(uint)m*18*256,256,q,k,v,qw,kw,packed,gate,kc,vc);else{gpu.Dispatch("batch_split_q",p,(uint)m*4096,128,q,split,gate);gpu.Dispatch("rmsnorm",new(){N=256,M=(uint)m*16,Flags=1,Epsilon=1e-6f},(uint)m*16*256,256,split,qw,split,dummy);gpu.Dispatch("rmsnorm",new(){N=256,M=(uint)m*2,Flags=1,Epsilon=1e-6f},(uint)m*2*256,256,k,kw,kn,dummy);gpu.Dispatch("batch_rope",p,(uint)m*4096,128,split,kn,v,rot,kc,vc);gpu.Dispatch("attention_query_grouped_half",p,(uint)m*1024,128,rot,packed);}gpu.End();}
        kc.Values<ushort>(cache+2).Fill(0x7e01);vc.Values<ushort>(cache+2).Fill(0x7e01);Encode(false);var pr=packed.Values<ushort>(m*4096).ToArray();var kr=kc.Values<ushort>(cache+2).ToArray();var vr=vc.Values<ushort>(cache+2).ToArray();var gr=gate.Values<float>(m*4096).ToArray();
        packed.Values<ushort>(m*4096+2).Fill(0x7e01);gate.Values<float>(m*4096+1).Fill(float.NaN);kc.Values<ushort>(cache+2).Fill(0x7e01);vc.Values<ushort>(cache+2).Fill(0x7e01);Encode(true);
        Check(gr,gate.Values<float>(m*4096+1),$"attention-gate-{m}-{position}");
        if(!pr.AsSpan().SequenceEqual(packed.Values<ushort>(m*4096))||packed.Values<ushort>(m*4096+2)[m*4096]!=0x7e01||!kr.AsSpan().SequenceEqual(kc.Values<ushort>(cache+2))||!vr.AsSpan().SequenceEqual(vc.Values<ushort>(cache+2)))throw new Exception("Attention packed query/cache mismatch");
        Console.WriteLine(JsonSerializer.Serialize(new{type="check",name="attention-packed",m,position,bitwise=true,prefix_and_guards=true}));
        if(m==2048&&position==8192&&runs>0)Bench(gpu,"attention",runs,Encode);
    }
}
