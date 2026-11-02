using System.Diagnostics;
using System.Text.Json;
namespace Bootstrap.Engine;
internal static class ExpertFusionTests
{
    public static void Run(string source, int runs)
    {
        using var gpu=new Metal(source);
        const int k=2048,n=512,experts=256,scales=n/128*(k/128),slot=n*k+scales*2;
        using var raw=gpu.Allocate((ulong)experts*2*slot);using var descriptors=gpu.Allocate(experts*48);
        for(int e=0;e<experts;e++)for(int choice=0;choice<2;choice++){
            ulong off=(ulong)((e*2+choice)*slot);var desc=descriptors.Values<ulong>(experts*6);
            desc[e*6+choice*2]=off;desc[e*6+choice*2+1]=off+n*k;
            var w=raw.Values<byte>(n*k,off);
            for(int i=0;i<w.Length;i++)w[i]=(byte)(((i*13+e*3+choice*7)%112)|((i+choice)%3==0?128:0));
            for(int i=0;i<scales;i++)raw.Values<ushort>(scales,off+n*k)[i]=(ushort)(0x3b00+(i%3)*128);
        }
        foreach(int m in new[]{32,257,2048})foreach(bool skew in new[]{false,true}){
            int rows=m*8,maxTiles=rows/16+256;
            using var x=gpu.Allocate((ulong)m*k*4);using var packed=gpu.Allocate((ulong)rows*k*2);
            using var tokens=gpu.Allocate((ulong)rows*4);using var tiles=gpu.Allocate((ulong)maxTiles*16);using var tileCount=gpu.Allocate(4);
            using var gate=gpu.Allocate((ulong)rows*n*4);using var up=gpu.Allocate((ulong)rows*n*4);using var activation=gpu.Allocate((ulong)rows*n*4);
            using var baseline=gpu.Allocate((ulong)(rows*n+1)*2);using var output=gpu.Allocate((ulong)(rows*n+1)*2);
            for(int i=0;i<m*k;i++)x.Values<float>(m*k)[i]=((i*7L+i/113)%67-33)/64f;
            for(int i=0;i<rows;i++)tokens.Values<uint>(rows)[i]=(uint)((i*31+7)%m);
            gpu.Begin();gpu.Dispatch("expert_input_half",new(){M=(uint)rows,K=k},(uint)(rows*k/4),128,x,tokens,packed);gpu.End();
            int[] counts=new int[experts];for(int i=0;i<rows;i++)counts[skew?(i%16==0?i%256:(i%128)*2+1):i%256]++;
            int[] firsts=new int[experts];for(int e=1;e<experts;e++)firsts[e]=firsts[e-1]+counts[e-1];
            int Setup(int tileRows){int nt=0;for(int e=0;e<experts;e++)for(int r=0;r<counts[e];r+=tileRows){var t=tiles.Values<uint>(maxTiles*4).Slice(nt++*4,4);t[0]=(uint)e;t[1]=(uint)(firsts[e]+r);t[2]=(uint)Math.Min(tileRows,counts[e]-r);}tileCount.Values<uint>(1)[0]=(uint)nt;return nt;}
            void Encode(string variant,int nt){
                bool old=variant=="baseline";var p=new KernelArgs{M=(uint)rows,N=n,K=k,Flags=1};
                gpu.Begin();
                if(variant.StartsWith("fused"))gpu.Dispatch2D(variant=="fused"?"expert_gate_up_staged":"expert_gate_up_pipeline",p,n/32*64,(uint)(nt+1),64,packed,raw,descriptors,tiles,tileCount,output);
                else{
                    string kernel=old?"expert_matrix_gathered":"expert_matrix_staged";uint threads=old?128u:64u;
                    gpu.Dispatch2D(kernel,p,n/64*threads,(uint)(nt+1),threads,packed,raw,descriptors,tiles,tileCount,gate);
                    p.Kind=1;gpu.Dispatch2D(kernel,p,n/64*threads,(uint)(nt+1),threads,packed,raw,descriptors,tiles,tileCount,up);
                    gpu.Dispatch("silu_product",new(){N=(uint)(rows*n)},(uint)(rows*n),128,gate,up,activation);
                    gpu.Dispatch("expert_input_half",new(){M=(uint)rows,K=n,Kind=2},(uint)(rows*n/4),128,activation,tokens,old?baseline:output);
                }
                gpu.End();
            }
            foreach(string variant in new[]{"baseline","staged","fused","fused_pipeline"}){
                int nt=Setup(variant=="baseline"?64:32);output.Values<Half>(rows*n+1).Fill(Half.NaN);baseline.Values<Half>(rows*n+1)[rows*n]=Half.NaN;
                Encode(variant,nt);
                if(variant=="baseline"){
                    // Independent CPU reference for distinct gate/up bytes and varying scales.
                    foreach(int e in new[]{0,1,127,255})if(counts[e]>0)foreach(int row in new[]{firsts[e],firsts[e]+counts[e]-1})foreach(int c in new[]{0,127,128,511}){
                        double[] sums=new double[2];for(int ch=0;ch<2;ch++)for(int j=0;j<k;j++){
                            ulong off=(ulong)((e*2+ch)*slot);byte b=raw.Values<byte>(n*k,off)[c*k+j];int exp=(b>>3)&15,man=b&7;
                            double w=(b<128?1:-1)*(exp==0?man/512.0:(1+man/8.0)*Math.Pow(2,exp-7));
                            float scale=BitConverter.UInt32BitsToSingle((uint)raw.Values<ushort>(scales,off+n*k)[c/128*(k/128)+j/128]<<16);
                            sums[ch]+=(float)(Half)(w*scale)*(double)(float)packed.Values<Half>(rows*k)[row*k+j];
                        }
                        double expected=(float)(Half)(sums[0]/(1+Math.Exp(-sums[0]))*sums[1]);double actual=(float)baseline.Values<Half>(rows*n)[row*n+c];
                        if(Math.Abs(expected-actual)>0.00005+0.002*Math.Abs(expected))throw new Exception("Fused fixture CPU reference mismatch");
                    }
                }else{
                    double max=0,sum=0;for(int i=0;i<rows*n;i++){float a=(float)output.Values<Half>(rows*n)[i],b=(float)baseline.Values<Half>(rows*n)[i];double error=Math.Abs(a-b);max=Math.Max(max,error);sum+=error*error;if(!float.IsFinite(a)||error>0.00005+0.002*Math.Abs(b))throw new Exception($"Fusion mismatch {variant}/{m}/{skew}/{i}: {a} vs {b}");}
                    if(!Half.IsNaN(output.Values<Half>(rows*n+1)[rows*n]))throw new Exception("Fusion guard overwritten");
                    Console.WriteLine(JsonSerializer.Serialize(new{type="fusion_check",variant,m,skew,max_abs_error=max,rmse=Math.Sqrt(sum/(rows*n))}));
                }
                for(int run=0;run<runs;run++){gpu.ResetWallTiming();var watch=Stopwatch.StartNew();Encode(variant,nt);double wall=watch.Elapsed.TotalMilliseconds;Console.WriteLine(JsonSerializer.Serialize(new{type="fusion_benchmark",variant,m,skew,run,wall_ms=wall,timing=gpu.ReadWallTiming()}));}
            }
        }
    }
}
