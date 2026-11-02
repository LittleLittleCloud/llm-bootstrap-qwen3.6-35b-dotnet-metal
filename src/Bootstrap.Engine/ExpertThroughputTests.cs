using System.Diagnostics;
using System.Text.Json;
namespace Bootstrap.Engine;

// Development benchmark: complete routed projections, with an independent CPU
// reference. It does not load a model or contribute leaderboard measurements.
internal static class ExpertThroughputTests
{
    public static void Run(string source, int runs, bool comparison = false)
    {
        if (runs < 1) throw new ArgumentOutOfRangeException(nameof(runs));
        var variants = comparison ? new[] { ("adaptive", true, true), ("adaptive_core", true, true), ("fp16_predecoded", true, false) } : new[] { ("baseline", false, false), ("gathered", true, false), ("adaptive", true, true), ("staged", true, true), ("pipeline", true, true), ("staged64", true, true), ("staged128", true, true) };
        using var gpu = new Metal(source);
        foreach (int k in new[] {2048,512})
        {
            int n=k==2048?512:2048, m=2048*8, experts=256, scaleCount=n/128*(k/128), slot=n*k+scaleCount*2;
            using var x=gpu.Allocate((ulong)m*(ulong)k*4);using var packedX=gpu.Allocate((ulong)m*(ulong)k*2);using var raw=gpu.Allocate((ulong)experts*(ulong)slot);
            using var descriptors=gpu.Allocate((ulong)experts*48);using var tiles=gpu.Allocate((ulong)experts*5*16);
            using var tileCount=gpu.Allocate(4);using var tokens=gpu.Allocate((ulong)m*4);using var y=gpu.Allocate((ulong)(m*n+1)*4);
            for(int i=0;i<m*k;i++) x.Values<float>(m*k)[i]=((i*7L+i/113)%67-33)/64f;
            for(int e=0;e<experts;e++)
            {
                ulong offset=(ulong)(e*slot);
                var desc=descriptors.Values<ulong>(experts*6);
                for(int choice=0;choice<3;choice++){desc[e*6+choice*2]=offset;desc[e*6+choice*2+1]=offset+(ulong)(n*k);}
                var w=raw.Values<byte>(n*k,offset);
                for(int i=0;i<w.Length;i++) w[i]=(byte)(((i*13+e*3)%112)|(i%3==0?128:0));
                raw.Values<ushort>(scaleCount,offset+(ulong)(n*k)).Fill(0x3b80);
            }
            for(int i=0;i<m;i++)tokens.Values<uint>(m)[i]=(uint)((i*31+7)%2048);
            using var decodedWeights = comparison ? gpu.Allocate((ulong)experts*(ulong)n*(ulong)k*2) : null;
            if (comparison) { gpu.Begin(); gpu.Dispatch("expert_unpack_half",new(){A=(uint)experts,N=(uint)n,K=(uint)k,Kind=0},(uint)(experts*n*k/4),128,raw,descriptors,decodedWeights!);gpu.End(); }
            foreach(bool skew in new[]{false,true})
            {
                int first=0,nt=0;
                // Paired experts have 1+127 rows in the skewed case: every row
                // is assigned once, with both small tails and full tiles.
                for(int e=0;e<experts;e++)
                {
                    int rows=skew?(e%2==0?1:127):64;
                    for(int r=0;r<rows;r+=64){var t=tiles.Values<uint>(experts*3*4);t[nt*4]=(uint)e;t[nt*4+1]=(uint)(first+r);t[nt*4+2]=(uint)Math.Min(64,rows-r);nt++;}
                    first+=rows;
                }
                tileCount.Values<uint>(1)[0]=(uint)nt;
                var pack=new KernelArgs{M=(uint)m,K=(uint)k,Kind=k==512?2u:0u};
                gpu.Begin();gpu.Dispatch("expert_input_half",pack,(uint)(m*k/4),128,x,tokens,packedX);gpu.End();
                foreach(var (name,packed,adaptive) in variants)
                {
                    bool staged = name.StartsWith("staged") || name == "pipeline";
                    first=0;nt=0;
                    int tileRows=staged?32:64;
                    for(int e=0;e<experts;e++) {
                        int rows=skew?(e%2==0?1:127):64;
                        for(int r=0;r<rows;r+=tileRows){var t=tiles.Values<uint>(experts*5*4);t[nt*4]=(uint)e;t[nt*4+1]=(uint)(first+r);t[nt*4+2]=(uint)Math.Min(tileRows,rows-r);nt++;}
                        first+=rows;
                    }
                    tileCount.Values<uint>(1)[0]=(uint)nt;
                    string kernel=packed?"expert_matrix_gathered":"expert_matrix_fp16_tile64_up";
                    int groups=staged?2:packed?4:8;
                    if(staged)kernel=name=="pipeline"?"expert_matrix_pipeline":"expert_matrix_"+name;
                    y.Values<float>(m*n+1).Fill(float.NaN);
                    var p=new KernelArgs{N=(uint)n,K=(uint)k,Kind=k==512?2u:0u,Flags=adaptive?1u:0u};
                    void Encode()
                    {
                        gpu.Begin();
                        if(packed && name!="adaptive_core")gpu.Dispatch("expert_input_half",pack,(uint)(m*k/4),128,x,tokens,packedX);
                        if(name=="fp16_predecoded")gpu.Dispatch2D("expert_matrix_predecoded",p,(uint)((n+63)/64*128),(uint)(nt+1),128,packedX,decodedWeights!,tiles,tileCount,y);
                        else if(packed)gpu.Dispatch2D(kernel,p,(uint)((n+63)/64*groups*32),(uint)(nt+1),(uint)(groups*32),packedX,raw,descriptors,tiles,tileCount,y);
                        else gpu.Dispatch2D(kernel,p,(uint)((n+63)/64*groups*32),(uint)(nt+1),(uint)(groups*32),x,raw,descriptors,tiles,tileCount,tokens,y);
                        gpu.End();
                    }
                    for(int run=0;run<=runs;run++)
                    {
                        if(comparison && run==1) { var warm=Stopwatch.StartNew();while(warm.Elapsed.TotalMilliseconds<250)Encode(); }
                        gpu.ResetWallTiming();var watch=Stopwatch.StartNew();Encode();
                        double wall=watch.Elapsed.TotalMilliseconds;
                        if(run==0)
                        {
                            double worst=0;
                            foreach(int tile in new[]{0,1,nt/2,nt-1}) foreach(int r in new[]{0,(int)tiles.Values<uint>(nt*4)[tile*4+2]-1}) foreach(int c in new[]{0,n/2,n-1})
                            {
                                int expert=(int)tiles.Values<uint>(nt*4)[tile*4],row=(int)tiles.Values<uint>(nt*4)[tile*4+1]+r,input=k==512?row:(int)tokens.Values<uint>(m)[row];
                                double expected=0,mag=0;
                                for(int j=0;j<k;j++)
                                {
                                    byte v=raw.Values<byte>(n*k,(ulong)(expert*slot))[c*k+j];int exp=(v>>3)&15, mant=v&7;
                                    double decoded=(v<128?1:-1)*(exp==0?mant/512.0:(1+mant/8.0)*Math.Pow(2,exp-7));
                                    double term=(float)(Half)x.Values<float>(m*k)[input*k+j]*(double)(float)(Half)(decoded/256);expected+=term;mag+=Math.Abs(term);
                                }
                                float actual=y.Values<float>(m*n)[row*n+c];double error=Math.Abs(actual-expected);worst=Math.Max(worst,error);
                                if(!float.IsFinite(actual)||error>1e-6+3e-6*mag)throw new Exception($"{name}/{k}/{skew}/{row}/{c}: {actual} vs {expected}");
                            }
                            if(!float.IsNaN(y.Values<float>(m*n+1)[m*n]))throw new Exception("Output guard overwritten");
                            Console.WriteLine(JsonSerializer.Serialize(new{type="expert_check",name,k,n,skew,max_abs_error=worst}));
                        }
                        Console.WriteLine(JsonSerializer.Serialize(new{type="expert_benchmark",name,k,n,m,skew,tiles=nt,run,wall_ms=wall,timing=gpu.ReadWallTiming()}));
                    }
                }
            }
        }
    }
}
