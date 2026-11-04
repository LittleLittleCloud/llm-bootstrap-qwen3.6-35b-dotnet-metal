#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <string>
#include <unordered_map>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <algorithm>
#include <mach/mach_time.h>

static double trace_now_ns() {
    static const double scale=[] { mach_timebase_info_data_t t; mach_timebase_info(&t); return double(t.numer)/t.denom; }();
    return double(mach_absolute_time())*scale;
}
struct TimelineEvent { std::string name, lane; double start, end; uint64_t command; };

static double wall_now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// Metal limits each timestamp buffer to 4096 samples on this device.
// Use multiple buffers for large prefill commands, without splitting GPU work.
static constexpr NSUInteger timestamp_samples_per_buffer = 4096;
static constexpr NSUInteger timestamp_buffer_count = 8;

// Original platform-only interop. All model operators are in kernels.metal.
struct Context {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLLibrary> library;
    id<MTLCommandBuffer> command;
    id<MTLComputeCommandEncoder> encoder;
    std::unordered_map<std::string,id<MTLComputePipelineState>> pipelines;
    std::string error;
    bool profile=false;
    std::unordered_map<std::string,std::pair<double,uint64_t>> timings;
    bool wall_enabled=false, attention_samples=false, all_samples=false, profile_shapes=false;
    double wall[7]={}; // GPU seconds, encoding seconds, commit seconds, wait seconds, commands, dispatches, sample-resolution seconds
    double attention[6]={}; // QK/softmax/AV seconds, followed by their counts
    double begin_time=0;
    NSMutableArray<id<MTLCounterSampleBuffer>> *samples;
    std::vector<int> sample_kinds;
    std::vector<std::string> sample_names;
    std::unordered_map<std::string,std::pair<double,uint64_t>> kernel_wall;
    std::string kernel_json;
    int timeline_run=-1; bool timeline=false; double timeline_start=0, timeline_begin=0;
    std::vector<TimelineEvent> timeline_events;
};
struct Buffer {
    id<MTLBuffer> value;
    void *mapping=nullptr;
    size_t mapped_size=0;
    ~Buffer() { value=nil; if(mapping) munmap(mapping,mapped_size); }
};
extern "C" {
void *boot_create(const char *source) { @autoreleasepool {
    Context *c=new Context(); c->device=MTLCreateSystemDefaultDevice();
    if(!c->device){delete c;return nullptr;}
    c->profile=getenv("BOOTSTRAP_PROFILE")!=nullptr;
    c->profile_shapes=getenv("BOOTSTRAP_PROFILE_SHAPES")!=nullptr;
    c->queue=[c->device newCommandQueue]; NSError *error=nil;
    MTLCompileOptions *options=[MTLCompileOptions new]; options.mathMode=MTLMathModeSafe; options.languageVersion=MTLLanguageVersion4_0;
    c->library=[c->device newLibraryWithSource:[NSString stringWithUTF8String:source] options:options error:&error];
    if(!c->library){fprintf(stderr,"Metal compile: %s\n",error.description.UTF8String);delete c;return nullptr;}
    return c;
}}
const char *boot_error(void *handle) { return ((Context*)handle)->error.c_str(); }
void boot_profile(void *handle,int enabled) { Context *c=(Context*)handle; c->profile=enabled!=0; c->timings.clear(); }
int boot_wall_reset(void *handle,int sample_attention) { @autoreleasepool {
    Context *c=(Context*)handle;
    if(c->command || c->profile){c->error="Wall profiling requires no active command and serial profiling disabled";return 0;}
    if(sample_attention && !c->samples){
        if(![c->device supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary]){c->error="Stage timestamp counters unsupported";return 0;}
        for(id<MTLCounterSet> set in c->device.counterSets)if([set.name isEqualToString:MTLCommonCounterSetTimestamp]){
            MTLCounterSampleBufferDescriptor *desc=[MTLCounterSampleBufferDescriptor new];
            desc.counterSet=set;desc.storageMode=MTLStorageModeShared;desc.sampleCount=timestamp_samples_per_buffer;
            c->samples=[NSMutableArray new];
            for(NSUInteger i=0;i<timestamp_buffer_count;i++){
                NSError *error=nil;id<MTLCounterSampleBuffer> buffer=[c->device newCounterSampleBufferWithDescriptor:desc error:&error];
                if(!buffer){c->samples=nil;c->error=error.description.UTF8String;return 0;}
                [c->samples addObject:buffer];
            }
            break;
        }
        if(!c->samples){c->error="Timestamp counter set unavailable";return 0;}
    }
    c->timeline_run++;
    c->timeline=getenv("BOOTSTRAP_TIMELINE_FILE") && c->timeline_run>=4;
    c->timeline_events.clear();c->timeline_start=trace_now_ns();
    c->wall_enabled=true;c->attention_samples=sample_attention!=0;c->all_samples=sample_attention==2;c->kernel_wall.clear();
    std::fill_n(c->wall,7,0.0);std::fill_n(c->attention,6,0.0);return 1;
}}
void boot_wall_read(void *handle,double *wall,double *attention) {
    Context *c=(Context*)handle;std::copy_n(c->wall,7,wall);std::copy_n(c->attention,6,attention);
    if(c->timeline) { @autoreleasepool {
        double end=trace_now_ns();NSMutableArray *events=[NSMutableArray new];
        for(auto &e:c->timeline_events)[events addObject:@{@"name":@(e.name.c_str()),@"lane":@(e.lane.c_str()),@"start_ns":@(e.start),@"duration_ns":@(e.end-e.start),@"command":@(e.command)}];
        auto record=@{@"run":@(c->timeline_run),@"sampled":@(c->all_samples),@"start_ns":@(c->timeline_start),@"end_ns":@(end),@"events":events};
        auto data=[NSJSONSerialization dataWithJSONObject:record options:0 error:nil];
        FILE *f=fopen(getenv("BOOTSTRAP_TIMELINE_FILE"),"a");if(!f){perror("timeline output");abort();}
        fwrite(data.bytes,1,data.length,f);fputc('\n',f);fclose(f);
    }}
}
const char *boot_kernel_wall_json(void *handle) { @autoreleasepool {
    Context *c=(Context*)handle;NSMutableArray *rows=[NSMutableArray array];
    for(auto &item:c->kernel_wall)[rows addObject:@{@"kernel":[NSString stringWithUTF8String:item.first.c_str()],@"gpu_ms":@(item.second.first*1000),@"count":@(item.second.second)}];
    NSData *data=[NSJSONSerialization dataWithJSONObject:rows options:0 error:nil];
    c->kernel_json.assign((const char*)data.bytes,data.length);return c->kernel_json.c_str();
}}
void boot_destroy(void *handle) {
    Context *c=(Context*)handle;
    if(c->profile)for(auto &entry:c->timings)fprintf(stderr,"GPU_DIAGNOSTIC %s count=%llu gpu_ms=%.6f\n",entry.first.c_str(),(unsigned long long)entry.second.second,entry.second.first*1000);
    delete c;
}
void *boot_buffer(void *handle,uint64_t bytes) { @autoreleasepool {
    Context *c=(Context*)handle; Buffer *b=new Buffer();
    b->value=[c->device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    if(!b->value){c->error="Metal buffer allocation failed";delete b;return nullptr;}
    memset(b->value.contents,0,bytes); return b;
}}
void *boot_map(void *handle,const char *path) { @autoreleasepool {
    Context *c=(Context*)handle; int fd=open(path,O_RDONLY);
    if(fd<0){c->error="Cannot open weight file";return nullptr;}
    struct stat info; if(fstat(fd,&info)){close(fd);c->error="Cannot stat weight file";return nullptr;}
    size_t page=(size_t)getpagesize(),length=((size_t)info.st_size+page-1)/page*page;
    void *memory=mmap(nullptr,length,PROT_READ|PROT_WRITE,MAP_PRIVATE,fd,0);close(fd);
    if(memory==MAP_FAILED){c->error="Weight mmap failed";return nullptr;}
    Buffer *b=new Buffer();b->mapping=memory;b->mapped_size=length;
    b->value=[c->device newBufferWithBytesNoCopy:memory length:length options:MTLResourceStorageModeShared deallocator:nil];
    if(!b->value){c->error="Metal no-copy weight buffer failed";delete b;return nullptr;}
    return b;
}}
void *boot_contents(void *handle) { return ((Buffer*)handle)->value.contents; }
void boot_release_buffer(void *handle) { delete (Buffer*)handle; }
uint64_t boot_allocated(void *handle) { return ((Context*)handle)->device.currentAllocatedSize; }
int boot_begin(void *handle) { @autoreleasepool {
    Context *c=(Context*)handle;
    if(c->command){c->error="An unfinished command is active";return 0;}
    if(c->wall_enabled)c->begin_time=wall_now();
    if(c->timeline)c->timeline_begin=trace_now_ns();
    c->sample_kinds.clear();c->sample_names.clear();c->command=[c->queue commandBuffer];return c->command!=nil;
}}
int boot_clear(void *handle,void *buffer) { @autoreleasepool {
    Context *c=(Context*)handle;
    if(c->encoder){[c->encoder endEncoding];c->encoder=nil;}
    id<MTLBlitCommandEncoder> blit=[c->command blitCommandEncoder];
    Buffer *b=(Buffer*)buffer;[blit fillBuffer:b->value range:NSMakeRange(0,b->value.length) value:0];[blit endEncoding];return 1;
}}
int boot_dispatch(void *handle,const char *name,void **buffers,const uint64_t *offsets,int count,const void *parameters,int parameter_bytes,uint32_t x,uint32_t y,uint32_t z,uint32_t tx,uint32_t ty,uint32_t tz,void *indirect,uint64_t indirect_offset) { @autoreleasepool {
    Context *c=(Context*)handle; double trace_start=c->timeline?trace_now_ns():0; id<MTLComputePipelineState> pipeline;
    auto found=c->pipelines.find(name);
    if(found==c->pipelines.end()){
        id<MTLFunction> function=[c->library newFunctionWithName:[NSString stringWithUTF8String:name]];
        if(!function){c->error=std::string("Unknown kernel: ")+name;return 0;}
        NSError *error=nil;pipeline=[c->device newComputePipelineStateWithFunction:function error:&error];
        if(!pipeline){c->error=error.description.UTF8String;return 0;}
        c->pipelines[name]=pipeline;
    } else pipeline=found->second;
    int sample_kind=-1;
    if(c->attention_samples){
        if((std::string(name)=="attention_scores_device" || std::string(name)=="batch_attention_scores" || std::string(name)=="batch_attention_scores_tiled" || std::string(name)=="batch_attention_scores_tiled_fp16" || std::string(name)=="batch_attention_scores_tiled_fp16_opt64"))sample_kind=0;
        else if(std::string(name)=="attention_softmax")sample_kind=1;
        else if((std::string(name)=="attention_values_device" || std::string(name)=="batch_attention_values" || std::string(name)=="batch_attention_values_tiled" || std::string(name)=="batch_attention_values_tiled_fp16" || std::string(name)=="batch_attention_values_tiled_fp16_opt64"))sample_kind=2;
    }
    if(c->all_samples || sample_kind>=0){
        if(c->encoder){[c->encoder endEncoding];c->encoder=nil;}
        NSUInteger at=c->sample_kinds.size()*2;
        if(at+1>=timestamp_samples_per_buffer*timestamp_buffer_count){c->error="Timestamp sample capacity exceeded";return 0;}
        MTLComputePassDescriptor *pass=[MTLComputePassDescriptor computePassDescriptor];
        pass.sampleBufferAttachments[0].sampleBuffer=c->samples[at/timestamp_samples_per_buffer];
        at%=timestamp_samples_per_buffer;
        pass.sampleBufferAttachments[0].startOfEncoderSampleIndex=at;
        pass.sampleBufferAttachments[0].endOfEncoderSampleIndex=at+1;
        c->encoder=[c->command computeCommandEncoderWithDescriptor:pass];c->sample_kinds.push_back(sample_kind);c->sample_names.emplace_back(name);
        // Diagnostic attribution only: the same dense kernel serves target,
        // shared-expert and draft projections. Preserve dispatch/pipeline/args.
        // KernelArgs is sixteen 32-bit words; copy to avoid alignment assumptions.
        if(c->profile_shapes && parameter_bytes==64){
            uint32_t p[16];memcpy(p,parameters,sizeof(p));
            c->sample_names.back()+="|N="+std::to_string(p[0])+",M="+std::to_string(p[1])+",K="+std::to_string(p[2])+
                ",Kind="+std::to_string(p[3])+",Flags="+std::to_string(p[5])+",A="+std::to_string(p[6])+",B="+std::to_string(p[7])+",C="+std::to_string(p[12]);
        }
    }
    if(!c->encoder)c->encoder=[c->command computeCommandEncoder];
    [c->encoder setComputePipelineState:pipeline];
    for(int i=0;i<count;++i){Buffer *b=(Buffer*)buffers[i];[c->encoder setBuffer:b->value offset:offsets[i] atIndex:i];}
    [c->encoder setBytes:parameters length:parameter_bytes atIndex:15];
    if(indirect){
        Buffer *args=(Buffer*)indirect;
        if(indirect_offset%4 || indirect_offset>args->value.length || args->value.length-indirect_offset<12){c->error="Invalid indirect dispatch arguments";return 0;}
        [c->encoder dispatchThreadgroupsWithIndirectBuffer:args->value indirectBufferOffset:indirect_offset threadsPerThreadgroup:MTLSizeMake(tx,ty,tz)];
    }else [c->encoder dispatchThreads:MTLSizeMake(x,y,z) threadsPerThreadgroup:MTLSizeMake(tx,ty,tz)];
    if(c->timeline)c->timeline_events.push_back({name,"CPU dispatch",trace_start,trace_now_ns(),uint64_t(c->wall[4])});
    if(c->wall_enabled)c->wall[5]++;
    if(c->all_samples || sample_kind>=0){[c->encoder endEncoding];c->encoder=nil;}
    // Opt-in diagnostic mode serializes submissions and is never used for formal timing.
    if(c->profile){
        [c->encoder endEncoding];c->encoder=nil;[c->command commit];[c->command waitUntilCompleted];
        if(c->command.status!=MTLCommandBufferStatusCompleted){c->error=c->command.error.description.UTF8String;return 0;}
        auto &sample=c->timings[name];sample.first+=c->command.GPUEndTime-c->command.GPUStartTime;sample.second++;
        c->command=[c->queue commandBuffer];
    }
    return 1;
}}
int boot_end(void *handle) { @autoreleasepool {
    Context *c=(Context*)handle;
    if(c->encoder){[c->encoder endEncoding];c->encoder=nil;}
    MTLTimestamp cpu0=0,gpu0=0,cpu1=0,gpu1=0;
    if(!c->sample_kinds.empty())[c->device sampleTimestamps:&cpu0 gpuTimestamp:&gpu0];
    double before=c->wall_enabled?wall_now():0;
    double trace_commit=c->timeline?trace_now_ns():0;
    [c->command commit];
    double committed=c->wall_enabled?wall_now():0;
    double trace_wait=c->timeline?trace_now_ns():0;
    [c->command waitUntilCompleted];
    double completed=c->wall_enabled?wall_now():0;
    if(c->timeline){
        auto id=uint64_t(c->wall[4]);
        c->timeline_events.push_back({"encode","CPU encode",c->timeline_begin,trace_commit,id});
        c->timeline_events.push_back({"commit","CPU commit",trace_commit,trace_wait,id});
        c->timeline_events.push_back({"waitUntilCompleted","CPU wait",trace_wait,trace_now_ns(),id});
        c->timeline_events.push_back({"command","GPU command",c->command.GPUStartTime*1e9,c->command.GPUEndTime*1e9,id});
    }
    if(c->wall_enabled){
        c->wall[0]+=c->command.GPUEndTime-c->command.GPUStartTime;
        c->wall[1]+=before-c->begin_time;c->wall[2]+=committed-before;c->wall[3]+=completed-committed;c->wall[4]++;
    }
    bool ok=c->command.status==MTLCommandBufferStatusCompleted;
    if(!ok)c->error=c->command.error.description.UTF8String;
    if(ok && !c->sample_kinds.empty()){
        [c->device sampleTimestamps:&cpu1 gpuTimestamp:&gpu1];
        NSMutableData *data=[NSMutableData new];
        NSUInteger total=c->sample_kinds.size()*2;
        for(NSUInteger first=0;first<total;first+=timestamp_samples_per_buffer){
            NSUInteger count=std::min(timestamp_samples_per_buffer,total-first);
            NSData *part=[c->samples[first/timestamp_samples_per_buffer] resolveCounterRange:NSMakeRange(0,count)];
            if(part)[data appendData:part];
        }
        if(data.length!=c->sample_kinds.size()*2*sizeof(MTLCounterResultTimestamp) || gpu1<=gpu0 || cpu1<=cpu0){c->error="Invalid timestamp resolution";ok=false;}
        else {
            // sampleTimestamps CPU values are nanoseconds on this Metal API,
            // not raw mach_absolute_time ticks. Validate against command GPU time.
            double seconds_per_tick=double(cpu1-cpu0)*1e-9/double(gpu1-gpu0);
            double sampled_seconds=0;
            std::vector<std::pair<uint64_t,uint64_t>> intervals;
            const auto *values=(const MTLCounterResultTimestamp*)data.bytes;
            for(size_t i=0;i<c->sample_kinds.size();++i){
                uint64_t start=values[i*2].timestamp,end=values[i*2+1].timestamp;
                if(start==MTLCounterErrorValue || end==MTLCounterErrorValue || end<start){c->error="Invalid GPU timestamp sample";ok=false;break;}
                intervals.emplace_back(start,end);
                if(c->timeline){
                    double a=double(cpu0)+double(start-gpu0)*seconds_per_tick*1e9;
                    c->timeline_events.push_back({c->sample_names[i],"GPU kernel",a,a+double(end-start)*seconds_per_tick*1e9,uint64_t(c->wall[4]-1)});
                }
                double seconds=(end-start)*seconds_per_tick;sampled_seconds+=seconds;
                int kind=c->sample_kinds[i];if(kind>=0){c->attention[kind]+=seconds;c->attention[3+kind]++;}
                auto &kernel=c->kernel_wall[c->sample_names[i]];kernel.first+=seconds;kernel.second++;
            }
            double command_seconds=c->command.GPUEndTime-c->command.GPUStartTime;
            // Encoder timestamp intervals can overlap. Validate their span rather
            // than their sum; summing overlap is not a command-buffer duration.
            if(ok && !intervals.empty()) {
                std::sort(intervals.begin(),intervals.end());
                uint64_t first=intervals.front().first,last=intervals.front().second;
                uint64_t union_start=first,union_end=last,union_ticks=0;size_t overlaps=0;
                for(size_t i=1;i<intervals.size();++i){auto [start,end]=intervals[i];last=std::max(last,end);
                    if(start<union_end){overlaps++;union_end=std::max(union_end,end);}
                    else {union_ticks+=union_end-union_start;union_start=start;union_end=end;}}
                union_ticks+=union_end-union_start;
                double span_seconds=(last-first)*seconds_per_tick;
                fprintf(stderr,"PROFILE_COUNTER command_ms=%.6f sampled_ms=%.6f span_ms=%.6f union_ms=%.6f overlapping_intervals=%zu scale=%.12g samples=%zu\n",command_seconds*1000,sampled_seconds*1000,span_seconds*1000,union_ticks*seconds_per_tick*1000,overlaps,seconds_per_tick,intervals.size());
                if(first<gpu0 || last>gpu1 || span_seconds>command_seconds*1.05+1e-5){c->error="Counter span exceeds command-buffer GPU duration or calibration bounds";ok=false;}
            }
        }
        c->wall[6]+=wall_now()-completed;
    }
    c->command=nil;return ok;
}}
}
