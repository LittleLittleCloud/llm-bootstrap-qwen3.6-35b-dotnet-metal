#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <string>
#include <unordered_map>

// Original platform-only interop. All model operators are in kernels.metal.
struct Context {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLLibrary> library;
    id<MTLCommandBuffer> command;
    id<MTLComputeCommandEncoder> encoder;
    std::unordered_map<std::string,id<MTLComputePipelineState>> pipelines;
    std::string error;
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
    c->queue=[c->device newCommandQueue]; NSError *error=nil;
    MTLCompileOptions *options=[MTLCompileOptions new]; options.mathMode=MTLMathModeSafe;
    c->library=[c->device newLibraryWithSource:[NSString stringWithUTF8String:source] options:options error:&error];
    if(!c->library){fprintf(stderr,"Metal compile: %s\n",error.description.UTF8String);delete c;return nullptr;}
    return c;
}}
const char *boot_error(void *handle) { return ((Context*)handle)->error.c_str(); }
void boot_destroy(void *handle) { delete (Context*)handle; }
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
    c->command=[c->queue commandBuffer];return c->command!=nil;
}}
int boot_clear(void *handle,void *buffer) { @autoreleasepool {
    Context *c=(Context*)handle;
    if(c->encoder){[c->encoder endEncoding];c->encoder=nil;}
    id<MTLBlitCommandEncoder> blit=[c->command blitCommandEncoder];
    Buffer *b=(Buffer*)buffer;[blit fillBuffer:b->value range:NSMakeRange(0,b->value.length) value:0];[blit endEncoding];return 1;
}}
int boot_dispatch(void *handle,const char *name,void **buffers,const uint64_t *offsets,int count,const void *parameters,int parameter_bytes,uint32_t x,uint32_t y,uint32_t z,uint32_t tx,uint32_t ty,uint32_t tz) { @autoreleasepool {
    Context *c=(Context*)handle; id<MTLComputePipelineState> pipeline;
    auto found=c->pipelines.find(name);
    if(found==c->pipelines.end()){
        id<MTLFunction> function=[c->library newFunctionWithName:[NSString stringWithUTF8String:name]];
        if(!function){c->error=std::string("Unknown kernel: ")+name;return 0;}
        NSError *error=nil;pipeline=[c->device newComputePipelineStateWithFunction:function error:&error];
        if(!pipeline){c->error=error.description.UTF8String;return 0;}
        c->pipelines[name]=pipeline;
    } else pipeline=found->second;
    if(!c->encoder)c->encoder=[c->command computeCommandEncoder];
    [c->encoder setComputePipelineState:pipeline];
    for(int i=0;i<count;++i){Buffer *b=(Buffer*)buffers[i];[c->encoder setBuffer:b->value offset:offsets[i] atIndex:i];}
    [c->encoder setBytes:parameters length:parameter_bytes atIndex:15];
    [c->encoder dispatchThreads:MTLSizeMake(x,y,z) threadsPerThreadgroup:MTLSizeMake(tx,ty,tz)];return 1;
}}
int boot_end(void *handle) { @autoreleasepool {
    Context *c=(Context*)handle;
    if(c->encoder){[c->encoder endEncoding];c->encoder=nil;}
    [c->command commit];[c->command waitUntilCompleted];
    bool ok=c->command.status==MTLCommandBufferStatusCompleted;
    if(!ok)c->error=c->command.error.description.UTF8String;
    c->command=nil;return ok;
}}
}
