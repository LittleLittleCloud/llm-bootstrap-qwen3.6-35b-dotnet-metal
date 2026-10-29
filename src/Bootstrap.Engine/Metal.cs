using System.Runtime.InteropServices;
using System.Runtime.CompilerServices;

namespace Bootstrap.Engine;

[StructLayout(LayoutKind.Sequential)]
internal struct KernelArgs
{
    public uint N, M, K, Kind, Position, Flags, A, B;
    public float Epsilon, Theta, Scale, Value;
    public uint C, D, E, F;
}

internal static unsafe class Native
{
    static Native()
    {
        NativeLibrary.SetDllImportResolver(typeof(Native).Assembly, (name, _, _) =>
        {
            if (name != "bootstrap_metal") return IntPtr.Zero;
            foreach (var path in new[] { Path.Combine(AppContext.BaseDirectory, "libbootstrap_metal.dylib"), Path.Combine(Environment.CurrentDirectory, "native/build/libbootstrap_metal.dylib") })
                if (File.Exists(path)) return NativeLibrary.Load(path);
            return IntPtr.Zero;
        });
    }
    [DllImport("bootstrap_metal", EntryPoint = "boot_create")] internal static extern nint Create([MarshalAs(UnmanagedType.LPUTF8Str)] string source);
    [DllImport("bootstrap_metal", EntryPoint = "boot_error")] internal static extern nint Error(nint context);
    [DllImport("bootstrap_metal", EntryPoint = "boot_destroy")] internal static extern void Destroy(nint context);
    [DllImport("bootstrap_metal", EntryPoint = "boot_buffer")] internal static extern nint Buffer(nint context, ulong bytes);
    [DllImport("bootstrap_metal", EntryPoint = "boot_map")] internal static extern nint Map(nint context, [MarshalAs(UnmanagedType.LPUTF8Str)] string path);
    [DllImport("bootstrap_metal", EntryPoint = "boot_contents")] internal static extern nint Contents(nint buffer);
    [DllImport("bootstrap_metal", EntryPoint = "boot_release_buffer")] internal static extern void Release(nint buffer);
    [DllImport("bootstrap_metal", EntryPoint = "boot_allocated")] internal static extern ulong Allocated(nint context);
    [DllImport("bootstrap_metal", EntryPoint = "boot_begin")] internal static extern int Begin(nint context);
    [DllImport("bootstrap_metal", EntryPoint = "boot_clear")] internal static extern int Clear(nint context, nint buffer);
    [DllImport("bootstrap_metal", EntryPoint = "boot_dispatch")] internal static extern int Dispatch(nint context, [MarshalAs(UnmanagedType.LPUTF8Str)] string name, nint* buffers, ulong* offsets, int count, in KernelArgs parameters, int parameterBytes, uint x, uint y, uint z, uint tx, uint ty, uint tz);
    [DllImport("bootstrap_metal", EntryPoint = "boot_end")] internal static extern int End(nint context);
}

internal sealed unsafe class GpuBuffer : IDisposable
{
    public nint Handle { get; private set; }
    public ulong Length { get; }
    public GpuBuffer(nint handle, ulong length) { Handle = handle; Length = length; }
    public Span<T> Values<T>(int count, ulong offset = 0) where T : unmanaged
    {
        if (offset + (ulong)count * (ulong)sizeof(T) > Length) throw new ArgumentOutOfRangeException(nameof(count));
        return new Span<T>((byte*)Native.Contents(Handle) + offset, count);
    }
    public void Dispose() { if (Handle != 0) { Native.Release(Handle); Handle = 0; } }
    public static implicit operator Binding(GpuBuffer buffer) => new(buffer, 0);
}

internal readonly record struct Binding(GpuBuffer Buffer, ulong Offset = 0);

internal sealed unsafe class Metal : IDisposable
{
    public nint Handle { get; private set; }
    public ulong AllocatedBytes => Native.Allocated(Handle);
    public Metal(string source)
    {
        Handle = Native.Create(source);
        if (Handle == 0) throw new InvalidOperationException("Metal device or runtime shader compilation failed; see compiler diagnostics.");
    }
    void Check(int result) { if (result == 0) throw new InvalidOperationException(Marshal.PtrToStringUTF8(Native.Error(Handle))); }
    public GpuBuffer Allocate(ulong bytes)
    {
        var handle = Native.Buffer(Handle, Math.Max(bytes, 4));
        if (handle == 0) Check(0);
        return new(handle, Math.Max(bytes, 4));
    }
    public GpuBuffer MapFile(string path)
    {
        var handle = Native.Map(Handle, path);
        if (handle == 0) Check(0);
        return new(handle, (ulong)new FileInfo(path).Length);
    }
    public void Begin() => Check(Native.Begin(Handle));
    public void Clear(GpuBuffer buffer) => Check(Native.Clear(Handle, buffer.Handle));
    public void End() => Check(Native.End(Handle));
    public void Dispatch(string kernel, KernelArgs parameters, uint threads, uint group, params Binding[] bindings)
        => Dispatch2D(kernel, parameters, threads, 1, group, bindings);
    public void Dispatch2D(string kernel, KernelArgs parameters, uint threadsX, uint threadsY, uint group, params Binding[] bindings)
    {
        nint* handles = stackalloc nint[bindings.Length];
        ulong* offsets = stackalloc ulong[bindings.Length];
        for (int i = 0; i < bindings.Length; i++) { handles[i] = bindings[i].Buffer.Handle; offsets[i] = bindings[i].Offset; }
        Check(Native.Dispatch(Handle, kernel, handles, offsets, bindings.Length, in parameters, sizeof(KernelArgs), threadsX, threadsY, 1, group, 1, 1));
    }
    public void Dispose() { if (Handle != 0) { Native.Destroy(Handle); Handle = 0; } }
}
