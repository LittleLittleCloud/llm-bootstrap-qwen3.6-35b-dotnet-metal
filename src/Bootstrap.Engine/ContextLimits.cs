namespace Bootstrap.Engine;

internal static class ContextLimits
{
    public const int NativeWindow = 262144;
    public const int AttentionWorkspaceFloats = 32 * 1024 * 1024;

    public static int OutputBudget(int promptTokens, int requested, int contextWindow)
    {
        if (promptTokens < 1 || promptTokens >= contextWindow || requested < 1)
            throw new ArgumentException("Context window exceeded or invalid output budget");
        return Math.Min(requested, contextWindow - promptTokens);
    }

    public static int PrefillCount(int position, int remaining)
    {
        int count = Math.Min(512, remaining);
        return Math.Min(count, AttentionWorkspaceFloats / 16 / checked(position + count));
    }
}
