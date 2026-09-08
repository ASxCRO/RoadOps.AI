using Microsoft.Extensions.VectorData;

namespace RoadOps.Rag;

public sealed record RagOptions
{
    public int ChunkSize { get; init; } = 300;
    public int ChunkOverlap { get; init; } = 50;
    public int TopK { get; init; } = 3;
    public string OllamaUrl { get; init; } = "http://localhost:11434";
    public string ChatModel { get; init; } = "llama3.2";
    public string EmbeddingModel { get; init; } = "nomic-embed-text";
    public string DocumentsPath { get; init; } = "docs";
}

public sealed record SourceChunk(string Id, string Document, string Text, double? Score);
public sealed record IngestionResult(int Documents, int Chunks);
public sealed record RagAnswer(string Answer, IReadOnlyList<SourceChunk> Sources);

public sealed class StoredChunk
{
    [VectorStoreKey] public string Id { get; init; } = string.Empty;
    [VectorStoreData(IsIndexed = true)] public string Document { get; init; } = string.Empty;
    [VectorStoreData] public string Text { get; init; } = string.Empty;
    [VectorStoreVector(768)] public ReadOnlyMemory<float> Embedding { get; init; }
}
