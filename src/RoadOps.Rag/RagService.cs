using CommunityToolkit.VectorData.InMemory;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.VectorData;

namespace RoadOps.Rag;

public sealed class RagService
{
    private readonly IChatClient _chat;
    private readonly IEmbeddingGenerator<string, Embedding<float>> _embeddings;
    private readonly RagOptions _options;
    private readonly InMemoryVectorStore _store;
    private readonly VectorStoreCollection<string, StoredChunk> _chunks;
    private bool _ingested;

    public RagService(IChatClient chat, IEmbeddingGenerator<string, Embedding<float>> embeddings, RagOptions options)
    {
        _chat = chat; _embeddings = embeddings; _options = options;
        _store = new InMemoryVectorStore(new() { EmbeddingGenerator = embeddings });
        _chunks = _store.GetCollection<string, StoredChunk>("roadops-documents");
    }

    public bool IsIngested => _ingested;

    public async Task<IngestionResult> IngestAsync(CancellationToken cancellationToken = default)
    {
        await _chunks.EnsureCollectionDeletedAsync(cancellationToken);
        await _chunks.EnsureCollectionExistsAsync(cancellationToken);
        var docs = Directory.GetFiles(_options.DocumentsPath, "*.md").Select(path => new FileInfo(path)).OrderBy(file => file.Name).ToArray();
        using var pipeline = RoadOpsIngestionPipeline.Create(_options, _chunks, _embeddings, out var writer);
        await foreach (var _ in pipeline.ProcessAsync(docs, cancellationToken)) { }
        _ingested = true; return new(docs.Length, writer.Count);
    }

    public async Task<IReadOnlyList<SourceChunk>> SearchAsync(string question, int? topK = null, CancellationToken cancellationToken = default)
    {
        EnsureIngested(); var embedding = await _embeddings.GenerateAsync(question, cancellationToken: cancellationToken);
        var results = new List<SourceChunk>();
        await foreach (var result in _chunks.SearchAsync(embedding.Vector, topK ?? _options.TopK, cancellationToken: cancellationToken))
            results.Add(new(result.Record.Id, result.Record.Document, result.Record.Text, result.Score));
        return results;
    }

    public async Task<RagAnswer> AskAsync(string question, int? topK = null, CancellationToken cancellationToken = default)
    {
        var sources = await SearchAsync(question, topK, cancellationToken);
        var context = string.Join("\n\n", sources.Select(s => $"[{s.Id}] {s.Text}"));
        var response = await _chat.GetResponseAsync([
            new(ChatRole.System, "Answer only from the supplied road-operations context. If it is insufficient, say so. Cite chunk IDs in square brackets."),
            new(ChatRole.User, $"Context:\n{context}\n\nQuestion: {question}")], cancellationToken: cancellationToken);
        return new(response.Text, sources);
    }

    private void EnsureIngested()
    {
        if (!_ingested) throw new InvalidOperationException("RAG index is empty. Call POST /rag/ingest first.");
    }
}
