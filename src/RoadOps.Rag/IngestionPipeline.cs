using Microsoft.Extensions.AI;
using Microsoft.Extensions.DataIngestion;
using Microsoft.Extensions.DataIngestion.Chunkers;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.Extensions.VectorData;
using Microsoft.ML.Tokenizers;

namespace RoadOps.Rag;

/// <summary>Converts the deliberately small Markdown corpus into DataIngestion's common document shape.</summary>
public sealed class MarkdownDocumentReader : IngestionDocumentReader
{
    public override async Task<IngestionDocument> ReadAsync(Stream source, string identifier, string mediaType, CancellationToken cancellationToken)
    {
        using var reader = new StreamReader(source, leaveOpen: true);
        var markdown = await reader.ReadToEndAsync(cancellationToken);
        var document = new IngestionDocument(identifier);
        var section = new IngestionDocumentSection();
        section.Elements.Add(new IngestionDocumentParagraph(markdown));
        document.Sections.Add(section);
        return document;
    }
}

/// <summary>Writes token chunks after embedding them through the provider-neutral AI abstraction.</summary>
public sealed class VectorChunkWriter(
    VectorStoreCollection<string, StoredChunk> collection,
    IEmbeddingGenerator<string, Embedding<float>> embeddings) : IngestionChunkWriter<string>
{
    public int Count { get; private set; }

    public override async Task WriteAsync(IAsyncEnumerable<IngestionChunk<string>> chunks, CancellationToken cancellationToken)
    {
        await foreach (var chunk in chunks.WithCancellation(cancellationToken))
        {
            var embedding = await embeddings.GenerateAsync(chunk.Content, cancellationToken: cancellationToken);
            await collection.UpsertAsync(new StoredChunk
            {
                Id = $"{chunk.Document.Identifier}-{Count}",
                Document = chunk.Document.Identifier,
                Text = chunk.Content,
                Embedding = embedding.Vector
            }, cancellationToken);
            Count++;
        }
    }
}

public static class RoadOpsIngestionPipeline
{
    public static IngestionPipeline<string> Create(
        RagOptions options,
        VectorStoreCollection<string, StoredChunk> collection,
        IEmbeddingGenerator<string, Embedding<float>> embeddings,
        out VectorChunkWriter writer)
    {
        var chunkOptions = new IngestionChunkerOptions(TiktokenTokenizer.CreateForEncoding("cl100k_base"))
        {
            MaxTokensPerChunk = options.ChunkSize,
            OverlapTokens = options.ChunkOverlap
        };
        writer = new VectorChunkWriter(collection, embeddings);
        return new IngestionPipeline<string>(new MarkdownDocumentReader(), new DocumentTokenChunker(chunkOptions), writer, new IngestionPipelineOptions(), NullLoggerFactory.Instance);
    }
}
