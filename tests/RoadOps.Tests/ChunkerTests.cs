using Microsoft.Extensions.DataIngestion;
using Microsoft.Extensions.DataIngestion.Chunkers;
using Microsoft.ML.Tokenizers;
using RoadOps.Rag;

namespace RoadOps.Tests;

public sealed class ChunkerTests
{
    [Fact]
    public async Task Document_token_chunker_preserves_document_identifier_and_overlap()
    {
        var document = new IngestionDocument("inspection");
        var section = new IngestionDocumentSection();
        section.Elements.Add(new IngestionDocumentParagraph(string.Join(' ', Enumerable.Repeat("Road maintenance teams inspect drainage after heavy rainfall and record hazards for repair.", 8))));
        document.Sections.Add(section);
        var options = new IngestionChunkerOptions(TiktokenTokenizer.CreateForEncoding("cl100k_base")) { MaxTokensPerChunk = 8, OverlapTokens = 2 };
        var chunks = new List<IngestionChunk<string>>();
        await foreach (var chunk in new DocumentTokenChunker(options).ProcessAsync(document)) chunks.Add(chunk);
        Assert.True(chunks.Count > 1);
        Assert.All(chunks, x => Assert.Equal("inspection", x.Document.Identifier));
        Assert.Contains("rainfall", chunks[0].Content + chunks[1].Content);
    }
}
