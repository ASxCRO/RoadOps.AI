using System.Net;
using System.Net.Http.Json;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.AspNetCore.TestHost;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection.Extensions;
using RoadOps.Rag;

namespace RoadOps.Tests;

public sealed class ApiTests
{
    [Fact]
    public async Task Predict_before_training_returns_conflict()
    {
        using var factory = new RoadOpsApiFactory(); using var client = factory.CreateClient();
        var response = await client.PostAsJsonAsync("/ml/predict", new { vehicleSpeed = 40, ambientTemperature = 10, rainfall = 0, roadTemperature = 12, vibration = 1 });
        Assert.Equal(HttpStatusCode.Conflict, response.StatusCode);
    }

    [Fact]
    public async Task Ask_before_ingest_returns_conflict()
    {
        using var factory = new RoadOpsApiFactory(); using var client = factory.CreateClient();
        var response = await client.PostAsJsonAsync("/rag/ask", new { question = "When should salt spreading begin?" });
        Assert.Equal(HttpStatusCode.Conflict, response.StatusCode);
    }

    [Fact]
    public async Task Ingest_then_ask_uses_provider_neutral_fakes()
    {
        using var factory = new RoadOpsApiFactory(); using var client = factory.CreateClient();
        var ingest = await client.PostAsync("/rag/ingest", null);
        Assert.Equal(HttpStatusCode.OK, ingest.StatusCode);
        Assert.DoesNotContain("\"chunks\":0", await ingest.Content.ReadAsStringAsync());
        var response = await client.PostAsJsonAsync("/rag/ask", new { question = "When should salt spreading begin?" });
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        Assert.Contains("[", await response.Content.ReadAsStringAsync());
    }

    [Fact]
    public async Task Unavailable_chat_provider_returns_503()
    {
        using var factory = new RoadOpsApiFactory(providerUnavailable: true); using var client = factory.CreateClient();
        Assert.Equal(HttpStatusCode.OK, (await client.PostAsync("/rag/ingest", null)).StatusCode);
        var response = await client.PostAsJsonAsync("/rag/ask", new { question = "When should salt spreading begin?" });
        Assert.Equal(HttpStatusCode.ServiceUnavailable, response.StatusCode);
    }
}

internal sealed class RoadOpsApiFactory(bool providerUnavailable = false) : WebApplicationFactory<Program>
{
    protected override void ConfigureWebHost(IWebHostBuilder builder)
    {
        builder.UseEnvironment("Testing");
        builder.ConfigureTestServices(services =>
        {
            services.RemoveAll<IChatClient>();
            services.RemoveAll<IEmbeddingGenerator<string, Embedding<float>>>();
            services.RemoveAll<RagService>();
            services.RemoveAll<RagEvaluationService>();
            services.RemoveAll<RoadOps.ML.MlService>();
            services.AddSingleton<IChatClient>(new FakeChatClient(providerUnavailable));
            services.AddSingleton<IEmbeddingGenerator<string, Embedding<float>>>(new FakeEmbeddingGenerator(false));
            services.AddSingleton<RagService>();
            services.AddSingleton<RagEvaluationService>();
            var root = Path.Combine(Path.GetTempPath(), $"roadops-api-test-{Guid.NewGuid()}");
            services.AddSingleton(new RoadOps.ML.MlService(new(Path.Combine(root, "telemetry.csv"), Path.Combine(root, "artifacts"), Epochs: 2)));
        });
    }
}

internal sealed class FakeChatClient(bool unavailable) : IChatClient
{
    public void Dispose() { }
    public Task<ChatResponse> GetResponseAsync(IEnumerable<ChatMessage> messages, ChatOptions? options = null, CancellationToken cancellationToken = default)
    {
        if (unavailable) throw new HttpRequestException("Ollama unavailable");
        var context = messages.Last().Text ?? "No context.";
        return Task.FromResult(new ChatResponse(new ChatMessage(ChatRole.Assistant, $"Answer from supplied context [fake-0]. {context}")));
    }
    public async IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(IEnumerable<ChatMessage> messages, ChatOptions? options = null, [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken = default) { yield break; }
    public object? GetService(Type serviceType, object? serviceKey = null) => null;
}

internal sealed class FakeEmbeddingGenerator(bool unavailable) : IEmbeddingGenerator<string, Embedding<float>>
{
    public void Dispose() { }
    public Task<GeneratedEmbeddings<Embedding<float>>> GenerateAsync(IEnumerable<string> values, EmbeddingGenerationOptions? options = null, CancellationToken cancellationToken = default)
    {
        if (unavailable) throw new HttpRequestException("Ollama unavailable");
        var embeddings = values.Select(value => new Embedding<float>(Vector(value))).ToArray();
        return Task.FromResult(new GeneratedEmbeddings<Embedding<float>>(embeddings));
    }
    public object? GetService(Type serviceType, object? serviceKey = null) => null;
    private static ReadOnlyMemory<float> Vector(string text)
    {
        var vector = new float[768];
        foreach (var character in text.ToLowerInvariant()) vector[character % vector.Length] += 1;
        return vector;
    }
}
