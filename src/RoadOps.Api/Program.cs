using Microsoft.Extensions.AI;
using OllamaSharp;
using RoadOps.ML;
using RoadOps.Rag;

var builder = WebApplication.CreateBuilder(args);
builder.Services.AddOpenApi();
var ragOptions = builder.Configuration.GetSection("Rag").Get<RagOptions>() ?? new RagOptions();
ragOptions = ragOptions with { DocumentsPath = Path.GetFullPath(Path.Combine(builder.Environment.ContentRootPath, "..", "..", ragOptions.DocumentsPath)) };
builder.Services.AddSingleton(ragOptions);
builder.Services.AddSingleton(new MlService(new(Path.Combine(builder.Environment.ContentRootPath, "data", "telemetry.csv"), Path.Combine(builder.Environment.ContentRootPath, "artifacts"))));
builder.Services.AddSingleton<IChatClient>(_ => (IChatClient)new OllamaApiClient(new Uri(ragOptions.OllamaUrl), ragOptions.ChatModel));
builder.Services.AddSingleton<IEmbeddingGenerator<string, Embedding<float>>>(_ => (IEmbeddingGenerator<string, Embedding<float>>)new OllamaApiClient(new Uri(ragOptions.OllamaUrl), ragOptions.EmbeddingModel));
builder.Services.AddSingleton<RagService>();
builder.Services.AddSingleton<RagEvaluationService>();

var app = builder.Build();
app.MapOpenApi();

app.MapPost("/ml/train", (MlService ml) => Results.Ok(ml.Train()));
app.MapPost("/ml/predict", (Telemetry request, MlService ml) =>
{
    try { return Results.Ok(ml.Predict(request)); }
    catch (InvalidOperationException ex) { return Results.Problem(ex.Message, statusCode: StatusCodes.Status409Conflict); }
});
app.MapPost("/rag/ingest", async (RagService rag, CancellationToken ct) =>
{
    try { return Results.Ok(await rag.IngestAsync(ct)); }
    catch (Exception ex) when (IsProviderFailure(ex)) { return Results.Problem("Ollama is unavailable. Run `ollama serve` and pull the configured models.", statusCode: StatusCodes.Status503ServiceUnavailable); }
});
app.MapPost("/rag/ask", async (AskRequest request, RagService rag, CancellationToken ct) =>
{
    try { return Results.Ok(await rag.AskAsync(request.Question, request.TopK, ct)); }
    catch (InvalidOperationException ex) { return Results.Problem(ex.Message, statusCode: StatusCodes.Status409Conflict); }
    catch (Exception ex) when (IsProviderFailure(ex)) { return Results.Problem("Ollama is unavailable. Run `ollama serve` and pull the configured models.", statusCode: StatusCodes.Status503ServiceUnavailable); }
});
app.MapPost("/evaluation/run", async (RagEvaluationService evaluation, CancellationToken ct) =>
{
    try { return Results.Ok(await evaluation.RunAsync(ct)); }
    catch (InvalidOperationException ex) { return Results.Problem(ex.Message, statusCode: StatusCodes.Status409Conflict); }
    catch (Exception ex) when (IsProviderFailure(ex)) { return Results.Problem("Ollama is unavailable. Run `ollama serve` and pull the configured models.", statusCode: StatusCodes.Status503ServiceUnavailable); }
});
app.Run();

static bool IsProviderFailure(Exception exception) => exception is HttpRequestException || exception.InnerException is HttpRequestException;
public sealed record AskRequest(string Question, int? TopK);
public partial class Program { }
