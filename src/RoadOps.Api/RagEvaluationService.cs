using System.Text.Json;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.AI.Evaluation;
using Microsoft.Extensions.AI.Evaluation.Quality;
using RoadOps.Rag;

public sealed record GoldenQuestion(string Question, string ExpectedDocument, string Reference);
public sealed record DeterministicRetrievalMetrics(bool ExpectedDocumentFound, double HitAtK, double MeanReciprocalRank);
public sealed record GoldenEvaluationResult(string Question, string ExpectedDocument, DeterministicRetrievalMetrics DeterministicRetrieval, IReadOnlyDictionary<string, EvaluationMetric> LlmJudge);

public sealed class RagEvaluationService(RagService rag, IChatClient judge, RagOptions options)
{
    public async Task<IReadOnlyList<GoldenEvaluationResult>> RunAsync(CancellationToken cancellationToken)
    {
        if (!rag.IsIngested) throw new InvalidOperationException("RAG index is empty. Call POST /rag/ingest first.");
        var path = Path.Combine(options.DocumentsPath, "evaluation-golden.json");
        var questions = JsonSerializer.Deserialize<List<GoldenQuestion>>(await File.ReadAllTextAsync(path, cancellationToken)) ?? [];
        var configuration = new ChatConfiguration(judge);
        var results = new List<GoldenEvaluationResult>();
        foreach (var item in questions)
        {
            var answer = await rag.AskAsync(item.Question, cancellationToken: cancellationToken);
            // The reference is explicit evaluator context, rather than context supplied to the answering model.
            var messages = new[] { new ChatMessage(ChatRole.System, $"Golden reference answer: {item.Reference}"), new ChatMessage(ChatRole.User, item.Question) };
            var response = new ChatResponse(new ChatMessage(ChatRole.Assistant, answer.Answer));
            var context = string.Join("\n", answer.Sources.Select(x => x.Text));
            var contexts = new EvaluationContext[] { new RetrievalEvaluatorContext(answer.Sources.Select(x => x.Text)), new GroundednessEvaluatorContext(context), new CompletenessEvaluatorContext(item.Reference) };
            // These quality evaluators call the configured judge model and return 1–5 ratings.
            var retrieval = await new RetrievalEvaluator().EvaluateAsync(messages, response, configuration, contexts, cancellationToken);
            var relevance = await new RelevanceEvaluator().EvaluateAsync(messages, response, configuration, contexts, cancellationToken);
            var groundedness = await new GroundednessEvaluator().EvaluateAsync(messages, response, configuration, contexts, cancellationToken);
            var rank = answer.Sources.Select((source, index) => new { source, index }).FirstOrDefault(x => x.source.Document == item.ExpectedDocument)?.index;
            var deterministic = new DeterministicRetrievalMetrics(rank is not null, rank is null ? 0 : 1, rank is null ? 0 : 1d / (rank.Value + 1));
            var judgeMetrics = retrieval.Metrics.Concat(relevance.Metrics).Concat(groundedness.Metrics).ToDictionary(x => x.Key, x => x.Value);
            results.Add(new(item.Question, item.ExpectedDocument, deterministic, judgeMetrics));
        }
        return results;
    }
}
