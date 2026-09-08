namespace RoadOps.ML;

public sealed record ClassMetrics(float Precision, float Recall, float F1, int Support);
public sealed record ClassificationMetrics(float Accuracy, IReadOnlyDictionary<RoadCondition, ClassMetrics> PerClass, int[][] ConfusionMatrix);

public static class Metrics
{
    public static ClassificationMetrics Calculate(IReadOnlyList<RoadCondition> actual, IReadOnlyList<RoadCondition> predicted)
    {
        var matrix = new int[3, 3];
        for (var i = 0; i < actual.Count; i++) matrix[(int)actual[i], (int)predicted[i]]++;
        var perClass = Enum.GetValues<RoadCondition>().ToDictionary(c => c, c =>
        {
            var i = (int)c; var tp = matrix[i, i]; var fp = Enumerable.Range(0, 3).Sum(row => matrix[row, i]) - tp; var fn = Enumerable.Range(0, 3).Sum(col => matrix[i, col]) - tp;
            var precision = tp + fp == 0 ? 0 : (float)tp / (tp + fp); var recall = tp + fn == 0 ? 0 : (float)tp / (tp + fn);
            return new ClassMetrics(precision, recall, precision + recall == 0 ? 0 : 2 * precision * recall / (precision + recall), Enumerable.Range(0, 3).Sum(col => matrix[i, col]));
        });
        return new(actual.Where((x, i) => x == predicted[i]).Count() / (float)actual.Count, perClass, Enumerable.Range(0, 3).Select(row => Enumerable.Range(0, 3).Select(column => matrix[row, column]).ToArray()).ToArray());
    }
}
