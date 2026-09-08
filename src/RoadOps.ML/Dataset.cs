using System.Globalization;

namespace RoadOps.ML;

public static class TelemetryDataset
{
    public static IReadOnlyList<LabeledTelemetry> LoadOrCreate(string path, int count = 360, int seed = 20260908)
    {
        if (!File.Exists(path))
        {
            Directory.CreateDirectory(Path.GetDirectoryName(path)!);
            var random = new Random(seed);
            var rows = Enumerable.Range(0, count).Select(_ => CreateRow(random)).ToArray();
            File.WriteAllLines(path, ["vehicleSpeed,ambientTemperature,rainfall,roadTemperature,vibration,label", .. rows.Select(ToCsv)]);
        }
        return File.ReadLines(path).Skip(1).Select(Parse).ToArray();
    }

    public static (IReadOnlyList<LabeledTelemetry> Train, IReadOnlyList<LabeledTelemetry> Test) Split(IReadOnlyList<LabeledTelemetry> rows, float testRatio = .2f, int seed = 17)
    {
        var random = new Random(seed);
        var train = new List<LabeledTelemetry>(); var test = new List<LabeledTelemetry>();
        foreach (var group in rows.GroupBy(x => x.Label))
        {
            var shuffled = group.OrderBy(_ => random.Next()).ToArray();
            var testCount = Math.Max(1, (int)Math.Round(shuffled.Length * testRatio));
            test.AddRange(shuffled[..testCount]); train.AddRange(shuffled[testCount..]);
        }
        return (train, test);
    }

    private static LabeledTelemetry CreateRow(Random r)
    {
        var severity = r.NextDouble();
        var rain = severity > .72 ? 14 + (float)r.NextDouble() * 28 : (float)r.NextDouble() * 12;
        var ambient = -12 + (float)r.NextDouble() * 38;
        var road = ambient - 4 + (float)r.NextDouble() * 8;
        var vibration = severity > .80 ? 5 + (float)r.NextDouble() * 8 : .2f + (float)r.NextDouble() * 5;
        var speed = severity > .80 ? 20 + (float)r.NextDouble() * 35 : 35 + (float)r.NextDouble() * 65;
        var score = rain * .08f + vibration * .8f + (road < 0 ? 1.8f : 0) + (speed > 85 && rain > 7 ? 1.5f : 0);
        var label = score >= 8 ? RoadCondition.Critical : score >= 4.2f ? RoadCondition.Warning : RoadCondition.Normal;
        return new(new(speed, ambient, rain, road, vibration), label);
    }

    private static string ToCsv(LabeledTelemetry x) => string.Join(',', new[] { x.Telemetry.VehicleSpeed, x.Telemetry.AmbientTemperature, x.Telemetry.Rainfall, x.Telemetry.RoadTemperature, x.Telemetry.Vibration }.Select(v => v.ToString(CultureInfo.InvariantCulture)).Append(x.Label.ToString()));
    private static LabeledTelemetry Parse(string line)
    {
        var p = line.Split(',');
        return new(new(float.Parse(p[0], CultureInfo.InvariantCulture), float.Parse(p[1], CultureInfo.InvariantCulture), float.Parse(p[2], CultureInfo.InvariantCulture), float.Parse(p[3], CultureInfo.InvariantCulture), float.Parse(p[4], CultureInfo.InvariantCulture)), Enum.Parse<RoadCondition>(p[5]));
    }
}
