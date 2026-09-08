Create a production-quality but intentionally small .NET 10 GitHub
repository named RoadOps.AI.

The purpose of the repository is educational: demonstrate a complete
AI engineering lifecycle for road infrastructure while keeping every
component small enough to understand.

Architecture:

1. RoadOps.ML
   Implement a small multi-class neural network using TorchSharp.

Input telemetry features:

- vehicle speed
- ambient temperature
- rainfall
- road temperature
- vibration

Output:

- Normal
- Warning
- Critical

Generate a deterministic synthetic CSV dataset if no dataset exists.

Implement:

- train/test split
- normalization
- TorchSharp nn.Module
- training loop
- CrossEntropyLoss
- Adam optimizer
- model.train()
- model.eval()
- torch.no_grad()
- accuracy
- per-class precision
- recall
- F1
- confusion matrix
- save/load model weights

Keep the neural network intentionally simple and heavily comment the
ML-specific code explaining the PyTorch/TorchSharp concepts.

2. RoadOps.Rag

Create five small realistic Markdown documents related to:

- winter road maintenance
- road inspection
- incident response
- fleet procedures
- road sensor maintenance

Use the current Microsoft .NET AI ecosystem.

Use:

- Microsoft.Extensions.AI
- Microsoft.Extensions.DataIngestion
- Microsoft.Extensions.VectorData
- CommunityToolkit.VectorData.InMemory
- Microsoft.ML.Tokenizers

Implement:
document -> token based chunking -> embeddings -> vector storage ->
Top-K semantic retrieval -> context -> LLM response.

Default chunk size: 300 tokens.
Default overlap: 50 tokens.
Make these configurable.

Use interfaces IChatClient and IEmbeddingGenerator so the application
is provider-independent.

Support local Ollama through OllamaSharp.
Do not couple application logic directly to Ollama.

3. Evaluation

Create a small golden evaluation dataset with at least five questions.

Use Microsoft.Extensions.AI.Evaluation where practical to demonstrate:

- Retrieval evaluation
- Relevance
- Groundedness

Also demonstrate deterministic ML evaluation separately from
generative-AI evaluation.

4. RoadOps.Api

Create a Minimal API exposing:

POST /ml/train
POST /ml/predict
POST /rag/ingest
POST /rag/ask
POST /evaluation/run

5. Documentation

README.md must explain:

- architecture diagram using Mermaid
- difference between training and inference
- precision vs recall vs F1
- what embeddings are
- why document chunking is required
- chunk size / overlap tradeoffs
- semantic retrieval
- RAG
- groundedness
- LLM-as-a-judge
- why IChatClient and IEmbeddingGenerator abstractions are useful
- what would need to change for production

Include exact dotnet and ollama commands required to run the project.

Important:
Keep the project small.
Prefer readable code over abstractions.
Do not implement unnecessary enterprise patterns.
Do not fake implementations.
Everything committed must build.
Add tests for core deterministic functionality.

Before finishing:
dotnet restore
dotnet build
dotnet test

Fix all compilation/test failures.
