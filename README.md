# Attention paper learning project

## Tasks

- Data Generator: (/gen/cmd/gen) Create thousands of random short lists of integers
- Embedding Layer: Map each integer to a 32 or 64-dimensional vector
- Positional Encoding: Implement the sin and cos formulas to give those numbers a "place" in the list.
- Single-Head attention: Start with one "head" to see the raw math before moving to "Multi-Head".

## Architecture

### Intent Model

```mermaid
flowchart LR
    A["Prompt text"] --> B["Tokenize + normalize"]
    B --> C["Feature hashing<br/>word unigrams<br/>word bigrams<br/>char 3-5 grams"]
    C --> D["Sparse feature vector<br/>8192 dims"]
    D --> E["Linear layer<br/>W1 + B1"]
    E --> F["ReLU<br/>128 hidden units"]
    F --> G["Linear layer<br/>W2 + B2"]
    G --> H["Softmax"]
    H --> I["Intent label<br/>SortAsc | SortDesc | Sum"]
```

The intent classifier is a small hashed-feature MLP defined in `internal/intent/model.go`.

### Embed / Generator Model

```mermaid
flowchart LR
    A["Input prompt tokens"] --> B["Embedding lookup"]
    B --> C["Sequence matrix"]
    C --> D["Positional encoding"]
    D --> E["Transformer block"]

    E --> E1["Multi-head attention<br/>WQ WK WV per head<br/>concat + WO"]
    E1 --> E2["Feed-forward<br/>W1 -> W2"]

    E2 --> F["Output projection<br/>cW"]
    F --> G["Token logits"]
    G --> H["Softmax / argmax"]
    H --> I["Generated target tokens"]
```

The generator path uses an embedding layer, one transformer block, and an output projection.

### Overall Architecture

```mermaid
flowchart LR
    A["User prompt"] --> B["Extract numbers"]
    A --> C["Intent classifier"]
    C --> D["Task + order"]
    B --> E["Structured encoder"]
    D --> E
    E --> F["Embedding + transformer + output projection"]
    F --> G["Predicted output"]
```

Today the project uses two separate learned components:

- `intent`: a lightweight classifier that routes prompts into `sort asc`, `sort desc`, or `sum`
- `embed`: the sequence model that produces the target output text
