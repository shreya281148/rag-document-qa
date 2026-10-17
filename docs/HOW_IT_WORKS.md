# How this project works: a guided tour

Read this once, then read the code in the order below with this page open next to it. Aim to be able to explain every file in your own words.

## 1. The problem RAG solves

A language model only knows what it saw in training, and it will confidently make things up when asked about something it never saw, such as *your* documents. There are two ways to give it your knowledge:

- **Fine-tuning** changes the model's weights. It is slow, needs a lot of data and compute, goes stale as documents change, and still can't tell you *where* an answer came from.
- **Retrieval-Augmented Generation (RAG)** leaves the model alone. At question time it **searches your documents**, picks the few most relevant passages, and puts them into the prompt. The model then answers *from that text*.

RAG wins here because new documents only need indexing (no retraining), answers can cite their sources, and the model has far less room to invent things.

## 2. The two phases

**Ingest (once per document):**
file → text with page numbers → chunks → embedding vectors → stored in a vector database.

**Ask (every question):**
question → embedding vector → nearest chunks → prompt → local language model → answer with citations.

## 3. Read the code in this order

| Step | File | What to understand |
| --- | --- | --- |
| 1 | `config.py`, `config.yaml` | All tunable settings live in one place; environment variables can override them. |
| 2 | `loaders.py` | Text comes out *with* its file name and page number, so every answer can be traced back. `clean_text` fixes PDF whitespace quirks. |
| 3 | `chunking.py` | Why chunks exist, how size and overlap trade off, and why ids are deterministic (so re-indexing never duplicates). |
| 4 | `embeddings.py` | An embedding turns text into numbers so that similar *meaning* means nearby vectors. Note the two interchangeable embedders. |
| 5 | `vectorstore.py` | ChromaDB stores vectors and finds the nearest ones. Cosine distance is converted to a similarity score. |
| 6 | `prompts.py` | How the retrieved text and the rules are presented to the model. |
| 7 | `llm.py` | The pluggable model layer: one interface, several backends. |
| 8 | `pipeline.py` | The whole system in about 100 lines. If you can explain this file, you can explain the project. |
| 9 | `evaluate.py` | How retrieval quality is measured. |
| 10 | `app.py`, `cli.py` | The thin user-facing layers on top of the pipeline. |

## 4. Core ideas, in plain words

**Embedding.** A model that maps text to a list of numbers (here, 768 of them for `nomic-embed-text`) so that texts with similar meaning get similar numbers. "How many vacation days?" lands near "Employees receive 28 days of annual leave" even though few words match.

**Cosine similarity.** A score for how closely two vectors point in the same direction: 1.0 = identical direction, 0 = unrelated, negative = opposite. It is what "nearest chunk" means.

**Chunk size and overlap.** The embedding of a chunk represents *one* meaning. Huge chunks blur many meanings together; tiny chunks lose the context needed to answer. Overlap repeats a little text between neighbours so a sentence on the boundary survives in at least one chunk.

**top-k.** How many chunks go into the prompt. Too few can miss the answer; too many dilute the prompt with noise and use up the context window.

**Similarity floor (`min_score`).** If even the best chunk scores below the floor, the app says "not found" *without calling the model*, so the model has no chance to guess. Important limit: with a weak embedder the scores of good and bad matches can overlap (the evaluation prints this), so a floor alone cannot be perfect.

**Grounding prompt.** The prompt tells the model to use only the numbered passages, gives it an explicit way out ("I could not find this..."), and asks for citations like [1]. Models guess less when refusing is an allowed answer.

**Temperature.** Randomness of the model's word choice. 0.1 keeps answers factual and repeatable.

**Context window (`num_ctx`).** The maximum text (in tokens) the model can read per request. The retrieved passages plus the question must fit, or the beginning gets cut off silently. Ollama's default is small, so this project requests 4096 explicitly.

**Quantisation.** Storing model weights with fewer bits (for example 4-bit instead of 16-bit) to cut memory use and speed up inference at a small cost in quality. It is why a 3B model fits on a laptop.

**Why a pluggable LLM layer.** `llm.py` defines two methods (`generate`, `stream`). Ollama, an OpenAI-compatible server and the no-model fallback all implement them, so the pipeline never changes when the model does. This is also what makes the code testable without a real model.

## 5. Experiments to run (this is how you really learn it)

1. **Chunk size.** Run `python -m ragqa.evaluate --offline`, then change `chunking.chunk_size` to 200 and then 2000 (re-run each time). Which metric moves, and why?
2. **top-k.** Compare `-k 1`, `-k 4`, `-k 8`. What happens to hit@k and to answer quality with a real model?
3. **Similarity floor.** `ragqa ask "What is the dress code?" --show-scores`. See the scores of an unanswerable question. Then try your own documents and find a floor that works.
4. **Swap models.** With Ollama running, compare `qwen2.5:3b` and `qwen2.5:7b` on the same questions. Which is more accurate, and how much slower?
5. **Break it on purpose.** Remove the "answer only from context" rule in `prompts.py` and ask something not in the documents. Does the model start inventing answers? (Then run `pytest` and see which tests notice.)
6. **Your own documents.** Index a few real PDFs of yours and write 10 questions with known answers into a new JSONL file, then evaluate.

## 6. Interview questions and answers

**What is RAG, in two sentences?**
It retrieves the most relevant passages from your own documents and gives them to a language model as context, so answers are based on your data instead of the model's memory. This reduces hallucinations and lets answers cite sources, without retraining.

**Why RAG instead of fine-tuning?**
Fine-tuning is expensive, goes stale when documents change, and can't point to sources. RAG only needs new documents indexed, and every answer can be traced to passages.

**Why did you run the model locally?**
Documents can be confidential, and online AI services may store or train on what you send. Running Ollama, the embeddings and the vector database on your own machine means the data never leaves it. The trade-off is that small local models are less capable than the largest hosted ones.

**Why Qwen 2.5?**
It is a strong open model family for its size, handles structured and technical text well, and supports long context. It also comes in several sizes, so the same code runs the 3B version on a laptop or the 7B version on better hardware. The model is one config line, so I can compare alternatives.

**How do you reduce hallucinations?**
Four things: retrieved context only, a prompt that forbids outside knowledge and allows "not found", a similarity floor that skips the model when nothing relevant is found, and citations so every claim can be checked. I also keep the temperature low.

**How did you choose chunk size and overlap?**
I started from about 800 characters with 15% overlap, a common baseline, and tested variations with the evaluation script instead of guessing. The right values depend on the documents.

**How do you evaluate it?**
I evaluate retrieval separately from generation, because if the right passage never reaches the model, no model can answer correctly. I measure hit rate at k and mean reciprocal rank on questions with known answers. For generation I check answers against reference phrases and review them by hand. I also report when the similarity scores of answerable and unanswerable questions overlap, because then no single threshold can be perfect.

**What are the weaknesses?**
No OCR for scanned PDFs, tables lose structure, small models can misread passages, answers that span many documents are hard, and a single similarity threshold cannot separate good from bad matches perfectly. In my evaluation the offline embedder showed exactly that overlap.

**What would you improve next?**
Hybrid search (keyword plus vector), a re-ranker, heading-aware chunking, OCR, an LLM-as-judge evaluation, and conversation memory.

**What is the context window, and why does it matter here?**
It is the maximum text the model can read at once. Retrieved passages and the question must fit; otherwise the start of the prompt is truncated silently. That is why the config sets `num_ctx` explicitly.

**How is the project tested without a model?**
Every backend sits behind a small interface, and the Ollama client is tested against a fake HTTP server that mimics the Ollama API. That verifies request format, options, streaming and error handling. I also wrote a few intentional bugs to check that the tests catch them. Real answer quality still has to be judged with a real model.

**What was the hardest bug or surprise?**
Pick a true one from your own runs. One real example from building this: a similarity threshold that worked for one embedder blocked every answer with another, because similarity scales differ between embedders. The fix was an `auto` default per embedder plus an evaluation report that shows the score ranges.

## 7. How to talk about this project honestly

- It is an **independent demo** that you studied and can explain. Say that it was built with AI assistance and that you worked through the code, ran the experiments above, and can modify it.
- It is **architecturally similar** to what you worked on during your internship (local LLM, chunking, embeddings, retrieval), but it is **not** that code. The internship code belongs to the company and the college project to the university; this repository contains neither.
- Only claim results you have reproduced yourself. After you run it with real Qwen on your computer, note what you measured. Interviewers value honest numbers and honest limitations far more than impressive claims.
