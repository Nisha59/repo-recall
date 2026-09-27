# Structure explained

```
repo-recall/
├── .env                # your existing GITHUB_TOKEN — gitignored, untouched
├── .env.example         # documents every config var, no secrets
├── .gitignore
├── requirements.txt
├── README.md
├── CLAUDE.md
├── data/
│   ├── raw/             # gitignored — cloned docs, raw fetched issues land here
│   └── processed/       # gitignored — cleaned/chunked data ready to embed
├── src/repo_recall/     # the installable package (src-layout, not flat)
│   ├── config.py         # placeholder — will hold pydantic-settings Settings class
│   ├── collection/       # clone docs repo, fetch GitHub issues+comments via API
│   ├── processing/       # clean raw text, chunk it for embedding
│   ├── indexing/         # embed chunks + a common vector-store interface
│   │   └── backends/     # postgres+pgvector impl and lancedb impl live here later
│   ├── retrieval/        # embed a query, do top-k similarity search
│   ├── generation/       # build the RAG prompt, call the LLM, attach citations
│   ├── api/               # FastAPI app + Gradio UI
│   └── evaluation/       # test-question harness, scores retrieval + answers
└── tests/                # pytest, will mirror the src/repo_recall layout
```

**Why `src/repo_recall/` instead of a flat layout:** it's the standard convention for an installable Python package — keeps imports clean (`from repo_recall.retrieval import ...`) and prevents accidentally importing from the working directory instead of the installed package, which is a common source of subtle bugs.

**Why each stage is its own subpackage:** `retrieval/` never touches the LLM, `generation/` never does similarity search — each has one job, so you can unit-test and explain them independently, and swap pieces (e.g. vector backend) without touching the others.

**One deviation from the original tech-stack list:** `requests` and `PyGithub` were added to `requirements.txt` — not explicitly named in the stack bullets, but the `collection/` folder can't fetch GitHub issues without a GitHub client, so they were included now rather than leaving that gap.

Everything else — `config.py`, all `__init__.py` files — is docstring-only, no logic, per the "don't implement features yet" instruction. `requirements.txt` is pinned to the versions actually resolved during install (not guessed).

In Step 2, 

you're collecting the Kubernetes documentation and turning it into clean, organized data that your assistant can search later. Think of it like building a library: before anyone can find a book, you need to bring the books in, remove the packaging, and label each one.

What happens, piece by piece

1. Download the docs.
The script copies the kubernetes/website repository to your computer. Now you have thousands of documentation pages as Markdown files (.md), which are plain text files with simple formatting like # for headings.

2. Read each page.
The script goes through the content/en/docs folder and opens every Markdown file one by one.

3. Clean the page.
The raw files contain extra things meant for the website builder, not for readers. There are two main kinds:

Front matter: a block at the top of the file with settings like the title and page order, for example:
  ---
  title: Pods
  weight: 10
  ---

The script reads the title from here, then removes the block from the text.

Hugo shortcodes: special tags the website uses to insert things like notes, tabs, or code examples, such as {{< note >}} or {{< glossary_tooltip ... >}}. Your assistant doesn't need these tags, so the script removes them while keeping any useful text inside them.

After cleaning, you're left with just the readable content.

4. Label each page.
For every page, the script saves four things:

title: so you know what the page is about
text: the cleaned content
path: where the file is in the repository
URL: the matching page on kubernetes.io, so your assistant can later show a link like "Source: kubernetes.io/docs/concepts/workloads/pods/"

5. Save everything into one file.
All pages go into data/processed/docs.jsonl. JSONL means "one JSON record per line," so each line is one documentation page, which makes the file easy to read and process in the next steps.

A line might look like this:

{"title": "Pods", "text": "Pods are the smallest deployable units of computing...", "path": "concepts/workloads/pods/_index.md", "url": "https://kubernetes.io/docs/concepts/workloads/pods/"}

6. Check the results.
The script prints how many pages it processed and skipped, and the tests make sure the cleaning works correctly, for example that shortcodes are really removed.

Why this step matters

Everything later depends on this data. If the text is messy, search results will be worse. If URLs are wrong, citations won't work. Clean data is the foundation of good RAG, and interviewers know that, so you can explain it as: "I built an ingestion pipeline that parses the Markdown docs, strips front matter and Hugo shortcodes, and maps each file to its public URL for citations."

## What was actually built (Step 2 done)

In plain words, here's what got added to the project:

1. **Made the project installable.** Added a `pyproject.toml` so Python actually recognizes `src/repo_recall` as a package you can import and run — without this, none of the code below could be executed.

2. **Turned on settings.** `config.py` now reads all the configurable bits (which repo to clone, where docs live inside it, the public URL to build citation links, where to save files) from your `.env` file, with sensible Kubernetes defaults if you don't set them. This is what makes the whole thing reusable for a different project later — just change the `.env` values.

3. **Wrote the "cleaning" functions** (`processing/markdown.py`) — small, testable functions that each do one job:
   - pull the title/settings out of the `---` front matter block at the top of a file
   - strip out Hugo's special tags (`{{< note >}}`, `{{< glossary_tooltip ... >}}`, etc.), keeping any real words they were displaying
   - remove leftover `<!-- overview -->` style HTML comments (found these while checking the real output — they're template markers, not content)
   - tidy up extra blank lines
   - turn a file's path into its real kubernetes.io URL
   - fall back to a readable title from the filename if a page has none

4. **Wrote the downloader/runner** (`collection/repo.py` + `collection/docs.py`) — clones the `kubernetes/website` repo (shallow, so it's fast and small), walks every `.md` file under `content/en/docs`, runs each one through the cleaning functions, and writes one line of JSON per page into `data/processed/docs.jsonl`. It logs how many pages it kept vs. skipped (skipped = pages with nothing but template junk, no real content).

5. **Added 12 unit tests** for the cleaning functions specifically (not the download part, since that needs the internet) — covering front matter parsing, shortcode removal, HTML comment removal, URL building, and title fallback. All pass.

6. **Ran it for real.** It cloned the docs, processed **1,650 pages**, skipped 69 empty ones, and produced `data/processed/docs.jsonl`. Spot-checked the "Pods" page and the text reads clean and correct.

Along the way, the first version of the shortcode-cleaner had a bug: it deleted a shortcode's text entirely instead of keeping the word it was meant to display (so "a group of one or more containers" was coming out as "a group of one or more, with shared storage" — missing "containers"). Caught this by actually reading the output instead of trusting the tests alone, and fixed it before calling the step done.

 Results

  - First run (--limit 50): wrote 50 issues and skipped 322 pull requests. The issues endpoint is mostly PRs in this repo.
  - Second run (--limit 10): resumed from the saved page, skipped 13 issues it already had, and wrote 10 new ones. The file now has 60 lines with 60 unique
    issue numbers, so there are no duplicates.
  - Rate limit: 4,933 requests left this hour after both runs.

  What was added

  - collection/github_client.py: a small GitHubClient built on requests:
    - It slows down before the hourly request limit runs out.
    - When GitHub says the limit is hit, it waits for the reset time or the Retry-After header.
    - It retries server errors (500/502/503/504) and network errors up to 5 times, waiting longer each time.
    - Any other error stops the run straight away.
  - collection/issues.py: the command-line script, run as python -m repo_recall.collection.issues [--limit N] [--since YYYY-MM-DD].
    - Output goes to data/raw/issues/kubernetes__kubernetes.jsonl, with the resume point saved in a .state.json file next to it.
    - Each issue is written as soon as its comments are fetched. A half-written last line from a crash gets removed on the next run.
    - --since means "closed on or after this date".
    - --limit counts new issues written in this run, not the total in the file.
  - config.py: added github_repo and github_bot_logins.

  Things to know

  - Bot filtering: Kubernetes' bots (k8s-ci-robot, k8s-triage-robot, etc.) are ordinary user accounts, so GitHub's bot flag misses them. They're listed in
    config instead. Another repo would need its own list.
  - Extra field: I added state_reason (completed / not_planned) to each record. It could be used later to rank "won't fix" issues lower.
  - Rate-limit count: GitHub's /rate_limit endpoint reported 5000 remaining after about 55 requests. I now read the count from the X-RateLimit-Remaining header
    on each response instead, which is accurate.
  - Starting over: there's no restart flag. To begin fresh, delete the .jsonl and .state.json files.

  Nothing is committed yet.

   Plan: chunk docs and issues into data/processed/chunks.jsonl

 Context

 Collection is done: data/processed/docs.jsonl holds 1,650 cleaned doc pages (title, text, path, url), and data/raw/issues/kubernetes__kubernetes.jsonl holds
 closed issues (number, title, body, comments[{author, body, created_at}], labels, created_at, closed_at, state_reason, url). The next step is to turn both
 into retrieval-sized chunks with metadata, ready to embed. Decision: each doc chunk's date is the date of the downloaded docs snapshot (the HEAD commit
 date).

 Key constraint that shapes sizes

 BAAI/bge-small-en-v1.5 has a 512-token max input, and anything past that is silently cut off. So token counts use the embedding model's own tokenizer, and
 the size limit applies to the whole embedded text, including the heading-path prefix. Defaults: chunk_max_tokens=480 (leaves room for the special tokens),
 chunk_overlap_tokens=64, chunk_min_tokens=40, and issue_max_comments=5, all configurable in .env.

 Files

 Modify
 - src/repo_recall/config.py: add the 4 settings above; document them in .env.example.
 - src/repo_recall/collection/repo.py: add head_commit_date(dest) -> str (git log -1 --format=%cI).
 - src/repo_recall/collection/docs.py: add "date" to each record (the snapshot date is passed into build_doc_record). Re-run python -m
   repo_recall.collection.docs to regenerate docs.jsonl.

 New (all in src/repo_recall/processing/)
 - tokens.py: a TokenCounter Protocol (count(text) -> int) plus HFTokenCounter (loads the AutoTokenizer for settings.embedding_model lazily, counts without
   special tokens). Tests use a fake whitespace counter, following the CLAUDE.md "swap behind an interface" rule.
 - chunk_model.py: a Chunk dataclass with chunk_id, source_type, title, url, date, text, heading_path, labels, token_count, extra and to_dict().
 - text_units.py: shared splitting and packing:
   - split_blocks(markdown) -> list[str]: paragraphs split on blank lines, never breaking inside a ``` fenced code block.
   - split_oversized(unit, counter, max_tokens) -> list[str]: split by lines, then by words, only when a single block is too big.
   - pack_units(units, counter, max_tokens, overlap_tokens) -> list[str]: greedily fill chunks; each new chunk starts with the trailing whole units of the
     previous chunk, up to overlap_tokens.
 - doc_chunker.py:
   - split_sections(text) -> list[Section(headings: tuple[str,...], body)]: tracks #–###### levels, ignores # lines inside code fences (bash comments).
   - build_breadcrumbs(records) -> dict[path, list[str]]: gets parent titles from the _index.md records already in docs.jsonl, e.g.
     concepts/workloads/pods/_index.md → ["Concepts", "Workloads", "Pods"].
   - merge_tiny_sections(...): a section under chunk_min_tokens is merged forward into the next section. Its heading stays as a line in the text, and the
     chunk's heading path is the two sections' shared prefix.
   - chunk_doc(record, breadcrumb, settings, counter) -> list[Chunk]: heading_path = breadcrumb + page title + in-page headings, e.g. Concepts > Workloads >
     Pods > Pod lifecycle > Container states. Text embedded = "{heading_path}\n\n{body}". chunk_id = doc:{path}#{i}.
 - issue_chunker.py:
   - clean_issue_body(text): strip <!-- template instructions --> and empty template sections (### Anything else…\n\n_No response_).
   - is_low_value_comment(body): drops very short comments and "+1 / same here / any update? / /remind" style comments (bots and slash commands were already
     removed at collection time).
   - score_comment(comment, position, issue_author): signals are a code block or stack trace, links to a PR or commit, resolution words (fixed, root cause,
     workaround, duplicate of, resolved by), and length (capped). The last substantive comment is always kept because it usually holds the resolution.
   - select_comments(...): top issue_max_comments by score, put back in chronological order so the thread still reads as a conversation.
   - chunk_issue(record, settings, counter) -> list[Chunk]: the units are cleaned body blocks, then each selected comment as "Comment by {author}:\n…" (split
     if oversized), packed with pack_units. Every chunk starts with the header Issue #N: {title}\nLabels: … (the header counts toward the limit). date =
     closed_at, labels set, extra = {number, state_reason, created_at}. chunk_id = issue:{repo}#{N}:{i}.
 - chunk_stats.py: compute_stats(chunks) gives the count per source_type plus token-length min/p25/median/p75/p90/max/mean per type and a bucket histogram
   (0–100, 100–200, … 400–480, over the limit). format_stats(...) -> str.
 - chunk_all.py: the CLI python -m repo_recall.processing.chunk_all. It reads docs.jsonl and the issues jsonl for the configured repo (reusing output_paths
   from collection/issues.py), writes data/processed/chunks.jsonl, logs progress, and prints the stats table at the end (final CLI output, so print is allowed
   under CLAUDE.md).

 Tests (tests/processing/, fake whitespace TokenCounter)

 - test_text_units.py: code fence kept whole; packing respects max; overlap repeats trailing units; an oversized unit gets split.
 - test_doc_chunker.py: heading path nesting and reset when going back up a level; # inside a code fence is not a heading; breadcrumb from _index titles; long
   section → several chunks, each with the heading prefix and ≤ max; tiny section merged; chunk_id/metadata fields.
 - test_issue_chunker.py: template comments and _No response_ stripped; "+1"/"any update?" dropped; last comment always kept; top-K kept in chronological
   order; long issue → multiple chunks, each with the Issue #N header and labels/date set.
 - test_chunk_stats.py: counts per type and percentiles on a known input.
 - tests/collection/: add a test that build_doc_record includes date.

 Why these rules (to include in the final explanation)

 - Heading-first splitting: doc sections are units the authors chose, so a chunk answers one sub-topic instead of spanning two.
 - Heading path in the text: a section body like "Set restartPolicy…" is ambiguous on its own. The breadcrumb gives the embedding (and later the LLM) the
   context, and makes citations readable.
 - About 480 tokens, measured with the model's own tokenizer: the limit sits just under bge-small's 512-token cutoff so nothing is cut off, and the chunk is
   still big enough to hold a full explanation plus its YAML example.
 - Overlap in whole blocks: only affects the splits inside long sections. Repeating whole paragraphs means a split point never cuts an idea in half, without
   the wasted tokens that repeating fixed windows costs.
 - Code fences are never split unless a single block is larger than the limit, because half a manifest is useless.
 - Tiny sections are merged, because a 10-token chunk ("See also: …") produces noisy, contextless matches.
 - Issues are packed around the problem and its resolution: the value of an issue is the question plus how it was answered. Filtering "+1" comments and
   keeping the last substantive comment keeps the signal. Repeating the title and labels in every split chunk means any piece still says which problem it
   belongs to.

 Verification

 1. source .venv/Scripts/activate && pytest tests/ -q: new and existing tests pass.
 2. Re-run python -m repo_recall.collection.docs and confirm the records have date.
 3. Run python -m repo_recall.processing.chunk_all and check the printed stats. No chunk should be over 480 tokens (the "over" bucket is 0).
 4. Spot-check chunks.jsonl: the Pods page chunks show Concepts > Workloads > Pods > …, a long issue's chunks each start with Issue #N, and code blocks are
    intact.


 Chunking is done. python -m repo_recall.processing.chunk_all writes data/processed/chunks.jsonl and prints the stats, and all 61 tests pass (the 26 new tests
  plus the existing ones).

  Results on the real data

  ┌───────┬────────┬──────┬────────┬─────┬─────┐
  │       │ chunks │ mean │ median │ p90 │ max │
  ├───────┼────────┼──────┼────────┼─────┼─────┤
  │ doc   │ 17,845 │ 284  │ 284    │ 470 │ 480 │
  ├───────┼────────┼──────┼────────┼─────┼─────┤
  │ issue │ 285    │ 385  │ 431    │ 475 │ 480 │
  └───────┴────────┴──────┴────────┴─────┴─────┘

  Lengths are counted in the embedding model's own tokens, and no chunk is over 480. Every chunk has a date: docs get the snapshot date (2026-09-24) and issues
  get their close date. The script also prints a histogram by 100-token band.

  What was added

  - src/repo_recall/processing/:
    - text_units.py: splits text into blocks without breaking code blocks, and packs blocks into chunks with overlap.
    - doc_chunker.py and issue_chunker.py: the chunking rules below.
    - chunk_stats.py and chunk_all.py: the statistics and the command-line script.
    - tokens.py: token counting behind a small interface. The real version uses the bge tokenizer; tests use a fake one.
  - Settings in config.py and .env.example: CHUNK_MAX_TOKENS=480, CHUNK_OVERLAP_TOKENS=64, CHUNK_MIN_TOKENS=40, ISSUE_MAX_COMMENTS=5.
  - Doc records now include date: the date of the downloaded docs version, as you chose.
  - Tests are in tests/processing/ (4 new files) and tests/collection/test_docs.py.

  Why these rules

  - Split docs at headings first. The authors already divided the pages into topics, so each chunk covers one topic instead of the end of one and the start of
    the next.
  - Put the heading path in the text that gets embedded. A section like "Set restartPolicy to…" is ambiguous on its own. Starting it with Concepts > Workloads
    > Pods > Pod Lifecycle > … gives the search the missing context and makes citations readable. The parent names come from each folder's _index.md title.
  - 480 tokens, not 500. bge-small-en-v1.5 silently cuts off anything past 512 tokens. Counting with the model's own tokenizer, and counting the heading path
    as part of the chunk, means the limit is a guarantee rather than an estimate.
  - Overlap by whole paragraphs, and never split code blocks. When a long section is split, the next chunk starts by repeating the previous chunk's last
    paragraph(s), up to 64 tokens, so no idea is cut in half. Half a YAML example is useless, so code blocks are only split if a single block is bigger than
    the whole limit.
  - Merge tiny sections into a neighbour, but only if the result still fits. A 10-token "See also" chunk matches searches for no real reason. Most of the
    roughly 2,000 chunks under 100 tokens are API-reference field entries, which are meant to stand alone.
  - Issues are built around the problem and its answer. Each issue's chunks contain its description (with empty issue-template sections and HTML comments
    removed) and up to 5 comments. Pings like "@x PTAL", "+1" and "any update?" are dropped. The remaining comments are ranked by whether they contain code or
    stack traces, links to PRs or commits, and wording like "fixed", "root cause" or "workaround". The last real comment is always kept because it usually
    holds the resolution, and the kept comments stay in their original order. Every chunk of a split issue starts with Issue #N: title and its labels, so a
    fragment still says which problem it belongs to.

  Two fixes outside the plan

  - collection/repo.py: updating the docs download failed. git pull can't update a one-commit (shallow) copy once the upstream repo moves on, which is an
    existing bug that would have hit any re-run. It now fetches the latest commit and resets onto it. That folder is only a read-only copy of the docs, so
    nothing local is lost.
  - Quieter logs: I turned off a misleading tokenizer warning ("longer than 512"), since the chunker measures long text on purpose before splitting it. I also
    hid the per-request download logs from Hugging Face.

  Nothing is committed yet.  


    The big picture

  Chunking is like cutting a long book into index cards. Each card has to be small enough to read on its own and has a label saying where it came from. To do
  that we need to:

  1. measure how big a piece of text is
  2. decide what a "card" looks like
  3. cut text into card-sized pieces
  4. apply special cutting rules for docs
  5. apply special cutting rules for issues
  6. check the results (the statistics)
  7. run all of the above in order

  Each new file does one of those jobs.

  What each file does

  ┌───────────────────┬───────────────────────────────────────────────────────────────┬────────────────────────────────────────────────────────────────────┐
  │       File        │                        In plain words                         │                           Why we need it                           │
  ├───────────────────┼───────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────┤
  │ markdown.py       │ Cleans raw doc pages: removes website-only tags and gets the  │ Used when the docs were collected, before chunking.                │
  │ (older)           │ title.                                                        │                                                                    │
  ├───────────────────┼───────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────┤
  │ tokens.py         │ The ruler. Counts how many tokens a piece of text is.         │ The search model can only read 512 tokens and silently ignores the │
  │                   │                                                               │  rest, so every card must be measured with the model's own ruler.  │
  ├───────────────────┼───────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────┤
  │ chunk_model.py    │ The card template. Defines what one chunk looks like: id,     │ Docs and issues both produce the same kind of card, so later steps │
  │                   │ text, title, URL, date, labels.                               │  (embedding, search) only have to understand one format.           │
  ├───────────────────┼───────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────┤
  │                   │ The scissors. Splits text into paragraphs (keeping code       │ Docs and issues both need this, so it's written once and shared    │
  │ text_units.py     │ examples whole) and packs paragraphs into cards up to the     │ instead of copied twice.                                           │
  │                   │ size limit, with a little overlap.                            │                                                                    │
  ├───────────────────┼───────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────┤
  │ doc_chunker.py    │ Rules for docs. Cuts pages at their headings and adds the     │ Docs have headings and a folder structure; issues don't.           │
  │                   │ "Concepts > Workloads > Pods" label.                          │                                                                    │
  ├───────────────────┼───────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────┤
  │                   │ Rules for issues. Keeps the title and problem description,    │ Issues are conversations, not structured pages, so they need       │
  │ issue_chunker.py  │ throws away "+1"/"PTAL" comments, keeps the useful ones, and  │ different rules.                                                   │
  │                   │ puts "Issue #N: title" on every card.                         │                                                                    │
  ├───────────────────┼───────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────┤
  │ chunk_stats.py    │ The report card. Counts the cards and shows how long they     │ You asked for statistics. It also shows at a glance whether        │
  │                   │ are.                                                          │ anything went over the size limit.                                 │
  ├───────────────────┼───────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────┤
  │ chunk_all.py      │ The "go" button. Reads the docs and issues, runs everything   │ It's the single command you run: python -m                         │
  │                   │ above, saves chunks.jsonl and prints the stats.               │ repo_recall.processing.chunk_all.                                  │
  └───────────────────┴───────────────────────────────────────────────────────────────┴────────────────────────────────────────────────────────────────────┘

  Why not one big file?

  It would work, but it would be around 600 lines doing seven different jobs. Separate files help in three ways:

  - Easier to test. Each piece is tested on its own, e.g. "does the issue chunker really drop '+1' comments?", without running everything.
  - Easier to change. To switch to a different embedding model, you only touch tokens.py. To change how issues are handled, you only touch issue_chunker.py.
  - Easier to explain. Your CLAUDE.md asks for small, single-purpose pieces that are clear enough to walk through in an interview. "The scissors are shared,
    and docs and issues each have their own rules on top" is a clean story to tell.

  If it still feels like too many files, the easiest merges would be chunk_model.py into text_units.py and chunk_stats.py into chunk_all.py. That would take it
  down to 7 files without mixing up the jobs much.
 

 ● Indexing is finished and everything checks out: all 18,130 chunks are in Postgres and both kinds of search work.

  What I checked

  ┌───────────────────────────────┬────────────────────────────────────────────────────────────────────────┐
  │             Check             │                                 Result                                 │
  ├───────────────────────────────┼────────────────────────────────────────────────────────────────────────┤
  │ Finishing the interrupted run │ 338 chunks embedded, 17,792 skipped, 58 seconds                        │
  ├───────────────────────────────┼────────────────────────────────────────────────────────────────────────┤
  │ Running it again              │ 0 embedded, 18,130 skipped, 0.6 seconds                                │
  ├───────────────────────────────┼────────────────────────────────────────────────────────────────────────┤
  │ What's stored                 │ 17,845 doc and 285 issue chunks, all dated, all from bge-small-en-v1.5 │
  ├───────────────────────────────┼────────────────────────────────────────────────────────────────────────┤
  │ Size                          │ 110 MB in total: HNSW index 35 MB, GIN index 6 MB                      │
  ├───────────────────────────────┼────────────────────────────────────────────────────────────────────────┤
  │ Tests                         │ 74 pass, including the 5 that use the real database                    │
  ├───────────────────────────────┼────────────────────────────────────────────────────────────────────────┤
  │ Vector search                 │ EXPLAIN shows Index Scan using chunks_embedding_hnsw                   │
  ├───────────────────────────────┼────────────────────────────────────────────────────────────────────────┤
  │ Keyword search                │ EXPLAIN shows Bitmap Index Scan on chunks_search_tsv_gin               │
  └───────────────────────────────┴────────────────────────────────────────────────────────────────────────┘

  A test search for "pod restart policy" returned Pods > Pod Lifecycle > … > Container restarts > Individual container restart policy and rules as the top
  result (similarity 0.835). The next results were the rest of that section, then Job "Pod failure policy". Those are the right pages.

  The schema

  One table, chunks, with one row per chunk. Its columns fall into four groups:

  1. Identity: chunk_id (primary key). It's the same ID the chunker creates, like doc:concepts/.../pod-lifecycle.md#19. Because the ID is stable, re-running
     means "update this row" instead of creating a duplicate.
  2. What to cite: title, url, date, heading_path, labels, extra. This is what the answer shows as sources.
     - source_type only accepts 'doc' or 'issue'; anything else is rejected.
     - heading_path and labels are real text arrays, so a later query can filter by label (e.g. only sig/node issues).
     - extra is jsonb for the details that differ by source: the doc path, or the issue number and how it was closed.
  3. What gets searched: text, embedding, search_tsv.
     - text is exactly what was embedded.
     - embedding holds the 384 numbers. The size is read from the model rather than hardcoded, and the script refuses to run if the table was built for a
       different model size.
     - search_tsv is the keyword version of the text, and Postgres computes it itself. Title words count more than body words.
  4. Bookkeeping: content_hash, embedding_model, indexed_at. The hash is a fingerprint of the chunk's text plus the model name. On each run the script compares
     hashes and embeds only chunks that are new or changed. That's why the second run took 0.6 seconds, and switching models later re-embeds everything
     automatically.

  Why these indexes

  - HNSW on embedding (vector search). Without an index, every question would compare against all 18,130 vectors. HNSW works like a map of "who is close to
    whom", so a search checks only a few hundred. I chose it over the other pgvector option, IVFFlat, because IVFFlat has to be built after all the data is
    loaded: it learns groupings from the existing rows. Our script adds rows in batches and later runs add more, and HNSW handles that without a rebuild. It
    uses cosine distance, which is how bge compares text.
  - GIN on search_tsv (keyword search). A GIN index is like the index at the back of a book: word → which rows contain it. Embeddings are good at meaning but
    weak on exact strings like error messages, flags such as --dry-run, or issue numbers. Keyword search catches those, and having both is what makes hybrid
    search possible in the retrieval step.
  - Nothing else yet. Every index slows down inserts. source_type has only two values, so an index on it wouldn't help, and nothing filters on labels or dates
    yet. They're easy to add once retrieval needs them.

  Worth knowing

  - Keyword ranking gives many ties. In the keyword search, the top 5 all scored 1.000, so ts_rank alone can't order them. That doesn't matter for storage, but
    the retrieval step should combine keyword and vector results rather than rely on keyword scores alone.
  - Stale rows aren't deleted. If re-chunking removes a chunk, its old row stays in the table. A cleanup option would be a small addition later.
  - Starting over: docker compose down -v deletes the database volume, and a fresh index takes about 55 minutes on this laptop.

  Nothing is committed yet.

   Retrieval works in all three modes, and all 91 tests pass (17 new), including the ones that run against the real database.

  How to use it:
  python -m repo_recall.retrieval.cli "why does my pod keep restarting"
  python -m repo_recall.retrieval.cli --source-type issue -k 8 "..."
  python -m repo_recall.retrieval.cli              # keeps asking; empty line quits
  The output has one column per mode and one row per rank. Each result shows a score, whether it's a doc or an issue, and a short label, and a source list with
  links follows the table. In the hybrid column, v6 k1 means the chunk was 6th in vector search and 1st in keyword search. In the app, the mode comes from
  RETRIEVAL_MODE in .env (default hybrid) and can be overridden per search.

  How each mode works

  - Vector: the question is embedded with bge's query instruction (documents are embedded without it, as bge intends), and the closest chunks by cosine
    similarity come back.
  - Keyword: Postgres full-text search, with one change: the question's words are joined with OR instead of AND. With AND, a question like "why does the
    kubelet ignore the scheduler" only matches chunks containing every word, which is almost none. With OR, chunks matching more words rank higher, and title
    matches count extra.
  - Hybrid: it takes the top 50 from each mode and merges them with Reciprocal Rank Fusion, which uses only each chunk's position in each list. Raw scores
    can't be mixed: cosine similarity is around 0.8 while keyword scores run from about 0.8 to 4. A chunk that both searches rank highly wins.

  What the real searches showed

  - "pod restart policy": all three modes put the Pod Lifecycle restart-policy section first.
  - "kubectl apply --dry-run=server": keyword search wins. It finds the server-side apply docs because they contain those exact words, while vector search
    returns more generic kubectl pages. Hybrid follows the keyword results.
  - "my app keeps crashing and coming back up": vector search wins. It understands this means container restarts and returns Pod Lifecycle first. Keyword
    search returns unrelated pages, because words like "crashing" and "coming back" don't appear in the right docs.

  Things you should know

  1. Hybrid can push down a result only one search found. In the crash question, vector search's #1 chunk dropped to #3 in hybrid because keyword search didn't
     find it at all. That's how this merging works when one search returns junk. The evaluation step should measure how often it happens before we decide on a
     default.
  2. One issue can fill all the top slots. With the issue filter, all 5 results were chunks of the same issue (#142328), because long issues are split into
     several chunks. The answer step would do better with a limit like "at most 2 chunks per source". (Done: see "Per-source limit" at the end.)
  3. The issue-filter protection isn't needed yet. I added a pgvector setting so that filtering to issues still returns a full top-k when the vector index is
     used. On today's data Postgres doesn't use the index for issues at all: they're only 285 rows, so it checks each one exactly (8 ms, nothing missed).
     Unfiltered and doc-only searches do use the index (about 2 ms). The setting starts to matter once you collect a few thousand more issues, and a test
     forces the index to prove it works.

  Code changes:
  - New in retrieval/:
    - models.py: the search modes and the result format.
    - query_encoder.py: turns a question into numbers.
    - postgres_search.py: the two database searches.
    - fusion.py: the hybrid merge.
    - retriever.py: picks the mode.
    - cli.py: the side-by-side tool.
  - Changed: five retrieval settings in config.py and .env.example. The database connection setup in indexing/backends/postgres.py is now a shared connect()
    function, so indexing and retrieval don't duplicate it.
  ├────────────────────────┼───────────────────────────────────────────────────┤
  │ Average chunk size     │ 385 tokens                                        │
  └────────────────────────┴───────────────────────────────────────────────────┘

  Compared with the docs: issues are 285 of 18,130 chunks (1.6%), and the docs have 17,845. That imbalance is why searches without a filter almost always
  return docs, and why one long issue could fill all 5 slots when filtering to issues.

  If you want more, python -m repo_recall.collection.issues --limit N continues from where it stopped:
  - GitHub requests: most items the issue list returns are actually pull requests, which get skipped, plus one request per issue for its comments. Your 5,000
    requests per hour should allow somewhere around a few thousand issues per hour. That's a rough estimate, not measured.
  - Indexing time: about 5 chunks per issue at ~5.7 chunks/s on your CPU, so about 1 second per issue. For example, 2,000 more issues ≈ 10,000 chunks ≈ 30
    minutes of indexing, and only the new chunks get embedded.

  Something like 1,000–3,000 issues would make issues a real part of the answers instead of a rare extra. That many issues would also make Postgres use the
  vector index for issue-only searches, where the filtering safeguard from the last step actually applies.

  Per-source limit

  Results now include at most 2 chunks from any one source, so a single long issue or docs page can't fill every slot. A source is identified by its URL:
  one URL per docs page, one per issue.

  How it works
  - limit_per_source() goes through the results best-first, skips a chunk once its URL has been used twice, stops at k, and renumbers the ranks 1..n.
  - Dropping chunks would leave fewer than k results, so when the limit is on, the retriever first fetches retrieval_candidates (50) rows and applies the
    limit afterwards. Hybrid already fetched 50 from each side, so only vector-only and keyword-only mode fetch more than before (a few milliseconds).
  - RETRIEVAL_MAX_PER_SOURCE in .env sets the limit (default 2). 0 turns it off and restores the old behaviour, including fetching exactly k rows.

  Worth knowing
  - Issue-only searches can now return fewer than k results when a question matches only a few issues. That's intended: 2 good chunks beat 5 copies of one
    issue.
  - Not yet checked on real data (Docker wasn't running). Check the crash question with --source-type issue: before, all 5 slots were #142328; now it should
    take at most 2.

  Code changes:
  - New: retrieval/diversity.py (the limit) and tests/retrieval/test_diversity.py.
  - Changed: retriever.py applies the limit after searching; config.py and .env.example have the new setting; test_retriever.py expects the larger fetch
    and has one end-to-end test of the limit.

  Answer generation

  Not yet run end to end: Ollama and Docker (Postgres) weren't running, so this is covered by unit tests only (113 pass, 26 new). To try it:
  docker compose up -d
  ollama pull llama3.1:8b
  python -m repo_recall.generation.cli "why does my pod keep crashing"
  python -m repo_recall.generation.cli --source-type issue -k 8 "..."
  It prints the answer, the cited sources with URLs, and a footer with model, retrieval and generation time, tokens in/out and estimated cost.

  Choosing a model
  - LLM_PROVIDER=ollama (default): a local model through Ollama's /api/chat. Costs nothing.
  - LLM_PROVIDER=openai: any API that speaks OpenAI's /chat/completions format — OpenAI itself, or local servers like vLLM and LM Studio. LLM_BASE_URL
    picks the server.
  - Both clients use requests (no new dependencies) and sit behind one small LLMClient interface, so tests use a fake. Network errors, error responses and
    oddly shaped replies all raise one LLMError.

  How the prompt works
  - Retrieved chunks are numbered by page or issue, not by chunk. The per-source limit allows 2 chunks from one page, and they share one number, so a
    citation points at something a person can open.
  - The rules: use only the sources, cite every claim like [2] or [1][3], and reply with one exact sentence when the sources don't contain the answer. If
    only part is covered, answer that part and say what's missing. Answers taken from an issue should say so, since an issue is one user's situation.
  - Docs excerpts carry their section path; issues carry their number and title.

  What comes back (Answer)
  - The answer text and the cited sources (number, title, URL, doc or issue), plus every retrieved source, which the evaluation step will need.
  - answered is false when the model gives the "no answer" sentence, and also when it cites nothing valid: an answer without citations isn't grounded.
    Citation numbers that don't match a source are dropped and logged as a warning.
  - Retrieval time, generation time, token counts and estimated cost.

  Tokens and cost
  - One log line per question with tokens, estimated cost, both timings and how many sources were cited.
  - Prices are in config (LLM_INPUT_COST_PER_MTOK / LLM_OUTPUT_COST_PER_MTOK, USD per million tokens) because they differ by provider and change over
    time. Set them to your provider's prices; they default to 0.
  - If the server doesn't report token counts, cost shows as unknown rather than a guess. If retrieval finds nothing, the model isn't called at all.

  Worth knowing
  - OLLAMA_NUM_CTX=8192: with Ollama's default context window, a prompt with 5 sources of ~400 tokens each could be cut off without any warning.
  - The OpenAI-compatible client sends max_tokens and temperature. Some of OpenAI's newer reasoning models reject those; using one would be a two-line change.
  - llama3.1:8b is only a placeholder default. Set LLM_MODEL to whatever you've pulled.

  Code changes:
  - New in generation/: llm.py (the two clients and the factory), prompt.py (numbering sources, building the prompt), citations.py (finding [n] markers),
    cost.py (the estimate), models.py (the result records), answerer.py (retrieve → prompt → LLM, timing and logging), cli.py.
  - New tests in tests/generation/.
  - Changed: config.py and .env.example have the LLM settings.

  Evaluation set

  200 questions with reference answers are in data/eval/eval_set.jsonl, built from resolved kind/support issues. These issues are kept out of the search index.

  In simple terms
  - Why: to measure how good RepoRecall is, we need real questions whose correct answers we already know. Then we can ask RepoRecall and compare.
  - Step 1, find questions: on GitHub, Kubernetes maintainers tag user questions "kind/support". We downloaded 1,200 of those closed issues with
    their comments.
  - Step 2, keep only well-answered ones: we kept an issue when someone other than the asker gave a real answer, and that person is a maintainer or
    linked to the docs or to a fix. Replies like "ask on Stack Overflow", "please give more details" or "same problem here" don't count.
  - Step 3, save them: for each kept issue we store the question, the best answer (the "reference answer"), and the links in that answer. We kept the
    200 strongest.
  - Step 4, no cheating: if these issues were in the search index, RepoRecall could just find the issue and copy its answer. So the indexer now skips
    them, and deletes them if they were indexed earlier. Tests check this.
  - Checking quality: in 5 random examples, 3 were good at first. We fixed the rules, and now about 21–22 of 25 random examples are real answers.

  How to rebuild:
  python -m repo_recall.collection.issue_search --limit 1200     # ~20 min, ~1,200 GitHub requests
  python -m repo_recall.evaluation.eval_set --seed 11            # offline, seconds; prints a summary and 5 random examples
  python -m repo_recall.evaluation.eval_set --show-only --show 10  # look at more examples without rebuilding

  Where the questions come from
  - The 60 issues in the index are recent bug reports and flakes, not questions. In kubernetes/kubernetes, maintainers label user questions kind/support
    (2,910 closed ones), so the candidates come from a GitHub search for that label, newest first: 1,200 issues created between late 2020 and 2026.
  - Search returns at most 1,000 results per query, so it searches one year at a time.
  - Candidates go to their own file (data/raw/eval_candidates/), not the issue file that gets indexed.
  - Collection now keeps each comment's author_association (OWNER, MEMBER, CONTRIBUTOR, NONE, ...). That's how "a maintainer answered" can be checked.

  Selection rules (the first rule an issue fails is its rejection reason)
  1. Not closed as a duplicate: the answer is in another issue.
  2. The title isn't CI or tracking work (flaky/failing test, tracking issue, KEP).
  3. Not filed by an OWNER/MEMBER/COLLABORATOR: those are work items and design discussions, not user questions.
  4. The question (title + essential body) has at least 15 words.
  5. There is at least one answer comment. An answer comment:
     - is by someone other than the asker;
     - has at least 25 words after removing slash commands (/close, /kind support) and quoted text;
     - isn't a non-answer: a redirect ("ask on Stack Overflow", "file this with your vendor"), a request for more information ("please provide", "can
       you share"), a reply that is mostly questions, a "same issue here", or triage chatter. The exception is when it links docs or a fix;
     - has a trust signal: a maintainer reply (OWNER/MEMBER/COLLABORATOR/CONTRIBUTOR), a docs link (kubernetes.io/docs or kubernetes/website), or a
       linked fix (PR or commit URL).
  The reference answer is the best-scoring answer comment: +2 maintainer, +2 docs link, +1 fix link, +1 if the asker's next comment says thanks or that
  it worked, plus up to +1 for length. When more than 200 issues qualify, the highest-scoring 200 are kept.

  Each example has: id, issue_url, title, question, reference_answer, answer_author (+ role), source_urls (every URL in the answer), doc_urls (English docs
  links, normalized to the index's URL format), labels, closed_at, signals, answer_score.

  Results (1,200 candidates)
  - 613 had no qualifying answer, 23 had too short a question, 19 were filed by maintainers, 1 had a CI title. That left 544 qualifying; the top 200 are kept.
  - Signals in the 200 kept: maintainer reply 198, docs link 140, asker confirmed 93, fix link 19.
  - 135 examples cite at least one English docs page, and 149 of the 161 cited doc URLs match a page in the index. That gives retrieval evaluation a
    known page to look for.

  How the rules were tuned
  - The first version kept replies that weren't answers: redirects, "please fill out the issue template", "we got the exact same issue", triage meeting
    notes, clarifying questions. The non-answer and maintainer-asker rules were added for those.
  - "Asker confirmed" was first worth +2, but 117 of 128 confirmations were just "thanks", which pushed polite threads above ones with a docs link. It's
    now +1.
  - Spot check of 25 random examples after tuning: about 21–22 are real answers. The misses are mostly "this setup isn't supported" replies and
    maintainers speculating. Those are hard to catch with rules; an LLM pass could filter them later.

  Keeping eval issues out of the index
  - index_chunks reads the eval set and (1) never embeds chunks whose URL is an eval issue, and (2) deletes rows already stored for those URLs, in case an
    issue was indexed before it joined the eval set. VectorStore has a new delete_by_urls for this. Matching on URL works for any collector or repo.
  - Today no eval issue is in the index (0 of 18,130 chunks). The check matters once more issues are collected. Tests cover skipping, deleting and reading
    the eval file, and a Postgres test covers the delete (it runs when Docker is up).

  Known limits
  - Another indexed issue that duplicates or quotes an eval issue could still leak its answer. Only the eval issue itself is excluded.
  - Answers from 2020–2023 may describe older Kubernetes versions than the current docs.
  - data/eval/ is gitignored like the rest of data/. The set is rebuilt from the candidates file, and the rules are deterministic.

  Code changes:
  - New: collection/issue_search.py, evaluation/eval_set.py, and tests for both.
  - Changed: collection/issues.py keeps author and author_association; indexing/index_chunks.py, store.py and backends/postgres.py handle the exclusion;
    config.py and .env.example have EVAL_* settings.

  Evaluation pipeline

  In simple terms
  - Why: we have 200 real questions with known good answers. This step asks RepoRecall each question and measures how well it does.
  - It checks three things for each search mode (vector, keyword, hybrid):
    1. Did the search find the right page? If the maintainer's answer linked a docs page, did that page come back in the top 5 or top 10 results?
    2. Was the answer good? A second AI model (the "judge") reads each answer and gives two scores from 1 to 5:
       - correctness: does it say the same thing as the maintainer's answer?
       - faithfulness: is everything it says backed by the sources it was given, or did it make things up?
    3. How fast and how expensive was it? Typical time (p50), slow-case time (p95), and cost per question.
  - It saves everything to a JSON file with a timestamp in the name, plus a short table in a Markdown file.
  - --sample N runs on N random questions only, for a quick and cheap check.
  - --max-cost stops the run before it spends more than the limit (we use $3).

  How to run
  python -m repo_recall.evaluation.run --sample 30 --max-cost 3     # quick check, ~$0.25, a few minutes
  python -m repo_recall.evaluation.run --max-cost 3                 # all 200 questions x 3 modes, ~$1.60
  python -m repo_recall.evaluation.run --retrieval-only             # search metrics only, no AI calls, free
  Results: data/eval/results/<timestamp>.json and <timestamp>.md

  The metrics
  - Relevant sources: the pages the reference answer links to that exist in the index. Questions without any (about a third) are skipped for search
    metrics but still count for answer quality.
  - Recall@5 / Recall@10: the share of relevant pages found in the top 5 / top 10 chunks. Recall@5 is exactly what the answer model sees.
  - MRR: 1 / (position of the first relevant chunk), averaged. 1.0 means it's always first; 0.5 means usually second.
  - Correctness and faithfulness (1–5): from the judge. The grading instructions are in src/repo_recall/evaluation/prompts/judge.md.
  - Latency: retrieval time plus answer time, per question. p50 is the typical case, p95 the slow case.
  - Cost per query: the answer model's cost only. The judge's cost is reported separately because it's part of testing, not of the product.

  Models and cost
  - Answers: gpt-6-luna ($0.10 / $0.50 per 1M tokens). Judge: gpt-5-mini ($0.25 / $2.00). Using a different model as the judge avoids a model grading
    its own answers too kindly.
  - Measured on the sample: about $0.0026 per question (answer + judging), so the full 600-answer run costs about $1.60.

  Problems found and fixed along the way
  - Newer OpenAI models reject two settings the client sent: max_tokens (they want max_completion_tokens) and temperature=0 (only the default is
    allowed). The client now notices the error, switches to what the model accepts, and remembers it.
  - With 8 answers in parallel, several requests hit that error at once. Some gave up because another thread had already switched. Each request now
    decides from what it sent itself.
  - gpt-6-luna and gpt-5-mini "think" before answering, and that uses tokens. With an 800-token limit, luna sometimes spent everything on thinking and
    returned an empty answer. The limits are now 4,000 for answers and 4,000 for the judge.
  - The client now retries rate limits (429) and server errors (5xx) up to 3 times, so one hiccup doesn't break a long run.
  - A stale LLM_PROVIDER=ollama line in .env was removed. The second line was already winning, but it was confusing.

  Worth knowing
  - Both models only allow the default randomness (temperature 1), so answers differ slightly between runs. Small differences between modes may be noise.
  - Relevance comes only from links a human happened to write, so recall undercounts: a correct page the maintainer didn't link counts as a miss.
  - The judge is one AI model, and its scores haven't been checked against human grading yet.

  Code changes:
  - New: evaluation/metrics.py (recall, MRR, percentiles), evaluation/judge.py and evaluation/prompts/judge.md (the judge), evaluation/run.py (the
    runner), and tests for each.
  - Changed:
    - generation/llm.py: parameter switching, retries, empty-answer warning.
    - generation/answerer.py: new answer_from_results, so each question is searched once per mode.
    - retrieval/postgres_search.py: new indexed_urls.
    - config.py, .env.example: EVAL_JUDGE_* and EVAL_RESULTS_DIR settings.
    - pyproject.toml: ships the prompt file.

  Test run (30 random questions, seed 1; too few to trust small differences). File: data/eval/results/20260926-214456.md
  | Mode    | Recall@5 | Recall@10 | MRR   | Correctness | Faithfulness | Answered | Latency p50 | Latency p95 | Cost / query |
  |---------|----------|-----------|-------|-------------|--------------|----------|-------------|-------------|--------------|
  | vector  | 0.395    | 0.447     | 0.340 | 2.70        | 4.30         | 90%      | 5.9s        | 10.8s       | $0.00046     |
  | keyword | 0.158    | 0.263     | 0.053 | 2.03        | 4.57         | 73%      | 18.8s       | 62.2s       | $0.00045     |
  | hybrid  | 0.289    | 0.342     | 0.187 | 2.70        | 4.07         | 90%      | 19.0s       | 58.8s       | $0.00048     |
  - Vector search does best on real questions. Keyword search is weak because real questions are long and full of logs. Hybrid is pulled down by
    keyword, the opposite of the earlier hand-picked examples.
  - Answers are faithful (they stick to their sources) but often not correct: when search misses the right page, the model answers honestly from what
    it got and misses the maintainer's point.
  - The test run cost $0.24.

  Finding: keyword search is very slow on long questions
  - A 332-word question becomes a search for 623 words joined with OR. That matches 18,101 of 18,130 chunks, and ranking them with ts_rank_cd takes
    81 seconds. ts_rank takes 0.85s for the same query (about 100x faster). A 5-word question takes 0.06s, which is why earlier manual tests never
    showed this.
  - Hybrid runs keyword search inside it, so it's slow too. Generation takes about 5s in every mode.
  - Planned fix, after the baseline so it gets its own before/after comparison: use ts_rank, and/or cap how many words of a long question go into the
    keyword search.

  Full baseline, first attempt (2026-09-26; lost to low memory, done in parts 1 and 2 below)
  - The first full run was stopped by Claude Code because the laptop ran out of memory (0.4 GB of 5.9 GB free). Vector mode had finished (spent
    $0.52), but the runner only saved at the end, so nothing was kept.
  - Fixed since: the runner now saves results after each mode, and --resume continues an interrupted run without redoing finished modes.
  - Next session:
    1. Free memory first (close VS Code windows / browser tabs; aim for 1.5 GB free). Keep Docker running for Postgres.
    2. docker compose up -d   (if Postgres isn't running)
    3. python -m repo_recall.evaluation.run --max-cost 3 --workers 4      (about $1.60, 2 to 2.5 hours because of slow keyword search)
       If it gets interrupted: python -m repo_recall.evaluation.run --resume data/eval/results/<timestamp>.json --max-cost 3
    4. Put the baseline table here.
    5. Then fix keyword search speed and re-run to compare.
  - Nothing is committed to git yet.

  Full baseline, part 1: vector mode (2026-09-27)
  - Memory was too tight for all 3 modes at once, so the baseline is run one mode at a time (--modes). Each mode is saved on its own, so a crash loses
    at most one mode.
  - Command: python -m repo_recall.evaluation.run --modes vector --max-cost 1 --workers 2
  - File: data/eval/results/20260927-140408.md (+ .json). All 200 questions, no errors, didn't stop early. Took 22 minutes and cost $0.53.
  | Mode    | Recall@5 | Recall@10 | MRR   | Correctness | Faithfulness | Answered | Latency p50 | Latency p95 | Cost / query |
  |---------|----------|-----------|-------|-------------|--------------|----------|-------------|-------------|--------------|
  | vector  | 0.389    | 0.486     | 0.295 | 2.73        | 4.43         | 84%      | 3.94s       | 7.73s       | $0.00044     |
  - Retrieval metrics are over the 128 questions whose reference answer links an indexed page.
  - Very close to the 30-question sample (Recall@5 0.395, correctness 2.70), so the sample was representative for vector.
  - Still to do: keyword and hybrid. Either run them as they are (slow, ~$0.53 each, same --modes command), or fix keyword speed first and run them on
    the fixed version (the slow "before" behaviour is already recorded above).

  Keyword speed fix (2026-09-27)
  - retrieval/postgres_search.py: ranks with ts_rank instead of ts_rank_cd. Search results are ranked differently, not filtered differently.
  - Free search-only check (data/eval/results/20260927-143557.md): keyword search p50 3.39s, p95 8.64s over 200 questions (was up to 81s for one
    long question).

  Full baseline, part 2: keyword and hybrid (2026-09-27), on the ts_rank version
  - keyword: data/eval/results/20260927-155856.md. 200 questions, no errors, $0.50.
  - hybrid:  data/eval/results/20260927-180533.md. 200 questions, no errors, $0.53, about 35 minutes.
  - A first hybrid attempt was killed by Claude Code at 130/200 for low memory (and the internet dropped at the same time). The runner only saved
    after each mode, so its $0.35 was lost. Fixed: see "Per-question progress" below.

  Full baseline, all modes (200 questions; retrieval metrics over the 128 with an indexed reference page)
  | Mode    | Recall@5 | Recall@10 | MRR   | Correctness | Faithfulness | Answered | Latency p50 | Latency p95 | Cost / query |
  |---------|----------|-----------|-------|-------------|--------------|----------|-------------|-------------|--------------|
  | vector  | 0.389    | 0.486     | 0.295 | 2.73        | 4.43         | 84%      | 3.94s       | 7.73s       | $0.00044     |
  | keyword | 0.217    | 0.310     | 0.159 | 2.10        | 4.22         | 66%      | 6.93s       | 12.99s      | $0.00041     |
  | hybrid  | 0.391    | 0.479     | 0.277 | 2.56        | 4.46         | 84%      | 7.58s       | 14.12s      | $0.00045     |
  - Vector and hybrid find the right page about equally often (Recall@5 0.389 vs 0.391); vector ranks it slightly higher (MRR 0.295 vs 0.277) and
    gets more correct answers (2.73 vs 2.56), at half the latency. Keyword alone is clearly worst on these long, log-heavy questions.
  - Hybrid is no longer pulled below vector as in the 30-question sample: with ts_rank, fusion roughly matches vector. It doesn't beat it yet.
  - Latency p50/p95 are end-to-end (search + answer). Hybrid search alone: p50 3.46s, p95 8.67s, so keyword search is still most of hybrid's time.
  - The model occasionally cites a source number that doesn't exist (3 times in the hybrid run, e.g. "[5]" with 4 sources). citations.py drops
    these with a warning; worth a prompt tweak later.
  - Total spend for the full baseline: $1.56 (+ $0.35 lost in the killed hybrid attempt).

  Per-question progress (2026-09-27)
  - evaluation/progress.py (new): each answered question is appended to <timestamp>.progress.jsonl and flushed to disk as it finishes. A line cut
    off by a crash is skipped on load.
  - evaluation/run.py: writes the results JSON at the start of a run (so a run killed in its first mode can be resumed); collects answers in
    finish order; --resume keeps logged questions of unfinished modes, runs only the rest, and adds their cost back to the budget. Questions that
    errored or were skipped for budget aren't logged, so they're retried.
  - A crash now loses at most the questions in flight (--workers of them, well under $0.01).
  - Tests: tests/evaluation/test_progress.py (new), tests/evaluation/test_run.py (kill mid-mode, then resume: only the unfinished question runs and
    each question is paid for once). 202 tests pass.

  Running on this laptop (5.9 GB RAM)
  - Docker (Postgres) is required for every mode. Minimize the Docker Desktop window, don't close it: closing it was followed by Docker shutting down
    and Postgres going away mid-session.
  - Teams (with its msedgewebview2 helpers) used over 1 GB; quit it from the tray before runs. Aim for ~0.8 GB free before starting.

  Next
  - Commit (nothing is in git yet).
  - Improve retrieval, then re-run to compare against this baseline. Ideas: cap how many words of a long question go into keyword search (faster
    and less noisy), tune hybrid fusion weights toward vector.

  API, UI and Docker (2026-09-27)
  - api/app.py: FastAPI app factory. POST /ask (answer, cited sources, retrieved_count, timing, usage/cost), GET /health (200, or 503 when
    Postgres is unreachable), / redirects to the Gradio UI at /ui. The embedding model, DB connection and LLM client are built once at startup
    (build_services); tests pass a factory with fakes. LLM failure -> 502 (details only in the log), lost DB -> 503, bad input -> 422.
  - api/schemas.py: request/response models. Questions are stripped, 1-20,000 chars (longest eval question is ~5,300); top_k 1-20.
  - api/ui.py: Gradio page. Sources are bullets labelled [n], not a numbered list: they're listed in citation order ([3] before [1]) and Markdown
    would renumber a numbered list, so the labels wouldn't match the answer.
  - api/__main__.py: python -m repo_recall.api runs it from .venv.
  - retrieval/postgres_search.py: one lock around the shared connection (API requests run on several threads), reconnects if the connection was
    lost (e.g. Postgres restarted), and ping() for /health.
  - Dockerfile (CPU-only torch, embedding model baked in, non-root user, healthcheck), .dockerignore (keeps .env and data/ out of the image),
    docker-compose.yml gains an `app` service: `docker compose up --build` runs db + app. .env is passed in at runtime; POSTGRES_DSN is
    overridden to the db service.
  - README: one-command run, API reference and example, config, building the index, dev commands, evaluation baseline.
  - Tests: tests/api (24); 226 in total pass.
  - Slow first question: the first search after startup took 21-28s (loading the vector index into memory); after that every mode takes ~0.1s
    in the container. Fix: build_services runs one warm-up hybrid search at startup (~12s, before the app reports ready). First real question
    after a fresh `docker compose up`: search 0.19s, total 2.6s.
  - Docker checked end to end: `docker compose up --build` from scratch, /health 200, UI 200, real /ask answers with cited sources. Image ~3 GB;
    running memory ~450 MB app + <150 MB Postgres. Gradio analytics and Hugging Face lookups are turned off in the container.
  - API smoke tests cost about $0.001 in total (4 real questions).
  - Note: the API uses RETRIEVAL_MODE from .env (currently hybrid). The baseline says vector is as accurate and twice as fast.
