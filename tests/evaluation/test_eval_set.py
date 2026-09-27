import pytest

from repo_recall.config import Settings
from repo_recall.evaluation.eval_set import (
    build_eval_set,
    build_question,
    cap_code_block,
    evaluate_issue,
    extract_urls,
    is_docs_url,
    normalize_doc_url,
    strip_comment_noise,
)

SETTINGS = Settings(_env_file=None, eval_max_examples=200)
QUESTION_BODY = (
    "My pods keep getting OOMKilled right after a rolling update even though the memory "
    "limit is set well above what the app uses. What am I missing here?"
)
DOCS = "https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/"
LONG_ANSWER = (
    "The limit applies to the whole container including page cache, so the working set can "
    "exceed what your app reports. Check container_memory_working_set_bytes and raise the "
    "limit or lower the heap size accordingly."
)


def comment(body: str, author: str = "maintainer", role: str = "MEMBER") -> dict:
    return {"author": author, "author_association": role, "body": body, "created_at": "2024"}


def issue(number: int = 1, comments=None, **overrides) -> dict:
    record = {
        "number": number,
        "title": "Pods OOMKilled after rolling update",
        "author": "asker",
        "author_association": "NONE",
        "body": QUESTION_BODY,
        "comments": comments if comments is not None else [comment(LONG_ANSWER)],
        "labels": ["kind/support"],
        "closed_at": "2024-05-01T00:00:00Z",
        "state_reason": "completed",
        "url": f"https://github.com/kubernetes/kubernetes/issues/{number}",
    }
    return {**record, **overrides}


def reason(record: dict) -> str | None:
    return evaluate_issue(record, SETTINGS)[1]


# --- Rules ------------------------------------------------------------------


def test_member_answer_qualifies_with_expected_fields():
    example, why = evaluate_issue(issue(42), SETTINGS)
    assert why is None
    assert example["id"] == "kubernetes/kubernetes#42"
    assert example["issue_url"].endswith("/issues/42")
    assert example["question"].startswith("Pods OOMKilled after rolling update\n\nMy pods")
    assert example["reference_answer"] == LONG_ANSWER
    assert example["answer_author_association"] == "MEMBER"
    assert example["signals"] == ["maintainer_reply"]


def test_duplicates_are_rejected():
    assert reason(issue(state_reason="duplicate")) == "duplicate"


@pytest.mark.parametrize(
    "title", ["[Flaky test] TestFoo times out", "Failing test: e2e-gce", "KEP-1234: tracking"]
)
def test_ci_and_tracking_titles_are_rejected(title):
    assert reason(issue(title=title)) == "ci_or_tracking_title"


def test_short_questions_are_rejected():
    assert reason(issue(title="help", body="it broke")) == "short_question"


def test_asker_answering_themselves_does_not_count():
    assert reason(issue(comments=[comment(LONG_ANSWER, author="asker")])) == "no_qualifying_answer"


def test_short_or_command_only_answers_do_not_count():
    comments = [comment("/kind support\n/close"), comment("Try restarting the kubelet.")]
    assert reason(issue(comments=comments)) == "no_qualifying_answer"


def test_redirect_without_links_does_not_count():
    redirect = (
        "This is a support question and GitHub issues are not the right place for it. "
        "Please ask on Stack Overflow or the Kubernetes Slack where many more people "
        "can help you with this kind of question."
    )
    assert reason(issue(comments=[comment(redirect)])) == "no_qualifying_answer"


@pytest.mark.parametrize(
    "non_answer",
    [
        "This issue tracker is for bugs in Kubernetes, not GKE; this seems appropriate for "
        "a GCP support ticket since Kubernetes is not responsible for the node images.",
        "Please provide more details about your setup, including the cluster version, "
        "the network plugin you use and the full output of the failing command.",
        "Could you share the kubelet logs from the affected node and the exact manifest "
        "you applied, otherwise it is really hard to tell what is going on here?",
        "We got the exact same issue on our platform with vanilla Kubernetes and it also "
        "results in high CPU consumption for the apiserver after a few hours.",
        "Hi we are at the sig api-machinery bug triage meeting and decided to review this "
        "issue next time to leave time for sig auth to take a look at it first.",
        "How many files and directory layers are mounted in your environment? How long does "
        "the permission change take? And which storage driver is backing the volume here?",
    ],
)
def test_redirects_info_requests_and_me_toos_do_not_count(non_answer):
    assert reason(issue(comments=[comment(non_answer)])) == "no_qualifying_answer"


def test_issues_filed_by_maintainers_are_rejected():
    assert reason(issue(author_association="MEMBER")) == "asked_by_maintainer"
    assert reason(issue(author_association="CONTRIBUTOR")) is None


def test_asker_confirmation_credits_only_the_reply_right_before_it():
    first = comment(LONG_ANSWER, author="m1")
    follow_up = comment("Here is the output you asked for, still failing.", author="asker")
    second = comment(f"{LONG_ANSWER} Use a larger limit.", author="m2")
    thanks = comment("Thanks, that worked!", author="asker")
    example, _ = evaluate_issue(issue(comments=[first, follow_up, second, thanks]), SETTINGS)
    assert example["answer_author"] == "m2"
    assert "asker_confirmed" in example["signals"]


def test_redirect_that_links_docs_still_counts():
    text = f"Please ask on Stack Overflow next time, but the answer is in {DOCS} " + LONG_ANSWER
    assert reason(issue(comments=[comment(text)])) is None


def test_non_member_answer_needs_a_docs_or_fix_link():
    plain = comment(LONG_ANSWER, author="stranger", role="NONE")
    with_docs = comment(f"{LONG_ANSWER} See {DOCS}", author="stranger", role="NONE")
    with_fix = comment(
        f"{LONG_ANSWER} Fixed by https://github.com/kubernetes/kubernetes/pull/12345.",
        author="stranger",
        role="NONE",
    )
    assert reason(issue(comments=[plain])) == "no_qualifying_answer"
    assert evaluate_issue(issue(comments=[with_docs]), SETTINGS)[0]["signals"] == ["docs_link"]
    assert evaluate_issue(issue(comments=[with_fix]), SETTINGS)[0]["signals"] == ["fix_link"]


def test_best_scoring_answer_is_the_reference_and_asker_confirmation_counts():
    weaker = comment(LONG_ANSWER, author="m1")
    stronger = comment(f"See {DOCS} for how limits work. {LONG_ANSWER}", author="m2")
    thanks = comment("Thanks, that worked!", author="asker", role="NONE")
    example, _ = evaluate_issue(issue(comments=[weaker, stronger, thanks]), SETTINGS)
    assert example["answer_author"] == "m2"
    assert example["signals"] == ["maintainer_reply", "docs_link", "asker_confirmed"]
    assert example["source_urls"] == [DOCS]
    assert example["doc_urls"] == [DOCS]


# --- Question text ----------------------------------------------------------


def test_question_drops_environment_boilerplate_and_keeps_the_problem():
    body = (
        "### What happened?\n\nPods restart every few minutes after the upgrade.\n\n"
        "### Kubernetes version\n\n```console\nv1.30.1\n```\n\n"
        "### Cloud provider\n\nAWS\n\n"
        "### Anything else we need to know?\n\nOnly on nodes with cgroup v2."
    )
    question = build_question(issue(body=body))
    assert "Pods restart every few minutes" in question
    assert "Only on nodes with cgroup v2." in question
    assert "v1.30.1" not in question and "AWS" not in question


def test_long_code_blocks_are_capped():
    block = "```\n" + "\n".join(f"line {i}" for i in range(50)) + "\n```"
    capped = cap_code_block(block, max_lines=30)
    assert "line 29" in capped and "line 30" not in capped
    assert "... (20 more lines)" in capped
    assert capped.endswith("```")


def test_long_questions_are_cut_on_a_block_boundary():
    body = "\n\n".join(f"Paragraph {i} " + "word " * 60 for i in range(10))
    question = build_question(issue(body=body))
    assert len(question.split()) <= 300 + len(issue()["title"].split())
    assert question.rstrip().endswith("word")


def test_comment_noise_removes_commands_and_quotes_but_keeps_paths():
    body = "/kind support\n> you said X\nEdit /etc/kubernetes/manifests/kube-apiserver.yaml\n/close"
    assert strip_comment_noise(body) == "Edit /etc/kubernetes/manifests/kube-apiserver.yaml"


# --- URLs -------------------------------------------------------------------


def test_urls_are_extracted_without_trailing_punctuation_and_deduplicated():
    text = f"See ({DOCS}). Also {DOCS}, and https://github.com/kubernetes/kubernetes/pull/9."
    assert extract_urls(text) == [DOCS, "https://github.com/kubernetes/kubernetes/pull/9"]


def test_docs_urls_include_translations_and_website_repo_but_normalize_english_only():
    assert is_docs_url("https://kubernetes.io/zh-cn/docs/concepts/", SETTINGS)
    assert is_docs_url("https://github.com/kubernetes/website/blob/main/x.md", SETTINGS)
    assert not is_docs_url("https://kubernetes.io/blog/2024/foo/", SETTINGS)
    assert (
        normalize_doc_url("http://www.kubernetes.io/docs/tasks/debug#pods?x=1", SETTINGS)
        == "https://kubernetes.io/docs/tasks/debug/"
    )
    assert normalize_doc_url("https://kubernetes.io/zh-cn/docs/concepts/", SETTINGS) is None


# --- Whole set --------------------------------------------------------------


def test_build_eval_set_keeps_top_scoring_up_to_the_cap_and_counts_rejections():
    strong = [
        issue(n, comments=[comment(f"{LONG_ANSWER} {DOCS}")]) for n in range(1, 4)
    ]
    weak = [issue(n) for n in range(10, 13)]
    rejected = [issue(20, state_reason="duplicate"), issue(21, comments=[])]
    settings = Settings(_env_file=None, eval_max_examples=4)

    examples, rejections = build_eval_set([*weak, *rejected, *strong], settings)

    assert [e["issue_number"] for e in examples] == [1, 2, 3, 10]
    assert rejections == {"duplicate": 1, "no_qualifying_answer": 1, "over_max_examples": 2}
