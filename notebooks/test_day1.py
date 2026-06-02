from pathlib import Path
import sys

# Allow running this file directly from `notebooks/` while importing project code.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# from pangaea_readiness.fetcher import fetch_dataset
# from pangaea_readiness.profiler import profile_dataset

# # Test with a known PANGAEA dataset
# result = fetch_dataset("10.1594/PANGAEA.736010")

# print("=== FETCH RESULT ===")
# print(f"Title: {result['title']}")
# print(f"Columns: {result['columns']}")
# print(f"Shape: {result['data'].shape}")
# print(f"Error: {result['error']}")

# print("\n=== PROFILE ===")
# profile = profile_dataset(result)
# for k, v in profile.items():
#     print(f"{k}: {v}")
import os
from pangaea_readiness.fetcher import fetch_dataset
from pangaea_readiness.profiler import profile_dataset
from pangaea_readiness.scorer import score_dataset
from pangaea_readiness.ai_evaluator import ai_evaluate


def _load_env_file() -> None:
    """Load .env values into process env when running this script directly."""
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        entry = line.strip()
        if not entry or entry.startswith("#") or "=" not in entry:
            continue
        key, value = entry.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        os.environ.setdefault(key, value)


_load_env_file()

# Quick API key check
if not os.environ.get("SAIA_API_KEY"):
    print("⚠️  WARNING: SAIA_API_KEY not set — AI evaluation will be skipped")
    RUN_AI = False
else:
    print("✅ API key found — AI evaluation enabled")
    RUN_AI = True

TEST_DOIS = {
    "Tara Oceans (large/text-heavy)":     "10.1594/PANGAEA.842227",
    "CALYPSO (small/clean)":              "10.1594/PANGAEA.787140",
    "Sediment datums (too small for ML)": "10.1594/PANGAEA.856647",
}

for name, doi in TEST_DOIS.items():

    # Step 1 — fetch
    fetch_result = fetch_dataset(doi)

    # Step 2 — profile
    profile_result = profile_dataset(fetch_result)

    # Step 3 — rule-based score
    score_result = score_dataset(profile_result)

    print(f"\n{'='*60}")
    print(f"📦 {name}")
    print(f"🔗 {doi}")
    print(f"📌 Title  : {fetch_result['title']}")
    print(f"📐 Shape  : {fetch_result['data'].shape}")
    print(f"\n🏆 SCORE  : {score_result.score}/100")
    print(f"🚦 FLAG   : {score_result.flag}")
    print(f"🤖 TASKS  : {', '.join(score_result.ml_tasks) if score_result.ml_tasks else 'none'}")

    print(f"\n📊 BREAKDOWN:")
    for criterion, detail in score_result.breakdown.items():
        print(
            f"  {detail['verdict']} "
            f"{criterion:20s} "
            f"{detail['score']:2d}/{detail['max']} — "
            f"{detail['detail']}"
        )

    if score_result.warnings:
        print(f"\n⚠️  WARNINGS:")
        for w in score_result.warnings:
            print(f"  • {w}")

    if score_result.recommendations:
        print(f"\n💡 RECOMMENDATIONS:")
        for r in score_result.recommendations:
            print(f"  • {r}")

    # Step 4 — AI evaluation
    if RUN_AI:
        print(f"\n🧠 AI EVALUATION:")
        ai_result = ai_evaluate(fetch_result, profile_result)

        if ai_result.error:
            print(f"  ❌ Error: {ai_result.error}")
        else:
            print(f"  🌍 Domain       : {ai_result.domain}")
            print(f"  🎯 Confidence   : {ai_result.confidence}")
            print(f"  📝 Suitability  : {ai_result.suitability}")
            if ai_result.ml_tasks:
                print(f"  🤖 ML Tasks     : {', '.join(ai_result.ml_tasks)}")
            if ai_result.ds_tasks:
                print(f"  📊 DS Tasks     : {', '.join(ai_result.ds_tasks)}")
            if ai_result.red_flags:
                print(f"  🚩 Red Flags    : {', '.join(ai_result.red_flags)}")
            if ai_result.opportunities:
                print(f"  💡 Opportunities: {', '.join(ai_result.opportunities)}")
    else:
        print(f"\n🧠 AI EVALUATION: skipped (no API key)")