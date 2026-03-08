"""
Research Continuity Example
============================

Demonstrates MindGraph's core value: persistent structured memory that lets
an agent resume research across sessions without starting from scratch.

Scenario: Evaluate vector databases (Pinecone, Weaviate, Qdrant) for a
RAG-based AI assistant. Session 1 captures initial findings. Session 2
retrieves that context and uses it to drive a final decision.

No LLM calls — all research data is hardcoded so you can inspect the
graph operations without needing an LLM key. In a real agent, each
capture/argue/inquire call would use LLM-generated content.

Usage:
    export MINDGRAPH_URL="https://api.mindgraph.cloud"
    export MINDGRAPH_API_KEY="mg_..."
    python examples/research_continuity.py
"""

import os
import sys

from mindgraph import MindGraph


def main():
    url = os.environ.get("MINDGRAPH_URL", "https://api.mindgraph.cloud")
    api_key = os.environ.get("MINDGRAPH_API_KEY")
    if not api_key:
        print("Error: set MINDGRAPH_API_KEY environment variable")
        sys.exit(1)

    with MindGraph(url, api_key=api_key) as graph:

        # ================================================================
        # SESSION 1: Initial Research
        # ================================================================
        print("=" * 60)
        print("SESSION 1: Initial Research")
        print("=" * 60)

        # Open a session to group all activity
        s1 = graph.session(action="open", label="Vector DB evaluation — initial research")
        session1_uid = s1["uid"]
        print(f"  Opened session: {session1_uid}")

        # --- Define research goal (Intent layer) ---
        goal = graph.commit(
            action="goal",
            label="Evaluate vector databases for RAG assistant",
            summary="Compare Pinecone, Weaviate, and Qdrant on cost, latency, "
                    "filtering, managed vs self-hosted, and ecosystem fit.",
        )
        goal_uid = goal["uid"]
        print(f"  Created goal: {goal['label']}")

        # --- Capture sources (Reality layer) ---
        # A real agent would fetch these from the web; we record them as sources.
        src_pinecone = graph.capture(
            action="source",
            label="Pinecone documentation & benchmarks",
            summary="Official Pinecone docs covering serverless architecture, "
                    "pricing tiers, metadata filtering, and published p99 latency numbers.",
        )
        src_weaviate = graph.capture(
            action="source",
            label="Weaviate documentation & community benchmarks",
            summary="Weaviate docs on hybrid search, HNSW tuning, module ecosystem, "
                    "and self-hosted vs cloud pricing.",
        )
        src_qdrant = graph.capture(
            action="source",
            label="Qdrant documentation & ANN benchmarks",
            summary="Qdrant docs on payload filtering, quantization, gRPC API, "
                    "and published ann-benchmarks results.",
        )
        print(f"  Captured 3 sources")

        # --- Record observations ---
        obs_pinecone_cost = graph.capture(
            action="observation",
            label="Pinecone serverless costs scale with reads",
            summary="Pinecone serverless charges per read unit. At 1M queries/month "
                    "the cost is ~$70/mo for 1M vectors, but jumps to ~$280/mo at 5M queries.",
        )
        obs_weaviate_hybrid = graph.capture(
            action="observation",
            label="Weaviate supports native hybrid search",
            summary="Weaviate combines BM25 keyword search with vector search in a "
                    "single query via its hybrid search operator, eliminating the need "
                    "for a separate keyword index.",
        )
        obs_qdrant_filtering = graph.capture(
            action="observation",
            label="Qdrant payload filtering is pre-search",
            summary="Qdrant applies payload (metadata) filters before HNSW traversal, "
                    "giving exact filter results without post-search recall loss. "
                    "Pinecone and Weaviate apply filters post-search by default.",
        )
        print(f"  Recorded 3 observations")

        # --- Build claims with evidence (Epistemic layer) ---
        # argue() creates a Claim node and links Evidence nodes to it.
        arg_qdrant = graph.argue(
            claim={
                "label": "Qdrant has the best metadata filtering for RAG",
                "summary": "Pre-search filtering preserves recall while enforcing "
                           "exact metadata constraints — critical for multi-tenant RAG.",
            },
            evidence=[
                {
                    "label": "Qdrant pre-search filtering benchmark",
                    "summary": "ANN-benchmarks show Qdrant maintains 95%+ recall with "
                               "payload filters vs 80-85% for post-filter approaches.",
                    "source_uid": src_qdrant["uid"],
                },
            ],
        )
        claim_qdrant_uid = arg_qdrant["claim_uid"]

        arg_weaviate = graph.argue(
            claim={
                "label": "Weaviate's hybrid search reduces retrieval pipeline complexity",
                "summary": "Built-in BM25 + vector fusion means one fewer service to "
                           "deploy and maintain compared to Pinecone + Elasticsearch.",
            },
            evidence=[
                {
                    "label": "Weaviate hybrid search documentation",
                    "summary": "Single API call combines keyword and vector results with "
                               "configurable alpha weighting.",
                    "source_uid": src_weaviate["uid"],
                },
            ],
        )
        claim_weaviate_uid = arg_weaviate["claim_uid"]
        print(f"  Built 2 claims with evidence")

        # --- Record open questions (Epistemic layer) ---
        q_latency = graph.inquire(
            action="open_question",
            label="How does Qdrant's p99 latency compare under concurrent load?",
            summary="Benchmarks are single-client. Need multi-client latency data "
                    "to confirm Qdrant can handle 100+ concurrent RAG queries.",
            props={"status": "open"},
        )
        q_cost = graph.inquire(
            action="open_question",
            label="What is the total cost of self-hosting Qdrant vs Pinecone serverless?",
            summary="Qdrant Cloud pricing is competitive, but self-hosted on k8s "
                    "requires ops effort. Need a 12-month TCO comparison.",
            props={"status": "open"},
        )
        q_migration = graph.inquire(
            action="open_question",
            label="How easy is it to migrate from Pinecone to Qdrant?",
            summary="If we start with Pinecone for speed, can we migrate later "
                    "without re-indexing all vectors?",
            props={"status": "open"},
        )
        print(f"  Recorded 3 open questions")

        # --- Journal summary (Memory layer) ---
        # Captures the researcher's unstructured thoughts about the session.
        graph.journal(
            "Session 1 wrap-up",
            summary="Initial research complete. Qdrant leads on filtering, Weaviate "
                    "on hybrid search simplicity, Pinecone on zero-ops. Three open "
                    "questions remain around latency, TCO, and migration feasibility. "
                    "Next session should resolve these before making a recommendation.",
            session_uid=session1_uid,
        )
        print(f"  Wrote journal entry")

        # Close session
        graph.session(action="close", session_uid=session1_uid)
        print(f"  Closed session 1\n")

        # Collect UIDs from Session 1 for later use in distill
        session1_uids = [
            goal_uid,
            src_pinecone["uid"], src_weaviate["uid"], src_qdrant["uid"],
            obs_pinecone_cost["uid"], obs_weaviate_hybrid["uid"], obs_qdrant_filtering["uid"],
            claim_qdrant_uid, claim_weaviate_uid,
            q_latency["uid"], q_cost["uid"], q_migration["uid"],
        ]

        # ================================================================
        # SESSION 2: Resume & Decide
        # ================================================================
        print("=" * 60)
        print("SESSION 2: Resume & Decide")
        print("=" * 60)

        s2 = graph.session(action="open", label="Vector DB evaluation — decision")
        session2_uid = s2["uid"]
        print(f"  Opened session: {session2_uid}\n")

        # --- RESUME CONTEXT BLOCK ---
        # This is the "money shot": retrieving prior context proves the agent
        # did not start from zero.
        print("  --- Resuming with prior context ---")

        # Retrieve active goals
        goals_resp = graph.retrieve(action="active_goals")
        goals = goals_resp.get("nodes", []) if isinstance(goals_resp, dict) else goals_resp
        active_goal = next(
            (g for g in goals if g.get("uid") == goal_uid),
            goals[0] if goals else None,
        )
        if active_goal:
            print(f"  Goal: {active_goal['label']}")

        # Retrieve open questions
        questions_resp = graph.retrieve(action="open_questions")
        questions = questions_resp.get("nodes", []) if isinstance(questions_resp, dict) else questions_resp
        print(f"  Open questions ({len(questions)}):")
        for q in questions:
            print(f"    - {q['label']}")

        # Search for prior findings about vector databases
        search_results = graph.search("Qdrant filtering Pinecone")
        print(f"  Prior findings ({len(search_results)} matches):")
        for r in search_results[:5]:
            node = r["node"] if "node" in r else r
            print(f"    - {node['label']}")

        print("  --- End resume context ---\n")

        # --- Address each open question with new observations ---
        # Retrieved questions drive what we research next.
        session2_uids = []

        # Question 1: Qdrant p99 latency under concurrent load
        obs_latency = graph.capture(
            action="observation",
            label="Qdrant p99 latency is 12ms at 100 concurrent clients",
            summary="Load testing with 100 concurrent clients on a 3-node Qdrant "
                    "cluster (1M vectors, 768 dims) shows p99 of 12ms. Pinecone "
                    "serverless shows p99 of 18ms under the same workload. "
                    f"Addresses question: '{q_latency['label']}'",
        )
        session2_uids.append(obs_latency["uid"])
        print(f"  Answered: {q_latency['label']}")

        # Question 2: TCO comparison
        obs_tco = graph.capture(
            action="observation",
            label="Qdrant Cloud TCO is 40% lower than Pinecone at scale",
            summary="12-month TCO for 5M vectors, 2M queries/month: Pinecone "
                    "serverless ~$3,360/yr, Qdrant Cloud ~$2,016/yr (3-node managed). "
                    "Self-hosted Qdrant on k8s ~$1,440/yr but adds ~8 hrs/month ops. "
                    f"Addresses question: '{q_cost['label']}'",
        )
        session2_uids.append(obs_tco["uid"])
        print(f"  Answered: {q_cost['label']}")

        # Question 3: Migration feasibility — partially answered
        obs_migration = graph.capture(
            action="observation",
            label="Pinecone-to-Qdrant migration requires re-indexing",
            summary="Pinecone does not support bulk vector export. Migration "
                    "requires re-embedding from source documents. Estimated effort: "
                    "2-3 days for 5M vectors. Not a dealbreaker but worth noting. "
                    f"Partially addresses: '{q_migration['label']}'",
        )
        session2_uids.append(obs_migration["uid"])
        print(f"  Partially answered: {q_migration['label']}\n")

        # --- Open a decision (Intent layer) ---
        # The decision summary references the retrieved goal.
        decision = graph.deliberate(
            action="open_decision",
            label="Choose vector database for RAG assistant",
            summary=f"Decision needed to fulfill goal: '{active_goal['label']}'. "
                    "Three candidates evaluated across cost, latency, filtering, "
                    "hybrid search, and ops overhead.",
        )
        decision_uid = decision["uid"]
        session2_uids.append(decision_uid)

        # --- Add options referencing prior claims ---
        opt_qdrant = graph.deliberate(
            action="add_option",
            decision_uid=decision_uid,
            label="Qdrant Cloud",
            summary="Best-in-class filtering (pre-search), lowest TCO at scale, "
                    "strong latency. Requires learning Qdrant's query DSL. "
                    f"Supported by: '{arg_qdrant['claim_uid']}' (filtering claim).",
        )

        opt_weaviate = graph.deliberate(
            action="add_option",
            decision_uid=decision_uid,
            label="Weaviate Cloud",
            summary="Native hybrid search simplifies architecture. Mid-range cost. "
                    f"Supported by: '{arg_weaviate['claim_uid']}' (hybrid search claim). "
                    "Weaker metadata filtering than Qdrant.",
        )

        opt_pinecone = graph.deliberate(
            action="add_option",
            decision_uid=decision_uid,
            label="Pinecone Serverless",
            summary="Zero-ops, fastest time to production. Highest cost at scale, "
                    "post-search filtering reduces recall. No bulk export for migration.",
        )
        print(f"  Opened decision with 3 options")

        # --- Resolve the decision ---
        # Chosen option justified by evidence gathered across both sessions.
        resolution = graph.deliberate(
            action="resolve",
            decision_uid=decision_uid,
            chosen_option_uid=opt_qdrant["uid"],
            summary="Choosing Qdrant Cloud. Pre-search filtering maintains 95%+ "
                    "recall (critical for multi-tenant RAG). 40% lower TCO than "
                    "Pinecone at projected scale. p99 latency of 12ms beats Pinecone's "
                    "18ms. Tradeoff: Weaviate's hybrid search is convenient, but we "
                    "can implement BM25 + vector fusion at the application layer. "
                    "Caveat: migration from any future provider will require re-indexing.",
        )
        session2_uids.append(opt_qdrant["uid"])
        session2_uids.append(opt_weaviate["uid"])
        session2_uids.append(opt_pinecone["uid"])
        print(f"  Resolved decision: Qdrant Cloud\n")

        # --- Distill final summary (Memory layer) ---
        # source_uids spans both sessions, summary synthesizes everything.
        all_uids = session1_uids + session2_uids
        distill_result = graph.distill(
            label="Vector DB Evaluation — Final Recommendation",
            summary=(
                "Recommendation: Qdrant Cloud for the RAG assistant.\n\n"
                "Key findings:\n"
                "- Qdrant's pre-search payload filtering maintains 95%+ recall with "
                "exact metadata constraints, critical for multi-tenant isolation.\n"
                "- 12-month TCO at projected scale (5M vectors, 2M queries/mo): "
                "Qdrant Cloud ~$2,016/yr vs Pinecone ~$3,360/yr (40% savings).\n"
                "- p99 latency under 100 concurrent clients: Qdrant 12ms vs "
                "Pinecone 18ms.\n\n"
                "Resolved open questions:\n"
                "- Latency under load: confirmed competitive (12ms p99).\n"
                "- TCO: Qdrant Cloud is 40% cheaper; self-hosted saves more but "
                "adds ops burden.\n\n"
                "Remaining caveat:\n"
                "- Migration requires re-embedding from source docs (~2-3 days for 5M "
                "vectors). Pinecone does not support bulk export. This is acceptable "
                "risk given the cost and performance advantages.\n\n"
                "Tradeoff accepted: Weaviate's built-in hybrid search is convenient "
                "but can be replicated at the application layer. Qdrant's filtering "
                "advantage is harder to work around."
            ),
            summarizes_uids=all_uids,
        )
        session2_uids.append(distill_result["uid"])
        print(f"  Distilled final summary: {distill_result['uid']}")

        # Close session
        graph.session(action="close", session_uid=session2_uid)
        print(f"  Closed session 2\n")

        # ================================================================
        # FINAL OUTPUT
        # ================================================================
        print("=" * 60)
        print("RECOMMENDATION")
        print("=" * 60)
        print()
        print("  Chosen: Qdrant Cloud")
        print()
        print("  Evidence (from Session 1):")
        print("    - Pre-search filtering preserves 95%+ recall")
        print("    - Hybrid search is nice-to-have but reproducible at app layer")
        print()
        print("  Questions resolved (in Session 2):")
        print("    - Latency under load: 12ms p99 at 100 concurrent clients")
        print("    - TCO: 40% cheaper than Pinecone at projected scale")
        print()
        print("  Remaining caveat:")
        print("    - Migration requires re-embedding (~2-3 days for 5M vectors)")
        print()
        print("  The agent resumed Session 2 with full context from Session 1.")
        print("  No research was repeated. Open questions drove targeted follow-up.")
        print()


if __name__ == "__main__":
    main()
