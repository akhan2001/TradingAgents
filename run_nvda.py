"""One-off Step-7 test run: NVDA, yesterday, Claude Sonnet 4.5, low depth.

Forces .env to win over any empty ANTHROPIC_API_KEY already in the
environment (the Claude Code bash sandbox injects a blank one).
"""
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True), override=True)

from langchain_core.callbacks import BaseCallbackHandler
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG


class TokenCounter(BaseCallbackHandler):
    def __init__(self):
        self.calls = 0
        self.in_tok = 0
        self.out_tok = 0
        self.cache_read = 0

    def on_llm_end(self, response, **kwargs):
        try:
            for gen_list in response.generations:
                for gen in gen_list:
                    msg = getattr(gen, "message", None)
                    um = getattr(msg, "usage_metadata", None) if msg else None
                    if um:
                        self.calls += 1
                        self.in_tok += um.get("input_tokens", 0)
                        self.out_tok += um.get("output_tokens", 0)
                        det = um.get("input_token_details") or {}
                        self.cache_read += det.get("cache_read", 0)
        except Exception as e:
            print("token-callback error:", e)


counter = TokenCounter()

config = DEFAULT_CONFIG.copy()
# Research depth "low" == 1 debate round + 1 risk round (already the default).
config["max_debate_rounds"] = 1
config["max_risk_discuss_rounds"] = 1

print("=== CONFIG ===")
print("provider:", config["llm_provider"], "| deep:", config["deep_think_llm"],
      "| quick:", config["quick_think_llm"])
print("results_dir:", config["results_dir"])
print("data_vendors:", config["data_vendors"])
print("=== STARTING NVDA 2026-05-28 (streaming) ===", flush=True)

ta = TradingAgentsGraph(debug=True, config=config, callbacks=[counter])
final_state, decision = ta.propagate("NVDA", "2026-05-28")

print("\n\n===== FINAL DECISION =====")
print(decision)

# Sonnet 4.5 list pricing: $3 / 1M input, $15 / 1M output.
in_cost = counter.in_tok / 1_000_000 * 3.0
out_cost = counter.out_tok / 1_000_000 * 15.0
print("\n===== TOKEN / COST REPORT =====")
print(f"LLM calls       : {counter.calls}")
print(f"Input tokens    : {counter.in_tok:,} (of which cache-read: {counter.cache_read:,})")
print(f"Output tokens   : {counter.out_tok:,}")
print(f"Est. input cost : ${in_cost:.4f}")
print(f"Est. output cost: ${out_cost:.4f}")
print(f"Est. TOTAL cost : ${in_cost + out_cost:.4f}")
print("=== DONE ===", flush=True)
