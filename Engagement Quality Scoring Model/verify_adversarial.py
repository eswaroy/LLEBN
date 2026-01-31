import sys
import os
import time
import random

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, "src"))

def log(msg):
    print(msg)
    try:
        with open("adversarial_report.log", "a") as f:
            f.write(str(msg) + "\n")
    except: pass

def run_simulation():
    from eqsm.pipeline import EQSMPipeline
    
    with open("adversarial_report.log", "w") as f: f.write("")
    log("=== Phase-3 Adversarial Simulation (6-Months) ===\n")
    
    pipeline = EQSMPipeline()
    
    # SETUP
    genesis_node = "genesis_node"
    good_user = "good_user"
    farm_bots = [f"bot_{i}" for i in range(10)]
    
    # 1. Establish Good History
    # Good user gets trusted by genesis and settles rewards
    t0 = time.time()
    pipeline.graph.mark_settled(genesis_node)
    pipeline.graph.add_rating(genesis_node, good_user, 1.0)
    pipeline.graph.mark_settled(good_user) # Cheat: Assume they settled before
    
    # 2. Detect "Cycle Laundering" (A->B->C->A)
    # Bots create a trust ring in short time window
    log("[Scenario 1] Rapid Trust Laundering Ring")
    ts = t0
    pipeline.temporal.record_interaction(farm_bots[0], farm_bots[1], 1.0, ts)
    pipeline.temporal.record_interaction(farm_bots[1], farm_bots[2], 1.0, ts + 3600)
    pipeline.temporal.record_interaction(farm_bots[2], farm_bots[0], 1.0, ts + 7200)
    
    penalty = pipeline.temporal.detect_temporal_anomalies(farm_bots[0], ts + 10000)
    log(f"Bot 0 Temporal Penalty: {penalty:.2f} (Expected > 0.5)")
    if penalty > 0.5:
        log("PASS: Cycle Detected.\n")
    else:
        log("FAIL: Cycle Ignored.\n")

    # 3. "Slow Burn" Farm w/ Unsettled Trust
    # Bots interact heavily BUT none have settled rewards from outside world
    log("[Scenario 2] Unsettled Farm Attack")
    
    # They rate each other massive amounts
    for i in range(len(farm_bots)):
        src = farm_bots[i]
        dest = farm_bots[(i+1) % len(farm_bots)]
        pipeline.graph.add_rating(src, dest, 1.0)
        
    # Check Trust
    trust_scores = pipeline.graph.compute_global_trust()
    bot_trust = trust_scores.get(farm_bots[0], 0.0)
    good_trust = trust_scores.get(good_user, 0.0)
    
    log(f"Good User Trust: {good_trust:.6f}")
    log(f"Bot Ring Trust:  {bot_trust:.6f}")
    
    if bot_trust < 0.0001:
        log("PASS: Unsettled farm has ZERO global trust influence.")
    else:
        log("FAIL: Unsettled farm gained trust.")

    # 4. Economic Failure
    # Bots try to cash out
    log("[Scenario 3] Economic Settlement Attempt")
    res = pipeline.process_content_v3(
        "Generic bot content generated here.", 
        farm_bots[0],
        feature_overrides={"is_author_human_verified": True, "timestamp": t0}
    )
    
    # Even if they got a score locally, check trust rank
    log(f"Bot Final Score: {res['score']:.4f}")
    log(f"Bot Trust Rank: {res['trust_rank']:.6f}")
    
    if res['trust_rank'] < 0.001:
        log("PASS: Bot effectively economically isolated due to lack of trust execution.")
    else:
        log("FAIL: Bot gained economic ground.")

    log("\nAll Adversarial Tests Complete.")

if __name__ == "__main__":
    try:
        run_simulation()
    except Exception as e:
        import traceback
        traceback.print_exc()
        with open("adversarial_error.log", "w") as f:
            f.write(traceback.format_exc())
