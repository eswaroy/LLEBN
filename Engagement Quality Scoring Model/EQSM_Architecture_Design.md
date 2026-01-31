# EQSM System Architecture & Design Specification

## 1. System Architecture Diagram (Textual Description)

```mermaid
graph TD
    subgraph "Input Layer"
        User[User Identity]
        Content[Post/Comment]
        Context[Topic/Community]
        Peers[Peer Interactions]
    end

    subgraph "Signal Processing (Real-time)"
        NLP[NLP Engine] --> |Semantic Depth, Toxicity| FeatStore
        Temp[Temporal Engine] --> |Latency, Burstiness| FeatStore
        Graph[Graph Engine] --> |Centrality, TrustRank| FeatStore
    end

    User --> Temp
    Content --> NLP
    Peers --> Graph

    subgraph "Scoring Engine (EQSM)"
        FeatStore[Feature Store (Redis/Cassandra)] --> MVP[Phase 1: GBT Model]
        FeatStore --> Adv[Phase 2: GNN Model]
        
        MVP --> |Raw Score| Norm[Normalization Layer]
        Adv --> |Raw Score| Norm
    end

    subgraph "Anti-Gaming & Safety"
        Ring[Ring Detector] --> |Penalty| Norm
        Bot[Bot Fingerprint] --> |Block/Penalty| Norm
    end

    subgraph "Output & Integration"
        Norm --> Fin[Final Contribution Score]
        Fin --> Reward[Reward Engine]
        Fin --> Gov[Governance Weight]
        Fin --> Lead[Leaderboards]
    end
```

## 2. Feature Definitions Table

| Feature Category | Feature Name | ID | Type | Description |
| :--- | :--- | :--- | :--- | :--- |
| **A. Content Quality** | `semantic_depth_score` | `F_C01` | Float (0-1) | Transformer-based embedding distance from "trivial" phrases. Higher for coherent, structured arguments. |
| | `novelty_index` | `F_C02` | Float (0-1) | Cosine dissimilarity to the parent post and top-5 existing comments. Penalizes repetition. |
| | `toxicity_probability` | `F_C03` | Float (0-1) | Likelihood of toxic/inflammatory content (blocks scoring if > threshold). |
| **B. Temporal** | `response_latency_s` | `F_T01` | Int (Seconds) | Time from parent post creation to this response. |
| | `session_burstiness` | `F_T02` | Float | Variance of inter-event times. Low variance = potential bot/script. |
| **C. Trust & Graph** | `peer_reputation_weighted` | `F_G01` | Float | Sum of rater's reputation scores (not raw count). |
| | `interaction_density` | `F_G02` | Float | Ratio of mutual interactions to total interactions (High = potential echo chamber/cabal). |
| **D. User History** | `domain_consistency` | `F_U01` | Float | Variance of user's contribution scores over trailing 30 days. |
| | `account_age_days` | `F_U02` | Int | Age of Verified Human ID binding. |
| **E. Audience & Reach** | `effective_reach` | `F_A01` | Int | Count of unique Verified Humans exposed to the content. |
| **F. Device Context** | `device_context_type` | `F_D01` | Enum | Input source: `DESKTOP`, `MOBILE`, `VOICE_ASSIST`. |
| **G. Evaluation Mode** | `evaluation_context_mode` | `F_M01` | Enum | Scoring mode: `STANDARD`, `GROUP_CHALLENGE`, `HUMAN_VS_AI`. |
| **H. Cold Start** | `confidence_interval_lower` | `F_S01` | Float | Lower bound of the 95% Wilson Score Interval for quality. |
| | `is_new_entity` | `F_S02` | Boolean | True if account age < 7 days or interaction count < 5. |

## 3. Scoring Formula (Conceptual)

The **Contribution Score ($S_c$)** is calculated as:

$$
S_c = \sigma \left( \frac{w_1 \cdot Q_{semantic} + w_2 \cdot T_{novelty} + w_3 \cdot P_{reputation}}{1 + \lambda \cdot Pen_{gaming}} \right) \cdot D_{time}
$$

Where:
*   $\sigma$: Sigmoid activation (squashes output to 0-1).
*   $Q_{semantic}$: Base content quality score from NLP model.
*   $T_{novelty}$: Novelty score (penalizes "me too" comments).
*   $P_{reputation}$: Peer feedback weighted by the *reputation of the rater*.
    *   $P_{reputation} = \sum (Vote_i \times UserRep_i \times Decay(Mutual_{i, author}))$
*   $Pen_{gaming}$: Penalty factor derived from Anti-Gaming module (0 if clean, high if suspicious).
*   $D_{time}$: Time-decay factor for aggregate user scores (not single content).

## 4. Model Training Pipeline

### Phase 1: Gradient Boosted Trees (XGBoost)
*   **Objective**: Regression (predicting "Community Value").
*   **Labeling ($Y$)**: Semi-supervised.
    *   **Gold Set**: 5% of data manually reviewing by trusted "Founding Members" or Moderators.
    *   **Silver Set**: High-confidence automated labels based on long-term thread survival and lack of block actions.
*   **Loss Function**: Squared Error with Monotonic Constraints (e.g., higher semantic depth $\rightarrow$ higher score).

### Phase 2: Graph Neural Network (GNN) (Heterogeneous)
*   **Nodes**: User, Post, Topic.
*   **Edges**: `AUTHORED`, `REPLIED_TO`, `RATED`.
*   **Algorithm**: GraphSAGE or GAT (Graph Attention Networks).
*   **Objective**: Link Prediction (predict probability of positive high-reputation interaction).

## 5. Deployment & Performance

*   **Inference Strategy**: Hybrid.
    *   **Real-time (Online)**: Lightweight GBT model runs on `PostCreation` to filter spam and give initial provisional score. (~20ms latency).
    *   **Batch (Offline)**: Heavy GNN and Reputation updates run every hour (or event-triggered) to finalize scores and process rewards.
*   **Scaling**:
    *   Feature Store: Redis for hot features (user recent history).
    *   Model Serving: ONNX Runtime for GBT; PyTorch Serve for GNN.

## 6. Anti-Gaming Specifications (Safety)

1.  **Mutual Interaction Decay**:
    *   If User A and User B rate each other $> N$ times in $T$ days, the weight of their ratings drops exponentially ($0.9^k$).
2.  **Linguistic Fingerprinting**:
    *   Detects template-based comments ("Great project!", "Nice!") used by farming rings.
3.  **Sybil Subgraph Detection**:
    *   Identifies isolated dense subgraphs (clusters of users who only interact with each other and not the wider graph).

## 7. Explainability

All scores expose a `ReasoningVector` privately to the user:
```json
{
  "score": 0.85,
  "breakdown": {
    "content_depth": "+0.3 (High detail)",
    "peer_response": "+0.4 (Verified peers)",
    "novelty": "+0.15",
    "penalty": "0.0"
  }
}
```
*Publicly*, only the aggregate Reputation Tier is shown to prevent reverse-engineering the exact weights.

## 8. Missing System Additions (EQSM Extensions)

### 8.1. Audience-Size Normalization (Safety Fix)
To prevent "rich-get-richer" dynamics while protecting high-quality popular content.

*   **Objective**: Decouple *Quality* from *Reach* without suppressing top creators.
*   **Logic**: **Bounded Engagement Efficiency Ratio (BEER)**.
    We apply a dampening factor that saturates, preventing the penalty from going below a safety floor $\alpha$.

    $$
    S_{normalized} = S_{c} \times \max\left(\alpha, \left( \frac{1}{\log(1 + R_e) + \epsilon} \right)^{\beta} \right)
    $$
    
    *   $R_e$: `effective_reach` (Verified unique viewers).
    *   $\alpha$: **Normalization Floor** (Range: 0.5 - 0.8). Ensures that even with massive reach, a high-quality post retains at least $\alpha \times$ its score.
    *   $\beta$: **Dampening Factor** (Range: 0.3 - 0.6). Controls how strictly we penalize audience size.

### 8.2. AI Content Reward Exclusion (Policy)
**CRITICAL**: AI-generated or AI-autonomous content is strictly **excluded** from the economic layer.

*   **Rule**: `if author_type == 'AI_AGENT' OR content_flag == 'AI_GENERATED': ContributionScore = 0.0`.
*   **Implementation**:
    *   **Data Model Layer**: The `RewardEngine` filters all entities where `is_human_verified == False`.
    *   **Service Layer**: EQSM calculates scores for AI purely for **benchmarking** purposes (Comparison Baseline), but these scores are flagged as `non_monetizable`.
    *   **Audit**: Any attempt to override this flag triggers a P0 Security Alert.

### 8.3. Device-Aware Normalization
Adjusting expectations based on input modality to prevent discrimination.

*   **Context**: `device_context_type` (Desktop vs Mobile vs Voice).
*   **Adaptation**:
    *   **Mobile**: Relax `semantic_depth` threshold by 20%. Reduce typo penalties by 50%.
    *   **Voice**: Subtract `voice_processing_latency` from `response_latency_s`.

### 8.4. Community-Specific Scoring Overrides
Allowing distinct norms while enforcing global safety.

*   **Mechanism**: `CommunityConfig` overlay.
*   **Global Safety Constraint**: Communities cannot disable `toxicity_probability` checks or `RingDetector` penalties.
*   **Adjustments**:
    *   *Technical Communities*: Higher weight on `semantic_depth`.
    *   *Creative Communities*: Higher weight on `novelty_index`.

### 8.5. Evaluation Context Modes
*   **Standard**: Default EQSM logic.
*   **Group Challenge**: Score is a function of `CompletionStatus` AND `PeerVerification`.
*   **Human vs AI**: Differential scoring. Score ($S_{diff}$) is relative to the AI baseline ($B_{AI}$) in the same thread:
    $$ S_{final} = \max(0, S_{human} - B_{AI}) $$

### 8.6. Cold-Start Handling
*   **Strategy**: **Lower Bound Confidence Scoring**.
    *   Use the lower bound of the 95% Confidence Interval until $N > 10$.
    *   **Feature Gating**: Scores are capped at 0.5 until specific identity milestones are met.

### 8.7. Reward Settlement Window
*   **Policy**: **T+7 Settlement**.
    *   Rewards are **Locked** for 7 days post-creation.
    *   If `Anti-Gaming` triggers during this window, rewards are burned.
    *   Leaderboards update fast (Metric: *Estimated Value*), Wallets update slow (Metric: *Settled Value*).
