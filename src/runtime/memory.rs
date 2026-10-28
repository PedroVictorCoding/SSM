//! Provenance-Aware Long-Term Retrieval, User Preferences, and Memory Isolation.
//!
//! Provides:
//! - Host-owned approved document and user preferences storage with source IDs, timestamps,
//!   versions, retention policies, and access scopes.
//! - Strict classification of knowledge records:
//!   - `ApprovedDocument`: Human-approved documents, manuals, and specifications.
//!   - `UserPreference`: Explicit user-configured preferences across sessions.
//!   - `VerifiedObservation`: Deterministically verified tool results and system observations.
//!   - `ModelHypothesis`: Speculative model outputs (NEVER automatically treated as verified facts
//!     or ingested into the training pipeline, preserving ALIGNMENT.md).
//! - Hybrid lexical (BM25) and semantic (cosine similarity) retrieval under bounded context budgets.
//! - Conflict, correction, and stale-record tracking (superseding older versions without silent overwrites).
//! - User inspection, deletion, and source revocation controls with immediate index purging.
//! - Strict multi-session access separation and storage growth telemetry.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};
use std::time::Instant;

use serde::{Deserialize, Serialize};

/// Classification of a memory record's epistemological source.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FactKind {
    /// Human-approved documents, specifications, or datasets.
    ApprovedDocument,
    /// Explicit human user preferences (e.g. style, theme, rules).
    UserPreference,
    /// Verified tool outputs or grounded observations with committed side effects.
    VerifiedObservation,
    /// Speculative model hypothesis or unverified inference.
    /// Strictly barred from automatic promotion to facts or training data per ALIGNMENT.md.
    ModelHypothesis,
}

/// Scope defining which sessions/users are permitted to retrieve this record.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AccessScope {
    /// Universally accessible across all users and sessions.
    Global,
    /// Restricted to a specific authenticated user across their sessions.
    User(String),
    /// Isolated to a single conversational session.
    Session(String),
}

/// Retention policy determining lifetime and expiration.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RetentionPolicy {
    /// Retained indefinitely until explicitly deleted by user.
    Permanent,
    /// Expires after a monotonic time duration in milliseconds.
    TtlMs(u64),
    /// Discarded immediately when session closes.
    SessionOnly,
}

/// Complete provenance trace of a stored memory record.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MemoryProvenance {
    pub record_id: String,
    pub source_id: String,
    pub author: String,
    pub created_at_ms: u64,
    pub updated_at_ms: u64,
    pub version: u32,
    pub is_verified: bool,
    pub superseded_by: Option<String>,
}

/// Stored memory record with provenance, access boundaries, and content.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MemoryRecord {
    pub id: String,
    pub key: String,
    pub content: String,
    pub embedding: Option<Vec<f32>>,
    pub kind: FactKind,
    pub scope: AccessScope,
    pub retention: RetentionPolicy,
    pub provenance: MemoryProvenance,
    pub is_deleted: bool,
    /// Safety flag: permanently false for hypotheses and runtime outputs.
    pub is_training_eligible: bool,
}

/// Detected conflict between records for the same key or topic.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MemoryConflict {
    pub key: String,
    pub active_record_id: String,
    pub prior_record_id: String,
    pub reason: String,
}

/// Query specifying constraints and search criteria for memory retrieval.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RetrievalQuery {
    pub query_text: String,
    pub query_embedding: Option<Vec<f32>>,
    pub user_id: Option<String>,
    pub session_id: Option<String>,
    pub current_timestamp_ms: u64,
    pub max_results: usize,
    pub max_tokens_budget: usize,
    pub min_score: f32,
    pub allowed_kinds: Option<Vec<FactKind>>,
}

impl Default for RetrievalQuery {
    fn default() -> Self {
        Self {
            query_text: String::new(),
            query_embedding: None,
            user_id: None,
            session_id: None,
            current_timestamp_ms: 0,
            max_results: 5,
            max_tokens_budget: 512,
            min_score: 0.1,
            allowed_kinds: None,
        }
    }
}

/// Individual scored result from memory retrieval.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RetrievedMemory {
    pub record: MemoryRecord,
    pub lexical_score: f32,
    pub semantic_score: f32,
    pub combined_score: f32,
    pub estimated_tokens: usize,
}

/// Structured response from long-term memory retrieval.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RetrievalResponse {
    pub results: Vec<RetrievedMemory>,
    pub total_tokens_used: usize,
    pub conflicts_detected: Vec<MemoryConflict>,
    pub latency_ms: f64,
    pub provenance_sources: Vec<String>,
}

/// Telemetry metrics on storage volume and index size.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MemoryStorageMetrics {
    pub total_records: usize,
    pub active_records: usize,
    pub deleted_records: usize,
    pub total_bytes_content: usize,
    pub inverted_index_terms: usize,
    pub unique_sources: usize,
}

/// Audit event for user inspection and compliance tracking.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MemoryAuditEvent {
    pub timestamp_ms: u64,
    pub action: String,
    pub record_id: String,
    pub key: String,
    pub details: String,
}

/// Internal indices for fast lexical lookups and scope filtering.
#[derive(Default)]
struct MemoryIndex {
    /// Inverted index: lowercase term -> list of active record_ids
    term_to_records: HashMap<String, HashSet<String>>,
    /// Active record for a key: key -> record_id
    key_to_active_id: HashMap<String, String>,
    /// Source mapping: source_id -> list of record_ids
    source_to_records: HashMap<String, HashSet<String>>,
}

/// Host-owned thread-safe long-term memory store.
pub struct LongTermMemoryStore {
    records: Arc<RwLock<HashMap<String, MemoryRecord>>>,
    index: Arc<RwLock<MemoryIndex>>,
    audit_log: Arc<RwLock<Vec<MemoryAuditEvent>>>,
}

impl Default for LongTermMemoryStore {
    fn default() -> Self {
        Self::new()
    }
}

impl LongTermMemoryStore {
    pub fn new() -> Self {
        Self {
            records: Arc::new(RwLock::new(HashMap::new())),
            index: Arc::new(RwLock::new(MemoryIndex::default())),
            audit_log: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Stores an approved document with full provenance.
    pub fn store_approved_document(
        &self,
        record_id: &str,
        key: &str,
        content: &str,
        embedding: Option<Vec<f32>>,
        source_id: &str,
        scope: AccessScope,
        timestamp_ms: u64,
    ) -> Result<String, String> {
        let provenance = MemoryProvenance {
            record_id: record_id.to_string(),
            source_id: source_id.to_string(),
            author: "human_approver".to_string(),
            created_at_ms: timestamp_ms,
            updated_at_ms: timestamp_ms,
            version: 1,
            is_verified: true,
            superseded_by: None,
        };

        self.insert_record_internal(
            record_id,
            key,
            content,
            embedding,
            FactKind::ApprovedDocument,
            scope,
            RetentionPolicy::Permanent,
            provenance,
            false, // is_training_eligible permanently false for runtime context
            timestamp_ms,
        )
    }

    /// Stores a user preference, superseding any prior version for that key.
    pub fn store_user_preference(
        &self,
        record_id: &str,
        key: &str,
        content: &str,
        embedding: Option<Vec<f32>>,
        user_id: &str,
        timestamp_ms: u64,
    ) -> Result<String, String> {
        let scope = AccessScope::User(user_id.to_string());
        let (version, prior_id) = {
            let idx = self.index.read().unwrap();
            if let Some(prev_id) = idx.key_to_active_id.get(key) {
                let recs = self.records.read().unwrap();
                let v = recs
                    .get(prev_id)
                    .map(|r| r.provenance.version + 1)
                    .unwrap_or(1);
                (v, Some(prev_id.clone()))
            } else {
                (1, None)
            }
        };

        // Supersede prior record
        if let Some(prev) = prior_id {
            let mut recs = self.records.write().unwrap();
            if let Some(r) = recs.get_mut(&prev) {
                r.provenance.superseded_by = Some(record_id.to_string());
            }
        }

        let provenance = MemoryProvenance {
            record_id: record_id.to_string(),
            source_id: format!("user_pref:{}", user_id),
            author: format!("user:{}", user_id),
            created_at_ms: timestamp_ms,
            updated_at_ms: timestamp_ms,
            version,
            is_verified: true,
            superseded_by: None,
        };

        self.insert_record_internal(
            record_id,
            key,
            content,
            embedding,
            FactKind::UserPreference,
            scope,
            RetentionPolicy::Permanent,
            provenance,
            false,
            timestamp_ms,
        )
    }

    /// Stores a verified tool or environment observation.
    pub fn store_verified_observation(
        &self,
        record_id: &str,
        key: &str,
        content: &str,
        embedding: Option<Vec<f32>>,
        source_id: &str,
        scope: AccessScope,
        retention: RetentionPolicy,
        timestamp_ms: u64,
    ) -> Result<String, String> {
        let provenance = MemoryProvenance {
            record_id: record_id.to_string(),
            source_id: source_id.to_string(),
            author: "verified_tool".to_string(),
            created_at_ms: timestamp_ms,
            updated_at_ms: timestamp_ms,
            version: 1,
            is_verified: true,
            superseded_by: None,
        };

        self.insert_record_internal(
            record_id,
            key,
            content,
            embedding,
            FactKind::VerifiedObservation,
            scope,
            retention,
            provenance,
            false,
            timestamp_ms,
        )
    }

    /// Stores a model speculative hypothesis.
    /// STRICT INVARIANT: Always marked unverified and permanently ineligible for training.
    pub fn store_model_hypothesis(
        &self,
        record_id: &str,
        key: &str,
        content: &str,
        embedding: Option<Vec<f32>>,
        scope: AccessScope,
        timestamp_ms: u64,
    ) -> Result<String, String> {
        let provenance = MemoryProvenance {
            record_id: record_id.to_string(),
            source_id: "model:autoregressive_guess".to_string(),
            author: "assistant_model".to_string(),
            created_at_ms: timestamp_ms,
            updated_at_ms: timestamp_ms,
            version: 1,
            is_verified: false,
            superseded_by: None,
        };

        self.insert_record_internal(
            record_id,
            key,
            content,
            embedding,
            FactKind::ModelHypothesis,
            scope,
            RetentionPolicy::SessionOnly,
            provenance,
            false, // Never eligible for training
            timestamp_ms,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn insert_record_internal(
        &self,
        record_id: &str,
        key: &str,
        content: &str,
        embedding: Option<Vec<f32>>,
        kind: FactKind,
        scope: AccessScope,
        retention: RetentionPolicy,
        provenance: MemoryProvenance,
        is_training_eligible: bool,
        timestamp_ms: u64,
    ) -> Result<String, String> {
        if is_training_eligible && kind == FactKind::ModelHypothesis {
            return Err("Model hypotheses and runtime retrieval records are strictly barred from training ingestion per ALIGNMENT.md".into());
        }

        let record = MemoryRecord {
            id: record_id.to_string(),
            key: key.to_string(),
            content: content.to_string(),
            embedding,
            kind,
            scope,
            retention,
            provenance,
            is_deleted: false,
            is_training_eligible,
        };

        // Write record
        {
            let mut recs = self.records.write().unwrap();
            recs.insert(record_id.to_string(), record);
        }

        // Update indices
        {
            let mut idx = self.index.write().unwrap();
            idx.key_to_active_id
                .insert(key.to_string(), record_id.to_string());
            idx.source_to_records
                .entry(record_id.to_string())
                .or_default()
                .insert(record_id.to_string());

            // Tokenize content for inverted index
            let terms = Self::tokenize(content);
            for term in terms {
                idx.term_to_records
                    .entry(term)
                    .or_default()
                    .insert(record_id.to_string());
            }
        }

        // Audit log
        {
            let mut audit = self.audit_log.write().unwrap();
            audit.push(MemoryAuditEvent {
                timestamp_ms,
                action: "INSERT".into(),
                record_id: record_id.to_string(),
                key: key.to_string(),
                details: format!("Kind: {:?}, Scope: {:?}", kind, record_id),
            });
        }

        Ok(record_id.to_string())
    }

    /// Retrieves relevant records combining lexical (BM25) and semantic scoring
    /// within access boundaries and token budgets.
    pub fn retrieve(&self, query: &RetrievalQuery) -> RetrievalResponse {
        let start = Instant::now();
        let recs = self.records.read().unwrap();
        let idx = self.index.read().unwrap();

        let query_terms = Self::tokenize(&query.query_text);
        let mut candidate_ids: HashSet<String> = HashSet::new();

        // 1. Gather lexical candidates from inverted index
        for term in &query_terms {
            if let Some(matched) = idx.term_to_records.get(term) {
                for id in matched {
                    candidate_ids.insert(id.clone());
                }
            }
        }

        // If query has embedding or empty terms, evaluate all active records permitted by scope
        if query_terms.is_empty() || query.query_embedding.is_some() {
            for id in recs.keys() {
                candidate_ids.insert(id.clone());
            }
        }

        let total_active_docs = recs.values().filter(|r| !r.is_deleted).count().max(1);
        let avg_doc_len = {
            let total_len: usize = recs
                .values()
                .filter(|r| !r.is_deleted)
                .map(|r| r.content.len())
                .sum();
            (total_len as f32 / total_active_docs as f32).max(1.0)
        };

        let mut scored_results: Vec<RetrievedMemory> = Vec::new();
        let mut conflicts: Vec<MemoryConflict> = Vec::new();

        for id in &candidate_ids {
            let record = match recs.get(id) {
                Some(r) => r,
                None => continue,
            };

            // Filter deleted or superseded
            if record.is_deleted || record.provenance.superseded_by.is_some() {
                continue;
            }

            // Check retention expiration
            if let RetentionPolicy::TtlMs(ttl) = record.retention {
                if query.current_timestamp_ms > record.provenance.created_at_ms + ttl {
                    continue; // Stale / expired
                }
            }

            // Scope filter
            if !self.is_scope_accessible(&record.scope, query) {
                continue;
            }

            // Allowed kinds filter
            if let Some(kinds) = &query.allowed_kinds {
                if !kinds.contains(&record.kind) {
                    continue;
                }
            }

            // 2. Lexical scoring (BM25)
            let lexical_score = self.compute_bm25_score(
                &query_terms,
                &record.content,
                avg_doc_len,
                total_active_docs,
                &idx,
            );

            // 3. Semantic scoring (cosine similarity if embeddings present)
            let semantic_score = match (&query.query_embedding, &record.embedding) {
                (Some(q_emb), Some(r_emb)) => Self::cosine_similarity(q_emb, r_emb),
                _ => {
                    // Fallback to token overlap Jaccard
                    let doc_terms = Self::tokenize(&record.content);
                    if query_terms.is_empty() || doc_terms.is_empty() {
                        0.0
                    } else {
                        let intersect = query_terms.intersection(&doc_terms).count();
                        let union = query_terms.union(&doc_terms).count();
                        intersect as f32 / union as f32
                    }
                }
            };

            // Combined score (50% lexical, 50% semantic)
            let combined = 0.5 * lexical_score + 0.5 * semantic_score;

            if combined >= query.min_score {
                let est_tokens = (record.content.len() / 4).max(1);
                scored_results.push(RetrievedMemory {
                    record: record.clone(),
                    lexical_score,
                    semantic_score,
                    combined_score: combined,
                    estimated_tokens: est_tokens,
                });
            }
        }

        // Sort descending by combined score
        scored_results.sort_by(|a, b| {
            b.combined_score
                .partial_cmp(&a.combined_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Detect conflicts among retrieved keys (e.g. conflicting preferences)
        let mut seen_keys: HashMap<String, String> = HashMap::new();
        for item in &scored_results {
            if let Some(prior_id) = seen_keys.get(&item.record.key) {
                conflicts.push(MemoryConflict {
                    key: item.record.key.clone(),
                    active_record_id: item.record.id.clone(),
                    prior_record_id: prior_id.clone(),
                    reason:
                        "Multiple active records share identical topic key with divergent content"
                            .into(),
                });
            } else {
                seen_keys.insert(item.record.key.clone(), item.record.id.clone());
            }
        }

        // Bounded budget packing
        let mut final_results = Vec::new();
        let mut token_sum = 0;
        let mut sources = Vec::new();

        for item in scored_results.into_iter().take(query.max_results) {
            if token_sum + item.estimated_tokens <= query.max_tokens_budget
                || final_results.is_empty()
            {
                token_sum += item.estimated_tokens;
                sources.push(item.record.provenance.source_id.clone());
                final_results.push(item);
            } else {
                break;
            }
        }

        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;

        RetrievalResponse {
            results: final_results,
            total_tokens_used: token_sum,
            conflicts_detected: conflicts,
            latency_ms,
            provenance_sources: sources,
        }
    }

    /// User deletion control: marks record as deleted and expunges from index immediately.
    pub fn delete_record(
        &self,
        record_id: &str,
        reason: &str,
        timestamp_ms: u64,
    ) -> Result<(), String> {
        let (key, content) = {
            let mut recs = self.records.write().unwrap();
            let record = recs
                .get_mut(record_id)
                .ok_or_else(|| format!("Record '{record_id}' not found"))?;
            record.is_deleted = true;
            (record.key.clone(), record.content.clone())
        };

        // Remove from indices
        {
            let mut idx = self.index.write().unwrap();
            if idx.key_to_active_id.get(&key) == Some(&record_id.to_string()) {
                idx.key_to_active_id.remove(&key);
            }
            let terms = Self::tokenize(&content);
            for term in terms {
                if let Some(set) = idx.term_to_records.get_mut(&term) {
                    set.remove(record_id);
                }
            }
        }

        // Audit log
        {
            let mut audit = self.audit_log.write().unwrap();
            audit.push(MemoryAuditEvent {
                timestamp_ms,
                action: "DELETE".into(),
                record_id: record_id.to_string(),
                key,
                details: format!("Reason: {reason}"),
            });
        }

        Ok(())
    }

    /// Revokes an entire data source, permanently expunging all its records from retrieval.
    pub fn revoke_source(&self, source_id: &str, timestamp_ms: u64) -> usize {
        let matching_ids: Vec<String> = {
            let recs = self.records.read().unwrap();
            recs.values()
                .filter(|r| r.provenance.source_id == source_id && !r.is_deleted)
                .map(|r| r.id.clone())
                .collect()
        };

        let count = matching_ids.len();
        for id in matching_ids {
            let _ = self.delete_record(&id, "Source access revoked by user", timestamp_ms);
        }

        count
    }

    /// Purges all expired records exceeding their TTL.
    pub fn purge_expired(&self, current_timestamp_ms: u64) -> usize {
        let expired_ids: Vec<String> = {
            let recs = self.records.read().unwrap();
            recs.values()
                .filter(|r| {
                    if r.is_deleted {
                        return false;
                    }
                    if let RetentionPolicy::TtlMs(ttl) = r.retention {
                        current_timestamp_ms > r.provenance.created_at_ms + ttl
                    } else {
                        false
                    }
                })
                .map(|r| r.id.clone())
                .collect()
        };

        let count = expired_ids.len();
        for id in expired_ids {
            let _ = self.delete_record(&id, "TTL expired", current_timestamp_ms);
        }

        count
    }

    /// Inspects stored records for user transparency and audit verification.
    pub fn inspect_records(&self, filter_scope: Option<&AccessScope>) -> Vec<MemoryRecord> {
        let recs = self.records.read().unwrap();
        recs.values()
            .filter(|r| {
                if r.is_deleted {
                    return false;
                }
                if let Some(scope) = filter_scope {
                    &r.scope == scope
                } else {
                    true
                }
            })
            .cloned()
            .collect()
    }

    /// Returns storage volume and index telemetry.
    pub fn get_storage_metrics(&self) -> MemoryStorageMetrics {
        let recs = self.records.read().unwrap();
        let idx = self.index.read().unwrap();

        let total = recs.len();
        let active = recs.values().filter(|r| !r.is_deleted).count();
        let deleted = total - active;
        let bytes = recs
            .values()
            .filter(|r| !r.is_deleted)
            .map(|r| r.content.len())
            .sum();
        let terms = idx.term_to_records.len();
        let sources = recs
            .values()
            .map(|r| &r.provenance.source_id)
            .collect::<HashSet<_>>()
            .len();

        MemoryStorageMetrics {
            total_records: total,
            active_records: active,
            deleted_records: deleted,
            total_bytes_content: bytes,
            inverted_index_terms: terms,
            unique_sources: sources,
        }
    }

    /// Exports full audit events.
    pub fn export_audit_log(&self) -> Vec<MemoryAuditEvent> {
        self.audit_log.read().unwrap().clone()
    }

    fn is_scope_accessible(&self, scope: &AccessScope, query: &RetrievalQuery) -> bool {
        match scope {
            AccessScope::Global => true,
            AccessScope::User(req_user) => query.user_id.as_deref() == Some(req_user.as_str()),
            AccessScope::Session(req_sess) => {
                query.session_id.as_deref() == Some(req_sess.as_str())
            }
        }
    }

    fn compute_bm25_score(
        &self,
        query_terms: &HashSet<String>,
        doc_content: &str,
        avg_doc_len: f32,
        total_docs: usize,
        idx: &MemoryIndex,
    ) -> f32 {
        if query_terms.is_empty() {
            return 0.0;
        }

        let doc_terms = Self::tokenize_list(doc_content);
        let doc_len = doc_content.len() as f32;

        let k1: f32 = 1.2;
        let b: f32 = 0.75;
        let mut score: f32 = 0.0;

        for q_term in query_terms {
            let df = idx
                .term_to_records
                .get(q_term)
                .map(|s| s.len())
                .unwrap_or(0);
            let idf = ((total_docs as f32 - df as f32 + 0.5) / (df as f32 + 0.5))
                .ln_1p()
                .max(0.1);

            let tf = doc_terms.iter().filter(|&t| t == q_term).count() as f32;
            if tf > 0.0 {
                let term_score =
                    idf * (tf * (k1 + 1.0)) / (tf + k1 * (1.0 - b + b * (doc_len / avg_doc_len)));
                score += term_score;
            }
        }

        // Normalize score monotonically into [0, 1]
        (score / (score + 1.0)).min(1.0)
    }

    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() || a.is_empty() {
            return 0.0;
        }

        let mut dot = 0.0;
        let mut norm_a = 0.0;
        let mut norm_b = 0.0;

        for (x, y) in a.iter().zip(b.iter()) {
            dot += x * y;
            norm_a += x * x;
            norm_b += y * y;
        }

        let denom = norm_a.sqrt() * norm_b.sqrt();
        if denom < 1e-9 {
            0.0
        } else {
            (dot / denom).clamp(-1.0, 1.0).max(0.0)
        }
    }

    fn tokenize(text: &str) -> HashSet<String> {
        text.split(|c: char| !c.is_alphanumeric())
            .filter(|w| w.len() >= 2)
            .map(|w| w.to_lowercase())
            .collect()
    }

    fn tokenize_list(text: &str) -> Vec<String> {
        text.split(|c: char| !c.is_alphanumeric())
            .filter(|w| w.len() >= 2)
            .map(|w| w.to_lowercase())
            .collect()
    }
}
