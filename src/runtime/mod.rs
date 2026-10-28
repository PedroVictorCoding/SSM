//! Native Host Runtime, Bounded Timestamped Event Bus, and Supervisor for SAIL.
//!
//! Provides concurrency and streaming infrastructure for multimodal assistants:
//! - [`event`]: Versioned attributable runtime events and reference-counted media buffers (`SharedAudioBuffer`, `SharedVideoBuffer`).
//! - [`bus`]: Bounded thread-safe event bus with explicit overflow policies (`DropOldest`, `DropNewest`, `Reject`), non-blocking audio callbacks, and hierarchical cancellation.
//! - [`supervisor`]: Worker supervision, health tracking, crash isolation, and graceful shutdown.
//! - [`replay`]: Deterministic headless event recording and replay for reproducible smoke testing.

pub mod adapter;
pub mod bus;
pub mod dialogue;
pub mod event;
pub mod memory;
pub mod multimodal_context;
pub mod replay;
pub mod resource;
pub mod segmented;
pub mod segmented_host;
pub mod session;
pub mod supervisor;
pub mod verification;

pub use adapter::{
    AssistantCoordinator, BackendCapabilities, BackendInfo, BackendModality, BackendRequestError,
    FinishReason, GenerateRequest, GenerationChunk, GenerationMetadata, MockModelAdapter,
    ModelBackendAdapter, ModelFormat, NativeSailAdapter, PretrainedLocalAdapter,
    StreamingUtf8Decoder, ToolProposal, validate_model_format, validate_request_capabilities,
};
pub use bus::{
    BoundedEventQueue, BusError, CancellationToken, EventBus, OverflowPolicy, SequenceDiscrepancy,
};
pub use dialogue::{
    DialogueEvent, DialogueState, FullDuplexCoordinator, StateTransitionEvent, ToolExecutionStatus,
    TrackedToolExecution, TurnRecord,
};
pub use event::{EventPayload, RuntimeEvent, SharedAudioBuffer, SharedVideoBuffer};
pub use memory::{
    AccessScope, FactKind, LongTermMemoryStore, MemoryAuditEvent, MemoryConflict, MemoryProvenance,
    MemoryRecord, MemoryStorageMetrics, RetentionPolicy, RetrievalQuery, RetrievalResponse,
    RetrievedMemory,
};
pub use multimodal_context::{
    ContextProvenanceTrace, ModalityBudgetBreakdown, MultimodalContextAssembler, ReferenceIntent,
    ReferenceResolutionResult, ResolvedTarget, SynchronizedContextPacket, TimelineEntry,
};
pub use replay::EventRecorder;
pub use resource::{
    ConcurrentLoadMetrics, DeviceBudgetConfig, HostReservationGuard, MemoryReservationGuard,
    PriorityTier, RealMemoryAllocation, ResourceScheduler, ResourceUtilizationSnapshot,
    SessionGuard,
};
pub use segmented::{
    SegmentedAssistantCoordinator, SegmentedAssistantError, SegmentedCancellationOutcome,
    SegmentedDegradation, SegmentedTurnDisposition, SegmentedTurnOutcome,
};
pub use segmented_host::{SegmentedAssistantHost, SegmentedHostError, SegmentedHostEvent};
pub use session::{
    BoundedSessionCache, SessionInspectionReport, SessionManager, SessionRecord, SessionTask,
    TaskLifecycleStatus, TaskRevision,
};
pub use supervisor::{Supervisor, WorkerHandle, WorkerRole, WorkerState};
pub use verification::{
    MultiStepTaskCoordinator, StepExecutionStatus, StepPostcondition, TaskPlan, TaskPlanStep,
    VerifiedTaskExecutionTrace,
};
