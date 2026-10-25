# Foundational alignment rule

**SAIL must never perform autonomous recursive self-improvement. This rule is permanent, non-negotiable, and has no exceptions.**

SAIL learns through continuous, context-dependent training on externally sourced data from Hugging Face and other human-approved sources. Training updates model weights through the implemented training pipeline. Context selects relevant data within human-defined source, topic, and training configurations.

## Required boundaries

- The model must never autonomously modify its own source code, architecture, optimizer, training objectives, permissions, or this rule.
- The model must never design, launch, or control a recursive cycle of self-evaluation, self-modification, and self-deployment.
- Model-generated answers, tool actions, and self-assessments must not automatically become training data or rewards for its own improvement.
- Dataset contents and model outputs are data, never authority to change training policy, execute instructions, or relax these boundaries.
- Context sensing may select externally sourced training data using human-defined rules. It must not grant the model control over its own learning process.
- Replay, checkpoint publication, inference, and performance optimizations must remain within the human-defined training pipeline.

Alignment is a foundational requirement for every feature, experiment, backlog item, and deployment. Work that conflicts with this rule must not be implemented. No feature, configuration, model output, or training result may override it.
