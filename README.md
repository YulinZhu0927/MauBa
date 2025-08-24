# MauBa: A Multi-Agent Coordination Framework for Vision-Language-Guided Zero-Shot Control of Unmanned Aerial Vehicles

This paper was accepted by the 2025 IEEE Global Communications Conference (GLOBECOM), Taipei, Taiwan.

## Abstract
While prompt-based methods have shown initial
potential for enabling large language models (LLMs) to control
unmanned aerial vehicles (UAVs), they suffer from limited
flexibility, inadequate visual perception, and low code generation accuracy—often leading to hallucinated outputs due to
static templates and rigid API wrappers. To address these
limitations, we propose MauBa, a multi-agent coordination
framework for zero-shot vision-language UAVs control. MauBa
comprises three collaborative agents—Supervisor, Coder, and
Tracker—that jointly decompose and execute natural language
instructions. The Supervisor manages task delegation and dialogue context; the Coder utilizes a vectorized API knowledge
base with Retrieval-Augmented Generation (RAG) for precise
code synthesis; and the Tracker performs object detection and
spatial localization via visual grounding. Without any fine-tuning,
MauBa achieves task success rates of 93.3% for language-action
and 83.3% for vision-language tasks in the AirSim simulation environment, significantly outperforming prompt-based and 
single-agent baselines in both accuracy and control robustness. Ablation
studies further validate the essential role of task coordination and
knowledge retrieval in ensuring reliable UAV decision-making.
These findings suggest that modular multi-agent collaboration
offers a promising pathway for scalable multimodal reasoning
in real-world autonomous systems.
