mode: bootstrap
purpose: disarm Ember v0.0.28 after successful one-time T4 submission; prevent duplicate paid GPU launches from later pushes
source_checkpoint_repo: Jmiller18899/ember-v0.0.27-t4
preflight_job: Jmiller18899/6a9f7034259f8e97255ed966
training_job: Jmiller18899/6a9f7175e686246ca69a9ca9
training_commit: 500368b020f2b98108bb88aa9761e8f9fbffcd42
source_checkpoint: checkpoints/ember-agent-v0.0.27-sequence-completion-repair-20260908T014043Z/best.pt
source_sha256: 98b3c9200e7558cc0bb4cb810e0a8b42f98c447c39c84db914faec0ae3e89c24
v027_best_step: 599
protected_baseline: v0.0.27 (expanded exact-copy 0.2222, expanded continuation 0.8205, continuation-only margin health -0.6846)
expanded_first_token_top1_rate: 0.8666666666666667
expanded_full_sequence_margin_health: -0.8376828384399412
sequences_within_reach: 0.17777777777777778 (16/90 in [-0.75, 0))
very_near_boundary: 4/90 in [-0.25, 0)
within_half_logit: 14/90 in [-0.50, 0)
weakest_decision_first_token: 13/90
weakest_decision_eos: 0/90
format_parity: PASS
boundary_mining: PASS
hardware: t4-small
learning_rate: 1.2e-6
max_steps: 600
next_action: observe the submitted v0.0.28 training job only; do not launch another T4 session
