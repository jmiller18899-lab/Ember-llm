# Ember v0.0.47 observed-envelope template result

Formal corrected run: `34304357331`, job `102317829988`, commit `f622b4e5a984fbd4001ac342084c8b61fae71fec`.

The CPU-only diagnostic completed successfully after 111 focused/inherited tests passed. The pinned v0.0.31 step-479 checkpoint and frozen residual prompts were unchanged.

Measured representation stability:

- successful envelope sources: 18;
- unique observed non-value generated-token templates: 1;
- successful sources reaching the query-value boundary without a value-bearing token: 18/18 (100%);
- matched structural controls reproducing the observed template leave-one-out: 13/13 (100%).

This validates the observed generated-token prefix as a stable value-free structural representation, unlike the re-tokenized compact JSON prefix tested in v0.0.46.

Four non-rescued failures split as follows:

- `system_target_short_code_02`: full observed non-value prefix is top-1; failure is after the prefix, at or after the value boundary.
- `system_target_path_05`: full observed non-value prefix is top-1; failure is after the prefix, at or after the value boundary.
- `system_target_short_code_03`: first divergence is the tool marker (rank 3, margin -1.6754); all later observed-prefix tokens are supported once that point is forced. v0.0.45 showed a one-marker force alone still does not rescue the whole envelope, so this case has entry competition plus a later failure.
- `system_target_long_code_08`: first divergence is the tool marker (rank 35, margin -1.1382); later observed-prefix tokens are supported once that point is forced. v0.0.45 likewise showed a one-marker force alone does not rescue the whole envelope.

The two separately understood pure entry-suppression failures remain:

- `system_target_long_code_02`: tool rank 226, one-marker rescue succeeds.
- `system_target_long_code_05`: tool rank 72, one-marker rescue succeeds.

Artifact: `ember-v047-safe-prefix-34304357331`, artifact ID `10086109841`, ZIP SHA256 `e6083ac8937743898e8d59e5a0ea66171bde4d5cda0abd435f9822f9e2b8fe36`.

No optimizer, GPU, training, promotion, deployment, or production integration ran.

Conclusion: v0.0.47 establishes a stable value-free structural representation. The next learning canary can therefore separate an envelope-entry objective from a value-boundary/placement objective instead of supervising one monolithic completion.
