# Verification Standard

Vega changes must be tested as real local product behavior, not only as isolated code.

## Required For Backend Agent Changes

For changes to chat, memory, retrieval, tools, routing, model calls, or traces:

- Run a real chat through the local API or UI.
- Use prompts that exercise the feature being changed.
- Inspect the saved conversation and trace events.
- Confirm failure paths are visible in the trace when something goes wrong.
- Keep test data local and delete throwaway smoke conversations when appropriate.

## Required For Frontend Changes

For rendered UI changes:

- Use Playwright or the Browser plugin to load the app.
- Capture screenshots for desktop and mobile when layout is affected.
- Click the relevant controls.
- Check console errors.
- Verify scroll, overflow, and responsive behavior when touching panels, drawers, sidebars, or lists.

## Parallel Verification

Use parallel subprocesses where practical:

- build and compile checks can run in parallel
- API checks can run alongside frontend builds
- Playwright checks should run after the relevant dev server is live

The goal is to make verification fast without making it shallow.
