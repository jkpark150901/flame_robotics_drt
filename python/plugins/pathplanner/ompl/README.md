# OMPL joint-space backend

This backend targets the official OMPL Python binding built from OMPL tag
`2.0.0`. The binding must expose `ompl.base` and `ompl.geometric` with RRT,
RRTConnect, RRTstar, InformedRRTstar, and BITstar.

OMPL is imported only when an OMPL-backed planner executes. The application can
therefore start without OMPL, but planning fails with an explicit dependency
error. Use the `legacy_rrt_star` plugin for rollback; OMPL planner names do not
silently fall back to the in-house implementation.

Joint states use normalized active dimensions in `[0, 1]`. Fixed joints are
removed from the OMPL state and restored by `JointStateCodec`. Every exact OMPL
solution is converted to full raw q, connected to the exact goal, and checked by
`PlannerBase.verify_path()` before it is returned.
