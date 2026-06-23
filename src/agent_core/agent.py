from agent_core.core import Flow, FlowRunResult, Payload, RunContext, TraceOptions


class Agent:
    def __init__(self, flow: Flow) -> None:
        self.flow = flow

    def run(
        self,
        payload: Payload = None,
        *,
        max_steps: int = 100,
        trace: TraceOptions | bool | None = None,
        context: RunContext | None = None,
    ) -> FlowRunResult:
        return self.flow.run(
            payload,
            max_steps=max_steps,
            trace=trace,
            context=context,
        )
