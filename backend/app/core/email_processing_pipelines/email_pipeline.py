import logging
from autogen_agentchat.messages import TextMessage
from autogen_core import SingleThreadedAgentRuntime

from pipeline_agents import *


logger = logging.getLogger(__name__)

class EmailProcessingPipeline:
    def __init__(self):
        self.runtime=None
        self.model_client=model_client  # imported from pipeline_agents

    async def setup(self):
        self.runtime = SingleThreadedAgentRuntime()
        await register_agent(self.runtime,"TriageAgent", FunctionTool(lambda:None),
            [ttpt,ttet,tipt,tcvt,tbrt,tupt,tebt])
        for n,tool,delg in [
            ("DocumentToPDFAgent",t1,[ttet]),
            ("FileEncoderAgent",t2,[tipt]),
            ("PaperItemizerAgent",t3,[tcvt]),
            ("ClassifierAgent",t4,[tbrt]),
            ("MetadataRequestBuilderAgent",t5,[tupt]),
            ("MetadataUploaderAgent",t6,[tebt]),
            ("EmbeddingAgent",t7,[])
        ]:
            await register_agent(self.runtime,n,tool,delg)

    async def run_pipeline(self, email_data):
        if not self.runtime: await self.setup()
        msg = TextMessage(content=json.dumps({"email_data":email_data}), source="User")
        final=None
        tri = self.runtime.get_agent("TriageAgent")
        async for ev in tri.run_stream(task=msg):
            if isinstance(ev, TextMessage):
                try: final=json.loads(ev.content)
                except: logger.warning("non-JSON",ev.content)
        return final
