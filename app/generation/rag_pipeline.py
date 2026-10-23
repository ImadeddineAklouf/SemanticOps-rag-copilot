from app.generation.context_builder import (
    ContextBuilder,
)
from app.generation.llm_service import (
    LLMService,
)
from app.generation.prompt_builder import (
    PromptBuilder,
)


class RAGPipeline:
    def __init__(self, retriever, llm_service=None,
                 context_builder=None, prompt_builder=None):
        self.retriever = retriever
        self.context_builder = context_builder or ContextBuilder()
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.llm_service = llm_service or LLMService()

    def ask(self, question: str) -> str:
        results = self.retriever.retrieve(question)
        context = self.context_builder.build(results)

        if not context.strip():
            return "Je n'ai trouvé aucune information pertinente dans les documents."

        prompt = self.prompt_builder.build(question, context)
        return self.llm_service.generate(prompt)