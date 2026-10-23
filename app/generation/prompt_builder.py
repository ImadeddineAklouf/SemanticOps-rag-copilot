class PromptBuilder:
    def build(self, question: str, context: str) -> str:
        return (
            f"Contexte :\n{context}\n\n"
            f"Question : {question}\n\n"
            "Réponse :"
        )