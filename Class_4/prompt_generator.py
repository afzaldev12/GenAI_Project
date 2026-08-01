from langchain_core.prompts import PromptTemplate

# template for the prompt
template =PromptTemplate(
    template="""
Please summarize the research paper titled "{paper_input}" with the following specifications:
Explanation Style: {style_input}
Explanation Length: {length_input}
1.Mathematical Details:
  - Include relevant mathematical equations, formulas, and derivations that are central to the research paper's findings.
  - Explain the mathematical concepts and techniques used in the paper, providing step-by-step explanations where necessary.
2.Analogies:
  - Use relatable analogies to help explain complex mathematical concepts in a more intuitive manner.
  If certain information is not available in the paper,.
respond with:"Insufficient information available" instead of guessing.
Ensure that the response is accurate and based on the content of the research paper.
""",
input_variables = ["paper_input", "style_input", "length_input"],
validate_template=True
)

template.save('template.json')
