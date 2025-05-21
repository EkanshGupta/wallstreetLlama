import helper
import globals

query = "For a beginner, how do I invest to maximize and optimize my portfolio?"
obj = helper.InferRAG()
obj.rag_generate(query)