from ragas import evaluate
from pathlib import Path
import json

from rag.qa_chain import make_chain

dataset = [json.loads(l) for l in Path("eval/queries.jsonl").read_text().splitlines()]
chain   = make_chain()

evaluate(dataset=dataset, chain=chain, lang="en", print_table=True)
